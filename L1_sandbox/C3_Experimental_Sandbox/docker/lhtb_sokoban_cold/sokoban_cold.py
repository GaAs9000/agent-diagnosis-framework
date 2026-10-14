#!/usr/bin/env python3
"""Cold restoration via the unchanged public CLI, with atomic native snapshots."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CLIENT = "/usr/local/bin/sokoban-native"
RUN = Path(os.environ["SOKOBAN_RUN_DIR"])
SNAPSHOT = RUN / "cold-state.json"
SUBMISSION = Path("/workspace/submission/moves.log")
# Never let daemon startup truncate the persistent checkpoint's accepted log.
WORK = Path("/tmp/sokoban-cold-native")
os.environ["SOKOBAN_RUN_DIR"] = str(WORK)
os.environ["SOKOBAN_DAEMON_LOG"] = str(WORK / "daemon.log")
sys.path.insert(0, "/opt/sokoban")
import sokoban_cli as native


def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".native.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def invoke(args, *, replay=False):
    return subprocess.run([CLIENT, *args], stdout=subprocess.PIPE if replay else None,
                          stderr=subprocess.PIPE if replay else None, check=False)


def saved():
    if SNAPSHOT.exists():
        state = json.loads(SNAPSHOT.read_text())
    else:
        # Import a pre-adaptation native checkpoint without editing its history.
        state = {"moves": (RUN / "moves.log").read_text() if (RUN / "moves.log").exists() else "",
                 "meta": (RUN / "meta.json").read_text() if (RUN / "meta.json").exists() else None,
                 "shutdown": False}
    tokens = state["moves"].splitlines()
    if any(t not in {"U", "D", "L", "R", "RESET"} for t in tokens):
        raise RuntimeError("invalid native accepted history; refusing cold restore")
    return state


def mirror(state):
    atomic(RUN / "moves.log", state["moves"].encode("ascii"))
    if state["meta"] is not None:
        atomic(RUN / "meta.json", state["meta"].encode("utf-8"))
    atomic(SUBMISSION, state["moves"].encode("ascii"))


def restore(state):
    if native._alive():
        matches = (not state["shutdown"] and (WORK / "moves.log").exists()
                   and (WORK / "meta.json").exists()
                   and (WORK / "moves.log").read_text() == state["moves"]
                   and (WORK / "meta.json").read_text() == state["meta"])
        if matches:
            return
        # A killed shim can leave a partially replayed or uncommitted daemon
        # alive inside this same tool container. Roll it back to the checkpoint.
        if invoke(["shutdown"], replay=True).returncode:
            raise RuntimeError("could not stop uncommitted native daemon")
        deadline = time.monotonic() + 5
        while native._alive() or Path(native.SOCKET_PATH).exists():
            if time.monotonic() >= deadline:
                raise RuntimeError("uncommitted native daemon did not stop")
            time.sleep(0.01)
    prior = state["moves"] if not state["shutdown"] else ""
    prior_meta = state["meta"] if not state["shutdown"] else None
    native._ensure_daemon()
    tokens = prior.splitlines()
    limit = int(os.environ.get("SOKOBAN_MAX_MOVES_PER_CMD", "30"))
    if limit < 1:
        raise RuntimeError("invalid native command move limit")
    while tokens:
        if tokens[0] == "RESET":
            args, tokens = ["reset"], tokens[1:]
        else:
            count = 0
            while count < min(limit, len(tokens)) and tokens[count] != "RESET":
                count += 1
            args, tokens = ["move", "".join(tokens[:count])], tokens[count:]
        if invoke(args, replay=True).returncode:
            raise RuntimeError("native CLI history replay failed")
    if (WORK / "moves.log").read_text() != prior:
        raise RuntimeError("native accepted history differs after cold restore")
    if prior_meta is not None and json.loads((WORK / "meta.json").read_text()) != json.loads(prior_meta):
        raise RuntimeError("native counters differ after cold restore")


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "client.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = saved()
        # The single atomic snapshot is authoritative, even if a previous mirror
        # publication or restoration was interrupted. Never publish replay prefixes.
        mirror(state)
        restore(state)
        result = invoke(sys.argv[1:])
        actual = {"moves": (WORK / "moves.log").read_text(),
                  "meta": (WORK / "meta.json").read_text(),
                  "shutdown": sys.argv[1:] == ["shutdown"] and result.returncode == 0}
        atomic(SNAPSHOT, json.dumps(actual, sort_keys=True).encode("utf-8"))
        mirror(actual)
        return result.returncode


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        sys.stderr.write(f"ERROR: sokoban cold compatibility: {exc}\n")
        sys.exit(1)
