"""Serial Docker execution and controller-owned cold workspace snapshots.

No host shell is used. This module is controller infrastructure, never a model tool.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import threading
import time
import uuid

MAX_WORKSPACE_BYTES = 256 * 1024 * 1024
MAX_PATHS = 20000
MAX_OUTPUT_BYTES = 1024 * 1024
SNAPSHOT_OUTPUT_BYTES = MAX_WORKSPACE_BYTES * 4 // 3 + MAX_PATHS * 1024


def snapshot_workspace(root: Path) -> dict:
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("workspace root must be a real directory")
    entries, total = [], 0
    def visit(directory):
        nonlocal total
        for path in sorted(directory.iterdir()):
            info = path.lstat()
            relative = path.relative_to(root).as_posix()
            entry = {"path": relative, "mode": stat.S_IMODE(info.st_mode)}
            if stat.S_ISDIR(info.st_mode):
                entry["kind"] = "directory"
                entries.append(entry)
                if len(entries) > MAX_PATHS:
                    raise ValueError("workspace path limit exceeded")
                visit(path)
            elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                total += info.st_size
                if total > MAX_WORKSPACE_BYTES:
                    raise ValueError("workspace byte limit exceeded")
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                with os.fdopen(fd, "rb") as stream:
                    opened = os.fstat(stream.fileno())
                    if (opened.st_ino, opened.st_dev, opened.st_nlink) != (info.st_ino, info.st_dev, 1):
                        raise ValueError("workspace changed during snapshot")
                    data = stream.read(MAX_WORKSPACE_BYTES - (total - info.st_size) + 1)
                if len(data) != info.st_size:
                    raise ValueError("workspace changed during snapshot")
                entry.update(kind="file", data=base64.b64encode(data).decode("ascii"))
                entries.append(entry)
            else:
                raise ValueError("symlink, hardlink or special file rejected: " + relative)
            if len(entries) > MAX_PATHS:
                raise ValueError("workspace path limit exceeded")
    visit(root)
    return {"version": 1, "root_mode": stat.S_IMODE(root.stat().st_mode), "entries": entries, "total_bytes": total}


def validate_snapshot(snapshot: dict) -> dict:
    if snapshot.get("version") != 1 or not isinstance(snapshot.get("entries"), list):
        raise ValueError("invalid snapshot")
    if len(snapshot["entries"]) > MAX_PATHS:
        raise ValueError("workspace path limit exceeded")
    seen, total = {}, 0
    for entry in snapshot["entries"]:
        name = entry.get("path")
        if not isinstance(name, str) or not name or name == "." or str(PurePosixPath(name)) != name or name.startswith("/") or ".." in PurePosixPath(name).parts or "\x00" in name:
            raise ValueError("unsafe snapshot path")
        if name in seen:
            raise ValueError("duplicate snapshot path")
        mode = entry.get("mode")
        if type(mode) is not int or not 0 <= mode <= 0o7777:
            raise ValueError("invalid snapshot mode")
        kind = entry.get("kind")
        if kind == "file":
            data = base64.b64decode(entry["data"], validate=True)
            total += len(data)
        elif kind != "directory":
            raise ValueError("invalid snapshot entry")
        seen[name] = kind
        if total > MAX_WORKSPACE_BYTES:
            raise ValueError("workspace byte limit exceeded")
    for name in seen:
        for parent in PurePosixPath(name).parents:
            if str(parent) != "." and seen.get(str(parent)) != "directory":
                raise ValueError("missing or non-directory parent")
    mode = snapshot.get("root_mode")
    if type(mode) is not int or not 0 <= mode <= 0o7777:
        raise ValueError("invalid root mode")
    if snapshot.get("total_bytes") != total:
        raise ValueError("snapshot byte count mismatch")
    return snapshot


def restore_snapshot(snapshot: dict, root: Path) -> None:
    validate_snapshot(snapshot)
    root = Path(root)
    if root.is_symlink():
        raise ValueError("symlink destination")
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("snapshot destination must be empty")
    for entry in sorted(snapshot["entries"], key=lambda item: (item["path"].count("/"), item["path"])):
        path = root / entry["path"]
        if entry["kind"] == "directory":
            path.mkdir(mode=0o700)
        else:
            with path.open("xb") as stream:
                stream.write(base64.b64decode(entry["data"], validate=True))
            path.chmod(entry["mode"])
    for entry in sorted(snapshot["entries"], key=lambda item: item["path"].count("/"), reverse=True):
        if entry["kind"] == "directory":
            (root / entry["path"]).chmod(entry["mode"])
    root.chmod(snapshot["root_mode"])


def _helper_source() -> str:
    # These trusted helpers are sent through stdin, not placed in agent storage.
    import inspect
    return "import base64,json,os,stat\nfrom pathlib import Path,PurePosixPath\n" + f"MAX_WORKSPACE_BYTES={MAX_WORKSPACE_BYTES}\nMAX_PATHS={MAX_PATHS}\n" + "\n".join(inspect.getsource(fn) for fn in (snapshot_workspace, validate_snapshot, restore_snapshot))


def boundary_mounts_valid(inspected: dict, state: dict) -> bool:
    """Require both Docker intent and kernel observations for writable mounts."""
    host = inspected.get("HostConfig", {})
    if host.get("Binds") or set(host.get("Tmpfs") or {}) != {"/workspace", "/tmp"}:
        return False
    if any(m.get("Type") != "tmpfs" or m.get("Destination") not in ("/workspace", "/tmp") for m in inspected.get("Mounts", [])):
        return False
    mounts = {}
    try:
        for line in state["mountinfo"].splitlines():
            before, after = line.split(" - ", 1)
            fields, filesystem = before.split(), after.split()
            destination = fields[4]
            if destination in mounts:
                return False
            mounts[destination] = (filesystem[0], set(fields[5].split(",")) | set(filesystem[2].split(",")))
        for destination in ("/workspace", "/tmp"):
            filesystem, options = mounts[destination]
            required = {"rw", "nosuid", "nodev"}
            if destination == "/tmp":
                required.add("noexec")
            if filesystem != "tmpfs" or not required <= options or "ro" in options:
                return False
            observed = state["mount_stats"][destination]
            if observed != {"uid": 1000, "gid": 1000, "mode": 0o700}:
                return False
            if any(path.startswith(destination + "/") for path in mounts):
                return False
    except (KeyError, ValueError, IndexError, TypeError):
        return False
    return True


class DockerRuntime:
    def __init__(self, image_id: str, run_dir: Path, *, is_grader=False):
        if not image_id or image_id.startswith("-"):
            raise ValueError("invalid image identifier")
        self.image_id, self.run_dir, self.is_grader = image_id, Path(run_dir), is_grader
        self.container_id = None
        self.invalidated = False
        self._lock = threading.Lock()
        self._event_number = 0
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _record(self, event):
        self._event_number += 1
        event = {"event_number": self._event_number, "time_ns": time.time_ns(), **event}
        with (self.run_dir / "runtime_events.jsonl").open("a") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")

    def _command(self, argv, input_bytes=None, timeout=30, cap=MAX_OUTPUT_BYTES):
        started = time.monotonic()
        process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        captured = [bytearray(), bytearray()]
        lengths = [0, 0]
        def drain(stream, index):
            while chunk := stream.read(65536):
                lengths[index] += len(chunk)
                captured[index].extend(chunk[:max(0, cap - len(captured[index]))])
        def feed():
            try:
                process.stdin.write(input_bytes or b"")
                process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        threads = [threading.Thread(target=drain, args=(process.stdout, 0)), threading.Thread(target=drain, args=(process.stderr, 1)), threading.Thread(target=feed)]
        for thread in threads:
            thread.start()
        error = None
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            error = "timeout_container_invalidated"
            process.kill()
            if self.container_id:
                self.invalidated = True
                subprocess.run(["docker", "rm", "-f", self.container_id], capture_output=True, timeout=15, check=False)
                self.container_id = None
            process.wait()
        for thread in threads:
            thread.join(timeout=5)
        result = {"exit_code": process.returncode, "stdout": captured[0].decode("utf-8", "replace"), "stderr": captured[1].decode("utf-8", "replace"), "elapsed": time.monotonic() - started, "truncated": any(length > cap for length in lengths), "error": error, "stdout_bytes": lengths[0], "stderr_bytes": lengths[1]}
        return result

    def start(self):
        if self.container_id or self.invalidated:
            raise RuntimeError("runtime already started or invalidated")
        name = "c3-r1-" + uuid.uuid4().hex
        argv = ["docker", "run", "--detach", "--name", name, "--network", "none", "--read-only", "--user", "1000:1000", "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true", "--cpus", "2", "--memory", "4g", "--memory-swap", "4g", "--pids-limit", "128", "--ipc", "none", "--tmpfs", "/workspace:rw,nosuid,nodev,size=256m,uid=1000,gid=1000,mode=0700", "--tmpfs", "/tmp:rw,nosuid,nodev,noexec,size=64m,uid=1000,gid=1000,mode=0700", "--workdir", "/workspace", "--label", "c3.r1.role=" + ("grader" if self.is_grader else "agent"), "--entrypoint", "sleep", self.image_id, "infinity"]
        result = self._command(argv)
        if result["exit_code"] != 0 or result["error"]:
            self._command(["docker", "rm", "-f", name])
            raise RuntimeError("Docker start failed: " + result["stderr"])
        self.container_id = result["stdout"].strip()
        inspected = self._inspect()
        (self.run_dir / "runtime_provenance.json").write_text(json.dumps({"requested_image": self.image_id, "effective_image_id": inspected["Image"], "container_id": self.container_id, "create_argv": argv, "inspect": self._filtered_inspect(inspected), "runtime_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2))
        self._record({"kind": "start", "container_id": self.container_id, "image_id": inspected["Image"]})
        return self

    def close(self):
        if self.container_id:
            result = self._command(["docker", "rm", "-f", self.container_id])
            self._record({"kind": "close", **result})
            if result["exit_code"] != 0:
                raise RuntimeError("container removal failed")
            self.container_id = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()

    def _execute(self, argv, input_bytes=None, timeout=30, cap=MAX_OUTPUT_BYTES):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("concurrent execution forbidden")
        try:
            if not self.container_id or self.invalidated:
                raise RuntimeError("runtime not active")
            result = self._command(["docker", "exec", "-i", self.container_id, *argv], input_bytes, timeout, cap)
            self._record({"kind": "exec", "argv": argv, "input_sha256": hashlib.sha256(input_bytes or b"").hexdigest(), **{key: value for key, value in result.items() if key not in ("stdout", "stderr")}})
            return result
        finally:
            self._lock.release()

    def exec_python(self, code: str, timeout=30) -> dict:
        result = self._execute(["python", "-I", "-u", "-c", code], timeout=timeout)
        self._record({"kind": "model_python", "code_sha256": hashlib.sha256(code.encode()).hexdigest(), **result})
        return result

    def exec_fixed(self, argv, input_bytes=None) -> dict:
        """Controller-only execution; never expose this API as a model tool."""
        if not isinstance(argv, (list, tuple)) or not argv or any(not isinstance(value, str) or "\x00" in value for value in argv):
            raise ValueError("argv must contain literal strings")
        result = self._execute(list(argv), input_bytes)
        self._record({"kind": "controller_fixed", "argv": list(argv), **result})
        return result

    def seed_workspace(self, source: Path):
        snapshot = snapshot_workspace(source)
        payload = json.dumps(snapshot).encode()
        result = self._execute(["python", "-I", "-c", _helper_source() + "\nimport sys\nrestore_snapshot(json.load(sys.stdin),Path('/workspace'))"], payload, timeout=120)
        if result["exit_code"] or result["error"]:
            raise RuntimeError("workspace seed failed: " + result["stderr"])
        return self.verify_workspace(snapshot)

    def _snapshot(self):
        result = self._execute(["python", "-I", "-c", _helper_source() + "\nprint(json.dumps(snapshot_workspace(Path('/workspace'))))"], timeout=120, cap=SNAPSHOT_OUTPUT_BYTES)
        if result["exit_code"] or result["error"] or result["truncated"]:
            raise RuntimeError("workspace snapshot failed: " + result["stderr"])
        return validate_snapshot(json.loads(result["stdout"]))

    def export_workspace(self, destination: Path) -> dict:
        snapshot = self._snapshot()
        restore_snapshot(snapshot, destination)
        digest = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()
        result = {"snapshot_sha256": digest, "total_bytes": snapshot["total_bytes"], "paths": len(snapshot["entries"]), "destination": str(destination)}
        self._record({"kind": "export", **result})
        return result

    def verify_workspace(self, source_or_snapshot) -> dict:
        expected = validate_snapshot(source_or_snapshot) if isinstance(source_or_snapshot, dict) else snapshot_workspace(Path(source_or_snapshot))
        actual = self._snapshot()
        result = {"verified": actual == expected, "expected_paths": len(expected["entries"]), "actual_paths": len(actual["entries"])}
        self._record({"kind": "verify", **result})
        if not result["verified"]:
            raise RuntimeError("workspace verification failed (including deleted files and modes)")
        return result

    def _inspect(self):
        result = self._command(["docker", "inspect", self.container_id])
        if result["exit_code"] or result["truncated"] or result["error"]:
            raise RuntimeError("docker inspect failed")
        return json.loads(result["stdout"])[0]

    @staticmethod
    def _filtered_inspect(data):
        # Deliberately do not persist Config.Env or host-wide information.
        return {"Image": data["Image"], "State": {key: data["State"].get(key) for key in ("Running", "Pid", "Status")}, "Config": {key: data["Config"].get(key) for key in ("User", "WorkingDir", "Labels")}, "HostConfig": {key: data["HostConfig"].get(key) for key in ("NetworkMode", "ReadonlyRootfs", "CapAdd", "CapDrop", "SecurityOpt", "NanoCpus", "Memory", "MemorySwap", "PidsLimit", "PidMode", "IpcMode", "Privileged", "Binds", "Tmpfs")}, "Mounts": data.get("Mounts", [])}

    def audit_boundary(self) -> dict:
        data = self._filtered_inspect(self._inspect())
        code = """import os,json,stat
from pathlib import Path
own=os.getpid(); processes=[]
for p in Path('/proc').iterdir():
 if p.name.isdigit() and int(p.name)!=own:
  try: processes.append({'pid':int(p.name),'cmdline':(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode('utf8','replace').strip()})
  except FileNotFoundError: pass
print(json.dumps({'processes':processes,'tmp_entries':[str(p) for p in Path('/tmp').rglob('*')],'namespaces':{p.name:os.readlink(p) for p in Path('/proc/self/ns').iterdir()},'status':Path('/proc/self/status').read_text(),'mountinfo':Path('/proc/self/mountinfo').read_text(),'mount_stats':{p:{'uid':os.stat(p).st_uid,'gid':os.stat(p).st_gid,'mode':stat.S_IMODE(os.stat(p).st_mode)} for p in ('/workspace','/tmp')}}))
"""
        result = self._execute(["python", "-I", "-c", code])
        if result["exit_code"] or result["error"] or result["truncated"]:
            raise RuntimeError("boundary audit failed")
        state = json.loads(result["stdout"])
        host = data["HostConfig"]
        unexpected = [p for p in state["processes"] if not (p["pid"] == 1 and p["cmdline"] == "sleep infinity")]
        mounts_ok = boundary_mounts_valid(data, state)
        effective_ok = (host["NetworkMode"] == "none" and host["ReadonlyRootfs"] and not host["Privileged"] and not host["CapAdd"] and "ALL" in (host["CapDrop"] or []) and "no-new-privileges:true" in (host["SecurityOpt"] or []) and host["NanoCpus"] == 2000000000 and host["Memory"] == 4294967296 and host["MemorySwap"] == 4294967296 and host["IpcMode"] == "none" and host["PidsLimit"] == 128 and not host["Binds"] and host["PidMode"] in ("", None) and data["Config"]["User"] == "1000:1000" and mounts_ok)
        status_lines = dict(line.split(":", 1) for line in state["status"].splitlines() if ":" in line)
        effective_ok = effective_ok and status_lines.get("CapEff", "").strip() == "0000000000000000" and status_lines.get("NoNewPrivs", "").strip() == "1"
        audit = {"qualified": bool(effective_ok and not unexpected and not state["tmp_entries"]), "effective_config_ok": bool(effective_ok), "unexpected_processes": unexpected, "inspect": data, **state, "claim_scope": "observed idle tool boundary; not a proof of escape resistance"}
        self._record({"kind": "boundary_audit", **audit})
        return audit
