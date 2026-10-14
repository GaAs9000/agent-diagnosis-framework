# Sokoban cold-container compatibility

This is an explicit L1 environment adaptation, not an unmodified LHTB runtime.
The task wrapper, public native engine/daemon/CLI, levels, and official verifier
are unchanged. No benchmark task assets or lock files are modified.

## Pinned build

Build context: `docker/lhtb_sokoban_cold/`.

```sh
docker build --pull=false --network=none --platform=linux/amd64 \
  -t l1-lhtb-sokoban-cold:0.2.1 docker/lhtb_sokoban_cold
```

The FROM reference pins locally cached
`zli12321/lhtb-sokoban@sha256:11ae8cbb0a45d8cad145278caf1dc6cde99dd6bc7479dd486696759e80d1c950`.
Its native image ID is
`sha256:b2a9f13cab4b017a58e2f9dd9abd1db475f52f17bae290e3dc464dfeb0668fb0`.
The locally built derived image ID is
`sha256:3ad9ed1024eff491a7d387e7a052f1aa0e63c7999fc3ed2ddb027fa830d5434a`.
It has no registry RepoDigest because it has not been pushed. Labels record the
base digest, base image ID, and `sokoban-public-cli-history-replay-v1` adaptation.
Record the resolved derived image ID with each run; the local tag is mutable.

## Behavior

The original `/usr/local/bin/sokoban` is renamed `sokoban-native`. The installed
shim takes its original path, so bare `sokoban` and the untouched task helper
`/workspace/input/bin/sokoban` both invoke it. Docker environment defaults match
the original task helper: `SOKOBAN_RUN_DIR=/workspace/runs`, daemon log in that
directory, and socket in transient `/tmp`.

A fresh native daemon truncates its log and starts a pristine game. Before
starting it, the shim reads the previous native accepted log and metadata from
the persistent workspace. It then replays accepted tokens through the original native CLI, suppressing
only replay output. Contiguous directional tokens are grouped within the native
per-command move limit; batches never cross RESET boundaries. `U`, `D`, `L`, `R`, and
`RESET` are the public daemon's actual log tokens. Directional commands blocked
by a wall are also native accepted/logged moves. Invalid or oversized sequences
are not logged by the native daemon. This shim implements no game transitions,
solutions, or scorer logic. It verifies exact accepted-log and native-counter
parity after replay before invoking the user's current command.

The authoritative checkpoint is `runs/cold-state.json`: one atomic snapshot
containing the exact public native accepted log and metadata bytes. Native
daemon working files reside in transient `/tmp/sokoban-cold-native`. The
in-memory board is reconstructed by the unchanged daemon. Daemon startup and
replay cannot truncate the persistent checkpoint. After replay verifies and the
actual CLI returns, one atomic snapshot replacement publishes the native result;
`runs/moves.log`, `runs/meta.json`, and submission are repaired from that snapshot.
Existing pre-adaptation native log/meta checkpoints can also be imported. Native commands
within one container reuse its live daemon only after exact working log/meta
comparison against the committed snapshot, guarded by a workspace client lock.
A surviving daemon with partially replayed or uncommitted command state is
shut down through the native CLI and reconstructed from the saved checkpoint.
Explicit `shutdown` records a flag in that snapshot so the next CLI starts a fresh game, matching
the continuous native CLI's shutdown/restart semantics. Cold container teardown
alone preserves the history for restoration.

After the actual CLI call and atomic checkpoint publication, the shim mirrors the native accepted log
into `/workspace/submission/moves.log`. It also normalizes that mirror before
restoration, removing the original wrapper's speculative preappend when the
persisted accepted history is valid. Malformed persisted history fails closed. Status/repeated status do not append tokens.
Native argparse exit codes and actual CLI stdout/stderr pass through unchanged.

## Evidence and limits

On 2026-10-14, the following finite, no-API test passed: **4 passed in 48.38s**.

```sh
L1_DOCKER_TESTS=1 .venv/bin/python -m pytest \
  tests/l1_live/test_sokoban_cold_semantics.py -q
```

The tests run actual `DockerRuntime.execute` calls under its standard non-root,
read-only root filesystem, network-none, capability-drop constraints. A separate
continuous container runs the pinned native base as the control. It compares
exact stdout/stderr, exit codes, native log bytes, native metadata bytes, and the
submission mirror across D, L, status, repeated status, reset plus move,
invalid/overlong/empty moves, argparse rejection, and shutdown/restart. A copied
workspace checkpoint containing `D\nL\n` is restored in another workspace and
continued through new cold containers; the checkpoint and parent remain unchanged.
An additional 300-move native history test interrupts replay with `timeout 0.3`,
verifies persistent log/meta and snapshot remain unchanged, and successfully
retries in a fresh container. It covers both imported native checkpoints and
committed atomic snapshots. Another regression executes
`timeout 0.3 sokoban status; sokoban status` within one container, proving a
surviving partial daemon cannot replace the 300-move checkpoint. It also invokes
a public native move directly into the transient working state without
committing, simulating acceptance before interrupted publication; the next shim
rolls that daemon back and returns the canonical 300-move native status.

A shell/task-wrapper failure or runtime interruption **before entering the shim**
cannot be repaired by the shim: the authoritative native history remains
unchanged, but speculative submission preappend may remain until the next shim
call. A dedicated test reproduces and records this limitation. Runtime failure
or incomplete output must retain the existing L1 execution-validity treatment.
The official evaluator remains independent and can accept model-authored output
under its original contract; this shim adds no action-output restriction.
Replay cost grows with accepted history length. These tests prove the listed
native command semantics and checkpoint reconstruction, not model performance,
universal gameplay equivalence, or a scientific capability claim.
