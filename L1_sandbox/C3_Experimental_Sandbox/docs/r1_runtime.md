# R1 serial Docker runtime

`DockerRuntime(image_id, run_dir, is_grader=False)` is controller infrastructure.
Only `exec_python(code, timeout=30)` and public inspection are eligible for model
tools. Do not expose `exec_fixed`, Docker CLI, seed/export, host paths or snapshot
metadata to a model. Grading uses a separate container built from the same pinned
package base; official grader code belongs only in its grader image. The controller
runs fixed arguments such as `python /opt/grader/r1_grader.py TASK /workspace/submission`.

The context manager starts a nonroot UID/GID 1000 container with a read-only root,
network disabled, all capabilities removed, no-new-privileges, no host mounts,
private PID namespace, 2 CPUs, 4 GiB RAM (no extra swap), and 128 PIDs.
`/workspace` (256 MiB) and `/tmp` (64 MiB) are writable tmpfs mounts. The agent image
accepts `BASE_IMAGE` so the controller can supply the shared pinned package image.

`seed_workspace(Path)` sends a trusted JSON/base64 snapshot through stdin. It never
uses `docker cp`, which does not reliably populate a tmpfs mount. `export_workspace`
reconstructs the complete tree in a new or empty host directory. Both preserve
hidden entries, empty directories, regular file bytes and permission modes; links,
hardlinks and special files fail closed. Snapshots are limited to 256 MiB of file
bytes and 20,000 entries. `verify_workspace(Path | snapshot)` compares the entire
actual tree, including missing files and modes, and raises on a difference.
Snapshots must remain in controller storage outside agent mounts.

Execution drains stdout/stderr with a 1 MiB cap per stream and UTF-8 replacement
for invalid bytes. Results include `exit_code`, `stdout`, `stderr`, `elapsed`,
`truncated`, `error`, and original byte counts. A timeout invalidates and forcibly
removes the container; execution is never retried. Controller helpers have separate
larger bounded output limits for transferring snapshots. Execution is serial;
concurrent requests fail immediately.

`audit_boundary()` records filtered effective Docker inspect configuration,
container namespaces, process status, process inventory and `/tmp` entries.
It also records the raw kernel `/proc/self/mountinfo` table and mount-root UID,
GID and mode. Qualification requires actual tmpfs mounts at `/workspace` and
`/tmp`, with `rw,nosuid,nodev`, plus `noexec` for `/tmp`, UID/GID 1000 and mode
0700. Docker tmpfs declarations alone do not qualify a boundary; empty Docker
`Mounts` metadata is supported when these kernel observations pass. Nested mounts
under either writable root and unexpected inspect mounts or binds are rejected.
Qualification requires the enforced configuration, the baseline PID 1
`sleep infinity`, no other persistent process, and an empty `/tmp`. The audit
process itself is excluded by PID. Qualification is an observation at an idle
boundary, not proof of escape resistance. Call it after a completed tool and before
accepting an R1 checkpoint; do not infer that background work is replayable.

`runtime_provenance.json` stores effective image ID, filtered configuration,
creation arguments and runtime source hash. `runtime_events.jsonl` stores raw
execution results for model Python, code/input hashes, lifecycle and audit events.
It deliberately omits host environment variables and Docker `Config.Env`.
Local unit tests exercise snapshot validation and reconstruction without Docker;
container behavior requires the controller's serial Docker validation.
