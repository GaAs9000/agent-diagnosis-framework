# Runtime negative acceptance

The controller calls `run_runtime_negatives(agent_image, run_dir)` explicitly,
serially, outside model episodes. It creates four independent containers through
`DockerRuntime`, retaining their provenance and runtime events in per-case folders.
No official assets are changed. A controller-only canary is created under
`run_dir/controller_private`, outside every container mount.

The probes require refusal of reads of the official manifest, grader, Docker
socket and unmounted controller canary; refusal of a write under `/opt`; and a
qualified clean boundary with the required effective configuration. Separate
containers then verify that a workspace symlink is rejected during export, a
persistent background `sleep` disqualifies the boundary, and a `/tmp` cache file
disqualifies the boundary. These states are unsupported by R1 cold reconstruction.
Container context managers remove each container before the next case starts.

`runtime_negative_report.json` preserves raw Python results, audits and expected
refusals. Failed cases are persisted before the runner raises. The report concerns
these observed operations and boundary states; it does not prove resistance to
kernel attacks or every possible escape. Unit tests use a fake runtime only;
actual acceptance requires root-controlled serial Docker execution.
