# Real serial OfficeVal baseline

`await run_baseline(task_id, manifest, agent_image, grader_image,
model_qualification, output_dir, seed=2026, max_turns=10)` performs one real native
tool baseline. Both image arguments must be immutable Docker SHA-256 IDs, and
the model receipt must identify the pinned provider/model with passed real
native-call/nonce qualification. The output directory must be new.

Only frozen public inputs enter the actor workspace, with editable `0644` files
and a `0700` root. The model receives the public task instruction and designated
deliverable name. The generic system prompt supplies no rubric or answer. Native
`workspace_python` calls are the only actor edits. Generation uses seed 2026,
temperature 0, 2048 output tokens, a 16384 context limit, serial calls and no
automatic model retry. Complete raw events and usage remain in the agent logs;
only delivered tool results are bounded by the qualification transform.

`grade_workspace(task_id, manifest, workspace, grader_image, run_dir)` copies
only the designated DOCX into `/workspace/submission` in a new independent grader
container and executes `python -I /opt/grader/r1_grader.py`. Callers must close
the actor first. Missing designated files are legitimate task failures. Reference
files and actor programs never enter the grader. Official verifier code executes
only inside the grader; the host checks JSON using the C3 adapter. Each grade
retains official raw JSON, complete process output, exit code, selected artifact
hash, image ID, stdout/stderr hashes and process JSON hash.

At each finished tool turn, a checkpoint retains complete workspace bytes/modes,
exact messages, raw source audit, runtime and checkpoint record. `SUPPORTED_STATE`
is the existing `cold-tool-boundary-v1` contract: no pending calls, persistent
processes, external state or caches. A failed source audit remains visible.
Checkpoints register as real (`is_synthetic=False`) in the default
`ReplayQualification`; all stay ineligible until the controller supplies an
evidence-backed `R1AdmissionAuthority`. This runner never creates admission.

The return object (also `result.json`) contains `task_id`, `seed`, `is_synthetic`,
`output_dir`, `system_identity`, `runtime`, `model_qualification`, `initial_grade`,
`final_grade`, `agent_result`, `checkpoints`, `replay_store`, `final_workspace`,
`execution_valid`, `task_completed`, `target_resolved`, `regression_free` and
`outcome`. Each checkpoint reference contains its ID, event index, workspace,
message/audit/checkpoint-record paths, source audit flag, qualification report and
runtime identity. The system identity hashes the complete model, agent, tool,
system prompt, sampling and runtime configuration. Runtime environment values
are captured from the image, never copied from the host.

Completion uses the explicit C3 official-rule mapping, never a score threshold.
`target_resolved` requires dimension 1 and all positive rules. `regression_free`
compares negative-rule hits against the initial grade, and is `None` when either
grade has unassessed coverage or unavailable rule assessment. `task_completed`
preserves `None` for incomplete official coverage. Outcomes are `success` for
complete valid execution, `partial` for unassessed completion or satisfied positive
targets with remaining penalties, `failure` for unmet targets, and
`infrastructure_invalid` for invalid execution/grading. No reruns or seed sweeps
are performed.
