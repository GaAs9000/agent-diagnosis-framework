# Real serial branch continuation

`RealBranchExecutor(replay, root, baseline, manifest, agent_image, grader_image,
rendered_probes, model_qualification, expected_grader_identity)` is a synchronous
`BranchSpec -> ExecutionOutcome` scheduler adapter. It starts the async native
continuation with `asyncio.run`; dispatch from a synchronous serial controller.

The controller must first admit checkpoints through `R1AdmissionAuthority`, review
and freeze the probe panel, and freeze the seed schedule. This executor rejects
synthetic specs and identity mismatches. Task identity uses the baseline's explicit
`task_identity`, or the canonical task-manifest hash. Images remain immutable IDs.
Runtime identity accepts either the checkpoint runtime hash or its complete runtime
dictionary. Probe identity is the registry identity; rendered text has a separate
checked hash and actual tokenizer count.

Every branch creates an exclusive directory named by its scheduler key. Restoration
checks workspace, exact prefix, system/runtime identity and real admission. Current
disk/server model fingerprint must match qualification. A fresh Docker runtime
verifies seeded bytes, passes a raw boundary audit, has a distinct source PID, and
matches the complete checkpoint environment. Admission is checked again immediately
before model execution.

The original arm injects nothing. Other arms append exactly one runtime-owned user
advisory containing the frozen rendered text. Base and injected prefixes, actual
token counts, probe record, branch sampling and runtime hash are controller evidence.
The agent uses the same native tool schemas and delivered-result transform as the
baseline. Its remaining turn allowance is 10 minus completed assistant turns in
the restored prefix. Branch seed comes from the frozen scheduler spec; generation
remains temperature 0, 2048 output tokens and a 16384 context window without retries.
Only actual model tool calls modify document bytes.

After final export and actor removal, an independent grader scores the designated
DOCX. Completion and regression flags use the baseline's C3 outcome adapter and
initial grade. Invalid execution returns no true success flags. The method-visible
result contains only four outcome flags, identities, evidence hashes and optional
raw total score; it exposes no controller paths, private records or grader details.
Full evidence stays in the controller directory.

Physical cost uses actual model prompt/completion usage. GPU seconds are unmeasured
(`None`). Environment time measures controller/tokenizer preparation, Docker setup,
tool execution, export/removal and grading, excluding the agent model-wait loop.
Wall time measures the full branch. The cost definition is persisted separately.
Exceptions write an indeterminate failure record and are rethrown for scheduler
protection. Existing attempts are never overwritten or automatically retried.

The unit tests use explicitly synthetic doubles for all external operations. They
do not provide real runtime qualification, model execution or admission evidence.
