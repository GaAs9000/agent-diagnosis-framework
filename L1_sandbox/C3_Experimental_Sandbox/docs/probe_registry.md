# C3-E2 v1.0 probe registry

`ProbeRegistry.from_file("configs/probes.v1.json")` loads the complete closed vocabulary.
`get(id)` returns a defensive copy; `identity(id)` returns SHA-256 of canonical JSON
containing the protocol version and complete specification, including `control_id` and
`leakage_policy`. `matched_control_id` is an equal compatibility alias for `control_id`. Extra fields, arbitrary text,
paths, answers, scripts, resource removals, temperature shifts and changed preconditions
are rejected. The frozen `leakage_policy` declares a closed generic public-observation presentation
allowlist, prohibits answer material, task-specific patches and target object paths, and
requires per-task review including controls. This metadata is a vocabulary restriction,
**not proof that task leakage is absent or actual rendering is safe**.

| Line | Visibility target | Feedback target |
|---|---|---|
| E2-S | `visibility.compact` | `feedback.aggregate` |
| E2-F | `visibility.expanded` | `feedback.structured` |

Visibility changes the rendering of already public observations: compact collapses
redundant views; expanded presents the same public content in separate sections.
Feedback changes the grouping of already public tool feedback: aggregate groups it;
structured separates its public fields. Neither mechanism reveals hidden verifier
results, adds task-specific hints, removes necessary information, or grants tools.
These are competing presentation mechanisms, not competing names for the same prompt.
Execution and checks that these transformations actually preserve content live in the
runtime adapter; this registry supplies no transformation implementation.

Each target has an ID ending `.matched` with the same mechanism, preconditions and
`matched_mode`, but a `sham` payload. The runtime must match the target's presentation
cost and opportunity while preserving the original information presentation. Matching
must be audited on actual execution; the name alone establishes no matched-control
validity. All arms preserve sampling (including seed and temperature), budgets, necessary
inputs and tools. No extra budget or sampling intervention is permitted in this version.

`experiment_panel(line, target_probe_id, competing_probe_id)` returns four specs with
roles `original`, `target`, `matched`, `competing`. The competing probe must use the
other mechanism and the same line. Every spec includes its identity,
`checkpoint_policy="same_checkpoint"`, and `requires_fresh_execution=True`.
The original arm is a fresh continuation from the same checkpoint; an archived original
trajectory cannot substitute for it. Downstream scheduling must enforce checkpoint,
runtime, sampling and budget equality. Every arm, including original and sham, needs
its own task-bound review.

## Case and review contract

`validate_for_case(id, case_contract, review)` returns `eligible`, `reasons`, `identity`.
The trusted controller supplies:

```python
case_contract = {
    "task_identity": "<64 lowercase hex characters>",
    "is_synthetic": True,  # mandatory; real cases use False
    "line": "E2-F",
    "available_inputs": ["public_input_id"], "necessary_inputs": ["public_input_id"],
    "available_tools": ["public_tool_id"], "necessary_tools": ["public_tool_id"],
    "observation_capabilities": ["redundant_observation_views", "public_feedback_fields"],
    "sampling": {"seed": 2026, "temperature": 0.1},
    "budgets": {"tokens": 1000},
}
```

Resource IDs are inventory metadata, never agent-facing target paths. Visibility requires
`redundant_observation_views`; feedback requires `public_feedback_fields`. The task
identity must cover the frozen task contract and assets. Missing review or mismatched
inventory, task identity, probe hash or experiment line denies eligibility.

The review has exactly the following fields. Evidence identities refer to controller-owned
review records that reviewers must inspect; hash syntax validation does not inspect or
certify the records' truth.

```python
from c3_sandbox.evidence._storage import encode, digest
review_body = {
    "task_identity": case_contract["task_identity"],
    "probe_identity": registry.identity(probe_id),
    "line": case_contract["line"],
    "source": {"kind": "synthetic_fixture", "signer_id": "fixture-author",
               "record_identity": "<review-source-record SHA-256>"},
    "findings": {
        "leakage": {"verdict": "no_task_answer_material",
                    "evidence_identities": ["<leakage-review-evidence SHA-256>"]},
        "solvability": {"verdict": "necessary_resources_preserved",
                        "evidence_identities": ["<solvability-review-evidence SHA-256>"]},
        "reviewed_inputs": case_contract["available_inputs"],
        "reviewed_tools": case_contract["available_tools"],
    },
}
# Only for synthetic fixtures, after review findings have actually been recorded:
review = dict(review_body, signature={"scheme": "sha256-fixture",
                                     "value": digest(encode(review_body))})
```

Real cases require `source.kind` equal to `human` or `task_verifier` and an authenticated
HMAC-SHA256 signature over `encode(review_body)`. The controller separately supplies
`case_contract["trusted_review_keys"][signer_id]`, a secret string of at least 32 characters.
The review signature is `hmac.new(key.encode(), encode(review_body), hashlib.sha256).hexdigest()`
and uses scheme `hmac-sha256`. The trust mapping must be provisioned by the controller,
never from model output or the review itself. Keys must never enter trajectories,
probe payloads, exports or task workspaces. HMAC authenticates the declared signer and
findings; it does not establish the reviewer's competence or turn those findings into
an automated safety certificate. Review source records must document reviewer identity,
authority, evidence inspected and actual task-specific solvability/leakage reasoning.

The OfficeVal 002 E2-F panel now has a task-bound closed-schema/resource review
and real rendering evidence in `runs/e2-r1/probe-audit-v2` and
`runs/e2-r1/scheduler/probe_review_source.json`. Other task/line reviews remain
pending. The runtime adds an explicitly labelled public observation advisory
after the unchanged prefix; it does not invent a tool call or replace historical
observations. Global statistics may change both information and attention, so
the effect cannot identify a unique internal mechanism. Necessary resources are
preserved; this is not a guarantee that the model will solve the task.
Passing synthetic tests alone still authorizes no real experiment.

The registry was cross-checked against `C3_E2_PROTOCOL_v1_0.md` D4, complete branch
identity, fresh-original and matched-control requirements. It supplies eligibility
contracts only; scheduler and replay implementations enforce actual execution boundaries.
