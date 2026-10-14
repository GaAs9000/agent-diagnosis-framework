# D3 cold checkpoint replay qualification

The original D1–D5 tests established synthetic infrastructure only. R1 now adds
controller-owned, evidence-bound admission through `r1/admission.py` and a real
executor in `r1/branch_executor.py`. OfficeVal 002 has passed actual cold Docker
reconstruction and eight real model continuations. No declaration or boolean
alone authorizes real replay. The overall R1 gate remains incomplete: no E2-S
original, and no new tool call was observed after restoration. See
[R1 acceptance](E2_R1_ACCEPTANCE.md) for the exact supported scope and evidence.

## Protocol and legacy audit

The controlling requirement is section 6 of
`Agent_Latent_Capability_Audit_Kit/decisions/C3_MINIMAL_VALIDATION_PROTOCOL_v0_1.md`:
verify complete workspace and exact conversation/tool prefix, environment,
dependencies, process/services and other state before every branch. Recovery
failure excludes the branch from valid experiments. Directory permissions and
command blacklists cannot establish OS isolation.

The old `src/officeval_branch_replayer.py` in the audit kit was inspected without
executing it. `reconstruct_checkpoint` stages current baseline assets, then
reruns logged bash actions through `SandboxedEnvironment`. That is command
reexecution, not snapshot recovery. It ignores returned command outcomes and
checks only the files named in the old checkpoint, leaving extra files
undetected. Its `verified_hashes` result is not checked by `execute_branch`
before evaluation or the callback. It deletes preexisting destinations and
does not bind an immutable complete conversation, model/tool configuration,
environment, runtime image, services or process state. These findings do not
invalidate the existence of historical artifacts; they invalidate treating
that mechanism alone as complete or isolated cold replay.

## Supported scope and exact declarations

`ReplayQualification` binds one supported filesystem snapshot to the complete
JSON message prefix, a controller supplied fixed system identity, and full
runtime JSON. All JSON fields are retained. Canonicalization uses the existing
`evidence._storage.encode`: sorted object keys, ASCII escapes, compact
separators, no NaN. JSON object key ordering is irrelevant; array order and all
message/tool fields are identity relevant. This recovers exact JSON context,
not tokenizer hidden state, model KV caches or original JSON whitespace.

Under frozen `docs/C3_E2_PROTOCOL_v1_0.md`, the system identity must be the
SHA-256 of the complete canonical system configuration (model, scaffold, tools,
prompt and sampling/runtime settings), encoded as 64 lowercase hexadecimal
characters without a prefix. Plain model names, labels, empty strings and
mutable aliases are unqualified. Digest syntax alone cannot prove that the
controller hashed all required fields; the trusted controller supplies and
audits that configuration.

The required state contract is exactly:

```json
{
  "scope": "cold-tool-boundary-v1",
  "at_tool_boundary": true,
  "pending_tool_calls": 0,
  "persistent_processes": false,
  "external_state": "none",
  "caches": "none",
  "environment": "runtime.environment"
}
```

Missing, extra or unsupported state fields prevent qualification. Boolean
`false` is not accepted as the integer `0`. Tools must have finished before
capture; structured tool calls must have matching result messages. A v1
runtime is a JSON object with:

| Field | Requirement |
|---|---|
| `image_digest` | `sha256:` followed by 64 lowercase hexadecimal characters; no mutable image tag |
| `config` | Object containing frozen execution configuration, including logical cwd, platform, dependency identity, limits and network/clock policy as applicable |
| `model_config` | Object containing fixed model/tokenizer identities, generation settings and budget as applicable |
| `tool_config` | Object containing tool interface/implementation identities and settings |
| `environment` | Complete explicitly provisioned environment, string keys and values; exclude credentials |

Additional runtime JSON fields are retained and identity bound. The controller
must supply actual immutable identities and complete configuration, not aliases
or placeholders. Hashing declarations does not establish their truth. Synthetic
fixtures use deterministic fake identities clearly marked `is_synthetic=true`.
Setting that marker on real data would be mislabeling, not attestation.

Filesystem coverage is precisely `WorkspaceStore`'s complete regular-file and
directory tree: bytes, paths, empty directories, hidden files and POSIX modes,
including the root. Extra/missing paths and changed content/type/mode fail.
Symlinks and special files are unsupported and make capture unqualified.
Ownership, ACLs, extended attributes, timestamps, inode/hardlink topology,
open file descriptors, process memory, services, mounts, host state and external
caches are not restored. Environment is bound separately and must be freshly
provisioned by a runtime; this module never changes the host environment.
External effects include unfrozen network, clock dependence, remote model
sessions and external filesystem writes. Such required state is outside v1;
do not hide it by declaring `external_state="none"`.

## API and trusted controller integration

```python
qualification = ReplayQualification(store_path_or_WorkspaceStore)
checkpoint_id = qualification.register_checkpoint(
    workspace, messages, system_identity, runtime, state_contract, is_synthetic
)
metadata = qualification.qualify(checkpoint_id)
restored = qualification.restore(checkpoint_id, new_or_empty_destination)
verified = qualification.verify_restoration(
    checkpoint_id, destination, restored["messages"],
    restored["system_identity"], actual_runtime
)
```

Records live outside the workspace under `replay-checkpoints/<sha256>.json`.
The checkpoint ID hashes the immutable record, including snapshot ID, complete
context, system identity, runtime identity, state contract, synthetic marker
and registration failures. Content addressed snapshot manifests and objects
are independently verified; reopening a service uses the same records. Writes
are exclusive and stored bytes are read back. Inputs are copied through JSON,
so later caller mutation cannot change stored context. Invalid JSON input
raises before an ID is issued; missing or unsupported state is recorded as
unqualified, not silently accepted.

`qualify` returns `eligible_for_intervention`, `checkpoint_identity`,
`system_identity`, `runtime_identity`, `is_synthetic`, and `reasons`. It validates
the record and performs an actual `WorkspaceStore.restore` to a fresh temporary
directory, followed by the store's complete verification. `restore` adds
`valid`, `workspace_report`, and, on success, copies of `messages`, `runtime`
and `state_contract`. A nonempty destination is rejected without deletion.
`verify_restoration` adds `valid` and `workspace_report` and rejects differences
in the supplied workspace, exact message prefix, system identity or full runtime
JSON. An unqualified or corrupt checkpoint cannot produce a successful restore.

The scheduler must use this trusted service directly, restore each branch to a
fresh destination, provision the bound runtime, then call `verify_restoration`
with actual context/runtime before the callback. Saved eligibility metadata is
an audit result, not a future authorization token: it must not be replaced with
caller supplied `restore_verified=true` or reused after branch modifications.
Runtime drift is detected against supplied actual runtime JSON; this module
does not query Docker, the OS or a remote model to prove that JSON matches
reality. Concurrent writes between verification and callback are not prevented
by this filesystem store. Trusted lifecycle/ownership and runtime acceptance
must establish that boundary.

## Focused validation

`tests/test_e2_replay.py` uses temporary files only; no legacy commands, Docker,
model calls, server jobs or host configuration changes. Tests cover complete
cold restore, binary/hidden files, empty directories and modes, persisted
identity, caller mutation, extra/missing/tampered files, symlinks, nonempty
destinations, message/tool prefix drift, system/model/tool/image/environment
drift, incomplete/persistent/external/cache state, boolean override attempts,
real-run blocking, and corrupted/missing record/manifest/object data.
