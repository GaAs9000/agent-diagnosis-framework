# Controlled R1 replay admission

Real replay remains blocked unless the trusted local controller supplies a
concrete `R1AdmissionAuthority` to both `ReplayQualification` and
`ExperimentScheduler`. Synthetic qualification is unchanged. Registration
checks structural checkpoint validity; admission happens after its immutable
checkpoint identity exists.

```python
authority = R1AdmissionAuthority(private_host_directory)
replay = ReplayQualification(evidence_store, real_admission=authority)
receipt_id = authority.issue(
    checkpoint_id, checkpoint_record, source_audit, restore_audit,
    restoration_report, model_qualification, evidence_paths,
)
scheduler = ExperimentScheduler(
    database, qualifier=replay.qualify, probe_authorizer=registry,
    executor=controller_executor, real_admission=authority,
)
```

The controller must perform the actual source boundary audit, cold restore into
a fresh Docker runtime, equality verification, and native model qualification
before calling `issue`. Admission itself makes no Docker or model calls. Never
mount the authority directory, key, receipts, evaluator or evidence artifacts
into the model workspace. Its directory is mode 0700, its randomly generated
HMAC key is mode 0600, and receipt/index files are immutable content addressed
records. Receipts bind the entire checkpoint, including filesystem snapshot,
exact message prefix, runtime and system identities.

Both audits use `DockerRuntime.audit_boundary()` raw dictionaries. Admission
re-evaluates the immutable image ID, user/workdir, running state, network,
read-only root, capabilities, resource limits, PID/IPC settings, absence of
binds, exact tmpfs paths/options, observed privilege status, idle process list,
and empty `/tmp`. `qualified=True` alone cannot admit a boundary. Source and
restore must have distinct positive host PIDs. The expected idle process is
PID 1 running `sleep infinity`.

Mount admission also requires raw kernel `mountinfo` and `mount_stats` from
the audit. The shared `boundary_mounts_valid` policy checks that `/workspace`
and `/tmp` are actual writable tmpfs mounts with nosuid/nodev, `/tmp` has
noexec, both directories have UID/GID 1000 and mode 0700, and neither has
nested mounts. Docker may report an empty `inspect.Mounts` list for tmpfs;
this is accepted only with the complete kernel proof. Docker tmpfs intent
alone, missing observations, or any reported bind mount remains blocked.

`restoration_report` contains `verified=True`, integer `expected_paths` and
`actual_paths` with equal values, and `messages_equal`, `runtime_equal`,
`system_equal`, each exactly `True`. The controller obtains these from actual
filesystem restoration and exact conversation/configuration comparison.

`model_qualification` contains nonempty strings `provider`, `model`,
`model_fingerprint` (64 lowercase hexadecimal SHA256), `tool_call_id`,
`observation_tool_call_id`, `nonce`, `observed_nonce`, and `independent_nonce`.
The call IDs must match, and all three nonce values must match an independently
checked tool observation. `native_tool_calls`, `prompt_tokens`, and
`completion_tokens` must be positive integers. Runtime `model_config` must
contain the same provider, model and fingerprint. The fingerprint identifies
the pinned model/tokenizer/provider configuration, not merely a model name.

`evidence_paths` contains at least six distinct nonempty regular host files:
the exact five JSON documents for checkpoint, source audit, restore audit,
equality report and model qualification, plus the original native model
request/response trace. JSON documents may be pretty printed. The authority
hashes file bytes itself. Every qualification verifies the signature, full
checkpoint binding, raw admission conditions and every referenced file hash;
missing or changed evidence fails closed. Successful qualification includes
`admission_receipt`, the immutable receipt SHA256. The scheduler requires the
actual bound `ReplayQualification.qualify` method and the identical authority
object, then independently rechecks admission before authorization and on
cache hits. Existing phase seed separation and branch cache identity remain
in force.

The branch executor must separately audit each fresh runtime and verify the
actual restored workspace and prefix before continuation. A stored admission
receipt describes the qualification boundary; it cannot attest a later
container. This is trusted-controller evidence integrity for an observed idle
boundary, not a claim of malicious-host resistance or a proof against Docker
escape. The admission unit tests use explicitly synthetic receipt data and
establish no real model qualification or scientific claim.
