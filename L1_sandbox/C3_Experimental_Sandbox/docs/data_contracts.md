# C3-E2 D1/D2 data contracts

Version `1.0`. These contracts provide data validation and development-only legacy adaptation. They do not establish independent task families, valid causal replay, or any scientific gate. No candidate task-family manifest is generated: grouping needs a source/template review.

## Trajectory

`validate_trajectory(record: dict) -> dict` returns a deep JSON copy and raises `ValueError` for incompatible data. Required top-level keys are `schema_version`, `episode_id`, `task_id`, `task_family_id`, `system_id`, `split`, `status`, `outcome`, `events`, `artifact_versions`, `checkpoint_refs`, and `provenance`. Unknown fields survive. Supported splits are `development`, `exploration`, `confirmation`, and `transfer` (the R4 new-task domain).

`outcome` contains exactly four independent `bool | null` values: `execution_valid`, `target_resolved`, `regression_free`, and `task_completed`. Null means evidence is unavailable for that field. Status retains the external scorer's completion label and its origin is recorded by `provenance.source` and `provenance.version`; it is not derived by counting true endpoint fields:

| Status | Exact condition |
|---|---|
| `success` | Requires `execution_valid=true` and `task_completed=true`. |
| `failure` | Requires `execution_valid=true` and `task_completed=false`; scorer labels failure. |
| `partial` | Requires `execution_valid=true` and `task_completed=false`; scorer labels partial completion. |
| `unknown` | Classification unavailable; compatible with endpoint values subject to the invalid-execution guard below. |

Valid execution by itself establishes neither task success nor partial progress. A failed task with valid execution remains `failure`. `target_resolved` and `regression_free` do not determine the label or force each other's values: their endpoints may differ from the scorer's whole-task criterion. Invalid or unknown execution only permits `unknown`. When `execution_valid=false`, none of the other three fields may be true, regardless of status; false/null preserve unsuccessful or unavailable endpoint evidence. Unknown execution does not force the remaining endpoint fields to null. The validator rejects inconsistent status rather than rewriting it or inferring a failure/partial distinction.

`events` and `artifact_versions` are ordered object lists. `checkpoint_refs` objects require `checkpoint_id` and boolean `replay_qualified`. A native qualified reference has `{checkpoint_id, checkpoint_identity, qualification_report_ref, replay_qualified: true}`. Both identities must be lowercase SHA256 strings addressing immutable checkpoint/report records. They are optional for an unqualified reference, but are hash-validated when supplied. D1 validates the representation and does not certify that a report is truthful or currently valid: the scheduler must load the referenced evidence and re-qualify through D3 before execution or cache reuse. Legacy references retain `{checkpoint_id, raw, replay_qualified: false, reason}` and cannot be promoted by adding these fields.

`provenance` requires nonempty `source` and `version` strings plus boolean `is_synthetic`. The JSON Schema is at `schemas/trajectory_schema.json`; byte/hash consistency and finite JSON checks additionally require the Python validator. JSON data must use string object keys and finite numbers; tuples and other Python-only values are rejected.

## Lossless legacy import

`import_legacy_episode(run_dir, *, task_family_id, system_id, split='development') -> dict` reads `summary.json`, `events.jsonl`, `artifacts.jsonl`, `checkpoints.jsonl`, and `verifier_results.jsonl`. Every file is required; no raw file is written. The caller supplies grouping/configuration labels; the importer cannot certify their reviewed validity. Any split other than `development` is rejected because these historical cases have already informed development.

For each filename, `provenance.raw_files[filename]` holds `sha256`, ASCII `base64`, and `records`. SHA256 covers the exact original bytes, including JSON whitespace, line terminators, blank lines and final newline. JSONL records preserve order and unknown keys. Summary `records` is an object. Strict parsing rejects duplicate JSON keys, nonfinite values, non-object JSONL rows and invalid UTF-8. Blank JSONL rows are absent from the parsed record list but preserved in base64. Raw bytes can be recovered with `base64.b64decode(entry['base64'])`; validation recomputes the hash and parses the original bytes to verify the retained records.

`events` and `artifact_versions` copy their corresponding original record arrays. Each legacy checkpoint is wrapped as `{checkpoint_id, raw, replay_qualified: false, reason}`. Partial file hashes and conversation prefixes do not establish complete workspace/process/environment/context restoration. Validation checks that arrays and checkpoint `raw` values still match the retained originals. All outcome fields remain null and status remains unknown. Old `completed`, `dim1_pass`, scores and SafeFix fields are retained as historical evidence, never converted into E2 target/regression/whole-task success.

## Family manifest

`validate_family_manifest(record: dict) -> dict` accepts:

```json
{"schema_version":"1.0","families":[{"task_family_id":"reviewed-family","split":"development","task_ids":["task-1"],"source_group_ids":["document-template-group"],"asset_hashes":{"source-asset-id":"<64 lowercase hex characters>"}}]}
```

Lists and asset maps must be nonempty. Family IDs and task IDs must be globally unique; IDs within a source-group list cannot repeat. A source group may cover multiple families in the same split, but cannot cross splits. An asset ID must always have one hash; identical bytes under different asset IDs cannot cross splits. The validator catches stated grouping violations, but does not discover semantic near-duplicates or prove that 40 independently sampled families exist.

## System manifest and identities

`validate_system_manifest(record: dict) -> dict` accepts `{schema_version: '1.0', systems: [system_entry, ...]}` with unique nonempty system IDs. Each entry requires `system_id`, `config`, and `config_sha256`. `config` contains exactly `model`, `agent`, `tools`, `prompt`, `sampling`, and `runtime`; each must explicitly describe that component (no null, empty string, or empty object; an empty tool list explicitly means no tools). Nested shapes remain configurable for adapters; pinned revisions, prompt content, seed/temperature, dependency/image/platform versions should be recorded by their caller.

`canonical_hash(value) -> str` hashes finite JSON serialized as UTF-8 with `sort_keys=True`, compact separators, `ensure_ascii=True`, and `allow_nan=False`. Array order and JSON numeric representation are significant; object key order is not. This matches the existing evidence store's canonical encoding. There is no Unicode normalization.

`system_identity(system_entry) -> str` verifies the required full config and its supplied SHA256, then returns that hash. The system ID is a label and is excluded from the config identity; changing any config component requires a new hash. Hashes establish integrity, not authenticated provenance or a reviewed experimental freeze.
