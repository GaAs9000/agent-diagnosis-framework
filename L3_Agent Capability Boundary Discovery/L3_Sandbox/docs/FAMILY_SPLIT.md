# D3: frozen operational semantic families

The accepted D3 definition treats the existing 28 semantic groups as operational
families. The canonical manifest is
`L1_sandbox/C3_Experimental_Sandbox/benchmarks/officeval/family_split.json`.
`family_split.draft.json` is retained unchanged as historical provenance.

All 91 locked tasks occur exactly once: nine development families contain 55
tasks, and 19 heldout families contain 36 tasks. No semantic group crosses the
split. Existing explored tasks and the budget pilot's semantic groups belong to
development. Sokoban remains development and is outside this OfficeVal manifest.

`family_id` is the existing `provisional_block_id`; no groups were merged or
renamed. `family_definition = semantic_operational_group` and
`family_verified = true` mean the accepted operational membership and split are
validated. They do **not** certify template lineage or statistical independence.
`lineage_status = unknown` is explicit, including for heldout families. The
manifest preserves the existing public metadata and asset hashes and binds to
`data/data.lock.json` by SHA-256 and dataset revision.

The current pilot uses five tasks across four development families:

| Task | Operational family |
| --- | --- |
| officeval_004 | word-template-formatting |
| officeval_010 | word-diagram-edit |
| officeval_050 | image-to-editable-ppt |
| officeval_085 | excel-chart-creation |
| officeval_095 | excel-chart-creation |

Both L3 example configurations reference this canonical manifest. Planning must
validate task membership, family ID, verification flag and split before native
loading or execution, and freeze the manifest's identity with the plan inputs.
Pilot plans reject heldout tasks; validation plans require heldout tasks. Future
baseline plans must reference the same partition rather than create another
split. The 45-slot budget pilot, 1,695 logical request maximum, and 6,780 HTTP
attempt hard cap are unchanged.

The heldout task pool includes 19 PowerPoint tasks and only three Excel tasks.
Report pooled validation with its family composition; this pool does not support
strong software-specific claims. Development runs can inform implementation and
configuration choices before heldout sampling begins.
