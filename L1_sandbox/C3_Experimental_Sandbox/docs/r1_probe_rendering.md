# R1 D4 observation rendering

`c3_sandbox.r1.probes` implements the frozen public observation vocabulary from
`configs/probes.v1.json`. It contains no hypothesis, diagnosis or D6–D8 logic.

```python
pair = render_pair("visibility.expanded", public_evidence,
                   token_counter=runtime_token_counter)
delivered = pair[probe_id].rendered_text
audit = pair[probe_id].to_dict()
```

`render_probe(probe_id, public_evidence, *, token_counter=None)` returns a
`RenderedProbe`. `render_pair(target_id, public_evidence, *, token_counter=None)`
returns a dictionary keyed by the target and its registered matched control.
The exact evidence schema is `{inventory: {...}, feedback: {...} | None}`.
Inventory requires `file_count`, `extension_counts`, `total_bytes`, `docx_count`,
`paragraph_count`, `table_count`, `section_count`, `header_count`, `omml_count`,
and `xml_parse_error_count`. Feedback requires `exit_code`, `stdout_bytes`,
`stderr_bytes`, `files_changed`, and `parse_error_count`. Counts are nonnegative
integers; the exit code may be signed. Booleans, extra fields, nested values and
free text are rejected. Extension keys are restricted to the module's fixed
`EXTENSIONS`; producers must group unknown suffixes as `other` and suffix-free
files as `none`. No path, object index, text, expected value or rubric can enter
this renderer.

Expanded visibility shows global counts in separate workspace, document and XML
sections. Compact visibility shows the same count inventory in one collapsed
view. Structured feedback exposes separate public execution fields. Aggregate
feedback combines output byte counts and reports zero/nonzero exit status with
global change and parse counts. Absent feedback is explicitly unreported.
Matched observations use the same field layout with neutral `unreported` values;
they do not expose the measured values. These templates contain no patch advice
or task-specific target information. Statistics can still be relevant to a task;
the closed schema does not replace per-task leakage review.

Original renders the empty string: the runtime must add no extra message for that
arm. Other observations begin with an explicit runtime-owned advisory marker.
The runtime inserts that advisory immediately after the unmodified checkpoint
prefix, at the same experiment position for each arm. It must not manufacture a
model tool call or alter earlier messages. The renderer does not perform message
insertion, grant or remove tools, change sampling, or change execution budgets.

Paired matching retains the full observation and adds explicitly labelled neutral
content filler to both arms. It checks equal UTF-8 byte length and searches at
most 1,026 deterministic filler variants per arm for identical counts under the
supplied tokenizer. Each of two families varies the number of literal ` x` pairs
from zero to 512; the remaining bytes are repeated `0` digits or `n` letters.
These neutral patterns provide different token densities across vocabularies.
The search stops at its first shared exact count, records the selected family
and number of candidates evaluated, and retains every observation byte.
At most two additional tokenizer calls count the initial unpadded observations. A failed match raises `ValueError`; there is no fallback to
whitespace padding, truncation or approximate token claims. Without a tokenizer,
`token_count` and `token_counts_equal` are `None`, and tokenizer verification is
false. Matching consumes the existing context budget and may affect behavior;
controls are never assumed inert. Passing a standalone `render_probe` result does
not establish pair matching. Production must supply the runtime's exact tokenizer
and account for message-envelope costs separately.

Every result retains a detached validated public evidence snapshot, its SHA-256
digest, registry payload, delivered text, text digest, token count and matching
audit. The runtime must separately log the actual raw tool result and this
delivered observation, bind both to task/checkpoint/branch identity, and retain
the unchanged checkpoint prefix. The public digest binds the sanitized input,
not arbitrary raw tool output. No raw private field should be copied into the
model advisory. All resource tools remain available; solvability and resource
preservation require separate task-bound review and execution audits. Synthetic
renderer tests certify neither real task eligibility nor scientific claim gates.
