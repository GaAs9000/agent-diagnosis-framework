# R1 public evidence collection

`c3_sandbox.r1.public_evidence.collect_public_evidence(actor, *, checkpoint=None,
preworkspace=None, last_tool_result=None)` performs a single serial controller
`actor.exec_fixed` call. It sends fixed trusted Python source to
`python -I -B -c` inside the active actor container and the prior-state hash map
as JSON on stdin (avoiding argument-size limits for large workspaces). It creates no script,
extracted archive, cache, subprocess, or temporary file. It reads only the
workspace and uses standard-library XML and ZIP parsers; it imports no code
from the agent workspace. The caller owns runtime creation, restore,
serialization, and provenance.

The return value is exactly the closed schema accepted by
`r1.probes.validate_public_evidence`: `inventory` and `feedback`. No filename,
path, document text, XML path, answer, rubric, diagnostic exception, or free
string is returned. Unknown suffixes become `other`; extensionless files become
`none`. Extension categories use the renderer's fixed allowlist.

Inventory includes all regular workspace files, bytes, and DOCX file counts.
OOXML structural counts include `w:p`, `w:tbl`, `w:sectPr`, and `m:oMath` in
`word/document.xml` and top-level `word/header*.xml` parts. Paragraphs in tables
and headers count. `header_count` counts valid header parts, including
unreferenced parts; it does not count rendered pages. Other XML parts are not
parsed. This is structural inventory, not task scoring or format verification.

Each rejected DOCX contributes one `xml_parse_error_count`, and none of its
partial structural counts. ZIP limits are 2,048 members, 32 MiB per member and total uncompressed
bytes, and a maximum 200:1 compression ratio per member. The 8 MiB XML limit
applies only to parsed document/header parts. Unparsed embedded fonts and
other assets may exceed 8 MiB while remaining within the ZIP bounds. Unsafe/absolute/traversal/backslash/drive-style paths, duplicate names,
special ZIP file types, encrypted members, XML DTD/entity declarations, invalid
XML, missing document parts, and wrong root namespaces are rejected. No archive
members are extracted. Workspace limits match runtime snapshots (20,000 entries
and 256 MiB). Symlinks, hardlinks, and special workspace entries abort collection
without disclosing their paths.

For feedback, supply either a validated `DockerRuntime` snapshot as `checkpoint`,
or a previously exported workspace directory as `preworkspace`, plus the actual
previous raw runtime result as `last_tool_result`. Integer `exit_code`,
`stdout_bytes`, and `stderr_bytes` are required; captured text length is never
substituted for actual byte counts, including truncated output. Feedback is
`None` if either previous state or result is absent. `files_changed` counts
added, removed, content-changed, or mode-changed regular files using SHA-256 and
modes; directory-only changes do not count. `parse_error_count` is the current
inspection's rejected-DOCX count, not an inferred prior stderr exception count.
Before-state paths/hashes are controller-private inputs, never returned facts.

```python
public = collect_public_evidence(restored_actor)
public_with_feedback = collect_public_evidence(
    restored_actor,
    preworkspace=before_export / "workspace",
    last_tool_result=previous_raw_runtime_result,
)
```

Focused tests execute the trusted script using a fake actor against temporary
fixtures. They exercise actual DOCX counts, schema closure, byte feedback, file
differences, unsafe archives/filesystem objects, and workspace snapshot
invariance. They do not establish Docker isolation or physical runtime
invariance. Before qualifying a real replay, the root runner must compare
workspace snapshots and runtime boundary audits before/after collection
(including `/tmp` and residual processes), retain the actual execution record,
and verify results came from the restored actor. Reads may update filesystem
access times; snapshots do not model atime. This collector supplies no human
review approval or scientific admission.
