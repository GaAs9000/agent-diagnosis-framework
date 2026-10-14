# R1.1 native tool protocol admission

This adapter is engineering admission only. Synthetic fixtures do not establish real
LLM tool calling, task success, fresh continuation, or G0–G4 acceptance. Existing
003 artifacts and all earlier runs remain untouched.

## Installed behavior checked

The pinned local sources are:

- `.venv/lib/python3.12/site-packages/inspect_ai/model/_providers/openai_compatible.py`:
  `completion.model_dump()` is saved to `ModelCall` before choice conversion.
- `inspect_ai/model/_openai.py`: OpenAI IDs are passed into native Inspect calls;
  `length` becomes Inspect `max_tokens`.
- `inspect_ai/model/_call_tools.py`: the installed parser may accept YAML or recover
  JSON with trailing quotes. Admission therefore verifies original wire arguments
  with strict JSON before allowing the resulting Inspect arguments to dispatch.
- `.model-venv/lib/python3.12/site-packages/mlx_lm/server.py`: the formatter skips
  parse failures; `made_tool_call` can still set the completion reason to
  `tool_calls`. No missing call is reconstructed. MLX itself may assign UUID IDs
  while formatting a parsed call; the adapter accepts only IDs present in the
  actual provider response, and never fabricates IDs.
- Installed OpenAI 3 uses `httpx2`; fixtures use its `MockTransport` rather than
  the incompatible legacy `httpx` transport.

## Evidence and immutable identity

Pass the controller's already verified identities to `NativeAgent`:

```python
agent = NativeAgent({
    "seed": 2026,
    "token_count": tokenizer.count_messages,
    "identity_context": {
        "model_fingerprint": fingerprint,
        "tokenizer_fingerprint": fingerprint["tokenizer"],
        "provenance": {"source_snapshot": snapshot_sha256},
    },
}, new_run_dir)
```

The context must be JSON serializable and contain both fingerprint keys. It is
frozen independently of caller dictionaries at initialization, hashed, and included
on every request. No new fingerprint HTTP request is made by the agent. Legacy
callers may omit it, in which case the request explicitly records `not_supplied`
and null fingerprints; such a request is insufficient for R1.1 identity acceptance.
The controller must supply actual verified fingerprints for real G0 evidence.
Sampling and endpoint restrictions are unchanged.

`model_calls.jsonl` is append-only and contains controller requests, exact HTTP
request and response bodies encoded in base64 (`wire_response`), SHA-256 of the
response bytes, the existing lossless SDK `provider_call`, native Inspect output,
and exceptions. Headers are not recorded. `protocol_audit.jsonl` contains parser
status and output, native dispatch IDs, dispatch arguments, and observations linked
by turn/call ID and event index to the full tool result in `events.jsonl`.

Malformed completions record `protocol_invalid`; legacy `stopped_reason` values
remain compatible. Provider errors/timeouts record `not_evaluated` rather than
being claimed valid. This status is separate from independent task scoring. Entire
batches are checked before the first dispatch; `length`/`max_tokens` rejects even
otherwise executable calls. Restored result IDs/functions must exactly match calls
and every call must have a result before generation can proceed.

## Controlled fixture interface

```python
result = await run_protocol_fixture(
    scripted_openai_response_bodies, new_fixture_dir,
    messages=public_messages,
    tools={"workspace_python": lambda code: runtime.exec_python(code)},
    tool_schemas=tool_schemas,
)
```

`run_protocol_fixture` lives in `c3_sandbox.r1.tool_protocol`. It routes scripted
HTTP responses through the actual installed OpenAI SDK, Inspect provider, and
`NativeAgent`; it cannot contact the model endpoint. Caller-provided tool callbacks
may operate an independently qualified Docker container. All request evidence and
the returned result are labeled `synthetic_protocol_fixture`, with
`result.is_synthetic=True`, even when the tool is real Docker. The fixed synthetic
counter is not a real tokenizer. Supply actual runtime identities separately if
needed; synthetic model evidence remains synthetic.

Focused regression command (no Docker, LLM, or network):

```sh
.venv/bin/python -m pytest -q tests/test_r11_tool_protocol.py tests/test_r1_agent.py tests/test_r1_native_agent.py
```

Coverage includes valid multi-turn call/observation/final text; pure text; tool finish
with empty payload; invalid JSON; missing IDs; mismatched result IDs before any
request; truncation with complete call payload; duplicate IDs; unknown functions;
invalid schema; YAML/trailing-quote recovery; duplicate JSON keys; nonfinite JSON;
immutable identity; and lossless wire/SDK evidence reconciliation. Invalid fixtures
assert zero dispatch and no checkpoint. Real qualification must be run separately
by the controller under the fixed model/configuration in a new development run.

## Dispatch snapshot hook

`NativeAgent.run(..., dispatch_callback=callback)` and the fixture runner accept a
sync or async callback. Each validated call emits `stage="start"` before tool
execution and `stage="end"` after the result is retained. Payload fields are
`turn`, `tool_call_id`, `function`, `arguments`, `raw_result`, and `error`. Start
has null result/error. A callback exception records `dispatch_callback_error`,
stops the loop, and prevents further dispatch/checkpoints. A failing start does
not invoke that tool; a failing end preserves the tool's actual result.
`result.tool_dispatch` lists only invoked tools with their actual results/errors.
