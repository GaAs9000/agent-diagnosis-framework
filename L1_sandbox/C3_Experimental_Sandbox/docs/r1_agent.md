# R1 native Inspect agent

`NativeAgent(config: dict, run_dir: Path).run(messages, tools, tool_schemas, *,
max_turns, checkpoint_callback=None)` is async. Tools are keyword-argument Python
callables; tools and callbacks can return values directly or awaitables. Schemas
are OpenAI function definitions or Inspect ToolInfo-shaped dictionaries. Argument
objects are validated with JSON Schema before any tool in a response executes.
Specify `additionalProperties: false` where extra arguments must be rejected.

Config requires `token_count(messages, tool_schemas) -> int`, using the deployed
model's tokenizer and exact chat template/tool representation. It must count the
entire prompt. The agent refuses a request when prompt + 2048 exceeds 16384;
it never truncates history. Optional `seed` is supplied by the controller.
Optional `tool_result_transform(name, arguments, raw_result)` returns the content
delivered to the model; raw output is independently retained. Tool results should
be JSON-serializable objects or strings. A transformation error becomes an explicit
tool error. No credentials are read from host environment by this module.

The fixed provider is Inspect `openai-api/mlx/default_model` at
`http://127.0.0.1:18080/v1`, with explicit `local-unused` API key, native tools,
Chat Completions, no streaming, and no strict-provider schema rewriting. Generation
uses temperature 0, max_tokens 2048, max_connections 1, top_k 1, min_p 0, no retries
or cached generation, and requests serial tool calls. The deployment controller
must verify MLX 0.32.0 and `mlx-community/Qwen3-4B-Instruct-2507-4bit` revision
`50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b` externally.

`generation_timeout_seconds` defaults to 120 and must be positive and at most 180.
The outer deadline covers Inspect semaphore and dispatch waits as well as HTTP;
Inspect attempt and retry-window timeouts are also set. Both Inspect and the
OpenAI-compatible SDK have `max_retries=0`, so the agent does not retry implicitly.
On expiry, the loop records `model_timeout`, code locations from the coroutine
await chain (file basename, function, line), and whether cancellation completed
within one second. It never records frame locals or source lines. Standalone
Inspect generation does not require an eval context. Inspect's connection pool
and hard-pause dispatch gate can block before any HTTP request, so the retained
await chain is needed to distinguish these waits from transport stalls.

The agent explicitly supplies OpenAI SDK `DefaultAsyncHttpxClient` with
`trust_env=False` and the bounded timeout to Inspect's supported `http_client`
provider argument. This bypasses environment and system proxy discovery for the
fixed loopback endpoint while preserving native Inspect request hooks. The client
is closed on completion or failure. No process environment or global proxy setting
is changed; request evidence records this client configuration.

Restored prefixes use Inspect's native ChatMessage dictionary representation,
including assistant `tool_calls` with `id`, `function`, and dictionary `arguments`,
and tool messages with `tool_call_id` and `function`. TypeAdapter validates the
prefix, which stays byte-equivalent at dictionary/value level in returned history.
Pending, duplicate, or incorrectly bound calls cannot be restored. Multiple calls
from one response run serially after the full batch validates. A checkpoint callback
receives a deep copy of complete history and the last event index after each fully
resolved turn. No checkpoint contains pending tool calls.

`model_calls.jsonl` appends complete pre-request messages, schemas, generation and
provider configuration, raw Inspect ModelOutput JSON (including native calls), usage,
and provider errors. It also records `provider_call` entries from Inspect 0.3.278's
model-event sink hook: SDK request/response bodies before transcript condensation
and native-message conversion, with error/time fields and no HTTP headers. The
hook forwards pending/completed events to an enclosing sink or the normal Inspect
transcript. Generation still uses ordinary Inspect `Model.generate`; no tool-call
parser or provider bypass is introduced. These SDK bodies are distinct from a
byte-for-byte server wire capture and cannot recover evidence absent from older runs.
`events.jsonl` appends indexed model and tool events, raw versus
delivered results, protocol errors and budget failures. These are Inspect-level
records; the controller must capture server wire logs separately if required.
Usage reports real Inspect output tokens; full prompt is input + cache_read +
cache_write. Missing cost is null. The return dictionary contains `messages`,
`events`, `modelcall_count`, `tokens`, and `stopped_reason`. A response without native
calls stops with `assistant_finished`, which is not a success grade. Invalid calls
stop explicitly with `invalid_protocol`; raw invalid output remains in the log and
is not committed to the replayable conversation. Provider failure, invalid model
output, exhausted turns and context exhaustion have distinct stop reasons.
An assistant response with `stop_reason=tool_calls` but no native calls, or a final
response with no non-whitespace text and no native calls, stops with `protocol_error`.
Its output, stop reason, events and token usage remain in evidence; it executes no
fallback action and creates no completion checkpoint.

`tests/test_r1_agent.py` contains synthetic mocked protocol checks only. It does
not establish native qualification or scientific validity. Qualification should
run a real nonce-consuming tool through this same loop and inspect the native
response, bound result, and server evidence before benchmark use.
