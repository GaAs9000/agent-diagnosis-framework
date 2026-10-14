# R1 local tokenizer bridge

From `C3_Experimental_Sandbox`, start the tokenizer-only process:

```sh
.model-venv/bin/python src/c3_sandbox/r1/tokenizer_server.py --model-path models/Qwen3-4B-Instruct-2507-4bit
```

The model directory must contain the externally pinned revision
`50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`. The service reads tokenizer files only,
uses `local_files_only=True` and `trust_remote_code=False`, and verifies the loaded
template matches the local `chat_template.jinja`. It loads no model weights and
performs no generation. Revision provenance belongs to the download manifest;
the response fingerprints actual local file contents and installed library versions.
The loader is `mlx_lm.tokenizer_utils.load`, the same local-file implementation
called by `mlx_lm.utils.load_tokenizer`; it avoids the latter's download resolver.

The service binds only `127.0.0.1:18081`. POST `/count` accepts
`{"messages": [OpenAI-compatible messages], "tools": [OpenAI tools]}` (tools may
be null). It applies MLX's installed `process_message_content`, then its tokenizer
wrapper's `apply_chat_template` with `tools`, `tokenize=True`, and
`add_generation_prompt=True`. No extra chat-template overrides are supported;
the generation server must likewise use its default empty `chat_template_args`.
POST `/text` accepts `{"text": "..."}` and calls the same tokenizer's `encode`
with default special-token behavior, matching the MLX completion path.

Both return `token_count`, `prompt_token_id_sha256`, and `tokenizer_fingerprint`.
The token-ID digest hashes the UTF-8 compact JSON array of integer IDs. The
fingerprint contains tokenizer/template/config file hashes, the declared pinned
revision, MLX/Transformers versions, default template arguments, and a digest of
that identity. Capture these via the client's `*_details` methods when recording
run provenance. The service suppresses request/error logs, requires a bounded
Content-Length (4 MiB), rejects chunked bodies, and emits generic error responses.

In the controller `.venv`:

```python
from c3_sandbox.r1.tokenizer_client import TokenizerClient

tokenizer = TokenizerClient()
config["token_count"] = tokenizer.count_messages  # async NativeAgent callback
# Pass tokenizer.text_count as render_pair's synchronous text counter.
```

`count_messages` accepts Inspect-native message dictionaries and the same tool
schemas as `NativeAgent.run`. It validates through Inspect's `ChatMessage` adapter
and reuses the local OpenAI-compatible provider's async `messages_to_openai` and
`tools_to_openai`, including schema exclusions and `strict=False`. Provider
construction makes no model call. HTTP requests bypass ambient proxies and
redirects; alternate endpoints are rejected. Each call uses an HTTP connection,
never a subprocess. Counts cover the entire supplied history and tools. Neither
client nor service truncates messages or enforces a context budget: the native
agent retains its existing whole-prompt budget check.

Focused validation (stub HTTP/pure conversion; no tokenizer model loaded):

```sh
.venv/bin/python -m pytest -q tests/test_r1_tokenizer_client.py
```
