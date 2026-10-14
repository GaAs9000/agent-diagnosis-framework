"""Tokenizer bridge contracts without loading a model or calling generation."""
import asyncio
import ast
import copy
import hashlib
import json
import threading
from pathlib import Path
from http.server import HTTPServer

import pytest

from c3_sandbox.r1.tokenizer_client import TokenizerClient
from c3_sandbox.r1.tokenizer_server import TokenizerEngine, make_handler


def test_installed_mlx_startup_helpers_exist_without_metal_import():
    """Check actual installed source exports without importing Metal/MLX."""
    project = Path(__file__).resolve().parents[1]
    packages = list((project / ".model-venv" / "lib").glob("python*/site-packages/mlx_lm"))
    if not packages:
        pytest.skip("Local MLX environment not installed")
    package = packages[0]
    source = ast.parse((project / "src/c3_sandbox/r1/tokenizer_server.py").read_text())
    imports = [node for node in ast.walk(source) if isinstance(node, ast.ImportFrom)
               and node.module in {"mlx_lm.tokenizer_utils", "mlx_lm.server"}]
    assert len(imports) == 2
    for statement in imports:
        installed = ast.parse((package / (statement.module.rsplit(".", 1)[1] + ".py")).read_text())
        functions = {node.name: node for node in installed.body
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for imported in statement.names:
            assert imported.name in functions, f"Missing installed MLX helper: {imported.name}"
            if imported.asname == "load_tokenizer":
                parameters = [arg.arg for arg in functions[imported.name].args.args]
                assert parameters[:2] == ["model_path", "tokenizer_config_extra"]


def test_inspect_conversion_matches_native_provider(monkeypatch):
    client = TokenizerClient()
    captured = {}
    messages = [
        {"role": "system", "content": "Whole history"},
        {"role": "user", "content": "中文"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function", "function": "execute",
             "arguments": {"command": "echo hi"}}]},
        {"role": "tool", "tool_call_id": "call_1", "function": "execute", "content": "hi"},
    ]
    tools = [{"type": "function", "function": {"name": "execute", "description": "Run",
              "parameters": {"type": "object", "properties": {
                  "command": {"type": "string", "minLength": 1}}, "required": ["command"]}}}]
    original = copy.deepcopy((messages, tools))

    def post(route, body):
        captured.update(route=route, body=body)
        return {"token_count": 123}

    monkeypatch.setattr(client, "_post", post)
    assert asyncio.run(client.count_messages(messages, tools)) == 123
    body = captured["body"]
    assert captured["route"] == "/count"
    assert len(body["messages"]) == 4
    call = body["messages"][2]["tool_calls"][0]
    assert call["function"]["name"] == "execute"
    assert json.loads(call["function"]["arguments"]) == {"command": "echo hi"}
    assert body["messages"][3]["tool_call_id"] == "call_1"
    definition = body["tools"][0]["function"]
    assert definition["strict"] is False
    assert "minLength" not in definition["parameters"]["properties"]["command"]
    assert (messages, tools) == original


def test_empty_tools_are_omitted_as_null(monkeypatch):
    client = TokenizerClient()
    def post(route, body):
        assert body["tools"] is None
        return {"token_count": 1}
    monkeypatch.setattr(client, "_post", post)
    assert asyncio.run(client.count_messages([{"role": "user", "content": "x"}], [])) == 1


def test_text_count_and_private_endpoint(monkeypatch):
    client = TokenizerClient()
    def post(route, body):
        assert (route, body) == ("/text", {"text": "中文"})
        return {"token_count": 2}
    monkeypatch.setattr(client, "_post", post)
    assert client.text_count("中文") == 2
    for endpoint in ("https://example.com", "http://localhost:18081", "http://127.0.0.1:18080"):
        with pytest.raises(ValueError):
            TokenizerClient(endpoint)


def test_no_body_truncation():
    with pytest.raises(ValueError, match="no truncation"):
        TokenizerClient().text_count("x" * (4 * 1024 * 1024))


@pytest.mark.parametrize("status,count", [(302, 2), (200, True), (200, -1)])
def test_client_rejects_redirects_and_invalid_counts(monkeypatch, status, count):
    import c3_sandbox.r1.tokenizer_client as module
    class Response:
        def read(self, limit):
            return json.dumps({"token_count": count, "prompt_token_id_sha256": "fake",
                               "tokenizer_fingerprint": {}}).encode()
    class Connection:
        def __init__(self, host, port, timeout):
            assert (host, port) == ("127.0.0.1", 18081)
            self.closed = False
        def request(self, method, route, body, headers):
            assert method == "POST"
        def getresponse(self):
            response = Response()
            response.status = status
            return response
        def close(self):
            self.closed = True
    monkeypatch.setattr(module.http.client, "HTTPConnection", Connection)
    with pytest.raises(RuntimeError):
        TokenizerClient().text_count("x")


def test_engine_hash_and_template_flags_without_loading():
    engine = TokenizerEngine.__new__(TokenizerEngine)
    class StubTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {"tools": None, "tokenize": True, "add_generation_prompt": True}
            assert messages == [{"role": "user", "content": "normalized"}]
            return [1, 2, 3]
        def encode(self, text):
            assert text == "raw"
            return [4, 5]
    engine.tokenizer = StubTokenizer()
    engine.fingerprint = {"sha256": "fake"}
    engine.normalize = lambda messages: messages[0].update(content="normalized")
    body = {"messages": [{"role": "user", "content": "original"}], "tools": None}
    result = engine.count("/count", body)
    assert result["token_count"] == 3
    assert result["prompt_token_id_sha256"] == hashlib.sha256(b"[1,2,3]").hexdigest()
    assert body["messages"][0]["content"] == "original"
    assert engine.count("/text", {"text": "raw"})["token_count"] == 2


def test_http_contract_with_stub_engine():
    import http.client
    class StubEngine:
        def count(self, route, body):
            return {"token_count": 2, "prompt_token_id_sha256": "fake",
                    "tokenizer_fingerprint": {"sha256": "fake"}}
    server = HTTPServer(("127.0.0.1", 0), make_handler(StubEngine()))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for route, payload, length, expected in (
            ("/text", b'{"text":"x"}', None, 200),
            ("/unknown", b"{}", None, 404),
            ("/count", b"not-json", None, 400),
            ("/text", b"", str(4 * 1024 * 1024 + 1), 413),
        ):
            connection = http.client.HTTPConnection(*server.server_address)
            headers = {"Content-Length": length} if length else {}
            connection.request("POST", route, payload, headers)
            response = connection.getresponse()
            assert response.status == expected
            response.read()
            connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
