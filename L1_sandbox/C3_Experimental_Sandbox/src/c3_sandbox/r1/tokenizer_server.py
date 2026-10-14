"""Local tokenizer-only service; executable directly in the MLX environment."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import inspect
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.metadata import version
from pathlib import Path

HOST = "127.0.0.1"
PORT = 18081
MAX_BODY_BYTES = 4 * 1024 * 1024
REVISION = "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"


class TokenizerEngine:
    def __init__(self, model_path: Path, *, revision: str = REVISION,
                 expected_parser: str | None = None):
        # Import only at startup: unit tests need neither MLX nor a model.
        from mlx_lm.server import process_message_content
        # utils.load_tokenizer delegates to this local-file loader. The public
        # function is named `load` in tokenizer_utils (MLX-LM 0.32.0).
        from mlx_lm.tokenizer_utils import load as load_tokenizer

        model_path = model_path.resolve(strict=True)
        template = (model_path / "chat_template.jinja").read_text(encoding="utf-8")
        self.tokenizer = load_tokenizer(model_path, tokenizer_config_extra={
            "local_files_only": True, "trust_remote_code": False,
        })
        if self.tokenizer.chat_template != template:
            raise ValueError("Loaded template differs from pinned chat_template.jinja")
        self.normalize = process_message_content
        parser = self.tokenizer.tool_parser
        parser_name = f"{parser.__module__}.{parser.__name__}" if parser else None
        parser_source = inspect.getsourcefile(parser) if parser else None
        parser_hash = hashlib.sha256(Path(parser_source).read_bytes()).hexdigest() if parser_source else None
        if expected_parser is not None:
            expected_name = (f"mlx_lm.tool_parsers.{expected_parser}.parse_tool_call"
                             if "." not in expected_parser else expected_parser)
            if parser_name != expected_name:
                raise ValueError("Actual inferred tool parser differs from expected parser")
            if expected_name == "mlx_lm.tool_parsers.qwen3_coder.parse_tool_call":
                if parser_hash != "2d212e4e6f175178bd80b7db9963ccad613e29419f976e6d6b3272417f953722":
                    raise ValueError("Actual qwen3_coder parser source differs from pinned source")
        files = ("chat_template.jinja", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
                 "config.json", "generation_config.json")
        hashes = {name: hashlib.sha256((model_path / name).read_bytes()).hexdigest()
                  for name in files if (model_path / name).is_file()}
        identity = {"revision": revision, "file_sha256": hashes,
                    "mlx_lm_version": version("mlx-lm"),
                    "transformers_version": version("transformers"),
                    "chat_template_args": {}}
        # Preserve legacy fingerprint shape unless a new system was explicitly requested.
        if revision != REVISION or expected_parser is not None:
            identity.update(tool_parser=parser_name, tool_parser_sha256=parser_hash)
        self.fingerprint = {**identity, "sha256": hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}

    def count(self, route: str, body: dict) -> dict:
        if type(body) is not dict:
            raise ValueError("Expected an object")
        if route == "/text":
            if set(body) != {"text"} or not isinstance(body["text"], str):
                raise ValueError("Expected text")
            # Same tokenizer.encode default as the MLX completion path.
            ids = self.tokenizer.encode(body["text"])
        elif route == "/count":
            if set(body) != {"messages", "tools"} or not isinstance(body["messages"], list):
                raise ValueError("Expected messages and tools")
            if not body["messages"] or not all(isinstance(m, dict) for m in body["messages"]):
                raise ValueError("Expected nonempty messages")
            if body["tools"] is not None and not isinstance(body["tools"], list):
                raise ValueError("Expected tools list or null")
            messages = copy.deepcopy(body["messages"])
            self.normalize(messages)
            ids = self.tokenizer.apply_chat_template(
                messages, tools=body["tools"], tokenize=True, add_generation_prompt=True)
        else:
            raise ValueError("Unknown route")
        # Stable hash convention: compact JSON list of integer token IDs, UTF-8.
        digest = hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()
        return {"token_count": len(ids), "prompt_token_id_sha256": digest,
                "tokenizer_fingerprint": self.fingerprint}


def make_handler(engine):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass  # No prompts, paths, tool results, or error details in logs.

        def reply(self, status, payload):
            data = json.dumps(payload, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            self.connection.settimeout(10)
            if self.path not in ("/count", "/text"):
                self.reply(404, {"error": "unknown_endpoint"})
                return
            if self.headers.get("Transfer-Encoding"):
                self.reply(400, {"error": "transfer_encoding_unsupported"})
                return
            try:
                length = int(self.headers.get("Content-Length", "-1"))
                if length < 0:
                    self.reply(411, {"error": "content_length_required"})
                    return
                if length > MAX_BODY_BYTES:
                    self.reply(413, {"error": "body_too_large"})
                    return
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("Incomplete body")
                result = engine.count(self.path, json.loads(raw))
            except Exception:
                self.reply(400, {"error": "invalid_tokenizer_request"})
                return
            self.reply(200, result)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--expected-parser")
    args = parser.parse_args()
    engine = TokenizerEngine(args.model_path, revision=args.revision, expected_parser=args.expected_parser)
    with HTTPServer((HOST, PORT), make_handler(engine)) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
