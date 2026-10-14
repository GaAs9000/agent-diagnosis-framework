"""Native Gemini generateContent adapter with deterministic evidence records.

The adapter uses Gemini's function calling wire format directly.  It is
transport-injectable so protocol tests never need network access.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping
from urllib import error as urlerror
from urllib import request as urlrequest

MODEL = "gemini-3.5-flash-lite"
ENDPOINT_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _schema_to_gemini(tool_schema: list[Mapping[str, Any]] | Mapping[str, Any]) -> list[dict[str, Any]]:
    """Convert OpenAI-like tools to Gemini functionDeclarations."""
    def schema(value: Any) -> Any:
        # Gemini's Schema protobuf is close to JSON Schema, but it rejects
        # OpenAI-only keywords such as additionalProperties.  Convert
        # recursively so nested object parameters obey the same contract.
        if isinstance(value, Mapping):
            return {
                str(key): schema(item)
                for key, item in value.items()
                if key not in {"additionalProperties", "$schema"}
            }
        if isinstance(value, list):
            return [schema(item) for item in value]
        return value

    tools = [tool_schema] if isinstance(tool_schema, Mapping) else list(tool_schema or [])
    declarations = []
    for tool in tools:
        fn = tool.get("function", tool) if isinstance(tool, Mapping) else {}
        if not isinstance(fn, Mapping) or not fn.get("name"):
            raise ValueError("malformed tool schema")
        item = {"name": str(fn["name"])}
        if fn.get("description") is not None:
            item["description"] = str(fn["description"])
        params = fn.get("parameters", fn.get("parametersJsonSchema"))
        if params is not None:
            item["parameters"] = schema(params)
        declarations.append(item)
    return declarations


def _parts_from_message(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    role = message.get("role", "user")
    content = message.get("content", "")
    if isinstance(content, list):
        text = "".join(str(x.get("text", "")) if isinstance(x, Mapping) else str(x) for x in content)
    else:
        text = str(content)
    return [{"role": "model" if role in {"assistant", "model"} else "user", "parts": [{"text": text}]}]


@dataclass(frozen=True)
class GeminiToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    part_index: int


@dataclass
class GeminiRecord:
    request_raw: bytes
    response_raw: bytes
    usage: dict[str, Any] | None
    evidence_ids: tuple[str, ...] = ()
    request_identity: str = ""
    response_identity: str = ""
    selector_invalid: bool = False
    error: str | None = None
    evidence_identity: str = ""

    def __post_init__(self):
        self.request_identity = self.request_identity or hashlib.sha256(self.request_raw).hexdigest()
        self.response_identity = self.response_identity or hashlib.sha256(self.response_raw).hexdigest()
        self.evidence_identity = self.evidence_identity or hashlib.sha256("|".join(self.evidence_ids).encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {"request_raw": self.request_raw.decode("utf-8", "replace"), "response_raw": self.response_raw.decode("utf-8", "replace"),
                "usage": self.usage, "evidence_ids": list(self.evidence_ids), "request_identity": self.request_identity,
                "response_identity": self.response_identity, "evidence_identity": self.evidence_identity,
                "selector_invalid": self.selector_invalid, "error": self.error}


class GeminiProtocolError(ValueError):
    pass


class GeminiNativeToolAgent:
    def __init__(self, *, transport: Callable[..., Any] | None = None, model: str = MODEL,
                 api_key: str | None = None, timeout: float = 60.0):
        if model != MODEL:
            raise ValueError(f"model is fixed to {MODEL}")
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.transport = transport or self._http_transport
        self.timeout = timeout
        self.records: list[GeminiRecord] = []

    def _http_transport(self, request_raw: bytes, *, api_key: str) -> bytes:
        req = urlrequest.Request(ENDPOINT_TEMPLATE.format(model=self.model), data=request_raw,
                                 headers={"Content-Type": "application/json", "x-goog-api-key": api_key}, method="POST")
        try:
            with urlrequest.urlopen(req, timeout=self.timeout) as response:
                return response.read()
        except urlerror.HTTPError as exc:
            # Preserve the provider's structured reason in the audit record;
            # never include request headers or the API key in the exception.
            body = exc.read(4096).decode("utf-8", "replace")
            raise RuntimeError(f"Gemini HTTP {exc.code}: {body}") from exc

    async def _send(self, raw: bytes) -> bytes:
        if not self.api_key:
            raise RuntimeError("Gemini API key is missing (set GEMINI_API_KEY or GOOGLE_API_KEY)")
        # Injected transports receive only bytes; the key is passed solely to
        # the private HTTP transport and therefore cannot enter evidence.
        if self.transport == self._http_transport:
            result = self.transport(raw, api_key=self.api_key)
        else:
            result = self.transport(raw)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, tuple):
            result = result[-1]
        if not isinstance(result, (bytes, bytearray)):
            raise TypeError("transport must return response bytes")
        return bytes(result)

    @staticmethod
    def _parse(raw: bytes, *, turn: int) -> tuple[list[GeminiToolCall], str, dict[str, Any] | None, bool, str | None, list[dict[str, Any]]]:
        try:
            obj = json.loads(raw)
            candidates = obj.get("candidates")
            if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], Mapping):
                raise GeminiProtocolError("missing candidates")
            parts = candidates[0].get("content", {}).get("parts", [])
            if not isinstance(parts, list):
                raise GeminiProtocolError("malformed parts")
            calls, text = [], []
            for i, part in enumerate(parts):
                if not isinstance(part, Mapping):
                    raise GeminiProtocolError("malformed part")
                if "text" in part: text.append(str(part["text"]))
                if "functionCall" in part:
                    fc = part["functionCall"]
                    if not isinstance(fc, Mapping) or not isinstance(fc.get("name"), str) or not isinstance(fc.get("args", {}), Mapping):
                        raise GeminiProtocolError("malformed functionCall")
                    stable = "gemini-call-" + hashlib.sha256(raw + f":{turn}:{i}".encode()).hexdigest()[:16]
                    calls.append(GeminiToolCall(stable, fc["name"], dict(fc.get("args", {})), i))
            usage = obj.get("usageMetadata")
            return calls, "".join(text), usage if isinstance(usage, Mapping) else None, False, None, parts
        except (json.JSONDecodeError, TypeError, KeyError, GeminiProtocolError) as exc:
            return [], "", None, True, str(exc), []

    async def run(self, *, system: str = "", user: str = "", messages: list[Mapping[str, Any]] | None = None,
                  tool_schema: list[Mapping[str, Any]] | Mapping[str, Any] = (), tool_callback: Callable[..., Any] | None = None,
                  max_turns: int = 8, evidence_ids: tuple[str, ...] = ()) -> dict[str, Any]:
        if max_turns < 1: raise ValueError("max_turns must be positive")
        declarations = _schema_to_gemini(tool_schema)
        known_tools = {d["name"] for d in declarations}
        contents = []
        system_instruction = system
        if messages is not None:
            for msg in messages:
                if msg.get("role") == "system":
                    system_instruction = str(msg.get("content", ""))
                else:
                    contents.extend(_parts_from_message(msg))
        else:
            contents.append({"role": "user", "parts": [{"text": user}]})
        for turn in range(max_turns):
            payload = {"contents": contents}
            if system_instruction:
                payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
            if declarations: payload["tools"] = [{"functionDeclarations": declarations}]
            request_raw = _json_bytes(payload)
            try: response_raw = await self._send(request_raw)
            except Exception as exc:
                self.records.append(GeminiRecord(request_raw, b"", None, evidence_ids, error=str(exc)))
                raise
            calls, text, usage, invalid, err, raw_parts = self._parse(response_raw, turn=turn)
            rec = GeminiRecord(request_raw, response_raw, usage, evidence_ids, selector_invalid=invalid, error=err)
            self.records.append(rec)
            if invalid: return {"text": "", "tool_calls": [], "selector_invalid": True, "stopped_reason": "protocol_error", "error": err, "record": rec}
            if any(c.name not in known_tools for c in calls):
                rec.selector_invalid = True
                rec.error = "unknown function call"
                return {"text": "", "tool_calls": calls, "selector_invalid": True, "stopped_reason": "protocol_error", "record": rec}
            if not calls: return {"text": text, "tool_calls": [], "selector_invalid": False, "stopped_reason": "assistant_finished", "record": rec}
            if tool_callback is None: return {"text": text, "tool_calls": calls, "selector_invalid": False, "stopped_reason": "assistant_finished", "record": rec}
            # Preserve model function call turn before functionResponse continuation.
            contents.append({"role": "model", "parts": raw_parts if raw_parts else [{"functionCall": {"name": c.name, "args": c.arguments}} for c in calls]})
            responses = []
            for call in calls:
                try:
                    value = tool_callback(call.name, call.arguments)
                    if inspect.isawaitable(value): value = await value
                    responses.append({"functionResponse": {"name": call.name, "response": value if isinstance(value, Mapping) else {"result": value}}})
                except Exception as exc:
                    return {"text": text, "tool_calls": calls, "selector_invalid": True, "stopped_reason": "tool_execution_error", "error": f"tool call failed: {exc}", "record": rec}
            contents.append({"role": "user", "parts": responses})
        return {"text": "", "tool_calls": [], "selector_invalid": False, "stopped_reason": "max_turns", "error": "max_turns exceeded", "record": self.records[-1]}

    def run_sync(self, **kwargs):
        return asyncio.run(self.run(**kwargs))
