"""Remote API model backend implementation for Gemini, OpenAI, and compatible APIs.

Enforces:
- Rule 1: Retains native wire protocols and adapts to unified ModelResponse.
- Rule 2: Fail-closed; no silent fallback on failure.
- Rule 3: Binds provider model identity and response IDs into experimental records.
- Rule 4: Records prompt/completion/cached tokens and estimated costs.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import time
from typing import Any, Callable, Mapping, Sequence
from urllib import error as urlerror
from urllib import request as urlrequest

from .backend import (
    BackendError,
    CostRecord,
    ModelBackend,
    ModelIdentity,
    ModelResponse,
    ModelToolCall,
    UnifiedMessage,
)

GEMINI_ENDPOINT_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
OPENAI_ENDPOINT_TEMPLATE = "https://api.openai.com/v1/chat/completions"


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _convert_schema_to_gemini(tool_schema: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-style tool schema to Gemini functionDeclarations, stripping unsupported keys."""
    def clean(val: Any) -> Any:
        if isinstance(val, Mapping):
            return {str(k): clean(v) for k, v in val.items() if k not in {"additionalProperties", "$schema"}}
        if isinstance(val, list):
            return [clean(v) for v in val]
        return val

    declarations = []
    for tool in tool_schema:
        fn = tool.get("function", tool) if isinstance(tool, Mapping) else {}
        name = fn.get("name")
        if not name:
            continue
        decl = {"name": str(name)}
        if fn.get("description"):
            decl["description"] = str(fn["description"])
        params = fn.get("parameters") or fn.get("parametersJsonSchema")
        if params:
            decl["parameters"] = clean(params)
        declarations.append(decl)
    return declarations


class ApiBackend(ModelBackend):
    """Remote API backend supporting Gemini and OpenAI-compatible endpoints."""

    def __init__(
        self,
        identity: ModelIdentity,
        config: Mapping[str, Any] | None = None,
        *,
        transport: Callable[..., Any] | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ):
        super().__init__(identity, config)
        self.transport = transport or self._http_transport
        self.timeout = timeout
        self._api_key = api_key or self._resolve_api_key()

    def _resolve_api_key(self) -> str | None:
        cfg = self._config
        prov = self.identity.provider.lower()
        if cfg.get("api_key"):
            return str(cfg["api_key"])
        if prov in {"gemini", "google"}:
            return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if prov in {"openai", "openai-compatible"}:
            return os.environ.get("OPENAI_API_KEY")
        if prov in {"anthropic"}:
            return os.environ.get("ANTHROPIC_API_KEY")
        return os.environ.get("MODEL_API_KEY")

    def check_availability(self) -> tuple[bool, str | None]:
        if not self._api_key:
            return False, f"API key missing for provider '{self.identity.provider}'"
        return True, None

    def _http_transport(self, url: str, headers: dict[str, str], request_raw: bytes) -> bytes:
        req = urlrequest.Request(url, data=request_raw, headers=headers, method="POST")
        try:
            with urlrequest.urlopen(req, timeout=self.timeout) as response:
                return response.read()
        except urlerror.HTTPError as exc:
            body = exc.read(4096).decode("utf-8", "replace")
            # Preserve provider's structured reason; never leak request headers or api_key.
            raise BackendError(f"HTTP {exc.code} from {self.identity.provider}: {body}") from exc
        except Exception as exc:
            raise BackendError(f"Network error calling {self.identity.provider}: {exc}") from exc

    async def _send_bytes(self, url: str, headers: dict[str, str], payload_raw: bytes) -> bytes:
        if self.transport == self._http_transport:
            result = self.transport(url, headers, payload_raw)
        else:
            # Injected transport receives raw bytes
            sig = inspect.signature(self.transport)
            if len(sig.parameters) >= 3:
                result = self.transport(url, headers, payload_raw)
            else:
                result = self.transport(payload_raw)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, tuple):
            result = result[-1]
        if not isinstance(result, (bytes, bytearray)):
            raise TypeError("Transport must return bytes")
        return bytes(result)

    async def generate(
        self,
        messages: Sequence[UnifiedMessage],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        sampling: Mapping[str, Any] | None = None,
    ) -> ModelResponse:
        ok, reason = self.check_availability()
        if not ok:
            raise BackendError(reason)

        prov = self.identity.provider.lower()
        start_time = time.perf_counter()

        if prov in {"gemini", "google"}:
            return await self._generate_gemini(messages, tools=tools, sampling=sampling, start_time=start_time)
        elif prov in {"openai", "openai-compatible"}:
            return await self._generate_openai(messages, tools=tools, sampling=sampling, start_time=start_time)
        else:
            raise BackendError(f"Unsupported API provider: {self.identity.provider}")

    async def _generate_gemini(
        self,
        messages: Sequence[UnifiedMessage],
        *,
        tools: Sequence[Mapping[str, Any]] | None,
        sampling: Mapping[str, Any] | None,
        start_time: float,
    ) -> ModelResponse:
        model = self.identity.model_name
        endpoint = self.identity.endpoint or GEMINI_ENDPOINT_TEMPLATE.format(model=model)
        headers = {"Content-Type": "application/json", "x-goog-api-key": self._api_key or ""}

        contents: list[dict[str, Any]] = []
        system_instruction: str | None = None

        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            elif msg.role in {"assistant", "model"}:
                # If native_parts exists (e.g. from previous Gemini response containing thoughtSignature), use verbatim
                if msg.native_parts:
                    contents.append({"role": "model", "parts": msg.native_parts})
                elif msg.tool_calls:
                    parts = [{"functionCall": {"name": c.name, "args": c.arguments}} for c in msg.tool_calls]
                    contents.append({"role": "model", "parts": parts})
                elif msg.content:
                    contents.append({"role": "model", "parts": [{"text": msg.content}]})
            elif msg.role in {"tool", "function"}:
                # Function response
                fn_name = msg.name or "workspace_python"
                try:
                    resp_obj = json.loads(msg.content)
                except Exception:
                    resp_obj = {"result": msg.content}
                fn_part = {"functionResponse": {"name": fn_name, "response": resp_obj}}
                if msg.tool_call_id:
                    fn_part["functionResponse"]["id"] = msg.tool_call_id
                contents.append({"role": "user", "parts": [fn_part]})
            else:  # user
                contents.append({"role": "user", "parts": [{"text": msg.content}]})

        payload: dict[str, Any] = {"contents": contents}
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        if tools:
            declarations = _convert_schema_to_gemini(tools)
            if declarations:
                payload["tools"] = [{"functionDeclarations": declarations}]

        request_raw = _json_bytes(payload)
        response_raw = await self._send_bytes(endpoint, headers, request_raw)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        try:
            data = json.loads(response_raw)
        except Exception as exc:
            return ModelResponse(
                text="", tool_calls=[], cost=CostRecord("api", latency_ms=latency_ms),
                raw_response=response_raw, selector_invalid=True, error=f"Invalid JSON: {exc}"
            )

        candidates = data.get("candidates", [])
        if not candidates or not isinstance(candidates[0], Mapping):
            return ModelResponse(
                text="", tool_calls=[], cost=CostRecord("api", latency_ms=latency_ms),
                raw_response=data, selector_invalid=True, error="Missing candidate in response"
            )

        first_cand = candidates[0]
        raw_parts = first_cand.get("content", {}).get("parts", [])
        text_parts: list[str] = []
        tool_calls: list[ModelToolCall] = []

        for idx, part in enumerate(raw_parts):
            if not isinstance(part, Mapping):
                continue
            if "text" in part:
                text_parts.append(str(part["text"]))
            if "functionCall" in part:
                fc = part["functionCall"]
                call_id = str(fc.get("id") or f"gemini-call-{int(start_time)}-{idx}")
                tool_calls.append(
                    ModelToolCall(
                        id=call_id,
                        name=str(fc.get("name", "")),
                        arguments=dict(fc.get("args", {})),
                        part_index=idx,
                        native_raw=part,
                    )
                )

        # Extract usage and compute costs
        usage = data.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount")
        completion_tokens = usage.get("candidatesTokenCount")
        cached_tokens = usage.get("cachedContentTokenCount")
        total_tokens = usage.get("totalTokenCount")

        # Gemini 3.5 Flash Lite pricing estimate ($0.075 / 1M prompt, $0.01875 / 1M cached, $0.30 / 1M output)
        cost_usd = None
        if prompt_tokens is not None and completion_tokens is not None:
            uncached = prompt_tokens - (cached_tokens or 0)
            cost_usd = (uncached * 0.075 + (cached_tokens or 0) * 0.01875 + completion_tokens * 0.30) / 1_000_000.0

        cost_record = CostRecord(
            backend_type="api",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
        )

        response_id = data.get("responseId")

        return ModelResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            cost=cost_record,
            raw_response=raw_parts,  # Pass raw_parts so multi-turn can preserve thoughtSignature
            response_id=response_id,
            selector_invalid=False,
        )

    async def _generate_openai(
        self,
        messages: Sequence[UnifiedMessage],
        *,
        tools: Sequence[Mapping[str, Any]] | None,
        sampling: Mapping[str, Any] | None,
        start_time: float,
    ) -> ModelResponse:
        endpoint = self.identity.endpoint or OPENAI_ENDPOINT_TEMPLATE
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key or ''}"}

        openai_messages: list[dict[str, Any]] = []
        for msg in messages:
            item: dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.role == "assistant" and msg.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                    }
                    for c in msg.tool_calls
                ]
            elif msg.role in {"tool", "function"}:
                item["role"] = "tool"
                if msg.tool_call_id:
                    item["tool_call_id"] = msg.tool_call_id
                if msg.name:
                    item["name"] = msg.name
            openai_messages.append(item)

        payload: dict[str, Any] = {
            "model": self.identity.model_name,
            "messages": openai_messages,
        }
        if sampling:
            for k in ("temperature", "top_p", "max_tokens", "seed"):
                if k in sampling:
                    payload[k] = sampling[k]

        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"

        request_raw = _json_bytes(payload)
        response_raw = await self._send_bytes(endpoint, headers, request_raw)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        try:
            data = json.loads(response_raw)
        except Exception as exc:
            return ModelResponse(
                text="", tool_calls=[], cost=CostRecord("api", latency_ms=latency_ms),
                raw_response=response_raw, selector_invalid=True, error=f"Invalid JSON: {exc}"
            )

        choices = data.get("choices", [])
        if not choices or not isinstance(choices[0], Mapping):
            return ModelResponse(
                text="", tool_calls=[], cost=CostRecord("api", latency_ms=latency_ms),
                raw_response=data, selector_invalid=True, error="Missing choices in response"
            )

        msg = choices[0].get("message", {})
        text = msg.get("content") or ""
        tool_calls: list[ModelToolCall] = []
        for idx, tc in enumerate(msg.get("tool_calls", [])):
            fn = tc.get("function", {})
            args_str = fn.get("arguments", "{}")
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except Exception:
                args = {"raw": args_str}
            tool_calls.append(
                ModelToolCall(
                    id=tc.get("id") or f"call-{idx}",
                    name=fn.get("name", ""),
                    arguments=args,
                    part_index=idx,
                    native_raw=tc,
                )
            )

        usage = data.get("usage", {})
        cost_record = CostRecord(
            backend_type="api",
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            latency_ms=latency_ms,
        )

        return ModelResponse(
            text=text,
            tool_calls=tool_calls,
            cost=cost_record,
            raw_response=data,
            response_id=data.get("id"),
            selector_invalid=False,
        )
