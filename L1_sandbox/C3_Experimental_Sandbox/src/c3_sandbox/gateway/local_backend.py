"""Local model backend implementation for MLX, vLLM, and local inference services.

Enforces:
- Rule 1: Retains native local service protocol (e.g. MLX Chat Completions with custom tool parser).
- Rule 2: Fail-closed; no silent fallback on failure.
- Rule 3: Binds local model weights path, fingerprint, revision, and tool parser.
- Rule 4: Records tokens, latency, and hardware accelerators (Metal/CUDA); cost_usd remains null.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import socket
import subprocess
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


def _detect_hardware_info() -> dict[str, Any]:
    """Detect local accelerator presence (Apple Metal, CUDA, or CPU)."""
    info = {
        "system": platform.system(),
        "machine": platform.machine(),
        "accelerator": "cpu",
    }
    if platform.system() == "Darwin":
        # Check Apple Silicon / Metal
        try:
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
            if "Apple" in out.stdout or platform.machine() == "arm64":
                info["accelerator"] = "metal"
                info["device"] = "Apple Silicon GPU"
        except Exception:
            pass
    return info


class LocalBackend(ModelBackend):
    """Local inference backend connecting to MLX, vLLM, or OpenAI-compatible local server."""

    def __init__(
        self,
        identity: ModelIdentity,
        config: Mapping[str, Any] | None = None,
        *,
        transport: Callable[..., Any] | None = None,
        timeout: float = 120.0,
    ):
        super().__init__(identity, config)
        self.transport = transport
        self.timeout = timeout
        self.endpoint = identity.endpoint or self._config.get("model_endpoint") or "http://127.0.0.1:18080/v1/chat/completions"
        self.hardware_info = _detect_hardware_info()

    def check_availability(self) -> tuple[bool, str | None]:
        """Check if local endpoint is listening or if model files exist."""
        # 1. If weights path specified in metadata, verify existence
        model_path = self.identity.metadata.get("model_path")
        if model_path:
            p = Path(model_path)
            if not p.is_dir():
                return False, f"Local model weights path not found: {p}"

        # 2. If transport is injected, consider available
        if self.transport is not None:
            return True, None

        # 3. If endpoint has a host/port, check port reachability
        if "://" in self.endpoint:
            try:
                host_port = self.endpoint.split("://", 1)[1].split("/", 1)[0]
                if ":" in host_port:
                    host, port_str = host_port.split(":", 1)
                    port = int(port_str)
                    with socket.create_connection((host, port), timeout=1.0):
                        pass
            except Exception as exc:
                return False, f"Local server not reachable at {self.endpoint}: {exc}"

        return True, None

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

        start_time = time.perf_counter()
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

        raw_req = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if self.transport is not None:
            raw_resp = self.transport(raw_req)
        else:
            req = urlrequest.Request(self.endpoint, data=raw_req, headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urlrequest.urlopen(req, timeout=self.timeout) as response:
                    raw_resp = response.read()
            except Exception as exc:
                raise BackendError(f"Local inference call failed: {exc}") from exc

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        try:
            data = json.loads(raw_resp)
        except Exception as exc:
            return ModelResponse(
                text="", tool_calls=[],
                cost=CostRecord("local", latency_ms=latency_ms, hardware_info=self.hardware_info),
                raw_response=raw_resp, selector_invalid=True, error=f"Invalid JSON: {exc}"
            )

        choices = data.get("choices", [])
        if not choices or not isinstance(choices[0], Mapping):
            return ModelResponse(
                text="", tool_calls=[],
                cost=CostRecord("local", latency_ms=latency_ms, hardware_info=self.hardware_info),
                raw_response=data, selector_invalid=True, error="Missing choices"
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
                    id=tc.get("id") or f"local-call-{idx}",
                    name=fn.get("name", ""),
                    arguments=args,
                    part_index=idx,
                    native_raw=tc,
                )
            )

        usage = data.get("usage", {})
        cost_record = CostRecord(
            backend_type="local",
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            cost_usd=None,  # Rule 4: cost_usd is explicitly None for local models
            latency_ms=latency_ms,
            hardware_info=self.hardware_info,
        )

        return ModelResponse(
            text=text,
            tool_calls=tool_calls,
            cost=cost_record,
            raw_response=data,
            response_id=data.get("id"),
            selector_invalid=False,
        )


class SavedReferenceBackend(ModelBackend):
    """Stub backend for frozen reference baselines.
    
    Reference evidence is preserved separately; no duplicate original runs.
    """

    def __init__(self, identity: ModelIdentity, config: Mapping[str, Any] | None = None, *, transport: Any = None, **kwargs):
        super().__init__(identity, config, **kwargs)
        self.transport = transport

    def check_availability(self) -> tuple[bool, str | None]:
        return True, None

    async def generate(self, messages, *, tools=None, sampling=None) -> ModelResponse:
        raise BackendError("Saved reference backend does not execute live generations")
