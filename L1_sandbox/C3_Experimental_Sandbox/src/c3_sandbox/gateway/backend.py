"""Unified ModelBackend interface and typed data structures for C3 Sandbox.

All models (Remote API and Local MLX/vLLM) enter C3 via the same invocation
contract. Upper-level protocols (Sandbox, Docker execution, Verifier, Experiment
Cards, D6-D8 diagnostics) are completely unified while native wire protocols are
retained per-backend and mapped through adapters.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
import hashlib
import json
import time
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ModelIdentity:
    """Immutable identity record binding model, backend, version, and schemas.
    
    Part of experimental identity: must be frozen with checkpoints and config locks.
    """
    backend: str                        # "api" | "local" | "saved_reference"
    provider: str                       # "gemini" | "openai" | "anthropic" | "mlx" | "saved_reference"
    model_name: str                     # e.g. "gemini-3.5-flash-lite", "Qwen3-Coder-30B-A3B-Instruct-4bit"
    fingerprint: str                    # sha256 or provider model version
    tokenizer_fingerprint: str | None = None
    revision: str | None = None
    endpoint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def identity_hash(self) -> str:
        raw = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ModelToolCall:
    """Unified representation of a tool call requested by the model."""
    id: str
    name: str
    arguments: dict[str, Any]
    part_index: int = 0
    native_raw: Any = None


@dataclass
class CostRecord:
    """Typed cost and resource accounting.
    
    Rule 4: API records request usage and cost/fees; local models record
    tokens, latency, and hardware resources; missing metrics remain None/null.
    """
    backend_type: str                   # "api" | "local" | "saved_reference"
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None       # Populated for API where rate is known
    latency_ms: float | None = None     # Wall-clock generation latency
    hardware_info: dict[str, Any] | None = None  # e.g. {"accelerator": "metal", "device": "Apple Silicon"}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UnifiedMessage:
    """Unified conversational message for multi-turn execution."""
    role: str                           # "system" | "user" | "model" | "assistant" | "tool"
    content: str = ""
    tool_calls: list[ModelToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    native_parts: Any = None            # Preserves provider-specific wire data (e.g. thoughtSignature)


@dataclass
class ModelResponse:
    """Standardized response from any ModelBackend."""
    text: str
    tool_calls: list[ModelToolCall]
    cost: CostRecord
    raw_response: Any = None
    response_id: str | None = None
    selector_invalid: bool = False
    error: str | None = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class BackendError(RuntimeError):
    """Base error raised when a backend fails closed."""
    pass


class ModelBackend(ABC):
    """Abstract interface that all model backends must implement."""

    def __init__(self, identity: ModelIdentity, config: Mapping[str, Any] | None = None, **kwargs):
        self._identity = identity
        self._config = dict(config or {})

    @property
    def identity(self) -> ModelIdentity:
        return self._identity

    @property
    def backend_type(self) -> str:
        return self._identity.backend

    @property
    def model_name(self) -> str:
        return self._identity.model_name

    @abstractmethod
    def check_availability(self) -> tuple[bool, str | None]:
        """Check whether the backend is available for live execution.
        
        Returns (is_available, error_or_reason_if_not).
        """
        pass

    @abstractmethod
    async def generate(
        self,
        messages: Sequence[UnifiedMessage],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        sampling: Mapping[str, Any] | None = None,
    ) -> ModelResponse:
        """Execute a single-turn generation against the model.
        
        Must return a valid ModelResponse or raise BackendError.
        Rule 2: Never silently fallback to another backend on error.
        """
        pass
