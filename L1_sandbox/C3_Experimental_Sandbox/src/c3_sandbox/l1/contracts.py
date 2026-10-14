"""L1-only port contracts. These interfaces do not claim live backend qualification.

Native protocol envelopes must be retained alongside normalized data. Neither a
model response nor a backend is authorized to fabricate a ToolExecution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class ExecutionBudget:
    max_model_requests: int
    max_tool_calls: int
    wall_time_seconds: float | None = None
    max_tokens: int | None = None
    max_cost_usd: float | None = None

    def __post_init__(self) -> None:
        for name in ("max_model_requests", "max_tool_calls"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("wall_time_seconds", "max_tokens", "max_cost_usd"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive or unknown")

        if self.max_tokens is not None and type(self.max_tokens) is not int:
            raise ValueError("max_tokens must be an integer")


@dataclass(frozen=True)
class PublicTask:
    """Only this view may be staged in the agent workspace; no hidden rubric."""
    benchmark: str
    task_id: str
    prompt: str
    assets: tuple[Mapping[str, Any], ...]
    expected_outputs: tuple[str, ...]
    source_revision: str | None


@dataclass(frozen=True)
class BackendIdentity:
    backend: str  # api | local; a fixture is not a qualified backend
    provider: str
    requested_model: str
    resolved_revision: str | None = None
    tokenizer_digest: str | None = None


@dataclass(frozen=True)
class ModelRequest:
    request_id: str
    native_body: bytes
    parameters: Mapping[str, Any]
    # Authentication is deliberately excluded; the backend obtains secrets privately.


@dataclass(frozen=True)
class ToolInvocation:
    tool_call_id: str
    request_id: str
    name: str
    arguments: Mapping[str, Any]
    native_id: str | None = None


@dataclass(frozen=True)
class ModelReply:
    request_id: str
    native_body: bytes
    tool_calls: tuple[ToolInvocation, ...] = ()
    usage: Mapping[str, Any] | None = None
    reported_model_version: str | None = None
    error: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ToolExecution:
    """Created by the sandbox boundary, never by the model backend."""
    tool_call_id: str
    raw_observation: Mapping[str, Any]
    started_at: str | None
    finished_at: str | None
    before_snapshot_id: str | None
    after_snapshot_id: str | None


@dataclass(frozen=True)
class CheckpointQualification:
    checkpoint_id: str
    eligible: bool
    scopes: tuple[str, ...]
    reason_codes: tuple[str, ...]
    evidence_refs: tuple[Mapping[str, Any], ...] = ()


class BenchmarkAdapter(Protocol):
    def public_task(self, task_id: str) -> PublicTask: ...
    def stage_public_assets(self, task: PublicTask, destination: Path) -> Mapping[str, Any]: ...
    def evaluate(self, task_id: str, submission: Path, output: Path) -> Mapping[str, Any]:
        """Run an independent evaluator; preserve raw output even on failure."""
        ...


class ModelBackend(Protocol):
    @property
    def identity(self) -> BackendIdentity: ...
    async def invoke(self, request: ModelRequest) -> ModelReply: ...


class IsolatedSandbox(Protocol):
    def provenance(self) -> Mapping[str, Any]: ...
    def execute(self, invocation: ToolInvocation) -> ToolExecution: ...
    def capture_checkpoint(self) -> CheckpointQualification: ...
    def restore(self, checkpoint_id: str, qualification: CheckpointQualification) -> Mapping[str, Any]: ...
