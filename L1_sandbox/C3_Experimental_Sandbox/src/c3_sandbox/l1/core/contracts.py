"""Versioned live ports. No provider, benchmark, or container imports here."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

VERSION = '0.2.0'

class L1Error(RuntimeError):
    code = 'infrastructure_error'
class ConfigurationError(L1Error):
    code = 'configuration_error'
class ProtocolError(L1Error):
    code = 'protocol_error'
class CapabilityError(L1Error):
    code = 'unsupported_capability'
class EvidenceError(L1Error):
    code = 'evidence_error'
class BudgetError(L1Error):
    code = 'budget_exhausted'

@dataclass(frozen=True)
class Asset:
    name: str
    path: Path
    sha256: str
    size_bytes: int

@dataclass(frozen=True)
class EvaluationSpec:
    kind: str
    source_root: Path
    entrypoint: str
    task_id: str
    metadata: dict = field(default_factory=dict)

@dataclass(frozen=True)
class Task:
    benchmark: str
    task_id: str
    prompt: str
    assets: tuple[Asset, ...]
    provenance: dict
    private_files: dict[str, Path]
    evaluation: EvaluationSpec
    requirements: dict = field(default_factory=dict)

@dataclass(frozen=True)
class Call:
    call_id: str
    name: str
    arguments: Any
    native_id: str | None

@dataclass(frozen=True)
class NativeValidatedCall(Call):
    """Internal execution capability issued only after pinned native validation."""

@dataclass(frozen=True)
class Reply:
    calls: tuple[Call, ...]
    native_message: dict
    usage: dict | None
    model_version: str | None
    finish_reason: str | None
    text: str | None

@dataclass(frozen=True)
class HTTPResult:
    status: int | None
    body: bytes
    metadata: dict

@dataclass(frozen=True)
class Result:
    observation: dict
    stdout: bytes = b''
    stderr: bytes = b''
    image: bytes | None = None
    image_mime: str | None = None
    provenance: dict = field(default_factory=dict)

class BenchmarkAdapter(Protocol):
    def list_tasks(self) -> list[dict]: ...
    def load(self, task_id: str) -> Task: ...

class ModelBackend(Protocol):
    def identity(self) -> dict: ...
    def initial(self, system: str, prompt: str) -> dict: ...
    def request(self, conversation: dict, tools: list[dict]) -> dict: ...
    def send(self, body: bytes, timeout: float) -> HTTPResult: ...
    def parse(self, body: bytes, request_id: str) -> Reply: ...
    def accept(self, conversation: dict, reply: Reply, results: list[tuple[Call, Result]]) -> dict: ...

class SandboxRuntime(Protocol):
    def preflight(self) -> dict: ...
    def execute(self, call: Call, workspace: Path, timeout: float) -> Result: ...
    def qualification(self) -> dict: ...

class Evaluator(Protocol):
    def evaluate(self, spec: EvaluationSpec, submission: Path, output: Path) -> dict: ...
