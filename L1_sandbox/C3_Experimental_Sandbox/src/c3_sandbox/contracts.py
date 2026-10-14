"""Small shared contracts; filesystem snapshots are not complete agent state."""
from dataclasses import dataclass, field
from typing import Protocol

FILESYSTEM_SCOPE = "filesystem-only: regular files, directories, bytes and permission modes; excludes process memory, services, environment and conversation/context"


@dataclass
class RestoreReport:
    valid: bool
    mismatches: list[str] = field(default_factory=list)
    scope: str = FILESYSTEM_SCOPE

    def to_dict(self) -> dict:
        return {"valid": self.valid, "mismatches": list(self.mismatches), "scope": self.scope}


@dataclass(frozen=True)
class StateTransition:
    """Observed object transition; no claim about an internal cognitive mechanism."""
    precondition: str
    operation: str
    object_id: str
    scope: str
    postcondition: str


@dataclass(frozen=True)
class ProbeSpec:
    probe_id: str
    checkpoint_id: str
    content: str
    matched_control_id: str
    leakage_review_status: str = "draft"


@dataclass(frozen=True)
class Prediction:
    case_id: str
    method_id: str
    budget: float
    candidate_universe: tuple[str, ...]
    probabilities: dict[str, float]
    is_synthetic: bool
    pre_confirmation: bool


@dataclass(frozen=True)
class Branch:
    case_id: str
    probe_id: str
    snapshot_id: str
    seed: int
    restore_verified: bool = False


class ModelAdapter(Protocol):
    """Extension boundary only; no model integration is supplied here."""
    def generate(self, messages: list[dict], budget: int) -> dict: ...


class StateAligner(Protocol):
    def align(self, transitions: list[StateTransition]) -> list[StateTransition]: ...


class ProbeSelector(Protocol):
    def predict(self, case_id: str, candidates: list[ProbeSpec], budget: int) -> Prediction: ...
