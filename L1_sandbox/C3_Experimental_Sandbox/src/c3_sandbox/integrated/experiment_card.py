"""Versioned, evidence-bound experiment cards for offline probe planning.

The workflow is deliberately declarative: compiling a card produces scheduler
branch dictionaries and never executes a model, script, or confirmation.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
import copy, hashlib, json
from typing import Any, Mapping, Iterable

OPERATORS = (
    "original", "visibility.compact", "visibility.compact.matched",
    "visibility.expanded", "visibility.expanded.matched",
    "feedback.aggregate", "feedback.aggregate.matched",
    "feedback.structured", "feedback.structured.matched",
)

class CardStatus(str, Enum):
    PENDING="pending"; COMPILED="compiled"; SEALED="sealed"; OBSERVED="observed"
    INDISTINGUISHABLE="indistinguishable"; UNSUPPORTED="unsupported"; REVIEW_REQUIRED="review_required"

class ExperimentCardError(ValueError): pass
class EvidenceMissingError(ExperimentCardError): pass
class OperatorRejectedError(ExperimentCardError): pass
class UnauthorizedRevealError(ExperimentCardError): pass


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str).encode()).hexdigest()

def _copy(value): return copy.deepcopy(value)

@dataclass(frozen=True)
class EvidenceReference:
    evidence_id: str
    role: str = "support"

@dataclass(frozen=True)
class CompetingExplanation:
    explanation_id: str
    description: str
    probability: float | None = None
    rationale: str = ""
    def __post_init__(self):
        if self.probability is not None and not 0 <= self.probability <= 1: raise ValueError("probability must be in [0,1]")

@dataclass(frozen=True)
class CandidateExperiment:
    experiment_id: str
    condition: Any
    operator: str
    checkpoint: Any
    control: Any
    endpoint: Any
    budget: Any
    applicability: Any = None

@dataclass(frozen=True)
class PriorPrediction:
    prediction_id: str
    experiment_id: str
    expected_outcome: Any
    probabilities: Mapping[str, float] = field(default_factory=dict)
    sealed: bool = False
    seal_identity: str | None = None

@dataclass(frozen=True)
class ExperimentCard:
    phenomenon: str
    evidence_references: tuple[EvidenceReference, ...]
    competing_explanations: tuple[CompetingExplanation, ...]
    candidate_experiments: tuple[CandidateExperiment, ...]
    prior_predictions: tuple[PriorPrediction, ...] = ()
    status: CardStatus = CardStatus.PENDING
    card_id: str = ""
    version: int = 1
    parent_card_id: str | None = None
    revealed_result: Any = field(default=None, repr=False, compare=False)
    update_note: str | None = None

    def __post_init__(self):
        if not self.phenomenon: raise ValueError("phenomenon required")
        def coerce(t, values):
            out=[]
            for value in values:
                if isinstance(value, t): out.append(value)
                elif t is EvidenceReference and isinstance(value, str): out.append(t(value))
                else: out.append(t(**value))
            return tuple(out)
        object.__setattr__(self, "evidence_references", coerce(EvidenceReference, self.evidence_references))
        object.__setattr__(self, "competing_explanations", coerce(CompetingExplanation, self.competing_explanations))
        object.__setattr__(self, "candidate_experiments", coerce(CandidateExperiment, self.candidate_experiments))
        object.__setattr__(self, "prior_predictions", coerce(PriorPrediction, self.prior_predictions))
        if not isinstance(self.status, CardStatus): object.__setattr__(self, "status", CardStatus(self.status))
        if not self.card_id: object.__setattr__(self, "card_id", self.identity_for(self.version, self.parent_card_id))
        if self.version < 1: raise ValueError("version must be positive")

    def identity_for(self, version=None, parent=None):
        payload = {"phenomenon":self.phenomenon, "evidence_references":[asdict(x) for x in self.evidence_references], "competing_explanations":[asdict(x) for x in self.competing_explanations], "candidate_experiments":[asdict(x) for x in self.candidate_experiments], "prior_predictions":[asdict(x) for x in self.prior_predictions], "version":version or self.version, "parent_card_id":parent if parent is not None else self.parent_card_id}
        return _digest(payload)

    @classmethod
    def create(cls, *, phenomenon, evidence_references=(), competing_explanations=(), candidate_experiments=(), prior_predictions=()):
        def conv(t, x):
            if isinstance(x, t): return x
            if t is EvidenceReference and isinstance(x, str): return t(x)
            return t(**x)
        return cls(phenomenon, tuple(conv(EvidenceReference,x) for x in evidence_references), tuple(conv(CompetingExplanation,x) for x in competing_explanations), tuple(conv(CandidateExperiment,x) for x in candidate_experiments), tuple(conv(PriorPrediction,x) for x in prior_predictions))

    def _version(self, **changes):
        data = {f.name:getattr(self,f.name) for f in self.__dataclass_fields__.values() if f.init and f.name not in {"card_id","version","parent_card_id","revealed_result"}}
        data.update(changes); data["version"] = self.version + 1; data["parent_card_id"] = self.card_id; data["card_id"] = ""
        return ExperimentCard(**data)

    def seal_predictions(self):
        preds=[]
        for p in self.prior_predictions:
            if p.sealed: preds.append(p); continue
            identity=_digest({"prediction_id":p.prediction_id,"experiment_id":p.experiment_id,"expected_outcome":p.expected_outcome,"probabilities":dict(p.probabilities),"card_id":self.card_id})
            preds.append(PriorPrediction(p.prediction_id,p.experiment_id,_copy(p.expected_outcome),dict(p.probabilities),True,identity))
        return self._version(prior_predictions=tuple(preds), status=CardStatus.SEALED)

    def update_after_reveal(self, result: Any, *, authorized_reveal=False, outcome_status: CardStatus|str=CardStatus.OBSERVED, explanation_updates: Mapping[str, Any] | None = None):
        if not authorized_reveal: raise UnauthorizedRevealError("confirmation result requires authorized reveal")
        status = CardStatus(outcome_status)
        exps=[]
        for e in self.competing_explanations:
            upd = (explanation_updates or {}).get(e.explanation_id, {})
            if isinstance(upd, Mapping): exps.append(CompetingExplanation(e.explanation_id, upd.get("description",e.description), upd.get("probability",e.probability), upd.get("rationale",e.rationale)))
            else: exps.append(e)
        card=self._version(competing_explanations=tuple(exps), status=status)
        object.__setattr__(card, "revealed_result", _copy(result))
        return card

class ProbeOperatorRegistry:
    def __init__(self, operators: Iterable[str] = OPERATORS): self._operators=frozenset(operators)
    def is_registered(self, operator): return operator in self._operators
    def require(self, operator):
        if not self.is_registered(operator): raise OperatorRejectedError(f"unsupported probe operator: {operator}")

class ExperimentCardCompiler:
    def __init__(self, registry: ProbeOperatorRegistry|None=None): self.registry=registry or ProbeOperatorRegistry()
    def compile(self, card: ExperimentCard, *, evidence: Mapping[str, Any] | Iterable[str] | None = None, evidence_ids: Iterable[str] | None = None, system_identity: Any, runtime_identity: Any, task_identity: Any = "task", line: str = "E2-S", phase: str = "exploration", grader_identity: Any = "grader") -> tuple[dict, ...]:
        if evidence is None: evidence = evidence_ids or ()
        ids = set(evidence.keys()) if isinstance(evidence, Mapping) else set(evidence)
        missing=[r.evidence_id for r in card.evidence_references if r.evidence_id not in ids]
        if missing: raise EvidenceMissingError(f"missing evidence: {missing}")
        branches=[]
        for x in card.candidate_experiments:
            self.registry.require(x.operator)
            cp = x.checkpoint if isinstance(x.checkpoint, Mapping) else {"checkpoint_id":str(x.checkpoint)}
            checkpoint_id=str(cp.get("checkpoint_id", cp.get("id", x.experiment_id)))
            checkpoint_identity=cp.get("checkpoint_identity", _digest(cp))
            probe_identity=_digest({"operator":x.operator,"experiment_id":x.experiment_id,"condition":x.condition})
            branches.append({"case_id":x.experiment_id,"task_identity":task_identity,"line":line,"system_identity":system_identity,"checkpoint_id":checkpoint_id,"checkpoint_identity":checkpoint_identity,"probe_id":x.operator,"probe_identity":probe_identity,"phase":phase,"runtime_identity":runtime_identity,"grader_identity":grader_identity,"sampling":{"operator":x.operator,"condition":_copy(x.condition)},"condition":_copy(x.condition),"repeat":1,"budget":_copy(x.budget),"endpoint":_copy(x.endpoint),"control":_copy(x.control),"applicability":_copy(x.applicability)})
        return tuple(branches)

def outcome_not_run(*, reason="not scheduled", **context): return {"status":"NOT_RUN", "reason":reason, **context}
def outcome_selector_invalid(*, reason="invalid selector", **context): return {"status":"selector_invalid", "reason":reason, **context}
def outcome_unknown(*, reason="unknown", **context): return {"status":"unknown", "reason":reason, **context}

__all__=["OPERATORS","CardStatus","ExperimentCard","EvidenceReference","CompetingExplanation","CandidateExperiment","PriorPrediction","ProbeOperatorRegistry","ExperimentCardCompiler","ExperimentCardError","EvidenceMissingError","OperatorRejectedError","UnauthorizedRevealError","outcome_not_run","outcome_selector_invalid","outcome_unknown"]
