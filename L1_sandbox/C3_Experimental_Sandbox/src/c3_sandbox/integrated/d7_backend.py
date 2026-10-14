"""D7 diagnostic backend contracts.

The backend is deliberately transport based: production runs can point at an
OpenAI-compatible local endpoint while tests inject a byte transport.  The
module never fabricates a prediction when a model response is malformed.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib import request as urlrequest

OWN_TEXT, ALL_TEXT, C3_STATE = "OWN_TEXT", "ALL_TEXT", "C3_STATE"
METHODS = (OWN_TEXT, ALL_TEXT, C3_STATE)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _episode_content(evidence: Mapping[str, Any], case_id: str) -> Any:
    """Return same-task reference episode content, never a reference count."""
    refs = evidence.get("reference_episodes", evidence.get("same_task_references", evidence.get("reference_episode", evidence.get("same_task_reference_episode", []))))
    if isinstance(refs, Mapping):
        item = refs.get(case_id, refs.get(str(case_id)))
        return item if item is not None else []
    if isinstance(refs, list):
        out = []
        for item in refs:
            if not isinstance(item, Mapping):
                continue
            owner = item.get("case_id", item.get("episode_id", item.get("task_id")))
            if owner is None or str(owner) == str(case_id):
                out.append(item)
        return out
    return []


def build_evidence_view(evidence: Mapping[str, Any], *, method: str, case_id: str) -> dict[str, Any]:
    """Build the sole allowed condition view from saved evidence."""
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    own = {"case_id": case_id, "text": evidence.get("own_text", evidence.get("text", "")),
           "json": evidence.get("own_json", evidence.get("json", {}))}
    if method == OWN_TEXT:
        return {"method": method, "case_id": case_id, "own": own}
    if method == ALL_TEXT:
        return {"method": method, "case_id": case_id, "own": own,
                "reference_episodes": _episode_content(evidence, case_id)}
    return {"method": method, "case_id": case_id,
            "state": evidence.get("state", evidence.get("c3_state", {})),
            "snapshot": evidence.get("snapshot", {}),
            "dispatch": evidence.get("dispatch", {}),
            "score": evidence.get("score", {})}


@dataclass(frozen=True)
class DiagnosticPrediction:
    case_id: str
    method: str
    candidate_universe: tuple[str, ...]
    probabilities: dict[str, float]
    status: str
    model: str
    request_raw: bytes
    response_raw: bytes
    usage: dict[str, Any] | None
    evidence_ids: tuple[str, ...]
    checkpoint_identity: str | None
    request_identity: str
    response_identity: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["request_raw"] = self.request_raw.decode("utf-8", errors="replace")
        value["response_raw"] = self.response_raw.decode("utf-8", errors="replace")
        value["candidate_universe"] = list(self.candidate_universe)
        value["evidence_ids"] = list(self.evidence_ids)
        return value


class DiagnosticBackend:
    """OpenAI-compatible diagnostic backend with an injectable byte transport."""
    def __init__(self, *, model: str, endpoint: str | None = None,
                 transport: Callable[[bytes], bytes] | None = None,
                 timeout: float = 60.0, persist_dir: str | Path | None = None):
        if not model:
            raise ValueError("model is required")
        if endpoint is None and transport is None:
            raise ValueError("endpoint or transport is required")
        self.model, self.endpoint, self.transport = model, endpoint, transport
        self.timeout, self.persist_dir = timeout, Path(persist_dir) if persist_dir else None
        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)

    def _send(self, payload: bytes) -> bytes:
        if self.transport is not None:
            raw = self.transport(payload)
            if not isinstance(raw, (bytes, bytearray)):
                raise TypeError("transport must return response bytes")
            return bytes(raw)
        req = urlrequest.Request(self.endpoint, data=payload,
                                 headers={"Content-Type": "application/json"}, method="POST")
        with urlrequest.urlopen(req, timeout=self.timeout) as response:
            return response.read()

    def predict(self, evidence: Mapping[str, Any], *, case_id: str,
                candidate_universe: Iterable[str], method: str,
                evidence_ids: Iterable[str] = (), checkpoint_identity: str | None = None) -> dict[str, Any]:
        candidates = list(candidate_universe)
        if not candidates or len(set(candidates)) != len(candidates):
            raise ValueError("candidate_universe must be unique and nonempty")
        view = build_evidence_view(evidence, method=method, case_id=case_id)
        payload_obj = {"model": self.model,
                       "messages": [{"role": "system", "content": "Return JSON with probabilities per candidate."},
                                    {"role": "user", "content": json.dumps({"evidence": view, "candidates": candidates}, ensure_ascii=False)}],
                       "response_format": {"type": "json_object"}}
        request_raw = _json_bytes(payload_obj)
        evidence_ids = tuple(str(x) for x in evidence_ids)
        try:
            response_raw = self._send(request_raw)
            outer = json.loads(response_raw)
            usage = outer.get("usage") if isinstance(outer, Mapping) else None
            content = outer.get("choices", [{}])[0].get("message", {}).get("content", "")
            parsed = json.loads(content) if isinstance(content, str) else content
            probs = parsed.get("probabilities") if isinstance(parsed, Mapping) else None
            if not isinstance(probs, Mapping):
                raise ValueError("missing probabilities object")
            probabilities = {c: float(probs[c]) for c in candidates}
            if any(not math.isfinite(v) or not 0 <= v <= 1 for v in probabilities.values()):
                raise ValueError("probabilities must be finite values in [0, 1]")
            status, error = "ok", None
        except Exception as exc:  # malformed model output is an explicit invalid result
            response_raw = locals().get("response_raw", b"")
            usage, probabilities, status, error = None, {}, "selector_invalid", str(exc)
        result = DiagnosticPrediction(case_id, method, tuple(candidates), probabilities, status,
                                      self.model, request_raw, response_raw, usage, evidence_ids,
                                      checkpoint_identity, _sha(request_raw), _sha(response_raw), error)
        if self.persist_dir:
            path = self.persist_dir / f"{case_id}-{method}.json"
            path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            path.with_suffix(".request.bin").write_bytes(request_raw)
            path.with_suffix(".response.bin").write_bytes(response_raw)
        return result.to_dict()


class FixturePredictor:
    """Explicit fixture-only baseline; never the formal diagnostic backend."""
    is_fixture = True

    def __init__(self, probabilities: Mapping[str, float]):
        self.probabilities = dict(probabilities)

    def predict(self, evidence: Mapping[str, Any], *, case_id: str,
                candidate_universe: Iterable[str], method: str, **_: Any) -> dict[str, Any]:
        candidates = list(candidate_universe)
        return {"case_id": case_id, "method": method, "candidate_universe": candidates,
                "probabilities": {c: float(self.probabilities.get(c, 0.0)) for c in candidates},
                "status": "fixture", "model": "fixture", "request_raw": "", "response_raw": "",
                "usage": None, "evidence_ids": [], "checkpoint_identity": None}


def select_condition(predictions: Iterable[Mapping[str, Any]], *, policy: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Select a candidate condition under an explicit policy.

    Predictions from several diagnostic methods are combined by candidate ID;
    the selected value is never a method name or the first list entry.
    """
    policy = dict(policy or {})
    threshold = float(policy.get("min_probability", 0.0))
    exploratory = bool(policy.get("allow_exploratory", False))
    rows = [p for p in predictions if isinstance(p, Mapping)]
    valid = [p for p in rows if p.get("status") in ("ok", "fixture") and isinstance(p.get("probabilities"), Mapping)]
    if len(valid) != len(rows) or not valid:
        return {"status": "selector_invalid", "condition": None, "reason": "invalid prediction"}
    scores: dict[str, list[float]] = {}
    for row in valid:
        for candidate, value in row["probabilities"].items():
            try:
                value = float(value)
            except (TypeError, ValueError):
                return {"status": "selector_invalid", "condition": None, "reason": "non-numeric probability"}
            if not math.isfinite(value) or not 0 <= value <= 1:
                return {"status": "selector_invalid", "condition": None, "reason": "probability outside [0,1]"}
            scores.setdefault(str(candidate), []).append(value)
    ranked = sorted(((candidate, sum(values) / len(values)) for candidate, values in scores.items()),
                    key=lambda item: (-item[1], item[0]))
    eligible = [row for row in ranked if row[1] >= threshold]
    if eligible:
        condition, score = eligible[0]
        return {"status": "selected", "condition": condition, "probability": score,
                "policy": policy, "candidate_scores": dict(ranked), "selection_reason": "max_mean_probability"}
    if exploratory and ranked:
        condition, score = ranked[0]
        return {"status": "exploratory", "condition": condition, "probability": score,
                "policy": policy, "candidate_scores": dict(ranked), "selection_reason": "explicit_exploration_policy"}
    return {"status": "abstain", "condition": None, "reason": "no condition met policy", "policy": policy}


# Descriptive production-facing name; retained alias keeps the interface simple.
OpenAICompatibleBackend = DiagnosticBackend
