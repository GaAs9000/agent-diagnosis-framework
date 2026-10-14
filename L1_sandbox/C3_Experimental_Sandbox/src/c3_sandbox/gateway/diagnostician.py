"""GatewayDiagnosticBackend: Bridges ModelBackend with D7 Diagnostic contracts.

Enforces:
- Rule 1: Converts evidence view and candidate universe into unified messages.
- Rule 2: Fail-closed; no silent fallback on model failure or malformed response.
- Rule 3: Preserves model identity, request/response SHA-256 hashes, and checkpoint IDs.
- Rule 4: Records cost and usage from CostRecord into diagnostic prediction.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from .backend import (
    BackendError,
    ModelBackend,
    ModelResponse,
    UnifiedMessage,
)
from ..integrated.d7_backend import (
    DiagnosticPrediction,
    METHODS,
    build_evidence_view,
)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class GatewayDiagnosticBackend:
    """Bridge connecting any ModelBackend (ApiBackend or LocalBackend) to D7 predictions."""

    def __init__(self, backend: ModelBackend, *, persist_dir: str | Path | None = None):
        self.backend = backend
        self.persist_dir = Path(persist_dir) if persist_dir else None
        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model(self) -> str:
        return self.backend.model_name

    def predict(
        self,
        evidence: Mapping[str, Any],
        *,
        case_id: str,
        candidate_universe: Iterable[str],
        method: str,
        evidence_ids: Iterable[str] = (),
        checkpoint_identity: str | None = None,
    ) -> dict[str, Any]:
        candidates = list(candidate_universe)
        if not candidates or len(set(candidates)) != len(candidates):
            raise ValueError("candidate_universe must be unique and nonempty")

        view = build_evidence_view(evidence, method=method, case_id=case_id)
        evidence_ids_tuple = tuple(str(x) for x in evidence_ids)

        system_content = "Return JSON with probabilities per candidate."
        user_content = json.dumps({"evidence": view, "candidates": candidates}, ensure_ascii=False)

        payload_obj = {
            "model": self.backend.model_name,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            "response_format": {"type": "json_object"},
        }
        request_raw = _json_bytes(payload_obj)

        try:
            ok, reason = self.backend.check_availability()
            if not ok:
                raise BackendError(f"Diagnostician backend unavailable: {reason}")

            messages = [
                UnifiedMessage(role="system", content=system_content),
                UnifiedMessage(role="user", content=user_content),
            ]
            response: ModelResponse = asyncio.run(self.backend.generate(messages))

            if response.selector_invalid or response.error:
                raise ValueError(response.error or "backend returned selector_invalid")

            response_raw = (
                _json_bytes(response.raw_response)
                if response.raw_response is not None
                else response.text.encode("utf-8")
            )

            # Extract probabilities
            content = response.text or ""
            parsed = None
            if content.strip().startswith("{"):
                try:
                    parsed = json.loads(content)
                except Exception:
                    parsed = None
            if not isinstance(parsed, Mapping) and "```json" in content:
                try:
                    block = content.split("```json", 1)[1].split("```", 1)[0].strip()
                    parsed = json.loads(block)
                except Exception:
                    parsed = None
            if not isinstance(parsed, Mapping) and isinstance(response.raw_response, Mapping):
                # Try OpenAI choices format
                choices = response.raw_response.get("choices", [])
                if choices and isinstance(choices[0], Mapping):
                    inner_content = choices[0].get("message", {}).get("content", "")
                    try:
                        parsed = json.loads(inner_content)
                    except Exception:
                        pass
                if not isinstance(parsed, Mapping):
                    parsed = response.raw_response

            probs = parsed.get("probabilities") if isinstance(parsed, Mapping) else None
            if not isinstance(probs, Mapping):
                raise ValueError(f"missing probabilities object in model response: {content[:200]}")

            probabilities = {c: float(probs[c]) for c in candidates}
            if any(not math.isfinite(v) or not 0 <= v <= 1 for v in probabilities.values()):
                raise ValueError("probabilities must be finite values in [0, 1]")

            usage = response.cost.to_dict()
            status, error = "ok", None

        except Exception as exc:
            response_raw = locals().get("response_raw", b"")
            usage, probabilities, status, error = None, {}, "selector_invalid", str(exc)

        result = DiagnosticPrediction(
            case_id=case_id,
            method=method,
            candidate_universe=tuple(candidates),
            probabilities=probabilities,
            status=status,
            model=self.backend.model_name,
            request_raw=request_raw,
            response_raw=response_raw,
            usage=usage,
            evidence_ids=evidence_ids_tuple,
            checkpoint_identity=checkpoint_identity,
            request_identity=_sha(request_raw),
            response_identity=_sha(response_raw),
            error=error,
        )

        if self.persist_dir:
            path = self.persist_dir / f"{case_id}-{method}.json"
            path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            path.with_suffix(".request.bin").write_bytes(request_raw)
            path.with_suffix(".response.bin").write_bytes(response_raw)

        return result.to_dict()
