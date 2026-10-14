"""Closed declarative probe vocabulary; this module executes no intervention."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import hmac
import json
import math
from pathlib import Path

from .evidence._storage import encode, digest, identity as check_identity


VERSION = "C3-E2-v1.0"


def _spec(probe_id, kind, mechanism, payload, lines, matched=None):
    return {
        "probe_id": probe_id, "kind": kind, "mechanism": mechanism,
        "payload": payload, "lines": lines, "control_id": matched, "matched_control_id": matched,
        "leakage_policy": {
            "allowlist": "closed_generic_public_observation_presentation",
            "no_answer_material": True, "no_task_specific_patch": True,
            "no_target_object_paths": True, "per_task_review": "required_including_controls",
        },
        "preconditions": [] if mechanism == "original" else [
            "redundant_observation_views" if mechanism == "visibility" else "public_feedback_fields"],
        "preserve": ["necessary_inputs", "necessary_tools", "sampling", "temperature", "budgets"],
        "requires_fresh_execution": True,
    }


def _finite_number(value, *, minimum=0, strict=False):
    if type(value) not in {int, float}:
        return False
    try:
        return math.isfinite(value) and (value > minimum if strict else value >= minimum)
    except OverflowError:
        return False


def _vocabulary():
    specs = [_spec("original", "original", "original", {}, ["E2-S", "E2-F"])]
    for line, visibility, feedback in [("E2-S", "compact", "aggregate"), ("E2-F", "expanded", "structured")]:
        for mechanism, mode in [("visibility", visibility), ("feedback", feedback)]:
            name = f"{mechanism}.{mode}"
            specs.append(_spec(name, "target", mechanism, {"mode": mode}, [line], name + ".matched"))
            specs.append(_spec(name + ".matched", "matched_control", mechanism,
                               {"mode": "sham", "matched_mode": mode}, [line]))
    return {spec["probe_id"]: spec for spec in specs}


class ProbeRegistry:
    def __init__(self, document: dict):
        if not isinstance(document, dict) or set(document) != {"version", "probes"} or document["version"] != VERSION:
            raise ValueError("invalid registry schema/version")
        if not isinstance(document["probes"], list):
            raise ValueError("probes must be a list")
        expected = _vocabulary()
        found = {}
        for spec in document["probes"]:
            if (not isinstance(spec, dict) or not isinstance(spec.get("probe_id"), str)
                    or spec["probe_id"] not in expected):
                raise ValueError("unknown probe")
            key = spec["probe_id"]
            try:
                matches = encode(spec) == encode(expected[key])
            except (TypeError, ValueError) as exc:
                raise ValueError("probe specification is not canonical JSON") from exc
            if key in found or not matches:
                raise ValueError("probe payload/preconditions differ from closed vocabulary")
            found[key] = deepcopy(spec)
        if set(found) != set(expected):
            raise ValueError("registry must contain the complete frozen panel vocabulary")
        self._specs = found

    @classmethod
    def from_file(cls, path):
        def unique(pairs):
            obj = {}
            for key, value in pairs:
                if key in obj:
                    raise ValueError("duplicate JSON key")
                obj[key] = value
            return obj
        return cls(json.loads(Path(path).read_text(), object_pairs_hook=unique))

    def get(self, probe_id: str) -> dict:
        return deepcopy(self._specs[probe_id])

    def identity(self, probe_id: str) -> str:
        return digest(encode({"version": VERSION, "probe": self.get(probe_id)}))

    def experiment_panel(self, line: str, target_probe_id: str, competing_probe_id: str) -> list[dict]:
        if line not in {"E2-S", "E2-F"}:
            raise ValueError("unknown experiment line")
        target, competing = self.get(target_probe_id), self.get(competing_probe_id)
        if any(p["kind"] != "target" or line not in p["lines"] for p in [target, competing]):
            raise ValueError("target and competing probes must be targets eligible for this line")
        if target["mechanism"] == competing["mechanism"]:
            raise ValueError("competing probe must test another mechanism")
        ids = ["original", target_probe_id, target["matched_control_id"], competing_probe_id]
        return [dict(self.get(key), identity=self.identity(key), role=role,
                     checkpoint_policy="same_checkpoint", line=line)
                for key, role in zip(ids, ["original", "target", "matched", "competing"])]

    def validate_for_case(self, probe_id: str, case_contract: dict, review: dict | None) -> dict:
        spec = self.get(probe_id)
        reasons = []
        case = case_contract
        if not isinstance(case, dict):
            return {"eligible": False, "reasons": ["invalid case contract"], "identity": self.identity(probe_id)}
        try:
            check_identity(case.get("task_identity"))
        except ValueError:
            reasons.append("task identity missing/invalid")
        if type(case.get("is_synthetic")) is not bool:
            reasons.append("synthetic status must be explicit")
        if case.get("line") not in spec["lines"]:
            reasons.append("probe not eligible for experiment line")
        for category in ["inputs", "tools"]:
            available, necessary = case.get("available_" + category), case.get("necessary_" + category)
            if (not isinstance(available, list) or not isinstance(necessary, list)
                    or any(not isinstance(x, str) or not x for x in (available or []) + (necessary or []))
                    or len(set(available or [])) != len(available or [])
                    or not set(necessary or []).issubset(available or [])):
                reasons.append("missing or removed necessary " + category)
        capabilities = case.get("observation_capabilities", [])
        if (not isinstance(capabilities, list) or any(not isinstance(x, str) for x in capabilities)
                or not set(spec["preconditions"]).issubset(capabilities)):
            reasons.append("probe preconditions not satisfied")
        sampling = case.get("sampling")
        temperature = sampling.get("temperature") if isinstance(sampling, dict) else None
        if not _finite_number(temperature):
            reasons.append("sampling/temperature contract missing or invalid")
        budgets = case.get("budgets")
        if (not isinstance(budgets, dict) or not budgets
                or any(not _finite_number(v, strict=True) for v in budgets.values())):
            reasons.append("budget contract missing or invalid")
        # Caller overrides are never a second arbitrary intervention channel.
        for field in ["removed_inputs", "removed_tools", "sampling_override", "temperature_override", "budget_override", "payload_override"]:
            if field in case:
                reasons.append("forbidden intervention override: " + field)
        reasons.extend(self._review_reasons(probe_id, case, review))
        return {"eligible": not reasons, "reasons": reasons, "identity": self.identity(probe_id)}

    def _review_reasons(self, probe_id, case, review):
        if not isinstance(review, dict):
            return ["per-task leakage/solvability review missing"]
        try:
            if set(review) != {"task_identity", "probe_identity", "line", "source", "findings", "signature"}:
                raise ValueError("review schema invalid")
            if (review["task_identity"] != case.get("task_identity") or review["probe_identity"] != self.identity(probe_id)
                    or review["line"] != case.get("line")):
                raise ValueError("stale or mismatched review binding")
            source = review["source"]
            if not isinstance(source, dict) or set(source) != {"kind", "signer_id", "record_identity"}:
                raise ValueError("review source provenance missing")
            check_identity(source["record_identity"])
            if not isinstance(source["signer_id"], str) or not source["signer_id"]:
                raise ValueError("review signer missing")
            findings = review["findings"]
            if set(findings) != {"leakage", "solvability", "reviewed_inputs", "reviewed_tools"}:
                raise ValueError("review findings invalid")
            for field, verdict in [("leakage", "no_task_answer_material"), ("solvability", "necessary_resources_preserved")]:
                finding = findings[field]
                if not isinstance(finding, dict) or set(finding) != {"verdict", "evidence_identities"} or finding["verdict"] != verdict:
                    raise ValueError("review rejected or incomplete: " + field)
                evidence = finding["evidence_identities"]
                if not isinstance(evidence, list) or not evidence:
                    raise ValueError("review evidence missing: " + field)
                for entry in evidence:
                    check_identity(entry)
            for category in ["inputs", "tools"]:
                if findings["reviewed_" + category] != case.get("available_" + category):
                    raise ValueError("review resource inventory mismatch")
            signature = review["signature"]
            if not isinstance(signature, dict) or set(signature) != {"scheme", "value"}:
                raise ValueError("review signature invalid")
            check_identity(signature["value"])
            body = encode({key: value for key, value in review.items() if key != "signature"})
            if source["kind"] == "synthetic_fixture":
                if case.get("is_synthetic") is not True or signature["scheme"] != "sha256-fixture":
                    raise ValueError("synthetic review cannot authorize real case")
                expected = digest(body)
            else:
                if source["kind"] not in {"human", "task_verifier"} or signature["scheme"] != "hmac-sha256":
                    raise ValueError("signed human/task verifier provenance required")
                key = case.get("trusted_review_keys", {}).get(source["signer_id"])
                if not isinstance(key, str) or len(key) < 32:
                    raise ValueError("trusted review signer missing")
                expected = hmac.new(key.encode(), body, hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expected, signature["value"]):
                raise ValueError("review signature mismatch")
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            return [str(exc)]
        return []
