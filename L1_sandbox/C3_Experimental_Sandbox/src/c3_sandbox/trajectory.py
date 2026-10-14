"""Validated trajectories and lossless, non-replay-qualified legacy import."""
import base64
import binascii
import hashlib
import json
from pathlib import Path

from .manifests import SCHEMA_VERSION, SPLITS, _copy, _hash, _object, _text, _version

OUTCOME_FIELDS = ("execution_valid", "target_resolved", "regression_free", "task_completed")


def _validate_status(status, outcome):
    # Failure versus partial is the external scorer's label, not a count of
    # positive endpoint fields. Unknown preserves incomplete classification.
    if outcome["execution_valid"] is False and any(
        outcome[field] is True for field in OUTCOME_FIELDS if field != "execution_valid"
    ):
        raise ValueError("invalid execution cannot establish verified successful endpoints")
    if status == "unknown":
        return
    if not isinstance(status, str) or status not in {"success", "failure", "partial"}:
        raise ValueError("unknown status")
    completed = status == "success"
    if outcome["execution_valid"] is not True or outcome["task_completed"] is not completed:
        raise ValueError("status is incompatible with outcome")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _loads(data):
    def reject(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    try:
        value = json.loads(data, parse_constant=reject, object_pairs_hook=_pairs)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid UTF-8 JSON evidence") from exc
    return _copy(value)


def _records(data):
    records = []
    for line in data.splitlines():
        if line.strip():
            value = _loads(line)
            if not isinstance(value, dict):
                raise ValueError("JSONL evidence must contain objects")
            records.append(value)
    return records


def validate_trajectory(record: dict) -> dict:
    """Return an independent copy; raise ValueError on incompatible evidence."""
    record = _copy(record)
    _object(record, {"schema_version", "episode_id", "task_id", "task_family_id", "system_id", "split", "status", "outcome", "events", "artifact_versions", "checkpoint_refs", "provenance"})
    _version(record)
    for name in ("episode_id", "task_id", "task_family_id", "system_id"):
        _text(record[name])
    if not isinstance(record["split"], str) or record["split"] not in SPLITS:
        raise ValueError("unknown split")
    outcome = record["outcome"]
    _object(outcome, set(OUTCOME_FIELDS))
    if set(outcome) != set(OUTCOME_FIELDS) or any(value is not None and type(value) is not bool for value in outcome.values()):
        raise ValueError("outcome requires exactly four boolean or null fields")
    _validate_status(record["status"], outcome)
    for name in ("events", "artifact_versions", "checkpoint_refs"):
        if not isinstance(record[name], list) or any(not isinstance(item, dict) for item in record[name]):
            raise ValueError(f"{name} must contain objects")
    provenance = record["provenance"]
    _object(provenance, {"source", "version", "is_synthetic"})
    _text(provenance["source"])
    _text(provenance["version"])
    if type(provenance["is_synthetic"]) is not bool:
        raise ValueError("is_synthetic must be boolean")
    raw_files = provenance.get("raw_files", {})
    if not isinstance(raw_files, dict):
        raise ValueError("raw_files must be an object")
    for name, raw in raw_files.items():
        _object(raw, {"sha256", "base64", "records"})
        _hash(raw["sha256"])
        try:
            data = base64.b64decode(raw["base64"], validate=True)
        except (ValueError, TypeError, binascii.Error) as exc:
            raise ValueError("invalid raw evidence base64") from exc
        if hashlib.sha256(data).hexdigest() != raw["sha256"]:
            raise ValueError("raw evidence hash mismatch")
        expected = _records(data) if name.endswith(".jsonl") else _loads(data)
        if expected != raw["records"]:
            raise ValueError("raw evidence records do not match original bytes")
    for checkpoint in record["checkpoint_refs"]:
        _object(checkpoint, {"checkpoint_id", "replay_qualified"})
        _text(checkpoint["checkpoint_id"])
        if type(checkpoint["replay_qualified"]) is not bool:
            raise ValueError("replay_qualified must be boolean")
        if checkpoint["replay_qualified"]:
            _object(checkpoint, {"checkpoint_identity", "qualification_report_ref"})
        for name in ("checkpoint_identity", "qualification_report_ref"):
            if name in checkpoint:
                _hash(checkpoint[name])
    if provenance["source"] == "legacy_officeval":
        required = {"summary.json", "events.jsonl", "artifacts.jsonl", "checkpoints.jsonl", "verifier_results.jsonl"}
        if not required <= raw_files.keys():
            raise ValueError("legacy import requires all five raw evidence files")
        if record["events"] != raw_files["events.jsonl"]["records"] or record["artifact_versions"] != raw_files["artifacts.jsonl"]["records"]:
            raise ValueError("legacy evidence arrays differ from retained records")
        originals = raw_files["checkpoints.jsonl"]["records"]
        if [ref.get("raw") for ref in record["checkpoint_refs"]] != originals:
            raise ValueError("legacy checkpoint references differ from retained records")
        if any(value is not None for value in outcome.values()):
            raise ValueError("legacy outcome cannot establish E2 endpoint fields")
        if record["status"] != "unknown" or any(ref["replay_qualified"] for ref in record["checkpoint_refs"]):
            raise ValueError("legacy status must remain unknown and checkpoints unqualified")
    return record


def import_legacy_episode(run_dir, *, task_family_id, system_id, split="development") -> dict:
    """Read evidence only. Legacy SafeFix/score/status do not establish E2 outcome."""
    run_dir = Path(run_dir).resolve()
    if split != "development":
        raise ValueError("legacy episodes are development-only materials")
    raw_files = {}
    for name in ("summary.json", "events.jsonl", "artifacts.jsonl", "checkpoints.jsonl", "verifier_results.jsonl"):
        data = (run_dir / name).read_bytes()
        raw_files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "base64": base64.b64encode(data).decode("ascii"), "records": _records(data) if name.endswith(".jsonl") else _loads(data)}
    summary = raw_files["summary.json"]["records"]
    _object(summary, {"episode_id", "task_id"})
    record = {
        "schema_version": SCHEMA_VERSION, "episode_id": summary["episode_id"],
        "task_id": summary["task_id"], "task_family_id": task_family_id, "system_id": system_id,
        "split": split, "status": "unknown", "outcome": dict.fromkeys(OUTCOME_FIELDS),
        "events": raw_files["events.jsonl"]["records"], "artifact_versions": raw_files["artifacts.jsonl"]["records"],
        "checkpoint_refs": [{"checkpoint_id": item["checkpoint_id"], "raw": item, "replay_qualified": False,
            "reason": "legacy partial file hashes/history; no verified complete state snapshot"} for item in raw_files["checkpoints.jsonl"]["records"]],
        "provenance": {"source": "legacy_officeval", "version": "legacy-import-v1", "is_synthetic": False,
                       "source_run_dir": str(run_dir), "raw_files": raw_files},
    }
    return validate_trajectory(record)
