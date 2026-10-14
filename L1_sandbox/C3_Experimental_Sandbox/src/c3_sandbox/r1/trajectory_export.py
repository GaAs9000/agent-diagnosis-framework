"""Read-only R1 evidence export into validated D1/D2 development contracts."""
from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path

from c3_sandbox.manifests import _hash, canonical_hash, validate_family_manifest, validate_system_manifest
from c3_sandbox.trajectory import OUTCOME_FIELDS, _loads, _records, validate_trajectory


def _raw(data: bytes, records) -> dict:
    return {"sha256": hashlib.sha256(data).hexdigest(),
            "base64": base64.b64encode(data).decode("ascii"), "records": records}


def _write(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _files(root: Path, version: str) -> list[dict]:
    records = []
    if not root.is_dir():
        return records
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Snapshot export requires regular files, not symlinks")
        if path.is_file():
            data = path.read_bytes()
            records.append({"version": version, "path": str(path.relative_to(root)),
                            "source_path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
                            "size_bytes": len(data)})
    return records


def export_baseline(result: dict, outputdir: Path, qualification_report: dict | None = None,
                    exclusion_reason: str | None = None) -> dict:
    """Preserve source bytes; audit exclusions override derived endpoints only."""
    source = Path(result["output_dir"]).resolve()
    outputdir = Path(outputdir).resolve()
    if outputdir == source or source in outputdir.parents:
        raise ValueError("Export destination must be outside the immutable source run")
    if exclusion_reason is not None and not exclusion_reason.strip():
        raise ValueError("exclusion_reason must be nonempty")
    raw_files, raw_streams = {}, {}
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError("Source evidence must not contain symlinks")
        if not path.is_file():
            continue
        name = str(path.relative_to(source))
        if path.suffix in {".json", ".jsonl"}:
            data = path.read_bytes()
            raw_files[name] = _raw(data, _records(data) if path.suffix == ".jsonl" else _loads(data))
        elif path.name in {"stdout.txt", "stderr.txt"}:
            # D1 raw_files is JSON-only; retain non-JSON streams independently.
            data = path.read_bytes()
            raw_streams[name] = _raw(data, None)
    required = {"result.json", "agent/events.jsonl", "agent/model_calls.jsonl"}
    if not required <= raw_files.keys():
        raise ValueError("Baseline export requires original result, events and model call files")
    if raw_files["result.json"]["records"] != result:
        raise ValueError("Supplied result differs from immutable result.json")
    events = raw_files["agent/events.jsonl"]["records"]
    if events != result["agent_result"]["events"]:
        raise ValueError("Agent result events differ from original event bytes")
    checkpoints = result.get("checkpoints", [])
    qualified = qualification_report is not None and qualification_report.get("eligible_for_intervention") is True
    if qualified:
        _hash(qualification_report.get("admission_receipt"))
        if qualification_report.get("receipt_identity", qualification_report["admission_receipt"]) != qualification_report["admission_receipt"]:
            raise ValueError("Qualification admission receipt identities disagree")
        if (qualification_report.get("is_synthetic") is not result.get("is_synthetic") or
            qualification_report.get("system_identity") != result["system_identity"] or
            not qualification_report.get("admission_receipt") or qualification_report.get("reasons")):
            raise ValueError("Qualification report is not admitted for this baseline system")
        matching = [cp for cp in checkpoints if cp["checkpoint_id"] == qualification_report.get("checkpoint_identity")]
        if len(matching) != 1 or matching[0]["runtime_identity"] != qualification_report.get("runtime_identity"):
            raise ValueError("Qualification report does not bind an exact checkpoint/runtime")
    refs = []
    artifacts = _files(source / "public-input", "initial") + _files(source / "final-workspace", "final")
    for cp in checkpoints:
        identifier = cp["checkpoint_id"]
        admitted = qualified and not exclusion_reason and identifier == qualification_report["checkpoint_identity"]
        ref = {"checkpoint_id": identifier, "checkpoint_identity": identifier,
               "replay_qualified": bool(admitted), "event_index": cp["event_index"],
               "raw": copy.deepcopy(cp)}
        if admitted:
            ref["qualification_report_ref"] = canonical_hash(qualification_report)
        else:
            ref["reason"] = exclusion_reason or "No matching admitted replay qualification supplied"
        refs.append(ref)
        workspace = Path(cp["workspace"]).resolve()
        if source not in workspace.parents:
            raise ValueError("Checkpoint workspace must be inside its original run")
        artifacts.extend(_files(workspace, "checkpoint:" + identifier))
    outcome = {name: result.get(name) for name in OUTCOME_FIELDS}
    status = result.get("outcome", "unknown")
    if exclusion_reason or outcome["execution_valid"] is not True:
        outcome = dict.fromkeys(OUTCOME_FIELDS)
        outcome["execution_valid"] = False if exclusion_reason or result.get("execution_valid") is False else None
        status = "unknown"
    record = {"schema_version": "1.0", "episode_id": result.get("episode_id", "r1-baseline-" + result["task_id"]),
              "task_id": result["task_id"], "task_family_id": "officeval-provisional-" + result["task_id"],
              "system_id": result["system_identity"], "split": "development", "status": status,
              "outcome": outcome, "events": events, "artifact_versions": artifacts, "checkpoint_refs": refs,
              "provenance": {"source": "r1_native_officeval", "version": "r1-export-v1",
                             "is_synthetic": result["is_synthetic"], "source_run_dir": str(source),
                             "raw_files": raw_files, "raw_stream_files": raw_streams,
                             "qualification_report": copy.deepcopy(qualification_report),
                             "audit_exclusion_reason": exclusion_reason,
                             "source_group_excluded_from_future_confirmation": True}}
    validated = validate_trajectory(record)
    _write(outputdir / "trajectory.json", validated)
    return validated


def export_manifests(baselines: list[dict], frozenmanifest: dict, outdir: Path) -> dict:
    """One frozen system; provisional task groupings are development materials."""
    if not baselines:
        raise ValueError("At least one baseline is required")
    baselines = [validate_trajectory(item) for item in baselines]
    configs, families = [], []
    source_group = "officeval-source-" + frozenmanifest["source_commit"]
    task_ids = set()
    for baseline in baselines:
        if baseline["split"] != "development" or baseline["task_id"] in task_ids:
            raise ValueError("Baselines must be unique development tasks")
        task_ids.add(baseline["task_id"])
        provenance = baseline["provenance"]
        original = provenance["raw_files"]["result.json"]["records"]
        runtime = original["runtime"]
        requests = [item for item in provenance["raw_files"]["agent/model_calls.jsonl"]["records"] if item.get("type") == "request"]
        if not requests:
            raise ValueError("Original model request required for full system configuration")
        request = requests[0]
        configs.append({"model": runtime["model_config"],
                        "agent": {"implementation": "r1.NativeAgent", "protocol": "inspect-native-tools",
                                  "request_provider": request["provider"]},
                        "tools": runtime["tool_config"],
                        "prompt": [message for message in request["messages"] if message["role"] == "system"],
                        "sampling": request["config"],
                        "runtime": {"image_digest": runtime["image_digest"], "config": runtime["config"],
                                    "environment": runtime["environment"]}})
        task = frozenmanifest["tasks"][baseline["task_id"]]
        asset_hashes = {item["path"]: item["sha256"] for item in task["input_files"]}
        observed = {item["path"]: item["sha256"] for item in baseline["artifact_versions"] if item["version"] == "initial"}
        if any(observed.get(item["name"]) != item["sha256"] for item in task["input_files"]):
            raise ValueError("Initial artifact hashes differ from the frozen task manifest")
        families.append({"task_family_id": baseline["task_family_id"], "task_ids": [baseline["task_id"]],
                         "source_group_ids": [source_group], "split": "development", "asset_hashes": asset_hashes,
                         "grouping_status": "provisional task-ID grouping; research-family independence not established",
                         "excluded_from_future_confirmation": True})
    if any(config != configs[0] for config in configs[1:]) or len({item["system_id"] for item in baselines}) != 1:
        raise ValueError("R1 single-model manifest requires identical frozen execution configurations")
    system = {"system_id": baselines[0]["system_id"], "config": configs[0], "config_sha256": canonical_hash(configs[0])}
    systems = validate_system_manifest({"schema_version": "1.0", "study_scope": "single-model development study",
                                        "systems": [system], "frozen_source_manifest_sha256": canonical_hash(frozenmanifest)})
    family_manifest = validate_family_manifest({"schema_version": "1.0", "families": families,
        "excluded_source_group_ids_from_future_confirmation": [source_group],
        "research_family_independence_established": False})
    _write(Path(outdir) / "system_manifest.json", systems)
    _write(Path(outdir) / "family_manifest.json", family_manifest)
    return {"system_manifest": systems, "family_manifest": family_manifest}
