"""Synthetic exporter tests; no model or container execution."""
import base64
import copy
import hashlib
import json

import pytest

from c3_sandbox.manifests import canonical_hash, validate_family_manifest, validate_system_manifest
from c3_sandbox.r1.trajectory_export import export_baseline, export_manifests
from c3_sandbox.trajectory import validate_trajectory


def fixture(tmp_path, task="officeval_002"):
    source = tmp_path / task
    (source / "agent").mkdir(parents=True)
    events = [{"event_index": 0, "type": "synthetic", "unknown": [1, 2]}]
    (source / "agent/events.jsonl").write_bytes(b' {"event_index":0,"type":"synthetic","unknown":[1,2]}\r\n\n')
    request = {"type": "request", "messages": [{"role": "system", "content": "fixed"},
        {"role": "user", "content": task}], "provider": {"model": "local-synthetic"},
        "config": {"temperature": 0, "seed": 1}}
    (source / "agent/model_calls.jsonl").write_text(json.dumps(request) + "\n")
    for name in ["public-input", "final-workspace", "checkpoints/c/workspace"]:
        directory = source / name
        directory.mkdir(parents=True)
        (directory / "document.txt").write_bytes(b'original bytes' if name == "public-input" else b'changed bytes')
    (source / "final-grade").mkdir()
    (source / "final-grade/grade.json").write_bytes(b'{"raw_score":0,"unknown":"retain"}\r\n')
    (source / "final-grade/stderr.txt").write_bytes(b'raw stderr\x00\xff')
    result = {"output_dir": str(source), "task_id": task, "is_synthetic": True,
              "system_identity": "1" * 64, "execution_valid": True, "target_resolved": False,
              "regression_free": True, "task_completed": False, "outcome": "failure",
              "agent_result": {"events": events}, "runtime": {"model_config": {"model": "local-synthetic"},
              "tool_config": {"schemas": []}, "image_digest": "sha256:fixed", "config": {"network": "none"},
              "environment": {"LANG": "C"}}, "checkpoints": [{"checkpoint_id": "2" * 64,
              "runtime_identity": "3" * 64, "event_index": 0, "workspace": str(source / "checkpoints/c/workspace")}]}
    (source / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    report = {"checkpoint_identity": "2" * 64, "runtime_identity": "3" * 64,
              "system_identity": "1" * 64, "is_synthetic": True, "eligible_for_intervention": True,
              "admission_receipt": "4" * 64, "reasons": []}
    return result, report


def test_export_lossless_bytes_artifacts_and_exact_qualification(tmp_path):
    result, report = fixture(tmp_path)
    source = tmp_path / result["task_id"]
    before = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    record = export_baseline(result, tmp_path / "export", report)
    assert validate_trajectory(record) == record
    for name, raw in record["provenance"]["raw_files"].items():
        assert base64.b64decode(raw["base64"]) == before[name]
        assert raw["sha256"] == hashlib.sha256(before[name]).hexdigest()
    stream = record["provenance"]["raw_stream_files"]["final-grade/stderr.txt"]
    assert base64.b64decode(stream["base64"]) == before["final-grade/stderr.txt"]
    assert record["events"][0]["unknown"] == [1, 2]
    assert record["checkpoint_refs"][0]["qualification_report_ref"] == canonical_hash(report)
    assert record["checkpoint_refs"][0]["replay_qualified"] is True
    assert {v["version"] for v in record["artifact_versions"]} == {"initial", "final", "checkpoint:" + "2" * 64}
    assert before == {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}


def test_audit_override_preserves_erroneously_valid_original(tmp_path):
    result, report = fixture(tmp_path)
    record = export_baseline(result, tmp_path / "excluded", report, exclusion_reason="native protocol contradiction")
    assert record["status"] == "unknown"
    assert record["outcome"] == {"execution_valid": False, "target_resolved": None,
                                  "regression_free": None, "task_completed": None}
    assert record["provenance"]["raw_files"]["result.json"]["records"]["execution_valid"] is True
    assert record["checkpoint_refs"][0]["replay_qualified"] is False


def test_nonadmitted_and_wrong_checkpoint_reports(tmp_path):
    result, report = fixture(tmp_path)
    unqualified = export_baseline(result, tmp_path / "no-report")
    assert unqualified["checkpoint_refs"][0]["replay_qualified"] is False
    report["checkpoint_identity"] = "9" * 64
    with pytest.raises(ValueError, match="exact checkpoint"):
        export_baseline(result, tmp_path / "wrong", report)
    report["checkpoint_identity"] = "2" * 64
    report["receipt_identity"] = "5" * 64
    with pytest.raises(ValueError, match="receipt identities"):
        export_baseline(result, tmp_path / "wrong-receipt", report)


def test_reject_modified_result_and_source_destination(tmp_path):
    result, _ = fixture(tmp_path)
    changed = copy.deepcopy(result)
    changed["outcome"] = "success"
    with pytest.raises(ValueError, match="differs"):
        export_baseline(changed, tmp_path / "tampered")
    with pytest.raises(ValueError, match="immutable"):
        export_baseline(result, tmp_path / result["task_id"] / "export")


def test_manifests_single_frozen_system_development_exclusions(tmp_path):
    baselines, tasks = [], {}
    for task in ["officeval_002", "officeval_003"]:
        result, _ = fixture(tmp_path, task)
        baselines.append(export_baseline(result, tmp_path / (task + "-export")))
        tasks[task] = {"input_files": [{"name": "document.txt", "path": "inputs/" + task + "/document.txt",
                       "sha256": hashlib.sha256(b'original bytes').hexdigest()}]}
    frozen = {"source_commit": "fixed", "tasks": tasks}
    exported = export_manifests(baselines, frozen, tmp_path / "manifests")
    systems, families = exported["system_manifest"], exported["family_manifest"]
    assert validate_system_manifest(systems) == systems
    assert validate_family_manifest(families) == families
    assert len(systems["systems"]) == 1
    assert systems["study_scope"] == "single-model development study"
    assert all(f["split"] == "development" and f["excluded_from_future_confirmation"] for f in families["families"])
    assert families["research_family_independence_established"] is False
    bad = copy.deepcopy(frozen)
    bad["tasks"]["officeval_002"]["input_files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Initial artifact"):
        export_manifests(baselines, bad, tmp_path / "bad-hash")
    changed = copy.deepcopy(baselines)
    changed[1]["system_id"] = "5" * 64
    with pytest.raises(ValueError, match="identical frozen"):
        export_manifests(changed, frozen, tmp_path / "bad-system")
