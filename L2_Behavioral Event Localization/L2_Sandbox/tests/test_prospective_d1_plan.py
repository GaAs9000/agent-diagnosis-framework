"""Prospective D1 input changes fail before any execution adapter can run."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("prospective_d1_ladder", ROOT / "scripts/v3_d1_ladder.py")
ladder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ladder)


def test_partial_prospective_arguments_reject_before_sampling(tmp_path):
    with pytest.raises(ValueError, match="requires cohort"):
        ladder._prospective_inputs(runs_dir=tmp_path, diag_dir=tmp_path,
            cohort_path=tmp_path / "cohort.json", collection_plan_path=None,
            prospective_freeze_path=None, e1_verification_path=None)


def test_changed_bound_data_rejected(tmp_path):
    path = tmp_path / "facts.jsonl"
    path.write_text("original")
    plan = {"prospective_binding": {"inputs": {str(path): ladder.file_digest(path)}}}
    ladder.verify_prospective_plan_inputs(plan)
    path.write_text("different")
    with pytest.raises(ValueError, match="input changed"):
        ladder.verify_prospective_plan_inputs(plan)


def test_changed_source_rejected_without_git_head_dependency(tmp_path):
    path = tmp_path / "source.py"
    path.write_text("pass\n")
    plan = {"prospective_binding": {"inputs": {str(path): ladder.file_digest(path)}}}
    ladder.verify_prospective_plan_inputs(plan)
    path.write_text("raise RuntimeError\n")
    with pytest.raises(ValueError, match="input changed"):
        ladder.verify_prospective_plan_inputs(plan)


@pytest.mark.parametrize("simulation", [False, True])
def test_prospective_plan_cannot_execute_even_with_adapter(tmp_path, simulation):
    called = []
    plan = {"prospective_binding": {"inputs": {}}, "execution_ready": False,
            "execution_authorized": False, "mode": "plan_only"}
    with pytest.raises(ValueError, match="plan-only"):
        ladder._execute_d1_locked(plan, runs_dir=tmp_path, receipts_dir=tmp_path,
            outcomes_out=tmp_path / "outcomes.json", freeze={},
            branch_runner=lambda *a: called.append(a), simulate_offline=simulation)
    assert called == []
    assert not (tmp_path / "admission.json").exists()


def test_incomplete_or_redefined_pool_rejected(tmp_path):
    collection = ladder.seal({"slots": []}, "plan_sha256")
    paths = []
    for name, doc in (("cohort", {"slots": {}}), ("collection", collection), ("freeze", {}), ("e1", {})):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(doc))
        paths.append(path)
    with pytest.raises(ValueError, match="exactly DEV9"):
        ladder._prospective_inputs(runs_dir=tmp_path, diag_dir=tmp_path,
            cohort_path=paths[0], collection_plan_path=paths[1],
            prospective_freeze_path=paths[2], e1_verification_path=paths[3])


@pytest.fixture
def full_pool(tmp_path, monkeypatch):
    monkeypatch.setattr(ladder, "REPO_ROOT", tmp_path)
    runs = tmp_path / "runs/fresh/episodes"
    diag = tmp_path / "diag"
    phase = tmp_path / "phase"
    phase.mkdir()
    (phase / "p1_controller_image_pin").mkdir()
    (phase / "p1_controller_image_pin/controller_source.tar").write_bytes(b"archive")
    (phase / "e1").mkdir()
    manifest = phase / "e1/manifest.json"
    manifest.write_text("{}")
    runtime = tmp_path / "runtime.json"
    runtime.write_text("{}")
    configs = {f"officeval_{n}": {"task": n} for n in ("003", "004", "010", "033", "050", "060", "070", "085", "090")}
    slots, e1rows = {}, []
    def put(path, doc):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc))
        return ladder.file_digest(path)
    for task, config in configs.items():
        for rep in range(5):
            label = f"{task}.r{rep:02d}"
            ep, out = runs / label, diag / label
            hashes = {key: put(ep / filename, doc) for filename, key, doc in [
                ("manifest.json", "manifest_sha256", {"episode_id": label}),
                ("evidence_index.json", "index_file_sha256", {}),
                ("raw/resolved_config.json", "config_sha256", config)]}
            put(ep / "official_grade.json", {})
            hashes["config_sha256"] = ladder.digest(config)
            slots[label] = dict(hashes, path=str(ep), study_valid=True, status="completed", task_id=task, episode_id=label)
            outputs = {name: put(out / name, {}) for name in ("grades.jsonl", "facts.jsonl", "defects.json")}
            receipt = put(out / "receipt.json", {})
            grading = put(out / "grading_receipt.json", {})
            e1rows.append(dict(slot=label, outputs=outputs, receipt_sha256=receipt, grading_receipt_sha256=grading,
                binding=dict(parent_index_sha256=hashes["index_file_sha256"], manifest_sha256=hashes["manifest_sha256"],
                    config_sha256=hashes["config_sha256"], episode_id=label, source_sha256="source")))
    freeze_path = phase / "freeze.json"
    put(freeze_path, {})
    collection = ladder.seal(dict(slots=list(slots), configs=configs, prospective_freeze_sha256=ladder.file_digest(freeze_path),
        expected_service_identity={"identity_receipt": {"path": str(runtime), "sha256": ladder.file_digest(runtime)}}), "plan_sha256")
    collection_path = phase / "collection.json"
    put(collection_path, collection)
    cohort_path = phase / "cohort.json"
    cohort_sha = put(cohort_path, dict(slots=slots, plan_sha256=collection["plan_sha256"]))
    for row in e1rows:
        row["binding"]["cohort_sha256"] = cohort_sha
    e1_path = phase / "verification.json"
    put(e1_path, dict(full_45_ready=True, cohort_sha256=cohort_sha, slots=e1rows,
        manifest_sha256=ladder.file_digest(manifest)))
    return dict(runs_dir=runs, diag_dir=diag, cohort_path=cohort_path,
                collection_plan_path=collection_path, prospective_freeze_path=freeze_path,
                e1_verification_path=e1_path)


def test_complete_pool_binds_full_configs_and_e1(full_pool):
    binding = ladder._prospective_inputs(**full_pool)
    assert len(binding["slot_ids"]) == 45
    assert len(binding["full_task_configs"]) == 9
    assert binding["dataset_id"].startswith("fresh:")
    assert "20261017" not in binding["dataset_id"]
    ladder.verify_prospective_plan_inputs({"prospective_binding": binding})


@pytest.mark.parametrize("filename", ["facts.jsonl", "grades.jsonl", "defects.json", "receipt.json", "grading_receipt.json"])
def test_e1_modified_after_verification_rejected(full_pool, filename):
    path = full_pool["diag_dir"] / "officeval_003.r00" / filename
    path.write_text("modified")
    with pytest.raises(ValueError, match="hash mismatch"):
        ladder._prospective_inputs(**full_pool)


def test_resolved_config_change_rejected(full_pool):
    path = full_pool["runs_dir"] / "officeval_003.r00/raw/resolved_config.json"
    path.write_text('{"task": "different"}')
    with pytest.raises(ValueError, match="hash mismatch"):
        ladder._prospective_inputs(**full_pool)


def test_canonical_config_digest_is_distinct_from_raw_file_hash(full_pool):
    path = full_pool["runs_dir"] / "officeval_003.r00/raw/resolved_config.json"
    doc = json.loads(path.read_text())
    path.write_text(json.dumps(doc, indent=4) + "\n")
    assert ladder.file_digest(path) != ladder.digest(doc)
    binding = ladder._prospective_inputs(**full_pool)
    assert binding["inputs"][str(path.resolve())] == ladder.file_digest(path)


def test_extra_manifest_episode_cannot_enter_sampling_pool(full_pool):
    extra = full_pool["runs_dir"] / "officeval_003.r99"
    extra.mkdir()
    (extra / "manifest.json").write_text('{"study_valid": true}')
    with pytest.raises(ValueError, match="outside frozen prospective cohort"):
        ladder._prospective_inputs(**full_pool)


def test_v2_gate_cannot_omit_prospective_inputs(tmp_path):
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"policy": "bounded-mechanism-contract-v2"}))
    with pytest.raises(ValueError, match="requires complete prospective inputs"):
        ladder.build_d1_units_and_plan(runs_dir=tmp_path, diag_dir=tmp_path,
            p0a_catalog_path=tmp_path / "catalog.json", gate_audit_path=gate)


def test_v2_policy_cannot_execute_when_prospective_binding_removed(tmp_path):
    plan = {"measurement_gate_binding": {"policy": "bounded-mechanism-contract-v2"},
            "execution_ready": True}
    with pytest.raises(ValueError, match="plan-only"):
        ladder._execute_d1_locked(plan, runs_dir=tmp_path, receipts_dir=tmp_path,
            outcomes_out=tmp_path / "outcomes.json", freeze={})
    assert not (tmp_path / "admission.json").exists()
