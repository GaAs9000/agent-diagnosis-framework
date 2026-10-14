from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from c3_sandbox.r1 import officeval


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    destination = tmp_path_factory.mktemp("official") / "frozen"
    return officeval.freeze_officeval(destination)


def grader_module():
    source = Path(__file__).resolve().parents[1] / "docker/r1_grader.py"
    spec = importlib.util.spec_from_file_location("r1_grader_test", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules["officeval"] = officeval
    spec.loader.exec_module(module)
    return module


def complete_raw(task_id, task):
    return {"id": task["official_id"], "file_name": "candidate.docx", "status": "ok", "error": None,
            "dim1_pass": True, "dim1_reason": "", "max_score": task["max_score"],
            "total_score": task["max_score"], "dim2_items": [
                {**rule, "hit": rule["max_delta"] > 0,
                 "delta": max(rule["max_delta"], 0), "detail": ""} for rule in task["expected_rules"]]}


def test_freeze_real_sources_and_public_view(frozen):
    root = Path(frozen["bundle_root"])
    assert list(frozen["tasks"]) == ["officeval_003", "officeval_002"]
    assert frozen["source_commit"] == officeval.SOURCE_COMMIT
    assert not list(root.rglob(".git"))
    for relative, metadata in frozen["files"].items():
        path = root / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["sha256"]
        assert not path.stat().st_mode & 0o222
    for task_id, maximum in [("officeval_002", 13), ("officeval_003", 15)]:
        assert frozen["tasks"][task_id]["max_score"] == maximum
        public = officeval.task_public_view(frozen, task_id)
        assert set(public) == {"task_prompt", "input_files", "expected_output"}
        assert "expected_rules" not in json.dumps(public)
        assert len(public["input_files"]) == 2
    with pytest.raises(FileExistsError):
        officeval.freeze_officeval(root)


@pytest.mark.parametrize("task_id", officeval.TASK_IDS)
def test_completion_and_legitimate_failure(frozen, task_id):
    task = frozen["tasks"][task_id]
    raw = complete_raw(task_id, task)
    result = officeval.normalize_official(raw, task_id, task)
    assert result["c3_operational_completed"] is True
    assert result["official_raw"] is raw
    raw["dim2_items"][0].update(hit=False, delta=0)
    raw["total_score"] -= task["expected_rules"][0]["max_delta"]
    assert officeval.normalize_official(raw, task_id, task)["outcome"] == "task_failed"


@pytest.mark.parametrize("mutation", ["wrong_type", "missing_rule", "extra_rule", "duplicate_rule", "wrong_score", "missing_field", "penalty_hit", "wrong_id"])
def test_strict_fail_closed(frozen, mutation):
    task_id = "officeval_003"
    task = frozen["tasks"][task_id]
    raw = deepcopy(complete_raw(task_id, task))
    if mutation == "wrong_type":
        raw["dim1_pass"] = "true"
    elif mutation == "missing_rule":
        raw["dim2_items"].pop()
    elif mutation == "extra_rule":
        raw["dim2_items"].append({"rule": "extra", "max_delta": 1, "delta": 0, "hit": False, "detail": ""})
    elif mutation == "duplicate_rule":
        raw["dim2_items"][-1] = deepcopy(raw["dim2_items"][0])
    elif mutation == "wrong_score":
        raw["total_score"] = 14
    elif mutation == "missing_field":
        del raw["dim2_items"][0]["hit"]
    elif mutation == "penalty_hit":
        raw["dim2_items"][-1]["hit"] = True  # Inconsistent delta must be rejected.
    else:
        raw["id"] = "officeval_002"
    result = officeval.normalize_official(raw, task_id, task)
    assert result["validity"] == "infrastructure_invalid"
    assert result["c3_operational_completed"] is False


def test_dimension1_short_circuit(frozen):
    task = frozen["tasks"]["officeval_003"]
    raw = complete_raw("officeval_003", task)
    raw.update(dim1_pass=False, total_score=0, max_score=0, dim2_items=[])
    assert officeval.normalize_official(raw, "officeval_003", task)["outcome"] == "task_failed"
    raw["dim1_pass"] = None
    assert officeval.normalize_official(raw, "officeval_003", task)["validity"] == "infrastructure_invalid"


@pytest.mark.parametrize("filename,content,reason", [(None, None, "missing_deliverable"), ("a.docx", b"", "empty_deliverable"), ("a.py", b"raise Exception()", "invalid_deliverable_inventory")])
def test_candidate_failures(frozen, tmp_path, filename, content, reason):
    if filename:
        (tmp_path / filename).write_bytes(content)
    result = grader_module().grade("officeval_003", tmp_path, Path(frozen["bundle_root"]))
    assert result["reason"] == reason
    assert result["outcome"] == "task_failed"


@pytest.mark.parametrize("task_id", officeval.TASK_IDS)
def test_real_official_initial_asset(frozen, tmp_path, task_id):
    task = frozen["tasks"][task_id]
    original = next(i for i in task["input_files"] if i["name"].endswith(".docx"))
    shutil.copyfile(Path(frozen["bundle_root"]) / original["path"], tmp_path / original["name"])
    result = grader_module().grade(task_id, tmp_path, Path(frozen["bundle_root"]))
    assert isinstance(result["official_raw"], dict)
    assert result["official_raw"]["id"] == task["official_id"]
    assert result["official_raw"]["status"] == "ok"
    assert result["c3_operational_completed"] is False
    if task_id == "officeval_003":
        assert result["validity"] == "valid"
        assert result["outcome"] == "task_failed"
        assert result["coverage_complete"] is False
        assert len(result["unassessed_rules"]) == 1
    else:
        assert result["validity"] == "valid", result


def test_grader_exception_is_infrastructure(frozen, tmp_path):
    result = grader_module().grade("officeval_003", tmp_path, tmp_path / "absent")
    assert result["outcome"] == "infrastructure_invalid"


def test_dirty_source_rejected(tmp_path, monkeypatch):
    def dirty_git(*args, **kwargs):
        return officeval.SOURCE_COMMIT if args[0][-2:] == ["rev-parse", "HEAD"] else " M verifiers/officeval_003_verifier.py"
    monkeypatch.setattr(officeval.subprocess, "check_output", dirty_git)
    with pytest.raises(ValueError, match="clean"):
        officeval.freeze_officeval(tmp_path / "frozen")
    assert not (tmp_path / "frozen").exists()


@pytest.mark.parametrize("known_failure", [False, True])
def test_pinned_unassessed_alias(frozen, known_failure):
    task_id = "officeval_003"
    task = frozen["tasks"][task_id]
    raw = complete_raw(task_id, task)
    alias = task["unassessed_aliases"][0]
    raw["dim2_items"][alias["slot"]]["rule"] = alias["alias"]
    if known_failure:
        raw["dim2_items"][0].update(hit=False, delta=0)
        raw["total_score"] -= 5
    untouched = deepcopy(raw)
    result = officeval.normalize_official(raw, task_id, task)
    assert raw == untouched
    assert result["validity"] == "valid"
    assert result["coverage_complete"] is False
    assert result["unassessed_rules"] == [alias["canonical_rule"]]
    assert result["c3_operational_completed"] is (False if known_failure else None)
    assert result["outcome"] == ("task_failed" if known_failure else "unassessed")


@pytest.mark.parametrize("mutation", ["unknown_alias", "wrong_slot", "alias_hit", "alias_delta", "missing_alias"])
def test_unassessed_alias_rejects_broadening(frozen, mutation):
    task_id = "officeval_003"
    task = frozen["tasks"][task_id]
    raw = complete_raw(task_id, task)
    alias = task["unassessed_aliases"][0]
    item = raw["dim2_items"][alias["slot"]]
    item["rule"] = alias["alias"]
    if mutation == "unknown_alias":
        item["rule"] += "!"
    elif mutation == "wrong_slot":
        raw["dim2_items"][6], raw["dim2_items"][7] = raw["dim2_items"][7], raw["dim2_items"][6]
    elif mutation == "alias_hit":
        item.update(hit=True, delta=-1)
        raw["total_score"] -= 1
    elif mutation == "alias_delta":
        item["delta"] = -1
    else:
        raw["dim2_items"].pop()
    assert officeval.normalize_official(raw, task_id, task)["validity"] == "infrastructure_invalid"


def test_standalone_grader_module_import(tmp_path):
    copied = tmp_path / "officeval.py"
    shutil.copyfile(Path(officeval.__file__), copied)
    code = """
import importlib.util
from pathlib import Path
import sys
spec = importlib.util.spec_from_file_location('standalone_officeval', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.SOURCE_ROOT is None
assert module._workspace_source_root(Path('/opt/grader/officeval.py')) is None
assert callable(module.normalize_official)
try:
    module.freeze_officeval(Path(sys.argv[1]).parent / 'must-not-create')
except RuntimeError as exc:
    assert 'workspace' in str(exc)
else:
    raise AssertionError('freeze outside workspace must fail')
"""
    subprocess.run([sys.executable, "-I", "-c", code, str(copied)], check=True,
                   capture_output=True, text=True)
    assert not (tmp_path / "must-not-create").exists()
