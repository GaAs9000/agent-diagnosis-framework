from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from c3_sandbox.r1 import officeval
from c3_sandbox.r1.officeval_selection import (
    SUPPORTED_TASK_IDS, freeze_selection, inspect_selection, static_inventory,
)

COUNTS = {
    "officeval_010": (32, 90), "officeval_012": (48, 58), "officeval_015": (45, 127),
    "officeval_018": (22, 55), "officeval_021": (30, 80), "officeval_031": (11, 21),
    "officeval_033": (8, 17), "officeval_036": (11, 18), "officeval_037": (21, 40),
}


def complete_raw(task):
    items = [{"rule": r["rule"], "max_delta": r["max_delta"],
              "delta": max(r["max_delta"], 0), "hit": r["max_delta"] > 0, "detail": ""}
             for r in task["expected_rules"]]
    return {"id": task["official_id"], "file_name": task["expected_output"],
            "status": "ok", "error": None, "dim1_pass": True, "dim1_reason": "",
            "max_score": task["official_max_score"], "total_score": task["completion_threshold"],
            "dim2_items": items}


@pytest.fixture(scope="module")
def review():
    return inspect_selection(SUPPORTED_TASK_IDS)


@pytest.mark.parametrize("task_id", SUPPORTED_TASK_IDS)
def test_pinned_ast_inventory(review, task_id):
    task = review[task_id]
    assert task["eligible"], task
    count, maximum = COUNTS[task_id]
    assert len(task["expected_rules"]) == count
    assert task["max_score"] == task["completion_threshold"] == maximum
    assert task["official_max_score"] == maximum
    task = {**task, "official_id": task_id.removeprefix("officeval_")}
    raw = complete_raw(task)
    assert officeval.normalize_official(raw, task_id, task)["outcome"] == "completed"
    raw["dim2_items"][0].update(hit=False, delta=0)
    raw["total_score"] -= raw["dim2_items"][0]["max_delta"]
    assert officeval.normalize_official(raw, task_id, task)["outcome"] == "task_failed"


def test_duplicate_rules_preserved(review):
    task = {**review["officeval_015"], "official_id": "015"}
    counts = Counter((r["rule"], r["max_delta"]) for r in task["expected_rules"])
    assert counts[("答题卡第一页左侧有四个文本框内容", 3)] == 2
    raw = complete_raw(task)
    raw["dim2_items"].pop(16)
    raw["total_score"] -= 3
    assert officeval.normalize_official(raw, "officeval_015", task)["reason"] == "unexpected_rule_inventory"


def test_official_signed_maximum_is_distinct_from_completion():
    task = {"official_id": "015", "expected_output": "a.docx", "max_score": 3,
            "completion_threshold": 3, "official_max_score": 1,
            "expected_rules": [{"rule": "positive", "max_delta": 3}, {"rule": "penalty", "max_delta": -2}]}
    raw = complete_raw(task)
    untouched = deepcopy(raw)
    assert officeval.normalize_official(raw, "officeval_015", task)["outcome"] == "completed"
    assert raw == untouched
    raw["max_score"] = 3
    assert officeval.normalize_official(raw, "officeval_015", task)["reason"] == "inconsistent_total_score"


def test_negative_hit_prevents_completion(review):
    task = {**review["officeval_033"], "official_id": "033"}
    raw = complete_raw(task)
    item = next(i for i in raw["dim2_items"] if i["max_delta"] < 0)
    item.update(hit=True, delta=item["max_delta"])
    raw["total_score"] += item["delta"]
    assert officeval.normalize_official(raw, "officeval_033", task)["outcome"] == "task_failed"


def test_failed_branch_alias_cannot_claim_success(review):
    task = {**review["officeval_037"], "official_id": "037"}
    alias = task["failed_rule_aliases"][0]
    raw = complete_raw(task)
    item = next(i for i in raw["dim2_items"] if i["rule"] == alias["canonical_rule"])
    item.update(rule=alias["alias"], hit=False, delta=0)
    raw["total_score"] -= item["max_delta"]
    result = officeval.normalize_official(raw, "officeval_037", task)
    assert result["validity"] == "valid" and result["outcome"] == "task_failed"
    item.update(hit=True, delta=item["max_delta"])
    raw["total_score"] += item["max_delta"]
    assert officeval.normalize_official(raw, "officeval_037", task)["reason"] == "invalid_failed_alias_score"


def test_missing_render_assessment_never_passes(review):
    task = {**review["officeval_010"], "official_id": "010", "required_unassessed_rules": ["render"]}
    result = officeval.normalize_official(complete_raw(task), "officeval_010", task)
    assert result["c3_operational_completed"] is None
    assert result["coverage_complete"] is False
    assert result["unassessed_rules"] == ["render"]


@pytest.mark.parametrize("task_id", ["officeval_002", "officeval_003", "officeval_006", "officeval_024", "officeval_030", "officeval_035", "officeval_038"])
def test_unsupported_reported_before_freeze(tmp_path, task_id):
    report = inspect_selection([task_id])
    assert report[task_id]["eligible"] is False
    with pytest.raises(ValueError, match="unsupported selection"):
        freeze_selection(tmp_path / "frozen", [task_id], tmp_path / "absent-assets")
    assert not (tmp_path / "frozen").exists()


def test_ast_does_not_execute_verifier(tmp_path):
    source = officeval.SOURCE_ROOT / "verifiers/officeval_015_verifier.py"
    target = tmp_path / "trap.py"
    target.write_text(source.read_text() + "\nraise RuntimeError('must never execute')\n")
    assert len(static_inventory("officeval_015", target)["expected_rules"]) == 45


def test_freeze_compatible_shape_order_and_readonly(tmp_path):
    assets = tmp_path / "assets"
    ids = ("officeval_015", "officeval_037")
    for task_id in ids:
        task = json.loads((officeval.SOURCE_ROOT / "data/tasks" / f"{task_id}.json").read_text())
        (assets / task_id).mkdir(parents=True)
        for entry in task["origin_files"]:
            (assets / task_id / entry["dest"]).write_bytes(b"test public asset, never scored")
    manifest = freeze_selection(tmp_path / "frozen", ids, assets)
    assert tuple(manifest["tasks"]) == ids
    assert manifest["source_commit"] == officeval.SOURCE_COMMIT
    assert manifest["tasks"]["officeval_037"]["expected_output"] == "澜桥市高三英语试题.docx"
    for task_id, task in manifest["tasks"].items():
        assert task["exact_output_name"] is True
        assert officeval.task_public_view(manifest, task_id)["expected_output"] == task["expected_output"]
        assert "expected_rules" not in officeval.task_public_view(manifest, task_id)
        assert task["official_id"] == task_id.removeprefix("officeval_")
    for relative, metadata in manifest["files"].items():
        file = Path(manifest["bundle_root"]) / relative
        assert hashlib.sha256(file.read_bytes()).hexdigest() == metadata["sha256"]
        assert file.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError):
        freeze_selection(tmp_path / "frozen", ids, assets)


@pytest.mark.parametrize("ids", [[], list(SUPPORTED_TASK_IDS) * 2, ["officeval_010"] * 2])
def test_invalid_pool_size_or_duplicates(tmp_path, ids):
    with pytest.raises(ValueError, match="one to twelve"):
        freeze_selection(tmp_path / "frozen", ids, tmp_path)


def test_missing_asset_is_atomic(tmp_path):
    with pytest.raises(ValueError, match="missing, empty"):
        freeze_selection(tmp_path / "frozen", ["officeval_010"], tmp_path / "absent")
    assert not (tmp_path / "frozen").exists()
    assert not list(tmp_path.glob(".officeval-selection-*"))


def test_dirty_source_rejected(tmp_path, monkeypatch):
    from c3_sandbox.r1 import officeval_selection
    def dirty_git(args, **kwargs):
        return officeval.SOURCE_COMMIT if args[-2:] == ["rev-parse", "HEAD"] else " M verifier.py"
    monkeypatch.setattr(officeval_selection.subprocess, "check_output", dirty_git)
    with pytest.raises(ValueError, match="clean at pinned"):
        freeze_selection(tmp_path / "frozen", ["officeval_010"], tmp_path)
