import asyncio
import json
from pathlib import Path
import pytest

from c3_sandbox.r1 import office_run as run
from c3_sandbox.r1.docker_runtime import restore_snapshot, snapshot_workspace

IMAGE = "sha256:" + "a" * 64
GRADER = "sha256:" + "b" * 64


def manifest(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "original.docx").write_bytes(b"original")
    (bundle / "reference.pdf").write_bytes(b"reference")
    files = [{"name": name, "path": name, "sha256": run._sha(bundle / name)}
             for name in ("original.docx", "reference.pdf")]
    return {"bundle_root": str(bundle), "tasks": {"officeval_003": {
        "official_id": "003", "expected_output": "original.docx",
        "task_prompt": "Remove blank rows.", "input_files": files,
        "expected_rules": [{"rule": "positive", "max_delta": 5},
                           {"rule": "penalty", "max_delta": -5}],
        "max_score": 5, "unassessed_aliases": []}}}


def official(hit=False, penalty=False):
    return {"status": "ok", "id": "003", "error": None,
            "dim1_pass": True, "dim1_reason": "ok", "file_name": "original.docx",
            "total_score": 5 * hit - 5 * penalty, "max_score": 5,
            "dim2_items": [{"rule": "positive", "max_delta": 5,
                            "delta": 5 * hit, "hit": hit, "detail": "ok"},
                           {"rule": "penalty", "max_delta": -5,
                            "delta": -5 * penalty, "hit": penalty, "detail": "ok"}]}


def receipt():
    return {"status": "passed", "is_synthetic": False,
            "provider": "openai-api/mlx/default_model", "model": run.MODEL_REPO,
            "model_fingerprint": "c" * 64, "native_tool_calls": 1,
            "prompt_tokens": 100, "completion_tokens": 20, "model_calls": 2,
            "tool_call_id": "nonce-read", "observation_tool_call_id": "nonce-read",
            "nonce": "abc", "observed_nonce": "abc", "independent_nonce": "abc"}


@pytest.fixture
def fake_runtime(monkeypatch, tmp_path):
    states = {"active": [], "grader_inputs": [], "actors_closed": 0}
    class Runtime:
        def __init__(self, image_id, run_dir, *, is_grader=False):
            self.image_id, self.run_dir, self.is_grader = image_id, Path(run_dir), is_grader
            self.workspace = self.run_dir / "fake-workspace"
            self.workspace.mkdir(parents=True)

        def __enter__(self):
            if self.is_grader:
                assert not any(not actor.is_grader for actor in states["active"])
            states["active"].append(self)
            return self

        def __exit__(self, *_):
            states["active"].remove(self)
            if not self.is_grader:
                states["actors_closed"] += 1

        def seed_workspace(self, source):
            restore_snapshot(snapshot_workspace(source), self.workspace)

        def _inspect(self):
            return {"Config": {"Env": ["HOME=/tmp", "PYTHONDONTWRITEBYTECODE=1"]}}

        def _snapshot(self):
            return snapshot_workspace(self.workspace)

        def export_workspace(self, destination):
            restore_snapshot(snapshot_workspace(self.workspace), destination)
            return {"destination": str(destination), "snapshot_sha256": "a" * 64}

        def audit_boundary(self):
            return {"qualified": False, "unexpected_processes": [{"pid": 88}]}

        def exec_python(self, code):
            # The test tool changes bytes only when the fake model requests it.
            assert code == "edit"
            (self.workspace / "original.docx").write_bytes(b"edited")
            (self.workspace / "evil.py").write_text("raise RuntimeError('must never be imported')")
            return {"exit_code": 0, "stdout": "x" * 9000, "stderr": "", "error": None}

        def exec_fixed(self, argv):
            assert argv == ["python", "-I", "/opt/grader/r1_grader.py", "officeval_003", "/workspace/submission"]
            entries = list((self.workspace / "submission").iterdir())
            states["grader_inputs"].append([p.name for p in entries])
            if not entries:
                grade = {"official_raw": None, "validity": "valid", "c3_operational_completed": False,
                         "outcome": "task_failed", "reason": "missing_deliverable",
                         "coverage_complete": False, "unassessed_rules": []}
            else:
                raw = official(hit=entries[0].read_bytes() == b"edited")
                grade = run.normalize_official(raw, "officeval_003", states["manifest"]["tasks"]["officeval_003"])
            return {"exit_code": 0, "error": None, "truncated": False,
                    "stdout": json.dumps(grade), "stderr": ""}
    monkeypatch.setattr(run, "DockerRuntime", Runtime)
    return states


def test_grader_delivers_only_original_name_and_missing_is_failure(tmp_path, fake_runtime):
    spec = manifest(tmp_path)
    fake_runtime["manifest"] = spec
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "renamed.docx").write_bytes(b"edited")
    (workspace / "evil.py").write_text("raise AssertionError")
    grade = run.grade_workspace("officeval_003", spec, workspace, GRADER, tmp_path / "grade")
    assert grade["validity"] == "valid"
    assert grade["reason"] == "missing_deliverable"
    assert grade["c3_operational_completed"] is False
    assert fake_runtime["grader_inputs"] == [[]]
    assert grade["stdout_sha256"] == run._sha(tmp_path / "grade/stdout.txt")
    with pytest.raises(FileExistsError):
        run.grade_workspace("officeval_003", spec, workspace, GRADER, tmp_path / "grade")


def test_baseline_checkpoint_retains_raw_and_stays_unadmitted(tmp_path, monkeypatch, fake_runtime):
    spec = manifest(tmp_path)
    fake_runtime["manifest"] = spec
    class Agent:
        def __init__(self, config, run_dir):
            assert config["temperature"] == 0 and config["max_tokens"] == 2048
            self.config = config
        async def run(self, messages, tools, schemas, *, max_turns, checkpoint_callback, dispatch_callback):
            dispatch_event = {"stage": "start", "turn": 0, "tool_call_id": "edit-call",
                              "function": "workspace_python", "arguments": {"code": "edit"},
                              "raw_result": None, "error": None}
            await dispatch_callback(dispatch_event)
            raw = tools["workspace_python"]("edit")
            await dispatch_callback({**dispatch_event, "stage": "end", "raw_result": raw})
            visible = self.config["tool_result_transform"]("workspace_python", {"code": "edit"}, raw)
            messages += [{"role": "assistant", "content": "", "tool_calls": [
                {"id": "edit-call", "function": "workspace_python", "arguments": {"code": "edit"}, "type": "function"}]},
                {"role": "tool", "content": visible, "function": "workspace_python", "tool_call_id": "edit-call"}]
            await checkpoint_callback(messages, 1)
            messages.append({"role": "assistant", "content": "done"})
            await checkpoint_callback(messages, 2)
            return {"messages": messages, "events": [{"type": "tool_result", "raw_result": raw}],
                    "tokens": {"output_tokens": 15}, "modelcall_count": 2, "stopped_reason": "assistant_finished"}
    monkeypatch.setattr(run, "NativeAgent", Agent)
    result = asyncio.run(run.run_baseline("officeval_003", spec, IMAGE, GRADER, receipt(), tmp_path / "run"))
    assert result["outcome"] == "success" and result["task_completed"] is True
    assert result["initial_grade"]["c3_operational_completed"] is False
    assert fake_runtime["grader_inputs"] == [["original.docx"], ["original.docx"]]
    assert fake_runtime["actors_closed"] == 1
    assert len(result["checkpoints"]) == 1
    checkpoint = result["checkpoints"][0]
    assert checkpoint["qualification"]["eligible_for_intervention"] is False
    assert "real replay blocked" in " ".join(checkpoint["qualification"]["reasons"])
    record = json.loads(Path(checkpoint["checkpoint_record_path"]).read_text())
    assert record["is_synthetic"] is False
    assert record["runtime"]["environment"] == {"HOME": "/tmp", "PYTHONDONTWRITEBYTECODE": "1"}
    assert Path(checkpoint["workspace"], "evil.py").is_file()
    assert len(result["agent_result"]["events"][0]["raw_result"]["stdout"]) == 9000
    assert json.loads(record["messages"][-1]["content"])["context_output_truncated"] is True
    assert (tmp_path / "run/public-input/original.docx").stat().st_mode & 0o777 == 0o644
    assert (tmp_path / "run/public-input").stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize("field,value", [("status", "failed"), ("native_tool_calls", 0),
                                         ("model", "other"), ("is_synthetic", True)])
def test_bad_qualification_rejected_before_output(tmp_path, field, value):
    qualification = receipt()
    qualification[field] = value
    with pytest.raises(ValueError):
        asyncio.run(run.run_baseline("officeval_003", {}, IMAGE, GRADER, qualification, tmp_path / "run"))
    assert not (tmp_path / "run").exists()


def test_outcome_does_not_use_zero_score_threshold():
    task = {"official_id": "003", "max_score": 5, "unassessed_aliases": [],
            "expected_rules": [{"rule": "positive", "max_delta": 5}, {"rule": "penalty", "max_delta": -5}]}
    grade = run.normalize_official(official(), "officeval_003", task)
    result = run._outcome(grade, grade, {"stopped_reason": "assistant_finished"})
    assert result["outcome"] == "failure" and result["task_completed"] is False
    grade["c3_operational_completed"] = None
    grade["unassessed_rules"] = ["penalty"]
    assert run._outcome(grade, grade, {"stopped_reason": "assistant_finished"})["regression_free"] is None
    invalid = {**grade, "validity": "infrastructure_invalid"}
    assert run._outcome(invalid, grade, {"stopped_reason": "assistant_finished"})["execution_valid"] is False


def test_regression_rejects_lost_existing_positive_with_unchanged_penalties():
    before = {"validity": "valid", "unassessed_rules": [], "official_raw": {
        "dim1_pass": True, "dim2_items": [
            {"rule": "existing-positive", "max_delta": 3, "hit": True},
            {"rule": "penalty", "max_delta": -5, "hit": False}]}}
    after = {"validity": "valid", "c3_operational_completed": False, "unassessed_rules": [],
             "official_raw": {"dim1_pass": True, "dim2_items": [
                 {"rule": "existing-positive", "max_delta": 3, "hit": False},
                 {"rule": "penalty", "max_delta": -5, "hit": False}]}}
    assert run._outcome(before, after, {"stopped_reason": "assistant_finished"})["regression_free"] is False
    after["official_raw"]["dim2_items"][0]["hit"] = True
    assert run._outcome(before, after, {"stopped_reason": "assistant_finished"})["regression_free"] is True
    after["official_raw"].update(dim1_pass=False, dim2_items=[])
    assert run._outcome(before, after, {"stopped_reason": "assistant_finished"})["regression_free"] is False
    after["unassessed_rules"] = ["unassessed"]
    assert run._outcome(before, after, {"stopped_reason": "assistant_finished"})["regression_free"] is None


@pytest.mark.parametrize("prior_hits,current_hits,expected", [
    ([True, True], [True, False], False),
    ([True, False], [False, True], False),
    ([True, False], [True, True], True),
    ([True, True], [True, True], True),
])
def test_regression_preserves_each_duplicate_positive_occurrence(prior_hits, current_hits, expected):
    def grade(hits):
        return {"validity": "valid", "unassessed_rules": [], "c3_operational_completed": False,
                "official_raw": {"dim1_pass": True, "dim2_items": [
                    {"rule": "答题卡第一页左侧有四个文本框内容", "max_delta": 3, "hit": hit}
                    for hit in hits]}}
    result = run._outcome(grade(prior_hits), grade(current_hits), {"stopped_reason": "assistant_finished"})
    assert result["regression_free"] is expected
    assert result["execution_valid"] is True and result["task_completed"] is False
    assert result["target_resolved"] is all(current_hits)


@pytest.mark.parametrize("prior_hits,current_hits,expected", [
    ([True, False], [False, True], False),
    ([False, True], [True, True], False),
    ([True, True], [False, True], True),
    ([True, False], [True, False], True),
])
def test_regression_distinguishes_duplicate_negative_occurrences(prior_hits, current_hits, expected):
    def grade(hits):
        return {"validity": "valid", "unassessed_rules": [], "c3_operational_completed": False,
                "official_raw": {"dim1_pass": True, "dim2_items": [
                    {"rule": "same penalty", "max_delta": -3, "hit": hit} for hit in hits]}}
    assert run._outcome(grade(prior_hits), grade(current_hits),
                        {"stopped_reason": "assistant_finished"})["regression_free"] is expected


def test_regression_rule_identity_includes_weight():
    def grade(first, second):
        return {"validity": "valid", "unassessed_rules": [], "c3_operational_completed": False,
                "official_raw": {"dim1_pass": True, "dim2_items": [
                    {"rule": "same label", "max_delta": 3, "hit": first},
                    {"rule": "same label", "max_delta": 5, "hit": second}]}}
    assert run._outcome(grade(True, False), grade(False, True),
                        {"stopped_reason": "assistant_finished"})["regression_free"] is False
