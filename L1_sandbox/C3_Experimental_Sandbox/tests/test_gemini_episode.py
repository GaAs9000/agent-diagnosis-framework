import json
from pathlib import Path
import pytest

from c3_sandbox.integrated.gemini_episode import run_gemini_episode_sync


def reply(parts, usage=None):
    value = {"candidates": [{"content": {"parts": parts}}]}
    if usage:
        value["usageMetadata"] = usage
    return json.dumps(value).encode()


class FakeRuntime:
    started = 0
    def __init__(self, image, run_dir):
        self.run_dir = Path(run_dir); self.workspace = self.run_dir / "workspace"
    def __enter__(self):
        type(self).started += 1; self.workspace.mkdir(parents=True); return self
    def __exit__(self, *args): pass
    def seed_workspace(self, source):
        self.workspace.joinpath("input.docx").write_bytes((Path(source) / "input.docx").read_bytes())
    def _snapshot(self): return {"version": 1, "root_mode": 0o700, "entries": [] , "total_bytes": 0}
    def exec_python(self, code): return {"exit_code": 0, "stdout": "ok", "stderr": "", "error": None, "truncated": False, "stdout_bytes": 2, "stderr_bytes": 0}
    def export_workspace(self, destination):
        Path(destination).mkdir(parents=True)
        (Path(destination) / "input.docx").write_bytes(b"x")


class FakeRecorder:
    def __init__(self, actor, root): self.records = []; Path(root).mkdir(parents=True)
    async def __call__(self, event):
        if event["stage"] == "end": self.records.append({"tool_call_id": event["tool_call_id"], "raw_result": event["raw_result"]})


def bundle(tmp_path):
    root = tmp_path / "bundle"; (root / "inputs").mkdir(parents=True)
    (root / "inputs" / "input.docx").write_bytes(b"x")
    return {"bundle_root": str(root), "task_ids": ["officeval_002"], "tasks": {"officeval_002": {
        "metadata": {"task_prompt": "edit", "input_files": [{"name": "input.docx", "path": "inputs/input.docx"}], "expected_output": "input.docx"}
    }}}


def grade(*args):
    Path(args[-1]).mkdir(parents=True)
    return {"validity": "valid", "c3_operational_completed": True, "outcome": "success",
            "official_raw": {"dim1_pass": True, "dim2_items": []}, "unassessed_rules": []}


def test_gemini_native_episode_preserves_records_and_dispatch(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret")
    replies = iter([reply([{"functionCall": {"name": "workspace_python", "args": {"code": "print(1)"}}}]),
                    reply([{"text": "done"}], {"totalTokenCount": 9})])
    result = run_gemini_episode_sync(bundle(tmp_path), "officeval_002", agent_image="agent", grader_image="grader",
        output_dir=tmp_path / "run", transport=lambda raw: next(replies), runtime_factory=FakeRuntime,
        grade_fn=grade, dispatch_factory=FakeRecorder)
    assert result["provider"] == "google-gemini" and result["model"] == "gemini-3.5-flash-lite"
    assert result["execution_valid"] is True and result["task_completed"] is True
    assert len(result["raw_gemini_records"]) == 2 and len(result["tool_dispatch"]) == 1
    assert "secret" not in (tmp_path / "run" / "result.json").read_text()


def test_gemini_missing_key_fails_before_runtime(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False); monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    FakeRuntime.started = 0
    with pytest.raises(RuntimeError, match="API key"):
        run_gemini_episode_sync(bundle(tmp_path), "officeval_002", agent_image="agent", grader_image="grader",
            output_dir=tmp_path / "run", runtime_factory=FakeRuntime, grade_fn=grade)
    assert FakeRuntime.started == 0 and not (tmp_path / "run").exists()
