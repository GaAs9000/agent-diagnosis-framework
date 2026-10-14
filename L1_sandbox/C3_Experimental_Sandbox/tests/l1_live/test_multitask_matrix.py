from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import pytest
from c3_sandbox.l1.core.config import resolve
from c3_sandbox.l1.core.contracts import Result
from c3_sandbox.l1.core.doctor import doctor
from c3_sandbox.l1.core.orchestrator import run
from c3_sandbox.l1.benchmarks.officeval.adapter import NativeOfficeVal
from c3_sandbox.l1.storage import read_json
from .conftest import ROOT, ScriptedGemini, FixtureEvaluator

TASKS = [
    ("officeval_003", "docx", "Restructure"),
    ("officeval_033", "docx", "Compute"),
    ("officeval_060", "pptx", "Restructure"),
    ("officeval_070", "pptx", "Beautify"),
    ("officeval_090", "xlsx", "Extract"),
]


class MultitaskFixtureRuntime:
    is_quiescent = True

    def preflight(self):
        return {"adapter": "test_fixture", "image_id": "sha256:" + "0" * 64, "is_test_double": True}

    def qualification(self):
        return {
            "eligible": True,
            "image_id": "sha256:" + "0" * 64,
            "scopes": ["workspace_files", "native_conversation"],
            "reason_codes": [],
            "test_only": True,
        }

    def execute(self, call, workspace, timeout):
        script = (
            "from pathlib import Path; import shutil; root=Path('.'); "
            "inputs = [p for p in (root/'input').iterdir() if p.is_file()]; "
            "shutil.copyfile(inputs[0], root/'submission'/inputs[0].name) if inputs else None; "
            "print('multitask fixture ok')"
        )
        p = subprocess.run([sys.executable, "-c", script], cwd=workspace, capture_output=True, check=True)
        return Result(
            {"stdout": p.stdout.decode(), "stderr": "", "exit_code": 0, "output_complete": True},
            p.stdout,
            b"",
            provenance={"test_fixture": True, "quiescent": True},
        )


@pytest.mark.parametrize("task_id,doc_ext,intent", TASKS)
def test_multitask_config_and_doctor_without_core_modification(task_id, doc_ext, intent, monkeypatch):
    if not (ROOT / "third_party/OmegaUse-OfficeVal/.git").exists() or not (ROOT / f"benchmarks/officeval/assets/{task_id}").exists():
        pytest.skip("Native OfficeVal third_party checkout or task assets not present in this checkout")
    monkeypatch.setenv("GEMINI_API_KEY", "test-preflight-key")
    cfg_file = (ROOT / f"configs/l1/officeval_{task_id[-3:]}.gemini-35lite.json"
                if task_id != "officeval_003"
                else ROOT / "configs/l1/officeval.gemini-35lite.json")
    assert cfg_file.exists(), f"Configuration file {cfg_file} must exist"

    cfg = resolve(cfg_file)
    assert cfg["benchmark"]["task_id"] == task_id
    assert cfg["benchmark"]["adapter"] == "officeval-native"

    # Verify adapter loading and locked hashes
    adapter = NativeOfficeVal(cfg["benchmark"])
    task = adapter.load(task_id)
    assert task.task_id == task_id
    assert task.evaluation.entrypoint == f"verifiers/{task_id}_verifier.py"
    assert len(task.assets) >= 1
    assert any(a.name.endswith(f".{doc_ext}") for a in task.assets)

    # Verify doctor preflight passes without contacting model
    doc = doctor(cfg)
    assert doc["ready"] is True
    assert doc["model_endpoint_contacted"] is False
    assert any(c["name"] == "native_task_and_assets" and c["passed"] for c in doc["checks"])


@pytest.mark.parametrize("task_id,doc_ext,intent", TASKS)
def test_multitask_fixture_orchestration(tmp_path, task_id, doc_ext, intent):
    if not (ROOT / "third_party/OmegaUse-OfficeVal/.git").exists() or not (ROOT / f"benchmarks/officeval/assets/{task_id}").exists():
        pytest.skip("Native OfficeVal third_party checkout or task assets not present in this checkout")
    cfg_file = (ROOT / f"configs/l1/officeval_{task_id[-3:]}.gemini-35lite.json"
                if task_id != "officeval_003"
                else ROOT / "configs/l1/officeval.gemini-35lite.json")
    cfg = resolve(cfg_file)

    injected = {
        "model": ScriptedGemini(cfg["model"]),
        "benchmark": NativeOfficeVal(cfg["benchmark"]),
        "runtime": MultitaskFixtureRuntime(),
        "evaluation": FixtureEvaluator(),
    }

    out = tmp_path / f"run_{task_id}"
    result = run(cfg, out, injected=injected)

    assert result["verification"]["verified"] is True
    manifest = read_json(out / "manifest.json")
    assert manifest["task"]["task_id"] == task_id
    assert manifest["task"]["provenance"]["adapter"] == "officeval-native"
    assert manifest["counts"]["model_requests"] == 2
    assert manifest["counts"]["tool_results"] == 1
    assert manifest["counts"]["snapshots"] >= 2
