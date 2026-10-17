from __future__ import annotations
import shutil
from pathlib import Path
import pytest
from c3_sandbox.l1.core.config import resolve
from c3_sandbox.l1.core.orchestrator import run
from c3_sandbox.l1.runtime.docker import DockerRuntime
from c3_sandbox.l1.evidence.live import verify
from c3_sandbox.l1.benchmarks.registry import registry as benchmark_registry
from c3_sandbox.l1.evaluation.registry import registry as eval_registry
from .conftest import ScriptedGemini, gemini_call, gemini_done, ROOT

TASK1_ID = "instance_ansible__ansible-0ea40e09d1b35bcb69ff4d9cecf3d0defa4b36e8-v30a923fb5c164d6cd18280c02422f75e611e8fb2"
TASK2_ID = "instance_ansible__ansible-0fd88717c953b92ed8a50495d55e630eb5d59166-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5"


def _require_swebench_assets():
    if not (ROOT / f"benchmarks/swebench_pro/data/tasks/{TASK1_ID}/instruction.md").exists():
        pytest.skip("SWE-bench Pro task data not present in this checkout")


def test_swebench_pro_task_load_and_provenance():
    _require_swebench_assets()
    reg = benchmark_registry()
    adapter = reg.create("swebench_pro-native", {"data_dir": str(ROOT / "benchmarks/swebench_pro/data")})
    tasks = adapter.list_tasks()
    assert any(t["task_id"] == TASK1_ID for t in tasks)
    assert any(t["task_id"] == TASK2_ID for t in tasks)

    task1 = adapter.load(TASK1_ID)
    assert task1.benchmark == "swebench_pro"
    assert task1.task_id == TASK1_ID
    assert task1.provenance["adapter"] == "swebench_pro-native"
    assert task1.provenance["data_lock_status"] == "sha256_locked"
    assert task1.provenance["evaluation_protocol"] == "official_swe_bench_pro_v2_verifier"
    assert task1.evaluation.kind == "swebench_pro-verifier"
    assert "source/tests/test_patch.patch" in task1.private_files
    assert "source/solution/gold_patch.diff" in task1.private_files

    task2 = adapter.load(TASK2_ID)
    assert task2.benchmark == "swebench_pro"
    assert task2.task_id == TASK2_ID
    assert task2.provenance["data_lock_status"] == "sha256_locked"


def test_swebench_pro_evaluator_scoring_matrix(tmp_path):
    _require_swebench_assets()
    bench_reg = benchmark_registry()
    eval_reg = eval_registry()
    adapter = bench_reg.create("swebench_pro-native", {"data_dir": str(ROOT / "benchmarks/swebench_pro/data")})

    for idx, tid in enumerate([TASK1_ID, TASK2_ID], start=1):
        task = adapter.load(tid)
        evaluator = eval_reg.create("swebench_pro-verifier", {"image": f"ghcr.io/scaleapi/swe-bench_pro-v2:{tid}"})

        # 1. Negative probe: empty patch
        sub_neg = tmp_path / f"sub_neg_{idx}"
        sub_neg.mkdir()
        out_neg = tmp_path / f"out_neg_{idx}"
        grade_neg = evaluator.evaluate(task.evaluation, sub_neg, out_neg)
        assert grade_neg["projection"]["evaluation_validity"] == "valid"
        assert grade_neg["projection"]["task_outcome"] == "failure"
        assert grade_neg["official_raw"]["passed"] is False
        assert grade_neg["official_raw"]["reward"] == 0.0

        # 2. Positive probe: gold patch
        sub_pos = tmp_path / f"sub_pos_{idx}"
        sub_pos.mkdir()
        out_pos = tmp_path / f"out_pos_{idx}"
        gold_patch_src = task.private_files["source/solution/gold_patch.diff"]
        shutil.copyfile(gold_patch_src, sub_pos / "gold_patch.diff")

        grade_pos = evaluator.evaluate(task.evaluation, sub_pos, out_pos)
        assert grade_pos["projection"]["evaluation_validity"] == "valid"
        assert grade_pos["projection"]["task_outcome"] == "success"
        assert grade_pos["official_raw"]["passed"] is True
        assert grade_pos["official_raw"]["reward"] == 1.0


def test_swebench_pro_end_to_end_scripted_execution(tmp_path):
    _require_swebench_assets()
    out = (tmp_path / "episode").resolve()
    cfg = resolve(ROOT / "configs/l1/swebench_pro.gemini-35lite.json")

    bench_reg = benchmark_registry()
    eval_reg = eval_registry()
    benchmark = bench_reg.create("swebench_pro-native", cfg["benchmark"])
    task = benchmark.load(TASK1_ID)
    evaluator = eval_reg.create("swebench_pro-verifier", cfg["evaluation"])

    # Prepare command for scripted model to deposit the gold patch in submission directory
    gold_patch_text = task.private_files["source/solution/gold_patch.diff"].read_text()
    escaped_patch = gold_patch_text.replace("'", "'\\''")
    cmd = f"mkdir -p /workspace/submission && printf '%s' '{escaped_patch}' > /workspace/submission/gold_patch.diff"

    model = ScriptedGemini(cfg["model"], [
        gemini_call("run", {"command": cmd}),
        gemini_done()
    ])
    runtime = DockerRuntime(cfg["runtime"])

    res = run(cfg, out, injected={
        "model": model,
        "runtime": runtime,
        "evaluation": evaluator,
        "benchmark": benchmark
    })

    assert res["status"]["task_outcome"] == "success"
    assert res["status"]["evaluation_validity"] == "valid"
    v = verify(out)
    assert v["verified"], f"SWE-bench Pro evidence verification errors: {v['errors']}"
    assert not v["errors"]


def test_swebench_pro_f27_writable_app_persistence_and_cold_resume(tmp_path):
    """F27: Verify that in-place edits inside /app persist across docker-cold steps,
    sync into /workspace/patch.diff and /workspace/submission/patch.diff, are captured
    in checkpoints, and survive cold resume with identical official and snapshot scores.
    """
    _require_swebench_assets()
    parent_out = (tmp_path / "parent_episode").resolve()
    cfg = resolve(ROOT / "configs/l1/swebench_pro.gemini-35lite.json")

    bench_reg = benchmark_registry()
    eval_reg = eval_registry()
    benchmark = bench_reg.create("swebench_pro-native", cfg["benchmark"])
    task = benchmark.load(TASK1_ID)
    evaluator = eval_reg.create("swebench_pro-verifier", cfg["evaluation"])

    # Apply gold patch directly inside /app (without manually writing /workspace/submission/*)
    gold_patch_text = task.private_files["source/solution/gold_patch.diff"].read_text()
    escaped_patch = gold_patch_text.replace("'", "'\\''")
    step1_cmd = f"printf '%s' '{escaped_patch}' | git -C /app apply -"
    step2_cmd = "git -C /app status --short && test -s /workspace/submission/patch.diff"

    parent_model = ScriptedGemini(cfg["model"], [
        gemini_call("run", {"command": step1_cmd}),
        gemini_call("run", {"command": step2_cmd}),
        gemini_done(),
    ])
    parent_runtime = DockerRuntime(cfg["runtime"])

    parent_res = run(cfg, parent_out, injected={
        "model": parent_model,
        "runtime": parent_runtime,
        "evaluation": evaluator,
        "benchmark": benchmark,
    })
    assert parent_res["status"]["task_outcome"] == "success"
    assert parent_res["status"]["evaluation_validity"] == "valid"
    parent_grade = __import__("json").loads((parent_out / "official_grade.json").read_text())
    assert parent_grade["official_raw"]["reward"] == 1.0
    pv = verify(parent_out)
    assert pv["verified"], f"Parent verification errors: {pv['errors']}"

    # Cold resume from cp000001 (right after Step 1 applied patch in /app)
    child_out = (tmp_path / "child_resume_cp000001").resolve()
    child_model = ScriptedGemini(cfg["model"], [
        gemini_call("run", {"command": "git -C /app diff --stat HEAD"}),
        gemini_done(),
    ])
    child_runtime = DockerRuntime(cfg["runtime"])
    child_res = run(
        cfg,
        child_out,
        injected={
            "model": child_model,
            "runtime": child_runtime,
            "evaluation": evaluator,
            "benchmark": benchmark,
        },
        parent=parent_out,
        checkpoint_id="cp000001",
    )
    assert child_res["status"]["task_outcome"] == "success"
    assert child_res["status"]["evaluation_validity"] == "valid"
    child_grade = __import__("json").loads((child_out / "official_grade.json").read_text())
    assert child_grade["official_raw"]["reward"] == 1.0
    cv = verify(child_out)
    assert cv["verified"], f"Child cold-resume verification errors: {cv['errors']}"

