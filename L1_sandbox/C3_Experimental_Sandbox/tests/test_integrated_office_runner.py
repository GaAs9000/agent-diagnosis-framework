import asyncio
import pytest

from c3_sandbox.integrated.office_runner import run_episode
from c3_sandbox.r1.office_run import runtime_system_identity


def test_runner_rejects_task_before_runtime_import(tmp_path):
    bundle = {"task_ids": ["officeval_002"], "bundle_root": str(tmp_path), "tasks": {}}
    with pytest.raises(ValueError, match="not in frozen bundle"):
        asyncio.run(run_episode(bundle, "officeval_003", agent_image="sha256:" + "a" * 64,
                                 grader_image="sha256:" + "b" * 64, model_profile={"x": 1},
                                 output_dir=tmp_path / "out", model_fingerprint="f" * 64,
                                 tokenizer_fingerprint="t" * 64))


def test_runner_rejects_missing_profile_before_runtime(tmp_path):
    bundle = {"task_ids": ["officeval_002"], "bundle_root": str(tmp_path),
              "tasks": {"officeval_002": {"expected_output": "x.docx"}}}
    with pytest.raises(ValueError, match="complete model profile"):
        asyncio.run(run_episode(bundle, "officeval_002", agent_image="sha256:" + "a" * 64,
                                 grader_image="sha256:" + "b" * 64, model_profile={},
                                 output_dir=tmp_path / "out", model_fingerprint="f" * 64,
                                 tokenizer_fingerprint="t" * 64))


def test_runtime_identity_binds_active_prompt():
    runtime = {"model_config": {"model": "m"}, "sampling": {"seed": 2026},
               "environment": {"PATH": "fixed"}}
    tools = {"schemas": [], "timeout_seconds": 30}
    assert runtime_system_identity(runtime, tools, "prompt-a") != runtime_system_identity(runtime, tools, "prompt-b")


def test_integrated_budget_is_fixed(tmp_path):
    bundle = {"task_ids": ["officeval_002"], "bundle_root": str(tmp_path),
              "tasks": {"officeval_002": {"expected_output": "x.docx"}}}
    with pytest.raises(ValueError, match="budget=100"):
        asyncio.run(run_episode(bundle, "officeval_002", agent_image="sha256:" + "a" * 64,
                                 grader_image="sha256:" + "b" * 64, model_profile={"x": 1},
                                 output_dir=tmp_path / "out", model_fingerprint="f" * 64,
                                 tokenizer_fingerprint="t" * 64, budget=1))
