"""Synthetic controller fixtures only: no real admission, Docker, or model claims."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import shutil

import pytest

from c3_sandbox.experiment_scheduler import BranchSpec, ProtocolError
from c3_sandbox.manifests import canonical_hash
from c3_sandbox.r1 import branch_executor as module
from c3_sandbox.r1.qualification import MODEL_REPO

IMAGE = "sha256:" + "a" * 64
GRADER = "sha256:" + "b" * 64


@pytest.fixture
def synthetic_fixture(tmp_path, monkeypatch):
    """Replace external operations with explicitly synthetic test doubles."""
    prefix = [{"role": "system", "content": "generic"}, {"role": "user", "content": "task"},
              {"role": "assistant", "content": "", "tool_calls": [{"id": "read", "function": "workspace_python", "arguments": {"code": "read"}}]},
              {"role": "tool", "content": "observed", "function": "workspace_python", "tool_call_id": "read"}]
    runtime = {"image_digest": IMAGE, "config": {},
               "model_config": {"provider": "openai-api/mlx/default_model", "model": MODEL_REPO, "model_fingerprint": "c" * 64},
               "tool_config": {"schemas": copy.deepcopy(module.TOOL_SCHEMAS)}, "environment": {"HOME": "/tmp"}}
    qualification = {**runtime["model_config"], "status": "passed", "is_synthetic": False,
                     "native_tool_calls": 1, "model_calls": 2, "prompt_tokens": 100, "completion_tokens": 20,
                     "tool_call_id": "nonce", "observation_tool_call_id": "nonce",
                     "nonce": "abc", "observed_nonce": "abc", "independent_nonce": "abc"}
    original = tmp_path / "source"
    original.mkdir()
    (original / "document.docx").write_bytes(b"original")
    (tmp_path / "messages.json").write_text(json.dumps(prefix))
    (tmp_path / "source_audit.json").write_text(json.dumps({"inspect": {"State": {"Pid": 20}}}))
    checkpoint = "d" * 64
    system = "e" * 64
    identity = canonical_hash(runtime)
    state = {"eligible": True, "calls": [], "environment": ["HOME=/tmp"], "pid": 21,
             "stopped_reason": "assistant_finished", "actor_closed": False}
    class Replay:
        def qualify(self, identifier):
            return {"eligible_for_intervention": state["eligible"], "is_synthetic": False,
                    "checkpoint_identity": identifier}
        def restore(self, identifier, workspace):
            shutil.copytree(original, workspace)
            return {**self.qualify(identifier), "valid": state["eligible"], "messages": copy.deepcopy(prefix),
                    "runtime": copy.deepcopy(runtime), "runtime_identity": identity, "system_identity": system}
        def verify_restoration(self, *args):
            return {"valid": True, "eligible_for_intervention": state["eligible"]}
    class Runtime:
        def __init__(self, image_id, run_dir):
            self.workspace = Path(run_dir) / "workspace"
        def __enter__(self):
            state["actor_closed"] = False
            return self
        def __exit__(self, *_):
            state["actor_closed"] = True
        def seed_workspace(self, workspace):
            shutil.copytree(workspace, self.workspace)
        def verify_workspace(self, workspace):
            assert (self.workspace / "document.docx").read_bytes() == (workspace / "document.docx").read_bytes()
            return {"verified": True}
        def audit_boundary(self):
            return {"qualified": True, "inspect": {"State": {"Pid": state["pid"]}}}
        def _inspect(self):
            return {"Config": {"Env": state["environment"]}}
        def exec_python(self, code):
            state["calls"].append(code)
            (self.workspace / "document.docx").write_bytes(b"model-edited")
            return {"exit_code": 0, "stdout": "real fixture output", "stderr": ""}
        def _snapshot(self):
            from c3_sandbox.r1.docker_runtime import snapshot_workspace
            return snapshot_workspace(self.workspace)

        def export_workspace(self, destination):
            shutil.copytree(self.workspace, destination)
            return {"snapshot_sha256": "f" * 64}
    class Tokenizer:
        async def count_messages_details(self, messages, tools):
            return {"token_count": len(messages), "prompt_token_id_sha256": "f" * 64}
        async def count_messages(self, messages, tools):
            return len(messages)
        def text_details(self, text):
            return {"token_count": len(text), "prompt_token_id_sha256": "f" * 64}
    class Agent:
        def __init__(self, config, run_dir):
            state["agent_config"] = config
        async def run(self, messages, tools, schemas, *, max_turns, dispatch_callback):
            state["model_prefix"] = copy.deepcopy(messages)
            state["remaining_turns"] = max_turns
            event = {"stage": "start", "turn": 0, "tool_call_id": "new-model-call",
                     "function": "workspace_python", "arguments": {"code": "model-command"},
                     "raw_result": None, "error": None}
            await dispatch_callback(event)
            raw = tools["workspace_python"]("model-command")
            await dispatch_callback({**event, "stage": "end", "raw_result": raw})
            return {"messages": messages, "events": [], "tokens": {"full_prompt_tokens": 200, "output_tokens": 30},
                    "stopped_reason": state["stopped_reason"]}
    def grade(task_id, manifest, workspace, image, folder):
        assert state["actor_closed"]
        assert (workspace / "document.docx").read_bytes() == b"model-edited"
        return {"validity": "valid", "c3_operational_completed": True,
                "official_raw": {"dim1_pass": True, "dim2_items": [{"rule": "positive", "max_delta": 5, "hit": True}], "total_score": 5},
                "unassessed_rules": []}
    monkeypatch.setattr(module, "DockerRuntime", Runtime)
    monkeypatch.setattr(module, "TokenizerClient", Tokenizer)
    monkeypatch.setattr(module, "NativeAgent", Agent)
    monkeypatch.setattr(module, "grade_workspace", grade)
    monkeypatch.setattr(module, "model_fingerprint", lambda: {"sha256": "c" * 64})
    monkeypatch.setattr(module, "_audit", lambda audit, image: None)
    manifest = {"tasks": {"officeval_002": {"public": "frozen fixture"}}}
    baseline = {"task_id": "officeval_002", "is_synthetic": False, "system_identity": system,
                "execution_valid": True, "task_completed": True,
                "runtime": runtime, "initial_grade": {
                    "validity": "valid", "official_raw": {"dim1_pass": True, "dim2_items": []}, "unassessed_rules": []},
                "checkpoints": [{"checkpoint_id": checkpoint, "messages_path": str(tmp_path / "messages.json"),
                                 "source_audit_path": str(tmp_path / "source_audit.json")} ]}
    # This fixture bypasses the renderer's public-evidence producer intentionally.
    probe = {"probe_id": "original", "rendered_text": "", "rendered_digest": module.hashlib.sha256(b"").hexdigest(), "token_count": 0}
    executor = module.RealBranchExecutor(Replay(), tmp_path / "branches", baseline, manifest, IMAGE, GRADER,
                                         {"original": probe}, qualification, "1" * 64)
    spec = BranchSpec("fixture", canonical_hash(manifest["tasks"]["officeval_002"]), "E2-S", system,
                      checkpoint, checkpoint, "original", executor.registry.identity("original"),
                      {"seed": 2027, "temperature": 0, "max_turns": 10}, 0, identity, "1" * 64, False, "exploration")
    return executor, spec, state, tmp_path


def test_synthetic_fixture_native_continuation_and_sanitized_result(synthetic_fixture):
    executor, spec, state, tmp_path = synthetic_fixture
    result = executor(spec)
    assert state["calls"] == ["model-command"]
    assert state["remaining_turns"] == 9
    assert state["agent_config"]["seed"] == 2027
    assert state["model_prefix"][-1]["role"] == "tool"  # original adds no message
    assert result.result["outcome"]["task_completed"] is True
    assert result.physical_cost["input_tokens"] == 200 and result.physical_cost["output_tokens"] == 30
    assert result.physical_cost["gpu_seconds"] is None
    assert str(tmp_path) not in json.dumps(result.result)
    assert (tmp_path / "branches" / spec.key / "base_prefix.json").is_file()
    with pytest.raises(FileExistsError):
        executor(spec)


@pytest.mark.parametrize("mutation", ["synthetic", "admission", "environment", "same_pid", "fingerprint"])
def test_synthetic_fixture_fail_closed_before_model(synthetic_fixture, mutation, monkeypatch):
    executor, spec, state, tmp_path = synthetic_fixture
    if mutation == "synthetic":
        spec = replace(spec, is_synthetic=True)
    elif mutation == "admission":
        state["eligible"] = False
    elif mutation == "environment":
        state["environment"] = ["HOME=/other"]
    elif mutation == "same_pid":
        state["pid"] = 20
    else:
        monkeypatch.setattr(module, "model_fingerprint", lambda: {"sha256": "0" * 64})
    with pytest.raises(ProtocolError):
        executor(spec)
    assert state["calls"] == []
    assert (tmp_path / "branches" / spec.key / "failure.json").is_file()


def test_synthetic_fixture_invalid_agent_has_no_success_flags(synthetic_fixture):
    executor, spec, state, _ = synthetic_fixture
    state["stopped_reason"] = "model_error"
    result = executor(spec)
    assert result.result["outcome"] == {"execution_valid": False, "target_resolved": None,
                                        "regression_free": None, "task_completed": None}


def test_synthetic_fixture_injects_exactly_one_runtime_user_advisory(synthetic_fixture):
    executor, spec, state, tmp_path = synthetic_fixture
    identifier = "visibility.compact"
    text = "Runtime-owned advisory: public observation presentation.\nWorkspace inventory: docx_count=1"
    executor.rendered_probes[identifier] = {"probe_id": identifier, "rendered_text": text,
        "rendered_digest": module.hashlib.sha256(text.encode()).hexdigest(), "token_count": len(text)}
    spec = replace(spec, probe_id=identifier, probe_identity=executor.registry.identity(identifier))
    executor(spec)
    assert state["model_prefix"][-1] == {"role": "user", "content": text}
    base = json.loads((tmp_path / "branches" / spec.key / "base_prefix.json").read_text())
    assert state["model_prefix"][:-1] == base


@pytest.mark.parametrize("valid,completed,line", [
    (False, True, "E2-S"), (True, None, "E2-S"),
    (True, False, "E2-S"), (True, True, "E2-F"),
])
def test_synthetic_fixture_rejects_invalid_or_wrong_baseline_line(synthetic_fixture, valid, completed, line):
    executor, spec, state, _ = synthetic_fixture
    executor.baseline.update(execution_valid=valid, task_completed=completed)
    with pytest.raises(ProtocolError):
        executor(replace(spec, line=line))
    assert state["calls"] == []


def test_synthetic_fixture_assessed_failure_uses_failure_line(synthetic_fixture):
    executor, spec, state, _ = synthetic_fixture
    executor.baseline["task_completed"] = False
    executor(replace(spec, line="E2-F"))
    assert state["calls"] == ["model-command"]
