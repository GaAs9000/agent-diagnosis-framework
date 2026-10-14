import copy
import json
from pathlib import Path

import pytest

from c3_sandbox.replay import ReplayQualification, STATE_CONTRACT_V1
from c3_sandbox.evidence._storage import digest, encode


SYSTEM_ID = digest(encode({"model": "deterministic-fixture", "scaffold": "fixture-v1",
                         "tools": "read-v1", "prompt": "fixture", "sampling": {"temperature": 0}}))


@pytest.fixture
def checkpoint(tmp_path):
    workspace = tmp_path / "source"
    workspace.mkdir()
    (workspace / "empty").mkdir()
    (workspace / ".hidden").write_bytes(b"\x00\xff")
    (workspace / ".hidden").chmod(0o640)
    messages = [{"role": "system", "content": "fixture"},
                {"role": "assistant", "content": None,
                 "tool_calls": [{"id": "one", "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "one", "content": "done"}]
    runtime = {"image_digest": "sha256:" + "a" * 64, "config": {"cwd": "/workspace", "network": "none"},
               "model_config": {"model": "deterministic-fixture", "temperature": 0},
               "tool_config": {"version": "fixture-v1"}, "environment": {"LANG": "C"}}
    service = ReplayQualification(tmp_path / "store")
    identity = service.register_checkpoint(workspace, messages, SYSTEM_ID, runtime, STATE_CONTRACT_V1, True)
    return service, identity, workspace, messages, runtime


def test_cold_restore_binds_complete_context(checkpoint, tmp_path):
    service, identity, _, messages, runtime = checkpoint
    restored = tmp_path / "branch"
    result = service.restore(identity, restored)
    assert result["valid"] and result["eligible_for_intervention"]
    assert result["messages"] == messages
    assert result["runtime"] == runtime
    assert (restored / "empty").is_dir()
    assert (restored / ".hidden").stat().st_mode & 0o777 == 0o640
    report = service.verify_restoration(identity, restored, messages, SYSTEM_ID, runtime)
    assert report["valid"]
    # Persistent records survive reopening; caller mutation cannot mutate them.
    messages[0]["content"] = "changed"
    assert ReplayQualification(service.workspace_store).qualify(identity)["eligible_for_intervention"]
    assert service.restore(identity, tmp_path / "second")["messages"][0]["content"] == "fixture"


@pytest.mark.parametrize("mutation", ["extra", "missing", "bytes", "mode", "symlink"])
def test_full_workspace_mismatch_rejected(checkpoint, tmp_path, mutation):
    service, identity, _, messages, runtime = checkpoint
    branch = tmp_path / "branch"
    assert service.restore(identity, branch)["valid"]
    file = branch / ".hidden"
    if mutation == "extra":
        (branch / "extra").write_text("surprise")
    elif mutation == "missing":
        file.unlink()
    elif mutation == "bytes":
        file.write_bytes(b"wrong")
    elif mutation == "mode":
        file.chmod(0o600)
    else:
        file.unlink()
        file.symlink_to("empty")
    result = service.verify_restoration(identity, branch, messages, SYSTEM_ID, runtime)
    assert not result["valid"] and not result["eligible_for_intervention"]
    assert result["reasons"]


@pytest.mark.parametrize("component", ["message", "tool_history", "system", "runtime", "image", "environment", "model", "tools"])
def test_context_identity_mismatch_rejected(checkpoint, tmp_path, component):
    service, identity, _, messages, runtime = checkpoint
    branch = tmp_path / "branch"
    assert service.restore(identity, branch)["valid"]
    messages, runtime = copy.deepcopy(messages), copy.deepcopy(runtime)
    system = SYSTEM_ID
    if component == "message":
        messages[0]["content"] = "other"
    elif component == "tool_history":
        messages[1]["tool_calls"][0]["function"]["arguments"] = '{"x":1}'
    elif component == "system":
        system = "other"
    elif component == "image":
        runtime["image_digest"] = "sha256:" + "b" * 64
    elif component == "environment":
        runtime["environment"]["LANG"] = "other"
    elif component == "model":
        runtime["model_config"]["temperature"] = 1
    elif component == "tools":
        runtime["tool_config"]["version"] = "other"
    else:
        runtime["config"]["network"] = "host"
    assert not service.verify_restoration(identity, branch, messages, system, runtime)["eligible_for_intervention"]


@pytest.mark.parametrize("key,value", [("persistent_processes", True), ("pending_tool_calls", 1),
                                      ("pending_tool_calls", False), ("external_state", "network"),
                                      ("caches", "model-kv"), ("at_tool_boundary", False),
                                      ("environment", "host"), ("scope", "all")])
def test_unsupported_state_unqualified(checkpoint, tmp_path, key, value):
    service, _, workspace, messages, runtime = checkpoint
    contract = {**STATE_CONTRACT_V1, key: value}
    identity = service.register_checkpoint(workspace, messages, SYSTEM_ID, runtime, contract, True)
    result = service.qualify(identity)
    assert not result["eligible_for_intervention"]
    assert any(key in reason for reason in result["reasons"])
    assert not service.restore(identity, tmp_path / "rejected")["valid"]
    assert not (tmp_path / "rejected").exists()


def test_missing_context_and_overrides_fail_closed(checkpoint):
    service, _, workspace, messages, runtime = checkpoint
    for key in STATE_CONTRACT_V1:
        contract = STATE_CONTRACT_V1.copy()
        del contract[key]
        identity = service.register_checkpoint(workspace, messages, SYSTEM_ID, runtime, contract, True)
        assert not service.qualify(identity)["eligible_for_intervention"]
    cases = [([], SYSTEM_ID, runtime, STATE_CONTRACT_V1, True),
             (messages, "", runtime, STATE_CONTRACT_V1, True),
             (messages, SYSTEM_ID, {}, STATE_CONTRACT_V1, True),
             (messages, SYSTEM_ID, runtime, {**STATE_CONTRACT_V1, "eligible_for_intervention": True}, True),
             (messages, SYSTEM_ID, runtime, STATE_CONTRACT_V1, False),
             (messages, SYSTEM_ID, runtime, STATE_CONTRACT_V1, "true"),
             (messages[:-1], SYSTEM_ID, runtime, STATE_CONTRACT_V1, True)]
    for args in cases:
        identity = service.register_checkpoint(workspace, *args)
        assert not service.qualify(identity)["eligible_for_intervention"]


@pytest.mark.parametrize("target", ["record", "manifest", "object", "missing_object"])
def test_corrupt_checkpoint_unqualified(checkpoint, tmp_path, target):
    service, identity, _, _, _ = checkpoint
    record = json.loads((service.records / (identity + ".json")).read_bytes())
    if target == "record":
        path = service.records / (identity + ".json")
    elif target == "manifest":
        path = service.workspace_store.manifests / (record["snapshot_id"] + ".json")
    else:
        path = next(service.workspace_store.objects.iterdir())
    if target == "missing_object":
        path.unlink()
    else:
        path.chmod(0o600)
        path.write_bytes(b"corrupt")
    assert not service.qualify(identity)["eligible_for_intervention"]
    assert not service.restore(identity, tmp_path / "branch")["valid"]


def test_nonempty_destination_not_deleted(checkpoint, tmp_path):
    service, identity, _, _, _ = checkpoint
    destination = tmp_path / "branch"
    destination.mkdir()
    (destination / "keep").write_text("keep")
    assert not service.restore(identity, destination)["valid"]
    assert (destination / "keep").read_text() == "keep"


def test_unsupported_filesystem_recorded_unqualified(checkpoint):
    service, _, workspace, messages, runtime = checkpoint
    (workspace / "link").symlink_to("empty")
    identity = service.register_checkpoint(workspace, messages, SYSTEM_ID, runtime, STATE_CONTRACT_V1, True)
    assert not service.qualify(identity)["eligible_for_intervention"]


@pytest.mark.parametrize("weak_identity", ["model-name", "", "A" * 64, "a" * 63, "sha256:" + "a" * 64])
def test_system_identity_requires_full_configuration_hash(checkpoint, weak_identity):
    service, _, workspace, messages, runtime = checkpoint
    invalid = service.register_checkpoint(workspace, messages, weak_identity, runtime, STATE_CONTRACT_V1, True)
    result = service.qualify(invalid)
    assert not result["eligible_for_intervention"]
    assert any("system identity" in reason for reason in result["reasons"])
    valid = service.register_checkpoint(workspace, messages, SYSTEM_ID, runtime, STATE_CONTRACT_V1, True)
    assert service.qualify(valid)["eligible_for_intervention"]
