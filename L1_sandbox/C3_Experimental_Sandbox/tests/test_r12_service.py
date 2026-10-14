"""Synthetic process observations; no live process or server qualification."""
import asyncio
import copy
import json
from types import SimpleNamespace

import pytest
from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ModelOutput
from c3_sandbox.r1 import agent, r12_service as service
from test_r12_system import contract, identity


def setup_observation(tmp_path, monkeypatch):
    root = tmp_path / "root with spaces"
    model = root / contract()["model"]["path"]
    model.mkdir(parents=True)
    monkeypatch.setattr(service, "ROOT", root)
    state = {"start": "Sun Oct 11 10:00:00 2026", "pid": "123",
             "command": f"{root}/.model-venv/bin/python -m mlx_lm.server --model {contract()['model']['path']} "
                        "--host 127.0.0.1 --port 18080 --prompt-cache-size 0 --decode-concurrency 1 "
                        "--prompt-concurrency 1 --max-tokens 4096 --temp 0 --log-level DEBUG",
             "listener": "p123\ncPython\nn127.0.0.1:18080\n", "cwd": str(root),
             "environment": {"source_sha256": "synthetic"}, "calls": []}
    def read(argv):
        state["calls"].append(argv)
        if argv[0] == "/bin/ps":
            return f"{state['pid']} {state['start']} {state['command']}\n"
        if "cwd" in argv:
            return f"p123\nn{state['cwd']}\n"
        return state["listener"]
    monkeypatch.setattr(service, "_read_command", read)
    monkeypatch.setattr(service, "_environment", lambda *args: copy.deepcopy(state["environment"]))
    return state, root


def test_observe_and_verify_relative_model_with_unquoted_space_path(tmp_path, monkeypatch):
    state, root = setup_observation(tmp_path, monkeypatch)
    frozen = service.observe_service(contract(), 123)
    assert frozen["argv"][0] == str(root / ".model-venv/bin/python")
    assert frozen["model_path"] == str(root / contract()["model"]["path"])
    assert frozen == service.verify_service(contract(), frozen)
    assert all("environ" not in arg for command in state["calls"] for arg in command)
    assert any("-iTCP:18080" in command for command in state["calls"])


@pytest.mark.parametrize("change", ["start", "environment", "model", "alias", "template",
                                     "trust", "cache", "owner", "wildcard", "extra_listener", "cwd", "marker"])
def test_verify_rejects_process_listener_and_source_drift(tmp_path, monkeypatch, change):
    state, _ = setup_observation(tmp_path, monkeypatch)
    frozen = service.observe_service(contract(), 123)
    if change == "start":
        state["start"] = "Sun Oct 11 11:00:00 2026"
    elif change == "environment":
        state["environment"]["source_sha256"] = "mutated"
    elif change == "model":
        state["command"] = state["command"].replace(contract()["model"]["path"], "models/other")
    elif change == "marker":
        state["command"] = state["command"].replace(contract()["model"]["path"], "__C3_APPROVED_MODEL_PATH__")
    elif change == "alias":
        state["command"] += " --served-model-name other"
    elif change == "template":
        state["command"] += " --chat-template alternative"
    elif change == "trust":
        state["command"] += " --trust-remote-code"
    elif change == "cache":
        state["command"] = state["command"].replace("--prompt-cache-size 0", "--prompt-cache-size 1")
    elif change == "owner":
        state["listener"] = "p456\nn127.0.0.1:18080\n"
    elif change == "wildcard":
        state["listener"] = "p123\nn*:18080\n"
    elif change == "extra_listener":
        state["listener"] += "p456\nn[::1]:18080\n"
    elif change == "cwd":
        state["cwd"] = "/other"
    with pytest.raises(ValueError):
        service.verify_service(contract(), frozen)


def test_bound_command_reader_rejects_errors_timeout_and_large_output(monkeypatch):
    calls = []
    def run(argv, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(returncode=0, stdout=b"x" * (service.MAX_OUTPUT_BYTES + 1))
    monkeypatch.setattr(service.subprocess, "run", run)
    with pytest.raises(ValueError, match="bounds"):
        service._read_command(["/bin/ps"])
    assert calls[0]["timeout"] == 5
    assert set(calls[0]["env"]) == {"LC_ALL", "PATH"}


def test_guard_rejection_prevents_generation_and_records_audit(tmp_path, monkeypatch):
    system = contract()
    generates = []
    class SyntheticModel:
        async def generate(self, *args, **kwargs):
            generates.append(True)
            pytest.fail("generation must not happen after failed service guard")
    monkeypatch.setattr(agent, "get_model", lambda **kwargs: SyntheticModel())
    def reject():
        raise ValueError("synthetic drift")
    native = agent.NativeAgent({"identity_context": identity(system), "token_count": lambda m,t: 1,
                               "generation_guard": reject}, tmp_path, system_contract=system)
    with pytest.raises(ValueError, match="synthetic drift"):
        asyncio.run(native.run([{"role": "system", "content": system["system_prompt"]},
            {"role": "user", "content": "Synthetic"}], {"execute": lambda: "ok"},
            system["tool_schemas"], max_turns=1))
    assert not generates and not (tmp_path / "model_calls.jsonl").exists()
    audit = json.loads((tmp_path / "protocol_audit.jsonl").read_text())
    assert audit["type"] == "generation_guard" and audit["status"] == "rejected"


def test_required_guard_and_fresh_observation_in_request(tmp_path, monkeypatch):
    system = contract()
    with pytest.raises(ValueError, match="generation_guard"):
        agent.NativeAgent({"identity_context": identity(system)}, tmp_path, system_contract=system)
    class SyntheticModel:
        async def generate(self, *args, **kwargs):
            return ModelOutput(choices=[ChatCompletionChoice(
                message=ChatMessageAssistant(content="Synthetic done"), stop_reason="stop")])
    monkeypatch.setattr(agent, "get_model", lambda **kwargs: SyntheticModel())
    guard_calls = []
    def guard():
        guard_calls.append(True)
        return {"pid": 123, "synthetic": True}
    native = agent.NativeAgent({"identity_context": identity(system), "token_count": lambda m,t: 1,
                               "generation_guard": guard}, tmp_path, system_contract=system)
    asyncio.run(native.run([{"role": "system", "content": system["system_prompt"]},
        {"role": "user", "content": "Synthetic"}], {"execute": lambda: "ok"},
        system["tool_schemas"], max_turns=1))
    records = [json.loads(line) for line in (tmp_path / "model_calls.jsonl").read_text().splitlines()]
    request = next(row for row in records if row["type"] == "request")
    assert len(guard_calls) == 1 and request["serving_observation"]["pid"] == 123


def test_environment_binds_actual_local_sources_lock_and_packages(tmp_path, monkeypatch):
    import hashlib
    root = tmp_path / "root"
    environment = root / ".model-venv"
    site = environment / "lib/python3.12/site-packages"
    parser = site / "mlx_lm/tool_parsers/qwen3_coder.py"
    parser.parent.mkdir(parents=True)
    parser.write_bytes(b"synthetic parser")
    (site / "mlx_lm/server.py").write_bytes(b"synthetic server")
    (site / "mlx_lm/tokenizer_utils.py").write_bytes(b"synthetic tokenizer")
    executable = environment / "bin/python"
    executable.parent.mkdir()
    executable.write_bytes(b"synthetic executable")
    (environment / "pyvenv.cfg").write_bytes(b"synthetic pyvenv")
    (root / contract()["server"]["packages_lock"]).write_bytes(b"synthetic lock")
    monkeypatch.setattr(service, "ROOT", root)
    monkeypatch.setattr(service, "TOOL_PARSER_SHA256", hashlib.sha256(parser.read_bytes()).hexdigest())
    packages = [SimpleNamespace(metadata={"Name": name}, version="synthetic")
                for name in ("mlx", "mlx-lm", "transformers")]
    monkeypatch.setattr(service, "distributions", lambda **kwargs: packages)
    observed = service._environment(executable, contract())
    assert observed["mlx_source_sha256"]["server.py"] == hashlib.sha256(b"synthetic server").hexdigest()
    assert observed["server_packages_lock_sha256"] == hashlib.sha256(b"synthetic lock").hexdigest()
    assert set(observed["packages"]) == {"mlx", "mlx-lm", "transformers"}
    parser.write_bytes(b"mutated parser")
    with pytest.raises(ValueError, match="parser source"):
        service._environment(executable, contract())
