"""Synthetic R1.2 contract tests; these do not establish G0 qualification."""
import asyncio
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ModelOutput
from c3_sandbox.r1 import agent, r12_system as system_module, tokenizer_server


def contract():
    return {"schema_version": "r12-system-v1", "system_id": "synthetic-r12",
            "model": {"repository": system_module.MODEL_REPOSITORY,
                      "revision": system_module.MODEL_REVISION,
                      "path": system_module.MODEL_PATH, "quantization": "4bit",
                      "tool_parser": system_module.TOOL_PARSER,
                      "tool_parser_sha256": system_module.TOOL_PARSER_SHA256},
            "sampling": copy.deepcopy(system_module.SAMPLING),
            "seed_domains": copy.deepcopy(system_module.SEED_DOMAINS),
            "system_prompt": "Synthetic qualification", "agent_image": "sha256:" + "a" * 64,
            "tool_schemas": [{"type": "function", "function": {"name": "execute",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}}],
            "server": {"host": "127.0.0.1", "port": 18080, "tokenizer_port": 18081,
                       "prompt_cache_size": 0, "prompt_concurrency": 1, "decode_concurrency": 1,
                       "packages_lock": "requirements.model.local-arm64.lock.txt"}}


def identity(system):
    return {"model_fingerprint": "b" * 64, "tokenizer_fingerprint": {"sha256": "c" * 64},
            "system_contract_sha256": system_module.system_identity(system)}


def test_contract_detaches_and_hashes_every_field():
    system = contract()
    detached = system_module.validated_system(system)
    sha = system_module.system_identity(system)
    system["model"]["revision"] = "changed"
    assert detached["model"]["revision"] == system_module.MODEL_REVISION
    detached["provenance"] = {"qualification": "synthetic"}
    assert system_module.system_identity(detached) != sha


@pytest.mark.parametrize("group,key,value", [
    ("model", "revision", "wrong"), ("model", "path", "../outside"),
    ("sampling", "max_tokens", 2048), ("sampling", "seed", True),
    ("sampling", "parallel_tool_calls", 0), ("sampling", "min_p", float("nan")),
    ("server", "prompt_cache_size", True), ("server", "port", 9999)])
def test_contract_rejects_configuration_drift(group, key, value):
    system = contract()
    system[group][key] = value
    with pytest.raises(ValueError):
        system_module.validated_system(system)


def fake_fingerprint(tmp_path, monkeypatch):
    monkeypatch.setattr(system_module, "ROOT", tmp_path)
    folder = tmp_path / system_module.MODEL_PATH
    folder.mkdir(parents=True)
    (folder / "weights.safetensors").write_bytes(b"synthetic weights")
    (folder / "tokenizer.json").write_bytes(b"synthetic tokenizer")
    (folder / "config.json").write_text('{"quantization":{"bits":4,"group_size":64}}')
    (tmp_path / "requirements.model.local-arm64.lock.txt").write_bytes(b"synthetic lock")
    observed = {"revision": system_module.MODEL_REVISION, "tool_parser": system_module.TOOL_PARSER,
                "tool_parser_sha256": system_module.TOOL_PARSER_SHA256,
                "file_sha256": {"tokenizer.json": hashlib.sha256(b"synthetic tokenizer").hexdigest()}}
    observed["sha256"] = hashlib.sha256(json.dumps(observed, sort_keys=True,
                                                separators=(",", ":")).encode()).hexdigest()
    return folder, SimpleNamespace(text_details=lambda text: {"tokenizer_fingerprint": observed}), observed


def test_full_model_fingerprint_and_file_drift(tmp_path, monkeypatch):
    folder, tokenizer, observed = fake_fingerprint(tmp_path, monkeypatch)
    original = system_module.fingerprint_model(contract(), tokenizer)
    assert set(original["files"]) == {"weights.safetensors", "tokenizer.json", "config.json"}
    assert original["tool_parser"] == system_module.TOOL_PARSER
    assert original["system_contract_sha256"] == system_module.system_identity(contract())
    (folder / "weights.safetensors").write_bytes(b"mutated weights")
    assert original["sha256"] != system_module.fingerprint_model(contract(), tokenizer)["sha256"]
    observed["tool_parser_sha256"] = "bad"
    with pytest.raises(ValueError, match="observed tokenizer"):
        system_module.fingerprint_model(contract(), tokenizer)


def test_model_external_symlink_rejected_before_tokenizer(tmp_path, monkeypatch):
    folder, tokenizer, _ = fake_fingerprint(tmp_path, monkeypatch)
    external = tmp_path / "outside"
    external.write_bytes(b"outside")
    (folder / "escape").symlink_to(external)
    with pytest.raises(ValueError, match="symlink escapes"):
        system_module.fingerprint_model(contract(), tokenizer)


def test_native_contract_budget_identity_and_freezing(tmp_path, monkeypatch):
    system = contract()
    saved = copy.deepcopy(system)
    provenance = identity(system)
    captured = {}
    class SyntheticModel:
        async def generate(self, *args, **kwargs):
            return ModelOutput(choices=[ChatCompletionChoice(
                message=ChatMessageAssistant(content="Synthetic completed"), stop_reason="stop")])
    def get_model(**kwargs):
        captured.update(kwargs)
        return SyntheticModel()
    monkeypatch.setattr(agent, "get_model", get_model)
    native = agent.NativeAgent({"token_count": lambda m, t: 20000,
                               "identity_context": provenance, "generation_guard": lambda: {"synthetic": True}}, tmp_path, system_contract=system)
    system["system_prompt"] = "mutated"
    provenance["model_fingerprint"] = "mutated"
    result = asyncio.run(native.run([
        {"role": "system", "content": saved["system_prompt"]},
        {"role": "user", "content": "Synthetic"}], {"execute": lambda: "ok"},
        saved["tool_schemas"], max_turns=1))
    assert result["modelcall_count"] == 1
    assert captured["config"].max_tokens == 4096
    assert captured["config"].timeout == 240
    assert captured["config"].seed == 2026
    records = [json.loads(line) for line in (tmp_path / "model_calls.jsonl").read_text().splitlines()]
    request = next(row for row in records if row["type"] == "request")
    assert request["system_contract_sha256"] == system_module.system_identity(saved)
    assert request["identity_context"]["model_fingerprint"] == "b" * 64


@pytest.mark.parametrize("override", [{"max_tokens": 2048}, {"seed": 8},
                                     {"generation_timeout_seconds": 120}])
def test_native_rejects_contract_overrides(tmp_path, override):
    system = contract()
    with pytest.raises(ValueError, match="override"):
        agent.NativeAgent({**override, "identity_context": identity(system), "generation_guard": lambda: {"synthetic": True}}, tmp_path,
                          system_contract=system)


def test_native_rejects_unbound_identity_and_prompt(tmp_path):
    system = contract()
    with pytest.raises(ValueError, match="identity_context"):
        agent.NativeAgent({}, tmp_path, system_contract=system)
    native = agent.NativeAgent({"identity_context": identity(system), "generation_guard": lambda: {"synthetic": True}}, tmp_path, system_contract=system)
    with pytest.raises(ValueError, match="system prompt"):
        asyncio.run(native.run([{"role": "system", "content": "wrong"}],
            {"execute": lambda: "ok"}, system["tool_schemas"], max_turns=1))


def test_tokenizer_observes_actual_inferred_parser(tmp_path, monkeypatch):
    # Test startup via synthetic loader; no MLX import, weights, or generation.
    parser_file = tmp_path / "parser.py"
    parser_file.write_text("synthetic parser source")
    def parse_tool_call():
        pass
    parse_tool_call.__module__ = "mlx_lm.tool_parsers.synthetic"
    server = ModuleType("mlx_lm.server")
    server.process_message_content = lambda messages: None
    loader = ModuleType("mlx_lm.tokenizer_utils")
    template = "synthetic template"
    loader.load = lambda *args, **kwargs: SimpleNamespace(chat_template=template,
                                                        tool_parser=parse_tool_call)
    monkeypatch.setitem(sys.modules, "mlx_lm.server", server)
    monkeypatch.setitem(sys.modules, "mlx_lm.tokenizer_utils", loader)
    monkeypatch.setattr(tokenizer_server.inspect, "getsourcefile", lambda parser: str(parser_file))
    monkeypatch.setattr(tokenizer_server, "version", lambda name: "synthetic")
    (tmp_path / "chat_template.jinja").write_text(template)
    engine = tokenizer_server.TokenizerEngine(tmp_path, revision="synthetic", expected_parser="synthetic")
    assert engine.fingerprint["tool_parser"] == "mlx_lm.tool_parsers.synthetic.parse_tool_call"
    assert engine.fingerprint["tool_parser_sha256"] == hashlib.sha256(parser_file.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="inferred tool parser"):
        tokenizer_server.TokenizerEngine(tmp_path, revision="synthetic", expected_parser="qwen3_coder")
    legacy = tokenizer_server.TokenizerEngine(tmp_path)
    assert "tool_parser" not in legacy.fingerprint


def test_frozen_seed_domain_and_confirmation_override(tmp_path):
    system = contract()
    seed = system_module.SEED_DOMAINS["confirmation"]
    assert seed != 2026
    native = agent.NativeAgent({"execution_phase": "confirmation", "seed": seed,
                               "identity_context": identity(system), "generation_guard": lambda: {"synthetic": True}}, tmp_path, system_contract=system)
    assert native.config["seed"] == seed
    with pytest.raises(ValueError, match="override"):
        agent.NativeAgent({"execution_phase": "confirmation", "seed": 2026,
                           "identity_context": identity(system), "generation_guard": lambda: {"synthetic": True}}, tmp_path, system_contract=system)
    system["seed_domains"]["confirmation"] += 1
    with pytest.raises(ValueError, match="seed_domains"):
        system_module.validated_system(system)


def test_actual_quantization_config_rejects_eight_bit(tmp_path, monkeypatch):
    folder, tokenizer, _ = fake_fingerprint(tmp_path, monkeypatch)
    (folder / "config.json").write_text('{"quantization":{"bits":8}}')
    with pytest.raises(ValueError, match="4bit quantization"):
        system_module.fingerprint_model(contract(), tokenizer)


def test_contract_context_rejects_before_request(tmp_path, monkeypatch):
    system = contract()
    native = agent.NativeAgent({"identity_context": identity(system),
                               "token_count": lambda m, t: 32768 - 4096 + 1, "generation_guard": lambda: {"synthetic": True}},
                              tmp_path, system_contract=system)
    class SyntheticModel:
        async def generate(self, *args, **kwargs):
            pytest.fail("generation must not run past frozen context")
    monkeypatch.setattr(agent, "get_model", lambda **kwargs: SyntheticModel())
    result = asyncio.run(native.run([{"role": "system", "content": system["system_prompt"]},
        {"role": "user", "content": "Synthetic"}], {"execute": lambda: "ok"},
        system["tool_schemas"], max_turns=1))
    assert result["stopped_reason"] == "context_budget_exceeded"
    assert result["modelcall_count"] == 0


def test_acquisition_cache_and_locks_do_not_change_fingerprint(tmp_path, monkeypatch):
    folder, tokenizer, _ = fake_fingerprint(tmp_path, monkeypatch)
    original = system_module.fingerprint_model(contract(), tokenizer)
    cache = folder / ".cache" / "huggingface" / "download"
    cache.mkdir(parents=True)
    metadata = cache / "weights.safetensors.metadata"
    metadata.write_text("commit\nhash\ninitial timestamp")
    (cache / "weights.safetensors.lock").write_text("initial lock")
    (folder / ".acquisition.lock").write_text("initial lock")
    assert original == system_module.fingerprint_model(contract(), tokenizer)
    metadata.write_text("commit\nhash\nchanged timestamp")
    (folder / ".acquisition.lock").write_text("changed lock")
    assert original == system_module.fingerprint_model(contract(), tokenizer)


def test_index_requires_every_referenced_shard(tmp_path, monkeypatch):
    folder, tokenizer, _ = fake_fingerprint(tmp_path, monkeypatch)
    index = folder / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"layer1": "weights.safetensors",
                                                "layer2": "missing.safetensors"}}))
    with pytest.raises(ValueError, match="missing or excluded shard"):
        system_module.fingerprint_model(contract(), tokenizer)
    (folder / "missing.safetensors").write_bytes(b"second synthetic shard")
    result = system_module.fingerprint_model(contract(), tokenizer)
    assert {"weights.safetensors", "missing.safetensors", index.name} <= result["files"].keys()


@pytest.mark.parametrize("shard", ["../outside.safetensors", ".cache/hidden.safetensors"])
def test_index_cannot_reference_outside_or_cache_shards(tmp_path, monkeypatch, shard):
    folder, tokenizer, _ = fake_fingerprint(tmp_path, monkeypatch)
    index = folder / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"layer": shard}}))
    with pytest.raises(ValueError, match="missing or excluded shard"):
        system_module.fingerprint_model(contract(), tokenizer)
