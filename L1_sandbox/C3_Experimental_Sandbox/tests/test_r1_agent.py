"""Synthetic protocol tests only: never contact a provider or Docker."""
import asyncio
import copy
import json

import pytest
from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ModelOutput, ModelUsage
from inspect_ai.tool import ToolCall

from c3_sandbox.r1 import agent


SCHEMA = [{"type": "function", "function": {"name": "nonce", "description": "consume nonce",
          "parameters": {"type": "object", "properties": {"value": {"type": "string"}},
                         "required": ["value"], "additionalProperties": False}}}]
PREFIX = [{"role": "system", "content": "synthetic test"}, {"role": "user", "content": "go"}]


def output(calls=None, usage=None):
    return ModelOutput(choices=[ChatCompletionChoice(message=ChatMessageAssistant(
        content="" if calls else "Finished synthetic test.", tool_calls=calls), stop_reason="tool_calls" if calls else "stop")], usage=usage)


def fixture_model(monkeypatch, outputs):
    class SyntheticModel:
        requests = []

        async def generate(self, messages, **kwargs):
            self.requests.append((copy.deepcopy(messages), kwargs))
            result = outputs.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

    model = SyntheticModel()
    monkeypatch.setattr(agent, "get_model", lambda **kwargs: model)
    return model


def run(tmp_path, tools, **kwargs):
    config = kwargs.pop("config", {"token_count": lambda messages, schemas: 10})
    return asyncio.run(agent.NativeAgent(config, tmp_path).run(
        kwargs.pop("messages", PREFIX), tools, SCHEMA, max_turns=kwargs.pop("max_turns", 3), **kwargs))


def test_native_serial_calls_usage_raw_log_checkpoint_and_replay(monkeypatch, tmp_path):
    model = fixture_model(monkeypatch, [output([ToolCall("id1", "nonce", {"value": "a"}),
                                               ToolCall("id2", "nonce", {"value": "b"})],
        ModelUsage(input_tokens=10, input_tokens_cache_read=3, input_tokens_cache_write=2,
                   output_tokens=4, total_tokens=19, total_cost=.01)), output()])
    consumed, checkpoints = [], []

    async def nonce(value):
        consumed.append(value)
        await asyncio.sleep(0)
        return {"nonce": value}

    result = run(tmp_path, {"nonce": nonce}, checkpoint_callback=lambda messages, index:
                 checkpoints.append((messages, index)), config={"token_count": lambda m, t: 10,
                 "tool_result_transform": lambda n, a, raw: "redacted"})
    assert consumed == ["a", "b"]
    assert result["messages"][:2] == PREFIX
    assert [m["tool_call_id"] for m in result["messages"] if m["role"] == "tool"] == ["id1", "id2"]
    assert result["tokens"]["full_prompt_tokens"] == 15
    assert result["tokens"]["output_tokens"] == 4
    assert result["stopped_reason"] == "assistant_finished"
    assert result["events"][1]["raw_result"] == {"nonce": "a"}
    assert result["events"][1]["delivered_content"] == "redacted"
    assert len(checkpoints) == 2
    records = [json.loads(line) for line in (tmp_path / "model_calls.jsonl").read_text().splitlines()]
    assert records[1]["model_output"]["choices"][0]["message"]["tool_calls"][0]["id"] == "id1"
    assert records[0]["provider"]["emulate_tools"] is False
    assert records[0]["provider"]["stream"] is False
    prefix = checkpoints[0][0]
    fixture_model(monkeypatch, [output()])
    restored = run(tmp_path / "restored", {"nonce": nonce}, messages=prefix)
    assert restored["messages"][:len(prefix)] == prefix
    assert model.requests[1][0][3].tool_call_id == "id1"


@pytest.mark.parametrize("call", [ToolCall("a", "unknown", {}),
    ToolCall("a", "nonce", {"value": 5}), ToolCall("a", "nonce", {"value": "a", "extra": 1}),
    ToolCall("a", "nonce", {}, parse_error="bad JSON")])
def test_invalid_protocol_has_no_fallback_action(monkeypatch, tmp_path, call):
    fixture_model(monkeypatch, [output([ToolCall("valid", "nonce", {"value": "good"}), call])])
    executed, checkpoints = [], []
    result = run(tmp_path, {"nonce": lambda value: executed.append(value)},
                 checkpoint_callback=lambda *args: checkpoints.append(args))
    assert result["stopped_reason"] == "invalid_protocol"
    assert not executed and not checkpoints
    assert result["messages"] == PREFIX


def test_context_budget_no_truncation_or_generation(monkeypatch, tmp_path):
    model = fixture_model(monkeypatch, [])
    result = run(tmp_path, {"nonce": lambda value: value},
                 config={"token_count": lambda m, t: 14337})
    assert not model.requests and result["modelcall_count"] == 0
    assert result["messages"] == PREFIX
    assert result["stopped_reason"] == "context_budget_exceeded"


def test_error_retained_and_tool_error_binding(monkeypatch, tmp_path):
    fixture_model(monkeypatch, [output([ToolCall("id", "nonce", {"value": "a"})]), RuntimeError("provider failure")])

    def failed(value):
        raise RuntimeError("tool failure")

    result = run(tmp_path, {"nonce": failed})
    assert result["messages"][-1]["tool_call_id"] == "id"
    assert "tool failure" in result["messages"][-1]["content"]
    assert result["stopped_reason"] == "model_error"
    assert "provider failure" in (tmp_path / "model_calls.jsonl").read_text()


def test_requires_counter_and_resolved_prefix(monkeypatch, tmp_path):
    fixture_model(monkeypatch, [])
    with pytest.raises(ValueError, match="token_count"):
        run(tmp_path, {"nonce": lambda value: value}, config={})
    pending = PREFIX + [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "a", "function": "nonce", "arguments": {"value": "a"}, "type": "function"}]}]
    with pytest.raises(ValueError, match="Pending"):
        run(tmp_path, {"nonce": lambda value: value}, messages=pending)


def test_transform_error_keeps_raw_evidence(monkeypatch, tmp_path):
    fixture_model(monkeypatch, [output([ToolCall("id", "nonce", {"value": "a"})])])

    def broken_transform(*args):
        raise ValueError("delivery failure")

    result = run(tmp_path, {"nonce": lambda value: {"raw": value}}, max_turns=1,
                 config={"token_count": lambda m, t: 10, "tool_result_transform": broken_transform})
    assert result["events"][1]["raw_result"] == {"raw": "a"}
    assert result["events"][1]["error"]["message"] == "delivery failure"


def test_hung_generation_timeout_stack_retained_no_retry(monkeypatch, tmp_path):
    class SyntheticHungModel:
        calls = 0
        cancelled = False

        async def waiting_leaf(self):
            secret_local = "do-not-log-sensitive-locals"
            await asyncio.Event().wait()

        async def generate(self, messages, **kwargs):
            self.calls += 1
            try:
                await self.waiting_leaf()
            finally:
                self.cancelled = True

    model = SyntheticHungModel()
    provider_configs = []

    def get_model(**kwargs):
        provider_configs.append(kwargs)
        return model

    monkeypatch.setattr(agent, "get_model", get_model)
    result = run(tmp_path, {"nonce": lambda value: value},
                 config={"token_count": lambda m, t: 10, "generation_timeout_seconds": .01})
    assert result["stopped_reason"] == "model_timeout"
    assert result["modelcall_count"] == model.calls == 1
    assert model.cancelled
    assert provider_configs[0]["max_retries"] == 0
    error = result["events"][0]["error"]
    assert error["cancellation_completed"] is True
    assert any(frame["function"] == "waiting_leaf" for frame in error["await_stack"])
    records = (tmp_path / "model_calls.jsonl").read_text()
    assert "GenerationTimeout" in records
    assert "do-not-log-sensitive-locals" not in records
    assert result["messages"] == PREFIX


@pytest.mark.parametrize("timeout", [0, -1, 181, True, float("inf")])
def test_generation_timeout_validation(tmp_path, timeout):
    with pytest.raises(ValueError, match="generation_timeout_seconds"):
        run(tmp_path, {"nonce": lambda value: value},
            config={"token_count": lambda m, t: 10, "generation_timeout_seconds": timeout})


def test_explicit_client_ignores_proxy_environment_and_closes(monkeypatch, tmp_path):
    monkeypatch.setenv("HTTP_PROXY", "http://synthetic.invalid:9999")
    monkeypatch.setenv("ALL_PROXY", "http://synthetic.invalid:9999")
    original = fixture_model(monkeypatch, [output()])
    captured = []

    def get_model(**kwargs):
        captured.append(kwargs["http_client"])
        assert kwargs["http_client"]._trust_env is False
        assert not kwargs["http_client"]._mounts
        return original

    monkeypatch.setattr(agent, "get_model", get_model)
    result = run(tmp_path, {"nonce": lambda value: value})
    assert result["stopped_reason"] == "assistant_finished"
    assert captured[0].is_closed
    request = json.loads((tmp_path / "model_calls.jsonl").read_text().splitlines()[0])
    assert request["http_client"]["trust_env"] is False
