"""Synthetic regressions for termination contradictions and SDK body evidence."""
import asyncio
import json
from types import SimpleNamespace

import pytest
from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ModelCall, ModelOutput, ModelUsage
from c3_sandbox.r1 import agent


@pytest.mark.parametrize("stop_reason,content", [
    ("tool_calls", ""), ("tool_calls", "claims completion"), ("stop", ""), ("stop", " \n ")])
def test_missing_native_calls_or_empty_final_is_protocol_error(monkeypatch, tmp_path, stop_reason, content):
    output = ModelOutput(choices=[ChatCompletionChoice(message=ChatMessageAssistant(content=content),
                         stop_reason=stop_reason)], usage=ModelUsage(input_tokens=20, output_tokens=694, total_tokens=714))

    class SyntheticModel:
        async def generate(self, *args, **kwargs):
            return output

    monkeypatch.setattr(agent, "get_model", lambda **kwargs: SyntheticModel())
    checkpoints = []
    result = asyncio.run(agent.NativeAgent({"token_count": lambda m, t: 20}, tmp_path).run(
        [{"role": "user", "content": "synthetic"}], {}, [], max_turns=2,
        checkpoint_callback=lambda *args: checkpoints.append(args)))
    assert result["stopped_reason"] == "protocol_error"
    assert result["modelcall_count"] == 1
    assert result["tokens"]["output_tokens"] == 694
    assert not checkpoints and len(result["messages"]) == 1
    assert result["events"][0]["stop_reason"] == stop_reason
    assert '"output_tokens": 694' in (tmp_path / "model_calls.jsonl").read_text()


def test_provider_body_sink_retains_unconverted_calls_and_forwards_events():
    records, forwarded = [], []
    parent = SimpleNamespace(on_pending=lambda event: forwarded.append("pending"),
                             on_complete=lambda event: forwarded.append("complete"))
    sink = agent._ProviderCallSink(lambda filename, record: records.append(record), 3, parent)
    event = SimpleNamespace(call=ModelCall(request={"messages": [{"role": "user", "content": "synthetic"}]},
        response={"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [
            {"id": "raw-id", "function": {"name": "nonce", "arguments": '{"value":"synthetic"}'}}]}}]}))
    sink.on_pending(event)
    sink.on_complete(event)
    assert forwarded == ["pending", "complete"]
    assert records[0]["response"] == event.call.response
    assert records[0]["type"] == "provider_call" and records[0]["turn"] == 3
    assert "headers" not in records[0]
    event.call.response.clear()
    assert records[0]["response"]["choices"]


def test_generate_task_inherits_body_capture_hook(monkeypatch, tmp_path):
    provider_response = {"choices": [{"finish_reason": "tool_calls", "message": {
        "content": "", "tool_calls": [{"id": "unconverted", "function": {
            "name": "nonce", "arguments": '{"value":"synthetic"}'}}]}}]}
    monkeypatch.setattr(agent, "transcript", lambda: SimpleNamespace(
        _event=lambda event: None, _event_updated=lambda event: None))

    class SyntheticModel:
        async def generate(self, *args, **kwargs):
            sink = agent._model_event_sink.get()
            event = SimpleNamespace(call=None)
            sink.on_pending(event)
            event.call = ModelCall(request={"messages": []}, response=provider_response)
            sink.on_complete(event)
            return ModelOutput(choices=[ChatCompletionChoice(message=ChatMessageAssistant(content=""),
                                                       stop_reason="tool_calls")])

    monkeypatch.setattr(agent, "get_model", lambda **kwargs: SyntheticModel())
    result = asyncio.run(agent.NativeAgent({"token_count": lambda m, t: 1}, tmp_path).run(
        [{"role": "user", "content": "synthetic"}], {}, [], max_turns=1))
    assert result["stopped_reason"] == "protocol_error"
    records = [json.loads(line) for line in (tmp_path / "model_calls.jsonl").read_text().splitlines()]
    captured = next(record for record in records if record["type"] == "provider_call")
    assert captured["response"] == provider_response
    assert agent._model_event_sink.get() is None
