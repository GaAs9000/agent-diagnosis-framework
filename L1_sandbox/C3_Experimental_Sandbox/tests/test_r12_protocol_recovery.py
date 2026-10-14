"""Bounded R12 native protocol recovery; synthetic provider only."""
import asyncio
import json

import httpx2 as httpx

from c3_sandbox.r1.agent import NativeAgent

SCHEMA = [{"type": "function", "function": {"name": "workspace_python",
          "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                         "required": ["code"], "additionalProperties": False}}}]
PREFIX = [{"role": "user", "content": "protocol recovery"}]


def response(calls=None, content="done", finish=None):
    calls = calls or []
    return {"model": "mlx/default_model", "choices": [{"finish_reason": finish or ("tool_calls" if calls else "stop"),
        "message": {"role": "assistant", "content": "" if calls else content, "tool_calls": calls}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}}


def unknown(i="c1", name="workspace=workspace_python"):
    return {"id": i, "type": "function", "function": {"name": name, "arguments": '{"code":"print(1)"}'}}


def run(tmp_path, responses, *, policy=None, max_turns=None):
    scripted = iter(responses)
    def respond(request):
        return httpx.Response(200, json=next(scripted), request=request)
    config = {"seed": 2026, "token_count": lambda messages, tools: 10,
              "identity_context": {"model_fingerprint": "synthetic", "tokenizer_fingerprint": "fixed"},
              "evidence_kind": "synthetic_protocol_fixture"}
    if policy is not None:
        config["protocol_recovery_policy"] = policy
    async def invoke():
        agent = NativeAgent(config, tmp_path, http_transport=httpx.MockTransport(respond))
        return await agent.run(PREFIX, {"workspace_python": lambda code: {"ran": code}}, SCHEMA,
                               max_turns=max_turns or len(responses))
    return asyncio.run(invoke())


def test_explicit_policy_recovers_unknown_without_normalization(tmp_path):
    result = run(tmp_path, [response([unknown()]), response()], policy="bounded")
    assert result["stopped_reason"] == "assistant_finished"
    assert result["protocol_recoveries"] == 1
    assert not result["tool_dispatch"]
    observation = result["messages"][2]
    assert observation["function"] == "workspace=workspace_python"
    assert json.loads(observation["content"])["available_tools"] == ["workspace_python"]


def test_default_behavior_remains_fail_closed(tmp_path):
    result = run(tmp_path, [response([unknown()])])
    assert result["protocol_recoveries"] == 0
    assert result["stopped_reason"] in {"invalid_protocol", "model_error"}
    assert result["messages"] == PREFIX


def test_recovery_limit_is_two(tmp_path):
    responses = [response([unknown(f"c{i}")]) for i in range(3)]
    result = run(tmp_path, responses, policy="bounded")
    assert result["protocol_recoveries"] == 2
    assert result["stopped_reason"] in {"invalid_protocol", "model_error"}


def test_malformed_missing_id_terminates_without_recovery(tmp_path):
    malformed = unknown()
    del malformed["id"]
    result = run(tmp_path, [response([malformed])], policy="bounded")
    assert result["protocol_recoveries"] == 0
    assert result["protocol_status"] == "protocol_invalid"
    assert result["messages"] == PREFIX


def test_recovery_counts_provider_budget(tmp_path):
    result = run(tmp_path, [response([unknown()]), response()], policy="bounded")
    assert result["tokens"]["total_tokens"] == 28
    audits = [(tmp_path / "protocol_audit.jsonl").read_text().splitlines()]
    assert any('"provider_raw_evidence_unavailable": false' in line for line in audits[0])
