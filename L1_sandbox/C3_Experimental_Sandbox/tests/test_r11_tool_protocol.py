"""Synthetic HTTP protocol tests through pinned OpenAI SDK + Inspect; no LLM/Docker."""
import asyncio
import base64
import json

import pytest

from c3_sandbox.r1 import agent
from c3_sandbox.r1.tool_protocol import run_protocol_fixture

SCHEMA = [{"type": "function", "function": {"name": "nonce", "description": "read fixture nonce",
          "parameters": {"type": "object", "properties": {"value": {"type": "string"}},
                         "required": ["value"], "additionalProperties": False}}}]
PREFIX = [{"role": "user", "content": "synthetic protocol test"}]


def wire(calls=None, finish=None, content=None):
    return {"id": "synthetic-completion", "object": "chat.completion", "created": 0,
            "model": "mlx/default_model", "choices": [{"index": 0,
            "finish_reason": finish or ("tool_calls" if calls else "stop"),
            "message": {"role": "assistant", "content": content if content is not None else
                        ("" if calls else "synthetic final"), "tool_calls": calls}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            "synthetic_unknown_sdk_field": {"preserve": [1, 2]}}


def call(identifier="fixture-call", name="nonce", arguments='{"value":"fixture"}'):
    return {"id": identifier, "type": "function", "function": {"name": name, "arguments": arguments}}


def records(path, filename):
    return [json.loads(line) for line in (path / filename).read_text().splitlines()]


def run(tmp_path, responses, executed, **kwargs):
    return asyncio.run(run_protocol_fixture(responses, tmp_path, messages=kwargs.pop("messages", PREFIX),
           tools={"nonce": lambda value: executed.append(value) or {"observed": value}},
           tool_schemas=SCHEMA, **kwargs))


def test_actual_sdk_inspect_dispatch_and_lossless_wire_identity(tmp_path):
    invoked = []
    response = wire([call()])
    result = run(tmp_path, [response, wire()], invoked,
                 identity_context={"model_fingerprint": {"sha256": "synthetic-model"},
                                   "tokenizer_fingerprint": {"sha256": "synthetic-tokenizer"}})
    assert invoked == ["fixture"]
    assert result["stopped_reason"] == "assistant_finished"
    assert result["is_synthetic"] and result["evidence_kind"] == "synthetic_protocol_fixture"
    assert result["messages"][2]["tool_call_id"] == "fixture-call"
    model_records = records(tmp_path, "model_calls.jsonl")
    requests = [r for r in model_records if r["type"] == "request"]
    assert len(requests) == 2 and all(r["identity_context"]["model_fingerprint"]["sha256"] ==
                                    "synthetic-model" for r in requests)
    sdk = next(r for r in model_records if r["type"] == "provider_call")
    assert sdk["response"]["synthetic_unknown_sdk_field"] == response["synthetic_unknown_sdk_field"]
    actual = next(r for r in model_records if r["type"] == "wire_response")
    assert json.loads(base64.b64decode(actual["response_body_base64"])) == response
    sent = json.loads(base64.b64decode(actual["request_body_base64"]))
    assert sent["tools"][0]["function"]["name"] == "nonce"
    assert sent["temperature"] == 0 and sent["seed"] == 2026 and sent["max_tokens"] == 2048
    assert sent.get("stream", False) is False
    audit = records(tmp_path, "protocol_audit.jsonl")
    parsed = next(r for r in audit if r["type"] == "parser_output")
    dispatch = next(r for r in audit if r["type"] == "tool_dispatch")
    observation = next(r for r in audit if r["type"] == "tool_observation")
    assert parsed["status"] == "tool_calls" and parsed["wire_captured"] and parsed["sdk_captured"]
    assert parsed["dispatch_call_ids"] == [dispatch["tool_call_id"]] == [observation["tool_call_id"]]
    assert parsed["turn"] == dispatch["turn"] == observation["turn"] == 0
    assert result["events"][observation["event_index"]]["raw_result"] == {"observed": "fixture"}


def test_text_final_no_dispatch(tmp_path):
    invoked = []
    result = run(tmp_path, [wire(content="Action: nonce\nAction Input: fake")], invoked)
    assert not invoked and result["stopped_reason"] == "assistant_finished"
    assert records(tmp_path, "protocol_audit.jsonl")[0]["status"] == "assistant_finished"


# Seven malformed response classes, each checked against actual SDK/Inspect path.
@pytest.mark.parametrize("response", [
    wire([], finish="tool_calls", content=""),
    wire([call(arguments='{"value":')]),
    wire([{k: v for k, v in call().items() if k != "id"}]),
    wire([call()], finish="length"),
    wire([call(), call()]),
    wire([call(name="unknown")]),
    wire([call(arguments='{"value":4}')]),
], ids=["empty_tool_payload", "invalid_json", "missing_id", "token_truncation",
        "duplicate_id", "unknown_function", "invalid_argument_schema"])
def test_seven_malformed_batches_never_dispatch(tmp_path, response):
    invoked, checkpoints = [], []
    result = run(tmp_path, [response], invoked, checkpoint_callback=lambda *args: checkpoints.append(args))
    assert not invoked and not checkpoints and result["messages"] == PREFIX
    assert result["protocol_status"] == "protocol_invalid"
    audit = records(tmp_path, "protocol_audit.jsonl")
    assert any(r.get("status") == "protocol_invalid" for r in audit)
    assert not any(r["type"] == "tool_dispatch" for r in audit)
    assert any(r["type"] == "wire_response" for r in records(tmp_path, "model_calls.jsonl"))


@pytest.mark.parametrize("arguments", ['fixture', '{"value":"fixture"}"',
                                       '{"value":"a","value":"b"}', '{"value":NaN}'])
def test_inspect_yaml_or_json_recovery_cannot_authorize_dispatch(tmp_path, arguments):
    invoked = []
    result = run(tmp_path, [wire([call(arguments=arguments)])], invoked)
    assert not invoked and result["protocol_status"] == "protocol_invalid"


def test_mismatched_result_id_rejected_before_request(tmp_path):
    prefix = PREFIX + [{"role": "assistant", "content": "", "tool_calls": [{
        "id": "expected", "function": "nonce", "arguments": {"value": "fixture"}, "type": "function"}]},
        {"role": "tool", "content": "synthetic result", "tool_call_id": "wrong", "function": "nonce"}]
    with pytest.raises(ValueError, match="result binding"):
        run(tmp_path, [wire()], [], messages=prefix)
    assert not (tmp_path / "model_calls.jsonl").exists()
    assert records(tmp_path, "protocol_audit.jsonl")[0]["status"] == "protocol_invalid"


def test_immutable_identity_context_and_transport_guard(tmp_path):
    context = {"model_fingerprint": {"sha256": "before"}, "tokenizer_fingerprint": {"sha256": "tokenizer"}}
    config = {"identity_context": context}
    instance = agent.NativeAgent(config, tmp_path)
    context["model_fingerprint"]["sha256"] = "after"
    instance.config["identity_context"]["tokenizer_fingerprint"]["sha256"] = "changed"
    frozen = json.loads(instance._identity_json)
    assert frozen["model_fingerprint"]["sha256"] == "before"
    assert frozen["tokenizer_fingerprint"]["sha256"] == "tokenizer"
    with pytest.raises(ValueError, match="synthetic_protocol_fixture"):
        agent.NativeAgent({}, tmp_path, http_transport=object())


@pytest.mark.parametrize("fail_stage", [None, "start", "end"])
def test_dispatch_snapshot_hook_and_fail_closed(tmp_path, fail_stage):
    invoked, snapshots = [], []

    async def snapshot(payload):
        snapshots.append(payload)
        if payload["stage"] == fail_stage:
            raise ValueError("snapshot rejected")

    result = run(tmp_path, [wire([call("first"), call("second")]), wire()], invoked,
                 dispatch_callback=snapshot)
    if fail_stage == "start":
        assert not invoked and not result["tool_dispatch"]
    elif fail_stage == "end":
        assert invoked == ["fixture"] and len(result["tool_dispatch"]) == 1
        assert snapshots[-1]["raw_result"] == {"observed": "fixture"}
    else:
        assert invoked == ["fixture", "fixture"]
        assert [s["stage"] for s in snapshots] == ["start", "end", "start", "end"]
        assert snapshots[1]["tool_call_id"] == result["tool_dispatch"][0]["tool_call_id"] == "first"
    if fail_stage:
        assert result["stopped_reason"] == "dispatch_callback_error"
        assert result["modelcall_count"] == 1
        assert result["events"][-1]["type"] == "dispatch_callback_error"


def test_invalid_payload_never_reaches_dispatch_callback(tmp_path):
    snapshots = []
    result = run(tmp_path, [wire([call()], finish="length")], [],
                 dispatch_callback=lambda payload: snapshots.append(payload))
    assert not snapshots and not result["tool_dispatch"]


def test_missing_id_sdk_failure_preserves_wire_usage(tmp_path):
    missing = {k: v for k, v in call().items() if k != "id"}
    result = run(tmp_path, [wire([missing])], [])
    assert result["protocol_status"] == "protocol_invalid"
    assert result["tokens"]["input_tokens"] == result["tokens"]["full_prompt_tokens"] == 10
    assert result["tokens"]["output_tokens"] == 4
    assert result["tokens"]["total_tokens"] == 14
    assert result["token_usage_unknown"] is False
    accounting = [r for r in records(tmp_path, "model_calls.jsonl") if r["type"] == "usage_accounting"]
    assert len(accounting) == 1 and accounting[0]["status"] == "observed"
    assert accounting[0]["source"] == "captured_wire_after_generation_error"


def test_success_usage_is_counted_once(tmp_path):
    result = run(tmp_path, [wire([call()]), wire()], [])
    assert result["tokens"]["total_tokens"] == 28
    assert result["tokens"]["output_tokens"] == 8
    assert result["token_usage_unknown"] is False
    accounting = [r for r in records(tmp_path, "model_calls.jsonl") if r["type"] == "usage_accounting"]
    assert len(accounting) == 2
    assert all(r["source"] == "inspect_model_output" for r in accounting)


@pytest.mark.parametrize("usage", [None, {}, {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 999}])
def test_sdk_failure_without_valid_metering_marks_unknown(tmp_path, usage):
    missing = {k: v for k, v in call().items() if k != "id"}
    response = wire([missing])
    response["usage"] = usage
    result = run(tmp_path, [response], [])
    assert result["token_usage_unknown"] is True
    assert result["unknown_usage_turns"] == [0]
    assert result["token_totals_scope"] == "observed_subtotal"
    assert result["tokens"]["total_tokens"] == 0  # subtotal, explicitly not observed zero usage


def test_wire_cache_metering_on_sdk_conversion_failure(tmp_path):
    missing = {k: v for k, v in call().items() if k != "id"}
    response = wire([missing])
    response["usage"]["prompt_tokens_details"] = {"cached_tokens": 3, "cache_write_tokens": 2}
    result = run(tmp_path, [response], [])
    assert result["tokens"]["input_tokens"] == 5
    assert result["tokens"]["input_tokens_cache_read"] == 3
    assert result["tokens"]["input_tokens_cache_write"] == 2
    assert result["tokens"]["full_prompt_tokens"] == 10
    assert result["tokens"]["total_tokens"] == 14
