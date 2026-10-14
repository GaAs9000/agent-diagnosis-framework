"""Strict native protocol admission. Synthetic fixtures are never LLM evidence."""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProtocolDecision:
    status: str
    reason: str | None = None

    def record(self) -> dict:
        return asdict(self)


def _invalid(reason: str) -> ProtocolDecision:
    return ProtocolDecision("protocol_invalid", reason)


def _strict_object(arguments: Any) -> dict:
    if not isinstance(arguments, str):
        raise ValueError("wire arguments must be a JSON string")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError("nonfinite JSON value")

    value = json.loads(arguments, object_pairs_hook=unique, parse_constant=nonfinite)
    if type(value) is not dict:
        raise ValueError("tool arguments must be an object")
    return value


def validate_wire_response(response: Any) -> ProtocolDecision:
    """Validate unconverted OpenAI-compatible response, without repairing it."""
    if type(response) is not dict or type(response.get("choices")) is not list or len(response["choices"]) != 1:
        return _invalid("expected exactly one wire choice")
    choice = response["choices"][0]
    if type(choice) is not dict or type(choice.get("message")) is not dict:
        return _invalid("missing native assistant message")
    message = choice["message"]
    if message.get("role") != "assistant":
        return _invalid("wire message role must be assistant")
    calls = message.get("tool_calls")
    if calls is None:
        calls = []
    if type(calls) is not list:
        return _invalid("tool_calls must be a list")
    finish = choice.get("finish_reason")
    if finish == "length":
        return _invalid("token truncation; no payload may dispatch")
    if finish == "tool_calls" and not calls:
        return _invalid("tool_calls stop reason without native tool calls")
    identifiers = set()
    for call in calls:
        if type(call) is not dict or not isinstance(call.get("id"), str) or not call["id"].strip():
            return _invalid("missing tool call ID")
        if call["id"] in identifiers:
            return _invalid("duplicate tool call ID")
        identifiers.add(call["id"])
        function = call.get("function")
        if call.get("type") != "function" or type(function) is not dict or not isinstance(function.get("name"), str):
            return _invalid("invalid native function shape")
        try:
            _strict_object(function.get("arguments"))
        except (ValueError, TypeError, RecursionError) as exc:
            return _invalid("invalid JSON arguments: " + str(exc))
    if calls:
        if finish not in ("stop", "tool_calls"):
            return _invalid("unsupported tool completion finish reason")
        return ProtocolDecision("tool_calls")
    content = message.get("content")
    if finish != "stop" or not isinstance(content, str) or not content.strip():
        return _invalid("empty or nonterminal final completion")
    return ProtocolDecision("assistant_finished")


def validate_native_output(output: Any, validators: dict, seen: set[str],
                           wire_response: Any = None) -> ProtocolDecision:
    """Validate full Inspect batch and, when captured, bind it to the wire calls."""
    if wire_response is not None:
        decision = validate_wire_response(wire_response)
        if decision.status == "protocol_invalid":
            return decision
    if output.error or len(output.choices) != 1:
        return _invalid("invalid Inspect model output")
    choice = output.choices[0]
    calls = choice.message.tool_calls or []
    if choice.stop_reason == "max_tokens":
        return _invalid("token truncation; no payload may dispatch")
    if not calls:
        if choice.stop_reason == "tool_calls" or not choice.message.text.strip():
            return _invalid("tool_calls stop reason without native tool calls" if choice.stop_reason == "tool_calls"
                            else "empty final completion without native tool calls")
        if choice.stop_reason != "stop":
            return _invalid("nonterminal completion")
    batch = set()
    for call in calls:
        if not isinstance(call.id, str) or not call.id.strip() or call.id in seen or call.id in batch:
            return _invalid("missing or duplicate tool call ID")
        batch.add(call.id)
        if call.type != "function" or call.parse_error:
            return _invalid(call.parse_error or "non-function tool call")
        if call.function not in validators:
            return _invalid("unknown tool name: " + call.function)
        if type(call.arguments) is not dict:
            return _invalid("tool arguments must be an object")
        errors = list(validators[call.function].iter_errors(call.arguments))
        if errors:
            return _invalid("; ".join(error.message for error in errors))
    if wire_response is not None:
        wire_calls = wire_response["choices"][0]["message"].get("tool_calls") or []
        if len(wire_calls) != len(calls):
            return _invalid("wire/Inspect tool call count mismatch")
        for wire, native in zip(wire_calls, calls):
            if (wire["id"] != native.id or wire["function"]["name"] != native.function
                    or _strict_object(wire["function"]["arguments"]) != native.arguments):
                return _invalid("wire/Inspect tool call binding mismatch")
    return ProtocolDecision("tool_calls" if calls else "assistant_finished")


def validate_history_bindings(history: list[dict]) -> set[str]:
    """A restored prefix must contain complete, exact native call/result pairs."""
    pending, seen = {}, set()
    for message in history:
        if message["role"] == "assistant":
            if pending:
                raise ValueError("Unresolved calls in restored prefix")
            for call in message.get("tool_calls") or []:
                identifier = call.get("id")
                if not isinstance(identifier, str) or not identifier.strip() or identifier in seen:
                    raise ValueError("Missing or duplicate tool call ID in restored prefix")
                pending[identifier] = call["function"]
                seen.add(identifier)
        elif message["role"] == "tool":
            identifier = message.get("tool_call_id")
            if identifier not in pending or message.get("function") != pending[identifier]:
                raise ValueError("Invalid restored tool result binding")
            del pending[identifier]
        elif pending:
            raise ValueError("Interrupted tool results in restored prefix")
    if pending:
        raise ValueError("Pending calls cannot be restored")
    return seen


async def run_protocol_fixture(responses: list[dict], run_dir: Path, *, messages: list[dict],
                               tools: dict, tool_schemas: list[dict], max_turns: int | None = None,
                               checkpoint_callback=None, dispatch_callback=None, identity_context: dict | None = None) -> dict:
    """Exercise actual SDK/Inspect/NativeAgent with scripted HTTP, injected tools.

    No model/server request is made. Tools may be real Docker callbacks provided by
    the caller. Every fixture request/result is explicitly synthetic protocol evidence.
    """
    import httpx2 as httpx
    from .agent import NativeAgent

    scripted = iter(copy.deepcopy(responses))

    def respond(request):
        if str(request.url) != "http://127.0.0.1:18080/v1/chat/completions":
            raise AssertionError("fixture attempted unexpected endpoint")
        try:
            response = next(scripted)
        except StopIteration as exc:
            raise AssertionError("fixture response script exhausted") from exc
        return httpx.Response(200, json=response, request=request)

    instance = NativeAgent({"seed": 2026, "token_count": lambda m, t: 10,
                            "identity_context": identity_context or {
                                "model_fingerprint": "synthetic-scripted-provider",
                                "tokenizer_fingerprint": "synthetic-fixed-counter"},
                            "evidence_kind": "synthetic_protocol_fixture"}, run_dir,
                           http_transport=httpx.MockTransport(respond))
    result = await instance.run(messages, tools, tool_schemas,
                               max_turns=max_turns or len(responses),
                               checkpoint_callback=checkpoint_callback, dispatch_callback=dispatch_callback)
    result["is_synthetic"] = True
    result["evidence_kind"] = "synthetic_protocol_fixture"
    return result
