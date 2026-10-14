"""Serial native Inspect tool calling; no action parsing or success grading."""
from __future__ import annotations

import copy
import base64
import hashlib
import asyncio
import inspect
import json
import math
from pathlib import Path
from typing import Any

from inspect_ai.model import ChatMessage, ChatMessageTool, GenerateConfig, get_model
from inspect_ai.model._model import _model_event_sink, use_model_event_sink
from inspect_ai.log._transcript import transcript
from inspect_ai.tool import ToolInfo
from jsonschema import Draft202012Validator
from pydantic import TypeAdapter
from openai import DefaultAsyncHttpxClient

from .tool_protocol import (validate_history_bindings, validate_native_output,
                             validate_wire_response)


async def _await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _await_stack(task: asyncio.Task) -> list[dict]:
    """Record code locations only; never inspect locals, arguments or source text."""
    stack = []
    current = task.get_coro()
    seen = set()
    while current is not None and id(current) not in seen and len(stack) < 64:
        seen.add(id(current))
        frame = getattr(current, "cr_frame", None) or getattr(current, "gi_frame", None)
        if frame is not None:
            stack.append({"file": Path(frame.f_code.co_filename).name,
                          "function": frame.f_code.co_name, "line": frame.f_lineno})
        current = getattr(current, "cr_await", None) or getattr(current, "gi_yieldfrom", None)
    return stack


class _ProviderCallSink:
    """Pinned Inspect event hook: capture SDK bodies before transcript condensation."""

    def __init__(self, write, turn, parent=None):
        self.write, self.turn, self.parent = write, turn, parent
        self.responses = []

    def on_pending(self, event):
        if self.parent is not None:
            self.parent.on_pending(event)
        else:
            transcript()._event(event)

    def on_complete(self, event):
        if event.call is not None:
            self.responses.append(copy.deepcopy(event.call.response))
            self.write("model_calls.jsonl", {"type": "provider_call", "turn": self.turn,
                       "request": copy.deepcopy(event.call.request),
                       "response": copy.deepcopy(event.call.response),
                       "error": event.call.error, "time": event.call.time})
        if self.parent is not None:
            self.parent.on_complete(event)
        else:
            transcript()._event_updated(event)


class NativeAgent:
    """Async native tool loop. Callbacks and tool functions may be sync or async."""

    def __init__(self, config: dict, run_dir: Path, *, http_transport: Any = None,
                 system_contract: dict | None = None):
        self.config = dict(config)
        self._system_json = None
        self._system_sha256 = None
        if system_contract is not None:
            from .r12_system import validated_system, system_identity
            system = validated_system(system_contract)
            self._system_json = json.dumps(system, sort_keys=True, allow_nan=False)
            self._system_sha256 = system_identity(system)
            phase = config.get("execution_phase", "qualification")
            if phase not in system["seed_domains"]:
                raise ValueError("unknown execution_phase")
            self._execution_phase = phase
            for key, value in system["sampling"].items():
                if key == "seed":
                    value = system["seed_domains"][phase]
                if key in config and (config[key] != value or
                                      (type(config[key]) is bool) != (type(value) is bool)):
                    raise ValueError(f"config override differs from system contract: {key}")
                self.config[key] = value
        # Freeze supplied provenance independently of mutable caller/config dictionaries.
        identity = config.get("identity_context", {"model_fingerprint": None,
                              "tokenizer_fingerprint": None, "status": "not_supplied"})
        if type(identity) is not dict or not {"model_fingerprint", "tokenizer_fingerprint"} <= identity.keys():
            raise ValueError("identity_context requires model_fingerprint and tokenizer_fingerprint")
        if self._system_json is not None:
            if (not identity["model_fingerprint"] or not identity["tokenizer_fingerprint"]
                    or identity.get("system_contract_sha256") != self._system_sha256):
                raise ValueError("R1.2 identity_context requires model/tokenizer and matching system contract hash")
        self._identity_json = json.dumps(identity, ensure_ascii=False, sort_keys=True, allow_nan=False)
        self._identity_sha256 = hashlib.sha256(self._identity_json.encode()).hexdigest()
        self._generation_guard = config.get("generation_guard")
        if self._system_json is not None and not callable(self._generation_guard):
            raise ValueError("R1.2 requires a generation_guard callable")
        self._http_transport = http_transport
        self._evidence_kind = config.get("evidence_kind", "native_provider")
        if http_transport is not None and self._evidence_kind != "synthetic_protocol_fixture":
            raise ValueError("injected HTTP transport requires synthetic_protocol_fixture identity")
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _write(self, filename: str, record: dict) -> None:
        if self._system_sha256 is not None:
            record = {**record, "system_contract_sha256": self._system_sha256,
                      "identity_context_sha256": self._identity_sha256}
        with (self.run_dir / filename).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    async def run(self, messages: list[dict], tools: dict[str, Any],
                  tool_schemas: list[dict], *, max_turns: int,
                  checkpoint_callback: Any = None, dispatch_callback: Any = None) -> dict:
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        if self._system_json is not None:
            system = json.loads(self._system_json)
            if type(max_turns) is not int or max_turns > system["sampling"]["max_turns"]:
                raise ValueError("max_turns exceeds system contract")
            if tool_schemas != system["tool_schemas"]:
                raise ValueError("tool_schemas differ from system contract")
            prompts = [m.get("content") for m in messages if m.get("role") == "system"]
            if prompts != [system["system_prompt"]]:
                raise ValueError("system prompt differs from system contract")
            # Revalidate runtime overrides even if caller mutated the public config.
            for key, value in system["sampling"].items():
                if key == "seed":
                    value = system["seed_domains"][self._execution_phase]
                if self.config.get(key) != value or (type(self.config.get(key)) is bool) != (type(value) is bool):
                    raise ValueError(f"runtime config differs from system contract: {key}")
        timeout = self.config.get("generation_timeout_seconds", 120)
        timeout_bound = 240 if self._system_json is not None else 180
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= timeout_bound:
            raise ValueError("generation_timeout_seconds must be positive and at most 180")
        counter = self.config.get("token_count")
        if not callable(counter):
            raise ValueError("token_count callback required; no context truncation is allowed")
        output_budget = 4096 if self._system_json is not None else 2048
        context_budget = 32768 if self._system_json is not None else 16384
        if self.config.get("max_tokens", output_budget) != output_budget:
            raise ValueError("R1 requires max_tokens=2048")
        if self.config.get("temperature", 0) != 0:
            raise ValueError("R1 requires temperature=0")
        infos = []
        validators = {}
        for schema in tool_schemas:
            definition = schema.get("function", schema)
            name = definition["name"]
            parameters = definition.get("parameters", {"type": "object", "properties": {}})
            if name in validators or name not in tools or parameters.get("type") != "object":
                raise ValueError("Tool schemas must uniquely match callable tools and object arguments")
            Draft202012Validator.check_schema(parameters)
            validators[name] = Draft202012Validator(parameters)
            infos.append(ToolInfo(name=name, description=definition.get("description", ""),
                                  parameters=parameters))
        if set(validators) != set(tools) or not all(callable(tool) for tool in tools.values()):
            raise ValueError("tools and tool_schemas must have identical callable names")
        history = copy.deepcopy(messages)
        adapter = TypeAdapter(list[ChatMessage])
        # Reject bad result IDs before any generation or tool execution.
        try:
            seen = validate_history_bindings(history)
        except ValueError as exc:
            self._write("protocol_audit.jsonl", {"type": "parser_output", "turn": None,
                        "status": "protocol_invalid", "reason": str(exc), "stage": "restored_prefix",
                        "raw_pre_parser_evidence_available": False,
                        "provider_raw_evidence_unavailable": True})
            raise
        adapter.validate_python(history)
        generation = GenerateConfig(temperature=0, max_tokens=output_budget, max_connections=1,
                                    seed=self.config.get("seed"), max_retries=0,
                                    timeout=math.ceil(timeout), attempt_timeout=math.ceil(timeout),
                                    parallel_tool_calls=False,
                                    extra_body={"top_k": 1, "min_p": 0})
        provider = dict(model=self.config.get("model", "openai-api/mlx/default_model"),
                        base_url=self.config.get("base_url", "http://127.0.0.1:18080/v1"),
                        api_key="local-unused", emulate_tools=False, responses_api=False,
                        stream=False, strict_tools=False, max_retries=0)
        if provider["base_url"] != "http://127.0.0.1:18080/v1" or provider["model"] != "openai-api/mlx/default_model":
            raise ValueError("R1 provider is restricted to the pinned local MLX endpoint/model")
        # Use the SDK's own httpx flavor. Inspect's default client reconstructs
        # system/environment proxy mounts even for the loopback endpoint.
        wire_responses = {}
        current_turn = None

        async def capture_wire(response):
            # Read without altering the body; SDK consumes the same cached bytes.
            body = await response.aread()
            request_body = response.request.content
            try:
                structure = json.loads(body)
            except (ValueError, UnicodeDecodeError):
                structure = None
            wire_responses[current_turn] = (structure, response.status_code)
            self._write("model_calls.jsonl", {"type": "wire_response", "turn": current_turn,
                        "request_body_base64": base64.b64encode(request_body).decode("ascii"),
                        "response_body_base64": base64.b64encode(body).decode("ascii"),
                        "response_sha256": hashlib.sha256(body).hexdigest(),
                        "status_code": response.status_code, "response": structure,
                        "identity_context_sha256": self._identity_sha256,
                        "evidence_kind": self._evidence_kind})

        client_options = {"trust_env": False, "timeout": timeout,
                          "event_hooks": {"response": [capture_wire]}}
        if self._http_transport is not None:
            client_options["transport"] = self._http_transport
        http_client = DefaultAsyncHttpxClient(**client_options)
        try:
            model = get_model(**provider, config=generation, memoize=False,
                              http_client=http_client)
        except BaseException:
            await http_client.aclose()
            raise
        events = []
        dispatches = []
        totals = dict(input_tokens=0, input_tokens_cache_read=0, input_tokens_cache_write=0,
                      full_prompt_tokens=0, output_tokens=0, total_tokens=0, total_cost=None)
        unknown_usage_turns = []
        metered_turns = set()
        recovery_count = 0

        # This is deliberately opt-in.  R12 runs are identified by the frozen
        # system contract; fixtures and callers may opt in explicitly without
        # changing the historical fail-closed R1 path.
        recovery_policy = self.config.get("protocol_recovery_policy")
        recovery_enabled = (bool(self._system_json) or recovery_policy in (True, "bounded", "r12")
                            or (isinstance(recovery_policy, dict)
                                and recovery_policy.get("enabled") is True))
        recovery_limit = 2

        def account_usage(turn: int, usage: Any, source: str) -> None:
            if turn in metered_turns:
                return
            required = ("input_tokens", "output_tokens", "total_tokens")
            valid = (type(usage) is dict and all(type(usage.get(k)) is int and usage[k] >= 0 for k in required)
                     and all(usage.get(k) is None or (type(usage[k]) is int and usage[k] >= 0)
                             for k in ("input_tokens_cache_read", "input_tokens_cache_write")))
            if not valid:
                unknown_usage_turns.append(turn)
                self._write("model_calls.jsonl", {"type": "usage_accounting", "turn": turn,
                            "status": "unknown", "source": source})
                return
            metered_turns.add(turn)
            for key in ("input_tokens", "input_tokens_cache_read", "input_tokens_cache_write",
                        "output_tokens", "total_tokens"):
                totals[key] += usage.get(key) or 0
            totals["full_prompt_tokens"] += sum(usage.get(key) or 0 for key in
                ("input_tokens", "input_tokens_cache_read", "input_tokens_cache_write"))
            if usage.get("total_cost") is not None:
                totals["total_cost"] = (totals["total_cost"] or 0) + usage["total_cost"]
            self._write("model_calls.jsonl", {"type": "usage_accounting", "turn": turn,
                        "status": "observed", "source": source, "usage": usage})

        def account_wire_usage(turn: int) -> None:
            wire = wire_responses.get(turn)
            body = wire[0] if wire and wire[1] == 200 else None
            raw_usage = body.get("usage") if type(body) is dict else None
            converted = None
            if type(raw_usage) is dict:
                prompt, completion, total = (raw_usage.get(key) for key in
                                            ("prompt_tokens", "completion_tokens", "total_tokens"))
                details = raw_usage.get("prompt_tokens_details") or {}
                cache_read = details.get("cached_tokens", 0) if type(details) is dict else None
                cache_write = details.get("cache_write_tokens", 0) if type(details) is dict else None
                if (all(type(n) is int and n >= 0 for n in (prompt, completion, total, cache_read, cache_write))
                        and cache_read + cache_write <= prompt and prompt + completion == total):
                    converted = {"input_tokens": prompt - cache_read - cache_write,
                                 "input_tokens_cache_read": cache_read,
                                 "input_tokens_cache_write": cache_write,
                                 "output_tokens": completion, "total_tokens": total}
            account_usage(turn, converted, "captured_wire_after_generation_error")

        calls = 0
        stopped = "max_turns"

        def event(record: dict) -> None:
            record = {"event_index": len(events), **record}
            events.append(record)
            self._write("events.jsonl", record)

        async def recover_unknown_wire_calls(wire_structure: Any, turn: int, reason: str) -> bool:
            """Recover a structurally valid unknown native call without executing it."""
            nonlocal recovery_count
            if not recovery_enabled or recovery_count >= recovery_limit:
                return False
            decision = validate_wire_response(wire_structure)
            if decision.status != "tool_calls":
                return False
            wire_calls = wire_structure["choices"][0]["message"].get("tool_calls") or []
            unknown = [c for c in wire_calls
                       if c.get("function", {}).get("name") not in validators]
            if not unknown or len(unknown) != len(wire_calls):
                return False
            for call in unknown:
                if recovery_count >= recovery_limit:
                    break
                name = call["function"]["name"]
                try:
                    arguments = json.loads(call["function"]["arguments"])
                except (TypeError, ValueError):
                    return False
                call_id = call["id"]
                # Preserve the provider's exact name and arguments.  In
                # particular, never turn workspace=workspace_python into a
                # registered callable.
                assistant_call = {"role": "assistant", "content": "",
                                  "tool_calls": [{"id": call_id, "type": "function",
                                                   "function": name,
                                                   "arguments": arguments}]}
                history.append(assistant_call)
                available = sorted(validators)
                content = json.dumps({"error": "unknown_tool", "tool": name,
                                      "available_tools": available},
                                     ensure_ascii=False, sort_keys=True)
                history.append(ChatMessageTool(content=content, tool_call_id=call_id,
                                               function=name).model_dump(mode="json"))
                recovery_count += 1
                self._write("protocol_audit.jsonl", {
                    "type": "protocol_recovery", "turn": turn,
                    "status": "recovered", "reason": reason,
                    "tool_call_id": call_id, "function": name,
                    "available_tools": available, "recovery_index": recovery_count,
                    "raw_pre_parser_evidence_available": wire_structure is not None,
                    "provider_raw_evidence_unavailable": wire_structure is None,
                    "evidence_kind": self._evidence_kind})
                event({"type": "protocol_recovery", "turn": turn,
                       "tool_call_id": call_id, "function": name,
                       "content": content, "recovery_index": recovery_count})
            return bool(unknown)

        try:
            for turn in range(max_turns):
                current_turn = turn
                prompt_tokens = await _await(counter(copy.deepcopy(history), copy.deepcopy(tool_schemas)))
                if type(prompt_tokens) is not int or prompt_tokens < 0:
                    raise ValueError("token_count must return a nonnegative integer")
                if prompt_tokens + output_budget > context_budget:
                    event({"type": "context_budget_exceeded", "prompt_tokens": prompt_tokens})
                    stopped = "context_budget_exceeded"
                    break
                serving_observation = None
                if self._generation_guard is not None:
                    try:
                        serving_observation = await _await(self._generation_guard())
                        guard_json = json.dumps(serving_observation, ensure_ascii=False, allow_nan=False)
                        if type(serving_observation) is not dict or len(guard_json.encode()) > 65536:
                            raise ValueError("generation_guard must return a bounded serving observation")
                        serving_observation = json.loads(guard_json)
                    except Exception as exc:
                        self._write("protocol_audit.jsonl", {"type": "generation_guard", "turn": turn,
                                    "status": "rejected", "error_type": type(exc).__name__})
                        raise
                    self._write("protocol_audit.jsonl", {"type": "generation_guard", "turn": turn,
                                "status": "verified", "serving_observation": serving_observation})
                request = {"turn": turn, "messages": copy.deepcopy(history),
                           "tools": copy.deepcopy(tool_schemas),
                           "config": generation.model_dump(mode="json"),
                           "provider": provider, "prompt_tokens_counted": prompt_tokens,
                           "http_client": {"trust_env": False, "timeout_seconds": timeout},
                           "generation_timeout_seconds": timeout,
                           "identity_context": json.loads(self._identity_json),
                           "identity_context_sha256": self._identity_sha256,
                           "evidence_kind": self._evidence_kind}
                if self._system_json is not None:
                    request["execution_phase"] = self._execution_phase
                if serving_observation is not None:
                    request["serving_observation"] = serving_observation
                self._write("model_calls.jsonl", {"type": "request", **request})
                calls += 1
                try:
                    sink = _ProviderCallSink(self._write, turn, _model_event_sink.get())
                    with use_model_event_sink(sink):
                        task = asyncio.create_task(model.generate(adapter.validate_python(history), tools=infos,
                                                  tool_choice="auto", config=generation, cache=False))
                    done, _ = await asyncio.wait({task}, timeout=timeout)
                    if not done:
                        stack = _await_stack(task)
                        task.cancel()
                        cancelled, _ = await asyncio.wait({task}, timeout=1)
                        if cancelled:
                            try:
                                task.result()
                            except (asyncio.CancelledError, Exception):
                                pass
                        error = {"type": "GenerationTimeout", "message": "Inspect generation exceeded bounded deadline",
                                 "timeout_seconds": timeout, "await_stack": stack,
                                 "cancellation_completed": bool(cancelled)}
                        self._write("model_calls.jsonl", {"type": "error", "turn": turn, "error": error})
                        self._write("protocol_audit.jsonl", {"type": "parser_output", "turn": turn,
                                    "status": "not_evaluated", "reason": "generation_timeout",
                                    "dispatch_count": 0,
                                    "raw_pre_parser_evidence_available": False,
                                    "provider_raw_evidence_unavailable": True})
                        event({"type": "model_timeout", "turn": turn, "error": error})
                        account_wire_usage(turn)
                        stopped = "model_timeout"
                        break
                    output = task.result()
                except Exception as exc:
                    error = {"type": type(exc).__name__, "message": str(exc)}
                    self._write("model_calls.jsonl", {"type": "error", "turn": turn, "error": error})
                    account_wire_usage(turn)
                    wire = wire_responses.get(turn)
                    decision = validate_wire_response(wire[0]) if wire and wire[1] == 200 else None
                    if decision and decision.status == "protocol_invalid":
                        recovered = await recover_unknown_wire_calls(
                            wire[0] if wire and wire[1] == 200 else None, turn, decision.reason)
                        self._write("protocol_audit.jsonl", {"type": "parser_output", "turn": turn,
                                    **decision.record(), "stage": "sdk_conversion", "dispatch_count": 0,
                                    "raw_pre_parser_evidence_available": bool(wire and wire[1] == 200),
                                    "provider_raw_evidence_unavailable": not bool(wire and wire[1] == 200)})
                        if recovered:
                            continue
                        event({"type": "invalid_protocol", "turn": turn, "error": decision.reason})
                        stopped = "invalid_protocol"
                    else:
                        self._write("protocol_audit.jsonl", {"type": "parser_output", "turn": turn,
                                    "status": "not_evaluated", "reason": "provider_error",
                                    "dispatch_count": 0,
                                    "raw_pre_parser_evidence_available": bool(wire and wire[1] == 200),
                                    "provider_raw_evidence_unavailable": not bool(wire and wire[1] == 200)})
                        event({"type": "model_error", "turn": turn, "error": error})
                        stopped = "model_error"
                    break
                raw = output.model_dump(mode="json")
                usage = raw.get("usage")
                self._write("model_calls.jsonl", {"type": "response", "turn": turn,
                                                 "model_output": raw, "usage": usage})
                account_usage(turn, usage, "inspect_model_output")
                # Keep the historical event shape/order, with stricter admission in
                # a separate append-only parser log linked by request turn.
                if output.choices:
                    assistant = output.choices[0].message
                    native_calls = assistant.tool_calls or []
                    stop_reason = output.choices[0].stop_reason
                else:
                    assistant, native_calls, stop_reason = None, [], None
                wire = wire_responses.get(turn)
                sdk_response = next((r for r in reversed(sink.responses) if r is not None), None)
                wire_structure = wire[0] if wire else sdk_response
                decision = validate_native_output(output, validators, seen, wire_structure)
                self._write("protocol_audit.jsonl", {"type": "parser_output", "turn": turn,
                            **decision.record(), "native_model_output": raw,
                            "wire_captured": wire is not None, "sdk_captured": sdk_response is not None,
                            "identity_context_sha256": self._identity_sha256,
                            "evidence_kind": self._evidence_kind,
                            "raw_pre_parser_evidence_available": bool(wire and wire[1] == 200),
                            "provider_raw_evidence_unavailable": not bool(wire and wire[1] == 200),
                            "dispatch_call_ids": [c.id for c in native_calls] if decision.status == "tool_calls" else []})
                if output.error or len(output.choices) != 1:
                    event({"type": "invalid_model_output", "turn": turn, "error": output.error,
                           "choice_count": len(output.choices)})
                    stopped = "invalid_model_output"
                    break
                event({"type": "model_response", "turn": turn,
                       "message": assistant.model_dump(mode="json"), "usage": usage,
                       "stop_reason": stop_reason})
                if decision.status == "protocol_invalid":
                    recovered = await recover_unknown_wire_calls(
                        wire[0] if wire and wire[1] == 200 else None, turn, decision.reason)
                    if recovered:
                        continue
                    legacy_empty = not native_calls and (stop_reason == "tool_calls" or not assistant.text.strip())
                    event({"type": "protocol_error" if legacy_empty else "invalid_protocol",
                           "turn": turn, "error": decision.reason, "stop_reason": stop_reason})
                    stopped = "protocol_error" if legacy_empty else "invalid_protocol"
                    break
                batch_ids = {c.id for c in native_calls}
                history.append(assistant.model_dump(mode="json"))
                seen.update(batch_ids)
                callback_failed = False
                for call in native_calls:
                    callback_payload = {"turn": turn, "tool_call_id": call.id,
                                        "function": call.function,
                                        "arguments": copy.deepcopy(call.arguments),
                                        "raw_result": None, "error": None}
                    if dispatch_callback:
                        try:
                            await _await(dispatch_callback({"stage": "start", **copy.deepcopy(callback_payload)}))
                        except Exception as exc:
                            error = {"type": type(exc).__name__, "message": str(exc)}
                            event({"type": "dispatch_callback_error", "stage": "start", **callback_payload, "error": error})
                            self._write("protocol_audit.jsonl", {"type": "dispatch_callback_error",
                                        "stage": "start", **callback_payload, "error": error})
                            stopped, callback_failed = "dispatch_callback_error", True
                            break
                    dispatches.append(copy.deepcopy(callback_payload))
                    self._write("protocol_audit.jsonl", {"type": "tool_dispatch", "turn": turn,
                                "tool_call_id": call.id, "function": call.function,
                                "arguments": copy.deepcopy(call.arguments),
                                "evidence_kind": self._evidence_kind})
                    raw_result = None
                    try:
                        raw_result = await _await(tools[call.function](**copy.deepcopy(call.arguments)))
                        transform = self.config.get("tool_result_transform")
                        delivered = await _await(transform(call.function, copy.deepcopy(call.arguments),
                                                          copy.deepcopy(raw_result))) if transform else raw_result
                        content = delivered if isinstance(delivered, str) else json.dumps(delivered, ensure_ascii=False)
                        error = None
                    except Exception as exc:
                        error = {"type": type(exc).__name__, "message": str(exc)}
                        content = json.dumps({"tool_execution_error": error})
                    history.append(ChatMessageTool(content=content, tool_call_id=call.id,
                                                   function=call.function).model_dump(mode="json"))
                    self._write("protocol_audit.jsonl", {"type": "tool_observation", "turn": turn,
                                "tool_call_id": call.id, "function": call.function,
                                "event_index": len(events), "error": error,
                                "evidence_kind": self._evidence_kind})
                    event({"type": "tool_result", "turn": turn, "tool_call_id": call.id,
                           "function": call.function, "arguments": call.arguments,
                           "raw_result": raw_result, "delivered_content": content, "error": error})
                    dispatches[-1].update(raw_result=copy.deepcopy(raw_result), error=copy.deepcopy(error))
                    if dispatch_callback:
                        try:
                            await _await(dispatch_callback({"stage": "end", **copy.deepcopy(dispatches[-1])}))
                        except Exception as exc:
                            callback_error = {"type": type(exc).__name__, "message": str(exc)}
                            event({"type": "dispatch_callback_error", "stage": "end", **callback_payload, "error": callback_error})
                            self._write("protocol_audit.jsonl", {"type": "dispatch_callback_error",
                                        "stage": "end", **callback_payload, "error": callback_error})
                            stopped, callback_failed = "dispatch_callback_error", True
                            break
                if callback_failed:
                    break
                if checkpoint_callback:
                    await _await(checkpoint_callback(copy.deepcopy(history), len(events) - 1))
                if not native_calls:
                    stopped = "assistant_finished"
                    break
        finally:
            await http_client.aclose()
        return {"messages": history, "events": events, "modelcall_count": calls,
                "tokens": totals, "stopped_reason": stopped,
                "protocol_status": "protocol_invalid" if stopped in
                    ("protocol_error", "invalid_protocol", "invalid_model_output") else
                    ("not_evaluated" if stopped in ("model_error", "model_timeout", "context_budget_exceeded") else "valid"),
                "token_usage_unknown": bool(unknown_usage_turns),
                "unknown_usage_turns": unknown_usage_turns,
                "protocol_recoveries": recovery_count,
                "token_totals_scope": "observed_subtotal" if unknown_usage_turns else "all_requests",
                "tool_dispatch": dispatches,
                **({"system_contract_sha256": self._system_sha256} if self._system_sha256 else {}),
                "identity_context_sha256": self._identity_sha256,
                "evidence_kind": self._evidence_kind}
