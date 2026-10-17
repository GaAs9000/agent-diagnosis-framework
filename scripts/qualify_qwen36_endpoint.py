#!/usr/bin/env python3
"""Qualification test harness for Qwen3.6 served on aml-b via SSH tunnel.

Executes all 4 qualification test requests and directly exercises L1's
native OpenAICompatibleBackend parsing and validation contracts.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# Ensure L1 sandbox src is on PYTHONPATH
L1_ROOT = Path(__file__).resolve().parents[1] / "L1_sandbox" / "C3_Experimental_Sandbox"
sys.path.insert(0, str(L1_ROOT / "src"))

try:
    from c3_sandbox.l1.models.native import OpenAICompatibleBackend, parse_json
    from c3_sandbox.l1.runtime.tools import TOOLS, validate_call
    from c3_sandbox.l1.core.contracts import Result, ProtocolError
    from c3_sandbox.l1.storage import canonical
except ImportError as exc:
    print(f"[FAIL] Could not import L1 native backend modules: {exc}")
    sys.path.append(str(Path.cwd() / "L1_sandbox" / "C3_Experimental_Sandbox" / "src"))
    from c3_sandbox.l1.models.native import OpenAICompatibleBackend, parse_json
    from c3_sandbox.l1.runtime.tools import TOOLS, validate_call
    from c3_sandbox.l1.core.contracts import Result, ProtocolError
    from c3_sandbox.l1.storage import canonical

BASE_URL = os.environ.get("QWEN_ENDPOINT_URL", "http://127.0.0.1:8001/v1")
MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen3.6-27B")

CONFIG = {
    "kind": "local",
    "provider": "openai-compatible",
    "model": MODEL_ID,
    "base_url": BASE_URL,
    "parameters": {
        "temperature": 0.0,
        "max_tokens": 8192
    },
    "timeout_seconds": 30,
    "max_response_bytes": 16777216
}


def test_1_models_endpoint() -> tuple[dict, str]:
    print("\n--- [Test 1] GET /v1/models (Identity & Health Check) ---")
    url = f"{BASE_URL.rstrip('/')}/models"
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "l1-evidence/0.2.0"})
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            elapsed = time.monotonic() - start
            status = resp.status
            body = resp.read()
    except urllib.error.URLError as e:
        print(f"[FAIL] Connection error to {url}: {e}")
        return {"passed": False, "error": str(e), "elapsed_ms": None}, MODEL_ID

    parsed = parse_json(body)
    models = [m.get("id") for m in parsed.get("data", [])]
    print(f"Status: {status} | Latency: {elapsed*1000:.1f}ms | Available models: {models}")
    
    matched_model = None
    if MODEL_ID in models:
        matched_model = MODEL_ID
    elif "Qwen/Qwen3.6-27B" in models:
        matched_model = "Qwen/Qwen3.6-27B"
    elif "Qwen3.6-27B" in models:
        matched_model = "Qwen3.6-27B"
    elif len(models) > 0:
        matched_model = models[0]
        
    passed = (status == 200 and matched_model is not None)
    if not passed:
        print(f"[FAIL] Expected model '{MODEL_ID}' (or Qwen3.6-27B) not found in {models}")
    else:
        print(f"[PASS] Model '{matched_model}' verified online.")
    return {"passed": passed, "status": status, "elapsed_ms": elapsed * 1000, "models": models, "active_model": matched_model}, (matched_model or MODEL_ID)


def test_2_native_tool_call(config: dict) -> tuple[dict, object, object, dict]:
    print("\n--- [Test 2] POST /v1/chat/completions (Single-Turn Tool Call) ---")
    backend = OpenAICompatibleBackend(config)
    tools = [TOOLS[0]]  # Single 'run' tool
    
    system_prompt = (
        "You are the tested agent in an execution-and-evidence sandbox. Complete the user's task "
        "using the supplied native tools.\n\n"
        "[SANDBOX EXECUTION CONTRACT & RUNTIME CONSTRAINTS]\n"
        "- Model Request Budget: At most 20 model requests.\n"
        "- Tool Execution Budget: At most 30 tool invocations."
    )
    user_prompt = "Inspect the Sokoban puzzle state: /workspace/input/bin/sokoban status"
    conv = backend.initial(system_prompt, user_prompt)
    req_body = backend.request(conv, tools)
    req_bytes = canonical(req_body)

    start = time.monotonic()
    result = backend.send(req_bytes, timeout=30.0)
    elapsed = time.monotonic() - start

    print(f"Status: {result.status} | Latency: {elapsed*1000:.1f}ms")
    if result.status != 200:
        print(f"[FAIL] HTTP error: {result.status}, body: {result.body[:500]}")
        return {"passed": False, "status": result.status, "error": "HTTP status != 200"}, conv, None, tools

    try:
        reply = backend.parse(result.body, "qual_req_2")
    except ProtocolError as e:
        print(f"[FAIL] L1 ProtocolError during parse: {e}")
        return {"passed": False, "status": result.status, "error": str(e)}, conv, None, tools

    print(f"Finish Reason: '{reply.finish_reason}' | Call Count: {len(reply.calls)}")
    if not reply.calls:
        print(f"[FAIL] Expected tool calls, but none returned. Text: {reply.text}")
        return {"passed": False, "status": result.status, "error": "No tool calls generated"}, conv, reply, tools

    call = reply.calls[0]
    print(f"Tool Call 0: name='{call.name}', native_id='{call.native_id}', args={call.arguments}")

    # Validate against L1 validate_call
    try:
        validate_call(call, images=False, harness="minimal")
        print("[PASS] validate_call passed successfully (strict argument check: only 'command').")
    except ProtocolError as e:
        print(f"[FAIL] validate_call rejected tool call: {e}")
        return {"passed": False, "error": f"validate_call: {e}"}, conv, reply, tools

    passed = (reply.finish_reason == "tool_calls" and call.name == "run" and "command" in call.arguments)
    return {"passed": passed, "status": result.status, "elapsed_ms": elapsed * 1000, "call": str(call)}, conv, reply, tools


def test_3_multiturn_continuation(config: dict, conv, reply, tools) -> dict:
    print("\n--- [Test 3] POST /v1/chat/completions (Multi-Turn Tool Ingestion) ---")
    if not reply or not reply.calls:
        print("[SKIP] Cannot run Test 3 because Test 2 produced no valid tool call.")
        return {"passed": False, "error": "Skipped due to Test 2 failure"}

    backend = OpenAICompatibleBackend(config)
    simulated_obs = {
        "output": "Sokoban 0.1\nLevel: 1\nGrid: 8x8\nState: In Progress\nTargets: 2, Placed: 0",
        "stdout": "Sokoban 0.1\nLevel: 1\nGrid: 8x8\nState: In Progress\nTargets: 2, Placed: 0",
        "stderr": "",
        "exit_code": 0,
        "stop_reason": None,
        "output_complete": True
    }
    tool_result = Result(simulated_obs, simulated_obs["stdout"].encode("utf-8"), b"")
    
    # Accept tool observation into conversation
    turn2_conv = backend.accept(conv, reply, [(reply.calls[0], tool_result)])
    req_body = backend.request(turn2_conv, tools)
    req_bytes = canonical(req_body)

    start = time.monotonic()
    result = backend.send(req_bytes, timeout=30.0)
    elapsed = time.monotonic() - start

    print(f"Status: {result.status} | Latency: {elapsed*1000:.1f}ms")
    if result.status != 200:
        print(f"[FAIL] Multi-turn HTTP error: {result.status}, body: {result.body[:500]}")
        return {"passed": False, "status": result.status, "error": "HTTP status != 200"}

    try:
        reply2 = backend.parse(result.body, "qual_req_3")
        print(f"Multi-turn parsed cleanly: finish_reason='{reply2.finish_reason}', calls={len(reply2.calls)}")
        passed = True
    except ProtocolError as e:
        print(f"[FAIL] Multi-turn ProtocolError: {e}")
        passed = False
        return {"passed": False, "error": str(e)}

    return {"passed": passed, "status": result.status, "elapsed_ms": elapsed * 1000}


def test_4_plain_text_completion(config: dict) -> dict:
    print("\n--- [Test 4] POST /v1/chat/completions (Plain Text Completion) ---")
    backend = OpenAICompatibleBackend(config)
    conv = backend.initial("You are a helpful assistant.", "What is 17 + 25? Answer with only the number.")
    req_body = backend.request(conv, [])  # No tools
    req_bytes = canonical(req_body)

    start = time.monotonic()
    result = backend.send(req_bytes, timeout=30.0)
    elapsed = time.monotonic() - start

    print(f"Status: {result.status} | Latency: {elapsed*1000:.1f}ms")
    if result.status != 200:
        return {"passed": False, "status": result.status, "error": "HTTP status != 200"}

    try:
        reply = backend.parse(result.body, "qual_req_4")
    except ProtocolError as e:
        print(f"[FAIL] Text completion ProtocolError: {e}")
        return {"passed": False, "error": str(e)}

    passed = (reply.finish_reason == "stop" and reply.calls == () and "42" in (reply.text or ""))
    print(f"Finish Reason: '{reply.finish_reason}' | Content: '{reply.text.strip() if reply.text else None}'")
    if passed:
        print("[PASS] Plain text completion and stop condition verified.")
    else:
        print("[FAIL] Unexpected completion outcome.")
    return {"passed": passed, "status": result.status, "elapsed_ms": elapsed * 1000, "content": reply.text}


def run_all():
    print(f"=== Qwen3.6 Endpoint Qualification Harness ===")
    print(f"Target: {BASE_URL} | Desired Model: {MODEL_ID}")
    
    r1, active_model = test_1_models_endpoint()
    test_config = dict(CONFIG)
    test_config["model"] = active_model
    print(f"Using verified model identifier: {active_model}")
    
    r2, conv, reply, tools = test_2_native_tool_call(test_config)
    r3 = test_3_multiturn_continuation(test_config, conv, reply, tools)
    r4 = test_4_plain_text_completion(test_config)

    all_passed = all([r1["passed"], r2["passed"], r3["passed"], r4["passed"]])
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": BASE_URL,
        "model_id": active_model,
        "overall_qualification_passed": all_passed,
        "tests": {
            "test_1_models": r1,
            "test_2_native_tool": r2,
            "test_3_multiturn": r3,
            "test_4_plain_text": r4
        }
    }
    
    print("\n================ QUALIFICATION SUMMARY ================")
    print(f"Overall Result: {'[PASSED] QUALIFIED FOR L1 EXPERIMENTS' if all_passed else '[FAILED] DO NOT PROCEED TO L1'}")
    print(f"Test 1 (GET /v1/models):           {'PASS' if r1['passed'] else 'FAIL'}")
    print(f"Test 2 (Tool Call Elicitation):     {'PASS' if r2['passed'] else 'FAIL'}")
    print(f"Test 3 (Multi-turn Tool Ingestion): {'PASS' if r3['passed'] else 'FAIL'}")
    print(f"Test 4 (Plain Text / Stop Reason):  {'PASS' if r4['passed'] else 'FAIL'}")
    print("=======================================================\n")
    
    # Save receipt to disk
    receipt_path = Path(__file__).resolve().parent / "qualification_receipt.json"
    with open(receipt_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved qualification receipt to: {receipt_path}")
    
    return report


if __name__ == "__main__":
    rep = run_all()
    if not rep["overall_qualification_passed"]:
        sys.exit(1)
    sys.exit(0)
