"""Comprehensive tests for C3 Unified Model Gateway and Dual-Backend Architecture.

Verifies:
1. Rule 1: Unified upper-level protocol; native wire protocols preserved per backend.
2. Rule 2: Strict fail-closed, NO silent fallback between backends on error.
3. Rule 3: Backend identity is experiment identity (frozen config, hashes, runtime metadata).
4. Rule 4: Unified evaluation with separate cost accounting (API USD + cache vs Local tokens + hardware).
5. Config-driven backend switching (switching config alone switches engine transparently).
6. Gateway-to-D7 diagnostic bridge (GatewayDiagnosticBackend).
"""
import asyncio
import json
from pathlib import Path
import pytest

from c3_sandbox.gateway import (
    ApiBackend,
    BackendError,
    BackendRegistry,
    CostRecord,
    GatewayDiagnosticBackend,
    LocalBackend,
    ModelBackend,
    ModelGateway,
    ModelIdentity,
    ModelResponse,
    ModelToolCall,
    SavedReferenceBackend,
    UnifiedMessage,
)


# ==============================================================================
# 1. Identity & Data Structure Invariants
# ==============================================================================

def test_model_identity_immutable_and_hashable():
    ident = ModelIdentity(
        backend="api",
        provider="gemini",
        model_name="gemini-3.5-flash-lite",
        fingerprint="fp_12345",
        tokenizer_fingerprint="tok_67890",
        metadata={"max_turns": 20},
    )
    d = ident.to_dict()
    assert d["backend"] == "api"
    assert d["provider"] == "gemini"
    assert d["model_name"] == "gemini-3.5-flash-lite"
    h1 = ident.identity_hash
    h2 = ident.identity_hash
    assert len(h1) == 64
    assert h1 == h2  # Deterministic SHA-256


def test_cost_record_separation_rule_4():
    # API cost record has cost_usd, cached_tokens, no hardware_info
    api_cost = CostRecord(
        backend_type="api",
        prompt_tokens=1000,
        completion_tokens=50,
        cached_tokens=800,
        total_tokens=1050,
        cost_usd=0.00035,
        latency_ms=450.0,
    )
    d_api = api_cost.to_dict()
    assert d_api["cost_usd"] == 0.00035
    assert d_api["cached_tokens"] == 800
    assert d_api["hardware_info"] is None

    # Local cost record has NO cost_usd (None/null), has hardware_info
    local_cost = CostRecord(
        backend_type="local",
        prompt_tokens=500,
        completion_tokens=120,
        total_tokens=620,
        cost_usd=None,
        latency_ms=1200.0,
        hardware_info={"accelerator": "metal", "device": "Apple Silicon GPU"},
    )
    d_local = local_cost.to_dict()
    assert d_local["cost_usd"] is None
    assert d_local["hardware_info"]["accelerator"] == "metal"


# ==============================================================================
# 2. Backend Registry & Config-Driven Switching
# ==============================================================================

def test_backend_registry_creates_api_and_local_and_saved():
    api_cfg = {
        "backend": "api",
        "provider": "gemini",
        "model": "gemini-3.5-flash-lite",
        "api_key": "test-key-123",
    }
    local_cfg = {
        "backend": "local",
        "provider": "mlx",
        "model": "Qwen3-Coder-30B",
        "endpoint": "http://127.0.0.1:18080/v1/chat/completions",
    }
    saved_cfg = {
        "backend": "saved_reference",
        "model": "Qwen3-4B",
    }

    backend_api = BackendRegistry.create(api_cfg)
    assert isinstance(backend_api, ApiBackend)
    assert backend_api.backend_type == "api"
    assert backend_api.model_name == "gemini-3.5-flash-lite"

    backend_local = BackendRegistry.create(local_cfg)
    assert isinstance(backend_local, LocalBackend)
    assert backend_local.backend_type == "local"
    assert backend_local.model_name == "Qwen3-Coder-30B"

    backend_saved = BackendRegistry.create(saved_cfg)
    assert isinstance(backend_saved, SavedReferenceBackend)
    assert backend_saved.backend_type == "saved_reference"


def test_backend_registry_rejects_unknown_backend():
    with pytest.raises(BackendError, match="Unknown model backend type"):
        BackendRegistry.create({"backend": "unsupported_magic_backend", "model": "test"})


def test_model_gateway_from_config_three_roles():
    config = {
        "models": {
            "executor": {
                "backend": "api",
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "sk-mock-key",
            },
            "reference": {
                "backend": "saved_reference",
                "model": "qwen3-4b-mlx",
            },
            "diagnostician": {
                "backend": "api",
                "provider": "gemini",
                "model": "gemini-3.5-flash-lite",
                "api_key": "AQ.MockKey",
            },
        }
    }
    gateway = ModelGateway.from_config(config)
    assert isinstance(gateway.executor, ApiBackend)
    assert gateway.executor.identity.provider == "openai"
    assert isinstance(gateway.reference, SavedReferenceBackend)
    assert isinstance(gateway.diagnostician, ApiBackend)
    assert gateway.diagnostician.identity.provider == "gemini"


# ==============================================================================
# 3. Rule 1: Native Wire Protocols & Tool Call Preservation
# ==============================================================================

def test_api_backend_gemini_native_wire_and_thought_signature():
    """Verify Gemini wire protocol preserves thoughtSignature and formats tools correctly."""
    sent_requests = []

    def mock_gemini_transport(payload_raw: bytes) -> bytes:
        sent_requests.append(json.loads(payload_raw.decode("utf-8")))
        return json.dumps({
            "responseId": "gemini_resp_001",
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "thought": "Thinking about the task...",
                            "thoughtSignature": "sig_valid_12345",
                            "functionCall": {
                                "id": "fc_001",
                                "name": "workspace_python",
                                "args": {"code": "print('hello world')"}
                            }
                        }
                    ]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 120,
                "candidatesTokenCount": 35,
                "cachedContentTokenCount": 80,
                "totalTokenCount": 155
            }
        }).encode("utf-8")

    ident = ModelIdentity(backend="api", provider="gemini", model_name="gemini-3.5-flash-lite", fingerprint="fp1")
    backend = ApiBackend(ident, api_key="valid-test-key", transport=mock_gemini_transport)

    tools = [{
        "type": "function",
        "function": {
            "name": "workspace_python",
            "description": "Execute python code",
            "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "additionalProperties": False},
        }
    }]
    messages = [
        UnifiedMessage(role="system", content="You are a document assistant."),
        UnifiedMessage(role="user", content="Inspect document.docx"),
    ]

    resp = asyncio.run(backend.generate(messages, tools=tools))

    assert not resp.selector_invalid
    assert len(resp.tool_calls) == 1
    call = resp.tool_calls[0]
    assert call.id == "fc_001"
    assert call.name == "workspace_python"
    assert call.arguments == {"code": "print('hello world')"}
    assert resp.cost.cached_tokens == 80
    assert resp.cost.cost_usd is not None  # USD cost calculated for Gemini
    assert resp.response_id == "gemini_resp_001"

    # Verify outbound wire payload: converted schema stripped additionalProperties
    sent = sent_requests[0]
    tools_wire = sent.get("tools", [])
    assert len(tools_wire) == 1
    decl = tools_wire[0]["functionDeclarations"][0]
    assert decl["name"] == "workspace_python"
    assert "additionalProperties" not in decl["parameters"]


def test_local_backend_native_wire_and_hardware_detection():
    """Verify LocalBackend formats messages for OpenAI-compatible endpoint and records hardware."""
    sent_requests = []

    def mock_local_transport(payload_raw: bytes) -> bytes:
        sent_requests.append(json.loads(payload_raw.decode("utf-8")))
        return json.dumps({
            "id": "local_resp_001",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Running analysis",
                    "tool_calls": [{
                        "id": "tc_local_1",
                        "type": "function",
                        "function": {
                            "name": "workspace_python",
                            "arguments": json.dumps({"code": "import docx; print(len(docx.Document().paragraphs))"})
                        }
                    }]
                }
            }],
            "usage": {
                "prompt_tokens": 200,
                "completion_tokens": 40,
                "total_tokens": 240
            }
        }).encode("utf-8")

    ident = ModelIdentity(
        backend="local",
        provider="mlx",
        model_name="Qwen3-Coder-30B",
        fingerprint="fp_local_mlx",
        endpoint="http://127.0.0.1:18080/v1/chat/completions",
    )
    backend = LocalBackend(ident, transport=mock_local_transport)

    resp = asyncio.run(backend.generate([
        UnifiedMessage(role="system", content="System instruction"),
        UnifiedMessage(role="user", content="Count paragraphs"),
    ]))

    assert not resp.selector_invalid
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "workspace_python"
    assert resp.cost.prompt_tokens == 200
    assert resp.cost.cost_usd is None  # Local models MUST keep cost_usd as None
    assert resp.cost.hardware_info is not None
    assert "accelerator" in resp.cost.hardware_info


# ==============================================================================
# 4. Rule 2: Strict Fail-Closed / NO Silent Fallback
# ==============================================================================

def test_api_backend_fails_closed_on_auth_missing():
    """Missing API key must fail closed and never proceed or fallback."""
    ident = ModelIdentity(backend="api", provider="gemini", model_name="gemini-3.5-flash-lite", fingerprint="fp1")
    backend = ApiBackend(ident, api_key=None)
    backend._api_key = None

    ok, reason = backend.check_availability()
    assert not ok
    assert "API key missing" in reason

    with pytest.raises(BackendError, match="API key missing"):
        asyncio.run(backend.generate([UnifiedMessage(role="user", content="hello")]))


def test_api_backend_fails_closed_on_http_error():
    """HTTP 401/429/500 must raise BackendError, never silently succeed or switch."""
    def error_transport(*args):
        raise BackendError("HTTP 401 from gemini: API_KEY_INVALID")

    ident = ModelIdentity(backend="api", provider="gemini", model_name="gemini-3.5-flash-lite", fingerprint="fp1")
    backend = ApiBackend(ident, api_key="dummy-key", transport=error_transport)

    with pytest.raises(BackendError, match="HTTP 401 from gemini"):
        asyncio.run(backend.generate([UnifiedMessage(role="user", content="hello")]))


def test_local_backend_fails_closed_when_unavailable():
    """Local backend pointing at closed port must return unavailable, not fallback to API."""
    ident = ModelIdentity(
        backend="local",
        provider="mlx",
        model_name="Qwen3-Coder-30B",
        fingerprint="fp1",
        endpoint="http://127.0.0.1:59999/v1/chat/completions",  # Port not listening
    )
    backend = LocalBackend(ident)
    ok, reason = backend.check_availability()
    assert not ok
    assert "not reachable" in reason

    with pytest.raises(BackendError, match="not reachable"):
        asyncio.run(backend.generate([UnifiedMessage(role="user", content="hello")]))


def test_saved_reference_backend_never_executes_live():
    """SavedReferenceBackend must fail closed if generation is attempted."""
    ident = ModelIdentity(backend="saved_reference", provider="saved_reference", model_name="ref", fingerprint="fp1")
    backend = SavedReferenceBackend(ident)
    ok, reason = backend.check_availability()
    assert ok

    with pytest.raises(BackendError, match="does not execute live generations"):
        asyncio.run(backend.generate([UnifiedMessage(role="user", content="test")]))


# ==============================================================================
# 5. Rule 4: ModelGateway Multi-Turn Execution & Cost Accounting
# ==============================================================================

class MockDockerActor:
    def __init__(self):
        self.files = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def seed_workspace(self, path):
        pass

    def exec_python(self, code: str):
        return {
            "exit_code": 0,
            "stdout": "script executed successfully\n",
            "stderr": "",
            "truncated": False,
            "stdout_bytes": 30,
            "stderr_bytes": 0,
        }

    def export_workspace(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "output.docx").write_text("modified")


class MockDispatchRecorder:
    def __init__(self, actor, path):
        self.records = []

    async def __call__(self, event):
        self.records.append(event)


def test_gateway_run_episode_with_api_backend(tmp_path):
    """Verify ModelGateway runs full multi-turn Docker loop with API backend and records costs."""
    turns_count = 0

    def mock_transport(raw: bytes) -> bytes:
        nonlocal turns_count
        turns_count += 1
        if turns_count == 1:
            # Turn 1: tool call
            return json.dumps({
                "responseId": "resp_turn_1",
                "candidates": [{
                    "content": {
                        "role": "model",
                        "parts": [{
                            "thoughtSignature": "sig1",
                            "functionCall": {
                                "id": "call_1",
                                "name": "workspace_python",
                                "args": {"code": "print('editing doc')"}
                            }
                        }]
                    }
                }],
                "usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 20, "cachedContentTokenCount": 30}
            }).encode()
        else:
            # Turn 2: finish
            return json.dumps({
                "responseId": "resp_turn_2",
                "candidates": [{
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Task finished successfully"}]
                    }
                }],
                "usageMetadata": {"promptTokenCount": 90, "candidatesTokenCount": 10, "cachedContentTokenCount": 50}
            }).encode()

    ident = ModelIdentity(backend="api", provider="gemini", model_name="gemini-3.5-flash-lite", fingerprint="fp_gemini")
    backend = ApiBackend(ident, api_key="test-key", transport=mock_transport)
    gateway = ModelGateway(executor=backend)

    # Prepare bundle & grading stubs
    bundle_root = tmp_path / "bundle"
    inputs_dir = bundle_root / "inputs"
    inputs_dir.mkdir(parents=True)
    (inputs_dir / "input.docx").write_text("dummy")

    bundle = {
        "bundle_root": str(bundle_root),
        "task_ids": ["officeval_002"],
        "tasks": {
            "officeval_002": {
                "task_id": "officeval_002",
                "task_prompt": "Fix margins in input.docx",
                "expected_output": "output.docx",
                "input_files": [{"name": "input.docx", "path": "inputs/input.docx"}],
            }
        }
    }

    def mock_grader(task_id, manifest, ws_dir, grader_image, out_dir):
        return {
            "validity": "valid",
            "c3_operational_completed": True,
            "official_raw": {"dim1_pass": True, "dim2_items": [{"hit": True, "max_delta": 1}]},
        }

    result = asyncio.run(gateway.run_episode(
        bundle=bundle,
        task_id="officeval_002",
        agent_image="mock-agent-image",
        grader_image="mock-grader-image",
        output_dir=tmp_path / "ep1",
        seed=2026,
        max_turns=5,
        budget=100,
        runtime_factory=lambda *a, **kw: MockDockerActor(),
        grade_fn=mock_grader,
        dispatch_factory=lambda *a, **kw: MockDispatchRecorder(*a, **kw),
    ))

    assert result["status"] == "complete"
    assert result["execution_valid"] is True
    assert result["task_completed"] is True
    assert result["model_identity"]["backend"] == "api"

    cost_summary = result["cost_summary"]
    assert cost_summary["backend"] == "api"
    assert cost_summary["prompt_tokens"] == 140
    assert cost_summary["completion_tokens"] == 30
    assert cost_summary["cached_tokens"] == 80
    assert cost_summary["cost_usd"] is not None
    assert cost_summary["total_calls"] == 2


def test_gateway_run_episode_fail_closed_on_model_error(tmp_path):
    """When backend fails during execution, gateway must mark failed status, NOT switch backend."""
    def failing_transport(*args):
        raise BackendError("HTTP 503 from provider: service unavailable")

    ident = ModelIdentity(backend="api", provider="gemini", model_name="gemini-3.5-flash-lite", fingerprint="fp1")
    backend = ApiBackend(ident, api_key="test-key", transport=failing_transport)
    gateway = ModelGateway(executor=backend)

    bundle_root = tmp_path / "bundle"
    inputs_dir = bundle_root / "inputs"
    inputs_dir.mkdir(parents=True)
    (inputs_dir / "input.docx").write_text("dummy")

    bundle = {
        "bundle_root": str(bundle_root),
        "task_ids": ["officeval_002"],
        "tasks": {
            "officeval_002": {
                "task_id": "officeval_002",
                "task_prompt": "Fix margins",
                "expected_output": "output.docx",
                "input_files": [{"name": "input.docx", "path": "inputs/input.docx"}],
            }
        }
    }

    def mock_grader(task_id, manifest, ws_dir, grader_image, out_dir):
        return {"validity": "valid", "c3_operational_completed": False}

    result = asyncio.run(gateway.run_episode(
        bundle=bundle,
        task_id="officeval_002",
        agent_image="mock-agent-image",
        grader_image="mock-grader-image",
        output_dir=tmp_path / "ep_err",
        seed=2026,
        max_turns=5,
        budget=100,
        runtime_factory=lambda *a, **kw: MockDockerActor(),
        grade_fn=mock_grader,
        dispatch_factory=lambda *a, **kw: MockDispatchRecorder(*a, **kw),
    ))

    # Must be marked failed, with stopped_reason = backend_error
    assert result["status"] == "failed"
    assert result["agent_result"]["stopped_reason"] == "backend_error"
    assert "HTTP 503" in result["agent_result"]["error"]


# ==============================================================================
# 6. Gateway D7 Diagnostic Bridge Tests
# ==============================================================================

def test_gateway_diagnostic_backend_generates_valid_prediction(tmp_path):
    """Verify GatewayDiagnosticBackend bridges ModelBackend to D7 Diagnostic contracts."""
    def mock_diag_transport(raw_req: bytes) -> bytes:
        return json.dumps({
            "responseId": "diag_resp_001",
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{
                        "text": json.dumps({
                            "probabilities": {
                                "original": 0.1,
                                "visibility.expanded": 0.7,
                                "feedback.structured": 0.2
                            }
                        })
                    }]
                }
            }],
            "usageMetadata": {"promptTokenCount": 300, "candidatesTokenCount": 45}
        }).encode("utf-8")

    ident = ModelIdentity(backend="api", provider="gemini", model_name="gemini-3.5-flash-lite", fingerprint="fp_diag")
    backend = ApiBackend(ident, api_key="mock-key", transport=mock_diag_transport)
    diag_bridge = GatewayDiagnosticBackend(backend, persist_dir=tmp_path)

    evidence = {
        "own_text": "Agent attempted to read non-existent Table Grid style",
        "reference_episodes": [{"case_id": "ref:officeval_003", "text": "Reference model used default table"}],
    }
    candidates = ["original", "visibility.expanded", "feedback.structured"]

    pred = diag_bridge.predict(
        evidence,
        case_id="gemini:officeval_003",
        candidate_universe=candidates,
        method="ALL_TEXT",
        evidence_ids=["gemini:officeval_003"],
        checkpoint_identity="chk_turn_2",
    )

    assert pred["status"] == "ok"
    assert pred["probabilities"]["visibility.expanded"] == 0.7
    assert pred["probabilities"]["original"] == 0.1
    assert pred["model"] == "gemini-3.5-flash-lite"
    assert pred["checkpoint_identity"] == "chk_turn_2"
    assert (tmp_path / "gemini:officeval_003-ALL_TEXT.json").is_file()
    assert (tmp_path / "gemini:officeval_003-ALL_TEXT.request.bin").is_file()
    assert (tmp_path / "gemini:officeval_003-ALL_TEXT.response.bin").is_file()


def test_gateway_diagnostic_backend_malformed_response_selector_invalid():
    """Malformed diagnostic response must yield selector_invalid, never crash."""
    def bad_transport(raw_req: bytes) -> bytes:
        return json.dumps({
            "choices": [{"message": {"content": "I don't know the probabilities"}}]
        }).encode()

    ident = ModelIdentity(backend="api", provider="openai", model_name="gpt-4o", fingerprint="fp_gpt")
    backend = ApiBackend(ident, api_key="mock-key", transport=bad_transport)
    diag_bridge = GatewayDiagnosticBackend(backend)

    pred = diag_bridge.predict(
        {"text": "evidence"},
        case_id="case_1",
        candidate_universe=["original", "visibility.expanded"],
        method="OWN_TEXT",
    )
    assert pred["status"] == "selector_invalid"
    assert pred["probabilities"] == {}
    assert pred["error"] is not None


# ==============================================================================
# 7. End-to-End Config Switching & Integrated Delivery
# ==============================================================================

import importlib.util
_SCRIPT = Path(__file__).parents[1] / "scripts" / "run_integrated_delivery.py"
_spec = importlib.util.spec_from_file_location("run_integrated_delivery", _SCRIPT)
delivery_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(delivery_mod)


def test_dual_backend_config_preflight_and_delivery(tmp_path):
    """Verify configs/integrated_dual_backend.json passes preflight and executes."""

    config_path = Path(__file__).parents[1] / "configs/integrated_dual_backend.json"
    cfg = json.loads(config_path.read_text())
    # Provide test key
    cfg["models"]["executor"]["api_key"] = "AQ.TestKeyMock"
    cfg["models"]["diagnostician"]["api_key"] = "AQ.TestKeyMock"

    # Preflight with mock root
    preflight_calls = []
    def mock_preflight(val):
        preflight_calls.append(val)
        return {
            "bundle": {"task_ids": val["task_ids"], "tasks": {t: {} for t in val["task_ids"]}},
            "bundle_root": str(tmp_path / "bundle"),
            "task_ids": val["task_ids"],
            "service_config": val["services"],
            "preflight_config": val,
        }

    # Mock transport so closeout diagnostician doesn't call real external network
    def mock_diag_transport(*args):
        return json.dumps({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{
                        "text": json.dumps({"probabilities": {"original": 0.5, "visibility.expanded": 0.3, "feedback.structured": 0.2}})
                    }]
                }
            }],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5}
        }).encode()

    mock_gateway = ModelGateway.from_config(cfg, transport=mock_diag_transport)

    runner_calls = []
    def mock_runner(context, task_id, model_key):
        runner_calls.append((task_id, model_key))
        return {
            "status": "complete",
            "execution_valid": True,
            "task_completed": True,
            "result": {"execution_valid": True, "task_id": task_id, "model_key": model_key},
        }

    manifest = delivery_mod.execute_delivery(cfg, tmp_path / "out", runner=mock_runner, preflight=mock_preflight, gateway=mock_gateway)

    assert manifest["status"] == "complete"
    assert len(manifest["runs"]) == 16  # 8 executor + 8 reference
    assert (tmp_path / "out" / "config.lock").is_file()
    assert (tmp_path / "out" / "run_manifest.json").is_file()
    # Check that executor runs were recorded with correct model name
    executor_runs = [r for r in manifest["runs"] if r["model_key"] in {"executor", "target"}]
    assert len(executor_runs) == 8
    assert all(r["model"] == "gemini-3.5-flash-lite" for r in executor_runs)


def test_delivery_anti_silent_fallback_on_runner_error(tmp_path):
    """If an API model fails with an unrecoverable error, the run row MUST be marked failed, never silently switched."""
    config_path = Path(__file__).parents[1] / "configs/integrated_dual_backend.json"
    cfg = json.loads(config_path.read_text())
    cfg["models"]["executor"]["api_key"] = "AQ.TestKeyMock"

    def mock_preflight(val):
        return {
            "bundle": {"task_ids": val["task_ids"], "tasks": {t: {} for t in val["task_ids"]}},
            "bundle_root": str(tmp_path / "bundle"),
            "task_ids": val["task_ids"],
            "service_config": val["services"],
            "preflight_config": val,
        }

    def failing_runner(context, task_id, model_key):
        if model_key in {"executor", "target"}:
            raise BackendError("Fatal API Authentication Failure: HTTP 401")
        return {"status": "NOT_RUN", "reason": "ref"}

    def mock_diag_transport(*args):
        return b"{}"
    mock_gateway = ModelGateway.from_config(cfg, transport=mock_diag_transport)

    manifest = delivery_mod.execute_delivery(cfg, tmp_path / "out_err", runner=failing_runner, preflight=mock_preflight, gateway=mock_gateway)

    # Must NOT be complete, executor tasks must have failed status
    assert manifest["status"] == "completed_with_failures"
    failed_rows = [r for r in manifest["runs"] if r["model_key"] in {"executor", "target"}]
    assert all(r["status"] == "failed" for r in failed_rows)
    assert all("Fatal API Authentication Failure" in r["error"] for r in failed_rows)


def test_config_switching_transparently_changes_backend(tmp_path):
    """Switching executor config backend between 'api' and 'local' changes execution without modifying sandbox."""
    base_config = {
        "models": {
            "executor": {
                "backend": "api",
                "provider": "gemini",
                "model": "gemini-3.5-flash-lite",
                "api_key": "mock-key",
            }
        }
    }
    gw_api = ModelGateway.from_config(base_config)
    assert isinstance(gw_api.executor, ApiBackend)
    assert gw_api.executor.backend_type == "api"

    # Switch only config: backend="local"
    local_config = {
        "models": {
            "executor": {
                "backend": "local",
                "provider": "mlx",
                "model": "Qwen3-Coder-30B",
                "endpoint": "http://127.0.0.1:18080/v1/chat/completions",
            }
        }
    }
    gw_local = ModelGateway.from_config(local_config)
    assert isinstance(gw_local.executor, LocalBackend)
    assert gw_local.executor.backend_type == "local"
    assert gw_local.executor.model_name == "Qwen3-Coder-30B"

