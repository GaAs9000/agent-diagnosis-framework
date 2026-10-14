"""ModelGateway: Unified gateway orchestrating model execution across C3 Sandbox.

All Sandbox tasks, diagnostics, and experiment card replays interact solely
with ModelGateway. It unifies conversational tool loops, event timelines,
provenance records, and cost accounting while allowing seamless config-based
backend switching.
"""
from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Mapping, Sequence

from .backend import (
    BackendError,
    CostRecord,
    ModelBackend,
    ModelIdentity,
    ModelResponse,
    ModelToolCall,
    UnifiedMessage,
)
from .registry import BackendRegistry
from ..r1.docker_runtime import DockerRuntime
from ..r1.officeval import task_public_view

SYSTEM_PROMPT = (
    "Complete the user's document task using the native workspace_python tool. "
    "Use the installed document libraries to inspect and edit the supplied files. "
    "Save the deliverable with its designated original filename. Preserve existing "
    "content and formatting unless the task requires a change. Tool observations "
    "are actual execution results. Explain briefly when finished."
)

TOOL_SCHEMAS = [{"type": "function", "function": {
    "name": "workspace_python",
    "description": "Execute Python in the isolated task workspace and return actual stdout, stderr and exit code.",
    "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                    "required": ["code"], "additionalProperties": False},
}}]


def delivered_tool_result(_name: str, _arguments: Mapping[str, Any], raw: Mapping[str, Any]) -> str:
    visible = {key: raw.get(key) for key in ("exit_code", "error", "truncated", "stdout_bytes", "stderr_bytes")}
    visible["stdout"] = str(raw.get("stdout", ""))[:6000]
    visible["stderr"] = str(raw.get("stderr", ""))[:2000]
    visible["context_output_truncated"] = len(str(raw.get("stdout", ""))) > 6000 or len(str(raw.get("stderr", ""))) > 2000
    return json.dumps(visible, ensure_ascii=False)


def _public_staging(bundle: Mapping[str, Any], task_id: str, destination: Path) -> dict:
    task = bundle["tasks"][task_id]
    task = task.get("metadata", task) if isinstance(task, Mapping) else task
    public = task_public_view({"tasks": {task_id: task}}, task_id)
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    root = Path(bundle["bundle_root"]).resolve()
    for entry in public["input_files"]:
        name, relative = entry["name"], Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe frozen public input path")
        source = (root / relative).resolve()
        if not source.is_relative_to(root) or source.is_symlink() or not source.is_file():
            raise ValueError("frozen public input unavailable")
        shutil.copyfile(source, destination / name)
        (destination / name).chmod(0o644)
    return public


def _episode_outcome(initial: Mapping[str, Any], final: Mapping[str, Any], stopped_reason: str) -> dict:
    valid = initial.get("validity") == final.get("validity") == "valid" and stopped_reason in {"assistant_finished", "max_turns"}
    completed = final.get("c3_operational_completed") if valid else None
    raw = final.get("official_raw")
    targets = bool(raw and raw.get("dim1_pass") and all(i.get("hit") for i in raw.get("dim2_items", []) if i.get("max_delta", 0) > 0)) if valid else None
    return {
        "execution_valid": valid,
        "task_completed": completed,
        "target_resolved": targets,
        "regression_free": None,
        "outcome": "success" if valid and completed is True else ("failure" if valid else "infrastructure_invalid"),
    }


class ModelGateway:
    """Unified model gateway coordinating executor, reference, and diagnostic backends."""

    def __init__(
        self,
        executor: ModelBackend,
        reference: ModelBackend | None = None,
        diagnostician: ModelBackend | None = None,
    ):
        self.executor = executor
        self.reference = reference or executor
        self.diagnostician = diagnostician or executor

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        root: str | Path | None = None,
        transport: Callable[..., Any] | None = None,
    ) -> ModelGateway:
        """Construct ModelGateway by reading the 'models' block in the config."""
        models = config.get("models", {})

        # 1. Resolve executor (alias: target)
        exec_cfg = models.get("executor") or models.get("target")
        if not exec_cfg:
            raise ValueError("Config missing required executor/target model section")
        executor = BackendRegistry.create(exec_cfg, root=root, transport=transport)

        # 2. Resolve reference
        ref_cfg = models.get("reference")
        reference = BackendRegistry.create(ref_cfg, root=root, transport=transport) if ref_cfg else None

        # 3. Resolve diagnostician
        diag_cfg = models.get("diagnostician") or config.get("diagnostic")
        diagnostician = BackendRegistry.create(diag_cfg, root=root, transport=transport) if diag_cfg else None

        return cls(executor=executor, reference=reference, diagnostician=diagnostician)

    def create_diagnostic_backend(self, persist_dir: str | Path | None = None) -> Any:
        """Create a D7-compatible diagnostic backend instance wrapping self.diagnostician."""
        from .diagnostician import GatewayDiagnosticBackend
        return GatewayDiagnosticBackend(self.diagnostician, persist_dir=persist_dir)


    async def run_episode(
        self,
        bundle: Mapping[str, Any],
        task_id: str,
        *,
        agent_image: str,
        grader_image: str,
        output_dir: str | Path,
        seed: int = 2026,
        max_turns: int = 10,
        budget: int = 100,
        runtime_factory: Callable[..., Any] = DockerRuntime,
        grade_fn: Callable[..., Mapping[str, Any]] | None = None,
        dispatch_factory: Callable[..., Any] | None = None,
        backend_override: ModelBackend | None = None,
    ) -> dict[str, Any]:
        """Run an end-to-end task episode in the isolated Docker workspace.
        
        Uses backend_override if provided, else self.executor.
        Enforces Rule 1 (Unified events), Rule 2 (No silent fallback), Rule 3 (Frozen identity),
        and Rule 4 (Separate cost accounting).
        """
        backend = backend_override or self.executor
        if task_id not in bundle.get("task_ids", []):
            raise ValueError("task is not in frozen bundle")
        if seed != 2026 or not isinstance(max_turns, int) or max_turns < 1 or max_turns > budget or budget != 100:
            raise ValueError("integrated delivery fixes seed=2026 and budget=100")

        # Check backend availability before creating containers
        ok, reason = backend.check_availability()
        if not ok:
            raise BackendError(f"Backend '{backend.identity.provider}' unavailable: {reason}")

        if grade_fn is None:
            from ..r1.grader import grade_workspace as grade_fn
        if dispatch_factory is None:
            from ..r1.r11_replay import DispatchRecorder as dispatch_factory

        output = Path(output_dir).absolute()
        output.mkdir(parents=True, exist_ok=False)
        staging = output / "public-input"
        public = _public_staging(bundle, task_id, staging)
        task_record = bundle["tasks"][task_id]
        task_metadata = task_record.get("metadata", task_record)
        manifest = {"bundle_root": bundle["bundle_root"], "tasks": {task_id: task_metadata}}

        initial = dict(grade_fn(task_id, manifest, staging, grader_image, output / "initial-grade"))
        events: list[dict[str, Any]] = []
        dispatch_records: list[dict[str, Any]] = []
        checkpoints: list[dict[str, Any]] = []
        costs: list[CostRecord] = []
        raw_model_records: list[dict[str, Any]] = []

        runtime_meta = {
            "backend": backend.identity.backend,
            "provider": backend.identity.provider,
            "model": backend.identity.model_name,
            "model_fingerprint": backend.identity.fingerprint,
            "tokenizer_fingerprint": backend.identity.tokenizer_fingerprint or "unified-wire-v1",
            "sampling": {"seed": seed, "max_turns": max_turns},
            "tool_config": {"schemas": copy.deepcopy(TOOL_SCHEMAS), "delivered_result": "r1.qualification.delivered_tool_result-v1"},
        }

        user_prompt = public["task_prompt"] + "\nDesignated deliverable: " + public["expected_output"]
        messages: list[UnifiedMessage] = [
            UnifiedMessage(role="system", content=SYSTEM_PROMPT),
            UnifiedMessage(role="user", content=user_prompt),
        ]

        stopped_reason = "max_turns"
        last_error = None
        selector_invalid = False

        with runtime_factory(agent_image, output / "actor-runtime") as actor:
            actor.seed_workspace(staging)
            recorder = dispatch_factory(actor, output / "tool-dispatch")

            for turn_idx in range(max_turns):
                try:
                    response: ModelResponse = await backend.generate(
                        messages,
                        tools=TOOL_SCHEMAS,
                        sampling={"seed": seed, "max_tokens": 4096},
                    )
                except Exception as exc:
                    selector_invalid = True
                    stopped_reason = "backend_error"
                    last_error = str(exc)
                    events.append({"type": "agent_error", "error": str(exc), "turn": turn_idx})
                    break

                costs.append(response.cost)
                raw_model_records.append({
                    "turn": turn_idx,
                    "response_id": response.response_id,
                    "selector_invalid": response.selector_invalid,
                    "error": response.error,
                    "cost": response.cost.to_dict(),
                })

                if response.selector_invalid:
                    selector_invalid = True
                    stopped_reason = "protocol_error"
                    last_error = response.error
                    events.append({"type": "protocol_error", "error": response.error, "turn": turn_idx})
                    break

                if not response.tool_calls:
                    # Model provided text answer and stopped
                    stopped_reason = "assistant_finished"
                    messages.append(UnifiedMessage(role="assistant", content=response.text))
                    events.append({"type": "assistant_message", "content": response.text, "turn": turn_idx})
                    break

                # Preserve assistant turn with its tool calls and provider-specific native parts (e.g. thoughtSignature)
                messages.append(UnifiedMessage(
                    role="assistant",
                    content=response.text,
                    tool_calls=response.tool_calls,
                    native_parts=response.raw_response,
                ))

                # Execute tool calls
                for call in response.tool_calls:
                    if call.name != "workspace_python" or not isinstance(call.arguments.get("code"), str):
                        selector_invalid = True
                        stopped_reason = "protocol_error"
                        last_error = f"Unsupported tool call: {call.name}"
                        events.append({"type": "protocol_error", "error": last_error, "turn": turn_idx})
                        break

                    start_event = {
                        "type": "tool_dispatch",
                        "stage": "start",
                        "turn": len(events),
                        "tool_call_id": call.id,
                        "name": call.name,
                        "arguments": call.arguments,
                    }
                    events.append(start_event)
                    await recorder(start_event)

                    raw_res = actor.exec_python(call.arguments["code"])

                    end_event = {**start_event, "stage": "end", "raw_result": raw_res}
                    await recorder(end_event)
                    dispatch_records.append(dict(recorder.records[-1]))

                    events.append({
                        "type": "tool_result",
                        "turn": len(events),
                        "tool_call_id": call.id,
                        "raw_result": raw_res,
                    })

                    delivered = delivered_tool_result(call.name, call.arguments, raw_res)
                    messages.append(UnifiedMessage(
                        role="tool",
                        name=call.name,
                        tool_call_id=call.id,
                        content=delivered,
                    ))

                if selector_invalid:
                    break

            actor.export_workspace(output / "final-workspace")

        final = dict(grade_fn(task_id, manifest, output / "final-workspace", grader_image, output / "final-grade"))
        protocol_status = "invalid" if selector_invalid else "valid"

        agent_record = {
            "events": events,
            "tool_dispatch": dispatch_records,
            "protocol_status": protocol_status,
            "stopped_reason": stopped_reason,
            "error": last_error,
        }

        # Calculate cost totals
        total_prompt = sum(c.prompt_tokens or 0 for c in costs)
        total_comp = sum(c.completion_tokens or 0 for c in costs)
        total_cached = sum(c.cached_tokens or 0 for c in costs)
        total_cost_usd = sum(c.cost_usd for c in costs if c.cost_usd is not None) if any(c.cost_usd is not None for c in costs) else None

        cost_summary = {
            "backend": backend.identity.backend,
            "prompt_tokens": total_prompt,
            "completion_tokens": total_comp,
            "cached_tokens": total_cached,
            "total_tokens": total_prompt + total_comp,
            "cost_usd": total_cost_usd,
            "total_calls": len(costs),
        }

        result = {
            "task_id": task_id,
            "provider": backend.identity.provider,
            "model": backend.identity.model_name,
            "model_identity": backend.identity.to_dict(),
            "seed": seed,
            "is_synthetic": False,
            "output_dir": str(output),
            "runtime": runtime_meta,
            "initial_grade": initial,
            "final_grade": final,
            "agent_result": agent_record,
            "tool_dispatch": dispatch_records,
            "checkpoints": checkpoints,
            "cost_summary": cost_summary,
            "raw_model_records": raw_model_records,
            "status": "complete" if protocol_status == "valid" else "failed",
            **_episode_outcome(initial, final, stopped_reason),
        }

        path_json = output / "result.json"
        path_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n")
        return result

    def run_episode_sync(self, *args, **kwargs) -> dict[str, Any]:
        return asyncio.run(self.run_episode(*args, **kwargs))
