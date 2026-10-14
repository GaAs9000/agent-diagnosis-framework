"""Native Gemini original-task episode runner.

This adapter deliberately does not reuse the MLX/OpenAI agent.  Gemini's
function-call wire records are kept verbatim while tool observations are
produced by the same isolated Docker runtime and independently graded by the
OfficeVal grader.
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping

from .gemini_api import GeminiNativeToolAgent, MODEL
from .office_runner import build_task_bundle
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

GEMINI_PROVIDER = "google-gemini"


def _api_key(value: str | None) -> str | None:
    return value or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


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


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def _episode_outcome(initial: Mapping[str, Any], final: Mapping[str, Any], stopped_reason: str) -> dict:
    valid = initial.get("validity") == final.get("validity") == "valid" and stopped_reason in {"assistant_finished", "max_turns"}
    completed = final.get("c3_operational_completed") if valid else None
    raw = final.get("official_raw")
    targets = bool(raw and raw.get("dim1_pass") and all(i.get("hit") for i in raw.get("dim2_items", []) if i.get("max_delta", 0) > 0)) if valid else None
    return {"execution_valid": valid, "task_completed": completed, "target_resolved": targets,
            "regression_free": None, "outcome": "success" if valid and completed is True else ("failure" if valid else "infrastructure_invalid")}


async def run_gemini_episode(
    bundle: Mapping[str, Any], task_id: str, *, agent_image: str, grader_image: str,
    output_dir: str | Path, model_fingerprint: str = "gemini-3.5-flash-lite",
    tokenizer_fingerprint: str = "gemini-native-wire-v1", seed: int = 2026,
    max_turns: int = 10, budget: int = 100, api_key: str | None = None,
    transport: Callable[..., Any] | None = None, runtime_factory: Callable[..., Any] = DockerRuntime,
    grade_fn: Callable[..., Mapping[str, Any]] | None = None,
    dispatch_factory: Callable[..., Any] | None = None,
) -> dict:
    """Run one Gemini native-function-call episode with fail-closed setup.

    ``transport`` and the factories are intentionally injectable for protocol
    tests; production defaults never simulate actions or grading.
    """
    if task_id not in bundle.get("task_ids", []):
        raise ValueError("task is not in frozen bundle")
    if seed != 2026 or not isinstance(max_turns, int) or max_turns < 1 or max_turns > budget or budget != 100:
        raise ValueError("integrated delivery fixes seed=2026 and budget=100")
    key = _api_key(api_key)
    if not key:
        # Check before creating output actors or grading containers.
        raise RuntimeError("Gemini API key is missing (set GEMINI_API_KEY or GOOGLE_API_KEY)")
    if not model_fingerprint or not tokenizer_fingerprint:
        raise ValueError("complete Gemini model and tokenizer fingerprints required")
    if grade_fn is None:
        # Keep the remote Gemini path independent of the local Inspect runtime;
        # only the pinned Docker grader is needed here.
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
    runtime = {"provider": GEMINI_PROVIDER, "model": MODEL, "model_fingerprint": model_fingerprint,
               "tokenizer_fingerprint": tokenizer_fingerprint, "sampling": {"seed": seed, "max_turns": max_turns},
               "tool_config": {"schemas": copy.deepcopy(TOOL_SCHEMAS), "delivered_result": "r1.qualification.delivered_tool_result-v1"}}

    with runtime_factory(agent_image, output / "actor-runtime") as actor:
        actor.seed_workspace(staging)
        recorder = dispatch_factory(actor, output / "tool-dispatch")

        async def callback(name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
            if name != "workspace_python" or not isinstance(arguments, Mapping) or not isinstance(arguments.get("code"), str):
                raise ValueError("unsupported Gemini function call")
            call_id = "gemini-call-" + str(len(dispatch_records))
            start = {"type": "tool_dispatch", "stage": "start", "turn": len(events), "tool_call_id": call_id,
                     "name": name, "arguments": dict(arguments)}
            events.append(start)
            await recorder(start)
            raw = actor.exec_python(arguments["code"])
            end = {**start, "stage": "end", "raw_result": raw}
            await recorder(end)
            dispatch_records.append(dict(recorder.records[-1]))
            events.append({"type": "tool_result", "turn": len(events), "tool_call_id": call_id, "raw_result": raw})
            return json.loads(delivered_tool_result(name, arguments, raw))

        agent = GeminiNativeToolAgent(transport=transport, api_key=key, model=MODEL)
        user = public["task_prompt"] + "\nDesignated deliverable: " + public["expected_output"]
        try:
            agent_result = await agent.run(system=SYSTEM_PROMPT, user=user, tool_schema=TOOL_SCHEMAS,
                                           tool_callback=callback, max_turns=max_turns,
                                           evidence_ids=(f"gemini:{task_id}",))
        except Exception as exc:
            agent_result = {"text": "", "tool_calls": [], "selector_invalid": True, "error": str(exc), "stopped_reason": "error"}
            events.append({"type": "agent_error", "error": str(exc)})
        actor.export_workspace(output / "final-workspace")
    final = dict(grade_fn(task_id, manifest, output / "final-workspace", grader_image, output / "final-grade"))
    protocol_status = "invalid" if agent_result.get("selector_invalid") else "valid"
    stopped_reason = agent_result.get("stopped_reason", "assistant_finished" if not agent_result.get("selector_invalid") else "protocol_error")
    agent_record = {"events": events, "tool_dispatch": dispatch_records, "protocol_status": protocol_status,
                    "text": agent_result.get("text", ""), "error": agent_result.get("error"),
                    "stopped_reason": stopped_reason,
                    "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in agent_result.get("tool_calls", [])]}
    result = {"task_id": task_id, "provider": GEMINI_PROVIDER, "model": MODEL, "seed": seed,
              "is_synthetic": False, "output_dir": str(output), "runtime": runtime,
              "initial_grade": initial, "final_grade": final, "agent_result": agent_record,
              "tool_dispatch": dispatch_records, "checkpoints": checkpoints,
              "raw_gemini_records": [record.to_dict() for record in agent.records],
              "gemini_usage": [record.usage for record in agent.records if record.usage is not None],
              "status": "complete" if protocol_status == "valid" else "failed",
              **_episode_outcome(initial, final, stopped_reason)}
    _json(output / "result.json", result)
    return result


def run_gemini_episode_sync(*args, **kwargs):
    return asyncio.run(run_gemini_episode(*args, **kwargs))
