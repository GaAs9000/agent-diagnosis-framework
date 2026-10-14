#!/usr/bin/env python3
"""Run the fixed eight-task target-model development matrix serially.

This is an explicitly invoked bounded run. It records infrastructure errors as
rows and never replaces a task or retries after seeing its outcome.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from c3_sandbox.integrated.office_runner import run_episode
from c3_sandbox.r1.r12_service import observe_service


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/integrated-delivery-001"))
    parser.add_argument("--model-pid", type=int, required=True)
    parser.add_argument("--max-turns", type=int, default=20)
    args = parser.parse_args()
    root = args.root.absolute()
    bundle = json.loads((root / "bundle/manifest.json").read_text())
    system_manifest = json.loads((Path("runs/e2-r1-2/system_manifest.json")).read_text())
    system = system_manifest["system"]
    fingerprint = system_manifest["model_fingerprint"]
    profile = {
        "status": "blocked_system_profile_reused_for_integrated_delivery",
        "provider": "openai-api/mlx/default_model",
        "model": system["model"]["repository"],
        "model_fingerprint": fingerprint["sha256"],
        "tokenizer_fingerprint": fingerprint["tokenizer"],
        "native_tool_calls": 0, "model_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
        "profile_source": "R1.2 frozen model identity; no new qualification gate",
    }
    grader_image = (root / "grader-image-id.txt").read_text().strip()
    results = []
    fragment_total = 0
    for index, task_id in enumerate(bundle["tasks"]):
        if fragment_total >= 100:
            result = {"task_id": task_id, "status": "not_run_budget_exhausted",
                      "reason": "integrated execution fragment budget exhausted",
                      "new_model_calls": 0}
            results.append(result)
            print(json.dumps({"event": "episode_not_run", "order": index, "task_id": task_id,
                              "reason": result["reason"]}, ensure_ascii=False), flush=True)
            continue
        output = root / "originals" / "target" / task_id
        print(json.dumps({"event": "episode_start", "order": index, "task_id": task_id}, ensure_ascii=False), flush=True)
        try:
            result = asyncio.run(run_episode(
                bundle, task_id, agent_image=system["agent_image"], grader_image=grader_image,
                model_profile=profile, output_dir=output,
                model_fingerprint=fingerprint["sha256"], tokenizer_fingerprint=fingerprint["tokenizer"],
                server_pid=args.model_pid, guard="r12_service_observation",
                seed=2026, max_turns=args.max_turns, budget=100,
                system_contract=system,
                generation_guard=lambda: observe_service(system, args.model_pid)))
        except Exception as exc:
            result = {"task_id": task_id, "status": "infrastructure_invalid",
                      "reason": type(exc).__name__ + ": " + str(exc),
                      "model": profile["model"], "model_fingerprint": profile["model_fingerprint"],
                      "new_model_calls": 0}
            output.mkdir(parents=True, exist_ok=True)
            (output / "execution_error.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        results.append(result)
        fragment_total += len(result.get("tool_dispatch", []))
        print(json.dumps({"event": "episode_finished", "order": index, "task_id": task_id,
                          "outcome": result.get("outcome"), "stopped_reason": result.get("agent_result", {}).get("stopped_reason"),
                          "execution_valid": result.get("execution_valid"), "model_calls": result.get("agent_result", {}).get("modelcall_count", 0)}, ensure_ascii=False), flush=True)
    (root / "originals/target_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"event": "matrix_finished", "episodes": len(results),
                      "model_calls": sum(r.get("agent_result", {}).get("modelcall_count", 0) for r in results),
                      "execution_fragments": fragment_total, "fragment_budget": 100}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
