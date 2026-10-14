"""Fixed two-task, one-model development run. No seed/model selection loop."""
import asyncio
import argparse
import hashlib
import json
from pathlib import Path

from c3_sandbox.evidence._storage import encode
from c3_sandbox.r1.office_run import run_baseline
from c3_sandbox.r1.qualification import image_identity

ROOT = Path(__file__).resolve().parents[1]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, default=ROOT / 'runs/e2-r1')
    parser.add_argument('--qualification-dir', type=Path, default=ROOT / 'runs/r1-native-qualification-007')
    args = parser.parse_args()
    output = args.run_root.absolute()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "runs/r1-build-context/frozen/manifest.json").read_bytes())
    qualification = json.loads((args.qualification_dir / 'qualification.json').read_bytes())
    agent = image_identity("c3-r1-agent:locked")
    grader = image_identity("c3-r1-grader:locked")
    plan = {"stage": "C3-E2-R1", "split": "development", "tasks": ["officeval_003", "officeval_002"],
            "model": qualification["model"], "model_fingerprint": qualification["model_fingerprint"],
            "master_seed": 2026, "temperature": 0, "max_tokens_per_call": 2048,
            "context_limit": 16384, "original_max_model_calls": 10, "concurrency": 1,
            "checkpoint_selection": "first idle complete-tool checkpoint with successful cold restoration",
            "exploration_seed": 2026,
            "confirmation_seed": int.from_bytes(hashlib.sha256(b"C3-E2-R1:2026:confirmation").digest()[:4], "big"),
            "agent_image": agent, "grader_image": grader,
            "completion_endpoint": "C3 operational mapping: all official rules assessed and met; original official scores retained",
            "no_outcome_based_seed_or_model_retry": True}
    plan['qualification_directory'] = str(args.qualification_dir.absolute())
    with (output / "frozen_plan.json").open("xb") as f:
        f.write(encode(plan))
    for task_id in plan["tasks"]:
        print("START", task_id, flush=True)
        result = await run_baseline(task_id, manifest, agent, grader, qualification,
                                    output / "originals" / task_id, seed=2026, max_turns=10)
        print("END", task_id, json.dumps({k: result.get(k) for k in ["execution_valid", "task_completed", "target_resolved", "regression_free", "outcome"]}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
