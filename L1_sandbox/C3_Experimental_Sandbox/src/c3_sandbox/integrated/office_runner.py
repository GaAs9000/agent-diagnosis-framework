"""Bounded adapter around the existing R1 OfficeVal runner.

The adapter owns only new integrated output directories; R1 code and runs stay
unchanged.  Validation happens before importing/starting any runtime actor.
"""
from __future__ import annotations
import ast, asyncio, hashlib, json
from pathlib import Path
from typing import Mapping

TASK_IDS = ("officeval_002", "officeval_003", "officeval_006", "officeval_012",
            "officeval_015", "officeval_018", "officeval_033", "officeval_037")

def _sha(path: Path) -> str:
    h = hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()

def build_task_bundle(source: str | Path, assets: str | Path, verifiers: str | Path,
                      *, pinned_manifest: Mapping | None = None) -> dict:
    """Build an immutable, hash-addressed bundle; fail closed on missing inputs."""
    root, assets, verifiers = Path(source), Path(assets), Path(verifiers)
    if not root.exists() or not assets.exists() or not verifiers.exists():
        raise FileNotFoundError("source, assets, and verifiers are all required")
    # A previously frozen integrated bundle is already authoritative.  Read it
    # back without reinterpreting task metadata or copying files again.
    frozen = root / "manifest.json"
    if frozen.is_file():
        manifest = json.loads(frozen.read_text())
        if tuple(manifest.get("task_ids", ())) != TASK_IDS or manifest.get("bundle_root") != str(root):
            raise ValueError("frozen bundle identity/order mismatch")
        return {"schema_version": "integrated-officeval-bundle-v1", "immutable": True,
                "task_ids": list(TASK_IDS), "bundle_root": str(root),
                "tasks": {task: {"metadata": manifest["tasks"][task],
                                 "verifier_sha256": manifest["files"][manifest["tasks"][task]["verifier_path"]]["sha256"]}
                           for task in TASK_IDS},
                "pinned_manifest": dict(pinned_manifest or {}),
                "manifest_sha256": _sha(frozen)}
    tasks = {}
    for task_id in TASK_IDS:
        task_file = assets / f"{task_id}.json"
        verifier = verifiers / f"{task_id}_verifier.py"
        if not task_file.is_file() or not verifier.is_file():
            raise FileNotFoundError(f"missing asset or verifier for {task_id}")
        try: metadata = json.loads(task_file.read_text())
        except (OSError, ValueError) as exc: raise ValueError(f"invalid metadata {task_id}") from exc
        # Parse verifier without importing/executing it.
        ast.parse(verifier.read_text(), filename=str(verifier))
        tasks[task_id] = {"metadata": metadata, "asset": str(task_file),
                          "asset_sha256": _sha(task_file), "verifier": str(verifier),
                          "verifier_sha256": _sha(verifier)}
    return {"schema_version": "integrated-officeval-bundle-v1", "immutable": True,
            "task_ids": list(TASK_IDS), "bundle_root": str(root), "tasks": tasks,
            "pinned_manifest": dict(pinned_manifest or {})}

async def run_episode(bundle: Mapping, task_id: str, *, agent_image: str,
                      grader_image: str, model_profile: Mapping, output_dir: str | Path,
                      model_fingerprint: str, tokenizer_fingerprint: str,
                      server_pid: int | None = None, guard: str | None = None,
                      seed: int = 2026, max_turns: int = 10, budget: int = 100,
                      system_contract: Mapping | None = None,
                      generation_guard=None) -> dict:
    """Run one fresh serial episode through R1's independently graded path."""
    if task_id not in bundle.get("task_ids", []): raise ValueError("task is not in frozen bundle")
    if seed != 2026 or max_turns < 1 or budget != 100 or max_turns > budget:
        raise ValueError("integrated delivery fixes budget=100 and requires max_turns <= budget")
    if not model_profile or not model_fingerprint or not tokenizer_fingerprint:
        raise ValueError("complete model profile and fingerprints required")
    if server_pid is not None and (type(server_pid) is not int or server_pid <= 0):
        raise ValueError("invalid server PID")
    task_record = bundle["tasks"][task_id]
    task_metadata = task_record.get("metadata", task_record)
    task_manifest = {"bundle_root": bundle["bundle_root"], "tasks": {task_id: task_metadata}}
    # Import only after all fail-closed validation, so missing profile cannot start Docker.
    from c3_sandbox.r1.office_run import run_baseline
    receipt = dict(model_profile)
    receipt.update(model_fingerprint=model_fingerprint, tokenizer_fingerprint=tokenizer_fingerprint)
    result = await run_baseline(task_id, task_manifest, agent_image, grader_image, receipt,
                                Path(output_dir), seed=seed, max_turns=max_turns,
                                system_contract=dict(system_contract) if system_contract else None,
                                generation_guard=generation_guard,
                                protocol_recovery_policy="bounded")
    used_fragments = len(result.get("tool_dispatch", []))
    if used_fragments > budget:
        raise RuntimeError("episode execution fragment budget exceeded")
    result["integrated"] = {"guard": guard, "server_pid": server_pid, "budget": budget,
                             "used_fragments": used_fragments,
                             "tokenizer_fingerprint": tokenizer_fingerprint}
    return result

def run_episode_sync(*args, **kwargs):
    return asyncio.run(run_episode(*args, **kwargs))
