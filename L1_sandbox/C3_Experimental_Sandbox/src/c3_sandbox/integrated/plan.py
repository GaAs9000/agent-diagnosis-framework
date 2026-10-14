"""Frozen integrated delivery plan for the bounded OfficeVal comparison.

This module is deliberately side-effect free: planning and analysis inspect saved
runs only.  Execution is exposed as a serial hook and is never started implicitly.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib, json
import re
from typing import Any, Callable, Iterable, Mapping

TASK_IDS = ("officeval_002", "officeval_003", "officeval_006", "officeval_012",
            "officeval_015", "officeval_018", "officeval_033", "officeval_037")
TARGET_MODEL = "Qwen3-Coder-30B-A3B"
REFERENCE_MODEL = "Qwen3-4B"
SEED = 2026
MAX_ORIGINAL_EPISODES = 16
MAX_ERROR_REVISIONS = 3
MAX_EXECUTION_FRAGMENTS = 100


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _hash_paths(paths: Iterable[str | Path]) -> dict[str, str]:
    result = {}
    for value in sorted((Path(p) for p in paths), key=lambda p: str(p)):
        if value.is_file(): result[str(value)] = sha256_file(value)
    return result


@dataclass(frozen=True)
class FrozenPlan:
    schema_version: str
    task_ids: tuple[str, ...]
    target_model: str
    reference_model: str
    seed: int
    max_original_episodes: int
    max_error_revisions: int
    max_execution_fragments: int
    software_delivery: dict
    runtime_coverage: dict
    natural_failure_side: dict
    natural_success_side: dict
    method_comparison: dict
    scientific_claim: dict
    hashes: dict
    excluded_tasks: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        value = asdict(self)
        value["task_ids"] = list(self.task_ids)
        value["excluded_tasks"] = list(self.excluded_tasks)
        return value


def build_plan(*, source_paths: Iterable[str | Path] = (), asset_paths: Iterable[str | Path] = (),
               verifier_paths: Iterable[str | Path] = (), task_ids: Iterable[str] = TASK_IDS,
               target_model: str = TARGET_MODEL, reference_model: str = REFERENCE_MODEL,
               seed: int = SEED, existing_runs: Iterable[Mapping] = ()) -> FrozenPlan:
    requested = tuple(task_ids)
    excluded = tuple(t for t in requested if t not in TASK_IDS)
    selected = tuple(t for t in TASK_IDS if t in requested)
    if not selected: raise ValueError("no qualifying fixed task IDs")
    runs = list(existing_runs)
    present = {str(r.get("task_id", r.get("episode_id", ""))) for r in runs if isinstance(r, Mapping)}
    return FrozenPlan(
        schema_version="integrated-delivery-freeze-v1", task_ids=selected,
        target_model=target_model, reference_model=reference_model, seed=seed,
        max_original_episodes=MAX_ORIGINAL_EPISODES, max_error_revisions=MAX_ERROR_REVISIONS,
        max_execution_fragments=MAX_EXECUTION_FRAGMENTS,
        software_delivery={"status": "planned", "analysis_only": True},
        runtime_coverage={"target": "unknown", "reference": "unknown", "saved_run_count": len(runs),
                          "present_task_ids": sorted(present & set(selected))},
        natural_failure_side={"status": "unknown", "source": "saved_runs_only"},
        natural_success_side={"status": "unknown", "source": "saved_runs_only"},
        method_comparison={"status": "planned", "methods": ["target", "reference"]},
        scientific_claim={"status": "not_established", "claim": None},
        hashes={"source": _hash_paths(source_paths), "task_ids": sha256_json(list(selected)),
                "asset": _hash_paths(asset_paths), "verifier": _hash_paths(verifier_paths)}, excluded_tasks=excluded)


def serial_execute(plan: FrozenPlan, runner: Callable[[str, str], Mapping], *, dry_run: bool = True) -> list[dict]:
    """Execute a bounded target/reference matrix serially through an injected hook.

    Tests can inject a recorder; the default dry-run cannot launch models or Docker.
    """
    if not dry_run and not callable(runner): raise ValueError("runner hook required")
    out = []
    for task_id in plan.task_ids:
        for model in (plan.target_model, plan.reference_model):
            out.append({"task_id": task_id, "model": model, "status": "planned"} if dry_run
                       else dict(runner(task_id, model)))
    if len(out) * 1 > plan.max_original_episodes: raise RuntimeError("episode budget exceeded")
    return out


def analyze_saved_runs(plan: FrozenPlan, runs: Iterable[Mapping]) -> dict:
    rows = []
    allowed = set(plan.task_ids)
    for run in runs:
        if not isinstance(run, Mapping): continue
        raw = str(run.get("task_id", run.get("episode_id", "")))
        match = re.search(r"officeval_\d{3}", raw)
        task_id = match.group(0) if match else raw
        if task_id not in allowed: continue
        rows.append(dict(run))
    return {"status": "ok" if rows else "unknown", "data_state": "observed" if rows else "no_data",
            "runs": rows, "plan_hash": sha256_json(plan.to_dict())}
