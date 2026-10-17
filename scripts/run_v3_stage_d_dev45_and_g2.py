#!/usr/bin/env python3
"""V3 Stage G & Stage D: OfficeVal 45-Episode Dev Cohort + V3-G2 20-Episode c0 Equivalence + E1 Census.

Strictly follows Notion V3 (§3.1-3.7, §5.1, §9.1-9.4):
- Unified Actor: Qwen/Qwen3.6-27B on aml-b (frozen weight_digest & tokenizer_digest)
- Harness: pi-coding-agent@0.73.1 with deterministic context compaction enabled
- Sampling: Temperature = 1.0, unseeded (fresh stochastic sampling per V3 §3.4)
- Dev45: 9 dev families x 5 replicates = 45 episodes
- V3-G2: 20 c0 (cp000000) cold-resumed episodes paired by task against fresh dev runs
  (also verifies 20/20 byte-identical r000000/request.json for V3-G1)
- E1 Census: Zero-request snapshot grading across all 45 episodes + V3-G3 terminal consistency check
"""
from __future__ import annotations

import concurrent.futures
import copy
import hashlib
import json
import random
import shutil
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
L1_ROOT = ROOT / "L1_sandbox" / "C3_Experimental_Sandbox"
L2_ROOT = ROOT / "L2_Behavioral Event Localization" / "L2_Sandbox"
sys.path.insert(0, str(L1_ROOT / "src"))
sys.path.insert(0, str(L2_ROOT / "src"))

from c3_sandbox.l1.core.config import resolve as resolve_config
from c3_sandbox.l1.core.orchestrator import run as l1_run
from c3_sandbox.l1.evaluation.study import study_outcome
from c3_sandbox.l1.evidence.live import verify as l1_verify
from c3_sandbox.l1.storage import read_json, utc_now
from l2_sandbox.defects import build_defect_table
from l2_sandbox.evidence.l1 import L1Package
from l2_sandbox.facts.build import build_facts
from l2_sandbox.grades import grade_episode_snapshots as grade_episode

CONFIG_PATH = L1_ROOT / "configs" / "l1" / "officeval.qwen36.pi.t1.json"
OUT_ROOT = L1_ROOT / "runs" / "v3_qwen36_pi_dev45_20261017"
PROGRESS_JSON = OUT_ROOT / "progress.json"
REPORT_JSON = L2_ROOT / "reports" / "v3_stage_d_dev45_and_g2_20261017.json"
REPORT_MD = L2_ROOT / "reports" / "V3_STAGE_D_DEV45_AND_G2_REPORT_20261017.md"

DEV_TASKS = [
    "officeval_003",
    "officeval_004",
    "officeval_010",
    "officeval_033",
    "officeval_050",
    "officeval_060",
    "officeval_070",
    "officeval_085",
    "officeval_090",
]
REPLICATES = ["r00", "r01", "r02", "r03", "r04"]

# Pre-registered 20-slot c0 cold-resume cohort for V3-G2 (2 per dev family + 060.r02 + 090.r02)
G2_SLOTS = [(t, r) for r in ("r00", "r01") for t in DEV_TASKS] + [
    ("officeval_060", "r02"),
    ("officeval_090", "r02"),
]

_LOCK = threading.Lock()


def wait_for_endpoint(url: str = "http://127.0.0.1:8001/v1/models", timeout_s: int = 180) -> dict:
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last_err = exc
        time.sleep(2)
    raise RuntimeError(f"Endpoint {url} not ready after {timeout_s}s: {last_err}")


def is_valid_episode(ep_dir: Path) -> tuple[bool, dict[str, Any]]:
    if not ep_dir.exists() or not (ep_dir / "manifest.json").exists():
        return False, {}
    try:
        v = l1_verify(ep_dir)
        if not v.get("verified"):
            return False, {"reason": "verify_failed"}
        manifest = read_json(ep_dir / "manifest.json")
        study = study_outcome(ep_dir)
        ok = (
            manifest.get("status", {}).get("execution_validity") == "valid"
            and bool(study.get("study_valid"))
        )
        return ok, {
            "episode_id": manifest.get("episode_id"),
            "index_sha256": v.get("index_sha256"),
            "study_score": float(study.get("study_outcome_y") or 0.0),
            "study_outcome": study.get("study_classification"),
            "study_valid": bool(study.get("study_valid")),
            "termination": manifest.get("status", {}).get("termination"),
            "model_requests": int(manifest.get("counts", {}).get("model_requests", 0)),
            "tool_calls": int(manifest.get("counts", {}).get("tool_calls", 0)),
            "agent_phase_seconds": float(manifest.get("usage", {}).get("agent_phase_seconds", 0.0)),
        }
    except Exception as exc:
        return False, {"reason": str(exc)}


def update_progress(state: dict[str, Any]) -> None:
    with _LOCK:
        state["updated_at"] = utc_now()
        tmp = PROGRESS_JSON.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(PROGRESS_JSON)


def run_dev_slot(base_cfg: dict, task_id: str, rep_id: str, progress_state: dict[str, Any]) -> dict[str, Any]:
    run_label = f"{task_id}.{rep_id}"
    ep_dir = OUT_ROOT / "episodes" / run_label
    ep_dir.parent.mkdir(parents=True, exist_ok=True)

    ok, info = is_valid_episode(ep_dir)
    if ok:
        print(f"[DEV45-SKIP] {run_label} already valid (score={info['study_score']:.4f})", flush=True)
        with _LOCK:
            progress_state["dev45_completed"][run_label] = {**info, "cached": True}
        update_progress(progress_state)
        return {"run_label": run_label, "task_id": task_id, "rep_id": rep_id, "ep_dir": str(ep_dir), **info, "cached": True}

    cfg = copy.deepcopy(base_cfg)
    cfg["benchmark"]["task_id"] = task_id
    # Ensure unseeded T=1.0 per V3 §3.4
    cfg["model"]["parameters"].pop("seed", None)

    for attempt in range(1, 4):
        if ep_dir.exists():
            failed_dir = OUT_ROOT / "failed_attempts" / f"{run_label}.attempt{attempt - 1}_{int(time.time())}"
            failed_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(ep_dir), str(failed_dir))

        t0 = time.monotonic()
        print(f"[DEV45-START] {run_label} (attempt {attempt})", flush=True)
        try:
            l1_run(cfg, ep_dir)
        except Exception as exc:
            print(f"[DEV45-WARN] {run_label} attempt {attempt} raised {exc}", flush=True)

        dt = time.monotonic() - t0
        ok, info = is_valid_episode(ep_dir)
        if ok:
            print(
                f"[DEV45-DONE] {run_label} in {dt:.1f}s | score={info['study_score']:.4f} | "
                f"reqs={info['model_requests']} | term={info['termination']}",
                flush=True,
            )
            res = {
                "run_label": run_label,
                "task_id": task_id,
                "rep_id": rep_id,
                "ep_dir": str(ep_dir),
                "wall_seconds": round(dt, 2),
                "attempts": attempt,
                "cached": False,
                **info,
            }
            with _LOCK:
                progress_state["dev45_completed"][run_label] = res
            update_progress(progress_state)
            return res
        print(f"[DEV45-RETRY] {run_label} attempt {attempt} invalid: {info}", flush=True)

    raise RuntimeError(f"Slot {run_label} failed after 3 attempts")


def run_g2_slot(task_id: str, rep_id: str, progress_state: dict[str, Any]) -> dict[str, Any]:
    run_label = f"{task_id}.{rep_id}"
    parent_dir = OUT_ROOT / "episodes" / run_label
    child_dir = OUT_ROOT / "v3_g2_c0_resumes" / f"{run_label}_c0"
    child_dir.parent.mkdir(parents=True, exist_ok=True)

    ok, info = is_valid_episode(child_dir)
    if not ok:
        parent_cfg = read_json(parent_dir / "raw" / "resolved_config.json")
        for attempt in range(1, 4):
            if child_dir.exists():
                shutil.rmtree(child_dir)
            t0 = time.monotonic()
            print(f"[V3-G2-START] {run_label}_c0 from cp000000 (attempt {attempt})", flush=True)
            try:
                l1_run(parent_cfg, child_dir, parent=parent_dir, checkpoint_id="cp000000")
            except Exception as exc:
                print(f"[V3-G2-WARN] {run_label}_c0 attempt {attempt} raised {exc}", flush=True)
            dt = time.monotonic() - t0
            ok, info = is_valid_episode(child_dir)
            if ok:
                info["wall_seconds"] = round(dt, 2)
                break
        if not ok:
            raise RuntimeError(f"V3-G2 slot {run_label}_c0 failed after 3 attempts")

    # Verify V3-G1 byte-identical first request r000000/request.json
    parent_req0 = parent_dir / "raw" / "model" / "r000000" / "request.json"
    child_req0 = child_dir / "raw" / "model" / "r000000" / "request.json"
    parent_sha = hashlib.sha256(parent_req0.read_bytes()).hexdigest()
    child_sha = hashlib.sha256(child_req0.read_bytes()).hexdigest()

    _, parent_info = is_valid_episode(parent_dir)
    res = {
        "run_label": run_label,
        "task_id": task_id,
        "rep_id": rep_id,
        "parent_episode_id": parent_info.get("episode_id"),
        "child_episode_id": info.get("episode_id"),
        "parent_score": parent_info.get("study_score", 0.0),
        "c0_resume_score": info.get("study_score", 0.0),
        "score_diff_c0_minus_parent": round(info.get("study_score", 0.0) - parent_info.get("study_score", 0.0), 6),
        "r000000_parent_sha256": parent_sha,
        "r000000_child_sha256": child_sha,
        "byte_identical_first_request": parent_sha == child_sha,
        "c0_model_requests": info.get("model_requests", 0),
        "c0_termination": info.get("termination"),
        "c3_l1_verified": True,
    }
    print(
        f"[V3-G2-DONE] {run_label}_c0 | parent={res['parent_score']:.4f} | "
        f"c0={res['c0_resume_score']:.4f} | byte_id={res['byte_identical_first_request']}",
        flush=True,
    )
    with _LOCK:
        progress_state["g2_completed"][run_label] = res
    update_progress(progress_state)
    return res


def analyze_dev45_episode(ep_dir: Path, run_label: str, task_id: str, rep_id: str) -> dict[str, Any]:
    _, info = is_valid_episode(ep_dir)
    pkg = L1Package(ep_dir)
    grade_rows = grade_episode(pkg, cache_dir=OUT_ROOT / "grade_cache", verify_terminal_submission=True)
    facts = build_facts(pkg)
    defects_res = build_defect_table(pkg, grade_rows, facts)

    term_row = grade_rows[0]
    cp_rows = [r for r in grade_rows[1:] if r.get("kind") == "checkpoint"]
    snap_scores = [float(r.get("score") or 0.0) for r in cp_rows]
    max_snap_score = max(snap_scores) if snap_scores else 0.0
    terminal_snap_score = snap_scores[-1] if snap_scores else 0.0
    v3_g3_abs_diff = abs(float(term_row.get("score") or 0.0) - info["study_score"])

    ever_had_candidate = any(r.get("candidate_status") == "ok" for r in cp_rows)
    any_cp_dim1_pass = any(bool(r.get("dim1_pass")) for r in cp_rows)
    has_terminal_sub = bool(term_row.get("in_submission"))
    term_dim1_pass = bool(term_row.get("dim1_pass"))

    defects = defects_res.get("defects") or []
    ded_by_kind = {"broken": 0.0, "undelivered": 0.0, "never": 0.0}
    cnt_by_kind = {"broken": 0, "undelivered": 0, "never": 0}
    for d in defects:
        k = d["kind"]
        ded_by_kind[k] = ded_by_kind.get(k, 0.0) + float(d["deduction"])
        cnt_by_kind[k] = cnt_by_kind.get(k, 0) + 1

    compaction_events = 0
    for line in (ep_dir / "trajectory.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip() and '"session.compacted"' in line:
            compaction_events += 1

    if info["study_score"] >= 1.0 - 1e-9:
        substate = "perfect_score"
    elif ded_by_kind["broken"] > 1e-9:
        substate = "e1_broken"
    elif ded_by_kind["undelivered"] > 1e-9:
        substate = "e1_undelivered"
    elif not ever_had_candidate:
        substate = "gate_no_candidate_ever"
    elif not has_terminal_sub:
        substate = "gate_unsubmitted_candidate"
    elif not term_dim1_pass:
        substate = "gate_submitted_dim1_fail"
    else:
        substate = "dim2_delivered_item_wrong"

    return {
        "run_label": run_label,
        "task_id": task_id,
        "rep_id": rep_id,
        **info,
        "max_snapshot_score": round(max_snap_score, 6),
        "terminal_snapshot_score": round(terminal_snap_score, 6),
        "delivery_gap_g": round(max(0.0, max_snap_score - info["study_score"]), 6),
        "v3_g3_abs_diff": v3_g3_abs_diff,
        "ever_had_candidate": ever_had_candidate,
        "any_cp_dim1_pass": any_cp_dim1_pass,
        "term_dim1_pass": term_dim1_pass,
        "compaction_events": compaction_events,
        "defects_count_by_kind": cnt_by_kind,
        "defects_deduction_by_kind": {k: round(v, 6) for k, v in ded_by_kind.items()},
        "substate": substate,
    }


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    wait_for_endpoint()
    base_cfg = resolve_config(CONFIG_PATH)

    progress_state: dict[str, Any] = {
        "started_at": utc_now(),
        "total_dev45_target": 45,
        "total_g2_target": 20,
        "dev45_completed": {},
        "g2_completed": {},
    }
    update_progress(progress_state)

    t0 = time.monotonic()
    dev_jobs = [(task_id, rep_id) for rep_id in REPLICATES for task_id in DEV_TASKS]
    print(f"[STAGE-D] Launching 45 OfficeVal Dev episodes (max_workers=3)...", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futs = [pool.submit(run_dev_slot, base_cfg, t, r, progress_state) for t, r in dev_jobs]
        for f in concurrent.futures.as_completed(futs):
            f.result()

    dev_wall = time.monotonic() - t0
    print(f"[STAGE-D] All 45 Dev episodes completed in {dev_wall:.1f}s!", flush=True)

    t_g2 = time.monotonic()
    print(f"[V3-G2] Launching 20 c0 cold-resume episodes (max_workers=3)...", flush=True)
    g2_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futs = [pool.submit(run_g2_slot, t, r, progress_state) for t, r in G2_SLOTS]
        for f in concurrent.futures.as_completed(futs):
            g2_results.append(f.result())
    g2_results.sort(key=lambda x: x["run_label"])
    g2_wall = time.monotonic() - t_g2
    print(f"[V3-G2] All 20 c0 cold-resume episodes completed in {g2_wall:.1f}s!", flush=True)

    # Run E1 Zero-Request Snapshot Census across all 45 dev episodes
    print(f"[STAGE-D] Running E1 zero-request snapshot census across 45 dev episodes...", flush=True)
    ep_Rows = []
    for rep_id in REPLICATES:
        for task_id in DEV_TASKS:
            run_label = f"{task_id}.{rep_id}"
            ep_dir = OUT_ROOT / "episodes" / run_label
            ep_Rows.append(analyze_dev45_episode(ep_dir, run_label, task_id, rep_id))
    ep_Rows.sort(key=lambda x: x["run_label"])

    # Compute V3-G2 equivalence statistics (paired by task_id across the 20 pairs & task-mean comparison)
    diffs = [r["score_diff_c0_minus_parent"] for r in g2_results]
    mean_diff = statistics.mean(diffs)
    rng = random.Random(20261017)
    boot_means = []
    for _ in range(10000):
        sample = [rng.choice(diffs) for _ in range(len(diffs))]
        boot_means.append(statistics.mean(sample))
    boot_means.sort()
    ci_low = boot_means[int(0.025 * len(boot_means))]
    ci_high = boot_means[int(0.975 * len(boot_means))]
    within_pm_005 = (ci_low >= -0.05) and (ci_high <= 0.05)

    payload = {
        "schema_version": "0.5.0",
        "generated_at": utc_now(),
        "dev45_wall_seconds": round(dev_wall, 2),
        "g2_wall_seconds": round(g2_wall, 2),
        "dev45_summary": {
            "total_episodes": len(ep_Rows),
            "verified_count": sum(1 for r in ep_Rows if r["study_valid"]),
            "mean_study_score": round(statistics.mean(r["study_score"] for r in ep_Rows), 6),
            "mean_max_snapshot_score": round(statistics.mean(r["max_snapshot_score"] for r in ep_Rows), 6),
            "nonzero_score_count": sum(1 for r in ep_Rows if r["study_score"] > 0),
            "perfect_score_count": sum(1 for r in ep_Rows if r["study_score"] >= 1.0 - 1e-9),
            "total_model_requests": sum(r["model_requests"] for r in ep_Rows),
            "total_tool_calls": sum(r["tool_calls"] for r in ep_Rows),
            "v3_g3_max_abs_diff": max(r["v3_g3_abs_diff"] for r in ep_Rows),
            "substate_counts": {
                s: sum(1 for r in ep_Rows if r["substate"] == s)
                for s in sorted({r["substate"] for r in ep_Rows})
            },
        },
        "v3_g2_summary": {
            "n_pairs": len(g2_results),
            "all_byte_identical_r000000": all(r["byte_identical_first_request"] for r in g2_results),
            "mean_parent_score": round(statistics.mean(r["parent_score"] for r in g2_results), 6),
            "mean_c0_resume_score": round(statistics.mean(r["c0_resume_score"] for r in g2_results), 6),
            "mean_diff_c0_minus_parent": round(mean_diff, 6),
            "bootstrap_95_ci": [round(ci_low, 6), round(ci_high, 6)],
            "within_pm_005_equivalence_bound": within_pm_005,
            "verdict": "PASS" if within_pm_005 else "INCONCLUSIVE_AT_N20",
            "pairs": g2_results,
        },
        "dev45_episodes": ep_Rows,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[COMPLETE] Saved {REPORT_JSON}", flush=True)


if __name__ == "__main__":
    main()
