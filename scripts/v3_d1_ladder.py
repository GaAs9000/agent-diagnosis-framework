#!/usr/bin/env python3
"""V3 Stage D1 Entrypoint (R15, R18): 3D Stratified Intervention Ladder, Single-Item R Localization, and R12 Harm Check.

Contract (Notion V3 §3.4, §8.2, R7–R12, R15, R18):
- Builds the 3D stratified sampling frame (`exhausted_budget x valid_delivery x rule_source`) strictly from
  the new `Qwen/Qwen3.6-27B + pi` development cohort (`v3_qwen36_pi_dev45_20261017`), rejecting legacy DeepSeek runs.
- Constructs whole-task `explicit` and `implicit` requirement lists (`build_task_requirement_lists`) and
  nested ladder messages (`P`, `D`, `S`, `R_explicit`, `R_implicit`, `R_single`) with SHA-256 hashes,
  marking empty requirement levels as `ineligible` (never falling back to `D`).
- Includes a passing-sample harm check (`R12`: comparing `D`, `S`, `R_explicit` against `P` on passing items/runs).
- Defaults to `--plan-only` (`R18`): outputs `v3_d1_ladder_plan_20261017.json` with exact request cap,
  wall-clock estimate, and stop conditions, initiating ZERO model requests unless `--execute --d0-confirmed` is passed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
L1_ROOT = REPO_ROOT / "L1_sandbox" / "C3_Experimental_Sandbox"
L2_ROOT = REPO_ROOT / "L2_Behavioral Event Localization" / "L2_Sandbox"
sys.path.insert(0, str(L1_ROOT / "src"))
sys.path.insert(0, str(L2_ROOT / "src"))

from c3_sandbox.l1.evaluation.study import study_outcome  # noqa: E402
from l2_sandbox.v05 import (  # noqa: E402
    DEFAULT_SELF_CHECK_DRAFT,
    DiagnosticUnitV05,
    E1Observation,
    E2ResponseVector,
    V05_SCHEMA_VERSION,
    build_interleaved_arm_schedule,
    build_ladder_intervention_messages,
    build_task_requirement_lists,
    build_v3_stratified_sampling_frame,
    select_v05_intervention_anchors,
    verify_nested_ladder_messages,
)

DEFAULT_DEV45_RUNS_DIR = L1_ROOT / "runs" / "v3_qwen36_pi_dev45_20261017" / "episodes"
DEFAULT_DEV45_DIAG_DIR = L2_ROOT / "diagnoses" / "v3_qwen36_pi_dev45_20261017"
DEFAULT_P0A_CATALOG = L2_ROOT / "reports" / "v3_p0a_audit_20261017.json"
DEFAULT_PLAN_OUT = L2_ROOT / "reports" / "v3_d1_ladder_plan_20261017.json"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _synthesized_cp_rows_from_package(ep_dir: Path, max_requests: int = 35) -> list[dict[str, Any]]:
    cp_dir = ep_dir / "checkpoints"
    rows: list[dict[str, Any]] = []
    if not cp_dir.exists():
        return rows
    for cp_file in sorted(cp_dir.glob("cp*.json")):
        cp_id = cp_file.stem
        k_val = int(cp_id[2:]) if cp_id[2:].isdigit() else 0
        rows.append({
            "kind": "checkpoint",
            "checkpoint": cp_id,
            "k": k_val,
            "remaining_requests": max(0, max_requests - k_val),
            "candidate_status": "no_candidate",
            "score": 0.0,
            "cold_resume_eligible": True,
        })
    return rows


def build_d1_units_and_plan(
    *,
    runs_dir: Path,
    diag_dir: Path,
    p0a_catalog_path: Path,
    n_per_arm: int = 12,
    r_b: int = 6,
    max_requests: int = 35,
    target_per_stratum: int = 3,
    passing_harm_sample_size: int = 4,
    self_check_text: str = DEFAULT_SELF_CHECK_DRAFT,
) -> dict[str, Any]:
    p0a = _load_json(p0a_catalog_path) if p0a_catalog_path.exists() else {}
    task_rule_catalog: dict[str, dict[str, dict[str, Any]]] = p0a.get("task_rule_catalog") or {}

    units: list[DiagnosticUnitV05] = []
    passing_candidates: list[dict[str, Any]] = []
    task_ladder_specs: dict[str, Any] = {}
    scanned_episodes: list[str] = []

    if runs_dir.exists():
        for ep_dir in sorted(runs_dir.iterdir()):
            if not ep_dir.is_dir() or not (ep_dir / "manifest.json").exists():
                continue
            st_out = study_outcome(ep_dir)
            if not st_out.get("study_valid"):
                continue

            manifest = _load_json(ep_dir / "manifest.json")
            og = _load_json(ep_dir / "official_grade.json")
            raw_og = og.get("official_raw") or {}

            ep_id = ep_dir.name
            scanned_episodes.append(ep_id)
            task_id = str((manifest.get("task") or {}).get("task_id") or ep_id.split(".")[0])
            short_task = task_id.split(":")[-1]
            termination = str((manifest.get("status") or {}).get("termination") or "")
            exhausted_budget = termination == "request_budget"

            dim1_pass = bool(raw_og.get("dim1_pass"))
            term_score = float(st_out.get("study_outcome_y") or 0.0)

            req_lists = build_task_requirement_lists(task_rule_catalog, short_task)
            ladder_msgs = build_ladder_intervention_messages(
                explicit_requirements=req_lists["explicit"],
                implicit_requirements=req_lists["implicit"],
                self_check_text=self_check_text,
            )
            nest_verification = verify_nested_ladder_messages(ladder_msgs)
            task_ladder_specs[short_task] = {
                "task_id": short_task,
                "explicit_count": len(req_lists["explicit"]),
                "implicit_count": len(req_lists["implicit"]),
                "ambiguous_count": len(req_lists["ambiguous"]),
                "ladder_messages": ladder_msgs,
                "nesting_check": nest_verification,
            }

            grades_path = diag_dir / ep_id / "grades.jsonl"
            cp_rows = _load_jsonl(grades_path)
            if not cp_rows:
                cp_rows = _synthesized_cp_rows_from_package(ep_dir, max_requests=max_requests)

            anchor_info = select_v05_intervention_anchors(
                cp_rows,
                max_requests=max_requests,
                r_b=r_b,
                termination=termination,
            )

            if term_score >= 0.999 and anchor_info["eligible"]:
                passing_candidates.append({
                    "unit_id": f"{ep_id}::PASSING_WHOLE_RUN",
                    "episode_id": ep_id,
                    "task_id": short_task,
                    "terminal_score_y": term_score,
                    "anchor_k": anchor_info["primary_anchor_k"],
                    "remaining_budget": anchor_info["primary_remaining_budget"],
                    "arms": ["P", "D", "S"] + (["R_explicit"] if ladder_msgs["R_explicit"]["eligible"] else []),
                })

            if not dim1_pass:
                units.append(
                    DiagnosticUnitV05(
                        unit_id=f"{ep_id}::GATE_NO_VALID_DELIVERY",
                        episode_id=ep_id,
                        task_id=short_task,
                        benchmark="officeval",
                        unit_level="gate_or_no_artifact",
                        defect_name="dim1_gate_failure",
                        deduction_weight=round(1.0 - term_score, 6),
                        rule_source="gate_or_no_artifact",
                        e1_observation=E1Observation(
                            e1_class="never_succeeded_on_disk",
                            candidate_substate="no_candidate",
                            k_star=0,
                            max_item_score=0.0,
                            terminal_item_score=0.0,
                            deduction=round(1.0 - term_score, 6),
                        ),
                        e2_response_vector=E2ResponseVector(
                            evaluated=False,
                            anchor_k=anchor_info["primary_anchor_k"],
                            anchor_type=anchor_info["primary_anchor_type"],
                            remaining_budget_at_anchor=anchor_info["primary_remaining_budget"],
                        ),
                        operational_label=None,
                        evidence_status="inconclusive" if anchor_info["eligible"] else "ineligible",
                        exhausted_budget=exhausted_budget,
                        valid_delivery=False,
                    )
                )
            else:
                dim2_items = raw_og.get("dim2_items") or []
                max_score = float(raw_og.get("max_score") or 1.0)
                if max_score <= 0:
                    max_score = 1.0
                for idx, item in enumerate(dim2_items):
                    max_d = float(item.get("max_delta") or 0.0)
                    act_d = float(item.get("delta") or 0.0)
                    if max_d > 0:
                        lost_pts = max(0.0, max_d - act_d)
                    elif max_d < 0:
                        lost_pts = max(0.0, -act_d)
                    else:
                        lost_pts = float(item.get("deduction") or 0.0) * max_score
                    ded = lost_pts / max_score
                    rule_txt = str(item.get("rule") or f"item_{idx}").strip()
                    if ded <= 1e-9:
                        if max_d > 0 and anchor_info["eligible"] and len(passing_candidates) < passing_harm_sample_size * 4:
                            passing_candidates.append({
                                "unit_id": f"{ep_id}::PASSING_ITEM::{idx}",
                                "episode_id": ep_id,
                                "task_id": short_task,
                                "rule": rule_txt,
                                "terminal_score_y": 1.0,
                                "anchor_k": anchor_info["primary_anchor_k"],
                                "remaining_budget": anchor_info["primary_remaining_budget"],
                                "arms": ["P", "D", "S"] + (["R_explicit"] if ladder_msgs["R_explicit"]["eligible"] else []),
                            })
                        continue
                    cat_map = task_rule_catalog.get(f"officeval:{short_task}") or task_rule_catalog.get(short_task) or {}
                    r_src = str((cat_map.get(rule_txt) or {}).get("source_category") or "ambiguous")
                    units.append(
                        DiagnosticUnitV05(
                            unit_id=f"{ep_id}::ITEM::{idx}",
                            episode_id=ep_id,
                            task_id=short_task,
                            benchmark="officeval",
                            unit_level="rubric_item",
                            defect_name=rule_txt,
                            deduction_weight=round(ded, 6),
                            rule_source=r_src,
                            e1_observation=E1Observation(
                                e1_class="never_succeeded_on_disk",
                                candidate_substate="delivered_item_wrong",
                                k_star=anchor_info["primary_anchor_k"] or 0,
                                max_item_score=0.0,
                                terminal_item_score=0.0,
                                deduction=round(ded, 6),
                            ),
                            e2_response_vector=E2ResponseVector(
                                evaluated=False,
                                anchor_k=anchor_info["primary_anchor_k"],
                                anchor_type=anchor_info["primary_anchor_type"],
                                remaining_budget_at_anchor=anchor_info["primary_remaining_budget"],
                            ),
                            operational_label=None,
                            evidence_status="inconclusive" if anchor_info["eligible"] else "ineligible",
                            exhausted_budget=exhausted_budget,
                            valid_delivery=True,
                        )
                    )

    frame = build_v3_stratified_sampling_frame(
        units,
        dataset_id="v3_qwen36_pi_dev45_20261017",
        actor_model="Qwen/Qwen3.6-27B",
        target_sample_per_stratum=target_per_stratum,
    )

    chosen_passing = passing_candidates[:passing_harm_sample_size]

    ladder_branch_count = 0
    estimated_actor_requests = 0
    unit_schedules: list[dict[str, Any]] = []

    for s_unit in frame["sampled_units"]:
        t_id = s_unit["task_id"]
        spec = task_ladder_specs.get(t_id, {})
        l_msgs = spec.get("ladder_messages") or {}
        arms = ["P", "D", "S"]
        if (l_msgs.get("R_explicit") or {}).get("eligible"):
            arms.append("R_explicit")
        if (l_msgs.get("R_implicit") or {}).get("eligible"):
            arms.append("R_implicit")
        if s_unit["rule_source"] != "gate_or_no_artifact":
            arms.append("R_single")
        rem_b = int(s_unit.get("remaining_budget") or r_b)
        ank = int(s_unit.get("anchor_k") or (max_requests - r_b))
        sched = build_interleaved_arm_schedule(arms, n_per_arm, anchor_k=ank, remaining_budget=rem_b)
        ladder_branch_count += len(sched)
        estimated_actor_requests += len(sched) * rem_b
        unit_schedules.append({
            "unit_id": s_unit["unit_id"],
            "task_id": t_id,
            "stratum_id": s_unit["stratum_id"],
            "inclusion_prob": s_unit["inclusion_prob"],
            "anchor_k": ank,
            "remaining_budget": rem_b,
            "eligible_arms": arms,
            "ineligible_arms": [
                a for a in ("R_explicit", "R_implicit")
                if not (l_msgs.get(a) or {}).get("eligible")
            ],
            "planned_branches": len(sched),
            "max_requests_cap": len(sched) * rem_b,
        })

    harm_branches = sum(len(p["arms"]) * n_per_arm for p in chosen_passing)
    harm_requests = sum(len(p["arms"]) * n_per_arm * int(p.get("remaining_budget") or r_b) for p in chosen_passing)

    total_planned_branches = ladder_branch_count + harm_branches
    total_max_requests = estimated_actor_requests + harm_requests
    measured_req_per_hour = 352.3

    return {
        "schema_version": V05_SCHEMA_VERSION,
        "stage": "D1",
        "mode": "plan_only",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Development-set descriptive planning artifact (R15, R18). "
            "Does NOT initiate any Actor or Analyst requests until D0 is frozen and user confirms --execute --d0-confirmed."
        ),
        "source_dataset": {
            "dataset_id": "v3_qwen36_pi_dev45_20261017",
            "runs_dir": str(runs_dir),
            "scanned_valid_episodes": len(scanned_episodes),
            "episode_ids": scanned_episodes,
        },
        "d0_candidate_parameters": {
            "n_per_arm": n_per_arm,
            "r_b": r_b,
            "delta": 0.10,
            "tau": 0.20,
            "alpha": 0.05,
            "self_check_S_draft": self_check_text,
            "self_check_S_status": "D0_CANDIDATE_PENDING_USER_CONFIRMATION",
        },
        "stratified_sampling_frame": frame,
        "r12_passing_harm_check_plan": {
            "candidate_pool_size": len(passing_candidates),
            "sampled_passing_unit_count": len(chosen_passing),
            "sampled_passing_units": chosen_passing,
            "planned_branches": harm_branches,
            "max_requests_cap": harm_requests,
        },
        "task_ladder_message_specs": task_ladder_specs,
        "unit_schedules": unit_schedules,
        "budget_and_stop_conditions": {
            "total_planned_branches": total_planned_branches,
            "max_actor_requests_cap": total_max_requests,
            "analyst_requests": 0,
            "measured_throughput_req_per_hour_3_concurrency": measured_req_per_hour,
            "estimated_wall_clock_hours_upper_bound": round(total_max_requests / measured_req_per_hour, 2),
            "estimated_api_cost_usd": 0.0,
            "stop_conditions": [
                "Abort immediately if invalid_attempt_rate (transport/protocol error) exceeds 15% over any 20 consecutive branches.",
                "Abort immediately if any continuation branch violates same-checkpoint or same-remaining-budget invariant.",
                "Abort if V3-G2 or V3-G3 gate fails or if user revokes execution authorization.",
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="V3 Stage D1 Ladder & R12 Harm Check (default: --plan-only)")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_DEV45_RUNS_DIR)
    parser.add_argument("--diag-dir", type=Path, default=DEFAULT_DEV45_DIAG_DIR)
    parser.add_argument("--p0a-catalog", type=Path, default=DEFAULT_P0A_CATALOG)
    parser.add_argument("--out", type=Path, default=DEFAULT_PLAN_OUT)
    parser.add_argument("--n-per-arm", type=int, default=12)
    parser.add_argument("--r-b", type=int, default=6)
    parser.add_argument("--target-per-stratum", type=int, default=3)
    parser.add_argument("--plan-only", action="store_true", default=True, help="Default: generate plan JSON without running requests")
    parser.add_argument("--execute", action="store_true", help="Execute planned branches (requires --d0-confirmed)")
    parser.add_argument("--d0-confirmed", action="store_true", help="Explicit flag confirming user has frozen D0")
    args = parser.parse_args()

    plan = build_d1_units_and_plan(
        runs_dir=args.runs_dir,
        diag_dir=args.diag_dir,
        p0a_catalog_path=args.p0a_catalog,
        n_per_arm=args.n_per_arm,
        r_b=args.r_b,
        target_per_stratum=args.target_per_stratum,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        f"[V3-D1 PLAN] Wrote {args.out}\n"
        f"  scanned_valid_episodes = {plan['source_dataset']['scanned_valid_episodes']}\n"
        f"  eligible_never_units   = {plan['stratified_sampling_frame']['total_eligible_never_units']}\n"
        f"  sampled_ladder_units   = {plan['stratified_sampling_frame']['total_sampled_units']}\n"
        f"  sampled_passing_harm   = {plan['r12_passing_harm_check_plan']['sampled_passing_unit_count']}\n"
        f"  total_planned_branches = {plan['budget_and_stop_conditions']['total_planned_branches']}\n"
        f"  max_actor_requests_cap = {plan['budget_and_stop_conditions']['max_actor_requests_cap']}\n"
        f"  est_wall_clock_hours   = {plan['budget_and_stop_conditions']['estimated_wall_clock_hours_upper_bound']} h"
    )

    if args.execute:
        if not args.d0_confirmed:
            sys.stderr.write(
                "BLOCKED [R18]: --execute requires --d0-confirmed after user freezes D0 parameters and authorizes budget.\n"
            )
            return 2
        sys.stderr.write(
            "BLOCKED [R18]: D0 is not yet frozen by user. Refusing to send Actor requests.\n"
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
