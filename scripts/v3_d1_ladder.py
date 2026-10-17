#!/usr/bin/env python3
"""V3 Stage D1 Entrypoint (R15, R18): 3D Stratified Intervention Ladder, Single-Item R Localization, and R12 Harm Check.

Contract (Notion V3 §3.4, §8.2, R7–R12, R15, R18):
- Builds the 3D stratified sampling frame (`exhausted_budget x valid_delivery x rule_source`) strictly from
  the new `Qwen/Qwen3.6-27B + pi` development cohort (`v3_qwen36_pi_dev45_20261017`), rejecting legacy DeepSeek runs.
- Constructs whole-task `explicit` and `implicit` requirement lists (`build_task_requirement_lists`, including
  splitting `ambiguous` rules into visible requirements for `R_explicit` and exact verifier tolerances for `R_implicit`)
  and nested ladder messages (`P`, `D`, `S`, `R_explicit`, `R_implicit`, `R_single`) with SHA-256 hashes,
  marking empty requirement levels as `ineligible` (never falling back to `D` or `S`).
- Preserves true checkpoint remaining budgets (`remaining_budget`) and per-unit `R_single` requirement messages.
- Includes a passing-sample harm check (`R12`: comparing `D`, `S`, `R_explicit` against `P` on passing items/runs).
- Defaults to `--plan-only` (`R18`): outputs `v3_d1_ladder_plan_20261017.json` with exact request cap,
  wall-clock estimate, and stop conditions, initiating ZERO model requests unless `--execute --d0-confirmed` is passed.
- Connects native cold resume and scoring with sealed receipts, failure retention and idempotent resume.
  Offline tests inject explicit adapters; simulation cannot fall back to native execution.
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
    adjudicate_e2_outcomes,
    build_interleaved_arm_schedule,
    build_ladder_intervention_messages,
    build_task_requirement_lists,
    build_v3_stratified_sampling_frame,
    compute_ipw_recoverability_ratio,
    evaluate_harm_on_passing_units,
    select_v05_intervention_anchors,
    sha256_text,
    verify_nested_ladder_messages,
)

from l2_sandbox.v3_execution import (Admission, approve, digest, exclusive, file_digest, identity, native_branch, read_json, seal, verify_branch, verify_seal, write_json)

DEFAULT_DEV45_RUNS_DIR = L1_ROOT / "runs" / "v3_qwen36_pi_dev45_20261017" / "episodes"
DEFAULT_DEV45_DIAG_DIR = L2_ROOT / "diagnoses" / "v3_qwen36_pi_dev45_20261017"
DEFAULT_P0A_CATALOG = L2_ROOT / "reports" / "v3_p0a_audit_20261017.json"
DEFAULT_PLAN_OUT = L2_ROOT / "reports" / "v3_d1_ladder_plan_20261017.json"
DEFAULT_OUTCOMES_OUT = L2_ROOT / "reports" / "v3_d1_ladder_outcomes_20261017.json"
DEFAULT_D0_FREEZE_FILE = L2_ROOT / "configs" / "v3_d0_freeze.json"
DEFAULT_RECEIPTS_DIR = L2_ROOT / "validation" / "v3_d1_receipts"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_checkpoint_remaining_budgets(ep_dir: Path, max_requests: int = 35) -> dict[int, dict[str, Any]]:
    """Read true `budget_used.model_requests` and `snapshot_id` from `checkpoints/cpNNNNNN.json`."""
    cp_dir = ep_dir / "checkpoints"
    info_by_k: dict[int, dict[str, Any]] = {}
    if not cp_dir.exists():
        return info_by_k
    for cp_file in sorted(cp_dir.glob("cp*.json")):
        cp_id = cp_file.stem
        k_val = int(cp_id[2:]) if cp_id[2:].isdigit() else 0
        try:
            cp_doc = _load_json(cp_file)
        except Exception:
            cp_doc = {}
        used_raw = (cp_doc.get("budget_used") or {}).get("model_requests")
        if type(used_raw) is not int:
            continue
        used_reqs = used_raw
        rem_reqs = max(0, max_requests - used_reqs)
        info_by_k[k_val] = {
            "checkpoint_id": cp_id,
            "k": k_val,
            "used_requests": used_reqs,
            "remaining_requests": rem_reqs,
            "snapshot_id": cp_doc.get("snapshot_id"),
            "cold_resume_eligible": (cp_doc.get("qualification") or {}).get("eligible") is True,
        }
    return info_by_k


def _enrich_cp_rows_with_package_budgets(
    cp_rows: list[dict[str, Any]],
    ep_dir: Path,
    max_requests: int = 35,
) -> list[dict[str, Any]]:
    cp_info = _read_checkpoint_remaining_budgets(ep_dir, max_requests=max_requests)
    enriched: list[dict[str, Any]] = []
    for r in cp_rows:
        if r.get("kind") != "checkpoint":
            enriched.append(r)
            continue
        k_val = int(r.get("k", 0))
        pkg_cp = cp_info.get(k_val)
        row_copy = dict(r)
        if pkg_cp is not None:
            row_copy["remaining_requests"] = int(pkg_cp["remaining_requests"])
            row_copy["cold_resume_eligible"] = bool(pkg_cp["cold_resume_eligible"])
            if not row_copy.get("snapshot_id") and pkg_cp.get("snapshot_id"):
                row_copy["snapshot_id"] = pkg_cp["snapshot_id"]
        else:
            row_copy["cold_resume_eligible"] = False
            row_copy.pop("remaining_requests", None)
        enriched.append(row_copy)
    return enriched


def build_d1_units_and_plan(
    *,
    runs_dir: Path,
    diag_dir: Path,
    p0a_catalog_path: Path,
    n_per_arm: int = 24,
    r_b: int = 6,
    max_requests: int = 35,
    target_per_stratum: int = 3,
    passing_harm_sample_size: int = 4,
    self_check_text: str = DEFAULT_SELF_CHECK_DRAFT,
) -> dict[str, Any]:
    p0a = _load_json(p0a_catalog_path) if p0a_catalog_path.exists() else {}
    family_path = L1_ROOT / "benchmarks/officeval/family_split.json"
    family_doc = read_json(family_path)
    families = {r["task_id"]: r["family_id"] for r in family_doc["tasks"]}
    if family_doc.get("status") != "frozen":
        raise ValueError("semantic family mapping must be frozen")

    units: list[DiagnosticUnitV05] = []
    passing_candidates: list[dict[str, Any]] = []
    task_ladder_specs: dict[str, Any] = {}
    task_req_cache: dict[str, dict[str, Any]] = {}
    scanned_episodes: list[str] = []
    readiness_errors: list[str] = []
    parent_bindings: dict[str, dict] = {}

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
            native_config = read_json(ep_dir / "raw/resolved_config.json")
            if native_config["model"].get("model") != "Qwen/Qwen3.6-27B" or native_config["agent"].get("harness") != "pi":
                readiness_errors.append(f"{ep_id}:actor_or_harness_identity_mismatch")
            if int(native_config["budget"]["max_model_requests"]) != max_requests:
                readiness_errors.append(f"{ep_id}:native_max_requests_differ_from_plan")
            termination = str((manifest.get("status") or {}).get("termination") or "")
            exhausted_budget = termination == "request_budget"

            dim1_pass = bool(raw_og.get("dim1_pass"))
            term_score = float(st_out.get("study_outcome_y") or 0.0)

            if short_task not in task_req_cache:
                req_lists = build_task_requirement_lists(p0a, short_task)
                task_req_cache[short_task] = req_lists
                ladder_msgs = build_ladder_intervention_messages(
                    explicit_requirements=req_lists["explicit"],
                    implicit_requirements=req_lists["implicit"],
                    self_check_text=self_check_text,
                )
                nest_verification = verify_nested_ladder_messages(ladder_msgs)
                task_ladder_specs[short_task] = {
                    "task_id": short_task,
                    "requirements_review": req_lists,
                    "explicit_count": len(req_lists["explicit"]),
                    "implicit_count": len(req_lists["implicit"]),
                    "ambiguous_count": len(req_lists["ambiguous"]),
                    "explicit_only_count": len(req_lists.get("explicit_only", [])),
                    "implicit_only_count": len(req_lists.get("implicit_only", [])),
                    "ladder_messages": ladder_msgs,
                    "nesting_check": nest_verification,
                }
            else:
                req_lists = task_req_cache[short_task]
                ladder_msgs = task_ladder_specs[short_task]["ladder_messages"]

            if not req_lists.get("review_complete"):
                readiness_errors.append(f"{short_task}:whole_task_requirements_not_reviewed")
            grades_path = diag_dir / ep_id / "grades.jsonl"
            cp_rows = _load_jsonl(grades_path)
            if cp_rows:
                expected_index = file_digest(ep_dir / "evidence_index.json")
                for grade_row in cp_rows:
                    if grade_row.get("episode_id") != manifest["episode_id"] or grade_row.get("episode_index_sha256") != expected_index:
                        readiness_errors.append(f"{ep_id}:snapshot_grades_parent_binding_mismatch")
                cp_rows = _enrich_cp_rows_with_package_budgets(cp_rows, ep_dir, max_requests=max_requests)
            else:
                readiness_errors.append(f"{ep_id}:missing_snapshot_grades")
                continue

            # Whole-run / peak anchor for passing runs
            peak_anchor_info = select_v05_intervention_anchors(
                cp_rows,
                max_requests=max_requests,
                r_b=r_b,
                termination=termination,
            )
            # Never-succeeded anchor (uses voluntary_pre_finish on assistant_finished or budget tail/first write)
            never_anchor_info = select_v05_intervention_anchors(
                cp_rows,
                max_requests=max_requests,
                r_b=r_b,
                termination=termination,
                e1_class="never_succeeded_on_disk",
            )

            if term_score >= 0.999 and peak_anchor_info["eligible"]:
                passing_candidates.append({
                    "unit_id": f"{ep_id}::PASSING_WHOLE_RUN",
                    "episode_id": ep_id,
                    "task_id": short_task,
                    "terminal_score_y": term_score,
                    "anchor_k": peak_anchor_info["primary_anchor_k"],
                    "anchor_type": peak_anchor_info["primary_anchor_type"],
                    "remaining_budget": peak_anchor_info["primary_remaining_budget"],
                    "arms": ["P", "D", "S"] + (["R_explicit"] if ladder_msgs["R_explicit"]["eligible"] else []),
                })

            if not dim1_pass:
                units.append(
                    DiagnosticUnitV05(
                        unit_id=f"{ep_id}::GATE_NO_VALID_DELIVERY",
                        episode_id=ep_id,
                        task_id=short_task,
                        benchmark="officeval",
                        family_id=families.get(short_task),
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
                            anchor_k=never_anchor_info["primary_anchor_k"],
                            anchor_type=never_anchor_info["primary_anchor_type"],
                            remaining_budget_at_anchor=never_anchor_info["primary_remaining_budget"],
                        ),
                        operational_label=None,
                        evidence_status="inconclusive" if never_anchor_info["eligible"] else "ineligible",
                        early_anchor_k=never_anchor_info.get("early_anchor_k"),
                        early_anchor_remaining_budget=never_anchor_info.get("early_remaining_budget"),
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
                        if max_d > 0 and peak_anchor_info["eligible"] and len(passing_candidates) < passing_harm_sample_size * 4:
                            passing_candidates.append({
                                "unit_id": f"{ep_id}::PASSING_ITEM::{idx}",
                                "episode_id": ep_id,
                                "task_id": short_task,
                                "rule": rule_txt,
                                "terminal_score_y": 1.0,
                                "anchor_k": peak_anchor_info["primary_anchor_k"],
                                "anchor_type": peak_anchor_info["primary_anchor_type"],
                                "remaining_budget": peak_anchor_info["primary_remaining_budget"],
                                "arms": ["P", "D", "S"] + (["R_explicit"] if ladder_msgs["R_explicit"]["eligible"] else []),
                            })
                        continue
                    terminal_item = max(0.0, min(1.0, act_d / max_d if max_d > 0 else 1.0 + act_d / abs(max_d))) if max_d else 0.0
                    observed = [(int(row["k"]), 1.0 - float(row["deductions"][rule_txt]) / (abs(max_d) / max_score)) for row in cp_rows if row.get("kind") == "checkpoint" and row.get("dim1_pass") is True and rule_txt in (row.get("deductions") or {}) and max_d]
                    prior_correct = [k for k, score in observed if score >= .999]
                    r_src = (req_lists.get("rule_sources") or {}).get(rule_txt, "ambiguous")
                    if isinstance(r_src, dict):
                        r_src = r_src.get("source_category", "ambiguous")
                    if rule_txt not in (req_lists.get("rule_sources") or {}):
                        readiness_errors.append(f"{short_task}:unreviewed_rule:{rule_txt}")
                    units.append(
                        DiagnosticUnitV05(
                            unit_id=f"{ep_id}::ITEM::{idx}",
                            episode_id=ep_id,
                            task_id=short_task,
                            benchmark="officeval",
                            family_id=families.get(short_task),
                            unit_level="rubric_item",
                            defect_name=rule_txt,
                            deduction_weight=round(ded, 6),
                            rule_source=r_src,
                            e1_observation=E1Observation(
                                e1_class="broken" if prior_correct else "never_succeeded_on_disk",
                                candidate_substate="delivered_item_wrong",
                                k_star=never_anchor_info["primary_anchor_k"] or 0,
                                max_item_score=max([terminal_item, *[score for _, score in observed]]),
                                terminal_item_score=terminal_item,
                                deduction=round(ded, 6),
                                first_candidate_write_turn=never_anchor_info.get("first_candidate_write_k"),
                            ),
                            e2_response_vector=E2ResponseVector(
                                evaluated=False,
                                anchor_k=never_anchor_info["primary_anchor_k"],
                                anchor_type=never_anchor_info["primary_anchor_type"],
                                remaining_budget_at_anchor=never_anchor_info["primary_remaining_budget"],
                            ),
                            operational_label=None,
                            evidence_status="inconclusive" if never_anchor_info["eligible"] else "ineligible",
                            early_anchor_k=never_anchor_info.get("early_anchor_k"),
                            early_anchor_remaining_budget=never_anchor_info.get("early_remaining_budget"),
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
    if len(chosen_passing) < passing_harm_sample_size:
        readiness_errors.append("insufficient_passing_harm_sample")

    ladder_branch_count = 0
    estimated_actor_requests = 0
    unit_schedules: list[dict[str, Any]] = []

    units_by_id = {u.unit_id: u for u in units}
    for s_unit in frame["sampled_units"]:
        t_id = s_unit["task_id"]
        req_lists = task_req_cache.get(t_id) or build_task_requirement_lists(p0a, t_id)
        single_req = (
            str(s_unit.get("defect_name") or "").strip()
            if s_unit["rule_source"] != "gate_or_no_artifact"
            else None
        )
        u_msgs = build_ladder_intervention_messages(
            explicit_requirements=req_lists["explicit"],
            implicit_requirements=req_lists["implicit"],
            single_rule_requirement=single_req,
            self_check_text=self_check_text,
        )
        u_nest = verify_nested_ladder_messages(u_msgs)
        arms = ["P", "D", "S"]
        for cand_arm in ("R_explicit", "R_implicit", "R_single"):
            if (u_msgs.get(cand_arm) or {}).get("eligible"):
                arms.append(cand_arm)

        rem_b = int(s_unit["remaining_budget"])
        ank = int(s_unit["anchor_k"])
        binding_key = f"{s_unit['episode_id']}:{ank}"
        try:
            parent_bindings[binding_key] = identity(runs_dir / s_unit["episode_id"], ank)
        except (ValueError, OSError, KeyError) as exc:
            readiness_errors.append(f"{binding_key}:parent_identity:{exc}")
        sched = build_interleaved_arm_schedule(arms, n_per_arm, anchor_k=ank, remaining_budget=rem_b)
        ladder_branch_count += len(sched)
        estimated_actor_requests += len(sched) * rem_b
        unit_schedules.append({
            "family_id": families.get(t_id),
            "parent_identity": parent_bindings.get(binding_key),
            "unit_id": s_unit["unit_id"],
            "episode_id": s_unit["episode_id"],
            "task_id": t_id,
            "unit_level": s_unit.get("unit_level", "rubric_item"),
            "defect_name": s_unit.get("defect_name", ""),
            "rule_source": s_unit["rule_source"],
            "deduction_weight": s_unit["deduction_weight"],
            "terminal_y": units_by_id[s_unit["unit_id"]].e1_observation.terminal_item_score,
            "e1_observation": units_by_id[s_unit["unit_id"]].e1_observation.__dict__,
            "stratum_id": s_unit["stratum_id"],
            "inclusion_prob": s_unit["inclusion_prob"],
            "anchor_k": ank,
            "anchor_type": s_unit.get("anchor_type") or "voluntary_pre_finish",
            "remaining_budget": rem_b,
            "eligible_arms": arms,
            "deployable_arms": [a for a in arms if a in ("P","D","S","R_explicit") or (a == "R_single" and s_unit["rule_source"] == "explicit")],
            "ineligible_arms": [
                a for a in ("R_explicit", "R_implicit", "R_single")
                if not (u_msgs.get(a) or {}).get("eligible")
            ],
            "arm_messages": u_msgs,
            "nesting_check": u_nest,
            "schedule": sched,
            "planned_branches": len(sched),
            "max_requests_cap": len(sched) * rem_b,
        })

    harm_branches = sum(len(p["arms"]) * n_per_arm for p in chosen_passing)
    harm_requests = sum(len(p["arms"]) * n_per_arm * int(p.get("remaining_budget") or r_b) for p in chosen_passing)

    total_planned_branches = ladder_branch_count + harm_branches
    total_max_requests = estimated_actor_requests + harm_requests
    measured_req_per_hour = 352.3

    plan_payload = {
        "schema_version": V05_SCHEMA_VERSION,
        "stage": "D1",
        "mode": "plan_only",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Development-set descriptive planning artifact (R15, R18). "
            "Does NOT initiate any Actor or Analyst requests until D0 is frozen and user confirms --execute --d0-confirmed."
        ),
        "family_mapping": {"path": str(family_path), "sha256": file_digest(family_path), "semantics": family_doc.get("group_semantics")},
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
            "ceiling_threshold": 0.05,
            "min_replicates_per_arm": 2,
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
    # Freeze the complete scientific and execution contract, including passing samples.
    for p_unit in chosen_passing:
        p_unit["family_id"] = families.get(p_unit["task_id"])
        p_unit["unit_level"] = "rubric_item" if p_unit.get("rule") else "whole_run"
        p_unit["defect_name"] = p_unit.get("rule", "whole_run")
        try:
            p_unit["parent_identity"] = identity(runs_dir / p_unit["episode_id"], int(p_unit["anchor_k"]))
        except (ValueError, OSError, KeyError) as exc:
            readiness_errors.append(f"{p_unit['unit_id']}:parent_identity:{exc}")
    comparisons = []
    levels = {"P": 0, "D": 1, "S": 1, "R_explicit": 2, "R_implicit": 3}
    for unit in unit_schedules:
        unit["comparison_ids"] = [f"{a}_vs_{b}" for a in unit["eligible_arms"] if a in levels and levels[a] > 0 for b in unit["eligible_arms"] if b in levels and levels[b] < levels[a]]
        if "S" in unit["eligible_arms"] and "D" in unit["eligible_arms"]:
            unit["comparison_ids"].append("S_vs_D")
        if "R_single" in unit["eligible_arms"]:
            unit["localization_comparison_ids"] = [f"R_single_vs_{a}" for a in ("P", "D") if a in unit["eligible_arms"]]
            unit["comparison_ids"].extend(unit["localization_comparison_ids"])
        comparisons.extend(f"{unit['unit_id']}::{c}" for c in unit["comparison_ids"])
    plan_payload["primary_test_family"] = {"comparison_ids": comparisons, "size": len(comparisons)}
    plan_payload["requirements_catalog_sha256"] = digest(p0a)
    for unit in [*unit_schedules, *chosen_passing]:
        if not unit.get("family_id"):
            readiness_errors.append(f"{unit['task_id']}:missing_semantic_family")
    plan_payload["readiness_errors"] = sorted(set(readiness_errors))
    plan_payload["execution_ready"] = bool(unit_schedules and not readiness_errors)
    return seal(plan_payload, "plan_sha256")


def _execute_d1_locked(plan, *, runs_dir, receipts_dir, outcomes_out, freeze, branch_runner=None, simulate_offline=False):
    if simulate_offline and branch_runner is None:
        raise ValueError('offline simulation requires an explicit injected adapter; native execution is forbidden')
    caps = approve(plan, freeze)
    if caps['max_requests'] < plan['budget_and_stop_conditions']['max_actor_requests_cap']:
        raise ValueError('approved request cap does not cover the frozen branch schedule')
    ledger = Admission(receipts_dir / 'admission.json', plan['plan_sha256'], caps)
    params = plan['d0_candidate_parameters']
    runner = branch_runner or (lambda u, b, m: native_branch(u, b, m, runs_dir=runs_dir, output=receipts_dir / 'branches' / digest([u['unit_id'], b])[:24], limiter=ledger))
    executed, cached = 0, 0
    failures = []
    all_rows, passing_rows = [], []
    schedules = [(u, False) for u in plan['unit_schedules']]
    for passing in plan['r12_passing_harm_check_plan']['sampled_passing_units']:
        u = dict(passing, eligible_arms=passing['arms'], arm_messages=plan['task_ladder_message_specs'][passing['task_id']]['ladder_messages'])
        u['schedule'] = build_interleaved_arm_schedule(u['arms'], params['n_per_arm'], anchor_k=u['anchor_k'], remaining_budget=u['remaining_budget'])
        schedules.append((u, True))
    for unit, is_passing in schedules:
        arm_outcomes = {a: [] for a in unit['eligible_arms']}
        receipts = {a: [] for a in unit['eligible_arms']}
        nest = verify_nested_ladder_messages(unit['arm_messages'])
        if not nest['valid']:
            raise ValueError(f'invalid frozen ladder: {nest}')
        for branch in unit['schedule']:
            arm = branch['arm']
            if branch['checkpoint_k'] != unit['anchor_k'] or branch['remaining_budget'] != unit['remaining_budget']:
                raise ValueError('same-state/same-budget schedule invariant violated')
            msg = unit['arm_messages'][arm]
            if not msg['eligible']:
                raise ValueError('ineligible arm in schedule')
            binding = dict(plan_sha256=plan['plan_sha256'], unit_id=unit['unit_id'], arm=arm, replicate=branch['replicate'], anchor_k=unit['anchor_k'], remaining_budget=unit['remaining_budget'], message_sha256=msg['message_sha256'], parent_identity=unit['parent_identity'], endpoint=dict(unit_level=unit.get('unit_level','rubric_item'), defect_name=unit.get('defect_name', unit.get('rule',''))))
            branch_id = digest(binding)
            binding['branch_id'] = branch_id
            path = receipts_dir / (branch_id + '.json')
            if path.exists():
                receipt = read_json(path)
                verify_branch(receipt, expected=binding)
                if bool(receipt.get('simulated_offline')) != simulate_offline:
                    raise ValueError('simulation/production receipt mismatch')
                cached += 1
            else:
                ledger.check()
                # Persist before the actor. A crash leaves an inflight receipt that cannot retry.
                write_json(path, seal(dict(binding, status='inflight', valid_attempt=False)))
                try:
                    result = runner(unit, branch, msg)
                    if result.get('valid_attempt') is not True or result.get('status') != 'completed':
                        raise ValueError('adapter did not return completed valid evidence')
                    if bool(result.get('simulated_offline')) != simulate_offline:
                        raise ValueError('adapter execution mode mismatch')
                    receipt = seal(dict(result, **binding))
                    write_json(path, receipt)
                    verify_branch(receipt, expected=binding)
                    executed += 1
                except Exception as exc:
                    ledger.invalid()
                    receipt = seal(dict(binding, status='failed', valid_attempt=False, error_type=type(exc).__name__, error=str(exc), simulated_offline=simulate_offline))
                    write_json(path, receipt)
                    failures.append(receipt)
                    write_json(outcomes_out, seal(dict(stage='D1', status='failed', plan_sha256=plan['plan_sha256'], simulated_offline=simulate_offline, failed_attempts=failures, evaluated_units=all_rows, passing_units=passing_rows)))
                    raise
            score = float(receipt['outcome_score'])
            if not 0 <= score <= 1:
                raise ValueError('invalid endpoint score')
            receipts[arm].append(receipt)
            arm_outcomes[arm].append(score)
        row = {k: v for k, v in unit.items() if k not in ('schedule', 'arm_messages')}
        row.update(arm_outcomes=arm_outcomes, branch_receipts=receipts, terminal_y=1.0 if is_passing else unit.get('terminal_y', 0.0))
        if is_passing:
            passing_rows.append(row)
        else:
            vec, label, status = adjudicate_e2_outcomes(unit.get('terminal_y', 0.0), arm_outcomes, anchor_k=unit['anchor_k'], anchor_type=unit['anchor_type'], remaining_budget=unit['remaining_budget'], delta=params['delta'], tau=params['tau'], alpha=params['alpha'], ineligible_arms=unit.get('ineligible_arms', ()), primary_family_size=plan['primary_test_family']['size'], comparison_ids=unit['comparison_ids'])
            row.update(e2_response_vector=vec.__dict__, operational_label=label, evidence_status=status)
            all_rows.append(row)
    ipw_rows = [dict(unit_id=r['unit_id'], family_id=r['family_id'], task_id=r['task_id'], stratum_id=r['stratum_id'], deduction=r['deduction_weight'], inclusion_prob=r['inclusion_prob'], recovered_by_P=r['evidence_status']=='plain_recoverable', recovered_by_D=r['operational_label']=='delivery_responsive', recovered_by_S=r['operational_label']=='self_check_responsive', recovered_by_R_explicit=r['operational_label']=='explicit_rule_responsive', recovered_by_R_implicit=r['operational_label']=='implicit_rule_responsive', inconclusive=r['evidence_status']=='inconclusive') for r in all_rows]
    payload = seal(dict(stage='D1', status='completed', mode='simulation' if simulate_offline else 'executed', simulated_offline=simulate_offline, plan_sha256=plan['plan_sha256'], d1_plan_sha256=plan['plan_sha256'], d0_parameters=params, primary_test_family=plan['primary_test_family'], execution_summary=dict(evaluated_unit_count=len(all_rows), total_branches_executed=executed, reused_cached_branches=cached, admission=ledger.state), evaluated_units=all_rows, passing_units=passing_rows, ipw_recoverability_summary=compute_ipw_recoverability_ratio(ipw_rows), r12_passing_harm_evaluation=evaluate_harm_on_passing_units(passing_rows, delta=params['delta'], alpha=params['alpha'])))
    write_json(outcomes_out, payload)
    return payload


def execute_d1_plan(plan, *, runs_dir, receipts_dir, outcomes_out, freeze=None, branch_runner=None, simulate_offline=False):
    with exclusive(receipts_dir / '.stage.lock'):
        return _execute_d1_locked(plan, runs_dir=runs_dir, receipts_dir=receipts_dir, outcomes_out=outcomes_out, freeze=freeze, branch_runner=branch_runner, simulate_offline=simulate_offline)


def main() -> int:
    parser = argparse.ArgumentParser(description="V3 Stage D1 Ladder & R12 Harm Check (default: --plan-only)")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_DEV45_RUNS_DIR)
    parser.add_argument("--diag-dir", type=Path, default=DEFAULT_DEV45_DIAG_DIR)
    parser.add_argument("--p0a-catalog", type=Path, default=DEFAULT_P0A_CATALOG)
    parser.add_argument("--out", type=Path, default=DEFAULT_PLAN_OUT)
    parser.add_argument("--outcomes-out", type=Path, default=DEFAULT_OUTCOMES_OUT)
    parser.add_argument("--receipts-dir", type=Path, default=DEFAULT_RECEIPTS_DIR)
    parser.add_argument("--approved-plan", type=Path, help="Execute this existing immutable plan; never rebuild on execute")
    parser.add_argument("--d0-freeze-file", type=Path, default=DEFAULT_D0_FREEZE_FILE)
    parser.add_argument("--n-per-arm", type=int, default=24)
    parser.add_argument("--r-b", type=int, default=6)
    parser.add_argument("--target-per-stratum", type=int, default=3)
    parser.add_argument("--plan-only", action="store_true", help="Generate plan JSON without running requests (default)")
    parser.add_argument("--execute", action="store_true", help="Execute planned branches (requires --d0-confirmed)")
    parser.add_argument("--d0-confirmed", action="store_true", help="Explicit flag confirming user has frozen D0")
    parser.add_argument("--simulate-offline", action="store_true", help="Reserved for injected test adapters; CLI refuses native fallback")
    args = parser.parse_args()
    if args.plan_only and args.execute:
        parser.error("choose --plan-only or --execute")

    if args.execute and args.approved_plan is None:
        sys.stderr.write("BLOCKED: --execute requires --approved-plan and exact frozen approval.\n")
        return 2
    plan = read_json(args.approved_plan) if args.execute else build_d1_units_and_plan(
        runs_dir=args.runs_dir,
        diag_dir=args.diag_dir,
        p0a_catalog_path=args.p0a_catalog,
        n_per_arm=args.n_per_arm,
        r_b=args.r_b,
        target_per_stratum=args.target_per_stratum,
    )
    if not args.execute:
        write_json(args.out, plan)

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
        if not args.simulate_offline:
            if not args.d0_freeze_file.is_file():
                sys.stderr.write(
                    f"BLOCKED [R18]: Missing frozen D0 file {args.d0_freeze_file} (pass --d0-freeze-file or --simulate-offline).\n"
                )
                return 2
            freeze_doc = _load_json(args.d0_freeze_file)
            if not freeze_doc.get("d0_frozen"):
                sys.stderr.write("BLOCKED [R18]: D0 freeze file does not have d0_frozen=true.\n")
                return 2
        if not plan.get("unit_schedules"):
            sys.stderr.write("BLOCKED [R15]: Plan has 0 sampled unit schedules to execute.\n")
            return 2
        try:
            outcomes = execute_d1_plan(
                plan,
                runs_dir=args.runs_dir,
                receipts_dir=args.receipts_dir,
                outcomes_out=args.outcomes_out,
                simulate_offline=args.simulate_offline,
                freeze=_load_json(args.d0_freeze_file),
            )
        except Exception as exc:
            sys.stderr.write(f"FAILED [R15]: D1 execution error: {exc}\n")
            return 2
        print(
            f"[V3-D1 EXECUTE] Wrote {args.outcomes_out}\n"
            f"  evaluated_unit_count     = {outcomes['execution_summary']['evaluated_unit_count']}\n"
            f"  total_branches_executed  = {outcomes['execution_summary']['total_branches_executed']}\n"
            f"  reused_cached_branches   = {outcomes['execution_summary']['reused_cached_branches']}\n"
            f"  level1_generic_share_lb  = {outcomes['ipw_recoverability_summary'].get('level1_generic_share_lower_bound')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
