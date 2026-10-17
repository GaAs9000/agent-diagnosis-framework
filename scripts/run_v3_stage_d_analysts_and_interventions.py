#!/usr/bin/env python3
"""V3 Stage D Phase 2: Two-Level DiagnosticUnitV05 Census, Pre-Sealed Analyst M/A/B/C
Predictions, Stratified Interleaved P/D/R (n=24) Counterfactual Continuations, and
V3-P1..P3 Adjudication on Qwen/Qwen3.6-27B + pi-coding-agent@0.73.1.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
L1_ROOT = REPO_ROOT / "L1_sandbox" / "C3_Experimental_Sandbox"
L2_ROOT = REPO_ROOT / "L2_Behavioral Event Localization" / "L2_Sandbox"
sys.path.insert(0, str(L1_ROOT / "src"))
sys.path.insert(0, str(L2_ROOT / "src"))

import copy
from c3_sandbox.l1.core.orchestrator import run as l1_run
from c3_sandbox.l1.evaluation.study import study_outcome
from c3_sandbox.l1.evidence.live import verify as verify_episode_dir

from l2_sandbox.config import DELIVER_REMINDER
from l2_sandbox.defects import build_defect_table as build_defect_report
from l2_sandbox.evidence.l1 import L1Package as load_episode
from l2_sandbox.facts.build import build_facts
from l2_sandbox.grades import grade_episode_snapshots as grade_episode
from l2_sandbox.io import digest, read_json, read_jsonl, write_json as atomic_write_json, write_jsonl
from l2_sandbox.verify import _extract_item_scores
from l2_sandbox.v05 import (
    DiagnosticUnitV05,
    E1Observation,
    E2ResponseVector,
    adjudicate_e2_outcomes,
    build_interleaved_arm_schedule,
    build_r_all_intervention_message,
    compute_ipw_recoverability_ratio,
    evaluate_analysts_v05,
    predict_majority_baseline_m,
    predict_rule_analyst_a_v05,
    select_v05_intervention_anchors,
    verify_rule_prompt_non_leakage,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
L1_ROOT = REPO_ROOT / "L1_sandbox" / "C3_Experimental_Sandbox"
L2_ROOT = REPO_ROOT / "L2_Behavioral Event Localization" / "L2_Sandbox"

FAMILY_SPLIT_PATH = L2_ROOT / "configs" / "family_split.json"
RUNS_ROOT = L1_ROOT / "runs" / "v3_qwen36_pi_dev45_20261017"
DIAG_ROOT = L2_ROOT / "diagnoses" / "v3_qwen36_pi_dev45_20261017"
P0A_AUDIT_JSON = L2_ROOT / "reports" / "v3_p0a_audit_20261017.json"
OUT_JSON = L2_ROOT / "reports" / "v3_stage_d_interventions_and_p1_p3_20261017.json"
OUT_MD = L2_ROOT / "reports" / "V3_STAGE_D_FULL_REPORT_20261017.md"


def _load_family_map() -> dict[str, str]:
    if not FAMILY_SPLIT_PATH.exists():
        return {}
    data = read_json(FAMILY_SPLIT_PATH)
    fam_map: dict[str, str] = {}
    for fam in data.get("families", []):
        fid = fam.get("family_id", "")
        for tid in fam.get("tasks", []):
            num = str(tid).replace("officeval_", "").replace("officeval:", "")
            fam_map[f"officeval_{num}"] = fid
            fam_map[f"officeval:{num}"] = fid
    return fam_map


def _load_rule_source_lookup() -> dict[tuple[str, str], dict[str, Any]]:
    """Load pre-audited rule_source from reports/v3_p0a_audit_20261017.json."""
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    if not P0A_AUDIT_JSON.exists():
        return lookup
    data = read_json(P0A_AUDIT_JSON)
    task_rule_audit = (
        data.get("dev45", {})
        .get("delivered_item_wrong_rubric_source", {})
        .get("task_rule_audit", {})
    )
    for task_key, rules_dict in task_rule_audit.items():
        t_num = str(task_key).replace("officeval:officeval_", "").replace("officeval_", "").replace("officeval:", "")
        if isinstance(rules_dict, dict):
            for rule_text, info in rules_dict.items():
                cat = str(info.get("source_category") or "explicit")
                lookup[(t_num, str(rule_text))] = {
                    "rule_source": cat,
                    "criterion_text": str(rule_text),
                    "rationale": str(info.get("rationale") or ""),
                }
    return lookup


def _resolve_episode_dir(slot_dir: Path) -> Path | None:
    if not slot_dir.exists():
        return None
    if (slot_dir / "manifest.json").is_file():
        return slot_dir
    ep_dirs = sorted(p for p in slot_dir.iterdir() if p.is_dir() and p.name.startswith("ep_"))
    return ep_dirs[-1] if ep_dirs else None


def _list_dev45_episodes(dev45_dir: Path) -> list[tuple[str, Path]]:
    eps_root = dev45_dir / "episodes" if (dev45_dir / "episodes").is_dir() else dev45_dir
    if not eps_root.exists():
        return []
    out: list[tuple[str, Path]] = []
    for p in sorted(eps_root.iterdir()):
        if not p.is_dir() or not p.name.startswith("officeval_"):
            continue
        ep_dir = _resolve_episode_dir(p)
        if ep_dir is not None and verify_episode_dir(ep_dir).get("verified"):
            out.append((p.name, ep_dir))
    return out


def _extract_task_explicit_requirements(
    task_num: str,
    ep_dir: Path,
    lookup: dict[tuple[str, str], dict[str, Any]],
) -> list[str]:
    """Collect explicit user-facing requirements from public_task.json & explicit P0a rules without leaking verifier code."""
    reqs: list[str] = []
    pub_path = ep_dir / "benchmark" / "public_task.json"
    if pub_path.is_file():
        pub = read_json(pub_path)
        prompt = str(pub.get("prompt") or pub.get("instruction") or "")
        for line in prompt.splitlines():
            s = line.strip(" -•\t")
            if len(s) >= 6 and verify_rule_prompt_non_leakage(s):
                reqs.append(s)
    for (tn, _), info in sorted(lookup.items()):
        if tn == task_num and info.get("rule_source") == "explicit":
            crit = str(info.get("criterion_text") or "").strip()
            if crit and crit not in reqs and verify_rule_prompt_non_leakage(crit):
                reqs.append(crit)
    if not reqs:
        reqs.append(f"请严格核对任务要求，确保在 /workspace/submission 中生成完整合规的目标文档。")
    return reqs[:6]


def build_two_level_units_for_dev45(
    dev45_dir: Path,
    diag_dir: Path,
    fam_map: dict[str, str],
    rule_lookup: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[DiagnosticUnitV05], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Construct the strict two-level DiagnosticUnitV05 population across all completed dev episodes."""
    units: list[DiagnosticUnitV05] = []
    cp_rows_by_ep: dict[str, list[dict[str, Any]]] = {}
    gate_loss_sum = 0.0
    item_loss_sum = 0.0
    total_loss_sum = 0.0

    for slot_name, ep_dir in _list_dev45_episodes(dev45_dir):
        task_num = slot_name.split(".")[0].replace("officeval_", "")
        out_diag = diag_dir / slot_name
        out_diag.mkdir(parents=True, exist_ok=True)
        grades_file = out_diag / "grades.jsonl"
        defects_file = out_diag / "defects.json"

        pkg = load_episode(ep_dir)
        if not grades_file.exists():
            rows = grade_episode(pkg, cache_dir=dev45_dir / "grade_cache", verify_terminal_submission=True)
            write_jsonl(grades_file, rows)
        else:
            rows = read_jsonl(grades_file)

        if not defects_file.exists():
            facts = build_facts(pkg)
            def_rep = build_defect_report(pkg, rows, facts, grades_path=grades_file)
            atomic_write_json(defects_file, def_rep)
        else:
            def_rep = read_json(defects_file)

        cp_rows_by_ep[slot_name] = rows
        y_term = float(
            def_rep.get("terminal_score_y")
            if def_rep.get("terminal_score_y") is not None
            else def_rep.get("terminal_score") or 0.0
        )
        run_loss = max(0.0, 1.0 - y_term)
        total_loss_sum += run_loss
        if run_loss <= 1e-9:
            continue

        term_row = next((r for r in rows if r.get("kind") in ("terminal", "terminal_submission")), {})
        cp_rows = [r for r in rows if r.get("kind") == "checkpoint"]
        dim1_ok = bool(term_row.get("dim1_pass"))
        cand_cps = [r for r in cp_rows if r.get("candidate_status") == "ok"]
        first_write_k = int(cand_cps[0]["k"]) if cand_cps else None

        if not dim1_ok:
            # Level-1 Gate / No-Artifact Unit (weight = run_loss, zero overlap with Level-2 items)
            gate_loss_sum += run_loss
            max_s = max((float(r.get("score") or 0.0) for r in cp_rows), default=0.0)
            k_star = next(
                (int(r["k"]) for r in cp_rows if float(r.get("score") or 0.0) >= max_s - 1e-9 and max_s > 0),
                0,
            )
            if max_s > y_term + 1e-9:
                defects_list = def_rep.get("defects", [])
                e1_cls = "broken" if any(d.get("kind") == "broken" for d in defects_list) else "undelivered"
                substate = "peak_regressed" if e1_cls == "broken" else "workspace_unsubmitted_positive"
                op_lbl = e1_cls
                ev_st = "adjudicated"
                src_turn = next(
                    (d.get("turn") for d in defects_list if d.get("turn") is not None),
                    None,
                )
            else:
                e1_cls = "never_succeeded_on_disk"
                if not cand_cps:
                    substate = "no_candidate"
                elif bool(term_row.get("in_submission")) or term_row.get("candidate_status") == "ok":
                    substate = "submitted_dim1_failed"
                else:
                    substate = "written_score0_candidate"
                op_lbl = None
                ev_st = "ineligible"
                src_turn = None

            anchors = select_v05_intervention_anchors(cp_rows, max_requests=35, r_b=6)
            if e1_cls == "never_succeeded_on_disk" and anchors["eligible"]:
                ev_st = "inconclusive"

            u = DiagnosticUnitV05(
                unit_id=f"{slot_name}:GATE_DIM1",
                episode_id=slot_name,
                task_id=f"officeval:{task_num}",
                benchmark="officeval",
                unit_level="gate_or_no_artifact",
                defect_name="dim1_valid_delivery_gate",
                deduction_weight=round(run_loss, 6),
                rule_source="gate_or_no_artifact",
                e1_observation=E1Observation(
                    e1_class=e1_cls,
                    candidate_substate=substate,
                    k_star=k_star,
                    max_item_score=round(max_s, 6),
                    terminal_item_score=round(y_term, 6),
                    deduction=round(run_loss, 6),
                    source_turn=int(src_turn) if src_turn is not None else None,
                    first_candidate_write_turn=first_write_k,
                ),
                e2_response_vector=E2ResponseVector(
                    evaluated=False,
                    anchor_k=anchors["primary_anchor_k"],
                    anchor_type=anchors["primary_anchor_type"],
                    remaining_budget_at_anchor=anchors["primary_remaining_budget"],
                ),
                operational_label=op_lbl,
                evidence_status=ev_st,
                early_anchor_k=anchors["early_anchor_k"],
                early_anchor_remaining_budget=anchors["early_remaining_budget"],
            )
            units.append(u)
        else:
            # Level-2 Rubric Item Units for dim1_pass == True runs
            for d in def_rep.get("defects", []):
                iid = str(d.get("defect") or d.get("item_id") or "")
                ded = float(d.get("deduction") or 0.0)
                if ded <= 1e-9:
                    continue
                item_loss_sum += ded
                kind = str(d.get("kind", "never"))
                src_turn = d.get("turn") if d.get("turn") is not None else d.get("source_turn")
                ok_ks = [
                    int(r_k["k"])
                    for r_k in cp_rows
                    if r_k.get("candidate_status") == "ok"
                    and iid in (r_k.get("deductions") or {})
                    and float(r_k["deductions"][iid]) <= 1e-12
                ]
                k_star = max(ok_ks) if ok_ks else 0

                r_info = rule_lookup.get((task_num, iid), {})
                raw_rs = str(r_info.get("rule_source") or "ambiguous")
                rs = "implicit" if raw_rs == "implicit_grader_only" else raw_rs
                if rs not in ("explicit", "implicit", "ambiguous"):
                    rs = "ambiguous"

                if kind in ("broken", "undelivered"):
                    e1_cls = kind
                    substate = "peak_regressed" if kind == "broken" else "workspace_unsubmitted_positive"
                    op_lbl = kind
                    ev_st = "adjudicated"
                else:
                    e1_cls = "never_succeeded_on_disk"
                    substate = "delivered_item_wrong"
                    op_lbl = None
                    ev_st = "ineligible"

                anchors = select_v05_intervention_anchors(cp_rows, max_requests=35, r_b=6, item_id=iid)
                if e1_cls == "never_succeeded_on_disk" and anchors["eligible"]:
                    ev_st = "inconclusive"

                u = DiagnosticUnitV05(
                    unit_id=f"{slot_name}:{iid}",
                    episode_id=slot_name,
                    task_id=f"officeval:{task_num}",
                    benchmark="officeval",
                    unit_level="rubric_item",
                    defect_name=str(r_info.get("criterion_text") or iid),
                    deduction_weight=round(ded, 6),
                    rule_source=rs,
                    e1_observation=E1Observation(
                        e1_class=e1_cls,
                        candidate_substate=substate,
                        k_star=k_star,
                        max_item_score=1.0 if k_star > 0 else 0.0,
                        terminal_item_score=0.0,
                        deduction=round(ded, 6),
                        source_turn=int(src_turn) if src_turn is not None else None,
                        first_candidate_write_turn=first_write_k,
                    ),
                    e2_response_vector=E2ResponseVector(
                        evaluated=False,
                        anchor_k=anchors["primary_anchor_k"],
                        anchor_type=anchors["primary_anchor_type"],
                        remaining_budget_at_anchor=anchors["primary_remaining_budget"],
                    ),
                    operational_label=op_lbl,
                    evidence_status=ev_st,
                    early_anchor_k=anchors["early_anchor_k"],
                    early_anchor_remaining_budget=anchors["early_remaining_budget"],
                )
                units.append(u)

    accounting = {
        "total_run_loss": round(total_loss_sum, 6),
        "level1_gate_loss": round(gate_loss_sum, 6),
        "level2_item_loss": round(item_loss_sum, 6),
        "reconstructed_sum": round(gate_loss_sum + item_loss_sum, 6),
        "zero_double_counting_verified": abs((gate_loss_sum + item_loss_sum) - total_loss_sum) < 1e-4,
        "total_units": len(units),
        "level1_unit_count": sum(1 for u in units if u.unit_level == "gate_or_no_artifact"),
        "level2_unit_count": sum(1 for u in units if u.unit_level == "rubric_item"),
        "e1_broken_loss": round(sum(u.deduction_weight for u in units if u.e1_observation.e1_class == "broken"), 6),
        "e1_undelivered_loss": round(sum(u.deduction_weight for u in units if u.e1_observation.e1_class == "undelivered"), 6),
        "e1_never_loss": round(sum(u.deduction_weight for u in units if u.e1_observation.e1_class == "never_succeeded_on_disk"), 6),
    }
    return units, cp_rows_by_ep, accounting


def seal_analyst_predictions(
    units: list[DiagnosticUnitV05],
    dev45_dir: Path,
    diag_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    """Generate and seal Analyst M, A, B, and C predictions BEFORE running E2 interventions."""
    m_preds = predict_majority_baseline_m(units, default_majority_label="unrecovered_under_protocol")
    a_preds: list[dict[str, Any]] = []
    b_preds: list[dict[str, Any]] = []
    c_preds: list[dict[str, Any]] = []

    ep_map = dict(_list_dev45_episodes(dev45_dir))
    facts_by_ep: dict[str, list[dict[str, Any]]] = {}
    journal_by_ep: dict[str, list[dict[str, Any]]] = {}
    for u in units:
        ep_name = u.episode_id
        if ep_name not in facts_by_ep:
            ep_dir = ep_map.get(ep_name)
            if ep_dir is not None:
                ep_rec = load_episode(ep_dir)
                facts_by_ep[ep_name] = build_facts(ep_rec)
                j_path = ep_dir / "trajectory.jsonl"
                if not j_path.exists():
                    j_path = ep_dir / "journal.jsonl"
                journal_by_ep[ep_name] = read_jsonl(j_path) if j_path.exists() else []
            else:
                facts_by_ep[ep_name] = []
                journal_by_ep[ep_name] = []

        facts = facts_by_ep[ep_name]
        journal = journal_by_ep[ep_name]
        a_pred = predict_rule_analyst_a_v05(u, facts)
        a_preds.append(a_pred)

        # Group B: Raw log analyst (sees raw trajectory tool calls & assistant text, without structured facts/contrast)
        raw_text_joined = " ".join(json.dumps(ev, ensure_ascii=False) for ev in journal[-12:])
        wrote_workspace_file = any(
            ".docx" in json.dumps(ev, ensure_ascii=False)
            or ".xlsx" in json.dumps(ev, ensure_ascii=False)
            or ".pptx" in json.dumps(ev, ensure_ascii=False)
            for ev in journal
        )
        wrote_submission = any("submission/" in json.dumps(ev, ensure_ascii=False) for ev in journal)
        last_step = max(1, len([ev for ev in journal if "model.request" in str(ev.get("event_type") or ev.get("kind") or "")]))

        if u.e1_observation.e1_class == "broken":
            b_preds.append({
                "unit_id": u.unit_id,
                "analyst_group": "B",
                "p_prob": 0.30,
                "d_prob": 0.60,
                "r_prob": 0.60,
                "recommended_arm": "D",
                "predicted_label": "broken",
                "confidence": 0.70,
                "abstain": False,
                "evidence_steps": [max(1, last_step - 1)],
            })
        elif wrote_workspace_file and not wrote_submission:
            b_preds.append({
                "unit_id": u.unit_id,
                "analyst_group": "B",
                "p_prob": 0.10,
                "d_prob": 0.75,
                "r_prob": 0.75,
                "recommended_arm": "D",
                "predicted_label": "delivery_responsive",
                "confidence": 0.75,
                "abstain": False,
                "evidence_steps": [max(1, last_step)],
            })
        elif wrote_submission and u.rule_source == "explicit":
            b_preds.append({
                "unit_id": u.unit_id,
                "analyst_group": "B",
                "p_prob": 0.05,
                "d_prob": 0.15,
                "r_prob": 0.62,
                "recommended_arm": "R",
                "predicted_label": "rule_responsive",
                "confidence": 0.62,
                "abstain": False,
                "evidence_steps": [max(1, last_step - 2)],
            })
        else:
            b_preds.append({
                "unit_id": u.unit_id,
                "analyst_group": "B",
                "p_prob": 0.05,
                "d_prob": 0.12,
                "r_prob": 0.18,
                "recommended_arm": "none",
                "predicted_label": "unrecovered_under_protocol",
                "confidence": 0.58,
                "abstain": False,
                "evidence_steps": [max(1, last_step)] if "error" in raw_text_joined.lower() else [],
            })

        # Group C: Structured evidence + contrast.json analyst (uses facts + candidate_substate + rule_source)
        sub = u.e1_observation.candidate_substate
        if u.e1_observation.e1_class == "broken":
            c_preds.append({
                "unit_id": u.unit_id,
                "analyst_group": "C",
                "p_prob": 0.35,
                "d_prob": 0.75,
                "r_prob": 0.75,
                "recommended_arm": "D",
                "predicted_label": "broken",
                "confidence": 0.85,
                "abstain": False,
                "evidence_steps": [u.e1_observation.source_turn or max(1, last_step - 1)],
            })
        elif sub in ("no_candidate", "workspace_unsubmitted_positive") and (u.e2_response_vector.remaining_budget_at_anchor or 0) >= 6:
            c_preds.append({
                "unit_id": u.unit_id,
                "analyst_group": "C",
                "p_prob": 0.08,
                "d_prob": 0.78,
                "r_prob": 0.80,
                "recommended_arm": "D",
                "predicted_label": "delivery_responsive",
                "confidence": 0.82,
                "abstain": False,
                "evidence_steps": [u.e2_response_vector.anchor_k or max(1, last_step)],
            })
        elif sub in ("submitted_dim1_failed", "written_score0_candidate", "delivered_item_wrong") and u.rule_source in ("explicit", "gate_or_no_artifact"):
            c_preds.append({
                "unit_id": u.unit_id,
                "analyst_group": "C",
                "p_prob": 0.05,
                "d_prob": 0.12,
                "r_prob": 0.72,
                "recommended_arm": "R",
                "predicted_label": "rule_responsive",
                "confidence": 0.76,
                "abstain": False,
                "evidence_steps": [u.e1_observation.first_candidate_write_turn or u.e2_response_vector.anchor_k or 1],
            })
        else:
            c_preds.append({
                "unit_id": u.unit_id,
                "analyst_group": "C",
                "p_prob": 0.03,
                "d_prob": 0.06,
                "r_prob": 0.10,
                "recommended_arm": "none",
                "predicted_label": "unrecovered_under_protocol",
                "confidence": 0.80,
                "abstain": False,
                "evidence_steps": [],
            })

    sealed = {"M": m_preds, "A": a_preds, "B": b_preds, "C": c_preds}
    sealed_payload = {
        "schema_version": "l2-0.5.0",
        "sealed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "predictions": sealed,
        "predictions_sha256": digest(sealed),
    }
    atomic_write_json(diag_dir / "sealed_analyst_predictions_v05.json", sealed_payload)
    return sealed


def select_stratified_parent_anchors(
    units: list[DiagnosticUnitV05],
    max_states: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Stratified selection of unique parent states (episode_id, anchor_k) covering:
    - Stratum 1: Level-1 `no_candidate` (k* = 0)
    - Stratum 2: Level-1 `submitted_dim1_failed` / `written_score0_candidate`
    - Stratum 3: Level-2 `delivered_item_wrong` (covering explicit / implicit / ambiguous)
    Returns (selected_states, inclusion_prob_by_unit_id).
    """
    eligible_never = [
        u for u in units
        if u.e1_observation.e1_class == "never_succeeded_on_disk"
        and u.e2_response_vector.anchor_k is not None
    ]

    by_state: dict[tuple[str, int], list[DiagnosticUnitV05]] = {}
    for u in eligible_never:
        key = (u.episode_id, int(u.e2_response_vector.anchor_k))  # type: ignore[arg-type]
        by_state.setdefault(key, []).append(u)

    stratum1_states: list[tuple[str, int]] = []
    stratum2_states: list[tuple[str, int]] = []
    stratum3_states: list[tuple[str, int]] = []

    for state_key, state_units in sorted(by_state.items()):
        substates = {u.e1_observation.candidate_substate for u in state_units}
        if "no_candidate" in substates:
            stratum1_states.append(state_key)
        elif "submitted_dim1_failed" in substates or "written_score0_candidate" in substates:
            stratum2_states.append(state_key)
        else:
            stratum3_states.append(state_key)

    k1 = min(len(stratum1_states), max(1, max_states // 3 + (1 if max_states % 3 >= 1 else 0))) if stratum1_states else 0
    k2 = min(len(stratum2_states), max(1, max_states // 3 + (1 if max_states % 3 >= 2 else 0))) if stratum2_states else 0
    k3 = min(len(stratum3_states), max(1, max_states - k1 - k2)) if stratum3_states else 0

    def _pick_diverse(states: list[tuple[str, int]], k: int) -> list[tuple[str, int]]:
        if k <= 0:
            return []
        if len(states) <= k:
            return list(states)
        picked: list[tuple[str, int]] = []
        seen_tasks: set[str] = set()
        for st in states:
            t_prefix = st[0].split(".")[0]
            if t_prefix not in seen_tasks and len(picked) < k:
                picked.append(st)
                seen_tasks.add(t_prefix)
        for st in states:
            if st not in picked and len(picked) < k:
                picked.append(st)
        return picked

    sel1 = _pick_diverse(stratum1_states, k1)
    sel2 = _pick_diverse(stratum2_states, k2)
    sel3 = _pick_diverse(stratum3_states, k3)

    pi1 = (len(sel1) / len(stratum1_states)) if stratum1_states else 0.0
    pi2 = (len(sel2) / len(stratum2_states)) if stratum2_states else 0.0
    pi3 = (len(sel3) / len(stratum3_states)) if stratum3_states else 0.0

    inclusion_prob: dict[str, float] = {}
    selected_states: list[dict[str, Any]] = []

    for stratum_name, sel_list, pi_val in (
        ("stratum1_no_candidate", sel1, pi1),
        ("stratum2_dim1_invalid", sel2, pi2),
        ("stratum3_dim2_item_wrong", sel3, pi3),
    ):
        for st in sel_list:
            st_units = by_state[st]
            for u in st_units:
                inclusion_prob[u.unit_id] = round(pi_val, 6)
            selected_states.append({
                "episode_id": st[0],
                "anchor_k": st[1],
                "stratum": stratum_name,
                "inclusion_prob": round(pi_val, 6),
                "unit_ids": [u.unit_id for u in st_units],
                "remaining_budget": st_units[0].e2_response_vector.remaining_budget_at_anchor or (35 - st[1]),
                "anchor_type": st_units[0].e2_response_vector.anchor_type or "budget_tail_no_candidate",
            })

    return selected_states, inclusion_prob


def execute_interleaved_pdr_for_state(
    state_info: dict[str, Any],
    *,
    dev45_dir: Path,
    branches_root: Path,
    rule_lookup: dict[tuple[str, str], dict[str, Any]],
    n_per_arm: int = 24,
    max_workers: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    """Run interleaved P, D, R continuation branches (n_per_arm each) from (episode_id, anchor_k)
    and grade both run-level and item-level outcomes on every branch.
    """
    ep_name = state_info["episode_id"]
    anchor_k = int(state_info["anchor_k"])
    rem_budget = int(state_info["remaining_budget"])
    task_num = ep_name.split(".")[0].replace("officeval_", "")
    cp_id = f"cp{anchor_k:06d}"

    ep_map = dict(_list_dev45_episodes(dev45_dir))
    parent_ep = ep_map[ep_name]
    base_cfg = read_json(parent_ep / "raw" / "resolved_config.json")

    explicit_reqs = _extract_task_explicit_requirements(task_num, parent_ep, rule_lookup)
    r_message = build_r_all_intervention_message(explicit_reqs)
    assert verify_rule_prompt_non_leakage(r_message), f"Leakage check failed for {ep_name}"

    schedule = build_interleaved_arm_schedule(
        ["P", "D", "R"],
        n_per_arm=n_per_arm,
        anchor_k=anchor_k,
        remaining_budget=rem_budget,
        seed=f"v3-stage-d-{ep_name}-k{anchor_k}",
    )

    state_out_dir = branches_root / f"{ep_name}_k{anchor_k:02d}"
    state_out_dir.mkdir(parents=True, exist_ok=True)

    def _run_one_branch(sched_item: dict[str, Any]) -> dict[str, Any]:
        arm = sched_item["arm"]
        rep = int(sched_item["replicate"])
        at_req = int(sched_item["at_request"])
        b_dir = state_out_dir / f"{arm}_r{rep:02d}"
        grade_cache = b_dir / "branch_grade_v05.json"

        if grade_cache.exists():
            return read_json(grade_cache)

        ep_dir = _resolve_episode_dir(b_dir)
        if ep_dir is None or not verify_episode_dir(ep_dir).get("verified"):
            cfg = copy.deepcopy(base_cfg)
            cfg.get("model", {}).get("parameters", {}).pop("seed", None)
            if arm == "D":
                cfg.setdefault("agent", {})["intervention"] = {
                    "at_request": at_req,
                    "message": DELIVER_REMINDER,
                }
            elif arm == "R":
                cfg.setdefault("agent", {})["intervention"] = {
                    "at_request": at_req,
                    "message": r_message,
                }
            else:
                cfg.get("agent", {}).pop("intervention", None)

            if b_dir.exists():
                import shutil
                shutil.rmtree(b_dir)
            l1_run(
                cfg,
                b_dir,
                parent=parent_ep,
                checkpoint_id=cp_id,
            )
            ep_dir = b_dir

        ver = verify_episode_dir(ep_dir)
        study_out = study_outcome(ep_dir)
        term_y = float(study_out.get("study_outcome_y") or 0.0)
        b_pkg = load_episode(ep_dir)
        item_passed_map = _extract_item_scores(b_pkg)
        og = b_pkg.json("official_grade.json") if (ep_dir / "official_grade.json").is_file() else {}
        dim1_pass = bool((og.get("official_raw") or {}).get("dim1_pass"))

        res = {
            "arm": arm,
            "replicate": rep,
            "episode_dir": str(ep_dir),
            "evaluation_validity": "valid" if ver.get("verified") else "invalid",
            "study_valid": bool(study_out.get("study_valid")),
            "terminal_score_y": round(term_y, 6),
            "dim1_pass": 1.0 if dim1_pass else 0.0,
            "item_scores": item_passed_map,
        }
        atomic_write_json(grade_cache, res)
        return res

    results_by_arm: dict[str, list[dict[str, Any]]] = {"P": [], "D": [], "R": []}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_run_one_branch, item): item for item in schedule}
        for fut in as_completed(futs):
            r = fut.result()
            results_by_arm[r["arm"]].append(r)

    for arm in results_by_arm:
        results_by_arm[arm].sort(key=lambda x: x["replicate"])
    return results_by_arm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-arm", type=int, default=24)
    parser.add_argument("--max-states", type=int, default=9)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    fam_map = _load_family_map()
    rule_lookup = _load_rule_source_lookup()

    units, cp_rows_by_ep, accounting = build_two_level_units_for_dev45(
        RUNS_ROOT, DIAG_ROOT, fam_map, rule_lookup
    )
    print(f"[STAGE-D-P2] Built {len(units)} two-level units. Zero double counting: {accounting['zero_double_counting_verified']}")

    sealed_preds = seal_analyst_predictions(units, RUNS_ROOT, DIAG_ROOT)
    print(f"[STAGE-D-P2] Sealed Analyst M/A/B/C predictions across {len(units)} units.")

    selected_states, inclusion_prob = select_stratified_parent_anchors(units, max_states=args.max_states)
    print(f"[STAGE-D-P2] Selected {len(selected_states)} stratified parent states covering {len(inclusion_prob)} units.")

    if args.dry_run:
        return

    branches_root = L1_ROOT / "runs" / "v3_qwen36_pi_dev45_interventions_20261017"
    branches_root.mkdir(parents=True, exist_ok=True)

    state_outcomes: dict[tuple[str, int], dict[str, list[dict[str, Any]]]] = {}
    for idx, st in enumerate(selected_states, start=1):
        print(f"[STAGE-D-P2] ({idx}/{len(selected_states)}) Running interleaved P/D/R (n={args.n_per_arm}) on {st['episode_id']} @ k={st['anchor_k']}...")
        arm_res = execute_interleaved_pdr_for_state(
            st,
            dev45_dir=RUNS_ROOT,
            branches_root=branches_root,
            rule_lookup=rule_lookup,
            n_per_arm=args.n_per_arm,
            max_workers=args.max_workers,
        )
        state_outcomes[(st["episode_id"], int(st["anchor_k"]))] = arm_res

    # Adjudicate sampled units and populate IPW rows
    ipw_rows: list[dict[str, Any]] = []
    unit_by_id = {u.unit_id: u for u in units}
    for st in selected_states:
        st_key = (st["episode_id"], int(st["anchor_k"]))
        arm_res = state_outcomes[st_key]
        for uid in st["unit_ids"]:
            u = unit_by_id[uid]
            if u.unit_level == "gate_or_no_artifact":
                p_vals = [float(x["terminal_score_y"]) for x in arm_res["P"]]
                d_vals = [float(x["terminal_score_y"]) for x in arm_res["D"]]
                r_vals = [float(x["terminal_score_y"]) for x in arm_res["R"]]
            else:
                iid = uid.split(":", 1)[1]
                p_vals = [float(x["item_scores"].get(iid, 0.0)) for x in arm_res["P"]]
                d_vals = [float(x["item_scores"].get(iid, 0.0)) for x in arm_res["D"]]
                r_vals = [float(x["item_scores"].get(iid, 0.0)) for x in arm_res["R"]]

            vec, op_lbl, ev_st = adjudicate_e2_outcomes(
                u.e1_observation.terminal_item_score,
                {"P": p_vals, "D": d_vals, "R": r_vals},
                anchor_k=int(st["anchor_k"]),
                anchor_type=st["anchor_type"],
                remaining_budget=int(st["remaining_budget"]),
            )
            u.e2_response_vector = vec
            u.operational_label = op_lbl
            u.evidence_status = ev_st

            ep_prefix = u.episode_id.split(".")[0]
            ipw_rows.append({
                "unit_id": u.unit_id,
                "episode_id": u.episode_id,
                "family_id": fam_map.get(ep_prefix, ep_prefix),
                "unit_level": u.unit_level,
                "rule_source": u.rule_source,
                "stratum": st["stratum"],
                "deduction": u.deduction_weight,
                "inclusion_prob": inclusion_prob[u.unit_id],
                "operational_label": op_lbl,
                "evidence_status": ev_st,
                "recovered_by_D": (op_lbl == "delivery_responsive" and ev_st == "adjudicated"),
                "recovered_by_R": (op_lbl == "rule_responsive" and ev_st == "adjudicated"),
                "inconclusive": (ev_st == "inconclusive"),
                "delta_D_vs_P": vec.delta_D_vs_P,
                "delta_R_vs_D": vec.delta_R_vs_D,
            })

    # V3-P1 IPW estimation
    ipw_summary = compute_ipw_recoverability_ratio(
        ipw_rows,
        total_never_deduction=accounting["e1_never_loss"],
    )

    # Mandatory rule_source breakdown (explicit / implicit / ambiguous / gate_or_no_artifact)
    rule_source_breakdown: dict[str, dict[str, Any]] = {}
    for rs_key in ("explicit", "implicit", "ambiguous", "gate_or_no_artifact"):
        rs_all = [u for u in units if u.rule_source == rs_key and u.e1_observation.e1_class == "never_succeeded_on_disk"]
        rs_sampled = [r for r in ipw_rows if r["rule_source"] == rs_key]
        w_sum = sum(r["deduction"] for r in rs_sampled)
        mean_r_vs_d = (
            sum(float(r["delta_R_vs_D"] or 0.0) * r["deduction"] for r in rs_sampled) / w_sum
            if w_sum > 0 else 0.0
        )
        rule_source_breakdown[rs_key] = {
            "population_unit_count": len(rs_all),
            "population_deduction": round(sum(u.deduction_weight for u in rs_all), 6),
            "sampled_unit_count": len(rs_sampled),
            "sampled_deduction": round(w_sum, 6),
            "weighted_mean_delta_R_vs_D": round(mean_r_vs_d, 6),
            "delivery_responsive_count": sum(1 for r in rs_sampled if r["recovered_by_D"]),
            "rule_responsive_count": sum(1 for r in rs_sampled if r["recovered_by_R"]),
            "inconclusive_count": sum(1 for r in rs_sampled if r["inconclusive"]),
        }

    # V3-P3 Analyst evaluation across M / A / B / C
    analyst_evals = {
        grp: evaluate_analysts_v05(units, preds)
        for grp, preds in sealed_preds.items()
    }

    summary = {
        "schema_version": "l2-0.5.0",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "two_level_accounting": accounting,
        "selected_states": selected_states,
        "ipw_v3_p1_summary": ipw_summary,
        "rule_source_breakdown_v3_p2": rule_source_breakdown,
        "analyst_evaluation_v3_p3": analyst_evals,
        "sampled_unit_results": ipw_rows,
    }
    atomic_write_json(OUT_JSON, summary)

    md = f"""# V3 Stage D Full Report: Two-Level Census, Stratified Interleaved `P/D/R` ($n={args.n_per_arm}$), and `V3-P1..P3` Adjudication

- **Completed At**: `{summary['completed_at']}`
- **Unified Actor**: `Qwen/Qwen3.6-27B (revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9)` + `pi-coding-agent@0.73.1`
- **Protocol**: `L2 v0.5` (`DiagnosticUnitV05`)

---

## 1. Two-Level Non-Overlapping Loss Accounting (V3 §3.1)

| Metric | Value |
|---|---:|
| Total Run Loss $\sum_r (1 - Y_r)$ | `{accounting['total_run_loss']:.6f}` |
| Level-1 Gate / No-Artifact Loss ($\text{{dim1\_pass}} = \text{{false}}$) | `{accounting['level1_gate_loss']:.6f}` (`{accounting['level1_unit_count']}` units) |
| Level-2 Rubric Item Loss ($\text{{dim1\_pass}} = \text{{true}}$) | `{accounting['level2_item_loss']:.6f}` (`{accounting['level2_unit_count']}` units) |
| Reconstructed Sum | `{accounting['reconstructed_sum']:.6f}` |
| Zero Double-Counting Verified | `{accounting['zero_double_counting_verified']}` |
| E1 `broken` Loss | `{accounting['e1_broken_loss']:.6f}` |
| E1 `undelivered` Loss | `{accounting['e1_undelivered_loss']:.6f}` |
| E1 `never_succeeded_on_disk` Loss | `{accounting['e1_never_loss']:.6f}` |

---

## 2. `V3-P1` & `V3-P2`: Inverse-Probability Weighted (IPW) Recoverability & Rule-Source Breakdown

- **Strict Lower-Bound Recoverable Ratio (`delivery_responsive` + `rule_responsive`, `inconclusive = 0`)**:
  - Point Estimate: **`{ipw_summary['strict_recoverable_ratio']:.4f}`** (`95%` Family-Clustered Bootstrap CI: `{ipw_summary['strict_ci_95']}`)
  - `delivery_responsive` Share: `{ipw_summary['delivery_responsive_ratio']:.4f}`
  - `rule_responsive` Share: `{ipw_summary['rule_responsive_ratio']:.4f}`
- **Upper-Bound Recoverable Ratio (Including `inconclusive`)**:
  - Point Estimate: **`{ipw_summary['upper_bound_including_inconclusive_ratio']:.4f}`** (`95%` CI: `{ipw_summary['upper_ci_95']}`)

### Mandatory `rule_source` Covariate Breakdown (V3 §3.3 & `V3-P2`)

| `rule_source` | Pop. Units | Pop. Deduction | Sampled Units | Sampled Deduction | Weighted Mean $\hat{{\Delta}}_{{R-D}}$ | `delivery_responsive` | `rule_responsive` | `inconclusive` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for rs_key, rs_val in rule_source_breakdown.items():
        md += (
            f"| `{rs_key}` | `{rs_val['population_unit_count']}` | `{rs_val['population_deduction']:.4f}` | "
            f"`{rs_val['sampled_unit_count']}` | `{rs_val['sampled_deduction']:.4f}` | "
            f"`{rs_val['weighted_mean_delta_R_vs_D']:+.4f}` | `{rs_val['delivery_responsive_count']}` | "
            f"`{rs_val['rule_responsive_count']}` | `{rs_val['inconclusive_count']}` |\n"
        )

    md += """
---

## 3. `V3-P3`: Pre-Sealed Analyst Evaluation (`M / A / B / C`) Against E2 Adjudicated Truth

| Analyst Group | Weighted 5-Class Macro-F1 | Weighted Accuracy | Mean Multi-Class Brier Score | Weighted Regret | False Localization Rate on `unrecovered_under_protocol` | Selective Risk @ 80% Coverage |
|---|---:|---:|---:|---:|---:|---:|
"""
    for grp in ("M", "A", "B", "C"):
        ev = analyst_evals[grp]
        md += (
            f"| **`{grp}`** | `{ev['weighted_macro_f1']:.4f}` | `{ev['weighted_accuracy']:.4f}` | "
            f"`{ev['mean_brier_score']:.4f}` | `{ev['weighted_regret']:.4f}` | "
            f"`{ev['false_localization_rate_on_unrecovered']:.4f}` | `{ev['selective_risk_at_80pct_coverage']:.4f}` |\n"
        )

    OUT_MD.write_text(md, encoding="utf-8")
    print(f"[STAGE-D-P2] Wrote full Stage D Phase 2 summary to {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
