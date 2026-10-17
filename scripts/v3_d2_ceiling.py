#!/usr/bin/env python3
"""V3 Stage D2 Entrypoint (R16, R18): 0-Request 2-Fold Cross-Fitted Diagnostic Value Ceiling (`V3-P3` Gate).

Contract (Notion V3 §3.7, §5.2 `V3-P3`, §8.2, R13, R16, R18):
- Exclusively reuses continuation branches already produced by Stage D1 (`actor_requests_issued = 0`,
  `analyst_requests_issued = 0`).
- Validates D1 provenance at the input boundary:
  * Verifies `d1_plan_sha256` binding against the Stage D1 plan (when provided).
  * Verifies per-arm `branch_receipts` with distinct `branch_id`s, `status == "completed"`, matching
    `anchor_k` / `remaining_budget`, and `>= min_replicates_per_arm` (at least 2) independent replicates
    so selection (`[:mid]`) and evaluation (`[mid:]`) never share a continuation branch.
- Splits evaluated units into two strictly disjoint halves `I_1` and `I_2` (`I_1 ∩ I_2 = ∅`), enforces
  distinct fixed-strategy (`P, D, S, R_explicit`) vs diagnostic (`P, D, S, R_explicit, R_single, R`) arm pools,
  preserves signed evaluation gains (never clamping negative gains to 0), and gates Stage D3 on the
  one-sided deduction-weighted 95% lower confidence bound (`deduction_weighted_lb95 > ceiling_threshold`).
- Supports both `--plan-only` and `--execute` CLI flags (`R18`), issuing 0 model requests in all modes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
L2_ROOT = REPO_ROOT / "L2_Behavioral Event Localization" / "L2_Sandbox"
L1_ROOT = REPO_ROOT / "L1_sandbox" / "C3_Experimental_Sandbox"
sys.path.insert(0, str(L1_ROOT / "src"))
sys.path.insert(0, str(L2_ROOT / "src"))

from l2_sandbox.v05 import (  # noqa: E402
    DiagnosticUnitV05,
    E1Observation,
    V05_SCHEMA_VERSION,
    adjudicate_e2_outcomes,
    compute_cross_fitted_diagnostic_value_ceiling,
    sha256_text,
)

from l2_sandbox.v3_execution import digest, seal, verify_seal, verify_branch, write_json

DEFAULT_D1_PLAN = L2_ROOT / "reports" / "v3_d1_ladder_plan_20261017.json"
DEFAULT_D1_OUTCOMES = L2_ROOT / "reports" / "v3_d1_ladder_outcomes_20261017.json"
DEFAULT_D2_OUT = L2_ROOT / "reports" / "v3_d2_ceiling_report_20261017.json"


def _verify_unit_d1_provenance(
    row: dict[str, Any],
    *,
    min_replicates_per_arm: int,
    require_branch_receipts: bool,
    planned_schedule_by_uid: dict[str, dict[str, Any]] | None,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    uid = str(row.get("unit_id") or "")
    if not uid:
        return False, ["missing_unit_id"]

    arm_outcomes = row.get("arm_outcomes")
    if not isinstance(arm_outcomes, dict) or "P" not in arm_outcomes or len(arm_outcomes) < 2:
        return False, ["missing_P_or_fewer_than_2_arms"]

    req_reps = max(2, int(min_replicates_per_arm))
    for arm, vals in arm_outcomes.items():
        if not isinstance(vals, list) or len(vals) < req_reps:
            reasons.append(f"arm_{arm}_insufficient_replicates:{len(vals) if isinstance(vals, list) else 0}<{req_reps}")

    if planned_schedule_by_uid is not None:
        if uid not in planned_schedule_by_uid:
            reasons.append("unit_id_not_in_d1_plan")
        else:
            sched = planned_schedule_by_uid[uid]
            if int(row.get("anchor_k", -1)) != int(sched.get("anchor_k", -2)):
                reasons.append(f"anchor_k_mismatch:{row.get('anchor_k')}!={sched.get('anchor_k')}")
            if int(row.get("remaining_budget", -1)) != int(sched.get("remaining_budget", -2)):
                reasons.append(
                    f"remaining_budget_mismatch:{row.get('remaining_budget')}!={sched.get('remaining_budget')}"
                )

    if require_branch_receipts:
        receipts_by_arm = row.get("branch_receipts")
        if not isinstance(receipts_by_arm, dict):
            reasons.append("missing_branch_receipts")
        else:
            for arm, vals in arm_outcomes.items():
                r_list = receipts_by_arm.get(arm)
                if not isinstance(r_list, list) or len(r_list) != len(vals):
                    reasons.append(f"arm_{arm}_receipt_count_mismatch")
                    continue
                branch_ids = [str(r.get("branch_id") or "") for r in r_list if isinstance(r, dict)]
                if len(branch_ids) != len(vals) or any(not bid for bid in branch_ids):
                    reasons.append(f"arm_{arm}_missing_branch_id")
                    continue
                if len(set(branch_ids)) != len(branch_ids):
                    reasons.append(f"arm_{arm}_duplicate_branch_id_leakage")
                mid = len(branch_ids) // 2
                if not set(branch_ids[:mid]).isdisjoint(set(branch_ids[mid:])):
                    reasons.append(f"arm_{arm}_selection_eval_branch_overlap")
                for r in r_list:
                    if str(r.get("status") or "") != "completed":
                        reasons.append(f"arm_{arm}_non_completed_receipt:{r.get('status')}")
                        break

    return len(reasons) == 0, reasons


def evaluate_d2_from_d1_payload(
    d1_payload: dict[str, Any],
    *,
    d1_plan_payload: dict[str, Any] | None = None,
    ceiling_threshold: float = 0.05,
    delta: float = 0.10,
    tau: float = 0.20,
    alpha: float = 0.05,
    min_replicates_per_arm: int = 2,
    require_branch_receipts: bool = True,
    mode: str = "execute",
) -> dict[str, Any]:
    raw_units = d1_payload.get("evaluated_units") or []
    req_reps = max(2, int(min_replicates_per_arm))

    plan_sha_expected: str | None = None
    planned_schedule_by_uid: dict[str, dict[str, Any]] | None = None
    plan_binding_verified = False
    plan_binding_errors: list[str] = []

    try:
        if not d1_plan_payload:
            raise ValueError('sealed D1 plan required')
        verify_seal(d1_plan_payload, 'plan_sha256')
        verify_seal(d1_payload)
        if d1_payload.get('status') != 'completed':
            raise ValueError('D1 execution not completed')
        plan_sha_expected = d1_plan_payload['plan_sha256']
        if d1_payload.get('d1_plan_sha256') != plan_sha_expected:
            raise ValueError('D1 plan/outcomes binding mismatch')
        if ceiling_threshold != d1_plan_payload['d0_candidate_parameters']['ceiling_threshold'] or min_replicates_per_arm != d1_plan_payload['d0_candidate_parameters']['min_replicates_per_arm']:
            raise ValueError('D2 threshold or minimum replication differs from frozen D0')
        if (delta, tau, alpha) != tuple(d1_plan_payload['d0_candidate_parameters'][k] for k in ('delta','tau','alpha')):
            raise ValueError('D2 method parameters differ from frozen D1')
        planned_schedule_by_uid = {r['unit_id']: r for r in d1_plan_payload['unit_schedules']}
        if set(r['unit_id'] for r in raw_units) != set(planned_schedule_by_uid):
            raise ValueError('D1 outcome coverage differs from planned units')
        if not require_branch_receipts:
            raise ValueError('unreceipted outcomes cannot qualify')
        seen_ids = set()
        seen_packages = set()
        for row in raw_units:
            planned = planned_schedule_by_uid[row['unit_id']]
            for key in ('episode_id','task_id','family_id','unit_level','defect_name','rule_source','deduction_weight','stratum_id','inclusion_prob','anchor_k','anchor_type','remaining_budget','comparison_ids','parent_identity','e1_observation','deployable_arms'):
                if row.get(key) != planned.get(key):
                    raise ValueError(f'unit metadata differs from frozen plan: {key}')
            if row.get('terminal_y') != planned.get('terminal_y', 0.0):
                raise ValueError('unit terminal endpoint differs from plan')
            for arm in planned['eligible_arms']:
                receipts = row['branch_receipts'][arm]
                branches = [b for b in planned['schedule'] if b['arm'] == arm]
                if len(receipts) != len(branches) or len(receipts) != len(row['arm_outcomes'][arm]):
                    raise ValueError('branch schedule/outcome coverage mismatch')
                for branch, receipt, score in zip(branches, receipts, row['arm_outcomes'][arm]):
                    expected = dict(plan_sha256=plan_sha_expected, unit_id=row['unit_id'], arm=arm, replicate=branch['replicate'], anchor_k=planned['anchor_k'], remaining_budget=planned['remaining_budget'], message_sha256=planned['arm_messages'][arm]['message_sha256'], parent_identity=planned['parent_identity'], endpoint=dict(unit_level=planned.get('unit_level','rubric_item'), defect_name=planned.get('defect_name',planned.get('rule',''))))
                    expected['branch_id'] = digest(expected)
                    if bool(receipt.get('simulated_offline')) != bool(d1_payload.get('simulated_offline')):
                        raise ValueError('receipt simulation mode differs from D1 payload')
                    verify_branch(receipt, expected=expected)
                    if receipt['branch_id'] in seen_ids:
                        raise ValueError('duplicate physical continuation branch')
                    seen_ids.add(receipt['branch_id'])
                    if not receipt.get('simulated_offline'):
                        if receipt['package_index_sha256'] in seen_packages:
                            raise ValueError('same physical evidence package reused across replicates')
                        seen_packages.add(receipt['package_index_sha256'])
                    if score != receipt['outcome_score']:
                        raise ValueError('outcome differs from bound branch receipt')
            if set(row['arm_outcomes']) != set(planned['eligible_arms']):
                raise ValueError('outcome arm pool differs from frozen schedule')
        plan_binding_verified = True
    except (ValueError, KeyError, TypeError, OSError) as exc:
        plan_binding_errors.append(str(exc))

    units: list[DiagnosticUnitV05] = []
    rejected_units: list[dict[str, Any]] = []

    if plan_binding_verified:
        for row in raw_units:
            if not isinstance(row, dict):
                rejected_units.append({"unit_id": "<non_dict>", "reasons": ["non_dict_unit_row"]})
                continue
            ok, reasons = _verify_unit_d1_provenance(
                row,
                min_replicates_per_arm=req_reps,
                require_branch_receipts=require_branch_receipts,
                planned_schedule_by_uid=planned_schedule_by_uid,
            )
            if not ok:
                rejected_units.append({"unit_id": str(row.get("unit_id") or "<unknown>"), "reasons": reasons})
                continue

            arm_outcomes = row["arm_outcomes"]
            vec, op_lbl, ev_st = adjudicate_e2_outcomes(
                float(row.get("terminal_y") or 0.0),
                arm_outcomes,
                anchor_k=int(row.get("anchor_k") or 0),
                anchor_type=str(row.get("anchor_type") or "voluntary_pre_finish"),
                remaining_budget=int(row.get("remaining_budget") or 6),
                delta=delta,
                tau=tau,
                alpha=alpha,
                primary_family_size=d1_plan_payload["primary_test_family"]["size"],
                comparison_ids=planned_schedule_by_uid[row["unit_id"]]["comparison_ids"],
            )
            units.append(
                DiagnosticUnitV05(
                    unit_id=str(row["unit_id"]),
                    episode_id=str(row.get("episode_id") or str(row["unit_id"]).split("::")[0]),
                    task_id=str(row.get("task_id") or "officeval"),
                    benchmark=str(row.get("benchmark") or "officeval"),
                    unit_level=str(row.get("unit_level") or "rubric_item"),
                    family_id=row.get("family_id"),
                    defect_name=str(row.get("defect_name") or "defect"),
                    deduction_weight=float(row.get("deduction_weight") or 1.0),
                    rule_source=str(row.get("rule_source") or "ambiguous"),
                    e1_observation=E1Observation(
                        e1_class="never_succeeded_on_disk",
                        candidate_substate=str(row.get("candidate_substate") or "no_candidate"),
                        k_star=int(row.get("anchor_k") or 0),
                        max_item_score=0.0,
                        terminal_item_score=float(row.get("terminal_y") or 0.0),
                        deduction=float(row.get("deduction_weight") or 1.0),
                    ),
                    e2_response_vector=vec,
                    operational_label=op_lbl,
                    evidence_status=ev_st,
                )
            )
    else:
        for row in raw_units:
            uid = str(row.get("unit_id") or "<unknown>") if isinstance(row, dict) else "<non_dict>"
            rejected_units.append({"unit_id": uid, "reasons": list(plan_binding_errors)})

    crossfit = compute_cross_fitted_diagnostic_value_ceiling(
        units,
        ceiling_threshold=ceiling_threshold,
        alpha=alpha,
        min_replicates_per_arm=req_reps,
    )

    if mode == "plan_only" and not raw_units:
        status = "plan_only_pending_d1_execution"
    elif raw_units and not units and (rejected_units or not plan_binding_verified):
        status = "rejected_unverified_d1_provenance"
    elif units:
        status = "evaluated"
    else:
        status = "pending_d1_execution"

    outcomes_sha256 = sha256_text(json.dumps(d1_payload, sort_keys=True, ensure_ascii=False))
    gate_open = bool(mode == "execute" and not d1_payload.get("simulated_offline") and status == "evaluated" and not rejected_units and crossfit.get("passes_v3_p3_gate"))

    return seal({
        "schema_version": V05_SCHEMA_VERSION,
        "stage": "D2",
        "d1_plan_sha256": plan_sha_expected,
        "d1_outcomes_sha256": d1_payload.get("artifact_sha256"),
        "simulated_offline": bool(d1_payload.get("simulated_offline")),
        "mode": mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "actor_requests_issued": 0,
        "analyst_requests_issued": 0,
        "status": status,
        "d0_parameters": {
            "ceiling_threshold": ceiling_threshold,
            "delta": delta,
            "tau": tau,
            "alpha": alpha,
            "min_replicates_per_arm": req_reps,
        },
        "d1_provenance_audit": {
            "d1_outcomes_sha256": outcomes_sha256,
            "d1_plan_sha256_expected": plan_sha_expected,
            "d1_plan_sha256_actual": d1_payload.get("d1_plan_sha256"),
            "plan_binding_verified": plan_binding_verified,
            "plan_binding_errors": plan_binding_errors,
            "require_branch_receipts": require_branch_receipts,
            "input_unit_count": len(raw_units),
            "verified_unit_count": len(units),
            "rejected_unit_count": len(rejected_units),
            "rejected_units": rejected_units,
        },
        "cross_fitted_ceiling": crossfit,
        "d3_prerequisite_gate_open": gate_open,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description="V3 Stage D2 0-Request Cross-Fitted Ceiling Evaluator")
    parser.add_argument("--d1-plan", type=Path, default=None)
    parser.add_argument("--d1-outcomes", type=Path, default=DEFAULT_D1_OUTCOMES)
    parser.add_argument("--out", type=Path, default=DEFAULT_D2_OUT)
    parser.add_argument("--ceiling-threshold", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument("--tau", type=float, default=0.20)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-replicates-per-arm", type=int, default=2)
    parser.add_argument("--allow-unreceipted", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if args.plan_only and args.execute:
        parser.error("choose --plan-only or --execute")
    mode = "execute" if args.execute else "plan_only"

    if args.execute and args.d1_outcomes.exists():
        d1_payload = json.loads(args.d1_outcomes.read_text(encoding="utf-8"))
    else:
        d1_payload = {"evaluated_units": []}

    d1_plan_payload: dict[str, Any] | None = None
    if args.d1_plan is not None and args.d1_plan.exists():
        d1_plan_payload = json.loads(args.d1_plan.read_text(encoding="utf-8"))
    elif args.d1_outcomes == DEFAULT_D1_OUTCOMES and DEFAULT_D1_PLAN.exists() and d1_payload.get("d1_plan_sha256"):
        d1_plan_payload = json.loads(DEFAULT_D1_PLAN.read_text(encoding="utf-8"))

    report = evaluate_d2_from_d1_payload(
        d1_payload,
        d1_plan_payload=d1_plan_payload,
        ceiling_threshold=args.ceiling_threshold,
        delta=args.delta,
        tau=args.tau,
        alpha=args.alpha,
        min_replicates_per_arm=args.min_replicates_per_arm,
        require_branch_receipts=not args.allow_unreceipted,
        mode=mode,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        f"[V3-D2 CEILING] Wrote {args.out}\n"
        f"  mode                    = {report['mode']}\n"
        f"  status                  = {report['status']}\n"
        f"  actor_requests_issued   = {report['actor_requests_issued']}\n"
        f"  verified_unit_count     = {report['d1_provenance_audit']['verified_unit_count']}\n"
        f"  rejected_unit_count     = {report['d1_provenance_audit']['rejected_unit_count']}\n"
        f"  eligible_unit_count     = {report['cross_fitted_ceiling']['eligible_unit_count']}\n"
        f"  unweighted_ceiling      = {report['cross_fitted_ceiling']['unweighted_ceiling']}\n"
        f"  deduction_weighted_ceil = {report['cross_fitted_ceiling']['deduction_weighted_ceiling']}\n"
        f"  deduction_weighted_lb95 = {report['cross_fitted_ceiling']['deduction_weighted_lb95']}\n"
        f"  d3_gate_open            = {report['d3_prerequisite_gate_open']}"
    )
    if args.execute and report["status"] == "rejected_unverified_d1_provenance":
        sys.stderr.write("BLOCKED [R16/R18]: D1 outcomes failed provenance or branch-isolation verification.\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
