#!/usr/bin/env python3
"""V3 Stage D2 Entrypoint (R16, R18): 0-Request 2-Fold Cross-Fitted Diagnostic Value Ceiling (`V3-P3` Gate).

Contract (Notion V3 §3.7, §5.2 `V3-P3`, §8.2, R13, R16, R18):
- Exclusively reuses continuation branches already produced by Stage D1 (`actor_requests_issued = 0`,
  `analyst_requests_issued = 0`).
- Splits evaluated units into two strictly disjoint halves `I_1` and `I_2` (`I_1 ∩ I_2 = ∅`) and uses
  split-branch evaluation within units so selection and gain evaluation never share continuation outcomes.
- Reports both unweighted (equal weight per unit) and deduction-weighted cross-fitted diagnostic value ceilings.
- Serves as the mandatory prerequisite gate for Stage D3 (`V3-P4` / `V3-P5`): if `D2` does not reach the
  pre-registered ceiling threshold (`ceiling_threshold`, draft `0.05`), Stage D3 Analyst evaluation is skipped.
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
sys.path.insert(0, str(L2_ROOT / "src"))

from l2_sandbox.v05 import (  # noqa: E402
    DiagnosticUnitV05,
    E1Observation,
    V05_SCHEMA_VERSION,
    adjudicate_e2_outcomes,
    compute_cross_fitted_diagnostic_value_ceiling,
)

DEFAULT_D1_OUTCOMES = L2_ROOT / "reports" / "v3_d1_ladder_outcomes_20261017.json"
DEFAULT_D2_OUT = L2_ROOT / "reports" / "v3_d2_ceiling_report_20261017.json"


def evaluate_d2_from_d1_payload(
    d1_payload: dict[str, Any],
    *,
    ceiling_threshold: float = 0.05,
    delta: float = 0.10,
    tau: float = 0.20,
    alpha: float = 0.05,
) -> dict[str, Any]:
    raw_units = d1_payload.get("evaluated_units") or []
    units: list[DiagnosticUnitV05] = []
    for row in raw_units:
        arm_outcomes = row.get("arm_outcomes") or {}
        if "P" not in arm_outcomes or len(arm_outcomes) < 2:
            continue
        vec, op_lbl, ev_st = adjudicate_e2_outcomes(
            float(row.get("terminal_y") or 0.0),
            arm_outcomes,
            anchor_k=int(row.get("anchor_k") or 0),
            anchor_type=str(row.get("anchor_type") or "budget_tail_no_candidate"),
            remaining_budget=int(row.get("remaining_budget") or 6),
            delta=delta,
            tau=tau,
            alpha=alpha,
        )
        units.append(
            DiagnosticUnitV05(
                unit_id=str(row["unit_id"]),
                episode_id=str(row.get("episode_id") or row["unit_id"].split("::")[0]),
                task_id=str(row.get("task_id") or "officeval"),
                benchmark=str(row.get("benchmark") or "officeval"),
                unit_level=str(row.get("unit_level") or "rubric_item"),
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

    crossfit = compute_cross_fitted_diagnostic_value_ceiling(
        units,
        ceiling_threshold=ceiling_threshold,
        alpha=alpha,
    )
    return {
        "schema_version": V05_SCHEMA_VERSION,
        "stage": "D2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "actor_requests_issued": 0,
        "analyst_requests_issued": 0,
        "status": "evaluated" if units else "pending_d1_execution",
        "d0_parameters": {
            "ceiling_threshold": ceiling_threshold,
            "delta": delta,
            "tau": tau,
            "alpha": alpha,
        },
        "cross_fitted_ceiling": crossfit,
        "d3_prerequisite_gate_open": bool(crossfit.get("passes_v3_p3_gate")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="V3 Stage D2 0-Request Cross-Fitted Ceiling Evaluator")
    parser.add_argument("--d1-outcomes", type=Path, default=DEFAULT_D1_OUTCOMES)
    parser.add_argument("--out", type=Path, default=DEFAULT_D2_OUT)
    parser.add_argument("--ceiling-threshold", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument("--tau", type=float, default=0.20)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    if args.d1_outcomes.exists():
        d1_payload = json.loads(args.d1_outcomes.read_text(encoding="utf-8"))
    else:
        d1_payload = {"evaluated_units": []}

    report = evaluate_d2_from_d1_payload(
        d1_payload,
        ceiling_threshold=args.ceiling_threshold,
        delta=args.delta,
        tau=args.tau,
        alpha=args.alpha,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        f"[V3-D2 CEILING] Wrote {args.out}\n"
        f"  status                  = {report['status']}\n"
        f"  actor_requests_issued   = {report['actor_requests_issued']}\n"
        f"  eligible_unit_count     = {report['cross_fitted_ceiling']['eligible_unit_count']}\n"
        f"  unweighted_ceiling      = {report['cross_fitted_ceiling']['unweighted_ceiling']}\n"
        f"  deduction_weighted_ceil = {report['cross_fitted_ceiling']['deduction_weighted_ceiling']}\n"
        f"  d3_gate_open            = {report['d3_prerequisite_gate_open']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
