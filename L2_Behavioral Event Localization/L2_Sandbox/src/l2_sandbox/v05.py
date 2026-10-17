"""L2 v0.5 Intervention-Verifiable Failure Diagnosis Protocol (Notion V3 §3.1–3.7, §5, §6).

Implements:
1. DiagnosticUnitV05 with strict separation of:
   - e1_observation (E1 zero-cost snapshot state)
   - e2_response_vector (E2 counterfactual continuation vector on P / D / R / R_all)
   - operational_label (broken, undelivered, delivery_responsive, rule_responsive, unrecovered_under_protocol)
   - evidence_status (adjudicated, plain_recoverable, inconclusive, ineligible, not_applicable)
   - rule_source (explicit, implicit, ambiguous, gate_or_no_artifact)
2. Anchor checkpoint selection including k* = 0 (no_candidate) via budget-tail anchor (remaining >= r_b)
   and optional earlier budget-sensitivity anchor (remaining >= 2 * r_b).
3. Same-checkpoint, same-remaining-budget interleaved arm scheduler for P / D / R / R_all.
4. Statistical decision rule with pre-registered (delta = 0.10, tau = 0.20, alpha = 0.05),
   one-sided 95% confidence bounds (lower_drop / upper_drop), and Holm step-down adjustment.
5. Analyst M / A / B / C intervention-prediction protocol and 4-tier evaluation metrics:
   - Tier 1: Verification-budget utility curve (deduction recovered per 100 requests) & AUUC
   - Tier 2: Deduction-weighted Macro-F1, accuracy, and selective accuracy-coverage curve
   - Tier 3: Multi-arm Brier Score & Expected Calibration Error (ECE)
   - Tier 4: Source-turn accuracy on `broken` defects + False Localization Rate on `unrecovered_under_protocol`
6. Cross-benchmark adaptation contracts for OfficeVal, LHTB Sokoban, and SWE-bench Pro V2.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import random
from typing import Any, Sequence

from .config import (
    DELIVER_REMINDER as DELIVER_INTERVENTION_MESSAGE,
    RULE_REMINDER_TEMPLATE as RULE_INTERVENTION_TEMPLATE,
)
from .io import L2Error, digest
from .stats import holm_adjust, lower_drop, permutation_one_sided, wilson_bounds

V05_SCHEMA_VERSION = "l2-0.5.0"

OPERATIONAL_LABELS = (
    "broken",
    "undelivered",
    "delivery_responsive",
    "rule_responsive",
    "unrecovered_under_protocol",
)

EVIDENCE_STATUSES = (
    "adjudicated",
    "plain_recoverable",
    "inconclusive",
    "ineligible",
    "not_applicable",
)

RULE_SOURCES = (
    "explicit",
    "implicit",
    "ambiguous",
    "gate_or_no_artifact",
)


@dataclass
class E1Observation:
    """Zero-cost snapshot trajectory observation (V3 §3.1 Field 1)."""
    e1_class: str  # "broken" | "undelivered" | "never_succeeded_on_disk"
    candidate_substate: str  # "no_candidate" | "written_score0_candidate" | "submitted_dim1_failed" | "delivered_item_wrong" | "workspace_unsubmitted_positive" | "peak_regressed"
    k_star: int
    max_item_score: float
    terminal_item_score: float
    deduction: float
    source_turn: int | None = None
    first_candidate_write_turn: int | None = None
    deadlock_turn: int | None = None  # For Sokoban irreversible_state_damage


@dataclass
class E2ResponseVector:
    """Counterfactual continuation response vector at anchor checkpoint k_int (V3 §3.1 Field 2)."""
    evaluated: bool = False
    anchor_k: int | None = None
    anchor_type: str | None = None  # "candidate_peak" | "candidate_first_write" | "budget_tail_no_candidate" | "early_budget_sensitivity"
    remaining_budget_at_anchor: int | None = None
    arm_outcomes: dict[str, list[float]] = field(default_factory=dict)
    mean_P: float | None = None
    mean_D: float | None = None
    mean_R: float | None = None
    mean_R_all: float | None = None
    delta_P_vs_Y: float | None = None
    delta_D_vs_P: float | None = None
    delta_R_vs_D: float | None = None
    delta_R_vs_P: float | None = None
    delta_R_single_vs_R_all: float | None = None
    lb95_P_vs_Y: float | None = None
    lb95_D_vs_P: float | None = None
    ub95_D_vs_P: float | None = None
    lb95_R_vs_D: float | None = None
    ub95_R_vs_D: float | None = None
    lb95_R_vs_P: float | None = None
    ub95_R_vs_P: float | None = None
    p_val_D_vs_P: float | None = None
    p_val_R_vs_D: float | None = None
    p_holm_D_vs_P: float | None = None
    p_holm_R_vs_D: float | None = None


@dataclass
class DiagnosticUnitV05:
    """Single V3 diagnostic unit (either Level-1 gate/no_artifact or Level-2 rubric item)."""
    unit_id: str
    episode_id: str
    task_id: str
    benchmark: str
    unit_level: str  # "gate_or_no_artifact" | "rubric_item"
    defect_name: str
    deduction_weight: float
    rule_source: str  # "explicit" | "implicit" | "ambiguous" | "gate_or_no_artifact"
    e1_observation: E1Observation
    e2_response_vector: E2ResponseVector
    operational_label: str | None
    evidence_status: str  # "adjudicated" | "plain_recoverable" | "inconclusive" | "ineligible" | "not_applicable"
    early_anchor_k: int | None = None
    early_anchor_remaining_budget: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def upper_drop(
    vals_treat: Sequence[float],
    vals_ctrl: Sequence[float] | float,
    alpha: float = 0.05,
    *,
    seed: str = "l2-v05-ub",
) -> float:
    """One-sided (1 - alpha) upper confidence bound for E[V(treat)] - E[V(ctrl)].

    By duality: UB_{1-alpha}(T - C) = -LB_{1-alpha}(C - T).
    When vals_ctrl is a scalar y_term, UB_{1-alpha}(T - y_term) = UB_{1-alpha}(T) - y_term.
    """
    if isinstance(vals_ctrl, (int, float)) and not isinstance(vals_ctrl, bool):
        y_term = float(vals_ctrl)
        _, hi_w = wilson_bounds(vals_treat, alpha=alpha, one_sided=True)
        if all(float(x) in (0.0, 1.0) for x in vals_treat):
            return hi_w - y_term
        n1 = len(vals_treat)
        if n1 == 0:
            return 1.0
        rng = random.Random(int(digest({"seed": seed, "t": sorted(round(float(x), 8) for x in vals_treat), "y": y_term, "a": alpha})[:16], 16))
        n_boot = 2000
        means = sorted((sum(rng.choice(vals_treat) for _ in range(n1)) / n1) - y_term for _ in range(n_boot))
        idx = min(n_boot - 1, max(0, int(math.ceil((1.0 - alpha) * (n_boot - 1)))))
        return max(means[idx], hi_w - y_term)
    return -lower_drop(vals_ctrl, vals_treat, alpha=alpha, seed=seed)


def select_v05_intervention_anchors(
    checkpoint_rows: Sequence[dict[str, Any]],
    *,
    max_requests: int = 35,
    r_b: int = 6,
    item_id: str | None = None,
) -> dict[str, Any]:
    """V3 §3.2 Anchor Selection Rule (including k* = 0 no_candidate runs & budget sensitivity).

    Returns:
      {
        "eligible": bool,
        "primary_anchor_k": int | None,
        "primary_anchor_type": str | None,
        "primary_remaining_budget": int | None,
        "early_anchor_k": int | None,
        "early_remaining_budget": int | None,
      }
    """
    eligible_cps: list[dict[str, Any]] = []
    for r in checkpoint_rows:
        if r.get("kind") != "checkpoint":
            continue
        k = int(r.get("k", 0))
        rem = int(r.get("remaining_requests", max_requests - k))
        if rem >= r_b and bool(r.get("cold_resume_eligible", True)):
            eligible_cps.append({**r, "k": k, "remaining_requests": rem})

    if not eligible_cps:
        return {
            "eligible": False,
            "primary_anchor_k": None,
            "primary_anchor_type": None,
            "primary_remaining_budget": None,
            "early_anchor_k": None,
            "early_remaining_budget": None,
        }

    eligible_cps.sort(key=lambda x: x["k"])
    cand_cps = [r for r in eligible_cps if r.get("candidate_status") == "ok"]

    # Helper to compute item or total score at checkpoint r
    def _score_at(r: dict[str, Any]) -> float:
        if item_id and isinstance(r.get("deductions"), dict):
            # If dim1_pass is True, item score is 1 - normalized_deduction / item_weight
            if not r.get("dim1_pass"):
                return 0.0
            ded = float(r["deductions"].get(item_id, 0.0))
            return 1.0 if ded <= 1e-9 else 0.0
        return float(r.get("score") or 0.0)

    if cand_cps:
        max_s = max(_score_at(r) for r in cand_cps)
        if max_s > 0.0:
            chosen = next(r for r in cand_cps if _score_at(r) >= max_s - 1e-9)
            anchor_type = "candidate_peak"
        else:
            chosen = cand_cps[0]
            anchor_type = "candidate_first_write"
    else:
        # Case 2 (k* = 0 / no_candidate in eligible prefix): Never skip!
        # Select latest eligible checkpoint with remaining_requests >= r_b
        chosen = eligible_cps[-1]
        anchor_type = "budget_tail_no_candidate"

    # Secondary earlier anchor for budget-starvation sensitivity analysis (remaining >= 2 * r_b)
    early_candidates = [
        r for r in eligible_cps
        if r["remaining_requests"] >= 2 * r_b and r["k"] < chosen["k"]
    ]
    early_chosen = early_candidates[-1] if early_candidates else None

    return {
        "eligible": True,
        "primary_anchor_k": int(chosen["k"]),
        "primary_anchor_type": anchor_type,
        "primary_remaining_budget": int(chosen["remaining_requests"]),
        "early_anchor_k": int(early_chosen["k"]) if early_chosen else None,
        "early_remaining_budget": int(early_chosen["remaining_requests"]) if early_chosen else None,
    }


def build_interleaved_arm_schedule(
    arms: Sequence[str],
    n_per_arm: int,
    *,
    anchor_k: int,
    remaining_budget: int,
    seed: str = "l2-v05-schedule",
) -> list[dict[str, Any]]:
    """Build an interleaved, randomized execution schedule across arms at the exact same
    checkpoint anchor_k and exact same remaining_budget (V3 §3.2 Invariant).
    """
    if n_per_arm < 1:
        raise L2Error("n_per_arm must be >= 1")
    schedule: list[dict[str, Any]] = []
    rng = random.Random(int(digest({"seed": seed, "arms": list(arms), "n": n_per_arm, "k": anchor_k})[:16], 16))
    for rep in range(n_per_arm):
        arm_order = list(arms)
        rng.shuffle(arm_order)
        for arm in arm_order:
            schedule.append({
                "arm": arm,
                "replicate": rep,
                "checkpoint_k": int(anchor_k),
                "at_request": int(anchor_k) + 1,
                "remaining_budget": int(remaining_budget),
            })
    return schedule


def build_r_all_intervention_message(requirements: Sequence[str]) -> str:
    """Build the whole-run screening intervention prompt (`R-all`, V3 §3.4) without leaking
    verifier code or variable names.
    """
    cleaned = [r.strip() for r in requirements if isinstance(r, str) and r.strip()]
    if not cleaned:
        return DELIVER_INTERVENTION_MESSAGE
    if len(cleaned) == 1:
        return RULE_INTERVENTION_TEMPLATE.format(rule=cleaned[0], requirement=cleaned[0])
    joined = "；".join(f"({idx}) {req}" for idx, req in enumerate(cleaned, start=1))
    return RULE_INTERVENTION_TEMPLATE.format(rule=joined, requirement=joined)


def adjudicate_e2_outcomes(
    terminal_y: float,
    arm_outcomes: dict[str, Sequence[float]],
    *,
    anchor_k: int,
    anchor_type: str = "budget_tail_no_candidate",
    remaining_budget: int = 6,
    delta: float = 0.10,
    tau: float = 0.20,
    alpha: float = 0.05,
) -> tuple[E2ResponseVector, str | None, str]:
    """V3 §3.2 Statistical Decision Rule with pre-registered (delta = 0.10, tau = 0.20)
    and Holm step-down adjustment across (D vs P) and (R vs D).

    Returns (e2_vector, operational_label, evidence_status).
    """
    p_vals = [float(x) for x in arm_outcomes.get("P", [])]
    d_vals = [float(x) for x in arm_outcomes.get("D", [])]
    r_vals = [float(x) for x in arm_outcomes.get("R", [])]
    r_all_vals = [float(x) for x in arm_outcomes.get("R_all", [])]

    if not p_vals or not d_vals or not r_vals:
        raise L2Error("E2 adjudication requires non-empty P, D, and R arm outcomes")

    mean_p = sum(p_vals) / len(p_vals)
    mean_d = sum(d_vals) / len(d_vals)
    mean_r = sum(r_vals) / len(r_vals)
    mean_r_all = (sum(r_all_vals) / len(r_all_vals)) if r_all_vals else None

    delta_p_y = mean_p - float(terminal_y)
    delta_d_p = mean_d - mean_p
    delta_r_d = mean_r - mean_d
    delta_r_p = mean_r - mean_p
    delta_r_single_all = (mean_r - mean_r_all) if mean_r_all is not None else None

    lb_p_y = lower_drop(p_vals, float(terminal_y), alpha=alpha, seed="v05-p-y")
    lb_d_p = lower_drop(d_vals, p_vals, alpha=alpha, seed="v05-d-p")
    ub_d_p = upper_drop(d_vals, p_vals, alpha=alpha, seed="v05-d-p-ub")
    lb_r_d = lower_drop(r_vals, d_vals, alpha=alpha, seed="v05-r-d")
    ub_r_d = upper_drop(r_vals, d_vals, alpha=alpha, seed="v05-r-d-ub")
    lb_r_p = lower_drop(r_vals, p_vals, alpha=alpha, seed="v05-r-p")
    ub_r_p = upper_drop(r_vals, p_vals, alpha=alpha, seed="v05-r-p-ub")

    # One-sided p-values for H1: D > P and H1: R > D
    p_d_vs_p = permutation_one_sided(d_vals, p_vals, direction=1.0, seed="v05-perm-dp")
    p_r_vs_d = permutation_one_sided(r_vals, d_vals, direction=1.0, seed="v05-perm-rd")
    p_holm_dp, p_holm_rd = holm_adjust([p_d_vs_p, p_r_vs_d])

    vec = E2ResponseVector(
        evaluated=True,
        anchor_k=anchor_k,
        anchor_type=anchor_type,
        remaining_budget_at_anchor=remaining_budget,
        arm_outcomes={k: [float(x) for x in v] for k, v in arm_outcomes.items()},
        mean_P=round(mean_p, 6),
        mean_D=round(mean_d, 6),
        mean_R=round(mean_r, 6),
        mean_R_all=round(mean_r_all, 6) if mean_r_all is not None else None,
        delta_P_vs_Y=round(delta_p_y, 6),
        delta_D_vs_P=round(delta_d_p, 6),
        delta_R_vs_D=round(delta_r_d, 6),
        delta_R_vs_P=round(delta_r_p, 6),
        delta_R_single_vs_R_all=round(delta_r_single_all, 6) if delta_r_single_all is not None else None,
        lb95_P_vs_Y=round(lb_p_y, 6),
        lb95_D_vs_P=round(lb_d_p, 6),
        ub95_D_vs_P=round(ub_d_p, 6),
        lb95_R_vs_D=round(lb_r_d, 6),
        ub95_R_vs_D=round(ub_r_d, 6),
        lb95_R_vs_P=round(lb_r_p, 6),
        ub95_R_vs_P=round(ub_r_p, 6),
        p_val_D_vs_P=round(p_d_vs_p, 6),
        p_val_R_vs_D=round(p_r_vs_d, 6),
        p_holm_D_vs_P=round(p_holm_dp, 6),
        p_holm_R_vs_D=round(p_holm_rd, 6),
    )

    # Step 1: Check if plain continuation P alone recovers the defect without intervention
    if delta_p_y >= tau and lb_p_y > delta:
        return vec, None, "plain_recoverable"

    # Step 2: Delivery-responsive (D > P)
    if delta_d_p >= tau and lb_d_p > delta and p_holm_dp <= alpha:
        return vec, "delivery_responsive", "adjudicated"

    # Step 3: Rule-responsive (D did not recover, R > D)
    if delta_d_p < tau and delta_r_d >= tau and lb_r_d > delta and p_holm_rd <= alpha:
        return vec, "rule_responsive", "adjudicated"

    # Step 4: Unrecovered under protocol (both D - P and R - P have 95% upper bound < tau)
    if ub_d_p < tau and ub_r_p < tau:
        return vec, "unrecovered_under_protocol", "adjudicated"

    # Step 5: Inconclusive (confidence interval straddles [delta, tau])
    return vec, None, "inconclusive"


def predict_majority_baseline_m(
    units: Sequence[DiagnosticUnitV05],
    *,
    default_majority_label: str = "unrecovered_under_protocol",
) -> list[dict[str, Any]]:
    """Baseline M (V3 §3.5): Zero-cost majority-class predictor."""
    preds: list[dict[str, Any]] = []
    for u in units:
        preds.append({
            "unit_id": u.unit_id,
            "analyst_group": "M",
            "p_prob": 0.05,
            "d_prob": 0.05 if default_majority_label != "delivery_responsive" else 0.80,
            "r_prob": 0.05 if default_majority_label != "rule_responsive" else 0.80,
            "recommended_arm": "none" if default_majority_label == "unrecovered_under_protocol" else (
                "D" if default_majority_label == "delivery_responsive" else "R"
            ),
            "predicted_label": default_majority_label,
            "confidence": 1.0,
            "abstain": False,
            "evidence_steps": [],
        })
    return preds


def predict_rule_analyst_a_v05(
    unit: DiagnosticUnitV05,
    facts: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Analyst Group A (V3 §3.5): Deterministic rule table on L2 facts (without seeing s(k)).

    Inspects whether the trajectory wrote files outside submission/, exhausted budget without
    final submission write, or modified submission/ followed by errors.
    """
    submission_writes = [
        int(f.get("step", 0))
        for f in facts
        if f.get("type") == "submission_artifact_Step"
        or (f.get("type") == "workspace_file_op" and str((f.get("value") or {}).get("path", "")).startswith("submission/"))
    ]
    workspace_writes = [
        int(f.get("step", 0))
        for f in facts
        if f.get("type") == "workspace_file_op"
        and not str((f.get("value") or {}).get("path", "")).startswith("submission/")
    ]
    tool_anomalies = [
        int(f.get("step", 0))
        for f in facts
        if f.get("type") in ("tool_error", "non_zero_exit")
    ]

    if not submission_writes and workspace_writes:
        # Agent worked in workspace/scripts without delivering to submission/ -> predict D
        return {
            "unit_id": unit.unit_id,
            "analyst_group": "A",
            "p_prob": 0.15,
            "d_prob": 0.65,
            "r_prob": 0.70,
            "recommended_arm": "D",
            "predicted_label": "delivery_responsive",
            "confidence": 0.65,
            "abstain": False,
            "evidence_steps": [workspace_writes[-1]],
        }
    if submission_writes and not tool_anomalies:
        # Agent cleanly wrote submission/ but missed specific rubric item -> predict R
        return {
            "unit_id": unit.unit_id,
            "analyst_group": "A",
            "p_prob": 0.05,
            "d_prob": 0.10,
            "r_prob": 0.55,
            "recommended_arm": "R",
            "predicted_label": "rule_responsive",
            "confidence": 0.55,
            "abstain": False,
            "evidence_steps": [submission_writes[-1]],
        }
    return {
        "unit_id": unit.unit_id,
        "analyst_group": "A",
        "p_prob": 0.05,
        "d_prob": 0.08,
        "r_prob": 0.15,
        "recommended_arm": "none",
        "predicted_label": "unrecovered_under_protocol",
        "confidence": 0.60,
        "abstain": False,
        "evidence_steps": [],
    }


def evaluate_analysts_v05(
    units: Sequence[DiagnosticUnitV05],
    predictions: Sequence[dict[str, Any]],
    *,
    cost_per_unit_requests: float = 144.0,
) -> dict[str, Any]:
    """Evaluate Analyst predictions (M / A / B / C) across V3 §3.5's 4 prioritized metric tiers.

    1. Primary Tier 1: Verification-Budget Utility Curve (expected deduction recovered per 100
       verification requests) and Area Under the Uplift Curve (AUUC).
    2. Primary Tier 2: Deduction-Weighted Macro-F1, Weighted Accuracy, and Selective Accuracy-Coverage.
    3. Primary Tier 3: Multi-arm Brier Score and Expected Calibration Error (ECE) on (p_P, p_D, p_R).
    4. Secondary Tier 4: Source-turn exact & +/-1 accuracy on `broken` units + False Localization Rate
       on `unrecovered_under_protocol` units.
    """
    by_id = {u.unit_id: u for u in units}
    pred_by_id = {p["unit_id"]: p for p in predictions if p.get("unit_id") in by_id}

    adjudicated_e2 = [
        u for u in units
        if u.evidence_status == "adjudicated"
        and u.operational_label in ("delivery_responsive", "rule_responsive", "unrecovered_under_protocol")
        and u.unit_id in pred_by_id
    ]

    # --- Tier 1: Verification-Budget Utility & AUUC ---
    # For each unit, if recommended_arm == "D", actual gain is max(0, delta_D_vs_P) * deduction_weight;
    # if recommended_arm == "R", actual gain is max(0, delta_R_vs_P) * deduction_weight.
    ranked_items: list[tuple[float, float, float]] = []  # (expected_priority, actual_recovered_deduction, request_cost)
    for u in adjudicated_e2:
        p = pred_by_id[u.unit_id]
        rec_arm = p.get("recommended_arm", "none")
        if p.get("abstain") or rec_arm == "none":
            priority = 0.0
        elif rec_arm == "D":
            priority = float(p.get("d_prob", 0.0)) * u.deduction_weight
        else:
            priority = float(p.get("r_prob", 0.0)) * u.deduction_weight

        vec = u.e2_response_vector
        if rec_arm == "D":
            gain_rate = max(0.0, float(vec.delta_D_vs_P or 0.0))
        elif rec_arm == "R":
            gain_rate = max(0.0, float(vec.delta_R_vs_P or 0.0))
        else:
            gain_rate = 0.0
        actual_gain = gain_rate * u.deduction_weight
        req_cost = 0.0 if rec_arm == "none" else float(cost_per_unit_requests)
        ranked_items.append((priority, actual_gain, req_cost))

    ranked_items.sort(key=lambda x: -x[0])
    total_active_requests = sum(c for _, _, c in ranked_items)
    total_actual_gain = sum(g for _, g, c in ranked_items if c > 0)
    utility_per_100_req = (total_actual_gain / total_active_requests * 100.0) if total_active_requests > 0 else 0.0

    # Compute normalized AUUC across K = 1..len(ranked_items)
    cum_gains: list[float] = []
    running_g = 0.0
    for _, g, _ in ranked_items:
        running_g += g
        cum_gains.append(running_g)
    max_possible_gain = sum(
        max(0.0, float(u.e2_response_vector.delta_D_vs_P or 0.0), float(u.e2_response_vector.delta_R_vs_P or 0.0)) * u.deduction_weight
        for u in adjudicated_e2
    )
    auuc = (sum(cum_gains) / (len(cum_gains) * max_possible_gain)) if (cum_gains and max_possible_gain > 0) else 0.0

    # --- Tier 2: Deduction-Weighted Macro-F1 & Weighted Accuracy ---
    classes = ("delivery_responsive", "rule_responsive", "unrecovered_under_protocol")
    total_w = sum(u.deduction_weight for u in adjudicated_e2)
    correct_w = 0.0
    f1_per_class: dict[str, float] = {}
    for u in adjudicated_e2:
        p = pred_by_id[u.unit_id]
        pred_lbl = p.get("predicted_label")
        if not pred_lbl:
            rec = p.get("recommended_arm", "none")
            pred_lbl = "delivery_responsive" if rec == "D" else ("rule_responsive" if rec == "R" else "unrecovered_under_protocol")
        if pred_lbl == u.operational_label:
            correct_w += u.deduction_weight

    for cls in classes:
        tp_w = sum(
            u.deduction_weight for u in adjudicated_e2
            if u.operational_label == cls and (
                pred_by_id[u.unit_id].get("predicted_label") == cls
                or (
                    not pred_by_id[u.unit_id].get("predicted_label")
                    and (
                        (cls == "delivery_responsive" and pred_by_id[u.unit_id].get("recommended_arm") == "D")
                        or (cls == "rule_responsive" and pred_by_id[u.unit_id].get("recommended_arm") == "R")
                        or (cls == "unrecovered_under_protocol" and pred_by_id[u.unit_id].get("recommended_arm", "none") == "none")
                    )
                )
            )
        )
        fp_w = sum(
            u.deduction_weight for u in adjudicated_e2
            if u.operational_label != cls and (
                pred_by_id[u.unit_id].get("predicted_label") == cls
            )
        )
        fn_w = sum(
            u.deduction_weight for u in adjudicated_e2
            if u.operational_label == cls and (
                pred_by_id[u.unit_id].get("predicted_label") != cls
            )
        )
        denom = 2.0 * tp_w + fp_w + fn_w
        f1_per_class[cls] = (2.0 * tp_w / denom) if denom > 0 else 0.0

    present_classes = [c for c in classes if any(u.operational_label == c for u in adjudicated_e2)]
    weighted_macro_f1 = (
        sum(f1_per_class[c] for c in present_classes) / len(present_classes)
        if present_classes else 0.0
    )
    weighted_acc = (correct_w / total_w) if total_w > 0 else 0.0

    # Selective accuracy (non-abstained)
    non_abstained = [u for u in adjudicated_e2 if not pred_by_id[u.unit_id].get("abstain", False)]
    coverage = (len(non_abstained) / len(adjudicated_e2)) if adjudicated_e2 else 0.0
    sel_w = sum(u.deduction_weight for u in non_abstained)
    sel_correct_w = sum(
        u.deduction_weight for u in non_abstained
        if pred_by_id[u.unit_id].get("predicted_label") == u.operational_label
    )
    selective_weighted_acc = (sel_correct_w / sel_w) if sel_w > 0 else 0.0

    # --- Tier 3: Brier Score & ECE on (p_P, p_D, p_R) ---
    brier_terms: list[float] = []
    ece_pairs: list[tuple[float, float]] = []
    for u in adjudicated_e2:
        p = pred_by_id[u.unit_id]
        vec = u.e2_response_vector
        if vec.mean_P is not None and vec.mean_D is not None and vec.mean_R is not None:
            for prob_key, actual_mean in (("p_prob", vec.mean_P), ("d_prob", vec.mean_D), ("r_prob", vec.mean_R)):
                pred_p = max(0.0, min(1.0, float(p.get(prob_key, 0.0))))
                act_v = max(0.0, min(1.0, float(actual_mean)))
                brier_terms.append((pred_p - act_v) ** 2)
                ece_pairs.append((pred_p, act_v))

    brier_score = (sum(brier_terms) / len(brier_terms)) if brier_terms else None
    ece_score = None
    if ece_pairs:
        n_bins = 5
        total_pts = len(ece_pairs)
        ece_val = 0.0
        for b_idx in range(n_bins):
            lo = b_idx / n_bins
            hi = (b_idx + 1) / n_bins
            bucket = [
                (pr, ac) for pr, ac in ece_pairs
                if (lo <= pr < hi) or (b_idx == n_bins - 1 and pr == 1.0)
            ]
            if bucket:
                avg_p = sum(x[0] for x in bucket) / len(bucket)
                avg_a = sum(x[1] for x in bucket) / len(bucket)
                ece_val += (len(bucket) / total_pts) * abs(avg_p - avg_a)
        ece_score = round(ece_val, 6)

    # --- Tier 4: Broken Source-Turn Accuracy & False Localization Rate ---
    broken_units = [u for u in units if u.operational_label == "broken" and u.e1_observation.source_turn is not None and u.unit_id in pred_by_id]
    exact_hits = 0
    pm1_hits = 0
    for u in broken_units:
        steps = pred_by_id[u.unit_id].get("evidence_steps") or []
        true_t = int(u.e1_observation.source_turn)  # type: ignore[arg-type]
        if steps:
            s0 = int(steps[0])
            if s0 == true_t:
                exact_hits += 1
            if abs(s0 - true_t) <= 1:
                pm1_hits += 1

    unrec_units = [u for u in adjudicated_e2 if u.operational_label == "unrecovered_under_protocol"]
    false_loc_count = sum(
        1 for u in unrec_units
        if bool(pred_by_id[u.unit_id].get("evidence_steps")) and not pred_by_id[u.unit_id].get("abstain", False)
    )

    return {
        "schema_version": V05_SCHEMA_VERSION,
        "adjudicated_e2_unit_count": len(adjudicated_e2),
        "tier1_utility": {
            "utility_per_100_requests": round(utility_per_100_req, 6),
            "auuc": round(auuc, 6),
            "total_actual_deduction_recovered": round(total_actual_gain, 6),
            "total_verification_requests": round(total_active_requests, 2),
        },
        "tier2_classification": {
            "weighted_macro_f1": round(weighted_macro_f1, 6),
            "weighted_accuracy": round(weighted_acc, 6),
            "f1_by_class": {k: round(v, 6) for k, v in f1_per_class.items()},
            "coverage": round(coverage, 6),
            "selective_weighted_accuracy": round(selective_weighted_acc, 6),
        },
        "tier3_calibration": {
            "brier_score": round(brier_score, 6) if brier_score is not None else None,
            "ece": ece_score,
        },
        "tier4_localization_and_hallucination": {
            "broken_unit_count": len(broken_units),
            "broken_source_turn_exact_acc": round(exact_hits / len(broken_units), 6) if broken_units else None,
            "broken_source_turn_pm1_acc": round(pm1_hits / len(broken_units), 6) if broken_units else None,
            "unrecovered_unit_count": len(unrec_units),
            "false_localization_rate_on_unrecovered": round(false_loc_count / len(unrec_units), 6) if unrec_units else None,
        },
    }


def build_sokoban_v05_unit(
    episode_id: str,
    *,
    terminal_passed: bool,
    max_band: int,
    terminal_band: int,
    k_star: int,
    first_deadlock_turn: int | None = None,
) -> DiagnosticUnitV05:
    """V3 §3.6.2 Sokoban Benchmark Adaptation Boundary:
    - Explicitly marks `undelivered` and arm `D` as `not_applicable`
    - Separates zero-cost BFS deadlock (`irreversible_state_damage`) from official `passed` (`band >= 1`).
    """
    y_val = 1.0 if terminal_passed else 0.0
    max_s = 1.0 if max_band >= 1 else 0.0
    ded = 1.0 - y_val

    if max_s > y_val or first_deadlock_turn is not None:
        e1_cls = "broken"
        substate = "irreversible_state_damage" if first_deadlock_turn is not None else "peak_regressed"
        op_label = "broken"
        ev_status = "adjudicated"
    else:
        e1_cls = "never_succeeded_on_disk"
        substate = "no_candidate" if k_star == 0 else "delivered_item_wrong"
        op_label = None
        ev_status = "not_applicable"

    return DiagnosticUnitV05(
        unit_id=f"{episode_id}:sokoban_pass",
        episode_id=episode_id,
        task_id="lhtb:sokoban",
        benchmark="lhtb",
        unit_level="rubric_item",
        defect_name="sokoban_official_pass_and_deadlock_free",
        deduction_weight=ded,
        rule_source="explicit",
        e1_observation=E1Observation(
            e1_class=e1_cls,
            candidate_substate=substate,
            k_star=k_star,
            max_item_score=max_s,
            terminal_item_score=y_val,
            deduction=ded,
            source_turn=first_deadlock_turn,
            deadlock_turn=first_deadlock_turn,
        ),
        e2_response_vector=E2ResponseVector(evaluated=False),
        operational_label=op_label,
        evidence_status=ev_status,
    )
