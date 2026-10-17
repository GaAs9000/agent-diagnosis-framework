"""Unit tests for L2 v0.5 Intervention-Verifiable Failure Diagnosis Protocol (Notion V3 §3.1–3.7)."""
from __future__ import annotations

import pytest

from l2_sandbox.v05 import (
    DiagnosticUnitV05,
    E1Observation,
    E2ResponseVector,
    EVIDENCE_STATUSES,
    OPERATIONAL_LABELS,
    RULE_SOURCES,
    V05_SCHEMA_VERSION,
    adjudicate_e2_outcomes,
    build_interleaved_arm_schedule,
    build_r_all_intervention_message,
    build_sokoban_v05_unit,
    evaluate_analysts_v05,
    predict_majority_baseline_m,
    predict_rule_analyst_a_v05,
    select_v05_intervention_anchors,
)


def test_v05_anchor_selection_includes_k_star_0_and_early_budget_anchor():
    # Case 1: k* = 0 (no candidate ever written across 35 steps)
    cps_no_cand = [
        {"kind": "checkpoint", "k": k, "candidate_status": "no_candidate", "score": 0.0, "cold_resume_eligible": True}
        for k in range(36)
    ]
    res0 = select_v05_intervention_anchors(cps_no_cand, max_requests=35, r_b=6)
    assert res0["eligible"] is True
    assert res0["primary_anchor_k"] == 29  # 35 - 29 = 6 remaining requests
    assert res0["primary_anchor_type"] == "budget_tail_no_candidate"
    assert res0["primary_remaining_budget"] == 6
    assert res0["early_anchor_k"] == 23  # 35 - 23 = 12 >= 2 * r_b remaining requests
    assert res0["early_remaining_budget"] == 12

    # Case 2: score-0 candidate written at k=18
    cps_score0 = [
        {
            "kind": "checkpoint",
            "k": k,
            "candidate_status": "ok" if k >= 18 else "no_candidate",
            "score": 0.0,
            "cold_resume_eligible": True,
        }
        for k in range(36)
    ]
    res_s0 = select_v05_intervention_anchors(cps_score0, max_requests=35, r_b=6)
    assert res_s0["eligible"] is True
    assert res_s0["primary_anchor_k"] == 18
    assert res_s0["primary_anchor_type"] == "candidate_first_write"
    assert res_s0["primary_remaining_budget"] == 17

    # Case 3: positive candidate peak at k=22
    cps_peak = [
        {
            "kind": "checkpoint",
            "k": k,
            "candidate_status": "ok" if k >= 15 else "no_candidate",
            "score": 0.8 if k >= 22 else (0.4 if k >= 15 else 0.0),
            "cold_resume_eligible": True,
        }
        for k in range(36)
    ]
    res_peak = select_v05_intervention_anchors(cps_peak, max_requests=35, r_b=6)
    assert res_peak["eligible"] is True
    assert res_peak["primary_anchor_k"] == 22
    assert res_peak["primary_anchor_type"] == "candidate_peak"

    # Case 4: Ineligible when no checkpoint has >= r_b remaining requests
    cps_short = [
        {"kind": "checkpoint", "k": 31, "remaining_requests": 4, "candidate_status": "no_candidate", "cold_resume_eligible": True}
    ]
    res_inel = select_v05_intervention_anchors(cps_short, max_requests=35, r_b=6)
    assert res_inel["eligible"] is False
    assert res_inel["primary_anchor_k"] is None


def test_v05_interleaved_schedule_preserves_same_checkpoint_and_budget():
    sched = build_interleaved_arm_schedule(["P", "D", "R", "R_all"], 6, anchor_k=29, remaining_budget=6)
    assert len(sched) == 24
    assert {item["checkpoint_k"] for item in sched} == {29}
    assert {item["at_request"] for item in sched} == {30}
    assert {item["remaining_budget"] for item in sched} == {6}
    for rep in range(6):
        block = sched[rep * 4 : (rep + 1) * 4]
        assert {b["arm"] for b in block} == {"P", "D", "R", "R_all"}


def test_v05_adjudicate_e2_all_five_regimes_with_delta_010_tau_020():
    # 1. Plain recoverable: P alone jumps from Y=0.0 to mean_P=0.75 (24 branches)
    vec_p, lbl_p, status_p = adjudicate_e2_outcomes(
        0.0,
        {
            "P": [1.0] * 18 + [0.0] * 6,
            "D": [1.0] * 19 + [0.0] * 5,
            "R": [1.0] * 19 + [0.0] * 5,
        },
        anchor_k=28,
        remaining_budget=7,
    )
    assert status_p == "plain_recoverable"
    assert lbl_p is None
    assert vec_p.delta_P_vs_Y == 0.75

    # 2. Delivery-responsive: P=0.05, D=0.80, R=0.80
    vec_d, lbl_d, status_d = adjudicate_e2_outcomes(
        0.0,
        {
            "P": [1.0] * 1 + [0.0] * 23,
            "D": [1.0] * 20 + [0.0] * 4,
            "R": [1.0] * 20 + [0.0] * 4,
        },
        anchor_k=29,
        remaining_budget=6,
    )
    assert status_d == "adjudicated"
    assert lbl_d == "delivery_responsive"
    assert vec_d.lb95_D_vs_P > 0.10
    assert vec_d.p_holm_D_vs_P <= 0.05

    # 3. Rule-responsive: P=0.0, D=0.04, R=0.75, R_all=0.71
    vec_r, lbl_r, status_r = adjudicate_e2_outcomes(
        0.0,
        {
            "P": [0.0] * 24,
            "D": [1.0] * 1 + [0.0] * 23,
            "R": [1.0] * 18 + [0.0] * 6,
            "R_all": [1.0] * 17 + [0.0] * 7,
        },
        anchor_k=22,
        remaining_budget=13,
    )
    assert status_r == "adjudicated"
    assert lbl_r == "rule_responsive"
    assert vec_r.lb95_R_vs_D > 0.10
    assert vec_r.p_holm_R_vs_D <= 0.05
    assert vec_r.delta_R_single_vs_R_all is not None

    # 4. Unrecovered under protocol: P=0.0, D=0.0, R=0.0 across 24 branches
    vec_u, lbl_u, status_u = adjudicate_e2_outcomes(
        0.0,
        {
            "P": [0.0] * 24,
            "D": [0.0] * 24,
            "R": [0.0] * 24,
        },
        anchor_k=29,
        remaining_budget=6,
    )
    assert status_u == "adjudicated"
    assert lbl_u == "unrecovered_under_protocol"
    assert vec_u.ub95_D_vs_P < 0.20
    assert vec_u.ub95_R_vs_P < 0.20

    # 5. Inconclusive: small sample (n=6) with 1/6 success in D -> CI straddles [0.10, 0.20]
    vec_inc, lbl_inc, status_inc = adjudicate_e2_outcomes(
        0.0,
        {
            "P": [0.0] * 6,
            "D": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "R": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        anchor_k=29,
        remaining_budget=6,
    )
    assert status_inc == "inconclusive"
    assert lbl_inc is None


def test_v05_analyst_evaluation_four_tier_metrics():
    # Build 3 adjudicated E2 units + 1 E1 broken unit
    v_d, lbl_d, st_d = adjudicate_e2_outcomes(
        0.0, {"P": [0.0] * 24, "D": [1.0] * 20 + [0.0] * 4, "R": [1.0] * 20 + [0.0] * 4}, anchor_k=29
    )
    v_r, lbl_r, st_r = adjudicate_e2_outcomes(
        0.0, {"P": [0.0] * 24, "D": [0.0] * 24, "R": [1.0] * 19 + [0.0] * 5}, anchor_k=24
    )
    v_u, lbl_u, st_u = adjudicate_e2_outcomes(
        0.0, {"P": [0.0] * 24, "D": [0.0] * 24, "R": [0.0] * 24}, anchor_k=29
    )

    units = [
        DiagnosticUnitV05(
            unit_id="u_deliv",
            episode_id="ep1",
            task_id="officeval_070",
            benchmark="officeval",
            unit_level="gate_or_no_artifact",
            defect_name="no_artifact",
            deduction_weight=1.0,
            rule_source="gate_or_no_artifact",
            e1_observation=E1Observation("never_succeeded_on_disk", "no_candidate", 0, 0.0, 0.0, 1.0),
            e2_response_vector=v_d,
            operational_label=lbl_d,
            evidence_status=st_d,
        ),
        DiagnosticUnitV05(
            unit_id="u_rule",
            episode_id="ep2",
            task_id="officeval_090",
            benchmark="officeval",
            unit_level="rubric_item",
            defect_name="IFERROR formula",
            deduction_weight=0.5,
            rule_source="implicit",
            e1_observation=E1Observation("never_succeeded_on_disk", "delivered_item_wrong", 10, 0.0, 0.0, 0.5),
            e2_response_vector=v_r,
            operational_label=lbl_r,
            evidence_status=st_r,
        ),
        DiagnosticUnitV05(
            unit_id="u_unrec",
            episode_id="ep3",
            task_id="officeval_010",
            benchmark="officeval",
            unit_level="rubric_item",
            defect_name="complex layout",
            deduction_weight=0.5,
            rule_source="ambiguous",
            e1_observation=E1Observation("never_succeeded_on_disk", "delivered_item_wrong", 20, 0.0, 0.0, 0.5),
            e2_response_vector=v_u,
            operational_label=lbl_u,
            evidence_status=st_u,
        ),
        DiagnosticUnitV05(
            unit_id="u_broken",
            episode_id="ep4",
            task_id="officeval_033",
            benchmark="officeval",
            unit_level="rubric_item",
            defect_name="price table overwritten",
            deduction_weight=0.4,
            rule_source="explicit",
            e1_observation=E1Observation("broken", "peak_regressed", 10, 1.0, 0.0, 0.4, source_turn=19),
            e2_response_vector=E2ResponseVector(evaluated=False),
            operational_label="broken",
            evidence_status="adjudicated",
        ),
    ]

    # Baseline M vs an informed Analyst C
    m_preds = predict_majority_baseline_m(units)
    m_eval = evaluate_analysts_v05(units, m_preds)
    assert m_eval["tier1_utility"]["utility_per_100_requests"] == 0.0
    assert m_eval["tier2_classification"]["weighted_accuracy"] == pytest.approx(0.5 / 2.0)

    c_preds = [
        {
            "unit_id": "u_deliv",
            "analyst_group": "C",
            "p_prob": 0.02,
            "d_prob": 0.82,
            "r_prob": 0.82,
            "recommended_arm": "D",
            "predicted_label": "delivery_responsive",
            "confidence": 0.90,
            "abstain": False,
            "evidence_steps": [28],
        },
        {
            "unit_id": "u_rule",
            "analyst_group": "C",
            "p_prob": 0.01,
            "d_prob": 0.02,
            "r_prob": 0.78,
            "recommended_arm": "R",
            "predicted_label": "rule_responsive",
            "confidence": 0.85,
            "abstain": False,
            "evidence_steps": [10],
        },
        {
            "unit_id": "u_unrec",
            "analyst_group": "C",
            "p_prob": 0.01,
            "d_prob": 0.02,
            "r_prob": 0.03,
            "recommended_arm": "none",
            "predicted_label": "unrecovered_under_protocol",
            "confidence": 0.80,
            "abstain": False,
            "evidence_steps": [],  # Does not fabricate a step on unrecovered!
        },
        {
            "unit_id": "u_broken",
            "analyst_group": "C",
            "p_prob": 0.50,
            "d_prob": 0.50,
            "r_prob": 0.50,
            "recommended_arm": "D",
            "predicted_label": "broken",
            "confidence": 0.95,
            "abstain": False,
            "evidence_steps": [19],
        },
    ]
    c_eval = evaluate_analysts_v05(units, c_preds)
    assert c_eval["tier2_classification"]["weighted_macro_f1"] == 1.0
    assert c_eval["tier2_classification"]["weighted_accuracy"] == 1.0
    assert c_eval["tier1_utility"]["utility_per_100_requests"] > 0.35
    assert c_eval["tier3_calibration"]["brier_score"] < 0.01
    assert c_eval["tier4_localization_and_hallucination"]["broken_source_turn_exact_acc"] == 1.0
    assert c_eval["tier4_localization_and_hallucination"]["false_localization_rate_on_unrecovered"] == 0.0


def test_v05_sokoban_adaptation_and_r_all_message():
    u_dead = build_sokoban_v05_unit(
        "sokoban.r01", terminal_passed=False, max_band=0, terminal_band=0, k_star=0, first_deadlock_turn=14
    )
    assert u_dead.operational_label == "broken"
    assert u_dead.e1_observation.candidate_substate == "irreversible_state_damage"
    assert u_dead.e1_observation.deadlock_turn == 14

    u_never = build_sokoban_v05_unit(
        "sokoban.r02", terminal_passed=False, max_band=0, terminal_band=0, k_star=5, first_deadlock_turn=None
    )
    assert u_never.operational_label is None
    assert u_never.evidence_status == "not_applicable"

    msg = build_r_all_intervention_message(["文档表格中没有空白行", "所有横线间距误差不超过0.15cm"])
    assert "(1) 文档表格中没有空白行" in msg
    assert "(2) 所有横线间距误差不超过0.15cm" in msg


def test_v05_ipw_recoverability_and_rule_non_leakage():
    from l2_sandbox.v05 import compute_ipw_recoverability_ratio, verify_rule_prompt_non_leakage

    clean_prompt = build_r_all_intervention_message(["文档表格中没有空白行"])
    assert verify_rule_prompt_non_leakage(clean_prompt, forbidden_tokens=["row_idx", "cell_text"]) is True
    leaky_prompt = "Check def verify_table in officeval_003_verifier.py and assert row_idx == 0"
    assert verify_rule_prompt_non_leakage(leaky_prompt, forbidden_tokens=["row_idx"]) is False

    sampled_rows = [
        {"family_id": "fam1", "deduction": 1.0, "inclusion_prob": 0.5, "recovered_by_D": True, "recovered_by_R": False, "inconclusive": False},
        {"family_id": "fam2", "deduction": 1.0, "inclusion_prob": 0.5, "recovered_by_D": False, "recovered_by_R": True, "inconclusive": False},
        {"family_id": "fam3", "deduction": 2.0, "inclusion_prob": 1.0, "recovered_by_D": False, "recovered_by_R": False, "inconclusive": True},
        {"family_id": "fam4", "deduction": 2.0, "inclusion_prob": 1.0, "recovered_by_D": False, "recovered_by_R": False, "inconclusive": False},
    ]
    # Total HT denominator = 1/0.5 + 1/0.5 + 2/1.0 + 2/1.0 = 8.0
    # D numerator = 2.0 -> lower bound = 2/8 = 0.25, upper bound (incl inconclusive 2.0) = 4/8 = 0.50
    res = compute_ipw_recoverability_ratio(sampled_rows, total_never_deduction=8.0)
    assert res["ht_denom_never_deduction"] == pytest.approx(8.0)
    assert res["delivery_responsive_share_lower_bound"] == pytest.approx(0.25)
    assert res["delivery_responsive_share_upper_bound_incl_inconclusive"] == pytest.approx(0.50)
    assert res["rule_responsive_share_lower_bound"] == pytest.approx(0.25)
    assert res["inconclusive_share"] == pytest.approx(0.25)

