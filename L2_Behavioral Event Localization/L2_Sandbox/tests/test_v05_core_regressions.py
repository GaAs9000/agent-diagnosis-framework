"""Offline counterexamples for the V3 protocol boundaries; no execution claims."""
from dataclasses import replace

import pytest

from l2_sandbox.io import L2Error
from l2_sandbox.v05 import (
    DiagnosticUnitV05, E1Observation, E2ResponseVector, _mean_bounds,
    adjudicate_e2_outcomes, build_task_requirement_lists,
    compute_cross_fitted_diagnostic_value_ceiling, evaluate_analysts_v05,
    select_v05_intervention_anchors, compute_ipw_recoverability_ratio,
)


def unit(uid, outcomes, **changes):
    value = DiagnosticUnitV05(
        unit_id=uid, episode_id=uid, task_id=uid, family_id=uid, benchmark="officeval",
        unit_level="rubric_item", defect_name="rule", deduction_weight=1.0,
        rule_source="explicit", e1_observation=E1Observation("never_succeeded_on_disk", "no_candidate", 0, 0, 0, 1),
        e2_response_vector=E2ResponseVector(evaluated=True, arm_outcomes=outcomes),
        operational_label=None, evidence_status="inconclusive",
    )
    return replace(value, **changes)


def test_exact_binary_bounds_do_not_collapse_on_all_zero_or_one():
    low, high = _mean_bounds([0.0] * 24, .05)
    assert low == 0
    assert high == pytest.approx(1 - .05**(1/24))
    low, high = _mean_bounds([1.0] * 24, .05)
    assert high == 1
    assert low == pytest.approx(.05**(1/24))


def test_global_family_widens_intervals_and_delta_null_is_not_zero_null():
    outcomes = {"P": [0.0] * 120, "D": [1.0] * 18 + [0.0] * 102}
    local, _, _ = adjudicate_e2_outcomes(0, outcomes, anchor_k=1)
    frozen, _, _ = adjudicate_e2_outcomes(0, outcomes, anchor_k=1, primary_family_size=500, comparison_ids=["u:D-P"])
    assert frozen.pairwise_lb95["D_vs_P"] < local.pairwise_lb95["D_vs_P"]
    assert frozen.inference["scope"] == "frozen_family"
    assert frozen.pairwise_p_holm["D_vs_P"] >= local.pairwise_p_holm["D_vs_P"]
    near, label, _ = adjudicate_e2_outcomes(0, {"P": [0.0]*100, "D": [1.0]*10 + [0.0]*90}, anchor_k=1)
    assert label != "delivery_responsive"
    assert near.pairwise_p_raw["D_vs_P"] > .05


def test_latest_anchor_uses_real_budget_even_if_an_early_candidate_scored_better():
    rows = [{"kind": "checkpoint", "k": k, "remaining_requests": 100-k,
             "candidate_status": "ok", "score": 1 if k == 2 else 0} for k in range(15)]
    anchor = select_v05_intervention_anchors(rows, termination="assistant_finished")
    assert anchor["primary_anchor_k"] == 14
    assert anchor["primary_remaining_budget"] == 86
    assert anchor["primary_anchor_type"] == "voluntary_pre_finish"


def test_unreviewed_ambiguous_text_never_becomes_task_visible_information():
    result = build_task_requirement_lists({"t": {"hidden formula IFERROR": {"source_category": "ambiguous"}}}, "t")
    assert result["explicit"] == result["implicit"] == []
    assert result["unresolved_rules"] == ["hidden formula IFERROR"]
    assert not result["review_complete"]
    reviewed = build_task_requirement_lists({"t": {"r": {
        "source_category": "ambiguous", "visible_requirement": "visible", "exact_verifier_specification": "hidden",
        "reviewer": "person", "reviewed_at": "2026-10-17", "source_citation": "task.md:2",
    }}}, "t")
    assert reviewed["review_complete"]
    assert reviewed["explicit"] == ["visible"]
    assert reviewed["implicit"] == ["hidden"]


def test_both_policies_select_on_same_half_and_swap_without_gate_from_two_families():
    units = [unit("a", {"P": [0, 0], "D": [1, 0], "S": [0, 1]}),
             unit("b", {"P": [0, 0], "D": [1, 0], "S": [0, 1]})]
    result = compute_cross_fitted_diagnostic_value_ceiling(units)
    assert result["folds"][0]["best_fixed_arm_weighted"] == "D"
    assert result["folds"][1]["best_fixed_arm_weighted"] == "S"
    assert result["deduction_weighted_ceiling"] == 0
    assert result["deduction_weighted_lb95"] < .05
    assert not result["passes_v3_p3_gate"]
    with pytest.raises(L2Error):
        compute_cross_fitted_diagnostic_value_ceiling(units, candidate_arms=["P", "R_implicit"])


def test_missing_prediction_and_inconclusive_harm_stay_in_policy_report():
    measured = unit("measured", {"P": [1, 1], "D": [0, 0]},
                    e2_response_vector=E2ResponseVector(evaluated=True, arm_means={"P": 1, "D": 0}))
    missing = unit("missing", {})
    report = evaluate_analysts_v05([measured, missing], [{"unit_id": "measured", "recommended_arm": "D", "verification_requests": 3}])
    assert report["missing_prediction_unit_count"] == 1
    assert report["inconclusive_unit_count"] == 2
    assert report["tier2_classification"]["full_sample_coverage"] == .5
    assert report["tier1_utility"]["total_actual_deduction_recovered"] == -1
    assert report["tier1_utility"]["total_verification_requests"] == 3
    abstained = evaluate_analysts_v05([measured], [{"unit_id": "measured", "recommended_arm": "D", "abstain": True}])
    assert abstained["tier1_utility"]["total_actual_deduction_recovered"] == 0
    assert abstained["tier1_utility"]["total_verification_requests"] == 0


def test_p_or_d_or_s_reports_union_and_uncertain_upper_bound():
    report = compute_ipw_recoverability_ratio([
        {"task_id": "a", "deduction": 1, "inclusion_prob": 1, "recovered_by_P": True},
        {"task_id": "b", "deduction": 1, "inclusion_prob": 1, "recovered_by_S": True},
        {"task_id": "c", "deduction": 1, "inclusion_prob": 1, "inconclusive": True},
    ])
    assert report["level0_1_combined_share_lower_bound"] == pytest.approx(2/3, abs=1e-6)
    assert report["level0_1_combined_share_upper_bound_incl_inconclusive"] == 1
    assert "family_cluster_ci95_level0_1_combined" in report


def test_recovery_and_negative_decisions_require_bounds_on_their_own_thresholds():
    # A point mean above tau does not establish absolute recovery.
    _, _, status = adjudicate_e2_outcomes(0, {"P": [1.0]*6 + [0.0]*18, "D": [1.0]*6 + [0.0]*18}, anchor_k=1)
    assert status == "inconclusive"
    # All absolute rates are below tau, but delta=.10 lies inside the gain CI.
    outcomes = {"P": [1.0]*10 + [0.0]*990, "D": [1.0]*110 + [0.0]*890}
    vec, label, status = adjudicate_e2_outcomes(0, outcomes, anchor_k=1)
    assert max(vec.arm_ub95_abs.values()) < .20
    assert vec.pairwise_lb95["D_vs_P"] < .10 < vec.pairwise_ub95["D_vs_P"]
    assert label is None and status == "inconclusive"


def test_unknown_semantic_family_cannot_open_diagnostic_gate():
    items = [unit(str(i), {"P": [0]*4, "D": [1]*4}, family_id=None) for i in range(4)]
    result = compute_cross_fitted_diagnostic_value_ceiling(items)
    assert result["family_ids_complete"] is False
    assert result["family_cluster_count"] == 1
    assert result["passes_v3_p3_gate"] is False


def test_unmeasured_rule_arm_does_not_borrow_another_arms_gain():
    measured = unit("r", {}, e2_response_vector=E2ResponseVector(
        evaluated=True, arm_means={"P": .1, "R_explicit": .8}, delta_R_vs_P=.7))
    report = evaluate_analysts_v05([measured], [{"unit_id": "r", "recommended_arm": "R_single"}])
    assert report["tier1_utility"]["unmeasured_policy_unit_count"] == 1
    assert report["tier1_utility"]["total_actual_deduction_recovered"] == 0


def test_no_sample_reports_uncertainty_instead_of_zero_width_recovery_interval():
    report = compute_ipw_recoverability_ratio([])
    assert report["interval_status"] == "no_valid_sample"
    assert report["family_cluster_ci95_level0_1_combined"] == [0, 1]
    assert report["level0_1_combined_share_upper_bound_incl_inconclusive"] == 1


def test_ladder_verifier_requires_canonical_identity_and_complete_spec():
    from l2_sandbox.v05 import build_ladder_intervention_messages, verify_nested_ladder_messages
    ladder = build_ladder_intervention_messages(explicit_requirements=["visible"])
    ladder["S"]["level"] = 0
    assert not verify_nested_ladder_messages(ladder)["valid"]
    ladder = build_ladder_intervention_messages()
    del ladder["S"]
    assert not verify_nested_ladder_messages(ladder)["valid"]


def test_single_rule_success_is_auxiliary_and_cannot_adjudicate_a_ladder_level():
    vec, label, status = adjudicate_e2_outcomes(0, {
        "P": [0]*160, "D": [0]*160, "S": [0]*160,
        "R_explicit": [0]*160, "R_single": [1]*160,
    }, anchor_k=1)
    assert label == "unrecovered_under_protocol"
    assert "R_single_vs_P" in vec.pairwise_deltas
    assert "R_single_vs_S" not in vec.pairwise_deltas


def test_missing_configured_fixed_policy_is_disclosed_and_blocks_gate():
    result = compute_cross_fitted_diagnostic_value_ceiling([
        unit("a", {"P": [0, 0], "D": [1, 1]}),
        unit("b", {"P": [0, 0], "D": [1, 1]}),
    ])
    assert not result["fixed_policy_pool_complete"]
    assert set(result["missing_fixed_arms"]) == {"S", "R_explicit"}
    assert result["evidence_status"] == "inconclusive"
    assert not result["passes_v3_p3_gate"]


def test_hidden_recommendation_has_no_deployable_utility_but_retains_calibration():
    hidden = unit("hidden", {}, rule_source="implicit", evidence_status="adjudicated",
                  operational_label="implicit_rule_responsive",
                  e2_response_vector=E2ResponseVector(evaluated=True,
                      arm_means={"P": 0, "R_implicit": 1}, pairwise_deltas={"R_implicit_vs_P": 1}))
    report = evaluate_analysts_v05([hidden], [{"unit_id": "hidden", "recommended_arm": "R_implicit",
                                            "r_implicit_prob": 1, "p_prob": 0}])
    assert report["tier1_utility"]["inadmissible_policy_unit_count"] == 1
    assert report["tier1_utility"]["measured_policy_unit_count"] == 0
    assert not report["tier1_utility"]["primary_policy_evaluation_complete"]
    assert report["tier1_utility"]["total_verification_requests"] == 0
    assert report["tier3_calibration"]["brier_score"] == 0


def test_single_rule_deployable_only_for_explicit_source_and_measured_arm():
    explicit = unit("explicit", {}, e2_response_vector=E2ResponseVector(evaluated=True,
        arm_means={"P": 0, "R_single": 1}, pairwise_deltas={"R_single_vs_P": 1}))
    allowed = evaluate_analysts_v05([explicit], [{"unit_id": "explicit", "recommended_arm": "R_single", "verification_requests": 4}])
    assert allowed["tier1_utility"]["total_actual_deduction_recovered"] == 1
    assert allowed["tier1_utility"]["inadmissible_policy_unit_count"] == 0
    hidden = replace(explicit, rule_source="implicit")
    blocked = evaluate_analysts_v05([hidden], [{"unit_id": "explicit", "recommended_arm": "R_single"}])
    assert blocked["tier1_utility"]["inadmissible_policy_unit_count"] == 1
    generic = evaluate_analysts_v05([explicit], [{"unit_id": "explicit", "recommended_arm": "R"}])
    assert generic["tier1_utility"]["inadmissible_policy_unit_count"] == 1


def test_rule_a_abstains_from_hidden_policy_while_preserving_attribution():
    from l2_sandbox.v05 import predict_rule_analyst_a_v05
    hidden = unit("hidden", {}, rule_source="implicit")
    prediction = predict_rule_analyst_a_v05(hidden, [{"type": "submission_artifact_Step", "step": 2}])
    assert prediction["abstain"] and prediction["recommended_arm"] == "P"
    assert prediction["predicted_label"] == "implicit_rule_responsive"
    assert prediction["predicted_attribution_arm"] == "R_implicit"
