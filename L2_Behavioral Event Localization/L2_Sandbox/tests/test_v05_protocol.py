"""Unit tests for L2 v0.5 Intervention-Verifiable Failure Diagnosis Protocol (Notion V3 §3.1–3.7, R7–R14b, R21)."""
from __future__ import annotations

import pytest

from l2_sandbox.io import L2Error
from l2_sandbox.v05 import (
    DEFAULT_SELF_CHECK_DRAFT,
    DiagnosticUnitV05,
    E1Observation,
    E2ResponseVector,
    EVIDENCE_STATUSES,
    LADDER_ARM_LEVELS,
    OPERATIONAL_LABELS,
    RULE_SOURCES,
    V05_SCHEMA_VERSION,
    V3_PROPOSITIONS,
    adjudicate_e2_outcomes,
    build_interleaved_arm_schedule,
    build_ladder_intervention_messages,
    build_r_all_intervention_message,
    build_sokoban_v05_unit,
    build_task_requirement_lists,
    build_v3_stratified_sampling_frame,
    compute_cross_fitted_diagnostic_value_ceiling,
    compute_ipw_recoverability_ratio,
    evaluate_analysts_v05,
    evaluate_harm_on_passing_units,
    predict_majority_baseline_m,
    predict_rule_analyst_a_v05,
    select_v05_intervention_anchors,
    sha256_text,
    verify_nested_ladder_messages,
    verify_normal_request_zero_diagnostic_leakage,
    verify_rule_prompt_non_leakage,
)


def test_r14a_v3_propositions_and_ladder_levels():
    assert V3_PROPOSITIONS == ("V3-P1", "V3-P2", "V3-P3", "V3-P4", "V3-P5")
    assert LADDER_ARM_LEVELS["P"] == 0
    assert LADDER_ARM_LEVELS["D"] == 1
    assert LADDER_ARM_LEVELS["S"] == 1
    assert LADDER_ARM_LEVELS["R_explicit"] == 2
    assert LADDER_ARM_LEVELS["R_implicit"] == 3
    assert len(DEFAULT_SELF_CHECK_DRAFT) > 20


def test_r7_r8_nested_ladder_messages_whole_task_requirements_and_empty_ineligible():
    # R8: Whole-task requirement list built from task catalog regardless of which item failed in a specific run
    catalog = {
        "officeval:officeval_070": {
            "标题居中加粗": {"source_category": "explicit"},
            "页边距上下2.54cm": {"source_category": "explicit"},
            "表格不得跨页断行": {"source_category": "implicit"},
        },
        "officeval:officeval_090": {
            "所有公式使用IFERROR包裹": {"source_category": "implicit"},
        },
    }
    req_070 = build_task_requirement_lists(catalog, "officeval_070")
    assert req_070["task_id"] == "officeval_070"
    assert len(req_070["explicit"]) == 2
    assert len(req_070["implicit"]) == 1

    # R7: Nested prefix verification when both explicit and implicit lists are non-empty
    msgs_070 = build_ladder_intervention_messages(
        explicit_requirements=req_070["explicit"],
        implicit_requirements=req_070["implicit"],
        single_rule_requirement="表格不得跨页断行",
    )
    nest_check = verify_nested_ladder_messages(msgs_070)
    assert nest_check["valid"] is True
    assert msgs_070["P"]["message"] is None
    assert len(msgs_070["D"]["message_sha256"]) == 64
    assert len(msgs_070["S"]["message_sha256"]) == 64
    assert msgs_070["R_explicit"]["message"].startswith(msgs_070["D"]["message"])
    assert msgs_070["R_implicit"]["message"].startswith(msgs_070["R_explicit"]["message"])
    assert msgs_070["R_single"]["message"].startswith(msgs_070["D"]["message"])

    # R8: When explicit requirement list is empty (e.g. officeval_090 has 0 explicit, 1 implicit),
    # R_explicit MUST be marked ineligible (message=None) and MUST NOT fall back to D's text!
    req_090 = build_task_requirement_lists(catalog, "officeval_090")
    assert len(req_090["explicit"]) == 0
    assert len(req_090["implicit"]) == 1
    msgs_090 = build_ladder_intervention_messages(
        explicit_requirements=req_090["explicit"],
        implicit_requirements=req_090["implicit"],
    )
    assert msgs_090["R_explicit"]["eligible"] is False
    assert msgs_090["R_explicit"]["evidence_status"] == "ineligible"
    assert msgs_090["R_explicit"]["message"] is None
    assert msgs_090["R_explicit"]["message"] != msgs_090["D"]["message"]
    assert msgs_090["R_implicit"]["eligible"] is True
    assert msgs_090["R_implicit"]["message"].startswith(msgs_090["D"]["message"])
    assert verify_nested_ladder_messages(msgs_090)["valid"] is True

    # Empty R_all also returns None (ineligible) instead of falling back to D
    assert build_r_all_intervention_message([]) is None


def test_r10_v05_anchor_selection_budget_tail_and_voluntary_pre_finish():
    # Case 1: k* = 0, budget exhausted (termination="request_budget") -> budget_tail_no_candidate at k=29 (35-29=6)
    cps_no_cand = [
        {"kind": "checkpoint", "k": k, "candidate_status": "no_candidate", "score": 0.0, "cold_resume_eligible": True}
        for k in range(36)
    ]
    res0 = select_v05_intervention_anchors(cps_no_cand, max_requests=35, r_b=6, termination="request_budget")
    assert res0["eligible"] is True
    assert res0["primary_anchor_k"] == 29
    assert res0["primary_anchor_type"] == "budget_tail_no_candidate"
    assert res0["primary_remaining_budget"] == 6
    assert res0["budget_tail_anchor_k"] == 29
    assert res0["voluntary_finish_anchor_k"] is None
    assert res0["early_anchor_k"] == 23  # 35 - 23 = 12 >= 2 * r_b

    # Case 2: Voluntary finish ("assistant_finished") after 14 requests with no candidate -> voluntary_pre_finish at k=14
    cps_vol = [
        {"kind": "checkpoint", "k": k, "candidate_status": "no_candidate", "score": 0.0, "cold_resume_eligible": True}
        for k in range(15)
    ]
    res_vol = select_v05_intervention_anchors(cps_vol, max_requests=35, r_b=6, termination="assistant_finished")
    assert res_vol["eligible"] is True
    assert res_vol["voluntary_finish_anchor_k"] == 14
    assert res_vol["primary_anchor_k"] == 14
    assert res_vol["primary_anchor_type"] == "voluntary_pre_finish"
    assert res_vol["primary_remaining_budget"] == 21

    # Case 3: score-0 candidate written at k=18, voluntary finish at k=20 -> primary=18, voluntary_finish_anchor_k=20
    cps_score0 = [
        {
            "kind": "checkpoint",
            "k": k,
            "candidate_status": "ok" if k >= 18 else "no_candidate",
            "score": 0.0,
            "cold_resume_eligible": True,
        }
        for k in range(21)
    ]
    res_s0 = select_v05_intervention_anchors(cps_score0, max_requests=35, r_b=6, termination="assistant_finished")
    assert res_s0["eligible"] is True
    assert res_s0["primary_anchor_k"] == 18
    assert res_s0["primary_anchor_type"] == "candidate_first_write"
    assert res_s0["voluntary_finish_anchor_k"] == 20

    # Case 4: Boundary check: remaining == r_b (eligible) vs remaining == r_b - 1 (ineligible)
    cps_exact = [
        {"kind": "checkpoint", "k": 29, "remaining_requests": 6, "candidate_status": "no_candidate", "cold_resume_eligible": True}
    ]
    assert select_v05_intervention_anchors(cps_exact, max_requests=35, r_b=6)["eligible"] is True
    cps_short = [
        {"kind": "checkpoint", "k": 30, "remaining_requests": 5, "candidate_status": "no_candidate", "cold_resume_eligible": True}
    ]
    assert select_v05_intervention_anchors(cps_short, max_requests=35, r_b=6)["eligible"] is False


def test_r9_adjudicate_e2_nested_ladder_regimes_and_edge_cases():
    # 1. Plain recoverable: P alone jumps from Y=0.0 to mean_P=0.75 (24 branches), D=0.75, S=0.75
    vec_p, lbl_p, status_p = adjudicate_e2_outcomes(
        0.0,
        {
            "P": [1.0] * 18 + [0.0] * 6,
            "D": [1.0] * 18 + [0.0] * 6,
            "S": [1.0] * 18 + [0.0] * 6,
            "R_explicit": [1.0] * 18 + [0.0] * 6,
        },
        anchor_k=28,
        remaining_budget=7,
    )
    assert status_p == "plain_recoverable"
    assert lbl_p is None
    assert vec_p.delta_P_vs_Y == 0.75

    # 2. Edge case: All-1 in P, D, S -> plain_recoverable
    vec_all1, lbl_all1, status_all1 = adjudicate_e2_outcomes(
        0.0,
        {"P": [1.0] * 24, "D": [1.0] * 24, "S": [1.0] * 24},
        anchor_k=28,
        remaining_budget=7,
    )
    assert status_all1 == "plain_recoverable"
    assert lbl_all1 is None

    # 3. Delivery-responsive (level 1 D): P=0.04, D=0.83, S=0.83
    vec_d, lbl_d, status_d = adjudicate_e2_outcomes(
        0.0,
        {
            "P": [1.0] * 1 + [0.0] * 23,
            "D": [1.0] * 20 + [0.0] * 4,
            "S": [1.0] * 20 + [0.0] * 4,
        },
        anchor_k=29,
        remaining_budget=6,
    )
    assert status_d == "adjudicated"
    assert lbl_d == "delivery_responsive"
    assert vec_d.lb95_D_vs_P > 0.10
    assert vec_d.p_holm_D_vs_P <= 0.05

    # 4. Self-check-responsive (level 1 S): P=0.0, D=0.04, S=0.79, R_explicit=0.79
    vec_s, lbl_s, status_s = adjudicate_e2_outcomes(
        0.0,
        {
            "P": [0.0] * 24,
            "D": [1.0] * 1 + [0.0] * 23,
            "S": [1.0] * 19 + [0.0] * 5,
            "R_explicit": [1.0] * 19 + [0.0] * 5,
        },
        anchor_k=14,
        remaining_budget=21,
    )
    assert status_s == "adjudicated"
    assert lbl_s == "self_check_responsive"
    assert vec_s.pairwise_lb95["S_vs_P"] > 0.10

    # 5. Explicit-rule-responsive (level 2 R_explicit): must beat P, D, AND S
    vec_re, lbl_re, status_re = adjudicate_e2_outcomes(
        0.0,
        {
            "P": [0.0] * 24,
            "D": [0.0] * 24,
            "S": [1.0] * 1 + [0.0] * 23,
            "R_explicit": [1.0] * 19 + [0.0] * 5,
            "R_implicit": [1.0] * 19 + [0.0] * 5,
        },
        anchor_k=20,
        remaining_budget=15,
    )
    assert status_re == "adjudicated"
    assert lbl_re == "explicit_rule_responsive"
    assert vec_re.pairwise_lb95["R_explicit_vs_S"] > 0.10

    # 6. Implicit-rule-responsive (level 3 R_implicit): must beat P, D, S, AND R_explicit
    vec_ri, lbl_ri, status_ri = adjudicate_e2_outcomes(
        0.0,
        {
            "P": [0.0] * 24,
            "D": [0.0] * 24,
            "S": [0.0] * 24,
            "R_explicit": [1.0] * 1 + [0.0] * 23,
            "R_implicit": [1.0] * 19 + [0.0] * 5,
            "R_single": [1.0] * 1 + [0.0] * 23,
            "R_all": [1.0] * 1 + [0.0] * 23,
        },
        anchor_k=20,
        remaining_budget=15,
    )
    assert status_ri == "adjudicated"
    assert lbl_ri == "implicit_rule_responsive"
    assert vec_ri.pairwise_lb95["R_implicit_vs_R_explicit"] > 0.10

    # 7. Edge case: All-0 across P, D, S, R_explicit, R_implicit -> unrecovered_under_protocol
    vec_u, lbl_u, status_u = adjudicate_e2_outcomes(
        0.0,
        {
            "P": [0.0] * 24,
            "D": [0.0] * 24,
            "S": [0.0] * 24,
            "R_explicit": [0.0] * 24,
            "R_implicit": [0.0] * 24,
        },
        anchor_k=29,
        remaining_budget=6,
    )
    assert status_u == "adjudicated"
    assert lbl_u == "unrecovered_under_protocol"
    assert vec_u.arm_ub95_abs["P"] < 0.20

    # 8. Critical guard: If P has moderate success (e.g. 3/24 = 0.125, UB95(P)=0.289 > tau=0.20)
    # and D, S also have 3/24, it MUST be inconclusive, NOT unrecovered_under_protocol!
    vec_mod, lbl_mod, status_mod = adjudicate_e2_outcomes(
        0.0,
        {
            "P": [1.0] * 3 + [0.0] * 21,
            "D": [1.0] * 3 + [0.0] * 21,
            "S": [1.0] * 3 + [0.0] * 21,
        },
        anchor_k=29,
        remaining_budget=6,
        delta=0.10,
        tau=0.20,
    )
    assert status_mod == "inconclusive"
    assert lbl_mod is None


def test_r11_3d_stratified_sampling_frame_and_ht_hajek_ipw():
    # Rejects legacy deepseek datasets
    with pytest.raises(L2Error, match="R11 violation"):
        build_v3_stratified_sampling_frame([], dataset_id="deepseek_dev45_20261015")

    units: list[DiagnosticUnitV05] = []
    for i in range(10):
        units.append(
            DiagnosticUnitV05(
                unit_id=f"u_ex_{i}",
                episode_id=f"ep_{i}",
                task_id=f"officeval_{i:03d}",
                benchmark="officeval",
                unit_level="rubric_item",
                defect_name="rule_ex",
                deduction_weight=1.0,
                rule_source="explicit",
                e1_observation=E1Observation("never_succeeded_on_disk", "no_candidate", 0, 0.0, 0.0, 1.0),
                e2_response_vector=E2ResponseVector(evaluated=False, anchor_k=29),
                operational_label=None,
                evidence_status="inconclusive",
                exhausted_budget=True,
                valid_delivery=False,
            )
        )
    for i in range(4):
        units.append(
            DiagnosticUnitV05(
                unit_id=f"u_im_{i}",
                episode_id=f"ep_im_{i}",
                task_id=f"officeval_1{i:02d}",
                benchmark="officeval",
                unit_level="rubric_item",
                defect_name="rule_im",
                deduction_weight=0.5,
                rule_source="implicit",
                e1_observation=E1Observation("never_succeeded_on_disk", "delivered_item_wrong", 10, 0.0, 0.0, 0.5),
                e2_response_vector=E2ResponseVector(evaluated=False, anchor_k=14),
                operational_label=None,
                evidence_status="inconclusive",
                exhausted_budget=False,
                valid_delivery=True,
            )
        )

    frame = build_v3_stratified_sampling_frame(
        units,
        dataset_id="v3_qwen36_pi_dev45_20261017",
        actor_model="Qwen/Qwen3.6-27B",
        target_sample_per_stratum=5,
    )
    assert frame["stratification_axes"] == ["exhausted_budget", "valid_delivery", "rule_source"]
    assert frame["total_eligible_never_units"] == 14
    assert frame["total_sampled_units"] == 9  # 5 from first stratum (pi=0.5), 4 from second (pi=1.0)

    sampled_rows = []
    for idx, row in enumerate(frame["sampled_units"]):
        r = dict(row)
        r["deduction"] = r["deduction_weight"]
        r["recovered_by_D"] = idx < 3
        r["recovered_by_S"] = idx == 3
        r["recovered_by_R_explicit"] = idx == 4
        r["recovered_by_R_implicit"] = False
        r["inconclusive"] = False
        sampled_rows.append(r)

    ipw = compute_ipw_recoverability_ratio(sampled_rows, total_never_deduction=12.0)
    assert ipw["ht_denom_never_deduction"] == pytest.approx(12.0)
    assert "ht_shares" in ipw and "hajek_shares" in ipw
    assert ipw["ht_shares"]["level1_D_or_S"] == pytest.approx(ipw["hajek_shares"]["level1_D_or_S"])


def test_r12_harm_evaluation_on_passing_units():
    passing_rows = [
        {
            "unit_id": "pass_1",
            "arm_outcomes": {
                "P": [1.0] * 24,
                "D": [1.0] * 24,
                "S": [1.0] * 24,
                "R_explicit": [1.0] * 23 + [0.0] * 1,
            },
        },
        {
            "unit_id": "pass_2",
            "arm_outcomes": {
                "P": [1.0] * 24,
                "D": [1.0] * 24,
                "S": [1.0] * 24,
                "R_explicit": [0.0] * 20 + [1.0] * 4,  # severe harm
            },
        },
    ]
    res = evaluate_harm_on_passing_units(passing_rows)
    assert res["evaluated_unit_count"] == 2
    assert res["summary_by_arm"]["D"]["significant_harm_count"] == 0
    assert res["summary_by_arm"]["S"]["significant_harm_count"] == 0
    assert res["summary_by_arm"]["R_explicit"]["significant_harm_count"] == 1


def test_r13_cross_fitted_diagnostic_value_ceiling_disjoint_halves():
    # Homogeneous units: D, S, R_explicit all have identical gain +0.8 over P -> ceiling = 0.0
    hom_units: list[DiagnosticUnitV05] = []
    for i in range(6):
        vec, lbl, st = adjudicate_e2_outcomes(
            0.0,
            {
                "P": [0.0] * 12,
                "D": [0.8] * 12,
                "S": [0.8] * 12,
                "R_explicit": [0.8] * 12,
            },
            anchor_k=20,
        )
        hom_units.append(
            DiagnosticUnitV05(
                unit_id=f"u_{i}",
                episode_id=f"ep_{i}",
                task_id=f"officeval_{i:03d}",
                benchmark="officeval",
                unit_level="rubric_item",
                defect_name="d",
                deduction_weight=1.0 if i % 2 == 0 else 0.5,
                rule_source="explicit",
                e1_observation=E1Observation("never_succeeded_on_disk", "no_candidate", 0, 0.0, 0.0, 1.0),
                e2_response_vector=vec,
                operational_label=lbl,
                evidence_status=st,
            )
        )
    ceiling = compute_cross_fitted_diagnostic_value_ceiling(hom_units, ceiling_threshold=0.05)
    assert ceiling["folds_disjoint"] is True
    assert set(ceiling["folds"][0]["train_unit_ids"]).isdisjoint(set(ceiling["folds"][0]["eval_unit_ids"]))
    assert ceiling["unweighted_ceiling"] == pytest.approx(0.0)
    assert ceiling["deduction_weighted_ceiling"] == pytest.approx(0.0)
    assert ceiling["passes_v3_p3_gate"] is False

    # Heterogeneous units: half respond ONLY to D, half respond ONLY to R_explicit -> ceiling = +0.45 >= 0.05
    het_units: list[DiagnosticUnitV05] = []
    for i in range(8):
        is_d_unit = i % 2 == 0
        vec, lbl, st = adjudicate_e2_outcomes(
            0.0,
            {
                "P": [0.0] * 12,
                "D": [0.9] * 12 if is_d_unit else [0.0] * 12,
                "S": [0.0] * 12,
                "R_explicit": [0.0] * 12 if is_d_unit else [0.9] * 12,
            },
            anchor_k=20,
        )
        het_units.append(
            DiagnosticUnitV05(
                unit_id=f"het_{i}",
                episode_id=f"ep_{i}",
                task_id=f"officeval_{i:03d}",
                benchmark="officeval",
                unit_level="rubric_item",
                defect_name="d",
                deduction_weight=1.0,
                rule_source="explicit",
                e1_observation=E1Observation("never_succeeded_on_disk", "no_candidate", 0, 0.0, 0.0, 1.0),
                e2_response_vector=vec,
                operational_label=lbl,
                evidence_status=st,
            )
        )
    het_ceiling = compute_cross_fitted_diagnostic_value_ceiling(het_units, ceiling_threshold=0.05)
    assert het_ceiling["unweighted_ceiling"] == pytest.approx(0.45)
    assert het_ceiling["deduction_weighted_ceiling"] == pytest.approx(0.45)
    assert het_ceiling["passes_v3_p3_gate"] is True


def test_r21_normal_request_zero_diagnostic_leakage():
    clean_req = {"messages": [{"role": "user", "content": "请根据 workspace 中的数据生成季度报表 docx。"}]}
    res_clean = verify_normal_request_zero_diagnostic_leakage(clean_req)
    assert res_clean["zero_leakage"] is True
    assert res_clean["leaked_markers"] == []

    leaky_req = {"messages": [{"role": "user", "content": "请同时逐条核对以下任务明确要求：officeval_070_verifier.py"}]}
    res_leaky = verify_normal_request_zero_diagnostic_leakage(leaky_req)
    assert res_leaky["zero_leakage"] is False
    assert len(res_leaky["leaked_markers"]) >= 1

    clean_prompt = build_r_all_intervention_message(["文档表格中没有空白行"])
    assert clean_prompt is not None
    assert verify_rule_prompt_non_leakage(clean_prompt, forbidden_tokens=["row_idx", "cell_text"]) is True
    leaky_prompt = "Check def verify_table in officeval_003_verifier.py and assert row_idx == 0"
    assert verify_rule_prompt_non_leakage(leaky_prompt, forbidden_tokens=["row_idx"]) is False


def test_v05_analyst_evaluation_four_tier_metrics_without_insample_ceiling():
    v_d, lbl_d, st_d = adjudicate_e2_outcomes(
        0.0, {"P": [0.0] * 24, "D": [1.0] * 20 + [0.0] * 4, "S": [1.0] * 20 + [0.0] * 4}, anchor_k=29
    )
    v_r, lbl_r, st_r = adjudicate_e2_outcomes(
        0.0,
        {"P": [0.0] * 24, "D": [0.0] * 24, "S": [0.0] * 24, "R_explicit": [1.0] * 19 + [0.0] * 5},
        anchor_k=24,
    )
    v_u, lbl_u, st_u = adjudicate_e2_outcomes(
        0.0, {"P": [0.0] * 24, "D": [0.0] * 24, "S": [0.0] * 24, "R_explicit": [0.0] * 24}, anchor_k=29
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
            rule_source="explicit",
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

    m_preds = predict_majority_baseline_m(units)
    m_eval = evaluate_analysts_v05(units, m_preds)
    assert m_eval["tier1_utility"]["utility_per_100_requests"] == 0.0
    assert "max_possible_gain" not in m_eval["tier1_utility"]
    assert "cross_fitted_ceiling" in m_eval["tier1_utility"]

    c_preds = [
        {
            "unit_id": "u_deliv",
            "analyst_group": "C",
            "p_prob": 0.02,
            "d_prob": 0.82,
            "s_prob": 0.82,
            "r_explicit_prob": 0.82,
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
            "s_prob": 0.02,
            "r_explicit_prob": 0.78,
            "r_prob": 0.78,
            "recommended_arm": "R_explicit",
            "predicted_label": "explicit_rule_responsive",
            "confidence": 0.85,
            "abstain": False,
            "evidence_steps": [10],
        },
        {
            "unit_id": "u_unrec",
            "analyst_group": "C",
            "p_prob": 0.01,
            "d_prob": 0.02,
            "s_prob": 0.02,
            "r_explicit_prob": 0.03,
            "r_prob": 0.03,
            "recommended_arm": "none",
            "predicted_label": "unrecovered_under_protocol",
            "confidence": 0.80,
            "abstain": False,
            "evidence_steps": [],
        },
        {
            "unit_id": "u_broken",
            "analyst_group": "C",
            "p_prob": 0.50,
            "d_prob": 0.50,
            "s_prob": 0.50,
            "r_explicit_prob": 0.50,
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


def test_v05_sokoban_adaptation_boundary():
    u_dead = build_sokoban_v05_unit(
        "sokoban.r01", terminal_passed=False, max_band=0, terminal_band=0, k_star=0, first_deadlock_turn=14
    )
    # R4: Static corner deadlock without band regression is reported as independent_state_damage, NOT broken!
    assert u_dead.operational_label is None
    assert u_dead.e1_observation.e1_class == "never_succeeded_on_disk"
    assert u_dead.e1_observation.independent_state_damage == "corner_deadlock_detected"
    assert u_dead.e1_observation.deadlock_turn == 14

    u_reg = build_sokoban_v05_unit(
        "sokoban.r02", terminal_passed=False, max_band=1, terminal_band=0, k_star=5, first_deadlock_turn=8
    )
    assert u_reg.operational_label == "broken"
    assert u_reg.e1_observation.e1_class == "broken"
