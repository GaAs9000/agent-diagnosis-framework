"""Independent offline regression probes for the R1-R21 acceptance contract.

Synthetic outcomes establish software behavior only; no model/scorer execution.
"""
from copy import deepcopy

import pytest

from l2_sandbox.io import L2Error
from l2_sandbox.v05 import (
    DiagnosticUnitV05, E1Observation, E2ResponseVector,
    build_ladder_intervention_messages, verify_nested_ladder_messages,
    select_v05_intervention_anchors, build_v3_stratified_sampling_frame,
    compute_cross_fitted_diagnostic_value_ceiling, evaluate_analysts_v05,
)


def unit(i, outcomes=None, *, status="adjudicated", source="explicit", weight=1.0):
    outcomes = outcomes or {"P": [0., 0.], "D": [1., 1.]}
    return DiagnosticUnitV05(
        unit_id=f"independent_{i}", episode_id=f"ep_{i}", task_id=f"task_{i}",
        benchmark="synthetic_independent", unit_level="rubric_item", defect_name="fixture",
        deduction_weight=weight, rule_source=source,
        e1_observation=E1Observation("never_succeeded_on_disk", "wrong_candidate", 0, 0., 0., 1.),
        e2_response_vector=E2ResponseVector(evaluated=True, arm_outcomes=outcomes,
            arm_means={a: sum(v)/len(v) for a,v in outcomes.items()},
            pairwise_deltas={"D_vs_P": sum(outcomes["D"])/len(outcomes["D"])-sum(outcomes["P"])/len(outcomes["P"])},
            anchor_k=9, remaining_budget_at_anchor=26, anchor_type="voluntary_pre_finish"),
        operational_label="delivery_responsive" if status=="adjudicated" else None,
        evidence_status=status, exhausted_budget=False, valid_delivery=True,
    )


@pytest.mark.parametrize("arm", ["D", "S", "R_explicit", "R_implicit", "R_single"])
def test_independent_all_message_hash_leaves_reject_tampering(arm):
    messages=build_ladder_intervention_messages(explicit_requirements=["PUBLIC"],
        implicit_requirements=["PRIVATE"], single_rule_requirement="SINGLE")
    assert verify_nested_ladder_messages(messages)["valid"]
    messages[arm]["message"] += "\nUNSIGNED_APPEND"
    assert not verify_nested_ladder_messages(messages)["valid"]


def test_independent_self_declared_parent_cannot_redefine_protocol():
    messages=build_ladder_intervention_messages(explicit_requirements=["PUBLIC"], implicit_requirements=["PRIVATE"])
    messages["R_implicit"]["prefix_arm"]="D"
    assert not verify_nested_ladder_messages(messages)["valid"]


def test_independent_actual_budget_wins_over_checkpoint_number_and_survives_sampling():
    rows=[{"kind":"checkpoint", "k":900, "remaining_requests":26, "cold_resume_eligible":True,
           "candidate_status":"ok", "score":0}]
    anchor=select_v05_intervention_anchors(rows,max_requests=35,termination="assistant_finished")
    assert anchor["primary_anchor_k"]==900
    assert anchor["primary_remaining_budget"]==26
    u=unit(1); u.e2_response_vector.anchor_k=900
    sampled=build_v3_stratified_sampling_frame([u],dataset_id="new_qwen36_pi_dev45")["sampled_units"][0]
    assert (sampled["anchor_k"], sampled["remaining_budget"])==(900,26)


def test_independent_singletons_never_open_d3_gate():
    result=compute_cross_fitted_diagnostic_value_ceiling([
        unit(1,{"P":[0.],"D":[1.]}), unit(2,{"P":[0.],"D":[0.]})])
    assert result["eligible_unit_count"]==0
    assert result["passes_v3_p3_gate"] is False


@pytest.mark.parametrize("kwargs", [dict(candidate_arms=("P","R_implicit")),dict(fixed_arm_pool=("P","R_single")),dict(candidate_arms=("P","R"))])
def test_independent_hidden_or_unclassified_arms_rejected(kwargs):
    with pytest.raises(L2Error):
        compute_cross_fitted_diagnostic_value_ceiling([unit(1)],**kwargs)


def test_independent_large_point_gain_sparse_families_does_not_open_gate():
    units=[unit(1,{"P":[0.,0.],"D":[1.,1.],"S":[0.,0.]}),
           unit(2,{"P":[0.,0.],"D":[0.,0.],"S":[1.,1.]})]
    result=compute_cross_fitted_diagnostic_value_ceiling(units)
    assert result["deduction_weighted_ceiling"]==.5
    assert result["deduction_weighted_lb95"]<=.05
    assert result["passes_v3_p3_gate"] is False
    for row in result["unit_evaluations"]:
        for split in row["branch_split_indices"].values():
            assert set(split["selection"]).isdisjoint(split["evaluation"])


def test_independent_abstention_costs_zero_and_signed_harm_is_retained():
    u=unit(1,{"P":[1.,1.],"D":[0.,0.]})
    abstained=evaluate_analysts_v05([u],[{"unit_id":u.unit_id,"recommended_arm":"D","abstain":True}])
    assert abstained["tier1_utility"]["total_verification_requests"]==0
    assert abstained["tier1_utility"]["total_actual_deduction_recovered"]==0
    harmful=evaluate_analysts_v05([u],[{"unit_id":u.unit_id,"recommended_arm":"D"}])
    assert harmful["tier1_utility"]["total_actual_deduction_recovered"]==-1


def test_independent_full_coverage_keeps_missing_and_unresolved_units():
    units=[unit(1),unit(2,status="inconclusive"),unit(3,status="ineligible"),unit(4)]
    result=evaluate_analysts_v05(units,[{"unit_id":units[0].unit_id,"recommended_arm":"D"},
        {"unit_id":units[2].unit_id,"recommended_arm":"D","abstain":True}])
    assert result["missing_prediction_unit_count"]==2
    assert result["tier2_classification"]["full_sample_coverage"]==.25
    # Evidence-status census must describe all input units, independent of prediction availability.
    assert result["inconclusive_unit_count"]==1
    assert result["not_applicable_unit_count"]==1


def test_independent_default_d1_d2_d3_modes_issue_zero_external_calls(tmp_path,monkeypatch):
    import sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root/'scripts'))
    import v3_d1_ladder as d1
    import v3_d2_ceiling as d2
    import v3_d3_analysts as d3
    from c3_sandbox.l1.core import orchestrator
    calls=[]
    def forbidden(*args,**kwargs):
        calls.append(1)
        raise AssertionError('external execution reached by default plan mode')
    monkeypatch.setattr(orchestrator,'run',forbidden)
    monkeypatch.setattr(d1,'native_branch',forbidden)
    monkeypatch.setattr(d3,'call_real_llm_analyst',forbidden)
    missing=tmp_path/'missing'
    monkeypatch.setattr(sys,'argv',['d1','--runs-dir',str(missing),'--diag-dir',str(missing),
        '--p0a-catalog',str(missing),'--out',str(tmp_path/'d1.json')])
    assert d1.main()==0
    monkeypatch.setattr(sys,'argv',['d2','--d1-outcomes',str(missing),'--d1-plan',str(missing),'--out',str(tmp_path/'d2.json')])
    assert d2.main()==0
    monkeypatch.setattr(sys,'argv',['d3','--d1-plan',str(missing),'--d2-report',str(missing),'--out',str(tmp_path/'d3.json')])
    assert d3.main()==0
    assert calls==[]


def test_independent_simulation_without_adapter_never_falls_through_to_native(tmp_path,monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]/'scripts'))
    import v3_d1_ladder as d1
    calls=[]
    def forbidden(*args,**kwargs):
        calls.append(1)
        raise AssertionError('simulation reached native Actor adapter')
    monkeypatch.setattr(d1,'native_branch',forbidden)
    with pytest.raises(ValueError,match='simulat|adapter'):
        d1.execute_d1_plan({},runs_dir=tmp_path,receipts_dir=tmp_path/'receipts',
            outcomes_out=tmp_path/'outcomes',freeze=None,simulate_offline=True)
    assert calls==[]
