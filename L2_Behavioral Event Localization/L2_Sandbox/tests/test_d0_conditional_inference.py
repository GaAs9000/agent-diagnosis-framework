"""Offline tests of the conditional frozen-pool estimand and physical draw unit."""
from copy import deepcopy
import hashlib

import pytest

from l2_sandbox.io import L2Error
from l2_sandbox.v05 import (
    DiagnosticUnitV05, E1Observation, E2ResponseVector,
    compute_cross_fitted_diagnostic_value_ceiling,
)


def unit(index, n=14, *, shared=None, weight=1.0):
    parent = f"parent-{index}" if shared is None else shared
    good = "D" if index % 2 == 0 else "S"
    outcomes = {arm: [float(arm == good)] * n for arm in ("P", "D", "S", "R_explicit", "R_single")}
    identities = {arm: [hashlib.sha256(f"{parent}/{arm}/{i}".encode()).hexdigest() for i in range(n)] for arm in outcomes}
    return DiagnosticUnitV05(
        unit_id=f"u{index}", episode_id=parent, task_id="task", benchmark="officeval",
        unit_level="rubric_item", defect_name=f"rule-{index}", deduction_weight=weight,
        rule_source="explicit", family_id="single-family", inclusion_prob=1.0,
        e1_observation=E1Observation("never_succeeded_on_disk", "wrong_candidate", 0, 0, 0, weight),
        e2_response_vector=E2ResponseVector(evaluated=True, arm_outcomes=outcomes, branch_identities=identities),
        operational_label=None, evidence_status="inconclusive",
    )


def ceiling(units, **kwargs):
    return compute_cross_fitted_diagnostic_value_ceiling(units, estimand="conditional_frozen_dev_pool", **kwargs)


def test_one_replicate_cannot_open_gate():
    result = ceiling([unit(0, n=1), unit(1, n=1)])
    assert not result["conditional_pool_complete"]
    assert not result["passes_v3_p3_gate"]
    assert result["deduction_weighted_lb95"] == -1


def test_fake_or_missing_physical_ids_rejected():
    u = unit(0)
    u.e2_response_vector.branch_identities["P"][0] = "fake-branch"
    with pytest.raises(L2Error, match="SHA256"):
        ceiling([u])
    u = unit(0)
    u.e2_response_vector.branch_identities = {}
    with pytest.raises(L2Error, match="aligned"):
        ceiling([u])


def test_physical_draw_cannot_cross_halves_globally():
    a, b = unit(0), unit(1)
    b.episode_id = a.episode_id
    b.e2_response_vector.branch_identities["P"][7] = a.e2_response_vector.branch_identities["P"][0]
    with pytest.raises(L2Error, match="across selection"):
        ceiling([a, b])


def test_shared_package_endpoints_are_one_dependent_group():
    originals = [unit(0), unit(1)]
    base = ceiling(originals)
    copies = []
    for u in originals:
        v = deepcopy(u)
        v.unit_id += "-other-item"
        v.defect_name += "-other-item"
        copies.extend([u, v])
    grouped = ceiling(copies)
    for direction, original in zip(grouped["direction_intervals"], base["direction_intervals"]):
        assert direction["weighted"]["physical_group_count"] == original["weighted"]["physical_group_count"]
        assert direction["weighted"]["lb95"] == pytest.approx(original["weighted"]["lb95"])


def test_same_physical_endpoint_requires_consistent_outcomes():
    a = unit(0)
    b = deepcopy(a)
    b.unit_id = "duplicate-endpoint"
    b.e2_response_vector.arm_outcomes["D"][0] = 0
    with pytest.raises(L2Error, match="Inconsistent outcome"):
        ceiling([a, b])


def test_zero_evaluation_variance_does_not_collapse_interval():
    units = [unit(0), unit(1)]
    for u in units:
        for arm in u.e2_response_vector.arm_outcomes:
            u.e2_response_vector.arm_outcomes[arm][7:] = [0.0] * 7
    first = ceiling(units)["direction_intervals"][0]["weighted"]
    assert first["estimate"] == 0
    assert first["lb95"] < 0 < first["ub95"]
    assert first["empirical_bernstein_radius"] > 0


def test_exact_policy_identity_is_zero_even_with_variable_observations():
    u = unit(0)
    result = ceiling([u])
    assert result["deduction_weighted_lb95"] == result["deduction_weighted_ub95"] == 0
    assert all(d["weighted"]["interval_method"] == "exact_policy_identity" for d in result["direction_intervals"])


def test_well_powered_conditional_gate_can_open_with_one_family():
    result = ceiling([unit(i, weight=1 + i % 3) for i in range(200)])
    assert result["passes_v3_p3_gate"]
    assert result["conditional_deduction_weighted_lb95"] > 0.05
    assert not result["family_population_auxiliary"]["passes_v3_p3_gate"]
    assert result["error_allocation"]["eta_per_direction_tail_method"] == 0.05 / 8
    assert not result["error_allocation"]["joint_weighted_unweighted_coverage_claimed"]


def test_swap_halves_preserves_estimate_and_interval():
    units = [unit(i) for i in range(8)]
    # Make the two halves differ while preserving distinct physical identities.
    units[0].e2_response_vector.arm_outcomes["D"][7:] = [0.2] * 7
    before = ceiling(units)
    for u in units:
        for arm, values in u.e2_response_vector.arm_outcomes.items():
            u.e2_response_vector.arm_outcomes[arm] = values[7:] + values[:7]
            ids = u.e2_response_vector.branch_identities[arm]
            u.e2_response_vector.branch_identities[arm] = ids[7:] + ids[:7]
    after = ceiling(units)
    assert after["deduction_weighted_ceiling"] == pytest.approx(before["deduction_weighted_ceiling"])
    assert after["deduction_weighted_lb95"] == pytest.approx(before["deduction_weighted_lb95"])


def test_missing_frozen_arm_closes_pool_even_if_globally_absent():
    units = [unit(0), unit(1)]
    for u in units:
        del u.e2_response_vector.arm_outcomes["S"]
        u.e2_response_vector.branch_identities["S"] = []
    result = ceiling(units)
    assert not result["conditional_pool_complete"]
    assert not result["passes_v3_p3_gate"]


def test_evaluation_harm_remains_signed_and_selection_uses_same_half():
    a, b = unit(0), unit(1)
    # Selection ties the fixed policies at 1/2 and picks D; b selects S.
    # Evaluation reverses b's response, creating negative policy utility.
    b.e2_response_vector.arm_outcomes["D"][7:] = [1.0] * 7
    b.e2_response_vector.arm_outcomes["S"][7:] = [0.0] * 7
    result = ceiling([a, b])
    first = result["direction_intervals"][0]["weighted"]
    assert first["estimate"] == -0.5
    assert first["lb95"] < 0
    first_rows = [r for r in result["unit_evaluations"] if r["fold"] == 1]
    assert first_rows[1]["selected_unit_arm"] == "S"
    assert first_rows[1]["train_best_fixed_arm_weighted"] == "D"


def test_hidden_rule_arm_cannot_enter_conditional_policy_pool():
    with pytest.raises(L2Error, match="hidden"):
        ceiling([unit(0)], candidate_arms=("P", "R_implicit"))
