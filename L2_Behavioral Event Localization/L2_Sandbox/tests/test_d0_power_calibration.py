"""Checks boundary resolution and actual conditional-target calibration inputs."""
import importlib.util
from pathlib import Path

import pytest

from l2_sandbox.v05 import _difference_bounds, _mean_bounds

spec = importlib.util.spec_from_file_location("calibration", Path(__file__).parents[1] / "scripts/calibrate_d0_power.py")
calibration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibration)


@pytest.mark.parametrize("m,k", [(1, 2), (10, 5), (288, 6)])
def test_closed_form_matches_production_exact_boundary_inversion(m, k):
    row = calibration.boundary_resolution(14, m, k)
    e = row["endpoint_error"]
    assert row["zero_success_upper"] == pytest.approx(_mean_bounds([0.] * 14, e)[1], abs=1e-12)
    assert row["best_difference_lower"] == pytest.approx(_difference_bounds([1.] * 14, [0.] * 14, e)[0], abs=1e-12)
    assert row["zero_success_upper"] > .2
    assert (row["best_difference_lower"] > .1) == (m == 1)


@pytest.mark.parametrize("m,k", [(1, 2), (10, 5), (288, 6)])
def test_strict_minimum_really_is_first_passing_integer(m, k):
    result = calibration.minimum_n(m, k)
    n = result["minimum_n_zero_success_upper_strictly_below_tau"]
    assert calibration.boundary_resolution(n, m, k)["zero_success_upper"] < .2
    assert calibration.boundary_resolution(n - 1, m, k)["zero_success_upper"] >= .2
    n = result["minimum_n_best_difference_lower_strictly_above_delta"]
    assert calibration.boundary_resolution(n, m, k)["best_difference_lower"] > .1
    assert calibration.boundary_resolution(n - 1, m, k)["best_difference_lower"] <= .1


def test_synthetic_physical_draws_are_distinct_and_reproducible():
    units, probs = calibration.synthetic_units("null", 14, "fixed-seed")
    again, _ = calibration.synthetic_units("null", 14, "fixed-seed")
    assert [u.to_dict() for u in units] == [u.to_dict() for u in again]
    assert len({u.family_id for u in units}) == 9
    assert len(units) == 24
    ids = [x for u in units for values in u.e2_response_vector.branch_identities.values() for x in values]
    assert len(ids) == len(set(ids)) == 24 * 5 * 14
    assert all(set(p.values()) == {.2} for p in probs.values())


def test_null_conditional_truth_is_exactly_zero_and_actual_split_is_seven():
    cell = calibration.calibrate_cell("null", 14, 2, "test")
    assert cell["split"] == [7, 7]
    assert cell["conditional_true_target"]["min"] == cell["conditional_true_target"]["max"] == 0
    assert cell["halfwidth"]["min"] > 0
    assert cell["gate_rate_interpretation"].startswith("false_positive")
