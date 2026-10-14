"""Synthetic regressions for honest prediction coverage and curve identifiability."""
import pytest

from l2_sandbox.evaluation.metrics import value_prediction_scores, evaluate_value_predictions_across_cases
from l2_sandbox.stats.curve import evaluate_curve, derive_stage2_interval
from l2_sandbox.stats.missing import apply_missing_imputation, missing_bounds


def rows(value, n=24, outcome_case='scored'):
    return [{'outcome': {'endpoint_defined': value is not None, 'value': value,
                         'outcome_case': outcome_case}} for _ in range(n)]


def test_empty_analyst_cannot_inherit_perfect_rules_predictions():
    receipts = [dict(rows(1.0, 1)[0], checkpoint_id='cp000000', stage='stage1')]
    result = value_prediction_scores('case', 'family', {},
                                     {'rules': {'cp000000': 1.0}, 'linear_decay': {'cp000000': 0.5}},
                                     receipts, 'passed')
    analyst = result['scores']['analyst']
    assert analyst['status'] == 'incomplete_predictions'
    assert analyst['n_expected_branches'] == analyst['n_unpredicted_branches'] == 1
    assert analyst['prediction_coverage'] == 0.0
    assert analyst['brier_or_mse'] is None
    assert result['brier_skill_score_vs_linear_decay'] is None
    assert result['scores']['rules']['brier_or_mse'] == 0.0
    clustered = evaluate_value_predictions_across_cases(
        [result, dict(result, case_id='case2', family_id='family2')], n_boot=20)
    assert clustered['analyst_brier_or_mse'] is None
    assert clustered['skill_score_vs_linear_decay'] is None
    assert clustered['linear_decay_brier_or_mse'] == 0.25
    assert clustered['analyst_unscored_case_count'] == 2


def test_partial_analyst_predictions_are_incomplete():
    receipts = [dict(rows(1.0, 1)[0], checkpoint_id=cp, stage='stage1')
                for cp in ('cp000000', 'cp000004')]
    result = value_prediction_scores('case', 'family', {'value_predictions': {'cp000000': 1.0}},
                                     {}, receipts, 'passed')
    analyst = result['scores']['analyst']
    assert analyst['n_scored_branches'] == analyst['n_unpredicted_branches'] == 1
    assert analyst['prediction_coverage'] == 0.5
    assert analyst['brier_or_mse'] is None


def test_canonical_submission_cases_preserve_primary_zero_and_sensitivity_bounds():
    samples = {0: rows(0.0, 1, 'empty_submission') + rows(0.0, 1, 'unscorable_submission')}
    assert apply_missing_imputation(samples, 1.0) == {0: [0.0, 0.0]}
    assert apply_missing_imputation(samples, 0.0, include_unscorable_as_missing=True) == {0: [0.0, 0.0]}
    assert apply_missing_imputation(samples, 1.0, include_unscorable_as_missing=True) == {0: [0.0, 1.0]}
    counts = missing_bounds(samples, {})
    assert counts['missing_count'] == 0
    assert counts['empty_submission_zero_count'] == counts['unscorable_submission_zero_count'] == 1
    legacy = {0: [{'outcome': {'endpoint_defined': True, 'value': 0.0,
                              'disposition': 'unscorable_submission_zero'}}]}
    assert apply_missing_imputation(legacy, 1.0, include_unscorable_as_missing=True) == {0: [1.0]}


def test_unscorable_sensitivity_can_make_concentration_not_identifiable():
    s1 = {0: rows(1.0), 4: rows(0.0)}
    s2 = {0: rows(1.0), **{k: rows(0.0, outcome_case='unscorable_submission') for k in (1, 2, 3, 4)}}
    result = evaluate_curve([0, 4], s1, s2, 8, 0.0)
    assert result['primary_inference']['verdict'] == 'concentrated_turn'
    assert result['missing_bounds']['missing_as_0_verdict'] == 'concentrated_turn'
    assert result['missing_bounds']['missing_as_1_verdict'] == 'diffuse_loss'
    assert result['curve_verdict'] == 'not_identifiable'
    assert 'sensitivity_bounds_change_verdict' in result['not_identifiable_reasons']


def test_out_of_reach_requires_initial_confidence_bound():
    insufficient = evaluate_curve([0], {0: rows(0.0, 1)}, {0: rows(0.0)}, 1, 0.0)
    assert insufficient['primary_inference']['v0_upper_95'] >= 0.25
    assert insufficient['curve_verdict'] == 'not_identifiable'
    assert insufficient['not_identifiable_reasons']
    established = evaluate_curve([0], {0: rows(0.0)}, {0: rows(0.0)}, 1, 0.0)
    assert established['primary_inference']['v0_upper_95'] < 0.25
    assert established['curve_verdict'] == 'out_of_reach'
    missing_initial = evaluate_curve([0], {0: []}, {0: rows(1.0)}, 1, 0.0)
    assert missing_initial['curve_verdict'] == 'not_identifiable'
    assert 'initial_value_evidence_missing' in missing_initial['not_identifiable_reasons']


@pytest.mark.parametrize('terminal', [None, float('nan'), float('inf'), -0.1, 1.1])
def test_undefined_parent_value_cannot_be_imputed(terminal):
    with pytest.raises(ValueError, match='defined finite parent terminal value'):
        derive_stage2_interval([0], {0: [1.0]}, 1, terminal)
    with pytest.raises(ValueError, match='defined finite parent terminal value'):
        evaluate_curve([0], {0: rows(1.0)}, {0: rows(1.0)}, 1, terminal)


def test_cluster_skill_uses_only_cases_with_both_predictors_scored():
    paired = {'case_id': 'paired', 'family_id': 'family1', 'scores': {
        'analyst': {'brier_or_mse': 0.1}, 'linear_decay': {'brier_or_mse': 0.2}}}
    analyst_only = {'case_id': 'analyst_only', 'family_id': 'family2', 'scores': {
        'analyst': {'brier_or_mse': 0.9}, 'linear_decay': {'brier_or_mse': None}}}
    baseline_only = {'case_id': 'baseline_only', 'family_id': 'family3', 'scores': {
        'analyst': {'brier_or_mse': None}, 'linear_decay': {'brier_or_mse': 0.01}}}
    result = evaluate_value_predictions_across_cases([paired, analyst_only, baseline_only], n_boot=20)
    assert result['skill_score_vs_linear_decay'] == 0.5
    assert result['paired_scored_case_count'] == 1
    assert result['status'] == 'incomplete_analyst_predictions'


def prediction_case(case_id, family_id, analyst, linear):
    return {'case_id': case_id, 'family_id': family_id, 'scores': {
        'analyst': {'brier_or_mse': analyst}, 'linear_decay': {'brier_or_mse': linear}}}


def test_unscored_families_cannot_create_cluster_confidence_intervals():
    result = evaluate_value_predictions_across_cases([
        prediction_case('a', 'A', 0.1, 0.2),
        prediction_case('b', 'B', None, 0.3),
    ], n_boot=100)
    assert result['family_count'] == 2
    assert result['analyst_scored_family_count'] == result['paired_scored_family_count'] == 1
    assert result['analyst_ci_95'] == result['skill_ci_95'] == [None, None]
    assert result['analyst_brier_or_mse'] == 0.1
    assert result['skill_score_vs_linear_decay'] == 0.5
    assert result['status'] == 'incomplete_analyst_predictions'


def test_each_cluster_interval_requires_two_families_eligible_for_that_metric():
    cases = [prediction_case('a', 'A', 0.1, 0.2), prediction_case('b', 'B', 0.3, None)]
    result = evaluate_value_predictions_across_cases(cases, n_boot=100)
    assert result['analyst_scored_family_count'] == 2
    assert result['paired_scored_family_count'] == 1
    assert result['analyst_ci_95'] == [0.1, 0.3]
    assert result['skill_ci_95'] == [None, None]

    cases.append(prediction_case('c', 'C', 0.3, 0.4))
    paired_result = evaluate_value_predictions_across_cases(cases, n_boot=100)
    assert paired_result['analyst_scored_family_count'] == 3
    assert paired_result['paired_scored_family_count'] == 2
    assert paired_result['skill_ci_95'] == pytest.approx([0.25, 0.5])

    # Adding fully unscored families must not change either eligible bootstrap population.
    extended = evaluate_value_predictions_across_cases(
        cases + [prediction_case('d', 'D', None, 0.9)], n_boot=100)
    assert extended['analyst_ci_95'] == paired_result['analyst_ci_95']
    assert extended['skill_ci_95'] == paired_result['skill_ci_95']
