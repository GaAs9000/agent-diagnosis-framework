"""D0 budget exhaustion retains the full frozen experiment and its evidence."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'L1_sandbox/C3_Experimental_Sandbox/src'))
from l2_sandbox import v3_execution as ex
from l2_sandbox.v05 import build_ladder_intervention_messages, build_interleaved_arm_schedule
import v3_d1_ladder as d1
import v3_d2_ceiling as d2


def plan():
    u = dict(unit_id='ep::item', episode_id='ep', task_id='task', family_id='f', unit_level='rubric_item', defect_name='target', rule_source='explicit', deduction_weight=.5, stratum_id='s', inclusion_prob=.25, anchor_k=2, anchor_type='voluntary_pre_finish', remaining_budget=3, parent_identity={'index': 'frozen'}, eligible_arms=['P', 'D', 'S'], ineligible_arms=['R_explicit', 'R_implicit', 'R_single'], arm_messages=build_ladder_intervention_messages(explicit_requirements=[], implicit_requirements=[]), comparison_ids=['D_vs_P', 'S_vs_P', 'S_vs_D'])
    u['schedule'] = build_interleaved_arm_schedule(u['eligible_arms'], 2, anchor_k=2, remaining_budget=3)
    return ex.seal(dict(pricing_mode='local_no_provider_charge', actor_identity=dict(model='Qwen/Qwen3.6-27B', harness='pi', endpoint='http://local.test/v1', provider='local'), stage='D1', execution_ready=True, unit_schedules=[u], primary_test_family=dict(size=3, comparison_ids=u['comparison_ids']), d0_candidate_parameters=dict(n_per_arm=2, delta=.1, tau=.2, alpha=.05, ceiling_threshold=.05, min_replicates_per_arm=2, estimand='conditional_frozen_dev_pool'), budget_and_stop_conditions=dict(max_actor_requests_cap=18), r12_passing_harm_check_plan=dict(sampled_passing_units=[]), task_ladder_message_specs={}), 'plan_sha256')


def frozen(p, requests=2):
    return dict(d0_frozen=True, approved_plan_sha256=p['plan_sha256'], limits=dict(max_requests=requests, max_tokens=10000, max_cost_usd=0, max_wall_seconds=86400, max_invalid_attempts=2, max_tokens_per_request=10, usd_per_token_upper_bound=0))


def execute(tmp_path, p, f, runner):
    return d1.execute_d1_plan(p, runs_dir=tmp_path, receipts_dir=tmp_path/'r', outcomes_out=tmp_path/'out', freeze=f, simulate_offline=True, branch_runner=runner)


def test_cap_mid_unit_seals_partial_and_resume_stays_capped(tmp_path):
    p = plan(); f = frozen(p); calls = []
    from unittest.mock import patch
    def runner_shared(u, b, m):
        calls.append(b)
        # Native adapter receives the stage Admission; locate it through patched constructor below.
        assert active[0].acquire(1)
        active[0].reconcile(1, 1)
        return dict(valid_attempt=True, status='completed', outcome_score=0, simulated_offline=True)
    active = []
    cls = d1.Admission
    def admission(*a, **kw):
        obj = cls(*a, **kw); active[:] = [obj]; return obj
    with patch.object(d1, 'Admission', admission):
        out = execute(tmp_path, p, f, runner_shared)
        assert out['status'] == 'incomplete' and len(calls) == 2
        assert sum(map(len, out['partial_current_unit']['branch_receipts'].values())) == 2
        assert len(out['pending_branches']) == 4
        ex.verify_seal(out)
        resumed = execute(tmp_path, p, f, runner_shared)
        assert resumed['status'] == 'incomplete' and len(calls) == 2
        assert resumed['execution_summary']['reused_cached_branches'] == 2
    report = d2.evaluate_d2_from_d1_payload(out, d1_plan_payload=p, estimand='conditional_frozen_dev_pool')
    assert not report['d3_prerequisite_gate_open']


def test_bad_freeze_never_calls_adapter(tmp_path):
    p = plan(); f = frozen(p); f['approved_plan_sha256'] = 'bad'; calls = []
    with pytest.raises(ValueError, match='approved_plan'):
        execute(tmp_path, p, f, lambda *a: calls.append(a))
    assert not calls


def test_zero_api_cost_does_not_relax_request_tokens_or_identity(tmp_path):
    p = plan(); f = frozen(p)
    caps = ex.approve(p, f)
    ledger = ex.Admission(tmp_path/'ledger', p['plan_sha256'], caps)
    assert ledger.acquire(1); ledger.reconcile(1, 1)
    assert ledger.acquire(1); ledger.reconcile(1, 1)
    assert not ledger.acquire(1)
    altered = dict(caps, max_requests=3)
    with pytest.raises(ValueError, match='limits differ'):
        ex.Admission(tmp_path/'ledger', p['plan_sha256'], altered)
    f['limits']['max_tokens'] = 0
    with pytest.raises(ValueError):
        ex.approve(p, f)


def test_d2_forwards_weights_physical_identity_and_rejects_other_estimand(tmp_path, monkeypatch):
    p = plan(); calls = []
    def runner(u, b, m):
        return dict(valid_attempt=True, status='completed', outcome_score=float(b['arm'] != 'P'), simulated_offline=True, package_index_sha256=ex.digest([b['arm'], b['replicate']]))
    out = execute(tmp_path, p, frozen(p, 20), runner)
    real = d2.compute_cross_fitted_diagnostic_value_ceiling
    def compute(units, **kwargs):
        calls.extend(units)
        return real(units, **kwargs)
    monkeypatch.setattr(d2, 'compute_cross_fitted_diagnostic_value_ceiling', compute)
    report = d2.evaluate_d2_from_d1_payload(out, d1_plan_payload=p, estimand='conditional_frozen_dev_pool')
    assert report['d1_provenance_audit']['plan_binding_verified']
    assert calls[0].inclusion_prob == .25 and calls[0].stratum_id == 's'
    assert calls[0].e2_response_vector.branch_identities['P'] == [ex.digest(['P', i]) for i in range(2)]
    assert report['cross_fitted_ceiling']['estimand'] == 'conditional_frozen_dev_pool'
    inferred = d2.evaluate_d2_from_d1_payload(out, d1_plan_payload=p)
    assert inferred['d1_provenance_audit']['plan_binding_verified']
    assert inferred['d0_parameters']['estimand'] == 'conditional_frozen_dev_pool'
    wrong = d2.evaluate_d2_from_d1_payload(out, d1_plan_payload=p, estimand='family_population')
    assert not wrong['d1_provenance_audit']['plan_binding_verified']
    assert not wrong['d3_prerequisite_gate_open']


def test_zero_price_without_local_identity_is_rejected():
    p = plan(); p.pop('actor_identity'); p = ex.seal(p, 'plan_sha256')
    with pytest.raises(ValueError, match='local provider identity'):
        ex.approve(p, frozen(p))


def test_three_workers_overlap_share_cap_and_preserve_partial_units(tmp_path, monkeypatch):
    import copy
    import threading
    p = plan()
    units = []
    for index in range(3):
        u = copy.deepcopy(p['unit_schedules'][0])
        u.update(unit_id=f'ep{index}::item', episode_id=f'ep{index}', stratum_id=f's{index}')
        units.append(u)
    p.update(unit_schedules=units, execution_concurrency=3)
    p = ex.seal(p, 'plan_sha256')
    f = frozen(p, 4)
    active = []; calls = []; overlap = [0, 0]
    lock = threading.Lock(); barrier = threading.Barrier(3, timeout=5)
    cls = d1.Admission
    def admission(*args, **kwargs):
        obj = cls(*args, **kwargs); active[:] = [obj]; return obj
    monkeypatch.setattr(d1, 'Admission', admission)
    def runner(u, b, m):
        ledger = active[0]
        assert ledger.acquire(1)
        with lock:
            calls.append((u['unit_id'], b['arm'], b['replicate']))
            ordinal = len(calls)
            overlap[0] += 1
            overlap[1] = max(overlap)
        if ordinal <= 3:
            barrier.wait()
        ledger.reconcile(1, ordinal)
        with lock:
            overlap[0] -= 1
        return dict(valid_attempt=True, status='completed', outcome_score=0, simulated_offline=True)
    out = execute(tmp_path, p, f, runner)
    assert overlap[1] == 3
    assert out['status'] == 'incomplete'
    assert len(calls) == out['budget_state']['requests'] == 4
    assert len(out['partial_units']) == 3
    assert {r['stratum_id'] for r in out['partial_units']} == {'s0', 's1', 's2'}
    assert sum(sum(a['status'] == 'completed' for a in r['attempt_receipts']) for r in out['partial_units']) == 4
    assert len(out['pending_branches']) == 14
    assert sorted(s['actual_tokens'] for s in out['budget_state']['sends']) == [1, 2, 3, 4]
    ex.verify_seal(out)
    resumed = execute(tmp_path, p, f, runner)
    assert len(calls) == 4
    assert resumed['execution_summary']['reused_cached_branches'] == 4
    assert len(resumed['partial_units']) == 3


def test_concurrency_must_match_frozen_plan_and_be_bounded():
    p = plan(); p['execution_concurrency'] = 4; p = ex.seal(p, 'plan_sha256')
    with pytest.raises(ValueError, match='execution_concurrency'):
        ex.approve(p, frozen(p))
    p['execution_concurrency'] = 3; p = ex.seal(p, 'plan_sha256')
    f = frozen(p); f['execution_concurrency'] = 1
    with pytest.raises(ValueError, match='concurrency differs'):
        ex.approve(p, f)


def test_concurrent_failure_drains_and_retains_other_completed_receipts(tmp_path, monkeypatch):
    import threading
    p = plan(); p['execution_concurrency'] = 3; p = ex.seal(p, 'plan_sha256')
    active = []; barrier = threading.Barrier(3, timeout=5)
    cls = d1.Admission
    def admission(*args, **kwargs):
        obj = cls(*args, **kwargs); active[:] = [obj]; return obj
    monkeypatch.setattr(d1, 'Admission', admission)
    def runner(u, b, m):
        assert active[0].acquire(1)
        active[0].reconcile(1, 1)
        barrier.wait()
        if b['arm'] == 'P':
            raise ValueError('invalid branch evidence')
        return dict(valid_attempt=True, status='completed', outcome_score=1, simulated_offline=True)
    with pytest.raises(ValueError, match='invalid branch evidence'):
        execute(tmp_path, p, frozen(p, 20), runner)
    out = ex.read_json(tmp_path/'out')
    assert out['status'] == 'failed'
    assert out['budget_state']['requests'] == 3
    assert len(out['failed_attempts']) == 1
    assert sum(map(len, out['partial_units'][0]['branch_receipts'].values())) == 2
    assert len(out['partial_units'][0]['attempt_receipts']) == 3
    assert len(out['pending_branches']) == 4
    ex.verify_seal(out)
