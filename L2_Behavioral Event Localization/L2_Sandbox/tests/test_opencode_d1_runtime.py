"""Offline cross-actor runner checks, executed on the AML CPU worker only."""
import copy
import threading
import time

import pytest

from l2_sandbox import opencode_d1_runtime as runtime
from l2_sandbox.opencode_admission import DEFAULT_CAPS
from l2_sandbox.v3_execution import file_digest, read_json, seal, write_json


@pytest.fixture
def plan(tmp_path):
    def unit(i, arms):
        messages = {a: dict(eligible=True, message=None, message_sha256=None) for a in arms}
        return dict(unit_id=f'u{i}', episode_id=f'ep_{i}', task_id=f't{i}',
            anchor_k=2, remaining_budget=33, parent_identity={'parent_episode_id': f'ep_{i}'},
            eligible_arms=arms, arms=arms, arm_messages=messages,
            schedule=runtime.build_interleaved_arm_schedule(arms, 14, anchor_k=2, remaining_budget=33))
    failures = [unit(i, ['P', 'D', 'S', 'R_explicit', 'R_implicit'] + (['R_single'] if i < 13 else [])) for i in range(24)]
    controls = [unit(i, ['P', 'D', 'S', 'R_explicit', 'R_single']) for i in range(24, 30)]
    science = seal(dict(d0_candidate_parameters=runtime.PARAMETERS,
        unit_schedules=failures, r12_passing_harm_check_plan=dict(sampled_passing_units=controls),
        task_ladder_message_specs={u['task_id']:dict(ladder_messages=u['arm_messages']) for u in controls}), 'plan_sha256')
    scientific_path = tmp_path / 'science.json'
    write_json(scientific_path, science)
    freeze_path = tmp_path / 'freeze.json'
    write_json(freeze_path, seal(dict(schema='d0-scientific-input-freeze-v2', d0_frozen=True,
        execution_authorized=False, plan_sha256=science['plan_sha256']), 'freeze_sha256'))
    source = tmp_path / 'source.py'
    source.write_text('# frozen source\n')
    return seal(dict(schema='cross-actor-d1-execution-v1',
        authorization=dict(scope='cross_actor_D1', execution_authorized=True,
                           exact_user_text='开始吧，直接用opencode'),
        execution_concurrency=3, automatic_d2=False,
        model=dict(adapter='opencode-chat', provider='opencode-go', model=runtime.MODEL,
            base_url='https://opencode.ai/inference/go/v1', timeout_seconds=600,
            options=dict(expected_reported_model=runtime.MODEL),
            parameters=dict(temperature=1, max_tokens=8192)),
        scientific_plan=science, source_scientific_plan=dict(path=str(scientific_path), sha256=file_digest(scientific_path)),
        source_scientific_freeze=dict(path=str(freeze_path), sha256=file_digest(freeze_path)),
        source_files={str(source):file_digest(source), **{str(p): file_digest(p) for p in runtime.CRITICAL_SOURCES}},
        bound_files={str(freeze_path):file_digest(freeze_path)},
        limits=dict(DEFAULT_CAPS), runs_dir=str(tmp_path / 'parents')))


def completed(*args, **kwargs):
    return dict(status='completed', valid_attempt=True, outcome_score=0.5)


@pytest.mark.parametrize('change', [
    {'schema': 'same-actor-d1'},
    {'authorization': dict(scope='same_actor_D1', execution_authorized=True, exact_user_text='开始吧，直接用opencode')},
    {'authorization': dict(scope='cross_actor_D1', execution_authorized=True, exact_user_text='go')},
    {'automatic_d2': True},
])
def test_wrong_authority_rejected_before_admission(plan, tmp_path, change):
    output = tmp_path / 'output'
    with pytest.raises(ValueError):
        runtime.execute(seal({**plan, **change}), output, branch_adapter=completed)
    assert not (output / 'admission.json').exists()


def test_changed_source_rejected_before_admission(plan, tmp_path):
    source = next(iter(plan['source_files']))
    from pathlib import Path
    Path(source).write_text('# changed\n')
    with pytest.raises(ValueError, match='changed'):
        runtime.execute(plan, tmp_path / 'output', branch_adapter=completed)
    assert not (tmp_path / 'output/admission.json').exists()


def test_critical_source_cannot_be_omitted_from_resealed_inventory(plan, tmp_path):
    modified = copy.deepcopy(plan)
    del modified['source_files'][str(runtime.CRITICAL_SOURCES[0])]
    with pytest.raises(ValueError, match='critical executable source'):
        runtime.execute(seal(modified), tmp_path / 'output', branch_adapter=completed)
    assert not (tmp_path / 'output/admission.json').exists()


def test_scientific_raw_identity_and_n14_are_required(plan, tmp_path):
    modified = copy.deepcopy(plan)
    modified['scientific_plan']['d0_candidate_parameters']['n_per_arm'] = 15
    modified['scientific_plan'] = seal(modified['scientific_plan'], 'plan_sha256')
    with pytest.raises(ValueError, match='raw reference'):
        runtime.execute(seal(modified), tmp_path / 'output', branch_adapter=completed)
    assert not (tmp_path / 'output/admission.json').exists()


def test_exact_original_interleaving_includes_all_controls(plan, tmp_path):
    jobs = runtime._jobs(plan, tmp_path / 'out')
    assert len(jobs) == 2282
    assert [j['unit']['unit_id'] for j in jobs[:30]] == [f'u{i}' for i in range(30)]
    for i, unit in enumerate(plan['scientific_plan']['unit_schedules']):
        assert [j['branch'] for j in jobs if j['unit']['unit_id'] == unit['unit_id']] == unit['schedule']
    controls = [j for j in jobs if j['binding']['passing_control']]
    assert len(controls) == 6 * 5 * 14
    assert len({j['binding']['branch_id'] for j in jobs}) == 2282


def subset(monkeypatch, plan, output, count):
    jobs = runtime._jobs(plan, output)[:count]
    monkeypatch.setattr(runtime, '_jobs', lambda p, o: jobs)
    return jobs


def test_serial_canary_then_three_parallel_and_reuse(plan, tmp_path, monkeypatch):
    output = tmp_path / 'out'
    jobs = subset(monkeypatch, plan, output, 7)
    lock = threading.Lock()
    barrier = threading.Barrier(3)
    active = 0
    peak = 0
    calls = []
    canary_done = False

    def adapter(unit, branch, message, **kwargs):
        nonlocal active, peak, canary_done
        with lock:
            index = len(calls)
            calls.append(unit['unit_id'])
            active += 1
            peak = max(peak, active)
            if index:
                assert canary_done
        if index in (1, 2, 3):
            barrier.wait(timeout=10)
        time.sleep(.005)
        with lock:
            active -= 1
            if index == 0:
                canary_done = True
        return completed()

    result = runtime.execute(plan, output, branch_adapter=adapter)
    assert result['status'] == 'simulated_completed'
    assert result['completed_branches'] == 7
    assert peak == 3
    assert result['d2_gate_open'] is False
    assert result['requests'] == 0
    assert all(read_json(j['path'])['simulated_offline'] for j in jobs)
    result = runtime.execute(plan, output, branch_adapter=lambda *a, **kw: pytest.fail('cached branch reran'))
    assert result['completed_branches'] == 7


def test_first_invalid_canary_stops_without_leaking_error(plan, tmp_path, monkeypatch):
    output = tmp_path / 'out'
    jobs = subset(monkeypatch, plan, output, 5)
    calls = []
    def broken(*args, **kwargs):
        calls.append(1)
        raise RuntimeError('SECRET provider body')
    result = runtime.execute(plan, output, branch_adapter=broken)
    assert calls == [1]
    assert result['status'] == 'incomplete'
    assert result['invalid_attempts'] == 1
    assert result['stop_reason'] == 'invalid_attempt'
    assert 'SECRET' not in jobs[0]['path'].read_text()
    with pytest.raises(ValueError, match='incomplete'):
        runtime.execute(plan, output, branch_adapter=completed)
    assert calls == [1]


def test_cap_stops_new_admission_and_preserves_inflight_charges(plan, tmp_path, monkeypatch):
    plan = seal({**plan, 'limits': {**plan['limits'], 'max_requests': 4}})
    output = tmp_path / 'out'
    subset(monkeypatch, plan, output, 12)
    calls = []
    def adapter(*args, limiter, **kwargs):
        calls.append(1)
        assert limiter.acquire(10)
        limiter.reconcile(10, None)
        return completed()
    result = runtime.execute(plan, output, branch_adapter=adapter)
    assert len(calls) == 4
    assert result['requests'] == 4
    assert result['completed_branches'] == 4
    assert result['invalid_attempts'] == 0
    assert result['status'] == 'incomplete'
    assert result['tokens'] == 4 * (1_048_576 + 8192)


def test_uncertain_package_never_retried(plan, tmp_path, monkeypatch):
    output = tmp_path / 'out'
    jobs = subset(monkeypatch, plan, output, 2)
    jobs[0]['package'].mkdir(parents=True)
    with pytest.raises(ValueError, match='uncertain'):
        runtime.execute(plan, output, branch_adapter=lambda *a, **kw: pytest.fail('uncertain retry'))
    assert not (output / 'admission.json').exists()
