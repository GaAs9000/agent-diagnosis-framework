"""A bound v2 grant removes cost admission only, retaining cumulative guards."""
import copy
from pathlib import Path
import time
from types import SimpleNamespace
import json

import pytest

from l2_sandbox import opencode_d1_runtime as runtime
from l2_sandbox.opencode_admission import Admission, DEFAULT_CAPS, validate_budget_authorization
from l2_sandbox.v3_execution import file_digest, read_json, seal, write_json
from test_opencode_d1_runtime import plan, successor
from test_opencode_multigeneration import chain
from test_opencode_budget_authorization import extension


@pytest.fixture
def no_cost(extension):
    current, jobs, grant, path = extension
    current, grant = copy.deepcopy(current), copy.deepcopy(grant)
    previous = read_json(Path(current['predecessor']['plan']['path']))
    grant.update(schema='opencode-d1-budget-authorization-v2',
                 scope='cross_actor_D1_remove_usage_equivalent_cost_cap',
                 exact_user_text='$50 等值用量 全部放开就行',
                 deadline_epoch=previous['deadline_epoch'])
    grant['phase_started_at_epoch'] = grant['deadline_epoch'] - 86400
    grant['cumulative_limits']['max_cost_usd'] = None
    grant = seal(grant)
    write_json(path, grant)
    current['budget_authorization']['sha256'] = file_digest(path)
    current['bound_files'][str(path)] = file_digest(path)
    current['deadline_epoch'] = grant['deadline_epoch']
    current['limits']['max_cost_usd'] = None
    return seal(current), jobs, grant, path


def test_null_cost_limit_requires_v2_authorization(tmp_path, extension):
    current, _, grant, _ = extension
    for authorization in (None, grant):
        with pytest.raises(ValueError, match='positive finite'):
            Admission(tmp_path / 'denied.json', 'a' * 64, dict(max_cost_usd=None),
                      absolute_deadline=current['deadline_epoch'], budget_authorization=authorization)
    assert DEFAULT_CAPS['max_cost_usd'] == 12.0


@pytest.mark.parametrize('field,value', [('scope', 'cross_actor_D1_budget_extension'),
    ('source', 'other'), ('exact_user_text', 'A'), ('predecessor_plan_sha256', '0' * 64)])
def test_v2_requires_exact_authorization(no_cost, field, value):
    _, _, grant, _ = no_cost
    changed = copy.deepcopy(grant)
    changed[field] = value
    with pytest.raises(ValueError, match='authorization'):
        validate_budget_authorization(seal(changed), grant['predecessor_plan_sha256'])


def test_v2_preserves_cumulative_lineage(no_cost, tmp_path):
    current, jobs, grant, _ = no_cost
    carried = runtime._predecessor_seeds(current, jobs, True)
    assert carried['requests'] == 75
    assert carried['tokens'] == 1729388
    assert carried['cost_usd'] == pytest.approx(.482426196)
    assert carried['completed'] == 7
    assert carried['infrastructure_invalid'] == 6
    ledger = Admission(tmp_path / 'admission.json', current['artifact_sha256'], current['limits'],
                       absolute_deadline=current['deadline_epoch'], budget_authorization=grant)
    assert ledger.caps['max_cost_usd'] is None
    assert ledger.caps['max_requests'] + carried['requests'] == 22000
    assert ledger.caps['max_tokens'] + carried['tokens'] == 1_000_000_000


@pytest.mark.parametrize('change', ['deadline', 'requests', 'tokens', 'cost', 'model'])
def test_v2_rejects_deadline_extension_and_cap_reset(no_cost, change):
    current, jobs, grant, path = no_cost
    current, grant = copy.deepcopy(current), copy.deepcopy(grant)
    if change == 'deadline':
        grant['deadline_epoch'] += 1
        grant['phase_started_at_epoch'] += 1
        write_json(path, seal(grant))
        current['budget_authorization']['sha256'] = file_digest(path)
        current['bound_files'][str(path)] = file_digest(path)
        current['deadline_epoch'] = grant['deadline_epoch']
    elif change == 'model':
        current['model']['options']['truncated_tool_call_policy'] = 'output_limit_error_feedback_v1'
    else:
        key = {'requests': 'max_requests', 'tokens': 'max_tokens', 'cost': 'max_cost_usd'}[change]
        current['limits'][key] = 50 if change == 'cost' else current['limits'][key] + 1
    with pytest.raises(ValueError):
        runtime._predecessor_seeds(seal(current), jobs, True)


def test_admission_above_50_keeps_metering_and_other_guards(no_cost, tmp_path):
    current, _, grant, _ = no_cost
    ledger = Admission(tmp_path / 'admission.json', current['artifact_sha256'], current['limits'],
                       absolute_deadline=current['deadline_epoch'], budget_authorization=grant)
    ledger.state['cost_usd'] = 51.0
    assert ledger.exhaustion_reason() is None
    assert ledger.acquire(1)
    assert ledger.snapshot()['cost_usd'] > 51
    ledger.observe(SimpleNamespace(status=200, body=json.dumps(dict(
        model='deepseek-v4.1-flash', usage=dict(prompt_tokens=10, completion_tokens=1))).encode()))
    ledger.reconcile(1, 11)
    assert ledger.snapshot()['cost_usd'] > 51
    assert ledger.snapshot()['sends'][0]['status'] == 'settled'
    ledger.state['requests'] = ledger.caps['max_requests']
    assert ledger.exhaustion_reason() == 'http_attempt_cap'
    ledger.state['requests'] = 0
    ledger.state['tokens'] = ledger.caps['max_tokens']
    assert ledger.exhaustion_reason() == 'token_cap'
    ledger.state['tokens'] = 0
    ledger.state['absolute_deadline'] = time.time() - 1
    assert ledger.exhaustion_reason() == 'wall_cap'


def test_v2_retains_usage_anomaly_stop(no_cost, tmp_path):
    current, _, grant, _ = no_cost
    ledger = Admission(tmp_path / 'admission.json', current['artifact_sha256'], current['limits'],
                       absolute_deadline=current['deadline_epoch'], budget_authorization=grant)
    assert ledger.acquire(1)
    ledger.observe(SimpleNamespace(status=200, body=json.dumps(dict(
        model='deepseek-v4.1-flash', usage=dict(prompt_tokens=10, completion_tokens=8193))).encode()))
    assert ledger.exhaustion_reason() == 'usage_exceeded_reservation'


def test_inherited_v2_preserves_null_cost_and_ancestor_usage(no_cost, tmp_path):
    previous, _, grant, _ = no_cost
    output = tmp_path / 'uncapped/execution'
    jobs = runtime._scheduled_jobs(previous, output)
    carried = runtime._predecessor_seeds(previous, jobs, True)
    for job in jobs:
        if 'predecessor_seed' in job:
            write_json(job['path'], job['predecessor_seed'])
    own = dict(requests=0, tokens=0, cost_usd=0.0, invalid_attempts=0,
               branch_invalid=0, fatal_invalid=0)
    write_json(output / 'admission.json', seal(dict(own,
        plan_sha256=previous['artifact_sha256'], limits=previous['limits'],
        stop_reason='closed', sends=[])))
    aggregate = {'aggregate_' + key: value for key, value in carried.items()
                 if key not in ('completed', 'infrastructure_invalid')}
    write_json(output / 'result.json', seal(dict(own, **aggregate,
        plan_sha256=previous['artifact_sha256'], status='incomplete', d2_gate_open=False,
        simulated_offline=True, completed_branches=7, invalid_or_incomplete_branches=6,
        planned_branches=len(jobs), pending_branches=len(jobs)-13, http_send_count=0)))
    write_json(output.parent / 'plan.json', previous)
    write_json(output.parent / 'supervisor.json', dict(deadline_epoch=previous['deadline_epoch'],
        started_at_epoch=grant['phase_started_at_epoch']))
    write_json(output.parent / 'supervisor_result.json', dict(controller_exit_code=0))
    current = copy.deepcopy(previous)
    current.pop('budget_authorization')
    refs = {name: dict(path=str(path), sha256=file_digest(path)) for name, path in (
        ('plan', output.parent / 'plan.json'), ('result', output / 'result.json'),
        ('admission', output / 'admission.json'), ('supervisor', output.parent / 'supervisor.json'),
        ('supervisor_result', output.parent / 'supervisor_result.json'))}
    refs['receipt_directory'] = str(output / 'receipts')
    current['predecessor'] = refs
    for ref in refs.values():
        if isinstance(ref, dict):
            current['bound_files'][ref['path']] = ref['sha256']
    for path in (output / 'receipts').glob('*.json'):
        current['bound_files'][str(path)] = file_digest(path)
    current = seal(current)
    updated_jobs = runtime._scheduled_jobs(current, tmp_path / 'next')
    assert runtime._predecessor_seeds(current, updated_jobs, True) == carried
    assert runtime._admission_authorization(current) == grant
    changed = copy.deepcopy(current)
    changed['limits']['max_cost_usd'] = 12
    with pytest.raises(ValueError, match='inherited v2'):
        runtime._predecessor_seeds(seal(changed), updated_jobs, True)
