"""Finish-only v3 grants retain all charges and close the 24-hour phase bound."""
import copy
from pathlib import Path
import time

import pytest

from l2_sandbox import opencode_d1_runtime as runtime
from l2_sandbox.opencode_admission import Admission, DEFAULT_CAPS, validate_budget_authorization
from l2_sandbox.v3_execution import file_digest, read_json, seal, write_json
from test_opencode_d1_runtime import plan, successor
from test_opencode_multigeneration import chain
from test_opencode_budget_authorization import extension


@pytest.fixture
def finish_remaining(extension):
    current, jobs, grant, path = extension
    current, grant = copy.deepcopy(current), copy.deepcopy(grant)
    supervisor = read_json(Path(current['predecessor']['supervisor']['path']))
    requested = supervisor['deadline_epoch'] + 86400
    start = time.time()
    source_path = path.with_name('authorization_source.txt')
    source_path.write_text('Original user attachment')
    current['bound_files'][str(source_path)] = file_digest(source_path)
    grant.update(schema='opencode-d1-budget-authorization-v3',
        scope='cross_actor_D1_finish_remaining', source='direct_user_attachment_20261020',
        exact_user_text='授权决定：批准启动 continuation_05，仅执行剩余 412 条 pending 分支。',
        request_attachment_sha256=file_digest(source_path),
        request_attachment=dict(path=str(source_path), sha256=file_digest(source_path)),
        requested_deadline_epoch=requested,
        predecessor_census=dict(completed_branches=7, invalid_or_incomplete_branches=6,
                                pending_branches=2269),
        authorized_deadline_extension_seconds=86400, phase_started_at_epoch=start,
        deadline_epoch=min(requested, start + 86400), max_wall_seconds=86400,
        cumulative_limits=dict(max_requests=25000, max_tokens=2_000_000_000, max_cost_usd=None))
    grant = seal(grant)
    write_json(path, grant)
    current['budget_authorization']['sha256'] = file_digest(path)
    current['bound_files'][str(path)] = file_digest(path)
    current['deadline_epoch'] = grant['deadline_epoch']
    current['limits'].update(max_requests=25000 - 75, max_tokens=2_000_000_000 - 1729388,
                             max_cost_usd=None, max_wall_seconds=86400)
    return seal(current), jobs, grant, path


def rebind(current, grant, path):
    write_json(path, seal(grant))
    current['budget_authorization']['sha256'] = file_digest(path)
    current['bound_files'][str(path)] = file_digest(path)
    current['deadline_epoch'] = grant['deadline_epoch']
    return seal(current)


@pytest.mark.parametrize('field,value', [
    ('source', 'direct_user_message_20261019'), ('scope', 'cross_actor_D1_budget_extension'),
    ('predecessor_plan_sha256', '0' * 64), ('request_attachment_sha256', 'bad'),
    ('exact_user_text', ''), ('authorized_deadline_extension_seconds', 86401),
    ('max_wall_seconds', 86399)])
def test_v3_rejects_wrong_authorization(finish_remaining, field, value):
    _, _, grant, _ = finish_remaining
    changed = copy.deepcopy(grant)
    changed[field] = value
    with pytest.raises(ValueError):
        validate_budget_authorization(seal(changed), grant['predecessor_plan_sha256'])


@pytest.mark.parametrize('cap,value', [('max_requests', 25001), ('max_tokens', 2_000_000_001),
                                      ('max_cost_usd', 50), ('max_requests', 22000)])
def test_v3_rejects_wrong_cumulative_caps(finish_remaining, cap, value):
    _, _, grant, _ = finish_remaining
    changed = copy.deepcopy(grant)
    changed['cumulative_limits'][cap] = value
    with pytest.raises(ValueError, match='authorization'):
        validate_budget_authorization(seal(changed), grant['predecessor_plan_sha256'])


def test_v3_carries_all_usage_and_preserves_completed_and_invalid(finish_remaining, tmp_path):
    current, jobs, grant, _ = finish_remaining
    carried = runtime._predecessor_seeds(current, jobs, True)
    assert carried['requests'] == 75
    assert carried['tokens'] == 1729388
    assert carried['cost_usd'] == pytest.approx(.482426196)
    assert carried['completed'] == 7
    assert carried['infrastructure_invalid'] == 6
    seeds = [job['predecessor_seed'] for job in jobs if 'predecessor_seed' in job]
    assert len(seeds) == 13
    assert all('outcome_score' not in seed for seed in seeds if seed['status'] != 'completed')
    assert not any(job['package'].exists() for job in jobs)
    ledger = Admission(tmp_path / 'admission.json', current['artifact_sha256'], current['limits'],
        absolute_deadline=current['deadline_epoch'], budget_authorization=grant)
    assert ledger.caps['max_requests'] + carried['requests'] == 25000
    assert ledger.caps['max_tokens'] + carried['tokens'] == 2_000_000_000
    assert ledger.caps['max_cost_usd'] is None
    assert ledger.snapshot()['requests'] == 0


def test_v3_allows_exact_caps_without_changing_defaults(finish_remaining, tmp_path):
    current, _, grant, _ = finish_remaining
    caps = dict(max_requests=25000, max_tokens=2_000_000_000, max_cost_usd=None)
    for name, value in caps.items():
        with pytest.raises(ValueError):
            Admission(tmp_path / (name + '.json'), 'a' * 64, {name: value})
    ledger = Admission(tmp_path / 'granted.json', current['artifact_sha256'], caps,
        absolute_deadline=current['deadline_epoch'], budget_authorization=grant)
    assert all(ledger.caps[name] == value for name, value in caps.items())
    assert DEFAULT_CAPS['max_requests'] == 15000
    assert DEFAULT_CAPS['max_tokens'] == 1_000_000_000
    assert DEFAULT_CAPS['max_cost_usd'] == 12.0


@pytest.mark.parametrize('change', ['requested', 'phase', 'deadline', 'requests', 'tokens', 'cost', 'model'])
def test_v3_rejects_deadline_and_usage_reset(finish_remaining, change):
    current, jobs, grant, path = finish_remaining
    current, grant = copy.deepcopy(current), copy.deepcopy(grant)
    if change == 'requested':
        grant['requested_deadline_epoch'] += 1
        grant['deadline_epoch'] = min(grant['requested_deadline_epoch'], grant['phase_started_at_epoch'] + 86400)
    elif change == 'phase':
        grant['deadline_epoch'] = grant['phase_started_at_epoch'] + 86401
    elif change == 'deadline':
        grant['deadline_epoch'] += 1
    elif change == 'model':
        current['model']['options']['truncated_tool_call_policy'] = 'output_limit_error_feedback_v1'
    else:
        cap = {'requests': 'max_requests', 'tokens': 'max_tokens', 'cost': 'max_cost_usd'}[change]
        current['limits'][cap] = 12 if change == 'cost' else current['limits'][cap] + 1
    with pytest.raises(ValueError):
        runtime._predecessor_seeds(rebind(current, grant, path), jobs, True)


def test_v3_accepts_requested_deadline_when_it_is_earlier_than_phase_end(finish_remaining):
    current, jobs, grant, path = finish_remaining
    grant = copy.deepcopy(grant)
    grant['phase_started_at_epoch'] = grant['requested_deadline_epoch'] - 3600
    grant['deadline_epoch'] = grant['requested_deadline_epoch']
    assert runtime._predecessor_seeds(rebind(copy.deepcopy(current), grant, path), jobs, True)['requests'] == 75


def test_inherited_v3_retains_null_cost_and_ancestor_usage(finish_remaining, tmp_path):
    previous, _, grant, _ = finish_remaining
    output = tmp_path / 'finished/execution'
    jobs = runtime._scheduled_jobs(previous, output)
    carried = runtime._predecessor_seeds(previous, jobs, True)
    for job in jobs:
        if 'predecessor_seed' in job:
            write_json(job['path'], job['predecessor_seed'])
    own = dict(requests=0, tokens=0, cost_usd=0.0, invalid_attempts=0,
               branch_invalid=0, fatal_invalid=0)
    write_json(output / 'admission.json', seal(dict(own, plan_sha256=previous['artifact_sha256'],
        limits=previous['limits'], stop_reason='closed', sends=[])))
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


@pytest.mark.parametrize('change', ['missing', 'tampered', 'digest'])
def test_v3_requires_bound_raw_authorization_source(finish_remaining, change):
    current, jobs, grant, path = finish_remaining
    current, grant = copy.deepcopy(current), copy.deepcopy(grant)
    source = Path(grant['request_attachment']['path'])
    if change == 'missing':
        current['bound_files'].pop(str(source))
    elif change == 'tampered':
        source.write_text('Changed user attachment')
    else:
        grant['request_attachment']['sha256'] = '0' * 64
    with pytest.raises(ValueError):
        runtime._predecessor_seeds(rebind(current, grant, path), jobs, True)


@pytest.mark.parametrize('change', ['missing', 'negative', 'boolean', 'total', 'mismatch'])
def test_v3_requires_exact_predecessor_census(finish_remaining, change):
    current, jobs, grant, path = finish_remaining
    current, grant = copy.deepcopy(current), copy.deepcopy(grant)
    if change == 'missing':
        grant.pop('predecessor_census')
    elif change == 'negative':
        grant['predecessor_census']['completed_branches'] = -1
    elif change == 'boolean':
        grant['predecessor_census']['completed_branches'] = True
    elif change == 'total':
        grant['predecessor_census']['pending_branches'] += 1
    else:
        grant['predecessor_census']['completed_branches'] += 1
        grant['predecessor_census']['pending_branches'] -= 1
    with pytest.raises(ValueError):
        runtime._predecessor_seeds(rebind(current, grant, path), jobs, True)
