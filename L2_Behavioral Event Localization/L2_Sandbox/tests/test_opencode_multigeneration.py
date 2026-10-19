"""Closed continuation chains preserve attempts, identities and cumulative usage."""
import copy
from pathlib import Path

import pytest

from l2_sandbox import opencode_d1_runtime as runtime
from l2_sandbox.v3_execution import file_digest, read_json, seal, write_json
from test_opencode_d1_runtime import plan, successor  # shared frozen scientific fixture


def bind_successor(previous, output, destination, *, truncation=True):
    updated = copy.deepcopy(previous)
    previous_path = output.parent / 'plan.json'
    write_json(previous_path, previous)
    refs = {name: dict(path=str(path), sha256=file_digest(path)) for name, path in (
        ('plan', previous_path), ('result', output / 'result.json'),
        ('admission', output / 'admission.json'), ('supervisor', output.parent / 'supervisor.json'),
        ('supervisor_result', output.parent / 'supervisor_result.json'))}
    refs['receipt_directory'] = str(output / 'receipts')
    updated['predecessor'] = refs
    if truncation:
        updated['model']['options']['truncated_tool_call_policy'] = 'output_limit_error_feedback_v1'
    own = read_json(output / 'admission.json')
    for cap, key in [('max_requests', 'requests'), ('max_tokens', 'tokens'), ('max_cost_usd', 'cost_usd')]:
        updated['limits'][cap] -= own[key]
    for ref in refs.values():
        if isinstance(ref, dict):
            updated['bound_files'][ref['path']] = ref['sha256']
    for path in (output / 'receipts').glob('*.json'):
        updated['bound_files'][str(path)] = file_digest(path)
    return seal(updated), runtime._scheduled_jobs(seal(updated), destination)


@pytest.fixture
def chain(successor, tmp_path):
    previous, root_jobs = successor
    output = tmp_path / 'middle/execution'
    jobs = runtime._scheduled_jobs(previous, output)
    runtime._predecessor_seeds(previous, jobs, True)
    for job in jobs[:6]:
        write_json(job['path'], job['predecessor_seed'])
    for i, job in enumerate(jobs[6:13]):
        if i < 4:
            write_json(job['package'] / 'raw/model/r000001/request.json',
                dict(model=runtime.MODEL, messages=[dict(role='user', content='Original task')]))
            receipt = seal(dict(job['binding'], status='completed', valid_attempt=True,
                simulated_offline=True, outcome_score=.5, package_path=str(job['package']), requests_used=1))
        else:
            receipt = seal(dict(job['binding'], status='branch_invalid', valid_attempt=False,
                simulated_offline=True, error_type='BranchFailure', attempt_package_path=str(job['package'])))
        write_json(job['path'], receipt)
    for job in [*root_jobs[:3], *jobs[6:10]]:
        request = next((job['package'] / 'raw/model').glob('r*/request.json'))
        write_json(request.with_name('response.bin'), dict(choices=[dict(finish_reason='stop')]))
    own = dict(requests=50, tokens=1000, cost_usd=.1078713, invalid_attempts=0,
               branch_invalid=3, fatal_invalid=0)
    write_json(output / 'admission.json', seal(dict(own,
        plan_sha256=previous['artifact_sha256'], limits=previous['limits'], stop_reason='http_attempt_cap',
        sends=[dict(transport_started=True, status='settled') for _ in range(50)])))
    aggregate = {f'aggregate_{k}': v for k, v in dict(requests=75, tokens=1729388,
        cost_usd=.482426196, http_send_count=75, invalid_attempts=1, branch_invalid=3, fatal_invalid=1).items()}
    write_json(output / 'result.json', seal(dict(own, **aggregate,
        plan_sha256=previous['artifact_sha256'], status='incomplete', d2_gate_open=False,
        simulated_offline=True, completed_branches=7, invalid_or_incomplete_branches=6,
        planned_branches=2282, pending_branches=2269, http_send_count=50)))
    write_json(output.parent / 'supervisor.json', dict(deadline_epoch=previous['deadline_epoch'],
        started_at_epoch=previous['deadline_epoch'] - previous['limits']['max_wall_seconds']))
    write_json(output.parent / 'supervisor_result.json', dict(controller_exit_code=0))
    current, current_jobs = bind_successor(previous, output, tmp_path / 'latest')
    return current, current_jobs, previous, jobs, root_jobs, output


def test_two_generations_charge_own_ledgers_once_and_preserve_attempts(chain):
    current, jobs, _, _, roots, _ = chain
    carried = runtime._predecessor_seeds(current, jobs, True)
    assert carried == dict(requests=75, tokens=1729388, cost_usd=pytest.approx(.482426196),
        http_send_count=75, invalid_attempts=1, completed=7, infrastructure_invalid=6,
        branch_invalid=3, fatal_invalid=1)
    seeds = [j['predecessor_seed'] for j in jobs if 'predecessor_seed' in j]
    assert len(seeds) == 13
    assert jobs[0]['predecessor_seed']['package_path'] == str(roots[0]['package'])
    assert sum(s['status'] == 'branch_invalid' for s in seeds) == 3
    assert all('outcome_score' not in s for s in seeds if s['status'] != 'completed')
    assert not any(j['package'].exists() for j in jobs)


@pytest.mark.parametrize('change', ['missing_ref', 'tampered_ref', 'mode', 'inflight', 'aggregate', 'missing_seed', 'seed_type'])
def test_chain_rejects_unbound_or_uncertain_evidence(chain, change):
    current, jobs, previous, middle_jobs, roots, output = chain
    if change in ('missing_ref', 'tampered_ref'):
        ref = current['predecessor']['supervisor_result']
        if change == 'missing_ref':
            current['bound_files'].pop(ref['path'])
        else:
            Path(ref['path']).write_text('{}')
    else:
        path = middle_jobs[0]['path'] if change in ('mode', 'seed_type') else output / ('admission.json' if change == 'inflight' else 'result.json')
        artifact = read_json(path)
        if change == 'mode':
            artifact['simulated_offline'] = False
        elif change == 'seed_type':
            artifact.update(status='infrastructure_invalid', valid_attempt=False)
            artifact.pop('outcome_score')
        elif change == 'inflight':
            artifact['sends'][0]['status'] = 'inflight'
        elif change == 'aggregate':
            artifact['aggregate_requests'] += 25
        elif change == 'missing_seed':
            middle_jobs[0]['path'].unlink()
        if change != 'missing_seed':
            write_json(path, seal(artifact))
        current, jobs = bind_successor(previous, output, jobs[0]['path'].parent.parent)
    with pytest.raises(ValueError):
        runtime._predecessor_seeds(seal(current), jobs, True)


def test_adding_recovery_rejects_old_length_response(chain):
    current, jobs, _, _, roots, _ = chain
    response = next((roots[0]['package'] / 'raw/model').glob('r*/response.bin'))
    write_json(response, dict(choices=[dict(finish_reason='length')]))
    with pytest.raises(ValueError, match='length response'):
        runtime._predecessor_seeds(current, jobs, True)


def test_unchanged_recovery_policy_accepts_length_response(tmp_path):
    package = tmp_path / 'package'
    write_json(package / 'raw/model/r000001/request.json', dict(model=runtime.MODEL, messages=[]))
    write_json(package / 'raw/model/r000001/response.bin', dict(choices=[dict(finish_reason='length')]))
    model = dict(model=runtime.MODEL, options=dict(truncated_tool_call_policy='output_limit_error_feedback_v1'))
    assert runtime._wire_compatibility(dict(package_path=str(package), requests_used=1), model, model)['all_bodies_unchanged']


@pytest.mark.parametrize('policy,value', [('history_projection_policy', None),
    ('history_projection_policy', 'other'), ('truncated_tool_call_policy', 'other')])
def test_policies_cannot_be_removed_or_changed(chain, policy, value):
    current, jobs, *_ = chain
    if value is None:
        current['model']['options'].pop(policy)
    else:
        current['model']['options'][policy] = value
    with pytest.raises(ValueError, match='polic'):
        runtime._predecessor_seeds(seal(current), jobs, True)


@pytest.mark.parametrize('reason', ['cycle_path', 'cycle_sha', 'depth'])
def test_recursive_cycle_and_depth_guard(chain, reason):
    current, jobs, *_ = chain
    ref = current['predecessor']['plan']
    chain_nodes = ((ref['path'], 'different'),) if reason == 'cycle_path' else (
        (('/different', ref['sha256']),) if reason == 'cycle_sha' else tuple((f'/p{i}', str(i)) for i in range(8)))
    with pytest.raises(ValueError, match='cycle or depth'):
        runtime._predecessor_seeds(current, jobs, True, _chain=chain_nodes)


@pytest.mark.parametrize('change', ['missing_response', 'missing_ancestor_ref', 'tampered_ancestor_ref'])
def test_recursive_raw_evidence_is_required(chain, change):
    current, jobs, previous, middle_jobs, roots, output = chain
    if change == 'missing_response':
        next((roots[0]['package'] / 'raw/model').glob('r*/response.bin')).unlink()
    else:
        ancestor_ref = previous['predecessor']['supervisor_result']
        if change == 'missing_ancestor_ref':
            previous['bound_files'].pop(ancestor_ref['path'])
            current, jobs = bind_successor(seal(previous), output, jobs[0]['path'].parent.parent)
            # Rebind closure identities so the failure is the missing ancestor inventory.
            for name in ('admission', 'result'):
                path = output / (name + '.json')
                artifact = read_json(path)
                artifact['plan_sha256'] = seal(previous)['artifact_sha256']
                write_json(path, seal(artifact))
            current, jobs = bind_successor(seal(previous), output, jobs[0]['path'].parent.parent)
        else:
            Path(ancestor_ref['path']).write_text('{}')
    with pytest.raises((ValueError, FileNotFoundError)):
        runtime._predecessor_seeds(current, jobs, True)


def test_native_branch_failure_metadata_survives_seed(chain):
    current, jobs, previous, middle_jobs, _, output = chain
    path = middle_jobs[10]['path']
    receipt = read_json(path)
    receipt.update(status='incomplete', failure_scope='branch', error_source='model_protocol',
                   failure_phase='execution', testable=False, no_sample_replacement=True)
    write_json(path, seal(receipt))
    current, jobs = bind_successor(previous, output, jobs[0]['path'].parent.parent)
    runtime._predecessor_seeds(current, jobs, True)
    seed = jobs[10]['predecessor_seed']
    assert seed['status'] == 'branch_invalid'
    assert seed['error_source'] == 'model_protocol'
    assert seed['failure_phase'] == 'execution'
    assert seed['testable'] is False
    assert seed['no_sample_replacement'] is True
    assert 'outcome_score' not in seed
