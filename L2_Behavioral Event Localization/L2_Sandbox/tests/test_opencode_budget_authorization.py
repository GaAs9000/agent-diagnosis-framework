"""Explicit finite budget extensions retain all predecessor charges and attempts."""
import copy
import time

import pytest

from l2_sandbox import opencode_d1_runtime as runtime
from l2_sandbox.opencode_admission import Admission, DEFAULT_CAPS, validate_budget_authorization
from l2_sandbox.v3_execution import file_digest, read_json, seal, write_json
from test_opencode_d1_runtime import plan, successor
from test_opencode_multigeneration import bind_successor, chain


@pytest.fixture
def extension(chain, tmp_path):
    current, jobs, previous, *_ = chain
    current = copy.deepcopy(current)
    current['model'] = copy.deepcopy(previous['model'])
    start = time.time()
    grant = seal(dict(schema='opencode-d1-budget-authorization-v1', execution_authorized=True,
        scope='cross_actor_D1_budget_extension', source='direct_user_message_20261019',
        exact_user_text='A', predecessor_plan_sha256=previous['artifact_sha256'],
        cumulative_limits=dict(max_requests=22000, max_tokens=1_000_000_000, max_cost_usd=50.0),
        phase_started_at_epoch=start, deadline_epoch=start + 86400, max_wall_seconds=86400))
    path = tmp_path / 'authorization.json'
    write_json(path, grant)
    current['budget_authorization'] = dict(path=str(path), sha256=file_digest(path))
    current['bound_files'][str(path)] = file_digest(path)
    current['deadline_epoch'] = grant['deadline_epoch']
    current['limits'].update(max_requests=22000 - 75, max_tokens=1_000_000_000 - 1729388,
                             max_cost_usd=50.0 - .482426196, max_wall_seconds=86400)
    current = seal(current)
    return current, runtime._scheduled_jobs(current, jobs[0]['path'].parent.parent), grant, path


def test_authorized_extension_subtracts_all_ancestors_once(extension, tmp_path):
    current, jobs, grant, _ = extension
    carried = runtime._predecessor_seeds(current, jobs, True)
    assert carried['requests'] == 75
    assert carried['tokens'] == 1729388
    assert carried['cost_usd'] == pytest.approx(.482426196)
    assert carried['completed'] == 7
    assert carried['infrastructure_invalid'] == 6
    assert sum('predecessor_seed' in job for job in jobs) == 13
    ledger = Admission(tmp_path / 'admission.json', current['artifact_sha256'], current['limits'],
                       absolute_deadline=current['deadline_epoch'],
                       budget_authorization=runtime._admission_authorization(current))
    assert ledger.caps['max_requests'] + carried['requests'] == 22000
    assert ledger.caps['max_tokens'] + carried['tokens'] == 1_000_000_000
    assert ledger.caps['max_cost_usd'] + carried['cost_usd'] == pytest.approx(50)
    assert ledger.remaining_seconds() <= 86400
    assert ledger.snapshot()['requests'] == 0


@pytest.mark.parametrize('change', ['missing', 'unbound', 'predecessor', 'deadline', 'wall',
                                  'cap', 'tokens', 'own_only', 'model', 'unsealed', 'unauthorized'])
def test_forged_or_mismatched_extension_rejected(extension, change):
    current, jobs, grant, path = extension
    current, grant = copy.deepcopy(current), copy.deepcopy(grant)
    if change == 'missing':
        current.pop('budget_authorization')
    elif change == 'unbound':
        current['bound_files'].pop(str(path))
    elif change in ('predecessor', 'wall', 'unsealed', 'unauthorized'):
        if change == 'predecessor':
            grant['predecessor_plan_sha256'] = '0' * 64
        elif change == 'wall':
            grant['deadline_epoch'] += 1
        elif change == 'unauthorized':
            grant['execution_authorized'] = False
        if change != 'unsealed':
            grant = seal(grant)
        else:
            grant['exact_user_text'] = 'tampered'
        write_json(path, grant)
        current['budget_authorization']['sha256'] = file_digest(path)
        current['bound_files'][str(path)] = file_digest(path)
    elif change == 'deadline':
        current['deadline_epoch'] += 1
    elif change == 'cap':
        current['limits']['max_requests'] += 1
    elif change == 'tokens':
        current['limits']['max_tokens'] = 1_000_000_000
    elif change == 'own_only':
        current['limits']['max_requests'] = 22000 - 50
    elif change == 'model':
        current['model']['options']['truncated_tool_call_policy'] = 'output_limit_error_feedback_v1'
    with pytest.raises(ValueError):
        runtime._predecessor_seeds(seal(current), jobs, True)


@pytest.mark.parametrize('caps', [dict(max_requests=15001), dict(max_cost_usd=12.01)])
def test_admission_keeps_original_hard_caps_without_authorization(tmp_path, caps):
    with pytest.raises(ValueError, match='hard limit'):
        Admission(tmp_path / 'admission.json', 'a' * 64, caps)
    assert not (tmp_path / 'admission.json').exists()


def test_grant_cannot_raise_other_frozen_admission_caps(extension, tmp_path):
    current, _, grant, _ = extension
    caps = {**current['limits'], 'max_tokens_per_request': DEFAULT_CAPS['max_tokens_per_request'] + 1}
    with pytest.raises(ValueError, match='hard limit'):
        Admission(tmp_path / 'admission.json', current['artifact_sha256'], caps,
                  absolute_deadline=current['deadline_epoch'], budget_authorization=grant)


def test_admission_rejects_deadline_beyond_authorized_phase(extension, tmp_path):
    current, _, grant, _ = extension
    with pytest.raises(ValueError, match='deadline'):
        Admission(tmp_path / 'admission.json', current['artifact_sha256'], current['limits'],
                  absolute_deadline=grant['deadline_epoch'] + 1, budget_authorization=grant)


def test_authorization_cannot_raise_cumulative_token_cap(extension):
    _, _, grant, _ = extension
    grant = copy.deepcopy(grant)
    grant['cumulative_limits']['max_tokens'] += 1
    with pytest.raises(ValueError, match='authorization'):
        validate_budget_authorization(seal(grant), grant['predecessor_plan_sha256'])


def test_branch_storage_changes_only_new_package_destination(plan, tmp_path):
    output = tmp_path / 'execution'
    standard = runtime._scheduled_jobs(plan, output)
    relocated = {**plan, 'branch_storage_root': str(tmp_path / 'external/branches')}
    # Operational location is not part of the scientific branch binding.
    moved = runtime._scheduled_jobs(relocated, output)
    assert [j['binding'] for j in moved] == [j['binding'] for j in standard]
    assert [j['path'] for j in moved] == [j['path'] for j in standard]
    assert all(j['package'].parent == tmp_path / 'external/branches' for j in moved)


def test_branch_storage_rejects_relative_and_symlink_ancestors(plan, tmp_path):
    link = tmp_path / 'linked'
    link.symlink_to(tmp_path, target_is_directory=True)
    for root in ('relative/branches', str(link / 'branches')):
        with pytest.raises(ValueError, match='storage root'):
            runtime._scheduled_jobs({**plan, 'branch_storage_root': root}, tmp_path / 'execution')


@pytest.mark.parametrize('stray_directory', ['branches', 'provenance'])
def test_external_predecessor_stray_attempt_is_rejected(chain, tmp_path, stray_directory):
    _, jobs, previous, _, _, output = chain
    previous = copy.deepcopy(previous)
    storage = tmp_path / 'external/branches'
    previous['branch_storage_root'] = str(storage)
    previous = seal(previous)
    for name in ('admission', 'result'):
        path = output / (name + '.json')
        artifact = read_json(path)
        artifact['plan_sha256'] = previous['artifact_sha256']
        write_json(path, seal(artifact))
    (storage.parent / stray_directory / 'unreceipted_attempt').mkdir(parents=True)
    current, current_jobs = bind_successor(previous, output, tmp_path / 'next')
    with pytest.raises(ValueError, match='uncertain predecessor branch'):
        runtime._predecessor_seeds(current, current_jobs, True)
