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
    monkeypatch.setattr(runtime, '_jobs', lambda p, o: copy.deepcopy(jobs))
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
        startup = read_json(output.parent / 'integrity_preflight.json')
        assert startup['plan_sha256'] == plan['artifact_sha256']
        assert startup['planned_jobs'] == 7
        assert startup['actor_requests'] == 0
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
    assert result['stop_reason'] == 'http_attempt_cap'
    assert result['execution_finished'] is True
    assert result['schedule_complete'] is False


def test_uncertain_package_never_retried(plan, tmp_path, monkeypatch):
    output = tmp_path / 'out'
    jobs = subset(monkeypatch, plan, output, 2)
    jobs[0]['package'].mkdir(parents=True)
    with pytest.raises(ValueError, match='uncertain'):
        runtime.execute(plan, output, branch_adapter=lambda *a, **kw: pytest.fail('uncertain retry'))
    assert not (output / 'admission.json').exists()


@pytest.fixture
def successor(plan, tmp_path):
    """Six retained attempts: three complete no-op bodies and three failures."""
    old_output = tmp_path / 'previous' / 'execution'
    old_jobs = runtime._scheduled_jobs(plan, old_output)
    for i, job in enumerate(old_jobs[:6]):
        if i < 3:
            package = job['package']
            request = package / 'raw/model/r000002/request.json'
            write_json(request, dict(model=runtime.MODEL, temperature=1, max_tokens=8192,
                messages=[dict(role='user', content='Original task')]))
            receipt = seal(dict(job['binding'], status='completed', valid_attempt=True,
                simulated_offline=True, outcome_score=.5, package_path=str(package), requests_used=1))
        else:
            receipt = seal(dict(job['binding'], status='incomplete', valid_attempt=False,
                simulated_offline=True, error_type='ValueError', attempt_package_path=str(job['package'])))
        write_json(job['path'], receipt)
    old_plan_path = tmp_path / 'previous/plan.json'
    write_json(old_plan_path, plan)
    old_admission_path = old_output / 'admission.json'
    write_json(old_admission_path, seal(dict(plan_sha256=plan['artifact_sha256'], limits=plan['limits'],
        requests=25, tokens=1728388, cost_usd=.374554896, invalid_attempts=1,
        stop_reason='invalid_attempt', sends=[dict(transport_started=True) for _ in range(25)])))
    old_result_path = old_output / 'result.json'
    write_json(old_result_path, seal(dict(plan_sha256=plan['artifact_sha256'], status='incomplete',
        d2_gate_open=False, simulated_offline=True, completed_branches=3,
        invalid_or_incomplete_branches=3, planned_branches=2282, pending_branches=2276,
        requests=25, tokens=1728388, cost_usd=.374554896, http_send_count=25)))
    supervisor_path = tmp_path / 'previous/supervisor.json'
    supervisor_result_path = tmp_path / 'previous/supervisor_result.json'
    deadline = time.time() + 1000
    write_json(supervisor_path, dict(deadline_epoch=deadline, started_at_epoch=deadline-86400))
    write_json(supervisor_result_path, dict(controller_exit_code=0))
    predecessor = {name: dict(path=str(path), sha256=file_digest(path)) for name, path in (
        ('plan', old_plan_path), ('result', old_result_path), ('admission', old_admission_path),
        ('supervisor', supervisor_path), ('supervisor_result', supervisor_result_path))}
    predecessor['receipt_directory'] = str(old_output / 'receipts')
    updated = copy.deepcopy(plan)
    updated['predecessor'] = predecessor
    updated['deadline_epoch'] = deadline
    updated['model']['options']['history_projection_policy'] = 'group_tool_results_before_images_v1'
    updated['limits'].update(max_requests=plan['limits']['max_requests']-25,
        max_tokens=plan['limits']['max_tokens']-1728388,
        max_cost_usd=plan['limits']['max_cost_usd']-.374554896, max_wall_seconds=1000)
    for ref in predecessor.values():
        if isinstance(ref, dict):
            updated['bound_files'][ref['path']] = ref['sha256']
    for job in old_jobs[:6]:
        updated['bound_files'][str(job['path'])] = file_digest(job['path'])
    return seal(updated), old_jobs


def test_successor_carries_six_attempts_without_duplicate_calls(successor, tmp_path, monkeypatch):
    plan, old_jobs = successor
    output = tmp_path / 'successor'
    jobs = subset(monkeypatch, plan, output, 8)
    calls = []
    def adapter(unit, *args, **kwargs):
        calls.append(unit['unit_id'])
        return completed()
    result = runtime.execute(plan, output, branch_adapter=adapter)
    assert calls == [j['unit']['unit_id'] for j in jobs[6:8]]
    assert result['completed_branches'] == 5
    assert result['carried_completed_branches'] == 3
    assert result['carried_infrastructure_invalid_branches'] == 3
    assert result['pending_branches'] == 0
    assert result['execution_finished'] is True
    assert result['status'] == 'incomplete'
    assert result['requests'] == 0
    assert result['aggregate_requests'] == result['aggregate_http_send_count'] == 25
    assert result['aggregate_tokens'] == 1728388
    assert result['aggregate_cost_usd'] == pytest.approx(.374554896)
    assert result['invalid_attempts'] == 0
    assert result['aggregate_invalid_branches'] == 3
    assert result['aggregate_invalid_attempts'] == 1
    assert result['d2_gate_open'] is False
    carried = read_json(jobs[0]['path'])
    assert carried['package_path'] == str(old_jobs[0]['package'])
    assert carried['wire_compatibility_proof']['all_bodies_unchanged'] is True
    assert carried['predecessor_target_model_sha256'] == old_jobs[0]['binding']['target_model_sha256']
    invalid = read_json(jobs[3]['path'])
    assert invalid['status'] == 'infrastructure_invalid'
    assert invalid['valid_attempt'] is False
    assert 'outcome_score' not in invalid
    second = runtime.execute(plan, output, branch_adapter=lambda *a, **kw: pytest.fail('carried or cached branch reran'))
    assert second['aggregate_requests'] == 25


def test_successor_rejects_non_noop_wire_before_admission(successor, tmp_path, monkeypatch):
    plan, old_jobs = successor
    from c3_sandbox.l1.models.opencode import project_tool_history
    request = old_jobs[0]['package'] / 'raw/model/r000002/request.json'
    original = read_json(request)
    original['messages'] = [
        dict(role='assistant', tool_calls=[dict(id='a'), dict(id='b')]),
        dict(role='tool', tool_call_id='a', content='image result'),
        dict(role='user', content=[dict(type='text', text='Image returned by tool call a'),
                                  dict(type='image_url', image_url=dict(url='data:image/png;base64,AA=='))]),
        dict(role='tool', tool_call_id='b', content='second result'),
    ]
    assert project_tool_history(original['messages'])[1]['changed'] is True
    write_json(request, original)
    output = tmp_path / 'successor'
    subset(monkeypatch, plan, output, 8)
    with pytest.raises(ValueError, match='would change'):
        runtime.execute(plan, output, branch_adapter=lambda *a, **kw: pytest.fail('paid work admitted'))
    assert not (output / 'admission.json').exists()
    assert not (output / 'receipts').exists()


@pytest.mark.parametrize('change', ['deadline', 'requests', 'tokens', 'cost', 'policy', 'receipt_binding'])
def test_successor_rejects_renewed_budget_or_changed_evidence(successor, tmp_path, monkeypatch, change):
    plan, old_jobs = successor
    altered = copy.deepcopy(plan)
    if change == 'deadline':
        altered['deadline_epoch'] += 1
    elif change == 'requests':
        altered['limits']['max_requests'] += 1
    elif change == 'tokens':
        altered['limits']['max_tokens'] += 1
    elif change == 'cost':
        altered['limits']['max_cost_usd'] += .01
    elif change == 'policy':
        altered['model']['parameters']['unapproved'] = True
    else:
        altered['bound_files'].pop(str(old_jobs[0]['path']))
    altered = seal(altered)
    output = tmp_path / 'successor'
    subset(monkeypatch, altered, output, 8)
    with pytest.raises(ValueError):
        runtime.execute(altered, output, branch_adapter=lambda *a, **kw: pytest.fail('paid work admitted'))
    assert not (output / 'admission.json').exists()


def test_successor_rejects_unreceipted_old_attempt(successor, tmp_path, monkeypatch):
    plan, old_jobs = successor
    (old_jobs[0]['package'].parent / 'unknown-attempt').mkdir()
    output = tmp_path / 'successor'
    subset(monkeypatch, plan, output, 8)
    with pytest.raises(ValueError, match='uncertain predecessor branch'):
        runtime.execute(plan, output, branch_adapter=lambda *a, **kw: pytest.fail('uncertain attempt admitted'))
    assert not (output / 'admission.json').exists()


def test_successor_rejects_unclosed_predecessor_controller(successor, tmp_path, monkeypatch):
    plan, _ = successor
    altered = copy.deepcopy(plan)
    from pathlib import Path
    ref = altered['predecessor']['supervisor_result']
    write_json(Path(ref['path']), dict(controller_exit_code=None))
    ref['sha256'] = file_digest(Path(ref['path']))
    altered['bound_files'][ref['path']] = ref['sha256']
    altered = seal(altered)
    output = tmp_path / 'successor'
    subset(monkeypatch, altered, output, 8)
    with pytest.raises(ValueError, match='must have exited'):
        runtime.execute(altered, output, branch_adapter=lambda *a, **kw: pytest.fail('overlapping batch admitted'))
    assert not (output / 'admission.json').exists()


@pytest.mark.parametrize('source,phase', [('model_protocol', 'execution'),
    ('model_transport', 'execution'), ('scorer', 'scoring'),
    ('branch_wall_time', 'execution')])
def test_known_branch_failure_continues_frozen_jobs_without_replacement(plan, tmp_path, monkeypatch, source, phase):
    output = tmp_path / 'out'
    jobs = subset(monkeypatch, plan, output, 7)
    calls = []
    def adapter(unit, *args, **kwargs):
        calls.append(unit['unit_id'])
        if len(calls) == 1:
            raise runtime.BranchFailure(source, phase)
        return completed()
    result = runtime.execute(plan, output, branch_adapter=adapter)
    assert sorted(calls) == sorted(j['unit']['unit_id'] for j in jobs)
    assert result['completed_branches'] == 6
    assert result['invalid_or_incomplete_branches'] == 1
    assert result['pending_branches'] == 0
    assert result['branch_invalid'] == result['aggregate_branch_invalid'] == 1
    assert result['invalid_attempts'] == result['fatal_invalid'] == 0
    assert result['stop_reason'] is None
    assert result['execution_finished'] is result['process_finished'] is result['schedule_complete'] is True
    invalid = read_json(jobs[0]['path'])
    assert invalid['failure_scope'] == 'branch'
    assert invalid['error_source'] == source
    assert invalid['failure_phase'] == phase
    assert invalid['testable'] is False
    assert invalid['no_sample_replacement'] is True
    assert invalid['no_resampling'] is True
    assert 'outcome_score' not in invalid
    with pytest.raises(ValueError, match='incomplete'):
        runtime.execute(plan, output, branch_adapter=completed)


def test_parallel_branch_failure_does_not_cancel_other_active_jobs(plan, tmp_path, monkeypatch):
    output = tmp_path / 'out'
    jobs = subset(monkeypatch, plan, output, 7)
    barrier = threading.Barrier(3)
    calls = []
    lock = threading.Lock()
    def adapter(unit, *args, **kwargs):
        with lock:
            index = len(calls)
            calls.append(unit['unit_id'])
        if index in (1, 2, 3):
            barrier.wait(timeout=10)
            if index == 1:
                raise runtime.BranchFailure('model_protocol')
            time.sleep(.02)
        return completed()
    result = runtime.execute(plan, output, branch_adapter=adapter)
    assert len(calls) == len(jobs)
    assert result['completed_branches'] == 6
    assert result['branch_invalid'] == 1
    assert result['stop_reason'] is None


@pytest.mark.parametrize('fault', ['integrity', 'provider'])
def test_global_failure_stops_admission_and_marks_process_finished(plan, tmp_path, monkeypatch, fault):
    output = tmp_path / 'out'
    jobs = subset(monkeypatch, plan, output, 5)
    calls = []
    def adapter(*args, limiter, **kwargs):
        calls.append(1)
        if fault == 'provider':
            import json
            from types import SimpleNamespace
            assert limiter.acquire(5)
            limiter.observe(SimpleNamespace(status=429, body=json.dumps({'error': 'SECRET quota'}).encode()))
            limiter.reconcile(5, None)
            raise runtime.BranchFailure('model_transport')
        raise ValueError('SECRET source/provenance mismatch')
    result = runtime.execute(plan, output, branch_adapter=adapter)
    assert calls == [1]
    assert result['completed_branches'] == 0
    assert result['pending_branches'] == 4
    assert result['fatal_invalid'] == 1
    assert result['branch_invalid'] == 0
    assert result['stop_reason'] == ('provider_status_429' if fault == 'provider' else 'invalid_attempt')
    assert result['execution_finished'] is result['process_finished'] is True
    assert result['schedule_complete'] is False
    assert 'SECRET' not in jobs[0]['path'].read_text()
    assert read_json(jobs[0]['path'])['failure_scope'] == 'global'


@pytest.mark.parametrize('termination,protocol,source', [
    ('protocol_error', 'invalid', 'model_protocol'),
    ('model_transport_error', 'valid', 'model_transport')])
def test_verified_manifest_classifies_only_known_model_failures(tmp_path, termination, protocol, source):
    manifest = dict(episode_id='episode', recording_mode='live', runtime=dict(adapter='docker-cold'),
        status=dict(termination=termination, protocol_validity=protocol,
                    execution_validity='invalid', evidence_completeness='complete'))
    with pytest.raises(runtime.BranchFailure) as caught:
        runtime._verified_branch_score(tmp_path, {}, manifest, dict(index_sha256='a'*64))
    assert caught.value.source == source
    assert caught.value.evidence['termination'] == termination
    manifest['status']['evidence_completeness'] = 'incomplete'
    with pytest.raises(ValueError, match='provenance'):
        runtime._verified_branch_score(tmp_path, {}, manifest, dict(index_sha256='a'*64))


def test_verified_score_preserves_delivery_zero_and_scoring_failures(tmp_path, monkeypatch):
    from c3_sandbox.l1.evaluation import study
    manifest = dict(episode_id='episode', recording_mode='live', runtime=dict(adapter='docker-cold'),
        status=dict(termination='assistant_finished', execution_validity='valid',
                    evaluation_validity='invalid', evidence_completeness='complete'))
    monkeypatch.setattr(study, 'study_outcome', lambda *a, **kw: dict(study_valid=True, study_outcome_y=0))
    monkeypatch.setattr(runtime, 'item_score', lambda *a, **kw: 0.0)
    assert runtime._verified_branch_score(tmp_path, {}, manifest, dict(index_sha256='a'*64)) == 0
    monkeypatch.setattr(study, 'study_outcome', lambda *a, **kw: dict(study_valid=False, study_classification='infrastructure_failure'))
    with pytest.raises(runtime.BranchFailure) as caught:
        runtime._verified_branch_score(tmp_path, {}, manifest, dict(index_sha256='a'*64))
    assert caught.value.source == 'scorer'
    monkeypatch.setattr(study, 'study_outcome', lambda *a, **kw: dict(study_valid=False, study_classification='raw_projection_mismatch'))
    with pytest.raises(ValueError, match='unrecognized'):
        runtime._verified_branch_score(tmp_path, {}, manifest, dict(index_sha256='a'*64))


@pytest.mark.parametrize('global_wall_expired', [False, True])
def test_branch_wall_budget_is_local_until_global_deadline(plan, tmp_path, monkeypatch, global_wall_expired):
    output = tmp_path / 'out'
    jobs = subset(monkeypatch, plan, output, 5)
    calls = []
    def adapter(*args, limiter, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            if global_wall_expired:
                # Expire the global ledger clock while this branch is running.
                monkeypatch.setattr('l2_sandbox.opencode_admission.time.time',
                    lambda: limiter.snapshot()['started_at'] + limiter.caps['max_wall_seconds'])
            manifest = dict(episode_id='episode', recording_mode='live', runtime=dict(adapter='docker-cold'),
                status=dict(termination='wall_time_budget', execution_validity='invalid', evidence_completeness='complete'))
            runtime._verified_branch_score(output, {}, manifest, dict(index_sha256='a'*64),
                global_stop_reason=limiter.exhaustion_reason())
        return completed()
    result = runtime.execute(plan, output, branch_adapter=adapter)
    invalid = read_json(jobs[0]['path'])
    assert invalid['testable'] is False
    assert 'outcome_score' not in invalid
    if global_wall_expired:
        assert calls == [1]
        assert result['stop_reason'] == 'wall_cap'
        assert result['completed_branches'] == 0
        assert invalid['failure_scope'] == 'global'
        assert result['schedule_complete'] is False
    else:
        assert len(calls) == 5
        assert result['stop_reason'] is None
        assert result['completed_branches'] == 4
        assert invalid['failure_scope'] == 'branch'
        assert invalid['error_source'] == 'branch_wall_time'
        assert result['schedule_complete'] is True
    assert result['process_finished'] is True


def test_real_study_grading_receipts_preserve_delivery_zero_and_classify_infra(tmp_path, monkeypatch):
    from c3_sandbox.l1 import package
    from c3_sandbox.l1.evaluation.officeval import interpret
    from c3_sandbox.l1.evaluation.study import study_outcome
    manifest = dict(episode_id='episode', recording_mode='live', runtime=dict(adapter='docker-cold'),
        task=dict(benchmark='officeval', task_id='officeval_003'),
        status=dict(termination='assistant_finished', execution_validity='valid',
                    evaluation_validity='invalid', evidence_completeness='complete'))
    raw = dict(id='003', file_name='', status='error', error='FileNotFoundError: no docx',
        dim1_pass=False, dim1_reason='', dim2_items=[], total_score=0, max_score=0)
    grade = dict(official_raw=raw, projection=interpret(raw, 'officeval_003'),
        process=dict(container_exit_code=0, stop_reason=None))
    write_json(tmp_path / 'manifest.json', manifest)
    write_json(tmp_path / 'official_grade.json', grade)
    write_json(tmp_path / 'grading/worker-output/worker.json', dict(official_returned=True, exception=None))
    monkeypatch.setattr(package, 'verify_package', lambda *a, **kw: dict(verified=True))
    outcome = study_outcome(tmp_path, verify_evidence=True)
    assert outcome['study_classification'] == 'valid_zero'
    assert outcome['zero_reason'] == 'missing_or_damaged_delivery'
    assert runtime._verified_branch_score(tmp_path, {}, manifest, dict(index_sha256='a'*64)) == 0
    grade.update(status='infra_error', error_kind='infrastructure')
    write_json(tmp_path / 'official_grade.json', grade)
    assert study_outcome(tmp_path, verify_evidence=True)['study_classification'] == 'infrastructure_failure'
    with pytest.raises(runtime.BranchFailure) as caught:
        runtime._verified_branch_score(tmp_path, {}, manifest, dict(index_sha256='a'*64))
    assert caught.value.source == 'scorer'
    assert caught.value.phase == 'scoring'
    # A bad package cannot qualify as a branch-local scorer failure.
    monkeypatch.setattr(package, 'verify_package', lambda *a, **kw: dict(verified=False))
    with pytest.raises(ValueError, match='unrecognized'):
        runtime._verified_branch_score(tmp_path, {}, manifest, dict(index_sha256='a'*64))
