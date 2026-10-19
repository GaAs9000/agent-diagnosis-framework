"""Bounded replay behavior using fake runtime/native validator, no Docker/model."""
import importlib.util
from pathlib import Path
import sys
import time

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('prospective_replay_test_target', SCRIPTS / 'replay_prospective_tools.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    class Validator:
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr(m.mini, 'NativePiValidator', Validator)
    plan = {'schema': m.SCHEMA, 'limits': m.LIMITS.copy(), 'source_identity': {},
            'native_identity': {}, 'package_indices': {'/fake': 'index'},
            'batches': [{'batch_id': 'one', 'package': '/fake', 'request_id': 'r0'}]}
    monkeypatch.setattr(m, 'source_identity', lambda: {})
    monkeypatch.setattr(m.mini, 'native_identity', lambda validator=None: {})
    monkeypatch.setattr(m.mini, 'verified', lambda *args: None)
    monkeypatch.setattr(m, 'select_batch', lambda b: {'calls': [{}], 'request_exact': {}})
    path = tmp_path / 'plan.json'; m.mini.write_json(tmp_path, 'plan.json', plan)
    return plan, path


def row(exact=True):
    return {'status': 'complete', 'cleanup': {'confirmed': True},
            'calls': [{'classification': 'exact' if exact else 'unexplained',
                       'unexplained_state_difference': not exact,
                       'unexplained_tool_difference': not exact}]}


def test_no_model_network_calls_and_no_second_exact_replay(frozen, tmp_path):
    _, path = frozen; seen = []
    def runner(batch, output, deadline, *, admit, **kw):
        seen.append(output.name)
        with pytest.raises(AssertionError):
            m.mini.OpenAICompatibleBackend.send(None, b'forbidden', 1)
        with pytest.raises(AssertionError):
            m.socket.create_connection(('model.invalid', 443))
        admit('one', 0, True)
        return row()
    report = m.execute(path, tmp_path / 'out', m.mini.file_digest(path), runner=runner)
    assert seen == ['one_replay1']
    assert report['status'] == 'complete' and report['mechanism_gate_pass']
    assert report['tool_attempts'] == 1 and report['actor_requests'] == 0
    with pytest.raises(ValueError, match='new output'):
        m.execute(path, tmp_path / 'out', m.mini.file_digest(path), runner=runner)


def test_second_replay_never_erases_first_failure(frozen, tmp_path):
    _, path = frozen; seen = []
    def runner(batch, output, deadline, *, admit, **kw):
        seen.append(output.name); admit('one', 0, True)
        return row(exact=len(seen) == 2)
    report = m.execute(path, tmp_path / 'out', m.mini.file_digest(path), runner=runner)
    assert len(seen) == 2 and report['status'] == 'blocked_unexplained_difference'
    assert not report['mechanism_gate_pass']
    assert report['batches'][0]['replays'][0]['calls'][0]['unexplained_state_difference']


def test_partial_failure_not_coverage(frozen, tmp_path):
    _, path = frozen
    def runner(*args, **kw):
        kw['admit']('one', 0, False)
        raise TimeoutError('tool stalled')
    report = m.execute(path, tmp_path / 'out', m.mini.file_digest(path), runner=runner)
    assert report['status'] == 'blocked_error' and report['not_run'] == ['one']
    assert report['tool_attempts'] == 1 and not report['mechanism_gate_pass']


@pytest.mark.parametrize('field', ['source_identity', 'native_identity', 'limits'])
def test_plan_drift_fails_before_output(frozen, tmp_path, field):
    plan, path = frozen
    plan[field] = {'tampered': True}; path.unlink(); m.mini.write_json(tmp_path, 'plan.json', plan)
    with pytest.raises(ValueError):
        m.execute(path, tmp_path / 'out', m.mini.file_digest(path))
    assert not (tmp_path / 'out').exists()


def test_original_truncation_contract():
    obs = {'stdout': '123456', 'stderr': 'bad', 'output': 'abcdef', 'exit_code': 1}
    assert m.delivered_observation(obs, 3) == {
        'stdout': '123', 'stderr': 'bad', 'output': 'abc', 'exit_code': 1,
        'observation_truncation': {'fields': ['stdout', 'output'], 'limit_chars': 3, 'raw_retained': True}}
    assert obs['stdout'] == '123456'


def test_cleanup_needs_each_created_container_removed():
    assert m.cleanup_receipt([])['confirmed']
    events = [{'kind': 'runtime.container.created', 'data': {'container_id': 'x'}}]
    assert not m.cleanup_receipt(events)['confirmed']
    events.append({'kind': 'runtime.container.removed', 'data': {'container_id': 'x', 'removed': True}})
    assert m.cleanup_receipt(events)['confirmed']


@pytest.mark.parametrize('borrowed', [False, True])
def test_native_invalid_sibling_runs_from_single_restored_workspace(tmp_path, monkeypatch, borrowed):
    config = {'runtime': {'max_workspace_bytes': 10000, 'max_files': 10, 'tool_timeout_seconds': 7},
              'recording': {'observation_limit_chars': 100}}
    expected = {'files': [], 'directories': [], 'total_bytes': 0, 'content_sha256': 'empty'}
    calls = []
    for i, valid in enumerate((False, True)):
        calls.append({'call': {'call_id': f'ep:c{i}', 'native_id': f'n{i}', 'name': 'write',
                               'arguments': {'ok': valid}}, 'native_validation_ok': valid})
    monkeypatch.setattr(m, 'select_batch', lambda batch: {'config': config, 'checkpoint': {
        'qualification': {'image_id': 'image'}}, 'before_snapshot': {}, 'calls': calls, 'request_exact': {}})
    monkeypatch.setattr(m.mini, 'referenced', lambda *args: tmp_path / 'fake')
    monkeypatch.setattr(m.mini, 'read_json', lambda path: expected)
    restored = []
    def restore(root, before, workspace):
        restored.append(workspace); workspace.mkdir(); (workspace / 'proof').write_text('same')
    monkeypatch.setattr(m.mini, 'restore', restore)
    monkeypatch.setattr(m.mini, 'capture', lambda *args: expected)
    monkeypatch.setattr(m.mini, 'native_identity', lambda validator: {})
    comparisons = []
    def compare(root, recorded, actual, *args):
        comparisons.append(actual.observation)
        return {'classification': 'exact'}
    monkeypatch.setattr(m, 'compare_call', compare)
    executed = []
    class Runtime:
        def __init__(self, config): pass
        def set_event_sink(self, sink): self.sink = sink
        def preflight(self): return {'image_id': 'image'}
        def execute(self, call, workspace, timeout):
            assert (workspace / 'proof').read_text() == 'same'
            assert 0 < timeout <= 7
            executed.append(call.call_id)
            return m.mini.Result({'exit_code': 0})
    lifecycle = []
    class Validator:
        def __enter__(self): lifecycle.append('open'); return self
        def __exit__(self, *args): lifecycle.append('close')
        def validate(self, name, args, native_id, **kw):
            return {'ok': args['ok'], 'validated_arguments': {}, 'native_result': {'error': 'native'}}
    admissions = []
    output = tmp_path / 'batch'
    report = m.run_batch({'package': str(tmp_path), 'batch_id': 'one'}, output, time.monotonic() + 60,
                         expected_native_identity={}, admit=lambda *args: admissions.append(args),
                         runtime_factory=Runtime, validator_factory=Validator,
                         shared_validator=Validator() if borrowed else None)
    assert lifecycle == ([] if borrowed else ['open', 'close'])
    assert len(restored) == 1 and executed == ['ep:c1']
    assert comparisons == [{'error': 'native'}, {'exit_code': 0}]
    assert [a[2] for a in admissions] == [False, True]
    event_paths = list(output.glob('runtime_event_*.json'))
    # Only the valid sibling emits a start; validation feedback does not.
    assert len(event_paths) == 1
    assert report['status'] == 'complete' and report['cleanup']['confirmed']


def test_hard_admission_cap_retains_exact_attempt_count(frozen, tmp_path):
    _, path = frozen
    def runner(batch, output, deadline, *, admit, **kw):
        for ordinal in range(6301):
            admit('one', ordinal, True)
        pytest.fail('cap must stop the 6301st call')
    report = m.execute(path, tmp_path / 'out', m.mini.file_digest(path), runner=runner)
    assert report['tool_attempts'] == 6300 and report['status'] == 'blocked_error'
    assert len(list((tmp_path / 'out/admissions').iterdir())) == 6300
    assert not report['mechanism_gate_pass']


@pytest.mark.parametrize('extension', ['.xlsx', '.pptx', '.txt'])
def test_other_state_variance_is_unexplained(tmp_path, monkeypatch, extension):
    expected = {'files': [{'path': 'out' + extension, 'sha256': 'old'}], 'directories': [],
                'total_bytes': 1, 'content_sha256': 'old'}
    actual_state = {**expected, 'files': [{'path': 'out' + extension, 'sha256': 'new'}],
                    'content_sha256': 'new'}
    for name in ('stdout', 'stderr'):
        (tmp_path / name).write_bytes(b'')
    monkeypatch.setattr(m.mini, 'referenced', lambda root, ref: tmp_path / ref)
    monkeypatch.setattr(m.mini, 'read_json', lambda path: expected)
    # Production ZIP classifier already rejects every non-docx extension.
    recorded = {'after_snapshot': 'state', 'original_result': {'stdout': 'stdout', 'stderr': 'stderr'},
                'original_observation': {'exit_code': 0}, 'original_delivered': {'exit_code': 0},
                'call': {'call_id': 'c', 'name': 'write', 'arguments': {}}}
    out = tmp_path / 'out'; out.mkdir()
    result = m.compare_call(tmp_path, recorded, m.mini.Result({'exit_code': 0}),
                            tmp_path, actual_state, out, {'recording': {'observation_limit_chars': 100}})
    assert result['classification'] == 'unexplained' and result['unexplained_state_difference']


def test_arbitrary_stdout_difference_is_not_ls_variance(tmp_path, monkeypatch):
    expected = {'files': [], 'directories': [], 'total_bytes': 0, 'content_sha256': 'same'}
    (tmp_path / 'stdout').write_bytes(b'old'); (tmp_path / 'stderr').write_bytes(b'')
    monkeypatch.setattr(m.mini, 'referenced', lambda root, ref: tmp_path / ref)
    monkeypatch.setattr(m.mini, 'read_json', lambda path: expected)
    recorded = {'after_snapshot': 'state', 'original_result': {'stdout': 'stdout', 'stderr': 'stderr'},
                'original_observation': {'exit_code': 0}, 'original_delivered': {'exit_code': 0},
                'call': {'call_id': 'c', 'name': 'bash', 'arguments': {'command': 'date'}}}
    out = tmp_path / 'out'; out.mkdir()
    result = m.compare_call(tmp_path, recorded, m.mini.Result({'exit_code': 0}, stdout=b'new'),
                            tmp_path, expected, out, {'recording': {'observation_limit_chars': 100}})
    assert result['classification'] == 'unexplained' and result['unexplained_tool_difference']


def test_actual_colon_batch_id_has_safe_progress_filename(frozen, tmp_path):
    plan, path = frozen
    plan['batches'][0]['batch_id'] = 'officeval_003.r00:cp000000'
    path.unlink(); m.mini.write_json(tmp_path, path.name, plan)
    def runner(batch, output, deadline, *, admit, **kw):
        admit(batch['batch_id'], 0, True)
        return row()
    report = m.execute(path, tmp_path / 'out', m.mini.file_digest(path), runner=runner)
    assert report['status'] == 'complete'
    assert len(list((tmp_path / 'out').glob('progress_*.json'))) == 1
    assert ':' not in next((tmp_path / 'out').glob('progress_*.json')).name


@pytest.fixture
def interrupted(tmp_path, monkeypatch):
    class Validator:
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr(m.mini, 'NativePiValidator', Validator)
    source = tmp_path / 'source'; source.mkdir()
    state = {'files': [], 'directories': [], 'total_bytes': 0, 'content_sha256': 'empty'}
    m.mini.write_json(source, 'before.json', state)
    selected = {'checkpoint': {'qualification': {'image_id': 'image'}},
                'before_snapshot': {'path': 'before.json'}, 'config': {}, 'request_exact': {},
                'calls': [{'call': {'call_id': 'ep:c0'}, 'native_validation_ok': True}]}
    monkeypatch.setattr(m, 'select_batch', lambda batch: selected)
    monkeypatch.setattr(m, 'source_identity', lambda: {'fixed': 'new'})
    monkeypatch.setattr(m.mini, 'native_identity', lambda validator=None: {})
    monkeypatch.setattr(m.mini, 'verified', lambda *args: None)
    monkeypatch.setattr(m.mini, 'referenced', lambda root, ref: root / ref['path'])
    monkeypatch.setattr(m.mini, 'validate_snapshot', lambda *args: None)
    comparison = row()['calls'][0]
    monkeypatch.setattr(m, 'compare_call', lambda *args, **kw: comparison)
    plan = {'schema': m.SCHEMA, 'limits': m.LIMITS.copy(), 'source_identity': {'fixed': 'old'},
            'native_identity': {}, 'package_indices': {str(source): 'index'},
            'batches': [{'batch_id': 'officeval_003.r00:cp000000', 'package': str(source), 'request_id': 'r0'},
                        {'batch_id': 'remaining', 'package': str(source), 'request_id': 'r1'}]}
    plan_path = tmp_path / 'old_plan.json'; m.mini.write_json(tmp_path, plan_path.name, plan)
    prior_dir = tmp_path / 'prior'; prior_dir.mkdir()
    batch_dir = prior_dir / (plan['batches'][0]['batch_id'] + '_replay1'); batch_dir.mkdir()
    replay = row(); replay.update(batch_id=plan['batches'][0]['batch_id'], request_exact={}, restore_exact=True)
    replay['cleanup'] = m.cleanup_receipt([])
    m.mini.write_json(batch_dir, 'batch_result.json', replay)
    m.mini.write_json(batch_dir, 'native_identity.json', {})
    m.mini.write_json(batch_dir, 'runtime_provenance.json', {'image_id': 'image'})
    m.mini.write_json(batch_dir, 'runtime_event_000000.json', {'kind': 'tool.started', 'data': {'call_id': 'ep:c0'}})
    m.mini.write_json(batch_dir, 'restore_capture/states/snapshots/s000000.json', state)
    call = batch_dir / 'call_0000'; call.mkdir()
    m.mini.write_json(call, 'native_validation.json', {'ok': True})
    m.mini.write_json(call, 'result.json', {'observation': {}, 'provenance': {}})
    m.mini.write_bytes(call, 'stdout.bin', b''); m.mini.write_bytes(call, 'stderr.bin', b'')
    m.mini.write_json(call, 'capture/states/snapshots/s000000.json', state)
    m.mini.write_json(call, 'comparison.json', comparison)
    m.mini.write_json(prior_dir, 'admissions/000001.json', {
        'batch_id': plan['batches'][0]['batch_id'], 'ordinal': 0, 'native_valid_execution': True})
    previous = {'status': 'blocked_error', 'error': {'type': 'ValueError',
                'message': "unsafe relative evidence path: 'progress_officeval_003.r00:cp000000.json'"},
                'plan_sha256': m.mini.file_digest(plan_path), 'actor_requests': 0, 'analyst_requests': 0,
                'tool_attempts': 1, 'runtime_executions': 1, 'native_feedback_only': 0,
                'batches': [{'batch_id': plan['batches'][0]['batch_id'], 'replays': [replay]}]}
    result_path = prior_dir / 'result.json'; m.mini.write_json(prior_dir, result_path.name, previous)
    started_path = tmp_path / 'started.json'
    m.mini.write_json(tmp_path, started_path.name, {'plan_sha256': m.mini.file_digest(plan_path),
                                                 'started_at': time.time() - 600})
    return plan_path, result_path, started_path


def test_resume_reuses_exact_prefix_and_carries_budget_and_deadline(interrupted, tmp_path):
    old, result, started = interrupted
    plan = m.build_continuation_plan(old, result, started)
    path = tmp_path / 'continuation.json'; m.mini.write_json(tmp_path, path.name, plan)
    seen = []
    def runner(batch, output, deadline, *, admit, **kw):
        seen.append(batch['batch_id'])
        assert 0 < deadline - time.monotonic() < 21100
        admit(batch['batch_id'], 0, True)
        return row()
    report = m.execute(path, tmp_path / 'new', m.mini.file_digest(path), runner=runner)
    assert seen == ['remaining'] and report['tool_attempts'] == 2
    assert report['carried_tool_attempts'] == 1 and report['mechanism_gate_pass']
    assert len(report['batches']) == 2
    assert len(list((tmp_path / 'new/admissions').glob('*.json'))) == 1


@pytest.mark.parametrize('tamper', ['state', 'admission', 'unknown_error', 'selection', 'deadline'])
def test_continuation_rejects_drift_and_expired_lease(interrupted, tmp_path, tamper):
    old, result, started = interrupted
    plan = m.build_continuation_plan(old, result, started)
    if tamper == 'state':
        state_path = next(result.parent.glob('*_replay1/call_0000/capture/states/snapshots/*.json'))
        state_path.write_text('{}')
    elif tamper == 'admission':
        (result.parent / 'admissions/000001.json').write_text('{}')
    elif tamper == 'unknown_error':
        old_result = m.mini.read_json(result); old_result['error']['message'] = 'different failure'
        result.unlink(); m.mini.write_json(result.parent, result.name, old_result)
        plan['continuation']['prior_result'] = m._ref(result)
        plan['continuation']['prior_files'] = m.prior_inventory(result.parent)
    elif tamper == 'selection':
        plan['batches'][1]['request_id'] = 'different'
    else:
        clock = m.mini.read_json(started); clock['started_at'] = time.time() - 21601
        started.unlink(); m.mini.write_json(tmp_path, started.name, clock)
        plan['continuation']['started_receipt'] = m._ref(started)
    path = tmp_path / 'continuation.json'; m.mini.write_json(tmp_path, path.name, plan)
    with pytest.raises((ValueError, TimeoutError)):
        m.execute(path, tmp_path / 'new', m.mini.file_digest(path))
    assert not (tmp_path / 'new').exists()


@pytest.mark.parametrize('command,declared', [
    ('ls -la /workspace/input/', True),
    ('ls -la /workspace/submission/', True),
    ('ls -la /workspace/input/; echo injected', False),
    ('ls -la /workspace/other/', False),
    ('ls -la /workspace/input/ --color', False),
])
def test_literal_input_ls_metadata_only_is_declared(tmp_path, monkeypatch, command, declared):
    expected = {'files': [], 'directories': [], 'total_bytes': 0, 'content_sha256': 'same'}
    left = b'total 4\n-rw-r--r-- 1 root root 100 Oct 18 12:30 file.docx\n'
    right = b'total 4\n-rw-r--r-- 1 1000 1000 100 Oct 19 13:40 file.docx\n'
    (tmp_path / 'stdout').write_bytes(left); (tmp_path / 'stderr').write_bytes(b'')
    monkeypatch.setattr(m.mini, 'referenced', lambda root, ref: tmp_path / ref)
    monkeypatch.setattr(m.mini, 'read_json', lambda path: expected)
    observed = m.mini.format_pi_bash_observation(left, b'', 0, None, None, None)
    actual_obs = m.mini.format_pi_bash_observation(right, b'', 0, None, None, None)
    recorded = {'after_snapshot': 'state', 'original_result': {'stdout': 'stdout', 'stderr': 'stderr'},
                'original_observation': observed, 'original_delivered': observed,
                'call': {'call_id': 'c', 'name': 'bash', 'arguments': {'command': command}}}
    out = tmp_path / 'out'; out.mkdir()
    result = m.compare_call(tmp_path, recorded, m.mini.Result(actual_obs, stdout=right),
                            tmp_path, expected, out, {'recording': {'observation_limit_chars': 10000}})
    assert result['ls_metadata_variance'] is declared
    assert result['classification'] == ('declared_runtime_variance' if declared else 'unexplained')


@pytest.fixture
def bridge_interrupted(interrupted, tmp_path, monkeypatch):
    import copy
    import shutil
    old_path, result_path, started_path = interrupted
    old = m.mini.read_json(old_path)
    old['batches'].append({'batch_id': 'last', 'package': old['batches'][0]['package'], 'request_id': 'r2'})
    old_path.unlink(); m.mini.write_json(tmp_path, old_path.name, old)
    old_result = m.mini.read_json(result_path); old_result['plan_sha256'] = m.mini.file_digest(old_path)
    result_path.unlink(); m.mini.write_json(result_path.parent, result_path.name, old_result)
    clock = m.mini.read_json(started_path); clock['plan_sha256'] = m.mini.file_digest(old_path)
    started_path.unlink(); m.mini.write_json(tmp_path, started_path.name, clock)
    first = m.build_continuation_plan(old_path, result_path, started_path)
    first_path = tmp_path / 'first_continuation_plan.json'; m.mini.write_json(tmp_path, first_path.name, first)
    current = tmp_path / 'current'; current.mkdir()
    original_dir = next(result_path.parent.glob('*_replay1'))
    extra_replays = []
    for i in range(2):
        directory = current / f'remaining_replay{i + 1}'
        shutil.copytree(original_dir, directory)
        replay = m.mini.read_json(directory / 'batch_result.json'); replay['batch_id'] = 'remaining'
        replay['calls'][0] = row(exact=False)['calls'][0]
        (directory / 'batch_result.json').unlink(); m.mini.write_json(directory, 'batch_result.json', replay)
        (directory / 'call_0000/comparison.json').unlink()
        m.mini.write_json(directory / 'call_0000', 'comparison.json', replay['calls'][0])
        (directory / 'call_0000/stdout.bin').write_bytes(b'variance')
        extra_replays.append(replay)
        m.mini.write_json(current, f'admissions/{i + 2:06d}.json', {
            'batch_id': 'remaining', 'ordinal': 0, 'native_valid_execution': True})
    def comparison(root, recorded, actual, *args, **kw):
        return row(exact=actual.stdout != b'variance')['calls'][0]
    monkeypatch.setattr(m, 'compare_call', comparison)
    previous = copy.deepcopy(old_result)
    previous.update(plan_sha256=m.mini.file_digest(first_path), tool_attempts=3, runtime_executions=3,
                    error={'type': 'NativePiInfrastructureError',
                           'message': 'Native Pi bridge closed or exceeded response limit'})
    previous['batches'].append({'batch_id': 'remaining', 'replays': extra_replays})
    m.mini.write_json(current, 'last_replay1/batch_result.json', {
        'batch_id': 'last', 'status': 'incomplete', 'calls': [], 'cleanup': m.cleanup_receipt([])})
    current_result = current / 'result.json'; m.mini.write_json(current, current_result.name, previous)
    return first_path, current_result, started_path


def test_second_continuation_retains_nonexact_and_two_replays_without_duplication(bridge_interrupted, tmp_path):
    prior, previous, clock = bridge_interrupted
    second = m.build_continuation_plan(prior, previous, clock)
    path = tmp_path / 'second.json'; m.mini.write_json(tmp_path, path.name, second)
    seen = []
    def runner(batch, output, deadline, *, admit, **kw):
        seen.append(batch['batch_id']); admit(batch['batch_id'], 0, True)
        return row()
    report = m.execute(path, tmp_path / 'resumed', m.mini.file_digest(path), runner=runner)
    assert seen == ['last'] and report['tool_attempts'] == 4 and report['carried_tool_attempts'] == 3
    assert len(report['batches'][1]['replays']) == 2
    assert report['batches'][1]['replays'][0]['calls'][0]['classification'] == 'unexplained'
    assert not report['mechanism_gate_pass'] and report['status'] == 'blocked_unexplained_difference'


def test_third_shared_session_repair_allowed_but_fourth_rejected(bridge_interrupted, tmp_path):
    import copy
    import shutil
    prior, previous, clock = bridge_interrupted
    second = m.build_continuation_plan(prior, previous, clock)
    second_path = tmp_path / 'second.json'; m.mini.write_json(tmp_path, second_path.name, second)
    third_root = tmp_path / 'third_failed'; third_root.mkdir()
    shutil.copytree(previous.parent / 'last_replay1', third_root / 'last_replay1')
    previous_data = copy.deepcopy(m.mini.read_json(previous))
    previous_data['plan_sha256'] = m.mini.file_digest(second_path)
    m.mini.write_json(third_root, 'result.json', previous_data)
    third = m.build_continuation_plan(second_path, third_root / 'result.json', clock)
    assert third['replay_operation'] == {'validator_lifetime': 'shared-execution-v1'}
    assert m.validate_continuation(third)['result']['tool_attempts'] == 3
    third_path = tmp_path / 'third.json'; m.mini.write_json(tmp_path, third_path.name, third)
    with pytest.raises(ValueError, match='at most three'):
        m.build_continuation_plan(third_path, third_root / 'result.json', clock)


def test_native_start_resume_rejects_any_partial_call_artifact(bridge_interrupted):
    prior, previous, clock = bridge_interrupted
    (previous.parent / 'last_replay1/call_0000').mkdir()
    with pytest.raises(ValueError, match='contains tool work'):
        m.build_continuation_plan(prior, previous, clock)


def test_reclassification_is_explicit_sidecar_and_never_overwrites_old_raw(bridge_interrupted, tmp_path, monkeypatch):
    import copy
    prior, previous, clock = bridge_interrupted
    aggregate = m.mini.read_json(previous)
    for index in range(2):
        replay = aggregate['batches'][1]['replays'][index]
        old = replay['calls'][0]
        old['unexplained_state_difference'] = False
        old['ls_metadata_variance'] = False
        directory = previous.parent / f'remaining_replay{index + 1}'
        for path, value in [(directory / 'batch_result.json', replay),
                            (directory / 'call_0000/comparison.json', old)]:
            path.unlink(); m.mini.write_json(path.parent, path.name, value)
    previous.unlink(); m.mini.write_json(previous.parent, previous.name, aggregate)
    original_select = m.select_batch
    def selected(batch):
        result = copy.deepcopy(original_select(batch))
        result['calls'][0]['call'].update(name='bash', arguments={'command': 'ls -la /workspace/input/'})
        return result
    monkeypatch.setattr(m, 'select_batch', selected)
    def recompute(root, recorded, actual, *args, **kw):
        if actual.stdout != b'variance':
            return row()['calls'][0]
        return {**row(exact=False)['calls'][0], 'classification': 'declared_runtime_variance',
                'unexplained_state_difference': False, 'unexplained_tool_difference': False,
                'ls_metadata_variance': True}
    monkeypatch.setattr(m, 'compare_call', recompute)
    original_raw = (previous.parent / 'remaining_replay1/call_0000/comparison.json').read_bytes()
    second = m.build_continuation_plan(prior, previous, clock)
    path = tmp_path / 'second.json'; m.mini.write_json(tmp_path, path.name, second)
    def runner(batch, output, deadline, *, admit, **kw):
        admit(batch['batch_id'], 0, True)
        return row()
    report = m.execute(path, tmp_path / 'resumed', m.mini.file_digest(path), runner=runner)
    assert report['mechanism_gate_pass'] and len(report['reclassifications']) == 2
    assert report['batches'][1]['replays'][0]['calls'][0]['classification'] == 'unexplained'
    sidecar = m.mini.read_json(tmp_path / 'resumed/reclassification_sidecar.json')
    assert sidecar['original_receipts_unchanged'] and sidecar['changes'][0]['after']['ls_metadata_variance']
    assert (previous.parent / 'remaining_replay1/call_0000/comparison.json').read_bytes() == original_raw


def test_saved_docx_zip_timestamp_receipt_json_roundtrip_is_not_new_variance(interrupted, tmp_path, monkeypatch):
    import copy
    old, result, started = interrupted
    previous = m.mini.read_json(result)
    fresh = {**row()['calls'][0], 'classification': 'declared_runtime_variance',
             'zip_timestamp_variance': [{'member_timestamp_changes': [{
                 'member': 'word/document.xml', 'expected': (2026, 10, 18, 16, 18, 58),
                 'actual': (2026, 10, 19, 3, 12, 12)}]}]}
    saved = __import__('json').loads(m.mini.canonical(fresh))
    previous['batches'][0]['replays'][0]['calls'][0] = saved
    directory = next(result.parent.glob('*_replay1'))
    for path, value in [(directory / 'batch_result.json', previous['batches'][0]['replays'][0]),
                        (directory / 'call_0000/comparison.json', saved), (result, previous)]:
        path.unlink(); m.mini.write_json(path.parent, path.name, value)
    monkeypatch.setattr(m, 'compare_call', lambda *args, **kw: copy.deepcopy(fresh))
    plan = m.build_continuation_plan(old, result, started)
    verified = m.validate_continuation(plan)
    assert verified['result']['tool_attempts'] == 1
    assert verified['reclassifications'] == []
    assert verified['effective_comparisons'][0] == saved
    assert fresh != saved  # The original Python representation really differs.


def test_execution_uses_one_native_open_close_for_ordered_independent_batches(frozen, tmp_path):
    plan, path = frozen
    plan['batches'].append({'batch_id': 'two', 'package': '/fake', 'request_id': 'r1'})
    path.unlink(); m.mini.write_json(tmp_path, path.name, plan)
    lifecycle = []; results = []
    class Validator:
        closed = False
        def __enter__(self): lifecycle.append('open'); return self
        def __exit__(self, *args): self.closed = True; lifecycle.append('close')
        def validate(self, name, args, call_id, **kwargs):
            assert not self.closed
            return {'ok': args.get('valid') is True, 'call_id': call_id, 'arguments': args.copy()}
    def runner(batch, output, deadline, *, admit, shared_validator, **kw):
        assert lifecycle == ['open']
        with pytest.raises(AssertionError):
            m.mini.OpenAICompatibleBackend.send(None, b'forbidden', 1)
        valid = batch['batch_id'] == 'two'
        result = shared_validator.validate('edit', {'valid': valid}, batch['batch_id'])
        results.append(result); admit(batch['batch_id'], 0, valid)
        return row()
    report = m.execute(path, tmp_path / 'out', m.mini.file_digest(path), runner=runner,
                       validator_factory=Validator)
    assert lifecycle == ['open', 'close']
    assert [r['call_id'] for r in results] == ['one', 'two']
    assert [r['ok'] for r in results] == [False, True]
    assert report['validator_lifetime'] == 'shared-execution-v1'
    assert report['tool_attempts'] == 2 and report['native_feedback_only'] == 1


def test_shared_session_closed_once_on_execution_failure(frozen, tmp_path):
    _, path = frozen; events = []
    class Validator:
        def __enter__(self): events.append('open'); return self
        def __exit__(self, *args): events.append('close')
    def runner(*args, **kwargs):
        raise RuntimeError('fatal session failure, no reconnect')
    report = m.execute(path, tmp_path / 'out', m.mini.file_digest(path), runner=runner,
                       validator_factory=Validator)
    assert events == ['open', 'close'] and report['status'] == 'blocked_error'


def test_dry_run_rejects_changed_retained_prefix(interrupted, tmp_path, monkeypatch):
    old, result, started = interrupted
    plan = m.build_continuation_plan(old, result, started)
    path = tmp_path / 'continuation.json'
    m.mini.write_json(tmp_path, path.name, plan)
    saved = next(result.parent.glob('*_replay1/call_0000/stdout.bin'))
    saved.write_bytes(b'changed after plan freeze')
    monkeypatch.setattr(sys, 'argv', ['replay', '--plan', str(path),
                                    '--plan-sha256', m.mini.file_digest(path)])
    with pytest.raises(ValueError, match='prior raw evidence inventory changed'):
        m.main()
