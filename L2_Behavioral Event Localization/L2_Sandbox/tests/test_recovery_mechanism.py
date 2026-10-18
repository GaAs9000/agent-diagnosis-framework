"""P0 replay contracts; fake runtime only, no Docker/model and no credentials."""
import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import time
import zipfile

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/check_recovery_mechanism.py'
spec = importlib.util.spec_from_file_location('recovery_mechanism_test_target', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    plan = {'schema': 'recovery_mechanism_mini_v1', 'limits': m.LIMITS.copy(),
            'source_identity': {}, 'native_identity': {'installed': 'pinned'}, 'package_indices': {},
            'cases': [{'case': s[0], 'package': str(tmp_path / 'source')} for s in m.SPECS]}
    monkeypatch.setattr(m, 'source_identity', lambda: {})
    monkeypatch.setattr(m, 'native_identity', lambda validator=None: {'installed': 'pinned'})
    monkeypatch.setattr(m, 'select_case', lambda root, label, *args:
                        {'case': label, 'package': str(root)})
    # Each real case must be package-bound, even in the fake transport tests.
    plan['package_indices'] = {str(tmp_path / 'source'): 'sealed-index'}
    monkeypatch.setattr(m, 'verified', lambda *args: None)
    path = tmp_path / 'plan.json'
    m.write_json(tmp_path, path.name, plan)
    return path, plan


def good_row(case):
    return {'case': case['case'], 'restore_exact': True, 'request_exact': True,
            'deterministic_tool_exact': True, 'declared_runtime_variance': False,
            'unexplained_state_difference': False, 'unexplained_tool_difference': False}


def save_plan(path, plan):
    path.unlink()
    m.write_json(path.parent, path.name, plan)


def test_exact_five_order_single_attempt_actor_guard(frozen, tmp_path):
    path, _ = frozen
    seen = []
    def replay(case, output, deadline, **kwargs):
        seen.append(case['case'])
        assert 0 < deadline - time.monotonic() <= 1800
        with pytest.raises(AssertionError, match='Actor send is forbidden'):
            m.OpenAICompatibleBackend.send(None, b'anything', 1)
        with pytest.raises(AssertionError, match='Actor send is forbidden'):
            m.GeminiBackend.send(None, b'anything', 1)
        return good_row(case)
    report = m.execute(path, tmp_path / 'out', m.file_digest(path), runner=replay)
    assert seen == [s[0] for s in m.SPECS]
    assert report['actor_requests'] == 0 and report['tool_attempts'] == 5
    assert report['status'] == 'mini_exact_with_limited_coverage'
    assert report['live_gate_pass'] is False
    with pytest.raises(ValueError, match='new output directory'):
        m.execute(path, tmp_path / 'out', m.file_digest(path), runner=replay)
    assert len(seen) == 5


@pytest.mark.parametrize('mutation', ['limits', 'six_calls', 'wrong_case', 'source', 'case_tamper', 'unbound_package'])
def test_invalid_plan_never_writes_or_calls(frozen, tmp_path, mutation):
    path, plan = frozen
    if mutation == 'limits': plan['limits']['tool_calls'] = 6
    if mutation == 'six_calls': plan['cases'].append(plan['cases'][0])
    if mutation == 'wrong_case': plan['cases'][0]['case'] = 'another'
    if mutation == 'source': plan['source_identity'] = {'unknown': 'hash'}
    if mutation == 'case_tamper': plan['cases'][0]['injected'] = True
    if mutation == 'unbound_package': plan['cases'][0]['package'] = str(tmp_path / 'not_bound')
    save_plan(path, plan)
    def forbidden(*args): pytest.fail('replay must not run')
    with pytest.raises(ValueError):
        m.execute(path, tmp_path / 'out', m.file_digest(path), runner=forbidden)
    assert not (tmp_path / 'out').exists()


def test_plan_sha_tamper_no_writes(frozen, tmp_path):
    path, _ = frozen
    sha = m.file_digest(path)
    with path.open('ab') as stream: stream.write(b' ')
    with pytest.raises(ValueError, match='SHA mismatch'):
        m.execute(path, tmp_path / 'out', sha)
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('kind', ['unexplained_state_difference', 'unexplained_tool_difference'])
def test_unexplained_difference_stops_without_retries(frozen, tmp_path, kind):
    path, _ = frozen
    def replay(case, *args, **kwargs):
        row = good_row(case); row[kind] = True
        return row
    report = m.execute(path, tmp_path / 'out', m.file_digest(path), runner=replay)
    assert report['tool_attempts'] == 1
    assert report['status'] == 'blocked_unexplained_difference'


def test_runtime_variance_remains_review_not_pass(frozen, tmp_path):
    path, _ = frozen
    def replay(case, *args, **kwargs):
        row = good_row(case)
        row.update(declared_runtime_variance=True, deterministic_tool_exact=False)
        return row
    report = m.execute(path, tmp_path / 'out', m.file_digest(path), runner=replay)
    assert report['status'] == 'review_runtime_variance'
    assert report['live_gate_pass'] is False
    assert all(not row['deterministic_tool_exact'] for row in report['cases'])


def test_runtime_failure_preserves_attempt_and_blocks_retry(frozen, tmp_path):
    path, _ = frozen
    def fail(*args, **kwargs): raise TimeoutError('runtime 60 seconds')
    report = m.execute(path, tmp_path / 'out', m.file_digest(path), runner=fail)
    assert report['tool_attempts'] == 1 and report['status'] == 'blocked_error'
    assert report['error']['type'] == 'TimeoutError'
    assert (tmp_path / 'out/started.json').exists()
    assert (tmp_path / 'out/result.json').exists()


def test_expired_wall_limit_makes_no_tool_attempt(frozen, tmp_path, monkeypatch):
    path, _ = frozen
    ticks = iter([0, 1801])
    monkeypatch.setattr(m.time, 'monotonic', lambda: next(ticks))
    report = m.execute(path, tmp_path / 'out', m.file_digest(path))
    assert report['tool_attempts'] == 0 and report['status'] == 'blocked_error'


def test_output_must_not_overlap_evidence_or_symlink(frozen, tmp_path):
    path, plan = frozen
    source = Path(next(iter(plan['package_indices'])))
    source.mkdir()
    with pytest.raises(ValueError, match='immutable evidence'):
        m.execute(path, source / 'new', m.file_digest(path))
    link = tmp_path / 'link'
    link.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        m.execute(path, link / 'new', m.file_digest(path))
    assert not (source / 'new').exists()


@pytest.fixture
def tool_case(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'native_identity', lambda validator=None: {'installed': 'pinned'})
    source = tmp_path / 'source'; source.mkdir()
    workspace = tmp_path / 'original'; workspace.mkdir()
    (workspace / 'file.txt').write_text('old')
    store = m.StateStore(source)
    before = store.capture(workspace, 'before')
    after = store.capture(workspace, 'after')
    config = {'runtime': {'max_workspace_bytes': 100000, 'max_files': 100},
              'model': {'supports_images': True}}
    m.write_json(source, 'raw/resolved_config.json', config)
    m.write_json(source, 'cp.json', {'qualification': {'image_id': 'sha256:pinned'}})
    m.write_json(source, 'original.json', {'observation': {'output': 'old', 'exit_code': 0}})
    m.write_bytes(source, 'stdout.bin', b''); m.write_bytes(source, 'stderr.bin', b'')
    def ref(path): return {'path': path, 'sha256': m.file_digest(source / path)}
    case = {'case': 'dev003_docx_read', 'package': str(source), 'checkpoint_ref': ref('cp.json'),
            'before_snapshot': ref('states/snapshots/s000000.json'),
            'after_snapshot': ref('states/snapshots/s000001.json'),
            'call': {'name': 'read', 'arguments': {'path': 'file.txt'}, 'call_id': 'ep:r000000:c0', 'native_id': 'native0'},
            'original_result': {'result': ref('original.json'), 'stdout': ref('stdout.bin'), 'stderr': ref('stderr.bin')}}
    class Runtime:
        calls = []
        def __init__(self, config): pass
        def set_event_sink(self, sink): pass
        def preflight(self): return {'image_id': 'sha256:pinned'}
        def execute(self, call, workspace, timeout, **kwargs):
            assert isinstance(call, m.NativeValidatedCall)
            assert timeout <= 60
            self.calls.append(call)
            return m.Result({'output': (workspace / 'file.txt').read_text(), 'exit_code': 0})
    class Validator:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def validate(self, name, arguments, cid, **kwargs):
            return {'ok': True, 'validated_arguments': arguments}
    out = tmp_path / 'output'; out.mkdir()
    return case, out, Runtime, Validator


def test_restore_bytes_modes_raw_streams_and_native_validation(tool_case):
    case, output, runtime, validator = tool_case
    result = m.run_case(case, output, time.monotonic() + 1800,
                        expected_native_identity={'installed': 'pinned'}, runtime_factory=runtime, validator_factory=validator)
    assert result['deterministic_tool_exact'] is True
    assert result['restore_exact'] and result['after_state_exact']
    assert len(runtime.calls) == 1
    assert (output / case['case'] / 'stdout.bin').read_bytes() == b''


def test_native_validation_error_never_executes_runtime(tool_case):
    case, output, runtime, validator = tool_case
    class Invalid(validator):
        def validate(self, *args, **kwargs):
            return {'ok': False, 'native_result': {'output': 'Validation failed', 'exit_code': 1}}
    row = m.run_case(case, output, time.monotonic() + 1800,
                     expected_native_identity={'installed': 'pinned'}, runtime_factory=runtime, validator_factory=Invalid)
    assert runtime.calls == []
    assert row['unexplained_tool_difference'] is True
    saved = m.read_json(output / case['case'] / 'result.json')
    assert saved['observation']['output'] == 'Validation failed'


def test_runtime_image_mismatch_blocks_tool(tool_case):
    case, output, runtime, validator = tool_case
    class WrongImage(runtime):
        def preflight(self): return {'image_id': 'wrong'}
    with pytest.raises(ValueError, match='runtime image'):
        m.run_case(case, output, time.monotonic() + 1800,
                   expected_native_identity={'installed': 'pinned'}, runtime_factory=WrongImage, validator_factory=validator)
    assert runtime.calls == []


def test_state_difference_not_hidden_by_matching_tool_output(tool_case):
    case, output, runtime, validator = tool_case
    class Mutates(runtime):
        def execute(self, call, workspace, timeout, **kwargs):
            (workspace / 'unexpected.txt').write_text('unexplained')
            return m.Result({'output': 'old', 'exit_code': 0})
    row = m.run_case(case, output, time.monotonic() + 1800,
                     expected_native_identity={'installed': 'pinned'}, runtime_factory=Mutates, validator_factory=validator)
    assert row['observation_exact'] is True
    assert row['unexplained_state_difference'] is True
    assert row['deterministic_tool_exact'] is False


def test_zip_timestamp_only_diagnostic_retains_different_raw_hashes(tmp_path):
    def zipped(timestamp, content):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as archive:
            archive.writestr(zipfile.ZipInfo('document.xml', timestamp), content)
        return stream.getvalue()
    original = zipped((2026, 10, 18, 12, 0, 0), b'document')
    replay = zipped((2026, 10, 18, 13, 0, 0), b'document')
    roots = []
    states = []
    for i, data in enumerate((original, replay)):
        root = tmp_path / f'capture{i}'; root.mkdir()
        ws = tmp_path / f'workspace{i}'; ws.mkdir()
        (ws / 'answer.docx').write_bytes(data)
        roots.append(root); states.append(m.StateStore(root).capture(ws, 'zip'))
    detail = m.zip_time_only(roots[0], states[0], roots[1], states[1])
    assert detail and detail[0]['expected_sha256'] != detail[0]['actual_sha256']
    assert not m.snapshot_equal(*states)
    # A different XML payload of equal byte length must remain unexplained.
    data = zipped((2026, 10, 18, 13, 0, 0), b'changed!')
    (tmp_path / 'workspace1/answer.docx').write_bytes(data)
    other_root = tmp_path / 'capture_other'; other_root.mkdir()
    other = m.StateStore(other_root).capture(tmp_path / 'workspace1', 'changed')
    assert m.zip_time_only(roots[0], states[0], other_root, other) == []


def test_ls_variance_only_metadata_never_names_modes_sizes_or_noise():
    original = b'total 8\n-rw-r--r-- 1 alice staff 42 Oct 18 12:00 report.docx\n'
    replay = b'total 8\n-rw-r--r-- 1 root root 42 Oct 18 23:00 report.docx\n'
    assert m.ls_metadata_only(original, replay)
    for bad in (replay.replace(b'report.docx', b'other.docx'),
                replay.replace(b'42', b'43'), replay.replace(b'-rw-r--r--', b'-rwxr--r--'),
                replay.replace(b'total 8', b'total 9'), replay + b'UNEXPECTED OUTPUT\n',
                b'arbitrary text\n', replay.replace(b'1 root', b'2 root')):
        assert not m.ls_metadata_only(original, bad)
    assert not m.ls_metadata_only(original, original)


def test_native_write_edit_tempfiles_and_invalid_arguments_are_not_executed(tmp_path):
    """Additional CPU tool coverage; not one of the five historical observations."""
    import os
    if not os.environ.get('C3_PI_NODE') or not os.environ.get('C3_PI_NATIVE_DIR'):
        pytest.skip('pinned native Node bridge must be explicitly supplied on CPU host')
    from c3_sandbox.l1.runtime.tools import execute_pi_write, execute_pi_edit
    with m.NativePiValidator() as validator:
        write = validator.validate('write', {'path': '/workspace/result.txt', 'content': 'alpha\nbeta\n'}, 'write-test')
        assert write['ok']
        call = m.NativeValidatedCall('write-test', 'write', write['validated_arguments'], 'native-write')
        assert execute_pi_write(call, tmp_path).observation['exit_code'] == 0
        assert (tmp_path / 'result.txt').read_text() == 'alpha\nbeta\n'
        edit = validator.validate('edit', {'path': '/workspace/result.txt',
                                  'edits': [{'oldText': 'beta', 'newText': 'gamma'}]}, 'edit-test')
        assert edit['ok']
        call = m.NativeValidatedCall('edit-test', 'edit', edit['validated_arguments'], 'native-edit')
        assert execute_pi_edit(call, tmp_path).observation['exit_code'] == 0
        assert (tmp_path / 'result.txt').read_text() == 'alpha\ngamma\n'
        bad = validator.validate('edit', {'edits': [{'path': '/workspace/result.txt',
                                 'oldText': 'gamma', 'newText': 'wrong'}]}, 'invalid-edit')
        assert not bad['ok'] and bad['native_result']['isError'] is True
        assert 'Validation failed' in bad['native_result']['content'][0]['text']
        # Only approved capabilities above execute; the rejected shape is pure feedback.
        assert (tmp_path / 'result.txt').read_text() == 'alpha\ngamma\n'


def test_changed_installed_native_identity_blocks_before_any_tool(frozen, tmp_path, monkeypatch):
    path, _ = frozen
    monkeypatch.setattr(m, 'native_identity', lambda validator=None: {'installed': 'changed'})
    def forbidden(*args, **kwargs): pytest.fail('tool must not run')
    with pytest.raises(ValueError, match='native validator differs'):
        m.execute(path, tmp_path / 'out', m.file_digest(path), runner=forbidden)
    assert not (tmp_path / 'out').exists()


def test_per_call_native_identity_drift_blocks_before_docker(tool_case):
    case, output, runtime, validator = tool_case
    class NoDocker(runtime):
        def preflight(self): pytest.fail('Docker must not be invoked')
    with pytest.raises(ValueError, match='native validator differs before'):
        m.run_case(case, output, time.monotonic() + 1800,
                   expected_native_identity={'installed': 'different'},
                   runtime_factory=NoDocker, validator_factory=validator)
    assert runtime.calls == []
