"""Pure CPU regression cases for census coverage and fixed recovery selection."""
import copy
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/audit_prospective_recovery.py'
spec = importlib.util.spec_from_file_location('prospective_recovery_target', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def event(kind, seq, data=None, **extra):
    return {'kind': kind, 'seq': seq, 'event_id': f'e{seq}', 'sha256': str(seq) * 64,
            'data': data or {}, **extra}


def row(number, seq, *, request=True, eligible=True, compactions=()):
    return {'checkpoint': {'checkpoint_id': f'cp{number:06d}', 'pending_calls': [],
                           'qualification': {'eligible': eligible, 'reason_codes': []},
                           'conversation': {'_pi_compaction': list(compactions)}},
            'checkpoint_event': event('checkpoint.created', seq, {'checkpoint': {'path': f'cp{number}', 'sha256': 'a'}}),
            'next_request': event('model.request', seq + 1, {'request_id': f'r{number}'}) if request else None,
            'snapshot_ref': {'path': f's{number}', 'sha256': 'a'}}


def test_compaction_binds_first_subsequent_checkpoint_not_every_later_one():
    data = {'summary': 'actual'}
    rows = [row(0, 0), row(1, 5, compactions=[data]), row(2, 9, compactions=[data])]
    actual = m.compression_bindings([event('session.compacted', 4, data)], rows)
    assert len(actual) == 1
    assert actual[0]['checkpoint_id'] == 'cp000001'
    assert actual[0]['request_id'] == 'r1'


def test_multiple_compactions_each_get_first_checkpoint():
    a, b = {'summary': 'a'}, {'summary': 'b'}
    actual = m.compression_bindings([event('session.compacted', 3, a), event('session.compacted', 8, b)],
                                    [row(0, 0), row(1, 4, compactions=[a]), row(2, 9, compactions=[a, b])])
    assert [v['checkpoint_id'] for v in actual] == ['cp000001', 'cp000002']


@pytest.mark.parametrize('rows', [[], [row(1, 5)]])
def test_missing_compression_checkpoint_or_session_state_fails(rows):
    with pytest.raises(ValueError):
        m.compression_bindings([event('session.compacted', 4, {'summary': 'a'})], rows)


def test_terminal_post_compaction_has_no_request_pass():
    data = {'summary': 'a'}
    result = m.compression_bindings([event('session.compacted', 4, data)],
                                    [row(1, 5, compactions=[data], request=False)])
    assert result[0]['request_status'] == 'not_applicable_terminal'
    assert result[0]['request_id'] is None


def test_selection_deduplicates_and_preserves_missing_cp0(monkeypatch):
    rows = [row(0, 0), row(1, 5), row(2, 9), row(3, 12)]
    monkeypatch.setattr(m, 'complete_batch', lambda root, es, r:
                        (None, 'following_response_has_no_tool_calls') if r is rows[0] else ([{'id': 1}], None))
    selected, reasons = m.select_batches(Path('.'), [], rows, [], True)
    assert [v['row']['checkpoint']['checkpoint_id'] for v in selected] == ['cp000002', 'cp000003']
    assert any(v.get('role') == 'cp0' for v in reasons)


def test_compaction_selection_adds_every_first_checkpoint_outside_fixed20(monkeypatch):
    rows = [row(i, i * 3) for i in range(4)]
    monkeypatch.setattr(m, 'complete_batch', lambda *args: ([{'id': 1}], None))
    selected, _ = m.select_batches(Path('.'), [], rows,
                                  [{'checkpoint_id': 'cp000001', 'event_id': 'e1'},
                                   {'checkpoint_id': 'cp000003', 'event_id': 'e9'}], False)
    assert [v['row']['checkpoint']['checkpoint_id'] for v in selected] == ['cp000001', 'cp000003']


def test_raw_request_bytes_cannot_be_normalized(monkeypatch, tmp_path):
    import check_recovery_mechanism as mini
    class Fake:
        def request(self, session, tools):
            return {'a': 1, 'b': 2}
    monkeypatch.setattr(mini, 'OpenAICompatibleBackend', Fake)
    monkeypatch.setattr(mini, 'tools_for_config', lambda _: [])
    path = tmp_path / 'request.json'
    path.write_bytes(b'{ "b": 2, "a": 1 }')
    request = event('model.request', 1, {'body': {'path': path.name, 'sha256': m.file_digest(path)}})
    with pytest.raises(ValueError, match='raw bytes differ'):
        mini.reconstruct(tmp_path, {'conversation': {}}, request, {'model': {'adapter': 'openai-compatible'}})
    path.write_bytes(m.canonical({'a': 1, 'b': 2}))
    request['data']['body']['sha256'] = m.file_digest(path)
    assert mini.reconstruct(tmp_path, {'conversation': {}}, request,
                            {'model': {'adapter': 'openai-compatible'}})['size_bytes'] == len(path.read_bytes())


def batch_fixture(monkeypatch, invalid=False):
    before = {'files': [], 'directories': [], 'total_bytes': 0, 'content_sha256': 'zero'}
    data = {'call_id': 'c1', 'request_id': 'r0', 'arguments': {'wrong': True}, 'name': 'edit'}
    calls = [event('tool.call', 2, data)]
    if not invalid:
        calls.append(event('tool.started', 3, {'call_id': 'c1', 'before_snapshot_id': 's0'}))
    result = event('tool.result', 4, {'call_id': 'c1', 'result': {'path': 'result', 'sha256': 'a'}}, links={'call': 'e2'})
    states = [event('state.snapshot', 0, {'snapshot': {'path': 's0', 'sha256': 'a'}}),
              event('state.snapshot', 5, {'snapshot': {'path': 's1', 'sha256': 'a'}}, links={'tool_result': 'e4'})]
    payloads = {'s0': dict(before, snapshot_id='s0'), 's1': dict(before, snapshot_id='s1'),
                'result': {'observation': {'error': 'native validation failed'},
                           'provenance': {'pi_native_validation_error': invalid}}}
    monkeypatch.setattr(m, 'referenced', lambda root, ref: ref['path'])
    monkeypatch.setattr(m, 'read_json', lambda path: payloads[path])
    monkeypatch.setattr(m, 'validate_snapshot', lambda *args: None)
    return states + calls + [result], payloads


def test_native_invalid_without_tool_started_is_retained(monkeypatch):
    es, _ = batch_fixture(monkeypatch, invalid=True)
    cp = row(0, 0); cp['snapshot_ref'] = {'path': 's0', 'sha256': 'a'}
    calls, reason = m.complete_batch(Path('.'), es, cp)
    assert reason is None and calls[0]['native_invalid_without_start'] is True
    assert calls[0]['original_observation']['error'] == 'native validation failed'


def test_missing_tool_after_snapshot_blocks(monkeypatch):
    es, _ = batch_fixture(monkeypatch)
    es = [e for e in es if not e.get('links', {}).get('tool_result')]
    cp = row(0, 0); cp['snapshot_ref'] = {'path': 's0', 'sha256': 'a'}
    with pytest.raises(ValueError, match='post-state'):
        m.complete_batch(Path('.'), es, cp)


def test_native_invalid_must_not_change_workspace(monkeypatch):
    es, payloads = batch_fixture(monkeypatch, invalid=True)
    payloads['s1']['content_sha256'] = 'changed'
    cp = row(0, 0); cp['snapshot_ref'] = {'path': 's0', 'sha256': 'a'}
    with pytest.raises(ValueError, match='changed workspace'):
        m.complete_batch(Path('.'), es, cp)


def test_fixed20_cover_all_families_and_no_holdout():
    assert len(set(m.REPLAY_SLOTS)) == 20
    assert {v.split('.')[0] for v in m.REPLAY_SLOTS} == set(m.DEV)
    assert len(m.SLOTS) == len(set(m.SLOTS)) == 45


def test_original_collection_archive_independent_of_new_analysis(monkeypatch, tmp_path):
    import io
    import tarfile
    l1 = 'L1_sandbox/C3_Experimental_Sandbox/src/controller.py'
    analysis = 'L2_Behavioral Event Localization/L2_Sandbox/scripts/analysis.py'
    original = {l1: b'original controller', analysis: b'old analysis'}
    archive = tmp_path / 'controller_source.tar'
    with tarfile.open(archive, 'w') as stream:
        for name, body in original.items():
            item = tarfile.TarInfo(name); item.size = len(body)
            stream.addfile(item, io.BytesIO(body))
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    for name, body in original.items():
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(body)
    frozen = {'bound_files': {str(archive): m.file_digest(archive)},
              'source_files': {name: m.hashlib.sha256(body).hexdigest() for name, body in original.items()}}
    (tmp_path / analysis).write_bytes(b'new reporting analysis does not rewrite old collection')
    m.verify_controller_archive(frozen)
    (tmp_path / l1).write_bytes(b'changed current mechanics')
    with pytest.raises(ValueError, match='actual imported L1 mechanics differ'):
        m.verify_controller_archive(frozen)


def test_archive_missing_original_source_does_not_pass(monkeypatch, tmp_path):
    import tarfile
    archive = tmp_path / 'controller_source.tar'
    with tarfile.open(archive, 'w'):
        pass
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    with pytest.raises(KeyError):
        m.verify_controller_archive({'bound_files': {str(archive): m.file_digest(archive)},
                                     'source_files': {'missing.py': 'a' * 64}})
