import asyncio
import copy
import hashlib
import json
from pathlib import Path

import pytest
from c3_sandbox.manifests import canonical_hash
from c3_sandbox.replay import ReplayQualification, STATE_CONTRACT_V1
from c3_sandbox.r1 import r11_replay as r11
from c3_sandbox.r1.docker_runtime import snapshot_workspace


def baseline(tmp_path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'file').write_bytes(b'original')
    replay = ReplayQualification(tmp_path / 'store')
    runtime = {'image_digest': 'sha256:' + 'a' * 64, 'config': {}, 'model_config': {},
               'tool_config': {}, 'environment': {}}
    messages = [{'role': 'user', 'content': 'task'}]
    cps = []
    events = []
    for turn in range(3):
        calls = [{'id': f'c{turn}-{i}', 'type': 'function', 'function': 'workspace_python',
                  'arguments': {'code': 'pass'}} for i in range(2)]
        messages.append({'role': 'assistant', 'content': '', 'tool_calls': calls})
        for call in calls:
            messages.append({'role': 'tool', 'content': 'ok', 'function': 'workspace_python', 'tool_call_id': call['id']})
            events.append({'event_index': len(events), 'type': 'tool_result', 'turn': turn,
                           'tool_call_id': call['id'], 'raw_result': {'exit_code': 0}})
        identifier = replay.register_checkpoint(workspace, messages, 'b' * 64, runtime,
                                                copy.deepcopy(STATE_CONTRACT_V1), is_synthetic=False)
        folder = tmp_path / f'cp{turn}'
        folder.mkdir()
        for name, value in [('messages', messages), ('checkpoint_record', replay._record(identifier)),
                            ('source_audit', {'qualified': True})]:
            (folder / (name + '.json')).write_text(json.dumps(value))
        cps.append({'checkpoint_id': identifier, 'event_index': len(events) - 1,
                    'source_boundary_qualified': True, **{name + '_path': str(folder / (name + '.json'))
                     for name in ('messages', 'checkpoint_record', 'source_audit')}})
    return {'task_id': 'task', 'is_synthetic': False, 'execution_valid': True, 'task_completed': True,
            'runtime': runtime, 'replay_store': str(tmp_path / 'store'), 'checkpoints': cps,
            'agent_result': {'events': events, 'tool_dispatch': copy.deepcopy(events), 'protocol_status': 'valid'}}


def test_selection_earliest_later_call_uses_completed_batch(tmp_path):
    result = baseline(tmp_path)
    selected = r11.select_checkpoint(result, 'T')
    assert selected['checkpoint_id'] == result['checkpoints'][0]['checkpoint_id']
    assert selected['subsequent_tool_call_id'] == 'c1-0'
    assert selected['remaining_turns'] == 9
    assert r11.select_checkpoint(result, 'S')['reference'] == result['checkpoints'][0]
    assert r11.select_checkpoint(result, 'S', max_turns=1) is None


def test_selection_rejects_tampered_and_unqualified_boundaries(tmp_path):
    result = baseline(tmp_path)
    Path(result['checkpoints'][0]['messages_path']).write_text('[]')
    result['checkpoints'][1]['source_boundary_qualified'] = False
    assert r11.select_checkpoint(result, 'T') is None
    assert r11.select_checkpoint(result, 'S')['reference'] == result['checkpoints'][2]
    result['agent_result']['protocol_status'] = 'protocol_invalid'
    assert r11.select_checkpoint(result, 'S') is None


def test_all_candidate_decisions_retained_and_first_selected(tmp_path):
    result = baseline(tmp_path)
    failed = {**result, 'task_id': 'failed', 'task_completed': False}
    chosen = r11.select_engineering_candidates([failed, result])
    assert len(chosen['decisions']) == 2
    assert chosen['S_candidate']['task_id'] == 'task'
    assert chosen['T_candidate']['task_id'] == 'failed'


def test_trace_hash_drift_rejected_before_docker(tmp_path, monkeypatch):
    monkeypatch.setattr(r11, '_qualification', lambda model: model)
    monkeypatch.setattr(r11, 'DockerRuntime', lambda *args: pytest.fail('Docker must not start'))
    trace = tmp_path / 'trace'
    trace.write_bytes(b'changed trace')
    with pytest.raises(ValueError, match='trace hash mismatch'):
        r11.admit_checkpoint({'model_qualification': {'model_trace_sha256': 'a' * 64}}, {},
                             tmp_path / 'admission', trace)
    assert not (tmp_path / 'admission').exists()


def test_dispatch_filesystem_delta_binds_actual_call_and_modes(tmp_path):
    workspace = tmp_path / 'ws'
    workspace.mkdir()
    (workspace / 'old').write_bytes(b'old')
    (workspace / 'mode').write_bytes(b'same')
    class Actor:
        def _snapshot(self):
            return snapshot_workspace(workspace)
    recorder = r11.DispatchRecorder(Actor(), tmp_path / 'dispatch')
    event = {'stage': 'start', 'turn': 0, 'tool_call_id': 'sdk-call', 'function': 'workspace_python',
             'arguments': {'code': 'model code'}, 'raw_result': None, 'error': None}
    asyncio.run(recorder(event))
    (workspace / 'old').unlink()
    (workspace / 'new').write_bytes(b'new')
    (workspace / 'mode').chmod(0o600)
    asyncio.run(recorder({**event, 'stage': 'end', 'raw_result': {'exit_code': 0}}))
    record = recorder.records[0]
    assert record['tool_call_id'] == 'sdk-call'
    assert record['code_sha256'] == hashlib.sha256(b'model code').hexdigest()
    assert {item['path'] for item in record['fs_delta']} == {'old', 'new', 'mode'}
    assert json.loads(Path(record['after_path']).read_text()) == snapshot_workspace(workspace)


def test_unsupported_snapshot_does_not_hide_dispatch_but_timeout_fails_closed(tmp_path):
    class Actor:
        invalidated = False
        def _snapshot(self):
            raise RuntimeError('workspace snapshot failed: symlink, hardlink or special file rejected: link')
    actor = Actor()
    recorder = r11.DispatchRecorder(actor, tmp_path / 'dispatch')
    event = {'stage': 'start', 'turn': 0, 'tool_call_id': 'call', 'function': 'workspace_python',
             'arguments': {'code': 'pass'}, 'raw_result': None, 'error': None}
    asyncio.run(recorder(event))
    asyncio.run(recorder({**event, 'stage': 'end'}))
    assert recorder.records[0]['fs_delta'] is None
    assert recorder.records[0]['snapshot_ineligible_reasons']
    actor.invalidated = True
    with pytest.raises(RuntimeError):
        asyncio.run(recorder({**event, 'tool_call_id': 'later'}))


def test_selection_requires_valid_protocol_and_actual_dispatch_link(tmp_path):
    result = baseline(tmp_path)
    result['agent_result']['tool_dispatch'] = []
    assert r11.select_checkpoint(result, 'T') is None
    del result['agent_result']['protocol_status']
    assert r11.select_checkpoint(result, 'S') is None


def test_t_does_not_skip_ineligible_nearest_completed_boundary(tmp_path):
    result = baseline(tmp_path)
    result['checkpoints'][1]['source_boundary_qualified'] = False
    result['agent_result']['tool_dispatch'] = [result['agent_result']['events'][4]]
    assert r11.select_checkpoint(result, 'T') is None
