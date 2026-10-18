"""Offline mechanics assertions; fixtures contain no model or grader execution."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/audit_g1_mechanical.py'
spec = importlib.util.spec_from_file_location('g1_mechanical_audit', SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')
    return {'path': str(path.name), 'sha256': audit.file_digest(path), 'size_bytes': path.stat().st_size}


def fixture(root):
    snapshot = {'snapshot_id': 's000007', 'files': [], 'directories': [], 'total_bytes': 0,
                'content_sha256': audit.digest({'files': [], 'directories': []})}
    snap_ref = save(root / 'state.json', snapshot)
    cp = {'checkpoint_id': 'cp000000', 'snapshot_id': snapshot['snapshot_id'],
          'budget_used': {'model_requests': 0}, 'qualification': {'eligible': True},
          'conversation': {'messages': []}}
    cp_ref = save(root / 'cp.json', cp)
    save(root / 'checkpoints/index.json', {'checkpoints': [{'path': 'cp.json', 'checkpoint_id': 'cp000000'}]})
    events = [{'kind': 'state.snapshot', 'seq': 0, 'event_id': 's', 'data': {'snapshot': snap_ref}},
              {'kind': 'checkpoint.created', 'seq': 1, 'event_id': 'c', 'data': {'checkpoint': cp_ref}},
              {'kind': 'model.request', 'seq': 2, 'event_id': 'r', 'data': {}}]
    return snapshot, cp, events


def test_checkpoint_snapshot_binding_is_not_ordinal(tmp_path):
    _, _, events = fixture(tmp_path)
    rows = audit.bound_checkpoints(tmp_path, events)
    assert rows[0]['checkpoint']['checkpoint_id'] == 'cp000000'
    assert rows[0]['snapshot']['snapshot_id'] == 's000007'
    assert rows[0]['genuine_post_compaction'] is False


def test_wrong_snapshot_binding_rejected(tmp_path):
    _, cp, events = fixture(tmp_path)
    cp['snapshot_id'] = 's000008'
    events[1]['data']['checkpoint'] = save(tmp_path / 'cp.json', cp)
    with pytest.raises(KeyError):
        audit.bound_checkpoints(tmp_path, events)


def test_old_snapshot_binding_rejected_even_when_present(tmp_path):
    snapshot, _, events = fixture(tmp_path)
    snapshot['snapshot_id'] = 's000008'
    ref = save(tmp_path / 'other_state.json', snapshot)
    events.insert(1, {'kind': 'state.snapshot', 'seq': .5, 'event_id': 'later', 'data': {'snapshot': ref}})
    with pytest.raises(ValueError, match='most recent'):
        audit.bound_checkpoints(tmp_path, events)


def test_missing_blob_fails_before_restore(tmp_path):
    snapshot = {'snapshot_id': 's000007', 'files': [{'path': 'a', 'blob': 'states/blobs/missing',
                'sha256': '0'*64, 'size_bytes': 1, 'mode': 0o640}], 'directories': [], 'total_bytes': 1}
    snapshot['content_sha256'] = audit.digest({'files': snapshot['files'], 'directories': []})
    with pytest.raises((ValueError, OSError)):
        audit.replay_workspace(tmp_path, snapshot, {'runtime': {'max_files': 10, 'max_workspace_bytes': 100}})


def test_request_mismatch_rejected_using_production_backend(tmp_path, monkeypatch):
    config = {'model': {'adapter': 'openai-compatible', 'model': 'example', 'parameters': {},
                        'api_key_env': None, 'base_url': 'http://127.0.0.1:1/v1', 'kind': 'local'}}
    monkeypatch.setattr(audit, 'tools_for_config', lambda config: [])
    ref = save(tmp_path / 'request.json', {'model': 'wrong', 'messages': [], 'stream': False, 'n': 1})
    row = {'checkpoint': {'conversation': {'messages': []}},
           'next_request': {'data': {'body': ref, 'request_id': 'r1'}}}
    with pytest.raises(ValueError, match='request reconstruction mismatch'):
        audit.reconstruct_request(tmp_path, config, row)
    ref = save(tmp_path / 'request.json', {'model': 'example', 'messages': [], 'stream': False, 'n': 1})
    row['next_request']['data']['body'] = ref
    assert audit.reconstruct_request(tmp_path, config, row)['following_request_id'] == 'r1'


def test_genuine_compaction_requires_exact_saved_receipt(tmp_path):
    _, cp, events = fixture(tmp_path)
    receipt = {'at_request': 0, 'summary_sha256': 'a'*64}
    events.insert(1, {'kind': 'session.compacted', 'seq': .5, 'event_id': 'comp', 'data': receipt})
    with pytest.raises(ValueError, match='compaction state'):
        audit.bound_checkpoints(tmp_path, events)
    cp['conversation']['_pi_compaction'] = [receipt]
    events[2]['data']['checkpoint'] = save(tmp_path / 'cp.json', cp)
    rows = audit.bound_checkpoints(tmp_path, events)
    assert 'genuine_post_compaction' in audit.selected_checkpoints(rows)[0]['roles']


def test_empty_workspace_roundtrip_preserves_modes(tmp_path):
    snapshot, _, _ = fixture(tmp_path)
    result = audit.replay_workspace(tmp_path, snapshot, {'runtime': {'max_files': 10, 'max_workspace_bytes': 100}})
    assert result['status'] == 'byte_and_mode_equal'


def test_nonempty_workspace_roundtrip_checks_actual_bytes_and_modes(tmp_path):
    content = b'original saved evidence\n'
    blob = tmp_path / 'states/blobs/example'
    blob.parent.mkdir(parents=True)
    blob.write_bytes(content)
    files = [{'path': 'submission/f.txt', 'blob': 'states/blobs/' + audit.file_digest(blob),
              'sha256': audit.file_digest(blob), 'size_bytes': len(content), 'mode': 0o640}]
    blob.rename(tmp_path / files[0]['blob'])
    directories = [{'path': 'submission', 'mode': 0o750}]
    snapshot = {'snapshot_id': 's42', 'files': files, 'directories': directories,
                'total_bytes': len(content), 'content_sha256': audit.digest({'files': files, 'directories': directories})}
    result = audit.replay_workspace(tmp_path, snapshot, {'runtime': {'max_files': 10, 'max_workspace_bytes': 100}})
    assert result['files'] == files
    assert result['directories'] == directories


def test_inventory_default_never_executes_or_promotes_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'audit_package', lambda *args: pytest.fail('must not restore without --execute'))
    output = tmp_path / 'report.json'
    monkeypatch.setattr('sys.argv', [str(SCRIPT), '--runs', str(tmp_path), '--output', str(output)])
    assert audit.main() == 0
    report = json.loads(output.read_text())
    assert report['actor_requests'] == 0
    assert report['applicable_configuration_verified'] is False
    assert report['live_g1_gate_pass'] is False
    assert len(report['slots']) == 45
