"""Offline adversarial checks for the additive evidence auditor; no runtime/model."""
import base64
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('r12_audit', ROOT / 'scripts/verify_r12_g0.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_strict_json_rejects_duplicate_keys_and_nonfinite():
    for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
        with pytest.raises(ValueError):
            audit.strict_parse(text)


def test_observation_requires_both_complete_objects_in_one_json():
    expected = {'manifest': {'source_file': 'x.json'}, 'source': {'items': [1, 2]}}
    assert audit.complete_observation('prefix\n' + json.dumps(expected) + '\nsuffix', expected)
    assert not audit.complete_observation(json.dumps(expected['manifest']) + '\n' + json.dumps(expected['source']), expected)
    assert not audit.complete_observation(json.dumps({'manifest': expected['manifest'], 'source': {'items': [1]}}), expected)
    assert not audit.complete_observation(json.dumps({**expected, 'unrequested': True}), expected)


def test_observation_comparison_preserves_numeric_json_types():
    assert not audit.complete_observation('{"quota": "24"}', {'quota': 24})
    assert not audit.complete_observation('{"quota": true}', {'quota': 1})


def test_independent_target_does_not_accept_recorded_type_or_order():
    g0 = audit.G0
    manifest = audit.load(g0 / 'inputs/manifest.json')
    expected = audit.expected_target(audit.load(g0 / 'inputs' / manifest['source_file']))
    actual = audit.load(g0 / 'final-workspace' / manifest['target_file'])
    assert list(expected['services_by_id']) == ['atlas', 'beacon', 'zenith']
    assert list(actual['services_by_id']) == ['zenith', 'atlas', 'beacon']
    assert all(type(v['quota']['hard']) is int for v in expected['services_by_id'].values())
    assert all(type(v['quota']['hard']) is str for v in actual['services_by_id'].values())
    assert audit.canonical(expected) != audit.canonical(actual)


def test_delta_recomputed_from_bytes_not_declared_receipt():
    def snapshot(data):
        return {'entries': [{'path': 'a.json', 'kind': 'file', 'mode': 420,
                             'data': base64.b64encode(data).decode()}]}
    before, after = snapshot(b'old'), snapshot(b'new')
    change = audit.snapshot_delta(before, after)
    assert len(change) == 1
    assert change[0]['before']['sha256'] != change[0]['after']['sha256']
    assert change[0]['after']['size_bytes'] == 3
    assert not audit.snapshot_delta(after, after)


def test_saved_writes_exist_despite_false_whole_grade_receipt():
    receipt = audit.load(audit.G0 / 'qualification.json')
    assert receipt['nontrivial_file_change_verified'] is False
    writes = [audit.load(audit.G0 / 'tool-dispatch' / f'call-{i:03d}/dispatch.json') for i in (4, 5)]
    assert all(d['raw_result']['exit_code'] == 0 and d['error'] is None for d in writes)
    assert [x['path'] for d in writes for x in d['fs_delta']] == ['service-map.v2.json', 'migration-report.json']
    assert [x['after']['size_bytes'] for d in writes for x in d['fs_delta']] == [862, 221]


def test_final_invalid_native_call_is_preserved_and_undispatched():
    native = audit.rows(audit.G0 / 'agent/model_calls.jsonl')
    wire = [r for r in native if r['type'] == 'wire_response']
    emitted = [c for r in wire for c in r['response']['choices'][0]['message']['tool_calls']]
    dispatches = [audit.load(p) for p in (audit.G0 / 'tool-dispatch').glob('*/dispatch.json')]
    rejected = [c for c in emitted if c['id'] not in {d['tool_call_id'] for d in dispatches}]
    assert len(wire) == len(emitted) == 15
    assert len(dispatches) == 14
    assert len(rejected) == 1
    assert rejected[0]['function']['name'] == 'workspace=workspace_python'


def test_saved_final_readback_is_complete_but_initial_observation_absent():
    manifest = audit.load(audit.G0 / 'inputs/manifest.json')
    source = audit.load(audit.G0 / 'inputs' / manifest['source_file'])
    dispatches = [audit.load(p) for p in sorted((audit.G0 / 'tool-dispatch').glob('*/dispatch.json'))]
    assert not any(audit.complete_observation(d['raw_result']['stdout'], {'manifest': manifest, 'source': source}) for d in dispatches[:4])
    target = audit.load(audit.G0 / 'final-workspace' / manifest['target_file'])
    report = audit.load(audit.G0 / 'final-workspace' / manifest['report_file'])
    expected = {'target': target, 'report': report, 'batch_id': source['batch_id'],
                'target_sha256': audit.digest(audit.G0 / 'final-workspace' / manifest['target_file']),
                'active_count': len(target['services_by_id'])}
    assert audit.complete_observation(dispatches[6]['raw_result']['stdout'], expected)
    assert not dispatches[6]['fs_delta']


def test_empty_stage_and_explicit_not_run_marker_are_not_executions(tmp_path):
    stage = tmp_path / 'originals'
    assert audit.no_original_executions(stage)
    stage.mkdir()
    assert audit.no_original_executions(stage)
    receipt = {'status': 'NOT_RUN_G0_BLOCKED', 'llm_calls': 0,
               'new_physical_branches': 0, 'prior_evidence_promoted': False}
    (stage / 'NOT_RUN.json').write_text(json.dumps(receipt))
    assert audit.no_original_executions(stage)
    assert audit.no_original_executions(audit.RUN / 'originals')
    (stage / 'trial.json').write_text('{}')
    assert not audit.no_original_executions(stage)


@pytest.mark.parametrize('override', [
    {'status': 'PASS'}, {'llm_calls': 1}, {'new_physical_branches': 1},
    {'prior_evidence_promoted': True}, {'llm_calls': False},
    {'new_physical_branches': '0'}, {'prior_evidence_promoted': 0},
])
def test_invalid_not_run_receipt_is_rejected(tmp_path, override):
    receipt = {'status': 'NOT_RUN_G0_BLOCKED', 'llm_calls': 0,
               'new_physical_branches': 0, 'prior_evidence_promoted': False}
    (tmp_path / 'NOT_RUN.json').write_text(json.dumps({**receipt, **override}))
    assert not audit.no_original_executions(tmp_path)


def test_unrecognized_or_malformed_stage_entries_are_rejected(tmp_path):
    marker = tmp_path / 'NOT_RUN.json'
    for text in ('{}', 'not json', '[]', '{"status":"NOT_RUN_G0_BLOCKED","status":"PASS"}'):
        marker.write_text(text)
        assert not audit.no_original_executions(tmp_path)
    marker.unlink()
    (tmp_path / 'trial').mkdir()
    assert not audit.no_original_executions(tmp_path)
