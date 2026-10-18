"""Strict R22 boundary and scope regressions; run on the CPU audit host."""
import importlib.util
from pathlib import Path
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/audit_r22_native_validation.py'
spec = importlib.util.spec_from_file_location('r22_audit_test', SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_exact_42_scope_rejects_unknown_missing_and_recovery():
    inventory = dict.fromkeys(audit.SLOTS, 'hash')
    audit.check_scope(inventory)
    for bad in ({**inventory, 'officeval_004.r02': 'hash'},
                {k: v for k, v in inventory.items() if k != audit.SLOTS[0]},
                {**inventory, 'holdout_001': 'hash'}):
        with pytest.raises(ValueError, match='exactly'):
            audit.check_scope(bad)


@pytest.mark.parametrize('raw,validated', [({'offset': '2'}, {'offset': 2}),
                                         ({'offset': True}, {'offset': 1}),
                                         ({'offset': 2}, {'offset': 2.0})])
def test_native_coercion_halts_even_python_equality(raw, validated):
    row = audit.compare_call({'name': 'read', 'arguments': raw},
                             {'ok': True, 'validated_arguments': validated}, lambda a: a)
    assert row['boundary_pass'] is False
    assert row['raw_equal'] is False


def test_legacy_transform_cannot_bless_raw_native_difference():
    raw = {'path': '/workspace/a', 'oldText': 'x', 'newText': 'y'}
    transformed = {'path': '/workspace/a', 'edits': [{'oldText': 'x', 'newText': 'y'}]}
    row = audit.compare_call({'name': 'edit', 'arguments': raw},
                             {'ok': True, 'validated_arguments': transformed}, lambda a: transformed)
    assert row['legacy_effective_equal'] is True
    assert row['boundary_pass'] is False


def test_native_failure_is_preserved():
    error = {'ok': False, 'validated_arguments': None, 'native_result': {'isError': True, 'content': ['missing path']}}
    row = audit.compare_call({'name': 'edit', 'arguments': {'edits': [{'path': 'a'}]}}, error, lambda a: a)
    assert row['native'] == error
    assert row['boundary_pass'] is False


def test_frozen_inventory_detects_tamper(tmp_path):
    path = tmp_path / 'evidence_index.json'
    path.write_text('original')
    expected = audit.sha(path)
    path.write_text('tampered')
    assert audit.sha(path) != expected


def test_seal_failure_prevents_native_validation(tmp_path):
    class ForbiddenValidator:
        def validate_batch(self, calls):
            pytest.fail('tampered package reached native validation')
    with pytest.raises(ValueError, match='full seal'):
        audit.audit_package(tmp_path, 'trusted-external-index', ForbiddenValidator(), lambda a: a,
                            lambda *a, **kw: {'verified': False, 'errors': ['file_inventory_mismatch']},
                            None, None, None, None)


def test_unknown_tool_failure_cannot_be_skipped():
    row = audit.compare_call({'name': 'unknown', 'arguments': {}},
                             {'ok': False, 'validated_arguments': None,
                              'native_result': {'isError': True, 'content': ['Tool not found']}}, lambda a: a)
    assert row['boundary_pass'] is False


def test_prefix_refuses_skipped_or_post_failure_calls(tmp_path):
    import json
    events = [{'kind': 'model.response', 'seq': 3, 'data': {'request_id': 'failed'}},
              {'kind': 'tool.call', 'seq': 4, 'data': {'call_id': 'bad'}}]
    (tmp_path / 'trajectory.jsonl').write_text('\n'.join(json.dumps(e) for e in events))
    with pytest.raises(ValueError, match='non-successful or post-failure'):
        audit.prefix_validation(tmp_path, 'failed', None, lambda a: a)


def test_prefix_requires_exact_failed_response_binding(tmp_path):
    (tmp_path / 'trajectory.jsonl').write_text('')
    with pytest.raises(ValueError, match='response binding'):
        audit.prefix_validation(tmp_path, 'missing', None, lambda a: a)


@pytest.mark.parametrize('validated,ok,expected', [({'path': 'a', 'offset': '2'}, True, 'pass'),
                                                ({'path': 'a', 'offset': 2}, True, 'protocol_halt'),
                                                (None, False, 'protocol_halt')])
def test_prefix_checks_every_successful_raw_call(tmp_path, validated, ok, expected):
    import json
    raw = {'path': 'a', 'offset': '2'}
    response = {'choices': [{'message': {'tool_calls': [{'id': 'native1', 'function': {'name': 'read', 'arguments': json.dumps(raw)}}]}}]}
    target = tmp_path / 'response.json'
    target.write_text(json.dumps(response))
    data = {'request_id': 'prior', 'call_id': 'prior:c0', 'native_id': 'native1', 'name': 'read', 'arguments': raw}
    events = [{'kind': 'model.response', 'seq': 0, 'data': {'request_id': 'prior', 'body': {'path': target.name, 'sha256': audit.sha(target)}}},
              {'kind': 'tool.call', 'seq': 1, 'data': data},
              {'kind': 'tool.result', 'seq': 2, 'data': {'call_id': 'prior:c0'}},
              {'kind': 'model.response', 'seq': 3, 'data': {'request_id': 'failed'}},
              {'kind': 'tool.call', 'seq': 4, 'data': {'request_id': 'failed', 'call_id': 'failed:c0'}}]
    (tmp_path / 'trajectory.jsonl').write_text('\n'.join(json.dumps(e) for e in events))
    class Validator:
        def dependency_identity(self):
            return {'identity': 'frozen'}
        def validate_batch(self, calls):
            assert calls == [{'name': 'read', 'arguments': raw, 'id': 'native1'}]
            return [{'ok': ok, 'raw_arguments': raw, 'validated_arguments': validated,
                     'telemetry': {'actor_requests': 0, 'native_tool_executions': 0, 'native_loop_observed_responses': 1},
                     'provenance': self.dependency_identity()}]
    report = audit.prefix_validation(tmp_path, 'failed', Validator(), lambda a: a)
    assert report['status'] == expected
    assert report['native_validation_calls'] == 1
    assert report['calls'][0]['raw_arguments_sha256'] == audit.value_hash(raw)
