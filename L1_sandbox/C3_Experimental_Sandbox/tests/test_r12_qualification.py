"""Synthetic qualification-oracle regression tests; no model or Docker."""
import copy
import hashlib
import json
from pathlib import Path
import shutil

from c3_sandbox.r1.r12_qualification import qualification_inputs, grade_migration, write, sha


def fixture(tmp_path):
    root = tmp_path / 'qualification'
    protocol = qualification_inputs(root, '918273')
    expected = json.loads((root / 'private/expected.json').read_bytes())
    workspace = root / 'final'
    shutil.copytree(root / 'inputs', workspace)
    write(workspace / 'service-map.v2.json', expected)
    target_hash = sha(workspace / 'service-map.v2.json')
    write(workspace / 'migration-report.json', {'batch_id': '918273',
        'source_sha256': protocol['input_hashes'][protocol['source_file']],
        'target_sha256': target_hash, 'active_count': 3})
    def call(turn, stdout, delta):
        return {'turn': turn, 'error': None, 'raw_result': {'exit_code': 0, 'error': None, 'stdout': stdout}, 'fs_delta': delta}
    initial = {'manifest': json.loads((workspace / 'manifest.json').read_bytes()),
               'source': json.loads((workspace / protocol['source_file']).read_bytes())}
    final = {'target': expected, 'report': json.loads((workspace / 'migration-report.json').read_bytes()),
             'batch_id': '918273', 'target_sha256': target_hash, 'active_count': 3}
    dispatches = [call(0, json.dumps(initial), []),
        call(1, 'written', [{'path': 'service-map.v2.json', 'after': {'kind': 'file'}}]),
        call(2, json.dumps(final), [])]
    agent = {'protocol_status': 'valid', 'stopped_reason': 'assistant_finished', 'token_usage_unknown': False}
    return protocol, expected, workspace, dispatches, agent


def test_real_artifact_properties_are_required(tmp_path):
    args = fixture(tmp_path)
    assert grade_migration(*args)['passed']
    target = args[2] / 'service-map.v2.json'
    obj = json.loads(target.read_bytes())
    obj['services_by_id']['atlas']['listen_port'] = '9001'
    target.write_text(json.dumps(obj))
    assert not grade_migration(*args)['passed']


def test_source_mutation_rejected(tmp_path):
    args = fixture(tmp_path)
    (args[2] / args[0]['source_file']).write_text('{}')
    result = grade_migration(*args)
    assert not result['passed'] and not result['checks']['inputs_preserved']


def test_separate_successful_tool_turns_and_write_are_required(tmp_path):
    protocol, expected, workspace, dispatches, agent = fixture(tmp_path)
    for event in dispatches:
        event['turn'] = 0
    assert not grade_migration(protocol, expected, workspace, dispatches, agent)['passed']
    dispatches[1]['raw_result']['exit_code'] = 1
    assert not grade_migration(protocol, expected, workspace, dispatches, agent)['checks']['successful_nontrivial_write']


def test_readback_must_match_actual_saved_bytes(tmp_path):
    protocol, expected, workspace, dispatches, agent = fixture(tmp_path)
    dispatches[-1]['raw_result']['stdout'] = 'I verified it.'
    assert not grade_migration(protocol, expected, workspace, dispatches, agent)['checks']['successful_fresh_readback']


def test_invalid_protocol_unknown_cost_and_duplicate_json_fail(tmp_path):
    protocol, expected, workspace, dispatches, agent = fixture(tmp_path)
    bad = dict(agent, protocol_status='protocol_invalid', token_usage_unknown=True)
    assert not grade_migration(protocol, expected, workspace, dispatches, bad)['passed']
    (workspace / 'migration-report.json').write_text('{"batch_id":"a","batch_id":"b"}')
    assert not grade_migration(protocol, expected, workspace, dispatches, agent)['passed']


def test_manifest_only_and_hash_only_observations_do_not_qualify(tmp_path):
    protocol, expected, workspace, dispatches, agent = fixture(tmp_path)
    dispatches[0]['raw_result']['stdout'] = (workspace / 'manifest.json').read_text()
    dispatches[-1]['raw_result']['stdout'] = sha(workspace / 'service-map.v2.json') + ' 918273 3'
    checks = grade_migration(protocol, expected, workspace, dispatches, agent)['checks']
    assert not checks['successful_observation_before_write']
    assert not checks['successful_fresh_readback']


def test_failed_output_mutations_still_bound_observation_order(tmp_path):
    protocol, expected, workspace, dispatches, agent = fixture(tmp_path)
    failed = {'turn': -1, 'error': None, 'raw_result': {'exit_code': 1, 'error': None, 'stdout': ''},
              'fs_delta': [{'path': 'migration-report.json', 'after': {'kind': 'file'}}]}
    checks = grade_migration(protocol, expected, workspace, [failed, *dispatches], agent)['checks']
    assert not checks['successful_observation_before_write']
    failed['turn'] = 3
    checks = grade_migration(protocol, expected, workspace, [*dispatches, failed], agent)['checks']
    assert not checks['successful_fresh_readback']
