"""Independent development qualification for the frozen R1.2 execution system."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time

from ..evidence._storage import encode
from .agent import NativeAgent
from .docker_runtime import DockerRuntime, snapshot_workspace
from .qualification import delivered_tool_result
from .r11_replay import DispatchRecorder
from .r12_system import fingerprint_model, system_identity, validated_system
from .r12_service import verify_service
from .tokenizer_client import TokenizerClient

SYSTEM_PROMPT = (
    "Complete the user's file-based task using the native workspace_python tool. "
    "Inspect the actual supplied files and relevant reference material before editing. "
    "Use observed filenames and contents, and use tool feedback to correct errors within the remaining budget. "
    "Preserve content and formatting that the task does not ask to change. "
    "Save each deliverable under the designated filename. "
    "After writing, reopen the saved outputs in a separate tool call and verify the requested properties. "
    "Finish with a short account of what was verified and any unresolved requirements."
)


def write(path, value):
    with Path(path).open('xb') as stream:
        stream.write(encode(value))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_migration(source):
    services = {}
    for entry in sorted(source['services'], key=lambda value: value['id']):
        if entry['enabled']:
            services[entry['id']] = {'listen_port': int(entry['port']),
                'quota': {key: int(entry['quota'][key]) for key in ('soft', 'hard')},
                'tags': sorted({tag.strip().lower() for tag in entry['tags']}), 'notes': entry['notes']}
    return {'version': 2, 'batch_id': source['batch_id'], 'owner': copy.deepcopy(source['owner']),
            'services_by_id': services, 'quota_total': sum(entry['quota']['soft'] for entry in services.values())}


def qualification_inputs(root, nonce):
    """Create a prospective public task and a private oracle before model execution."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    public = root / 'inputs'
    public.mkdir(mode=0o700)
    source_name = 'legacy-services-' + nonce + '.json'
    source = {'version': 1, 'batch_id': nonce,
        'owner': {'name': '演示团队', 'preserve': [True, 0, '007', {'note': '保留原样 α'}]},
        'services': [
            {'id': 'zenith', 'enabled': True, 'port': ' 8102 ', 'quota': {'soft': '12', 'hard': '24'}, 'tags': [' API ', 'blue', 'api'], 'notes': 'Keep Z exactly.'},
            {'id': 'dormant', 'enabled': False, 'port': 'unused', 'quota': {'soft': 'unused', 'hard': 'unused'}, 'tags': ['ignore'], 'notes': 'Inactive.'},
            {'id': 'atlas', 'enabled': True, 'port': '09001', 'quota': {'soft': ' 8 ', 'hard': '016'}, 'tags': [' Prod ', 'PROD', ' edge'], 'notes': '保留 A 原文。'},
            {'id': 'beacon', 'enabled': True, 'port': '8200', 'quota': {'soft': '5', 'hard': '10'}, 'tags': ['service', ' API', 'service '], 'notes': 'Keep B exactly.'},
            {'id': 'retired', 'enabled': False, 'port': '0', 'quota': {'soft': '0', 'hard': '0'}, 'tags': [], 'notes': 'Inactive two.'}]}
    write(public / source_name, source)
    manifest = {'source_file': source_name, 'source_sha256': sha(public / source_name),
                'target_file': 'service-map.v2.json', 'report_file': 'migration-report.json', 'batch_id': nonce}
    write(public / 'manifest.json', manifest)
    task = (
        "This is a separate development qualification task, not an OfficeVal case. "
        "Use at least three separate tool turns: first inspect manifest.json and its complete named source file, "
        "printing one JSON object with keys manifest and source containing both complete parsed objects; "
        "then perform the migration; finally reopen the written files and verify them. "
        "Create the manifest's target JSON with exactly version=2, batch_id and owner copied unchanged, "
        "services_by_id containing enabled services only in ID order, and quota_total equal to the sum of their soft quotas. "
        "Each service value must contain listen_port (integer), quota.soft and quota.hard (integers), "
        "tags (trimmed, lowercase, deduplicated and sorted), and notes copied exactly. "
        "Create the manifest's report JSON with exactly batch_id, source_sha256, target_sha256 computed from the actual saved bytes, "
        "and active_count. Preserve both input files byte-for-byte. "
        "In the final verification tool turn, independently reopen both output files and print one JSON object "
        "with exactly target (the complete parsed target), report (the complete parsed report), "
        "batch_id, target_sha256 computed from disk bytes, and active_count computed from the reopened target."
    )
    private = root / 'private'
    private.mkdir(mode=0o700)
    write(private / 'expected.json', expected_migration(source))
    protocol = {'family': 'independent-json-service-configuration-migration-v1', 'not_officeval': True,
        'excluded_from_future_candidates_and_confirmation': True, 'task_prompt': task,
        'source_file': source_name, 'input_hashes': {name: sha(public / name) for name in ('manifest.json', source_name)},
        'required_distinct_tool_turns': 3, 'require_successful_inspection_write_and_readback': True,
        'failure_action': 'BLOCKED_SYSTEM; do not freeze or execute candidate tasks; no alternate model',
        'expected_sha256': sha(private / 'expected.json')}
    write(root / 'qualification_protocol.json', protocol)
    return protocol


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def observation_object(stdout, expected):
    """Accept a complete observed JSON object, with no guessed substring matches."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(stdout):
        if character != '{':
            continue
        try:
            value, _ = decoder.raw_decode(stdout[index:])
            if encode(value) == encode(expected):
                return True
        except (ValueError, TypeError):
            continue
    return False


def grade_migration(protocol, expected, workspace, dispatches, agent):
    """Deterministic controller evaluation of artifacts and actual process records."""
    workspace = Path(workspace)
    target, report = workspace / 'service-map.v2.json', workspace / 'migration-report.json'
    checks = {'valid_native_protocol': agent.get('protocol_status') == 'valid',
        'normal_termination': agent.get('stopped_reason') == 'assistant_finished',
        'metered_requests': not agent.get('token_usage_unknown', True),
        'separate_tool_turns': len({entry['turn'] for entry in dispatches}) >= 3,
        'inputs_preserved': all((workspace / name).is_file() and sha(workspace / name) == digest
                               for name, digest in protocol['input_hashes'].items()),
        'correct_nested_migration': False, 'correct_report': False,
        'successful_observation_before_write': False, 'successful_nontrivial_write': False,
        'successful_fresh_readback': False}
    detail = {}
    try:
        actual, summary = strict_json(target), strict_json(report)
        checks['correct_nested_migration'] = (encode(actual) == encode(expected)
            and list(actual['services_by_id']) == sorted(expected['services_by_id']))
        target_hash = sha(target)
        source_hash = protocol['input_hashes'][protocol['source_file']]
        wanted = {'batch_id': expected['batch_id'], 'source_sha256': source_hash,
                  'target_sha256': target_hash, 'active_count': len(expected['services_by_id'])}
        checks['correct_report'] = encode(summary) == encode(wanted)
        detail = {'target_sha256': target_hash, 'target_size_bytes': target.stat().st_size,
                  'report_sha256': sha(report), 'active_count': wanted['active_count']}
        successful = [entry for entry in dispatches if entry.get('error') is None and
            isinstance(entry.get('raw_result'), dict) and entry['raw_result'].get('exit_code') == 0
            and entry['raw_result'].get('error') is None and entry.get('fs_delta') is not None]
        writes = [entry for entry in successful if any(delta['path'] == 'service-map.v2.json' and delta['after']
                  for delta in entry['fs_delta'])]
        checks['successful_nontrivial_write'] = bool(writes) and checks['correct_nested_migration']
        if writes:
            mutations = [entry for entry in dispatches if any(delta['path'] in
                {'service-map.v2.json', 'migration-report.json'} for delta in (entry.get('fs_delta') or []))]
            first_write = min(entry['turn'] for entry in mutations)
            last_write = max(entry['turn'] for entry in mutations)
            initial_observation = {'manifest': strict_json(workspace / 'manifest.json'),
                                   'source': strict_json(workspace / protocol['source_file'])}
            final_observation = {'target': actual, 'report': summary, 'batch_id': expected['batch_id'],
                                 'target_sha256': target_hash, 'active_count': wanted['active_count']}
            checks['successful_observation_before_write'] = any(entry['turn'] < first_write
                and observation_object(entry['raw_result']['stdout'], initial_observation) and not entry['fs_delta']
                for entry in successful)
            checks['successful_fresh_readback'] = any(entry['turn'] > last_write and not entry['fs_delta']
                and observation_object(entry['raw_result']['stdout'], final_observation)
                for entry in successful)
    except (OSError, ValueError, KeyError, TypeError) as error:
        detail['grading_error'] = type(error).__name__ + ': ' + str(error)
    return {'passed': all(checks.values()), 'checks': checks, 'detail': detail,
            'grader_kind': 'independent deterministic controller; no model judgment or candidate code executed'}


async def execute_qualification(root, system_manifest):
    root = Path(root).absolute()
    protocol = strict_json(root / 'qualification_protocol.json')
    system = validated_system(system_manifest['system'])
    if system_identity(system) != system_manifest['system_contract_sha256']:
        raise ValueError('frozen system identity mismatch')
    fingerprint = fingerprint_model(system)
    if fingerprint != system_manifest['model_fingerprint']:
        raise ValueError('model/tokenizer fingerprint changed after system freeze')
    if (root / 'result.json').exists() or (root / 'agent').exists():
        raise FileExistsError('qualification attempt already exists; no automatic retry')
    started = time.perf_counter()
    with DockerRuntime(system['agent_image'], root / 'runtime') as actor:
        actor.seed_workspace(root / 'inputs')
        audit = actor.audit_boundary()
        write(root / 'boundary_before.json', audit)
        if not audit['qualified']:
            raise RuntimeError('qualification actor boundary is unsupported')
        dispatch = DispatchRecorder(actor, root / 'tool-dispatch')
        tokenizer = TokenizerClient()
        agent = NativeAgent({**system['sampling'], 'execution_phase': 'qualification',
            'token_count': tokenizer.count_messages, 'tool_result_transform': delivered_tool_result,
            'generation_guard': lambda: verify_service(system, system_manifest['serving_observation']),
            'identity_context': {'model_fingerprint': fingerprint['sha256'],
                'tokenizer_fingerprint': fingerprint['tokenizer'],
                'system_contract_sha256': system_identity(system)}}, root / 'agent', system_contract=system)
        result = await agent.run([{'role': 'system', 'content': system['system_prompt']},
            {'role': 'user', 'content': protocol['task_prompt']}], {'workspace_python': actor.exec_python},
            system['tool_schemas'], max_turns=system['sampling']['max_turns'], dispatch_callback=dispatch)
        write(root / 'result.json', result)
        actor.export_workspace(root / 'final-workspace')
        end_audit = actor.audit_boundary()
        write(root / 'boundary_after.json', end_audit)
    expected = strict_json(root / 'private/expected.json')
    if sha(root / 'private/expected.json') != protocol['expected_sha256']:
        raise ValueError('private qualification oracle changed')
    grade = grade_migration(protocol, expected, root / 'final-workspace', dispatch.records, result)
    grade['checks']['supported_final_boundary'] = end_audit['qualified']
    grade['passed'] = all(grade['checks'].values())
    write(root / 'independent_grade.json', grade)
    first = dispatch.records[0] if dispatch.records else None
    receipt = {'schema_version': 'r12-qualification-v1', 'status': 'passed' if grade['passed'] else 'blocked_system',
        'is_synthetic': False, 'task_data_kind': 'synthetic independent development qualification',
        'provider': 'openai-api/mlx/default_model', 'model': system['model']['repository'],
        'model_fingerprint': fingerprint['sha256'], 'tokenizer_fingerprint': fingerprint['tokenizer'],
        'system_contract_sha256': system_identity(system), 'system_contract': system,
        'native_tool_calls': len(dispatch.records), 'model_calls': result['modelcall_count'],
        'prompt_tokens': result['tokens']['full_prompt_tokens'], 'completion_tokens': result['tokens']['output_tokens'],
        'tool_call_id': first['tool_call_id'] if first else None,
        'observation_tool_call_id': first['tool_call_id'] if first else None,
        'nonce': expected['batch_id'],
        'observed_nonce': expected['batch_id'] if first and expected['batch_id'] in str(first['raw_result']) else None,
        'independent_nonce': expected['batch_id'], 'model_trace_sha256': sha(root / 'agent/model_calls.jsonl'),
        'qualification_protocol_sha256': sha(root / 'qualification_protocol.json'),
        'independent_grade_sha256': sha(root / 'independent_grade.json'), 'nontrivial_file_change_verified': grade['passed'],
        'wall_seconds': time.perf_counter() - started, 'gpu_seconds': None,
        'candidate_execution_admitted': grade['passed']}
    write(root / 'qualification.json', receipt)
    return receipt
