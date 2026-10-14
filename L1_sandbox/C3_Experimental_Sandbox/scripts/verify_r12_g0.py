"""Read-only independent audit of a saved R12 G0 attempt; no model/runtime calls."""
import base64
import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/e2-r1-2'
G0 = RUN / 'system_qualification'

def load(path):
    return strict_parse(path.read_text())

def strict_parse(text):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError('duplicate JSON key: ' + key)
            value[key] = item
        return value
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]

def complete_observation(text, expected):
    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char == '{':
            try:
                value, _ = decoder.raw_decode(text[i:])
                if canonical(value) == canonical(expected):
                    return True
            except ValueError:
                pass
    return False

def expected_target(source):
    services = {}
    for item in sorted(source['services'], key=lambda s: s['id']):
        if item['enabled']:
            services[item['id']] = {
                'listen_port': int(item['port']),
                'quota': {key: int(item['quota'][key]) for key in ('soft', 'hard')},
                'tags': sorted({tag.strip().lower() for tag in item['tags']}),
                'notes': item['notes']}
    return {'version': 2, 'batch_id': source['batch_id'], 'owner': source['owner'],
            'services_by_id': services, 'quota_total': sum(s['quota']['soft'] for s in services.values())}

def snapshot_entries(snapshot):
    entries = {}
    for item in snapshot['entries']:
        value = {k: v for k, v in item.items() if k not in ('path', 'data')}
        if item['kind'] == 'file':
            data = base64.b64decode(item['data'], validate=True)
            value.update(sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))
        entries[item['path']] = value
    return entries

def snapshot_delta(before, after):
    a, b = snapshot_entries(before), snapshot_entries(after)
    return [{'path': name, 'before': a.get(name), 'after': b.get(name)}
            for name in sorted(a.keys() | b.keys()) if a.get(name) != b.get(name)]

def no_original_executions(path):
    """An empty stage or one explicit zero-execution receipt is not a trial."""
    if not path.exists():
        return True
    if not path.is_dir() or path.is_symlink():
        return False
    entries = list(path.iterdir())
    if not entries:
        return True
    marker = path / 'NOT_RUN.json'
    if entries != [marker] or not marker.is_file() or marker.is_symlink():
        return False
    try:
        receipt = load(marker)
        return (isinstance(receipt, dict)
                and receipt.get('status') == 'NOT_RUN_G0_BLOCKED'
                and type(receipt.get('llm_calls')) is int and receipt['llm_calls'] == 0
                and type(receipt.get('new_physical_branches')) is int and receipt['new_physical_branches'] == 0
                and receipt.get('prior_evidence_promoted') is False)
    except (OSError, ValueError):
        return False

def verify():
    if not (G0 / 'qualification.json').exists():
        raise SystemExit('PENDING: final qualification receipt not yet saved')
    manifest = load(RUN / 'system_manifest.json')
    definition = load(RUN / 'system_definition.json')
    protocol = load(G0 / 'qualification_protocol.json')
    freeze = load(G0 / 'freeze_event.json')
    checks, details = {}, {}
    checks['system_definition_equals_frozen_system'] = definition == manifest['system']
    checks['manifest_freeze_hash'] = digest(RUN / 'system_manifest.json') == freeze['system_manifest_sha256']
    checks['protocol_freeze_hash'] = digest(G0 / 'qualification_protocol.json') == freeze['qualification_protocol_sha256']
    sources = definition['provenance']['scaffold_source_files']
    checks['all_frozen_source_snapshot_hashes'] = all(digest(G0 / 'source_snapshot' / name) == sha for name, sha in sources.items())
    checks['all_current_source_hashes'] = all(digest(ROOT / name) == sha for name, sha in sources.items())
    model_dir = ROOT / definition['model']['path']
    checks['model_weights_tokenizer_disk_hashes'] = all(digest(model_dir / name) == sha for name, sha in manifest['model_fingerprint']['files'].items())
    site = Path(manifest['serving_observation']['python_environment']['site_packages'])
    checks['server_parser_disk_hashes'] = all(digest(site / 'mlx_lm' / name) == sha for name, sha in manifest['serving_observation']['python_environment']['mlx_source_sha256'].items())
    native = rows(G0 / 'agent/model_calls.jsonl')
    events = rows(G0 / 'agent/events.jsonl')
    runtime = rows(G0 / 'runtime/runtime_events.jsonl')
    dispatches = [load(p) for p in sorted((G0 / 'tool-dispatch').glob('call-*/dispatch.json'))]
    requests = [r for r in native if r['type'] == 'request']
    wire = {r['turn']: r for r in native if r['type'] == 'wire_response'}
    sdk = {r['turn']: r for r in native if r['type'] == 'provider_call'}
    responses = {r['turn']: r for r in events if r['type'] == 'model_response'}
    checks['saved_request_service_identity'] = all(r['serving_observation'] == manifest['serving_observation'] and r['system_contract_sha256'] == manifest['system_contract_sha256'] for r in requests)
    checks['local_provider_no_paid_api'] = all(r['provider']['base_url'] == 'http://127.0.0.1:18080/v1' and r['provider']['emulate_tools'] is False for r in requests)
    checks['wire_saved_bytes_hashes'] = all(hashlib.sha256(base64.b64decode(r['response_body_base64'])).hexdigest() == r['response_sha256'] and json.loads(base64.b64decode(r['response_body_base64'])) == r['response'] for r in wire.values())
    links = []
    for d in dispatches:
        t = d['turn']
        wc = next(c for c in wire[t]['response']['choices'][0]['message']['tool_calls'] if c['id'] == d['tool_call_id'])
        sc = next(c for c in sdk[t]['response']['choices'][0]['message']['tool_calls'] if c['id'] == d['tool_call_id'])
        nc = next(c for c in responses[t]['message']['tool_calls'] if c['id'] == d['tool_call_id'])
        er = next(r for r in events if r['type'] == 'tool_result' and r['tool_call_id'] == d['tool_call_id'])
        rr = [r for r in runtime if r['kind'] == 'model_python' and r['code_sha256'] == d['code_sha256']]
        links.append(json.loads(wc['function']['arguments']) == d['arguments'] == nc['arguments'] == json.loads(sc['function']['arguments']) and hashlib.sha256(d['arguments']['code'].encode()).hexdigest() == d['code_sha256'] and er['raw_result'] == d['raw_result'] and any(r['exit_code'] == d['raw_result']['exit_code'] and r['stdout'] == d['raw_result']['stdout'] and r['stderr'] == d['raw_result']['stderr'] for r in rr))
    checks['wire_sdk_native_runtime_observation_links'] = all(links)
    checks['three_distinct_tool_feedback_turns'] = len({d['turn'] for d in dispatches}) >= 3
    initial = load(G0 / 'inputs/manifest.json')
    source = load(G0 / 'inputs' / initial['source_file'])
    final = G0 / 'final-workspace'
    target = load(final / initial['target_file'])
    report = load(final / initial['report_file'])
    expected = expected_target(source)
    services = expected['services_by_id']
    checks['independently_correct_nested_target'] = canonical(target) == canonical(expected)
    checks['service_id_order'] = list(target['services_by_id']) == list(services)
    checks['source_inputs_preserved'] = all(digest(final / name) == sha for name, sha in protocol['input_hashes'].items())
    wanted_report = {'batch_id': source['batch_id'], 'source_sha256': digest(final / initial['source_file']), 'target_sha256': digest(final / initial['target_file']), 'active_count': len(services)}
    checks['report_from_actual_bytes'] = canonical(report) == canonical(wanted_report)
    mutations = [d for d in dispatches if any(delta['path'] in {initial['target_file'], initial['report_file']} for delta in d['fs_delta'])]
    successful = [d for d in dispatches if d['raw_result']['exit_code'] == 0 and not d['error'] and not d['raw_result']['error']]
    first, last = min(d['turn'] for d in mutations), max(d['turn'] for d in mutations)
    checks['complete_observation_before_any_output_mutation'] = any(d['turn'] < first and not d['fs_delta'] and complete_observation(d['raw_result']['stdout'], {'manifest': initial, 'source': source}) for d in successful)
    checks['successful_nontrivial_write'] = any(d['fs_delta'] for d in successful)
    observed = {'target': target, 'report': report, 'batch_id': source['batch_id'], 'target_sha256': wanted_report['target_sha256'], 'active_count': len(target['services_by_id'])}
    checks['fresh_complete_readback_after_all_mutations'] = any(d['turn'] > last and not d['fs_delta'] and complete_observation(d['raw_result']['stdout'], observed) for d in successful)
    details.update(first_output_mutation_turn=first, last_output_mutation_turn=last, actual_service_order=list(target['services_by_id']), expected_service_order=list(services), hard_quota_types={k: type(v['quota']['hard']).__name__ for k,v in target['services_by_id'].items()}, target_sha256=wanted_report['target_sha256'])
    usage = [r for r in native if r['type'] == 'usage_accounting']
    result = load(G0 / 'result.json')
    counts = {'requests': len(requests), 'wire_responses': len(wire), 'sdk_responses': len(sdk), 'tool_calls': len(dispatches), 'tool_exit_zero': sum(d['raw_result']['exit_code'] == 0 for d in dispatches), 'prompt_tokens': sum(r['usage']['input_tokens'] for r in usage), 'completion_tokens': sum(r['usage']['output_tokens'] for r in usage)}
    checks['usage_complete_and_reconciled'] = len(usage) == len(requests) == result['modelcall_count'] and counts['prompt_tokens'] == result['tokens']['full_prompt_tokens'] and counts['completion_tokens'] == result['tokens']['output_tokens']
    candidate_path = RUN / 'candidate_manifest.json'
    candidate_marker = load(candidate_path) if candidate_path.exists() else None
    checks['candidate_pool_unfrozen'] = manifest['candidate_pool_frozen'] is False and (
        candidate_marker is None or (candidate_marker.get('candidate_pool_frozen') is False
                                    and candidate_marker.get('selection_performed') is False
                                    and candidate_marker.get('candidates') == []))
    checks['no_officeval_originals_saved'] = no_original_executions(RUN / 'originals')
    # Audit integrity is distinct from whether the agent met its qualification contract.
    contract_keys = {'independently_correct_nested_target', 'service_id_order',
                     'complete_observation_before_any_output_mutation',
                     'fresh_complete_readback_after_all_mutations', 'successful_nontrivial_write'}
    evidence_checks = {k: v for k, v in checks.items() if k not in contract_keys}
    protocol_audit = rows(G0 / 'agent/protocol_audit.jsonl')
    guards = [r for r in protocol_audit if r['type'] == 'generation_guard']
    evidence_checks['every_request_service_guard_verified'] = (
        [g['turn'] for g in guards] == [r['turn'] for r in requests]
        and all(g['status'] == 'verified' and g['serving_observation'] == manifest['serving_observation']
                and g['system_contract_sha256'] == manifest['system_contract_sha256'] for g in guards))
    evidence_checks['all_dispatch_snapshot_deltas_recomputed'] = all(
        snapshot_delta(load(G0 / 'tool-dispatch' / f'call-{i:03d}' / 'before.json'),
                       load(G0 / 'tool-dispatch' / f'call-{i:03d}' / 'after.json')) == d['fs_delta']
        for i, d in enumerate(dispatches))
    snapshots = [(load(G0 / 'tool-dispatch' / f'call-{i:03d}' / 'before.json'),
                  load(G0 / 'tool-dispatch' / f'call-{i:03d}' / 'after.json'))
                 for i in range(len(dispatches))]
    evidence_checks['snapshots_form_contiguous_chain'] = all(
        snapshots[i][1] == snapshots[i + 1][0] for i in range(len(snapshots) - 1))
    final_entries = snapshot_entries(snapshots[-1][1])
    evidence_checks['final_files_match_last_snapshot'] = (
        set(final_entries) == {str(p.relative_to(final)) for p in final.rglob('*')}
        and all(digest(final / name) == item['sha256'] and (final / name).stat().st_size == item['size_bytes']
                for name, item in final_entries.items() if item['kind'] == 'file'))
    evidence_checks['oracle_hash_and_public_derivation'] = (
        digest(G0 / 'private/expected.json') == protocol['expected_sha256']
        and canonical(load(G0 / 'private/expected.json')) == canonical(expected))
    boundaries = [load(G0 / name) for name in ('boundary_before.json', 'boundary_after.json')]
    evidence_checks['saved_isolated_boundaries_and_oracle_exclusion'] = all(
        b['qualified'] and b['effective_config_ok'] and not b['inspect']['Mounts']
        and b['inspect']['HostConfig']['NetworkMode'] == 'none'
        and b['inspect']['HostConfig']['ReadonlyRootfs']
        and b['inspect']['Image'] == definition['agent_image']
        for b in boundaries)
    evidence_checks['only_public_task_and_inputs_supplied'] = (
        requests[0]['messages'] == [{'role': 'system', 'content': definition['system_prompt']},
                                   {'role': 'user', 'content': protocol['task_prompt']}]
        and set(snapshot_entries(snapshots[0][0])) == set(protocol['input_hashes']))
    runtime_tools = [r for r in runtime if r['kind'] == 'model_python']
    evidence_checks['runtime_dispatch_order_exact'] = len(runtime_tools) == len(dispatches) and all(
        r['code_sha256'] == d['code_sha256'] and all(r[key] == d['raw_result'][key]
            for key in ('exit_code', 'error', 'stdout', 'stderr', 'truncated', 'stdout_bytes', 'stderr_bytes'))
        for r, d in zip(runtime_tools, dispatches))
    wire_requests = {t: json.loads(base64.b64decode(w['request_body_base64'])) for t, w in wire.items()}
    evidence_checks['wire_sdk_request_messages_and_parameters'] = all(
        wire_requests[t]['messages'] == sdk[t]['request']['messages']
        and all(wire_requests[t][key] == sdk[t]['request'][key]
                for key in ('model', 'max_tokens', 'seed', 'temperature', 'parallel_tool_calls'))
        and wire_requests[t]['tools'] == sdk[t]['request']['tools']
        for t in wire)
    evidence_checks['every_dispatch_feedback_delivered_next_request'] = all(
        any(m.get('role') == 'tool' and m.get('tool_call_id') == d['tool_call_id']
            and m['content'] == next(e['delivered_content'] for e in events
                if e['type'] == 'tool_result' and e['tool_call_id'] == d['tool_call_id'])
            for m in wire_requests[d['turn'] + 1]['messages']) for d in dispatches)
    evidence_checks['one_response_per_turn_no_retries'] = (
        [r['turn'] for r in requests] == list(range(len(requests)))
        and all(sum(r['type'] == kind and r['turn'] == t for r in native) == 1
                for t in range(len(requests)) for kind in ('request', 'wire_response', 'provider_call', 'response', 'usage_accounting'))
        and all(r['config']['max_retries'] == 0 and not r['config'].get('fallback_models') for r in requests)
        and all(w['status_code'] == 200 and sdk[t]['error'] is None for t, w in wire.items()))
    evidence_checks['per_turn_usage_matches_http_and_sdk'] = all(
        u['usage']['input_tokens'] == wire[u['turn']]['response']['usage']['prompt_tokens']
        and u['usage']['output_tokens'] == wire[u['turn']]['response']['usage']['completion_tokens']
        and all(wire[u['turn']]['response']['usage'][key] == sdk[u['turn']]['response']['usage'][key]
                for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'))
        and wire[u['turn']]['response']['usage']['prompt_tokens_details']['cached_tokens'] ==
                sdk[u['turn']]['response']['usage']['prompt_tokens_details']['cached_tokens']
        for u in usage)
    evidence_checks['freeze_precedes_first_response'] = (
        datetime.fromisoformat(freeze['timestamp_utc']).timestamp() < wire[0]['response']['created'])
    raw_calls = [c for w in wire.values() for c in w['response']['choices'][0]['message'].get('tool_calls', [])]
    rejected = [c for c in raw_calls if c['id'] not in {d['tool_call_id'] for d in dispatches}]
    evidence_checks['all_native_calls_accounted_including_rejection'] = (
        len(raw_calls) == len(dispatches) + 1 and len(rejected) == 1
        and rejected[0]['function']['name'] not in {s['function']['name'] for s in definition['tool_schemas']}
        and result['protocol_status'] == 'protocol_invalid' and result['stopped_reason'] == 'invalid_protocol')
    parsed = [r for r in protocol_audit if r['type'] == 'parser_output']
    evidence_checks['parser_rejection_matches_raw_unknown_name'] = (
        len(parsed) == len(requests) and parsed[-1]['status'] == 'protocol_invalid'
        and parsed[-1]['reason'] == 'unknown tool name: ' + rejected[0]['function']['name']
        and parsed[-1]['dispatch_call_ids'] == [] and parsed[-1]['wire_captured'] and parsed[-1]['sdk_captured'])
    pre_turns = [d['turn'] for d in successful if d['turn'] < first and not d['fs_delta']
                 and complete_observation(d['raw_result']['stdout'], {'manifest': initial, 'source': source})]
    post_turns = [d['turn'] for d in successful if d['turn'] > last and not d['fs_delta']
                  and complete_observation(d['raw_result']['stdout'], observed)]
    target_writes = [d for d in successful if any(x['path'] == initial['target_file'] and x['after'] for x in d['fs_delta'])]
    recomputed_grade = {
        'valid_native_protocol': result['protocol_status'] == 'valid',
        'normal_termination': result['stopped_reason'] == 'assistant_finished',
        'metered_requests': not result['token_usage_unknown'],
        'separate_tool_turns': checks['three_distinct_tool_feedback_turns'],
        'inputs_preserved': checks['source_inputs_preserved'],
        'correct_nested_migration': checks['independently_correct_nested_target'] and checks['service_id_order'],
        'correct_report': checks['report_from_actual_bytes'],
        'successful_observation_before_write': bool(pre_turns),
        'successful_nontrivial_write': bool(target_writes) and checks['independently_correct_nested_target'] and checks['service_id_order'],
        'successful_fresh_readback': bool(post_turns),
        'supported_final_boundary': boundaries[-1]['qualified']}
    recorded_grade, receipt, gate = [load(G0 / name) for name in ('independent_grade.json', 'qualification.json', 'gate.json')]
    evidence_checks['independent_grade_exactly_reproduced'] = recomputed_grade == recorded_grade['checks'] and all(recomputed_grade.values()) == recorded_grade['passed']
    evidence_checks['grade_and_trace_receipt_hashes'] = (
        digest(G0 / 'independent_grade.json') == receipt['independent_grade_sha256']
        and digest(G0 / 'agent/model_calls.jsonl') == receipt['model_trace_sha256'])
    evidence_checks['blocked_gate_consistent_and_candidates_not_admitted'] = (
        not recorded_grade['passed'] and receipt['status'] == gate['status'].lower() == 'blocked_system'
        and receipt['candidate_execution_admitted'] is gate['candidate_execution_admitted'] is False)
    counts.update(native_calls_emitted=len(raw_calls), native_calls_rejected=len(rejected),
                  verified_service_guard_count=len(guards),
                  tool_exit_nonzero=len(dispatches) - counts['tool_exit_zero'],
                  successful_artifact_write_calls=len([d for d in successful if d['fs_delta']]),
                  all_contract_qualified_modification_calls=len(target_writes) if all(recomputed_grade.values()) else 0,
                  total_tokens=counts['prompt_tokens'] + counts['completion_tokens'],
                  officeval_candidate_executions=0)
    details.update(pre_observation_qualified_turns=pre_turns, post_observation_qualified_turns=post_turns,
                   rejected_native_calls=rejected,
                   successful_artifact_writes=[{'turn': d['turn'], 'tool_call_id': d['tool_call_id'], 'fs_delta': d['fs_delta']} for d in successful if d['fs_delta']],
                   receipt_semantics='nontrivial_file_change_verified is assigned whole-grade passed; false does not mean no file writes',
                   failed_grade_predicates=[k for k, v in recomputed_grade.items() if not v],
                   identity_source_count=len(sources), model_fingerprint_files=len(manifest['model_fingerprint']['files']))
    return {'schema_version': 'r12-g0-independent-verification-v1',
            'audit_status': 'PASS' if all(evidence_checks.values()) else 'FAIL',
            'qualification_status': 'PASS' if all(recomputed_grade.values()) else 'BLOCKED_SYSTEM',
            'evidence_checks': evidence_checks, 'recomputed_grade': recomputed_grade,
            'contract_checks': {k: checks[k] for k in sorted(contract_keys)}, 'counts': counts, 'details': details,
            'scope': 'one synthetic independent development task; actual local model execution; not OfficeVal or E2-S',
            'coverage_limit': 'Offline saved evidence audit. Disk source/model/parser hashes checked; no new process/network/container inspection or new model calls. Boundary records are observations, not proof of escape resistance. GPU time and monetary cost are unavailable; tokens are reconciled. No semantic claim about unlogged actions or scientific efficacy.'}

if __name__ == '__main__':
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
