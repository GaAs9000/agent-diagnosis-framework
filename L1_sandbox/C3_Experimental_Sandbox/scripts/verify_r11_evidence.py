"""Read-only multi-turn R1.1 evidence audit. Never creates runtimes or calls a model.

Only a requested output file is written, exclusively. Frozen run code is checked
against source_snapshot before its pure selection/normalization helpers are used.
HMAC verifies trusted-controller consistency, not independence from that controller.
"""
from __future__ import annotations
import argparse
import base64
from collections import Counter
import hashlib
import hmac
import json
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lines(path):
    return [json.loads(line) for line in Path(path).read_bytes().splitlines()] if Path(path).exists() else []


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def journal(root):
    records = lines(root / 'run_journal.jsonl')
    previous = '0' * 64
    for index, record in enumerate(records):
        body = {k: v for k, v in record.items() if k != 'event_sha256'}
        require(record['sequence'] == index and record['previous_event_sha256'] == previous
                and digest(body) == record['event_sha256'], 'journal chain/order/hash mismatch')
        previous = record['event_sha256']
    return records


def verify_planned_order(plan, events):
    expected = [(c['order'], c['task_id']) for c in plan['candidates']]
    require(plan['candidate_count'] == len(expected) <= 12 and
            [c['order'] for c in plan['candidates']] == list(range(len(expected))) and
            len({task for _, task in expected}) == len(expected), 'planned order/count mismatch')
    starts = [e for e in events if e['kind'] == 'original_started']
    finishes = [e for e in events if e['kind'] == 'original_finished']
    require([(e['order'], e['task_id']) for e in starts] == expected ==
            [(e['order'], e['task_id']) for e in finishes], 'incomplete or reordered original denominator')
    require([(e['kind'], e['task_id']) for e in events if e['kind'] in ('original_started', 'original_finished')] ==
            [(kind, task) for _, task in expected for kind in ('original_started', 'original_finished')],
            'original execution not serial')
    require(all(e.get('candidate_order_sha256') == events[0]['candidate_order_sha256'] for e in starts),
            'original start used a different candidate pool')
    return expected, finishes


def inventory(snapshot):
    require(snapshot is not None, 'snapshot unavailable')
    result = {'.': {'kind': 'directory', 'mode': snapshot['root_mode']}}
    byte_count = 0
    for entry in snapshot['entries']:
        path = entry['path']
        require(path not in result and not Path(path).is_absolute() and '..' not in Path(path).parts,
                'snapshot duplicate or unsafe path')
        item = {key: entry[key] for key in ('kind', 'mode')}
        require(item['kind'] in ('file', 'directory'), 'snapshot unsupported file type')
        if item['kind'] == 'file':
            data = base64.b64decode(entry['data'], validate=True)
            byte_count += len(data)
            item.update(sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))
        result[path] = item
    require(byte_count == snapshot['total_bytes'], 'snapshot byte total mismatch')
    return result


def delta(before, after):
    before, after = inventory(before), inventory(after)
    return [{'path': p, 'before': before.get(p), 'after': after.get(p)}
            for p in sorted(before.keys() | after.keys()) if before.get(p) != after.get(p)]


def wire_calls(body):
    return body['choices'][0]['message'].get('tool_calls') or []


def call_binding(call, native=False):
    return (call['id'], call['function'] if native else call['function']['name'],
            call['arguments'] if native else json.loads(call['function']['arguments']))


def verify_trace(folder, fingerprint, result=None, *, synthetic=False, dispatch_snapshots=True, legacy_accounting=False, agent_subdir="agent"):
    """Audit every request including invalid final turns; no success-only filtering."""
    folder = Path(folder)
    records = lines(folder / agent_subdir / 'model_calls.jsonl')
    parser = lines(folder / agent_subdir / 'protocol_audit.jsonl')
    events = lines(folder / agent_subdir / 'events.jsonl')
    requests = [r for r in records if r['type'] == 'request']
    require(requests and [r['turn'] for r in requests] == list(range(len(requests))), 'request turn sequence mismatch')
    by_turn = {}
    for record in records:
        key = (record.get('turn'), record['type'])
        by_turn.setdefault(key, []).append(record)
    totals = Counter()
    unknown = []
    dispatch_count = 0
    previous_prefix = None
    for request in requests:
        turn = request['turn']
        identity = request['identity_context']
        require(identity.get('model_fingerprint') == fingerprint['sha256'] and
                identity.get('tokenizer_fingerprint') == fingerprint['tokenizer'], 'request identity fingerprint mismatch/null')
        # NativeAgent uses noncompact sorted Unicode JSON for this particular hash.
        identity_sha = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()
        require(request['identity_context_sha256'] == identity_sha, 'request identity context hash mismatch')
        kind = 'synthetic_protocol_fixture' if synthetic else 'native_provider'
        require(request['evidence_kind'] == kind, 'synthetic/native evidence relabelled')
        cfg = request['config']
        require(cfg['temperature'] == 0 and cfg['max_tokens'] == 2048 and cfg['max_retries'] == 0
                and cfg['parallel_tool_calls'] is False and cfg['extra_body'] == {'top_k': 1, 'min_p': 0}, 'request sampling drift')
        if previous_prefix is not None:
            require(request['messages'] == previous_prefix, 'conversation prefix continuity mismatch')
        wires = by_turn.get((turn, 'wire_response'), [])
        responses = by_turn.get((turn, 'response'), [])
        providers = by_turn.get((turn, 'provider_call'), [])
        accounts = by_turn.get((turn, 'usage_accounting'), [])
        decisions = [r for r in parser if r['type'] == 'parser_output' and r.get('turn') == turn]
        dispatches = [r for r in parser if r['type'] == 'tool_dispatch' and r.get('turn') == turn]
        observations = [r for r in events if r['type'] == 'tool_result' and r.get('turn') == turn]
        if not accounts and legacy_accounting and len(responses) == 1:
            accounts = [{'status': 'observed', 'usage': responses[0]['usage'], 'source': 'legacy_ModelOutput_reconciliation'}]
        require(len(decisions) == 1 and len(accounts) == 1, 'missing/duplicate parser or cost accounting')
        decision = decisions[0]
        require(len(wires) <= 1 and len(responses) <= 1, 'duplicate wire/model output')
        if wires:
            wire = wires[0]
            body_bytes = base64.b64decode(wire['response_body_base64'], validate=True)
            body = json.loads(body_bytes)
            from c3_sandbox.r1.tool_protocol import validate_wire_response
            wire_decision = validate_wire_response(body)
            if decision['status'] in ('tool_calls', 'assistant_finished'):
                require(wire_decision.status == decision['status'], 'parser admitted invalid wire protocol')
            elif decision['status'] == 'protocol_invalid':
                require(wire_decision.status == 'protocol_invalid' or bool(responses), 'parser invalid status not supported by evidence')
            sent = json.loads(base64.b64decode(wire['request_body_base64'], validate=True))
            require(hashlib.sha256(body_bytes).hexdigest() == wire['response_sha256'] and body == wire['response'], 'wire response hash/structure mismatch')
            require(wire['identity_context_sha256'] == identity_sha and wire['evidence_kind'] == kind, 'wire identity mismatch')
            require(all(sent.get(k) == cfg[k] for k in ('temperature', 'max_tokens', 'seed', 'parallel_tool_calls')), 'wire sampling differs from request')
            if providers:
                sdk = providers[-1]
                require(sdk['request']['messages'] == sent['messages'], 'SDK/wire request mismatch')
                sdk_body = sdk['response']
                if sdk.get('error'):
                    require(decision['status'] == 'protocol_invalid' and by_turn.get((turn, 'error')), 'SDK conversion error silently accepted')
                elif sdk_body is not None:
                    require(sdk_body['id'] == body['id'] and sdk_body['model'] == body['model']
                            and sdk_body['choices'][0]['finish_reason'] == body['choices'][0]['finish_reason'], 'SDK response identity mismatch')
                    if wire_decision.status != 'protocol_invalid':
                        require([call_binding(c) for c in wire_calls(sdk_body)] == [call_binding(c) for c in wire_calls(body)], 'SDK/wire tool binding mismatch')
            if responses:
                require(providers, 'ModelOutput lacks raw SDK record')
                output = responses[0]['model_output']
                require(output['response_id'] == body['id'], 'ModelOutput response identity mismatch')
                if decision['status'] in ('tool_calls', 'assistant_finished'):
                    native = output['choices'][0]['message'].get('tool_calls') or []
                    require([call_binding(c, True) for c in native] == [call_binding(c) for c in wire_calls(body)], 'ModelOutput/wire tool binding mismatch')
                    require(decision['native_model_output'] == output, 'parser ModelOutput mismatch')
            account = accounts[0]
            if account['status'] == 'observed':
                raw = body.get('usage') or {}
                usage = account['usage']
                full = sum(usage.get(k) or 0 for k in ('input_tokens', 'input_tokens_cache_read', 'input_tokens_cache_write'))
                require((full, usage['output_tokens'], usage['total_tokens']) ==
                        (raw.get('prompt_tokens'), raw.get('completion_tokens'), raw.get('total_tokens')),
                        'observed token cost differs from raw wire usage')
                require(full + usage['output_tokens'] == usage['total_tokens'], 'raw token sum mismatch')
        elif responses or dispatches:
            raise ValueError('model output/dispatch without raw wire response')
        account = accounts[0]
        if account['status'] == 'unknown':
            unknown.append(turn)
        else:
            require(account['status'] == 'observed', 'invalid cost accounting status')
            u = account['usage']
            for k in ('input_tokens', 'output_tokens', 'total_tokens', 'input_tokens_cache_read', 'input_tokens_cache_write'):
                totals[k] += u.get(k) or 0
            totals['full_prompt_tokens'] += sum(u.get(k) or 0 for k in ('input_tokens', 'input_tokens_cache_read', 'input_tokens_cache_write'))
        if decision['status'] != 'tool_calls':
            require(not dispatches and not observations, 'invalid/text response dispatched tool')
        else:
            require(wires and responses, 'dispatch lacks native wire and ModelOutput')
            expected = [call_binding(c) for c in wire_calls(body)]
            require(all(name == 'workspace_python' and type(args) is dict and set(args) == {'code'} and
                        isinstance(args['code'], str) for _, name, args in expected), 'parser admitted invalid tool payload/schema')
            actual = [(r['tool_call_id'], r['function'], r['arguments']) for r in dispatches]
            require(actual == expected and [r['tool_call_id'] for r in observations] == [c[0] for c in expected], 'dispatch/observation ID or arguments mismatch')
            require(decision['dispatch_call_ids'] == [c[0] for c in expected], 'parser dispatch identity mismatch')
        if not synthetic:
            from c3_sandbox.r1.qualification import delivered_tool_result
            for observation in observations:
                if observation['error'] is None:
                    visible = delivered_tool_result(observation['function'], observation['arguments'], observation['raw_result'])
                    require(json.loads(observation['delivered_content']) == json.loads(visible), 'tool observation differs from actual transformed raw output')
        dispatch_count += len(dispatches)
        if responses and decision['status'] in ('tool_calls', 'assistant_finished'):
            previous_prefix = request['messages'] + [responses[0]['model_output']['choices'][0]['message']]
            # Obtain exact serialized Tool message from next request/final result;
            # its ID/content is cross-checked against independent tool events.
            next_messages = requests[turn + 1]['messages'] if turn + 1 < len(requests) else (result or {}).get('messages')
            if next_messages is not None:
                require(next_messages[:len(previous_prefix)] == previous_prefix, 'assistant prefix differs from native output')
                tools = next_messages[len(previous_prefix):]
                require(len(tools) == len(observations), 'tool message count mismatch')
                for m, e in zip(tools, observations):
                    require(m['role'] == 'tool' and m['tool_call_id'] == e['tool_call_id'] and m['function'] == e['function']
                            and m['content'] == e['delivered_content'], 'tool result message binding mismatch')
                previous_prefix = next_messages
    if result is not None:
        require(result['modelcall_count'] == len(requests), 'request denominator mismatch')
        require(result['events'] == events, 'agent result/event transcript mismatch')
        for k, n in totals.items():
            require(result['tokens'][k] == n, 'agent token cost total mismatch: ' + k)
        require(result.get('token_usage_unknown', False) == bool(unknown) and result.get('unknown_usage_turns', []) == unknown,
                'unknown cost label mismatch')
        if any(p.get('status') == 'protocol_invalid' for p in parser):
            require(result['protocol_status'] == 'protocol_invalid', 'protocol-invalid turn excluded from summary')
    if dispatch_snapshots:
        docker = [r for r in lines(folder / 'actor-runtime/runtime_events.jsonl') if r.get('kind') == 'model_python']
        saved = sorted((folder / 'tool-dispatch').glob('call-*/dispatch.json'))
        require(len(saved) == dispatch_count == len(docker), 'Docker/snapshot/native dispatch count mismatch')
        native_events = [e for e in events if e['type'] == 'tool_result']
        for path, actual, event in zip(saved, docker, native_events):
            record = read(path)
            code_hash = hashlib.sha256(event['arguments']['code'].encode()).hexdigest()
            require(record['tool_call_id'] == event['tool_call_id'] and record['arguments'] == event['arguments']
                    and record['code_sha256'] == actual['code_sha256'] == code_hash, 'actual Docker code hash/ID mismatch')
            require(record['raw_result'] == event['raw_result'], 'dispatch raw result mismatch')
            raw = event['raw_result']
            for k in ('stdout', 'stderr', 'stdout_bytes', 'stderr_bytes', 'exit_code', 'error', 'truncated'):
                require(actual[k] == raw[k], 'Docker output differs from raw tool result')
            for channel in ('stdout', 'stderr'):
                require(type(raw[channel + '_bytes']) is int and raw[channel + '_bytes'] >= len(raw[channel].encode())
                        or '\ufffd' in raw[channel], 'output byte count impossible')
            before, after = read(record['before_path']), read(record['after_path'])
            if before is None or after is None:
                require(record['fs_delta'] is None and record['snapshot_ineligible_reasons'], 'missing snapshot silently accepted')
            else:
                require(record['fs_delta'] == delta(before, after), 'filesystem delta mismatch')
    return {'legacy_usage_reconciled': legacy_accounting, 'requests': len(requests), 'dispatches': dispatch_count, 'tokens': dict(totals),
            'unknown_usage_turns': unknown, 'protocol_invalid_turns': sum(p.get('status') == 'protocol_invalid' for p in parser)}


def tree_inventory(root):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), 'restored tree missing/unsafe')
    result = {'.': {'kind': 'directory', 'mode': stat.S_IMODE(root.stat().st_mode)}}
    for path in sorted(root.rglob('*')):
        info = path.lstat()
        require(not path.is_symlink() and (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)), 'restored special file/symlink')
        item = {'kind': 'directory' if path.is_dir() else 'file', 'mode': stat.S_IMODE(info.st_mode)}
        if item['kind'] == 'file':
            require(info.st_nlink == 1, 'restored hardlinked file')
            item['sha256'] = sha(path)
        result[path.relative_to(root).as_posix()] = item
    return result


def verify_checkpoint_files(reference, baseline):
    record = read(reference['checkpoint_record_path'])
    require(digest(record) == reference['checkpoint_id'] and digest(record['messages']) == record['message_identity'] and
            digest(record['runtime']) == record['runtime_identity'], 'checkpoint component identity mismatch')
    require(read(reference['messages_path']) == record['messages'], 'checkpoint message file mismatch')
    store = Path(baseline['replay_store'])
    snapshot_path = store / 'snapshots' / (record['snapshot_id'] + '.json')
    require(sha(snapshot_path) == record['snapshot_id'], 'checkpoint snapshot manifest hash mismatch')
    entries = read(snapshot_path)['entries']
    for entry in entries.values():
        if entry['kind'] == 'file':
            require(sha(store / 'objects' / entry['sha256']) == entry['sha256'], 'checkpoint snapshot blob changed')
    require(tree_inventory(reference['workspace']) == entries, 'checkpoint source file bytes/paths/types/modes changed')
    return entries


def verify_admission(folder, checkpoint):
    folder = Path(folder)
    private = folder / 'private'
    receipt_id = (private / (checkpoint + '.receipt')).read_text()
    path = private / 'receipts' / (receipt_id + '.json')
    require(sha(path) == receipt_id, 'admission receipt content hash mismatch')
    receipt = read(path)
    expected = hmac.new((private / 'authority.key').read_bytes(), encoded(receipt['payload']), hashlib.sha256).hexdigest()
    require(hmac.compare_digest(expected, receipt['signature']), 'admission HMAC mismatch')
    payload = receipt['payload']
    require(payload['checkpoint_identity'] == digest(payload['checkpoint_record']) == checkpoint, 'admission checkpoint identity mismatch')
    for evidence in payload['evidence']:
        require(sha(evidence['path']) == evidence['sha256'], 'admission evidence/qualification trace changed')
    from c3_sandbox.r1.admission import _validate
    _validate(payload['checkpoint_record'], payload['source_audit'], payload['restore_audit'],
              payload['restoration_report'], payload['model_qualification'])
    return {'checkpoint_identity': checkpoint, 'receipt_identity': receipt_id, 'HMAC_verified': True}


def verify_grade(folder, grade, manifest, task):
    from c3_sandbox.r1.officeval import normalize_official
    folder = Path(folder)
    for name in ('process', 'stdout', 'stderr'):
        suffix = '.json' if name == 'process' else '.txt'
        require(sha(folder / (name + suffix)) == grade[name + '_sha256'], 'grader raw artifact hash mismatch')
    require(read(folder / 'process.json') == grade['raw_process'], 'grader process differs from saved raw')
    require((folder / 'stdout.txt').read_text() == grade['raw_process']['stdout'], 'grader stdout mismatch')
    if grade['official_raw'] is not None:
        normalized = normalize_official(grade['official_raw'], task, manifest['tasks'][task])
        require(all(grade.get(k) == v for k, v in normalized.items()), 'official rule inventory/completion diluted')
        parsed = json.loads(grade['raw_process']['stdout'])
        require(parsed == grade['grader_json'] and parsed['official_raw'] == grade['official_raw'], 'official raw score detached from grader process')


def verify_branch(folder, baseline, plan, manifest):
    folder = Path(folder)
    agent = read(folder / 'agent_result.json')
    trace = verify_trace(folder, plan['model_fingerprint'], agent)
    original_ids = {e['tool_call_id'] for e in baseline['agent_result']['events'] if e['type'] == 'tool_result'}
    new_events = [e for e in agent['events'] if e['type'] == 'tool_result']
    require(not original_ids.intersection(e['tool_call_id'] for e in new_events), 'branch reused original suffix tool IDs')
    spec, result = read(folder / 'attempt.json'), read(folder / 'result.json')
    reference = next(cp for cp in baseline['checkpoints'] if cp['checkpoint_id'] == spec['checkpoint_id'])
    require(tree_inventory(folder / 'workspace') == verify_checkpoint_files(reference, baseline), 'branch staging differs from checkpoint bytes/paths/types/modes')
    require(spec['is_synthetic'] is False and spec['checkpoint_id'] == result['checkpoint_identity'], 'branch identity mismatch')
    for key, filename in [('agent_result', 'agent_result.json'), ('final_grade', 'final_grade.json'),
                          ('base_prefix', 'base_prefix.json'), ('injected_prefix', 'injected_prefix.json')]:
        require(result['evidence_hashes'][key] == sha(folder / filename), 'branch evidence hash mismatch: ' + key)
    require(read(folder / 'base_prefix.json') == next(read(cp['messages_path']) for cp in baseline['checkpoints']
                if cp['checkpoint_id'] == spec['checkpoint_id']), 'restored prefix differs from source')
    for file, key in [('verification.json', 'valid'), ('seed_verification.json', 'verified')]:
        require(read(folder / file)[key] is True, 'branch restoration/seed verification failed')
    require(digest(read(folder / 'verification.json')) == result['evidence_hashes']['restoration'], 'restoration evidence identity mismatch')
    from c3_sandbox.r1.admission import _audit
    _audit(read(folder / 'boundary_audit.json'), plan['agent_image'])
    grade = read(folder / 'final_grade.json')
    verify_grade(folder / 'grade', grade, manifest, baseline['task_id'])
    from c3_sandbox.r1.office_run import _outcome
    flags = _outcome(baseline['initial_grade'], grade, agent)
    require(result['outcome'] == {k: flags[k] for k in ('execution_valid', 'target_resolved', 'regression_free', 'task_completed')}, 'branch score flags diluted')
    require(result['tool_dispatch_count'] == trace['dispatches'], 'new tool call count mismatch')
    require(result['tool_dispatch_identity'] == digest([read(p) for p in sorted((folder / 'tool-dispatch').glob('call-*/dispatch.json'))]), 'branch dispatch identity mismatch')
    cost = read(folder / 'cost.json')
    require(cost['gpu_seconds'] is None, 'unmeasured GPU cost presented as measured')
    for key, token in [('input_tokens', 'full_prompt_tokens'), ('output_tokens', 'output_tokens')]:
        require(cost[key] == (None if agent['token_usage_unknown'] else agent['tokens'][token]), 'branch measured token cost mismatch')
    return {**trace, 'cost': cost, 'outcome': result['outcome'], 'branch_key': result['branch_key']}


def verify_run(root):
    root = Path(root).resolve()
    report = {'schema': 'independent-r11-evidence-v1', 'run_root': str(root), 'read_only': True,
              'verifier_source_sha256': sha(__file__), 'verifier_is_postrun_native_source': False,
              'gates': {}, 'findings': [], 'originals': [], 'D6_D8': 'CLOSED',
              'limitations': ['Saved observations under a trusted controller; no new Docker/model execution.',
                              'Source-family independence and method effectiveness are not established.']}
    def gate(name, fallback, action):
        try:
            detail = action()
            report['gates'][name] = {'status': 'PASS', 'detail': detail}
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
            report['gates'][name] = {'status': fallback, 'reason': str(error)}
            report['findings'].append(name + ': ' + str(error))
    plan = read(root / 'candidate_order.json')
    events = []
    manifest = read(plan['manifest_path'])
    results = []
    def freeze():
        nonlocal events
        events = journal(root)
        require(events and events[0]['kind'] == 'candidate_pool_frozen' and
                events[0]['candidate_order_sha256'] == sha(root / 'candidate_order.json'), 'candidate file seal mismatch')
        require(digest(plan['implementation_lock']) == plan['implementation_identity'], 'frozen implementation identity mismatch')
        for relative, expected in plan['implementation_lock'].items():
            require(sha(root / 'source_snapshot' / relative) == expected and sha(ROOT / relative) == expected,
                    'frozen source snapshot/live code mismatch: ' + relative)
        require(sha(plan['manifest_path']) == plan['manifest_sha256'], 'manifest hash mismatch')
        for relative, item in manifest['files'].items():
            require(sha(Path(manifest['bundle_root']) / relative) == item['sha256'], 'OfficeVal file hash mismatch')
        for relative, key in [('source_audit-v2/source_review.json','source_review_sha256'),
                ('tool_protocol_audit/gate-v2.json','G0_audit_sha256'),
                ('grader-dependency-preflight/report.json','dependency_audit_sha256'),
                ('server_observation.json','serving_observation_sha256')]:
            require(sha(root / relative) == plan[key], 'frozen prerequisite changed: ' + relative)
        require(plan['D6_D8_open'] is False and plan['sampling']['seed'] == 2026 and plan['sampling']['temperature'] == 0,
                'research claim or frozen sampling drift')
        return {'source_files': len(plan['implementation_lock']), 'candidate_order_sha256': sha(root / 'candidate_order.json')}
    gate('freeze_integrity', 'INVALID', freeze)
    if report['gates']['freeze_integrity']['status'] != 'PASS':
        for name, state in [('G0','BLOCKED'), ('G1','INVALID'), ('G2','INCOMPLETE'), ('G3','INCOMPLETE'), ('G4','INVALID')]:
            report['gates'][name] = {'status': state, 'reason': 'frozen integrity prerequisite failed'}
        report['status'] = 'INVALID'
        return report
    def g0():
        receipt = read(plan['model_qualification_path'])
        require(sha(plan['model_qualification_path']) == plan['model_qualification_sha256'] and
                sha(plan['model_qualification_trace']) == plan['model_qualification_trace_sha256'] == receipt['model_trace_sha256'], 'native qualification seal mismatch')
        audit = read(root / 'tool_protocol_audit/gate-v2.json')
        cases = audit['controlled_cases']
        require(len(cases) == 7 and all(c['passed'] and c['is_synthetic_model_evidence'] for c in cases), 'controlled fixture evidence missing/relabelled')
        require(all(c['actual_docker_dispatches'] == (1 if c['case'] == 'valid_native' else 0) for c in cases), 'invalid fixture dispatched')
        fixture_root = root / 'tool_protocol_audit/controlled_fixtures-v2'
        fixture_details = {}
        for case in cases:
            case_folder = fixture_root / case['case']
            if case['case'] == 'mismatched_result_id':
                require(not [r for r in lines(case_folder / 'model_calls.jsonl') if r['type'] == 'request'], 'mismatched prefix reached model')
                require(any(r.get('status') == 'protocol_invalid' for r in lines(case_folder / 'protocol_audit.jsonl')), 'mismatched prefix did not reject')
                continue
            fixture_result = read(case_folder / 'result.json')
            require(fixture_result['is_synthetic'] is True, 'fixture synthetic label lost')
            detail = verify_trace(case_folder, {'sha256': 'synthetic-scripted-provider', 'tokenizer': 'synthetic-fixed-counter'},
                                  fixture_result, synthetic=True, dispatch_snapshots=False, agent_subdir='')
            require(detail['dispatches'] == case['actual_docker_dispatches'], 'controlled fixture dispatch count mismatch')
            fixture_details[case['case']] = detail
        fixture_python = [r for r in lines(fixture_root / 'runtime/runtime_events.jsonl') if r.get('kind') == 'model_python']
        require(len(fixture_python) == 1 and fixture_python[0]['exit_code'] == 0 and fixture_python[0]['error'] is None, 'fixture actual dispatch missing')
        qual_folder = Path(plan['model_qualification_path']).parent
        trace = verify_trace(qual_folder, plan['model_fingerprint'], read(qual_folder / 'result.json'), dispatch_snapshots=False, legacy_accounting=True)
        docker = [r for r in lines(qual_folder / 'runtime/runtime_events.jsonl') if r.get('kind') == 'model_python']
        observations = [r for r in lines(qual_folder / 'agent/events.jsonl') if r['type'] == 'tool_result']
        require(len(docker) == len(observations) == 1 and docker[0]['code_sha256'] == hashlib.sha256(observations[0]['arguments']['code'].encode()).hexdigest(), 'native qualification actual Docker code mismatch')
        require(docker[0]['stdout'].strip() == receipt['nonce'] == receipt['observed_nonce'] == receipt['independent_nonce'], 'native qualification nonce mismatch')
        require(trace['tokens']['full_prompt_tokens'] == receipt['prompt_tokens'] and trace['tokens']['output_tokens'] == receipt['completion_tokens'], 'native qualification measured cost mismatch')
        require(trace['requests'] == audit['genuine_model_calls'] == 2 and trace['dispatches'] == audit['genuine_native_tool_calls'] == 1, 'native qualification count mismatch')
        return {'controlled_synthetic_cases': len(cases), 'controlled_fixture_details': fixture_details, 'actual_native': trace}
    gate('G0','BLOCKED',g0)
    def g1():
        expected, finishes = verify_planned_order(plan, events)
        for c, e in zip(plan['candidates'], finishes):
            folder = root / 'originals' / c['task_id']
            path = folder / ('result.json' if (folder / 'result.json').exists() else 'attempt_failure.json')
            require(sha(path) == e['result_sha256'], 'original result journal hash mismatch')
            r = read(path)
            require(r['task_id'] == c['task_id'] and r['is_synthetic'] is False, 'original identity mismatch')
            task = manifest['tasks'][c['task_id']]
            require(c['asset_hashes'] == {i['name']: i['sha256'] for i in task['input_files']} and
                    c['verifier_sha256'] == task['verifier_sha256'], 'candidate asset/scorer differs from frozen manifest')
            require(c['source_group_id'] == 'officeval-assets-' + digest(c['asset_hashes']) and
                    c['split'] == 'development' and c['excluded_from_future_confirmation'] is True, 'candidate source/split relabelled')
            results.append(r)
            if path.name == 'result.json':
                require(r['seed'] == 2026 and r['runtime']['image_digest'] == plan['agent_image'] and
                        r['runtime']['model_config']['model_fingerprint'] == plan['model_fingerprint']['sha256'], 'original runtime/model/seed drift')
                require(all(q['config']['seed'] == 2026 for q in lines(folder / 'agent/model_calls.jsonl') if q['type'] == 'request'), 'original request seed changed')
                trace = verify_trace(folder, plan['model_fingerprint'], r['agent_result'])
                for phase in ('initial', 'final'):
                    require(r[phase + '_grade']['grader_image'] == plan['grader_image'], 'grader image differs from freeze')
                    verify_grade(folder / (phase + '-grade'), r[phase + '_grade'], manifest, r['task_id'])
                from c3_sandbox.r1.office_run import _outcome
                require(all(r[k] == v for k,v in _outcome(r['initial_grade'], r['final_grade'], r['agent_result']).items()), 'original completion/outcome flags diluted')
                if r['agent_result']['protocol_status'] == 'protocol_invalid':
                    require(r['execution_valid'] is False and r['task_completed'] is None, 'protocol invalid included as task outcome')
            else:
                trace = {'execution_error': r['error_type'], 'cost': 'unknown unless retained trace accounts available'}
            report['originals'].append({'task_id': r['task_id'], 'outcome': r['outcome'], 'task_completed': r['task_completed'], **trace})
        index = read(root / 'original_results_index.json')
        require(index['attempted_count'] == index['outcome_denominator'] == len(expected) and index['tasks'] == [t for _,t in expected], 'reported denominator excludes attempts')
        from c3_sandbox.r1.r11_replay import select_engineering_candidates
        selection = read(root / 'coverage_selection.json')
        require(selection == select_engineering_candidates(results), 'selection differs from first S/T qualified rule')
        completed = [e for e in events if e['kind'] == 'original_pool_finished']
        require(len(completed) == 1 and completed[0]['selection_sha256'] == sha(root / 'coverage_selection.json'), 'selection journal seal mismatch')
        return {'attempts': len(expected), 'tasks': index['tasks'], 'S_candidate': selection['S_candidate'], 'T_candidate': selection['T_candidate']}
    gate('G1','INVALID',g1)
    def coverage(kind):
        require(report['gates']['G1']['status'] == 'PASS', 'complete originals prerequisite unavailable')
        selection = read(root / 'coverage_selection.json')[kind + '_candidate']
        folder = root / ('replay_tool_continuation' if kind == 'T' else 'success_preparation')
        if selection is None:
            gap = read(folder / 'coverage_gap.json')
            require(gap['status'] == 'INCOMPLETE' and gap['reason'], 'missing coverage gap reason')
            if kind == 'S':
                for phase in ('e2_s_exploration','e2_s_confirmation','scheduler'):
                    require(read(root / phase / 'not_run.json')['status'] == 'INCOMPLETE', 'missing S not_run record')
            raise ValueError('no original ' + kind + '_candidate; documented coverage gap')
        baseline = next(r for r in results if r['task_id'] == selection['task_id'])
        snapshot_entries = verify_checkpoint_files(selection['reference'], baseline)
        admission = verify_admission(folder / 'admission', selection['checkpoint_id'])
        require(tree_inventory(folder / 'admission/restored-workspace') == snapshot_entries and
                tree_inventory(folder / 'admission/staging') == snapshot_entries, 'admitted restore bytes/paths/types/modes differ from checkpoint')
        if kind == 'T':
            result = read(folder / 'result.json')
            detail = verify_branch(folder / 'branches' / result['branch_key'], baseline, plan, manifest)
            require(result['original_suffix_replayed'] is False and result['alternate_seed_or_probe_attempted'] is False,
                    'T replayed old suffix or adapted sampling')
            require(result['new_tool_calls'] == detail['dispatches'], 'T count differs from new branch')
            branch_folder = folder / 'branches' / result['branch_key']
            completed = [r for r in read(branch_folder / 'agent_result.json')['events'] if r['type'] == 'tool_result' and
                         isinstance(r.get('raw_result'), dict) and type(r['raw_result'].get('exit_code')) is int and
                         r['raw_result'].get('error') is None]
            require(completed, 'tool_continuation_not_observed: no completed new execution (transport/timeout excluded)')
            require(read(branch_folder / 'attempt.json')['sampling']['seed'] == 2026, 'T seed adapted')
            return {'admission': admission, 'new_continuation': detail}
        require(baseline['task_completed'] is True, 'S did not originate from official complete success')
        branches = []
        for phase in ('exploration','confirmation'):
            index = read(root / ('e2_s_' + phase) / 'index.json')
            require(index['checkpoint_identity'] == selection['checkpoint_id'] and len(index['branches']) == 4, 'S four-arm phase missing')
            for item in index['branches']:
                require(sha(root / item['result_path']) == item['result_sha256'] and sha(root / item['cost_path']) == item['cost_sha256'], 'S index result/cost seal mismatch')
                branches.append(verify_branch((root / item['result_path']).parent, baseline, plan, manifest))
        summary = read(root / 'scheduler/integration_checks.json')
        require(summary['physical_executions'] == 8 and summary['logical_charges'] == 5
                and summary['D6_D8'] == 'CLOSED' and all(c['passed'] for c in summary['checks']), 'S scheduler/cache/claim gate mismatch')
        ledger = read(root / 'scheduler/private/controller_ledger.json')
        require(len(ledger['physical']) == 8 and sum(m['spent'] for m in ledger['methods']) == 5, 'cache erased logical cost')
        require(plan['exploration_seed'] != plan['confirmation_seed'], 'phase seed domains overlap')
        return {'admission': admission, 'branches': branches, 'physical_executions': 8, 'logical_charges': 5}
    gate('G2','INCOMPLETE',lambda: coverage('T'))
    gate('G3','INCOMPLETE',lambda: coverage('S'))
    # Require independent runtime negatives and image audit, supplied after serial runs.
    def g4():
        evidence = read(root / 'verification/g4_runtime_audit.json')
        require(evidence['status'] == 'PASS' and evidence.get('checks') and all(c['passed'] for c in evidence['checks']), 'G4 negative/isolation evidence failed')
        require(evidence.get('agent_image') == plan['agent_image'] and evidence.get('grader_image') == plan['grader_image'], 'G4 image identity mismatch')
        require(evidence.get('evidence_files'), 'G4 raw evidence hashes missing')
        for item in evidence['evidence_files']:
            path = Path(item['path'])
            if not path.is_absolute():
                path = root / path
            require(sha(path) == item['sha256'], 'G4 raw evidence changed')
        return evidence
    gate('G4','INVALID',g4)
    report['original_totals'] = {'attempts': len(report['originals']),
        'requests': sum(r.get('requests', 0) for r in report['originals']),
        'dispatches': sum(r.get('dispatches', 0) for r in report['originals']),
        'tokens_observed_subtotal': {k: sum(r.get('tokens', {}).get(k, 0) for r in report['originals'])
                                   for k in ('full_prompt_tokens', 'output_tokens', 'total_tokens')},
        'outcomes': dict(Counter(r['outcome'] for r in report['originals'])),
        'cost_scope': 'all_requests' if all(not r.get('unknown_usage_turns') and 'tokens' in r for r in report['originals']) else 'observed_subtotal',
        'gpu_seconds': None, 'gpu_cost_label': 'unmeasured'}
    report['status'] = 'VERIFIED' if all(report['gates'][g]['status'] == 'PASS' for g in ('G0','G1','G2','G3','G4')) else 'PARTIAL_VERIFIED'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=ROOT / 'runs/e2-r1-1')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = verify_run(args.run_root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write('\n')
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
