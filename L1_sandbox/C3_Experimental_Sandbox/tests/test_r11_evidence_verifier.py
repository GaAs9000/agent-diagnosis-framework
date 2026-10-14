"""Adversarial saved-record verification; no model, Docker, or network calls."""
import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('r11_verifier', Path(__file__).parents[1] / 'scripts/verify_r11_evidence.py')
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_lines(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(x) + '\n' for x in values))


@pytest.fixture
def evidence(tmp_path):
    fp = {'sha256': 'model-pinned', 'tokenizer': {'sha256': 'tokenizer-pinned'}}
    identity = {'model_fingerprint': fp['sha256'], 'tokenizer_fingerprint': fp['tokenizer']}
    identity_hash = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    cfg = {'temperature': 0, 'max_tokens': 2048, 'seed': 2026, 'parallel_tool_calls': False,
           'max_retries': 0, 'extra_body': {'top_k': 1, 'min_p': 0}}
    # First call writes a file; the next billed response announces tool_calls
    # with empty payload, so it must be excluded from task outcomes, not costs.
    code = "open('result.txt','w').write('answer')"
    call = {'id': 'fresh-call-1', 'function': 'workspace_python', 'arguments': {'code': code}, 'type': 'function'}
    native_msg = {'role': 'assistant', 'content': '', 'tool_calls': [call]}
    raw = {'stdout': '', 'stderr': '', 'stdout_bytes': 0, 'stderr_bytes': 0, 'exit_code': 0, 'error': None, 'truncated': False}
    event = {'type': 'tool_result', 'event_index': 0, 'turn': 0, 'tool_call_id': call['id'],
             'function': call['function'], 'arguments': call['arguments'], 'raw_result': raw,
             'delivered_content': json.dumps({**raw, 'context_output_truncated': False}), 'error': None}
    prefix = [{'role': 'system', 'content': 'Complete task'}, {'role': 'user', 'content': 'Write result'}]
    end_prefix = prefix + [native_msg, {'role': 'tool', 'tool_call_id': call['id'], 'function': call['function'], 'content': event['delivered_content']}]
    records, parser = [], []
    for turn in (0, 1):
        wirecall = {'id': call['id'], 'type': 'function', 'function': {'name': call['function'], 'arguments': json.dumps(call['arguments'])}}
        wire_msg = {'role': 'assistant', 'content': None, 'tool_calls': [wirecall] if turn == 0 else []}
        wire = {'id': 'response-' + str(turn), 'model': 'default_model', 'choices': [{'finish_reason': 'tool_calls', 'message': wire_msg}],
                'usage': {'prompt_tokens': 100 + turn, 'completion_tokens': 20, 'total_tokens': 120 + turn}}
        usage = {'input_tokens': 100 + turn, 'output_tokens': 20, 'total_tokens': 120 + turn,
                 'input_tokens_cache_read': 0, 'input_tokens_cache_write': 0}
        sent = {k: cfg[k] for k in ('temperature', 'max_tokens', 'seed', 'parallel_tool_calls')}
        sent['messages'] = prefix if turn == 0 else end_prefix
        output = {'response_id': wire['id'], 'choices': [{'message': native_msg if turn == 0 else {'role': 'assistant', 'content': '', 'tool_calls': []}}], 'usage': usage}
        records += [{'type': 'request', 'turn': turn, 'messages': prefix if turn == 0 else end_prefix,
                     'config': cfg, 'identity_context': identity, 'identity_context_sha256': identity_hash, 'evidence_kind': 'native_provider'},
                    {'type': 'wire_response', 'turn': turn, 'request_body_base64': base64.b64encode(json.dumps(sent).encode()).decode(),
                     'response_body_base64': base64.b64encode(json.dumps(wire).encode()).decode(),
                     'response_sha256': hashlib.sha256(json.dumps(wire).encode()).hexdigest(), 'response': wire,
                     'identity_context_sha256': identity_hash, 'evidence_kind': 'native_provider'},
                    {'type': 'provider_call', 'turn': turn, 'request': sent, 'response': wire},
                    {'type': 'response', 'turn': turn, 'model_output': output, 'usage': usage},
                    {'type': 'usage_accounting', 'turn': turn, 'status': 'observed', 'usage': usage}]
        parser += [{'type': 'parser_output', 'turn': turn, 'status': 'tool_calls' if turn == 0 else 'protocol_invalid',
                    'native_model_output': output, 'dispatch_call_ids': [call['id']] if turn == 0 else []}]
        if turn == 0:
            parser += [{'type': 'tool_dispatch', 'turn': 0, 'tool_call_id': call['id'], 'function': call['function'], 'arguments': call['arguments']}]
    before = {'root_mode': 448, 'entries': [], 'total_bytes': 0}
    after = {'root_mode': 448, 'entries': [{'path': 'result.txt', 'kind': 'file', 'mode': 420, 'data': base64.b64encode(b'answer').decode()}], 'total_bytes': 6}
    dispatch_dir = tmp_path / 'tool-dispatch/call-000'
    save(dispatch_dir / 'before.json', before)
    save(dispatch_dir / 'after.json', after)
    dispatch = {**event, 'before_path': str(dispatch_dir / 'before.json'), 'after_path': str(dispatch_dir / 'after.json'),
                'code_sha256': hashlib.sha256(code.encode()).hexdigest(), 'fs_delta': v.delta(before, after)}
    save(dispatch_dir / 'dispatch.json', dispatch)
    write_lines(tmp_path / 'actor-runtime/runtime_events.jsonl', [{'kind': 'model_python', 'code_sha256': dispatch['code_sha256'], **raw}])
    write_lines(tmp_path / 'agent/model_calls.jsonl', records)
    write_lines(tmp_path / 'agent/protocol_audit.jsonl', parser)
    write_lines(tmp_path / 'agent/events.jsonl', [event])
    result = {'modelcall_count': 2, 'events': [event], 'messages': end_prefix,
              'tokens': {'input_tokens': 201, 'output_tokens': 40, 'total_tokens': 241, 'full_prompt_tokens': 201,
                         'input_tokens_cache_read': 0, 'input_tokens_cache_write': 0},
              'token_usage_unknown': False, 'unknown_usage_turns': [], 'protocol_status': 'protocol_invalid'}
    return tmp_path, fp, result


def test_multiturn_invalid_final_retains_measured_cost(evidence):
    folder, fp, result = evidence
    audit = v.verify_trace(folder, fp, result)
    assert (audit['requests'], audit['dispatches'], audit['protocol_invalid_turns']) == (2, 1, 1)
    assert audit['tokens']['total_tokens'] == 241


@pytest.mark.parametrize('attack,reason', [
    ('wire_hash', 'wire response hash'), ('fingerprint', 'fingerprint'),
    ('tool_id', 'tool binding'), ('code', 'Docker code'),
    ('cost', 'cost differs'), ('delta', 'filesystem delta'),
    ('drop_invalid_cost', 'cost accounting'), ('empty_payload_dispatch', 'invalid/text'),
    ('response_identity', 'response identity'), ('prefix', 'prefix'),
])
def test_adversarial_trace_tampering(evidence, attack, reason):
    folder, fp, result = evidence
    records = v.lines(folder / 'agent/model_calls.jsonl')
    parser = v.lines(folder / 'agent/protocol_audit.jsonl')
    if attack == 'wire_hash':
        next(r for r in records if r['type'] == 'wire_response')['response_sha256'] = '0' * 64
    elif attack == 'fingerprint':
        next(r for r in records if r['type'] == 'request')['identity_context']['tokenizer_fingerprint'] = None
    elif attack == 'tool_id':
        next(r for r in records if r['type'] == 'response')['model_output']['choices'][0]['message']['tool_calls'][0]['id'] = 'old-suffix-call'
    elif attack == 'code':
        docker = v.lines(folder / 'actor-runtime/runtime_events.jsonl')
        docker[0]['code_sha256'] = hashlib.sha256(b'prewritten fake code').hexdigest()
        write_lines(folder / 'actor-runtime/runtime_events.jsonl', docker)
    elif attack == 'cost':
        next(r for r in records if r['type'] == 'usage_accounting')['usage']['input_tokens'] = 0
    elif attack == 'delta':
        p = folder / 'tool-dispatch/call-000/dispatch.json'
        dispatch = v.read(p); dispatch['fs_delta'] = []; save(p, dispatch)
    elif attack == 'drop_invalid_cost':
        records = [r for r in records if not (r['turn'] == 1 and r['type'] == 'usage_accounting')]
    elif attack == 'empty_payload_dispatch':
        parser.append({'type': 'tool_dispatch', 'turn': 1, 'tool_call_id': 'fabricated', 'function': 'workspace_python', 'arguments': {'code': 'print(1)'}})
    elif attack == 'response_identity':
        next(r for r in records if r['type'] == 'response')['model_output']['response_id'] = 'other-response'
    elif attack == 'prefix':
        next(r for r in records if r['type'] == 'request' and r['turn'] == 1)['messages'][0]['content'] = 'Changed system prompt'
    write_lines(folder / 'agent/model_calls.jsonl', records)
    write_lines(folder / 'agent/protocol_audit.jsonl', parser)
    with pytest.raises(ValueError, match=reason):
        v.verify_trace(folder, fp, result)


def test_summary_cannot_discard_invalid_request_cost(evidence):
    folder, fp, result = evidence
    result['tokens']['output_tokens'] = 20
    with pytest.raises(ValueError, match='token cost total'):
        v.verify_trace(folder, fp, result)


def test_journal_reorder_and_candidate_hash_tamper(tmp_path):
    event = {'sequence': 0, 'kind': 'candidate_pool_frozen', 'previous_event_sha256': '0' * 64,
             'candidate_order_sha256': 'f' * 64}
    event['event_sha256'] = v.digest(event)
    write_lines(tmp_path / 'run_journal.jsonl', [event])
    assert v.journal(tmp_path) == [event]
    for key, value in [('sequence', 2), ('candidate_order_sha256', 'a' * 64), ('previous_event_sha256', 'b' * 64)]:
        tampered = copy.deepcopy(event); tampered[key] = value
        write_lines(tmp_path / 'run_journal.jsonl', [tampered])
        with pytest.raises(ValueError, match='journal chain'):
            v.journal(tmp_path)


def test_snapshot_bytes_and_duplicate_path_rejected():
    snap = {'root_mode': 448, 'entries': [{'path': 'a', 'kind': 'file', 'mode': 420, 'data': 'eA=='}], 'total_bytes': 1}
    assert v.inventory(snap)['a']['size_bytes'] == 1
    snap['total_bytes'] = 2
    with pytest.raises(ValueError, match='byte total'):
        v.inventory(snap)
    snap['entries'].append(snap['entries'][0])
    with pytest.raises(ValueError, match='duplicate'):
        v.inventory(snap)


def test_planned_order_cannot_hide_reorder_skip_or_parallel_attempts():
    plan = {'candidate_count': 2, 'candidates': [{'order': 0, 'task_id': 'first'}, {'order': 1, 'task_id': 'second'}]}
    events = [{'kind': 'candidate_pool_frozen', 'candidate_order_sha256': 'sealed'}]
    for candidate in plan['candidates']:
        events += [{'kind': 'original_started', 'candidate_order_sha256': 'sealed', **candidate},
                   {'kind': 'original_finished', **candidate}]
    assert v.verify_planned_order(plan, events)[0] == [(0, 'first'), (1, 'second')]
    attacks = [events[:3], [events[0], events[3], events[4], events[1], events[2]],
               [events[0], events[1], events[3], events[2], events[4]]]
    for attacked in attacks:
        with pytest.raises(ValueError, match='denominator|serial'):
            v.verify_planned_order(plan, attacked)
    tampered = copy.deepcopy(plan); tampered['candidates'][1]['order'] = 7
    with pytest.raises(ValueError, match='planned order'):
        v.verify_planned_order(tampered, events)
    tampered_events = copy.deepcopy(events); tampered_events[1]['candidate_order_sha256'] = 'different-pool'
    with pytest.raises(ValueError, match='different candidate pool'):
        v.verify_planned_order(plan, tampered_events)


def test_content_rehashed_receipt_still_rejects_forged_hmac(tmp_path):
    checkpoint = 'a' * 64
    private = tmp_path / 'private'
    (private / 'receipts').mkdir(parents=True)
    (private / 'authority.key').write_bytes(b'x' * 32)
    receipt = {'payload': {'checkpoint_identity': checkpoint}, 'signature': '0' * 64}
    raw = v.encoded(receipt); receipt_id = hashlib.sha256(raw).hexdigest()
    (private / 'receipts' / (receipt_id + '.json')).write_bytes(raw)
    (private / (checkpoint + '.receipt')).write_text(receipt_id)
    with pytest.raises(ValueError, match='HMAC'):
        v.verify_admission(tmp_path, checkpoint)


@pytest.mark.parametrize('mutation', ['bytes', 'mode', 'extra_file'])
def test_checkpoint_tree_tampering_rejected(tmp_path, mutation):
    workspace = tmp_path / 'source'
    workspace.mkdir(mode=0o700)
    target = workspace / 'document.docx'
    target.write_bytes(b'frozen-document')
    target.chmod(0o644)
    entries = v.tree_inventory(workspace)
    manifest = {'schema_version': 1, 'scope': 'filesystem', 'entries': entries}
    snapshot_id = v.digest(manifest)
    store = tmp_path / 'store'
    (store / 'snapshots').mkdir(parents=True)
    (store / 'objects').mkdir()
    (store / 'snapshots' / (snapshot_id + '.json')).write_bytes(v.encoded(manifest))
    (store / 'objects' / v.sha(target)).write_bytes(target.read_bytes())
    record = {'snapshot_id': snapshot_id, 'messages': [], 'runtime': {},
              'message_identity': v.digest([]), 'runtime_identity': v.digest({})}
    record_id = v.digest(record)
    record_path, messages_path = tmp_path / 'checkpoint.json', tmp_path / 'messages.json'
    save(record_path, record); save(messages_path, [])
    reference = {'checkpoint_record_path': str(record_path), 'checkpoint_id': record_id,
                 'messages_path': str(messages_path), 'workspace': str(workspace)}
    assert v.verify_checkpoint_files(reference, {'replay_store': str(store)}) == entries
    if mutation == 'bytes':
        target.write_bytes(b'modified-document')
    elif mutation == 'mode':
        target.chmod(0o600)
    else:
        (workspace / 'extra.txt').write_text('injected')
    with pytest.raises(ValueError, match='bytes/paths/types/modes'):
        v.verify_checkpoint_files(reference, {'replay_store': str(store)})
