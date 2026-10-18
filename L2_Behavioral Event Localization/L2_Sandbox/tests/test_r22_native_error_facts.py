"""Native validation evidence, reconstructed without model/validator execution."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

from l2_sandbox.evidence.l1 import L1Package
from l2_sandbox.facts.build import build_facts

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


g4 = script('audit_g4_provenance')
g1 = script('audit_g1_mechanical')
ERROR = 'Validation failed for tool "edit":\n  - unexpected property "unknown"'
NATIVE = {'content': [{'type': 'text', 'text': ERROR}], 'details': {}, 'isError': True}


def native_episode(arguments=None):
    config = {'agent': {'system_prompt': 'Public contract', 'harness': 'pi',
                        'validation_policy': 'pi-native-0.73.1'},
              'model': {'adapter': 'openai-compatible', 'model': 'fixture',
                        'parameters': {}, 'supports_images': True},
              'recording': {'observation_limit_chars': 4}}
    public = {'task_id': 'officeval_003', 'prompt': 'Edit public document', 'assets': []}
    model = g4.backend(config)
    conversation = g4.initial_conversation(config, public)
    store, events = {}, []

    def save(name, obj):
        store[name] = json.dumps(obj, sort_keys=True).encode()
        return {'path': name}

    rid = 'ep:r000000'
    events.append({'kind': 'model.request', 'event_id': 'r0', 'data': {
        'request_id': rid, 'body': save('request0', model.request(conversation, g4.tools_for_config(config)))}})
    response = {'choices': [{'message': {'role': 'assistant', 'content': None, 'tool_calls': [
        {'id': 'native-call', 'type': 'function', 'function': {'name': 'edit', 'arguments': json.dumps(arguments)}}]},
        'finish_reason': 'tool_calls'}], 'usage': {'completion_tokens': 19, 'total_tokens': 33}}
    events.append({'kind': 'model.response', 'data': {'request_id': rid, 'body': save('response0', response)}})
    events.append({'kind': 'model.parsed', 'data': {'request_id': rid, 'finish_reason': 'tool_calls',
                                                 'usage': response['usage']}})
    reply = model.parse(store['response0'], rid)
    call = reply.calls[0]
    events.append({'kind': 'tool.call', 'data': {'call_id': call.call_id, 'request_id': rid,
                                               'name': 'edit', 'arguments': arguments}})
    validation = {'call_id': call.call_id, 'ok': False, 'raw_arguments': arguments,
                  'validated_arguments': None, 'native_result': copy.deepcopy(NATIVE),
                  'provenance': {'validator': 'pi-native'}, 'telemetry': {}}
    events.append({'kind': 'tool.validation', 'data': validation})
    provenance = {'pi_native_validation_error': True, 'validation': validation['provenance'], 'quiescent': True}
    events.append({'kind': 'tool.result', 'data': {'call_id': call.call_id,
                  'result': save('result0', {'call_id': call.call_id, 'observation': NATIVE, 'provenance': provenance})}})
    conversation = model.accept(conversation, reply, [(call, g4.Result(copy.deepcopy(NATIVE), provenance=provenance))])
    events.append({'kind': 'model.request', 'event_id': 'r1', 'data': {'request_id': 'ep:r000001',
                  'body': save('request1', model.request(conversation, g4.tools_for_config(config)))}})
    return config, public, events, store, conversation


@pytest.mark.parametrize('arguments', [None, [], {'unknown': 'retained'}])
def test_native_full_request_reconstruction_preserves_exact_error_wire(arguments):
    config, public, events, store, _ = native_episode(arguments)
    body = json.loads(store['request1'])
    assert body['messages'][-1]['content'] == ERROR
    rows = g4.reconstruct_requests(config, public, events, lambda ref: store[ref['path']])
    assert [r['status'] for r in rows] == ['verified', 'verified']
    body['messages'][-1]['content'] = json.dumps(NATIVE)
    store['request1'] = json.dumps(body).encode()
    assert g4.reconstruct_requests(config, public, events, lambda ref: store[ref['path']])[1]['status'] == 'mismatch'


@pytest.mark.parametrize('change', ['marker', 'shape', 'validation'])
def test_native_error_cannot_be_inferred_from_an_unbound_observation(change):
    config, public, events, store, _ = native_episode()
    raw = json.loads(store['result0'])
    if change == 'marker':
        raw['provenance'].pop('pi_native_validation_error')
    elif change == 'shape':
        raw['observation']['extra'] = 'must be rejected'
    else:
        next(e for e in events if e['kind'] == 'tool.validation')['data']['native_result'] = {}
    store['result0'] = json.dumps(raw).encode()
    rows = g4.reconstruct_requests(config, public, events, lambda ref: store[ref['path']])
    assert rows[1]['status'] == 'unverified'


def test_non_c0_consumption_is_bound_to_saved_request_and_response_bytes():
    config, public, events, store, _ = native_episode()
    cp = {'budget_used': {'model_requests': 3}, 'conversation': g4.initial_conversation(config, public)}
    events[0]['data'].update(source='saved_response', http_attempts=0)
    events.insert(0, {'kind': 'response.bound', 'data': {'request': {'path': 'source_request'},
                                                     'response': {'path': 'source_response'}}})
    store['source_request'] = store['request0']
    store['source_response'] = store['response0']
    rows = g4.reconstruct_requests(config, public, events, lambda ref: store[ref['path']], parent_checkpoint=cp)
    assert [r['status'] for r in rows] == ['verified', 'verified']
    store['source_response'] = b'{}'
    assert g4.reconstruct_requests(config, public, events, lambda ref: store[ref['path']], parent_checkpoint=cp)[0]['status'] == 'mismatch'


def test_g1_reconstructs_checkpoint_native_error_message_and_rejects_changed_wire(tmp_path):
    config, _, _, store, conversation = native_episode()
    target = tmp_path / 'request.json'
    target.write_bytes(store['request1'])
    row = {'checkpoint': {'conversation': conversation}, 'next_request': {'data': {
        'request_id': 'ep:r000001', 'body': {'path': 'request.json', 'sha256': g1.file_digest(target)}}}}
    assert g1.reconstruct_request(tmp_path, config, row)['following_request_id'] == 'ep:r000001'
    row['checkpoint']['conversation']['messages'][-1]['content'] += '!'
    with pytest.raises(ValueError, match='reconstruction mismatch'):
        g1.reconstruct_request(tmp_path, config, row)


@pytest.mark.parametrize('arguments', [None, [], {'unknown': 'exact'}])
@pytest.mark.parametrize('next_state', ['valid', 'invalid', 'changed_tool', 'no_response'])
def test_facts_preserve_raw_arguments_and_observational_next_response(tmp_path, arguments, next_state):
    _, _, events, store, _ = native_episode(arguments)
    if next_state != 'no_response':
        events.extend([
            {'kind': 'model.parsed', 'data': {'request_id': 'ep:r000001', 'finish_reason': 'tool_calls',
                                            'usage': {'completion_tokens': 7}}},
            {'kind': 'tool.call', 'data': {'call_id': 'retry', 'request_id': 'ep:r000001',
                                          'name': 'read' if next_state == 'changed_tool' else 'edit',
                                          'arguments': {'path': 'public.docx'}}},
            {'kind': 'tool.validation', 'data': {'call_id': 'retry', 'ok': next_state != 'invalid'}},
        ])
    (tmp_path / 'manifest.json').write_text(json.dumps({'episode_id': 'fixture', 'recording_mode': 'test'}))
    (tmp_path / 'trajectory.jsonl').write_text('\n'.join(json.dumps(e) for e in events))
    for name, value in store.items():
        (tmp_path / name).write_bytes(value)
    facts = build_facts(L1Package(tmp_path))
    value = next(f['value'] for f in facts if f['type'] == 'tool_validation_error')
    assert value['raw_arguments'] == arguments
    assert next(f['value']['arguments'] for f in facts if f['type'] == 'tool_requested') == arguments
    assert value['native_result'] == NATIVE
    assert value['error_text'] == ERROR
    assert value['finish_reason'] == 'tool_calls'
    assert value['completion_tokens'] == 19
    assert value['next_call_corrected'] == {'valid': True, 'invalid': False}.get(next_state)
    assert value['next_call_changed'] == (True if next_state in ('valid', 'invalid') else None)
    if next_state == 'no_response':
        assert value['next_response'] is None
    else:
        assert value['next_response']['usage']['completion_tokens'] == 7
