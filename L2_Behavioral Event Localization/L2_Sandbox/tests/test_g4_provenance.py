"""Offline constructive audit tests; run on the AML CPU verification worker."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/audit_g4_provenance.py'
spec = importlib.util.spec_from_file_location('audit_g4_provenance', SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
EPISODE = 'ep_' + 'a' * 32


def fixture():
    config = {'agent': {'system_prompt': 'Public execution contract', 'harness': 'pi'},
              'model': {'adapter': 'openai-compatible', 'model': 'fixture-only',
                        'parameters': {'temperature': 1, 'max_tokens': 8192}, 'supports_images': True},
              'recording': {'observation_limit_chars': 4}}
    public = {'task_id': 'officeval_003', 'prompt': 'Use the public image to make a document',
              'assets': [{'name': 'public.png', 'sha256': 'public-image'}]}
    model = audit.backend(config)
    conversation = audit.initial_conversation(config, public)
    store = {}

    def save(name, value):
        store[name] = json.dumps(value, sort_keys=True).encode()
        return {'path': name}

    events = []
    for turn in range(3):
        rid = f'ep:r{turn:06d}'
        events.append({'kind': 'model.request', 'event_id': str(turn), 'data': {
            'request_id': rid, 'body': save(f'request{turn}', model.request(conversation, audit.tools_for_config(config)))}})
        message = {'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': f'call{turn}', 'type': 'function', 'function': {'name': 'bash', 'arguments': '{"command":"ls"}'}}]}
        response = {'choices': [{'message': message, 'finish_reason': 'tool_calls'}], 'usage': {'total_tokens': 30}}
        events.append({'kind': 'model.response', 'data': {'request_id': rid, 'body': save(f'response{turn}', response)}})
        reply = model.parse(store[f'response{turn}'], rid)
        call = reply.calls[0]
        observation = audit.format_pi_bash_observation(b'abcdef', b'', 0, None, True, False)
        raw = {'call_id': call.call_id, 'observation': observation,
               'provenance': {'container_exit_code': 0, 'stop_reason': None, 'output_complete': True, 'oom_killed': False}}
        store[f'stdout{turn}'] = b'abcdef'
        store[f'stderr{turn}'] = b''
        events.append({'kind': 'tool.result', 'data': {'call_id': call.call_id, 'result': save(f'result{turn}', raw),
                       'stdout': {'path': f'stdout{turn}'}, 'stderr': {'path': f'stderr{turn}'}}})
        delivered = copy.deepcopy(observation)
        delivered['stdout'] = 'abcd'
        delivered['output'] = 'abcd'
        delivered['observation_truncation'] = {'fields': ['stdout', 'output'], 'limit_chars': 4, 'raw_retained': True}
        conversation = model.accept(conversation, reply, [(call, audit.Result(delivered))])
    return config, public, events, store


def reconstruct(config, public, events, store, **kwargs):
    return audit.reconstruct_requests(config, public, events, lambda ref: store[ref['path']], **kwargs)


def test_exact_full_requests_and_actual_observation_truncation():
    config, public, events, store = fixture()
    rows = reconstruct(config, public, events, store)
    assert [r['status'] for r in rows] == ['verified'] * 3
    assert rows[0]['continuation_sources'][0]['delivered_observation_sha256']


@pytest.mark.parametrize('field', ['historical_message', 'tools', 'temperature', 'extra'])
def test_changes_to_any_full_body_field_are_detected(field):
    config, public, events, store = fixture()
    body = json.loads(store['request1'])
    if field == 'historical_message':
        body['messages'][0]['content'] += ' hidden grader instruction'
    elif field == 'tools':
        body['tools'][0]['function']['description'] += ' injected rule'
    elif field == 'temperature':
        body['temperature'] = 0
    else:
        body['unexpected'] = 'injection'
    store['request1'] = json.dumps(body).encode()
    rows = reconstruct(config, public, events, store)
    assert [r['status'] for r in rows] == ['verified', 'mismatch', 'unverified']


def test_missing_actual_tool_result_does_not_trust_next_saved_history():
    config, public, events, store = fixture()
    del store['result0']
    rows = reconstruct(config, public, events, store)
    assert rows[0]['status'] == 'verified'
    assert all(r['status'] == 'unverified' for r in rows[1:])


def test_plain_c0_parent_is_reconstructed_from_public_inputs():
    config, public, events, store = fixture()
    cp = {'budget_used': {'model_requests': 0}, 'conversation': audit.initial_conversation(config, public)}
    assert reconstruct(config, public, events, store, parent_checkpoint=cp)[0]['status'] == 'verified'
    cp['conversation']['messages'].append({'role': 'user', 'content': 'hidden input'})
    assert reconstruct(config, public, events, store, parent_checkpoint=cp)[0]['status'] == 'mismatch'


def test_non_c0_and_intervention_histories_are_not_silently_accepted():
    config, public, events, store = fixture()
    cp = {'budget_used': {'model_requests': 2}, 'conversation': {}}
    assert reconstruct(config, public, events, store, parent_checkpoint=cp)[0]['status'] == 'unverified'
    config['agent']['intervention'] = {'message': 'not plain'}
    assert reconstruct(config, public, events, store)[0]['status'] == 'unverified'


def test_history_rewrite_is_replayed_instead_of_prefix_comparison(monkeypatch):
    config, public, events, store = fixture()
    def rewrite(conversation, *_):
        conv = copy.deepcopy(conversation)
        conv['messages'] = conv['messages'][:2] + [{'role': 'user', 'content': 'derived compact summary'}]
        return conv, {'summary_sha256': audit.digest(conv['messages'][-1])}
    monkeypatch.setattr(audit, '_maybe_compact_pi_conversation', rewrite)
    model = audit.backend(config)
    initial = audit.initial_conversation(config, public)
    rewritten, receipt = rewrite(initial)
    for turn in (1, 2):
        store[f'request{turn}'] = json.dumps(model.request(rewritten, audit.tools_for_config(config))).encode()
    ordered = []
    for event in events:
        if event['kind'] == 'model.request' and event['data']['request_id'] != 'ep:r000000':
            ordered.append({'kind': 'session.compacted', 'data': copy.deepcopy(receipt)})
        ordered.append(event)
    events = ordered
    assert all(r['status'] == 'verified' for r in reconstruct(config, public, events, store))
    next(e for e in events if e['kind'] == 'session.compacted')['data'] = {'summary_sha256': 'forged'}
    assert reconstruct(config, public, events, store)[-1]['status'] == 'mismatch'


def test_public_or_independently_generated_rubric_constants_are_not_leak_verdicts():
    config, public, events, store = fixture()
    public['prompt'] = 'Public image contains 0.15 cm and IFERROR'
    response = json.loads(store['response0'])
    response['choices'][0]['message']['content'] = 'I independently propose hidden-looking constant 50%'
    store['response0'] = json.dumps(response).encode()
    for turn in range(3):
        body = json.loads(store[f'request{turn}'])
        body['messages'][1] = audit.initial_conversation(config, public)['messages'][1]
        if turn:
            body['messages'][2]['content'] = response['choices'][0]['message']['content']
        store[f'request{turn}'] = json.dumps(body).encode()
    assert all(r['status'] == 'verified' for r in reconstruct(config, public, events, store))


def test_mount_claims_and_image_or_controller_attestation_are_separate():
    events = [{'kind': 'runtime.container.created', 'data': {'container_id': 'c', 'mounts': [
        {'target': '/workspace', 'source_name': '.l1-work-' + EPISODE, 'read_only': False}]}}]
    graph = audit.runtime_graph(events, {'image_id': 'sha256:image', 'python_environment': {'grader_worker_sha256': 'generic'}}, episode_id=EPISODE)
    assert graph['recorded_actor_mounts']['status'] == 'verified'
    assert graph['image_contents']['status'] == 'unverified'
    assert graph['historical_controller']['status'] == 'unverified'
    assert graph['external_input_exclusion']['status'] == 'unverified'
    events[0]['data']['mounts'].append({'target': '/verifiers', 'source_name': 'source', 'read_only': True})
    assert audit.runtime_graph(events, {}, episode_id=EPISODE)['recorded_actor_mounts']['status'] == 'mismatch'


def test_different_episode_workspace_mount_is_rejected():
    events = [{'kind': 'runtime.container.created', 'data': {'container_id': 'c', 'mounts': [
        {'target': '/workspace', 'source_name': '.l1-work-ep_other', 'read_only': False}]}}]
    assert audit.runtime_graph(events, {}, episode_id=EPISODE)['recorded_actor_mounts']['status'] == 'mismatch'


@pytest.mark.parametrize('data', [
    {'container_id': 'actor'},
    {'container_id': 'actor', 'mounts': []},
    {'container_id': 'actor', 'mounts': None},
    {'container_id': 'actor', 'mounts': [{}]},
    {'mounts': [{'target': '/workspace', 'source_name': '.l1-work-' + EPISODE, 'read_only': False}]},
])
def test_missing_actor_mount_evidence_is_unverified_not_a_leak(data):
    graph = audit.runtime_graph([{'kind': 'runtime.container.created', 'data': data}], {}, episode_id=EPISODE)
    assert graph['recorded_actor_mounts']['status'] == 'unverified'
    assert graph['recorded_actor_mounts']['unexpected'] == []


@pytest.mark.parametrize('episode', [None, '', 'ep_wrong', '../ep_bad'])
def test_missing_or_invalid_episode_identity_cannot_verify_mounts(episode):
    events = [{'kind': 'runtime.container.created', 'data': {'container_id': 'actor', 'mounts': [
        {'target': '/workspace', 'source_name': '.l1-work-' + EPISODE, 'read_only': False}]}}]
    assert audit.runtime_graph(events, {}, episode_id=episode)['recorded_actor_mounts']['status'] == 'unverified'


def test_only_positive_image_probe_identity_allows_empty_mounts():
    events = [{'kind': 'runtime.container.created', 'data': {'container_id': 'probe', 'mounts': []}},
              {'kind': 'runtime.container.created', 'data': {'container_id': 'actor', 'mounts': [
                  {'target': '/workspace', 'source_name': '.l1-work-' + EPISODE, 'read_only': False}]}}]
    assert audit.runtime_graph(events, {}, episode_id=EPISODE)['recorded_actor_mounts']['status'] == 'unverified'
    runtime = {'environment_probe': {'container_id': 'probe'}}
    result = audit.runtime_graph(events, runtime, episode_id=EPISODE)['recorded_actor_mounts']
    assert result['status'] == 'verified'
    assert result['identified_probe_count'] == 1
    assert result['actor_container_count'] == 1
    events[0]['data']['mounts'] = [{'target': '/private', 'source_name': 'grader', 'read_only': True}]
    assert audit.runtime_graph(events, runtime, episode_id=EPISODE)['recorded_actor_mounts']['status'] == 'mismatch'


def test_probe_only_is_insufficient_actor_evidence():
    events = [{'kind': 'runtime.container.created', 'data': {'container_id': 'probe', 'mounts': []}}]
    runtime = {'environment_probe': {'container_id': 'probe'}}
    assert audit.runtime_graph(events, runtime, episode_id=EPISODE)['recorded_actor_mounts']['status'] == 'unverified'


def test_image_read_serialization_restores_production_order_without_changing_values():
    observation = {'output': 'Read image file [image/png]', 'stdout': 'Read image file [image/png]', 'stderr': '',
                   'path': 'input/image.png', 'mime_type': 'image/png', 'size_bytes': 12,
                   'exit_code': 0, 'stop_reason': None, 'output_complete': True}
    canonicalized = json.loads(json.dumps(observation, sort_keys=True))
    call = type('Call', (), {'name': 'read'})()
    restored = audit.reconstruct_observation(call, {'observation': canonicalized}, {}, lambda _: None)
    assert restored == observation
    assert json.dumps(restored) == json.dumps(observation)
    assert json.dumps(restored) != json.dumps(canonicalized)


def test_changed_bash_stream_is_rejected_even_when_saved_observation_matches_request():
    config, public, events, store = fixture()
    store['stdout0'] = b'poisoned output'
    rows = reconstruct(config, public, events, store)
    assert rows[1]['status'] == 'unverified'
    assert 'differs from actual streams' in rows[1]['reason']


def test_no_transport_is_constructed(monkeypatch):
    monkeypatch.setattr(audit.OpenAICompatibleBackend, '__init__', lambda *_: pytest.fail('HTTP backend constructed'))
    config, public, events, store = fixture()
    assert reconstruct(config, public, events, store)[0]['status'] == 'verified'
