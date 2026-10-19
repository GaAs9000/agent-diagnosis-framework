"""Recorded length-stop feedback, with no live provider or container execution."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from c3_sandbox.l1.core.contracts import ConfigurationError, HTTPResult, ProtocolError
from c3_sandbox.l1.core.orchestrator import run
from c3_sandbox.l1.models.opencode import OpenCodeChatBackend, TRUNCATED_TOOL_CALL_POLICY
from c3_sandbox.l1.storage import canonical, read_json, utc_now
from .test_opencode_protocols import _base_model_cfg
from .test_pi_response_continuation import events, fake_live_components, native_bridge, pi_config
from .test_pi_harness import PiFixtureRuntime


def backend(*, policy=TRUNCATED_TOOL_CALL_POLICY, native=True):
    cfg = _base_model_cfg('opencode-chat')
    if policy is not None:
        cfg['truncated_tool_call_policy'] = policy
    model = OpenCodeChatBackend(cfg)
    model.native_pi_validation = native
    return model


def response(arguments='{"path":"unfinished', finish='length'):
    return {'model':'deepseek-v4.1-flash', 'choices':[{'message':{
        'role':'assistant', 'content':None, 'tool_calls':[{'type':'function',
        'id':'original-id', 'function':{'name':'write', 'arguments':arguments}}]},
        'finish_reason':finish}], 'usage':{'prompt_tokens':12, 'completion_tokens':8}}


@pytest.fixture
def recorded_response():
    default = (Path(__file__).resolve().parents[4] /
        'L2_Behavioral Event Localization/L2_Sandbox/validation/opencode_d1_cross_actor_20261019/'
        'successor_01/execution/branches/'
        '8a32900a898f9ed4a4cea68c1664423442ad757b932c0c9683e20ba1b2ba366e/'
        'raw/model/r000023/response.bin')
    source = Path(os.environ.get('C3_OPENCODE_LENGTH_RESPONSE', default))
    if not source.is_file():
        pytest.skip('Set C3_OPENCODE_LENGTH_RESPONSE to the recorded HTTP 200 length response')
    raw = source.read_bytes()
    doc = json.loads(raw)
    assert doc['choices'][0]['finish_reason'] == 'length'
    assert doc['usage']['completion_tokens'] == 8192
    assert doc['usage']['total_tokens'] == 39446
    return raw


@pytest.mark.parametrize('policy,native', [(None,False),(None,True),(TRUNCATED_TOOL_CALL_POLICY,False)])
def test_feedback_requires_both_policy_and_native_validation(policy, native):
    with pytest.raises(ProtocolError):
        backend(policy=policy, native=native).parse(canonical(response()), 'r0')


def test_unknown_policy_is_configuration_error():
    with pytest.raises(ConfigurationError, match='truncated_tool_call_policy'):
        backend(policy='repair-json')


def test_non_length_malformed_arguments_stay_protocol_error():
    with pytest.raises(ProtocolError):
        backend().parse(canonical(response(finish='tool_calls')), 'r0')


def test_length_preserves_wire_message_and_valid_calls_in_mixed_batch():
    doc = response()
    good = copy.deepcopy(doc['choices'][0]['message']['tool_calls'][0])
    good['id'] = 'complete-id'
    good['function']['arguments'] = '{"path":"complete.txt","content":"complete"}'
    doc['choices'][0]['message']['tool_calls'].append(good)
    reply = backend().parse(canonical(doc), 'r0')
    assert reply.finish_reason == 'length'
    assert reply.native_message == doc['choices'][0]['message']
    assert reply.calls[0].arguments == '{"path":"unfinished'
    assert reply.calls[1].arguments == {'path':'complete.txt','content':'complete'}
    assert reply.usage['total_tokens'] == 20


def test_no_call_length_is_preserved_for_orchestrator_recovery():
    doc = response()
    doc['choices'][0]['message'].pop('tool_calls')
    reply = backend().parse(canonical(doc), 'r0')
    assert reply.calls == ()
    assert reply.finish_reason == 'length'
    assert reply.native_message == doc['choices'][0]['message']


@pytest.mark.parametrize('damage', ['missing_id','duplicate_id','empty_name','bad_type'])
def test_length_without_identifiable_calls_cannot_execute(damage):
    doc = response()
    calls = doc['choices'][0]['message']['tool_calls']
    if damage == 'missing_id':
        calls[0].pop('id')
    elif damage == 'duplicate_id':
        calls.append(copy.deepcopy(calls[0]))
    elif damage == 'empty_name':
        calls[0]['function']['name'] = ''
    else:
        calls[0]['type'] = 'other'
    with pytest.raises(ProtocolError):
        backend().parse(canonical(doc), 'r0')


@pytest.mark.parametrize('mode', ['recorded','complete_arguments','no_calls'])
@pytest.mark.parametrize('last_slot', [False, True])
def test_recorded_length_response_feedback_and_corrected_same_id(
        tmp_path, config, monkeypatch, native_bridge, recorded_response, mode, last_slot):
    original = json.loads(recorded_response)
    raw_call = copy.deepcopy(original['choices'][0]['message']['tool_calls'][0])
    if mode == 'complete_arguments':
        original['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = (
            '{"path":"should-not-exist.txt","content":"truncated turn"}')
    elif mode == 'no_calls':
        original['choices'][0]['message'].pop('tool_calls')
        original['choices'][0]['message']['content'] = 'Unfinished text'
    first_body = recorded_response if mode == 'recorded' else canonical(original)
    cfg = pi_config(config, native=True)
    cfg['model'].update(adapter='opencode-chat', provider='opencode', model='deepseek-v4.1-flash',
        base_url='https://opencode.ai/inference/openai/v1',
        options={'truncated_tool_call_policy':TRUNCATED_TOOL_CALL_POLICY})
    cfg['budget'].update(max_model_requests=1 if last_slot else 3, max_tool_calls=2)
    cfg['recording']['observation_limit_chars'] = 100000
    settings = copy.deepcopy(cfg['model'])
    settings.update(settings.pop('options'))
    model = OpenCodeChatBackend(settings)
    sent = []
    corrected = response('{"path":"recovered.txt","content":"complete"}', 'tool_calls')
    corrected['choices'][0]['message']['tool_calls'][0]['id'] = raw_call['id']
    done = {'model':'deepseek-v4.1-flash', 'choices':[{
        'message':{'role':'assistant','content':'done'}, 'finish_reason':'stop'}],
        'usage':{'total_tokens':7}}
    replies = [first_body, canonical(corrected), canonical(done)]

    def post(url, body, timeout, **kwargs):
        sent.append(json.loads(body))
        if len(sent) == 2:
            assert runtime.executed == []
            assert sent[-1]['messages'][-2] == original['choices'][0]['message']
            feedback = sent[-1]['messages'][-1]
            if mode != 'no_calls':
                assert feedback['tool_call_id'] == raw_call['id']
            assert 'Model output reached its token limit.' in feedback['content']
        return HTTPResult(200, replies.pop(0), {'started_at':utc_now(), 'finished_at':utc_now(),
            'transport_error':None,'body_truncated':False})

    monkeypatch.setattr(model.http, 'post', post)

    class RecordingRuntime(PiFixtureRuntime):
        def __init__(self): self.executed = []
        def execute(self, call, workspace, timeout):
            self.executed.append(call)
            return super().execute(call, workspace, timeout)

    runtime = RecordingRuntime()
    fake_live_components(monkeypatch, model, runtime)
    output = tmp_path/'episode'
    result = run(cfg, output)
    assert result['status']['termination'] == ('request_budget' if last_slot else 'assistant_finished'), result
    assert result['verification']['verified'], result
    assert len(sent) == (1 if last_slot else 3)
    assert len(runtime.executed) == (0 if last_slot else 1)
    if not last_slot:
        assert runtime.executed[0].native_id == raw_call['id']
        assert runtime.executed[0].arguments == {'path':'recovered.txt','content':'complete'}
    assert (output/'raw/model/r000000/response.bin').read_bytes() == first_body
    http = read_json(output/'raw/model/r000000/http.json')
    assert http['native_finish_reason'] == 'length'
    assert http['truncated_tool_call_feedback'] == {'policy':TRUNCATED_TOOL_CALL_POLICY,
        'raw_arguments_retained_call_ids':[raw_call['id']] if mode == 'recorded' else []}
    validations = [e['data'] for e in events(output) if e['kind']=='tool.validation']
    assert [v['ok'] for v in validations] == ([] if last_slot else [True])
    usage = read_json(output/'manifest.json')['usage']
    assert usage['model_requests_including_parent'] == usage['new_http_attempts'] == (1 if last_slot else 3)
    assert usage['tool_calls_including_parent'] == ((0 if mode == 'no_calls' else 1)
        + (0 if last_slot else 1))
    assert usage['known_total_tokens_including_parent'] == 39446 + (0 if last_slot else 27)
    if last_slot:
        saved = read_json(output/'checkpoints/cp000001.json')['conversation']
        assert saved['_output_limit_feedback_pending'][0]['delivery_role'] == (
            'user' if mode == 'no_calls' else 'tool')
        assert saved['messages'][-2] == original['choices'][0]['message']
        assert 'Model output reached its token limit.' in saved['messages'][-1]['content']
