"""Lossless, opt-in projection of native interleaved tool image histories."""
import copy
import pytest

from c3_sandbox.l1.core.contracts import ConfigurationError, HTTPResult
from c3_sandbox.l1.models.opencode import HISTORY_PROJECTION_POLICY, OpenCodeChatBackend, project_tool_history
from c3_sandbox.l1.storage import canonical, digest


def backend(policy=HISTORY_PROJECTION_POLICY):
    config={'kind':'local','provider':'opencode-go','model':'deepseek-v4.1-flash',
        'base_url':'http://localhost/v1','api_key_env':None,'parameters':{'temperature':1},
        'max_response_bytes':100000}
    if policy is not None:
        config['history_projection_policy']=policy
    return OpenCodeChatBackend(config)


def image(call_id,payload):
    return {'role':'user','content':[{'type':'text','text':f'Image returned by tool call {call_id}'},
        {'type':'image_url','image_url':{'url':payload}}]}


def history():
    return [{'role':'system','content':'Preserve everything. 保留'},
        {'role':'user','content':'Inspect both.'},
        {'role':'assistant','content':None,'reasoning_content':'original reasoning',
            'tool_calls':[{'id':call_id,'type':'function','function':{'name':'read',
                'arguments':'{ "path": "图像 '+call_id+'.png" }'}} for call_id in ['call-a','call-b']]},
        {'role':'tool','tool_call_id':'call-a','content':'exact stdout\n{"original": true}'},
        image('call-a','data:image/png;base64,Zmlyc3Q='),
        {'role':'tool','tool_call_id':'call-b','content':'second stdout 保留'},
        image('call-b','data:image/jpeg;base64,c2Vjb25k'),
        {'role':'user','content':'Continue with this intervention.'}]


def test_interleaved_projection_preserves_every_message_and_audits_wire(monkeypatch):
    adapter=backend(); messages=history(); original=copy.deepcopy(messages)
    conversation={'messages':messages,'_pi_context_usage':{'estimate_multiplier':2}}
    tools=[{'name':'read','parameters':{'type':'object'}}]
    body=adapter.request(conversation,tools)
    order=[0,1,2,3,5,4,6,7]
    assert body['messages']==[original[index] for index in order]
    assert conversation['messages']==original
    assert conversation['_pi_context_usage']=={'estimate_multiplier':2}
    assert set(conversation)=={'messages','_pi_context_usage','_opencode_session_id'}
    assert body['tools']==[{'type':'function','function':tools[0]}]
    posted=[]
    def post(url,payload,timeout,**kwargs):
        posted.append(payload)
        return HTTPResult(400,b'{}',{'retained':'metadata'})
    monkeypatch.setattr(adapter.http,'post',post)
    encoded=canonical(body)
    receipt=adapter.send(encoded,1)
    assert posted==[encoded]
    assert receipt.metadata['retained']=='metadata'
    assert receipt.metadata['history_projection']=={
        'policy':HISTORY_PROJECTION_POLICY,'source_messages_sha256':digest(original),
        'wire_messages_sha256':digest(body['messages']),'changed':True,
        'original_position_mapping':[{'wire_position':wire,'source_position':source}
            for wire,source in enumerate(order)]}
    # Reconstruct original messages directly from the persisted mapping.
    restored=[None]*len(original)
    for item in receipt.metadata['history_projection']['original_position_mapping']:
        restored[item['source_position']]=body['messages'][item['wire_position']]
    assert canonical(restored)==canonical(original)
    assert adapter.send(encoded,1).metadata['history_projection']==receipt.metadata['history_projection']


def test_already_grouped_history_is_byte_identical_noop(monkeypatch):
    messages=history(); messages[4],messages[5]=messages[5],messages[4]
    adapter=backend(); body=adapter.request({'messages':messages},[])
    assert canonical(body['messages'])==canonical(messages)
    monkeypatch.setattr(adapter.http,'post',lambda *a,**k:HTTPResult(200,b'{}',{}))
    audit=adapter.send(canonical(body),1).metadata['history_projection']
    assert audit['changed'] is False
    assert audit['source_messages_sha256']==audit['wire_messages_sha256']
    assert [item['source_position'] for item in audit['original_position_mapping']]==list(range(len(messages)))


@pytest.mark.parametrize('mutation',[
    'missing_result','duplicate_result','unmatched_result','unknown_user','unknown_user_image',
    'duplicate_call','missing_call_id','image_before_result','unmatched_image','malformed_image',
    'orphan_result','orphan_image'])
def test_rejects_ambiguous_histories_before_http(monkeypatch,mutation):
    messages=history()
    if mutation=='missing_result': del messages[5:7]
    elif mutation=='duplicate_result': messages.insert(4,copy.deepcopy(messages[3]))
    elif mutation=='unmatched_result': messages[5]['tool_call_id']='unknown'
    elif mutation=='unknown_user': messages[4]={'role':'user','content':'unknown interruption'}
    elif mutation=='unknown_user_image': messages[4]['content'][0]['text']='Unknown image origin'
    elif mutation=='duplicate_call': messages[2]['tool_calls'][1]['id']='call-a'
    elif mutation=='missing_call_id': del messages[2]['tool_calls'][1]['id']
    elif mutation=='image_before_result': messages[3],messages[4]=messages[4],messages[3]
    elif mutation=='unmatched_image': messages[4]['content'][0]['text']='Image returned by tool call unknown'
    elif mutation=='malformed_image': messages[4]['extra']='ambiguous'
    elif mutation=='orphan_result': messages.append(copy.deepcopy(messages[3]))
    elif mutation=='orphan_image': messages.insert(0,copy.deepcopy(messages[4]))
    adapter=backend()
    def forbidden(*args,**kwargs): pytest.fail('Malformed history reached HTTP')
    monkeypatch.setattr(adapter.http,'post',forbidden)
    with pytest.raises(ConfigurationError): adapter.request({'messages':messages},[])


def test_default_remains_unchanged_without_policy(monkeypatch):
    adapter=backend(None); messages=history(); body=adapter.request({'messages':messages},[])
    assert canonical(body['messages'])==canonical(messages)
    monkeypatch.setattr(adapter.http,'post',lambda *a,**k:HTTPResult(200,b'{}',{}))
    assert 'history_projection' not in adapter.send(canonical(body),1).metadata


def test_projection_receipt_cannot_bind_different_wire_history(monkeypatch):
    adapter=backend(); body=adapter.request({'messages':history()},[])
    body['messages'][0]['content']='tampered'
    def forbidden(*args,**kwargs): pytest.fail('Mismatched projection reached HTTP')
    monkeypatch.setattr(adapter.http,'post',forbidden)
    with pytest.raises(ConfigurationError,match='differs from the bound'):
        adapter.send(canonical(body),1)


def test_unknown_policy_rejected():
    with pytest.raises(ConfigurationError,match='history_projection_policy'): backend('unknown')


def test_multiple_turns_and_reversed_tool_order_remain_lossless():
    messages=history(); messages[3],messages[5]=messages[5],messages[3]
    messages[4],messages[6]=messages[6],messages[4]
    messages+=copy.deepcopy(history()[2:])
    adapter=backend(); body=adapter.request({'messages':messages},[])
    assert len(body['messages'])==len(messages)
    assert [m['tool_call_id'] for m in body['messages'] if m['role']=='tool']==['call-b','call-a','call-a','call-b']
    assert sorted(canonical(m) for m in body['messages'])==sorted(canonical(m) for m in messages)


def test_public_preflight_helper_is_pure_and_noop_proof_is_repeatable():
    messages=history(); original=copy.deepcopy(messages)
    wire,audit=project_tool_history(messages)
    assert messages==original
    replay,noop=project_tool_history(wire)
    assert canonical(replay)==canonical(wire)
    assert noop['changed'] is False
    assert noop['source_messages_sha256']==noop['wire_messages_sha256']==audit['wire_messages_sha256']
    wire[0]['content']='edited output'
    assert messages==original
