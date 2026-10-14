import copy
import json
import pytest
from c3_sandbox.l1.core.contracts import ProtocolError, Result
from c3_sandbox.l1.models.native import GeminiBackend, OpenAICompatibleBackend
from c3_sandbox.l1.storage import canonical
from .conftest import gemini_call,gemini_done

def test_gemini_native_signature_and_id_roundtrip(config):
    m=GeminiBackend(config['model']);raw=gemini_call();r=m.parse(canonical(raw),'r1')
    conv=m.accept(m.initial('system','task'),r,[(r.calls[0],Result({'stdout':'real'}))])
    body=m.request(conv,[])
    assert body['contents'][1]==raw['candidates'][0]['content']
    assert body['contents'][2]['parts'][0]['functionResponse']['id']=='native-call-1'

def test_gemini_missing_native_id_is_not_invented(config):
    m=GeminiBackend(config['model']);r=m.parse(canonical(gemini_call(native_id=None)),'req')
    assert r.calls[0].native_id is None
    body=m.accept(m.initial('s','t'),r,[(r.calls[0],Result({'x':1}))])
    assert 'id' not in body['contents'][-1]['parts'][0]['functionResponse']

def test_gemini_parallel_calls(config):
    raw=gemini_call(); p=copy.deepcopy(raw['candidates'][0]['content']['parts'][0]);p['functionCall']['id']='n2'
    raw['candidates'][0]['content']['parts'].append(p)
    r=GeminiBackend(config['model']).parse(canonical(raw),'r')
    assert len(r.calls)==2 and r.calls[0].call_id!=r.calls[1].call_id

@pytest.mark.parametrize('raw',[b'not json',b'{"x":1,"x":2}',b'[]',b'{"x":NaN}',b'{"error":{"message":"bad"}}',b'{"candidates":[]}'])
def test_invalid_native_json(config,raw):
    with pytest.raises(ProtocolError): GeminiBackend(config['model']).parse(raw,'r')

def chat(calls=True):
    msg={'role':'assistant','content':None,'extra_content':{'signature':'opaque'}}
    if calls:msg['tool_calls']=[{'type':'function','id':'t1','function':{'name':'run','arguments':'{"command":"ls"}'}}]
    else:msg['content']='finished'
    return {'choices':[{'message':msg,'finish_reason':'tool_calls' if calls else 'stop'}],'usage':{'total_tokens':10},'model':'local-model'}

def test_chat_native_message_roundtrip(config):
    m=OpenAICompatibleBackend(config['model']);v=chat();r=m.parse(canonical(v),'r')
    c=m.accept(m.initial('s','t'),r,[(r.calls[0],Result({'stdout':'files'}))])
    assert c['messages'][2]==v['choices'][0]['message']
    assert c['messages'][-1]['tool_call_id']=='t1'
    assert json.loads(c['messages'][-1]['content'])=={'stdout':'files'}

@pytest.mark.parametrize('reason',['length','content_filter',None])
def test_incomplete_chat_response_rejected(config,reason):
    m=OpenAICompatibleBackend(config['model']);v=chat();v['choices'][0]['finish_reason']=reason
    with pytest.raises(ProtocolError):m.parse(canonical(v),'r')

def test_no_parsing_react_text_as_native_call(config):
    m=GeminiBackend(config['model']);v=gemini_done();v['candidates'][0]['content']['parts'][0]['text']='Action: run\nAction Input: ls'
    assert m.parse(canonical(v),'r').calls==()

def test_images_are_actual_bytes(config):
    data=b'\x89PNG\r\n\x1a\nactual'
    for backend,raw in [(GeminiBackend,gemini_call()),(OpenAICompatibleBackend,chat())]:
        m=backend(config['model']);r=m.parse(canonical(raw),'r')
        conv=m.accept(m.initial('s','t'),r,[(r.calls[0],Result({'mime':'image/png'},image=data,image_mime='image/png'))])
        import base64
        assert base64.b64encode(data).decode() in json.dumps(conv)
