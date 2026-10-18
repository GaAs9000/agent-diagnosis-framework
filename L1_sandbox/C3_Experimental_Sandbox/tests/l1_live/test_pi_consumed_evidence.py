"""Consumed responses must remain bound to the original sealed wire evidence."""
import copy

import pytest

from c3_sandbox.l1.core.orchestrator import run
from c3_sandbox.l1.evidence.live import observation_in_request, seal, verify
from c3_sandbox.l1.storage import canonical, digest, read_json
from .conftest import ScriptedGemini, gemini_done
from .test_pi_harness import PiFixtureRuntime
from .test_pi_response_continuation import (
    events, fake_live_components, make_parent, native_bridge, pi_config,
)


def refresh_child_seal(root):
    rows=events(root)
    previous=None
    for row in rows:
        def update_refs(value):
            if isinstance(value,dict):
                if set(value)=={'path','sha256','size_bytes'}:
                    import hashlib
                    data=(root/value['path']).read_bytes()
                    value.update(sha256=hashlib.sha256(data).hexdigest(),size_bytes=len(data))
                else:
                    for item in value.values(): update_refs(item)
            elif isinstance(value,list):
                for item in value: update_refs(item)
        update_refs(row)
        row['previous_sha256']=previous
        row['sha256']=digest({k:v for k,v in row.items() if k!='sha256'})
        previous=row['sha256']
    (root/'trajectory.jsonl').write_bytes(b''.join(canonical(row)+b'\n' for row in rows))
    (root/'evidence_index.json').unlink()
    seal(root,rows)


@pytest.mark.parametrize('damage',['source_manifest','claim_new_http','source_call_id','wrong_checkpoint'])
def test_resealed_child_cannot_detach_consumed_prefix(tmp_path,config,monkeypatch,native_bridge,damage):
    parent=make_parent(tmp_path,config,monkeypatch)
    cfg=pi_config(config,native=True)
    model=ScriptedGemini(cfg['model'],[gemini_done()])
    fake_live_components(monkeypatch,model,PiFixtureRuntime())
    child=tmp_path/'child'
    result=run(cfg,child,parent=parent,checkpoint_id='cp000001',
               resume_response_id='r000001',allow_pi_native_upgrade=True)
    assert result['verification']['verified'],result
    if damage=='source_manifest':
        path=child/'lineage/source_manifest.json'
        value=read_json(path)
        value['usage']['known_total_tokens_including_parent']+=1
        (child/'lineage/source_manifest.json').write_bytes(canonical(value)+b'\n')
    elif damage=='claim_new_http':
        rows=events(child)
        next(row for row in rows if row['kind']=='response.consumed')['data']['new_http_attempts']=1
        (child/'trajectory.jsonl').write_bytes(b''.join(canonical(row)+b'\n' for row in rows))
    elif damage=='source_call_id':
        manifest=read_json(child/'manifest.json')
        manifest['lineage']['response_use']['source_call_id']='fabricated-source-call'
        (child/'manifest.json').write_bytes(canonical(manifest)+b'\n')
        (child/'lineage/response_use.json').write_bytes(canonical(manifest['lineage']['response_use'])+b'\n')
        rows=events(child)
        next(row for row in rows if row['kind']=='response.prefix_accounted')['data']['source_call_id']='fabricated-source-call'
        (child/'trajectory.jsonl').write_bytes(b''.join(canonical(row)+b'\n' for row in rows))
    else:
        import hashlib
        manifest=read_json(child/'manifest.json')
        data=(parent/'checkpoints/cp000000.json').read_bytes()
        (child/'lineage/parent_checkpoint.json').write_bytes(data)
        manifest['lineage'].update(parent_checkpoint_id='cp000000',parent_checkpoint_sha256=hashlib.sha256(data).hexdigest())
        (child/'manifest.json').write_bytes(canonical(manifest)+b'\n')
    refresh_child_seal(child)
    report=verify(child)
    assert report['verified'] is False
    assert any('Consumed source differs' in str(error) or 'transport/count binding mismatch' in str(error) or
               'source call graph mismatch' in str(error) or 'does not immediately precede' in str(error)
               for error in report['errors']),report


@pytest.mark.parametrize('provider',['chat','gemini','anthropic','responses'])
def test_native_error_delivery_checks_exact_wire_text_and_id(provider):
    text='Validation failed for tool "edit": native error'
    obs={'content':[{'type':'text','text':text}],'details':{},'isError':True}
    if provider=='chat':
        body={'messages':[{'role':'tool','tool_call_id':'c1','content':text}]}
    elif provider=='gemini':
        body={'contents':[{'role':'user','parts':[{'functionResponse':{'name':'edit','id':'c1','response':{'error':text}}}]}]}
    elif provider=='anthropic':
        body={'messages':[{'role':'user','content':[{'type':'tool_result','tool_use_id':'c1','content':text,'is_error':True}]}]}
    else:
        body={'input':[{'type':'function_call_output','call_id':'c1','output':text}]}
    assert observation_in_request(body,obs,'c1','edit')
    assert not observation_in_request(body,obs,'other','edit')
    wrong=copy.deepcopy(obs); wrong['content'][0]['text']+=' changed'
    assert not observation_in_request(body,wrong,'c1','edit')
