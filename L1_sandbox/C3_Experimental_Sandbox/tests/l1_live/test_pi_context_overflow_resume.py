"""Exact spent-context-rejection continuation, using CPU fixture ports only."""
import copy
import json

import pytest

from c3_sandbox.l1.core.contracts import ConfigurationError, EvidenceError, HTTPResult
from c3_sandbox.l1.core.orchestrator import run
from c3_sandbox.l1.evidence.live import seal, verify
from c3_sandbox.l1.models.native import OpenAICompatibleBackend
from c3_sandbox.l1.storage import canonical, digest, file_digest, read_json
from .test_pi_response_continuation import fake_live_components, events, native_bridge
from .test_pi_harness import PiFixtureRuntime


def write_json(root,relative,value):
    (root/relative).write_bytes(canonical(value)+b'\n')


class ScriptedOpenAI(OpenAICompatibleBackend):
    def __init__(self, config, replies):
        super().__init__(config)
        self.replies=list(replies); self.bodies=[]

    def send(self, body, timeout):
        self.bodies.append(json.loads(body))
        reply=self.replies.pop(0)
        status=400 if 'error' in reply else 200
        return HTTPResult(status,canonical(reply),dict(status_code=status,transport_error=None,body_truncated=False))


def reply(index=None):
    message={'role':'assistant','content':'done'}
    if index is not None:
        message['tool_calls']=[{'id':f'write-{index}','type':'function','function':{
            'name':'write','arguments':json.dumps({'path':f'note-{index}.txt','content':'x'*4000})}}]
    return {'choices':[{'message':message,'finish_reason':'tool_calls' if index is not None else 'stop'}],
            'usage':{'prompt_tokens':30000,'completion_tokens':10,'total_tokens':30010}}


def overflow():
    return {'error':{'type':'BadRequestError','code':400,'param':'input_tokens',
                     'message':"This model's maximum context length is 131072 tokens. parameter=input_tokens"}}


def cfg_for(config):
    cfg=copy.deepcopy(config)
    cfg['model'].update(adapter='openai-compatible',timeout_seconds=600)
    cfg['agent']={'name':'pi-coding-agent','version':'0.73.1','harness':'pi',
                  'validation_policy':'pi-native-0.73.1','harness_revision':'l1-pi-native-errors-context-v2',
                  'compaction':dict(enabled=True,context_window_tokens=25000,reserve_tokens=2000,keep_recent_tokens=3000)}
    cfg['budget'].update(max_model_requests=7,max_tool_calls=10)
    return cfg


def reseal(root, mutate=None):
    """Re-anchor synthetic fixture edits, preserving graph and native-request checks."""
    if mutate: mutate()
    history=events(root); previous=None
    def refs(value):
        if isinstance(value,dict):
            if set(value)=={'path','sha256','size_bytes'}:
                path=root/value['path']; value.update(sha256=file_digest(path),size_bytes=path.stat().st_size)
            else:
                for item in value.values(): refs(item)
        elif isinstance(value,list):
            for item in value: refs(item)
    for event in history:
        refs(event); event['previous_sha256']=previous
        event['sha256']=digest({k:v for k,v in event.items() if k!='sha256'}); previous=event['sha256']
    (root/'trajectory.jsonl').write_bytes(b''.join(canonical(e)+b'\n' for e in history))
    (root/'evidence_index.json').unlink()
    seal(root,history)


def parent_fixture(tmp_path,config,monkeypatch,bad=None,request_limit=7,unsatisfiable=False):
    import c3_sandbox.l1.core.orchestrator as module
    cfg=cfg_for(config)
    cfg['budget']['max_model_requests']=request_limit
    if unsatisfiable:
        cfg['agent']['compaction'].update(context_window_tokens=5000,reserve_tokens=2000,keep_recent_tokens=1000)
    model=ScriptedOpenAI(cfg['model'],[reply(0),reply(1),reply(2),bad or overflow()])
    fake_live_components(monkeypatch,model,PiFixtureRuntime())
    # Reconstruct the historical v1 compactor behavior without changing production
    # execution's explicit v2 gate. Re-anchor the resulting synthetic v1 source.
    with monkeypatch.context() as patch:
        patch.setattr(module,'_maybe_compact_pi_conversation',lambda conv,*a,**kw:(conv,None))
        result=run(cfg,tmp_path/'parent')
    assert result['status']['termination']=='model_transport_error',result
    parent=tmp_path/'parent'
    resolved=read_json(parent/'raw/resolved_config.json')
    resolved['agent']['harness_revision']='l1-pi-native-errors-v1'
    write_json(parent,'raw/resolved_config.json',resolved)
    manifest=read_json(parent/'manifest.json'); manifest['config_sha256']=digest(resolved)
    manifest['agent']['configuration']=resolved['agent']; write_json(parent,'manifest.json',manifest)
    for path in (parent/'checkpoints').glob('cp*.json'):
        cp=read_json(path); cp['config_sha256']=digest(resolved); write_json(parent,path.relative_to(parent).as_posix(),cp)
    reseal(parent)
    assert verify(parent)['verified'],verify(parent)
    return parent,cfg


def resume(parent,cfg,child,monkeypatch,replies=None,**kwargs):
    model=ScriptedOpenAI(cfg['model'],replies if replies is not None else [reply()])
    fake_live_components(monkeypatch,model,PiFixtureRuntime())
    result=run(cfg,child,parent=parent,checkpoint_id='cp000003',
               resume_context_overflow_id='r000003',allow_pi_native_upgrade=True,**kwargs)
    return result,model


def test_context_resume_compacts_before_fresh_request_and_preserves_spent_unknown(tmp_path,config,monkeypatch,native_bridge):
    parent,cfg=parent_fixture(tmp_path,config,monkeypatch)
    original=(parent/'evidence_index.json').read_bytes()
    result,model=resume(parent,cfg,tmp_path/'child',monkeypatch)
    assert result['verification']['verified'],result
    assert result['status']['termination']=='assistant_finished',result
    child=tmp_path/'child'; manifest=read_json(child/'manifest.json')
    assert manifest['usage']['model_requests_including_parent']==5
    assert manifest['usage']['new_http_attempts']==1
    assert manifest['usage']['saved_response_consumptions']==0
    assert manifest['usage']['all_requests_have_usage'] is False
    assert manifest['usage']['known_total_tokens_including_parent']==4*30010
    assert len(model.bodies)==1
    fresh=[e for e in events(child) if e['kind']=='model.request']
    assert fresh[0]['data']['request_id'].endswith(':r000004')
    assert model.bodies[0]!=read_json(parent/'raw/model/r000003/request.json')
    compacted=[e for e in events(child) if e['kind']=='session.compacted']
    assert compacted and compacted[0]['seq']<fresh[0]['seq']
    cp=read_json(child/'checkpoints/cp000000.json')
    assert cp['conversation']['_pi_context_usage']['source_messages_sha256']==digest(read_json(parent/'raw/model/r000002/request.json')['messages'])
    assert (parent/'evidence_index.json').read_bytes()==original
    assert verify(child)['verified']


@pytest.mark.parametrize('damage',['body','usage','checkpoint','config','budget','timeout'])
def test_context_resume_refuses_tampered_or_unapproved_source(tmp_path,config,monkeypatch,native_bridge,damage):
    parent,cfg=parent_fixture(tmp_path,config,monkeypatch)
    if damage=='body': (parent/'raw/model/r000003/response.bin').write_bytes(b'{}')
    if damage=='usage': (parent/'raw/model/r000002/response.bin').write_bytes(b'{}')
    if damage=='checkpoint':
        cp=read_json(parent/'checkpoints/cp000003.json'); cp['pending_calls']=['pending']; write_json(parent,'checkpoints/cp000003.json',cp)
    if damage=='config': cfg['model']['parameters']['temperature']=0.9
    if damage=='budget': cfg['budget']['max_model_requests']+=1
    if damage=='timeout': cfg['model']['timeout_seconds']=300
    model=ScriptedOpenAI(cfg['model'],[reply()]); fake_live_components(monkeypatch,model,PiFixtureRuntime())
    with pytest.raises((EvidenceError,ConfigurationError)):
        run(cfg,tmp_path/'blocked',parent=parent,checkpoint_id='cp000003',resume_context_overflow_id='r000003',allow_pi_native_upgrade=True)
    assert model.bodies==[]
    assert not (tmp_path/'blocked').exists()


@pytest.mark.parametrize('cause',['authentication','param','truncated','candidate'])
def test_other_400_is_not_context_resume(tmp_path,config,monkeypatch,native_bridge,cause):
    bad=overflow()
    if cause=='authentication': bad['error']['message']='invalid API key'
    if cause=='param': bad['error']['param']='max_tokens'
    if cause=='candidate': bad['choices']=[{'message':{'content':'answer'}}]
    parent,cfg=parent_fixture(tmp_path,config,monkeypatch,bad)
    if cause=='truncated':
        metadata=read_json(parent/'raw/model/r000003/http.json');metadata['body_truncated']=True
        write_json(parent,'raw/model/r000003/http.json',metadata);reseal(parent)
    with pytest.raises(EvidenceError): resume(parent,cfg,tmp_path/'blocked',monkeypatch)


def test_copied_source_cannot_be_changed_even_with_child_reseal(tmp_path,config,monkeypatch,native_bridge):
    parent,cfg=parent_fixture(tmp_path,config,monkeypatch)
    result,_=resume(parent,cfg,tmp_path/'child',monkeypatch)
    assert result['verification']['verified'],result
    copied=tmp_path/'child/lineage/context_source/raw/model/r000002/request.json'
    copied.write_bytes(copied.read_bytes()+b' ')
    reseal(tmp_path/'child')
    assert not verify(tmp_path/'child')['verified']


def test_external_deadline_sends_nothing_but_keeps_spent_parent(tmp_path,config,monkeypatch,native_bridge):
    parent,cfg=parent_fixture(tmp_path,config,monkeypatch)
    class Expired:
        def remaining_seconds(self): return 0
    result,model=resume(parent,cfg,tmp_path/'child',monkeypatch,request_limiter=Expired())
    assert result['status']['termination']=='wall_time_budget',result
    assert result['verification']['verified'],result
    assert model.bodies==[]
    usage=read_json(tmp_path/'child/manifest.json')['usage']
    assert usage['model_requests_including_parent']==4 and usage['new_http_attempts']==0


def test_exhausted_logical_budget_preserves_failed_request_without_send(tmp_path,config,monkeypatch,native_bridge):
    parent,cfg=parent_fixture(tmp_path,config,monkeypatch,request_limit=4)
    result,model=resume(parent,cfg,tmp_path/'child',monkeypatch)
    assert result['status']['termination']=='request_budget',result
    assert result['verification']['verified'],result
    assert model.bodies==[]
    assert read_json(tmp_path/'child/manifest.json')['usage']['model_requests_including_parent']==4


def test_unsatisfiable_context_remains_invalid_complete_and_sends_nothing(tmp_path,config,monkeypatch,native_bridge):
    parent,cfg=parent_fixture(tmp_path,config,monkeypatch,unsatisfiable=True)
    result,model=resume(parent,cfg,tmp_path/'child',monkeypatch)
    assert result['status']['termination']=='pi_context_unsatisfiable',result
    assert result['status']['execution_validity']=='invalid'
    assert result['status']['evidence_completeness']=='complete'
    assert result['verification']['verified'],result
    assert model.bodies==[]


def tool_child(tmp_path,config,monkeypatch):
    parent,cfg=parent_fixture(tmp_path,config,monkeypatch)
    child=tmp_path/'child'
    first=reply(4); first['usage'].update(prompt_tokens=4000,total_tokens=4010)
    result,model=resume(parent,cfg,child,monkeypatch,replies=[first,reply()])
    assert result['status']['termination']=='assistant_finished',result
    assert result['verification']['verified'],result
    return parent,child,model


def test_executed_tool_has_exact_charge_and_checkpoint_budgets(tmp_path,config,monkeypatch,native_bridge):
    parent,child,model=tool_child(tmp_path,config,monkeypatch)
    history=events(child)
    charges=[e for e in history if e['kind']=='tool.charged']
    assert len(charges)==1 and charges[0]['data']['tool_calls_including_parent']==4
    actions=[e for e in history if e['kind'] in ('tool.validation','tool.started','tool.result')]
    assert actions and all(e['seq']>charges[0]['seq'] for e in actions)
    assert read_json(child/'manifest.json')['usage']['tool_calls_including_parent']==4
    assert read_json(child/'checkpoints/cp000000.json')['budget_used']['tool_calls']==3
    assert read_json(child/'checkpoints/cp000001.json')['budget_used']['tool_calls']==4
    assert len(model.bodies)==2


@pytest.mark.parametrize('target,field,delta',[
    ('manifest','tool_calls_including_parent',-1),
    ('manifest','tool_calls_including_parent',1),
    ('cp000000','tool_calls',1),
    ('cp000001','tool_calls',-1),
    ('cp000001','tool_calls',1),
    ('cp000001','model_requests',-1),
    ('cp000001','known_total_tokens',-1),
])
def test_resealed_spent_tool_or_checkpoint_budget_tamper_fails(tmp_path,config,monkeypatch,native_bridge,target,field,delta):
    _,child,_=tool_child(tmp_path,config,monkeypatch)
    path='manifest.json' if target=='manifest' else 'checkpoints/'+target+'.json'
    value=read_json(child/path)
    value['usage' if target=='manifest' else 'budget_used'][field]+=delta
    write_json(child,path,value); reseal(child)
    report=verify(child)
    assert not report['verified'],report
    assert any('budget' in row['detail'].lower() for row in report['errors']),report


def test_charge_survives_deadline_before_validation_receipt(tmp_path,config,monkeypatch,native_bridge):
    parent,cfg=parent_fixture(tmp_path,config,monkeypatch)
    import c3_sandbox.l1.core.orchestrator as module
    import c3_sandbox.l1.runtime.pi_native_validation as validation_module
    factory=validation_module.NativePiValidator

    class DeadlineValidator:
        def __init__(self): self.delegate=factory()
        def __getattr__(self,name): return getattr(self.delegate,name)
        def validate(self,name,arguments,*,call_id,deadline_monotonic):
            monkeypatch.setattr(module.time,'monotonic',lambda:deadline_monotonic+1)
            raise TimeoutError('fixture deadline before native validation receipt')

    monkeypatch.setattr(validation_module,'NativePiValidator',DeadlineValidator)
    child=tmp_path/'child'
    result,model=resume(parent,cfg,child,monkeypatch,replies=[reply(4)])
    assert result['status']['termination']=='wall_time_budget',result
    assert result['verification']['verified'],result
    history=events(child)
    assert len([e for e in history if e['kind']=='tool.charged'])==1
    assert not any(e['kind'] in ('tool.validation','tool.started','tool.error','tool.result') for e in history)
    assert any(e['kind']=='tool.skipped' for e in history)
    assert read_json(child/'manifest.json')['usage']['tool_calls_including_parent']==4
