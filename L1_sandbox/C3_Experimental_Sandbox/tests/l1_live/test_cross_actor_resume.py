"""Offline actor transitions preserve sealed chat history and inherited budgets."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from c3_sandbox.l1.core.config import DEFAULTS
from c3_sandbox.l1.core.contracts import ConfigurationError, EvaluationSpec, HTTPResult, Result, Task
from c3_sandbox.l1.core.orchestrator import _validate_actor_transition, build_execution_contract_prompt, run
from c3_sandbox.l1.models.native import OpenAICompatibleBackend
from c3_sandbox.l1.models.opencode import OpenCodeChatBackend
from c3_sandbox.l1.storage import canonical, digest, file_digest, read_json
from .conftest import FixtureEvaluator, TrustedFixtureRuntime


def configs():
    source=copy.deepcopy(DEFAULTS)
    source['model'].update(adapter='openai-compatible',kind='local',provider='vllm',
        model='Qwen/Qwen2.5-32B-Instruct',api_key_env=None,base_url='http://localhost:8000/v1')
    source['agent'].update(name='pi-coding-agent',version='0.73.1',harness='pi',
        compaction={'enabled':False})
    source['budget'].update(max_model_requests=3,max_tool_calls=3,wall_time_seconds=60,max_total_tokens=1000)
    source['agent']['system_prompt']=build_execution_contract_prompt(source)
    target=copy.deepcopy(source)
    target['model'].update(adapter='opencode-chat',kind='api',provider='opencode-go',
        model='deepseek-v4.1-flash',base_url='https://opencode.ai/inference/go/v1')
    return source,target


def binding(source,target,index='a'*64,checkpoint='b'*64):
    return {'schema_version':'cross-actor-model-transition-v1',
        'source_config_sha256':digest(source),'target_config_sha256':digest(target),
        'source_model':copy.deepcopy(source['model']),'target_model':copy.deepcopy(target['model']),
        'parent_index_sha256':index,'parent_checkpoint_sha256':checkpoint}


def test_exact_binding_does_not_mutate_history_or_models():
    source,target=configs()
    target['agent']['intervention']={'at_request':2,'message':'Continue from this evidence.'}
    cp={'conversation':{'messages':[{'role':'user','content':[{'type':'image_url',
        'image_url':{'url':'data:image/png;base64,AA=='}}]},
        {'role':'assistant','tool_calls':[{'id':'opaque','type':'function',
        'function':{'name':'bash','arguments':'{"command":"ls"}'}}]},
        {'role':'tool','tool_call_id':'opaque','content':'unchanged output'}]}}
    originals=copy.deepcopy((source,target,cp))
    expected=binding(source,target)
    assert _validate_actor_transition(expected,source,target,cp,'a'*64,'b'*64)==expected
    assert (source,target,cp)==originals
    assert build_execution_contract_prompt(target)==source['agent']['system_prompt']


@pytest.mark.parametrize('field', ['schema_version','source_config_sha256','target_config_sha256',
    'source_model','target_model','parent_index_sha256','parent_checkpoint_sha256'])
def test_rejects_inexact_binding(field):
    source,target=configs(); contract=binding(source,target); contract[field]='tampered'
    with pytest.raises(ConfigurationError,match='exactly bind'):
        _validate_actor_transition(contract,source,target,{'conversation':{'messages':[{'role':'user'}]}},'a'*64,'b'*64)


@pytest.mark.parametrize('section,key,value', [
    ('budget','max_total_tokens',2000),('runtime','tool_timeout_seconds',61),
    ('recording','observation_limit_chars',42),('agent','harness','minimal'),
    ('agent','compaction',{'enabled':True}),('agent','system_prompt','changed'),
    ('evaluation','timeout_seconds',121),('plugins','load_installed',True)])
def test_rejects_other_config_changes_even_when_explicitly_bound(section,key,value):
    source,target=configs(); target[section][key]=value
    with pytest.raises(ConfigurationError,match='only model'):
        _validate_actor_transition(binding(source,target),source,target,
            {'conversation':{'messages':[{'role':'user'}]}},'a'*64,'b'*64)


def test_rejects_changed_image_support():
    source,target=configs(); target['model']['supports_images']=False
    with pytest.raises(ConfigurationError,match='preserve image support'):
        _validate_actor_transition(binding(source,target),source,target,
            {'conversation':{'messages':[{'role':'user'}]}},'a'*64,'b'*64)


@pytest.mark.parametrize('source_adapter,target_adapter,provider,model,history', [
    ('gemini','opencode-chat','opencode-go','deepseek-v4.1-flash',{'contents':[]}),
    ('opencode-responses','opencode-chat','opencode-go','deepseek-v4.1-flash',{'input':[]}),
    ('openai-compatible','opencode-chat','opencode','deepseek-v4.1-flash',{'messages':[{'role':'user'}]}),
    ('openai-compatible','opencode-chat','opencode-go','other',{'messages':[{'role':'user'}]}),
    ('openai-compatible','opencode-responses','opencode-go','deepseek-v4.1-flash',{'messages':[{'role':'user'}]}),
    ('openai-compatible','opencode-chat','opencode-go','deepseek-v4.1-flash',{'input':[],'messages':[{'role':'user'}]}),
    ('openai-compatible','opencode-chat','opencode-go','deepseek-v4.1-flash',{'messages':[]}),
])
def test_rejects_incompatible_protocol_or_target(source_adapter,target_adapter,provider,model,history):
    source,target=configs(); source['model']['adapter']=source_adapter
    target['model'].update(adapter=target_adapter,provider=provider,model=model)
    with pytest.raises(ConfigurationError):
        _validate_actor_transition(binding(source,target),source,target,{'conversation':history},'a'*64,'b'*64)


@pytest.mark.parametrize('kwargs', [{},{'parent':Path('/unused')},
    {'parent':Path('/unused'),'checkpoint_id':'cp000000','allow_pi_native_upgrade':True},
    {'parent':Path('/unused'),'checkpoint_id':'cp000000','resume_response_id':'r000000'},
    {'parent':Path('/unused'),'checkpoint_id':'cp000000','resume_context_overflow_id':'r000000'}])
def test_requires_standalone_cold_resume(tmp_path,kwargs):
    _,target=configs()
    with pytest.raises(ConfigurationError,match='Actor transition requires'):
        run(target,tmp_path/'never-created',actor_transition={},**kwargs)
    assert not (tmp_path/'never-created').exists()


class Benchmark:
    def load(self,task_id):
        return Task('officeval',task_id,'Keep the fixture history.',(),{'adapter':'test_fixture'}, {},
            EvaluationSpec('fixture',Path('.'),'unused.py',task_id))


class Runtime(TrustedFixtureRuntime):
    def execute(self,call,workspace,timeout):
        (workspace/'submission'/'fixture.txt').write_text('preserved')
        return Result({'stdout':'unaltered tool output','stderr':'','exit_code':0},
            b'unaltered tool output',provenance={'test_fixture':True,'quiescent':True})


class ScriptedChat:
    def send(self,body,timeout):
        self.bodies.append(json.loads(body))
        message=({'role':'assistant','content':None,'tool_calls':[{'id':'native-opaque-id','type':'function',
            'function':{'name':'bash','arguments':'{"command":"ls"}'}}]} if len(self.bodies)==1
            and self.config['adapter']=='openai-compatible' else {'role':'assistant','content':'Done'})
        return HTTPResult(200,canonical({'choices':[{'message':message,
            'finish_reason':'tool_calls' if message.get('tool_calls') else 'stop'}],
            'usage':{'total_tokens':12},'model':self.config['model']}),{})


class Source(ScriptedChat,OpenAICompatibleBackend):
    def __init__(self,config):
        super().__init__(config); self.bodies=[]


class Target(ScriptedChat,OpenCodeChatBackend):
    def __init__(self,config):
        super().__init__(config); self.bodies=[]


@pytest.mark.parametrize('intervention', [False,True])
def test_cold_resume_preserves_parent_history_budget_and_sealed_lineage(tmp_path,intervention):
    source,target=configs(); parent=tmp_path/'parent'; child=tmp_path/'child'
    ports={'benchmark':Benchmark(),'model':Source(source['model']),
        'runtime':Runtime(),'evaluation':FixtureEvaluator()}
    assert run(source,parent,injected=ports)['verification']['verified']
    before={str(p.relative_to(parent)):file_digest(p) for p in parent.rglob('*') if p.is_file()}
    cp=read_json(parent/'checkpoints/cp000001.json')
    source=read_json(parent/'raw/resolved_config.json')
    if intervention:
        target['agent']['intervention']={'at_request':2,'message':'Use the supplied evidence.'}
    contract=binding(source,target,file_digest(parent/'evidence_index.json'),file_digest(parent/'checkpoints/cp000001.json'))
    backend=Target(target['model'])
    child_ports={**ports,'model':backend,'runtime':Runtime()}
    with pytest.raises(ConfigurationError,match='same resolved config'):
        run(target,child,injected=child_ports,parent=parent,checkpoint_id='cp000001')
    illegal=copy.deepcopy(target)
    illegal['agent']['intervention']={'at_request':1,'message':'Too early'}
    illegal_binding=binding(source,illegal,contract['parent_index_sha256'],contract['parent_checkpoint_sha256'])
    with pytest.raises(ConfigurationError,match='strictly after the checkpoint'):
        run(illegal,child,injected=child_ports,parent=parent,checkpoint_id='cp000001',actor_transition=illegal_binding)
    result=run(target,child,injected=child_ports,parent=parent,checkpoint_id='cp000001',actor_transition=contract)
    assert result['verification']['verified'], result['verification']
    expected=copy.deepcopy(cp['conversation']['messages'])
    if intervention:
        expected.append({'role':'user','content':'Use the supplied evidence.'})
    assert backend.bodies[0]['messages']==expected
    assert read_json(child/'lineage/parent_checkpoint.json')==cp
    assert (child/'artifacts/submission/fixture.txt').read_text()=='preserved'
    manifest=read_json(child/'manifest.json')
    assert manifest['lineage']['actor_transition']==contract
    assert manifest['usage']['model_requests_including_parent']==2
    assert manifest['usage']['known_total_tokens_including_parent']==24
    assert read_json(child/'raw/resolved_config.json')['budget']==source['budget']
    events=[json.loads(line) for line in (child/'trajectory.jsonl').read_text().splitlines()]
    event=next(e for e in events if e['kind']=='actor.transitioned')
    assert event['data']['transition']['sha256']==file_digest(child/'lineage/actor_transition.json')
    assert before=={str(p.relative_to(parent)):file_digest(p) for p in parent.rglob('*') if p.is_file()}


def test_same_actor_cold_resume_still_needs_no_transition(tmp_path):
    source,_=configs(); parent=tmp_path/'parent'
    ports={'benchmark':Benchmark(),'model':Source(source['model']),
        'runtime':Runtime(),'evaluation':FixtureEvaluator()}
    assert run(source,parent,injected=ports)['verification']['verified']
    cp=read_json(parent/'checkpoints/cp000001.json')
    backend=Source(source['model'])
    result=run(source,tmp_path/'child',injected={**ports,'model':backend,'runtime':Runtime()},
        parent=parent,checkpoint_id='cp000001')
    assert result['verification']['verified']
    assert backend.bodies[0]['messages']==cp['conversation']['messages']
    assert 'actor_transition' not in read_json(tmp_path/'child/manifest.json')['lineage']


@pytest.mark.parametrize('cold_resume',[False,True])
def test_trusted_opencode_factory_with_installed_plugins_disabled(tmp_path,monkeypatch,cold_resume):
    import importlib
    from c3_sandbox.l1.core.registry import Registry
    orchestrator=importlib.import_module('c3_sandbox.l1.core.orchestrator')
    source,target=configs()
    assert target['plugins']['load_installed'] is False
    parent=tmp_path/'parent'
    if cold_resume:
        # Exercise production construction without making any external calls.
        with monkeypatch.context() as setup:
            setup.setattr(orchestrator,'components',lambda cfg: {
                'benchmark':Benchmark(),'model':Source(cfg['model']),
                'runtime':Runtime(),'evaluation':FixtureEvaluator()})
            assert run(source,parent)['verification']['verified']
        source=read_json(parent/'raw/resolved_config.json')
        contract=binding(source,target,file_digest(parent/'evidence_index.json'),
            file_digest(parent/'checkpoints/cp000001.json'))
    for section,factory in [('benchmark',Benchmark),('runtime',Runtime),('evaluation',FixtureEvaluator)]:
        module=importlib.import_module(f'c3_sandbox.l1.{"benchmarks" if section=="benchmark" else section}.registry')
        def registry(section=section,factory=factory):
            instance=Registry('fixture-'+section)
            instance.register(target[section]['adapter'],lambda settings:factory())
            return instance
        monkeypatch.setattr(module,'registry',registry)
    def forbidden_load(self):
        pytest.fail('Installed entry points must remain disabled')
    monkeypatch.setattr(Registry,'load_installed',forbidden_load)
    if not cold_resume:
        ports=orchestrator.components(target)
        assert type(ports['model']) is OpenCodeChatBackend
        assert ports['model'].identity()['provider']=='opencode-go'
    else:
        def done(self,body,timeout):
            return HTTPResult(200,canonical({'choices':[{'message':{'role':'assistant','content':'Done'},
                'finish_reason':'stop'}],'usage':{'total_tokens':12},'model':self.config['model']}),{})
        monkeypatch.setattr(OpenCodeChatBackend,'send',done)
        result=run(target,tmp_path/'child',parent=parent,checkpoint_id='cp000001',actor_transition=contract)
        assert result['verification']['verified'],result['verification']
        manifest=read_json(tmp_path/'child/manifest.json')
        assert manifest['status']['termination']=='assistant_finished'
        assert manifest['agent']['identity']['adapter']=='opencode-chat'
