import copy
import json
from pathlib import Path
import shutil
import pytest
from c3_sandbox.l1.core.orchestrator import run
from c3_sandbox.l1.core.contracts import HTTPResult, EvidenceError
from c3_sandbox.l1.evidence.live import verify,scan
from c3_sandbox.l1.storage import read_json,canonical
from .conftest import ScriptedGemini,TrustedFixtureRuntime,FixtureEvaluator,gemini_call,gemini_done


def test_complete_evidence_independent_of_model_quality(tmp_path,config,components):
    out=tmp_path/'episode'; result=run(config,out,injected=components)
    assert result['verification']['verified'],result
    assert result['status']['termination']=='assistant_finished'
    m=read_json(out/'manifest.json')
    assert m['recording_mode']=='test_fixture'
    assert not result['ready_for_live_claim']
    assert m['counts']['model_requests']==2 and m['counts']['tool_results']==1
    assert m['counts']['observations_delivered']==1
    assert m['counts']['snapshots']==3
    events=scan(out/'trajectory.jsonl')['events']
    raw=next(e for e in events if e['kind']=='tool.result')
    assert len((out/raw['data']['stdout']['path']).read_text())>20
    # Only the delivered observation is truncated; raw stdout is unchanged.
    assert components['model'].bodies[-1]['contents'][-1]['parts'][0]['functionResponse']['response']['observation_truncation']
    assert read_json(out/'artifacts/inventory.json')['files']
    moved=tmp_path/'moved';shutil.copytree(out,moved)
    assert verify(moved)['verified']

@pytest.mark.parametrize('outcome',['success','partial','failure'])
def test_multi_outcome_saved(tmp_path,config,components,outcome):
    components['evaluation']=FixtureEvaluator(outcome)
    r=run(config,tmp_path/outcome,injected=components)
    assert r['verification']['verified'] and r['status']['task_outcome']==outcome


def test_last_tool_not_fabricated_as_delivered(tmp_path,config,components):
    config['budget']['max_model_requests']=1
    r=run(config,tmp_path/'budget',injected=components)
    assert r['verification']['verified'],r
    assert r['status']['termination']=='request_budget'
    m=read_json(tmp_path/'budget/manifest.json')
    assert m['counts']['tool_results']==1 and m['counts']['observations_delivered']==0


def test_protocol_error_retained_and_graded(tmp_path,config,components):
    components['model']=ScriptedGemini(config['model'],[gemini_call('imaginary_tool')])
    r=run(config,tmp_path/'protocol',injected=components)
    assert r['verification']['verified'],r
    assert r['status']['protocol_validity']=='invalid'
    assert r['status']['evaluation_validity']=='valid'
    assert not components['runtime'].executed


def test_tool_error_retained(tmp_path,config,components):
    components['runtime']=TrustedFixtureRuntime(fail=True)
    r=run(config,tmp_path/'toolerror',injected=components)
    assert r['verification']['verified'],r
    events=scan(tmp_path/'toolerror/trajectory.jsonl')['events']
    assert any(e['kind']=='tool.error' for e in events)
    assert not any(e['kind']=='tool.result' for e in events)


def test_grader_exception_does_not_erase_agent_stop(tmp_path,config,components):
    components['evaluation']=FixtureEvaluator(fail=True)
    r=run(config,tmp_path/'badgrade',injected=components)
    assert r['verification']['verified'],r
    assert r['status']['termination']=='assistant_finished' and r['status']['evaluation_validity']=='invalid'

@pytest.mark.parametrize('receipt',[
 HTTPResult(429,b'{"error":"limited"}',{'transport_error':None,'body_truncated':False}),
 HTTPResult(None,b'',{'transport_error':{'type':'TimeoutError'},'body_truncated':False}),
 HTTPResult(200,b'{"candidates":',{'transport_error':None,'body_truncated':True}),
])
def test_transport_problems_preserved(tmp_path,config,components,receipt):
    components['model']=ScriptedGemini(config['model'],[receipt])
    r=run(config,tmp_path/'http',injected=components)
    assert r['verification']['verified'],r
    assert r['status']['termination']=='model_transport_error'
    assert next((tmp_path/'http/raw/model').rglob('response.bin')).read_bytes()==receipt.body


def test_malformed_response_preserved(tmp_path,config,components):
    components['model']=ScriptedGemini(config['model'],[HTTPResult(200,b'garbage',{'transport_error':None})])
    r=run(config,tmp_path/'malformed',injected=components)
    assert r['verification']['verified'],r
    assert r['status']['protocol_validity']=='invalid'


def test_partial_parallel_batch_records_skips(tmp_path,config,components):
    config['budget']['max_tool_calls']=1
    value=gemini_call();second=copy.deepcopy(value['candidates'][0]['content']['parts'][0]);second['functionCall']['id']='n2'
    value['candidates'][0]['content']['parts'].append(second)
    components['model']=ScriptedGemini(config['model'],[value])
    r=run(config,tmp_path/'parallel',injected=components)
    assert r['verification']['verified'],r
    assert r['status']['termination']=='tool_budget'
    events=scan(tmp_path/'parallel/trajectory.jsonl')['events']
    assert len([e for e in events if e['kind']=='tool.skipped'])==1
    assert len(read_json(tmp_path/'parallel/checkpoints/index.json')['checkpoints'])==1


def test_cold_resume_into_new_branch(tmp_path,config,components):
    parent=tmp_path/'parent';r=run(config,parent,injected=components)
    assert r['verification']['verified']
    cp='cp000001'
    model=ScriptedGemini(config['model'],[gemini_call(native_id='resume-native'),gemini_done()])
    child=tmp_path/'child';components['model']=model
    r=run(config,child,injected=components,parent=parent,checkpoint_id=cp)
    assert r['verification']['verified'],r
    m=read_json(child/'manifest.json')
    assert m['lineage']['parent_checkpoint_id']==cp
    assert m['counts']['model_requests']==2 and m['counts']['tool_results']==1
    assert model.bodies[0]['contents'][-1]['parts'][0]['functionResponse']['id']=='native-call-1'
    assert verify(parent)['verified']


def test_resume_refuses_fixture_in_production(tmp_path,config,components):
    parent=tmp_path/'p';run(config,parent,injected=components)
    with pytest.raises(EvidenceError):run(config,tmp_path/'c',parent=parent,checkpoint_id='cp000000')


def test_no_overwrite_existing_output(tmp_path,config,components):
    out=tmp_path/'e';out.mkdir()
    with pytest.raises(FileExistsError):run(config,out,injected=components)


def test_tool_timeout_is_recorded_not_model_failure_inference(tmp_path,config,components):
    components['runtime']=TrustedFixtureRuntime(timeout=True)
    r=run(config,tmp_path/'timeout',injected=components)
    assert r['verification']['verified'] and r['status']['termination']=='assistant_finished'
    raw=next((tmp_path/'timeout/raw/tools').rglob('result.json'))
    assert read_json(raw)['observation']['stop_reason']=='timeout'


def test_capped_output_marks_incomplete(tmp_path,config,components):
    components['runtime']=TrustedFixtureRuntime(cap=True)
    r=run(config,tmp_path/'cap',injected=components)
    assert r['verification']['verified'] and r['status']['evidence_completeness']=='incomplete'


def test_symlink_state_fail_closed(tmp_path,config,components):
    components['runtime']=TrustedFixtureRuntime(symlink=True)
    r=run(config,tmp_path/'link',injected=components)
    assert r['verification']['verified'],r
    assert r['status']['evidence_completeness']=='incomplete'
    assert not any(p.is_symlink() for p in (tmp_path/'link').rglob('*'))


def test_cold_resume_under_setgid_host_directory(tmp_path,config,components):
    import os
    parent_dir=tmp_path/'shared';parent_dir.mkdir();os.chmod(parent_dir,0o2775)
    first=parent_dir/'p';run(config,first,injected=components)
    components['model']=ScriptedGemini(config['model'])
    result=run(config,parent_dir/'c',injected=components,parent=first,checkpoint_id='cp000001')
    assert result['verification']['verified'],result
    assert read_json(parent_dir/'c/manifest.json')['counts']['tool_results']==1


def test_special_permission_state_not_qualified(tmp_path,config,components):
    import os
    class SpecialMode(TrustedFixtureRuntime):
        def execute(self,call,workspace,timeout):
            result=super().execute(call,workspace,timeout)
            os.chmod(workspace/'submission',0o2755)
            return result
    components['runtime']=SpecialMode()
    package=tmp_path/'mode';r=run(config,package,injected=components)
    assert r['verification']['verified'],r
    cp=read_json(package/'checkpoints/cp000001.json')
    assert not cp['qualification']['eligible']
    assert 'unsupported_special_permission_bits' in cp['qualification']['reason_codes']
