from pathlib import Path
import copy
import json
import os
import pytest
from l2_sandbox.pipeline import analyze
from l2_sandbox.config import load,validate
from l2_sandbox.io import L2Error,canonical,read_json,read_jsonl
from l2_sandbox.package import verify
from l2_sandbox.analyst.backend import doctor
from conftest import FixtureBackend,diagnosis_from_tool_results


def test_native_port_retrieval_and_frozen_prediction(source,tmp_path,config):
    config['mode']='analyst';backend=FixtureBackend();out=tmp_path/'analyst'
    r=analyze(source,out,config,backend=backend)
    assert r['status']=='completed_analyst',r
    assert r['mode']=='test_fixture' and r['counts']['analyst_model_requests']==2 and r['verification']['verified'],r
    trace=read_jsonl(out/'analyst_trace.jsonl')
    assert any(x['kind']=='output.accepted' for x in trace)
    assert len(backend.requests)==2
    assert backend.requests[1]['contents'][1]['parts'][0]['thoughtSignature']=='test-signature'
    assert read_json(out/'diagnosis.json')['hypotheses'][0]['verdict']=='untested'
    assert b'PRIVATE_RUBRIC_SENTINEL' not in (out/'raw/analyst/r001/request.bin').read_bytes()


def test_model_not_implicitly_authorized(source,tmp_path,config):
    config['mode']='analyst'
    with pytest.raises(L2Error,match='allow-model'):analyze(source,tmp_path/'forbidden',config)
    assert not (tmp_path/'forbidden').exists()


def test_invalid_prediction_is_repaired_not_hidden(source,tmp_path,config):
    config['mode']='analyst'
    def malformed(body):
        d=diagnosis_from_tool_results(body);d['hypotheses'][0]['verdict']='supported'
        return {'calls':[{'name':'submit_diagnosis','args':{'diagnosis_json':canonical(d).decode()}}]}
    backend=FixtureBackend([{'calls':[{'name':'get_requirement_evidence','args':{'limit':30}}]},malformed])
    out=tmp_path/'repair';r=analyze(source,out,config,backend=backend)
    assert r['status']=='completed_analyst' and r['verification']['verified'],r
    trace=read_jsonl(out/'analyst_trace.jsonl')
    assert sum(x['kind']=='output.rejected' for x in trace)==1
    assert r['counts']['analyst_model_requests']==3
    assert len(list((out/'raw/analyst/submissions').glob('*.txt')))==2


def test_unread_fact_and_fake_event_rejected(source,tmp_path,config):
    config['mode']='analyst';config['budget']['max_output_repairs']=0
    def bad(body):
        d=diagnosis_from_tool_results(body);d['localizations'][0]['event_ids']=['imaginary:event']
        return {'calls':[{'name':'submit_diagnosis','args':{'diagnosis_json':canonical(d).decode()}}]}
    backend=FixtureBackend([{'calls':[{'name':'get_requirement_evidence','args':{'limit':30}}]},bad])
    out=tmp_path/'bad';r=analyze(source,out,config,backend=backend)
    assert r['status']=='analysis_failed' and r['verification']['verified'],r
    assert read_json(out/'diagnosis.json')['hypotheses']==[]
    assert 'repair budget' in r['failure']['message']


def test_prompt_injection_shell_request_is_denied(source,tmp_path,config):
    config['mode']='analyst'
    backend=FixtureBackend([{'calls':[{'name':'run','args':{'command':'touch /tmp/L2_SHOULD_NEVER_EXIST'}}]},
                            {'calls':[{'name':'get_requirement_evidence','args':{'limit':30}}]}])
    out=tmp_path/'denied';r=analyze(source,out,config,backend=backend)
    assert r['status']=='completed_analyst' and r['verification']['verified'],r
    first=read_json(out/'raw/analyst/tools/r000_c000.json')
    assert first['error']=='AccessDenied' and first['not_executed']
    assert not Path('/tmp/L2_SHOULD_NEVER_EXIST').exists()


def test_raw_malformed_model_response_retained(source,tmp_path,config):
    config['mode']='analyst';out=tmp_path/'malformed'
    r=analyze(source,out,config,backend=FixtureBackend([b'not-json']))
    assert r['status']=='analysis_failed' and r['verification']['verified'],r
    assert (out/'raw/analyst/r000/response.bin').read_bytes()==b'not-json'


def test_network_error_does_not_fabricate_response(source,tmp_path,config):
    config['mode']='analyst';out=tmp_path/'httpfail'
    r=analyze(source,out,config,backend=FixtureBackend([TimeoutError('fixture timeout')]))
    assert r['status']=='analysis_failed' and r['verification']['verified'],r
    assert (out/'raw/analyst/r000/request.bin').exists()
    assert not (out/'raw/analyst/r000/response.bin').exists()


def test_request_budget_stops(source,tmp_path,config):
    config['mode']='analyst';config['budget']['max_model_requests']=1
    r=analyze(source,tmp_path/'budget',config,backend=FixtureBackend())
    assert r['status']=='analysis_failed' and r['failure']['type']=='BudgetExceeded'
    assert r['counts']['analyst_model_requests']==1 and r['verification']['verified']


def test_structured_once_has_no_retrieval(source,tmp_path,config):
    config['mode']='structured-once'
    def answer(body):
        prompt=json.loads(body['contents'][0]['parts'][0]['text'])
        proxy={'contents':[{'parts':[{'functionResponse':{'response':{'items':prompt['facts']}}}]}]}
        d=diagnosis_from_tool_results(proxy)
        return {'calls':[{'name':'submit_diagnosis','args':{'diagnosis_json':canonical(d).decode()}}]}
    r=analyze(source,tmp_path/'once',config,backend=FixtureBackend([answer]))
    assert r['status']=='completed_analyst' and r['verification']['verified'],r
    assert read_json(tmp_path/'once/cost.json')['retrieval_calls']==0

@pytest.mark.parametrize('key,value',[('analysis_mode','prefix'),('source_policy','partial'),('mode','arbitrary_shell'),('expected_source_index_sha256','wrong')])
def test_unsupported_configuration(config,key,value):
    config[key]=value
    with pytest.raises(L2Error):validate(config)


def test_inline_secret_not_allowed(config):
    config['model']['parameters']={'api_key':'secret'}
    with pytest.raises(L2Error):validate(config)


def test_budget_integer_required(config):
    config['budget']['max_model_requests']=True
    with pytest.raises(L2Error):validate(config)


def test_unknown_config_section(config):
    config['private_rubric']=True
    with pytest.raises(L2Error):validate(config)

@pytest.mark.live
@pytest.mark.skipif(not(os.environ.get('L2_LIVE_CONFIG') and os.environ.get('L2_LIVE_SOURCE')),
                    reason='Explicit real Analyst config/source required; no API/model credentials supplied by this delivery')
def test_real_analyst_opt_in(tmp_path):
    config=load(Path(os.environ['L2_LIVE_CONFIG']))
    assert config['mode']=='analyst'
    r=analyze(Path(os.environ['L2_LIVE_SOURCE']),tmp_path/'real_analyst',config,allow_model=True)
    assert r['status']=='completed_analyst' and r['verification']['verified'],r
    assert r['counts']['analyst_model_requests']>0 and r['counts']['analyst_tool_results']>0


def test_raw_analyst_bytes_cannot_be_resealed_as_valid(source,tmp_path,config):
    from l2_sandbox.package import seal
    config['mode']='analyst';out=tmp_path/'raw'
    r=analyze(source,out,config,backend=FixtureBackend())
    assert r['verification']['verified'],r
    (out/'raw/analyst/r000/request.bin').write_bytes(b'{"forged":true}')
    (out/'evidence_index.json').unlink();seal(out)
    assert not verify(out)['verified']


def test_failed_attempt_keeps_usage_from_successful_prior_responses(source,tmp_path,config):
    config['mode']='analyst';config['budget']['max_model_requests']=1
    out=tmp_path/'failed_with_usage';r=analyze(source,out,config,backend=FixtureBackend())
    assert r['status']=='analysis_failed'
    assert read_json(out/'cost.json')['usage_records']==[{'totalTokenCount':10}]
