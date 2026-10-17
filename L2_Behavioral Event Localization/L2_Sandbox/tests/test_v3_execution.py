"""Exercise the actual stage drivers with transport/native adapters replaced only at boundaries."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'L1_sandbox/C3_Experimental_Sandbox/src'))
from l2_sandbox import v3_execution as ex
from l2_sandbox.v05 import build_ladder_intervention_messages, build_interleaved_arm_schedule
import v3_d1_ladder as d1
import v3_d2_ceiling as d2
import v3_d3_analysts as d3


def fixture_plan():
    messages = build_ladder_intervention_messages(explicit_requirements=[], implicit_requirements=[])
    u = dict(unit_id='ep::item', episode_id='ep', task_id='task', family_id='semantic-family', unit_level='rubric_item', defect_name='target', rule_source='explicit', deduction_weight=1.0, stratum_id='a', inclusion_prob=1.0, anchor_k=2, anchor_type='voluntary_pre_finish', remaining_budget=3, parent_identity={'index':'frozen'}, eligible_arms=['P','D','S'], ineligible_arms=['R_explicit','R_implicit','R_single'], arm_messages=messages, comparison_ids=['D_vs_P','S_vs_P','S_vs_D'])
    u['schedule'] = build_interleaved_arm_schedule(u['eligible_arms'], 2, anchor_k=2, remaining_budget=3)
    p = dict(stage='D1', execution_ready=True, unit_schedules=[u], primary_test_family=dict(size=3,comparison_ids=u['comparison_ids']), d0_candidate_parameters=dict(n_per_arm=2,delta=.1,tau=.2,alpha=.05,ceiling_threshold=.05,min_replicates_per_arm=2), budget_and_stop_conditions=dict(max_actor_requests_cap=18), r12_passing_harm_check_plan=dict(sampled_passing_units=[]), task_ladder_message_specs={})
    return ex.seal(p, 'plan_sha256')


def freeze(plan):
    return dict(d0_frozen=True, approved_plan_sha256=plan['plan_sha256'], limits=dict(max_requests=30,max_tokens=30000,max_cost_usd=30,max_wall_seconds=60,max_invalid_attempts=2,max_tokens_per_request=10000,usd_per_token_upper_bound=.001))


def adapter(calls):
    def run(u,b,m):
        calls.append((u['unit_id'],b['arm'],b['replicate']))
        return dict(valid_attempt=True,status='completed',outcome_score=float(b['arm']!='P'),simulated_offline=True)
    return run


def test_exact_freeze_and_all_contract_fields_bound(tmp_path):
    p = fixture_plan()
    altered = json.loads(json.dumps(p)); altered['unit_schedules'][0]['remaining_budget'] = 9
    with pytest.raises(ValueError, match='seal'):
        ex.approve(altered,freeze(p))
    with pytest.raises(ValueError,match='approved_plan'):
        ex.approve(p,dict(d0_frozen=True,approved_plan_sha256='wrong'))


def test_driver_resume_and_tamper_rejection(tmp_path):
    p=fixture_plan(); calls=[]
    kwargs=dict(runs_dir=tmp_path,receipts_dir=tmp_path/'receipts',outcomes_out=tmp_path/'out.json',freeze=freeze(p),simulate_offline=True,branch_runner=adapter(calls))
    out=d1.execute_d1_plan(p,**kwargs)
    assert len(calls)==6 and out['execution_summary']['total_branches_executed']==6
    assert all(r['status']=='completed' for rs in out['evaluated_units'][0]['branch_receipts'].values() for r in rs)
    resumed=d1.execute_d1_plan(p,**kwargs)
    assert len(calls)==6 and resumed['execution_summary']['reused_cached_branches']==6
    path=next((tmp_path/'receipts').glob('*.json'))
    if path.name=='admission.json': path=next(x for x in (tmp_path/'receipts').glob('*.json') if x.name!='admission.json')
    receipt=ex.read_json(path); receipt['remaining_budget']=200; ex.write_json(path,receipt)
    with pytest.raises(ValueError): d1.execute_d1_plan(p,**kwargs)
    assert len(calls)==6


def test_failure_retained_and_never_silently_retried(tmp_path):
    p=fixture_plan(); calls=[]
    def bad(*args): calls.append(1); raise RuntimeError('transport failed')
    kwargs=dict(runs_dir=tmp_path,receipts_dir=tmp_path/'receipts',outcomes_out=tmp_path/'out.json',freeze=freeze(p),simulate_offline=True,branch_runner=bad)
    with pytest.raises(RuntimeError): d1.execute_d1_plan(p,**kwargs)
    out=ex.read_json(tmp_path/'out.json'); assert out['status']=='failed' and out['failed_attempts'][0]['error']=='transport failed'
    with pytest.raises(ValueError,match='incomplete'): d1.execute_d1_plan(p,**kwargs)
    assert len(calls)==1


def test_passing_harm_uses_same_bound_adapter_receipts(tmp_path):
    p=fixture_plan(); passing=dict(p['unit_schedules'][0],unit_id='ep::passing',arms=['P','D','S'])
    p['r12_passing_harm_check_plan']['sampled_passing_units']=[passing]
    p['task_ladder_message_specs']={'task':{'ladder_messages':passing['arm_messages']}}
    p['budget_and_stop_conditions']['max_actor_requests_cap']=36
    p=ex.seal(p,'plan_sha256'); f=freeze(p); f['limits']['max_requests']=40
    calls=[]; out=d1.execute_d1_plan(p,runs_dir=tmp_path,receipts_dir=tmp_path/'r',outcomes_out=tmp_path/'o',freeze=f,simulate_offline=True,branch_runner=adapter(calls))
    assert len(calls)==12 and len(out['passing_units'])==1
    assert out['r12_passing_harm_evaluation']['evaluated_unit_count']==1


def test_d2_rejects_missing_plan_mutation_and_simulation_gate(tmp_path):
    p=fixture_plan(); out=d1.execute_d1_plan(p,runs_dir=tmp_path,receipts_dir=tmp_path/'r',outcomes_out=tmp_path/'o',freeze=freeze(p),simulate_offline=True,branch_runner=adapter([]))
    missing=d2.evaluate_d2_from_d1_payload(out)
    assert not missing['d3_prerequisite_gate_open'] and not missing['d1_provenance_audit']['plan_binding_verified']
    report=d2.evaluate_d2_from_d1_payload(out,d1_plan_payload=p)
    assert report['d1_provenance_audit']['plan_binding_verified']
    assert not report['d3_prerequisite_gate_open']
    out['evaluated_units'][0]['arm_outcomes']['D'][0]=.123
    out=ex.seal(out)
    assert not d2.evaluate_d2_from_d1_payload(out,d1_plan_payload=p)['d1_provenance_audit']['plan_binding_verified']


def test_admission_reserves_unknown_usage_and_hard_stops(tmp_path):
    p=fixture_plan(); caps=freeze(p)['limits']; caps['max_requests']=1
    ledger=ex.Admission(tmp_path/'ledger',p['plan_sha256'],caps)
    assert ledger.acquire(5)
    ledger.reconcile(5,None)
    assert not ledger.acquire(5)
    again=ex.Admission(tmp_path/'ledger',p['plan_sha256'],caps)
    assert again.state['tokens']==10000 and again.state['requests']==1


@pytest.mark.parametrize("wallcap", [False, True])
def test_native_adapter_calls_real_orchestrator_contract_with_frozen_config(tmp_path,monkeypatch,wallcap):
    from c3_sandbox.l1.core import orchestrator
    from c3_sandbox.l1 import package
    p=tmp_path/'runs/ep'; (p/'raw').mkdir(parents=True); (p/'checkpoints').mkdir()
    cfg=dict(agent={'harness':'pi'},budget={'max_model_requests':35})
    ex.write_json(p/'raw/resolved_config.json',cfg)
    ex.write_json(p/'checkpoints/cp000002.json',{'budget_used':{'model_requests':29}})
    binding={'index':'frozen'}
    monkeypatch.setattr(ex,'identity',lambda *a:binding)
    calls=[]
    def run(config,output,**kw):
        calls.append((config,kw)); output.mkdir(); ex.write_json(output/'manifest.json',{'episode_id':'child','counts':{'model_requests':3}}); ex.write_json(output/'official_grade.json',{'official':True})
        return {'verification':{'verified':True}}
    monkeypatch.setattr(orchestrator,'run',run)
    monkeypatch.setattr(package,'verify_package',lambda *a,**k:dict(verified=True,index_sha256='child-index'))
    monkeypatch.setattr(ex,'item_score',lambda *a:.25)
    def check(self):
        if wallcap: raise RuntimeError('frozen wall-clock cap reached')
    limiter=type('Limiter',(),{'check':check})()
    out=ex.native_branch(dict(episode_id='ep',parent_identity=binding),dict(checkpoint_k=2,remaining_budget=6),dict(message='check'),runs_dir=tmp_path/'runs',output=tmp_path/'child',limiter=limiter)
    assert out['requests_used']==3
    assert bool(out['cap_exceeded']) == wallcap
    assert out['outcome_score']==.25 and calls[0][0]['agent']['intervention']==dict(at_request=30,message='check')
    assert calls[0][1]['parent']==p and calls[0][1]['request_limiter'] is limiter
    assert ex.read_json(p/'raw/resolved_config.json')==cfg


def test_d3_calls_B_C_and_seals_before_reading_outcomes(tmp_path,monkeypatch):
    base=fixture_plan(); row=base['unit_schedules'][0]
    outcome=ex.seal(dict(stage='D1',status='completed',simulated_offline=False,evaluated_units=[dict(row,terminal_y=0,arm_outcomes={'P':[0,0],'D':[1,1],'S':[1,1]})]))
    report=ex.seal(dict(d1_plan_sha256=base['plan_sha256'],mode='execute',d3_prerequisite_gate_open=True,d0_parameters={'ceiling_threshold':.05}))
    entry=dict(unit=row,B_prompt='raw prefix',C_prompt='raw prefix + facts',facts=[])
    p=ex.seal(dict(execution_ready=True,d1_plan_sha256=base['plan_sha256'],d2_report_sha256=report['artifact_sha256'],d1_outcomes_sha256=outcome['artifact_sha256'],analyst_inputs=[entry],analyst_model='test-model',majority_label='unrecovered_under_protocol',max_output_tokens=64),'plan_sha256')
    outcomes_path=tmp_path/'outcomes'; ex.write_json(outcomes_path,outcome)
    calls=[]
    def call(**kw):
        calls.append(kw['analyst_group']); assert not (tmp_path/'r/sealed_predictions.json').exists()
        return dict(prediction=dict(p_prob=.1,d_prob=.8,s_prob=.5,r_explicit_prob=.5,r_implicit_prob=.5,recommended_arm='D',predicted_label='delivery_responsive',confidence=.8,abstain=False,evidence_steps=[]),usage={'total_tokens':10},raw_response={'mocked':True})
    original=d3.read_json
    def read(path):
        if path==outcomes_path: assert (tmp_path/'r/sealed_predictions.json').exists()
        return original(path)
    monkeypatch.setattr(d3,'read_json',read)
    monkeypatch.setattr(d2,'evaluate_d2_from_d1_payload',lambda *a,**k: {'d1_provenance_audit':{'plan_binding_verified':True}})
    out=d3.execute_d3_plan(p,freeze=freeze(p),d1_plan=base,d2_report=report,outcomes_path=outcomes_path,receipts_dir=tmp_path/'r',results_out=tmp_path/'result',llm_call=call)
    assert calls==['B','C'] and set(out['results'])=={'M','A','B','C'}
    d3.execute_d3_plan(p,freeze=freeze(p),d1_plan=base,d2_report=report,outcomes_path=outcomes_path,receipts_dir=tmp_path/'r',results_out=tmp_path/'result',llm_call=call)
    assert calls==['B','C']


def test_simulation_without_adapter_never_enters_native(tmp_path, monkeypatch):
    calls=[]
    monkeypatch.setattr(d1, 'native_branch', lambda *a,**k: calls.append(1))
    p=fixture_plan()
    with pytest.raises(ValueError, match='native execution is forbidden'):
        d1.execute_d1_plan(p,runs_dir=tmp_path,receipts_dir=tmp_path/'r',outcomes_out=tmp_path/'o',freeze=freeze(p),simulate_offline=True)
    assert calls == []


def test_official_item_endpoint_uses_target_not_whole_score(tmp_path, monkeypatch):
    from c3_sandbox.l1.evaluation import study
    monkeypatch.setattr(study,'study_outcome', lambda *a,**k: dict(study_valid=True,study_outcome_y=.9))
    ex.write_json(tmp_path/'official_grade.json',dict(official_raw=dict(dim1_pass=True,dim2_items=[dict(rule='target',max_delta=4,delta=1),dict(rule='other',max_delta=6,delta=6)])))
    assert ex.item_score(tmp_path,dict(unit_level='rubric_item',defect_name='target')) == .25
    assert ex.item_score(tmp_path,dict(unit_level='whole_run')) == .9
    with pytest.raises(ValueError,match='missing or ambiguous'):
        ex.item_score(tmp_path,dict(unit_level='rubric_item',defect_name='missing'))


@pytest.mark.parametrize('mutation', ['simulation', 'family_id', 'deduction_weight'])
def test_d2_rejects_resealed_mode_or_scientific_metadata_laundering(tmp_path, mutation):
    p=fixture_plan()
    out=d1.execute_d1_plan(p,runs_dir=tmp_path,receipts_dir=tmp_path/'r',outcomes_out=tmp_path/'o',freeze=freeze(p),simulate_offline=True,branch_runner=adapter([]))
    if mutation == 'simulation':
        out['simulated_offline']=False
    else:
        out['evaluated_units'][0][mutation] = 'invented-family' if mutation=='family_id' else 999
    out=ex.seal(out)
    report=d2.evaluate_d2_from_d1_payload(out,d1_plan_payload=p)
    assert not report['d1_provenance_audit']['plan_binding_verified']
    assert not report['d3_prerequisite_gate_open']


def test_d3_majority_and_best_fixed_are_development_derived(tmp_path, monkeypatch):
    p=fixture_plan(); ex.write_json(tmp_path/'plan',p)
    report=ex.seal(dict(d1_plan_sha256=p['plan_sha256'],mode='execute',status='evaluated',d3_prerequisite_gate_open=True,simulated_offline=False))
    ex.write_json(tmp_path/'d2',report)
    monkeypatch.setattr(d3,'_analyst_input',lambda u,*a,**k:dict(unit=u))
    baseline=ex.seal(dict(split='development', actor_model='Qwen/Qwen3.6-27B',harness='pi',source_artifact_sha256='frozen-calibration',training_unit_ids=['calibration-other'],label_counts={'delivery_responsive':9,'unrecovered_under_protocol':1},deduction_weighted_label_counts={'delivery_responsive':7.5,'unrecovered_under_protocol':.5},fixed_arm_development_scores={'P':.1,'D':.9,'S':.4}))
    ex.write_json(tmp_path/'baseline',baseline)
    plan=d3.build_d3_plan(d1_plan_path=tmp_path/'plan',d2_report_path=tmp_path/'d2',baseline_path=tmp_path/'baseline')
    assert not plan['execution_ready'] and plan['majority_label'] is None
    assert any('source_plan_path' in x for x in plan['readiness_errors'])
    absent=d3.build_d3_plan(d1_plan_path=tmp_path/'plan',d2_report_path=tmp_path/'d2',baseline_path=tmp_path/'absent')
    assert not absent['execution_ready'] and absent['majority_label'] is None


def test_real_helper_preserves_http_raw_and_driver_validates_schema(monkeypatch):
    class Response:
        status=200
        def __enter__(self): return self
        def __exit__(self,*a): pass
        def read(self): return b'{"id":"test","usage":{"total_tokens":4},"choices":[{"message":{"content":"invalid-json"}}]}'
    monkeypatch.setattr(d3.urllib.request,'urlopen',lambda *a,**k:Response())
    result=d3.call_real_llm_analyst(analyst_group='B',unit_id='u',system_prompt='s',user_prompt='p',api_key='local-test-dummy',max_tokens=16)
    assert result['raw_response']['id']=='test' and result['prediction'] is None
    assert result['request_payload']['max_tokens']==16 and 'Authorization' not in json.dumps(result)


@pytest.mark.parametrize('stage', [d1,d2,d3])
def test_conflicting_cli_flags_refuse_before_any_stage_work(monkeypatch,stage):
    monkeypatch.setattr(sys,'argv',['stage','--plan-only','--execute'])
    with pytest.raises(SystemExit) as exc:
        stage.main()
    assert exc.value.code==2


def test_M_source_recomputed_from_disjoint_verified_calibration(tmp_path,monkeypatch):
    source_plan=fixture_plan(); row=dict(source_plan['unit_schedules'][0],unit_id='cal::item',episode_id='cal',family_id='cal-family',parent_identity={'config_sha256':'frozen-config'},terminal_y=0,arm_outcomes={'P':[0,0],'D':[1,1],'S':[0,0]})
    source_plan['source_dataset']={'runs_dir':str(tmp_path/'runs')}
    source_plan=ex.seal(source_plan,'plan_sha256')
    outcomes=ex.seal(dict(status='completed',simulated_offline=False,evaluated_units=[row]))
    ex.write_json(tmp_path/'source-plan',source_plan);ex.write_json(tmp_path/'source-outcomes',outcomes)
    ex.write_json(tmp_path/'runs/cal/raw/resolved_config.json',dict(model={'model':'Qwen/Qwen3.6-27B'},agent={'harness':'pi'}))
    baseline=ex.seal(dict(split='development',actor_model='Qwen/Qwen3.6-27B',harness='pi',source_plan_path=str(tmp_path/'source-plan'),source_outcomes_path=str(tmp_path/'source-outcomes'),source_plan_sha256=source_plan['plan_sha256'],source_outcomes_sha256=outcomes['artifact_sha256'],source_config_sha256s=['frozen-config'],calibration_pool={'episode_ids':['cal'],'family_ids':['cal-family']},training_unit_ids=['cal::item'],label_counts={'delivery_responsive':1},deduction_weighted_label_counts={'delivery_responsive':1.0},fixed_arm_development_scores={'P':0,'D':1,'S':0}))
    monkeypatch.setattr(d2,'evaluate_d2_from_d1_payload',lambda *a,**k: {'d1_provenance_audit':{'plan_binding_verified':True,'rejected_unit_count':0}})
    monkeypatch.setattr(d3,'adjudicate_e2_outcomes',lambda *a,**k:(None,'delivery_responsive','adjudicated'))
    assert d3.validate_development_baseline(baseline,fixture_plan()['unit_schedules']) == ('delivery_responsive','D')
    mutated=dict(baseline,label_counts={'unrecovered_under_protocol':999});mutated=ex.seal(mutated)
    with pytest.raises(ValueError,match='frequencies'):d3.validate_development_baseline(mutated,fixture_plan()['unit_schedules'])


def test_d3_policy_validator_excludes_attribution_only_hidden_actions():
    pred=dict(p_prob=.1,d_prob=.1,s_prob=.1,r_explicit_prob=.1,r_implicit_prob=.9,recommended_arm='R_implicit',predicted_label='implicit_rule_responsive',confidence=.9,abstain=False,evidence_steps=[])
    with pytest.raises(ValueError,match='action'):
        d3._validate_prediction(pred,'u','C',['P','D','S','R_explicit'])
    pred['recommended_arm']='P';pred['abstain']=True
    assert d3._validate_prediction(pred,'u','C',['P','D','S','R_explicit'])['r_implicit_prob']==.9
