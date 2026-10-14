"""Immutable experiment plans; only unchanged cold resume has a built-in executor.

Information injection is deliberately blocked against L1's unchanged-resume
contract. A text plan is not an implemented intervention, and a historical retry
is not accepted as a prospective randomized control.
"""
from __future__ import annotations
import copy
from pathlib import Path
from .. import SCHEMA_VERSION
from ..io import (L2Error,IntegrityError,digest,sha_file,read_json,safe,now,write_json,write_new,inventory)

TEMPLATES={
 'resume_unchanged':{'implemented':True,'effect_contrast':False,'description':'Fresh continuation of the same checkpoint under inherited policy/budget'},
 'provide_public_state_summary':{'implemented':False,'effect_contrast':True,'description':'Requires an audited L1 experiment-injection API; not a fabricated past Observation'},
 'generic_reminder':{'implemented':False,'effect_contrast':True,'description':'Requires explicit injected-message provenance and matched-cost protocol'},
}
ENDPOINTS=('submission_present','official_dim1','official_normalized_score')


def compile_plan(analysis:Path,source:Path,checkpoint:str,output:Path,*,hypothesis_id=None,
                 template='resume_unchanged',replicates=3,endpoint='submission_present'):
    from ..package import verify,seal
    from ..evidence.l1 import L1Reader
    if template not in TEMPLATES or endpoint not in ENDPOINTS:raise L2Error('Unknown intervention template or endpoint')
    if type(replicates) is not int or not 1<=replicates<=64:raise L2Error('replicates must be integer 1..64')
    if not __import__('re').fullmatch(r'cp[0-9]{6}',checkpoint):raise L2Error('Explicit checkpoint cpXXXXXX required')
    av=verify(analysis)
    if not av['verified'] or not av['status'].startswith('completed'):raise L2Error('A verified accepted diagnosis is required')
    am=read_json(analysis/'manifest.json');r=L1Reader(source,expected_index=am['source_index_sha256'])
    if r.version!='0.2.0':raise L2Error('Legacy source has no qualified live experiment bridge')
    diag=read_json(analysis/'diagnosis.json');hyp=None
    if hypothesis_id:
        hyp=next((h for h in diag['hypotheses'] if h['hypothesis_id']==hypothesis_id),None)
        if hyp is None:raise L2Error('Unknown hypothesis ID')
    if template!='resume_unchanged' and hyp is None:raise L2Error('A mechanism experiment requires a frozen hypothesis')
    cp_path=f'checkpoints/{checkpoint}.json';cp=r.json(cp_path)
    cfg=r.json('raw/resolved_config.json');remaining={
        'model_requests':cfg['budget']['max_model_requests']-cp['budget_used']['model_requests'],
        'tool_calls':cfg['budget']['max_tool_calls']-cp['budget_used']['tool_calls'],
        'wall_seconds':cfg['budget']['wall_time_seconds']-cp['budget_used']['agent_seconds']}
    blockers=[]
    if not cp['qualification']['eligible']:blockers+=['source_checkpoint_ineligible']
    if any(v<=0 for v in remaining.values()):blockers+=['checkpoint_budget_exhausted']
    if not TEMPLATES[template]['implemented']:blockers+=['L1_unchanged_resume_does_not_support_explicit_information_injection']
    if r.manifest.get('recording_mode')!='live':blockers+=['source_not_live_recording']
    # No final grading validity gate: a failed task may have a qualified checkpoint.
    arms=[{'arm_id':'control','template':'resume_unchanged','replicates':replicates}]
    if template!='resume_unchanged':arms.append({'arm_id':'treatment','template':template,'replicates':replicates})
    plan={'schema_version':SCHEMA_VERSION,'package_kind':'experiment_plan','created_at':now(),
        'analysis_id':am['analysis_id'],'analysis_index_sha256':sha_file(analysis/'evidence_index.json'),
        'prediction_commitment_sha256':sha_file(analysis/'predictions/frozen_manifest.json'),
        'source_index_sha256':r.index_sha256,'source_episode_id':r.episode_id,
        'checkpoint_id':checkpoint,'checkpoint_sha256':r.files[cp_path]['sha256'],
        'checkpoint_snapshot_id':cp['snapshot_id'],'config_sha256':digest(cfg),'hypothesis_id':hypothesis_id,
        'hypothesis_sha256':digest(hyp),'template':template,'arms':arms,'endpoint':endpoint,
        'assignment':'fixed_predeclared_alternating_arms_no_adaptive_stopping','replicates_per_arm':replicates,
        'remaining_budget_per_branch':remaining,'budget_policy':'inherit_parent_budget_no_reset',
        'failure_policy':'retain_all_attempts; endpoint undefined on infrastructure_invalid; report undefined count; no causal verdict if missing outcomes',
        'plan_state':'blocked' if blockers else 'frozen','blockers':blockers,'hypothesis_verdict':'untested',
        'execution_authorized':False,'contrast_identifiable':len(arms)>1 and not blockers,
        'private_rubric_injection':False,'trusted_external_timestamp':False}
    output=Path(output).absolute()
    for other in (analysis,source):
        if output.resolve().is_relative_to(Path(other).resolve()) or Path(other).resolve().is_relative_to(output.resolve()):
            raise L2Error('Plan output must be separate from source and diagnosis')
    output.mkdir(parents=True,exist_ok=False,mode=0o700)
    write_json(output,'plan.json',plan);write_json(output,'hypothesis.json',hyp)
    write_new(output,'prediction_commitment.json',(analysis/'predictions/frozen_manifest.json').read_bytes())
    write_new(output,'analysis_index.json',(analysis/'evidence_index.json').read_bytes())
    write_new(output,'source_index.json',r.read('evidence_index.json'))
    write_json(output,'checkpoint_qualification.json',{'checkpoint_id':checkpoint,'qualification':cp['qualification'],'budget_used':cp['budget_used']})
    write_new(output,'plan.md',(
        '# L2 冻结实验计划\n\n状态：'+plan['plan_state']+'。模板：'+template+'。\n\n'
        '这是计划，不是实验结果；调用 analyze/plan 不会付费调用被测模型。\n\n'
        '仅 resume_unchanged 有内置执行桥；它能建立新基线续跑，但不能单独识别干预效应。\n\n'
        '阻塞项：'+str(blockers)+'\n').encode())
    index=seal(output,'experiment_plan')
    return {'output':str(output),'index_sha256':index,'plan':plan}


def verify_experiment_package(root,idx,index_hash):
    kind=idx['package_kind']
    try:
        if kind=='experiment_plan':
            plan=read_json(root/'plan.json')
            if plan['package_kind']!=kind or plan['template'] not in TEMPLATES:raise IntegrityError('Invalid plan kind/template')
            if sha_file(root/'source_index.json')!=plan['source_index_sha256'] or sha_file(root/'analysis_index.json')!=plan['analysis_index_sha256']:
                raise IntegrityError('Plan source anchors differ')
            if sha_file(root/'prediction_commitment.json')!=plan['prediction_commitment_sha256'] or digest(read_json(root/'hypothesis.json'))!=plan['hypothesis_sha256']:
                raise IntegrityError('Frozen prediction binding differs')
            if plan['execution_authorized'] or plan['hypothesis_verdict']!='untested':raise IntegrityError('Frozen plan cannot assert authorized execution or causal support')
            if not TEMPLATES[plan['template']]['implemented'] and plan['plan_state']!='blocked':raise IntegrityError('Unsupported intervention must remain blocked')
            if plan['plan_state']=='frozen':
                q=read_json(root/'checkpoint_qualification.json')
                if plan['blockers'] or not q['qualification']['eligible'] or any(v<=0 for v in plan['remaining_budget_per_branch'].values()):
                    raise IntegrityError('Blocked/ineligible plan cannot be frozen as executable')
            if len(plan['arms'])==1 and plan['contrast_identifiable']:raise IntegrityError('Single arm has no intervention contrast')
            status=plan['plan_state']
        elif kind in ('experiment_execution','experiment_validation'):
            from .controller import check_execution_or_validation
            status=check_execution_or_validation(root,kind)
        else:raise IntegrityError('Unknown experiment package kind')
        return {'verified':True,'index_sha256':index_hash,'package_kind':kind,'status':status,
                'warnings':['integrity_only_not_causal_truth'],'errors':[]}
    except (OSError,KeyError,TypeError,ValueError,IntegrityError) as exc:
        return {'verified':False,'package_kind':kind,'errors':[{'type':type(exc).__name__,'message':str(exc)}]}
