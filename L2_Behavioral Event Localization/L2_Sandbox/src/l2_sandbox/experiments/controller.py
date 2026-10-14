"""Explicitly authorized unchanged-resume controller and descriptive receipt import.

No arbitrary executor script, model command or analysis-authored patch is run.
Uses the user's installed L1 run(parent=..., checkpoint_id=...) implementation.
"""
from __future__ import annotations
import copy
import inspect
from pathlib import Path
from ..io import (L2Error,Unsupported,IntegrityError,Journal,digest,now,read_json,read_jsonl,
                  sha_file,safe,write_json,write_jsonl,write_new)
from .plans import TEMPLATES


def outcome(reader,endpoint):
    m=reader.manifest;g=reader.json('official_grade.json');status=m['status']
    record={'execution_valid':status.get('execution_validity')=='valid','evaluation_available':status.get('evaluation_validity')=='valid',
            'endpoint_defined':False,'value':None,'termination':status.get('termination'),'official_projection':g.get('projection'),
            'official_raw':g.get('official_raw'),'error_reason':None}
    if not record['execution_valid']:
        record['error_reason']='execution_invalid';return record
    if endpoint=='submission_present':
        record.update(endpoint_defined=True,value=bool(reader.json('artifacts/inventory.json')['files']))
    elif not record['evaluation_available'] or not isinstance(g.get('official_raw'),dict):record['error_reason']='official_evaluation_unavailable'
    elif endpoint=='official_dim1':
        value=g['official_raw'].get('dim1_pass')
        if type(value) is bool:record.update(endpoint_defined=True,value=value)
        else:record['error_reason']='dim1_undefined'
    elif endpoint=='official_normalized_score':
        raw=g['official_raw'];score=raw.get('total_score');maximum=raw.get('max_score')
        if type(score) in (int,float) and type(maximum) in (int,float) and maximum>0:
            record.update(endpoint_defined=True,value=score/maximum)
        else:record['error_reason']='score_denominator_undefined'
    return record


def execute(plan_dir:Path,source:Path,output:Path,*,allow_execution=False,_test_runner=None):
    from ..package import verify,seal
    from ..evidence.l1 import L1Reader
    if not allow_execution:raise L2Error('Execution may spend real model/API budget; --allow-execution is required')
    v=verify(plan_dir)
    if not v['verified']:raise L2Error('Plan did not verify')
    plan=read_json(plan_dir/'plan.json')
    if plan['plan_state']!='frozen' or plan['template']!='resume_unchanged':raise Unsupported('Only a frozen resume_unchanged plan can be executed by this release')
    reader=L1Reader(source,expected_index=plan['source_index_sha256'])
    cp_path=f"checkpoints/{plan['checkpoint_id']}.json"
    if reader.files[cp_path]['sha256']!=plan['checkpoint_sha256']:raise IntegrityError('Checkpoint differs from frozen plan')
    cfg=reader.json('raw/resolved_config.json')
    if digest(cfg)!=plan['config_sha256']:raise IntegrityError('Agent configuration differs')
    if _test_runner is None:
        try:from c3_sandbox.l1.core.orchestrator import run as runner
        except ImportError as exc:raise Unsupported('Install existing L1 0.2 in this virtualenv; no host fallback') from exc
    else:runner=_test_runner
    if not {'parent','checkpoint_id'}<=set(inspect.signature(runner).parameters):raise Unsupported('Installed L1 resume signature differs')
    output=Path(output).absolute()
    for p in (plan_dir,source):
        if output.resolve().is_relative_to(Path(p).resolve()) or Path(p).resolve().is_relative_to(output.resolve()):
            raise L2Error('Experiment output must be outside immutable inputs')
    output.mkdir(parents=True,exist_ok=False,mode=0o700)
    write_new(output,'frozen_plan.json',(plan_dir/'plan.json').read_bytes())
    write_json(output,'authorization.json',{'authorized_at':now(),'explicit_allow_execution':True,'plan_index_sha256':v['index_sha256'],
        'plan_sha256':sha_file(plan_dir/'plan.json'),'mode':'test_fixture' if _test_runner else 'live_l1_bridge',
        'authorized_branches':sum(x['replicates'] for x in plan['arms']),'allowed_templates':['resume_unchanged']})
    journal=Journal(output,'experiment_trace.jsonl');receipts=[]
    journal.emit('experiment.started',{'plan_sha256':sha_file(plan_dir/'plan.json'),'source_index_sha256':reader.index_sha256})
    for i in range(plan['replicates_per_arm']):
        for arm in plan['arms']:
            name=f"branches/{arm['arm_id']}_{i:03d}"
            intent={'trial_id':f"{arm['arm_id']}_{i:03d}",'arm_id':arm['arm_id'],'template':arm['template'],'source_index_sha256':reader.index_sha256,
                    'checkpoint_sha256':plan['checkpoint_sha256'],'checkpoint_id':plan['checkpoint_id'],'replicate':i,'branch_path':name}
            journal.emit('branch.intent',intent)
            try:
                returned=runner(copy.deepcopy(cfg),output/name,parent=Path(source),checkpoint_id=plan['checkpoint_id'])
                branch=L1Reader(output/name)
                lineage=branch.manifest.get('lineage') or {}
                if lineage.get('parent_index_sha256')!=reader.index_sha256 or lineage.get('parent_checkpoint_sha256')!=plan['checkpoint_sha256']:
                    raise IntegrityError('New branch lineage differs from frozen plan')
                if branch.manifest['episode_id']==reader.episode_id:raise IntegrityError('Resume did not create a new episode')
                row={**intent,'attempted':True,'new_episode_id':branch.episode_id,'branch_index_sha256':branch.index_sha256,
                     'branch_verified':True,'counts':branch.manifest['counts'],'outcome':outcome(branch,plan['endpoint']),
                     'controller_error':None,'source_unchanged':reader.verify_unchanged()}
            except Exception as exc:
                row={**intent,'attempted':True,'new_episode_id':None,'branch_index_sha256':None,'branch_verified':False,'counts':None,
                     'outcome':{'execution_valid':False,'evaluation_available':False,'endpoint_defined':False,'value':None,
                                'error_reason':'controller_or_branch_error'},'controller_error':{'type':type(exc).__name__,'message':str(exc)},
                     'source_unchanged':reader.verify_unchanged()}
            receipts.append(row);journal.emit('branch.receipt',row)
            if not row['source_unchanged']:raise IntegrityError('Parent source changed; experiment aborted with journal retained')
    write_jsonl(output,'branch_receipts.jsonl',receipts)
    summary=summarize(receipts,plan,fixture=_test_runner is not None)
    write_json(output,'results.json',summary);journal.emit('experiment.finished',summary)
    seal(output,'experiment_execution')
    return {'output':str(output),'results':summary,'verification':verify(output)}


def summarize(receipts,plan,*,fixture=False):
    groups={}
    for arm in plan['arms']:
        rows=[r for r in receipts if r['arm_id']==arm['arm_id']]
        vals=[r['outcome']['value'] for r in rows if r['outcome']['endpoint_defined']]
        groups[arm['arm_id']]={'attempted':len(rows),'execution_valid':sum(r['outcome']['execution_valid'] for r in rows),
            'evaluation_available':sum(r['outcome']['evaluation_available'] for r in rows),
            'endpoint_defined':len(vals),'endpoint_undefined':len(rows)-len(vals),
            'mean_on_defined_only':sum(vals)/len(vals) if vals else None,
            'errors':[r.get('controller_error') or r['outcome'].get('error_reason') for r in rows if not r['outcome']['endpoint_defined']]}
    return {'plan_state':'completed' if all(r['branch_verified'] for r in receipts) else 'execution_invalid',
            'hypothesis_verdict':'untested' if fixture else 'not_identifiable','groups':groups,'endpoint':plan['endpoint'],
            'reason':'Test fixtures are not real intervention evidence' if fixture else 'Only unchanged-resume observations; no intervention contrast is identified',
            'mode':'test_fixture' if fixture else 'live_l1_bridge',
            'causal_effect':None,'statistical_policy':'descriptive_only_no_significance_or_ground_truth_label',
            'missing_outcomes_not_discarded':True}


def import_receipts(analysis:Path,execution:Path,output:Path):
    from ..package import verify,seal
    av=verify(analysis);ev=verify(execution)
    if not av['verified'] or not ev['verified']:raise L2Error('Verified diagnosis and controller receipt package required')
    plan=read_json(execution/'frozen_plan.json');am=read_json(analysis/'manifest.json')
    if plan['analysis_index_sha256']!=av['index_sha256'] or plan['analysis_id']!=am['analysis_id']:
        raise IntegrityError('Experiment does not belong to the frozen analysis')
    out=Path(output).absolute()
    if out.resolve().is_relative_to(analysis.resolve()) or out.resolve().is_relative_to(execution.resolve()):raise L2Error('Use separate validation output')
    out.mkdir(parents=True,exist_ok=False,mode=0o700)
    for src,dst in [('frozen_plan.json','frozen_plan.json'),('branch_receipts.jsonl','branch_receipts.jsonl'),('results.json','results.json')]:
        write_new(out,dst,(execution/src).read_bytes())
    write_json(out,'validation_binding.json',{'analysis_index_sha256':av['index_sha256'],'execution_index_sha256':ev['index_sha256'],
        'imported_at':now(),'original_prediction_modified':False,'execution_package_required_for_branch_revalidation':True})
    write_json(out,'statistical_analysis.json',{'hypothesis_verdict':'not_identifiable','causal_effect':None,
        'reason':'No implemented treatment arm; receipt import cannot upgrade a baseline to causal support'})
    write_new(out,'validation_report.md',b'# L2 receipt validation\n\nBaseline continuation receipts imported. No causal intervention contrast is identified. Original predictions were not modified.\n')
    seal(out,'experiment_validation')
    return verify(out)


def check_execution_or_validation(root,kind):
    from ..io import check_journal
    plan=read_json(root/'frozen_plan.json');receipts=read_jsonl(root/'branch_receipts.jsonl');results=read_json(root/'results.json')
    if plan['template']!='resume_unchanged' or plan['plan_state']!='frozen':raise IntegrityError('Unsupported executed plan')
    if len(receipts)!=sum(a['replicates'] for a in plan['arms']):raise IntegrityError('Missing declared trial receipts')
    if len({r['trial_id'] for r in receipts})!=len(receipts):raise IntegrityError('Duplicate trial receipt')
    for r in receipts:
        if r['source_index_sha256']!=plan['source_index_sha256'] or r['checkpoint_sha256']!=plan['checkpoint_sha256']:
            raise IntegrityError('Receipt provenance differs from plan')
    if kind=='experiment_execution':
        from ..evidence.l1 import L1Reader
        auth=read_json(root/'authorization.json');trace=read_jsonl(root/'experiment_trace.jsonl');check_journal(trace)
        if sha_file(root/'frozen_plan.json')!=auth['plan_sha256'] or not auth['explicit_allow_execution']:
            raise IntegrityError('Plan/authorization mismatch')
        if [e['data'] for e in trace if e['kind']=='branch.receipt']!=receipts:raise IntegrityError('Journal/receipt mismatch')
        for r in receipts:
            if r['branch_verified']:
                branch=L1Reader(safe(root,r['branch_path']),expected_index=r['branch_index_sha256'])
                if outcome(branch,plan['endpoint'])!=r['outcome']:raise IntegrityError('Receipt endpoint differs from branch')
                if branch.manifest['episode_id']!=r['new_episode_id']:raise IntegrityError('Receipt branch identity differs')
        expected=summarize(receipts,plan,fixture=auth['mode']=='test_fixture')
        if expected!=results:raise IntegrityError('Experiment aggregate differs')
    else:
        if read_json(root/'statistical_analysis.json')['hypothesis_verdict']!='not_identifiable':
            raise IntegrityError('Baseline import cannot claim a verified mechanism')
    if results['hypothesis_verdict'] not in ('untested','not_identifiable'):raise IntegrityError('Unsupported causal verdict')
    return results['plan_state']
