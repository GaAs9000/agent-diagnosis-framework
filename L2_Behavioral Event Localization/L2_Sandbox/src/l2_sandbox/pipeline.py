"""New-only, read-only-source analysis lifecycle with failure receipts."""
from __future__ import annotations
import copy
import uuid
from pathlib import Path
from . import __version__,SCHEMA_VERSION
from .io import (Journal,L2Error,IntegrityError,Unsupported,InvalidOutput,now,digest,safe,sha_file,write_json,write_jsonl,write_new)
from .config import validate
from .evidence.l1 import L1Reader
from .evidence.view import EvidenceView
from .facts.build import FactBuilder
from .analyst.rules import diagnose
from .validation import check_diagnosis
from .reporting import report
from .package import seal,verify


def analyze(source:Path,output:Path,config:dict,*,allow_model=False,backend=None):
    config=copy.deepcopy(validate(config));source=Path(source).absolute();output=Path(output).absolute()
    if output.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(output.resolve()):
        raise L2Error('Output must be outside the source tree and not its ancestor')
    safe(output,'manifest.json')
    if config['mode']!='rules' and not allow_model and backend is None:
        raise L2Error('Real Analyst inference requires --allow-model; rules mode never contacts a model')
    output.mkdir(parents=True,exist_ok=False,mode=0o700)
    aid='analysis_'+uuid.uuid4().hex;created=now();journal=Journal(output)
    write_json(output,'analyst_config.json',config)
    journal.emit('analysis.started',{'analysis_id':aid,'source_directory_hint':str(source),'config_sha256':digest(config),
                                     'allow_model':allow_model,'test_backend_injected':backend is not None})
    reader=None;builder=None;view=None;failure=None;checks=[];status='initializing';unchanged=False
    mode='test_fixture' if backend is not None else config['mode'];api_result={}
    empty={'summary':'本次分析未完成，未发布有效诊断。','localizations':[],'hypotheses':[],
           'abstentions':['需要解决记录在 failure 中的问题后，新建分析版本重新运行。']}
    diagnosis=copy.deepcopy(empty)
    try:
        reader=L1Reader(source,expected_index=config['expected_source_index_sha256'],require_l1=config['require_l1_verifier'])
        write_new(output,'source_index.json',reader.read('evidence_index.json'))
        write_json(output,'source_packages.json',{'sources':[reader.source_record()],
                   'retained_scope':'cited_source_subset; private verifier and sibling results are not exposed to Analyst'})
        journal.emit('source.accepted',reader.report)
        builder=FactBuilder(reader,output,config).build()
        view=EvidenceView(builder,config)
        journal.emit('view.built',{'view_sha256':view.manifest['view_sha256'],'facts':len(builder.facts),'object_versions':len(builder.objects)})
        if config['mode']=='rules':
            diagnosis=diagnose(builder,config['budget']['max_findings'])
            checks=check_diagnosis(diagnosis,builder.facts,builder.events,max_findings=config['budget']['max_findings'])
            status='completed_rules'
            journal.emit('rules.completed',{'diagnosis_sha256':digest(diagnosis),'model_requests':0})
        else:
            from .analyst.run import run
            api_result=run(view,config,output,journal,backend=backend)
            diagnosis=api_result['diagnosis'];checks=api_result['checks'];status='completed_analyst'
        unchanged=reader.verify_unchanged()
        if not unchanged:raise IntegrityError('Source changed during analysis; output is invalidated')
    except KeyboardInterrupt:
        failure={'type':'KeyboardInterrupt','message':'Interrupted; raw attempts and partial facts retained'};status='interrupted';diagnosis=copy.deepcopy(empty);checks=[]
        journal.emit('analysis.error',failure)
    except Exception as exc:
        failure={'type':type(exc).__name__,'message':str(exc)};status='analysis_failed';diagnosis=copy.deepcopy(empty);checks=[]
        journal.emit('analysis.error',failure)
    if reader is not None:
        try:unchanged=reader.verify_unchanged()
        except Exception:unchanged=False
    if reader is not None and not unchanged and status.startswith('completed'):
        failure={'type':'IntegrityError','message':'Source changed during final verification; diagnosis is invalidated'}
        status='analysis_failed';diagnosis=copy.deepcopy(empty);checks=[]
        journal.emit('analysis.error',failure)
    facts=builder.facts if builder else [];events=builder.events if builder else []
    refs=list(builder.refs.records.values()) if builder else [];coverage=builder.coverage if builder else []
    if failure:coverage=[*coverage,{'area':'analysis','status':'failed','reason':failure}]
    if not (output/'source_packages.json').exists():write_json(output,'source_packages.json',{'sources':[],'status':'source_not_accepted'})
    write_jsonl(output,'facts.jsonl',facts);write_json(output,'references.json',refs);write_json(output,'events.json',events)
    write_json(output,'coverage.json',coverage);write_json(output,'analysis_view.json',view.manifest if view else {})
    write_json(output,'checkpoints.json',builder.checkpoints if builder else [])
    write_json(output,'diagnosis.json',diagnosis);write_jsonl(output,'localizations.jsonl',diagnosis['localizations'])
    write_jsonl(output,'hypotheses.jsonl',diagnosis['hypotheses']);write_jsonl(output,'claim_checks.jsonl',checks)
    # These are suggestions, not executable/authorized plans. CLI plan compiles them.
    suggestions=[{'hypothesis_id':h['hypothesis_id'],'prediction_index':i,'prediction':p,'plan_state':'proposed',
                  'execution_authorized':False,'hypothesis_verdict':'untested'} for h in diagnosis['hypotheses'] for i,p in enumerate(h['predictions'])]
    write_jsonl(output,'intervention_plans.jsonl',suggestions)
    source_sha=reader.index_sha256 if reader else '0'*64
    write_json(output,'predictions/frozen_manifest.json',{'schema_version':SCHEMA_VERSION,'analysis_id':aid,
        'source_index_sha256':source_sha,'diagnosis_sha256':digest(diagnosis),'hypotheses_sha256':digest(diagnosis['hypotheses']),
        'frozen_at':now(),'submission_status':'accepted' if status.startswith('completed') else 'not_submitted',
        'trusted_external_timestamp':False,'experiment_results_seen':False})
    counts={'facts':len(facts),'events':len(events),'references':len(refs),'object_versions':len(builder.objects) if builder else 0,
            'localizations':len(diagnosis['localizations']),'hypotheses':len(diagnosis['hypotheses']),
            'analyst_model_requests':sum(e['kind']=='model.request' for e in journal.records),
            'analyst_tool_results':sum(e['kind']=='tool.result' for e in journal.records),'experiments_executed':0}
    manifest={'schema_version':SCHEMA_VERSION,'package_kind':'diagnosis','analysis_id':aid,'created_at':created,
        'producer':{'name':'l2-evidence-analyst','version':__version__,'code_sha256':digest({str(p.relative_to(Path(__file__).parent)):sha_file(p) for p in sorted(Path(__file__).parent.rglob('*.py'))})},
        'source_index_sha256':source_sha,'source_episode_id':reader.episode_id if reader else 'unknown','mode':mode,'status':status,
        'failure':failure,'counts':counts,'claim_guarantees':{'references':'checked','semantic_entailment':'not_certified','causal_validation':'untested','capability_profile':'not_produced'},
        'source_unchanged':unchanged,'config_sha256':digest(config)}
    write_json(output,'manifest.json',manifest)
    write_json(output,'cost.json',{'model_requests':counts['analyst_model_requests'],'usage_records':[e['data'].get('usage') for e in journal.records if e['kind']=='model.parsed'],
        'retrieval_calls':view.calls if view else 0,'retrieval_bytes':view.bytes_read if view else 0,'actual_cost_usd':None,
        'cost_scope':'Analyst only; L1 source and future experiments not combined'})
    write_new(output,'report.md',report(manifest,diagnosis,facts,coverage).encode('utf8'))
    journal.emit('analysis.finished',{'status':status,'diagnosis_sha256':digest(diagnosis),'source_unchanged':unchanged})
    index_hash=seal(output)
    result=verify(output)
    return {'analysis_id':aid,'output':str(output),'status':status,'mode':mode,'counts':counts,'failure':failure,
            'index_sha256':index_hash,'verification':result}
