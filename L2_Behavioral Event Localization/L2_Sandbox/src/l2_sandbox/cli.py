from __future__ import annotations
import argparse
import json
from pathlib import Path
from . import __version__
from .io import L2Error,read_json,read_jsonl,write_json,safe,bytes_read
from .config import load


def main(argv=None):
    p=argparse.ArgumentParser(prog='c3-l2',description='Read-only L1 evidence localization, predictions and explicit experiment plans')
    p.add_argument('--version',action='version',version=__version__)
    sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('inspect');s.add_argument('source',type=Path);s.add_argument('--require-l1-verifier',action='store_true')
    for name in ('analyze','batch'):
        s=sub.add_parser(name);s.add_argument('--source',type=Path,required=True);s.add_argument('--config',type=Path)
        s.add_argument('--output',type=Path,required=True);s.add_argument('--allow-model',action='store_true')
        if name=='batch':s.add_argument('--pattern',default='0*/*/manifest.json')
    s=sub.add_parser('verify');s.add_argument('package',type=Path);s.add_argument('--source',type=Path);s.add_argument('--expected-index-sha256')
    s=sub.add_parser('show-fact');s.add_argument('package',type=Path);s.add_argument('--fact',required=True)
    s=sub.add_parser('doctor');s.add_argument('--config',type=Path,required=True)
    s=sub.add_parser('plan');s.add_argument('--analysis',type=Path,required=True);s.add_argument('--source',type=Path,required=True)
    s.add_argument('--checkpoint',required=True);s.add_argument('--hypothesis');s.add_argument('--template',default='resume_unchanged')
    s.add_argument('--replicates',type=int,default=3);s.add_argument('--endpoint',default='submission_present');s.add_argument('--output',type=Path,required=True)
    s=sub.add_parser('execute');s.add_argument('--plan',type=Path,required=True);s.add_argument('--source',type=Path,required=True)
    s.add_argument('--output',type=Path,required=True);s.add_argument('--allow-execution',action='store_true')
    s=sub.add_parser('validate');s.add_argument('--analysis',type=Path,required=True);s.add_argument('--receipts',type=Path,required=True);s.add_argument('--output',type=Path,required=True)
    s=sub.add_parser('pack');s.add_argument('package',type=Path);s.add_argument('--output',type=Path,required=True)
    s=sub.add_parser('unpack');s.add_argument('archive',type=Path);s.add_argument('--output',type=Path,required=True);s.add_argument('--repair-utf8-names',action='store_true')
    s=sub.add_parser('evaluate');s.add_argument('--predictions',type=Path,required=True);s.add_argument('--gold',type=Path,required=True);s.add_argument('--top-k',type=int,default=3)
    args=p.parse_args(argv)
    def emit(v):print(json.dumps(v,ensure_ascii=False,indent=2))
    try:
        if args.command=='inspect':
            from .evidence.l1 import inspect
            r=inspect(args.source,require_l1=args.require_l1_verifier);emit(r);return 0 if r['verified'] else 2
        if args.command in ('analyze','batch'):
            from .pipeline import analyze
            config=load(args.config)
            if args.command=='analyze':
                r=analyze(args.source,args.output,config,allow_model=args.allow_model);emit(r)
                return 0 if r['verification']['verified'] and r['status'].startswith('completed') else 2
            if config['mode']!='rules' and not args.allow_model:raise L2Error('Batch model calls need explicit --allow-model')
            paths=sorted(args.source.glob(args.pattern))
            if not paths:raise L2Error('No matching source manifests')
            args.output.mkdir(parents=True,exist_ok=False,mode=0o700);rows=[]
            for i,mp in enumerate(paths):
                try:
                    r=analyze(mp.parent,args.output/f'analysis_{i:03d}',config,allow_model=args.allow_model)
                    rows.append({'source_relative':str(mp.parent.relative_to(args.source)),**r})
                except Exception as exc:rows.append({'source_relative':str(mp.parent.relative_to(args.source)),'error':type(exc).__name__+': '+str(exc)})
            result={'mode':config['mode'],'input_cases':len(paths),'results':rows,'research_accuracy_claimed':False,
                    'note':'Sibling episodes from one task are one task family, not independent test cases'}
            write_json(args.output,'batch_receipt.json',result);emit(result)
            return 0 if all(r.get('verification',{}).get('verified') and r.get('status','').startswith('completed') for r in rows) else 2
        if args.command=='verify':
            from .package import verify
            r=verify(args.package,source=args.source,expected_index=args.expected_index_sha256);emit(r);return 0 if r['verified'] else 2
        if args.command=='show-fact':
            from .package import verify
            from .evidence.references import select
            if not verify(args.package)['verified']:raise L2Error('Package verification failed')
            f=next((f for f in read_jsonl(args.package/'facts.jsonl') if f['fact_id']==args.fact),None)
            if f is None:raise L2Error('Unknown fact ID')
            refs={r['ref_id']:r for r in read_json(args.package/'references.json')}
            emit({'fact':f,'sources':[{'reference':refs[x],'selected_value':select(bytes_read(safe(args.package,refs[x]['local_blob'])),refs[x]['selector'])} for x in f['source_ref_ids']]});return 0
        if args.command=='doctor':
            from .analyst.backend import doctor
            r=doctor(load(args.config));emit(r);return 0 if r['ready'] else 2
        if args.command=='plan':
            from .experiments.plans import compile_plan
            r=compile_plan(args.analysis,args.source,args.checkpoint,args.output,hypothesis_id=args.hypothesis,
                           template=args.template,replicates=args.replicates,endpoint=args.endpoint);emit(r)
            return 0 if r['plan']['plan_state']=='frozen' else 2
        if args.command=='execute':
            from .experiments.controller import execute
            r=execute(args.plan,args.source,args.output,allow_execution=args.allow_execution);emit(r)
            return 0 if r['verification']['verified'] and r['results']['plan_state']=='completed' else 2
        if args.command=='validate':
            from .experiments.controller import import_receipts
            r=import_receipts(args.analysis,args.receipts,args.output);emit(r);return 0 if r['verified'] else 2
        if args.command=='pack':
            from .portable import pack
            emit(pack(args.package,args.output));return 0
        if args.command=='unpack':
            from .portable import unpack
            emit(unpack(args.archive,args.output,repair_utf8_names=args.repair_utf8_names));return 0
        if args.command=='evaluate':
            from .evaluation.metrics import localization_metrics
            emit(localization_metrics(read_json(args.predictions),read_json(args.gold),args.top_k));return 0
    except (L2Error,OSError,ValueError,KeyError,TypeError) as exc:
        emit({'ok':False,'error_type':type(exc).__name__,'error':str(exc)});return 2
    return 2

if __name__=='__main__':raise SystemExit(main())
