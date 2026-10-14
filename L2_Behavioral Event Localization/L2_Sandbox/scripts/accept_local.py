#!/usr/bin/env python3
"""Local M1/M2 acceptance, never an automatic branch experiment or causal claim."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
from l2_sandbox.config import load
from l2_sandbox.evidence.l1 import inspect
from l2_sandbox.analyst.backend import doctor
from l2_sandbox.pipeline import analyze
from l2_sandbox.io import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-model',action='store_true')
    a=p.parse_args();cfg=load(a.config);a.output.mkdir(parents=True,exist_ok=False,mode=0o700)
    receipt={'schema_version':'l2-0.1.0','source_check':inspect(a.source),
             'model_call_authorized':a.allow_model,'causal_validation_performed':False,'research_accuracy_established':False,
             'M1_passed':False,'M2_live_analyst_passed':False}
    code=2
    try:
        if not receipt['source_check']['verified']:raise RuntimeError('Source rejected')
        if cfg['mode']!='rules':
            receipt['model_preflight']=doctor(cfg)
            if not a.allow_model:
                receipt['reason']='preflight_only_no_model_invocation'
                code=0 if receipt['model_preflight']['ready'] else 2
                write_json(a.output,'acceptance.json',receipt);print(json.dumps(receipt,ensure_ascii=False,indent=2));return code
            if not receipt['model_preflight']['ready']:raise RuntimeError('Model backend/credential preflight failed')
        result=analyze(a.source,a.output/'diagnosis',cfg,allow_model=a.allow_model)
        receipt['analysis']=result
        receipt['M1_passed']=result['verification']['verified'] and result['counts']['facts']>0 and result['status'].startswith('completed')
        receipt['M2_live_analyst_passed']=(receipt['M1_passed'] and result['mode']=='analyst' and result['status']=='completed_analyst'
            and result['counts']['analyst_model_requests']>0 and result['counts']['analyst_tool_results']>0)
        code=0 if receipt['M1_passed'] and (cfg['mode']=='rules' or receipt['M2_live_analyst_passed'] or cfg['mode']=='structured-once') else 2
    except Exception as exc:receipt['error']={'type':type(exc).__name__,'message':str(exc)}
    write_json(a.output,'acceptance.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False,indent=2));return code

if __name__=='__main__':raise SystemExit(main())
