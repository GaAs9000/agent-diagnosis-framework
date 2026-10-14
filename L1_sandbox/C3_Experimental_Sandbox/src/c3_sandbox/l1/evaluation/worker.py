"""Trusted image-side worker. Never imported from the submission directory.

Invokes the upstream single-verifier API, not a reimplementation of its rubric.
Writes raw returned JSON separately from diagnostic stdout/stderr/process errors.
"""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback


def main(argv=None):
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True); p.add_argument('--entrypoint',required=True)
    p.add_argument('--submission',required=True); p.add_argument('--output',required=True)
    a=p.parse_args(argv)
    source=Path(a.source).resolve(); entry=(source/a.entrypoint).resolve(); output=Path(a.output)
    if not entry.is_relative_to(source) or not entry.is_file(): raise ValueError('invalid verifier path')
    output.mkdir(parents=True,exist_ok=True)
    # Only trusted upstream source is added. -I keeps cwd/PYTHONPATH out.
    sys.path.insert(0,str(source))
    os.environ['OMEGAUSE_OFFICEVAL_DISABLE_COM']='1'
    receipt={'worker_version':'0.2.0','entrypoint':a.entrypoint,'protocol':'evaluate(directory)',
             'exception':None,'official_returned':False}
    try:
        spec=importlib.util.spec_from_file_location('l1_upstream_verifier',entry)
        module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module)
        raw=module.evaluate(str(Path(a.submission)))
        with (output/'official.json').open('x',encoding='utf-8') as f:
            json.dump(raw,f,ensure_ascii=False,allow_nan=False); f.write('\n')
        receipt['official_returned']=True
        code=0
    except Exception as exc:
        receipt['exception']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
        code=1
    (output/'worker.json').write_text(json.dumps(receipt,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    return code

if __name__=='__main__': raise SystemExit(main())
