#!/usr/bin/env python3
"""Verify every L2 example; nested L1 branch receipts are checked by their parent."""
from pathlib import Path
import argparse,json
from l2_sandbox.io import read_json
from l2_sandbox.package import verify

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path,nargs='?',default=Path('examples'));a=p.parse_args()
    rows=[]
    for idx in sorted(a.directory.rglob('evidence_index.json')):
        if read_json(idx).get('schema_version')!='l2-0.1.0':continue
        result=verify(idx.parent);rows.append({'package':str(idx.parent),**result})
    print(json.dumps({'packages':len(rows),'all_verified':bool(rows) and all(r['verified'] for r in rows),'results':rows},ensure_ascii=False,indent=2))
    return 0 if rows and all(r['verified'] for r in rows) else 2
if __name__=='__main__':raise SystemExit(main())
