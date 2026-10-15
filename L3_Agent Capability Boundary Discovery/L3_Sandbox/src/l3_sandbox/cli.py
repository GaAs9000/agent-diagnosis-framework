"""Explicit prepare / execute / report commands; model execution is opt-in."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .common import L3Error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('plan', help='Freeze an offline task/condition matrix')
    prepare.add_argument('spec', type=Path)
    prepare.add_argument('--output', type=Path, required=True)
    execute = sub.add_parser('run', help='Verify workload; send model requests only with --execute')
    execute.add_argument('directory', type=Path)
    execute.add_argument('--execute', action='store_true')
    report = sub.add_parser('summarize', help='Summarize accepted evidence, missing cells and contrasts')
    report.add_argument('directory', type=Path)
    check = sub.add_parser('check-plan', help='Check frozen code, inputs, configuration and local images')
    check.add_argument('directory', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'plan':
            from .plan import create_plan
            plan = create_plan(args.spec, args.output)
            result = {'directory': str(args.output.absolute()), 'status': 'unexecuted',
                      'experiment_id': plan['experiment_id'], **plan['workload']}
        elif args.command == 'run':
            from .run import run_plan
            result = run_plan(args.directory, execute=args.execute)
        elif args.command == 'summarize':
            from .report import summarize
            result = summarize(args.directory)
        else:
            from .plan import verify_plan
            plan = verify_plan(args.directory)
            result = {'verified': True, 'experiment_id': plan['experiment_id'], **plan['workload']}
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        if args.command == 'run' and args.execute and result.get('status') not in ('complete', 'completed'):
            return 2
        return 0
    except (L3Error, ValueError, OSError) as exc:
        print(json.dumps({'error': type(exc).__name__, 'detail': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
