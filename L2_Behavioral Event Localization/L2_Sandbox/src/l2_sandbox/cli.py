"""L2 v0.3 CLI (`l2 run` and `l2 summarize`, Section 7.1)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from . import __version__
from .io import L2Error, write_json
from .report import run_episode
from .score import summarize_runs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog='l2',
        description='L2 v0.3 Behavioral Event Localization: value curves, predicate hypotheses, and scoring.',
    )
    parser.add_argument('--version', action='version', version=__version__)
    sub = parser.add_subparsers(dest='command', required=True)

    run_p = sub.add_parser('run', help='Run L2 on an L1 parent package (idempotent)')
    run_p.add_argument('l1_run', type=Path, help='Path to parent L1 package directory')
    run_p.add_argument('--analysts', default='rules', help='Comma-separated analysts, e.g. rules,llm:config.json')
    run_p.add_argument('--execute', action='store_true', help='Execute L1 continuations to measure the value curve')
    run_p.add_argument('--output', type=Path, default=None, help='Output directory (defaults to l2/<episode_id>/)')
    run_p.add_argument('--p0', type=float, default=0.5, help='Task baseline outcome p_0 (excluding parent trajectory)')
    run_p.add_argument('--n1', type=int, default=12, help='Stage 1 replicates per grid checkpoint')
    run_p.add_argument('--n2', type=int, default=24, help='Stage 2 replicates per selected interval checkpoint')
    run_p.add_argument('--stride', type=int, default=4, help='Stage 1 checkpoint grid stride')
    run_p.add_argument('--alpha', type=float, default=0.05, help='One-sided significance level')
    run_p.add_argument('--max-concurrency', type=int, default=4, help='Bounded thread pool concurrency (4..8)')

    sum_p = sub.add_parser('summarize', help='Summarize and pool scores across L2 episode directories')
    sum_p.add_argument('l2_dirs', type=Path, nargs='+', help='One or more l2/<episode_id> directories')
    sum_p.add_argument('--output', type=Path, default=None, help='Optional path to write summary JSON')

    args = parser.parse_args(argv)

    def emit(obj):
        print(json.dumps(obj, ensure_ascii=False, indent=2))

    try:
        if args.command == 'run':
            analyst_list = [x.strip() for x in args.analysts.split(',') if x.strip()]
            res = run_episode(
                args.l1_run,
                output_dir=args.output,
                analysts=analyst_list,
                execute=args.execute,
                p0=args.p0,
                n1=args.n1,
                n2=args.n2,
                stride=args.stride,
                alpha=args.alpha,
                max_concurrency=args.max_concurrency,
            )
            emit(res)
            return 0
        if args.command == 'summarize':
            res = summarize_runs(args.l2_dirs)
            if args.output:
                write_json(args.output, res)
            emit(res)
            return 0
    except (L2Error, OSError, ValueError, KeyError, TypeError) as exc:
        emit({'ok': False, 'error_type': type(exc).__name__, 'error': str(exc)})
        return 2
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
