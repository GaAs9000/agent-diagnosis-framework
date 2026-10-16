"""L2 v0.4 CLI (`l2 diagnose`, `l2 verify`, and `l2 summarize`, Section 9.1)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from . import __version__
from .config import load as load_config
from .io import L2Error, write_json
from .report import diagnose_episode, resolve_m2_baseline_p0, verify_episode
from .score import summarize_runs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog='l2',
        description='L2 v0.4 Defect Provenance & Behavioral Localization: snapshot grading, defect tables, origins, hypotheses, and targeted verification.',
    )
    parser.add_argument('--version', action='version', version=__version__)
    sub = parser.add_subparsers(dest='command', required=True)

    # 1. l2 diagnose (and alias l2 run)
    for cmd_name, cmd_help in (
        ('diagnose', 'Zero-continuation diagnosis: snapshot grading, defect table, and Analyst Call 1/2'),
        ('run', 'Alias for l2 diagnose (backward compatibility)'),
    ):
        diag_p = sub.add_parser(cmd_name, help=cmd_help)
        diag_p.add_argument('l1_run', type=Path, help='Path to parent L1 package directory')
        diag_p.add_argument('--analysts', default='rules', help='Comma-separated analysts, e.g. rules,llm_raw:config.json,llm:config.json')
        diag_p.add_argument('--cohort', type=Path, default=None, help='Optional M2 cohort directory for same-task cross-run contrast')
        diag_p.add_argument('--execute', action='store_true', help='Authorize live LLM Analyst API calls when not cached')
        diag_p.add_argument('--cache-only', action='store_true', help='Require cached LLM Analyst outputs')
        diag_p.add_argument('--output', type=Path, default=None, help='Output directory (defaults to l2/<episode_id>/)')
        diag_p.add_argument('--cache-dir', type=Path, default=None, help='Optional snapshot grade & LLM cache directory')
        diag_p.add_argument('--reuse-from', type=Path, action='append', default=[], help='Optional existing L2 directory to reuse branches from')
        diag_p.add_argument('--alpha', type=float, default=0.05, help='One-sided significance level')
        p0_group = diag_p.add_mutually_exclusive_group(required=False)
        p0_group.add_argument('--p0', type=float, default=None, help='Optional external baseline p_0 in [0,1]')
        p0_group.add_argument('--baseline', type=Path, default=None, help='Optional L3 M2 baseline directory or summary JSON')
        diag_p.add_argument('--baseline-condition', default=None, help='Optional condition_id when resolving --baseline')

    # 2. l2 verify
    ver_p = sub.add_parser('verify', help='Targeted continuation verification: partition, intervention, or two_point')
    ver_p.add_argument('l1_run', type=Path, help='Path to parent L1 package directory')
    ver_p.add_argument('--mode', choices=('partition', 'intervention', 'two_point'), default='intervention', help='Verification mode')
    ver_p.add_argument('--checkpoint', '--k', dest='checkpoint', type=int, required=True, help='Target checkpoint index k')
    ver_p.add_argument('--arms', default='plain,deliver', help='Comma-separated continuation arms, e.g. plain,deliver or plain,rule')
    ver_p.add_argument('--requirement', default=None, help='Rule requirement text when running the `rule` intervention arm')
    ver_p.add_argument('--n', type=int, default=24, help='Branches per arm (default: 24)')
    ver_p.add_argument('--outcome', default='total', help='Outcome to evaluate: `total` or a specific defect/item ID')
    ver_p.add_argument('--execute', action='store_true', help='Authorize paid L1 continuation branches')
    ver_p.add_argument('--cache-only', action='store_true', help='Require cached/reused continuation branches')
    ver_p.add_argument('--reuse-from', type=Path, action='append', default=[], help='Optional existing L2 directory to reuse plain branches from')
    ver_p.add_argument('--output', type=Path, default=None, help='Output directory (defaults to l2/<episode_id>/)')
    ver_p.add_argument('--cache-dir', type=Path, default=None, help='Optional snapshot grade cache directory')
    ver_p.add_argument('--alpha', type=float, default=0.05, help='One-sided significance level')
    ver_p.add_argument('--max-concurrency', type=int, default=4, help='Bounded thread pool concurrency')
    ver_p.add_argument('--config', type=Path, default=None, help='Optional rate-limit configuration')
    ver_p.add_argument('--rpm', type=float, default=None, help='Global branch model requests per minute')
    ver_p.add_argument('--tpm', type=float, default=None, help='Global branch model tokens per minute')

    # 3. l2 summarize
    sum_p = sub.add_parser('summarize', help='Summarize and pool defect census and analyst scores across L2 directories')
    sum_p.add_argument('l2_dirs', type=Path, nargs='+', help='One or more l2/<episode_id> directories')
    sum_p.add_argument('--output', type=Path, default=None, help='Optional path to write summary JSON')
    sum_p.add_argument('--holdout-family', action='append', default=[], help='Explicit held-out task family; repeat to select several')

    args = parser.parse_args(argv)

    def emit(obj):
        print(json.dumps(obj, ensure_ascii=False, indent=2))

    try:
        if args.command in ('diagnose', 'run'):
            analyst_list = [x.strip() for x in args.analysts.split(',') if x.strip()]
            p0_val = args.p0
            if args.baseline is not None:
                resolved_b = resolve_m2_baseline_p0(args.baseline, args.l1_run, condition_id=args.baseline_condition)
                p0_val = resolved_b['p0']
            res = diagnose_episode(
                args.l1_run,
                output_dir=args.output,
                analysts=analyst_list,
                execute=args.execute,
                cache_only=args.cache_only,
                cohort_dir=args.cohort,
                p0=p0_val,
                alpha=args.alpha,
                cache_dir=args.cache_dir,
                reuse_dirs=args.reuse_from,
            )
            emit(res)
            return 0

        if args.command == 'verify':
            arm_list = [x.strip() for x in args.arms.split(',') if x.strip()]
            cfg = load_config(args.config) if args.config else {'curve': {}}
            res = verify_episode(
                args.l1_run,
                output_dir=args.output,
                mode=args.mode,
                k=args.checkpoint,
                arms=arm_list,
                n=args.n,
                outcome=args.outcome,
                requirement=args.requirement,
                execute=args.execute,
                cache_only=args.cache_only,
                alpha=args.alpha,
                max_concurrency=args.max_concurrency,
                rpm=args.rpm if args.rpm is not None else cfg.get('curve', {}).get('rpm'),
                tpm=args.tpm if args.tpm is not None else cfg.get('curve', {}).get('tpm'),
                reuse_dirs=args.reuse_from,
                cache_dir=args.cache_dir,
            )
            emit(res)
            return 0

        if args.command == 'summarize':
            res = summarize_runs(args.l2_dirs, heldout_families=args.holdout_family)
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
