#!/usr/bin/env python3
"""Preflight recovery destinations, then reuse the owner-checked R22 supervisor."""
import argparse
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve()
spec = importlib.util.spec_from_file_location('context004_supervisor', SCRIPT.with_name('supervise_r22_native_alignment.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.SCRIPT = SCRIPT
module.DRIVER = SCRIPT.with_name('run_context004_recovery.py')
driver = module.load(module.DRIVER, 'context004_supervisor_preflight')


def preflight(phase, plan_path, freeze_path, *destinations):
    plan, freeze = module.read(plan_path), module.read(freeze_path)
    driver.validate_contract(plan)
    driver.assert_write_paths(plan, phase, plan_path, freeze_path, Path(plan['run_root']), *destinations)
    driver.validate_freeze(plan, freeze)
    driver.assert_sources(plan)
    driver.assert_originals(plan)


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--phase', type=Path); parser.add_argument('--plan', type=Path)
    parser.add_argument('--freeze', type=Path); parser.add_argument('--service', type=Path)
    parser.add_argument('--child', type=Path)
    args, _ = parser.parse_known_args(argv)
    if args.execute and args.child:
        binding = module.read(args.child)
        phase = Path(binding['phase']).resolve()
        if args.child.resolve() != phase / 'supervisor_started.json':
            raise ValueError('child binding must be the exact phase startup marker')
        preflight(phase, Path(binding['plan']), Path(binding['freeze']), args.child)
        module.assert_startup(binding)
    elif args.execute and args.phase and args.freeze and args.service:
        preflight(args.phase, args.plan or args.phase/'plan.json', args.freeze, args.service)
    module.main(argv)


if __name__ == '__main__':
    main()
