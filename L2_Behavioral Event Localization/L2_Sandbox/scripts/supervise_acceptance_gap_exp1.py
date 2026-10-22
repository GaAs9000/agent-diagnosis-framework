#!/usr/bin/env python3
"""Supervise one Exp1 block: preflight, run the driver in a child, then release containers and the GPU.

Reuses supervise_r22_native_alignment.py for exact-owner container labelling,
bounded child waiting and verified service release. Default invocation is inert.
The phase directory must be validation/acceptance_gap_exp1_20261022/block_NN and
contain plan.json, freeze.json (phase approval) and service.json. Analysis is a
separate zero-request step (scripts/analyze_acceptance_gap_exp1.py).
"""
import argparse
import importlib.util
from pathlib import Path
import re

SCRIPT = Path(__file__).resolve()
spec = importlib.util.spec_from_file_location('acceptance_gap_resource_supervisor',
                                              SCRIPT.with_name('supervise_r22_native_alignment.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.SCRIPT = SCRIPT
module.DRIVER = SCRIPT.with_name('run_acceptance_gap_exp1.py')
driver = module.load(module.DRIVER, 'acceptance_gap_supervisor_driver')

RESERVED = ('supervisor_started.json', 'supervisor_child.log', 'supervisor_child.json',
            'supervisor_result.json', 'result.json', 'score_inputs.json', 'cohort.json',
            'executed_plan.json', 'executed_freeze.json', 'e1', 'execution')


def phase_paths(phase, plan_path, freeze_path, service_path, *, child=False):
    phase = Path(phase).absolute()
    allowed = driver.PHASE_ROOT.resolve()
    if (phase.is_symlink() or any(p.is_symlink() for p in phase.parents)
            or phase.resolve().parent != allowed or not re.fullmatch(r'block_\d{2}', phase.name)):
        raise ValueError('supervisor phase must be an unaliased validation/acceptance_gap_exp1_20261022/block_NN')
    phase = phase.resolve()
    for path, name in ((plan_path, 'plan.json'), (freeze_path, 'freeze.json'), (service_path, 'service.json')):
        if driver.base.regular(path) != phase / name:
            raise ValueError('supervisor startup metadata must use exact phase paths')
    for ancestor in phase.parents:
        if ((ancestor / 'evidence_index.json').exists() or (ancestor / 'attempt_started.json').exists()
                or (ancestor / 'supervisor_started.json').exists()):
            raise ValueError('supervisor phase overlaps an existing package or execution')
    reserved = [name for name in RESERVED if not child or name not in
                ('supervisor_started.json', 'supervisor_child.log', 'supervisor_child.json')]
    if any((phase / name).exists() or (phase / name).is_symlink() for name in reserved):
        raise ValueError('supervisor restart/overwrite forbidden')
    return phase


def preflight(phase, plan_path, freeze_path, service_path, *, child=False):
    phase = phase_paths(phase, plan_path, freeze_path, service_path, child=child)
    plan, freeze, service = map(module.read, (plan_path, freeze_path, service_path))
    if phase.name != f"block_{plan.get('block', -1):02d}":
        raise ValueError('phase directory and plan block differ')
    driver.validate_plan(plan)
    driver.fresh_paths(Path(plan['run_root']), plan['block'], phase / 'execution')
    driver.validate_service(plan, service_path)
    # Phase approval, exact startup sources, clock calibration, process-group
    # identity and the independent GNU timeout derived from the original lease.
    module.validate(plan, freeze, service, module.time.time())
    if freeze.get('review_digest') != plan['review_digest'] or freeze.get('block') != plan['block']:
        raise ValueError('phase approval does not bind this reviewed block plan')
    driver.assert_sources(plan)
    return phase, plan, freeze


def child(binding_path):
    binding_path = driver.base.regular(binding_path)
    binding = module.read(binding_path)
    phase = Path(binding['phase']).resolve()
    if binding_path != phase / 'supervisor_started.json':
        raise ValueError('child binding must be the exact phase startup marker')
    expected_files = {str(path.resolve()) for path in
                      (SCRIPT, module.DRIVER, module.HELPER, phase / 'plan.json', phase / 'freeze.json',
                       phase / 'service.json')}
    if (set(binding.get('files', {})) != expected_files
            or binding.get('plan') != str(phase / 'plan.json')
            or binding.get('freeze') != str(phase / 'freeze.json')
            or not re.fullmatch('[a-f0-9]{64}', binding.get('resource_owner', ''))):
        raise ValueError('child startup file/resource binding differs')
    module.assert_startup(binding)
    phase, plan, freeze = preflight(phase, Path(binding['plan']), Path(binding['freeze']),
                                    phase / 'service.json', child=True)
    if binding.get('plan_sha256') != plan['plan_sha256']:
        raise ValueError('child plan binding differs')
    helper = module.load(module.HELPER, 'acceptance_gap_cleanup_child')
    module.save(phase / 'executed_plan.json', plan)
    module.save(phase / 'executed_freeze.json', freeze)
    with module.phase_containers(helper, phase, binding['resource_owner']):
        driver.execute(plan, phase / 'execution', phase / 'service.json')
    for name in ('result.json', 'cohort.json', 'score_inputs.json'):
        source = phase / 'execution' / name
        if source.is_file():
            module.save(phase / name, module.read(source))


def cpu_summary(phase, binding, deadline, helper):
    source = Path(phase) / 'execution/score_inputs.json'
    return dict(status='NOT_RUN', reason='separate zero-request analysis: scripts/analyze_acceptance_gap_exp1.py',
                score_inputs_sha256=module.sha(source) if source.is_file() else None,
                actor_requests=0, analyst_requests=0, d1_execution=False)


module.child = child
module.cpu_summary = cpu_summary


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--phase', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--freeze', type=Path)
    parser.add_argument('--service', type=Path)
    parser.add_argument('--child', type=Path)
    args, _ = parser.parse_known_args(argv)
    if args.execute and not args.child:
        if not all((args.phase, args.freeze, args.service)):
            raise ValueError('--execute requires --phase --freeze --service')
        preflight(args.phase, args.plan or args.phase / 'plan.json', args.freeze, args.service)
    module.main(argv)


if __name__ == '__main__':
    main()
