#!/usr/bin/env python3
"""One-shot Mac-to-aml-b CPU postprocessing; never sends model requests.

Run --execute on the Mac after review. The internal --remote-data mode runs
only offline audits on aml-b. Failed/in-progress finalizations are not retried.
"""
import argparse
from contextvars import ContextVar
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
L2 = Path('L2_Behavioral Event Localization/L2_Sandbox')
COHORT = 'v3_qwen36_pi_dev45_20261017'
CODE = '/home/jiasheren2/projects/agent_diagnosis_d0_20261018/Users/jiashen/Desktop/问题诊断'
PYTHON = '/home/jiasheren2/envs/agent-diagnosis-d0-py312/bin/python'
REMOTE_BASE = '/home/jiasheren2/projects/d0_serial_postprocess'
AUDITS = ('g1.json', 'g2.json', 'g4.json', 'status.json')
DEV = {'officeval_' + n for n in ('003', '004', '010', '033', '050', '060', '070', '085', '090')}
COMMAND_DEADLINE = ContextVar('postprocessing_deadline', default=None)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def command(args, **kwargs):
    deadline = COMMAND_DEADLINE.get()
    remaining = 1800 if deadline is None else deadline - time.time()
    if remaining <= 0:
        raise TimeoutError('shared postprocessing deadline expired; no command admitted')
    return subprocess.run(args, check=True, timeout=min(1800, remaining), **kwargs)


def ssh(args):
    return command(['ssh', '-oBatchMode=yes', '-oConnectTimeout=15', 'aml-b',
                    shlex.join(args)])


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_inventory(root):
    paths = set((root / 'L1_sandbox/C3_Experimental_Sandbox/src').rglob('*.py'))
    paths.update((root / 'L1_sandbox/C3_Experimental_Sandbox/src').rglob('*.json'))
    paths.update((root / L2 / 'src').rglob('*.py'))
    paths.update(root / L2 / 'scripts' / name for name in (
        'audit_g1_mechanical.py', 'audit_g4_provenance.py', 'finalize_d0_serial.py'))
    paths.add(root / 'scripts/v3_evaluate_gates_g2_g3_g4.py')
    return {str(p.relative_to(root)): sha(p) for p in sorted(paths)}


def assert_sources(expected):
    if source_inventory(ROOT) != expected:
        raise ValueError('audit source inventory differs from launch-pinned bytes')


def remote_audits(data):
    """No transport adapters are constructed by these offline audit entrypoints."""
    if not re.fullmatch(re.escape(REMOTE_BASE) + r'/[0-9a-f]{64}', str(data)):
        raise ValueError('unexpected isolated data directory')
    if sha(data / 'finalize_d0_serial.py') != sha(Path(__file__).resolve()):
        raise ValueError('remote orchestration source differs from reviewed copy')
    sources = read(data / 'sources.json')
    assert_sources(sources)  # Before importing any project code.
    output = data / 'reports'
    output.mkdir(exist_ok=False)
    (data / 'tmp').mkdir(exist_ok=False)
    scripts = ROOT / L2 / 'scripts'
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    status = {'actor_requests': 0, 'analyst_requests': 0, 'd0_frozen': False,
              'gates_promoted': False, 'commands': {}, 'reports_sha256': {},
              'sources': sources, 'sources_manifest_sha256': sha(data / 'sources.json')}
    tasks = [
        ('g1', ['--runs', str(data / COHORT / 'episodes'), '--execute', '--temp-parent', str(data / 'tmp')]),
        ('g4', [])]
    sys.path[:0] = [str(ROOT / 'L1_sandbox/C3_Experimental_Sandbox/src'), str(ROOT / L2 / 'src')]
    from c3_sandbox.l1.evaluation.study import study_outcome
    packages = sorted((data / COHORT / 'episodes').glob('*'))
    packages += sorted((data / COHORT / 'v3_g2_c0_resumes').glob('*'))
    valid, skipped = [], []
    for package in packages:
        try:
            if study_outcome(package, verify_evidence=True).get('study_valid'):
                valid += ['--package', str(package)]
            else:
                skipped.append(str(package))
        except Exception as exc:
            skipped.append({'package': str(package), 'error': str(exc)})
    tasks[1] = ('g4', valid)
    for name, extra in tasks:
        assert_sources(sources)
        if name == 'g4' and not valid:
            (output / 'g4.log').write_text('No study-valid packages; G4 was not invoked.\n')
            atomic(output / 'g4.json', {'gate_pass': False, 'status': 'no_valid_packages', 'skipped': skipped})
            continue
        script = scripts / ('audit_g1_mechanical.py' if name == 'g1' else 'audit_g4_provenance.py')
        with (output / f'{name}.log').open('wb') as log:
            try:
                result = subprocess.run([PYTHON, str(script), *extra, '--output', str(output / f'{name}.json')],
                                        env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=800)
                status['commands'][name] = {'exit_code': result.returncode, 'source_sha256': sha(script)}
            except subprocess.TimeoutExpired:
                status['commands'][name] = {'status': 'timeout'}
        if not (output / f'{name}.json').exists():
            atomic(output / f'{name}.json', {'gate_pass': False, 'status': 'audit_failed_or_timed_out'})
    evaluator_path = ROOT / 'scripts/v3_evaluate_gates_g2_g3_g4.py'
    assert_sources(sources)
    evaluator = load('postprocess_g2', evaluator_path)
    atomic(output / 'g2.json', evaluator.evaluate_r19_g2(data))
    status['g2_evaluator_sha256'] = sha(evaluator_path)
    status['g4_skipped_not_study_valid'] = skipped
    status['reports_sha256'] = {name: sha(output / name) for name in AUDITS[:-1]}
    assert_sources(sources)
    atomic(output / 'status.json', status)


def finalize(phase, output):
    token = COMMAND_DEADLINE.set(time.time() + 7 * 3600)
    try:
        return _finalize(phase, output)
    finally:
        COMMAND_DEADLINE.reset(token)


def _finalize(phase, output):
    """Lock persists after failure: a second execution cannot resubmit audits."""
    output.mkdir(parents=True, exist_ok=True)
    final = output / 'manifest.json'
    if final.exists():
        receipt = read(final)
        if receipt['identity']['plan_sha256'] != sha(phase / 'executed_plan.json'):
            raise ValueError('cached finalization belongs to a different requested phase')
        for name, expected in receipt['files_sha256'].items():
            if sha(output / name) != expected:
                raise ValueError('saved finalization bytes changed')
        return receipt
    (output / 'one_shot.lock').mkdir()
    sources = source_inventory(ROOT)
    atomic(output / 'sources.json', sources)  # Pin now, before the bounded wait.
    plan = read(phase / 'executed_plan.json')
    deadline = min(plan['service_stop_epoch'] + 120, time.time() + 21600, COMMAND_DEADLINE.get())
    while not (phase / 'supervisor_result.json').exists():
        if time.time() >= deadline:
            raise TimeoutError('serial supervisor did not finish within bounded wait')
        time.sleep(min(30, max(0, deadline - time.time())))
    # All copies, SSH calls, audits, and report fetches share this one budget.
    COMMAND_DEADLINE.set(min(COMMAND_DEADLINE.get(), time.time() + 3600))
    supervisor = read(phase / 'supervisor_result.json')
    source = ROOT / 'L1_sandbox/C3_Experimental_Sandbox/runs' / COHORT
    inventory = {}
    for group in ('episodes', 'v3_g2_c0_resumes'):
        for package in sorted((source / group).glob('*')):
            if (package.name.split('.r')[0] in DEV and not package.is_symlink()
                    and re.fullmatch(r'officeval_\d{3}\.r0[0-4](?:_c0)?', package.name)
                    and (package / 'evidence_index.json').is_file()):
                inventory[f'{group}/{package.name}'] = sha(package / 'evidence_index.json')
    identity = {'plan_sha256': sha(phase / 'executed_plan.json'),
                'supervisor_sha256': sha(phase / 'supervisor_result.json'), 'packages': inventory,
                'sources': sources,
                'result_sha256': sha(phase / 'result.json') if (phase / 'result.json').exists() else None}
    assert_sources(sources)
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    remote = REMOTE_BASE + '/' + run_id
    ssh(['mkdir', '-p', REMOTE_BASE])
    ssh(['mkdir', remote])  # Refuse to overwrite any previous data/audit mirror.
    for relative, expected in inventory.items():
        target = remote + '/' + COHORT + '/' + relative
        ssh(['mkdir', '-p', target])
        command(['rsync', '-a', str(source / relative) + '/', 'aml-b:' + target + '/'])
        if sha(source / relative / 'evidence_index.json') != expected:
            raise ValueError('source index changed during copy')
    # This new script is the only code copied; existing audit/source mirror is unchanged.
    command(['rsync', '-a', str(Path(__file__).resolve()), 'aml-b:' + remote + '/finalize_d0_serial.py'])
    command(['rsync', '-a', str(output / 'sources.json'), 'aml-b:' + remote + '/sources.json'])
    # Run the mirrored repository entrypoint (root-relative imports); root installs this
    # new file there before launch, and the isolated copy retains its exact bytes.
    remote_seconds = min(1800, int(COMMAND_DEADLINE.get() - time.time() - 5))
    if remote_seconds < 1:
        raise TimeoutError('insufficient shared budget for remote audit and cleanup')
    ssh(['timeout', str(remote_seconds), 'env', 'CUDA_VISIBLE_DEVICES=', 'OMP_NUM_THREADS=4',
         'OPENBLAS_NUM_THREADS=4', 'MKL_NUM_THREADS=4', 'NUMEXPR_NUM_THREADS=4',
         PYTHON, CODE + '/' + str(L2 / 'scripts/finalize_d0_serial.py'), '--remote-data', remote])
    for name in (*AUDITS, 'g1.log', 'g4.log'):
        command(['rsync', '-a', 'aml-b:' + remote + '/reports/' + name, str(output / name)])
    status = read(output / 'status.json')
    if status['sources'] != sources or status['sources_manifest_sha256'] != sha(output / 'sources.json'):
        raise ValueError('fetched source binding mismatch')
    for name, expected in status['reports_sha256'].items():
        if sha(output / name) != expected:
            raise ValueError('fetched audit hash mismatch')
    receipt = dict(identity=identity, remote_data=remote, supervisor=supervisor,
                   status='offline_reports_fetched', d0_frozen=False, gates_promoted=False,
                   total_wall_cap_seconds=25200, post_child_wall_cap_seconds=3600,
                   files_sha256={name: sha(output / name) for name in (*AUDITS, 'g1.log', 'g4.log', 'sources.json')})
    atomic(final, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--phase', type=Path, default=ROOT / L2 / 'validation/d0_20261018/serial_resume')
    parser.add_argument('--output', type=Path, default=ROOT / L2 / 'validation/d0_20261018/serial_postprocess')
    parser.add_argument('--remote-data', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.remote_data:
        remote_audits(args.remote_data)
    elif args.execute:
        finalize(args.phase, args.output)
    else:
        print('Planning only. --execute waits for the serial supervisor, copies sealed packages, and runs offline CPU audits.')


if __name__ == '__main__':
    main()
