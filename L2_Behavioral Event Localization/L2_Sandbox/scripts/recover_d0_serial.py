#!/usr/bin/env python3
"""Explicit offline recovery of serial postprocessing; never constructs a model client.

Preserves the original one-shot lock and partial mirror. Reuses only independently
verified packages from the original server mirror, with separate ordinary copies.
Run tests/audits on aml-b; local mode performs only file binding and SSH transport.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
L2 = Path('L2_Behavioral Event Localization/L2_Sandbox')
COHORT = 'v3_qwen36_pi_dev45_20261017'
CODE = Path('/home/jiasheren2/projects/agent_diagnosis_d0_20261018/Users/jiashen/Desktop/问题诊断')
PYTHON = '/home/jiasheren2/envs/agent-diagnosis-d0-py312/bin/python'
REMOTE = Path('/home/jiasheren2/projects/d0_serial_postprocess_recovery')
CHANGED = {'officeval_010.r00', 'officeval_050.r04'}
DEV = {'officeval_' + n for n in ('003', '004', '010', '033', '050', '060', '070', '085', '090')}
REPORTS = ('g1.json', 'g2.json', 'g4.json', 'status.json', 'verification.json', 'g1.log', 'g4.log')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def run(args, timeout=600, **kwargs):
    return subprocess.run(args, check=True, timeout=timeout, **kwargs)


def ssh(args, timeout=600, **kwargs):
    return run(['ssh', '-oBatchMode=yes', '-oConnectTimeout=15',
                '-oServerAliveInterval=15', '-oServerAliveCountMax=3',
                'aml-b', shlex.join([str(x) for x in args])], timeout=timeout, **kwargs)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def validate_inventory(packages):
    expected = {f'{task}.r0{n}' for task in DEV for n in range(5)}
    if set(packages) != expected:
        raise ValueError('recovery requires the exact 45 DEV parent slots')
    if any(not re.fullmatch(r'[0-9a-f]{64}', digest) for digest in packages.values()):
        raise ValueError('invalid externally anchored index digest')


def check_sources(binding, root):
    for section in ('sources', 'frozen_sources'):
        for relative, digest in binding[section].items():
            if sha(root / relative) != digest:
                raise ValueError(f'{section} changed: {relative}')
    if sha(Path(__file__).resolve()) != binding['recovery_script_sha256']:
        raise ValueError('recovery script changed')


def verify_one(package, expected):
    # Full inventory, every file hash, journal chain, checkpoint references and grade.
    from c3_sandbox.l1.package import verify_package
    from c3_sandbox.l1.evaluation.study import study_outcome
    check = verify_package(package, expected_index_sha256=expected)
    if not check.get('verified'):
        raise ValueError(f'full sealed verification failed: {package}: {check}')
    outcome = study_outcome(package, verify_evidence=True)
    return {'index_sha256': expected, 'full_evidence_valid': True, 'study_outcome': outcome}


def remote_stage(data, stage):
    import sys
    if data.parent != REMOTE or not re.fullmatch(r'[0-9a-f]{64}', data.name):
        raise ValueError('unexpected recovery directory')
    binding = read(data / 'binding.json')
    validate_inventory(binding['packages'])
    check_sources(binding, ROOT)
    sys.path[:0] = [str(ROOT / 'L1_sandbox/C3_Experimental_Sandbox/src'), str(ROOT / L2 / 'src')]
    episodes = data / COHORT / 'episodes'
    if stage == 'prepare':
        episodes.mkdir(parents=True, exist_ok=False)
        reused = {}
        mirror = ROOT / 'L1_sandbox/C3_Experimental_Sandbox/runs' / COHORT / 'episodes'
        for name, digest in binding['packages'].items():
            if name in CHANGED:
                continue
            reused[name] = verify_one(mirror / name, digest)
            if not reused[name]['study_outcome']['study_valid']:
                raise ValueError('reusable original parent is not study-valid')
            # No shared inodes: neither later transfer nor audit can mutate originals.
            shutil.copytree(mirror / name, episodes / name)
        check_sources(binding, ROOT)
        write(data / 'reuse_verification.json', reused)
        return
    (data / 'audit_once.lock').mkdir()
    verified = {name: verify_one(episodes / name, digest) for name, digest in binding['packages'].items()}
    write(data / 'verification.json', verified)
    valid = [name for name, row in verified.items() if row['study_outcome']['study_valid']]
    if len(valid) != 44 or set(verified) - set(valid) != {'officeval_050.r04'}:
        raise ValueError('recovery terminal inventory differs from 44 valid and 050.r04 invalid')
    if (data / COHORT / 'v3_g2_c0_resumes').exists():
        raise ValueError('unexpected G2 children: cannot silently omit them')
    output = data / 'reports'
    output.mkdir()
    (data / 'tmp').mkdir()
    shutil.copyfile(data / 'verification.json', output / 'verification.json')
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    commands = {}
    for name, extra in (
        ('g1', ['--runs', str(episodes), '--execute', '--temp-parent', str(data / 'tmp')]),
        ('g4', [x for name in valid for x in ('--package', str(episodes / name))])):
        check_sources(binding, ROOT)
        script = ROOT / L2 / 'scripts' / ('audit_g1_mechanical.py' if name == 'g1' else 'audit_g4_provenance.py')
        args = [PYTHON, str(script), *extra, '--output', str(output / f'{name}.json')]
        with (output / f'{name}.log').open('xb') as log:
            completed = run(args, timeout=900, env=environment, stdout=log, stderr=subprocess.STDOUT)
        commands[name] = {'argv': args, 'exit_code': completed.returncode, 'script_sha256': sha(script)}
    evaluator = load(ROOT / 'scripts/v3_evaluate_gates_g2_g3_g4.py', 'recovery_g2')
    write(output / 'g2.json', evaluator.evaluate_r19_g2(data))
    # Verify data again after audits; inputs must remain sealed and source-bound.
    for name, digest in binding['packages'].items():
        verify_one(episodes / name, digest)
    check_sources(binding, ROOT)
    write(output / 'status.json', {
        'status': 'offline_audits_completed', 'actor_requests': 0, 'analyst_requests': 0,
        'd0_frozen': False, 'gates_promoted': False, 'sealed_parents': 45, 'study_valid_parents': len(valid),
        'g2_children': 0, 'binding_sha256': sha(data / 'binding.json'), 'commands': commands,
        'reports_sha256': {name: sha(output / name) for name in REPORTS if name != 'status.json'}})


def transfer(source, target, log):
    """At most two bounded attempts; partial files are resumed, never unbounded retries."""
    args = ['rsync', '-az', '--partial', '--timeout=45', '--stats',
            '-e', 'ssh -oBatchMode=yes -oConnectTimeout=15 -oServerAliveInterval=15 -oServerAliveCountMax=3',
            str(source) + '/', 'aml-b:' + str(target) + '/']
    attempts = []
    with log.open('xb') as stream:
        for attempt in range(2):
            started = time.time()
            try:
                result = run(args, timeout=600, stdout=stream, stderr=subprocess.STDOUT)
                attempts.append({'attempt': attempt + 1, 'elapsed_seconds': time.time() - started,
                                 'exit_code': result.returncode})
                return attempts
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                attempts.append({'attempt': attempt + 1, 'elapsed_seconds': time.time() - started,
                                 'error': str(exc)})
        raise RuntimeError(f'bounded resumable transfer failed: {attempts}')


def recover(output):
    output.mkdir(parents=True, exist_ok=False)
    (output / 'recovery_once.lock').mkdir()
    phase = ROOT / L2 / 'validation/d0_20261018/serial_resume'
    old = ROOT / L2 / 'validation/d0_20261018/serial_postprocess'
    prereq = ROOT / L2 / 'validation/d0_20261018/prerequisite_resume/executed_plan.json'
    # Terminal supervisor and original non-success receipt are required before copying.
    if read(phase / 'supervisor_result.json')['exit_code'] != 0:
        raise ValueError('serial supervisor did not terminate normally')
    if not (old / 'one_shot.lock').is_dir() or (old / 'manifest.json').exists():
        raise ValueError('original failure lock/history missing or already completed')
    helper = load(ROOT / L2 / 'scripts/finalize_d0_serial.py', 'original_finalizer')
    runs = ROOT / 'L1_sandbox/C3_Experimental_Sandbox/runs' / COHORT
    packages = {p.name: sha(p / 'evidence_index.json') for p in sorted((runs / 'episodes').glob('*'))
                if re.fullmatch(r'officeval_\d{3}\.r0[0-4]', p.name) and p.name.split('.r')[0] in DEV}
    validate_inventory(packages)
    if any((runs / 'v3_g2_c0_resumes').glob('*/manifest.json')):
        raise ValueError('G2 children exist: recovery needs explicit inventory expansion')
    binding = {'packages': packages, 'sources': helper.source_inventory(ROOT),
               'frozen_sources': read(prereq)['source_fingerprints'],
               'recovery_script_sha256': sha(Path(__file__).resolve()),
               'serial_plan_sha256': sha(phase / 'executed_plan.json'),
               'serial_result_sha256': sha(phase / 'result.json'),
               'serial_supervisor_sha256': sha(phase / 'supervisor_result.json'),
               'original_failure_sha256': sha(old / 'execution.log'),
               'original_failure_supervisor_sha256': sha(old / 'supervisor_result.json'),
               'actor_requests': 0, 'analyst_requests': 0, 'd0_frozen': False, 'gates_promoted': False}
    check_sources(binding, ROOT)
    write(output / 'binding.json', binding)
    run_id = sha(output / 'binding.json')
    remote = REMOTE / run_id
    ssh(['mkdir', '-p', REMOTE])
    ssh(['mkdir', remote])
    run(['rsync', '-az', str(output / 'binding.json'), 'aml-b:' + str(remote) + '/binding.json'])
    # Installation handled explicitly before launch; both copies checked by binding.
    remote_entry = CODE / L2 / 'scripts/recover_d0_serial.py'
    env = ['env', 'CUDA_VISIBLE_DEVICES=', 'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4',
           'MKL_NUM_THREADS=4', 'NUMEXPR_NUM_THREADS=4']
    ssh(['timeout', '900', *env, PYTHON, remote_entry, '--remote-data', remote, '--stage', 'prepare'], timeout=930)
    attempts = {}
    for name in sorted(CHANGED):
        attempts[name] = transfer(runs / 'episodes' / name, remote / COHORT / 'episodes' / name,
                                  output / f'transfer_{name}.log')
        if sha(runs / 'episodes' / name / 'evidence_index.json') != packages[name]:
            raise ValueError('changed source index during transfer')
    write(output / 'transfers.json', attempts)
    check_sources(binding, ROOT)
    ssh(['timeout', '2400', *env, PYTHON, remote_entry, '--remote-data', remote, '--stage', 'audit'], timeout=2430)
    for name in REPORTS:
        run(['rsync', '-az', '--timeout=45', 'aml-b:' + str(remote / 'reports' / name), str(output / name)], timeout=90)
    status = read(output / 'status.json')
    if status['binding_sha256'] != sha(output / 'binding.json'):
        raise ValueError('remote input/source binding mismatch')
    for name, digest in status['reports_sha256'].items():
        if sha(output / name) != digest:
            raise ValueError('fetched report digest mismatch')
    check_sources(binding, ROOT)
    write(output / 'manifest.json', {
        'status': 'offline_reports_fetched', 'remote_data': str(remote),
        'actor_requests': 0, 'analyst_requests': 0, 'd0_frozen': False, 'gates_promoted': False,
        'binding_sha256': sha(output / 'binding.json'), 'original_failure_preserved': True,
        'reused_verified_parents': 43, 'transferred_changed_parents': sorted(CHANGED),
        'sealed_parents': 45, 'study_valid_parents': status['study_valid_parents'],
        'g2': read(output / 'g2.json'),
        'files_sha256': {name: sha(output / name) for name in (*REPORTS, 'binding.json', 'transfers.json')}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / L2 / 'validation/d0_20261018/serial_postprocess_recovery')
    parser.add_argument('--remote-data', type=Path)
    parser.add_argument('--stage', choices=('prepare', 'audit'))
    args = parser.parse_args()
    if args.remote_data:
        remote_stage(args.remote_data, args.stage)
    elif args.execute:
        recover(args.output)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
