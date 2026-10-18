#!/usr/bin/env python3
"""One-shot, zero-Actor T2 bookkeeping; optional new-package server audits.

The sole local analysis exception is the authorized original ARM64 Docker E1
scorer. All new evidence verification and G1/G4 analyses run on aml-b CPU4.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
L2 = Path('L2_Behavioral Event Localization/L2_Sandbox')
BASE = ROOT / L2 / 'validation/d0_20261018'
LABEL = 'officeval_050.r04'
REMOTE = Path('/home/jiasheren2/projects/d0_timeout600_postprocess')
PYTHON = '/home/jiasheren2/envs/agent-diagnosis-d0-py312/bin/python'
CODE = Path('/home/jiasheren2/projects/agent_diagnosis_d0_20261018/Users/jiashen/Desktop/问题诊断')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def assert_binding(binding):
    for relative, expected in binding['sources'].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f'pinned source changed: {relative}')


def wait_for(path, deadline):
    while not path.is_file():
        if time.time() >= deadline:
            return False
        time.sleep(min(15, max(0, deadline - time.time())))
    return True


def bounded(args, deadline, seconds=900, termination_grace=5, **kwargs):
    remaining = min(seconds, deadline - time.time())
    if remaining <= 0:
        raise TimeoutError('postprocessing hard deadline expired')
    process = subprocess.Popen(args, start_new_session=True, **kwargs)
    try:
        process.communicate(timeout=remaining)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.communicate(timeout=termination_grace)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
        raise
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, args)
    return subprocess.CompletedProcess(args, process.returncode)


class E1Terminated(BaseException):
    """Unwind the original Docker runtime's finally blocks on supervisor TERM."""


@contextmanager
def owned_e1_containers(phase, owner):
    """Only this E1 subprocess changes Docker create's bookkeeping flags."""
    original = subprocess.run
    previous = signal.getsignal(signal.SIGTERM)
    ids = phase / 'e1/container_ids'
    ids.mkdir(parents=True, exist_ok=True)
    sequence = 0

    def tracked(args, *positional, **kwargs):
        nonlocal sequence
        if isinstance(args, (list, tuple)) and list(args[:2]) == ['docker', 'create']:
            cidfile = ids / f'{sequence:06d}.cid'
            sequence += 1
            args = [*args[:2], '--label', f'd0.e1.owner={owner}', '--cidfile', str(cidfile), *args[2:]]
        return original(args, *positional, **kwargs)

    def terminate(signum, frame):
        raise E1Terminated('E1 supervisor requested termination')

    subprocess.run = tracked
    signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        subprocess.run = original
        signal.signal(signal.SIGTERM, previous)


def cleanup_e1(phase, owner, deadline):
    """Discover/remove only the exact unique owner label; require empty recheck.

    The label query also finds daemon-created containers when a killed docker
    create client had not yet returned its ID or written its cidfile.
    """
    if not re.fullmatch('[a-f0-9]{64}', owner):
        raise ValueError('invalid bound E1 owner')
    record = {'owner': owner, 'status': 'unconfirmed', 'owned_ids': [], 'actor_requests': 0}
    selector = ['docker', 'ps', '-aq', '--no-trunc', '--filter', f'label=d0.e1.owner={owner}']

    def docker(args):
        remaining = min(20, deadline - time.time())
        if remaining <= 0:
            raise TimeoutError('owned E1 cleanup deadline expired')
        return subprocess.run(args, capture_output=True, check=True, timeout=remaining)

    try:
        owned = docker(selector).stdout.decode('ascii').split()
        if any(not re.fullmatch('[a-f0-9]{64}', cid) for cid in owned):
            raise ValueError('Docker returned malformed owned container ID')
        record['owned_ids'] = owned
        # Persist the exact selected IDs before performing a destructive action.
        save(phase / 'e1/cleanup.json', record)
        if owned:
            docker(['docker', 'rm', '-f', *owned])
        remaining = docker(selector).stdout.decode('ascii').split()
        record.update(status='confirmed_empty' if not remaining else 'unconfirmed', remaining_ids=remaining)
    except Exception as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
    save(phase / 'e1/cleanup.json', record)
    return record


def seal_check(value):
    """Verify the driver's canonical JSON seal without importing its execution code."""
    clean = {k: v for k, v in value.items() if k != 'artifact_sha256'}
    digest = hashlib.sha256(json.dumps(clean, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    if value.get('artifact_sha256') != digest:
        raise ValueError('T2 receipt seal mismatch')


def snapshot(phase, plan):
    """Hash saved files; validity remains the driver's declaration until server verification."""
    runs = Path(plan['run_root'])
    receipts = {}
    for path in sorted(phase.glob('receipt_*.json')):
        row = read(path)
        seal_check(row)
        receipts[path.name] = {'sha256': sha(path), 'row': row}
    packages = {}
    for group in ('episodes', 'v3_g2_c0_resumes'):
        for package in sorted((runs / group).glob('*')):
            if package.is_symlink() or not re.fullmatch(r'officeval_\d{3}\.r0[0-4](?:_c0)?', package.name):
                raise ValueError('unexpected T2 package path')
            if not (package / 'evidence_index.json').is_file():
                packages[f'{group}/{package.name}'] = {'sealed': False, 'status': 'unsealed_or_partial'}
                continue
            config = read(package / 'raw/resolved_config.json')
            matching = receipts.get(f'receipt_{package.name}.json', {}).get('row')
            digest = sha(package / 'evidence_index.json')
            if matching and matching['package'].get('index_file_sha256') != digest:
                raise ValueError('receipt/package index binding mismatch')
            packages[f'{group}/{package.name}'] = {
                'sealed': True, 'index_sha256': digest, 'full_evidence_verified': False,
                'effective_timeout_seconds': config['model']['timeout_seconds'],
                'parent_timeout_seconds': matching.get('parent_timeout_seconds') if matching else None,
                'driver_study_valid': matching['package']['valid'] if matching else None,
                'comparison': matching.get('comparison') if matching else None}
    cohort = {}
    for label, record in plan['parents'].items():
        path = runs / 'episodes' / LABEL if label == LABEL else Path(record['package']['path'])
        cohort[label] = {'path': str(path), 'index_sha256': sha(path / 'evidence_index.json') if (path / 'evidence_index.json').is_file() else None,
                         'status': 'sealed_not_reclassified' if (path / 'evidence_index.json').is_file() else 'missing_or_partial'}
    saved = {p.name: sha(p) for p in phase.iterdir() if p.is_file() and p.suffix == '.json'}
    return {'cohort': cohort, 'new_packages': packages, 'receipts': receipts, 'phase_files_sha256': saved,
            'g2': {'purpose': 'joint cold restore plus timeout regime comparison; not pure equivalence',
                   'receipt_count': sum(n.endswith('_c0.json') for n in receipts), 'gate': 'undecided',
                   'pure_timeout300_equivalence_validator': 'not_run',
                   'note': 'receipt count includes invalid attempts; not a valid-pair count'}}


def e1_only(binding_path):
    binding = read(binding_path)
    assert_binding(binding)
    phase = Path(binding['phase'])
    (phase / 'e1_child_once.lock').mkdir()
    admitted = read(binding_path.parent / 'e1_input.json')
    if admitted['owner'] != sha(binding_path):
        raise ValueError('E1 ownership differs from bound one-shot phase')
    receipt_path = phase / f'receipt_{LABEL}.json'
    parent_index = Path(binding['run_root']) / 'episodes' / LABEL / 'evidence_index.json'
    if sha(receipt_path) != admitted['receipt_sha256'] or sha(parent_index) != admitted['index_sha256']:
        raise ValueError('E1 parent changed after receipt admission')
    census = load(ROOT / L2 / 'scripts/build_d0_census.py', 't2_original_census')
    runs = Path(binding['run_root'])
    plan = census.preflight(runs / 'episodes')
    ready = [r['slot'] for r in plan['slots'] if r['status'] == 'ready']
    if ready != [LABEL] or plan['study_valid'] != 1:
        raise ValueError('incremental E1 requires exactly the single new valid 050 parent')
    assert_binding(binding)
    save(phase / 'e1/plan.json', plan)
    with owned_e1_containers(phase, admitted['owner']):
        result = census.execute(plan, diag_dir=phase / 'e1/diagnoses', validation=phase / 'e1/validation')
    if result['completed_count'] != 1 or result['full_45_ready']:
        raise ValueError('incremental E1 incorrectly claims full cohort completion')
    assert_binding(binding)
    if sha(receipt_path) != admitted['receipt_sha256'] or sha(parent_index) != admitted['index_sha256']:
        raise ValueError('E1 parent changed during grading')


def remote_audits(data):
    if data.parent != REMOTE or not re.fullmatch(r'[0-9a-f]{64}', data.name):
        raise ValueError('unexpected isolated T2 audit directory')
    binding = read(data / 'binding.json')
    assert_binding(binding)
    sys.path[:0] = [str(ROOT / 'L1_sandbox/C3_Experimental_Sandbox/src'), str(ROOT / L2 / 'src')]
    recovery = load(ROOT / L2 / 'scripts/recover_d0_serial.py', 't2_recovery_helpers')
    selected = read(data / 'inventory.json')
    (data / 'audit_once.lock').mkdir()
    checked = {}
    for relative, row in selected.items():
        if not re.fullmatch(r'(episodes|v3_g2_c0_resumes)/officeval_\d{3}\.r0[0-4](?:_c0)?', relative):
            raise ValueError('unexpected audit package name')
        checked[relative] = recovery.verify_one(data / 'runs' / relative, row['index_sha256'])
    save(data / 'reports/verification.json', checked)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    for name, extra in (
        ('g1', ['--runs', str(data / 'runs/episodes'), '--execute', '--temp-parent', str(data / 'tmp')]),
        ('g4', [x for relative, row in checked.items() if row['study_outcome']['study_valid'] for x in ('--package', str(data / 'runs' / relative))])):
        if not extra or (name == 'g4' and not any(row['study_outcome']['study_valid'] for row in checked.values())):
            save(data / 'reports' / f'{name}.json', {'status': 'no_valid_new_packages', 'gate_pass': False})
            continue
        (data / 'tmp').mkdir(exist_ok=True)
        assert_binding(binding)
        script = ROOT / L2 / 'scripts' / ('audit_g1_mechanical.py' if name == 'g1' else 'audit_g4_provenance.py')
        with (data / 'reports' / f'{name}.log').open('xb') as log:
            result = subprocess.run([PYTHON, str(script), *extra, '--output', str(data / 'reports' / f'{name}.json')],
                                    env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600)
        if result.returncode:
            raise RuntimeError(f'{name} audit exited {result.returncode}')
    for relative, row in selected.items():
        recovery.verify_one(data / 'runs' / relative, row['index_sha256'])
    assert_binding(binding)
    save(data / 'reports/status.json', {'status': 'scoped_new_package_audits_completed',
        'actor_requests': 0, 'analyst_requests': 0, 'scope': sorted(selected),
        'per_audit_scopes': {
            'sealed_verification': sorted(selected),
            'g1': sorted(n for n in selected if n.startswith('episodes/')),
            'g4': sorted(n for n, row in checked.items() if row['study_outcome']['study_valid'])},
        'old44_audits': binding['old44_audits'], 'g2_gate': 'undecided', 'd0_frozen': False,
        'binding_sha256': sha(data / 'binding.json'), 'inventory_sha256': sha(data / 'inventory.json'),
        'files_sha256': {p.name: sha(p) for p in (data / 'reports').iterdir() if p.is_file()}})


def dispatch(binding_path, state, deadline):
    binding = read(binding_path)
    assert_binding(binding)
    out = binding_path.parent
    selected = {n: r for n, r in state['new_packages'].items() if r['sealed']}
    if not selected:
        return {'status': 'not_run_no_sealed_new_packages'}
    inventory_path = out / 'remote_inventory.json'
    save(inventory_path, selected)
    remote = REMOTE / sha(binding_path)
    def remote_call(args, seconds):
        return bounded(['ssh', '-oBatchMode=yes', '-oConnectTimeout=15', '-oServerAliveInterval=15',
                        '-oServerAliveCountMax=3', 'aml-b', shlex.join([str(x) for x in args])], deadline, seconds)
    remote_call(['mkdir', '-p', REMOTE], 30)
    remote_call(['mkdir', remote], 30)
    for local, remote_name in ((binding_path, 'binding.json'), (inventory_path, 'inventory.json')):
        bounded(['rsync', '-az', '--timeout=45', str(local), 'aml-b:' + str(remote / remote_name)], deadline, 90)
    transfers = {}
    for relative in selected:
        remaining = deadline - time.time()
        if remaining < 130:
            raise TimeoutError('insufficient bounded transfer/audit budget')
        # Single capped attempt keeps the aggregate deadline; partial bytes retained.
        args = ['rsync', '-az', '--partial', '--timeout=45', '-e', 'ssh -oBatchMode=yes -oConnectTimeout=15 -oServerAliveInterval=15 -oServerAliveCountMax=3',
                str(Path(binding['run_root']) / relative) + '/', 'aml-b:' + str(remote / 'runs' / relative) + '/']
        remote_call(['mkdir', '-p', remote / 'runs' / relative], 30)
        with (out / ('transfer_' + relative.replace('/', '_') + '.log')).open('xb') as log:
            bounded(args, deadline, 120, stdout=log, stderr=subprocess.STDOUT)
        transfers[relative] = selected[relative]['index_sha256']
    seconds = min(1500, int(deadline - time.time() - 90))
    if seconds < 1:
        raise TimeoutError('insufficient remote audit budget')
    remote_call(['timeout', str(seconds), 'env', 'CUDA_VISIBLE_DEVICES=', 'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4',
                  'MKL_NUM_THREADS=4', 'NUMEXPR_NUM_THREADS=4', PYTHON, CODE / L2 / 'scripts/finalize_d0_timeout600.py', '--remote-data', remote], seconds + 15)
    bounded(['rsync', '-az', '--timeout=45', 'aml-b:' + str(remote / 'reports') + '/', str(out / 'remote_reports') + '/'], deadline, 60)
    status = read(out / 'remote_reports/status.json')
    if status['binding_sha256'] != sha(binding_path) or status['inventory_sha256'] != sha(inventory_path):
        raise ValueError('remote binding mismatch')
    for name, digest in status['files_sha256'].items():
        if sha(out / 'remote_reports' / name) != digest:
            raise ValueError('fetched report bytes differ')
    assert_binding(binding)
    return {'status': status['status'], 'remote_data': str(remote), 'transfers': transfers,
            'scope': sorted(selected), 'per_audit_scopes': status['per_audit_scopes']}


def finalize(phase):
    plan = read(phase / 'executed_plan.json')
    out = phase / 'postprocess'
    out.mkdir(exist_ok=True)
    (out / 'one_shot.lock').mkdir()
    helper = load(ROOT / L2 / 'scripts/finalize_d0_serial.py', 't2_source_inventory')
    sources = helper.source_inventory(ROOT)
    sources.update(plan['source_fingerprints'])
    sources.update(read(BASE / 'prerequisite_resume/executed_plan.json')['source_fingerprints'])
    for name in ('build_d0_census.py', 'recover_d0_serial.py', 'finalize_d0_timeout600.py'):
        path = L2 / 'scripts' / name
        sources[str(path)] = sha(ROOT / path)
    old = BASE / 'serial_postprocess_recovery/manifest.json'
    binding = {'phase': str(phase), 'run_root': plan['run_root'], 'sources': sources,
               'plan_sha256': sha(phase / 'executed_plan.json'), 'service_stop_epoch': plan['service_stop_epoch'],
               'old44_audits': {'manifest': str(old), 'sha256': sha(old), 'scope': '44 valid serial parents; separate from new T2 audits'}}
    assert_binding(binding)
    binding_path = out / 'binding.json'
    save(binding_path, binding)
    deadline = plan['service_stop_epoch'] + 3600
    state = {'status': 'waiting_for_050_receipt', 'actor_requests': 0, 'analyst_requests': 0,
             'd0_frozen': False, 'gates_promoted': False, 'binding_sha256': sha(binding_path),
             'e1': {'status': 'pending', 'scope': 'new050only'}, 'remote_audits': {'status': 'pending'}}
    save(out / 'manifest.json', state)
    try:
        receipt = phase / f'receipt_{LABEL}.json'
        if wait_for(receipt, min(deadline, plan['service_stop_epoch'] + 120)):
            row = read(receipt); seal_check(row)
            if row['package']['valid']:
                parent_index = Path(plan['run_root']) / 'episodes' / LABEL / 'evidence_index.json'
                if sha(parent_index) != row['package']['index_file_sha256']:
                    raise ValueError('050 receipt index differs from E1 source')
                owner = sha(binding_path)
                save(out / 'e1_input.json', {'receipt_sha256': sha(receipt), 'index_sha256': sha(parent_index),
                    'owner': owner, 'docker_label': f'd0.e1.owner={owner}',
                    'cid_directory': str(phase / 'e1/container_ids')})
                state['e1'] = {'status': 'started', 'scope': 'single_new050_parent'}
                save(out / 'manifest.json', state)
                try:
                    with (out / 'e1.log').open('xb') as log:
                        bounded([sys.executable, str(Path(__file__).resolve()), '--e1-binding', str(binding_path)],
                                min(deadline, plan['service_stop_epoch'] + 25), 900, termination_grace=30,
                                stdout=log, stderr=subprocess.STDOUT)
                    manifest = phase / 'e1/validation/manifest.json'
                    state['e1'] = {'status': 'completed', 'scope': 'single_new050_parent', 'manifest_sha256': sha(manifest),
                                   'full_45_ready': False}
                except Exception as exc:
                    state['e1'] = {'status': 'failed_or_timed_out', 'error': f'{type(exc).__name__}: {exc}', 'full_45_ready': False}
                finally:
                    cleanup = cleanup_e1(phase, owner, min(deadline, plan['service_stop_epoch'] + 115))
                    state['e1']['container_cleanup'] = cleanup
                    if cleanup['status'] != 'confirmed_empty':
                        state['e1']['status'] = 'cleanup_unconfirmed'
            else:
                state['e1'] = {'status': 'not_run_study_invalid', 'scope': 'new050only'}
        else:
            state['e1'] = {'status': 'not_run_receipt_missing_at_deadline'}
        ended = wait_for(phase / 'supervisor_result.json', min(deadline, plan['service_stop_epoch'] + 120))
        assert_binding(binding)
        if sha(phase / 'executed_plan.json') != binding['plan_sha256']:
            raise ValueError('executed T2 plan changed during wait')
        state.update(snapshot(phase, plan))
        state['status'] = 'terminal_snapshot' if ended else 'partial_snapshot_supervisor_missing'
        save(out / 'manifest.json', state)  # Persist promptly before optional remote work.
        try:
            state['remote_audits'] = dispatch(binding_path, state, deadline)
        except Exception as exc:
            state['remote_audits'] = {'status': 'pending_failed_or_deadline', 'error': f'{type(exc).__name__}: {exc}'}
        assert_binding(binding)
    except Exception as exc:
        state.update(status='partial_postprocessing_error', error=f'{type(exc).__name__}: {exc}')
    finally:
        state['files_sha256'] = {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json'}
        save(out / 'manifest.json', state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--phase', type=Path, default=BASE / 'timeout600_t2')
    parser.add_argument('--e1-binding', type=Path)
    parser.add_argument('--remote-data', type=Path)
    args = parser.parse_args()
    if args.e1_binding:
        e1_only(args.e1_binding)
    elif args.remote_data:
        remote_audits(args.remote_data)
    elif args.execute:
        print(json.dumps(finalize(args.phase), ensure_ascii=False))
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
