#!/usr/bin/env python3
"""Operator helper for one Exp1 block. prepare is local; start-service and launch need --execute.

  prepare --block N --authorization A   lease (starts the six-hour clock), sealed plan, phase approval
  start-service --block N --execute     start the frozen vLLM service on aml-b GPU4 under that lease
  launch --block N --execute            wait for the tunnelled endpoint, preflight, start the supervisor
  status --block N                      read-only progress summary

The service owner tag is the block plan hash. The supervisor releases the
service and the block's containers when the block ends or the lease runs out.
No step renews a lease, restarts a block or resamples a slot.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import urllib.request

SCRIPT = Path(__file__).resolve()
DRIVER_PATH = SCRIPT.with_name('run_acceptance_gap_exp1.py')
SUPERVISOR = SCRIPT.with_name('supervise_acceptance_gap_exp1.py')
REMOTE_LAUNCHER_LOCAL = SCRIPT.with_name('start_owned_native_service.py')
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'aml-b']
READY_SECONDS = 600


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


driver = _load(DRIVER_PATH, 'acceptance_gap_launch_driver')
ROOT, L1 = driver.ROOT, driver.L1


def phase_dir(block):
    if type(block) is not int or not 0 <= block < 5:
        raise ValueError('block out of range')
    return driver.PHASE_ROOT / f'block_{block:02d}'


def exclusive_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return path


def native_env(spec):
    env = spec['service']['native_validator_env']
    return {key: str((ROOT / value).resolve()) for key, value in env.items()}


def git_state():
    head = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=30)
    dirty = subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain', '--untracked-files=no'],
                           capture_output=True, text=True, timeout=30)
    return dict(head=head.stdout.strip() or None, tracked_changes=bool(dirty.stdout.strip()))


def prepare(block, authorization, now=None):
    spec, _ = driver.load_spec()
    os.environ.update(native_env(spec))
    phase = phase_dir(block)
    if phase.exists() or phase.is_symlink():
        raise ValueError('phase already exists; a block is never re-prepared or restarted')
    phase.mkdir(parents=True, exist_ok=False)
    start = float(int(time.time() if now is None else now))
    lease = dict(started_epoch=start, hard_stop_epoch=start + driver.base.WALL,
                 service_stop_epoch=start + driver.base.WALL - 600)
    exclusive_json(phase / 'lease.json', lease)
    run_root = L1 / 'runs' / f'{driver.RUN_PREFIX}{block:02d}'
    try:
        plan = driver.build_block_plan(block=block, authorization=Path(authorization),
                                       service_lease=phase / 'lease.json', run_root=run_root)
    except Exception:
        # Nothing was started: remove the unused lease so a corrected input can be prepared.
        (phase / 'lease.json').unlink()
        phase.rmdir()
        raise
    exclusive_json(phase / 'plan.json', plan)
    auth = driver.read_json(Path(authorization))
    freeze = dict(schema='acceptance-gap-exp1-phase-approval-1', experiment_id=plan['experiment_id'], block=block,
                  phase_authorized=True, phase_frozen=True, internal_plan_reviewed=True,
                  review_basis='plan review_digest equals the user-approved preview digest in the authorization record',
                  review_digest=plan['review_digest'], approved_plan_sha256=plan['plan_sha256'], d0_frozen=False,
                  user_authorization_record=dict(exact_text=auth['exact_text'], source=auth['source'],
                                                 path=str(Path(authorization).resolve()),
                                                 sha256=driver.file_digest(Path(authorization))),
                  source_commit=git_state())
    exclusive_json(phase / 'freeze.json', freeze)
    return dict(phase=str(phase), plan_sha256=plan['plan_sha256'], lease=lease,
                next='start-service --block %d --execute' % block)


def remote(args, runner=subprocess.run, timeout=60, stdin=None):
    completed = runner([*SSH, shlex.join(args)], input=stdin, capture_output=True, text=True, timeout=timeout)
    if completed.returncode:
        raise RuntimeError('remote command failed: %s' % completed.stderr.strip()[-500:])
    return completed


def start_service(block, execute=False, runner=subprocess.run, clock=time.time):
    spec, _ = driver.load_spec()
    os.environ.update(native_env(spec))
    phase = phase_dir(block)
    for name in ('service.json', 'service_start_attempt.json', 'service_start_transport.json'):
        if (phase / name).exists():
            raise ValueError('service start was already attempted for this block: ' + name)
    plan, lease = driver.read_json(phase / 'plan.json'), driver.read_json(phase / 'lease.json')
    driver.validate_plan(plan)
    service = spec['service']
    python, code = service['remote_python'], Path(service['remote_code_root'])
    launcher = code / REMOTE_LAUNCHER_LOCAL.relative_to(ROOT)
    if not execute:
        return dict(status='not_executed', remote_launcher=str(launcher), plan_sha256=plan['plan_sha256'])
    expected = driver.file_digest(REMOTE_LAUNCHER_LOCAL)
    check = remote([python, '-c', 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())',
                    str(launcher)], runner=runner)
    if check.stdout.strip() != expected:
        raise ValueError('remote service launcher differs from the local frozen copy; sync it first')
    sent = clock()
    remote_epoch = float(remote([python, '-c', 'import time;print(repr(time.time()))'], runner=runner).stdout.strip())
    received = clock()
    payload = dict(phase_authorized=True, phase_frozen=True, plan_sha256=plan['plan_sha256'],
                   local_start_epoch=lease['started_epoch'], service_stop_epoch=lease['service_stop_epoch'],
                   clock_calibration=dict(local_sent_epoch=sent, local_received_epoch=received, remote_epoch=remote_epoch),
                   identity_receipt=service['remote_identity_receipt'])
    exclusive_json(phase / 'service_start_attempt.json', payload)
    completed = runner([*SSH, shlex.join([python, str(launcher), '--execute'])], input=json.dumps(payload),
                       capture_output=True, text=True, timeout=240)
    exclusive_json(phase / 'service_start_transport.json',
                   dict(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr))
    if completed.returncode:
        raise RuntimeError('remote service start failed; see service_start_transport.json')
    receipt = json.loads(completed.stdout.strip().splitlines()[-1])
    exclusive_json(phase / 'service.json', receipt)
    driver.validate_service(plan, phase / 'service.json')
    return dict(status='service_started', leader_pid=receipt.get('leader_pid'),
                next='bash scripts/start_tunnel.sh, then launch --block %d --execute' % block)


def wait_ready(opener=urllib.request.urlopen, seconds=None, sleep=time.sleep):
    until = time.monotonic() + (READY_SECONDS if seconds is None else seconds)
    while time.monotonic() < until:
        try:
            with opener('http://127.0.0.1:8001/health', timeout=3) as response:
                if response.status == 200:
                    with opener('http://127.0.0.1:8001/v1/models', timeout=3) as models_response:
                        models = json.load(models_response)
                    if {'Qwen/Qwen3.6-27B', 'Qwen3.6-27B'} <= {item['id'] for item in models['data']}:
                        return models
        except (OSError, ValueError, KeyError):
            pass
        sleep(min(5, max(0, until - time.monotonic())))
    raise TimeoutError('service readiness timeout through the local tunnel')


def launch(block, execute=False, opener=urllib.request.urlopen, popen=subprocess.Popen):
    spec, _ = driver.load_spec()
    env_update = native_env(spec)
    os.environ.update(env_update)
    phase = phase_dir(block)
    if any((phase / name).exists() for name in ('supervisor_launch.json', 'supervisor_started.json', 'startup_failed.json')):
        raise ValueError('block supervisor was already launched or failed to start')
    sup = _load(SUPERVISOR, 'acceptance_gap_launch_supervisor')
    names = [phase / n for n in ('plan.json', 'freeze.json', 'service.json')]
    plan, _, service = map(driver.read_json, names)
    if not execute:
        return dict(status='not_executed', plan_sha256=plan['plan_sha256'])
    # A readiness timeout leaves the service running under its own GNU timeout:
    # fix the tunnel and retry launch. Only a failed preflight releases it here.
    models = wait_ready(opener)
    launched = False
    try:
        sup.preflight(phase, *names)
        exclusive_json(phase / 'service_readiness.json', dict(status='ready', models=models,
                                                               checked_local_epoch=time.time(), actor_requests=0))
        env = os.environ.copy()
        env.update(env_update, PYTHONPATH=os.pathsep.join([str(L1 / 'src'), str(driver.L2 / 'src')]))
        args = [sys.executable, str(SUPERVISOR), '--execute', '--phase', str(phase), '--plan', str(names[0]),
                '--freeze', str(names[1]), '--service', str(names[2])]
        with (phase / 'supervisor.log').open('xb') as log:
            process = popen(args, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                            start_new_session=True)
        launched = True
        awake = None
        if Path('/usr/bin/caffeinate').exists():
            awake = popen(['/usr/bin/caffeinate', '-i', '-w', str(process.pid)], stdin=subprocess.DEVNULL,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        record = dict(pid=process.pid, pgid=process.pid, caffeinate_pid=getattr(awake, 'pid', None),
                      plan_sha256=plan['plan_sha256'], started_epoch=time.time(), command=args)
        exclusive_json(phase / 'supervisor_launch.json', record)
        return record
    finally:
        if not launched:
            try:
                release = sup.module.stop_service(service, time.time() + 35)
            except Exception as exc:
                release = dict(status='cleanup_unconfirmed', error=f'{type(exc).__name__}: {exc}')
            exclusive_json(phase / 'startup_failed.json', dict(status='setup_failed', release=release, actor_requests=0))


def status(block):
    phase = phase_dir(block)
    out = dict(phase=str(phase), exists=phase.exists())
    execution = phase / 'execution'
    if (execution / 'admission.json').is_file():
        admission = driver.read_json(execution / 'admission.json')
        out['http_admitted'] = admission.get('requests')
        out['invalid_attempts'] = admission.get('invalid_attempts')
        out['stop_reason'] = admission.get('stop_reason')
    receipts = sorted(execution.glob('receipt_*.json')) if execution.is_dir() else []
    out['slots_terminal'] = len(receipts)
    out['study_valid'] = sum(bool(driver.read_json(p).get('study_valid')) for p in receipts)
    for name in ('result.json', 'supervisor_result.json', 'startup_failed.json'):
        if (phase / name).is_file():
            value = driver.read_json(phase / name)
            out[name] = {k: value.get(k) for k in ('status', 'all_study_valid', 'study_valid_count', 'exit_code',
                                                     'wall_timeout', 'service_cleanup', 'container_cleanup') if k in value}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--block', type=int, required=True)
    p.add_argument('--authorization', type=Path, required=True)
    s = sub.add_parser('start-service')
    s.add_argument('--block', type=int, required=True)
    s.add_argument('--execute', action='store_true')
    lch = sub.add_parser('launch')
    lch.add_argument('--block', type=int, required=True)
    lch.add_argument('--execute', action='store_true')
    st = sub.add_parser('status')
    st.add_argument('--block', type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        value = prepare(args.block, args.authorization)
    elif args.command == 'start-service':
        value = start_service(args.block, execute=args.execute)
    elif args.command == 'launch':
        value = launch(args.block, execute=args.execute)
    else:
        value = status(args.block)
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
