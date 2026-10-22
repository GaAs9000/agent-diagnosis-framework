#!/usr/bin/env python3
"""Run the authorized Exp1 blocks one after another. Inert unless --execute.

For each block:
  1. wait until aml-b GPU4 is idle on several consecutive polls, you hold at most one
     other GPU, remote port 8001 is free, and local disk/Docker are fine;
  2. launch_acceptance_gap_block.py prepare -> start-service -> (tunnel) -> launch;
  3. wait for the block supervisor to finish and release the GPU, then check the block.

The chain stops, and never restarts a block that has started its service, when:
  - the STOP file exists before a block (a running block always finishes),
  - a block has more than 6 of 33 study-invalid runs, an integrity error, a non-zero
    driver exit or unconfirmed container/service cleanup,
  - the pre-registered interim analysis after block 2 returns stop_futility,
  - GPU4 stays unavailable longer than --max-gpu-wait-hours, or aml-b is unreachable,
  - anything unexpected happens.
A service start that provably launched nothing (no remote owner directory) is archived
under failed_starts/ and the block is prepared again after the next idle window.
Analyses run on aml-b CPU (analyze_acceptance_gap_exp1.py --inputs); locally the chain
only collects sealed receipts and orchestrates over SSH.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback

SCRIPT = Path(__file__).resolve()
L2 = SCRIPT.parents[1]
ROOT = L2.parents[1]
sys.path.insert(0, str(L2 / 'src'))
from l2_sandbox.v3_execution import file_digest, read_json, seal, verify_seal, write_json  # noqa: E402

LAUNCHER = SCRIPT.with_name('launch_acceptance_gap_block.py')
ANALYSIS = SCRIPT.with_name('analyze_acceptance_gap_exp1.py')
SUPERVISOR = SCRIPT.with_name('supervise_acceptance_gap_exp1.py')
SERVICE_LAUNCHER = SCRIPT.with_name('start_owned_native_service.py')
L2_INIT = L2 / 'src/l2_sandbox/__init__.py'
V3_EXECUTION = L2 / 'src/l2_sandbox/v3_execution.py'
SPEC = L2 / 'configs/acceptance_gap_exp1_20261022.json'
PHASE_ROOT = L2 / 'validation/acceptance_gap_exp1_20261022'
CHAIN_ROOT = PHASE_ROOT / 'chain'
STOP_FILE = PHASE_ROOT / 'STOP'
TUNNEL = ROOT / 'scripts/start_tunnel.sh'
TUNNEL_SOCKET = Path.home() / '.ssh/sockets/aml-b-8001.sock'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'aml-b']
SCP = ['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15']
REMOTE_BASE = '/home/jiasheren2/projects/acceptance_gap_exp1_20261022'
SLOTS_PER_BLOCK = 33
MAX_INVALID = 6          # more than 20% of 33 runs invalid stops the chain
INTERIM_AFTER = 2        # pre-registered interim analysis after blocks 0-2
INTERIM_TAG = 'interim_b00_b02'
FINAL_TAG = 'final_b00_b04'
MIN_FREE_BYTES = 10 * 1024 ** 3

PROBE = r'''
import json, os, pwd, subprocess
def query(args):
    return subprocess.run(['nvidia-smi', *args, '--format=csv,noheader,nounits'], capture_output=True,
                          text=True, timeout=20, check=True).stdout
def number(value):
    try:
        return int(float(value))
    except ValueError:
        return None
gpus = []
for line in query(['--query-gpu=index,uuid,memory.used,utilization.gpu']).splitlines():
    if line.strip():
        index, uuid, memory, util = [x.strip() for x in line.split(',')]
        gpus.append(dict(index=index, uuid=uuid, memory_mib=number(memory), util=number(util)))
apps = []
for line in query(['--query-compute-apps=pid,gpu_uuid']).splitlines():
    if line.strip():
        pid, uuid = [x.strip() for x in line.split(',')]
        try:
            user = pwd.getpwuid(os.stat('/proc/' + pid).st_uid).pw_name
        except Exception:
            user = None
        apps.append(dict(pid=int(pid), uuid=uuid, user=user))
listening = subprocess.run(['ss', '-ltnH', 'sport = :8001'], capture_output=True, text=True, timeout=10).stdout.strip()
print(json.dumps(dict(user=pwd.getpwuid(os.getuid()).pw_name, gpus=gpus, apps=apps,
                      port8001_listening=bool(listening))))
'''


class ChainStop(Exception):
    def __init__(self, reason, detail=None):
        super().__init__(reason)
        self.reason, self.detail = reason, detail


def gpu_ready(probe, gpu_index, gpu_uuid):
    """Pure decision on one probe: (ready, reasons)."""
    reasons = []
    gpu = next((g for g in probe.get('gpus', []) if g.get('index') == gpu_index), None)
    if gpu is None or gpu.get('uuid') != gpu_uuid:
        reasons.append('GPU%s identity differs from the frozen UUID' % gpu_index)
    elif gpu.get('memory_mib') is None or gpu.get('util') is None or gpu['memory_mib'] > 4 or gpu['util'] != 0:
        reasons.append('GPU%s busy (%s MiB, %s%%)' % (gpu_index, gpu.get('memory_mib'), gpu.get('util')))
    if any(app.get('uuid') == gpu_uuid for app in probe.get('apps', [])):
        reasons.append('compute process present on GPU%s' % gpu_index)
    own = {app['uuid'] for app in probe.get('apps', []) if app.get('user') == probe.get('user') and app.get('uuid') != gpu_uuid}
    if len(own) > 1:
        reasons.append('already holding %d other GPUs (cap is 2 in total)' % len(own))
    if probe.get('port8001_listening'):
        reasons.append('remote port 8001 is in use')
    return not reasons, reasons


def evaluate_block(phase):
    """Check a finished block phase; returns a dict with ok and reasons."""
    phase = Path(phase)
    reasons, out = [], dict(phase=str(phase))
    supervisor = phase / 'supervisor_result.json'
    if not supervisor.is_file():
        return dict(out, ok=False, reasons=['supervisor result missing'])
    sr = read_json(supervisor)
    if sr.get('exit_code') != 0:
        reasons.append('block driver exit code %r' % sr.get('exit_code'))
    if sr.get('wall_timeout'):
        reasons.append('block hit the supervisor wall-clock limit')
    if (sr.get('service_cleanup') or {}).get('status') != 'confirmed_absent':
        reasons.append('GPU service release unconfirmed')
    if (sr.get('container_cleanup') or {}).get('status') != 'confirmed_empty':
        reasons.append('container cleanup unconfirmed')
    result_path = phase / 'result.json'
    if not result_path.is_file():
        reasons.append('block result missing')
    else:
        result = read_json(result_path)
        try:
            verify_seal(result)
        except ValueError:
            reasons.append('block result seal invalid')
        if result.get('integrity_error'):
            reasons.append('integrity error: %s' % result['integrity_error'])
        if result.get('slots_terminal') != SLOTS_PER_BLOCK:
            reasons.append('only %r of %d slots terminal' % (result.get('slots_terminal'), SLOTS_PER_BLOCK))
        valid = int(result.get('study_valid_count') or 0)
        out.update(study_valid=valid, invalid=SLOTS_PER_BLOCK - valid, by_condition=result.get('by_condition'),
                   http_admitted=(result.get('admissions') or {}).get('requests'))
        if SLOTS_PER_BLOCK - valid > MAX_INVALID:
            reasons.append('%d of %d runs study-invalid (more than 20%%)' % (SLOTS_PER_BLOCK - valid, SLOTS_PER_BLOCK))
    return dict(out, ok=not reasons, reasons=reasons)


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def block_progress(phase):
    execution = Path(phase) / 'execution'
    out = dict(slots_terminal=len(list(execution.glob('receipt_*.json'))) if execution.is_dir() else 0)
    try:
        out['http_admitted'] = read_json(execution / 'admission.json').get('requests')
    except (OSError, ValueError):
        pass
    return out


class Chain:
    def __init__(self, args, run_dir, clock=time.time, sleep=time.sleep, runner=subprocess.run):
        self.args, self.run_dir, self.clock, self.sleep, self.runner = args, Path(run_dir), clock, sleep, runner
        self.auth = Path(args.authorization).resolve()
        self.spec = read_json(SPEC)
        self.spec_sha = file_digest(SPEC)
        service = self.spec['service']
        self.gpu_index, self.gpu_uuid = str(service['gpu_index']), service['gpu_uuid']
        self.remote_py = service['remote_python']
        self.state = dict(stage='starting', blocks=list(args.blocks), tunnel_restarts=0)

    # ------------------------------------------------------------ bookkeeping

    def event(self, kind, **data):
        line = json.dumps(dict(at=self.clock(), kind=kind, **data), ensure_ascii=False, default=str)
        with (self.run_dir / 'events.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(line + '\n')

    def status(self, **fields):
        self.state.update(fields, updated_at=self.clock())
        write_json(self.run_dir / 'status.json', self.state)

    def run(self, cmd, timeout, label=None):
        try:
            completed = self.runner(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            completed = subprocess.CompletedProcess(cmd, -9, exc.stdout or '', 'timeout after %ss' % timeout)
        self.event('command', label=label or cmd[0], returncode=completed.returncode,
                   stdout=(completed.stdout or '')[-1500:], stderr=(completed.stderr or '')[-1500:])
        return completed

    def must(self, cmd, timeout, reason, label=None):
        completed = self.run(cmd, timeout, label)
        if completed.returncode:
            raise ChainStop(reason, (completed.stderr or completed.stdout or '')[-800:])
        return completed

    def cli(self, *args, timeout):
        return self.run([sys.executable, str(LAUNCHER), *args], timeout, label='launcher ' + args[0])

    def phase(self, block):
        return PHASE_ROOT / ('block_%02d' % block)

    def check_stop(self):
        if STOP_FILE.exists():
            raise ChainStop('operator_stop', 'STOP file present: ' + str(STOP_FILE))

    # ------------------------------------------------------------ remote state

    def probe(self):
        completed = self.run(SSH + [shlex.join([self.remote_py, '-c', PROBE])], 60, label='gpu probe')
        if completed.returncode:
            raise RuntimeError((completed.stderr or 'probe failed')[-300:])
        return json.loads(completed.stdout.strip().splitlines()[-1])

    def remote_exists(self, path):
        completed = self.run(SSH + [shlex.join(['test', '-e', path]) + ' && echo yes || echo no'], 60,
                             label='remote exists')
        answer = (completed.stdout or '').strip()
        return {'yes': True, 'no': False}.get(answer)

    def remote_hashes(self, expected, reason):
        completed = self.must(SSH + [shlex.join(['sha256sum', *expected])], 120, reason, label='remote sha256')
        actual = {}
        for line in completed.stdout.splitlines():
            digest_value, _, name = line.partition('  ')
            actual[name.strip()] = digest_value.strip()
        if actual != expected:
            raise ChainStop(reason, dict(expected=expected, actual=actual))

    def wait_for_gpu(self, block):
        deadline = self.clock() + self.args.max_gpu_wait_hours * 3600
        consecutive = failures = 0
        while True:
            self.check_stop()
            try:
                probe = self.probe()
                failures = 0
                ready, reasons = gpu_ready(probe, self.gpu_index, self.gpu_uuid)
            except Exception as exc:
                failures += 1
                ready, reasons, probe = False, ['probe failed: %s' % exc], None
                if failures >= self.args.max_probe_failures:
                    raise ChainStop('aml_unreachable', reasons[0])
            consecutive = consecutive + 1 if ready else 0
            gpu = next((g for g in (probe or {}).get('gpus', []) if g.get('index') == self.gpu_index), None)
            self.status(stage='waiting_for_gpu', block=block, consecutive_idle_polls=consecutive,
                        required_idle_polls=self.args.idle_polls, not_ready_reasons=reasons, gpu=gpu)
            if consecutive >= self.args.idle_polls:
                self.event('gpu_ready', block=block, gpu=gpu)
                return probe
            if self.clock() >= deadline:
                raise ChainStop('gpu_wait_timeout', reasons)
            self.sleep(self.args.poll_seconds)

    def local_ready(self):
        if shutil.disk_usage(ROOT).free < MIN_FREE_BYTES:
            raise ChainStop('local_disk_low', 'less than 10 GiB free')
        if self.run(['docker', 'info', '--format', '{{.ServerVersion}}'], 60, label='docker info').returncode:
            raise ChainStop('docker_unavailable')

    def tunnel_alive(self):
        if not TUNNEL_SOCKET.exists():
            return False
        return self.run(['ssh', '-S', str(TUNNEL_SOCKET), '-O', 'check', 'aml-b'], 20, label='tunnel check').returncode == 0

    def ensure_tunnel(self):
        if self.tunnel_alive():
            return False
        # start_tunnel.sh leaves a backgrounded `ssh -f` master; never capture its output
        # through pipes (the master would keep them open), log it to a file instead.
        log = self.run_dir / 'tunnel.log'
        with log.open('a', encoding='utf-8') as handle:
            try:
                code = self.runner(['bash', str(TUNNEL)], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle,
                                   stderr=handle, timeout=120).returncode
            except subprocess.TimeoutExpired:
                code = -9
        self.event('command', label='start tunnel', returncode=code, log=str(log))
        if code or not self.tunnel_alive():
            raise ChainStop('tunnel_failed', 'see ' + str(log))
        self.state['tunnel_restarts'] += 1
        self.event('tunnel_started')
        return True

    # ------------------------------------------------------------ one block

    def service_root(self):
        spec = importlib.util.spec_from_file_location('chain_service_launcher', SERVICE_LAUNCHER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return str(module.SERVICE_ROOT)

    def safe_to_archive(self, phase):
        """True only when no GPU service can have been launched for this phase."""
        if (phase / 'service.json').exists() or not (phase / 'plan.json').exists():
            return False
        attempt, transport = phase / 'service_start_attempt.json', phase / 'service_start_transport.json'
        if attempt.exists() and not transport.exists():
            return False
        if transport.exists() and read_json(transport).get('returncode') == 0:
            return False
        owner = read_json(phase / 'plan.json')['plan_sha256']
        return self.remote_exists(self.service_root() + '/' + owner) is False

    def archive(self, phase, attempt):
        target = PHASE_ROOT / 'failed_starts' / ('%s.attempt%d.%s' % (phase.name, attempt, time.strftime('%Y%m%dT%H%M%S')))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(phase), str(target))
        self.event('service_start_archived', phase=str(phase), archived_to=str(target))

    def start_block(self, block):
        phase = self.phase(block)
        for attempt in range(1, self.args.max_start_attempts + 1):
            self.wait_for_gpu(block)
            self.local_ready()
            if phase.exists():
                raise ChainStop('phase_exists', str(phase))
            self.status(stage='preparing', block=block, start_attempt=attempt)
            if self.cli('prepare', '--block', str(block), '--authorization', str(self.auth), timeout=180).returncode:
                raise ChainStop('prepare_failed')
            self.status(stage='starting_service', block=block)
            started = self.cli('start-service', '--block', str(block), '--execute', timeout=420)
            if started.returncode == 0 and (phase / 'service.json').exists():
                return phase
            if (phase / 'service.json').exists():
                raise ChainStop('service_start_failed', self.emergency_release(phase))
            if not self.safe_to_archive(phase):
                raise ChainStop('service_start_failed',
                                'a service may have been launched without a local receipt; check aml-b before continuing')
            self.archive(phase, attempt)
        raise ChainStop('service_start_failed', '%d attempts' % self.args.max_start_attempts)

    def launch_block(self, block, phase):
        for attempt in range(1, 4):
            self.ensure_tunnel()
            self.status(stage='launching', block=block, launch_attempt=attempt)
            launched = self.cli('launch', '--block', str(block), '--execute', timeout=900)
            if launched.returncode == 0 and (phase / 'supervisor_launch.json').exists():
                return read_json(phase / 'supervisor_launch.json')
            if (phase / 'startup_failed.json').exists():
                raise ChainStop('startup_failed', read_json(phase / 'startup_failed.json'))
            if (phase / 'supervisor_launch.json').exists() or (phase / 'supervisor_started.json').exists():
                raise ChainStop('launch_state_unclear', 'supervisor files exist after a failed launch command')
            self.event('launch_retry', block=block, attempt=attempt)
        raise ChainStop('service_not_ready', self.emergency_release(phase))

    def emergency_release(self, phase):
        """Release the GPU service and this block's containers by their recorded owners."""
        record = dict(at=self.clock())
        spec = importlib.util.spec_from_file_location('chain_release_supervisor', SUPERVISOR)
        sup = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sup)
        if (phase / 'service.json').exists():
            try:
                record['service'] = sup.module.stop_service(read_json(phase / 'service.json'), time.time() + 45)
            except Exception as exc:
                record['service'] = dict(status='cleanup_unconfirmed', error='%s: %s' % (type(exc).__name__, exc))
        started = phase / 'supervisor_started.json'
        if started.exists():
            try:
                helper = sup.module.load(sup.module.HELPER, 'chain_release_helper')
                record['containers'] = helper.cleanup_e1(phase, read_json(started)['resource_owner'], time.time() + 60)
            except Exception as exc:
                record['containers'] = dict(status='cleanup_unconfirmed', error='%s: %s' % (type(exc).__name__, exc))
        target = phase / ('chain_emergency_release_%d.json' % int(self.clock()))
        write_json(target, record)
        self.event('emergency_release', phase=str(phase), record=record)
        return record

    def wait_supervisor(self, block, phase, launch):
        pid = launch['pid']
        deadline = read_json(phase / 'plan.json')['service_stop_epoch'] + 3600
        gone_since = None
        while True:
            result_exists = (phase / 'supervisor_result.json').exists()
            alive = pid_alive(pid)
            if result_exists and not alive:
                return
            if not alive:
                gone_since = gone_since or self.clock()
                if self.clock() - gone_since > 180:
                    raise ChainStop('supervisor_ended_without_result', self.emergency_release(phase))
            if self.clock() > deadline:
                raise ChainStop('supervisor_overran_lease', self.emergency_release(phase))
            if alive and not self.tunnel_alive():
                try:
                    self.ensure_tunnel()
                except ChainStop as exc:
                    self.event('tunnel_restart_failed', detail=exc.detail)
            self.status(stage='block_running', block=block, supervisor_pid=pid, progress=block_progress(phase))
            self.sleep(60)

    def run_block(self, block):
        phase = self.start_block(block)
        launch = self.launch_block(block, phase)
        self.event('block_launched', block=block, supervisor_pid=launch['pid'], plan_sha256=launch.get('plan_sha256'))
        self.wait_supervisor(block, phase, launch)
        verdict = evaluate_block(phase)
        self.event('block_finished', block=block, verdict=verdict)
        if not verdict['ok']:
            raise ChainStop('block_check_failed', verdict)
        return verdict

    # ------------------------------------------------------------ analysis on aml-b CPU

    def analysis(self, tag, blocks):
        local = PHASE_ROOT / ('analysis_' + tag)
        if (local / 'summary.json').exists():
            value = read_json(local / 'summary.json')
            verify_seal(value)
            return value
        local.mkdir(parents=True, exist_ok=True)
        inputs = local / 'inputs.json'
        if not inputs.exists():
            phases = [str(self.phase(b)) for b in blocks]
            self.must([sys.executable, str(ANALYSIS), '--export-inputs', str(inputs), '--blocks', *phases], 900,
                      'analysis_export_failed', label='export analysis inputs')
        remote = '%s/analysis/%s_%s' % (REMOTE_BASE, tag, time.strftime('%Y%m%dT%H%M%S'))
        layout = {ANALYSIS: remote + '/repo/x/L2_Sandbox/scripts/' + ANALYSIS.name,
                  L2_INIT: remote + '/repo/x/L2_Sandbox/src/l2_sandbox/__init__.py',
                  V3_EXECUTION: remote + '/repo/x/L2_Sandbox/src/l2_sandbox/v3_execution.py',
                  SPEC: remote + '/spec.json', inputs: remote + '/inputs.json'}
        directories = sorted({str(Path(p).parent) for p in layout.values()})
        self.must(SSH + [shlex.join(['mkdir', '-p', *directories])], 60, 'analysis_remote_prepare_failed')
        for source, target in layout.items():
            self.must(SCP + [str(source), 'aml-b:' + target], 300, 'analysis_upload_failed', label='upload')
        uploaded = {target: file_digest(source) for source, target in layout.items()}
        self.remote_hashes(uploaded, 'analysis_upload_mismatch')
        command = ['timeout', '900', 'env', 'CUDA_VISIBLE_DEVICES=', 'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4',
                   'MKL_NUM_THREADS=4', self.remote_py, layout[ANALYSIS], '--inputs', layout[inputs],
                   '--spec', layout[SPEC], '--output', remote + '/out']
        self.must(SSH + [shlex.join(command)], 960, 'analysis_remote_failed', label='remote analysis')
        for name in ('summary.json', 'report.md'):
            self.must(SCP + ['aml-b:%s/out/%s' % (remote, name), str(local / name)], 120, 'analysis_download_failed',
                      label='download')
        self.remote_hashes({'%s/out/%s' % (remote, n): file_digest(local / n) for n in ('summary.json', 'report.md')},
                           'analysis_download_mismatch')
        value = read_json(local / 'summary.json')
        verify_seal(value)
        if value.get('spec_sha256') != self.spec_sha:
            raise ChainStop('analysis_spec_mismatch')
        write_json(local / 'remote_receipt.json', dict(remote_directory=remote, uploaded=uploaded,
                                                       summary_sha256=file_digest(local / 'summary.json'),
                                                       execution='aml-b CPU, CUDA_VISIBLE_DEVICES empty, 4 threads',
                                                       actor_requests=0, analyst_requests=0))
        self.event('analysis_done', tag=tag, decision=value.get('decision'))
        return value

    # ------------------------------------------------------------ sequence

    def preflight(self):
        auth = read_json(self.auth)
        if (auth.get('experiment_id') != self.spec['experiment_id'] or auth.get('spec_sha256') != self.spec_sha
                or not auth.get('exact_text') or not set(self.args.blocks) <= set(auth.get('authorized_blocks') or [])):
            raise ChainStop('authorization_mismatch')
        if list(self.args.blocks) != sorted(set(self.args.blocks)) or not all(0 <= b < 5 for b in self.args.blocks):
            raise ChainStop('invalid_block_list')

    def check_previous(self, block):
        for earlier in range(block):
            verdict = evaluate_block(self.phase(earlier))
            if not verdict['ok']:
                raise ChainStop('previous_block_not_ok', verdict)

    def interim(self):
        summary = self.analysis(INTERIM_TAG, [0, 1, 2])
        decision = summary.get('decision')
        self.status(interim_decision=decision)
        if decision == 'stop_futility':
            raise ChainStop('futility_stop', summary.get('decision_reasons'))
        if decision not in ('interim_continue', 'insufficient_data'):
            raise ChainStop('unexpected_interim_decision', decision)
        return decision

    def execute(self):
        self.preflight()
        done, final = [], None
        for block in self.args.blocks:
            self.check_previous(block)
            if block > INTERIM_AFTER:
                self.interim()
            self.check_stop()
            done.append(dict(block=block, verdict=self.run_block(block)))
            if block == INTERIM_AFTER:
                self.interim()
        if self.args.blocks and self.args.blocks[-1] == 4:
            self.status(stage='final_analysis')
            final = self.analysis(FINAL_TAG, [0, 1, 2, 3, 4])
        return dict(blocks=done, final_decision=(final or {}).get('decision'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--authorization', type=Path, required=True)
    parser.add_argument('--blocks', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--poll-seconds', type=float, default=120)
    parser.add_argument('--idle-polls', type=int, default=3)
    parser.add_argument('--max-gpu-wait-hours', type=float, default=24)
    parser.add_argument('--max-probe-failures', type=int, default=30)
    parser.add_argument('--max-start-attempts', type=int, default=3)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status='not_executed', blocks=args.blocks, authorization=str(args.authorization),
                              stop_file=str(STOP_FILE), requires='--execute'), indent=2))
        return 0
    CHAIN_ROOT.mkdir(parents=True, exist_ok=True)
    lock = (CHAIN_ROOT / 'lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print('another chain run holds the lock', file=sys.stderr)
        return 2
    try:
        os.setsid()
    except OSError:
        pass
    run_dir = CHAIN_ROOT / 'runs' / time.strftime('%Y%m%dT%H%M%S')
    run_dir.mkdir(parents=True)
    chain = Chain(args, run_dir)
    write_json(run_dir / 'started.json', dict(pid=os.getpid(), args=vars(args) | dict(authorization=str(args.authorization)),
                                              chain_sha256=file_digest(SCRIPT), started_at=time.time()))
    chain.event('chain_started', pid=os.getpid())
    try:
        outcome = chain.execute()
        result = dict(status='completed', **outcome)
    except ChainStop as stop:
        result = dict(status='stopped', reason=stop.reason, detail=stop.detail)
    except Exception as exc:
        result = dict(status='stopped', reason='unexpected_error', detail='%s: %s' % (type(exc).__name__, exc),
                      traceback=traceback.format_exc())
    result.update(finished_at=time.time(), state=chain.state)
    write_json(run_dir / 'result.json', seal(json.loads(json.dumps(result, default=str))))
    chain.event('chain_finished', status=result['status'], reason=result.get('reason'))
    chain.status(stage='finished', result_status=result['status'], result_reason=result.get('reason'))
    return 0 if result['status'] == 'completed' else 3


if __name__ == '__main__':
    raise SystemExit(main())
