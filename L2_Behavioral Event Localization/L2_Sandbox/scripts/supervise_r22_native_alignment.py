#!/usr/bin/env python3
"""One-shot R22 supervision, exact-owner release, and bounded CPU-only summary."""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
import time

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[3]
DRIVER = SCRIPT.with_name('run_r22_native_alignment.py')
HELPER = SCRIPT.with_name('finalize_d0_timeout600.py')
CODE = Path('/home/jiasheren2/projects/agent_diagnosis_d0_20261018/Users/jiashen/Desktop/问题诊断')
PYTHON = '/home/jiasheren2/envs/agent-diagnosis-d0-py312/bin/python'
REMOTE = Path('/home/jiasheren2/projects/d0_r22_postprocess')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate(plan, freeze, service, now):
    stop = plan['service_stop_epoch']
    if not math.isfinite(stop) or not 645 <= stop - now <= 21600:
        raise ValueError('invalid bounded phase deadline')
    if (freeze.get('approved_plan_sha256') != plan['plan_sha256'] or
            any(freeze.get(key) is not True for key in ('phase_authorized', 'phase_frozen', 'internal_plan_reviewed')) or
            freeze.get('d0_frozen') is not False or not freeze.get('user_authorization_record', {}).get('exact_text') or
            not freeze.get('user_authorization_record', {}).get('source')):
        raise ValueError('explicit reviewed phase freeze required')
    for path in (SCRIPT, DRIVER, HELPER):
        if plan['bound_files'].get(str(path)) != sha(path) and plan['source_fingerprints'].get(str(path.relative_to(ROOT))) != sha(path):
            raise ValueError('supervisor/driver/helper not bound to plan')
    if any(service.get(key) != expected for key, expected in {
        'remote_host': 'aml-b', 'device': 4, 'owner_tag': plan['plan_sha256'],
        'server_plan_hash': plan['plan_sha256'], 'service_stop_epoch': stop}.items()):
        raise ValueError('service receipt does not match phase')
    pid, pgid = service['leader_pid'], service['pgid']
    if type(pid) is not int or pid <= 1 or pid != pgid:
        raise ValueError('service leader must own its process group')
    started, hard_stop = service['started_epoch'], service['hard_stop_epoch']
    if not all(math.isfinite(x) for x in (started, hard_stop)) or not 0 < hard_stop-started <= 21600 or stop > hard_stop-600 or started > now:
        raise ValueError('service lease must be independent and bounded to six hours')
    origin = plan.get('service_lease_origin')
    if not isinstance(origin, dict) or any(service.get(key) != origin.get(key) for key in
            ('started_epoch', 'hard_stop_epoch', 'service_stop_epoch')) or origin.get('service_stop_epoch') != stop:
        raise ValueError('service receipt changed frozen original lease')
    calibration = service.get('clock_calibration', {})
    sent, received, remote = [calibration.get(key) for key in
        ('local_sent_epoch', 'local_received_epoch', 'remote_epoch')]
    launch, remote_stop = service.get('remote_launch_epoch'), service.get('remote_hard_stop_epoch')
    values = (sent, received, remote, launch, remote_stop)
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError('finite clock calibration and remote launch required')
    if not 0 <= received-sent <= 10 or received > now or launch < remote:
        raise ValueError('invalid or stale launch clock calibration')
    offset_max = received-remote
    if remote_stop != hard_stop-offset_max-60:
        raise ValueError('remote deadline must derive from original lease and clock margin')
    duration, grace = service.get('timeout_seconds'), service.get('kill_grace_seconds')
    if (type(duration) is not int or not 645 < duration <= 21600 or grace != 30 or
            duration != int(remote_stop-launch-30-5) or launch+duration+30 > remote_stop):
        raise ValueError('actual timeout exceeds frozen lease or omits termination grace')
    command = service.get('command')
    if not isinstance(command, list) or len(command) < 5 or command[:4] != [
            'timeout', '--signal=TERM', '--kill-after=30s', str(duration)+'s']:
        raise ValueError('actual timeout command does not match bounded receipt')


@contextmanager
def phase_containers(helper, phase, owner):
    # The historical E1 label covers every Actor and grader container here.
    with helper.owned_e1_containers(phase, owner):
        tracked = subprocess.run
        lock = threading.Lock()
        def serialized_create(args, *positional, **kwargs):
            if isinstance(args, (list, tuple)) and list(args[:2]) == ['docker', 'create']:
                with lock:
                    return tracked(args, *positional, **kwargs)
            return tracked(args, *positional, **kwargs)
        subprocess.run = serialized_create
        try:
            yield
        finally:
            subprocess.run = tracked


def assert_startup(binding):
    for path, digest in binding['files'].items():
        if sha(path) != digest:
            raise ValueError('startup source or metadata changed: ' + path)


def child(binding_path):
    binding = read(binding_path)
    assert_startup(binding)
    helper = load(HELPER, 'r22_cleanup_child')
    driver = load(DRIVER, 'r22_execution_child')
    phase = Path(binding['phase'])
    with phase_containers(helper, phase, binding['resource_owner']):
        driver.execute(read(binding['plan']), read(binding['freeze']), phase)


def wait_child(process, deadline):
    try:
        return process.wait(timeout=max(0, deadline-time.time())), False
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            code = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait(timeout=10)
        return code, True


# Remote code receives only the bound receipt. It never searches/kills by name.
STOP_CODE = r'''
import json, os, re, signal, subprocess, sys, time
PROC_ROOT='/proc'

def process(n):
    base=PROC_ROOT+'/'+str(n)
    with open(base+'/stat') as f: raw=f.read()
    fields=raw[raw.rindex(')')+2:].split()
    try:
        with open(base+'/environ','rb') as f: env=f.read().split(b'\0')
    except PermissionError:
        # A nondumpable EngineCore can hide environ. Keep its tag unavailable;
        # prove() still requires readable owner tags for every other role.
        env=None
    with open(base+'/cmdline','rb') as f: args=f.read().rstrip(b'\0').split(b'\0')
    return dict(pid=n, uid=os.stat(base).st_uid, ppid=int(fields[1]),
        pgid=int(fields[2]), start=int(fields[19]), state=fields[0], env=env, args=args)

def members(pgid):
    found={}
    for entry in os.scandir(PROC_ROOT):
        if entry.name.isdigit():
            try:
                with open(PROC_ROOT+'/'+entry.name+'/stat') as f: raw=f.read()
                fields=raw[raw.rindex(')')+2:].split()
                if int(fields[2])==pgid and fields[0]!='Z':
                    p=process(int(entry.name))
                    if p['pgid']!=pgid: raise ValueError('process group changed during read')
                    found[p['pid']]=p
            except FileNotFoundError: pass
    return found

def gpu_query(args):
    r=subprocess.run(['nvidia-smi']+args+['--format=csv,noheader,nounits'],
        capture_output=True,text=True,timeout=5,check=True)
    return r.stdout.strip()

def gpu_processes():
    rows=gpu_query(['--query-compute-apps=pid,gpu_uuid'])
    return [(int(n.strip()), uuid.strip()) for row in rows.splitlines()
        for n,uuid in [row.split(',',1)]]

def identity(p):
    return tuple(p[k] for k in ('pid','uid','ppid','pgid','start'))

def api_command(s, args):
    expected=[v.encode() for v in s['command'][4:]]
    # nice execs its command; a console script's shebang adds the interpreter.
    # Admit only the recorded wrapper and a Python in that exact script's bin.
    if expected[:3]==[b'nice',b'-n',b'10']: expected=expected[3:]
    actual=args
    if expected and expected[0].endswith(b'/vllm') and actual!=expected and len(actual)==len(expected)+1:
        directory,_,name=actual[0].rpartition(b'/')
        if directory==expected[0].rsplit(b'/',1)[0] and re.fullmatch(br'python(?:3(?:\.[0-9]+)?)?',name):
            actual=actual[1:]
    if actual!=expected or (s.get('api_argv') is not None and args!=[v.encode() for v in s['api_argv']]):
        return False
    return (b'8001' in actual and b'qwen' in b' '.join(actual).lower() and
        (b'vllm.entrypoints.openai.api_server' in actual or any(
            v.rsplit(b'/',1)[-1]==b'vllm' and actual[i+1:i+2]==[b'serve']
            for i,v in enumerate(actual))))

def prove(s, group, gpu, uuid):
    pid=s['leader_pid']; owner=('C3_R22_SERVICE_OWNER='+s['owner_tag']).encode()
    if pid!=s['pgid'] or pid not in group: raise ValueError('leader missing or foreign group')
    api=[]
    for n,p in group.items():
        if p['uid']!=os.getuid(): raise ValueError('foreign UID')
        tags=[v for v in (p['env'] or []) if v.startswith(b'C3_R22_SERVICE_OWNER=')]
        if tags and tags!=[owner]: raise ValueError('owner mismatch')
        args=p['args']; cmd=b' '.join(args)
        if n==pid:
            if tags!=[owner] or [v.decode() for v in args[:4]]!=s['command'][:4]:
                raise ValueError('leader identity changed')
        elif api_command(s,args):
            if tags!=[owner] or p['ppid']!=pid: raise ValueError('API anchor mismatch')
            if s.get('api_pid',n)!=n or s.get('api_start_ticks',p['start'])!=p['start']:
                raise ValueError('API birth identity changed')
            api.append(n)
        elif args and re.fullmatch(br'VLLM::EngineCore(?:_DP[0-9]+)?',args[0].rstrip(b' ')):
            if (n,uuid) not in gpu: raise ValueError('EngineCore exact GPU identity missing')
        elif b'from multiprocessing.resource_tracker import main' in cmd:
            if tags!=[owner]: raise ValueError('tracker owner missing')
        else: raise ValueError('unknown group member')
    if len(api)!=1: raise ValueError('unique Qwen 8001 API anchor missing')
    # Only the observed API's direct EngineCore/tracker children are admitted.
    for n,p in group.items():
        if n not in (pid,api[0]) and p['ppid']!=api[0]:
            raise ValueError('descendant chain changed')
        if n!=pid and p['start']<group[p['ppid']]['start']:
            raise ValueError('parent process identity changed')
    with open(PROC_ROOT+'/stat') as f:
        boot=next(int(line.split()[1]) for line in f if line.startswith('btime '))
    started=boot+group[pid]['start']/os.sysconf('SC_CLK_TCK')
    if not s['remote_launch_epoch']-2<=started<=s['remote_launch_epoch']+10:
        raise ValueError('leader launch identity changed')
    if s.get('leader_start_ticks',group[pid]['start'])!=group[pid]['start']:
        raise ValueError('leader start identity changed')
    for n,u in gpu:
        if (n in group and u!=uuid) or (u==uuid and n not in group):
            raise ValueError('foreign GPU process')

def cleanup(s):
    pid=s['leader_pid']; pgid=s['pgid']; handles={}
    result=dict(status='cleanup_unconfirmed',leader_pid=pid,pgid=pgid)
    try:
        uuid=gpu_query(['-i','4','--query-gpu=uuid'])
        if not uuid or '\n' in uuid or s.get('gpu_uuid')!=uuid:
            raise ValueError('GPU UUID mismatch')
        group=members(pgid); gpu=gpu_processes()
        if group:
            prove(s,group,gpu,uuid)
            # Open stable process handles, then prove the entire unchanged graph again.
            # No signal can hit a reused PID, even during the final syscall race.
            for n in group: handles[n]=os.pidfd_open(n)
            current=members(pgid)
            if {n:identity(p) for n,p in current.items()}!={n:identity(p) for n,p in group.items()}:
                raise ValueError('process identity changed before signal')
            prove(s,current,gpu_processes(),uuid)
            for n in sorted(group,key=lambda n: (group[n]['start'],n),reverse=True):
                try: signal.pidfd_send_signal(handles[n],signal.SIGTERM)
                except ProcessLookupError: pass
            deadline=time.monotonic()+20
            while members(pgid) and time.monotonic()<deadline: time.sleep(.25)
        remaining=members(pgid)
        gpu=gpu_processes()
        result.update(remaining_pids=sorted(remaining),gpu4_processes=[n for n,u in gpu if u==uuid])
        if not remaining and not result['gpu4_processes'] and not any(n in group for n,u in gpu):
            result['status']='confirmed_absent'
    except Exception as exc: result['error']=type(exc).__name__+': '+str(exc)
    finally:
        for fd in handles.values(): os.close(fd)
    try: result['gpu4_memory_used_mib_readback']=gpu_query(['-i','4','--query-gpu=memory.used'])
    except Exception: result['gpu4_memory_used_mib_readback']=None
    return result

if __name__=='__main__': print(json.dumps(cleanup(json.load(sys.stdin))))
'''


def stop_service(service, deadline):
    remaining = min(35, deadline-time.time())
    if remaining <= 0:
        return {'status': 'cleanup_unconfirmed', 'error': 'release deadline expired'}
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'aml-b',
        shlex.join([PYTHON, '-c', STOP_CODE])], input=json.dumps(service), text=True,
        capture_output=True, timeout=remaining, check=True)
    return json.loads(result.stdout)


def cpu_summary(phase, binding, deadline, helper):
    source = phase / 'score_inputs.json'
    if not source.is_file():
        return {'status': 'pending', 'reason': 'no sealed score inputs available'}
    remote = REMOTE / binding['resource_owner']
    digest = sha(source)
    def command(args, seconds=60):
        helper.bounded(args, deadline, seconds, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    command(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'aml-b', shlex.join(['mkdir', '-p', str(remote)])])
    command(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', str(source), 'aml-b:'+str(remote/'score_inputs.json')], 180)
    # Check server driver and input bytes before running exactly the frozen CLI.
    server_driver = CODE / DRIVER.relative_to(ROOT)
    server_sources = [(CODE / relative, expected) for relative, expected in read(binding['plan'])['source_fingerprints'].items()]
    verify = "import hashlib,pathlib;" + ';'.join(
        "assert hashlib.sha256(pathlib.Path(%r).read_bytes()).hexdigest()==%r" % (str(path), expected)
        for path, expected in [*server_sources, (server_driver, binding['files'][str(DRIVER)]), (remote/'score_inputs.json', digest)])
    command(['ssh', 'aml-b', shlex.join([PYTHON, '-c', verify])])
    seconds = int(min(900, deadline-time.time()-90))
    if seconds <= 0:
        raise TimeoutError('CPU summary deadline expired')
    command(['ssh', 'aml-b', shlex.join(['timeout', str(seconds), 'env', 'CUDA_VISIBLE_DEVICES=',
        'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4', 'MKL_NUM_THREADS=4', 'NUMEXPR_NUM_THREADS=4',
        PYTHON, str(server_driver), '--summarize-cpu', str(remote/'score_inputs.json'), '--output', str(remote/'reports')])], seconds+15)
    receipt_code = ("import hashlib,json,pathlib; p=pathlib.Path(%r);" % str(remote) +
        "h=lambda q:hashlib.sha256(q.read_bytes()).hexdigest();" +
        "(p/'reports/receipt.json').write_text(json.dumps(dict(score_inputs_sha256=h(p/'score_inputs.json'),summary_sha256=h(p/'reports/sensitivity_summary.json'),driver_sha256=h(pathlib.Path(%r)))))" % str(server_driver))
    command(['ssh', 'aml-b', shlex.join([PYTHON, '-c', receipt_code])])
    reports = phase/'reports'
    reports.mkdir(exist_ok=True)
    command(['scp', '-q', 'aml-b:'+str(remote/'reports/sensitivity_summary.json'), str(reports/'summary.json')])
    command(['scp', '-q', 'aml-b:'+str(remote/'reports/receipt.json'), str(reports/'receipt.json')])
    receipt = read(reports/'receipt.json')
    if receipt != dict(score_inputs_sha256=digest, summary_sha256=sha(reports/'summary.json'), driver_sha256=binding['files'][str(DRIVER)]):
        raise ValueError('CPU summary receipt binding mismatch')
    return {'status': 'completed', 'score_inputs_sha256': digest, 'summary_sha256': sha(reports/'summary.json'), 'remote_directory': str(remote), 'execution': 'aml-b CPU4', 'actor_requests': 0, 'analyst_requests': 0}


class SupervisorTerminated(BaseException):
    pass


def supervise(phase, plan_path, freeze_path, service_path):
    phase = phase.resolve()
    plan_path, freeze_path, service_path = [x.resolve() for x in (plan_path, freeze_path, service_path)]
    plan, freeze, service = map(read, (plan_path, freeze_path, service_path))
    started = time.time()
    validate(plan, freeze, service, started)
    helper = load(HELPER, 'r22_cleanup_parent')
    files = {str(path): sha(path) for path in (SCRIPT, DRIVER, HELPER, plan_path, freeze_path, service_path)}
    owner = hashlib.sha256(json.dumps({'files': files, 'started_ns': time.time_ns()}, sort_keys=True).encode()).hexdigest()
    binding = dict(phase=str(phase), plan=str(plan_path), freeze=str(freeze_path), files=files,
        plan_sha256=plan['plan_sha256'], started_epoch=started, resource_owner=owner,
        resource_owner_label='d0.e1.owner', resource_scope='all phase Actor and grader containers')
    phase.mkdir(parents=True, exist_ok=True)
    marker = phase/'supervisor_started.json'
    with marker.open('x') as handle:
        json.dump(binding, handle, indent=2)
    state = dict(status='running', plan_sha256=plan['plan_sha256'], resource_owner=owner, resource_owner_label='d0.e1.owner')
    process = None
    previous_term = signal.getsignal(signal.SIGTERM)
    def terminate(signum, frame):
        raise SupervisorTerminated()
    signal.signal(signal.SIGTERM, terminate)
    try:
        assert_startup(binding)
        with (phase/'supervisor_child.log').open('xb') as log:
            process = subprocess.Popen([sys.executable, str(SCRIPT), '--execute', '--child', str(marker)], start_new_session=True,
                stdin=subprocess.DEVNULL, stdout=log, stderr=log)
            save(phase/'supervisor_child.json', dict(pid=process.pid, pgid=process.pid, binding_sha256=sha(marker)))
            state['exit_code'], state['wall_timeout'] = wait_child(process, plan['service_stop_epoch']+180)
        state['status'] = 'terminal_or_partial'
    except BaseException as exc:
        state.update(status='terminal_or_partial', error=type(exc).__name__)
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            wait_child(process, time.time()+30)
    finally:
        # Historical helper's actor_requests=0 describes only the cleanup call.
        try:
            cleanup = helper.cleanup_e1(phase, owner, min(time.time()+60, plan['service_stop_epoch']+300))
            cleanup.pop('actor_requests', None)
            state['container_cleanup'] = cleanup
        except Exception as exc:
            state['container_cleanup'] = {'status': 'cleanup_unconfirmed', 'error': type(exc).__name__}
        try:
            assert_startup(binding)
            state['startup_binding'] = {'status': 'unchanged'}
        except Exception as exc:
            state['startup_binding'] = {'status': 'drift_detected', 'error': type(exc).__name__}
        try:
            # The captured, validated receipt remains authoritative for release
            # even if execution sources or phase metadata have since drifted.
            state['service_cleanup'] = stop_service(service, min(time.time()+45, plan['service_stop_epoch']+360))
        except Exception as exc:
            state['service_cleanup'] = {'status': 'cleanup_unconfirmed', 'error': type(exc).__name__}
        state['artifacts'] = {str(path.relative_to(phase)): sha(path) for path in
            [phase/'result.json', phase/'score_inputs.json', phase/'executed_plan.json', phase/'executed_freeze.json', *phase.glob('receipt*.json')] if path.is_file()}
        save(phase/'supervisor_result.json', state)
        signal.signal(signal.SIGTERM, previous_term)
    try:
        state['cpu_summary'] = cpu_summary(phase, binding, plan['service_stop_epoch']+3600, helper)
    except Exception as exc:
        state['cpu_summary'] = {'status': 'pending_failed_or_deadline', 'error': type(exc).__name__}
    save(phase/'supervisor_result.json', state)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--phase', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--freeze', type=Path)
    parser.add_argument('--service', type=Path)
    parser.add_argument('--child', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        if not args.execute:
            parser.error('internal child also requires --execute')
        child(args.child)
    elif args.execute:
        if not all((args.phase, args.freeze, args.service)):
            parser.error('--execute requires --phase --freeze --service')
        supervise(args.phase, args.plan or args.phase/'plan.json', args.freeze, args.service)
    else:
        print(json.dumps({'status': 'not_executed', 'requires': '--execute --phase --freeze --service'}))


if __name__ == '__main__':
    main()
