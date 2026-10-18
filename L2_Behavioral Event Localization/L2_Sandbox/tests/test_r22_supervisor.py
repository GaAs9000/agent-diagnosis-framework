"""R22 lifecycle checks use fake processes/Docker; no Actor or GPU execution."""
import importlib.util
import json
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor
import time

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/supervise_r22_native_alignment.py'
spec = importlib.util.spec_from_file_location('r22_supervisor_test', SCRIPT)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


def contract():
    now = time.time()
    stop = now + 20000
    plan = dict(service_stop_epoch=stop, plan_sha256='a'*64, source_fingerprints={},
        bound_files={str(p): s.sha(p) for p in (s.SCRIPT, s.DRIVER, s.HELPER)})
    freeze = dict(approved_plan_sha256='a'*64, phase_authorized=True, phase_frozen=True,
        internal_plan_reviewed=True, d0_frozen=False, user_authorization_record={'exact_text': 'approved', 'source': 'direct'})
    service = dict(remote_host='aml-b', device=4, owner_tag='a'*64, server_plan_hash='a'*64,
        service_stop_epoch=stop, leader_pid=1234, pgid=1234, started_epoch=now-10, hard_stop_epoch=stop+600)
    plan['service_lease_origin'] = {key: service[key] for key in
        ('started_epoch', 'hard_stop_epoch', 'service_stop_epoch')}
    service['clock_calibration'] = dict(local_sent_epoch=now-2, local_received_epoch=now-1, remote_epoch=now-101)
    service['remote_launch_epoch'] = now-100
    service['remote_hard_stop_epoch'] = service['hard_stop_epoch']-100-60
    service['timeout_seconds'] = int(service['remote_hard_stop_epoch']-service['remote_launch_epoch']-35)
    service['kill_grace_seconds'] = 30
    service['command'] = ['timeout', '--signal=TERM', '--kill-after=30s', str(service['timeout_seconds'])+'s', 'qwen-server']
    return now, plan, freeze, service


def test_default_never_executes(monkeypatch, capsys):
    monkeypatch.setattr(s, 'supervise', lambda *a: pytest.fail('execution not authorized'))
    monkeypatch.setattr(s, 'child', lambda *a: pytest.fail('child not authorized'))
    s.main([])
    assert json.loads(capsys.readouterr().out)['status'] == 'not_executed'
    with pytest.raises(SystemExit):
        s.main(['--child', 'anything'])


def test_bounded_receipt_and_source_binding():
    now, plan, freeze, service = contract()
    s.validate(plan, freeze, service, now)
    service['owner_tag'] = 'b'*64
    with pytest.raises(ValueError, match='receipt'):
        s.validate(plan, freeze, service, now)


@pytest.mark.parametrize('mutate', [
    lambda p, f, v, now: p.update(service_stop_epoch=now+86401),
    lambda p, f, v, now: v.update(hard_stop_epoch=v['started_epoch']+86401),
    lambda p, f, v, now: f.update(phase_authorized=False),
    lambda p, f, v, now: p['bound_files'].pop(str(s.HELPER)),
])
def test_rejects_unbounded_or_unbound(mutate):
    now, plan, freeze, service = contract()
    mutate(plan, freeze, service, now)
    with pytest.raises(ValueError):
        s.validate(plan, freeze, service, now)


def test_normal_child_does_not_signal(monkeypatch):
    class Process:
        pid = 123
        def wait(self, timeout): return 0
    monkeypatch.setattr(s.os, 'killpg', lambda *a: pytest.fail('normal completion signalled'))
    assert s.wait_child(Process(), time.time()+10) == (0, False)


def test_timeout_terminates_only_child_group(monkeypatch):
    class Process:
        pid = 321
        attempts = 0
        def wait(self, timeout):
            self.attempts += 1
            if self.attempts < 3: raise subprocess.TimeoutExpired('fake', timeout)
            return -9
    killed = []
    monkeypatch.setattr(s.os, 'killpg', lambda *a: killed.append(a))
    assert s.wait_child(Process(), time.time()-1) == (-9, True)
    assert killed == [(321, s.signal.SIGTERM), (321, s.signal.SIGKILL)]


def test_concurrent_actor_and_grader_containers_owned_unrelated_preserved(tmp_path, monkeypatch):
    helper = s.load(s.HELPER, 'cleanup_fake_test')
    owner = 'a'*64
    unrelated = 'b'*64
    containers = {unrelated: 'another-project'}
    cidfiles = []
    def docker(args, *a, **kw):
        if args[:2] == ['docker', 'create']:
            label = args[args.index('--label')+1].split('=', 1)[1]
            cid = format(len(cidfiles)+10, '064x')
            file = args[args.index('--cidfile')+1]
            cidfiles.append(file)
            Path(file).write_text(cid)
            containers[cid] = label
            return subprocess.CompletedProcess(args, 0, stdout=cid.encode())
        if args[:2] == ['docker', 'ps']:
            exact = args[-1].split('=', 2)[-1]
            return subprocess.CompletedProcess(args, 0, stdout='\n'.join(c for c, label in containers.items() if label==exact).encode())
        if args[:3] == ['docker', 'rm', '-f']:
            for cid in args[3:]: del containers[cid]
            return subprocess.CompletedProcess(args, 0, stdout=b'')
        pytest.fail('unexpected command')
    monkeypatch.setattr(s.subprocess, 'run', docker)
    with s.phase_containers(helper, tmp_path, owner):
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda n: s.subprocess.run(['docker','create', 'actor-or-grader']), range(30)))
    assert len(set(cidfiles)) == 30
    result = helper.cleanup_e1(tmp_path, owner, time.time()+30)
    assert result['status'] == 'confirmed_empty'
    assert containers == {unrelated: 'another-project'}


def test_absent_inputs_remain_pending(tmp_path):
    assert s.cpu_summary(tmp_path, {}, time.time()+30, None)['status'] == 'pending'


def test_failed_child_still_releases_and_preserves_partial_evidence(tmp_path, monkeypatch):
    now, plan, freeze, service = contract()
    # Startup hashes include actual plan and freeze bytes, independent of seals.
    for name, value in [('plan.json', plan), ('freeze.json', freeze), ('service.json', service)]:
        s.save(tmp_path/name, value)
    calls = []
    class Helper:
        def cleanup_e1(self, phase, owner, deadline):
            calls.append('containers')
            return {'status': 'confirmed_empty', 'actor_requests': 0}
    class Process:
        pid = 99
        def __init__(self, *a, **kw):
            assert kw['start_new_session'] is True
            assert '--execute' in a[0]
        def wait(self, timeout): return 7
    monkeypatch.setattr(s, 'load', lambda *a: Helper())
    monkeypatch.setattr(s.subprocess, 'Popen', Process)
    monkeypatch.setattr(s, 'stop_service', lambda *a: calls.append('service') or {'status': 'confirmed_absent'})
    state = s.supervise(tmp_path, tmp_path/'plan.json', tmp_path/'freeze.json', tmp_path/'service.json')
    assert state['exit_code'] == 7
    assert state['status'] == 'terminal_or_partial'
    assert state['cpu_summary']['status'] == 'pending'
    assert calls == ['containers', 'service']
    assert 'actor_requests' not in state['container_cleanup']
    assert json.loads((tmp_path/'supervisor_result.json').read_text())['exit_code'] == 7
    with pytest.raises(FileExistsError):
        s.supervise(tmp_path, tmp_path/'plan.json', tmp_path/'freeze.json', tmp_path/'service.json')


@pytest.mark.parametrize('drift', ['driver', 'plan', None])
@pytest.mark.parametrize('cleanup_error', [False, True])
def test_release_uses_validated_receipt_despite_drift_or_cleanup_failure(tmp_path, monkeypatch, drift, cleanup_error):
    # Substitute a private driver copy so this test never mutates frozen source.
    driver = tmp_path/'private_driver.py'
    driver.write_text('original driver bytes')
    monkeypatch.setattr(s, 'DRIVER', driver)
    now, plan, freeze, service = contract()
    for name, value in [('plan.json', plan), ('freeze.json', freeze), ('service.json', service)]:
        s.save(tmp_path/name, value)
    released = []
    class Helper:
        def cleanup_e1(self, phase, owner, deadline):
            if cleanup_error:
                raise OSError('cannot persist Docker cleanup receipt')
            return {'status': 'confirmed_empty'}
    class Process:
        pid = 99
        def __init__(self, *a, **kw): pass
        def wait(self, timeout):
            if drift == 'driver': driver.write_text('changed driver bytes')
            if drift == 'plan': s.save(tmp_path/'plan.json', {'tampered': True})
            # An on-disk replacement must never replace the validated identity.
            s.save(tmp_path/'service.json', {'owner_tag': 'other-project'})
            return 7
    monkeypatch.setattr(s, 'load', lambda *a: Helper())
    monkeypatch.setattr(s.subprocess, 'Popen', Process)
    monkeypatch.setattr(s, 'stop_service', lambda receipt, deadline: released.append(receipt.copy()) or {'status': 'confirmed_absent'})
    state = s.supervise(tmp_path, tmp_path/'plan.json', tmp_path/'freeze.json', tmp_path/'service.json')
    assert released == [service]
    assert state['service_cleanup']['status'] == 'confirmed_absent'
    assert state['startup_binding']['status'] == 'drift_detected'
    assert state['container_cleanup']['status'] == ('cleanup_unconfirmed' if cleanup_error else 'confirmed_empty')


@pytest.mark.parametrize('change', ['stale_duration', 'full_six_hours', 'kill_grace', 'original_lease', 'original_start', 'clock_margin', 'command', 'rtt', 'nan_clock'])
def test_actual_launcher_cannot_extend_original_frozen_lease(change):
    now, plan, freeze, service = contract()
    if change == 'stale_duration':
        service['remote_launch_epoch'] += 60
    elif change == 'full_six_hours':
        service['timeout_seconds'] = 21600
        service['command'][3] = '21600s'
    elif change == 'kill_grace':
        service['kill_grace_seconds'] = 0
    elif change == 'original_lease':
        service['hard_stop_epoch'] += 60
    elif change == 'original_start':
        service['started_epoch'] -= 60
    elif change == 'clock_margin':
        service['remote_hard_stop_epoch'] += 60
    elif change == 'command':
        service['command'][2] = '--kill-after=90s'
    elif change == 'rtt':
        service['clock_calibration']['local_sent_epoch'] -= 20
    elif change == 'nan_clock':
        service['clock_calibration']['remote_epoch'] = float('nan')
    with pytest.raises(ValueError):
        s.validate(plan, freeze, service, now)


@pytest.fixture
def stop_runtime(tmp_path, monkeypatch):
    """Execute the shipped remote code against proc files and fake pidfd/GPU APIs."""
    import types
    runtime = types.ModuleType('remote_stop_under_test')
    exec(s.STOP_CODE, runtime.__dict__)
    runtime.PROC_ROOT = str(tmp_path)
    (tmp_path/'stat').write_text('btime 1000\n')
    owner = 'a'*64
    receipt = dict(leader_pid=100, pgid=100, owner_tag=owner,
        remote_launch_epoch=1010, gpu_uuid='GPU-exact4',
        command=['timeout', '--signal=TERM', '--kill-after=30s', '900s', 'python'])
    rows = [(103, 'GPU-exact4')]
    sent = []
    closed = []
    uids = {}
    starts = {100: 1000, 101: 1001, 102: 1002, 103: 1003}
    parents = {100: 1, 101: 100, 102: 101, 103: 101}
    commands = {100: receipt['command'], 101: ['/home/jiasheren2/envs/idea-n1/bin/python3.11',
        '/home/jiasheren2/envs/idea-n1/bin/vllm', 'serve',
        '/home/jiasheren2/hf_cache/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd',
        '--served-model-name', 'Qwen/Qwen3.6-27B', 'Qwen3.6-27B',
        '--port', '8001', '--host', '127.0.0.1'],
        102: ['python', '-c', 'from multiprocessing.resource_tracker import main;main(8)'],
        103: ['VLLM::EngineCore_DP0']}
    receipt['command'][4:] = commands[101]
    commands[100] = receipt['command']
    def write(n, *, tag=True):
        base = tmp_path/str(n)
        base.mkdir(exist_ok=True)
        fields = ['S', str(parents[n]), '100'] + ['0']*16 + [str(starts[n])]
        (base/'stat').write_text(str(n)+' (process with spaces) '+' '.join(fields))
        env = ('C3_R22_SERVICE_OWNER='+owner).encode() if tag else b''
        (base/'environ').write_bytes(env+b'\0')
        (base/'cmdline').write_bytes(b'\0'.join(v.encode() for v in commands[n])+b'\0')
    for n in commands: write(n, tag=n!=103)
    fake_os = types.SimpleNamespace(scandir=runtime.os.scandir,
        getuid=lambda: 42, sysconf=lambda _: 100,
        stat=lambda p: types.SimpleNamespace(st_uid=uids.get(int(Path(p).name),42)),
        pidfd_open=lambda n: n+1000, close=closed.append)
    runtime.os = fake_os
    def query(args, **kwargs):
        if '--query-compute-apps=pid,gpu_uuid' in args:
            output='\n'.join(f'{n}, {u}' for n,u in rows)
        elif '--query-gpu=uuid' in args: output='GPU-exact4'
        elif '--query-gpu=memory.used' in args: output='3'
        else: pytest.fail('unexpected GPU command')
        return subprocess.CompletedProcess(args, 0, stdout=output)
    runtime.subprocess = types.SimpleNamespace(run=query)
    def term(fd, sig):
        import shutil
        n=fd-1000
        sent.append((n, sig))
        shutil.rmtree(tmp_path/str(n))
        rows[:] = [(p,u) for p,u in rows if p!=n]
    runtime.signal = types.SimpleNamespace(SIGTERM=s.signal.SIGTERM, pidfd_send_signal=term)
    return types.SimpleNamespace(code=runtime, receipt=receipt, rows=rows, sent=sent,
        closed=closed, uids=uids, starts=starts, parents=parents, commands=commands, write=write, root=tmp_path)


def test_remote_stop_accepts_untagged_enginecore_with_exact_descendant_proof(stop_runtime):
    f=stop_runtime
    result=f.code.cleanup(f.receipt)
    assert result['status']=='confirmed_absent'
    assert result['gpu4_processes']==[]
    assert {n for n,sig in f.sent}=={100,101,102,103}
    assert len(f.closed)==4


@pytest.mark.parametrize('case', ['different_owner', 'reused_pid', 'unknown_member',
    'foreign_gpu', 'foreign_gpu4_member', 'foreign_uid', 'reparented', 'foreign_group',
    'missing_api_tag', 'missing_leader_tag', 'wrong_gpu_uuid', 'missing_gpu_engine'])
def test_remote_stop_refuses_unproven_members_without_any_signal(stop_runtime, case):
    f=stop_runtime
    if case=='different_owner':
        (f.root/'103'/'environ').write_bytes(b'C3_R22_SERVICE_OWNER=other\0')
    elif case=='reused_pid':
        f.starts[100]=4000; f.write(100)
    elif case=='unknown_member':
        f.commands[103]=['unrelated-worker']; f.write(103)
    elif case=='foreign_gpu': f.rows[:]=[(103,'GPU-other')]
    elif case=='foreign_gpu4_member': f.rows.append((999,'GPU-exact4'))
    elif case=='foreign_uid': f.uids[103]=43
    elif case=='reparented': f.parents[103]=1; f.write(103,tag=False)
    elif case=='foreign_group':
        path=f.root/'103'/'stat'
        path.write_text(path.read_text().replace('S 101 100 ', 'S 101 999 '))
    elif case=='missing_api_tag': f.write(101,tag=False)
    elif case=='missing_leader_tag': f.write(100,tag=False)
    elif case=='wrong_gpu_uuid': f.receipt['gpu_uuid']='GPU-other'
    elif case=='missing_gpu_engine': f.rows.clear()
    result=f.code.cleanup(f.receipt)
    assert result['status']=='cleanup_unconfirmed'
    assert 'error' in result
    assert f.sent==[]


def test_remote_stop_refuses_pid_reuse_between_proof_and_pidfd(stop_runtime):
    f=stop_runtime
    def open_fd(n):
        if n==103:
            f.starts[103]+=1
            f.write(103,tag=False)
        return n+1000
    f.code.os.pidfd_open=open_fd
    result=f.code.cleanup(f.receipt)
    assert result['status']=='cleanup_unconfirmed'
    assert 'identity changed' in result['error']
    assert f.sent==[]
    assert len(f.closed)==4


def test_remote_stop_does_not_confirm_gpu_orphan_after_group_disappears(stop_runtime):
    f=stop_runtime
    original=f.code.signal.pidfd_send_signal
    def signal_without_gpu_release(fd,sig):
        original(fd,sig)
        if fd==1100: f.rows.append((103,'GPU-exact4'))
    f.code.signal.pidfd_send_signal=signal_without_gpu_release
    result=f.code.cleanup(f.receipt)
    assert result['remaining_pids']==[]
    assert result['gpu4_processes']==[103]
    assert result['status']=='cleanup_unconfirmed'


def test_remote_stop_checks_explicit_leader_birth_receipt(stop_runtime):
    f=stop_runtime
    f.receipt['leader_start_ticks']=999
    assert 'start identity changed' in f.code.cleanup(f.receipt)['error']
    assert f.sent==[]


def test_remote_stop_checks_exact_api_command_not_just_qwen_name(stop_runtime):
    f=stop_runtime
    f.commands[101]=f.commands[101]+['--unapproved-option']
    f.write(101)
    assert f.code.cleanup(f.receipt)['status']=='cleanup_unconfirmed'
    assert f.sent==[]


def test_remote_stop_pidfd_failure_never_falls_back_to_pid_or_group_signal(stop_runtime):
    f=stop_runtime
    def unavailable(n): raise OSError('pidfd unavailable')
    f.code.os.pidfd_open=unavailable
    assert f.code.cleanup(f.receipt)['status']=='cleanup_unconfirmed'
    assert f.sent==[]


def test_remote_stop_confirmed_empty_is_idempotent(stop_runtime):
    f=stop_runtime
    assert f.code.cleanup(f.receipt)['status']=='confirmed_absent'
    f.sent.clear()
    assert f.code.cleanup(f.receipt)['status']=='confirmed_absent'
    assert f.sent==[]


def test_remote_stop_term_is_bounded_and_never_claims_remaining_processes_absent(stop_runtime):
    f=stop_runtime
    f.code.signal.pidfd_send_signal=lambda fd,sig: f.sent.append((fd-1000,sig))
    clock=[0]
    def sleep(seconds): clock[0]+=seconds
    f.code.time=type('Clock',(),dict(monotonic=lambda:clock[0],sleep=sleep))
    result=f.code.cleanup(f.receipt)
    assert result['status']=='cleanup_unconfirmed'
    assert result['remaining_pids']==[100,101,102,103]
    assert clock[0]==20
    assert all(sig==s.signal.SIGTERM for _,sig in f.sent)


# Literal archived receipt and first inspection record from the failed cleanup.
# The flattened cmd strings preserve the observed shebang and title padding.
ARCHIVED_SERVICE = json.loads(r'''
{
  "remote_host": "aml-b",
  "device": 4,
  "gpu_uuid": "GPU-6d3f754b-b139-9354-0cbc-42bf22208180",
  "leader_pid": 3527475,
  "pgid": 3527475,
  "owner_tag": "2b6eeb1334028881d18c17c78bd4474849c47807f2da491ff921999e7a99b8d8",
  "server_plan_hash": "2b6eeb1334028881d18c17c78bd4474849c47807f2da491ff921999e7a99b8d8",
  "started_epoch": 1792311066.870997,
  "hard_stop_epoch": 1792332666.870997,
  "service_stop_epoch": 1792331961.820034,
  "remote_started_epoch": 1792311650.2563725,
  "command": [
    "timeout",
    "--signal=TERM",
    "--kill-after=30s",
    "20672s",
    "nice",
    "-n",
    "10",
    "/home/jiasheren2/envs/idea-n1/bin/vllm",
    "serve",
    "/home/jiasheren2/hf_cache/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
    "--download-dir",
    "/home/jiasheren2/hf_cache",
    "--served-model-name",
    "Qwen/Qwen3.6-27B",
    "Qwen3.6-27B",
    "--port",
    "8001",
    "--host",
    "127.0.0.1",
    "--enable-auto-tool-choice",
    "--tool-call-parser",
    "qwen3_xml",
    "--reasoning-parser",
    "qwen3",
    "--max-model-len",
    "131072",
    "--max-num-seqs",
    "64",
    "--gpu-memory-utilization",
    "0.85",
    "--dtype",
    "bfloat16"
  ],
  "remote_directory": "/home/jiasheren2/projects/agent_diagnosis_d0_20261018/r22_service/2b6eeb1334028881d18c17c78bd4474849c47807f2da491ff921999e7a99b8d8",
  "other_owned_gpu_uuids": [
    "GPU-3c272abb-5a57-ba28-c029-aa01b0eb7a2a"
  ],
  "compiler_tools": {
    "nvcc": "/usr/local/cuda-13.0/bin/nvcc",
    "ninja": "/home/jiasheren2/envs/idea-n1/bin/ninja",
    "c++": "/usr/bin/c++",
    "gcc": "/usr/bin/gcc",
    "g++": "/usr/bin/g++"
  },
  "cuda_home": "/usr/local/cuda-13.0",
  "compiler_path": "/usr/local/cuda-13.0/bin/nvcc",
  "clock_calibration": {
    "local_sent_epoch": 1792311898.617934,
    "local_received_epoch": 1792311898.821444,
    "remote_epoch": 1792311649.8732052
  },
  "remote_launch_epoch": 1792311650.2552824,
  "remote_hard_stop_epoch": 1792332357.922758,
  "timeout_seconds": 20672,
  "kill_grace_seconds": 30,
  "independent_hard_stop": "GNU timeout remaining original lease/KILL+30s"
}
''')
ARCHIVED_INSPECTION = json.loads(r'''
[{"pid": 3527475, "pgid": 3527475, "uid": 1029, "owner_fields": ["C3_R22_SERVICE_OWNER=2b6eeb1334028881d18c17c78bd4474849c47807f2da491ff921999e7a99b8d8"], "cmd": "timeout --signal=TERM --kill-after=30s 20672s nice -n 10 /home/jiasheren2/envs/idea-n1/bin/vllm serve /home/jiasheren2/hf_cache/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 --download-dir /home/jiasheren2/hf_cache --served-model-name Qwen/Qwen3.6-27B Qwen3.6-27B --port 8001 --host 127.0.0.1 --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 --max-model-len 131072 --max-num-seqs 64 --gpu-memory-utilization 0.85 --dtype bfloat16 "}, {"pid": 3527476, "pgid": 3527475, "uid": 1029, "owner_fields": ["C3_R22_SERVICE_OWNER=2b6eeb1334028881d18c17c78bd4474849c47807f2da491ff921999e7a99b8d8"], "cmd": "/home/jiasheren2/envs/idea-n1/bin/python3.11 /home/jiasheren2/envs/idea-n1/bin/vllm serve /home/jiasheren2/hf_cache/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 --download-dir /home/jiasheren2/hf_cache --served-model-name Qwen/Qwen3.6-27B Qwen3.6-27B --port 8001 --host 127.0.0.1 --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 --max-model-len 131072 --max-num-seqs 64 --gpu-memory-utilization 0.85 --dtype bfloat16 "}, {"pid": 3527736, "pgid": 3527475, "uid": 1029, "owner_fields": ["C3_R22_SERVICE_OWNER=2b6eeb1334028881d18c17c78bd4474849c47807f2da491ff921999e7a99b8d8"], "cmd": "/home/jiasheren2/envs/idea-n1/bin/python3.11 -c from multiprocessing.resource_tracker import main;main(72) "}, {"pid": 3527737, "pgid": 3527475, "uid": 1029, "owner_fields": [], "cmd": "VLLM::EngineCore                                                                                                                                               "}]
''')


@pytest.fixture
def archived_stop_runtime(stop_runtime):
    import shutil
    import shlex
    f=stop_runtime
    for n in (100,101,102,103): shutil.rmtree(f.root/str(n))
    f.receipt=json.loads(json.dumps(ARCHIVED_SERVICE))
    leader=3527475
    api=3527476
    f.code.os.getuid=lambda: 1029
    (f.root/'stat').write_text('btime '+str(int(f.receipt['remote_launch_epoch'])-10)+'\n')
    for i,p in enumerate(ARCHIVED_INSPECTION):
        n=p['pid']
        base=f.root/str(n)
        base.mkdir()
        # The inspection did not archive stat birth ticks; supply monotonic births
        # and the separately verified direct-child graph while replaying literal cmds/tags.
        ppid=1 if n==leader else leader if n==api else api
        fields=['S',str(ppid),str(p['pgid'])]+['0']*16+[str(1000+i)]
        (base/'stat').write_text(str(n)+' (archived) '+' '.join(fields))
        (base/'environ').write_bytes(b'\0'.join(v.encode() for v in p['owner_fields'])+b'\0')
        cmd=p['cmd']
        if n==3527737: args=[cmd]  # Keep process-title trailing spaces exactly.
        elif n==3527736:
            interpreter,script=cmd.split(' -c ',1)
            args=[interpreter,'-c',script.rstrip()]
        else: args=shlex.split(cmd)
        (base/'cmdline').write_bytes(b'\0'.join(v.encode() for v in args)+b'\0')
        f.uids[n]=p['uid']
    f.rows[:]=[(3527737,f.receipt['gpu_uuid'])]
    original=f.code.subprocess.run
    def query(args,**kwargs):
        if '--query-gpu=uuid' in args:
            return subprocess.CompletedProcess(args,0,stdout=f.receipt['gpu_uuid'])
        return original(args,**kwargs)
    f.code.subprocess.run=query
    return f


def test_remote_stop_replays_literal_archived_missing_tag_incident(archived_stop_runtime):
    f=archived_stop_runtime
    result=f.code.cleanup(f.receipt)
    assert result['status']=='confirmed_absent', result
    assert {n for n,sig in f.sent}=={p['pid'] for p in ARCHIVED_INSPECTION}
    assert result['gpu4_processes']==[]


@pytest.mark.parametrize('change', ['interpreter_env', 'script_path', 'nice_priority',
    'api_argv', 'api_pid', 'api_start_ticks', 'title_suffix'])
def test_archived_cleanup_rejects_unapproved_exec_transform(archived_stop_runtime,change):
    f=archived_stop_runtime
    path=f.root/'3527476'/'cmdline'
    args=path.read_bytes().rstrip(b'\0').split(b'\0')
    if change=='interpreter_env': args[0]=b'/other/env/bin/python3.11'
    elif change=='script_path': args[1]=b'/other/env/bin/vllm'
    elif change=='nice_priority': f.receipt['command'][6]='11'
    elif change=='api_argv': f.receipt['api_argv']=['unexpected']
    elif change=='api_pid': f.receipt['api_pid']=3527477
    elif change=='api_start_ticks': f.receipt['api_start_ticks']=1
    elif change=='title_suffix':
        (f.root/'3527737'/'cmdline').write_bytes(b'VLLM::EngineCore_unrelated   \0')
    path.write_bytes(b'\0'.join(args)+b'\0')
    result=f.code.cleanup(f.receipt)
    assert result['status']=='cleanup_unconfirmed'
    assert f.sent==[]


def test_archived_cleanup_accepts_explicit_api_birth_and_argv(archived_stop_runtime):
    f=archived_stop_runtime
    f.receipt.update(api_pid=3527476,api_start_ticks=1001,leader_start_ticks=1000,
        api_argv=[v.decode() for v in (f.root/'3527476'/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')])
    assert f.code.cleanup(f.receipt)['status']=='confirmed_absent'
