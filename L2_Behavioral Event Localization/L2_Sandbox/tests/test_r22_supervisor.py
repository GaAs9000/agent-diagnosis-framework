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
