"""Exp1 chain runner contracts with fixture ports; never contacts aml-b, Docker or a model."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/chain_acceptance_gap_exp1.py'
spec = importlib.util.spec_from_file_location('exp1_chain_test', SCRIPT)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)

UUID = 'GPU-6d3f754b-b139-9354-0cbc-42bf22208180'


def probe(memory=3, util=0, apps=(), port=False, uuid=UUID, user='jiasheren2'):
    return dict(user=user, gpus=[dict(index='3', uuid='GPU-other', memory_mib=2000, util=30),
                                 dict(index='4', uuid=uuid, memory_mib=memory, util=util)],
                apps=list(apps), port8001_listening=port)


class Clock:
    def __init__(self):
        self.now = 1_000_000.0
        self.sleeps = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = tmp_path / 'phase_root'
    root.mkdir()
    monkeypatch.setattr(c, 'PHASE_ROOT', root)
    monkeypatch.setattr(c, 'CHAIN_ROOT', root / 'chain')
    monkeypatch.setattr(c, 'STOP_FILE', root / 'STOP')
    real = json.loads(c.SPEC.read_text())
    auth = tmp_path / 'authorization.json'
    auth.write_text(json.dumps(dict(exact_text='推送开跑吧', source='fixture', experiment_id=real['experiment_id'],
                                    spec_sha256=c.file_digest(c.SPEC), authorized_blocks=[0, 1, 2, 3, 4])))
    run_dir = root / 'chain/runs/test'
    run_dir.mkdir(parents=True)
    clock = Clock()
    args = SimpleNamespace(authorization=auth, blocks=[0, 1, 2, 3, 4], poll_seconds=120, idle_polls=3,
                           max_gpu_wait_hours=1, max_probe_failures=3, max_start_attempts=3)
    chain = c.Chain(args, run_dir, clock=clock.time, sleep=clock.sleep, runner=lambda *a, **k: None)
    return SimpleNamespace(chain=chain, root=root, clock=clock, args=args, auth=auth, run_dir=run_dir)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def finished_phase(root, block, valid=33, service='confirmed_absent', containers='confirmed_empty',
                   integrity=None, exit_code=0):
    phase = root / ('block_%02d' % block)
    write(phase / 'supervisor_result.json', dict(status='terminal_or_partial', exit_code=exit_code, wall_timeout=False,
                                                 service_cleanup=dict(status=service),
                                                 container_cleanup=dict(status=containers)))
    result = c.seal(dict(slots_terminal=33, study_valid_count=valid, integrity_error=integrity,
                         admissions=dict(requests=700), by_condition={}))
    write(phase / 'result.json', result)
    return phase


# ---------------------------------------------------------------- pure decisions

def test_gpu_ready_requires_idle_frozen_gpu_free_port_and_the_cap():
    assert c.gpu_ready(probe(), '4', UUID) == (True, [])
    assert not c.gpu_ready(probe(memory=68000, util=95), '4', UUID)[0]
    assert not c.gpu_ready(probe(util=3), '4', UUID)[0]
    assert not c.gpu_ready(probe(uuid='GPU-somewhere-else'), '4', UUID)[0]
    assert not c.gpu_ready(probe(apps=[dict(pid=1, uuid=UUID, user='someone')]), '4', UUID)[0]
    assert not c.gpu_ready(probe(port=True), '4', UUID)[0]
    one = [dict(pid=1, uuid='GPU-other', user='jiasheren2')]
    two = one + [dict(pid=2, uuid='GPU-third', user='jiasheren2')]
    assert c.gpu_ready(probe(apps=one), '4', UUID)[0]
    ready, reasons = c.gpu_ready(probe(apps=two), '4', UUID)
    assert not ready and 'other GPUs' in reasons[0]
    assert c.gpu_ready(probe(apps=two, user='other'), '4', UUID)[0]


def test_evaluate_block_accepts_clean_blocks_and_flags_problems(env):
    assert c.evaluate_block(finished_phase(env.root, 0))['ok']
    assert c.evaluate_block(finished_phase(env.root, 1, valid=27))['ok']
    verdict = c.evaluate_block(finished_phase(env.root, 2, valid=26))
    assert not verdict['ok'] and verdict['invalid'] == 7
    assert not c.evaluate_block(finished_phase(env.root, 3, service='cleanup_unconfirmed'))['ok']
    assert not c.evaluate_block(finished_phase(env.root, 4, containers='unconfirmed'))['ok']
    assert not c.evaluate_block(finished_phase(env.root, 0, integrity={'type': 'ValueError'}))['ok']
    assert not c.evaluate_block(finished_phase(env.root, 1, exit_code=1))['ok']
    assert not c.evaluate_block(env.root / 'block_09')['ok']


# ---------------------------------------------------------------- waiting for GPU4

def test_wait_needs_consecutive_idle_polls(env):
    probes = iter([probe(memory=60000, util=90), probe(), probe(), probe(util=5), probe(), probe(), probe()])
    env.chain.probe = lambda: next(probes)
    env.chain.wait_for_gpu(0)
    assert len(env.clock.sleeps) == 6
    status = json.loads((env.run_dir / 'status.json').read_text())
    assert status['stage'] == 'waiting_for_gpu' and status['consecutive_idle_polls'] == 3


def test_wait_times_out_and_honours_stop_and_probe_failures(env):
    env.chain.probe = lambda: probe(memory=60000, util=90)
    with pytest.raises(c.ChainStop) as stop:
        env.chain.wait_for_gpu(0)
    assert stop.value.reason == 'gpu_wait_timeout'

    def broken():
        raise RuntimeError('ssh: connect timed out')

    env.chain.probe = broken
    with pytest.raises(c.ChainStop) as stop:
        env.chain.wait_for_gpu(0)
    assert stop.value.reason == 'aml_unreachable'
    c.STOP_FILE.write_text('stop')
    with pytest.raises(c.ChainStop) as stop:
        env.chain.wait_for_gpu(0)
    assert stop.value.reason == 'operator_stop'


# ---------------------------------------------------------------- starting a block

def fake_cli(env, start_outcomes):
    calls = []
    outcomes = iter(start_outcomes)

    def cli(*args, timeout):
        calls.append(args)
        phase = env.root / ('block_%02d' % int(args[2]))
        if args[0] == 'prepare':
            write(phase / 'plan.json', dict(plan_sha256='a' * 64, service_stop_epoch=env.clock.now + 20000))
            return subprocess.CompletedProcess(args, 0, '{}', '')
        if args[0] == 'start-service':
            kind = next(outcomes)
            if kind == 'ok':
                write(phase / 'service.json', dict(leader_pid=7))
                return subprocess.CompletedProcess(args, 0, '{}', '')
            if kind == 'remote_refused':
                write(phase / 'service_start_attempt.json', {})
                write(phase / 'service_start_transport.json', dict(returncode=1))
            if kind == 'lost_transport':
                write(phase / 'service_start_attempt.json', {})
            if kind == 'receipt_rejected':
                write(phase / 'service.json', dict(leader_pid=8))
            return subprocess.CompletedProcess(args, 1, '', 'failed')
        raise AssertionError(args)

    env.chain.cli = cli
    env.chain.wait_for_gpu = lambda block: probe()
    env.chain.local_ready = lambda: None
    env.chain.service_root = lambda: '/srv/root'
    return calls


def test_failed_start_that_launched_nothing_is_archived_and_retried(env):
    calls = fake_cli(env, ['remote_refused', 'ok'])
    asked = []
    env.chain.remote_exists = lambda path: asked.append(path) or False
    phase = env.chain.start_block(0)
    assert (phase / 'service.json').exists() and asked == ['/srv/root/' + 'a' * 64]
    archived = list((env.root / 'failed_starts').iterdir())
    assert len(archived) == 1 and (archived[0] / 'service_start_transport.json').exists()
    assert [a[0] for a in calls] == ['prepare', 'start-service', 'prepare', 'start-service']


@pytest.mark.parametrize('outcome,remote', [('lost_transport', False), ('remote_refused', True), ('remote_refused', None)])
def test_start_failure_without_proof_of_nothing_launched_stops(env, outcome, remote):
    fake_cli(env, [outcome])
    env.chain.remote_exists = lambda path: remote
    with pytest.raises(c.ChainStop) as stop:
        env.chain.start_block(0)
    assert stop.value.reason == 'service_start_failed'
    assert not (env.root / 'failed_starts').exists() and (env.root / 'block_00').exists()


def test_rejected_receipt_releases_the_service(env):
    fake_cli(env, ['receipt_rejected'])
    released = []
    env.chain.emergency_release = lambda phase: released.append(phase) or {'service': 'released'}
    with pytest.raises(c.ChainStop) as stop:
        env.chain.start_block(0)
    assert stop.value.reason == 'service_start_failed' and released == [env.root / 'block_00']


def test_start_attempts_are_bounded(env):
    fake_cli(env, ['remote_refused'] * 3)
    env.chain.remote_exists = lambda path: False
    with pytest.raises(c.ChainStop) as stop:
        env.chain.start_block(0)
    assert stop.value.reason == 'service_start_failed' and len(list((env.root / 'failed_starts').iterdir())) == 3


# ---------------------------------------------------------------- running and sequencing

def test_supervisor_loss_triggers_emergency_release(env, monkeypatch):
    phase = env.root / 'block_00'
    write(phase / 'plan.json', dict(plan_sha256='a' * 64, service_stop_epoch=env.clock.now + 20000))
    monkeypatch.setattr(c, 'pid_alive', lambda pid: False)
    released = []
    env.chain.emergency_release = lambda p: released.append(p) or {}
    with pytest.raises(c.ChainStop) as stop:
        env.chain.wait_supervisor(0, phase, dict(pid=123))
    assert stop.value.reason == 'supervisor_ended_without_result' and released == [phase]


def test_finished_supervisor_returns(env, monkeypatch):
    phase = finished_phase(env.root, 0)
    write(phase / 'plan.json', dict(service_stop_epoch=env.clock.now + 20000))
    monkeypatch.setattr(c, 'pid_alive', lambda pid: False)
    env.chain.wait_supervisor(0, phase, dict(pid=123))
    assert env.clock.sleeps == []


def sequence(env, decisions):
    ran, tags = [], []

    def run_block(block):
        ran.append(block)
        return c.evaluate_block(finished_phase(env.root, block))

    def analysis(tag, blocks):
        tags.append((tag, blocks))
        return dict(decision=decisions.get(tag), decision_reasons={})

    env.chain.run_block = run_block
    env.chain.analysis = analysis
    return ran, tags


def test_full_sequence_runs_interim_after_block_2_and_final_after_block_4(env):
    ran, tags = sequence(env, {c.INTERIM_TAG: 'interim_continue', c.FINAL_TAG: 'go_method_development'})
    outcome = env.chain.execute()
    assert ran == [0, 1, 2, 3, 4] and outcome['final_decision'] == 'go_method_development'
    assert tags[0] == (c.INTERIM_TAG, [0, 1, 2]) and tags[-1] == (c.FINAL_TAG, [0, 1, 2, 3, 4])


def test_futility_after_block_2_stops_before_block_3(env):
    ran, _ = sequence(env, {c.INTERIM_TAG: 'stop_futility'})
    with pytest.raises(c.ChainStop) as stop:
        env.chain.execute()
    assert stop.value.reason == 'futility_stop' and ran == [0, 1, 2]


def test_stop_file_halts_between_blocks(env):
    ran, _ = sequence(env, {c.INTERIM_TAG: 'interim_continue'})
    original = env.chain.run_block

    def run_then_stop(block):
        verdict = original(block)
        if block == 1:
            c.STOP_FILE.write_text('stop')
        return verdict

    env.chain.run_block = run_then_stop
    with pytest.raises(c.ChainStop) as stop:
        env.chain.execute()
    assert stop.value.reason == 'operator_stop' and ran == [0, 1]


def test_later_blocks_require_clean_earlier_blocks(env):
    finished_phase(env.root, 0, valid=20)
    env.args.blocks = [1, 2, 3, 4]
    sequence(env, {})
    with pytest.raises(c.ChainStop) as stop:
        env.chain.execute()
    assert stop.value.reason == 'previous_block_not_ok'


def test_authorization_must_cover_spec_and_blocks(env):
    value = json.loads(env.auth.read_text())
    env.auth.write_text(json.dumps({**value, 'authorized_blocks': [0, 1]}))
    with pytest.raises(c.ChainStop) as stop:
        env.chain.preflight()
    assert stop.value.reason == 'authorization_mismatch'
    env.auth.write_text(json.dumps({**value, 'spec_sha256': '0' * 64}))
    with pytest.raises(c.ChainStop):
        env.chain.preflight()


def test_main_is_inert_without_execute(capsys, env):
    assert c.main(['--authorization', str(env.auth)]) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'not_executed'
    assert not (env.root / 'chain/lock').exists()
