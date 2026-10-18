"""Offline fixtures: execute this suite on the lab CPU host, never the Mac."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/finalize_d0_serial.py'
SPEC = importlib.util.spec_from_file_location('serial_finalization_test', SCRIPT)
d = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def phase(tmp_path, monkeypatch):
    monkeypatch.setattr(d, 'ROOT', tmp_path)
    monkeypatch.setattr(d, 'source_inventory', lambda root: {'fixture.py': 'pinned'})
    phase = tmp_path / 'phase'
    write(phase / 'executed_plan.json', {'service_stop_epoch': 1000})
    write(phase / 'supervisor_result.json', {'exit_code': 1, 'wall_timeout': False})
    monkeypatch.setattr(d.time, 'time', lambda: 100)
    return phase


def test_wait_timeout_makes_no_network_calls(phase, tmp_path, monkeypatch):
    (phase / 'supervisor_result.json').unlink()
    monkeypatch.setattr(d.time, 'time', lambda: 2000)
    monkeypatch.setattr(d, 'command', lambda *a, **k: pytest.fail('network called'))
    with pytest.raises(TimeoutError):
        d.finalize(phase, tmp_path / 'out')


def test_in_progress_rejected_without_network(phase, tmp_path, monkeypatch):
    out = tmp_path / 'out'
    (out / 'one_shot.lock').mkdir(parents=True)
    monkeypatch.setattr(d, 'command', lambda *a, **k: pytest.fail('network called'))
    with pytest.raises(FileExistsError):
        d.finalize(phase, out)


def test_completed_receipt_is_idempotent_and_checks_hashes(phase, tmp_path, monkeypatch):
    out = tmp_path / 'out'
    write(out / 'g2.json', {'status': 'undecided'})
    receipt = {'identity': {'plan_sha256': d.sha(phase / 'executed_plan.json')},
               'files_sha256': {'g2.json': d.sha(out / 'g2.json')}}
    write(out / 'manifest.json', receipt)
    monkeypatch.setattr(d, 'command', lambda *a, **k: pytest.fail('network called'))
    assert d.finalize(phase, out) == receipt
    write(out / 'g2.json', {'status': 'pass'})
    with pytest.raises(ValueError, match='bytes changed'):
        d.finalize(phase, out)


def test_finalization_keeps_failed_phase_and_false_gates(phase, tmp_path, monkeypatch):
    runs = tmp_path / 'L1_sandbox/C3_Experimental_Sandbox/runs' / d.COHORT
    for relative in ('episodes/officeval_003.r00', 'episodes/officeval_034.r00',
                     'v3_g2_c0_resumes/officeval_003.r00_c0'):
        write(runs / relative / 'evidence_index.json', {})
    # An unfinished DEV slot has no seal and must not be copied.
    write(runs / 'episodes/officeval_004.r00/manifest.json', {})
    commands = []

    def command(args, **kwargs):
        commands.append(args)
        if args[0] == 'rsync' and args[-2].startswith('aml-b:'):
            target = Path(args[-1])
            name = target.name
            if name == 'status.json':
                write(target, {'reports_sha256': {n: d.sha(target.parent / n) for n in d.AUDITS[:-1]},
                               'sources': d.read(target.parent / 'sources.json'),
                               'sources_manifest_sha256': d.sha(target.parent / 'sources.json')})
            elif name.endswith('.json'):
                write(target, {'gate_pass': False, 'status': 'unverified'})
            else:
                target.write_text('raw offline audit log\n')

    monkeypatch.setattr(d, 'command', command)
    out = tmp_path / 'out'
    result = d.finalize(phase, out)
    assert result['d0_frozen'] is False and result['gates_promoted'] is False
    assert result['supervisor']['exit_code'] == 1
    assert result['identity']['result_sha256'] is None
    assert set(result['identity']['packages']) == {
        'episodes/officeval_003.r00', 'v3_g2_c0_resumes/officeval_003.r00_c0'}
    assert not any('officeval_034' in ' '.join(c) for c in commands)
    remote_commands = [' '.join(c) for c in commands if c[0] == 'ssh']
    assert any('timeout 1800' in c and 'CUDA_VISIBLE_DEVICES=' in c for c in remote_commands)
    assert all('--delete' not in c for c in commands)
    assert all('--protect-args' not in c for c in commands)
    assert result['identity']['sources'] == {'fixture.py': 'pinned'}
    count = len(commands)
    assert d.finalize(phase, out) == result
    assert len(commands) == count


def test_ssh_quotes_each_argument(monkeypatch):
    seen = []
    monkeypatch.setattr(d, 'command', lambda args: seen.append(args))
    d.ssh(['python', '/path with spaces/审查.py', '--data', '/safe/a;echo unsafe'])
    import shlex
    assert shlex.split(seen[0][-1]) == ['python', '/path with spaces/审查.py', '--data', '/safe/a;echo unsafe']


def test_remote_mode_rejects_unapproved_path_before_commands(tmp_path, monkeypatch):
    monkeypatch.setattr(d, 'command', lambda *a, **k: pytest.fail('command called'))
    with pytest.raises(ValueError, match='unexpected isolated'):
        d.remote_audits(tmp_path)


def test_source_inventory_binds_all_project_sources_and_rejects_additions(tmp_path, monkeypatch):
    required = [tmp_path / d.L2 / 'scripts' / name for name in (
        'audit_g1_mechanical.py', 'audit_g4_provenance.py', 'finalize_d0_serial.py')]
    required += [tmp_path / 'scripts/v3_evaluate_gates_g2_g3_g4.py',
                 tmp_path / 'L1_sandbox/C3_Experimental_Sandbox/src/nested/module.py',
                 tmp_path / 'L1_sandbox/C3_Experimental_Sandbox/src/nested/schema.json',
                 tmp_path / d.L2 / 'src/package/module.py']
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
    monkeypatch.setattr(d, 'ROOT', tmp_path)
    pinned = d.source_inventory(tmp_path)
    assert len(pinned) == 7
    d.assert_sources(pinned)
    required[-1].write_text('changed')
    with pytest.raises(ValueError, match='source inventory differs'):
        d.assert_sources(pinned)


def test_sources_pinned_before_wait_and_fail_closed_on_change(phase, tmp_path, monkeypatch):
    seen = 0

    def inventory(root):
        nonlocal seen
        seen += 1
        return {'fixture.py': 'before' if seen == 1 else 'after'}

    monkeypatch.setattr(d, 'source_inventory', inventory)
    monkeypatch.setattr(d, 'command', lambda *a, **k: pytest.fail('network called'))
    with pytest.raises(ValueError, match='source inventory differs'):
        d.finalize(phase, tmp_path / 'out')
    assert d.read(tmp_path / 'out/sources.json') == {'fixture.py': 'before'}


def test_cached_result_rejects_another_requested_phase(phase, tmp_path, monkeypatch):
    out = tmp_path / 'out'
    write(out / 'manifest.json', {'identity': {'plan_sha256': 'another-phase'}, 'files_sha256': {}})
    monkeypatch.setattr(d, 'command', lambda *a, **k: pytest.fail('network called'))
    with pytest.raises(ValueError, match='different requested phase'):
        d.finalize(phase, out)


def test_shared_budget_exhaustion_admits_no_more_commands(monkeypatch):
    calls = []
    monkeypatch.setattr(d.time, 'time', lambda: 100)
    monkeypatch.setattr(d.subprocess, 'run', lambda *a, **k: calls.append(k['timeout']))
    token = d.COMMAND_DEADLINE.set(110)
    try:
        d.ssh(['mkdir', '/safe'])
        assert calls == [10]
        monkeypatch.setattr(d.time, 'time', lambda: 111)
        with pytest.raises(TimeoutError, match='no command admitted'):
            d.ssh(['mkdir', '/safe/next'])
        assert calls == [10]
    finally:
        d.COMMAND_DEADLINE.reset(token)
