"""L2 v0.4 CLI tests: optional p0/baseline, diagnose, verify, and summarize."""
from pathlib import Path

import pytest

from l2_sandbox import cli


def test_diagnose_runs_without_requiring_p0(tmp_path, monkeypatch, capsys):
    calls = []

    def fake_diagnose(parent, **kwargs):
        calls.append((parent, kwargs))
        return {'executed': False}

    monkeypatch.setattr(cli, 'diagnose_episode', fake_diagnose)
    parent = tmp_path / 'parent'
    assert cli.main(['diagnose', str(parent)]) == 0
    assert calls[0][0] == Path(parent)
    assert calls[0][1]['p0'] is None
    assert calls[0][1]['execute'] is False
    assert '"executed": false' in capsys.readouterr().out


@pytest.mark.parametrize('p0', [0.0, 0.37, 1.0])
def test_explicit_p0_is_forwarded(tmp_path, monkeypatch, capsys, p0):
    calls = []

    def fake_diagnose(parent, **kwargs):
        calls.append((parent, kwargs))
        return {'executed': False}

    monkeypatch.setattr(cli, 'diagnose_episode', fake_diagnose)
    parent = tmp_path / 'parent'
    assert cli.main(['diagnose', str(parent), '--p0', str(p0)]) == 0
    assert calls[0][0] == Path(parent)
    assert calls[0][1]['p0'] == p0
    assert calls[0][1]['execute'] is False


def test_verify_cli_forwards_mode_checkpoint_and_arms(tmp_path, monkeypatch, capsys):
    calls = []

    def fake_verify(parent, **kwargs):
        calls.append((parent, kwargs))
        return {'executed': False, 'mode': kwargs['mode']}

    monkeypatch.setattr(cli, 'verify_episode', fake_verify)
    parent = tmp_path / 'parent'
    assert cli.main(['verify', str(parent), '--mode', 'intervention', '--checkpoint', '29', '--arms', 'plain,deliver', '--n', '24']) == 0
    assert calls[0][0] == Path(parent)
    assert calls[0][1]['mode'] == 'intervention'
    assert calls[0][1]['k'] == 29
    assert calls[0][1]['arms'] == ['plain', 'deliver']
    assert calls[0][1]['n'] == 24
    assert calls[0][1]['execute'] is False


def test_unbound_baseline_json_is_rejected(tmp_path):
    pytest.importorskip('l3_sandbox')
    from l2_sandbox.io import L2Error, write_json
    from l2_sandbox.report import resolve_m2_baseline_p0
    summary = tmp_path / 'forged.json'
    write_json(summary, {'complete_cohorts': [{'n': 3, 'p0': .4}]})
    with pytest.raises(L2Error, match='Unverified baseline JSON'):
        resolve_m2_baseline_p0(summary, tmp_path / 'parent')


def test_baseline_without_l3_has_clear_install_error(tmp_path, monkeypatch):
    import builtins
    from l2_sandbox.io import L2Error
    from l2_sandbox.report import resolve_m2_baseline_p0
    original = builtins.__import__
    def without_l3(name, *args, **kwargs):
        if name.startswith('l3_sandbox'):
            raise ImportError('not installed')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', without_l3)
    with pytest.raises(L2Error, match='install the L3_Sandbox package'):
        resolve_m2_baseline_p0(tmp_path, tmp_path)
