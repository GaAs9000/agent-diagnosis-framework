"""The public run command never silently selects an unmeasured baseline."""
from pathlib import Path

import pytest

from l2_sandbox import cli


@pytest.mark.parametrize('execute', [False, True])
def test_run_requires_p0_before_config_or_execution(tmp_path, monkeypatch, capsys, execute):
    touched = []
    monkeypatch.setattr(cli, 'load_config', lambda *_: touched.append('config'))
    monkeypatch.setattr(cli, 'run_episode', lambda *a, **kw: touched.append('run'))
    output = tmp_path / 'output'
    argv = ['run', str(tmp_path / 'missing-parent'), '--output', str(output),
            '--analysts', 'llm:missing-config.json', '--config', 'missing-config.json']
    if execute:
        argv.append('--execute')

    with pytest.raises(SystemExit) as exc:
        cli.main(argv)

    assert exc.value.code == 2
    assert '--p0' in capsys.readouterr().err
    assert touched == []
    assert not output.exists()


@pytest.mark.parametrize('p0', [0.0, 0.37, 1.0])
def test_explicit_p0_and_provenance_are_forwarded(tmp_path, monkeypatch, capsys, p0):
    calls = []

    def fake_run(parent, **kwargs):
        calls.append((parent, kwargs))
        return {'executed': False}

    monkeypatch.setattr(cli, 'run_episode', fake_run)
    parent = tmp_path / 'parent'
    provenance = 'frozen M2 baseline cohort; parent excluded'
    assert cli.main(['run', str(parent), '--p0', str(p0),
                     '--baseline-provenance', provenance]) == 0
    assert calls[0][0] == Path(parent)
    assert calls[0][1]['p0'] == p0
    assert calls[0][1]['baseline_provenance'] == provenance
    assert calls[0][1]['execute'] is False
    assert '"executed": false' in capsys.readouterr().out


def test_unbound_baseline_json_is_rejected(tmp_path):
    pytest.importorskip('l3_sandbox')
    from l2_sandbox.io import L2Error, write_json
    from l2_sandbox.report import resolve_m2_baseline_p0
    summary = tmp_path / 'forged.json'
    write_json(summary, {'complete_cohorts': [{'n': 3, 'p0': .4}]})
    with pytest.raises(L2Error, match='Unverified baseline JSON'):
        resolve_m2_baseline_p0(summary, tmp_path / 'parent')


def test_wrapper_delegates_all_semantics_and_cli_provenance(tmp_path, monkeypatch):
    l3_report = pytest.importorskip('l3_sandbox.report')
    from l2_sandbox.report import resolve_m2_baseline_p0
    calls = []
    resolved = {'p0': .6, 'cohort_verified': True, 'used_n': 2,
                'provenance': 'sealed source; excluded_parent_trial=t0'}
    def resolve(source, parent, *, condition_id=None):
        calls.append((source, parent, condition_id))
        return resolved
    monkeypatch.setattr(l3_report, 'resolve_baseline_p0', resolve)
    source, parent = tmp_path / 'source', tmp_path / 'parent'
    assert resolve_m2_baseline_p0(source, parent, condition_id='requests_035') == resolved
    assert calls == [(source, parent, 'requests_035')]
    run_calls = []
    monkeypatch.setattr(cli, 'run_episode', lambda *a, **kw: run_calls.append(kw) or {'executed': False})
    assert cli.main(['run', str(parent), '--baseline', str(source)]) == 0
    assert run_calls[0]['p0'] == .6
    assert run_calls[0]['cohort_verified'] is True
    assert run_calls[0]['baseline_provenance'] == resolved['provenance']


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
