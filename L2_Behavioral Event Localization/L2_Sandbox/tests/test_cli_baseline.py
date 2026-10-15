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
