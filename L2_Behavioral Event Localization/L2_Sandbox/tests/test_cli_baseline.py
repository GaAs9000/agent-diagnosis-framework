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


def test_baseline_option_resolves_parent_excluded_p0(tmp_path):
    from conftest import make_l1_episode
    from l2_sandbox.evidence.l1 import L1Package
    from l2_sandbox.io import L2Error, digest, read_json, write_json
    from l2_sandbox.report import resolve_m2_baseline_p0

    p1 = L1Package(make_l1_episode(tmp_path / 'ep1', num_turns=2, task_id='officeval_004'))
    p2 = L1Package(make_l1_episode(tmp_path / 'ep2', num_turns=2, task_id='officeval_004'))
    p_ext = L1Package(make_l1_episode(tmp_path / 'ep_ext', num_turns=2, task_id='officeval_004'))
    cfg_sha = digest(p1.json('raw/resolved_config.json'))
    summary_path = tmp_path / 'summary.json'
    write_json(summary_path, {
        'm2_baseline': {
            'exported': True,
            'complete_cohorts': [{
                'key': 'cohort_004_35',
                'task_id': 'officeval_004',
                'condition_id': 'requests_035',
                'config_sha256': cfg_sha,
                'model_version': p1.model_version(),
                'outcome_endpoint_version': 'study-outcome-v1',
                'n': 3,
                'p0': 0.4,
                'selected_parent_exclusion_supported': True,
                'trials': [
                    {'trial_id': 't0', 'episode_id': p1.episode_id, 'expected_index_sha256': p1.index_sha256, 'study_outcome_y': 0.0, 'p0_excluding_self': 0.6},
                    {'trial_id': 't1', 'episode_id': p2.episode_id, 'expected_index_sha256': p2.index_sha256, 'study_outcome_y': 0.4, 'p0_excluding_self': 0.4},
                    {'trial_id': 't2', 'episode_id': 'ep3', 'expected_index_sha256': 'c' * 64, 'study_outcome_y': 0.8, 'p0_excluding_self': 0.2},
                ],
            }],
        }
    })
    # Parent p1 is in the cohort (y=0.0): p0 excludes p1 and averages {0.4, 0.8} = 0.6
    r_in = resolve_m2_baseline_p0(summary_path, p1.root)
    assert r_in['p0'] == pytest.approx(0.6)
    assert r_in['parent_in_cohort'] is True
    assert r_in['excluded_trial_id'] == 't0'
    assert r_in['used_n'] == 2

    # External parent p_ext is distinct from all cohort runs: averages all 3 runs = 0.4
    r_ext = resolve_m2_baseline_p0(summary_path, p_ext.root)
    assert r_ext['p0'] == pytest.approx(0.4)
    assert r_ext['parent_in_cohort'] is False
    assert r_ext['used_n'] == 3

    # CLI --baseline writes cohort_verified=True to baseline.json
    out_dir = tmp_path / 'l2_out'
    assert cli.main(['run', str(p1.root), '--baseline', str(summary_path), '--output', str(out_dir)]) == 0
    b_doc = read_json(out_dir / 'baseline.json')
    assert b_doc['p0'] == pytest.approx(0.6)
    assert b_doc['cohort_verified'] is True
    assert 'excluded_parent_trial=t0' in b_doc['provenance']

