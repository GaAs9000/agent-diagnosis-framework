"""Terminal report retries perform no model, scorer, or real network calls."""
import importlib.util
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from l2_sandbox.v3_execution import file_digest, read_json, seal, write_json


@pytest.fixture
def archive_case(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / 'scripts/archive_opencode_terminal.py'
    spec = importlib.util.spec_from_file_location('archive_recovery_helper', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / 'workspace'
    phase = root / 'validation/phase'
    phase.mkdir(parents=True)
    plan = seal({'schema': 'test-plan'})
    write_json(phase / 'execution_plan.json', plan)
    monkeypatch.setattr(module, 'ROOT', root)
    monkeypatch.setattr(module, 'L2', root / 'L2_Behavioral Event Localization/L2_Sandbox')
    monkeypatch.setattr(module, 'collect_inputs', lambda p: (
        plan, {phase / 'execution_plan.json': file_digest(phase / 'execution_plan.json')}))
    remote_report = tmp_path / 'remote_report'
    remote_report.mkdir()
    summary = seal({'census': {'planned': 2282, 'valid': 1792,
                              'invalid_or_incomplete': 78, 'unstarted': 412}})
    write_json(remote_report / 'summary.json', summary)
    (remote_report / 'report.md').write_text('A sealed descriptive report.\n')
    write_json(remote_report / 'manifest.json', seal(dict(
        plan_sha256=plan['artifact_sha256'], summary_sha256=summary['artifact_sha256'],
        generated_files={n: file_digest(remote_report / n) for n in ('summary.json', 'report.md')},
        input_snapshots=[])))
    return module, phase, remote_report


@pytest.mark.parametrize('failure', ['before_upload', 'after_remote_report', 'during_download'])
def test_retry_after_transient_failure_uses_new_attempt_and_atomic_promotion(archive_case, monkeypatch, failure):
    module, phase, remote_report = archive_case
    should_fail = [True]
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        is_ssh = args[0] == 'ssh'
        is_download = args[0] == 'rsync' and args[-2].startswith('aml-b:')
        is_report = is_ssh and 'summarize_opencode_finish.py' in args[-1]
        if should_fail[0] and ((failure == 'before_upload' and is_ssh)
                or (failure == 'after_remote_report' and is_report)):
            should_fail[0] = False
            raise OSError('simulated transient connection failure')
        if is_download:
            destination = Path(args[-1])
            if should_fail[0] and failure == 'during_download':
                (destination / 'partial').write_text('partial transfer')
                should_fail[0] = False
                raise OSError('simulated partial download')
            shutil.copytree(remote_report, destination, dirs_exist_ok=True)
        return SimpleNamespace(returncode=0, stdout='{}\n', stderr='')

    monkeypatch.setattr(module.subprocess, 'run', run)
    with pytest.raises(OSError):
        module.archive(phase)
    assert not (phase / 'terminal_report').exists()
    assert read_json(phase / 'postprocess/result.json')['success'] is False
    result = module.archive(phase)
    assert result['success'] is True
    assert result['actor_requests'] == result['scorer_invocations'] == 0
    attempts = list((phase / 'postprocess').glob('attempt_*'))
    assert len(attempts) == 2
    assert (phase / 'terminal_report/report.md').read_bytes() == (remote_report / 'report.md').read_bytes()
    assert not (phase / 'terminal_report/partial').exists()
    monkeypatch.setattr(module, '_archive_attempt', lambda *a: pytest.fail('completed report reran'))
    reused = module.archive(phase)
    assert reused['success'] and reused['reused_completed_report']
    (phase / 'terminal_report/report.md').write_text('changed')
    with pytest.raises(ValueError, match='hash differs'):
        module.archive(phase)
