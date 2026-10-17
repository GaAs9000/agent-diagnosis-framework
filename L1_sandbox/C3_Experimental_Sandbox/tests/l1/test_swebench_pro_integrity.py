"""Offline SWE scoring and asset-lock integrity regressions; no Docker/model calls."""
import copy
import json
import subprocess
import sys

import pytest

from c3_sandbox.l1.benchmarks.swebench_pro.adapter import NativeSWEBenchPro
from c3_sandbox.l1.core.contracts import ConfigurationError, EvaluationSpec
from c3_sandbox.l1.evaluation import swebench_pro as swe
from c3_sandbox.l1.storage import file_digest


def official(passed=True):
    return {
        'task_id': 'task', 'passed': passed, 'reward': float(passed),
        'score': float(passed), 'status': 'ok' if passed else 'failed',
        'exit_code': 0 if passed else 1,
        'protocol': {'reward_text': str(int(passed)), 'test_exit_text': str(int(not passed)),
                     'test_run_exit_text': '0', 'output_export': {'truncated': False}},
        'test_config': {'fail_to_pass': ['new'], 'pass_to_pass': ['old']},
        'test_results': {'tests': [{'name': 'new', 'status': 'PASSED' if passed else 'FAILED'},
                                   {'name': 'old', 'status': 'PASSED'}]},
    }


@pytest.mark.parametrize('passed', [True, False])
def test_interpret_healthy_official_result(passed):
    result = swe.interpret_swebench_pro(official(passed), 'task')
    assert result['evaluation_validity'] == 'valid'
    assert result['task_outcome'] == ('success' if passed else 'failure')


@pytest.mark.parametrize('raw', [None, [], {}, {'passed': False},
    *[dict(official(), **change) for change in [
        {'task_id': 'other'}, {'passed': 'false'}, {'passed': 1}, {'reward': True},
        {'reward': '1'}, {'reward': float('nan')}, {'reward': float('inf')},
        {'reward': 0.5}, {'reward': 10**1000}, {'score': False}, {'score': 0}, {'status': 'failed'},
        {'exit_code': True}, {'exit_code': 2}, {'protocol': None},
        {'protocol': {'reward_text': None}}, {'test_results': {}},
        {'test_results': {'tests': []}}, {'test_results': {'tests': [None]}},
        {'test_results': {'tests': [{'name': 'new', 'status': 'UNKNOWN'}]}},
        {'test_config': {}}, {'test_config': {'fail_to_pass': [], 'pass_to_pass': []}},
        {'error': 'timeout'},
    ]]])
def test_interpret_rejects_malformed_or_inconsistent_raw(raw):
    assert swe.interpret_swebench_pro(raw, 'task')['task_outcome'] == 'unknown'


@pytest.fixture
def evaluate(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    (source / 'tests').mkdir(parents=True)
    (source / 'tests/config.json').write_text(json.dumps(official()['test_config']))
    submission = tmp_path / 'submission'
    submission.mkdir()
    spec = EvaluationSpec(kind='swebench_pro-verifier', source_root=source,
                          entrypoint='tests/test.sh', task_id='task', metadata={})

    def run(*, passed=True, artifacts=None, process_changes=None, output_limit=8388608):
        raw = official(passed)
        files = {'reward.txt': raw['protocol']['reward_text'],
                 'test_exit_code.txt': raw['protocol']['test_exit_text'],
                 'test_run_exit.txt': '0', 'output.json': json.dumps(raw['test_results']),
                 'output_export.json': json.dumps({'limit_bytes': output_limit, 'truncated': False})}
        files.update(artifacts or {})
        process = {'stdout': b'raw stdout', 'stderr': b'raw stderr',
                   'container_exit_code': raw['exit_code'], 'stop_reason': None,
                   'oom_killed': False, 'output_complete': True}
        process.update(process_changes or {})

        class Runtime:
            def __init__(self, config):
                pass

            def preflight(self):
                return {'image_id': 'offline'}

            def container(self, mounts, command, timeout, **kwargs):
                run.command = command
                for name, content in files.items():
                    if content is not None:
                        (mounts[-1][0] / name).write_text(content)
                return copy.deepcopy(process)

        monkeypatch.setattr(swe, 'DockerRuntime', Runtime)
        output = tmp_path / 'evaluation'
        grade = swe.SWEBenchProEvaluator({'output_limit_bytes': output_limit}).evaluate(spec, submission, output)
        assert (output / 'stdout.bin').read_bytes() == b'raw stdout'
        assert (output / 'stderr.bin').read_bytes() == b'raw stderr'
        assert json.loads((output / 'process.json').read_text())['container_exit_code'] == process['container_exit_code']
        assert (output / 'worker-output/official.json').is_file()
        return grade

    return run


@pytest.mark.parametrize('passed', [True, False])
def test_evaluator_valid_pass_and_failure(evaluate, passed):
    grade = evaluate(passed=passed)
    assert grade['projection']['evaluation_validity'] == 'valid'
    assert grade['official_raw']['exit_code'] == (0 if passed else 1)


@pytest.mark.parametrize('artifacts,process', [
    ({'output_export.json': None}, {}),
    ({'output_export.json': '{}'}, {}),
    ({'output_export.json': '{"limit_bytes": 8388608, "truncated": true}'}, {}),
    ({'reward.txt': None}, {}),  # An exit-code receipt alone is never a score.
    ({'reward.txt': ''}, {}), ({'reward.txt': 'garbage'}, {}),
    ({'reward.txt': 'nan'}, {}), ({'reward.txt': 'true'}, {}),
    ({'reward.txt': '0'}, {}), ({'output.json': None}, {}),
    ({'output.json': '{broken'}, {}), ({'output.json': '{}'}, {}),
    ({'output.json': '{"tests": []}'}, {}),
    ({'test_exit_code.txt': None}, {}), ({'test_exit_code.txt': '1'}, {}),
    ({'test_run_exit.txt': None}, {}), ({'test_run_exit.txt': '137'}, {}),
    ({}, {'stop_reason': 'timeout', 'container_exit_code': 137}),
    ({}, {'oom_killed': True}), ({}, {'oom_killed': None}), ({}, {'output_complete': False}),
    ({}, {'container_exit_code': 2}), ({}, {'container_exit_code': None}),
])
def test_evaluator_preserves_invalid_results_as_unknown(evaluate, artifacts, process):
    grade = evaluate(artifacts=artifacts, process_changes=process)
    assert grade['status'] == 'infra_error'
    assert grade['error_kind'] == 'infrastructure'
    assert grade['projection']['evaluation_validity'] == 'invalid'
    assert grade['projection']['task_outcome'] == 'unknown'
    assert grade['official_raw']['exit_code'] == process.get('container_exit_code', 0)


def test_missing_required_test_is_a_legitimate_official_negative(evaluate):
    grade = evaluate(passed=False, artifacts={'output.json': json.dumps({
        'tests': [{'name': 'old', 'status': 'PASSED'}]})})
    assert grade['projection']['evaluation_validity'] == 'valid'
    assert grade['projection']['completion_rate'] == 0


@pytest.mark.parametrize('run_exit', ['0', '1'])
def test_agent_collection_failure_is_a_legitimate_official_negative(evaluate, run_exit):
    grade = evaluate(passed=False, artifacts={'output.json': '{"tests": []}',
                                             'test_run_exit.txt': run_exit})
    assert grade['projection']['evaluation_validity'] == 'valid'
    assert grade['projection']['task_outcome'] == 'failure'
    assert grade['official_raw']['reward'] == 0


@pytest.fixture
def assets(tmp_path):
    task = tmp_path / 'tasks/task'
    (task / 'tests').mkdir(parents=True)
    for name, content in [('instruction.md', 'Task'), ('task.toml', 'name="task"'),
                          ('tests/config.json', '{}')]:
        (task / name).write_text(content)
    entries = {p.relative_to(task).as_posix(): {'sha256': file_digest(p), 'size_bytes': p.stat().st_size}
               for p in task.rglob('*') if p.is_file()}
    lock = {'tasks': {'task': entries}}
    lock_path = tmp_path / 'data.lock.json'
    lock_path.write_text(json.dumps(lock))
    return NativeSWEBenchPro({'data_dir': str(tmp_path)}), task, lock, lock_path


def test_adapter_lock_verified(assets):
    adapter, _, _, _ = assets
    assert adapter.load('task').provenance['data_lock_status'] == 'sha256_locked'


@pytest.mark.parametrize('change', ['hash', 'size', 'size_bool', 'extra', 'missing', 'missing_task',
                                    'traversal', 'absolute', 'symlink_file', 'symlink_dir', 'hardlink'])
def test_adapter_rejects_asset_drift(assets, tmp_path, change):
    adapter, task, lock, lock_path = assets
    entries = lock['tasks']['task']
    if change == 'hash':
        (task / 'tests/config.json').write_text('[]')  # Same byte count, different hash.
    elif change == 'size':
        entries['tests/config.json']['size_bytes'] += 1
    elif change == 'size_bool':
        entries['tests/config.json']['size_bytes'] = True
    elif change == 'extra':
        (task / 'extra.txt').write_text('extra')
    elif change == 'missing':
        (task / 'tests/config.json').unlink()
    elif change == 'missing_task':
        lock['tasks'] = {}
    elif change in ('traversal', 'absolute'):
        entries['../outside' if change == 'traversal' else '/outside'] = entries.pop('tests/config.json')
    elif change == 'symlink_file':
        (task / 'tests/config.json').unlink()
        (task / 'tests/config.json').symlink_to(task / 'task.toml')
    elif change == 'symlink_dir':
        (task / 'linked').symlink_to(task / 'tests', target_is_directory=True)
    elif change == 'hardlink':
        (tmp_path / 'linked').hardlink_to(task / 'tests/config.json')
    lock_path.write_text(json.dumps(lock))
    with pytest.raises((ConfigurationError, ValueError)):
        adapter.load('task')


@pytest.mark.parametrize('task_id', ['../task', '/tmp/task', 'task/../task'])
def test_adapter_rejects_task_path_escape(assets, task_id):
    with pytest.raises(ValueError):
        assets[0].load(task_id)


@pytest.mark.parametrize('oversized', ['stdout.log', 'output.json'])
def test_worker_exports_share_a_bound_and_overlimit_is_invalid(tmp_path, evaluate, oversized):
    source = tmp_path / 'worker-tmp'
    target = tmp_path / 'export'
    source.mkdir()
    target.mkdir()
    for name in ['stdout.log', 'stderr.log', 'output.json']:
        (source / name).write_bytes(b'x' * (256 if name == oversized else 2))
    subprocess.run([sys.executable, '-c', swe._BOUNDED_EXPORT, str(source), str(target), '64'], check=True)
    files = ['test_stdout.log', 'test_stderr.log', 'output.json']
    assert sum((target / name).stat().st_size for name in files) <= 64
    receipt = json.loads((target / 'output_export.json').read_text())
    assert receipt['truncated'] is True
    assert receipt['limit_bytes'] == 64
    grade = evaluate(artifacts={'output_export.json': json.dumps(receipt)}, output_limit=64)
    assert grade['status'] == 'infra_error'
    assert grade['error'] == 'verifier_output_limit'
    assert grade['projection']['task_outcome'] == 'unknown'


def test_worker_shell_syntax_and_tmpfs_log_redirection(evaluate):
    evaluate()
    command = evaluate.command[2]
    subprocess.run(['/bin/bash', '-n'], input=command, text=True, check=True)
    assert '> /tmp/stdout.log 2> /tmp/stderr.log' in command
    assert '> /logs/verifier/test_stdout.log' not in command
    assert 'cp /tmp/output.json' not in command


def test_study_outcome_swebench_pro_passed(tmp_path):
    from c3_sandbox.l1.evaluation.study import study_outcome
    manifest = {
        'recording_mode': 'live',
        'task': {'benchmark': 'swebench_pro', 'task_id': 'task'},
        'status': {'execution_validity': 'valid', 'evaluation_validity': 'valid', 'evidence_completeness': 'complete'}
    }
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    raw = official(passed=True)
    grade = {
        'official_raw': raw,
        'projection': swe.interpret_swebench_pro(raw, 'task'),
        'process': {'container_exit_code': 0, 'stop_reason': None, 'output_complete': True}
    }
    (tmp_path / 'official_grade.json').write_text(json.dumps(grade))
    res = study_outcome(tmp_path)
    assert res['study_valid'] is True
    assert res['study_outcome_y'] == 1.0
    assert res['zero_reason'] is None
    assert res['study_classification'] == 'valid_score'


def test_study_outcome_swebench_pro_failed(tmp_path):
    from c3_sandbox.l1.evaluation.study import study_outcome
    manifest = {
        'recording_mode': 'live',
        'task': {'benchmark': 'swebench_pro', 'task_id': 'task'},
        'status': {'execution_validity': 'valid', 'evaluation_validity': 'valid', 'evidence_completeness': 'complete'}
    }
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    raw = official(passed=False)
    grade = {
        'official_raw': raw,
        'projection': swe.interpret_swebench_pro(raw, 'task'),
        'process': {'container_exit_code': 1, 'stop_reason': None, 'output_complete': True}
    }
    (tmp_path / 'official_grade.json').write_text(json.dumps(grade))
    res = study_outcome(tmp_path)
    assert res['study_valid'] is True
    assert res['study_outcome_y'] == 0.0
    assert res['zero_reason'] == 'official_zero'
    assert res['study_classification'] == 'valid_zero'


def test_study_outcome_swebench_pro_inconsistent(tmp_path):
    from c3_sandbox.l1.evaluation.study import study_outcome
    manifest = {
        'recording_mode': 'live',
        'task': {'benchmark': 'swebench_pro', 'task_id': 'task'},
        'status': {'execution_validity': 'valid', 'evaluation_validity': 'valid', 'evidence_completeness': 'complete'}
    }
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    raw = official(passed=False)
    grade = {
        'official_raw': raw,
        'projection': swe.interpret_swebench_pro(raw, 'task'),
        'process': {'container_exit_code': 0, 'stop_reason': None, 'output_complete': True}  # Mismatched exit code!
    }
    (tmp_path / 'official_grade.json').write_text(json.dumps(grade))
    res = study_outcome(tmp_path)
    assert res['study_valid'] is False

