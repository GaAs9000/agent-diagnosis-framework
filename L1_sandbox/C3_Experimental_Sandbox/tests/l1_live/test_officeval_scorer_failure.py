from pathlib import Path
import pytest
from c3_sandbox.l1.core.contracts import EvaluationSpec
from c3_sandbox.l1.evaluation import evaluator
from c3_sandbox.l1.storage import file_digest, write_json


def official_result(status='ok'):
    return {'id': '003', 'file_name': 'answer.docx', 'status': status,
            'error': None if status == 'ok' else 'missing submission',
            'dim1_pass': status == 'ok', 'dim1_reason': 'official check',
            'dim2_items': [], 'total_score': 0, 'max_score': 1}


def grade(tmp_path, monkeypatch, raw, **process_overrides):
    class Runtime:
        def __init__(self, config):
            pass
        def preflight(self):
            return {'python_environment': {'grader_worker_sha256': file_digest(Path(evaluator.__file__).with_name('worker.py'))}}
        def container(self, mounts, command, timeout, cwd):
            output = next(source for source, target, _ in mounts if target == '/grade')
            if isinstance(raw, str):
                (output / 'official.json').write_text(raw)
            elif raw is not None:
                write_json(output, 'official.json', raw)
            return {'stdout': b'official stdout', 'stderr': b'', 'container_exit_code': 0,
                    'stop_reason': None, 'oom_killed': False, 'output_complete': True,
                    **process_overrides}
    monkeypatch.setattr(evaluator, 'DockerRuntime', Runtime)
    sub = tmp_path / 'submission'
    sub.mkdir()
    return evaluator.DockerEvaluator({'image': 'fixture', 'timeout_seconds': 5, 'outcome_policy': 'official_dimensions'}).evaluate(
        EvaluationSpec(kind='python-verifier', source_root=tmp_path, entrypoint='verifier.py', task_id='officeval_003'),
        sub, tmp_path / 'grade')


@pytest.mark.parametrize('raw,process', [
    (official_result(), {'stop_reason': 'timeout'}),
    (official_result(), {'container_exit_code': 1}),
    (official_result(), {'oom_killed': True}),
    (official_result(), {'output_complete': False}),
    ({'status': 'ok'}, {}),
    ('{malformed', {}),
    (None, {}),
])
def test_unexpected_scorer_failures_are_infrastructure(tmp_path, monkeypatch, raw, process):
    result = grade(tmp_path, monkeypatch, raw, **process)
    assert result['status'] == 'infra_error'
    assert result['error_kind'] == 'infrastructure'
    assert result['projection']['evaluation_validity'] == 'invalid'
    assert result['error'] == result['projection']['reason']


def test_native_official_error_remains_agent_result(tmp_path, monkeypatch):
    result = grade(tmp_path, monkeypatch, official_result('error'))
    assert result['projection']['reason'] == 'official_error'
    assert 'error_kind' not in result
    assert 'status' not in result


def test_l2_rejects_scorer_stop_with_nonempty_official_result(tmp_path, monkeypatch):
    l2 = pytest.importorskip('l2_sandbox.evidence.l1')
    result = grade(tmp_path, monkeypatch, official_result(), stop_reason='timeout')
    package = tmp_path / 'package'
    package.mkdir()
    write_json(package, 'manifest.json', {
        'episode_id': 'officeval-scorer-failure-fixture', 'benchmark': 'officeval',
        'status': {'execution_validity': 'valid', 'evaluation_validity': 'invalid'}})
    write_json(package, 'artifacts/inventory.json', {'files': [{'path': 'answer.docx'}]})
    write_json(package, 'official_grade.json', result)
    outcome = l2.L1Package(package).outcome()
    assert outcome['valid'] is False
    assert outcome['y'] is None
    assert outcome['reason'] == 'scorer_exception'
