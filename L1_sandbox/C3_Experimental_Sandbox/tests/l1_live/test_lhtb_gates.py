from pathlib import Path
import xml.etree.ElementTree as ET
import pytest
from c3_sandbox.l1.benchmarks.lhtb.adapter import NativeLHTB, PIN
from c3_sandbox.l1.core.contracts import ConfigurationError, EvaluationSpec
from c3_sandbox.l1.evaluation import lhtb
from c3_sandbox.l1.storage import file_digest, write_json


@pytest.fixture
def locked_task(tmp_path):
    data = tmp_path / 'data'
    task = data / 'tasks/sokoban'
    for name in ('instruction.md', 'task.toml', 'tests/test_outputs.py', 'bin/sokoban'):
        path = task / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    entries = {p.relative_to(task).as_posix(): {'sha256': file_digest(p), 'size_bytes': p.stat().st_size}
               for p in task.rglob('*') if p.is_file()}
    write_json(data, 'data.lock.json', {'source_commit': PIN, 'tasks': {'sokoban': entries}})
    return data, task


@pytest.mark.parametrize('change', ['hash', 'size', 'extra', 'missing', 'commit'])
def test_admission_rejects_invalid_lock(locked_task, change):
    data, task = locked_task
    config = {'data_dir': str(data)}
    if change == 'hash':
        (task / 'bin/sokoban').write_text('X' * len('bin/sokoban'))
    elif change == 'size':
        (task / 'bin/sokoban').write_text('changed size')
    elif change == 'extra':
        (task / 'tests/extra.py').write_text('extra')
    elif change == 'missing':
        (task / 'bin/sokoban').unlink()
    else:
        config['source_commit'] = 'wrong'
    with pytest.raises(ConfigurationError, match='Invalid LHTB data lock'):
        NativeLHTB(config).load('sokoban')


def test_admission_verified_lock(locked_task):
    data, _ = locked_task
    assert NativeLHTB({'data_dir': str(data)}).load('sokoban').provenance['data_lock_status'] == 'sha256_locked'


@pytest.mark.parametrize('raw', [{}, {'status': 'error', 'reward': 0, 'raw_reward': 0},
    {'status': 'ok', 'reward': 0}, {'status': 'ok', 'error': 'failed', 'reward': 0, 'raw_reward': 0}, {'status': 'ok', 'reward': '0', 'raw_reward': 0},
    {'status': 'ok', 'reward': 0, 'raw_reward': float('nan')},
    {'status': 'ok', 'reward': float('inf'), 'raw_reward': 0}])
def test_interpret_rejects_invalid_score(raw):
    assert lhtb.interpret_lhtb(raw, 'sokoban')['evaluation_validity'] == 'invalid'


def fake_runtime(monkeypatch, files, *, exit_code=0, stop=None, failure=None):
    class Runtime:
        def __init__(self, config):
            pass
        def preflight(self):
            return {'image_id': 'fixture'}
        def container(self, mounts, command, timeout, cwd):
            grade = next(source for source, target, _ in mounts if target == '/logs/verifier')
            for name, text in files.items():
                (grade / name).write_text(text)
            if failure:
                root = ET.Element('testsuites')
                case = ET.SubElement(ET.SubElement(root, 'testsuite'), 'testcase', name='test_sokoban_progress')
                ET.SubElement(case, 'failure', message=failure)
                ET.ElementTree(root).write(grade / 'pytest.xml')
            return {'stdout': b'raw stdout', 'stderr': b'raw stderr', 'container_exit_code': exit_code,
                    'stop_reason': stop, 'output_complete': True, 'oom_killed': False}
    monkeypatch.setattr(lhtb, 'DockerRuntime', Runtime)


def evaluate(tmp_path, files, monkeypatch, **kwargs):
    fake_runtime(monkeypatch, files, **kwargs)
    sub = tmp_path / 'submission'
    sub.mkdir()
    (sub / 'actor.txt').write_text('agent')
    source = tmp_path / 'source/tests'
    source.mkdir(parents=True)
    (source / 'test_outputs.py').write_text('official fixture')
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in sub.iterdir()}
    result = lhtb.LHTBEvaluator({}).evaluate(
        EvaluationSpec(kind='lhtb-verifier', source_root=source.parent, entrypoint='tests/test_outputs.py', task_id='sokoban'),
        sub, tmp_path / 'out')
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in sub.iterdir()} == before
    assert (tmp_path / 'out/stdout.bin').read_bytes() == b'raw stdout'
    return result


@pytest.mark.parametrize('files,kwargs', [
    ({'pytest_exit.txt': '0'}, {}),
    ({'pytest_exit.txt': '0', 'reward.txt': 'oops', 'raw_reward.txt': '1'}, {}),
    ({'pytest_exit.txt': '0', 'reward.txt': 'nan', 'raw_reward.txt': '1'}, {}),
    ({'pytest_exit.txt': '0', 'reward.txt': '.1', 'raw_reward.txt': 'inf'}, {}),
    ({'pytest_exit.txt': '0', 'reward.txt': '.1'}, {}),
    ({'pytest_exit.txt': '2', 'reward.txt': '.1', 'raw_reward.txt': '1'}, {'exit_code': 2}),
    ({'reward.txt': '.1', 'raw_reward.txt': '1'}, {}),
    ({'pytest_exit.txt': '1', 'reward.txt': '0', 'raw_reward.txt': '0'}, {'exit_code': 1, 'failure': 'RuntimeError: scorer broke'}),
    ({'pytest_exit.txt': '0', 'reward.txt': '.1', 'raw_reward.txt': '1'}, {'stop': 'timeout'}),
    ({'pytest_exit.txt': '0', 'reward.txt': '.1', 'raw_reward.txt': '1'}, {'exit_code': 137}),
])
def test_evaluator_invalid_protocol(tmp_path, monkeypatch, files, kwargs):
    grade = evaluate(tmp_path, files, monkeypatch, **kwargs)
    assert grade['projection']['evaluation_validity'] == 'invalid'
    assert grade['official_raw']['status'] == 'error'
    assert grade['status'] == 'infra_error'
    assert grade['error_kind'] == 'infrastructure'
    assert grade['error'] == grade['projection']['reason']


def test_official_no_moves_preserves_missing_raw_and_submission(tmp_path, monkeypatch):
    grade = evaluate(tmp_path, {'pytest_exit.txt': '1', 'reward.txt': '0'}, monkeypatch,
                     exit_code=1, failure='AssertionError: No moves found at /submission/moves.log. The agent did not play')
    assert grade['projection']['task_outcome'] == 'failure'
    assert 'error_kind' not in grade
    assert grade['projection']['reason'] == 'official_no_moves'
    assert grade['official_raw']['raw_reward'] is None
    assert grade['official_raw']['grading_input']['synthesized_files'] == []


@pytest.mark.parametrize('score,exit_code,failure,outcome', [
    ('1.6667', 0, None, 'success'),
    ('0.5', 1, 'AssertionError: Band 0 < pass threshold 1. Solved 0', 'partial')])
def test_official_scores(tmp_path, monkeypatch, score, exit_code, failure, outcome):
    grade = evaluate(tmp_path, {'pytest_exit.txt': str(exit_code), 'reward.txt': '.01', 'raw_reward.txt': score},
                     monkeypatch, exit_code=exit_code, failure=failure)
    assert grade['projection']['evaluation_validity'] == 'valid'
    assert 'error_kind' not in grade
    assert grade['projection']['task_outcome'] == outcome


def test_l2_rejects_infrastructure_grade(tmp_path, monkeypatch):
    l2 = pytest.importorskip('l2_sandbox.evidence.l1')
    grade = evaluate(tmp_path, {'pytest_exit.txt': '0'}, monkeypatch)
    package = tmp_path / 'l2-package'
    package.mkdir()
    write_json(package, 'manifest.json', {
        'episode_id': 'lhtb-scorer-failure-fixture', 'benchmark': 'lhtb',
        'status': {'execution_validity': 'valid', 'evaluation_validity': 'invalid'}
    })
    write_json(package, 'artifacts/inventory.json', {'files': [{'path': 'moves.log'}]})
    write_json(package, 'official_grade.json', grade)
    outcome = l2.L1Package(package).outcome()
    assert outcome['valid'] is False
    assert outcome['y'] is None
    assert outcome['reason'] == 'scorer_exception'
