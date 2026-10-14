"""M1 gates must reject drift before any paid model execution."""
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('m1_runner', Path(__file__).parents[1] / 'scripts/run_m1.py')
m1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m1)


def frozen(tmp_path, monkeypatch):
    source = tmp_path / 'source.py'
    source.write_text('frozen source')
    config = tmp_path / 'configs/task.json'
    m1.write(config, {'test': True})
    monkeypatch.setattr(m1, 'sources', lambda: {str(source): m1.sha(source)})
    monkeypatch.setattr(m1, 'image_id', lambda image: 'image-id')
    plan = {'source_sha256': m1.sources(), 'templates_sha256': {}, 'locks_sha256': {},
            'configs_sha256': {str(config): m1.sha(config)}, 'configs': {'task': {'test': True}},
            'images': {'tag': 'image-id'}}
    m1.write(tmp_path / 'frozen_plan.json', plan)
    (tmp_path / 'frozen_plan.sha256').write_text(m1.sha(tmp_path / 'frozen_plan.json'))
    return source, config


@pytest.mark.parametrize('mutation', ['source', 'config', 'added_source', 'image'])
def test_frozen_drift_rejects_before_credentials_or_send(tmp_path, monkeypatch, mutation):
    source, config = frozen(tmp_path, monkeypatch)
    assert m1.verify_plan(tmp_path)
    if mutation == 'source':
        source.write_text('changed')
    elif mutation == 'config':
        config.write_text('{}')
    elif mutation == 'added_source':
        previous = m1.sources()
        monkeypatch.setattr(m1, 'sources', lambda: {**previous, 'new.py': 'new'})
    else:
        monkeypatch.setattr(m1, 'image_id', lambda image: 'changed')
    touched = []
    monkeypatch.setattr(m1, 'load_credentials', lambda: touched.append('credentials'))
    with pytest.raises(m1.GateError):
        m1.execute(tmp_path, 'parents')
    assert touched == []


@pytest.mark.parametrize('selected', [True, False])
def test_failure_selection_delegates_to_package(selected):
    class Package:
        def is_failed_trajectory(self):
            return selected
    assert m1.failure_selected(Package()) is selected


def statuses(count, invalid=0, wall=0):
    rows = [{'execution_validity': 'valid', 'evaluation_validity': 'valid'} for _ in range(count)]
    for row in rows[:invalid]:
        row['execution_validity'] = 'invalid'
    for row in rows[:wall]:
        row['termination'] = 'wall_time_budget'
    return rows


def test_rates_are_strict_and_include_invalid_trials():
    assert not m1.rate_summary([])['passed']
    assert not m1.rate_summary(statuses(20, invalid=1))['passed']
    assert m1.rate_summary(statuses(21, invalid=1))['passed']
    assert not m1.rate_summary(statuses(100, wall=1))['passed']
    assert m1.rate_summary(statuses(101, wall=1))['passed']


def test_auth_guard_latches_and_blocks_admission(tmp_path):
    class Limiter:
        called = False
        def acquire(self, *args, **kwargs):
            self.called = True
            return True
    limiter = Limiter()
    guard = m1.AuthGuardLimiter(limiter, [tmp_path])
    m1.write(tmp_path / 'raw/model/r000000/http.json', {'status_code': 403})
    with pytest.raises(m1.GateError):
        guard.acquire()
    assert not limiter.called
    (tmp_path / 'raw/model/r000000/http.json').unlink()
    with pytest.raises(m1.GateError):
        guard.acquire()


def test_submission_identity_catches_mode_and_mtime(tmp_path):
    path = tmp_path / 'answer'
    path.write_text('answer')
    before = m1.tree_identity(tmp_path)
    path.chmod(0o600)
    assert m1.tree_identity(tmp_path) != before


def test_missing_paid_flag_rejects(monkeypatch):
    touched = []
    monkeypatch.setattr(m1, 'execute', lambda *a: touched.append(True))
    with pytest.raises(SystemExit):
        m1.main(['--id', 'test', '--phase', 'parents'])
    assert not touched


def test_http_admission_cap_is_persisted_across_phases(tmp_path):
    class Limiter:
        def acquire(self, *args, **kwargs):
            return True
    counter = tmp_path / 'admissions.json'
    guard = m1.AuthGuardLimiter(Limiter(), [tmp_path], counter_path=counter, cap=2)
    assert guard.acquire()
    assert guard.acquire()
    restarted = m1.AuthGuardLimiter(Limiter(), [tmp_path], counter_path=counter, cap=2)
    with pytest.raises(m1.GateError):
        restarted.acquire()
    assert m1.read(counter) == {'admitted': 2, 'cap': 2}


def test_summary_cannot_pass_partial_and_counts_fidelity_invalid(tmp_path, monkeypatch):
    monkeypatch.setattr(m1, 'L1', tmp_path / 'l1')
    directory = tmp_path / 'validation'
    parent = m1.L1 / 'runs/m1_test/parents/officeval_003'
    status = {'execution_validity': 'valid', 'evaluation_validity': 'valid'}
    m1.write(parent / 'manifest.json', {'status': status})
    branch = directory / 'officeval_003/branches/k000-s1-i00'
    m1.write(branch / 'manifest.json', {'status': status})
    (branch.parent.parent / 'outcomes.jsonl').write_text('{"trial":"k000-s1-i00","valid":false}\n')
    result = m1.summarize(directory, {'id': 'test'})
    assert result['denominator'] == 2
    assert result['invalid'] == 1
    assert result['all_phases_complete'] is False
    assert result['milestone_passed'] is False


def office_missing_grade():
    from c3_sandbox.l1.evaluation.officeval import interpret
    raw = {'id': '003', 'file_name': 'missing.docx', 'status': 'error',
           'error': 'Required file not found', 'dim1_pass': False, 'dim1_reason': '',
           'dim2_items': [], 'total_score': 0, 'max_score': 15}
    return {'official_raw': raw, 'projection': interpret(raw, 'officeval_003'),
            'process': {'container_exit_code': 0, 'stop_reason': None}}


def test_empty_office_parent_is_valid_zero_without_rewriting_native_invalid():
    class Package:
        manifest = {'recording_mode': 'live', 'status': {'execution_validity': 'valid',
                    'evaluation_validity': 'invalid', 'evidence_completeness': 'complete'}}
        def outcome(self):
            return {'valid': True, 'y': 0., 'reason': 'empty_submission_zero'}
    pkg = Package()
    assert m1.valid_parent(pkg)
    assert pkg.manifest['status']['evaluation_validity'] == 'invalid'


@pytest.mark.parametrize('fault', [None, 'missing_raw', 'process_crash', 'worker_crash', 'wrong_id', 'infrastructure'])
def test_office_nonok_requires_healthy_complete_official_return(tmp_path, fault):
    grade = office_missing_grade()
    worker = {'official_returned': True, 'exception': None}
    if fault == 'missing_raw':
        grade['official_raw'] = None
    elif fault == 'process_crash':
        grade['process']['container_exit_code'] = 1
    elif fault == 'worker_crash':
        worker['exception'] = {'type': 'ImportError'}
    elif fault == 'wrong_id':
        grade['official_raw']['id'] = '090'
    elif fault == 'infrastructure':
        grade['error_kind'] = 'infrastructure'
    m1.write(tmp_path / 'worker-output/worker.json', worker)
    assert m1.grader_healthy('officeval', 'officeval_003', grade, tmp_path) is (fault is None)


def test_outcomes_without_trial_directory_still_count(tmp_path, monkeypatch):
    monkeypatch.setattr(m1, 'L1', tmp_path / 'l1')
    directory = tmp_path / 'validation'
    task = directory / 'officeval_003'
    task.mkdir(parents=True)
    (task / 'outcomes.jsonl').write_text('{"trial":"k000-s1-i00","valid":false}\n')
    result = m1.summarize(directory, {'id': 'test'})
    assert result['denominator'] == 1
    assert result['invalid'] == 1
