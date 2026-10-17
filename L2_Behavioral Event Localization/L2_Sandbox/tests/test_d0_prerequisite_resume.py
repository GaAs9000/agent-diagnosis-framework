"""Offline operational resume tests: never invoke model transport."""
import copy
import importlib.util
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/resume_d0_prerequisites.py'
spec = importlib.util.spec_from_file_location('resume_d0', SCRIPT)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


@pytest.fixture
def environment(tmp_path, monkeypatch):
    c = driver.collector
    monkeypatch.setattr(c, 'OUT_ROOT', tmp_path / 'runs')
    monkeypatch.setattr(c, 'PROGRESS_JSON', tmp_path / 'runs/progress.json')
    monkeypatch.setattr(driver, 'source_fingerprints', lambda: {'controller': 'test'})
    cfg = c.resolve_config(c.CONFIG_PATH)
    monkeypatch.setattr(c, 'resolve_config', lambda _: copy.deepcopy(cfg))
    invalid = set(driver.REPLACEMENTS)
    calls = []
    lock = threading.Lock()
    def record(path):
        label = path.name.removesuffix('_c0')
        valid = label not in invalid and path.exists()
        return valid, dict(episode_id=label, index_sha256='index-' + label, study_score=0)
    monkeypatch.setattr(c, 'is_valid_episode', record)
    for rep in c.REPLICATES:
        for task in c.DEV_TASKS:
            label = f'{task}.{rep}'
            p = c.OUT_ROOT / 'episodes' / label
            p.mkdir(parents=True)
            driver.write_json(p / 'manifest.json', dict(episode_id=label))
            driver.write_json(p / 'evidence_index.json', {})
            driver.write_json(p / 'checkpoints/cp000000.json', {})
            parent_cfg = copy.deepcopy(cfg)
            parent_cfg['benchmark']['task_id'] = task
            driver.write_json(p / 'raw/resolved_config.json', parent_cfg)
            (p / 'raw/model/r000000').mkdir(parents=True)
            driver.write_json(p / 'raw/model/r000000/request.json', {})
    def run(config, output, **kwargs):
        with lock:
            calls.append((config, output, kwargs))
        limiter = kwargs['request_limiter']
        assert limiter.acquire(1)
        limiter.reconcile(1, 1)
        output.mkdir(parents=True)
        label = output.name.removesuffix('_c0')
        lineage = {}
        if 'parent' in kwargs:
            parent = kwargs['parent']
            assert kwargs['checkpoint_id'] == 'cp000000'
            assert config == driver.read_json(parent / 'raw/resolved_config.json')
            lineage = dict(parent_episode_id=label, parent_index_sha256='index-' + label,
                parent_checkpoint_id='cp000000', parent_checkpoint_sha256=driver.file_digest(parent / 'checkpoints/cp000000.json'))
            (output / 'raw/model/r000000').mkdir(parents=True)
            (output / 'raw/model/r000000/request.json').write_text('{}')
        driver.write_json(output / 'raw/resolved_config.json', config)
        driver.write_json(output / 'checkpoints/cp000000.json', {})
        driver.write_json(output / 'raw/model/r000000/request.json', {})
        driver.write_json(output / 'manifest.json', dict(episode_id=label, lineage=lineage, counts=dict(model_requests=1)))
        invalid.discard(label)
    monkeypatch.setattr(c, 'l1_run', run)
    plan = driver.build_plan()
    freeze = dict(phase_authorized=True, phase_frozen=True, d0_frozen=False, approved_plan_sha256=plan['plan_sha256'], limits=driver.LIMITS,
        internal_plan_reviewed=True, user_authorization_record=dict(exact_text='fixture authorization', source='offline fixture'))
    return plan, freeze, calls, invalid, tmp_path / 'output'


def test_plan_zero_inference_and_fixed_workload(environment):
    plan, _, calls, _, _ = environment
    assert not calls
    assert sum(row['valid'] for row in plan['parents'].values()) == 41
    assert plan['attempt_limit_per_slot'] == 1
    assert len(plan['g2_slots']) == 20
    assert plan['limits']['max_requests'] == 4 * 35 * 4 + 20 * 35 * 4


def test_four_single_attempts_then_twenty_fresh_cold_parents(environment):
    plan, freeze, calls, _, output = environment
    before = {k: v for k, v in plan['parents'].items() if v['valid']}
    result = driver.execute(plan, freeze, output)
    assert len(calls) == 24
    assert len({id(row[2]['request_limiter']) for row in calls}) == 1
    assert result['g2_status'] == 'completed'
    assert result['admission']['requests'] == 24
    assert all(row['attempts'] == 1 for row in result['replacements'])
    driver.assert_unchanged(before)
    assert len(list(output.glob('archive_*.json'))) == 4
    for path in output.glob('archive_*.json'):
        mapping = driver.read_json(path)
        assert Path(mapping['archived_path']).exists()
        assert mapping['old']['index_file_sha256'] == driver.file_digest(Path(mapping['archived_path']) / 'evidence_index.json')
    driver.execute(plan, freeze, output)
    assert len(calls) == 24


def test_partial_stops_before_g2_and_never_retries(environment, monkeypatch):
    plan, freeze, calls, invalid, output = environment
    original = driver.collector.l1_run
    def fail(config, path, **kw):
        original(config, path, **kw)
        invalid.add(path.name)
    monkeypatch.setattr(driver.collector, 'l1_run', fail)
    result = driver.execute(plan, freeze, output)
    assert len(calls) == 4
    assert result['g2_status'] == 'blocked_partial_dev45'
    assert not result['g2']


def test_cached_replacements_skip_without_actor(environment):
    _, _, calls, invalid, output = environment
    invalid.clear()
    plan = driver.build_plan()
    freeze = dict(phase_authorized=True, phase_frozen=True, d0_frozen=False, approved_plan_sha256=plan['plan_sha256'], limits=driver.LIMITS,
        internal_plan_reviewed=True, user_authorization_record=dict(exact_text='fixture', source='fixture'))
    result = driver.execute(plan, freeze, output)
    assert all(row['cached'] for row in result['replacements'])
    assert len(calls) == 20


def test_parent_change_rejected_before_actor(environment):
    plan, freeze, calls, _, output = environment
    parent = Path(plan['parents']['officeval_003.r00']['path'])
    (parent / 'manifest.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='parent changed'):
        driver.execute(plan, freeze, output)
    assert not calls


def test_explicit_review_and_authorization_required(environment):
    plan, freeze, calls, _, output = environment
    freeze.pop('internal_plan_reviewed')
    with pytest.raises(ValueError, match='internal plan review'):
        driver.execute(plan, freeze, output)
    assert not calls


def test_method_d0_cannot_be_claimed_frozen(environment):
    plan, freeze, calls, _, output = environment
    freeze['d0_frozen'] = True
    with pytest.raises(ValueError, match='method D0 remains unfrozen'):
        driver.execute(plan, freeze, output)
    assert not calls


def test_source_guard_before_every_http_and_shared_ledger_cap(tmp_path, monkeypatch):
    plan = dict(source_fingerprints={'controller': 'good'})
    monkeypatch.setattr(driver, 'source_fingerprints', lambda: {'controller': 'good'})
    caps = {**driver.LIMITS, 'max_requests': 3}
    admission = driver.Admission(tmp_path / 'ledger.json', 'plan', caps)
    guard = driver.GuardedAdmission(admission, plan)
    def send(_):
        if guard.acquire(10):
            guard.reconcile(10, 1)
            return True
        return False
    with ThreadPoolExecutor(max_workers=3) as pool:
        assert sum(pool.map(send, range(9))) == 3
    assert admission.snapshot()['requests'] == 3
    monkeypatch.setattr(driver, 'source_fingerprints', lambda: {'controller': 'changed'})
    with pytest.raises(ValueError, match='source/config changed'):
        guard.acquire(10)
