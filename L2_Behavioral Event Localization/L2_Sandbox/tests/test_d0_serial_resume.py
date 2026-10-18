"""Offline-only serial continuation contracts; run on the lab CPU test host."""
import copy
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/resume_d0_serial.py'
spec = importlib.util.spec_from_file_location('d0_serial_test', SCRIPT)
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


@pytest.fixture
def environment(tmp_path, monkeypatch):
    c, p = d.c, d.prior
    monkeypatch.setattr(c, 'OUT_ROOT', tmp_path / 'runs')
    monkeypatch.setattr(d, 'assert_sources', lambda _: None)
    monkeypatch.setattr(d.time, 'time', lambda: d.SERVICE_STOP - 10000)
    config = c.resolve_config(c.CONFIG_PATH)
    invalid = set(d.REPLACEMENTS)
    calls = []

    def validity(path):
        label = path.name.removesuffix('_c0')
        return path.exists() and label not in invalid, dict(episode_id=label, index_sha256='index-' + label)

    monkeypatch.setattr(c, 'is_valid_episode', validity)
    parents, configs = {}, {}
    for rep in c.REPLICATES:
        for task in c.DEV_TASKS:
            label = f'{task}.{rep}'
            path = c.OUT_ROOT / 'episodes' / label
            task_config = copy.deepcopy(config)
            task_config['benchmark']['task_id'] = task
            configs[label] = task_config
            for relative, value in [('manifest.json', {}), ('evidence_index.json', {}),
                    ('checkpoints/cp000000.json', {}), ('raw/resolved_config.json', task_config),
                    ('raw/model/r000000/request.json', {})]:
                d.write_json(path / relative, value)
            parents[label] = d.identity(path)

    def run(cfg, output, **kwargs):
        # Each fake transport send must use the same, source-guarded admission.
        calls.append((output.name, kwargs['request_limiter']))
        kwargs['request_limiter'].acquire(1)
        kwargs['request_limiter'].reconcile(1, 1)
        label = output.name.removesuffix('_c0')
        lineage = {}
        if 'parent' in kwargs:
            parent = kwargs['parent']
            assert kwargs['checkpoint_id'] == 'cp000000'
            assert cfg == d.read_json(parent / 'raw/resolved_config.json')
            lineage = dict(parent_episode_id=label, parent_index_sha256='index-' + label,
                parent_checkpoint_id='cp000000', parent_checkpoint_sha256=d.file_digest(parent / 'checkpoints/cp000000.json'))
        for relative, value in [('manifest.json', dict(lineage=lineage)), ('evidence_index.json', {}),
                ('raw/resolved_config.json', cfg), ('raw/model/r000000/request.json', {}),
                ('checkpoints/cp000000.json', {})]:
            d.write_json(output / relative, value)
        invalid.discard(label)

    monkeypatch.setattr(c, 'l1_run', run)
    limits = {**p.LIMITS, 'max_requests': 3080, 'max_tokens': 3080 * p.GIB,
        'max_wall_seconds': d.MAX_NEW_WALL, 'max_invalid_attempts': 22}
    plan = d.seal(dict(execution_ready=True, execution_concurrency=3, d0_frozen=False,
        replacement_concurrency=1, g2_concurrency=3,
        limits=limits, pricing_mode='local_no_provider_charge', actor_identity=p.actor_identity(config),
        replacements=list(d.REPLACEMENTS), old_http_attempts=93, aggregate_http_cap=3360,
        service_stop_epoch=d.SERVICE_STOP, parents=parents,
        frozen_base_config=config, frozen_parent_configs=configs,
        g2_slots=[f'{t}.{r}' for t, r in c.G2_SLOTS]), 'plan_sha256')
    freeze = dict(phase_authorized=True, phase_frozen=True, d0_frozen=False,
        internal_plan_reviewed=True, approved_plan_sha256=plan['plan_sha256'], limits=limits,
        user_authorization_record=dict(exact_text='fixture direct approval', source='offline test'))
    return plan, freeze, calls, invalid, tmp_path / 'output'


def test_serial_two_then_original_twenty_and_idempotent_result(environment):
    plan, freeze, calls, _, output = environment
    result = d.execute(plan, freeze, output)
    assert [name for name, _ in calls[:2]] == list(d.REPLACEMENTS)
    assert sorted(name for name, _ in calls[2:]) == sorted(f'{label}_c0' for label in plan['g2_slots'])
    assert len(calls) == 22
    assert len({id(limiter) for _, limiter in calls}) == 1
    assert result['g2_status'] == 'completed'
    assert result['aggregate_http_attempts'] == 115
    assert len(list(output.glob('archive_*.json'))) == 2
    for path in output.glob('archive_*.json'):
        mapping = d.read_json(path)
        assert d.tree_digest(Path(mapping['archived_path'])) == mapping['tree_sha256']
    assert d.execute(plan, freeze, output) == result
    assert len(calls) == 22


def test_invalid_replacement_blocks_all_g2(environment, monkeypatch):
    plan, freeze, calls, invalid, output = environment
    original = d.c.l1_run
    def fail(cfg, path, **kwargs):
        original(cfg, path, **kwargs)
        invalid.add(path.name)
    monkeypatch.setattr(d.c, 'l1_run', fail)
    result = d.execute(plan, freeze, output)
    assert len(calls) == 2
    assert not result['all_45_study_valid']
    assert result['g2'] == []


def test_interrupted_ledger_forbids_restart(environment):
    plan, freeze, calls, _, output = environment
    d.write_json(output / 'admission.json', {})
    with pytest.raises(ValueError, match='interrupted serial ledger'):
        d.execute(plan, freeze, output)
    assert not calls


def test_protected_parent_change_rejected_before_any_actor(environment):
    plan, freeze, calls, _, output = environment
    valid = next(row for row in plan['parents'].values() if row['package']['valid'])
    Path(valid['package']['path'], 'unexpected.txt').write_text('change')
    with pytest.raises(ValueError, match='parent evidence changed'):
        d.execute(plan, freeze, output)
    assert not calls


def test_concurrency_or_cap_change_rejected(environment):
    plan, freeze, calls, _, output = environment
    plan['replacement_concurrency'] = 3
    plan = d.seal(plan, 'plan_sha256')
    freeze['approved_plan_sha256'] = plan['plan_sha256']
    with pytest.raises(ValueError, match='exact serial workload'):
        d.execute(plan, freeze, output)
    assert not calls


def test_existing_service_deadline_cannot_renew(environment, monkeypatch):
    plan, freeze, calls, _, output = environment
    monkeypatch.setattr(d.time, 'time', lambda: d.SERVICE_STOP + 1)
    with pytest.raises(ValueError, match='service deadline reached'):
        d.execute(plan, freeze, output)
    assert not calls


def test_guard_checks_every_retry_and_aggregate_budget(tmp_path, monkeypatch):
    checked = []
    monkeypatch.setattr(d, 'assert_sources', lambda _: checked.append(True))
    class FakeAdmission:
        def remaining_seconds(self): return 100
        def snapshot(self): return {'requests': 3267}
        def acquire(self, *args): raise AssertionError('cap must stop transport')
    guard = d.GuardedAdmission(FakeAdmission(), dict(service_stop_epoch=d.time.time() + 100,
        old_http_attempts=93, aggregate_http_cap=3360), {})
    for _ in range(2):
        with pytest.raises(RuntimeError, match='aggregate HTTP cap'):
            guard.acquire(1)
    assert len(checked) == 2


def test_original_source_missing_or_changed_rejected(tmp_path, monkeypatch):
    file = tmp_path / 'controller.py'
    file.write_text('old')
    monkeypatch.setattr(d.prior, 'ROOT', tmp_path)
    plan = dict(source_fingerprints={'controller.py': d.file_digest(file)}, predecessor_files={})
    file.write_text('changed')
    with pytest.raises(ValueError, match='execution source changed'):
        d.assert_sources(plan)
