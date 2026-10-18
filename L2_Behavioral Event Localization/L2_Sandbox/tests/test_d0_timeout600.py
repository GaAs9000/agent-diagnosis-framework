"""T2 offline contracts. Execute on aml-b CPU; no Actor calls."""
import copy
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/resume_d0_timeout600.py'
spec = importlib.util.spec_from_file_location('test_timeout600_driver', SCRIPT)
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


@pytest.fixture
def environment(tmp_path, monkeypatch):
    monkeypatch.setattr(d, 'assert_sources', lambda _: None)
    monkeypatch.setattr(d, 'validate_protocol', lambda _: None)
    monkeypatch.setattr(d, 'validate_pair', lambda *_: {'valid': True})
    monkeypatch.setattr(d.time, 'time', lambda: d.STOP - 10000)
    config = d.c.resolve_config(d.c.CONFIG_PATH)
    old_root = tmp_path / 'original'
    parents, configs = {}, {}
    calls = []
    invalid = {old_root / 'episodes' / d.LABEL}

    def validity(path):
        return path.exists() and path not in invalid, dict(episode_id=path.name.removesuffix('_c0'), index_sha256='index')
    monkeypatch.setattr(d.c, 'is_valid_episode', validity)
    for rep in d.c.REPLICATES:
        for task in d.c.DEV_TASKS:
            label = f'{task}.{rep}'
            cfg = copy.deepcopy(config); cfg['benchmark']['task_id'] = task
            path = old_root / 'episodes' / label
            for name, value in [('manifest.json', {}), ('evidence_index.json', {}),
                ('raw/resolved_config.json', cfg), ('checkpoints/cp000000.json', {}),
                ('raw/model/r000000/request.json', dict(first=True))]:
                d.write_json(path / name, value)
            configs[label] = cfg; parents[label] = d.s.identity(path)
    revised = copy.deepcopy(config); revised['model']['timeout_seconds'] = 600
    limits = {**d.p.LIMITS, 'max_requests': 2940, 'max_tokens': 2940*d.p.GIB,
        'max_wall_seconds': 15600, 'max_invalid_attempts': 21}
    plan = d.seal(dict(execution_ready=True, execution_concurrency=3, replacement_concurrency=1,
        g2_concurrency=3, d0_frozen=False, replacements=[d.LABEL], attempt_limit_per_slot=1,
        g2_checkpoint='cp000000', g2_slots=[f'{t}.{r}' for t, r in d.c.G2_SLOTS],
        timeout_seconds=600, request_safety_seconds=45, service_stop_epoch=d.STOP,
        old_http_attempts=134, old_active_seconds=6000, aggregate_http_cap=3360,
        limits=limits, pricing_mode='local_no_provider_charge', actor_identity=d.p.actor_identity(config),
        frozen_base_config=revised, frozen_parent_configs=configs, parents=parents,
        original_run_root=str(old_root), original_tree_sha256=d.s.tree_digest(old_root),
        run_root=str(tmp_path / 't2'), protocol={}, derivative=d.derivative_record()), 'plan_sha256')
    freeze = dict(phase_authorized=True, phase_frozen=True, d0_frozen=False,
        internal_plan_reviewed=True, approved_plan_sha256=plan['plan_sha256'], limits=limits,
        user_authorization_record=dict(exact_text='offline fixture', source='test'))

    def fake_run(cfg, output, **kwargs):
        calls.append((cfg, output, kwargs))
        assert cfg['model']['timeout_seconds'] == 600
        limiter = kwargs['request_limiter']
        assert limiter.acquire(1, timeout=700)
        limiter.reconcile(1, 1)
        lineage = {}
        if 'parent' in kwargs:
            assert kwargs['checkpoint_id'] == 'cp000000'
            parent = kwargs['parent']
            lineage = dict(parent_episode_id=parent.name, parent_index_sha256='index',
                parent_checkpoint_id='cp000000', parent_checkpoint_sha256=d.file_digest(parent / 'checkpoints/cp000000.json'))
        for name, value in [('manifest.json', dict(lineage=lineage)), ('evidence_index.json', {}),
            ('raw/resolved_config.json', cfg), ('checkpoints/cp000000.json', {}),
            ('raw/model/r000000/request.json', dict(first=True))]:
            d.write_json(output / name, value)
    monkeypatch.setattr(d.c, 'l1_run', fake_run)
    monkeypatch.setattr(d, 'derivative_runner', lambda plan, output: (
        d.write_json(output / 'GENERATED_SOURCE.py', {}),
        d.write_json(output / 'source_transform.json', {}), fake_run)[-1])
    original_guard = d.GuardedAdmission.__init__
    monkeypatch.setattr(d.GuardedAdmission, '__init__', lambda self, admission, plan, metadata, output=None:
        original_guard(self, admission, plan, metadata, output, metrics_reader=lambda: {'running': 0, 'waiting': 0}))
    return plan, freeze, calls, invalid, tmp_path / 'receipts'


def test_independent_repair_then_original_twenty_and_gate_undecided(environment):
    plan, freeze, calls, _, output = environment
    before = d.s.tree_digest(Path(plan['original_run_root']))
    result = d.execute(plan, freeze, output)
    assert len(calls) == 21
    assert calls[0][1] == Path(plan['run_root']) / 'episodes' / d.LABEL
    assert result['g2_status'] == 'completed' and result['g2_gate'] == 'undecided'
    assert result['aggregate_http_attempts'] == 155
    assert d.s.tree_digest(Path(plan['original_run_root'])) == before
    assert len(list(output.glob('telemetry/http_*.json'))) == 21
    for row in result['g2']:
        assert row['byte_identical_first_request'] and row['cold_parent_binding_valid']
        assert row['continuation_timeout_seconds'] == 600
        assert row['parent_timeout_seconds'] in (300, 600)
    assert d.execute(plan, freeze, output) == result
    assert len(calls) == 21


def test_failed_new_parent_blocks_g2_preserves_original(environment, monkeypatch):
    plan, freeze, calls, invalid, output = environment
    invalid.add(Path(plan['run_root']) / 'episodes' / d.LABEL)
    result = d.execute(plan, freeze, output)
    assert len(calls) == 1 and result['g2'] == []
    assert not result['all_45_study_valid']
    assert Path(plan['original_run_root'], 'episodes', d.LABEL).exists()


def test_original_invalid_parent_is_also_protected(environment):
    plan, freeze, calls, _, output = environment
    Path(plan['original_run_root'], 'episodes', d.LABEL, 'extra').write_text('changed')
    with pytest.raises(ValueError, match='parent evidence changed'):
        d.execute(plan, freeze, output)
    assert not calls


def test_interrupted_marker_never_restarts(environment):
    plan, freeze, calls, _, output = environment
    d.write_json(output / 'attempt_started.json', {})
    with pytest.raises(ValueError, match='automatic restart forbidden'):
        d.execute(plan, freeze, output)
    assert not calls


def test_short_service_window_refuses_before_attempt(environment, monkeypatch):
    plan, freeze, calls, _, output = environment
    monkeypatch.setattr(d.time, 'time', lambda: d.STOP - 644)
    with pytest.raises(ValueError, match='insufficient existing service'):
        d.execute(plan, freeze, output)
    assert not calls and not (output / 'attempt_started.json').exists()


def test_every_retry_requires_full_window(environment, monkeypatch):
    plan, _, _, _, output = environment
    admission = d.p.Admission(output / 'admission.json', plan['plan_sha256'], plan['limits'])
    guard = d.GuardedAdmission(admission, plan, {})
    assert guard.acquire(1, timeout=700)
    guard.reconcile(1, None)
    with pytest.raises(RuntimeError, match='safety window'):
        guard.acquire(1, timeout=644)
    assert admission.snapshot()['requests'] == 1


def test_delta_helper_allows_only_timeout():
    old = {'model': {'timeout_seconds': 300, 'parameters': {'temperature': 1}}}
    new = copy.deepcopy(old); new['model']['timeout_seconds'] = 600
    assert d._t2_resume_configs_match(new, old)
    assert d._t2_resume_configs_match(new, new)
    new['model']['parameters']['temperature'] = 0
    assert not d._t2_resume_configs_match(new, old)
    old['model']['timeout_seconds'] = 400
    assert not d._t2_resume_configs_match(new, old)


def test_generated_source_is_explicit_single_guard_revision(tmp_path):
    record = d.derivative_record()
    runner = d.derivative_runner({'derivative': record}, tmp_path)
    original = Path(record['original_path']).read_text()
    generated = (tmp_path / 'GENERATED_SOURCE.py').read_text()
    assert generated.replace(record['new_guard'], record['old_guard']) == original
    assert d.file_digest(tmp_path / 'GENERATED_SOURCE.py') == record['generated_sha256']
    with pytest.raises(ValueError, match='cp000000'):
        runner({}, tmp_path / 'run', parent=tmp_path, checkpoint_id='cp000001')


def test_real_sealed_parent_passes_only_t2_guard_without_model_send(tmp_path, monkeypatch):
    """Exercise the real guard on the server mirror, stopping before components."""
    import inspect
    from c3_sandbox.l1.core.contracts import ConfigurationError
    from c3_sandbox.l1.models.native import OpenAICompatibleBackend

    parent = d.c.OUT_ROOT / 'episodes/officeval_003.r00'
    if not (parent / 'manifest.json').is_file():
        pytest.skip('real sealed DEV parent not installed on this test host')
    cfg = d.read_json(parent / 'raw/resolved_config.json')
    cfg['model']['timeout_seconds'] = 600
    runner = d.derivative_runner({'derivative': d.derivative_record()}, tmp_path)
    module = inspect.getclosurevars(runner).nonlocals['module']

    class ReachedJournal(Exception):
        pass

    def stop_before_components(*args, **kwargs):
        raise ReachedJournal('real config guard accepted; no components or model created')

    def forbidden_send(*args, **kwargs):
        raise AssertionError('model send forbidden during CPU preflight')

    monkeypatch.setattr(module, 'Journal', stop_before_components)
    monkeypatch.setattr(OpenAICompatibleBackend, 'send', forbidden_send)
    with pytest.raises(ConfigurationError, match='same resolved config'):
        d.c.l1_run(cfg, tmp_path / 'old_reject', parent=parent, checkpoint_id='cp000000')
    with pytest.raises(ReachedJournal, match='real config guard accepted'):
        runner(cfg, tmp_path / 'new_accept', parent=parent, checkpoint_id='cp000000')
    cfg['model']['parameters']['temperature'] = 0
    with pytest.raises(ConfigurationError, match='same resolved config'):
        runner(cfg, tmp_path / 'other_delta_reject', parent=parent, checkpoint_id='cp000000')
