"""Bounded R22 contracts. Run on the server CPU; all transports are fakes."""
import copy
import importlib.util
from pathlib import Path
import threading

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/run_r22_native_alignment.py'
spec = importlib.util.spec_from_file_location('r22_driver_tests', SCRIPT)
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(d, 'L1', tmp_path / 'l1')
    monkeypatch.setattr(d.c, 'OUT_ROOT', d.L1 / 'runs/v3_qwen36_pi_dev45_20261017')
    monkeypatch.setattr(d, 'assert_sources', lambda _: None)
    cfg = d.c.resolve_config(d.c.CONFIG_PATH)
    origins, preserved, demoted = {}, {}, {}
    calls, invalid = [], set()

    def validity(path):
        valid = path.exists() and path not in invalid
        return valid, dict(episode_id=d.read_json(path / 'manifest.json')['episode_id'] if path.exists() else None,
                           index_sha256=d.file_digest(path / 'evidence_index.json') if path.exists() else None)
    monkeypatch.setattr(d.c, 'is_valid_episode', validity)

    def write_package(path, config, eid, used=0, lineage=None, cp=None):
        cp = cp or dict(conversation={'messages': []}, pending_calls=[], pending_observations=[], qualification={'eligible': True},
                        snapshot_id='s000000', budget_used=dict(model_requests=0, tool_calls=0, known_total_tokens=0))
        for name, value in [('manifest.json', dict(episode_id=eid, lineage=lineage, usage={'model_requests_including_parent': used})),
                            ('evidence_index.json', {'identity': eid}), ('checkpoints/cp000000.json', cp),
                            ('raw/resolved_config.json', config), ('benchmark/public_task.json', {'task_id': config['benchmark']['task_id']}),
                            ('states/snapshots/s000000.json', dict(content_sha256='workspace', files=[], directories=[], scope='workspace', total_bytes=0)),
                            ('raw/model/r000000/request.json', {'actual': 'first wire request'})]:
            d.write_json(path / name, value)

    for label in d.SLOTS:
        selected = d.SELECTED.get(label)
        path = d.L1 / selected[0] if selected else d.c.OUT_ROOT / 'episodes' / label
        eid = selected[1] if selected else 'ep_' + label
        old = copy.deepcopy(cfg); old['benchmark']['task_id'] = label.rsplit('.', 1)[0]
        write_package(path, old, eid)
        origin = dict(path=str(path), episode_id=eid, index_sha256=d.file_digest(path / 'evidence_index.json'), source_config=old,
                      parent_timeout_seconds=300, cp0=dict(checkpoint_id='cp000000', checkpoint_sha256=d.file_digest(path / 'checkpoints/cp000000.json')))
        if selected:
            _, _, k, remaining = selected
            d.write_json(path / 'checkpoints' / f'cp{k:06d}.json', {'checkpoint': k})
            origin.update(continuation_checkpoint=dict(checkpoint_id=f'cp{k:06d}', checkpoint_sha256=d.file_digest(path / 'checkpoints' / f'cp{k:06d}.json')),
                          resume_response_id=f'r{k:06d}', max_new_logical=remaining, response_sha256='saved-response')
            invalid.add(path)
        origins[label] = origin; preserved[label] = d.s.identity(path)
        if label in d.DEMOTED:
            control = d.c.OUT_ROOT / 'episodes' / label
            write_package(control, old, 'demoted_' + label)
            demoted[label] = d.s.identity(control)
    roots = {d.c.OUT_ROOT, d.L1 / 'runs/v3_qwen36_pi_dev45_t2_20261018'}
    plan = d.seal(dict(execution_ready=True, selected=list(d.SELECTED), origins=origins, preserved=preserved, demoted=demoted,
        g2_slots=d.G2_SLOTS, http_caps=d.HTTP_CAPS, logical_caps=d.LOGICAL_CAPS, old_http_attempts=144,
        aggregate_http_cap=3360, aggregate_authorized_max=3324, max_new_generation=63, wall_seconds=21600,
        continuation_concurrency=1, g2_concurrency=3, attempt_limit_per_slot=1, d0_frozen=False, service_renewal_authorized=False,
        service_stop_epoch=d.time.time()+20000, run_root=str(tmp_path / 'new'),
        immutable_trees={str(path): d.s.tree_digest(path) for path in roots}), 'plan_sha256')
    freeze = dict(approved_plan_sha256=plan['plan_sha256'], phase_authorized=True, phase_frozen=True,
                  internal_plan_reviewed=True, d0_frozen=False, user_authorization_record=dict(exact_text='authorized fixture', source='test'))

    def fake_run(config, output, **kwargs):
        calls.append((config, output, kwargs))
        assert kwargs['allow_pi_native_upgrade'] is True
        assert kwargs['request_limiter'].acquire(1, timeout=700)
        kwargs['request_limiter'].reconcile(1, None)
        parent = kwargs['parent']; old = d.read_json(parent / 'raw/resolved_config.json')
        d.assert_delta(old, config)
        origin = origins[output.name]
        cp = kwargs['checkpoint_id']
        lineage = dict(parent_episode_id=origin['episode_id'], parent_index_sha256=origin['index_sha256'], parent_checkpoint_id=cp,
                       parent_checkpoint_sha256=d.file_digest(parent / 'checkpoints' / (cp+'.json')))
        used = 1
        if 'resume_response_id' in kwargs:
            lineage['response_use'] = dict(source_response_sha256='saved-response', new_http_attempts_during_consume=0)
            used += int(kwargs['resume_response_id'][1:])+1
        write_package(output, config, 'new_' + output.name, used, lineage)
        if 'response_use' in lineage:
            d.write_json(output / 'lineage/response_use.json', lineage['response_use'])
    monkeypatch.setattr(d.c, 'l1_run', fake_run)
    monkeypatch.setattr(d, 'score_record', lambda label, path: dict(label=label, endpoint=str(path), endpoint_index_sha256=d.file_digest(path / 'evidence_index.json'),
                                                                  valid=path not in invalid, score=None if path in invalid else .5))
    return plan, freeze, calls, invalid, tmp_path / 'output'


def test_saved_attempt_chains_then_twenty_real_origin_cp0(env):
    plan, freeze, calls, _, output = env
    result = d.execute(plan, freeze, output)
    assert result['all_45_study_valid'] and result['g2_status'] == 'completed'
    assert result['aggregate_http_attempts'] == 167  # 144 + 3 new sends + 20 cold sends.
    assert len(calls) == 23
    for (_, path, kwargs), label in zip(calls[:3], d.SELECTED):
        assert path.parent.name == 'segments' and kwargs['resume_response_id'] == plan['origins'][label]['resume_response_id']
    cold_calls = {path.name: kwargs for _, path, kwargs in calls[3:]}
    for label, kwargs in cold_calls.items():
        assert kwargs['checkpoint_id'] == 'cp000000' and 'resume_response_id' not in kwargs
        assert str(kwargs['parent']) == plan['origins'][label]['path']
    cohort = d.read_json(output / 'cohort.json')
    assert len(cohort['slots']) == 45
    for label in d.SELECTED:
        chain = cohort['slots'][label]
        assert chain['attempt_id'] == plan['origins'][label]['episode_id']
        assert chain['origin_prefix'] != chain['terminal_endpoint']
        assert chain['continuation_segment'] == chain['terminal_endpoint']
    assert all(row['status']=='nonformal_control' for row in cohort['demoted_controls'].values())
    repaired_pair = next(row for row in result['g2'] if row['label']=='officeval_010.r00')
    assert repaired_pair['origin'] != repaired_pair['terminal_endpoint']
    assert result['g2_gate']=='undecided' and not result['pure_equivalence_eligible']
    d.assert_originals(plan)


def test_invalid_continuation_never_resamples_or_opens_g2(env):
    plan, freeze, calls, invalid, output = env
    invalid.add(Path(plan['run_root']) / 'segments' / next(iter(d.SELECTED)))
    result = d.execute(plan, freeze, output)
    assert len(calls)==3 and result['g2']==[] and not result['all_45_study_valid']
    assert len(list(output.glob('receipt_continuations_*.json')))==3
    assert [path.name for _, path, _ in calls] == list(d.SELECTED)
    cohort = d.read_json(output / 'cohort.json')['slots']
    failed = cohort[next(iter(d.SELECTED))]
    assert failed['status']=='invalid_continuation'
    assert failed['continuation_segment']==failed['terminal_endpoint']
    assert failed['terminal_endpoint']!=failed['origin_prefix']
    assert Path(failed['continuation_receipt']).is_file()
    assert all(cohort[label]['status']=='formal' for label in list(d.SELECTED)[1:])


def test_interrupted_stage_refuses_before_send(env):
    plan, freeze, calls, _, output = env
    d.write_json(output / 'attempt_started.json', {})
    with pytest.raises(ValueError, match='automatic restart forbidden'):
        d.execute(plan, freeze, output)
    assert not calls


def test_mutated_original_blocks_execution(env):
    plan, freeze, calls, _, output = env
    Path(plan['origins']['officeval_004.r02']['path'], 'tampered').write_text('change')
    with pytest.raises(ValueError, match='immutable original'):
        d.execute(plan, freeze, output)
    assert not calls


def test_structural_g2_mismatch_keeps_receipt_and_stops(env, monkeypatch):
    plan, freeze, calls, _, output = env
    original = d.c.l1_run
    def broken(config, path, **kwargs):
        original(config, path, **kwargs)
        if path.parent.name=='g2':
            d.write_json(path / 'raw/model/r000000/request.json', {'changed': True})
    monkeypatch.setattr(d.c, 'l1_run', broken)
    result = d.execute(plan, freeze, output)
    assert result['g2_status']=='incomplete' and result['stop_reason']
    rows = [d.read_json(path) for path in output.glob('receipt_g2_*.json')]
    assert rows and any('first request bytes differ' in row.get('structural_error','') for row in rows)
    assert len(calls) <= 6  # Three active workers maximum after the three continuations.


def test_every_retry_has_source_guard_and_window(env, monkeypatch):
    plan, _, _, _, output = env
    admission = d.p.Admission(output/'ledger.json', plan['plan_sha256'], {**d.p.LIMITS, 'max_requests':2})
    count = []
    monkeypatch.setattr(d, 'assert_sources', lambda _: count.append(1))
    guard = d.GuardedAdmission(admission, plan, {}, d.time.time(), threading.Event())
    assert guard.acquire(1, 700); guard.reconcile(1, None)
    assert guard.acquire(1, 700); guard.reconcile(1, None)
    assert not guard.acquire(1, 700)
    assert len(count)==3 and admission.snapshot()['requests']==2
    with pytest.raises(RuntimeError, match='safety window'):
        guard.acquire(1, 644)
    assert admission.snapshot()['requests']==2


def test_short_service_never_creates_attempt_marker(env, monkeypatch):
    plan, freeze, calls, _, output = env
    monkeypatch.setattr(d.time, 'time', lambda: plan['service_stop_epoch']-644)
    with pytest.raises(ValueError, match='insufficient bounded'):
        d.execute(plan, freeze, output)
    assert not calls and not (output/'attempt_started.json').exists()


def test_config_delta_rejects_sampling_drift(env):
    plan, *_ = env
    old = plan['origins']['officeval_003.r00']['source_config']; new=d.native_config(old)
    d.assert_delta(old, new)
    new['model']['parameters']['temperature']=0
    with pytest.raises(ValueError, match='explicit timeout/native-policy delta'):
        d.assert_delta(old, new)


def test_missing_or_different_audit_is_fatal():
    with pytest.raises(ValueError, match='native audit failed'):
        d.validate_audit({'status':'protocol_halt'})


def test_sensitivity_preserves_invalid_unknown_and_gate():
    good=dict(label='officeval_003.r00', valid=True, score=0)
    unknown=dict(label='officeval_004.r02', valid=False, score=None)
    inputs=dict(primary=[good,unknown], sensitivity_a=[good,good], sensitivity_b=[unknown], g2=[])
    result=d.summarize_cpu(inputs)
    assert result['primary']['mean']==0 and result['primary']['n_invalid_or_unknown']==1
    assert result['sensitivity_b']['mean'] is None
    assert result['g2']['mean_difference'] is None and result['g2']['interval95'] is None
    assert result['g2']['gate']=='undecided' and result['d0_frozen'] is False


def test_audit_conversion_difference_blocks_even_if_status_claims_pass():
    report = dict(status='pass', passed_packages=42, inventory={label:'hash' for label in set(d.SLOTS)-set(d.SELECTED)},
                  native_validation_failures=0, raw_differences=1, legacy_effective_differences=0,
                  sources_unchanged=True, inventory_unchanged=True, native_identity_before={'id':'pinned'}, native_identity_after={'id':'pinned'})
    with pytest.raises(ValueError, match='argument differences'):
        d.validate_audit(report)


def test_g2_interval_uses_original_equal_family_formula():
    def pair(label, parent, child):
        return dict(label=label, comparison_valid=True, endpoint_score={'valid':True,'score':parent},
                    child_score={'valid':True,'score':child})
    rows=[pair('officeval_003.r00',0,1),pair('officeval_003.r01',0,1),pair('officeval_004.r00',1,0)]
    result=d.summarize_cpu(dict(primary=[],sensitivity_a=[],sensitivity_b=[],g2=rows))['g2']
    assert result['mean_difference']==0  # Family means +1 and -1, each weight 1/2.
    assert result['family_weights']=={'officeval_003':.5,'officeval_004':.5}
    assert result['hoeffding_radius']==pytest.approx(d.math.sqrt(2*d.math.log(40)/2))
    assert result['gate']=='undecided'
