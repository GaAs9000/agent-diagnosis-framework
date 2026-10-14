"""Synthetic reproductions of controller identity, eligibility and accounting defects."""
from __future__ import annotations
import concurrent.futures
import copy
import time
import pytest

from test_value_curve_v02 import make_curve_episode, _overwrite_and_reseal
from l2_sandbox.analyst.baselines import extract_terminal_outcome
from l2_sandbox.config import load
from l2_sandbox.evidence.l1 import L1Reader
from l2_sandbox.experiments import controller
from l2_sandbox.experiments.plans import compile_curve_plan
from l2_sandbox.io import L2Error, Unsupported, read_json, read_jsonl
from l2_sandbox.package import verify
from l2_sandbox.pipeline import analyze, resolve_family_id


def setup_curve(tmp_path, **kwargs):
    source = make_curve_episode(tmp_path / 'parent', num_turns=6, **kwargs)
    analysis = tmp_path / 'analysis'
    analyzed = analyze(source, analysis, load())
    assert analyzed['verification']['verified'], analyzed['verification']
    plan = tmp_path / 'plan'
    result = compile_curve_plan(analysis, source, plan, stride=2, stage1_replicates=1,
                                stage2_replicates=1, max_concurrency=2, max_replacements=0)
    return source, analysis, plan, result['plan']


def branch_runner(source):
    reader = L1Reader(source)
    convs = {k: reader.json(f'checkpoints/cp{k:06d}.json')['conversation'] for k in range(6)}
    def runner(cfg, out, *, parent, checkpoint_id):
        assert cfg == reader.json('raw/resolved_config.json')
        make_curve_episode(out, num_turns=6, resume_from_turn=int(checkpoint_id[2:]),
            parent_conversations=convs, first_turn_behavior='readback', lineage={
                'parent_episode_id': reader.episode_id,
                'parent_index_sha256': reader.index_sha256,
                'parent_checkpoint_id': checkpoint_id,
                'parent_checkpoint_sha256': reader.files[f'checkpoints/{checkpoint_id}.json']['sha256']})
    return runner


def test_changed_stats_policy_rejects_old_frozen_plan_before_execution(tmp_path):
    source, _, plan, frozen = setup_curve(tmp_path)
    frozen['stats_policy_version'] = '0.2.0'
    _overwrite_and_reseal(plan, 'plan.json', frozen, 'value_curve_plan')
    assert verify(plan)['verified']  # The old plan still has valid integrity.
    output = tmp_path / 'execution'
    with pytest.raises(Unsupported, match='stats_policy_version'):
        controller.execute_curve(plan, source, output, allow_execution=True,
            _test_runner=lambda cfg, out, *, parent, checkpoint_id: pytest.fail('old policy must not execute'))
    assert not output.exists()


def test_nonzero_checkpoint_labels_first_new_request_not_later_local_turn(tmp_path):
    source, analysis, plan, _ = setup_curve(tmp_path)
    execution = tmp_path / 'execution'
    result = controller.execute_curve(plan, source, execution, allow_execution=True,
                                       _test_runner=branch_runner(source))
    assert result['verification']['verified'], result['verification']
    rows = [r for r in read_jsonl(execution / 'branch_receipts.jsonl') if r['checkpoint_id'] == 'cp000002']
    assert rows
    for row in rows:
        assert row['first_turn_index'] == 3
        assert 'submission_readback' in row['first_turn_tags']
        assert 'assistant_finished_turn' not in row['first_turn_tags']
        branch = L1Reader(execution / row['branch_path'])
        first_id = next(e['data']['request_id'] for e in branch.events if e['kind'] == 'model.request')
        assert row['first_request_id'] == first_id == row['first_request_identity']['branch_request_id']
    # Change tags, receipt journal and aggregates together: actual branch evidence still rejects it.
    rows = read_jsonl(execution / 'branch_receipts.jsonl')
    rows[0]['first_turn_tags'] = ['assistant_finished_turn']
    trace = read_jsonl(execution / 'experiment_trace.jsonl')
    from l2_sandbox.io import digest, write_jsonl
    for event in trace:
        if event['kind'] == 'branch.receipt' and event['data']['trial_id'] == rows[0]['trial_id']:
            event['data']['first_turn_tags'] = rows[0]['first_turn_tags']
    previous = None
    for event in trace:
        event['previous_sha256'] = previous
        event.pop('sha256')
        event['sha256'] = digest(event)
        previous = event['sha256']
    (execution / 'experiment_trace.jsonl').unlink()
    write_jsonl(execution, 'experiment_trace.jsonl', trace)
    _overwrite_and_reseal(execution, 'branch_receipts.jsonl', rows, 'value_curve_execution', is_jsonl=True)
    verification = verify(execution)
    assert not verification['verified']
    assert 'first new request/tag binding' in str(verification['errors'])


@pytest.mark.parametrize('case', ['transport_invalid', 'scorer_crash'])
def test_undefined_parent_blocks_plan_and_retains_none(tmp_path, case):
    source, _, plan, frozen = setup_curve(tmp_path, terminal_case=case)
    assert extract_terminal_outcome(L1Reader(source)) is None
    assert frozen['terminal_value'] is None
    assert frozen['grid']['terminal'] != 'exact_parent_grade'
    assert frozen['plan_state'] == 'blocked' and 'parent_endpoint_undefined' in frozen['blockers']
    assert verify(plan)['verified']
    baselines = read_json(plan / 'baselines.json')
    assert baselines['linear_decay']['value_predictions'] == {}
    # A resealed package must not turn an unknown parent into a frozen exact endpoint.
    forged = copy.deepcopy(frozen)
    forged.update(plan_state='frozen', blockers=[])
    _overwrite_and_reseal(plan, 'plan.json', forged, 'value_curve_plan')
    assert not verify(plan)['verified']
    with pytest.raises(L2Error):
        controller.execute_curve(plan, source, tmp_path / 'exec', allow_execution=True,
                                  _test_runner=lambda *a, **k: pytest.fail('must not execute'))
    # Even a fabricated zero in a resealed plan cannot override the source's unknown grade.
    forged.update(terminal_value=0.0)
    forged['grid']['terminal'] = 'exact_parent_grade'
    _overwrite_and_reseal(plan, 'plan.json', forged, 'value_curve_plan')
    with pytest.raises(Unsupported, match='parent_endpoint_undefined'):
        controller.execute_curve(plan, source, tmp_path / 'exec_zero', allow_execution=True,
                                  _test_runner=branch_runner(source))
    assert not (tmp_path / 'exec_zero').exists()


def test_empty_submission_is_valid_zero_parent(tmp_path):
    source, _, _, plan = setup_curve(tmp_path, terminal_case='empty_submission')
    assert extract_terminal_outcome(L1Reader(source)) == 0.0
    assert plan['terminal_value'] == 0.0 and plan['plan_state'] == 'frozen'


def test_interior_checkpoint_ineligible_blocks_before_stage1(tmp_path):
    source, _, plan, frozen = setup_curve(tmp_path, ineligible_cp='cp000001')
    assert 'cp000001' not in frozen['grid']['checkpoints']
    assert frozen['plan_state'] == 'blocked' and 'stage2_checkpoint_ineligible' in frozen['blockers']
    assert verify(plan)['verified']
    assert read_json(plan / 'grid_qualifications.json')['cp000001']['eligible'] is False
    with pytest.raises(Unsupported):
        controller.execute_curve(plan, source, tmp_path / 'exec', allow_execution=True,
                                  _test_runner=lambda *a, **k: pytest.fail('must not execute'))


def test_same_task_sibling_reports_share_family_and_legacy_retained_identity(tmp_path):
    source, analysis, plan, _ = setup_curve(tmp_path)
    sibling = tmp_path / 'sibling'
    analyze(source, sibling, load())
    first, second = read_json(analysis / 'manifest.json'), read_json(sibling / 'manifest.json')
    assert first['analysis_id'] != second['analysis_id']
    assert first['task']['family_id'] == second['task']['family_id'] == 'officeval:officeval_003'
    execution = tmp_path / 'execution'
    controller.execute_curve(plan, source, execution, allow_execution=True, _test_runner=branch_runner(source))
    assert controller.report_curve(analysis, execution, tmp_path / 'report')['verified']
    assert read_json(tmp_path / 'report/value_evaluation.json')['family_id'] == 'officeval:officeval_003'
    sibling_plan = tmp_path / 'sibling_plan'
    compile_curve_plan(sibling, source, sibling_plan, stride=2, stage1_replicates=1, stage2_replicates=1, max_replacements=0)
    sibling_execution = tmp_path / 'sibling_execution'
    controller.execute_curve(sibling_plan, source, sibling_execution, allow_execution=True, _test_runner=branch_runner(source))
    assert controller.report_curve(sibling, sibling_execution, tmp_path / 'sibling_report')['verified']
    assert read_json(tmp_path / 'sibling_report/value_evaluation.json')['family_id'] == 'officeval:officeval_003'
    first.pop('task')
    _overwrite_and_reseal(analysis, 'manifest.json', first, 'diagnosis')
    assert resolve_family_id(first, read_jsonl(analysis / 'facts.jsonl')) == 'officeval:officeval_003'
    assert resolve_family_id({'task': {'family_id': 'explicit-family'}}, []) == 'explicit-family'
    with pytest.raises(L2Error, match='identity unavailable'):
        resolve_family_id({'analysis_id': 'tempting_but_wrong'}, [])


def test_request_hook_accounts_multiple_sends_and_429_unknown_usage(tmp_path, monkeypatch):
    source, _, plan, _ = setup_curve(tmp_path)
    base_runner = branch_runner(source)
    limiters, attempts = [], []
    def runner(cfg, out, *, parent, checkpoint_id, request_limiter):
        limiters.append(request_limiter)
        for status, usage in [(200, 10), (429, None), (200, 20)]:
            assert request_limiter.acquire(100, timeout=1)
            attempts.append(status)
            request_limiter.reconcile(100, usage)
        base_runner(cfg, out, parent=parent, checkpoint_id=checkpoint_id)
    result = controller.execute_curve(plan, source, tmp_path / 'exec', allow_execution=True,
        rpm=10000, tpm=100000, _test_runner=runner)
    assert result['verification']['verified']
    receipts = read_jsonl(tmp_path / 'exec/branch_receipts.jsonl')
    assert len(attempts) == 3 * len(receipts) and attempts.count(429) == len(receipts)
    assert len({id(limiter) for limiter in limiters}) == 1
    # Known usage refunds 90+80; the unknown 429 keeps its 100 reservation.
    limiter = controller.TokenBucketRateLimiter(tpm=1000)
    monkeypatch.setattr(controller.time, 'monotonic', lambda: 0.0)
    limiter._last = 0.0
    for usage in (10, None, 20):
        assert limiter.acquire(100, timeout=0)
        limiter.reconcile(100, usage)
    assert limiter._tok_tokens == 870


def test_requested_rate_limit_requires_hook_legacy_runner_and_concurrency_bound(tmp_path):
    source, _, plan, _ = setup_curve(tmp_path)
    runner = branch_runner(source)
    with pytest.raises(Unsupported, match='request_limiter hook'):
        controller.execute_curve(plan, source, tmp_path / 'missing_hook', allow_execution=True, rpm=10, _test_runner=runner)
    assert not (tmp_path / 'missing_hook').exists()
    with pytest.raises(L2Error, match='only lower'):
        controller.execute_curve(plan, source, tmp_path / 'too_wide', allow_execution=True, max_concurrency=3, _test_runner=runner)
    assert controller.execute_curve(plan, source, tmp_path / 'legacy', allow_execution=True,
                                    max_concurrency=1, _test_runner=runner)['verification']['verified']


def test_limiter_oversized_timeout_and_shared_reservations():
    limiter = controller.TokenBucketRateLimiter(rpm=1, tpm=100)
    with pytest.raises(L2Error, match='exceeds tpm'):
        limiter.acquire(101, timeout=0)
    assert limiter.acquire(100, timeout=0)
    limiter.reconcile(100, None)
    start = time.monotonic()
    assert not limiter.acquire(1, timeout=0.01)
    assert time.monotonic() - start < 0.1
    limiter = controller.TokenBucketRateLimiter(tpm=100)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        admitted = list(pool.map(lambda _: limiter.acquire(25, timeout=0), range(8)))
    assert sum(admitted) == 4


def test_dim2_is_descriptive_and_uses_real_terminal_hits():
    plan = {'num_turns': 2, 'terminal_value': 1.0, 'terminal_dim2_hits': {'item': 1.0},
        'inference': {'alpha': 0.05}, 'grid': {'checkpoints': ['cp000000']},
        'endpoint': {'name': 'official_completion_rate'}, 'stats_policy_version': 'fixture',
        'strata': {'kinds': [], 'min_per_class': 4}}
    def row(stage, cp):
        return {'stage': stage, 'checkpoint_id': cp, 'trial_id': stage+cp,
                'outcome': {'endpoint_defined': True, 'value': 1.0, 'dim2_hits': {'item': 1.0}},
                'first_turn_tags': []}
    receipts = [row('stage1', 'cp000000'), row('stage2', 'cp000000'), row('stage2', 'cp000001')]
    result = controller.summarize_curve(receipts, plan, {'selected_interval': [0, 2],
        'checkpoints': ['cp000000', 'cp000001']}, fixture=True)
    item = result['exploratory_dim2_item_curves']['item']
    assert item['mean_start'] == item['mean_end'] == 1.0 and item['drop'] == 0.0
    assert item['p_value_holm'] is None and item['inference_status'] == 'not_tested'
    del plan['terminal_dim2_hits']
    unknown = controller.summarize_curve(receipts, plan, {'selected_interval': [0, 2],
        'checkpoints': ['cp000000', 'cp000001']}, fixture=True)['exploratory_dim2_item_curves']['item']
    assert unknown['mean_end'] is None and unknown['drop'] is None


@pytest.mark.parametrize('budget_key,cap', [('max_model_requests', 1), ('max_tool_calls', 1),
                                         ('wall_time_seconds', 1), ('max_total_tokens', 10)])
def test_all_interior_remaining_budgets_are_checked(tmp_path, budget_key, cap):
    """A sealed source can advertise eligible checkpoints despite an exhausted budget."""
    from l2_sandbox.io import digest, inventory, sha_file, write_json, write_jsonl
    source = make_curve_episode(tmp_path / 'source', num_turns=6)
    cfg = read_json(source / 'raw/resolved_config.json')
    cfg['budget'][budget_key] = cap
    def replace(path, data):
        (source / path).unlink()
        write_json(source, path, data)
    replace('raw/resolved_config.json', cfg)
    for k in range(6):
        path = f'checkpoints/cp{k:06d}.json'
        cp = read_json(source / path)
        cp['config_sha256'] = digest(cfg)
        replace(path, cp)
    manifest = read_json(source / 'manifest.json')
    manifest['config_sha256'] = digest(cfg)
    replace('manifest.json', manifest)
    events = read_jsonl(source / 'trajectory.jsonl')
    def refresh_refs(value):
        if isinstance(value, dict):
            if {'path', 'sha256', 'size_bytes'} <= value.keys():
                path = source / value['path']
                value.update(sha256=sha_file(path), size_bytes=path.stat().st_size)
            for child in value.values(): refresh_refs(child)
        elif isinstance(value, list):
            for child in value: refresh_refs(child)
    previous = None
    for event in events:
        refresh_refs(event['data'])
        event['previous_sha256'] = previous
        event.pop('sha256')
        event['sha256'] = digest(event)
        previous = event['sha256']
    (source / 'trajectory.jsonl').unlink()
    write_jsonl(source, 'trajectory.jsonl', events)
    index = read_json(source / 'evidence_index.json')
    (source / 'evidence_index.json').unlink()
    index.update(files=inventory(source), trajectory_head_sha256=events[-1]['sha256'])
    write_json(source, 'evidence_index.json', index)
    assert L1Reader(source).report['verified']
    analysis = tmp_path / 'analysis'
    assert analyze(source, analysis, load())['verification']['verified']
    compiled = compile_curve_plan(analysis, source, tmp_path / 'plan', stride=2,
                                  stage1_replicates=1, stage2_replicates=1)['plan']
    assert compiled['plan_state'] == 'blocked'
    assert 'stage2_checkpoint_ineligible' in compiled['blockers']
    qualifications = read_json(tmp_path / 'plan/grid_qualifications.json')
    assert qualifications['cp000001']['eligible'] is False
    assert any(v == 0 for v in qualifications['cp000001']['remaining_budget'].values())
