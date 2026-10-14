"""Offline regressions for immutable curve attempts and request-level admission."""
from pathlib import Path
import time
import pytest
from conftest import make_l1_episode
from l2_sandbox.curve import TokenBucketRateLimiter, build_curve_plan, ensure_plan_identity, execute_curve
from l2_sandbox.evidence.l1 import L1Package
from l2_sandbox.io import IntegrityError, Unsupported, read_json, read_jsonl, write_json, sha_file, validate_output_location


def reseal(root):
    """Reseal only generated synthetic fixtures after intentional semantic edits."""
    manifest = read_json(root / 'manifest.json')
    assert manifest['recording_mode'] == 'test_fixture'
    files = [{'path': p.relative_to(root).as_posix(), 'sha256': sha_file(p), 'size_bytes': p.stat().st_size}
             for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'evidence_index.json']
    write_json(root / 'evidence_index.json', {'episode_id': manifest['episode_id'], 'files': files})


def parent_at(path):
    return L1Package(make_l1_episode(path, num_turns=2))


def make_branch(parent, k, path, case='scored'):
    make_l1_episode(path, num_turns=2, resume_from_k=k, terminal_case=case, completion_rate=0.5, parent_pkg=parent)


def run(parent, output, runner, **kwargs):
    plan = build_curve_plan(parent, n1=1, n2=1, stride=2, max_concurrency=1)
    return execute_curve(parent, output, plan, resume_fn=runner, max_concurrency=1, **kwargs)


def test_retry_history_is_immutable_and_rerun_exclusion_identical(tmp_path):
    parent = parent_at(tmp_path / 'source')
    before = parent.content_identity()
    calls = []
    def runner(p, k, path):
        calls.append(path.name)
        make_branch(p, k, path, 'execution_invalid' if '-r' not in path.name else 'scored')
    output = tmp_path / 'output'
    first = run(parent, output, runner)
    files = {p.relative_to(output): p.read_bytes() for p in (output / 'branches').rglob('*') if p.is_file()}
    history = (output / 'outcomes.jsonl').read_bytes()
    count = len(calls)
    second = run(parent, output, runner)
    assert len(calls) == count
    assert (output / 'outcomes.jsonl').read_bytes() == history
    assert first['invalid_branches'] == second['invalid_branches'] > 0
    assert first['total_attempted_branches'] == second['total_attempted_branches']
    assert first['excluded_high_invalid_rate'] == second['excluded_high_invalid_rate'] is True
    assert parent.content_identity() == before
    assert files == {p.relative_to(output): p.read_bytes() for p in (output / 'branches').rglob('*') if p.is_file()}


def test_interrupted_attempt_and_exhaustion_are_retained(tmp_path):
    parent = parent_at(tmp_path / 'source')
    output = tmp_path / 'output'
    plan = build_curve_plan(parent, n1=1, n2=1, stride=2, max_concurrency=1)
    write_json(output / 'plan.json', plan)
    failed = output / 'branches/k000-s1-i00'
    failed.mkdir(parents=True)
    (failed / 'partial.txt').write_text('untouched')
    calls = []
    def broken(p, k, path):
        calls.append(path.name)
        path.mkdir()
        (path / 'partial.txt').write_text('keep')
        raise RuntimeError('transport')
    first = execute_curve(parent, output, plan, resume_fn=broken, max_retries_per_slot=1)
    count = len(calls)
    second = execute_curve(parent, output, plan, resume_fn=broken, max_retries_per_slot=1)
    assert len(calls) == count
    assert count > 0
    assert (failed / 'partial.txt').read_text() == 'untouched'
    assert first['invalid_branches'] == second['invalid_branches'] == count + 1
    assert len(read_jsonl(output / 'outcomes.jsonl')) == count + 1


def test_wrong_parent_plan_or_cached_lineage_rejected_before_write_or_send(tmp_path):
    parent = parent_at(tmp_path / 'source')
    output = tmp_path / 'output'
    run(parent, output, make_branch)
    before = {p.relative_to(output): sha_file(p) for p in output.rglob('*') if p.is_file()}
    def never(*args):
        pytest.fail('must reject before any send')
    other = parent_at(tmp_path / 'other')
    with pytest.raises(IntegrityError):
        run(other, output, never)
    assert before == {p.relative_to(output): sha_file(p) for p in output.rglob('*') if p.is_file()}
    plan = build_curve_plan(parent, n1=2, n2=1, stride=2)
    with pytest.raises(IntegrityError):
        ensure_plan_identity(output, plan)
    branch = next((output / 'branches').glob('*/manifest.json'))
    obj = read_json(branch)
    obj['lineage']['parent_index_sha256'] = 'wrong'
    write_json(branch, obj)
    before = (output / 'outcomes.jsonl').read_bytes()
    with pytest.raises(IntegrityError):
        run(parent, output, never)
    assert (output / 'outcomes.jsonl').read_bytes() == before


@pytest.mark.parametrize('location', ['source', 'source/nested', '.'])
def test_output_overlap_rejected(tmp_path, location):
    parent = parent_at(tmp_path / 'source')
    with pytest.raises(IntegrityError):
        validate_output_location(parent.root, tmp_path / location)


def test_symlink_artifact_rejected(tmp_path):
    parent = parent_at(tmp_path / 'source')
    output = tmp_path / 'output'
    output.mkdir()
    (output / 'plan.json').symlink_to(parent.root / 'manifest.json')
    with pytest.raises(IntegrityError):
        validate_output_location(parent.root, output)


def test_stage2_interior_unqualified_or_exhausted_checkpoint_rejected(tmp_path):
    parent = parent_at(tmp_path / 'source')
    cp = parent.root / 'checkpoints/cp000001.json'
    obj = read_json(cp)
    obj['qualification']['eligible'] = False
    write_json(cp, obj)
    reseal(parent.root)
    with pytest.raises(IntegrityError, match='unqualified'):
        build_curve_plan(parent, stride=2)
    obj['qualification']['eligible'] = True
    obj['budget_used']['model_requests'] = 35
    write_json(cp, obj)
    reseal(parent.root)
    with pytest.raises(IntegrityError, match='remaining'):
        build_curve_plan(parent, stride=2)


def test_optional_limits_must_reach_request_hooks(tmp_path):
    parent = parent_at(tmp_path / 'source')
    with pytest.raises(Unsupported):
        run(parent, tmp_path / 'nohooks', make_branch, rpm=60)
    seen = []
    def runner(p, k, path, *, request_limiter):
        # Model adapter retries use the same limiter, with one reconciliation per HTTP attempt.
        seen.append(request_limiter)
        for actual in (None, 12, 0):
            assert request_limiter.acquire(10, timeout=0.1)
            request_limiter.reconcile(10, actual)
        make_branch(p, k, path)
    run(parent, tmp_path / 'hooks', runner, rpm=100, tpm=1000)
    assert len(seen) > 1 and all(x is seen[0] for x in seen)


def test_admission_deadline_oversized_and_actual_usage_debt():
    limiter = TokenBucketRateLimiter(rpm=60, tpm=20)
    start = time.monotonic()
    assert not limiter.acquire(21, timeout=10)
    assert time.monotonic() - start < 0.1
    assert limiter.acquire(10, timeout=0)
    limiter.reconcile(10, 30)
    assert limiter._tok_tokens == -10
    assert not limiter.acquire(1, timeout=0.01)
    limiter.reconcile(0, None)
    assert limiter._tok_tokens < 0


@pytest.mark.parametrize('tamper', ['change', 'add', 'remove'])
def test_declared_l1_inventory_verified_before_read_or_send(tmp_path, tamper):
    parent = parent_at(tmp_path / 'source')
    target = parent.root / 'official_grade.json'
    if tamper == 'change':
        target.write_text('{}')
    elif tamper == 'add':
        (parent.root / 'extra.txt').write_text('unexpected')
    else:
        target.unlink()
    with pytest.raises(IntegrityError, match='inventory'):
        L1Package(parent.root)



def test_successful_higher_retry_reused_with_lower_retry_budget(tmp_path):
    parent = parent_at(tmp_path / 'source')
    output = tmp_path / 'output'
    plan = build_curve_plan(parent, n1=1, n2=1, stride=2, max_concurrency=1)
    write_json(output / 'plan.json', plan)
    for attempt in range(4):
        name = 'k000-s1-i00' + (f'-r{attempt}' if attempt else '')
        make_branch(parent, 0, output / 'branches' / name, 'scored' if attempt == 3 else 'execution_invalid')
    calls = []
    def runner(p, k, path):
        calls.append(path.name)
        make_branch(p, k, path)
    curve = execute_curve(parent, output, plan, resume_fn=runner, max_retries_per_slot=0)
    assert not any('-s1-' in name for name in calls)
    assert curve['invalid_branches'] == 3
    rows = read_jsonl(output / 'outcomes.jsonl')
    assert sum(row['stage'] == 's1' for row in rows) == 4


def test_parent_or_checkpoint_config_change_invalidates_plan(tmp_path):
    parent = parent_at(tmp_path / 'source')
    plan = build_curve_plan(parent, n1=1, n2=1, stride=2)
    output = tmp_path / 'output'
    write_json(output / 'plan.json', plan)
    grade = parent.root / 'official_grade.json'
    obj = read_json(grade)
    obj['extra'] = 'changed'
    write_json(grade, obj)
    reseal(parent.root)
    with pytest.raises(IntegrityError, match='no longer matches'):
        execute_curve(parent, output, plan, resume_fn=make_branch)
    assert not (output / 'branches').exists()



def test_default_l1_bridge_forwards_shared_http_limiter(tmp_path, monkeypatch):
    import sys
    from types import ModuleType
    parent = parent_at(tmp_path / 'source')
    seen = []
    fake = ModuleType('c3_sandbox.l1.core.orchestrator')
    def l1_run(cfg, path, *, parent, checkpoint_id, request_limiter):
        pkg = L1Package(parent)
        seen.append(request_limiter)
        assert request_limiter.acquire(1, timeout=0.1)
        request_limiter.reconcile(1, 2)
        make_branch(pkg, int(checkpoint_id[2:]), path)
    fake.run = l1_run
    monkeypatch.setitem(sys.modules, 'c3_sandbox.l1.core.orchestrator', fake)
    plan = build_curve_plan(parent, n1=1, n2=1, stride=2, max_concurrency=1)
    execute_curve(parent, tmp_path / 'output', plan, rpm=100, tpm=100)
    assert len(seen) > 1 and all(limiter is seen[0] for limiter in seen)


@pytest.mark.parametrize('parameters', [{}, {'temperature': 0}, {'temperature': True},
    {'temperature': 1.0, 'seed': 0}, {'temperature': 1.0, 'random_seed': 3},
    {'temperature': 1.0, 'randomSeed': 4}])
def test_plan_rejects_nonsampling_actor_policy_before_writes(tmp_path, parameters):
    from l2_sandbox.io import L2Error
    parent = L1Package(make_l1_episode(tmp_path / 'source', num_turns=2, model_parameters=parameters))
    before = parent.content_identity()
    with pytest.raises(L2Error, match='temperature=1.0'):
        build_curve_plan(parent)
    assert parent.content_identity() == before
    assert not (tmp_path / 'output').exists()


def test_real_parsed_model_version_preferred_over_declared_config(tmp_path):
    parent = parent_at(tmp_path / 'source')
    parent.events = [{'kind': 'model.parsed', 'data': {'reported_model_version': 'real-model-version'}}]
    assert parent.model_version() == 'real-model-version'
    assert parent.outcome()['modelVersion'] == 'real-model-version'
