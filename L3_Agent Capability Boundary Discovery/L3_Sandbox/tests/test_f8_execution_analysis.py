"""F8: sealed native executions survive independent analysis revisions."""
from contextlib import nullcontext
from pathlib import Path

import pytest

from l3_sandbox import plan, report, run
from l3_sandbox.common import L3Error, digest, read, sha_file, write


def test_execution_inventory_covers_actor_transport_restore_and_excludes_analysis():
    current = plan.source_inventory()
    assert current
    for suffix in ('models/native.py', 'models/http.py', 'models/registry.py',
                   'core/orchestrator.py', 'runtime/docker.py', 'recording/state.py'):
        assert any(name.endswith(suffix) for name in current)
    assert not any(name.endswith('/evaluation/study.py') for name in current)
    for name in ('report.py', 'outcomes.py', 'plan.py'):
        assert str(Path(plan.__file__).parent / name) not in current
    assert str(Path(run.__file__)) in current


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    import c3_sandbox.l1
    l1 = Path(c3_sandbox.l1.__file__).parent
    actor = str(l1 / 'models/native.py')
    analysis = str(Path(report.__file__))
    endpoint = str(l1 / 'evaluation/study.py')
    versions = {actor: 'actor-v1', analysis: 'report-v1', endpoint: 'endpoint-v1'}
    inventory = lambda: dict(versions)
    monkeypatch.setattr(plan, 'provenance_inventory', inventory)
    monkeypatch.setattr(run, 'provenance_inventory', inventory)
    monkeypatch.setattr(report, 'provenance_inventory', inventory)
    monkeypatch.setattr(plan, 'image_id', lambda value: value)
    cfg = {'model': {'kind': 'offline'}}
    task = {'task_id': 'officeval_003', 'family_id': 'family', 'family_verified': True,
            'split': 'development'}
    job = {**task, 'trial_id': 'trial', 'condition_id': 'control', 'replicate': 0,
           'config': cfg, 'config_sha256': digest(cfg)}
    frozen_plan = {'schema_version': plan.VERSION, 'experiment_id': 'offline', 'purpose': 'pilot',
        'design': 'budget', 'tasks': [task], 'conditions': [{'condition_id': 'control'}],
        'jobs': [job], 'analysis': {}, 'limits': {'max_concurrency': 1,
        'max_http_attempts': 4, 'max_replacements_per_slot': 0, 'rpm': None, 'tpm': None},
        'source_scope': 'execution-v1', 'source_sha256': plan.source_inventory(),
        'code_provenance': inventory(), 'inputs_sha256': {}, 'images': {}}
    write(tmp_path / 'plan.json', frozen_plan)
    (tmp_path / 'plan.sha256').write_text(sha_file(tmp_path / 'plan.json'))
    monkeypatch.setattr(run, '_credentials', lambda _: nullcontext())
    monkeypatch.setattr(run, '_transport_guard', lambda _: nullcontext())
    monkeypatch.setattr(run, '_verify_package', lambda *a: {'verified': True})
    calls = []
    def execute(config, output, *, request_limiter):
        calls.append(output)
        assert request_limiter.acquire(1, timeout=1)
        request_limiter.reconcile(1, 1)
        output.mkdir()
        write(output / 'manifest.json', {'recording_mode': 'live', 'config_sha256': digest(config),
            'task': {'task_id': task['task_id'], 'benchmark': 'officeval'},
            'status': {'execution_validity': 'valid', 'evidence_completeness': 'complete'}})
        write(output / 'raw/resolved_config.json', config)
        write(output / 'official_grade.json', {'process': {'container_exit_code': 0}})
        write(output / 'grading/worker-output/worker.json', {'official_returned': True, 'exception': None})
        for name in ('artifacts/inventory.json', 'evidence_index.json'):
            write(output / name, {})
        for name in ('trajectory.jsonl', 'state_changes.jsonl'):
            (output / name).write_text('')
    monkeypatch.setattr(run, '_native_run', execute)
    result = {'study_valid': True, 'study_outcome_y': 0.0, 'endpoint_version': 'v1',
              'model_versions': ['offline-test-version']}
    monkeypatch.setattr(run, '_summarize', lambda *a: dict(result))
    return tmp_path, frozen_plan, versions, actor, analysis, endpoint, result, calls


@pytest.mark.parametrize('legacy', [False, True])
def test_reanalysis_reuses_sealed_execution_without_rewriting_receipt(frozen, legacy):
    directory, frozen_plan, versions, actor, analysis, endpoint, result, calls = frozen
    if legacy:
        frozen_plan.pop('source_scope'); frozen_plan.pop('code_provenance')
        frozen_plan['source_sha256'] = dict(versions)
        write(directory / 'plan.json', frozen_plan)
        (directory / 'plan.sha256').write_text(sha_file(directory / 'plan.json'))
    run.run_plan(directory, execute=True)
    receipt = calls[0].parent / 'receipt.json'
    if legacy:
        # Actual old receipts have no per-attempt provenance. Preserve the old
        # full plan inventory as the read-only migration anchor.
        saved = read(receipt); saved['row'].pop('code_provenance')
        run._save_receipt(receipt, saved['row'])
        write(directory / 'run_results.json', {'rows': [saved['row']]})
    receipt_bytes = receipt.read_bytes()
    plan_bytes = (directory / 'plan.json').read_bytes()
    versions[analysis] = 'report-v2'
    versions[endpoint] = 'endpoint-v2'
    result.update(study_outcome_y=.75, endpoint_version='v2')
    assert report.summarize(directory)['cells'][0]['mean'] == .75
    resumed = run.run_plan(directory, execute=True)
    assert resumed['rows'][0]['study_outcome_y'] == .75
    assert resumed['http_admissions']['count'] == 1
    assert len(calls) == 1
    assert receipt.read_bytes() == receipt_bytes
    assert (directory / 'plan.json').read_bytes() == plan_bytes
    assert report.summarize(directory)['cells'][0]['mean'] == .75
    # A cached run table from v2 is disposable when a third endpoint arrives.
    versions[endpoint] = 'endpoint-v3'
    result.update(study_outcome_y=.25, endpoint_version='v3')
    assert report.summarize(directory)['cells'][0]['mean'] == .25
    assert receipt.read_bytes() == receipt_bytes
    # Even a later actor revision does not prevent read-only historical analysis.
    versions[actor] = 'actor-v2'
    assert report.summarize(directory)['cells'][0]['mean'] == .25
    with pytest.raises(L3Error, match='Execution source'):
        run.run_plan(directory, execute=True)
    assert len(calls) == 1


def test_changed_endpoint_validity_does_not_buy_a_replacement(frozen):
    directory, _, versions, _, _, endpoint, result, calls = frozen
    run.run_plan(directory, execute=True)
    versions[endpoint] = 'new-policy'
    result.update(study_valid=False, study_outcome_y=None, endpoint_version='v2')
    resumed = run.run_plan(directory, execute=True)
    assert resumed['status'] == 'complete'
    assert resumed['rows'][0]['accepted'] is True
    assert len(calls) == 1
    summary = report.summarize(directory)
    assert summary['nvalid'] == 0 and summary['nmissing'] == 1
