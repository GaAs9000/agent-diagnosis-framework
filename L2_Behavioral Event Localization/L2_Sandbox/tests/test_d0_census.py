"""DEV census boundaries and receipt reuse, without Docker or Actor requests."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/build_d0_census.py'
spec = importlib.util.spec_from_file_location('d0_census', SCRIPT)
census = importlib.util.module_from_spec(spec)
spec.loader.exec_module(census)
from l2_sandbox.v3_execution import seal


def test_default_is_plan_only(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(census, 'VALIDATION', tmp_path)
    monkeypatch.setattr(census, 'preflight', lambda: dict(study_valid=41, sealed_verified=44, expected_slots=45, actor_requests=0))
    monkeypatch.setattr(census, 'execute', lambda plan: calls.append(plan))
    monkeypatch.setattr('sys.argv', [str(SCRIPT)])
    census.main()
    assert calls == []
    assert (tmp_path / 'plan.json').is_file()


def test_dev_scope_has_exact_nine_tasks():
    assert census.DEV == tuple('officeval_' + n for n in ('003', '004', '010', '033', '050', '060', '070', '085', '090'))


def test_synthetic_terminal_preserved_but_not_anchor():
    class Package:
        turns = 2

        def checkpoints(self):
            return {0: {'eligible': True}, 1: {'eligible': False}}

        def validate_resume_checkpoint(self, k):
            if k == 1:
                raise census.L2Error('unqualified')

    rows = [dict(kind='terminal_submission', k=2), *[dict(kind='checkpoint', k=k) for k in range(3)]]
    actual = census.annotate_checkpoints(Package(), rows)
    assert len(actual) == 4
    assert actual[1]['anchor_eligible'] is True
    assert actual[2]['actual_checkpoint_present'] is True
    assert actual[2]['anchor_eligible'] is False
    assert actual[3]['synthetic_terminal_checkpoint'] is True
    assert actual[3]['anchor_eligible'] is False


def test_reuse_requires_matching_provenance_and_unchanged_outputs(tmp_path):
    outputs = {}
    for name in ('facts.jsonl', 'grades.jsonl', 'defects.json'):
        (tmp_path / name).write_text('{}\n')
        outputs[name] = census.file_digest(tmp_path / name)
    raw = tmp_path / 'raw.json'
    raw.write_text('{}')
    binding = {'parent_index_sha256': 'sealed', 'analysis_sha256': 'source', 'study_rule_version': 'v1.1'}
    receipt = dict(status='complete', binding=binding, outputs=outputs,
                   official_evidence_inventory={str(raw): census.file_digest(raw)})
    census.write_json(tmp_path / 'receipt.json', receipt)
    assert census.reusable(tmp_path, binding)
    assert not census.reusable(tmp_path, {**binding, 'parent_index_sha256': 'other'})
    assert not census.reusable(tmp_path, {**binding, 'analysis_sha256': 'changed'})
    raw.write_text('{"tampered": true}')
    assert not census.reusable(tmp_path, binding)
    raw.write_text('{}')
    (tmp_path / 'grades.jsonl').write_text('changed')
    assert not census.reusable(tmp_path, binding)


def test_preserves_raw_scorer_files_and_restores_adapter(tmp_path, monkeypatch):
    class Evaluator:
        def __init__(self, config):
            assert config['image'] == 'sha256:frozen'

        def evaluate(self, spec, submission, output):
            output.mkdir()
            (output / 'official.json').write_text('{"score": 1}')
            return {'projection': {'completion_rate': 1}, 'official_raw_sha256': 'raw'}

    monkeypatch.setattr(census.grading, 'DockerEvaluator', Evaluator)
    spec = type('Spec', (), dict(task_id='officeval_003', entrypoint='verifier.py'))()
    with census.preserve_grading_evidence(tmp_path / 'evidence', 'sha256:frozen') as calls:
        census.grading.DockerEvaluator({}).evaluate(spec, tmp_path, tmp_path / 'temp_grade')
    assert census.grading.DockerEvaluator is Evaluator
    assert len(calls) == 1
    assert (tmp_path / 'evidence/invocation_000000/grading/official.json').is_file()
    assert (tmp_path / 'evidence/invocation_000000/result.json').is_file()


def test_preflight_failure_prevents_execute(monkeypatch, tmp_path):
    def fail():
        raise ValueError('seal mismatch')
    monkeypatch.setattr(census, 'preflight', fail)
    monkeypatch.setattr('sys.argv', [str(SCRIPT), '--execute'])
    monkeypatch.setattr(census, 'execute', lambda plan: pytest.fail('must not execute'))
    with pytest.raises(ValueError, match='seal mismatch'):
        census.main()


def cohort_records():
    return {f'{task}.r{rep:02d}': dict(label=f'{task}.r{rep:02d}', task_id=task,
             status='completed', study_valid=True, episode_id=f'ep_{task}_{rep}',
             index_file_sha256='index', manifest_sha256='manifest', config_sha256='config')
            for task in census.DEV for rep in range(5)}


def test_prospective_cohort_requires_exact_valid_formal_roots(tmp_path):
    path = tmp_path / 'cohort.json'
    records = cohort_records()
    census.write_json(path, seal(dict(slots=records, plan_sha256='plan')))
    assert census.cohort_identity(path)['sha256'] == census.file_digest(path)
    label = next(iter(records))
    records[label]['study_valid'] = False
    census.write_json(path, seal(dict(slots=records)))
    with pytest.raises(ValueError, match='valid completed root'):
        census.cohort_identity(path)
    del records[label]
    census.write_json(path, seal(dict(slots=records)))
    with pytest.raises(ValueError, match='exactly DEV9'):
        census.cohort_identity(path)


def test_prospective_cli_requires_isolated_explicit_paths(monkeypatch):
    monkeypatch.setattr('sys.argv', [str(SCRIPT), '--cohort', 'new.json'])
    with pytest.raises(SystemExit):
        census.main()


def test_analysis_plan_relocation_checks_bindings_without_docker(monkeypatch, tmp_path):
    frozen = dict(analysis={'sha256': 'analysis'}, cohort={'sha256': 'cohort'},
                  slots=[dict(slot='one', status='ready', parent='/mac/one',
                              binding={'parent_index_sha256': 'index'}, source={'sha256': 'source'})],
                  study_valid=45, sealed_verified=45, expected_slots=45, actor_requests=0)
    relocated = {**frozen, 'slots': [{**frozen['slots'][0], 'parent': '/server/one'}]}
    path = tmp_path / 'frozen.json'
    census.write_json(path, frozen)
    calls = []
    def preflight(runs, cohort, *, inspect_images):
        assert not inspect_images
        return relocated
    monkeypatch.setattr(census, 'preflight', preflight)
    monkeypatch.setattr(census, 'execute', lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setattr('sys.argv', [str(SCRIPT), '--runs', 'server', '--cohort', 'cohort.json',
                                   '--diagnoses', str(tmp_path / 'diagnoses'), '--validation', str(tmp_path),
                                   '--stage', 'analysis', '--plan', str(path)])
    census.main()
    assert calls == []  # Explicit execute remains required even for the server-only stage.
    assert (tmp_path / 'analysis_plan.json').is_file()
    assert census.read_json(path)['slots'][0]['parent'] == '/mac/one'
    relocated['slots'][0]['binding'] = {'parent_index_sha256': 'changed'}
    with pytest.raises(ValueError, match='binding differs'):
        census.bind_frozen_plan(relocated, frozen)


def test_portable_grading_receipt_detects_tampered_raw_evidence(tmp_path):
    output = tmp_path / 'diagnoses/slot'
    output.mkdir(parents=True)
    validation = tmp_path / 'validation'
    raw = validation / 'official_evidence/namespace/invocation_000000/result.json'
    raw.parent.mkdir(parents=True)
    raw.write_text('{"official": true}')
    invocation = raw.with_name('invocation.json')
    invocation.write_text('{"status": "returned"}')
    (output / 'grades.jsonl').write_text('{}\n')
    binding = {'cohort_sha256': 'new', 'parent_index_sha256': 'index'}
    receipt = dict(status='grades_complete', binding=binding, official_scorer_invocations=1,
                   grades_sha256=census.file_digest(output / 'grades.jsonl'),
                   official_evidence_inventory_relative={str(p.relative_to(validation)): census.file_digest(p) for p in (raw, invocation)})
    census.write_json(output / 'grading_receipt.json', receipt)
    assert census.grade_receipt(output, binding, validation)
    assert census.grade_receipt(output, {**binding, 'cohort_sha256': 'old'}, validation) is None
    raw.write_text('changed')
    assert census.grade_receipt(output, binding, validation) is None


def test_mac_prospective_analysis_is_blocked_before_any_work(monkeypatch, tmp_path):
    monkeypatch.setattr(census.sys, 'platform', 'darwin')
    with pytest.raises(ValueError, match='lab server'):
        census.execute({'cohort': {'sha256': 'new'}}, tmp_path, tmp_path, stage='analysis')
    assert not list(tmp_path.iterdir())


def test_missing_grading_receipt_never_invokes_docker_in_analysis(monkeypatch, tmp_path):
    package = type('Package', (), {'root': tmp_path})()
    plan = {'analysis': {'sha256': 'analysis'}, 'cohort': {'sha256': 'new'},
            'slots': [dict(slot='one', status='ready', parent=str(tmp_path),
                           parent_index_sha256='index', binding={'cohort_sha256': 'new'})]}
    monkeypatch.setattr(census.sys, 'platform', 'linux')
    monkeypatch.setattr(census, 'L1Package', lambda path: package)
    monkeypatch.setattr(census, 'verify_package', lambda *args, **kwargs: {'verified': True})
    monkeypatch.setattr(census, 'analysis_identity', lambda: plan['analysis'])
    monkeypatch.setattr(census.grading, 'grade_episode_snapshots', lambda *args, **kwargs: pytest.fail('Docker called'))
    with pytest.raises(ValueError, match='valid grading receipt'):
        census.execute(plan, tmp_path / 'diagnoses', tmp_path / 'validation', stage='analysis')


@pytest.mark.parametrize('mismatch', [None, 'index_file_sha256', 'manifest_sha256', 'config_sha256', 'episode_id'])
def test_preflight_binds_every_prospective_slot_to_sealed_ledger(monkeypatch, tmp_path, mismatch):
    config = {'model': {'model': 'Qwen/Qwen3.6-27B'}, 'agent': {'harness': 'pi'},
              'evaluation': {'image': 'sha256:image'}}
    records = cohort_records()
    for label, record in records.items():
        path = tmp_path / 'episodes' / label
        path.mkdir(parents=True)
        (path / 'manifest.json').write_text('{}')
        (path / 'evidence_index.json').write_text('{}')
        record.update(index_file_sha256=census.file_digest(path / 'evidence_index.json'),
                      manifest_sha256=census.file_digest(path / 'manifest.json'), config_sha256=census.digest(config))
    if mismatch:
        records[next(iter(records))][mismatch] = 'wrong'
    cohort = tmp_path / 'cohort.json'
    census.write_json(cohort, seal({'slots': records}))
    class Package:
        def __init__(self, path):
            self.root = path
            self.task_id = path.name.split('.')[0]
            self.episode_id = 'ep_' + self.task_id + '_' + str(int(path.name[-2:]))
            self.manifest = {}
        def outcome(self):
            return {'valid': True}
        def json(self, name):
            if name == 'raw/resolved_config.json':
                return config
            return {'runtime': {'image_id': 'sha256:image', 'python_environment': {'grader_worker_sha256': 'worker'}}}
    real_digest = census.file_digest
    monkeypatch.setattr(census, 'file_digest', lambda path: 'worker' if path.name == 'worker.py' else real_digest(path))
    monkeypatch.setattr(census, 'L1Package', Package)
    monkeypatch.setattr(census, 'verify_package', lambda *args, **kwargs: {'verified': True})
    monkeypatch.setattr(census, 'analysis_identity', lambda: {'sha256': 'analysis'})
    monkeypatch.setattr(census, 'source_identity', lambda pkg: {'sha256': 'source'})
    monkeypatch.setattr(census.subprocess, 'run', lambda *args, **kwargs: pytest.fail('Docker called on server'))
    if mismatch:
        with pytest.raises(ValueError, match='differs from sealed cohort'):
            census.preflight(tmp_path / 'episodes', cohort, inspect_images=False)
    else:
        plan = census.preflight(tmp_path / 'episodes', cohort, inspect_images=False)
        assert plan['study_valid'] == 45
        assert len(plan['slots']) == 45
        assert all(row['binding']['cohort_sha256'] == real_digest(cohort) for row in plan['slots'])


def test_split_stages_keep_analysis_off_grading_host_and_preserve_receipt(monkeypatch, tmp_path):
    package = type('Package', (), {'root': tmp_path})()
    binding = {'cohort_sha256': 'new', 'parent_index_sha256': 'index'}
    slot = dict(slot='one', status='ready', parent=str(tmp_path), parent_index_sha256='index',
                binding=binding, image_id='sha256:image', source={'sha256': 'source'})
    plan = {'analysis': {'sha256': 'analysis'}, 'cohort': {'sha256': 'new'}, 'slots': [slot]}
    monkeypatch.setattr(census, 'L1Package', lambda path: package)
    monkeypatch.setattr(census, 'verify_package', lambda *args, **kwargs: {'verified': True})
    monkeypatch.setattr(census, 'analysis_identity', lambda: plan['analysis'])
    monkeypatch.setattr(census, 'annotate_checkpoints', lambda pkg, rows: rows)
    scorer_calls = []
    def grade(*args, **kwargs):
        scorer_calls.append(True)
        return [{'kind': 'checkpoint', 'k': 0}]
    monkeypatch.setattr(census.grading, 'grade_episode_snapshots', grade)
    monkeypatch.setattr(census, 'build_facts', lambda pkg: pytest.fail('analysis ran on grading host'))
    monkeypatch.setattr(census.sys, 'platform', 'darwin')
    diagnoses, validation = tmp_path / 'diagnoses', tmp_path / 'validation'
    namespace = census.digest({k: v for k, v in binding.items() if k != 'parent_index_sha256'})
    cache = validation / 'cache' / namespace / 'grades'
    cache.mkdir(parents=True)
    (cache / 'cached.json').write_text('{}')
    graded = census.execute(plan, diagnoses, validation, stage='grades')
    assert graded['full_45_ready'] is False
    assert not (diagnoses / 'one/facts.jsonl').exists()
    assert not (diagnoses / 'one/receipt.json').exists()
    assert (diagnoses / 'one/grading_receipt.json').is_file()
    monkeypatch.setattr(census.sys, 'platform', 'linux')
    monkeypatch.setattr(census, 'build_facts', lambda pkg: [{'turn': 0, 'type': 'test', 'value': True}])
    monkeypatch.setattr(census, 'build_defect_table', lambda *args, **kwargs: {'rows': []})
    census.execute(plan, diagnoses, validation, stage='analysis')
    assert len(scorer_calls) == 1
    assert census.reusable(diagnoses / 'one', binding)
    census.execute(plan, diagnoses, validation, stage='analysis')
    assert len(scorer_calls) == 1
    grade_receipt_path = diagnoses / 'one/grading_receipt.json'
    original_receipt = grade_receipt_path.read_text()
    grade_receipt_path.unlink()
    with pytest.raises(ValueError, match='valid grading receipt'):
        census.execute(plan, diagnoses, validation, stage='analysis')
    grade_receipt_path.write_text(original_receipt)
    altered = census.read_json(grade_receipt_path)
    altered['grades_sha256'] = 'tampered'
    census.write_json(grade_receipt_path, altered)
    with pytest.raises(ValueError, match='valid grading receipt'):
        census.execute(plan, diagnoses, validation, stage='analysis')
    assert len(scorer_calls) == 1


def test_cohort_rejects_missing_or_altered_artifact_seal(tmp_path):
    path = tmp_path / 'cohort.json'
    value = {'slots': cohort_records()}
    census.write_json(path, value)
    with pytest.raises(ValueError, match='artifact_sha256 seal'):
        census.cohort_identity(path)
    value = seal(value)
    value['slots']['officeval_003.r00']['episode_id'] = 'altered'
    census.write_json(path, value)
    with pytest.raises(ValueError, match='artifact_sha256 seal'):
        census.cohort_identity(path)


def test_nonzero_invocation_count_requires_raw_proof_pair(tmp_path):
    output = tmp_path / 'diagnoses'
    output.mkdir()
    (output / 'grades.jsonl').write_text('{}\n')
    receipt = dict(status='grades_complete', binding={}, official_scorer_invocations=1,
                   grades_sha256=census.file_digest(output / 'grades.jsonl'),
                   official_evidence_inventory_relative={})
    census.write_json(output / 'grading_receipt.json', receipt)
    assert census.grade_receipt(output, {}, tmp_path) is None
    receipt.update(official_scorer_invocations=0)
    census.write_json(output / 'grading_receipt.json', receipt)
    assert census.grade_receipt(output, {}, tmp_path) is None
    receipt.update(cache_entries_before=1, zero_invocations_reason='all_snapshot_grades_reused_from_bound_cache')
    census.write_json(output / 'grading_receipt.json', receipt)
    assert census.grade_receipt(output, {}, tmp_path)
