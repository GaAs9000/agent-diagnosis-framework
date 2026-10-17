"""DEV census boundaries and receipt reuse, without Docker or Actor requests."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/build_d0_census.py'
spec = importlib.util.spec_from_file_location('d0_census', SCRIPT)
census = importlib.util.module_from_spec(spec)
spec.loader.exec_module(census)


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
