"""Prospective-selection integrity; no model, Docker, or external evidence."""
import importlib.util
import json
from pathlib import Path

import pytest


def runner():
    path = Path(__file__).resolve().parents[1] / 'scripts/run_r11.py'
    spec = importlib.util.spec_from_file_location('r11_runner_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(tmp_path, monkeypatch):
    module = runner()
    plan = {'candidates': [{'order': 0, 'task_id': 'fixture-first'}, {'order': 1, 'task_id': 'fixture-second'}]}
    monkeypatch.setattr(module, 'select_engineering_candidates', lambda results: {'chosen': results[0]['task_id']})
    module.append_event(tmp_path, 'candidate_pool_frozen')
    for candidate in plan['candidates']:
        folder = tmp_path / 'originals' / candidate['task_id']
        folder.mkdir(parents=True)
        module.write(folder / 'result.json', {'task_id': candidate['task_id']})
        module.append_event(tmp_path, 'original_started', **candidate)
        module.append_event(tmp_path, 'original_finished', **candidate, result_sha256=module.sha(folder / 'result.json'))
    module.write(tmp_path / 'coverage_selection.json', {'chosen': 'fixture-first'})
    module.append_event(tmp_path, 'original_pool_finished', selection_sha256=module.sha(tmp_path / 'coverage_selection.json'))
    return module, plan


def test_original_and_selection_seals(tmp_path, monkeypatch):
    module, plan = fixture(tmp_path, monkeypatch)
    assert module.verified_selection(tmp_path, plan) == {'chosen': 'fixture-first'}
    (tmp_path / 'originals/fixture-first/result.json').write_text('{"task_id":"substituted"}')
    with pytest.raises(RuntimeError, match='original result changed'):
        module.verified_selection(tmp_path, plan)


def test_substituted_later_selection_rejected(tmp_path, monkeypatch):
    module, plan = fixture(tmp_path, monkeypatch)
    (tmp_path / 'coverage_selection.json').write_text('{"chosen":"fixture-second"}')
    with pytest.raises(RuntimeError, match='selection changed'):
        module.verified_selection(tmp_path, plan)


def test_selection_is_recomputed(tmp_path, monkeypatch):
    module, plan = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(module, 'select_engineering_candidates', lambda results: {'chosen': 'different-rule'})
    with pytest.raises(RuntimeError, match='first-S/T'):
        module.verified_selection(tmp_path, plan)


def test_journal_mutation_rejected(tmp_path, monkeypatch):
    module, plan = fixture(tmp_path, monkeypatch)
    path = tmp_path / 'run_journal.jsonl'
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    rows[1]['task_id'] = 'another-task'
    path.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
    with pytest.raises(RuntimeError, match='journal integrity'):
        module.verified_selection(tmp_path, plan)
