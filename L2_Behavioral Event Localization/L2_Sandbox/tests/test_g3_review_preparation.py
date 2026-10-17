"""Pure fixture checks for G3 preparation; no Docker or model calls."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/prepare_g3_review.py'
spec = importlib.util.spec_from_file_location('prepare_g3_review', SCRIPT)
g3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g3)


def fixture_rows(tmp_path):
    sheet = tmp_path / 'audit.md'
    sheet.write_text('frozen audit sheet')
    rows = []
    for i in range(1, 21):
        snapshot = tmp_path / f's{i:06d}.json'
        snapshot.write_text(json.dumps({'snapshot_id': snapshot.stem}))
        rows.append({'row_id': i, 'episode_id': f'episode{i}', 'checkpoint_id': f'cp{i:06d}',
                     'snapshot_id': snapshot.stem, 'snapshot_path': snapshot.name,
                     'expected_snapshot_sha256': g3.sha(snapshot),
                     'source': {'path': sheet.name, 'sha256': g3.sha(sheet)}})
    return {'rows': rows}


def test_frozen_20_point_identity_and_tamper_rejection(tmp_path):
    data = fixture_rows(tmp_path)
    assert len(g3.checked_rows(data, tmp_path)) == 20
    (tmp_path / data['rows'][4]['snapshot_path']).write_text('{}')
    with pytest.raises(ValueError, match='Snapshot identity mismatch: row 5'):
        g3.checked_rows(data, tmp_path)


def test_rejects_missing_or_duplicate_point(tmp_path):
    data = fixture_rows(tmp_path)
    with pytest.raises(ValueError, match='exactly'):
        g3.checked_rows({'rows': data['rows'][:-1]}, tmp_path)
    data['rows'][1]['episode_id'] = data['rows'][0]['episode_id']
    data['rows'][1]['checkpoint_id'] = data['rows'][0]['checkpoint_id']
    with pytest.raises(ValueError, match='Duplicate'):
        g3.checked_rows(data, tmp_path)


def test_infrastructure_error_cannot_win_candidate_selection():
    record = {'candidate': {'in_submission': True, 'last_written_turn': 2, 'tier': 0, 'candidate_id': 'submission/'},
              'grade': {'projection': {'completion_rate': 1.0}, 'status': 'infra_error'}}
    assert g3.rank(record) is None
    record['grade'].pop('status')
    assert g3.rank(record)[0] == 1.0
    record['grade']['projection']['completion_rate'] = None
    assert g3.rank(record) is None


def test_summary_retains_actual_exit_and_unknown_official_grade(tmp_path, monkeypatch):
    monkeypatch.setattr(g3, 'ROOT', tmp_path)
    point = tmp_path / 'point'
    candidate = {'paths': ['submission/damaged.pptx']}
    error = "AttributeError: 'NoneType' object has no attribute 'val'"
    record = {'candidate': candidate, 'grade': {'projection': {'completion_rate': None},
              'process': {'container_exit_code': 0}, 'official_raw': {'error': error}}}
    result = g3.summarize({'prior_auto_snapshot_score': 'unknown'}, point, [candidate], [record])
    assert result['scorer_exit_code'] == 0
    assert result['selected_candidate'] == candidate
    assert result['official_snapshot_grade']['projection']['completion_rate'] is None
    assert result['selection_status'] == 'unranked_unknown_grades'
    assert result['delivery_evidence']['matched_markers'] == [error]
    assert result['delivery_evidence']['study_outcome_evaluated'] is False
    assert result['reviewer'] == result['reviewed_at'] == result['human_conclusion'] == ''


def test_deleted_indexed_scorer_dependency_rejects_before_docker(tmp_path, monkeypatch):
    from types import SimpleNamespace
    source = tmp_path / 'benchmark/source'
    source.mkdir(parents=True)
    main = source / 'main.py'
    required = source / 'required_data.json'
    main.write_text('from pathlib import Path\ndata = Path(__file__).with_name("required_data.json").read_text()\n')
    required.write_text('{"required": true}')
    index = {'files': [{'path': p.relative_to(tmp_path).as_posix(), 'sha256': g3.sha(p),
                        'size_bytes': p.stat().st_size} for p in (main, required)]}
    pkg = SimpleNamespace(root=tmp_path, json=lambda name: index)
    assert len(g3.verify_official_source(pkg)) == 2
    required.unlink()
    monkeypatch.setattr(g3, 'L1Package', lambda path: pkg)
    docker_calls = []

    def unexpected_command(*args, **kwargs):
        docker_calls.append(args)
        raise AssertionError('Docker must not run after missing scorer dependency')

    monkeypatch.setattr(g3.subprocess, 'run', unexpected_command)
    with pytest.raises(ValueError, match='source inventory or digest mismatch'):
        g3.preflight([{'episode_id': 'fixture', 'snapshot_path': 'states/snapshots/s000000.json'}])
    assert docker_calls == []
