import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('auto_audit', ROOT / 'L2_Behavioral Event Localization/L2_Sandbox/scripts/build_automatic_review_audit.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_restored_inventory_rejects_deleted_and_injected_files(tmp_path):
    path = tmp_path / 'document.docx'
    path.write_bytes(b'original')
    rows = [{'path': path.name, 'sha256': audit.file_digest(path), 'size_bytes': path.stat().st_size}]
    audit.verify_inventory(tmp_path, rows)
    (tmp_path / 'extra').write_text('injected')
    with pytest.raises(ValueError, match='inventory changed'):
        audit.verify_inventory(tmp_path, rows)
    (tmp_path / 'extra').unlink()
    path.unlink()
    with pytest.raises(ValueError, match='inventory changed'):
        audit.verify_inventory(tmp_path, rows)


def test_evidence_path_cannot_escape_root(tmp_path):
    with pytest.raises(ValueError, match='outside'):
        audit.checked_path(tmp_path, '../private')


def test_empty_inventory_does_not_hide_missing_restoration(tmp_path):
    with pytest.raises(ValueError, match='directory missing'):
        audit.verify_inventory(tmp_path / 'missing', [])


def test_restored_symlinks_rejected(tmp_path):
    (tmp_path / 'link').symlink_to('/tmp')
    with pytest.raises(ValueError, match='symlinks'):
        audit.verify_inventory(tmp_path, [])


def test_missing_scoring_inventory_is_not_verified(tmp_path):
    rows = [{'row_id': i, 'episode_id': f'episode{i}', 'checkpoint_id': 'cp0'} for i in range(1, 21)]
    audit.write_json(tmp_path / 'report.json', {'rows': rows})
    audit.write_json(tmp_path / 'evidence_hashes.json', [])
    with pytest.raises(ValueError, match='inventory is empty'):
        audit.verify_saved_g3(tmp_path / 'report.json')


def test_checkpoint_without_actual_scorer_invocation_is_not_verified(tmp_path):
    receipt = tmp_path / 'receipt'
    receipt.write_bytes(b'receipt')
    rows = [{'row_id': i, 'episode_id': f'episode{i}', 'checkpoint_id': 'cp0',
             'candidate_evaluations': []} for i in range(1, 21)]
    audit.write_json(tmp_path / 'report.json', {'rows': rows})
    audit.write_json(tmp_path / 'evidence_hashes.json', [{'path': receipt.name,
        'sha256': audit.file_digest(receipt), 'size_bytes': receipt.stat().st_size}])
    with pytest.raises(ValueError, match='saved official scorer invocation'):
        audit.verify_saved_g3(tmp_path / 'report.json')
