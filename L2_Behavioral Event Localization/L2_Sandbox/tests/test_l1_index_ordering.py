"""A sealed live inventory is order-independent and remains content-strict."""
from pathlib import Path
import pytest
from l2_sandbox.evidence.l1 import L1Package
from l2_sandbox.io import IntegrityError, sha_file, write_json


def sealed_live_package(root):
    write_json(root / 'manifest.json', {'episode_id': 'ep_ordering', 'recording_mode': 'live'})
    for sibling in ('rubrics', 'rubrics-en'):
        write_json(root / 'benchmark/source/data' / sibling / 'officeval_001.json', {'rubric': sibling})
    files = [{'path': p.relative_to(root).as_posix(), 'sha256': sha_file(p), 'size_bytes': p.stat().st_size}
             for p in sorted(root.rglob('*')) if p.is_file()]
    # Both files are inventoried, but pathlib and serialized full strings differ.
    assert files != sorted(files, key=lambda row: row['path'])
    write_json(root / 'evidence_index.json', {'episode_id': 'ep_ordering',
                                           'files': sorted(files, key=lambda row: row['path'])})
    return root


def test_live_inventory_hyphenated_siblings_compare_in_same_order(tmp_path):
    root = sealed_live_package(tmp_path / 'episode')
    assert L1Package(root).episode_id == 'ep_ordering'


def test_l2_uses_declared_seal_without_rewalking_inventory(tmp_path, monkeypatch):
    root = sealed_live_package(tmp_path / 'episode')
    def no_scan(*args, **kwargs):
        pytest.fail('L2 must leave full evidence verification to L1')
    monkeypatch.setattr(Path, 'rglob', no_scan)
    assert L1Package(root).index_sha256
