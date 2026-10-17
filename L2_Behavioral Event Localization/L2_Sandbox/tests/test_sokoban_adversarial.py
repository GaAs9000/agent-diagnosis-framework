"""Adversarial challenge test suite for L2 v0.4 Sokoban grader and candidate discovery."""
from __future__ import annotations

import copy
import shutil
import tempfile
from pathlib import Path
import pytest

from l2_sandbox.evidence.l1 import L1Package
from l2_sandbox.grades import (
    GATE_DEFECT_ID,
    compute_item_deductions,
    grade_episode_snapshots,
    select_snapshot_candidates,
    _evaluate_staged_candidate,
)
from l2_sandbox.io import IntegrityError, read_json, write_json


def test_adversarial_empty_and_zero_byte_moves_log():
    """Empty (0-byte) moves.log in any directory must yield no_candidate and not crash."""
    # 1. 0-byte in submission/
    snap_sub_empty = {
        'files': [
            {'path': 'submission/moves.log', 'sha256': 'e3b0c442', 'blob': 'b0', 'size_bytes': 0},
        ]
    }
    cands = select_snapshot_candidates(
        snap_sub_empty, set(), ['moves.log'], {'.log'}, {}, task_id='sokoban', spec_kind='lhtb-verifier'
    )
    assert cands == [], f'Expected [], got {cands}'

    # 2. 0-byte in runs/ and root
    snap_runs_empty = {
        'files': [
            {'path': 'runs/moves.log', 'sha256': 'e3b0c442', 'blob': 'b0', 'size_bytes': 0},
            {'path': 'moves.log', 'sha256': 'e3b0c442', 'blob': 'b0', 'size_bytes': 0},
        ]
    }
    cands = select_snapshot_candidates(
        snap_runs_empty, set(), ['moves.log'], {'.log'}, {}, task_id='sokoban', spec_kind='lhtb-verifier'
    )
    assert cands == []

    # 3. 0-byte submission file + valid runs file: 0-byte must be skipped, valid kept
    snap_mixed = {
        'files': [
            {'path': 'submission/moves.log', 'sha256': 'e3b0c442', 'blob': 'b0', 'size_bytes': 0},
            {'path': 'runs/moves.log', 'sha256': 'valid_sha', 'blob': 'b1', 'size_bytes': 42},
        ]
    }
    cands = select_snapshot_candidates(
        snap_mixed, set(), ['moves.log'], {'.log'}, {}, task_id='sokoban', spec_kind='lhtb-verifier'
    )
    assert len(cands) == 1
    assert cands[0]['candidate_id'] == 'runs/moves.log'
    assert cands[0]['tier'] == 1
    assert cands[0]['in_submission'] is False

    # 4. Negative size or missing size_bytes
    snap_neg = {
        'files': [
            {'path': 'submission/moves.log', 'sha256': 'e3b0c442', 'blob': 'b0', 'size_bytes': -1},
            {'path': 'runs/moves.log', 'sha256': 'e3b0c442', 'blob': 'b0'},
        ]
    }
    cands = select_snapshot_candidates(
        snap_neg, set(), ['moves.log'], {'.log'}, {}, task_id='sokoban', spec_kind='lhtb-verifier'
    )
    assert cands == []


def test_adversarial_directory_tier_priority_and_filtering():
    """Verify rigorous filtering of input/ and ~$* files, and strict tier priority."""
    files = [
        # Must be filtered: input directory
        {'path': 'input/moves.log', 'sha256': 'in1', 'blob': 'b_in1', 'size_bytes': 100},
        {'path': 'input/submission/moves.log', 'sha256': 'in2', 'blob': 'b_in2', 'size_bytes': 100},
        {'path': 'input/runs/moves.log', 'sha256': 'in3', 'blob': 'b_in3', 'size_bytes': 100},

        # Must be filtered: lock files (~$*)
        {'path': '~$moves.log', 'sha256': 'lock1', 'blob': 'b_l1', 'size_bytes': 100},
        {'path': 'submission/~$moves.log', 'sha256': 'lock2', 'blob': 'b_l2', 'size_bytes': 100},

        # Must be filtered: zero bytes
        {'path': 'submission/zero.log', 'sha256': 'zero1', 'blob': 'b_z1', 'size_bytes': 0},

        # Tier 2: other .log files
        {'path': 'debug/agent.log', 'sha256': 't2_sha', 'blob': 'b_t2', 'size_bytes': 200},

        # Tier 1: root moves.log and runs/moves.log
        {'path': 'moves.log', 'sha256': 'root_sha', 'blob': 'b_root', 'size_bytes': 150},
        {'path': 'runs/moves.log', 'sha256': 'runs_sha', 'blob': 'b_runs', 'size_bytes': 180},

        # Tier 0: submission/moves.log
        {'path': 'submission/moves.log', 'sha256': 'sub_sha', 'blob': 'b_sub', 'size_bytes': 120},
    ]

    lwt = {
        'moves.log': 5,
        'runs/moves.log': 10,
        'submission/moves.log': 8,
        'debug/agent.log': 2,
    }

    cands = select_snapshot_candidates(
        {'files': files},
        initial_input_hashes=set(),
        office_targets=['moves.log'],
        target_exts={'.log'},
        last_written_turn=lwt,
        task_id='sokoban',
        spec_kind='lhtb-verifier',
    )

    assert len(cands) == 4
    # Tier 0 first
    assert cands[0]['candidate_id'] == 'submission/moves.log'
    assert cands[0]['tier'] == 0
    assert cands[0]['in_submission'] is True

    # Tier 1 next, ordered by last_written_turn descending (10 > 5)
    assert cands[1]['candidate_id'] == 'runs/moves.log'
    assert cands[1]['tier'] == 1
    assert cands[1]['last_written_turn'] == 10
    assert cands[1]['in_submission'] is False

    assert cands[2]['candidate_id'] == 'moves.log'
    assert cands[2]['tier'] == 1
    assert cands[2]['last_written_turn'] == 5
    assert cands[2]['in_submission'] is False

    # Tier 2 last
    assert cands[3]['candidate_id'] == 'debug/agent.log'
    assert cands[3]['tier'] == 2
    assert cands[3]['in_submission'] is False


def test_adversarial_content_deduplication():
    """Verify candidates with identical staged content hash are deduplicated across tiers."""
    snapshot = {
        'files': [
            {'path': 'submission/moves.log', 'sha256': 'identical_sha', 'blob': 'b1', 'size_bytes': 100},
            {'path': 'runs/moves.log', 'sha256': 'identical_sha', 'blob': 'b1', 'size_bytes': 100},
            {'path': 'moves.log', 'sha256': 'identical_sha', 'blob': 'b1', 'size_bytes': 100},
            {'path': 'debug/agent.log', 'sha256': 'identical_sha', 'blob': 'b1', 'size_bytes': 100},
        ]
    }
    cands = select_snapshot_candidates(
        snapshot, set(), ['moves.log'], {'.log'}, {}, task_id='sokoban', spec_kind='lhtb-verifier'
    )
    assert len(cands) == 1
    assert cands[0]['candidate_id'] == 'submission/moves.log'
    assert cands[0]['tier'] == 0


def test_adversarial_cache_missing_dir_auto_creation(tmp_path):
    """Missing cache directories must be automatically created without error."""
    cohort_dir = Path(__file__).resolve().parents[3] / 'L1_sandbox' / 'C3_Experimental_Sandbox' / 'runs' / 'lhtb_sokoban_qwen36_cohort'
    pkg_dir = cohort_dir / 'run_seed42'
    if not pkg_dir.is_dir():
        pytest.skip('run_seed42 baseline not present')

    pkg = L1Package(pkg_dir)
    # Deeply nested, non-existent cache directory
    non_existent_cache = tmp_path / 'deep' / 'nested' / 'non_existent' / 'cache'
    assert not non_existent_cache.exists()

    # Pre-copy one cache file if possible, or run live
    # To test directory creation on cache write:
    source_cache = Path(__file__).resolve().parents[1] / 'cache' / 'grades'
    # Copy all cache files to non_existent_cache / 'grades'
    shutil.copytree(source_cache, non_existent_cache / 'grades')

    rows = grade_episode_snapshots(pkg, cache_dir=non_existent_cache)
    assert len(rows) == 22
    assert (non_existent_cache / 'grades').is_dir()


def test_adversarial_semantic_cache_corruption(tmp_path):
    """If a cache file has valid JSON but corrupted schema or wrong key, it must re-evaluate."""
    cohort_dir = Path(__file__).resolve().parents[3] / 'L1_sandbox' / 'C3_Experimental_Sandbox' / 'runs' / 'lhtb_sokoban_qwen36_cohort'
    pkg_dir = cohort_dir / 'run_seed42'
    if not pkg_dir.is_dir():
        pytest.skip('run_seed42 baseline not present')

    pkg = L1Package(pkg_dir)
    source_cache = Path(__file__).resolve().parents[1] / 'cache' / 'grades'
    cache_dir = tmp_path / 'corrupt_cache'
    shutil.copytree(source_cache, cache_dir / 'grades')

    # Target key accessed during run_seed42 evaluation
    target_key = 'c22582a0ea84ab13495c092768645db77304ad90bf403a90fd9289ed06274ac2'
    target_file = cache_dir / 'grades' / f'{target_key}.json'
    assert target_file.is_file()

    # Corrupt semantically: mismatched cache_key and null grade
    target_file.write_text('{"cache_key": "corrupted_mismatched_key", "grade": null}')

    # Must re-evaluate and recover
    rows = grade_episode_snapshots(pkg, cache_dir=cache_dir)
    assert len(rows) == 22
    assert abs(rows[0]['score'] - 0.003226) < 1e-6

    # Verify that target_file was regenerated and overwritten with valid grade
    recovered_data = read_json(target_file)
    assert recovered_data.get('cache_key') == target_key
    assert isinstance(recovered_data.get('grade'), dict)


def test_adversarial_syntactic_cache_corruption(tmp_path):
    """Document behavior when cache file is truncated or 0 bytes."""
    cohort_dir = Path(__file__).resolve().parents[3] / 'L1_sandbox' / 'C3_Experimental_Sandbox' / 'runs' / 'lhtb_sokoban_qwen36_cohort'
    pkg_dir = cohort_dir / 'run_seed42'
    if not pkg_dir.is_dir():
        pytest.skip('run_seed42 baseline not present')

    pkg = L1Package(pkg_dir)
    source_cache = Path(__file__).resolve().parents[1] / 'cache' / 'grades'
    cache_dir = tmp_path / 'syntactic_corrupt_cache'
    shutil.copytree(source_cache, cache_dir / 'grades')

    target_key = 'c22582a0ea84ab13495c092768645db77304ad90bf403a90fd9289ed06274ac2'
    target_file = cache_dir / 'grades' / f'{target_key}.json'
    # Truncated JSON
    target_file.write_text('{"cache_key": ')

    # read_json raises IntegrityError on invalid JSON syntax
    with pytest.raises(IntegrityError):
        grade_episode_snapshots(pkg, cache_dir=cache_dir)


def test_adversarial_sokoban_deduction_invariants():
    """Verify sum of deductions equals 1 - score across all boundary cases."""
    # Boundary 1: Complete success (score=1.0, band=1)
    raw_perfect = {'status': 'ok', 'task_id': 'sokoban', 'band': 1, 'raw_reward': 155.0, 'reward': 1.0}
    ded, m = compute_item_deductions(raw_perfect, 1.0)
    assert abs(sum(ded.values()) - 0.0) < 1e-12
    assert ded.get('sokoban_progress') == 0.0
    assert ded.get(GATE_DEFECT_ID) == 0.0

    # Boundary 2: Complete failure (score=0.0, band=0)
    raw_zero = {'status': 'ok', 'task_id': 'sokoban', 'band': 0, 'raw_reward': 0.0, 'reward': 0.0}
    ded, m = compute_item_deductions(raw_zero, 0.0)
    assert abs(sum(ded.values()) - 1.0) < 1e-12
    assert ded[GATE_DEFECT_ID] == 1.0
    assert 'sokoban_progress' not in ded

    # Boundary 3: Band missing, raw_reward=1.8 (float)
    raw_float_band = {'status': 'ok', 'task_id': 'sokoban', 'raw_reward': 1.8, 'reward': 0.0116}
    ded, m = compute_item_deductions(raw_float_band, 0.0116)
    assert abs(sum(ded.values()) - (1.0 - 0.0116)) < 1e-12
    assert ded[GATE_DEFECT_ID] == 0.0
    assert abs(ded['sokoban_progress'] - (1.0 - 0.0116)) < 1e-12

    # Boundary 4: Band missing, raw_reward=0.8 (float, band 0)
    raw_float_band0 = {'status': 'ok', 'task_id': 'sokoban', 'raw_reward': 0.8, 'reward': 0.005}
    ded, m = compute_item_deductions(raw_float_band0, 0.005)
    assert abs(sum(ded.values()) - (1.0 - 0.005)) < 1e-12
    assert abs(ded[GATE_DEFECT_ID] - (1.0 - 0.005)) < 1e-12
    assert 'sokoban_progress' not in ded

    # Boundary 5: Error status with arbitrary reward
    raw_err = {'status': 'error', 'task_id': 'sokoban', 'error': 'docker killed', 'reward': 0.2}
    ded, m = compute_item_deductions(raw_err, 0.2)
    assert abs(sum(ded.values()) - 0.8) < 1e-12
    assert abs(ded[GATE_DEFECT_ID] - 0.8) < 1e-12


def test_adversarial_malformed_moves_log_verifier_resilience(tmp_path):
    """Verify that verifier handles malformed moves.log gracefully without crashing."""
    # We test the evaluator directly with synthetic submission containing malformed moves.log
    from c3_sandbox.l1.evaluation.lhtb import LHTBEvaluator
    from c3_sandbox.l1.core.contracts import EvaluationSpec

    cohort_dir = Path(__file__).resolve().parents[3] / 'L1_sandbox' / 'C3_Experimental_Sandbox' / 'runs' / 'lhtb_sokoban_qwen36_cohort'
    pkg_dir = cohort_dir / 'run_seed42'
    if not pkg_dir.is_dir():
        pytest.skip('run_seed42 baseline not present')

    pkg = L1Package(pkg_dir)
    spec = EvaluationSpec(
        'lhtb-verifier',
        pkg.root / 'benchmark/source',
        'tests/test_outputs.py',
        'sokoban',
        {'image': 'zli12321/lhtb-sokoban:20260615'},
    )
    evaluator = LHTBEvaluator({'image': 'zli12321/lhtb-sokoban:20260615', 'timeout_seconds': 30})

    # Case 1: Garbage move strings (not valid directions)
    sub1 = tmp_path / 'sub_garbage'
    sub1.mkdir(parents=True)
    (sub1 / 'moves.log').write_text("INVALID_COMMAND\nHELLO_SOKOBAN\n12345\n\n   \nFOOBAR\n")
    out1 = tmp_path / 'out_garbage'
    res1 = evaluator.evaluate(spec, sub1, out1)

    # Must produce valid structure with 0 reward, official_no_moves failure, without crashing
    assert res1['official_raw']['task_id'] == 'sokoban'
    assert res1['official_raw']['reward'] == 0.0
    assert res1['official_raw']['scoring_case'] == 'official_no_moves'
    assert res1['projection']['completion_rate'] == 0.0
    assert res1['projection']['official_dimensions']['passed'] is False

    # Case 2: Binary garbage / invalid UTF-8 bytes
    sub2 = tmp_path / 'sub_binary'
    sub2.mkdir(parents=True)
    (sub2 / 'moves.log').write_bytes(b"\x00\xff\xfe\x80\xaa\xbb\xcc\xdd\xee")
    out2 = tmp_path / 'out_binary'
    res2 = evaluator.evaluate(spec, sub2, out2)

    # Must handle unexpected test failure gracefully (status='error' or 'infra_error') without crashing python
    assert res2['official_raw']['task_id'] == 'sokoban'
    assert res2['official_raw']['status'] == 'error'
    assert res2['official_raw']['error'] == 'unexpected_official_test_failure'
    assert res2['projection']['evaluation_validity'] == 'invalid'

    # Verify that l2_sandbox.grades consumes this error payload safely without crashing
    from l2_sandbox.grades import _extract_dim1_pass, _extract_total_score
    proj2 = res2['projection']
    raw2 = res2['official_raw']
    s_val = float(proj2['completion_rate']) if isinstance(proj2.get('completion_rate'), (int, float)) else 0.0
    assert s_val == 0.0
    deductions, m_val = compute_item_deductions(raw2, s_val)
    assert deductions == {GATE_DEFECT_ID: 1.0}
    assert m_val == 1.0
    assert _extract_dim1_pass(raw2, proj2) is False
    assert _extract_total_score(raw2) == 0
