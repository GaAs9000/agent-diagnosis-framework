"""Comprehensive unit tests for L2 v0.4 Sokoban snapshot grader and defect tables."""
from __future__ import annotations

from pathlib import Path
import time
import pytest

from l2_sandbox.defects import build_defect_table
from l2_sandbox.evidence.l1 import L1Package
from l2_sandbox.facts.build import build_facts
from l2_sandbox.grades import (
    GATE_DEFECT_ID,
    compute_item_deductions,
    grade_episode_snapshots,
    select_snapshot_candidates,
)


def test_sokoban_candidate_discovery_tiers():
    """Verify tier ordering (submission/ -> runs/ -> other), deduplication, and filtering."""
    # Synthetic snapshot with multiple candidate files
    snapshot = {
        'files': [
            {'path': 'submission/moves.log', 'sha256': 'sha_sub_1', 'blob': 'blobs/1', 'size_bytes': 100},
            {'path': 'runs/moves.log', 'sha256': 'sha_runs_1', 'blob': 'blobs/2', 'size_bytes': 100},
            {'path': 'moves.log', 'sha256': 'sha_root_1', 'blob': 'blobs/3', 'size_bytes': 100},
            {'path': 'debug/agent_moves.log', 'sha256': 'sha_debug_1', 'blob': 'blobs/4', 'size_bytes': 50},
            # Should be filtered out:
            {'path': 'input/moves.log', 'sha256': 'sha_in', 'blob': 'blobs/5', 'size_bytes': 100},
            {'path': 'submission/~$moves.log', 'sha256': 'sha_tmp', 'blob': 'blobs/6', 'size_bytes': 100},
            {'path': 'runs/empty.log', 'sha256': 'sha_empty', 'blob': 'blobs/7', 'size_bytes': 0},
            {'path': 'other.txt', 'sha256': 'sha_txt', 'blob': 'blobs/8', 'size_bytes': 100},
        ]
    }
    last_written = {
        'submission/moves.log': 10,
        'runs/moves.log': 12,
        'moves.log': 5,
        'debug/agent_moves.log': 8,
    }

    cands = select_snapshot_candidates(
        snapshot,
        initial_input_hashes=set(),
        office_targets=['moves.log'],
        target_exts={'.log'},
        last_written_turn=last_written,
        task_id='sokoban',
        spec_kind='lhtb-verifier',
    )

    assert len(cands) == 4
    # Tier 0: submission/moves.log
    assert cands[0]['candidate_id'] == 'submission/moves.log'
    assert cands[0]['tier'] == 0
    assert cands[0]['in_submission'] is True
    assert cands[0]['staged'] == [('moves.log', 'submission/moves.log', 'sha_sub_1', 'blobs/1')]

    # Tier 1: runs/moves.log (turn 12) before moves.log (turn 5)
    assert cands[1]['candidate_id'] == 'runs/moves.log'
    assert cands[1]['tier'] == 1
    assert cands[1]['in_submission'] is False

    assert cands[2]['candidate_id'] == 'moves.log'
    assert cands[2]['tier'] == 1
    assert cands[2]['in_submission'] is False

    # Tier 2: debug/agent_moves.log
    assert cands[3]['candidate_id'] == 'debug/agent_moves.log'
    assert cands[3]['tier'] == 2
    assert cands[3]['in_submission'] is False

    # Content deduplication check: if runs/moves.log has identical sha256 to submission/moves.log,
    # and both map to staged 'moves.log', the second is deduplicated.
    snapshot_dup = {
        'files': [
            {'path': 'submission/moves.log', 'sha256': 'identical_sha', 'blob': 'blobs/1', 'size_bytes': 100},
            {'path': 'runs/moves.log', 'sha256': 'identical_sha', 'blob': 'blobs/1', 'size_bytes': 100},
        ]
    }
    cands_dup = select_snapshot_candidates(
        snapshot_dup,
        initial_input_hashes=set(),
        office_targets=['moves.log'],
        target_exts={'.log'},
        last_written_turn={},
        task_id='sokoban',
        spec_kind='lhtb-verifier',
    )
    assert len(cands_dup) == 1
    assert cands_dup[0]['candidate_id'] == 'submission/moves.log'

    # Empty candidate list when no matching non-empty files exist
    empty_snapshot = {
        'files': [
            {'path': 'input/moves.log', 'sha256': 'sha_in', 'blob': 'blobs/5', 'size_bytes': 100},
            {'path': 'moves.log', 'sha256': 'sha_zero', 'blob': 'blobs/6', 'size_bytes': 0},
        ]
    }
    cands_empty = select_snapshot_candidates(
        empty_snapshot,
        initial_input_hashes=set(),
        office_targets=['moves.log'],
        target_exts={'.log'},
        last_written_turn={},
        task_id='sokoban',
        spec_kind='lhtb-verifier',
    )
    assert cands_empty == []


def test_sokoban_compute_item_deductions():
    """Verify Sokoban deduction allocation for band 0, band 1, and error status."""
    # 1. Band 0 (< 1): Gate failed, all deduction 1 - score goes to dim1_gate
    raw_band0 = {
        'status': 'ok',
        'task_id': 'sokoban',
        'band': 0,
        'raw_reward': 0.5,
        'reward': 0.003226,
    }
    ded0, m0 = compute_item_deductions(raw_band0, 0.003226)
    assert m0 == 1.0
    assert abs(sum(ded0.values()) - (1.0 - 0.003226)) < 1e-12
    assert abs(ded0[GATE_DEFECT_ID] - (1.0 - 0.003226)) < 1e-12
    assert 'sokoban_progress' not in ded0

    # 2. Band 1 (>= 1): Gate passed, gate deduction 0.0, progress deduction 1 - score
    raw_band1 = {
        'status': 'ok',
        'task_id': 'sokoban',
        'band': 1,
        'raw_reward': 1.0,
        'reward': 0.65,
    }
    ded1, m1 = compute_item_deductions(raw_band1, 0.65)
    assert m1 == 1.0
    assert abs(sum(ded1.values()) - (1.0 - 0.65)) < 1e-12
    assert ded1[GATE_DEFECT_ID] == 0.0
    assert abs(ded1['sokoban_progress'] - 0.35) < 1e-12

    # 3. Error status: Gate failed, deduction 1.0 to dim1_gate
    raw_err = {
        'status': 'error',
        'task_id': 'sokoban',
        'error': 'timeout during execution',
    }
    ded_err, m_err = compute_item_deductions(raw_err, 0.0)
    assert m_err == 1.0
    assert ded_err == {GATE_DEFECT_ID: 1.0}


def test_sokoban_snapshot_grading_and_caching(tmp_path):
    """Test live snapshot grading and caching on baseline Sokoban run_seed42."""
    cohort_dir = Path(__file__).resolve().parents[3] / 'L1_sandbox' / 'C3_Experimental_Sandbox' / 'runs' / 'lhtb_sokoban_qwen36_cohort'
    pkg_dir = cohort_dir / 'run_seed42'
    if not pkg_dir.is_dir():
        pytest.skip('run_seed42 baseline not present')

    pkg = L1Package(pkg_dir)
    cache_dir = tmp_path / 'cache'

    # Run 1: Cold evaluation
    rows = grade_episode_snapshots(pkg, cache_dir=cache_dir)
    assert len(rows) == 22  # row 0 + k = 0..20

    # Row 0: Terminal submission
    term_row = rows[0]
    assert term_row['kind'] == 'terminal_submission'
    assert term_row['task_id'] == 'lhtb:sokoban'
    assert abs(term_row['score'] - 0.003226) < 1e-6
    assert term_row['dim1_pass'] is False
    assert term_row['band'] == 0
    assert term_row['raw_reward'] == 0.5
    assert abs(term_row['deductions'][GATE_DEFECT_ID] - 0.996774) < 1e-6

    # Row 1 (k=0): Before agent writes moves.log, status is 'no_candidate'
    cp0 = rows[1]
    assert cp0['kind'] == 'checkpoint'
    assert cp0['k'] == 0
    assert cp0['candidate_status'] == 'no_candidate'
    assert cp0['candidate'] == []
    assert cp0['score'] == 0.0

    # Row 5 (k=4): First appearance of moves.log candidate
    cp4 = rows[5]
    assert cp4['kind'] == 'checkpoint'
    assert cp4['k'] == 4
    assert cp4['candidate_status'] == 'ok'
    assert cp4['candidate'] == ['submission/moves.log']
    assert abs(cp4['score'] - 0.003226) < 1e-6

    # Row 21 (k=20): Terminal checkpoint
    cp20 = rows[21]
    assert cp20['kind'] == 'checkpoint'
    assert cp20['k'] == 20
    assert cp20['candidate_status'] == 'ok'
    assert abs(cp20['score'] - 0.003226) < 1e-6

    # Verify cache files were created in cache_dir / 'grades'
    cache_files = list((cache_dir / 'grades').glob('*.json'))
    assert len(cache_files) > 0

    # Run 2: Cached evaluation (< 1.0s)
    t0 = time.monotonic()
    rows_cached = grade_episode_snapshots(pkg, cache_dir=cache_dir)
    elapsed = time.monotonic() - t0
    assert elapsed < 1.0
    assert len(rows_cached) == len(rows)

    # Validate cached scores and deductions match identically
    for r1, r2 in zip(rows[1:], rows_cached[1:], strict=True):
        assert r1['k'] == r2['k']
        assert r1['candidate'] == r2['candidate']
        assert abs(r1['score'] - r2['score']) < 1e-12
        assert r1['deductions'] == r2['deductions']


def test_sokoban_defect_table_generation():
    """Verify defect table generation on real run_seed42 and synthetic trajectories."""
    cohort_dir = Path(__file__).resolve().parents[3] / 'L1_sandbox' / 'C3_Experimental_Sandbox' / 'runs' / 'lhtb_sokoban_qwen36_cohort'
    pkg_dir = cohort_dir / 'run_seed42'
    if not pkg_dir.is_dir():
        pytest.skip('run_seed42 baseline not present')

    pkg = L1Package(pkg_dir)
    facts = build_facts(pkg)

    # 1. Real cohort run_seed42: Never achieved pass threshold, gap g = 0
    real_rows = [
        {
            'kind': 'terminal_submission',
            'episode_id': pkg.episode_id,
            'task_id': pkg.family_id,
            'k': 20,
            'score': 0.003226,
            'dim1_pass': False,
            'deductions': {GATE_DEFECT_ID: 0.996774},
        },
        *[
            {
                'kind': 'checkpoint',
                'k': k,
                'candidate_status': 'ok',
                'candidate': ['submission/moves.log'],
                'score': 0.003226,
                'deductions': {GATE_DEFECT_ID: 0.996774},
            }
            for k in range(21)
        ],
    ]
    dt_real = build_defect_table(pkg, real_rows, facts)
    assert dt_real['terminal_score_y'] == 0.003226
    assert dt_real['delivery_gap_g'] == 0.0
    assert dt_real['undelivered_gap'] == 0.0
    assert abs(dt_real['total_deduction'] - 0.996774) < 1e-6
    assert dt_real['deduction_by_kind']['never'] == 0.996774
    assert dt_real['defects'][0]['defect'] == GATE_DEFECT_ID
    assert dt_real['defects'][0]['kind'] == 'never'
    assert dt_real['defects'][0]['turn'] is None

    # 2. Synthetic trajectory with undelivered defect (runs/moves.log got 0.5, submission got 0.003226)
    synth_undelivered_rows = [
        {
            'kind': 'terminal_submission',
            'episode_id': pkg.episode_id,
            'task_id': pkg.family_id,
            'k': 5,
            'score': 0.003226,
            'dim1_pass': False,
            'deductions': {GATE_DEFECT_ID: 0.996774},
        },
        {'kind': 'checkpoint', 'k': 0, 'candidate_status': 'ok', 'candidate': ['submission/moves.log'], 'score': 0.003226, 'deductions': {GATE_DEFECT_ID: 0.996774}},
        {'kind': 'checkpoint', 'k': 1, 'candidate_status': 'ok', 'candidate': ['submission/moves.log'], 'score': 0.003226, 'deductions': {GATE_DEFECT_ID: 0.996774}},
        {'kind': 'checkpoint', 'k': 2, 'candidate_status': 'ok', 'candidate': ['runs/moves.log'], 'score': 0.5, 'deductions': {'sokoban_progress': 0.5, GATE_DEFECT_ID: 0.0}},
        {'kind': 'checkpoint', 'k': 3, 'candidate_status': 'ok', 'candidate': ['runs/moves.log'], 'score': 0.5, 'deductions': {'sokoban_progress': 0.5, GATE_DEFECT_ID: 0.0}},
        {'kind': 'checkpoint', 'k': 4, 'candidate_status': 'ok', 'candidate': ['runs/moves.log'], 'score': 0.5, 'deductions': {'sokoban_progress': 0.5, GATE_DEFECT_ID: 0.0}},
        {'kind': 'checkpoint', 'k': 5, 'candidate_status': 'ok', 'candidate': ['runs/moves.log'], 'score': 0.5, 'deductions': {'sokoban_progress': 0.5, GATE_DEFECT_ID: 0.0}},
    ]
    dt_undeliv = build_defect_table(pkg, synth_undelivered_rows, facts)
    expected_g = 0.5 - 0.003226
    assert abs(dt_undeliv['delivery_gap_g'] - expected_g) < 1e-6
    assert abs(dt_undeliv['undelivered_gap'] - expected_g) < 1e-6
    assert any(d['kind'] == 'undelivered' for d in dt_undeliv['defects'])
    undeliv_defect = next(d for d in dt_undeliv['defects'] if d['kind'] == 'undelivered')
    assert undeliv_defect['turn'] == 2  # Earliest k where score was achieved and sustained

    # 3. Synthetic trajectory with broken defect (achieved 0.8 at k=1, dropped to 0.0 at k=2)
    synth_broken_rows = [
        {
            'kind': 'terminal_submission',
            'episode_id': pkg.episode_id,
            'task_id': pkg.family_id,
            'k': 3,
            'score': 0.0,
            'dim1_pass': False,
            'deductions': {GATE_DEFECT_ID: 1.0},
        },
        {'kind': 'checkpoint', 'k': 0, 'candidate_status': 'ok', 'candidate': ['moves.log'], 'score': 0.0, 'deductions': {GATE_DEFECT_ID: 1.0}},
        {'kind': 'checkpoint', 'k': 1, 'candidate_status': 'ok', 'candidate': ['moves.log'], 'score': 0.8, 'deductions': {'sokoban_progress': 0.2, GATE_DEFECT_ID: 0.0}},
        {'kind': 'checkpoint', 'k': 2, 'candidate_status': 'ok', 'candidate': ['moves.log'], 'score': 0.0, 'deductions': {GATE_DEFECT_ID: 1.0}},
        {'kind': 'checkpoint', 'k': 3, 'candidate_status': 'ok', 'candidate': ['moves.log'], 'score': 0.0, 'deductions': {GATE_DEFECT_ID: 1.0}},
    ]
    dt_broken = build_defect_table(pkg, synth_broken_rows, facts)
    assert any(d['kind'] == 'broken' for d in dt_broken['defects'])
    broken_defect = next(d for d in dt_broken['defects'] if d['kind'] == 'broken')
    assert broken_defect['turn'] == 2  # Broke at turn 2 (was ok at turn 1)
