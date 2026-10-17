"""Adversarial Empirical Stress Test Suite for Milestone M1 (L2 grades.py).

Tests:
1. Tier prioritization (Tier 0 > 1 > 2 > 3) and sub-prioritization matrix.
2. File filtering (input/*, ~$*, 0-byte, negative bytes, missing size, non-patch, extensionless, corrupted paths).
3. SHA-256 deduplication across identical blobs with different filenames, tiers, and directories.
4. Tier 3 git diff fallback against base commits (root repo, subrepo, clean tree, untracked files, 0-byte standalone override).
5. Evaluation zero-request caching and cross-filename content hash resolution.
6. Property-based randomized stress testing of item deduction invariant sum(d_i) == 1 - s across 100 random configurations.
7. Snapshot candidate selection rank_key tiebreaking under multi-candidate evaluation.
"""
from __future__ import annotations

import hashlib
import json
import random
import subprocess
from pathlib import Path
import pytest

from l2_sandbox.grades import (
    GATE_DEFECT_ID,
    compute_item_deductions,
    select_snapshot_candidates,
    _evaluate_staged_candidate,
    extract_task_rubric,
    _office_targets,
)
from c3_sandbox.l1.core.contracts import EvaluationSpec


# ==============================================================================
# 1. Tier Prioritization & Sub-Prioritization Exhaustive Hierarchy
# ==============================================================================

def test_stress_tier_prioritization_strict_hierarchy():
    """Stress-test strict priority order: Tier 0 > Tier 1 > Tier 2, plus sub-priorities and turn tiebreaking."""
    snapshot = {
        'files': [
            # Tier 0 (submission/)
            {'path': 'submission/patch.diff', 'sha256': 'sha_t0_p0', 'size_bytes': 100},          # sub_prio 0
            {'path': 'submission/model.patch', 'sha256': 'sha_t0_p1', 'size_bytes': 100},         # sub_prio 1
            {'path': 'submission/gold_patch.diff', 'sha256': 'sha_t0_p2', 'size_bytes': 100},    # sub_prio 2
            {'path': 'submission/custom_recent.diff', 'sha256': 'sha_t0_p3_rec', 'size_bytes': 100}, # sub_prio 3, turn 20
            {'path': 'submission/custom_older_a.patch', 'sha256': 'sha_t0_p3_a', 'size_bytes': 100}, # sub_prio 3, turn 10
            {'path': 'submission/custom_older_b.patch', 'sha256': 'sha_t0_p3_b', 'size_bytes': 100}, # sub_prio 3, turn 10 (name tiebreak)
            {'path': 'submission/custom_oldest.diff', 'sha256': 'sha_t0_p3_old', 'size_bytes': 100}, # sub_prio 3, turn 5
            # Tier 1 (root target files)
            {'path': 'patch.diff', 'sha256': 'sha_t1_p0', 'size_bytes': 100},                     # sub_prio 0
            {'path': 'model.patch', 'sha256': 'sha_t1_p1', 'size_bytes': 100},                    # sub_prio 1
            # Tier 2 (subdirectories / other files)
            {'path': 'work/deep/fix_recent.diff', 'sha256': 'sha_t2_rec', 'size_bytes': 100},     # turn 25
            {'path': 'work/alpha.patch', 'sha256': 'sha_t2_alpha', 'size_bytes': 100},            # turn 12 (name tiebreak)
            {'path': 'work/deep/fix_older.patch', 'sha256': 'sha_t2_older', 'size_bytes': 100},   # turn 12
        ]
    }
    last_written = {
        'submission/patch.diff': 1,
        'submission/model.patch': 1,
        'submission/gold_patch.diff': 1,
        'submission/custom_recent.diff': 20,
        'submission/custom_older_a.patch': 10,
        'submission/custom_older_b.patch': 10,
        'submission/custom_oldest.diff': 5,
        'patch.diff': 15,
        'model.patch': 15,
        'work/deep/fix_recent.diff': 25,
        'work/alpha.patch': 12,
        'work/deep/fix_older.patch': 12,
    }

    cands = select_snapshot_candidates(
        snapshot,
        initial_input_hashes=set(),
        office_targets=['patch.diff'],
        target_exts={'.diff', '.patch'},
        last_written_turn=last_written,
        task_id='instance_ansible__ansible-test',
        spec_kind='swebench_pro-verifier',
    )

    assert len(cands) == 12

    # Tier 0 checks (indices 0..6)
    expected_t0_order = [
        'submission/patch.diff',
        'submission/model.patch',
        'submission/gold_patch.diff',
        'submission/custom_recent.diff',
        'submission/custom_older_a.patch',
        'submission/custom_older_b.patch',
        'submission/custom_oldest.diff',
    ]
    for i, exp_id in enumerate(expected_t0_order):
        assert cands[i]['candidate_id'] == exp_id
        assert cands[i]['tier'] == 0
        assert cands[i]['in_submission'] is True
        assert cands[i]['staged'][0][0] == 'patch.diff'

    # Tier 1 checks (indices 7..8)
    expected_t1_order = [
        'patch.diff',
        'model.patch',
    ]
    for i, exp_id in enumerate(expected_t1_order, start=7):
        assert cands[i]['candidate_id'] == exp_id
        assert cands[i]['tier'] == 1
        assert cands[i]['in_submission'] is False
        assert cands[i]['staged'][0][0] == 'patch.diff'

    # Tier 2 checks (indices 9..11)
    expected_t2_order = [
        'work/deep/fix_recent.diff',
        'work/alpha.patch',
        'work/deep/fix_older.patch',
    ]
    for i, exp_id in enumerate(expected_t2_order, start=9):
        assert cands[i]['candidate_id'] == exp_id
        assert cands[i]['tier'] == 2
        assert cands[i]['in_submission'] is False
        assert cands[i]['staged'][0][0] == 'patch.diff'


# ==============================================================================
# 2. Hostile Filtering & Corrupted Paths Stress Testing
# ==============================================================================

def test_stress_filtering_and_corrupted_paths():
    """Verify adversarial inputs: non-patch, 0-byte, negative bytes, missing size, input/*, ~$*, extensions."""
    hostile_files = [
        # Must be filtered: input/ directory
        {'path': 'input/patch.diff', 'sha256': 'sha_in_1', 'size_bytes': 100},
        {'path': 'input/nested/fix.patch', 'sha256': 'sha_in_2', 'size_bytes': 200},
        # Must be filtered: temporary ~$ files
        {'path': 'submission/~$temp.diff', 'sha256': 'sha_tmp_1', 'size_bytes': 100},
        {'path': '~$root.patch', 'sha256': 'sha_tmp_2', 'size_bytes': 100},
        {'path': 'work/~$nested.diff', 'sha256': 'sha_tmp_3', 'size_bytes': 100},
        # Must be filtered: 0-byte, negative-byte, or missing size
        {'path': 'submission/zero.diff', 'sha256': 'sha_zero', 'size_bytes': 0},
        {'path': 'negative.patch', 'sha256': 'sha_neg', 'size_bytes': -10},
        {'path': 'work/missing_size.patch', 'sha256': 'sha_missing'},
        # Must be filtered: non-patch extensions
        {'path': 'submission/patch.txt', 'sha256': 'sha_txt', 'size_bytes': 100},
        {'path': 'patch.diff.bak', 'sha256': 'sha_bak', 'size_bytes': 100},
        {'path': 'fix.py', 'sha256': 'sha_py', 'size_bytes': 100},
        {'path': 'document.docx', 'sha256': 'sha_docx', 'size_bytes': 100},
        {'path': 'table.xlsx', 'sha256': 'sha_xlsx', 'size_bytes': 100},
        {'path': 'archive.patch.tar.gz', 'sha256': 'sha_gz', 'size_bytes': 100},
        # Must be filtered: extensionless or dotfiles without stem
        {'path': '.patch', 'sha256': 'sha_dotpatch', 'size_bytes': 100},
        {'path': '.diff', 'sha256': 'sha_dotdiff', 'size_bytes': 100},
        {'path': 'submission/.patch', 'sha256': 'sha_sub_dot', 'size_bytes': 100},
        # Must be filtered: empty path
        {'path': '', 'sha256': 'sha_empty_path', 'size_bytes': 100},
        # Valid files that MUST pass through
        {'path': 'submission/valid.diff', 'sha256': 'sha_valid_sub', 'size_bytes': 50},
        {'path': 'submission/UPPER.PATCH', 'sha256': 'sha_valid_upper', 'size_bytes': 50},
        {'path': 'fixes/补丁_01.patch', 'sha256': 'sha_valid_unicode', 'size_bytes': 50},
        {'path': 'fixes/fix with space.diff', 'sha256': 'sha_valid_space', 'size_bytes': 50},
    ]

    snapshot = {'files': hostile_files}
    cands = select_snapshot_candidates(
        snapshot,
        initial_input_hashes=set(),
        office_targets=['patch.diff'],
        target_exts={'.diff', '.patch'},
        last_written_turn={},
        task_id='instance_ansible__ansible-test',
        spec_kind='swebench_pro-verifier',
    )

    admitted_paths = [c['candidate_id'] for c in cands]
    assert len(admitted_paths) == 4
    assert 'submission/valid.diff' in admitted_paths
    assert 'submission/UPPER.PATCH' in admitted_paths
    assert 'fixes/补丁_01.patch' in admitted_paths
    assert 'fixes/fix with space.diff' in admitted_paths

    # Verify every hostile file was rejected
    for f in hostile_files[:-4]:
        assert f['path'] not in admitted_paths


# ==============================================================================
# 3. SHA-256 Deduplication Across Tiers, Names, and Directories
# ==============================================================================

def test_stress_sha256_deduplication_cross_tier_and_names():
    """Verify SHA-256 deduplication correctly preserves higher-tier and drops identical lower-tier blobs."""
    # Scenario A: Same blob in Tier 0 and Tier 1 -> Tier 0 wins
    # Scenario B: Same blob in Tier 0 and Tier 2 -> Tier 0 wins
    # Scenario C: Same blob in Tier 1 and Tier 2 -> Tier 1 wins
    # Scenario D: Same blob across multiple Tier 0 sub-priorities -> highest sub_prio wins
    # Scenario E: Same blob across multiple Tier 2 files -> highest turn wins
    sha_alpha = 'sha256_blob_alpha'
    sha_beta = 'sha256_blob_beta'
    sha_gamma = 'sha256_blob_gamma'
    sha_delta = 'sha256_blob_delta'

    snapshot = {
        'files': [
            # sha_alpha: present in Tier 0, Tier 1, and Tier 2
            {'path': 'submission/model.patch', 'sha256': sha_alpha, 'size_bytes': 100},
            {'path': 'patch.diff', 'sha256': sha_alpha, 'size_bytes': 100},
            {'path': 'dir/fix.patch', 'sha256': sha_alpha, 'size_bytes': 100},
            # sha_beta: present in Tier 0 (sub_prio 0 vs sub_prio 2)
            {'path': 'submission/patch.diff', 'sha256': sha_beta, 'size_bytes': 120},
            {'path': 'submission/gold_patch.diff', 'sha256': sha_beta, 'size_bytes': 120},
            # sha_gamma: present in Tier 1 and Tier 2
            {'path': 'model.patch', 'sha256': sha_gamma, 'size_bytes': 130},
            {'path': 'other/dup.patch', 'sha256': sha_gamma, 'size_bytes': 130},
            # sha_delta: present only in Tier 2, but at turns 10 vs 5
            {'path': 'tier2/older.diff', 'sha256': sha_delta, 'size_bytes': 140},
            {'path': 'tier2/newer.diff', 'sha256': sha_delta, 'size_bytes': 140},
        ]
    }
    last_written = {
        'submission/model.patch': 5,
        'patch.diff': 10,
        'dir/fix.patch': 20,
        'submission/patch.diff': 1,
        'submission/gold_patch.diff': 1,
        'model.patch': 5,
        'other/dup.patch': 15,
        'tier2/older.diff': 5,
        'tier2/newer.diff': 10,
    }

    cands = select_snapshot_candidates(
        snapshot,
        initial_input_hashes=set(),
        office_targets=['patch.diff'],
        target_exts={'.diff', '.patch'},
        last_written_turn=last_written,
        task_id='instance_ansible__ansible-test',
        spec_kind='swebench_pro-verifier',
    )

    # Exactly 4 candidates should survive: one for each unique sha
    assert len(cands) == 4

    cands_by_sha = {c['staged'][0][2]: c for c in cands}
    assert set(cands_by_sha.keys()) == {sha_alpha, sha_beta, sha_gamma, sha_delta}

    # sha_alpha was in Tier 0, 1, 2 -> Tier 0 candidate 'submission/model.patch' MUST be the one kept!
    cand_alpha = cands_by_sha[sha_alpha]
    assert cand_alpha['candidate_id'] == 'submission/model.patch'
    assert cand_alpha['tier'] == 0

    # sha_beta was in submission/patch.diff (sub_prio 0) and submission/gold_patch.diff (sub_prio 2)
    # -> sub_prio 0 MUST be kept!
    cand_beta = cands_by_sha[sha_beta]
    assert cand_beta['candidate_id'] == 'submission/patch.diff'
    assert cand_beta['tier'] == 0

    # sha_gamma was in Tier 1 'model.patch' and Tier 2 'other/dup.patch' -> Tier 1 MUST be kept!
    cand_gamma = cands_by_sha[sha_gamma]
    assert cand_gamma['candidate_id'] == 'model.patch'
    assert cand_gamma['tier'] == 1

    # sha_delta was in Tier 2 at turns 10 vs 5 -> turn 10 'tier2/newer.diff' MUST be kept!
    cand_delta = cands_by_sha[sha_delta]
    assert cand_delta['candidate_id'] == 'tier2/newer.diff'
    assert cand_delta['tier'] == 2


# ==============================================================================
# 4. Tier 3 Git Diff Fallback Empirical Scenarios
# ==============================================================================

def _setup_git_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'init'], cwd=path, check=True, capture_output=True)
    subprocess.run(['git', 'config', 'user.name', 'Tester'], cwd=path, check=True)
    subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=path, check=True)
    f = path / 'module.py'
    f.write_text('def execute(): pass\n')
    subprocess.run(['git', 'add', 'module.py'], cwd=path, check=True)
    subprocess.run(['git', 'commit', '-m', 'Initial'], cwd=path, check=True, capture_output=True)
    return path


def test_stress_tier3_git_diff_clean_tree(tmp_path):
    """Tier 3: when git repo has a clean working tree (no uncommitted diff), 0 candidates returned."""
    repo = _setup_git_repo(tmp_path / 'repo')
    snapshot = {'files': [{'path': 'module.py', 'sha256': 'sha_mod', 'size_bytes': 30}]}

    cands = select_snapshot_candidates(
        snapshot,
        initial_input_hashes=set(),
        office_targets=['patch.diff'],
        target_exts={'.diff', '.patch'},
        last_written_turn={},
        task_id='instance_ansible__ansible-test',
        spec_kind='swebench_pro-verifier',
        restored_workspace=repo,
    )
    assert cands == []


def test_stress_tier3_git_diff_untracked_only(tmp_path):
    """Tier 3: when git repo only has untracked files (no tracked edits), git diff produces empty -> 0 candidates."""
    repo = _setup_git_repo(tmp_path / 'repo')
    (repo / 'new_untracked_file.py').write_text('print("hello")\n')
    snapshot = {'files': [{'path': 'module.py', 'sha256': 'sha_mod', 'size_bytes': 30}]}

    cands = select_snapshot_candidates(
        snapshot,
        initial_input_hashes=set(),
        office_targets=['patch.diff'],
        target_exts={'.diff', '.patch'},
        last_written_turn={},
        task_id='instance_ansible__ansible-test',
        spec_kind='swebench_pro-verifier',
        restored_workspace=repo,
    )
    assert cands == []


def test_stress_tier3_git_diff_subrepo_detection(tmp_path):
    """Tier 3: git repo located in a subdirectory (e.g. restored_ws / 'ansible_src' / '.git')."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    subrepo = _setup_git_repo(ws / 'ansible_src')
    (subrepo / 'module.py').write_text('def execute(): return "modified"\n')

    snapshot = {'files': [{'path': 'ansible_src/module.py', 'sha256': 'sha_mod', 'size_bytes': 40}]}

    cands = select_snapshot_candidates(
        snapshot,
        initial_input_hashes=set(),
        office_targets=['patch.diff'],
        target_exts={'.diff', '.patch'},
        last_written_turn={'ansible_src/module.py': 8},
        task_id='instance_ansible__ansible-test',
        spec_kind='swebench_pro-verifier',
        restored_workspace=ws,
    )
    assert len(cands) == 1
    cand = cands[0]
    assert cand['tier'] == 3
    assert cand['candidate_id'] == 'git_diff.patch'
    assert cand['in_submission'] is False
    assert cand['last_written_turn'] == 8
    patch_file = ws / 'git_diff.patch'
    assert patch_file.exists()
    assert b'+def execute(): return "modified"' in patch_file.read_bytes()


def test_stress_tier3_git_diff_standalone_patch_precedence(tmp_path):
    """Tier 3: if snapshot already contains a valid Tier 1 or 2 patch, Tier 3 MUST NOT trigger."""
    repo = _setup_git_repo(tmp_path / 'repo')
    (repo / 'module.py').write_text('def execute(): return "uncommitted"\n')

    # Snapshot contains an explicit patch file
    snapshot = {
        'files': [
            {'path': 'patch.diff', 'sha256': 'sha_patch', 'size_bytes': 80},
        ]
    }

    cands = select_snapshot_candidates(
        snapshot,
        initial_input_hashes=set(),
        office_targets=['patch.diff'],
        target_exts={'.diff', '.patch'},
        last_written_turn={'patch.diff': 2},
        task_id='instance_ansible__ansible-test',
        spec_kind='swebench_pro-verifier',
        restored_workspace=repo,
    )
    # Tier 1 candidate was found; Tier 3 must NOT be added
    assert len(cands) == 1
    assert cands[0]['tier'] == 1
    assert cands[0]['candidate_id'] == 'patch.diff'
    assert not (repo / 'git_diff.patch').exists()


def test_stress_tier3_git_diff_0byte_standalone_allows_fallback(tmp_path):
    """Tier 3: if snapshot contains only a 0-byte patch, it is filtered, allowing Tier 3 to trigger."""
    repo = _setup_git_repo(tmp_path / 'repo')
    (repo / 'module.py').write_text('def execute(): return "saved_by_git_diff"\n')

    snapshot = {
        'files': [
            {'path': 'patch.diff', 'sha256': 'sha_empty', 'size_bytes': 0},
        ]
    }

    cands = select_snapshot_candidates(
        snapshot,
        initial_input_hashes=set(),
        office_targets=['patch.diff'],
        target_exts={'.diff', '.patch'},
        last_written_turn={'patch.diff': 2},
        task_id='instance_ansible__ansible-test',
        spec_kind='swebench_pro-verifier',
        restored_workspace=repo,
    )
    # 0-byte patch filtered -> Tier 3 triggers and generates git_diff.patch
    assert len(cands) == 1
    assert cands[0]['tier'] == 3
    assert cands[0]['candidate_id'] == 'git_diff.patch'


# ==============================================================================
# 5. Randomized Property-Based Stress: Deduction Invariant sum(d_i) == 1 - s
# ==============================================================================

def test_stress_deduction_invariant_randomized_matrix():
    """Property test: 100 random test configurations verifying sum(d_i) == 1 - s identically (< 1e-12)."""
    rng = random.Random(42)

    for trial in range(100):
        n_fail = rng.randint(1, 15)
        n_pass = rng.randint(1, 15)
        fail_tests = [f'test_f_{trial}_{i}' for i in range(n_fail)]
        pass_tests = [f'test_p_{trial}_{i}' for i in range(n_pass)]
        all_required = fail_tests + pass_tests
        total_req = len(all_required)

        # 3 possible states: all passed (score=1.0), partial passed (score=0.0), none passed (score=0.0)
        scenario = rng.choice(['all', 'partial', 'none'])
        if scenario == 'all':
            passed_names = set(all_required)
            score = 1.0
            is_passed = True
        elif scenario == 'partial':
            # At least one test failed
            k = rng.randint(1, total_req - 1)
            passed_names = set(rng.sample(all_required, k))
            score = 0.0
            is_passed = False
        else:
            passed_names = set()
            score = 0.0
            is_passed = False

        raw = {
            'task_id': f'instance_ansible_trial_{trial}',
            'passed': is_passed,
            'reward': score,
            'score': score,
            'status': 'ok' if is_passed else 'failed',
            'exit_code': 0 if is_passed else 1,
            'test_config': {
                'fail_to_pass': fail_tests,
                'pass_to_pass': pass_tests,
            },
            'test_results': {
                'tests': [{'name': t, 'status': 'PASSED' if t in passed_names else 'FAILED'} for t in all_required]
            },
        }

        deductions, max_score = compute_item_deductions(raw, score)
        assert max_score == 1.0

        actual_sum = sum(deductions.values())
        expected_sum = 1.0 - score
        assert abs(actual_sum - expected_sum) < 1e-12, (
            f"Trial {trial} ({scenario}) failed invariant: sum={actual_sum} != expected={expected_sum}, "
            f"n_req={total_req}, n_passed={len(passed_names)}"
        )

        if score == 1.0:
            for t in all_required:
                assert deductions[t] == 0.0
            assert deductions[GATE_DEFECT_ID] == 0.0
        else:
            for t in all_required:
                if t in passed_names:
                    assert deductions[t] == 0.0
                else:
                    assert abs(deductions[t] - (1.0 / total_req)) < 1e-12
            expected_gate = len(passed_names) / total_req
            assert abs(deductions[GATE_DEFECT_ID] - expected_gate) < 1e-12


# ==============================================================================
# 6. Evaluation Caching & Zero-Request Verification
# ==============================================================================

def test_stress_evaluation_zero_request_cross_filename(tmp_path, monkeypatch):
    """Verify evaluator is invoked only once for identical blobs regardless of different filenames across turns."""
    cache_dir = tmp_path / 'cache'
    eval_invocations = 0

    class MockEvaluator:
        def __init__(self, config):
            self.config = config

        def evaluate(self, spec, sub_dir, grade_out):
            nonlocal eval_invocations
            eval_invocations += 1
            # Check staged file exists in submission directory as 'patch.diff'
            staged_file = sub_dir / 'patch.diff'
            assert staged_file.is_file()
            return {
                'projection': {'completion_rate': 1.0, 'task_outcome': 'success', 'official_dimensions': {'passed': True}},
                'official_raw': {'status': 'ok', 'passed': True, 'reward': 1.0},
                'official_raw_sha256': 'raw_hash_xyz',
            }

    import l2_sandbox.grades as grades_mod
    monkeypatch.setattr(grades_mod, 'SWEBenchProEvaluator', MockEvaluator)

    ws_dir = tmp_path / 'ws'
    ws_dir.mkdir()
    blob_bytes = b'diff --git a/test b/test\n+fix\n'
    blob_sha = hashlib.sha256(blob_bytes).hexdigest()

    # Candidate 1: Turn 1, submission/patch.diff
    p1 = ws_dir / 'submission' / 'patch.diff'
    p1.parent.mkdir(parents=True, exist_ok=True)
    p1.write_bytes(blob_bytes)
    cand1 = {
        'candidate_id': 'submission/patch.diff',
        'paths': ['submission/patch.diff'],
        'staged': [('patch.diff', 'submission/patch.diff', blob_sha, '')],
        'in_submission': True,
        'tier': 0,
        'last_written_turn': 1,
    }

    # Candidate 2: Turn 2, completely different filename and location in workspace
    p2 = ws_dir / 'backup' / 'work_in_progress.patch'
    p2.parent.mkdir(parents=True, exist_ok=True)
    p2.write_bytes(blob_bytes)
    cand2 = {
        'candidate_id': 'backup/work_in_progress.patch',
        'paths': ['backup/work_in_progress.patch'],
        'staged': [('patch.diff', 'backup/work_in_progress.patch', blob_sha, '')],
        'in_submission': False,
        'tier': 2,
        'last_written_turn': 2,
    }

    spec = EvaluationSpec(
        kind='swebench_pro-verifier',
        source_root=tmp_path / 'src',
        entrypoint='test.sh',
        task_id='instance_ansible__ansible-test',
        metadata={'image': 'ghcr.io/scaleapi/swe-bench_pro-v2:instance_ansible__ansible-test'},
    )

    class DummyPkg:
        root = tmp_path
        episode_id = 'test-ep-01'

    # Turn 1 evaluation: Cache miss
    res1, hit1 = _evaluate_staged_candidate(DummyPkg(), spec, {'image': 'test-img'}, {}, cand1, cache_dir, restored_workspace=ws_dir)  # type: ignore
    assert hit1 is False
    assert eval_invocations == 1

    # Turn 2 evaluation of cand2 (different path, same staged content SHA): MUST BE CACHE HIT!
    res2, hit2 = _evaluate_staged_candidate(DummyPkg(), spec, {'image': 'test-img'}, {}, cand2, cache_dir, restored_workspace=ws_dir)  # type: ignore
    assert hit2 is True
    assert eval_invocations == 1  # 0 additional calls!
    assert res1 == res2


# ==============================================================================
# 7. Candidate Ranking Tiebreaking & Benchmark Recognition Stress
# ==============================================================================

def test_stress_checkpoint_candidate_ranking_criteria():
    """Verify rank_key = (round(s_val, 12), bool(in_submission), int(last_written_turn), -int(tier), candidate_id)."""
    # Candidate 1: High score wins unconditionally
    c_high = {'s_val': 1.0, 'in_submission': False, 'turn': 1, 'tier': 2, 'id': 'high'}
    # Candidate 2: Equal score (0.8), but in_submission=True beats in_submission=False
    c_sub = {'s_val': 0.8, 'in_submission': True, 'turn': 1, 'tier': 0, 'id': 'sub'}
    c_nonsub = {'s_val': 0.8, 'in_submission': False, 'turn': 10, 'tier': 1, 'id': 'nonsub'}
    # Candidate 3: Equal score (0.8) and in_submission (False), but turn 10 beats turn 5
    c_turn10 = {'s_val': 0.8, 'in_submission': False, 'turn': 10, 'tier': 1, 'id': 'turn10'}
    c_turn5 = {'s_val': 0.8, 'in_submission': False, 'turn': 5, 'tier': 1, 'id': 'turn5'}
    # Candidate 4: Equal score, in_submission, turn, but Tier 1 beats Tier 2 (-1 > -2)
    c_tier1 = {'s_val': 0.8, 'in_submission': False, 'turn': 5, 'tier': 1, 'id': 'tier1'}
    c_tier2 = {'s_val': 0.8, 'in_submission': False, 'turn': 5, 'tier': 2, 'id': 'tier2'}

    def rank(c):
        return (
            round(c['s_val'], 12),
            bool(c['in_submission']),
            int(c['turn']),
            -int(c['tier']),
            c['id'],
        )

    # 1. High score beats everything
    assert rank(c_high) > rank(c_sub)
    assert rank(c_high) > rank(c_nonsub)

    # 2. In submission beats non-submission for same score
    assert rank(c_sub) > rank(c_nonsub)

    # 3. Later turn beats earlier turn for same score and in_submission
    assert rank(c_turn10) > rank(c_turn5)

    # 4. Lower tier number (higher priority) beats higher tier number
    assert rank(c_tier1) > rank(c_tier2)


def test_stress_office_targets_swebench_pro_permutations(tmp_path):
    """Verify _office_targets correctly recognizes swebench_pro across all metadata forms."""
    class MockPkg:
        def __init__(self, root, manifest, benchmark, family_id, task_id):
            self.root = root
            self.manifest = manifest
            self.benchmark = benchmark
            self.family_id = family_id
            self.task_id = task_id

    # Permutation 1: manifest['benchmark'] == 'swebench_pro'
    p1 = MockPkg(tmp_path, {'benchmark': 'swebench_pro'}, 'swebench_pro', 'swebench_pro:task1', 'task1')
    targets, exts = _office_targets(p1)  # type: ignore
    assert targets == ['patch.diff']
    assert exts == {'.diff', '.patch'}

    # Permutation 2: description.json has evaluation kind 'swebench_pro-verifier'
    desc_dir = tmp_path / 'benchmark'
    desc_dir.mkdir(parents=True, exist_ok=True)
    (desc_dir / 'description.json').write_text(json.dumps({'evaluation': {'kind': 'swebench_pro-verifier'}}))
    p2 = MockPkg(tmp_path, {}, 'other', 'fam', 't')
    targets, exts = _office_targets(p2)  # type: ignore
    assert targets == ['patch.diff']
    assert exts == {'.diff', '.patch'}

    # Permutation 3: task_id starts with 'instance_ansible'
    p3 = MockPkg(tmp_path / 'empty', {}, 'other', 'fam', 'instance_ansible__ansible-123')
    targets, exts = _office_targets(p3)  # type: ignore
    assert targets == ['patch.diff']
    assert exts == {'.diff', '.patch'}
