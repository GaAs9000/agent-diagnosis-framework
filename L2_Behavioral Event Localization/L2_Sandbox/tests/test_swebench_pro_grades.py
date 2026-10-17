"""Comprehensive unit test suite for SWE-bench Pro V2 snapshot grading, deduplication, item deductions, and defect provenance."""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
import pytest

from l2_sandbox.grades import (
    GATE_DEFECT_ID,
    compute_item_deductions,
    extract_task_rubric,
    select_snapshot_candidates,
    _evaluate_staged_candidate,
)
from l2_sandbox.defects import build_defect_table
from l2_sandbox.evidence.l1 import L1Package


def test_swebench_pro_candidate_discovery_tiers(tmp_path):
    """Verify patch discovery across Tiers 0, 1, 2, 3, deduplication, and filtering."""
    # Synthetic snapshot with files across all tiers + files that must be filtered out
    snapshot = {
        'files': [
            # Tier 0 candidates in submission/
            {'path': 'submission/patch.diff', 'sha256': 'sha_sub_diff', 'blob': 'blobs/1', 'size_bytes': 150},
            {'path': 'submission/model.patch', 'sha256': 'sha_sub_model', 'blob': 'blobs/2', 'size_bytes': 150},
            {'path': 'submission/gold_patch.diff', 'sha256': 'sha_sub_gold', 'blob': 'blobs/3', 'size_bytes': 150},
            {'path': 'submission/extra.patch', 'sha256': 'sha_sub_extra', 'blob': 'blobs/4', 'size_bytes': 120},
            # Tier 1 candidates at root
            {'path': 'patch.diff', 'sha256': 'sha_root_diff', 'blob': 'blobs/5', 'size_bytes': 140},
            {'path': 'model.patch', 'sha256': 'sha_root_model', 'blob': 'blobs/6', 'size_bytes': 140},
            # Tier 2 candidates in subdirectories
            {'path': 'fixes/work.patch', 'sha256': 'sha_work_patch', 'blob': 'blobs/7', 'size_bytes': 110},
            {'path': 'diffs/test.diff', 'sha256': 'sha_test_diff', 'blob': 'blobs/8', 'size_bytes': 110},
            # Files to be filtered out:
            {'path': 'input/reference.diff', 'sha256': 'sha_in', 'blob': 'blobs/9', 'size_bytes': 200},
            {'path': 'submission/~$temp.diff', 'sha256': 'sha_tmp', 'blob': 'blobs/10', 'size_bytes': 100},
            {'path': 'submission/empty.patch', 'sha256': 'sha_empty', 'blob': 'blobs/11', 'size_bytes': 0},
            {'path': 'patch.txt', 'sha256': 'sha_txt', 'blob': 'blobs/12', 'size_bytes': 100},
        ]
    }
    last_written = {
        'submission/patch.diff': 10,
        'submission/model.patch': 9,
        'submission/gold_patch.diff': 8,
        'submission/extra.patch': 7,
        'patch.diff': 6,
        'model.patch': 5,
        'fixes/work.patch': 4,
        'diffs/test.diff': 3,
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

    # 4 (Tier 0) + 2 (Tier 1) + 2 (Tier 2) = 8 candidates total
    assert len(cands) == 8

    # Tier 0 checks
    for idx in range(4):
        assert cands[idx]['tier'] == 0
        assert cands[idx]['in_submission'] is True
        assert cands[idx]['staged'][0][0] == 'patch.diff'
    assert cands[0]['candidate_id'] == 'submission/patch.diff'
    assert cands[1]['candidate_id'] == 'submission/model.patch'
    assert cands[2]['candidate_id'] == 'submission/gold_patch.diff'
    assert cands[3]['candidate_id'] == 'submission/extra.patch'

    # Tier 1 checks
    assert cands[4]['tier'] == 1
    assert cands[4]['in_submission'] is False
    assert cands[4]['candidate_id'] == 'patch.diff'
    assert cands[5]['tier'] == 1
    assert cands[5]['in_submission'] is False
    assert cands[5]['candidate_id'] == 'model.patch'

    # Tier 2 checks
    assert cands[6]['tier'] == 2
    assert cands[6]['in_submission'] is False
    assert cands[6]['candidate_id'] == 'fixes/work.patch'
    assert cands[7]['tier'] == 2
    assert cands[7]['in_submission'] is False
    assert cands[7]['candidate_id'] == 'diffs/test.diff'


def test_swebench_pro_sha256_deduplication():
    """Verify that candidate patches with identical content hashes are deduplicated."""
    snapshot = {
        'files': [
            {'path': 'submission/patch.diff', 'sha256': 'same_content_sha', 'blob': 'blobs/1', 'size_bytes': 100},
            # Different path, identical sha256 -> must be deduplicated
            {'path': 'patch.diff', 'sha256': 'same_content_sha', 'blob': 'blobs/1', 'size_bytes': 100},
            {'path': 'work/backup.patch', 'sha256': 'same_content_sha', 'blob': 'blobs/1', 'size_bytes': 100},
            # Distinct patch
            {'path': 'fixes/unique.diff', 'sha256': 'unique_sha', 'blob': 'blobs/2', 'size_bytes': 120},
        ]
    }
    cands = select_snapshot_candidates(
        snapshot,
        initial_input_hashes=set(),
        office_targets=['patch.diff'],
        target_exts={'.diff', '.patch'},
        last_written_turn={'submission/patch.diff': 5, 'fixes/unique.diff': 6},
        task_id='instance_ansible__ansible-test',
        spec_kind='swebench_pro-verifier',
    )
    # Exactly 2 candidates: submission/patch.diff (tier 0) and fixes/unique.diff (tier 2)
    assert len(cands) == 2
    assert cands[0]['candidate_id'] == 'submission/patch.diff'
    assert cands[0]['tier'] == 0
    assert cands[1]['candidate_id'] == 'fixes/unique.diff'
    assert cands[1]['tier'] == 2


def test_swebench_pro_candidate_discovery_tier3_git_diff(tmp_path):
    """Verify Tier 3: when no standalone diff file exists, extract git diff from restored repository."""
    # Create a git repo with committed base file and uncommitted working-tree change
    repo_dir = tmp_path / 'repo'
    repo_dir.mkdir()
    subprocess.run(['git', 'init'], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(['git', 'config', 'user.name', 'Tester'], cwd=repo_dir, check=True)
    subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=repo_dir, check=True)
    code_file = repo_dir / 'ansible_module.py'
    code_file.write_text('def run(): return False\n')
    subprocess.run(['git', 'add', 'ansible_module.py'], cwd=repo_dir, check=True)
    subprocess.run(['git', 'commit', '-m', 'Initial commit'], cwd=repo_dir, check=True, capture_output=True)

    # Modify the file without creating a standalone .diff/.patch file
    code_file.write_text('def run(): return True\n')

    # Snapshot only contains python file, no .diff/.patch
    snapshot = {
        'files': [
            {'path': 'ansible_module.py', 'sha256': 'sha_mod', 'blob': 'blobs/mod', 'size_bytes': 50},
        ]
    }

    cands = select_snapshot_candidates(
        snapshot,
        initial_input_hashes=set(),
        office_targets=['patch.diff'],
        target_exts={'.diff', '.patch'},
        last_written_turn={'ansible_module.py': 3},
        task_id='instance_ansible__ansible-test',
        spec_kind='swebench_pro-verifier',
        restored_workspace=repo_dir,
    )

    assert len(cands) == 1
    t3 = cands[0]
    assert t3['tier'] == 3
    assert t3['candidate_id'] == 'git_diff.patch'
    assert t3['in_submission'] is False
    assert t3['staged'][0][0] == 'patch.diff'
    assert t3['staged'][0][1] == 'git_diff.patch'
    # Verify the generated diff patch content matches git diff
    patch_path = repo_dir / 'git_diff.patch'
    assert patch_path.is_file()
    assert b'+def run(): return True' in patch_path.read_bytes()


@pytest.mark.parametrize(
    'scenario,passed,passed_names,score',
    [
        ('pass_all', True, ['test_f1', 'test_p1', 'test_p2', 'test_p3'], 1.0),
        ('partial_pass', False, ['test_p1', 'test_p2'], 0.0),
        ('zero_pass', False, [], 0.0),
    ],
)
def test_swebench_pro_item_deductions_mathematical_invariant(scenario, passed, passed_names, score):
    """Verify item deductions and identical invariant sum(d_i) == 1 - s across pass, partial, and zero."""
    fail_to_pass = ['test_f1']
    pass_to_pass = ['test_p1', 'test_p2', 'test_p3']
    all_required = fail_to_pass + pass_to_pass
    N = len(all_required)  # N = 4

    raw = {
        'task_id': 'instance_ansible__ansible-test',
        'passed': passed,
        'reward': score,
        'score': score,
        'status': 'ok' if passed else 'failed',
        'exit_code': 0 if passed else 1,
        'test_config': {
            'fail_to_pass': fail_to_pass,
            'pass_to_pass': pass_to_pass,
        },
        'test_results': {
            'tests': [{'name': name, 'status': 'PASSED' if name in passed_names else 'FAILED'} for name in all_required]
        },
    }

    deductions, max_score = compute_item_deductions(raw, score)
    assert max_score == 1.0

    # Invariant: sum of all deductions identically equals 1 - score
    expected_sum = 1.0 - score
    actual_sum = sum(deductions.values())
    assert abs(actual_sum - expected_sum) < 1e-12, f'Invariant failed: {actual_sum} != {expected_sum}'

    if scenario == 'pass_all':
        for t in all_required:
            assert deductions[t] == 0.0
        assert deductions[GATE_DEFECT_ID] == 0.0

    elif scenario == 'partial_pass':
        # Passed tests have d_t = 0.0
        for p in passed_names:
            assert deductions[p] == 0.0
        # Failed tests have d_t = 1/N
        failed_tests = [t for t in all_required if t not in passed_names]
        for f in failed_tests:
            assert abs(deductions[f] - (1.0 / N)) < 1e-12
        # Gate penalty absorbs the wiped-out passed tests: d_gate = N_passed / N
        assert abs(deductions[GATE_DEFECT_ID] - (len(passed_names) / N)) < 1e-12
        assert abs(actual_sum - 1.0) < 1e-12

    elif scenario == 'zero_pass':
        # All tests failed -> each gets 1/N
        for t in all_required:
            assert abs(deductions[t] - (1.0 / N)) < 1e-12
        # Gate penalty = 0 since N_passed = 0
        assert deductions[GATE_DEFECT_ID] == 0.0
        assert abs(actual_sum - 1.0) < 1e-12


def test_swebench_pro_rubric_extraction(tmp_path):
    """Verify extract_task_rubric extracts required test rules from test_config."""
    from l2_sandbox.evidence.l1 import L1Package

    raw = {
        'status': 'ok',
        'test_config': {
            'fail_to_pass': ['test_fix_bug'],
            'pass_to_pass': ['test_reg_1', 'test_reg_2', 'test_reg_3'],
        },
    }
    (tmp_path / 'official_grade.json').write_text(json.dumps({'official_raw': raw}))

    class DummyPkg:
        def __init__(self, root):
            self.root = root

        def json(self, name):
            return json.loads((self.root / name).read_text())

    rubric = extract_task_rubric(DummyPkg(tmp_path))  # type: ignore
    assert len(rubric) == 4
    names = [r for r, _ in rubric]
    assert names == ['test_fix_bug', 'test_reg_1', 'test_reg_2', 'test_reg_3']
    for _, weight in rubric:
        assert abs(weight - 0.25) < 1e-12


def test_swebench_pro_evaluation_caching(tmp_path, monkeypatch):
    """Verify that _evaluate_staged_candidate caches evaluations by staged SHA-256."""
    from c3_sandbox.l1.core.contracts import EvaluationSpec

    cache_dir = tmp_path / 'cache'
    eval_call_count = 0

    class MockSWEBenchProEvaluator:
        def __init__(self, config):
            self.config = config

        def evaluate(self, spec, sub_dir, grade_out):
            nonlocal eval_call_count
            eval_call_count += 1
            return {
                'projection': {'completion_rate': 1.0, 'task_outcome': 'success', 'official_dimensions': {'passed': True}},
                'official_raw': {'status': 'ok', 'passed': True, 'reward': 1.0},
                'official_raw_sha256': 'raw_sha_123',
            }

    import l2_sandbox.grades as grades_mod
    monkeypatch.setattr(grades_mod, 'SWEBenchProEvaluator', MockSWEBenchProEvaluator)

    # Candidate file in workspace
    ws_dir = tmp_path / 'ws'
    ws_dir.mkdir()
    patch_file = ws_dir / 'submission' / 'patch.diff'
    patch_file.parent.mkdir()
    patch_file.write_text('diff --git a/file b/file\n')
    patch_sha = hashlib.sha256(patch_file.read_bytes()).hexdigest()

    spec = EvaluationSpec(
        kind='swebench_pro-verifier',
        source_root=tmp_path / 'source',
        entrypoint='tests/test.sh',
        task_id='instance_ansible__ansible-test',
        metadata={'image': 'ghcr.io/scaleapi/swe-bench_pro-v2:instance_ansible__ansible-test'},
    )
    cand = {
        'candidate_id': 'submission/patch.diff',
        'paths': ['submission/patch.diff'],
        'staged': [('patch.diff', 'submission/patch.diff', patch_sha, '')],
        'in_submission': True,
        'tier': 0,
        'last_written_turn': 1,
    }

    class DummyPkg:
        root = tmp_path
        episode_id = 'test-ep-01'

    # Call 1: cold evaluation
    grade1, hit1 = _evaluate_staged_candidate(
        DummyPkg(), spec, {'image': 'test-image'}, {}, cand, cache_dir, restored_workspace=ws_dir  # type: ignore
    )
    assert hit1 is False
    assert eval_call_count == 1
    assert grade1['projection']['completion_rate'] == 1.0

    # Call 2: repeated evaluation with identical staged content -> cache hit!
    grade2, hit2 = _evaluate_staged_candidate(
        DummyPkg(), spec, {'image': 'test-image'}, {}, cand, cache_dir, restored_workspace=ws_dir  # type: ignore
    )
    assert hit2 is True
    assert eval_call_count == 1  # 0 redundant evaluator calls!
    assert grade2 == grade1


def test_swebench_pro_defect_categorization_and_delivery_gap(tmp_path):
    """Verify defect taxonomy (undelivered, broken, never) and delivery gap g on SWE-bench Pro."""
    # Synthetic trajectory rows for 3 checkpoints: k = 0, 1, 2
    # Defect rules: test_f1, test_f2, test_p1
    # Checkpoint 0: all failed (never passed yet)
    # Checkpoint 1: test_f1 and test_p1 pass, test_f2 fails (s(1) = 0.0 due to test_f2)
    # Checkpoint 2 (final cp): test_f1 broke (failed), test_p1 passed, test_f2 never passed
    # Terminal submission: s(T) = 0.0, Y = 0.0
    pass_rubric = [('test_f1', 1.0 / 3), ('test_f2', 1.0 / 3), ('test_p1', 1.0 / 3)]

    # Case A: Undelivered defect test
    # Suppose at checkpoint 2 (final), s(2) = 1.0 (all passed), but terminal submission omitted patch (Y = 0.0)
    cp_rows_undeliv = [
        {
            'kind': 'terminal_submission',
            'episode_id': 'test-ep',
            'task_id': 'swebench_pro:instance_test',
            'k': 2,
            'score': 0.0,
            'dim1_pass': False,
            'deductions': {GATE_DEFECT_ID: 1.0},
        },
        {
            'kind': 'checkpoint',
            'k': 0,
            'candidate_status': 'ok',
            'score': 0.0,
            'candidate': ['submission/patch.diff'],
            'deductions': {'test_f1': 1.0 / 3, 'test_f2': 1.0 / 3, 'test_p1': 1.0 / 3, GATE_DEFECT_ID: 0.0},
        },
        {
            'kind': 'checkpoint',
            'k': 1,
            'candidate_status': 'ok',
            'score': 1.0,
            'candidate': ['submission/patch.diff'],
            'deductions': {'test_f1': 0.0, 'test_f2': 0.0, 'test_p1': 0.0, GATE_DEFECT_ID: 0.0},
        },
        {
            'kind': 'checkpoint',
            'k': 2,
            'candidate_status': 'ok',
            'score': 1.0,
            'candidate': ['submission/patch.diff'],
            'deductions': {'test_f1': 0.0, 'test_f2': 0.0, 'test_p1': 0.0, GATE_DEFECT_ID: 0.0},
        },
    ]

    og_data = {
        'official_raw': {
            'status': 'ok',
            'test_config': {'fail_to_pass': ['test_f1', 'test_f2'], 'pass_to_pass': ['test_p1']},
        }
    }
    (tmp_path / 'official_grade.json').write_text(json.dumps(og_data))

    class FakeL1Pkg:
        root = tmp_path
        episode_id = 'test-ep'
        turns = 2
        manifest = {'benchmark': 'swebench_pro', 'task': {'task_id': 'instance_test', 'benchmark': 'swebench_pro'}}
        family_id = 'swebench_pro:instance_test'
        index_sha256 = 'abc'
        benchmark = 'swebench_pro'
        task_id = 'instance_test'

        def termination(self):
            return 'completed'

        def json(self, p):
            if p == 'benchmark/description.json':
                return {'benchmark': 'swebench_pro', 'task_id': 'instance_test', 'evaluation': {'kind': 'swebench_pro-verifier'}}
            if p == 'official_grade.json':
                return og_data
            if p == 'raw/resolved_config.json':
                return {'budget': {'max_model_requests': 10}}
            return {}

    fake_pkg = FakeL1Pkg()

    # Case A: Undelivered defect test (s(2) = 1.0, Y = 0.0 -> delivery gap g = 1.0)
    table_undeliv = build_defect_table(fake_pkg, cp_rows_undeliv, [])  # type: ignore
    gap_undeliv = table_undeliv['delivery_gap_g']
    defects_undeliv = table_undeliv['defects']
    assert abs(gap_undeliv - 1.0) < 1e-12
    kinds_undeliv = {d['kind'] for d in defects_undeliv}
    assert 'undelivered' in kinds_undeliv
    assert abs(sum(d['deduction'] for d in defects_undeliv) - 1.0) < 1e-12

    # Case B & C: Broken and Never defects
    # k=0: all 3 tests fail (s=0.0)
    # k=1: test_f1 and test_p1 pass (d=0), test_f2 fails (s=0.0)
    # k=2: test_f1 broke (d>0), test_p1 passes (d=0), test_f2 never passed (d>0), s=0.0, Y=0.0
    cp_rows_broken = [
        {
            'kind': 'terminal_submission',
            'episode_id': 'test-ep',
            'task_id': 'swebench_pro:instance_test',
            'k': 2,
            'score': 0.0,
            'dim1_pass': False,
            'deductions': {'test_f1': 1.0 / 3, 'test_f2': 1.0 / 3, GATE_DEFECT_ID: 1.0 / 3},
        },
        {
            'kind': 'checkpoint',
            'k': 0,
            'candidate_status': 'ok',
            'score': 0.0,
            'candidate': ['submission/patch.diff'],
            'deductions': {'test_f1': 1.0 / 3, 'test_f2': 1.0 / 3, 'test_p1': 1.0 / 3, GATE_DEFECT_ID: 0.0},
        },
        {
            'kind': 'checkpoint',
            'k': 1,
            'candidate_status': 'ok',
            'score': 0.0,
            'candidate': ['submission/patch.diff'],
            'deductions': {'test_f1': 0.0, 'test_f2': 1.0 / 3, 'test_p1': 0.0, GATE_DEFECT_ID: 2.0 / 3},
        },
        {
            'kind': 'checkpoint',
            'k': 2,
            'candidate_status': 'ok',
            'score': 0.0,
            'candidate': ['submission/patch.diff'],
            'deductions': {'test_f1': 1.0 / 3, 'test_f2': 1.0 / 3, 'test_p1': 0.0, GATE_DEFECT_ID: 1.0 / 3},
        },
    ]

    table_bn = build_defect_table(fake_pkg, cp_rows_broken, [])  # type: ignore
    gap_bn = table_bn['delivery_gap_g']
    defects_bn = table_bn['defects']
    kinds_by_defect = {d['defect']: d['kind'] for d in defects_bn}
    # test_f1 was satisfied at k=1 (d=0) and failed at k=2 (d>0) -> broken!
    assert kinds_by_defect.get('test_f1') == 'broken'
    # test_f2 was never satisfied at any checkpoint -> never!
    assert kinds_by_defect.get('test_f2') == 'never'
    assert abs(gap_bn - 0.0) < 1e-12


def test_verify_extract_item_scores():
    """Verify that verify._extract_item_scores accurately extracts binary per-test scores."""
    from l2_sandbox.verify import _extract_item_scores

    class FakePkg:
        def __init__(self, raw):
            self._raw = raw

        def json(self, p):
            return {'official_raw': self._raw}

    raw = {
        'test_config': {
            'fail_to_pass': ['test_1', 'test_2'],
            'pass_to_pass': ['test_3', 'test_4'],
        },
        'test_results': {
            'tests': [
                {'name': 'test_1', 'status': 'PASSED'},
                {'name': 'test_2', 'status': 'FAILED'},
                {'name': 'test_3', 'status': 'PASSED'},
                {'name': 'test_4', 'status': 'SKIPPED'},
            ]
        },
    }
    scores = _extract_item_scores(FakePkg(raw))  # type: ignore
    assert scores == {
        'test_1': 1.0,
        'test_2': 0.0,
        'test_3': 1.0,
        'test_4': 0.0,
    }
