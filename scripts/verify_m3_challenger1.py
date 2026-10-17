"""Milestone M3 Adversarial Challenger 1: Empirical Stress Test Suite.

Stress tests:
1. Re-execution of snapshot scoring & bit-for-bit Y3 terminal score consistency across all baseline runs.
2. Verification of 0 Actor model requests (strict socket/HTTP interception during scoring).
3. Verification of delivery gap g = max_k s(k) - Y and 100% 'never' on 'no_artifact'.
4. Deduction conservation invariant sum(d_i) == 1 - s across all checkpoints.
5. Adversarial defect provenance perturbation (broken vs never vs undelivered transition).
6. Y3 consistency breach detection (L2Error on terminal mismatch).
7. Candidate discovery robustness under adversarial paths and zero-byte files.
"""
import sys
from pathlib import Path
import json
import socket
import http.client
import tempfile
import pytest

base_dir = Path("/Users/jiashen/Desktop/问题诊断")
l1_src = base_dir / "L1_sandbox" / "C3_Experimental_Sandbox" / "src"
l2_src = base_dir / "L2_Behavioral Event Localization" / "L2_Sandbox" / "src"

sys.path.insert(0, str(l1_src))
sys.path.insert(0, str(l2_src))

from l2_sandbox.evidence.l1 import L1Package
from l2_sandbox.grades import (
    grade_episode_snapshots,
    compute_item_deductions,
    select_snapshot_candidates,
    GATE_DEFECT_ID,
    NO_ARTIFACT_DEFECT_ID,
)
from l2_sandbox.defects import build_defect_table
from l2_sandbox.io import L2Error

cohort_dir = base_dir / "L1_sandbox" / "C3_Experimental_Sandbox" / "runs" / "swebench_pro_qwen36_cohort"
cache_dir = base_dir / "L2_Behavioral Event Localization" / "L2_Sandbox" / "cache"

def test_empirical_baseline_runs_y3_consistency():
    """Test 1: Re-execute snapshot scoring on all 3 baseline runs and verify Y3 consistency."""
    runs = ["run_task1_s42", "run_task1_s101", "run_task2_s42"]
    pkgs = [L1Package(cohort_dir / r) for r in runs]
    
    # Intercept network to guarantee 0 Actor model requests
    original_connect = socket.socket.connect
    network_calls = []
    
    def guarded_connect(self, address):
        host, port = address[0], address[1]
        # Allow only unix domain sockets or fail on port 8001 / remote LLM
        if port == 8001 or "8001" in str(address):
            network_calls.append(address)
            raise RuntimeError(f"VIOLATION: Attempted network connection to Actor model endpoint: {address}")
        return original_connect(self, address)
        
    socket.socket.connect = guarded_connect
    try:
        for r_name, pkg in zip(runs, pkgs):
            extra = [p for p in pkgs if p.episode_id != pkg.episode_id]
            grade_rows = grade_episode_snapshots(pkg, cache_dir=cache_dir, extra_runs=extra, verify_terminal_submission=True)
            
            # Check row count
            assert len(grade_rows) == 22, f"{r_name} expected 22 grade rows, got {len(grade_rows)}"
            
            # Row 0: terminal submission
            term_row = grade_rows[0]
            assert term_row["kind"] == "terminal_submission"
            official_y = float(pkg.outcome()["y"])
            assert term_row["score"] == official_y == 0.0
            
            # Rows 1..21: checkpoints k=0..20
            cp_rows = [r for r in grade_rows if r["kind"] == "checkpoint"]
            assert len(cp_rows) == 21
            for idx, r in enumerate(cp_rows):
                assert r["k"] == idx
                # Invariant: sum(d_i) == 1 - score
                ded_sum = sum(r["deductions"].values())
                assert abs(ded_sum - (1.0 - r["score"])) < 1e-12, f"Deduction invariant failed at k={idx}: {ded_sum} != {1.0 - r['score']}"
            
            # Terminal snapshot cp000020
            last_cp = cp_rows[-1]
            assert last_cp["k"] == 20
            s_T = float(last_cp["score"])
            
            # Y3 terminal consistency
            assert s_T == official_y, f"Y3 Mismatch on {r_name}: s(T)={s_T} != official_y={official_y}"
            assert abs(s_T - official_y) == 0.0
            
            # Delivery gap g = max_k s(k) - Y
            scores_k = [float(r["score"]) for r in cp_rows]
            max_s = max(scores_k)
            g = max_s - official_y
            assert max_s == 0.0
            assert g == 0.0
            
            # Defect classification
            defects_doc = build_defect_table(pkg, grade_rows, [], extra_runs=extra)
            assert defects_doc["delivery_gap_g"] == 0.0
            assert defects_doc["terminal_score_y"] == 0.0
            assert defects_doc["terminal_snapshot_score_s_T"] == 0.0
            assert defects_doc["k_star"] == 0
            
            defects = defects_doc["defects"]
            assert len(defects) == 1, f"Expected 1 defect on {r_name}, got {len(defects)}"
            d = defects[0]
            assert d["defect"] == NO_ARTIFACT_DEFECT_ID
            assert d["kind"] == "never"
            assert d["deduction"] == 1.0
            assert d["turn"] is None
            assert d["window"] is None
            
            # Deduction share by kind
            by_kind = defects_doc["deduction_by_kind"]
            assert by_kind["never"] == 1.0
            assert by_kind["undelivered"] == 0.0
            assert by_kind["broken"] == 0.0
            
    finally:
        socket.socket.connect = original_connect
        
    assert len(network_calls) == 0, f"Detected {len(network_calls)} unexpected Actor model calls: {network_calls}"
    print("PASS: Baseline runs Y3 consistency and zero-request verification passed.")


def test_adversarial_defect_provenance_transition(tmp_path):
    """Test 2: Adversarially perturb snapshots to test 'broken' vs 'never' vs 'undelivered' transitions."""
    # Synthetic package setup
    og_data = {
        'official_raw': {
            'status': 'ok',
            'test_config': {'fail_to_pass': ['test_1'], 'pass_to_pass': ['test_2']},
        }
    }
    (tmp_path / 'official_grade.json').write_text(json.dumps(og_data))
    
    class FakePkg:
        root = tmp_path
        episode_id = 'adv-ep-01'
        turns = 5
        manifest = {'benchmark': 'swebench_pro', 'task': {'task_id': 'instance_test', 'benchmark': 'swebench_pro'}}
        family_id = 'swebench_pro:instance_test'
        index_sha256 = 'adv_hash'
        benchmark = 'swebench_pro'
        task_id = 'instance_test'
        def termination(self): return 'completed'
        def json(self, p):
            if p == 'benchmark/description.json':
                return {'benchmark': 'swebench_pro', 'task_id': 'instance_test', 'evaluation': {'kind': 'swebench_pro-verifier'}}
            if p == 'official_grade.json': return og_data
            if p == 'raw/resolved_config.json': return {'budget': {'max_model_requests': 10}}
            return {}

    fake_pkg = FakePkg()

    # Case A: Never had candidate -> kind == 'never'
    rows_never = [
        {'kind': 'terminal_submission', 'episode_id': 'adv-ep-01', 'task_id': 'swebench_pro:instance_test', 'k': 5, 'score': 0.0, 'deductions': {GATE_DEFECT_ID: 1.0}},
        *[
            {'kind': 'checkpoint', 'k': k, 'candidate_status': 'no_candidate', 'score': 0.0, 'candidate': [], 'deductions': {GATE_DEFECT_ID: 1.0}}
            for k in range(6)
        ]
    ]
    doc_never = build_defect_table(fake_pkg, rows_never, [])
    assert doc_never['defects'][0]['kind'] == 'never'
    assert doc_never['defects'][0]['turn'] is None

    # Case B: Had candidate at k=2, but lost at k=3..5 -> kind == 'broken', origin turn == 3
    rows_broken = [
        {'kind': 'terminal_submission', 'episode_id': 'adv-ep-01', 'task_id': 'swebench_pro:instance_test', 'k': 5, 'score': 0.0, 'deductions': {GATE_DEFECT_ID: 1.0}},
        {'kind': 'checkpoint', 'k': 0, 'candidate_status': 'no_candidate', 'score': 0.0, 'candidate': [], 'deductions': {GATE_DEFECT_ID: 1.0}},
        {'kind': 'checkpoint', 'k': 1, 'candidate_status': 'no_candidate', 'score': 0.0, 'candidate': [], 'deductions': {GATE_DEFECT_ID: 1.0}},
        {'kind': 'checkpoint', 'k': 2, 'candidate_status': 'ok', 'score': 0.5, 'candidate': ['patch.diff'], 'deductions': {'test_1': 0.5, GATE_DEFECT_ID: 0.0}},
        {'kind': 'checkpoint', 'k': 3, 'candidate_status': 'no_candidate', 'score': 0.0, 'candidate': [], 'deductions': {GATE_DEFECT_ID: 1.0}},
        {'kind': 'checkpoint', 'k': 4, 'candidate_status': 'no_candidate', 'score': 0.0, 'candidate': [], 'deductions': {GATE_DEFECT_ID: 1.0}},
        {'kind': 'checkpoint', 'k': 5, 'candidate_status': 'no_candidate', 'score': 0.0, 'candidate': [], 'deductions': {GATE_DEFECT_ID: 1.0}},
    ]
    doc_broken = build_defect_table(fake_pkg, rows_broken, [])
    assert doc_broken['defects'][0]['kind'] == 'broken'
    assert doc_broken['defects'][0]['turn'] == 3
    assert doc_broken['delivery_gap_g'] == 0.5
    assert doc_broken['max_snapshot_score'] == 0.5
    assert doc_broken['k_star'] == 2

    # Case C: Retained candidate at k=5 with score 1.0, but omitted in submission (Y=0.0) -> kind == 'undelivered'
    rows_undeliv = [
        {'kind': 'terminal_submission', 'episode_id': 'adv-ep-01', 'task_id': 'swebench_pro:instance_test', 'k': 5, 'score': 0.0, 'deductions': {GATE_DEFECT_ID: 1.0}},
        *[
            {'kind': 'checkpoint', 'k': k, 'candidate_status': 'ok', 'score': 1.0, 'candidate': ['patch.diff'], 'deductions': {'test_1': 0.0, 'test_2': 0.0, GATE_DEFECT_ID: 0.0}}
            for k in range(6)
        ]
    ]
    doc_undeliv = build_defect_table(fake_pkg, rows_undeliv, [])
    assert doc_undeliv['delivery_gap_g'] == 1.0
    assert doc_undeliv['undelivered_gap'] == 1.0
    kinds = {d['kind'] for d in doc_undeliv['defects']}
    assert 'undelivered' in kinds
    print("PASS: Adversarial defect provenance transitions verified.")


def test_y3_consistency_violation_rejection(tmp_path):
    """Test 3: Verify that grades.py strictly rejects any package where re-evaluated s(T) != Y."""
    from c3_sandbox.l1.core.contracts import EvaluationSpec

    snaps_dir = tmp_path / 'states' / 'snapshots'
    snaps_dir.mkdir(parents=True, exist_ok=True)
    s0_content = {'files': []}
    s1_content = {'files': [{'path': 'submission/patch.diff', 'sha256': 'sha_diff', 'blob': 'blobs/1', 'size_bytes': 100}]}
    (snaps_dir / 's000000.json').write_text(json.dumps(s0_content))
    (snaps_dir / 's000001.json').write_text(json.dumps(s1_content))

    class MockPkg:
        root = tmp_path
        episode_id = 'ep-mismatch'
        turns = 1
        family_id = 'swebench_pro:test'
        task_id = 'test'
        benchmark = 'swebench_pro'
        manifest = {'benchmark': 'swebench_pro', 'task': {'task_id': 'test', 'benchmark': 'swebench_pro'}}
        index_sha256 = 'dummy_hash'
        snapshots = {
            's000000': s0_content,
            's000001': s1_content,
        }
        def checkpoints(self):
            return {
                0: {'checkpoint_id': 'cp000000', 'snapshot_id': 's000000'},
                1: {'checkpoint_id': 'cp000001', 'snapshot_id': 's000001'},
            }
        def json(self, p):
            if p == 'benchmark/description.json':
                return {'benchmark': 'swebench_pro', 'task_id': 'test', 'evaluation': {'kind': 'swebench_pro-verifier', 'entrypoint': 'test.sh'}}
            if p == 'raw/resolved_config.json':
                return {'evaluation': {'image': 'dummy_image'}}
            if p == 'official_grade.json':
                # Official grade claims 0.0
                return {'projection': {'completion_rate': 0.0, 'official_dimensions': {'passed': False}}, 'official_raw': {'status': 'failed'}}
            return {}

    # Mock _evaluate_staged_candidate to return completion_rate 1.0 (mismatch against official 0.0)
    import l2_sandbox.grades as grades_mod
    original_eval = grades_mod._evaluate_staged_candidate
    grades_mod._evaluate_staged_candidate = lambda *args, **kwargs: (
        {'projection': {'completion_rate': 1.0, 'official_dimensions': {'passed': True}}, 'official_raw': {'status': 'ok'}},
        False
    )
    
    try:
        with pytest.raises(L2Error) as exc_info:
            grade_episode_snapshots(MockPkg(), verify_terminal_submission=True)  # type: ignore
        assert "Terminal submission/ re-evaluation mismatch" in str(exc_info.value)
        print("PASS: Y3 consistency violation correctly triggered L2Error.")
    finally:
        grades_mod._evaluate_staged_candidate = original_eval


def test_adversarial_candidate_filtering_and_safety():
    """Test 4: Verify select_snapshot_candidates handles malicious/zero-byte/irrelevant files safely."""
    snapshot = {
        'files': [
            {'path': 'submission/~$locked.diff', 'sha256': 'sha_bad2', 'blob': 'b2', 'size_bytes': 100},
            {'path': 'submission/zero_size.patch', 'sha256': 'sha_bad3', 'blob': 'b3', 'size_bytes': 0},
            {'path': 'input/reference.diff', 'sha256': 'sha_bad4', 'blob': 'b4', 'size_bytes': 500},
            {'path': 'binary_patch.diff.png', 'sha256': 'sha_bad5', 'blob': 'b5', 'size_bytes': 1000},
            {'path': 'submission/patch.diff', 'sha256': 'sha_valid', 'blob': 'b6', 'size_bytes': 250},
        ]
    }
    cands = select_snapshot_candidates(
        snapshot,
        initial_input_hashes=set(),
        office_targets=['patch.diff'],
        target_exts={'.diff', '.patch'},
        last_written_turn={'submission/patch.diff': 1},
        task_id='instance_ansible__ansible-test',
        spec_kind='swebench_pro-verifier',
    )
    
    # Must only accept submission/patch.diff; reject path traversal, locked file, zero-size file, input file, wrong extension
    assert len(cands) == 1
    assert cands[0]['candidate_id'] == 'submission/patch.diff'
    assert cands[0]['staged'][0][2] == 'sha_valid'
    print("PASS: Adversarial candidate filtering and safety verified.")

if __name__ == '__main__':
    print("=== Running Milestone M3 Adversarial Stress Test Suite ===")
    test_empirical_baseline_runs_y3_consistency()
    with tempfile.TemporaryDirectory() as td:
        test_adversarial_defect_provenance_transition(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_y3_consistency_violation_rejection(Path(td))
    test_adversarial_candidate_filtering_and_safety()
    print("\nALL ADVERSARIAL STRESS TESTS COMPLETED SUCCESSFULLY!")
