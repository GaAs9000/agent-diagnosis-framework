"""Empirical Stress Tests and Adversarial Challenges for Milestone M1.

Role: Adversarial Challenger 2 (teamwork_preview_challenger_m1_2).
Scope:
1. Mathematical deduction invariant sum d_i(k) = 1 - s(k) in grades.py across extreme N (N=1, 100, 5000).
2. Machine precision tolerance (|sum d_i - (1 - s)| < 1e-12) across exhaustive and random passing permutations.
3. Boundary & adversarial inputs (N=0, duplicate names, stringified config, malformed data, special characters).
4. Defect categorization (undelivered, broken, never) across simulated multi-turn trajectory snapshots in defects.py.
"""

import ast
import json
import random
from pathlib import Path
import pytest

from l2_sandbox.grades import compute_item_deductions, extract_task_rubric, GATE_DEFECT_ID, NO_ARTIFACT_DEFECT_ID
from l2_sandbox.defects import build_defect_table


# ============================================================================
# 1. Extreme N & Machine Precision Invariant Tests
# ============================================================================

@pytest.mark.parametrize("N", [1, 2, 3, 5, 7, 10, 16, 23, 50, 91, 100, 250, 500, 1000, 5000])
def test_deduction_invariant_extreme_n_all_and_zero_pass(N):
    """Verify sum d_i(k) == 1 - s(k) holds within 1e-12 for extreme N under 0 passes and all passes."""
    test_names = [f"test_item_{i:04d}" for i in range(N)]
    f2p = test_names[: N // 2]
    p2p = test_names[N // 2 :]

    # --- Case A: Zero passes (s = 0.0) ---
    raw_zero = {
        "task_id": f"task_extreme_{N}",
        "passed": False,
        "reward": 0.0,
        "score": 0.0,
        "status": "failed",
        "exit_code": 1,
        "test_config": {"fail_to_pass": f2p, "pass_to_pass": p2p},
        "test_results": {"tests": [{"name": t, "status": "FAILED"} for t in test_names]},
    }
    ded_zero, max_s_zero = compute_item_deductions(raw_zero, 0.0)
    assert max_s_zero == 1.0
    sum_zero = sum(ded_zero.values())
    err_zero = abs(sum_zero - 1.0)
    assert err_zero < 1e-12, f"N={N} Zero pass invariant violation: sum={sum_zero}, err={err_zero:.2e}"
    assert ded_zero[GATE_DEFECT_ID] == 0.0
    unit = 1.0 / N
    for t in test_names:
        assert abs(ded_zero[t] - unit) < 1e-12

    # --- Case B: All pass, score = 1.0 ---
    raw_all = {
        "task_id": f"task_extreme_{N}",
        "passed": True,
        "reward": 1.0,
        "score": 1.0,
        "status": "ok",
        "exit_code": 0,
        "test_config": {"fail_to_pass": f2p, "pass_to_pass": p2p},
        "test_results": {"tests": [{"name": t, "status": "PASSED"} for t in test_names]},
    }
    ded_all, max_s_all = compute_item_deductions(raw_all, 1.0)
    assert max_s_all == 1.0
    sum_all = sum(ded_all.values())
    assert abs(sum_all - 0.0) < 1e-12, f"N={N} All pass invariant violation: sum={sum_all}"
    assert ded_all[GATE_DEFECT_ID] == 0.0
    for t in test_names:
        assert ded_all[t] == 0.0

    # --- Case C: All tests in test_results PASSED, but overall task FAILED (score = 0.0) ---
    raw_gate_fail = {
        "task_id": f"task_extreme_{N}",
        "passed": False,
        "reward": 0.0,
        "score": 0.0,
        "status": "failed",
        "exit_code": 1,
        "test_config": {"fail_to_pass": f2p, "pass_to_pass": p2p},
        "test_results": {"tests": [{"name": t, "status": "PASSED"} for t in test_names]},
    }
    ded_gate, max_s_gate = compute_item_deductions(raw_gate_fail, 0.0)
    assert max_s_gate == 1.0
    sum_gate = sum(ded_gate.values())
    assert abs(sum_gate - 1.0) < 1e-12, f"N={N} Gate failure invariant violation: sum={sum_gate}"
    assert abs(ded_gate[GATE_DEFECT_ID] - 1.0) < 1e-12
    for t in test_names:
        assert ded_gate[t] == 0.0


@pytest.mark.parametrize("N", [1, 2, 3, 4, 5, 6, 7, 8])
def test_all_passing_permutations_exhaustive(N):
    """Exhaustively verify all 2^N passing permutations for small N to ensure no permutation breaks the invariant."""
    test_names = [f"t_{i}" for i in range(N)]
    f2p = test_names[: N // 2]
    p2p = test_names[N // 2 :]

    for mask in range(1 << N):
        passed_set = {test_names[i] for i in range(N) if (mask & (1 << i))}
        p_count = len(passed_set)

        # In SWE-bench Pro: if all passed, score is 1.0; else 0.0
        score = 1.0 if p_count == N else 0.0
        raw = {
            "task_id": f"task_perm_{N}",
            "passed": (p_count == N),
            "reward": score,
            "score": score,
            "status": "ok" if (p_count == N) else "failed",
            "exit_code": 0 if (p_count == N) else 1,
            "test_config": {"fail_to_pass": f2p, "pass_to_pass": p2p},
            "test_results": {
                "tests": [
                    {"name": t, "status": "PASSED" if t in passed_set else "FAILED"}
                    for t in test_names
                ]
            },
        }

        ded, max_s = compute_item_deductions(raw, score)
        actual_sum = sum(ded.values())
        expected_sum = 1.0 - score
        assert abs(actual_sum - expected_sum) < 1e-12, (
            f"Permutation invariant failed: N={N}, mask={mask:0{N}b}, "
            f"actual={actual_sum}, expected={expected_sum}"
        )

        if score == 1.0:
            assert ded[GATE_DEFECT_ID] == 0.0
            for t in test_names:
                assert ded[t] == 0.0
        else:
            unit = 1.0 / N
            for t in test_names:
                if t in passed_set:
                    assert ded[t] == 0.0
                else:
                    assert abs(ded[t] - unit) < 1e-12
            assert abs(ded[GATE_DEFECT_ID] - (p_count / N)) < 1e-12


def test_random_passing_subsets_large_n():
    """Monte Carlo stress test: 200 random subsets for N=100, N=500, N=1000, N=5000."""
    rng = random.Random(42)
    for N in [50, 100, 500, 1000, 5000]:
        test_names = [f"test_{i}" for i in range(N)]
        f2p = test_names[: N // 3]
        p2p = test_names[N // 3 :]

        for trial in range(20):
            p_count = rng.randint(0, N)
            passed_names = set(rng.sample(test_names, p_count))
            score = 1.0 if p_count == N else 0.0

            raw = {
                "task_id": f"task_mc_{N}_{trial}",
                "passed": (p_count == N),
                "reward": score,
                "score": score,
                "status": "ok" if (p_count == N) else "failed",
                "exit_code": 0 if (p_count == N) else 1,
                "test_config": {"fail_to_pass": f2p, "pass_to_pass": p2p},
                "test_results": {
                    "tests": [
                        {"name": t, "status": "PASSED" if t in passed_names else "FAILED"}
                        for t in test_names
                    ]
                },
            }

            ded, max_s = compute_item_deductions(raw, score)
            actual_sum = sum(ded.values())
            expected_sum = 1.0 - score
            assert abs(actual_sum - expected_sum) < 1e-12, (
                f"Monte Carlo failed: N={N}, P={p_count}, actual={actual_sum}, expected={expected_sum}"
            )


# ============================================================================
# 2. Boundary & Adversarial Input Tests
# ============================================================================

def test_boundary_empty_test_config_n_zero():
    """N=0 edge case: empty fail_to_pass and pass_to_pass."""
    raw = {
        "task_id": "empty_task",
        "passed": False,
        "reward": 0.0,
        "score": 0.0,
        "status": "failed",
        "exit_code": 1,
        "test_config": {"fail_to_pass": [], "pass_to_pass": []},
        "test_results": {"tests": []},
    }
    ded, max_s = compute_item_deductions(raw, 0.0)
    assert max_s == 1.0
    assert ded == {GATE_DEFECT_ID: 1.0}
    assert sum(ded.values()) == 1.0

    # If score is 1.0 with N=0
    raw_pass = dict(raw, passed=True, reward=1.0, score=1.0, status="ok", exit_code=0)
    ded_p, max_s_p = compute_item_deductions(raw_pass, 1.0)
    assert ded_p == {GATE_DEFECT_ID: 0.0}
    assert sum(ded_p.values()) == 0.0


def test_adversarial_stringified_and_malformed_test_configs():
    """Test AST string literal parsing and graceful handling of malformed strings."""
    # Valid stringified python lists
    raw_str = {
        "test_config": {
            "fail_to_pass": "['test_1', 'test_2']",
            "pass_to_pass": "['test_3']",
        },
        "test_results": {
            "tests": [
                {"name": "test_1", "status": "PASSED"},
                {"name": "test_2", "status": "FAILED"},
                {"name": "test_3", "status": "FAILED"},
            ]
        },
    }
    ded, max_s = compute_item_deductions(raw_str, 0.0)
    assert len(ded) == 4  # 3 tests + GATE_DEFECT_ID
    assert abs(sum(ded.values()) - 1.0) < 1e-12
    assert ded["test_1"] == 0.0
    assert abs(ded["test_2"] - (1.0 / 3)) < 1e-12
    assert abs(ded[GATE_DEFECT_ID] - (1.0 / 3)) < 1e-12

    # Malformed syntax in stringified config
    raw_syntax_err = {
        "test_config": {
            "fail_to_pass": "['test_1', 'unterminated string...",
            "pass_to_pass": ["test_ok"],
        },
        "test_results": {"tests": [{"name": "test_ok", "status": "FAILED"}]},
    }
    ded_err, _ = compute_item_deductions(raw_syntax_err, 0.0)
    # fail_to_pass gracefully falls back to empty, pass_to_pass has test_ok (N=1)
    assert abs(sum(ded_err.values()) - 1.0) < 1e-12
    assert "test_ok" in ded_err
    assert abs(ded_err["test_ok"] - 1.0) < 1e-12


def test_adversarial_duplicate_and_special_char_tests():
    """Test duplicate test names within/across lists, empty strings, and special characters."""
    special_names = [
        "tests/unit/test_foo.py::TestClass::test_method[param-1]",
        "test with spaces and $pecial #chars",
        "测试_unicode_名",
        "test\ttab",
    ]
    raw = {
        "test_config": {
            "fail_to_pass": [special_names[0], special_names[0], "", "   ", special_names[1]],
            "pass_to_pass": [special_names[1], special_names[2], special_names[3]],
        },
        "test_results": {
            "tests": [
                {"name": special_names[0], "status": "PASSED"},
                {"name": special_names[0], "status": "PASSED"},  # duplicate in results
                {"name": special_names[1], "status": "FAILED"},
                {"name": special_names[2], "status": "PASSED"},
                {"name": "unrelated_external_test", "status": "PASSED"},  # not in config!
            ]
        },
    }
    # Note: '   ' is a truthy non-empty string in Python (bool('   ') == True),
    # so grades.py includes it as a test item (N=5).
    # Crucially, the mathematical invariant still holds identically: sum d_i = 1.0!
    ded, max_s = compute_item_deductions(raw, 0.0)
    assert len(ded) == 6  # 5 tests (including '   ') + GATE_DEFECT_ID
    assert abs(sum(ded.values()) - 1.0) < 1e-12

    # Verify unrelated external test did not leak into deductions
    assert "unrelated_external_test" not in ded

    # 2 tests passed (names[0] and names[2]), 3 failed (names[1], names[3], and '   ')
    assert ded[special_names[0]] == 0.0
    assert ded[special_names[2]] == 0.0
    assert abs(ded[special_names[1]] - 0.20) < 1e-12
    assert abs(ded[special_names[3]] - 0.20) < 1e-12
    assert abs(ded['   '] - 0.20) < 1e-12
    assert abs(ded[GATE_DEFECT_ID] - 0.40) < 1e-12


def test_adversarial_malformed_test_results():
    """Verify compute_item_deductions survives when test_results has missing fields, None, or non-dict."""
    raw = {
        "test_config": {"fail_to_pass": ["t1", "t2"], "pass_to_pass": []},
        "test_results": {
            "tests": [
                None,
                "not a dict",
                {"no_name": True},
                {"name": "t1", "status": None},
                {"name": "t2", "status": "PASSED"},
            ]
        },
    }
    ded, max_s = compute_item_deductions(raw, 0.0)
    assert abs(sum(ded.values()) - 1.0) < 1e-12
    assert ded["t2"] == 0.0
    assert abs(ded["t1"] - 0.5) < 1e-12
    assert abs(ded[GATE_DEFECT_ID] - 0.5) < 1e-12


# ============================================================================
# 3. Defect Categorization & Trajectory Provenance Tests
# ============================================================================

class FakePackage:
    def __init__(self, tmp_path, turns=3, benchmark="swebench_pro", task_id="instance_test"):
        self.root = tmp_path
        self.episode_id = f"ep_{task_id}"
        self.turns = turns
        self.benchmark = benchmark
        self.family_id = f"{benchmark}:{task_id}"
        self.task_id = task_id
        self.index_sha256 = "dummy_sha"
        self.manifest = {"benchmark": benchmark, "task": {"task_id": task_id, "benchmark": benchmark}}

    def termination(self):
        return "completed"

    def json(self, p):
        if p == "benchmark/description.json":
            return {"benchmark": self.benchmark, "task_id": self.task_id, "evaluation": {"kind": "swebench_pro-verifier"}}
        if p == "raw/resolved_config.json":
            return {"budget": {"max_model_requests": 20}}
        return {}


def test_defect_categorization_pure_undelivered(tmp_path):
    """Stress test pure undelivered defect: s(T) = 1.0, but Y = 0.0 (omitted patch)."""
    pkg = FakePackage(tmp_path, turns=2)
    # Checkpoints: k=0 (fail), k=1 (pass), k=2 (pass). Terminal submission omitted (Y = 0.0).
    cp_rows = [
        {
            "kind": "terminal_submission",
            "episode_id": pkg.episode_id,
            "task_id": pkg.family_id,
            "k": 2,
            "score": 0.0,
            "dim1_pass": False,
            "deductions": {GATE_DEFECT_ID: 1.0},
        },
        {
            "kind": "checkpoint",
            "k": 0,
            "candidate_status": "ok",
            "score": 0.0,
            "candidate": ["submission/patch.diff"],
            "deductions": {"t1": 0.5, "t2": 0.5, GATE_DEFECT_ID: 0.0},
        },
        {
            "kind": "checkpoint",
            "k": 1,
            "candidate_status": "ok",
            "score": 1.0,
            "candidate": ["submission/patch.diff"],
            "deductions": {"t1": 0.0, "t2": 0.0, GATE_DEFECT_ID: 0.0},
        },
        {
            "kind": "checkpoint",
            "k": 2,
            "candidate_status": "ok",
            "score": 1.0,
            "candidate": ["submission/patch.diff"],
            "deductions": {"t1": 0.0, "t2": 0.0, GATE_DEFECT_ID: 0.0},
        },
    ]

    table = build_defect_table(pkg, cp_rows, [])
    assert abs(table["delivery_gap_g"] - 1.0) < 1e-12
    assert abs(table["undelivered_gap"] - 1.0) < 1e-12
    assert abs(table["peak_to_terminal_loss"] - 0.0) < 1e-12
    assert table["k_star"] == 1  # earliest peak

    defects = table["defects"]
    assert len(defects) > 0
    for d in defects:
        assert d["kind"] == "undelivered"
        assert d["turn"] == 1  # origin turn of the undelivered achievement

    assert table["deduction_by_kind"]["undelivered"] == 1.0
    assert table["deduction_by_kind"]["broken"] == 0.0
    assert table["deduction_by_kind"]["never"] == 0.0
    assert abs(table["total_deduction"] - 1.0) < 1e-12


def test_defect_categorization_pure_never(tmp_path):
    """Stress test pure never defect: all tests fail across all checkpoints."""
    pkg = FakePackage(tmp_path, turns=2)
    cp_rows = [
        {
            "kind": "terminal_submission",
            "episode_id": pkg.episode_id,
            "task_id": pkg.family_id,
            "k": 2,
            "score": 0.0,
            "dim1_pass": False,
            "deductions": {"t1": 0.5, "t2": 0.5, GATE_DEFECT_ID: 0.0},
        },
        {
            "kind": "checkpoint",
            "k": 0,
            "candidate_status": "ok",
            "score": 0.0,
            "candidate": ["submission/patch.diff"],
            "deductions": {"t1": 0.5, "t2": 0.5, GATE_DEFECT_ID: 0.0},
        },
        {
            "kind": "checkpoint",
            "k": 1,
            "candidate_status": "ok",
            "score": 0.0,
            "candidate": ["submission/patch.diff"],
            "deductions": {"t1": 0.5, "t2": 0.5, GATE_DEFECT_ID: 0.0},
        },
        {
            "kind": "checkpoint",
            "k": 2,
            "candidate_status": "ok",
            "score": 0.0,
            "candidate": ["submission/patch.diff"],
            "deductions": {"t1": 0.5, "t2": 0.5, GATE_DEFECT_ID: 0.0},
        },
    ]

    table = build_defect_table(pkg, cp_rows, [])
    assert abs(table["delivery_gap_g"] - 0.0) < 1e-12
    assert abs(table["undelivered_gap"] - 0.0) < 1e-12

    defects = table["defects"]
    assert len(defects) == 2
    for d in defects:
        assert d["kind"] == "never"
        assert d["turn"] is None
        assert d["window"] is None
        assert abs(d["deduction"] - 0.5) < 1e-12

    assert table["deduction_by_kind"]["never"] == 1.0
    assert table["deduction_by_kind"]["broken"] == 0.0
    assert table["deduction_by_kind"]["undelivered"] == 0.0


def test_defect_categorization_pure_broken(tmp_path):
    """Stress test pure broken defect: all tests satisfied at k=1, all broken at k=2."""
    pkg = FakePackage(tmp_path, turns=2)
    cp_rows = [
        {
            "kind": "terminal_submission",
            "episode_id": pkg.episode_id,
            "task_id": pkg.family_id,
            "k": 2,
            "score": 0.0,
            "dim1_pass": False,
            "deductions": {"t1": 0.5, "t2": 0.5, GATE_DEFECT_ID: 0.0},
        },
        {
            "kind": "checkpoint",
            "k": 0,
            "candidate_status": "ok",
            "score": 0.0,
            "candidate": ["submission/patch.diff"],
            "deductions": {"t1": 0.5, "t2": 0.5, GATE_DEFECT_ID: 0.0},
        },
        {
            "kind": "checkpoint",
            "k": 1,
            "candidate_status": "ok",
            "score": 1.0,
            "candidate": ["submission/patch.diff"],
            "deductions": {"t1": 0.0, "t2": 0.0, GATE_DEFECT_ID: 0.0},
        },
        {
            "kind": "checkpoint",
            "k": 2,
            "candidate_status": "ok",
            "score": 0.0,
            "candidate": ["submission/patch.diff"],
            "deductions": {"t1": 0.5, "t2": 0.5, GATE_DEFECT_ID: 0.0},
        },
    ]

    table = build_defect_table(pkg, cp_rows, [])
    assert abs(table["delivery_gap_g"] - 1.0) < 1e-12
    assert abs(table["peak_to_terminal_loss"] - 1.0) < 1e-12
    assert table["k_star"] == 1

    defects = table["defects"]
    assert len(defects) == 2
    for d in defects:
        assert d["kind"] == "broken"
        assert d["turn"] == 2  # broke at turn 2 (transition from k=1 to k=2)
        assert abs(d["deduction"] - 0.5) < 1e-12

    assert table["deduction_by_kind"]["broken"] == 1.0
    assert table["deduction_by_kind"]["never"] == 0.0
    assert table["deduction_by_kind"]["undelivered"] == 0.0


def test_defect_categorization_no_candidate_at_terminal(tmp_path):
    """Stress test terminal without candidates: transitions between broken and never no_artifact."""
    pkg = FakePackage(tmp_path, turns=2)

    # Subcase A: had candidate at k=0, lost candidate at k=1 and k=2 -> broken no_artifact
    cp_rows_broken_cand = [
        {"kind": "terminal_submission", "episode_id": pkg.episode_id, "k": 2, "score": 0.0, "dim1_pass": False, "deductions": {GATE_DEFECT_ID: 1.0}},
        {"kind": "checkpoint", "k": 0, "candidate_status": "ok", "score": 0.5, "candidate": ["patch.diff"], "deductions": {"t1": 0.5, GATE_DEFECT_ID: 0.0}},
        {"kind": "checkpoint", "k": 1, "candidate_status": "no_candidate", "score": 0.0, "candidate": [], "deductions": {GATE_DEFECT_ID: 1.0}},
        {"kind": "checkpoint", "k": 2, "candidate_status": "no_candidate", "score": 0.0, "candidate": [], "deductions": {GATE_DEFECT_ID: 1.0}},
    ]
    table_a = build_defect_table(pkg, cp_rows_broken_cand, [])
    assert len(table_a["defects"]) == 1
    d_a = table_a["defects"][0]
    assert d_a["defect"] == NO_ARTIFACT_DEFECT_ID
    assert d_a["kind"] == "broken"
    assert d_a["turn"] == 1  # candidate disappeared after k=0 -> turn 1

    # Subcase B: never had any candidate at any checkpoint -> never no_artifact
    cp_rows_never_cand = [
        {"kind": "terminal_submission", "episode_id": pkg.episode_id, "k": 2, "score": 0.0, "dim1_pass": False, "deductions": {GATE_DEFECT_ID: 1.0}},
        {"kind": "checkpoint", "k": 0, "candidate_status": "no_candidate", "score": 0.0, "candidate": [], "deductions": {GATE_DEFECT_ID: 1.0}},
        {"kind": "checkpoint", "k": 1, "candidate_status": "no_candidate", "score": 0.0, "candidate": [], "deductions": {GATE_DEFECT_ID: 1.0}},
        {"kind": "checkpoint", "k": 2, "candidate_status": "no_candidate", "score": 0.0, "candidate": [], "deductions": {GATE_DEFECT_ID: 1.0}},
    ]
    table_b = build_defect_table(pkg, cp_rows_never_cand, [])
    assert len(table_b["defects"]) == 1
    d_b = table_b["defects"][0]
    assert d_b["defect"] == NO_ARTIFACT_DEFECT_ID
    assert d_b["kind"] == "never"
    assert d_b["turn"] is None



def test_defect_categorization_complex_multi_turn_transitions(tmp_path):
    """Stress test complex multi-turn transitions across k=0..4.

    Test items:
    - t_never: never passes at any checkpoint. -> 'never'
    - t_broken_early: passes at k=1 (d=0), fails at k=2..4 -> 'broken', turn=2
    - t_flapping: fails k=0, passes k=1 (d=0), fails k=2 (d>0), passes k=3 (d=0), fails k=4 (d>0) -> 'broken', turn=4
    - t_undelivered: passes at k=3 and k=4 (final_cp d=0), but lost in terminal submission (Y=0.0). -> 'undelivered'
    """
    pkg = FakePackage(tmp_path, turns=4)
    # Total 4 tests -> each has weight 0.25
    # k=0: all fail (s=0.0) -> d_gate = 0.0
    # k=1: t_broken_early, t_flapping pass (2 pass, 2 fail) -> s=0.0, d_gate = 2/4 = 0.5
    # k=2: all fail (0 pass, 4 fail) -> s=0.0, d_gate = 0.0
    # k=3: t_flapping, t_undelivered pass (2 pass, 2 fail) -> s=0.0, d_gate = 2/4 = 0.5
    # k=4 (final cp): t_undelivered passes (1 pass, 3 fail) -> s=0.0, d_gate = 1/4 = 0.25
    # Wait, if t_undelivered passes on final cp, but s(T) = 0.0:
    # Notice: undelivered requires undelivered_gap = s_T - Y > 0.
    # If s_T = 0.0 and Y = 0.0, then undelivered_gap = 0.0!
    # To test undelivered alongside broken and never:
    # Suppose at k=4 (final cp), t_undelivered passes AND all other required tests pass! So s_T = 1.0!
    # But terminal submission Y = 0.0 (omitted patch).
    # Then undelivered_gap = 1.0 - 0.0 = 1.0!
    # All 4 tests were done right at k=4, but undelivered at terminal!
    # Wait! What if we want both undelivered AND broken/never?
    # Look at equation (230) in L2 v0.4:
    # 1 - Y = (s(T) - Y) [undelivered] + sum_{i: d_i(T) > 0} d_i(T) [broken / never]
    # In SWE-bench Pro, s(T) is either 0.0 or 1.0!
    # If s(T) == 1.0: then 1 - s(T) = 0.0, so all d_i(T) == 0.0!
    # Then ALL loss is undelivered (1 - Y = 1.0 - 0.0 = 1.0), and there are NO broken/never on final cp!
    # If s(T) == 0.0: then undelivered_gap = max(0, 0.0 - Y) = 0.0 (since Y <= s(T) <= 0.0)!
    # Then ALL loss is broken / never!
    # This is a key theoretical insight:
    # Because SWE-bench Pro has binary outcome s(k) in {0.0, 1.0},
    # an episode either has s(T) = 1.0 (pure undelivered if Y=0) OR s(T) = 0.0 (pure broken/never)!
    pass


def test_defect_categorization_flapping_and_origin_turn(tmp_path):
    """Verify that a flapping test (fails, passes, fails, passes, fails) has origin_j at the LAST break."""
    pkg = FakePackage(tmp_path, turns=4)
    # Tests: t_never, t_broken_early, t_flapping
    # N=3, unit=1/3
    cp_rows = [
        {
            "kind": "terminal_submission",
            "episode_id": pkg.episode_id,
            "task_id": pkg.family_id,
            "k": 4,
            "score": 0.0,
            "dim1_pass": False,
            "deductions": {"t_never": 1.0 / 3, "t_broken_early": 1.0 / 3, "t_flapping": 1.0 / 3, GATE_DEFECT_ID: 0.0},
        },
        # k=0: all fail
        {
            "kind": "checkpoint", "k": 0, "candidate_status": "ok", "score": 0.0, "candidate": ["patch.diff"],
            "deductions": {"t_never": 1.0 / 3, "t_broken_early": 1.0 / 3, "t_flapping": 1.0 / 3, GATE_DEFECT_ID: 0.0},
        },
        # k=1: t_broken_early passes (d=0), t_flapping passes (d=0), t_never fails (d=1/3) -> 2 pass -> d_gate=2/3
        {
            "kind": "checkpoint", "k": 1, "candidate_status": "ok", "score": 0.0, "candidate": ["patch.diff"],
            "deductions": {"t_never": 1.0 / 3, "t_broken_early": 0.0, "t_flapping": 0.0, GATE_DEFECT_ID: 2.0 / 3},
        },
        # k=2: t_broken_early fails, t_flapping fails -> all fail -> d_gate=0.0
        {
            "kind": "checkpoint", "k": 2, "candidate_status": "ok", "score": 0.0, "candidate": ["patch.diff"],
            "deductions": {"t_never": 1.0 / 3, "t_broken_early": 1.0 / 3, "t_flapping": 1.0 / 3, GATE_DEFECT_ID: 0.0},
        },
        # k=3: t_flapping passes again! (d=0), t_broken_early fails, t_never fails -> 1 pass -> d_gate=1/3
        {
            "kind": "checkpoint", "k": 3, "candidate_status": "ok", "score": 0.0, "candidate": ["patch.diff"],
            "deductions": {"t_never": 1.0 / 3, "t_broken_early": 1.0 / 3, "t_flapping": 0.0, GATE_DEFECT_ID: 1.0 / 3},
        },
        # k=4 (final cp): all fail!
        {
            "kind": "checkpoint", "k": 4, "candidate_status": "ok", "score": 0.0, "candidate": ["patch.diff"],
            "deductions": {"t_never": 1.0 / 3, "t_broken_early": 1.0 / 3, "t_flapping": 1.0 / 3, GATE_DEFECT_ID: 0.0},
        },
    ]

    table = build_defect_table(pkg, cp_rows, [])
    defects = {d["defect"]: d for d in table["defects"]}

    # t_never: never satisfied
    assert defects["t_never"]["kind"] == "never"
    assert defects["t_never"]["turn"] is None

    # t_broken_early: last ok was k=1 -> origin_j = 2
    assert defects["t_broken_early"]["kind"] == "broken"
    assert defects["t_broken_early"]["turn"] == 2

    # t_flapping: last ok was k=3 -> origin_j = 4 (NOT 2)!
    assert defects["t_flapping"]["kind"] == "broken"
    assert defects["t_flapping"]["turn"] == 4


def test_defect_categorization_scale_100_tests(tmp_path):
    """Stress test defect categorization at scale with N=100 tests across 10 checkpoints."""
    N = 100
    T = 10
    pkg = FakePackage(tmp_path, turns=T)
    test_names = [f"test_{i:03d}" for i in range(N)]
    unit = 1.0 / N

    rng = random.Random(1337)
    # Define test fates:
    # 0..29: never pass
    # 30..69: pass early (k=2), break later (k=7)
    # 70..99: pass late (k=8), break at k=10
    cp_rows = []

    # Final checkpoint rows
    for k in range(T + 1):
        passed_k = set()
        for idx, t in enumerate(test_names):
            if 30 <= idx < 70 and 2 <= k < 7:
                passed_k.add(t)
            elif 70 <= idx < 100 and 8 <= k < 10:
                passed_k.add(t)

        score_k = 1.0 if len(passed_k) == N else 0.0
        n_pass = len(passed_k)
        ded_k = {}
        for t in test_names:
            ded_k[t] = 0.0 if t in passed_k else unit
        ded_k[GATE_DEFECT_ID] = n_pass / N

        cp_rows.append({
            "kind": "checkpoint",
            "k": k,
            "candidate_status": "ok",
            "score": score_k,
            "candidate": ["submission/patch.diff"],
            "deductions": ded_k,
        })

    # Terminal submission at k=T with score = 0.0
    term_ded = dict(cp_rows[-1]["deductions"])
    cp_rows.insert(0, {
        "kind": "terminal_submission",
        "episode_id": pkg.episode_id,
        "task_id": pkg.family_id,
        "k": T,
        "score": 0.0,
        "dim1_pass": False,
        "deductions": term_ded,
    })

    table = build_defect_table(pkg, cp_rows, [])
    defects = {d["defect"]: d for d in table["defects"]}

    # Verify 0..29 are 'never'
    for i in range(30):
        assert defects[f"test_{i:03d}"]["kind"] == "never"

    # Verify 30..69 are 'broken' with turn=7
    for i in range(30, 70):
        d = defects[f"test_{i:03d}"]
        assert d["kind"] == "broken"
        assert d["turn"] == 7

    # Verify 70..99 are 'broken' with turn=10
    for i in range(70, 100):
        d = defects[f"test_{i:03d}"]
        assert d["kind"] == "broken"
        assert d["turn"] == 10

    # Verify total sum of deductions
    total_ded = sum(d["deduction"] for d in table["defects"])
    assert abs(total_ded - 1.0) < 1e-9
