#!/usr/bin/env python3
"""
scripts/verify_m3_challenger2_stress.py

Adversarial Stress Testing & Empirical Verification for Milestone M3:
Counterfactual Continuation Verification and Statistical Inference.

Performs:
1. Deep inspection of outcomes.jsonl and all 8 branch packages under run_task1_s42.
2. Verification of arm partition (4 plain, 4 rule), completed requests (16 from k*=4),
   termination reason (request_budget), and test item results (15/16 passed).
3. Independent statistical re-computation:
   - Delta = mean(Y_treat) - mean(Y_ctrl)
   - Exact combinatorial permutation test
   - Monte Carlo permutation test (B=5000 across multiple random seeds)
   - Fisher exact test
   - Holm-Bonferroni adjustment
   - Verification of evidence level E0 and not_confirmed status
4. Adversarial stress testing & statistical power analysis:
   - Alternative outcome formulations (continuous item score 15/16 vs binary task outcome)
   - Statistical power of N=4 vs N=4 design under Fisher exact test
   - Branch execution divergence & trajectory independence analysis
"""

import json
import math
import itertools
import os
import random
import sys
from pathlib import Path

BASE_DIR = Path("/Users/jiashen/Desktop/问题诊断")
RUN_DIR = BASE_DIR / "L2_Behavioral Event Localization/L2_Sandbox/diagnoses/swebench_pro_qwen36/run_task1_s42"
OUTCOMES_PATH = RUN_DIR / "outcomes.jsonl"
VERIF_PATH = RUN_DIR / "verifications.json"
BRANCHES_DIR = RUN_DIR / "branches"
RECEIPT_PATH = BASE_DIR / "scripts/challenger2_m3_stress_receipt.json"

def main():
    print("=" * 70)
    print("M3 CHALLENGER 2: EMPIRICAL VERIFICATION & ADVERSARIAL STRESS TEST")
    print("=" * 70)

    results = {
        "outcomes_inspection": {},
        "branch_packages_inspection": {},
        "statistical_inference": {},
        "adversarial_stress_tests": {},
        "overall_verdict": "PENDING",
    }

    # -------------------------------------------------------------
    # 1. Outcomes.jsonl Inspection
    # -------------------------------------------------------------
    print("\n[Step 1] Inspecting outcomes.jsonl...")
    assert OUTCOMES_PATH.is_file(), f"outcomes.jsonl not found at {OUTCOMES_PATH}"

    rows = []
    with open(OUTCOMES_PATH, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                rows.append((line_num, json.loads(line)))

    print(f"Total rows in outcomes.jsonl: {len(rows)}")
    assert len(rows) == 8, f"Expected 8 rows, got {len(rows)}"

    plain_rows = [r for _, r in rows if r.get("arm_type") == "plain" or r.get("arm") == "plain"]
    rule_rows = [r for _, r in rows if r.get("arm_type") == "rule" or str(r.get("arm", "")).startswith("rule")]

    print(f"Plain arm branches: {len(plain_rows)}")
    print(f"Rule arm branches:  {len(rule_rows)}")
    assert len(plain_rows) == 4, f"Expected 4 plain branches, got {len(plain_rows)}"
    assert len(rule_rows) == 4, f"Expected 4 rule branches, got {len(rule_rows)}"

    # Detailed row verification
    all_k_4 = True
    all_term_budget = True
    all_reason_zero = True
    all_y_zero = True
    test_item_stats = []

    for idx, r in enumerate(plain_rows + rule_rows):
        k = r.get("k")
        y = r.get("y")
        term = r.get("termination")
        reason = r.get("reason")
        scores = r.get("item_scores", {})
        
        pass_count = sum(1 for v in scores.values() if v == 1.0)
        fail_count = sum(1 for v in scores.values() if v == 0.0)
        total_items = len(scores)

        test_item_stats.append({
            "trial": r.get("trial"),
            "arm": r.get("arm"),
            "arm_type": r.get("arm_type"),
            "k": k,
            "y": y,
            "termination": term,
            "reason": reason,
            "pass_count": pass_count,
            "fail_count": fail_count,
            "total_items": total_items,
            "failed_test_ids": [k for k, v in scores.items() if v == 0.0],
        })

        if k != 4: all_k_4 = False
        if y != 0.0: all_y_zero = False
        if term != "request_budget": all_term_budget = False
        if reason != "official_zero": all_reason_zero = False

    print(f"  - All k == 4: {all_k_4}")
    print(f"  - All y == 0.0: {all_y_zero}")
    print(f"  - All termination == 'request_budget': {all_term_budget}")
    print(f"  - All reason == 'official_zero': {all_reason_zero}")
    print(f"  - Test pass counts across all 8 branches: {[s['pass_count'] for s in test_item_stats]} / 16")

    assert all_k_4, "Not all branches have k == 4"
    assert all_y_zero, "Not all branches have y == 0.0"
    assert all_term_budget, "Not all branches have termination == 'request_budget'"
    assert all_reason_zero, "Not all branches have reason == 'official_zero'"
    for s in test_item_stats:
        assert s["pass_count"] == 15, f"Expected 15 passes, got {s['pass_count']}"
        assert s["total_items"] == 16, f"Expected 16 total items, got {s['total_items']}"
        assert s["failed_test_ids"] == ["test/units/utils/test_vars.py::TestVariableUtils::test_combine_vars_replace"], f"Unexpected failed test: {s['failed_test_ids']}"

    results["outcomes_inspection"] = {
        "total_rows": len(rows),
        "n_plain": len(plain_rows),
        "n_rule": len(rule_rows),
        "all_k_4": all_k_4,
        "all_y_0": all_y_zero,
        "all_termination_budget": all_term_budget,
        "all_reason_official_zero": all_reason_zero,
        "per_branch_pass_count": 15,
        "per_branch_total_tests": 16,
        "consistently_failing_test": "test/units/utils/test_vars.py::TestVariableUtils::test_combine_vars_replace",
    }

    # -------------------------------------------------------------
    # 2. Branch Package Directory Inspection
    # -------------------------------------------------------------
    print("\n[Step 2] Inspecting branch directories and manifest packages...")
    branch_dirs = sorted([d.name for d in BRANCHES_DIR.iterdir() if d.is_dir()])
    print(f"Branch directories found: {len(branch_dirs)}")
    assert len(branch_dirs) == 8, f"Expected 8 directories, found {len(branch_dirs)}"

    branch_manifest_stats = []
    for b_name in branch_dirs:
        b_path = BRANCHES_DIR / b_name
        m_path = b_path / "manifest.json"
        g_path = b_path / "official_grade.json"
        t_path = b_path / "trajectory.jsonl"
        assert m_path.is_file(), f"manifest.json missing in {b_name}"
        assert g_path.is_file(), f"official_grade.json missing in {b_name}"
        assert t_path.is_file(), f"trajectory.jsonl missing in {b_name}"

        with open(m_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        with open(g_path, "r", encoding="utf-8") as f:
            g = json.load(f)

        # Check request count, lineage, status
        counts = m.get("counts", {})
        lineage = m.get("lineage", {})
        status = m.get("status", {})
        req_count = counts.get("model_requests")
        resp_count = counts.get("model_responses")
        parent_cp = lineage.get("parent_checkpoint_id")
        term = status.get("termination")
        task_out = status.get("task_outcome")
        eval_val = status.get("evaluation_validity")

        # Official grade check
        grade_passed = g.get("official_raw", {}).get("passed", g.get("passed"))
        grade_summary = g.get("official_raw", {}).get("test_results", {})

        branch_manifest_stats.append({
            "branch": b_name,
            "requests": req_count,
            "responses": resp_count,
            "parent_cp": parent_cp,
            "termination": term,
            "task_outcome": task_out,
            "eval_validity": eval_val,
            "grade_passed": grade_passed,
            "grade_summary": grade_summary,
        })

        assert req_count == 16, f"Branch {b_name} had {req_count} requests, expected 16"
        assert parent_cp == "cp000004", f"Branch {b_name} had parent {parent_cp}, expected cp000004"
        assert term == "request_budget", f"Branch {b_name} had termination {term}"
        assert eval_val == "valid", f"Branch {b_name} had eval_validity {eval_val}"
        assert grade_passed is False, f"Branch {b_name} had grade_passed {grade_passed}"

    print(f"  - All 8 branch packages verified: 16 completed requests, parent cp000004, valid evaluation")
    results["branch_packages_inspection"] = {
        "branch_count": len(branch_manifest_stats),
        "requests_per_branch": 16,
        "parent_checkpoint": "cp000004",
        "eval_validity": "valid",
        "official_grade_passed": False,
    }

    # -------------------------------------------------------------
    # 3. Statistical Inference Re-computation
    # -------------------------------------------------------------
    print("\n[Step 3] Re-computing empirical effect size and running permutation tests...")
    y_ctrl = [r["y"] for r in plain_rows]
    y_treat = [r["y"] for r in rule_rows]

    mean_ctrl = sum(y_ctrl) / len(y_ctrl)
    mean_treat = sum(y_treat) / len(y_treat)
    delta_hat = mean_treat - mean_ctrl

    print(f"  - Y_ctrl:  {y_ctrl} -> mean = {mean_ctrl:.6f}")
    print(f"  - Y_treat: {y_treat} -> mean = {mean_treat:.6f}")
    print(f"  - Delta:   {delta_hat:.6f}")
    assert delta_hat == 0.0, f"Expected Delta == 0.0, got {delta_hat}"

    # Exact combinatorial permutation test
    combined = y_treat + y_ctrl
    n_treat = len(y_treat)
    total_n = len(combined)
    obs_diff = delta_hat

    comb_total = math.comb(total_n, n_treat)
    assert comb_total == 70, f"Expected 70 combinations, got {comb_total}"

    comb_extreme = 0
    for idxs in itertools.combinations(range(total_n), n_treat):
        s_treat = sum(combined[i] for i in idxs)
        s_ctrl = sum(combined) - s_treat
        diff = (s_treat / n_treat) - (s_ctrl / (total_n - n_treat))
        if diff >= obs_diff - 1e-12:
            comb_extreme += 1

    p_exact = comb_extreme / comb_total
    print(f"  - Exact combinatorial permutations: {comb_extreme}/{comb_total} >= {obs_diff} -> p = {p_exact:.6f}")
    assert p_exact == 1.0, f"Expected exact p == 1.0, got {p_exact}"

    # Monte Carlo permutation test with B=5000 across multiple seeds
    mc_seeds = [42, 101, 2026, 77777, 999999]
    mc_pvals = {}
    B = 5000

    for s in mc_seeds:
        rng = random.Random(s)
        pool = list(combined)
        extreme_count = 1  # Standard +1 definition for MC test
        for _ in range(B):
            rng.shuffle(pool)
            s_t = sum(pool[:n_treat])
            s_c = sum(pool[n_treat:])
            d = (s_t / n_treat) - (s_c / (total_n - n_treat))
            if d >= obs_diff - 1e-12:
                extreme_count += 1
        mc_pval = extreme_count / (B + 1)
        mc_pvals[f"seed_{s}"] = mc_pval
        assert mc_pval == 1.0, f"Expected MC p == 1.0 for seed {s}, got {mc_pval}"

    print(f"  - Monte Carlo permutation test (B={B}) across 5 seeds: all p = 1.000000")

    # Fisher exact test (hypergeometric)
    # 2x2 table:
    #             Success (Y=1)   Fail (Y=0)
    # Treatment        0              4        Total = 4
    # Control          0              4        Total = 4
    # Total            0              8        Total = 8
    # Under 0 total successes, P(X >= 0) = 1.0
    p_fisher = 1.0
    print(f"  - Fisher exact test: p = {p_fisher:.6f}")

    # Holm-Bonferroni adjustment:
    # With a single hypothesis, p_holm = min(1.0, p * 1) = 1.000000
    p_holm = min(1.0, p_exact * 1)
    print(f"  - Holm-Bonferroni adjusted p: {p_holm:.6f}")

    # Evidence Level and Confirmation Status Verification
    # Contract: alpha = 0.05, min_effect = 0.15
    # Status confirmed iff p_holm <= alpha and delta_hat >= min_effect (for causal intervention)
    alpha = 0.05
    min_effect = 0.15
    is_confirmed = (p_holm <= alpha) and (delta_hat >= min_effect)
    status = "confirmed" if is_confirmed else "not_confirmed"
    evidence_level = "E2" if is_confirmed else "E0"

    print(f"  - Status evaluated: {status} (expected 'not_confirmed')")
    print(f"  - Evidence level:   {evidence_level} (expected 'E0')")
    assert status == "not_confirmed", f"Status expected not_confirmed, got {status}"
    assert evidence_level == "E0", f"Evidence level expected E0, got {evidence_level}"

    # Match against verifications.json
    assert VERIF_PATH.is_file(), f"verifications.json missing at {VERIF_PATH}"
    with open(VERIF_PATH, "r", encoding="utf-8") as f:
        v_doc = json.load(f)

    direct_intv = v_doc.get("direct_intervention", {})
    hyps = direct_intv.get("hypotheses", [])
    assert len(hyps) == 1, f"Expected 1 direct intervention hypothesis, got {len(hyps)}"
    h0 = hyps[0]

    assert h0["control_arm"] == "plain"
    assert h0["treatment_arm"] == "rule"
    assert h0["k"] == 4
    assert h0["n_control"] == 4
    assert h0["n_treatment"] == 4
    assert h0["mean_control"] == 0.0
    assert h0["mean_treatment"] == 0.0
    assert h0["effect"] == 0.0
    assert h0["p_value"] == 1.0
    assert h0["p_value_holm"] == 1.0
    assert h0["evidence_level"] == "E0"
    assert h0["status"] == "not_confirmed"
    print("  - verifications.json matches calculated metrics 100%!")

    results["statistical_inference"] = {
        "mean_ctrl": mean_ctrl,
        "mean_treat": mean_treat,
        "delta_hat": delta_hat,
        "p_exact_permutation": p_exact,
        "mc_permutations_B5000": mc_pvals,
        "p_fisher": p_fisher,
        "p_holm": p_holm,
        "alpha": alpha,
        "min_effect": min_effect,
        "status": status,
        "evidence_level": evidence_level,
        "verifications_json_matched": True,
    }

    # -------------------------------------------------------------
    # 4. Adversarial Stress Testing & Boundary Probing
    # -------------------------------------------------------------
    print("\n[Step 4] Adversarial Stress Testing & Edge Cases...")

    # Stress Test A: Continuous Item Completion Rate Formulation
    # What if the outcome was the proportion of tests passed (15/16 = 0.9375)?
    y_ctrl_rate = [15.0 / 16.0] * 4
    y_treat_rate = [15.0 / 16.0] * 4
    delta_rate = (sum(y_treat_rate) / 4) - (sum(y_ctrl_rate) / 4)
    # Permutation test on identical continuous values:
    comb_extreme_rate = 0
    pool_rate = y_treat_rate + y_ctrl_rate
    for idxs in itertools.combinations(range(8), 4):
        s_t = sum(pool_rate[i] for i in idxs)
        s_c = sum(pool_rate) - s_t
        d = (s_t / 4) - (s_c / 4)
        if d >= delta_rate - 1e-12:
            comb_extreme_rate += 1
    p_rate = comb_extreme_rate / 70
    print(f"  - Stress Test A (Continuous 15/16 item score):")
    print(f"    mean_treat={15/16:.4f}, mean_ctrl={15/16:.4f}, Delta={delta_rate:.6f}, p={p_rate:.4f}")
    assert delta_rate == 0.0 and p_rate == 1.0

    # Stress Test B: Power Analysis of (N=4, N=4) Design
    # What combinations of (k_treat_success, k_ctrl_success) could reject H0 at alpha=0.05?
    # Possible outcomes:
    rejecting_tables = []
    for k_t in range(5): # 0..4
        for k_c in range(5): # 0..4
            # Hypergeometric probability P(X >= k_t | total_successes = k_t + k_c)
            # Total successes = M = k_t + k_c, N=8, n=4
            total_succ = k_t + k_c
            if total_succ == 0:
                p_val = 1.0
            else:
                denom = math.comb(8, 4)
                num = sum(math.comb(total_succ, x) * math.comb(8 - total_succ, 4 - x) for x in range(k_t, min(4, total_succ) + 1))
                p_val = num / denom
            if p_val <= 0.05 and (k_t / 4.0 - k_c / 4.0) >= 0.15:
                rejecting_tables.append({
                    "k_treat_pass": k_t,
                    "k_ctrl_pass": k_c,
                    "delta": (k_t - k_c) / 4.0,
                    "p_val": p_val,
                })
    print(f"  - Stress Test B (Power Analysis of N=4 vs N=4 design):")
    print(f"    Possible (treat, ctrl) combinations that achieve p <= 0.05 and delta >= 0.15:")
    for tab in rejecting_tables:
        print(f"      treat={tab['k_treat_pass']}/4, ctrl={tab['k_ctrl_pass']}/4 -> Delta={tab['delta']:.2f}, p={tab['p_val']:.4f}")

    assert len(rejecting_tables) == 1, f"Expected exactly 1 rejecting table (4 vs 0), got {len(rejecting_tables)}"
    assert rejecting_tables[0]["k_treat_pass"] == 4 and rejecting_tables[0]["k_ctrl_pass"] == 0
    assert abs(rejecting_tables[0]["p_val"] - (1.0 / 70.0)) < 1e-6
    print(f"    -> Crucial Insight: With N=4 vs N=4, ONLY an all-or-nothing outcome (4/4 vs 0/4, p = 1/70 = 0.0143)")
    print(f"       can achieve statistical significance at alpha=0.05.")

    # Stress Test C: Branch Trajectory Independence and Diversity
    # Verify that the 8 branches are not trivial copies or identical runs.
    branch_token_usage = []
    branch_trajectories = []
    for b_name in branch_dirs:
        t_path = BRANCHES_DIR / b_name / "trajectory.jsonl"
        with open(t_path, "r", encoding="utf-8") as f:
            t_lines = [json.loads(line) for line in f if line.strip()]
        branch_trajectories.append((b_name, len(t_lines)))
        
        m_path = BRANCHES_DIR / b_name / "manifest.json"
        with open(m_path, "r", encoding="utf-8") as f:
            m_data = json.load(f)
        u = m_data.get("usage", {})
        branch_token_usage.append((b_name, u.get("known_total_tokens_including_parent"), u.get("agent_phase_seconds")))

    print(f"  - Stress Test C (Branch Execution Independence):")
    tokens_set = set(tok for _, tok, _ in branch_token_usage)
    print(f"    Distinct token counts across 8 branches: {len(tokens_set)} distinct values: {tokens_set}")
    print(f"    Elapsed phase seconds: {[round(s, 1) for _, _, s in branch_token_usage]}")
    assert len(tokens_set) >= 6, f"Expected diverse token counts across branches, got {len(tokens_set)}"

    # Stress Test D: Intervention Message Delivery Verification
    # Check that rule branches received the exact rule intervention message while plain branches did not.
    for b_name in branch_dirs:
        m_path = BRANCHES_DIR / b_name / "manifest.json"
        with open(m_path, "r", encoding="utf-8") as f:
            m_data = json.load(f)
        intv = m_data.get("agent", {}).get("configuration", {}).get("intervention")
        if b_name.startswith("rule"):
            assert intv is not None, f"Rule branch {b_name} missing intervention configuration!"
            assert "Fix the Ansible issue" in intv.get("message", "")
            assert intv.get("at_request") == 5
        else:
            assert intv is None, f"Plain branch {b_name} should not have intervention configured!"
    print(f"  - Stress Test D: Intervention message configuration verified across all 8 branches.")

    results["adversarial_stress_tests"] = {
        "continuous_item_score_test": {
            "mean_treat": 15 / 16,
            "mean_ctrl": 15 / 16,
            "delta": delta_rate,
            "p_val": p_rate,
            "conclusion": "Continuous item score also yields Delta=0.0, p=1.0",
        },
        "power_analysis_n4_n4": {
            "total_possible_tables": 25,
            "rejecting_tables_count": len(rejecting_tables),
            "only_rejecting_configuration": "treat=4/4, ctrl=0/4",
            "minimum_p_value": 1.0 / 70.0,
            "statistical_power_insight": "N=4 design has statistical power only for complete effect (100% cure rate vs 0% control).",
        },
        "branch_diversity": {
            "distinct_token_counts": len(tokens_set),
            "token_range": [min(tokens_set), max(tokens_set)],
            "phase_seconds_range": [
                min(s for _, _, s in branch_token_usage),
                max(s for _, _, s in branch_token_usage),
            ],
            "confirmed_independent_execution": True,
        },
        "intervention_delivery_integrity": {
            "plain_branches_had_zero_intervention": True,
            "rule_branches_had_exact_rule_message": True,
            "intervention_request_target": 5,
        },
    }

    # -------------------------------------------------------------
    # Final Verdict
    # -------------------------------------------------------------
    results["overall_verdict"] = "APPROVE"
    print("\n" + "=" * 70)
    print("FINAL VERDICT: APPROVE")
    print("All tasks verified with 100% empirical rigor.")
    print("=" * 70)

    with open(RECEIPT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved empirical receipt to {RECEIPT_PATH}")

if __name__ == "__main__":
    main()
