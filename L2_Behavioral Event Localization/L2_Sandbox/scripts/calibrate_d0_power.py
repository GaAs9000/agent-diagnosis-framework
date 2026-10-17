#!/usr/bin/env python3
"""Finite, offline synthetic procedure calibration; never benchmark efficacy.

Run from L2_Sandbox with PYTHONPATH=src. No model, scorer, network or Git calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

import l2_sandbox.v05 as protocol

from l2_sandbox.v05 import (
    DiagnosticUnitV05, E1Observation, E2ResponseVector,
    compute_cross_fitted_diagnostic_value_ceiling,
)

ARMS = ("P", "D", "S", "R_explicit", "R_single")
NS = (12, 14, 24, 48, 64)
SOURCES = [
    {"url": "https://www.cs.mcgill.ca/~colt2009/papers/012.pdf",
     "use": "Maurer and Pontil (2009), Theorem 11; production physical-group empirical Bernstein bound"},
    {"url": "https://itl.nist.gov/div898/software/dataplot/refman2/auxillar/exacbici.htm",
     "use": "Exact binomial inversion; boundary formulas below are one-sided tail error e"},
]


def boundary_resolution(n, m, k, alpha=0.05):
    """Match R9: e=alpha/[8(M+K)], difference endpoints receive e/2."""
    e = alpha / (8 * (m + k))
    return {"n": n, "M": m, "K": k, "endpoint_error": e,
            "zero_success_upper": 1 - e ** (1 / n),
            "best_difference_lower": 2 * (e / 2) ** (1 / n) - 1}


def minimum_n(m, k, tau=0.2, delta=0.1):
    # Search integers with the same strict inequalities as the production rule.
    n_abs = next(n for n in range(1, 100001)
                 if boundary_resolution(n, m, k)["zero_success_upper"] < tau)
    n_diff = next(n for n in range(1, 100001)
                  if boundary_resolution(n, m, k)["best_difference_lower"] > delta)
    return {"M": m, "K": k, "tau": tau, "delta": delta,
            "minimum_n_zero_success_upper_strictly_below_tau": n_abs,
            "minimum_n_best_difference_lower_strictly_above_delta": n_diff}


def synthetic_units(scenario, n, seed, unequal_weights=False):
    rng = random.Random(seed)
    units, probabilities = [], {}
    for i in range(24):
        good = "D" if i % 2 == 0 else "S"
        if scenario == "null":
            probs = dict.fromkeys(ARMS, 0.2)
        else:
            high, low = {"moderate": (0.65, 0.15), "strong": (0.8, 0.1)}[scenario]
            probs = dict.fromkeys(ARMS, low)
            probs[good] = high
        probabilities[f"u{i}"] = probs
        # A distinct PRNG draw for every branch; hashes label simulated packages,
        # they are NOT hashes or verification claims about real L1 evidence.
        outcomes = {a: [float(rng.random() < probs[a]) for _ in range(n)] for a in ARMS}
        ids = {a: [hashlib.sha256(f"synthetic/{seed}/{i}/{a}/{j}".encode()).hexdigest()
                   for j in range(n)] for a in ARMS}
        weight = float(1 + i % 3) if unequal_weights else 1.0
        units.append(DiagnosticUnitV05(
            unit_id=f"u{i}", episode_id=f"synthetic-parent-{i}", task_id=f"synthetic-task-{i}",
            benchmark="SYNTHETIC_ONLY", unit_level="rubric_item", defect_name=f"rule-{i}",
            deduction_weight=weight, rule_source="explicit", family_id=f"family-{i % 9}",
            inclusion_prob=1.0,
            e1_observation=E1Observation("never_succeeded_on_disk", "wrong_candidate", 0, 0, 0, weight),
            e2_response_vector=E2ResponseVector(evaluated=True, arm_outcomes=outcomes, branch_identities=ids),
            operational_label=None, evidence_status="inconclusive"))
    return units, probabilities


def summarize(values):
    ordered = sorted(values)
    def q(p):
        pos = (len(ordered) - 1) * p
        lo = int(pos)
        return ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (pos - lo)
    return {"min": ordered[0], "p05": q(0.05), "median": statistics.median(ordered),
            "p95": q(0.95), "max": ordered[-1], "mean": statistics.mean(ordered)}


def calibrate_cell(scenario, n, repetitions, seed, unequal_weights=False):
    samples = []
    for rep in range(repetitions):
        run_seed = f"{seed}/{scenario}/{n}/{rep}/{unequal_weights}"
        units, probs = synthetic_units(scenario, n, run_seed, unequal_weights)
        result = compute_cross_fitted_diagnostic_value_ceiling(
            units, estimand="conditional_frozen_dev_pool")
        weights = {u.unit_id: u.deduction_weight for u in units}
        denom = 2 * sum(weights.values())
        truth = sum(weights[r["unit_id"]] * (
            probs[r["unit_id"]][r["selected_unit_arm"]]
            - probs[r["unit_id"]][r["train_best_fixed_arm_weighted"]])
            for r in result["unit_evaluations"]) / denom
        selected_errors = sum(
            probs[r["unit_id"]][r["selected_unit_arm"]] < max(probs[r["unit_id"]].values())
            for r in result["unit_evaluations"]) / 48
        lb, ub = result["deduction_weighted_lb95"], result["deduction_weighted_ub95"]
        aux = result["family_population_auxiliary"]
        samples.append({"replicate": rep, "conditional_true_target": truth,
                        "estimate": result["deduction_weighted_ceiling"], "lb": lb, "ub": ub,
                        "halfwidth": (ub - lb) / 2, "gate": result["passes_v3_p3_gate"],
                        "covers_conditional_target": lb <= truth <= ub,
                        "selection_error_fraction": selected_errors,
                        "auxiliary_halfwidth": (aux["deduction_weighted_ub95"] - aux["deduction_weighted_lb95"]) / 2,
                        "auxiliary_gate": aux["passes_v3_p3_gate"]})
    gates = sum(s["gate"] for s in samples)
    # With 0/200 events the one-sided 95% MC upper bound is still about 1.49%.
    rate_se = math.sqrt((gates / repetitions) * (1 - gates / repetitions) / repetitions)
    return {"scenario": scenario, "n_per_arm": n, "split": [n // 2, n - n // 2],
            "repetitions": repetitions, "unequal_weights": unequal_weights,
            "oracle_gap": {"null": 0.0, "moderate": 0.25, "strong": 0.35}[scenario],
            "gate_events": gates, "gate_rate": gates / repetitions, "gate_rate_mc_se": rate_se,
            "zero_events_one_sided_95_mc_upper": 1 - 0.05 ** (1 / repetitions) if gates == 0 else None,
            "gate_rate_interpretation": "false_positive_rate_at_true_zero_vs_0.05_threshold" if scenario == "null" else "power_for_synthetic_heterogeneity_including_selection_errors",
            "empirical_conditional_target_coverage": sum(s["covers_conditional_target"] for s in samples) / repetitions,
            **{name: summarize([s[name] for s in samples]) for name in (
                "halfwidth", "conditional_true_target", "selection_error_fraction", "auxiliary_halfwidth")},
            "auxiliary_gate_events": sum(s["auxiliary_gate"] for s in samples),
            "samples": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--seed", default="approvedD0n14-20261018")
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 1000:
        parser.error("repetitions must lie in [1,1000]")
    families = ((1, 2), (10, 5), (288, 6))
    report = {
        "status": "synthetic_procedure_calibration_not_benchmark_efficacy",
        "scientific_efficacy": "not_tested", "seed": args.seed,
        "python_version": sys.version,
        "source_sha256": {"scripts/calibrate_d0_power.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          "src/l2_sandbox/v05.py": hashlib.sha256(Path(protocol.__file__).read_bytes()).hexdigest()},
        "execution": {"workers": 1, "model_calls": 0, "scorer_calls": 0},
        "sources": SOURCES,
        "definitions": {"R9": "e=.05/[8(M+K)]; zero UB=1-e^(1/n); best delta LB=2(e/2)^(1/n)-1; strict tau=.2 and delta=.1. These are necessary endpoint resolution checks, not sufficient power or full Holm adjudication.",
                        "D2": "Production conditional_frozen_dev_pool, alpha=.05, eta=alpha/8 per direction/tail/method and per target. 24 fixed synthetic units, nine families, five explicit arms, independent Bernoulli package draws. Own selection half picks per-unit best and global fixed best, evaluation half estimates their signed difference, then swap.",
                        "target": "Average of two conditional true policy values given their respective selections; computed from known probabilities, differs from oracle gap under selection errors.",
                        "limitations": "Finite Monte Carlo is not guaranteed coverage. Physical independence is assumed in simulation; real packages and correlated endpoints are not validated. Family auxiliary is descriptive only. Weighted and unweighted targets have separate coverage, not a joint claim."},
        "unadjusted_zero_of_14_upper_95": 1 - 0.05 ** (1 / 14),
        "resolution": [boundary_resolution(n, m, k) for m, k in families for n in NS],
        "strict_minimum_n": [minimum_n(m, k) for m, k in families],
        "monte_carlo": [calibrate_cell(s, n, args.repetitions, args.seed)
                        for s in ("null", "moderate", "strong") for n in NS],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "calibration.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Offline synthetic D0 procedure calibration", "", report["status"], "",
             "No benchmark/model/scorer execution. No guaranteed coverage claim. One CPU worker.", "",
             "R9 resolution is necessary, not sufficient, for complete adjudication.", "",
             "| M | K | n | zero-success UB | best difference LB |", "|---|---|---|---|---|"]
    lines += [f"| {r['M']} | {r['K']} | {r['n']} | {r['zero_success_upper']:.6f} | {r['best_difference_lower']:.6f} |" for r in report["resolution"]]
    lines += ["", "Strict minimum n for UB < .2 / best difference LB > .1:", ""]
    lines += [f"- M={r['M']}, K={r['K']}: {r['minimum_n_zero_success_upper_strictly_below_tau']} / {r['minimum_n_best_difference_lower_strictly_above_delta']}" for r in report["strict_minimum_n"]]
    lines += ["", "D2 conditional target uses alpha/8; n14 splits 7/7. MC gate is weighted LB > .05.", "", "| scenario | n | gate events/reps | halfwidth median [p05,p95] | mean selection error |", "|---|---|---|---|---|"]
    for r in report["monte_carlo"]:
        h = r["halfwidth"]
        lines.append(f"| {r['scenario']} | {r['n_per_arm']} | {r['gate_events']}/{r['repetitions']} | {h['median']:.4f} [{h['p05']:.4f},{h['p95']:.4f}] | {r['selection_error_fraction']['mean']:.4f} |")
    lines += ["", "All distributions are synthetic; family auxiliary and raw replicate results are in calibration.json.", "", "Sources:"]
    lines += [f"- {s['url']} — {s['use']}" for s in SOURCES]
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(args.output_dir), "cells": len(report['monte_carlo']), "repetitions_per_cell": args.repetitions}))


if __name__ == "__main__":
    main()
