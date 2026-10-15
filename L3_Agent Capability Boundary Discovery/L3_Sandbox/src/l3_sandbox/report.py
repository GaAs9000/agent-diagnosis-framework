"""Offline, denominator-preserving summaries of frozen L3 trials.

Attempts are operational records, never independent statistical observations.
"""
from __future__ import annotations

import csv
import io
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, variance

from .common import L3Error, digest, read, safe_path, write
from .plan import verify_plan


def _versions(row):
    value = row.get("model_versions", [])
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return sorted(str(k) for k in value)
    return sorted(set(str(v) for v in value if v is not None))


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _bootstrap(values, samples, seed):
    """Resample whole families, retaining all their task contrasts together."""
    rng = random.Random(seed)
    families = sorted(values)
    estimates = []
    for _ in range(samples):
        draws = [rng.choice(families) for _ in families]
        estimates.append(mean(v for family in draws for v in values[family]))
    estimates.sort()
    return [estimates[int((len(estimates) - 1) * .025)],
            estimates[int((len(estimates) - 1) * .975)]]


def _contrast(name, weights, tasks, cells, analysis, drift, unknown_versions=False):
    differences = []
    clusters = defaultdict(list)
    missing = []
    unverified = []
    for task in tasks:
        tid = task["task_id"]
        required = [cells[(tid, cid)] for cid in weights]
        if any(cell["status"] != "complete" for cell in required):
            missing.append(tid)
            continue
        value = sum(weights[cid] * cells[(tid, cid)]["mean"] for cid in weights)
        differences.append({"task_id": tid, "effect": value})
        if task.get("family_verified") is True and task.get("family_id"):
            clusters[task["family_id"]].append(value)
        else:
            unverified.append(tid)
    complete = not missing
    result = {"name": name, "weights": weights, "status": "complete" if complete else "incomplete",
              "n_tasks_expected": len(tasks), "n_tasks_paired": len(differences),
              "missing_task_ids": missing, "unverified_family_task_ids": unverified,
              "task_effects": differences, "estimate": mean(d["effect"] for d in differences) if differences else None,
              "n_verified_families": len(clusters), "ci95": None, "ci_reason": None,
              "minimum_effect": analysis.get("minimum_effect"), "claim": "descriptive_only"}
    if drift:
        result.update(estimate=None, task_effects=[], ci_reason="model_version_drift_requires_stratified_design")
    elif unknown_versions:
        result["ci_reason"] = "unknown_model_version"
    elif missing:
        result["ci_reason"] = "incomplete_predetermined_cohort"
    elif unverified:
        result["ci_reason"] = "unverified_task_families"
    elif len(clusters) < 2:
        result["ci_reason"] = "fewer_than_two_verified_families"
    elif int(analysis.get("bootstrap_samples", 0)) < 1:
        result["ci_reason"] = "bootstrap_not_configured"
    else:
        result["ci95"] = _bootstrap(clusters, int(analysis["bootstrap_samples"]), int(analysis.get("seed", 0)))
    return result


def _condition_axes(condition):
    return ({"original": "O", "placebo": "P", "structure": "S"}.get(condition.get("context"), condition.get("structure", condition.get("structure_id"))),
            condition.get("max_model_requests", condition.get("budget", condition.get("budget_id"))))


def _verified_rows(directory, plan, rows):
    """Revalidate mutable result projections against receipts and sealed evidence.

    The runner is imported lazily: this path cannot invoke native execution.
    Missing receipt gaps are allowed only for the runner's exact interrupted-row
    projection and remain unobserved outcomes.
    """
    from . import run

    run._safe_tree(directory / "runs")
    jobs = {job["trial_id"]: job for job in plan["jobs"]}
    represented = set()
    verified = []
    for row in rows:
        job = jobs.get(row.get("trial_id"))
        index = row.get("attempt")
        if job is None or type(index) is not int or index < 0:
            raise L3Error("Result has unknown trial or invalid attempt identity")
        if index > plan.get("limits", {}).get("max_replacements_per_slot", 2):
            raise L3Error("Attempt exceeds frozen replacement limit")
        key = (job["trial_id"], index)
        if key in represented:
            raise L3Error("Duplicate result attempt")
        represented.add(key)
        attempt = directory / "runs" / job["trial_id"] / f"attempt_{index:03d}"
        if not attempt.is_dir():
            raise L3Error("Result attempt directory is missing")
        binding = run._binding(plan, job)
        receipt = attempt / "receipt.json"
        if not receipt.exists():
            interrupted = {**job, "config": None, "attempt": index, "binding": binding,
                           "accepted": False, "status": "interrupted",
                           "replacement_of": index - 1 if index else None}
            if digest(row) != digest(interrupted):
                raise L3Error("Unreceipted attempt is not a trusted interrupted projection")
            verified.append(row)
            continue
        saved = run._load_receipt(receipt, binding)
        projected = dict(saved)
        if saved.get("status") == "running" and row.get("status") == "interrupted":
            projected["status"] = "interrupted"
        if digest(row) != digest(projected):
            raise L3Error("Result row differs from attempt receipt")
        for field in ("trial_id", "task_id", "condition_id", "replicate", "config_sha256",
                      "family_id", "family_verified", "split"):
            if row.get(field) != job.get(field):
                raise L3Error("Receipt identity differs from frozen job: " + field)
        if row.get("config") is not None and digest(row["config"]) != job["config_sha256"]:
            raise L3Error("Receipt configuration differs from frozen job")
        if row.get("accepted") is True or row.get("status") == "execution_invalid":
            expected = row.get("expected_index_sha256")
            if not isinstance(expected, str) or len(expected) != 64:
                raise L3Error("Outcome receipt lacks an external evidence index digest")
            package = attempt / "episode"
            run._evidence(package, job, expected)
            derived = run._summarize(package, job, index)
            if row.get("accepted") is True:
                run._verify_summary(row, derived, job, index)
            elif any(key not in row or digest(row[key]) != digest(value) for key, value in derived.items()):
                raise L3Error("Receipt outcome differs from independently derived evidence")
        elif row.get("study_valid") is True or row.get("study_outcome_y") is not None:
            raise L3Error("Unaccepted attempt carries an unverified statistical outcome")
        verified.append(row)
    actual = set()
    for trial_root in (directory / "runs").glob("*"):
        if not trial_root.is_dir() or trial_root.name not in jobs:
            raise L3Error("Unexpected trial directory")
        for attempt in trial_root.iterdir():
            if not attempt.is_dir() or not attempt.name.startswith("attempt_"):
                raise L3Error("Unexpected attempt directory")
            try:
                index = int(attempt.name.removeprefix("attempt_"))
            except ValueError as exc:
                raise L3Error("Invalid attempt directory") from exc
            if attempt.name != f"attempt_{index:03d}":
                raise L3Error("Invalid attempt directory numbering")
            actual.add((trial_root.name, index))
    if actual != represented:
        raise L3Error("Run results omit existing attempt evidence")
    return verified


def summarize(directory: Path) -> dict:
    """Verify a frozen plan and write only derived report artifacts beside it."""
    directory = Path(directory)
    verify_plan(directory, verify_images=False)
    plan = read(directory / "plan.json")
    results_path = directory / "run_results.json"
    rows = read(results_path).get("rows", []) if results_path.exists() else []
    rows = _verified_rows(directory, plan, rows)
    jobs = plan["jobs"]
    tasks = plan["tasks"]
    if isinstance(tasks, dict):
        tasks = [dict(value, task_id=key) for key, value in tasks.items()]
    conditions = plan["conditions"]
    if isinstance(conditions, dict):
        conditions = [dict(value, condition_id=key) for key, value in conditions.items()]
    expected = {job["trial_id"]: job for job in jobs}
    if len(expected) != len(jobs):
        raise L3Error("Duplicate frozen trial identifiers")
    attempts = defaultdict(list)
    for row in rows:
        trial = row.get("trial_id")
        if trial not in expected:
            raise L3Error(f"Result has unknown trial_id: {trial}")
        job = expected[trial]
        for field in ("task_id", "condition_id", "replicate"):
            if row.get(field) != job[field]:
                raise L3Error(f"Result identity differs from frozen trial: {trial}/{field}")
        attempts[trial].append(row)
    selected = {}
    ignored_valid = 0
    invalid = []
    replacement_trials = []
    for trial, records in attempts.items():
        ordered = sorted(records, key=lambda row: row.get("attempt", 0))
        indices = [row.get("attempt", 0) for row in ordered]
        if len(indices) != len(set(indices)):
            raise L3Error(f"Duplicate attempt number: {trial}")
        valid = []
        for row in ordered:
            if row.get("study_valid") is True and row.get("accepted") is True:
                y = row.get("study_outcome_y")
                if not _number(y) or not 0 <= y <= 1:
                    raise L3Error(f"Valid trial lacks finite outcome in [0,1]: {trial}")
                valid.append(row)
            else:
                invalid.append(row)
        if valid:
            selected[trial] = valid[0]
            ignored_valid += len(valid) - 1
            if any(row.get("attempt", 0) < valid[0].get("attempt", 0) for row in ordered):
                replacement_trials.append(trial)
    cells = {}
    for task in tasks:
        for condition in conditions:
            tid, cid = task["task_id"], condition["condition_id"]
            frozen = [job for job in jobs if job["task_id"] == tid and job["condition_id"] == cid]
            observed = [selected[job["trial_id"]] for job in frozen if job["trial_id"] in selected]
            ys = [row["study_outcome_y"] for row in observed]
            delivered = [row.get("delivery_exists") for row in observed]
            dim1 = [row.get("dim1_pass", row.get("legal_delivery")) for row in observed]
            cells[(tid, cid)] = {"task_id": tid, "condition_id": cid, "family_id": task.get("family_id"),
                "nexpected": len(frozen), "nvalid": len(ys), "nmissing": len(frozen) - len(ys),
                "status": "complete" if len(ys) == len(frozen) and frozen else "incomplete",
                "mean": mean(ys) if ys else None, "sample_variance": variance(ys) if len(ys) > 1 else None,
                "nzero": sum(y == 0 for y in ys),
                "delivery_rate": mean(delivered) if delivered and all(isinstance(x, bool) for x in delivered) else None,
                "dim1_rate": mean(dim1) if dim1 and all(isinstance(x, bool) for x in dim1) else None,
                "delivery_unknown": sum(not isinstance(x, bool) for x in delivered),
                "dim1_unknown": sum(not isinstance(x, bool) for x in dim1),
                "terminations": dict(Counter(str(row.get("termination", "unknown")) for row in observed)),
                "zero_reasons": dict(Counter(str(row.get("zero_reason", "unspecified")) for row in observed if row["study_outcome_y"] == 0)),
                "model_versions": sorted({v for row in observed for v in _versions(row)}),
                "model_requests": _usage(observed, "model_requests"), "tool_calls": _usage(observed, "tool_calls"),
                "model_version_strata": _version_strata(observed),
                "tokens": _usage(observed, "known_total_tokens"),
                "unknown_usage_count": sum(row.get("unknown_usage_count", 0) for row in observed),
                "http_attempt_count": _usage(observed, "http_attempt_count"),
                "first_persistent_write_turn": [row.get("first_persistent_write_turn") for row in observed],
                "first_submission_write_turn": [row.get("first_submission_write_turn") for row in observed]}
    version_set = sorted({v for row in selected.values() for v in _versions(row)})
    drift = len(version_set) > 1
    unknown_versions = any(not _versions(row) for row in selected.values())
    analysis = plan.get("analysis", {})
    contrasts = []
    comparison = analysis.get("primary_comparison")
    if comparison:
        contrasts.append(_contrast("primary", {comparison[0]: 1, comparison[1]: -1}, tasks, cells, analysis, drift, unknown_versions))
    if plan.get("design") == "structure_budget":
        axes = {_condition_axes(condition): condition["condition_id"] for condition in conditions}
        budgets = sorted({axis[1] for axis in axes})
        for budget in budgets:
            for a, b in (("S", "P"), ("P", "O"), ("S", "O")):
                if (a, budget) in axes and (b, budget) in axes:
                    contrasts.append(_contrast(f"{a}-{b}@{budget}", {axes[(a, budget)]: 1, axes[(b, budget)]: -1}, tasks, cells, analysis, drift, unknown_versions))
        if len(budgets) == 2:
            low, high = budgets
            for a, b in (("S", "P"), ("P", "O"), ("S", "O")):
                if all((s, budget) in axes for s in (a, b) for budget in budgets):
                    weights = {axes[(a, high)]: 1, axes[(b, high)]: -1, axes[(a, low)]: -1, axes[(b, low)]: 1}
                    contrasts.append(_contrast(f"interaction:{a}-{b}", weights, tasks, cells, analysis, drift, unknown_versions))
    summary = {"experiment_id": plan["experiment_id"], "purpose": plan["purpose"],
        "interpretation": "exploratory_pilot_only" if plan["purpose"] == "pilot" else "controlled_validation_requires_independent_review",
        "automatic_capability_claim": False, "nexpected": len(jobs), "nvalid": len(selected),
        "nmissing": len(jobs) - len(selected), "nattempts": len(rows), "ninvalid_attempts": len(invalid),
        "attempt_usage": {field: _usage(rows, field) for field in ("known_total_tokens", "http_attempt_count", "model_requests", "tool_calls")},
        "invalid_classifications": dict(Counter(str(row.get("study_classification", "unknown")) for row in invalid)),
        "replacement_trials": replacement_trials, "nreplacement_trials": len(replacement_trials),
        "ignored_valid_attempts": ignored_valid, "missing_trial_ids": sorted(set(expected) - set(selected)),
        "model_versions": version_set, "model_version_drift": drift,
        "unknown_model_version_trials": sorted(trial for trial, row in selected.items() if not _versions(row)),
        "cells": list(cells.values()), "contrasts": contrasts,
        "m2_baseline": {"exported": False, "rows": [], "complete_cohorts": _baseline_cohorts(jobs, selected, cells),
                        "reason": "independent_parent_identity_export_not_enabled"}}
    write(directory / "summary.json", summary)
    out = io.StringIO()
    fieldnames = list(next(iter(cells.values()))) if cells else ["task_id", "condition_id"]
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    for cell in cells.values():
        writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in cell.items()})
    safe_path(directory / "summary.csv").write_text(out.getvalue(), encoding="utf-8")
    safe_path(directory / "report.md").write_text(_markdown(summary), encoding="utf-8")
    return summary


def _version_strata(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(_versions(row))].append(row["study_outcome_y"])
    return [{"model_versions": list(versions), "nvalid": len(ys), "mean": mean(ys)}
            for versions, ys in sorted(groups.items())]


def _baseline_cohorts(jobs, selected, cells):
    """Actual complete cohorts, retained for a future identity-aware M2 export."""
    cohorts = []
    for (tid, cid), cell in cells.items():
        if cell["status"] != "complete":
            continue
        frozen = [job for job in jobs if job["task_id"] == tid and job["condition_id"] == cid]
        configs = [job.get("config") for job in frozen]
        if not configs or any(not isinstance(config, dict) for config in configs):
            continue
        if len({digest(config) for config in configs}) != 1:
            continue
        observed = [selected[job["trial_id"]] for job in frozen]
        versions = {v for row in observed for v in _versions(row)}
        if len(versions) != 1 or any(not _versions(row) for row in observed):
            continue
        endpoint_versions = {row.get("endpoint_version") for row in observed}
        if len(endpoint_versions) != 1 or None in endpoint_versions:
            continue
        config = configs[0]
        version = next(iter(versions))
        endpoint = config.get("model", {}).get("base_url", config.get("model", {}).get("endpoint"))
        identity = {"task_id": tid, "condition_id": cid, "config": config,
                    "config_sha256": digest(config), "endpoint": endpoint, "model_version": version,
                    "outcome_endpoint_version": next(iter(endpoint_versions))}
        cohorts.append({**identity, "key": digest(identity), "n": cell["nvalid"], "p0": cell["mean"],
                        "sample_variance": cell["sample_variance"],
                        "trial_ids": [job["trial_id"] for job in frozen],
                        "selected_parent_exclusion_supported": False})
    return cohorts


def _usage(rows, field):
    values = [row.get(field) for row in rows]
    return {"known_total": sum(value for value in values if _number(value)),
            "unknown_trials": sum(not _number(value) for value in values)}


def _markdown(summary):
    lines = [f"# L3 report: {summary['experiment_id']}", "", summary["interpretation"], "",
             f"Frozen slots: {summary['nexpected']}; valid: {summary['nvalid']}; missing: {summary['nmissing']}.",
             f"Attempts: {summary['nattempts']}; invalid: {summary['ninvalid_attempts']}; replacement slots: {summary['nreplacement_trials']}.",
             "", "Attempts are not independent samples. Missing outcomes are not zero and do not establish no effect.",
             "Model version drift: " + str(summary["model_version_drift"]), "",
             "| Contrast | Paired tasks | Estimate | 95% family bootstrap CI | Status |",
             "|---|---:|---:|---|---|"]
    for contrast in summary["contrasts"]:
        lines.append(f"| {contrast['name']} | {contrast['n_tasks_paired']}/{contrast['n_tasks_expected']} | {contrast['estimate']} | {contrast['ci95'] or contrast['ci_reason']} | {contrast['status']} |")
    lines += ["", "No automatic E3/E4 significance, causal mechanism, or capability conclusion is produced.",
              "M2 export is disabled until independent parent identity and full configuration matching are supported.", ""]
    return "\n".join(lines)
