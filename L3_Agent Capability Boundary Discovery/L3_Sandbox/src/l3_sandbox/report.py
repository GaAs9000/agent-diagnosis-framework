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
from .plan import verify_plan, provenance_inventory


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
        # The run table may hold the original receipt projection or the current
        # reanalysis. Verify either below; never trust its numerical fields.
        original_projection = digest(row) == digest(projected)
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
            projected = run._project_receipt(saved, derived, job, index, plan, package)
            if not original_projection and digest(row) != digest(projected):
                if run._analysis_matches(plan, saved):
                    raise L3Error("Result row differs from attempt receipt or current projection")
                # Older derived tables can contain a previous endpoint version.
                # Their scientific fields are discarded, but execution anchors
                # must still agree with the immutable attempt receipt.
                for field in ('binding', 'expected_index_sha256', 'episode', 'code_provenance',
                              'status', 'replacement_of'):
                    if row.get(field) != saved.get(field):
                        raise L3Error("Result execution anchor differs from receipt: " + field)
        elif row.get("study_valid") is True or row.get("study_outcome_y") is not None:
            raise L3Error("Unaccepted attempt carries an unverified statistical outcome")
        else:
            if not original_projection:
                raise L3Error("Result row differs from attempt receipt")
        verified.append(projected)
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
    verify_plan(directory, verify_images=False, verify_execution=False)
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
    if isinstance(comparison, list) and len(comparison) == 2:
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
        "analysis_code_provenance": provenance_inventory(),
        "automatic_capability_claim": False, "nexpected": len(jobs), "nvalid": len(selected),
        "nmissing": len(jobs) - len(selected), "nattempts": len(rows), "ninvalid_attempts": len(invalid),
        "attempt_usage": {field: _usage(rows, field) for field in ("known_total_tokens", "http_attempt_count", "model_requests", "tool_calls")},
        "invalid_classifications": dict(Counter(str(row.get("study_classification", "unknown")) for row in invalid)),
        "replacement_trials": replacement_trials, "nreplacement_trials": len(replacement_trials),
        "ignored_valid_attempts": ignored_valid, "missing_trial_ids": sorted(set(expected) - set(selected)),
        "model_versions": version_set, "model_version_drift": drift,
        "unknown_model_version_trials": sorted(trial for trial, row in selected.items() if not _versions(row)),
        "cells": list(cells.values()), "contrasts": contrasts,
        "m2_baseline": _m2_baseline_summary(plan, jobs, selected, cells)}
    family = plan.get("family_manifest", {})
    manifest = family.get("manifest", {})
    summary["family_metadata"] = {
        "definition": manifest.get("family_definition", "unknown"),
        "manifest_sha256": family.get("sha256"),
        "manifest_path": family.get("path"),
        "lineage_status": manifest.get("lineage_status", "unknown"),
        "verified_meaning": manifest.get("family_verified_meaning", "unknown"),
        "n_tasks": len(tasks),
        "n_families": len({task["family_id"] for task in tasks if task.get("family_id")}),
    }
    write(directory / "summary.json", summary)
    if summary["m2_baseline"]["exported"]:
        write(directory / "m2_baseline.json", summary["m2_baseline"])
    out = io.StringIO()
    fieldnames = list(next(iter(cells.values()))) if cells else ["task_id", "condition_id"]
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    for cell in cells.values():
        writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in cell.items()})
    safe_path(directory / "summary.csv").write_text(out.getvalue(), encoding="utf-8")
    safe_path(directory / "report.md").write_text(_markdown(summary, plan), encoding="utf-8")
    return summary


def _version_strata(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(_versions(row))].append(row["study_outcome_y"])
    return [{"model_versions": list(versions), "nvalid": len(ys), "mean": mean(ys)}
            for versions, ys in sorted(groups.items())]


def _baseline_cohorts(jobs, selected, cells):
    """Complete task/condition cohorts with per-trial anchors for parent-excluded p0."""
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
        has_anchors = (
            len(observed) >= 2
            and all(isinstance(r.get("expected_index_sha256"), str) and len(r["expected_index_sha256"]) == 64
                    for r in observed)
            and len({r["expected_index_sha256"] for r in observed}) == len(observed)
        )
        trials = []
        for job, row in zip(frozen, observed):
            others = [r["study_outcome_y"] for r in observed if r["trial_id"] != job["trial_id"]]
            trials.append({
                "trial_id": job["trial_id"],
                "replicate": job.get("replicate"),
                "attempt": row.get("attempt", 0),
                "episode_id": row.get("episode_id"),
                "expected_index_sha256": row.get("expected_index_sha256"),
                "package_path": row.get("package_path"),
                "study_outcome_y": row["study_outcome_y"],
                "p0_excluding_self": mean(others) if others and has_anchors else None,
            })
        cohorts.append({**identity, "key": digest(identity), "n": cell["nvalid"], "p0": cell["mean"],
                        "sample_variance": cell["sample_variance"],
                        "trial_ids": [job["trial_id"] for job in frozen],
                        "trials": trials,
                        "selected_parent_exclusion_supported": has_anchors})
    return cohorts


def _m2_baseline_summary(plan, jobs, selected, cells):
    cohorts = _baseline_cohorts(jobs, selected, cells)
    exportable = [c for c in cohorts if c["selected_parent_exclusion_supported"]]
    rows = [
        {
            "cohort_key": c["key"],
            "task_id": c["task_id"],
            "condition_id": c["condition_id"],
            "config_sha256": c["config_sha256"],
            "model_version": c["model_version"],
            "outcome_endpoint_version": c["outcome_endpoint_version"],
            "trial_id": t["trial_id"],
            "episode_id": t["episode_id"],
            "expected_index_sha256": t["expected_index_sha256"],
            "study_outcome_y": t["study_outcome_y"],
            "p0_excluding_self": t["p0_excluding_self"],
            "n_excluding_self": c["n"] - 1,
        }
        for c in exportable
        for t in c["trials"]
    ]
    exported = bool(plan.get("design") == "baseline" and cohorts and len(exportable) == len(cohorts))
    return {
        "exported": exported,
        "rows": rows if exported else [],
        "complete_cohorts": cohorts,
        "reason": None if exported else "independent_parent_identity_export_not_enabled",
    }


def resolve_baseline_p0(source: Path, parent_episode: Path, *, condition_id: str | None = None) -> dict:
    """Resolve task/config-matched baseline p_0 while strictly excluding the parent episode itself."""
    import hashlib
    from c3_sandbox.l1.evaluation.study import ENDPOINT_VERSION

    source = safe_path(source)
    parent_episode = safe_path(parent_episode)
    if source.is_dir():
        summary_doc = read(source / "summary.json") if (source / "summary.json").is_file() else summarize(source)
        baseline_doc = summary_doc.get("m2_baseline") or {}
    else:
        doc = read(source)
        baseline_doc = doc.get("m2_baseline") if isinstance(doc, dict) and "m2_baseline" in doc else doc
    cohorts = (baseline_doc or {}).get("complete_cohorts") or []
    if not cohorts:
        raise L3Error("No complete baseline cohorts available in source")

    manifest_path = parent_episode / "manifest.json"
    cfg_path = parent_episode / "raw/resolved_config.json"
    idx_path = parent_episode / "evidence_index.json"
    if not (manifest_path.is_file() and cfg_path.is_file() and idx_path.is_file()):
        raise L3Error("Parent episode lacks manifest.json, raw/resolved_config.json, or evidence_index.json")
    manifest = read(manifest_path)
    resolved_cfg = read(cfg_path)
    parent_cfg_sha = digest(resolved_cfg)
    if manifest.get("config_sha256") != parent_cfg_sha:
        raise L3Error("Parent episode manifest config_sha256 disagrees with raw/resolved_config.json")
    parent_task_id = (manifest.get("task") or {}).get("task_id") or (manifest.get("task") or {}).get("id")
    parent_episode_id = manifest.get("episode_id")
    parent_index_sha = hashlib.sha256(idx_path.read_bytes()).hexdigest()

    traj_path = parent_episode / "trajectory.jsonl"
    parent_versions = sorted({
        json.loads(line).get("data", {}).get("reported_model_version")
        for line in (traj_path.read_text(encoding="utf-8").splitlines() if traj_path.is_file() else [])
        if line.strip() and json.loads(line).get("kind") == "model.parsed"
        and json.loads(line).get("data", {}).get("reported_model_version")
    })

    matched = [
        c for c in cohorts
        if c.get("task_id") == parent_task_id
        and c.get("config_sha256") == parent_cfg_sha
        and (condition_id is None or c.get("condition_id") == condition_id)
    ]
    if not matched:
        raise L3Error(f"No complete baseline cohort matches task={parent_task_id} and config_sha256={parent_cfg_sha}")
    if len(matched) > 1:
        raise L3Error("Multiple baseline cohorts match parent task/config; specify condition_id")
    cohort = matched[0]
    if not cohort.get("selected_parent_exclusion_supported"):
        raise L3Error("Matched cohort lacks sealed episode anchors required for parent exclusion")
    if parent_versions and cohort.get("model_version") not in parent_versions:
        raise L3Error("Parent episode modelVersion differs from baseline cohort modelVersion")
    if cohort.get("outcome_endpoint_version") != ENDPOINT_VERSION:
        raise L3Error("Baseline cohort outcome_endpoint_version differs from current study endpoint")

    trials = cohort.get("trials") or []
    excluded = [
        t for t in trials
        if t.get("expected_index_sha256") == parent_index_sha
        or (parent_episode_id and t.get("episode_id") and t.get("episode_id") == parent_episode_id)
    ]
    if len(excluded) > 1:
        raise L3Error("Parent episode matches multiple cohort slots; duplicate baseline evidence")
    remaining = [t for t in trials if t not in excluded]
    if not remaining:
        raise L3Error("No baseline trials remain after excluding parent episode")

    p0_val = mean(float(t["study_outcome_y"]) for t in remaining)
    excluded_id = excluded[0]["trial_id"] if excluded else None
    return {
        "p0": p0_val,
        "task_id": parent_task_id,
        "condition_id": cohort["condition_id"],
        "config_sha256": parent_cfg_sha,
        "model_version": cohort["model_version"],
        "outcome_endpoint_version": cohort["outcome_endpoint_version"],
        "cohort_key": cohort["key"],
        "cohort_n": len(trials),
        "used_n": len(remaining),
        "parent_in_cohort": bool(excluded),
        "excluded_trial_id": excluded_id,
        "parent_index_sha256": parent_index_sha,
        "provenance": (
            f"L3 cohort {cohort['key'][:16]} ({cohort['condition_id']}); "
            f"used {len(remaining)}/{len(trials)} valid runs; "
            f"excluded_parent_trial={excluded_id or 'external_verified_distinct'}"
        ),
    }


def _usage(rows, field):
    values = [row.get(field) for row in rows]
    return {"known_total": sum(value for value in values if _number(value)),
            "unknown_trials": sum(not _number(value) for value in values)}


def _display(value):
    return "unknown" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)


def _counts(values):
    return "; ".join(f"{key}: {value}" for key, value in sorted(values.items())) or "none observed"


def _pooled(cells):
    """Descriptive slot-weighted means, never an average of unequal cell means."""
    n = sum(cell["nvalid"] for cell in cells)
    result = {field: sum(cell[field] for cell in cells) for field in ("nexpected", "nvalid", "nmissing", "nzero")}
    for field, unknown in (("mean", None), ("delivery_rate", "delivery_unknown"), ("dim1_rate", "dim1_unknown")):
        known = all(not cell["nvalid"] or (cell[field] is not None and (unknown is None or not cell[unknown])) for cell in cells)
        result[field] = sum(cell[field] * cell["nvalid"] for cell in cells if cell["nvalid"]) / n if n and known else None
    result["terminations"] = dict(sum((Counter(cell["terminations"]) for cell in cells), Counter()))
    for field in ("model_requests", "tool_calls"):
        result[field] = {key: sum(cell[field][key] for cell in cells) for key in ("known_total", "unknown_trials")}
    return result


def _use(value):
    return f"{value['known_total']} (unknown trials: {value['unknown_trials']})"


def _markdown(summary, plan=None):
    plan = plan or {}
    lines = [f"# L3 report: {summary['experiment_id']}", "", summary["interpretation"], "",
             f"Frozen slots: {summary['nexpected']}; valid: {summary['nvalid']}; missing: {summary['nmissing']}.",
             f"Attempts: {summary['nattempts']}; invalid: {summary['ninvalid_attempts']}; replacement slots: {summary['nreplacement_trials']}.",
             "", "Attempts are not independent samples. Missing outcomes are not zero and do not establish no effect.",
             "Observed means and delivery/legal rates below describe valid observed slots only; incomplete cells are not complete-cohort estimates.",
             "Model version drift: " + str(summary["model_version_drift"]), ""]
    family = summary.get("family_metadata", {})
    lines += [f"Families: {family.get('n_tasks', 'unknown')} tasks / {family.get('n_families', 'unknown')} families; definition: {family.get('definition', 'unknown')}.",
              f"Frozen manifest SHA-256: {family.get('manifest_sha256') or 'unknown'}; path: {family.get('manifest_path') or 'unknown'}.",
              f"Template lineage: {family.get('lineage_status', 'unknown')}; statistical independence is not certified.",
              f"Family verification meaning: {family.get('verified_meaning', 'unknown')}.", "",
              "## Per-condition observations", "",
              "| Condition | Expected | Valid | Missing | Observed mean completion | Delivery rate | Legal rate | Zeros | Requests used | Tools used | Terminations (all observed valid slots) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|"]
    groups = defaultdict(list)
    for cell in summary["cells"]:
        groups[cell["condition_id"]].append(cell)
    for cid, cells in groups.items():
        cell = _pooled(cells)
        lines.append(f"| {cid} | {cell['nexpected']} | {cell['nvalid']} | {cell['nmissing']} | {_display(cell['mean'])} | {_display(cell['delivery_rate'])} | {_display(cell['dim1_rate'])} | {cell['nzero']} | {_use(cell['model_requests'])} | {_use(cell['tool_calls'])} | {_counts(cell['terminations'])} |")
    budgets = {json.dumps(job.get("config", {}).get("budget", {}), sort_keys=True) for job in plan.get("jobs", [])}
    lines += ["", "Frozen execution budgets (check tool and wall-clock ceilings alongside termination counts):"]
    lines += [f"- `{budget}`" for budget in sorted(budgets)]
    lines += ["", "Request/tool totals above include selected valid outcomes only; attempt totals below include failed attempts and replacements.",
              "", "## Per-task observations", "",
              "| Task | Family | Condition | Valid / expected | Missing | Observed mean | Delivery rate | Legal rate | Zeros | Terminations |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for cell in summary["cells"]:
        lines.append(f"| {cell['task_id']} | {cell.get('family_id') or 'unknown'} | {cell['condition_id']} | {cell['nvalid']}/{cell['nexpected']} | {cell['nmissing']} | {_display(cell['mean'])} | {_display(cell['delivery_rate'])} | {_display(cell['dim1_rate'])} | {cell['nzero']} | {_counts(cell['terminations'])} |")
    lines += ["", "## Predetermined contrasts", "",
              "Contrasts use paired complete task cells; they differ from pooled observed-slot means when cells are incomplete.", "",
              "| Contrast (condition weights) | Paired tasks | Estimate | Minimum meaningful effect | 95% family bootstrap CI | Status |",
              "|---|---:|---:|---:|---|---|"]
    for contrast in summary["contrasts"]:
        weights = _counts(contrast["weights"])
        lines.append(f"| {contrast['name']} ({weights}) | {contrast['n_tasks_paired']}/{contrast['n_tasks_expected']} | {_display(contrast['estimate'])} | {_display(contrast['minimum_effect'])} | {contrast['ci95'] or contrast['ci_reason']} | {contrast['status']} |")
    lines += ["", "## Usage across all attempts", "",
              "| Measure | Known total | Trials with unknown usage |", "|---|---:|---:|"]
    for field, value in summary["attempt_usage"].items():
        lines.append(f"| {field} | {value['known_total']} | {value['unknown_trials']} |")
    unknown_receipts = sum(cell.get("unknown_usage_count", 0) for cell in summary["cells"])
    lines += ["", f"Selected valid outcomes contain {unknown_receipts} HTTP attempts with unknown usage; known token totals may be incomplete.",
              "Actual monetary cost is unknown; reconcile with the provider bill before estimating subsequent batches.",
              "No automatic E3/E4 significance, causal mechanism, or capability conclusion is produced.",
              "M2 export is disabled until independent parent identity and full configuration matching are supported.", ""]
    return "\n".join(lines)
