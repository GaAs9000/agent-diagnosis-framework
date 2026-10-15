import json
from pathlib import Path

import pytest

from l3_sandbox import report


def experiment(tmp_path, monkeypatch, *, tasks=2, replicates=2, structure=False):
    monkeypatch.setattr(report, "verify_plan", lambda *a, **kw: None)
    monkeypatch.setattr(report, "_verified_rows", lambda directory, plan, rows: rows)
    task_rows = [{"task_id": f"t{i}", "family_id": f"f{i}", "family_verified": True, "split": "development"} for i in range(tasks)]
    conditions = ([{"condition_id": f"{context}-{budget}", "context": context, "max_model_requests": budget}
                   for context in ("original", "placebo", "structure") for budget in (5, 10)] if structure else
                  [{"condition_id": "control"}, {"condition_id": "treatment"}])
    jobs, rows = [], []
    for task in task_rows:
        for condition in conditions:
            for replicate in range(replicates):
                job = {**task, "condition_id": condition["condition_id"], "replicate": replicate,
                       "trial_id": f"{task['task_id']}-{condition['condition_id']}-{replicate}"}
                jobs.append(job)
                y = .5 if condition["condition_id"] == "treatment" else 0
                if structure:
                    y = {"original-5": 0, "placebo-5": .1, "structure-5": .2,
                         "original-10": .1, "placebo-10": .3, "structure-10": .8}[condition["condition_id"]]
                rows.append({**job, "attempt": 1, "accepted": True, "study_valid": True, "study_outcome_y": y,
                             "endpoint_version": "study-outcome-v1",
                             "model_versions": ["v1"], "delivery_exists": bool(y), "legal_delivery": bool(y), "dim1_pass": bool(y),
                             "known_total_tokens": 12, "unknown_usage_count": 0,
                             "termination": "request_budget", "model_requests": 5, "tool_calls": 4})
    plan = {"experiment_id": "test", "purpose": "pilot", "design": "structure_budget" if structure else "budget",
            "tasks": task_rows, "conditions": conditions, "jobs": jobs, "replicates": replicates,
            "analysis": {"primary_comparison": [conditions[-1]["condition_id"], conditions[0]["condition_id"]],
                         "bootstrap_samples": 200, "seed": 9, "minimum_effect": .1}}
    def save():
        (tmp_path / "plan.json").write_text(json.dumps(plan))
        (tmp_path / "run_results.json").write_text(json.dumps({"rows": rows}))
    save()
    return plan, rows, save


def test_zero_denominator_and_outputs(tmp_path, monkeypatch):
    experiment(tmp_path, monkeypatch)
    result = report.summarize(tmp_path)
    control = result["cells"][0]
    assert (control["nvalid"], control["nzero"], control["mean"], control["sample_variance"]) == (2, 2, 0, 0)
    assert control["delivery_rate"] == control["dim1_rate"] == 0
    assert result["contrasts"][0]["estimate"] == .5
    assert result["contrasts"][0]["ci95"] == [.5, .5]
    assert result["automatic_capability_claim"] is False
    assert result["interpretation"] == "exploratory_pilot_only"
    assert all((tmp_path / name).exists() for name in ("summary.json", "summary.csv", "report.md"))


def test_missing_invalid_and_replacement_are_not_samples(tmp_path, monkeypatch):
    plan, rows, save = experiment(tmp_path, monkeypatch)
    original = rows.pop(0)
    rows.append({**original, "study_valid": False, "study_outcome_y": None, "study_classification": "infrastructure_invalid"})
    rows.append({**original, "attempt": 2})
    rows.append({**original, "attempt": 3, "study_outcome_y": 1})
    rows.pop(0)  # a different slot remains missing
    save()
    result = report.summarize(tmp_path)
    assert result["nvalid"] == len(plan["jobs"]) - 1
    assert result["ninvalid_attempts"] == result["nreplacement_trials"] == result["ignored_valid_attempts"] == 1
    assert result["nmissing"] == 1
    assert result["cells"][0]["mean"] == 0  # no best-of retry selection
    assert result["contrasts"][0]["status"] == "incomplete"
    assert result["contrasts"][0]["n_tasks_paired"] == 1
    assert result["contrasts"][0]["ci95"] is None
    assert result["contrasts"][0]["ci_reason"] == "incomplete_predetermined_cohort"


def test_pair_by_task_not_attempt_or_replicate(tmp_path, monkeypatch):
    plan, rows, save = experiment(tmp_path, monkeypatch, tasks=3)
    for row in rows:
        if row["condition_id"] == "treatment":
            row["study_outcome_y"] = int(row["task_id"][1]) / 2
    save()
    result = report.summarize(tmp_path)
    assert result["contrasts"][0]["n_tasks_paired"] == 3
    assert result["contrasts"][0]["estimate"] == .5
    assert report.summarize(tmp_path) == result


def test_family_cluster_bootstrap_keeps_sibling_tasks_together():
    values = {"familyA": [0, 0], "familyB": [1]}
    ci = report._bootstrap(values, 1000, 33)
    assert ci == [0, 1]
    assert report._bootstrap(values, 1000, 33) == ci


@pytest.mark.parametrize("verified", [True, False])
def test_one_family_or_unverified_has_no_ci(tmp_path, monkeypatch, verified):
    plan, rows, save = experiment(tmp_path, monkeypatch)
    for task in plan["tasks"]:
        task.update(family_id="same-family", family_verified=verified)
    save()
    contrast = report.summarize(tmp_path)["contrasts"][0]
    assert contrast["ci95"] is None
    assert contrast["estimate"] == .5
    assert contrast["ci_reason"] == ("fewer_than_two_verified_families" if verified else "unverified_task_families")


def test_drift_not_silently_pooled(tmp_path, monkeypatch):
    _, rows, save = experiment(tmp_path, monkeypatch)
    rows[0]["model_versions"] = ["v1", "v2"]
    save()
    result = report.summarize(tmp_path)
    assert result["model_version_drift"]
    assert result["model_versions"] == ["v1", "v2"]
    assert result["contrasts"][0]["estimate"] is None
    assert result["contrasts"][0]["ci_reason"] == "model_version_drift_requires_stratified_design"


def test_six_cell_interaction_is_difference_of_differences(tmp_path, monkeypatch):
    experiment(tmp_path, monkeypatch, structure=True)
    contrasts = {c["name"]: c for c in report.summarize(tmp_path)["contrasts"]}
    assert contrasts["S-P@5"]["estimate"] == pytest.approx(.1)
    assert contrasts["S-P@10"]["estimate"] == pytest.approx(.5)
    assert contrasts["interaction:S-P"]["estimate"] == pytest.approx(.4)
    assert contrasts["interaction:S-O"]["estimate"] == pytest.approx(.5)


def test_unknown_slot_and_duplicate_attempt_fail(tmp_path, monkeypatch):
    _, rows, save = experiment(tmp_path, monkeypatch)
    rows.append(dict(rows[0]))
    save()
    with pytest.raises(report.L3Error, match="Duplicate attempt"):
        report.summarize(tmp_path)
    rows[-1]["trial_id"] = "invented"
    save()
    with pytest.raises(report.L3Error, match="unknown trial"):
        report.summarize(tmp_path)


def test_rejected_evidence_and_unobserved_slots_are_missing(tmp_path, monkeypatch):
    plan, rows, save = experiment(tmp_path, monkeypatch)
    for row in rows:
        row["accepted"] = False
    save()
    result = report.summarize(tmp_path)
    assert result["nvalid"] == 0
    assert result["nmissing"] == len(plan["jobs"])
    assert all(cell["mean"] is None for cell in result["cells"])
    assert result["contrasts"][0]["estimate"] is None
    assert result["contrasts"][0]["status"] == "incomplete"
    assert result["m2_baseline"]["complete_cohorts"] == []


def test_baseline_cohorts_require_full_observation_and_configuration(tmp_path, monkeypatch):
    plan, rows, save = experiment(tmp_path, monkeypatch)
    for job in plan["jobs"]:
        job["config"] = {"model": {"endpoint": "https://example.invalid/v1", "temperature": 1}, "budget": {"max_model_requests": 5}}
    save()
    result = report.summarize(tmp_path)
    cohorts = result["m2_baseline"]["complete_cohorts"]
    assert len(cohorts) == 4
    assert cohorts[0]["n"] == 2
    assert cohorts[0]["p0"] == 0
    assert cohorts[0]["endpoint"] == "https://example.invalid/v1"
    assert cohorts[0]["outcome_endpoint_version"] == "study-outcome-v1"
    assert result["m2_baseline"]["exported"] is False
    assert len({cohort["key"] for cohort in cohorts}) == 4
    first_key = cohorts[0]["key"]
    for job in plan["jobs"]:
        job["config"]["model"]["temperature"] = .5
    rows.pop(0)
    save()
    cohorts = report.summarize(tmp_path)["m2_baseline"]["complete_cohorts"]
    assert len(cohorts) == 3
    assert first_key not in {cohort["key"] for cohort in cohorts}


def test_baseline_cohort_does_not_mix_research_endpoint_versions(tmp_path, monkeypatch):
    plan, rows, save = experiment(tmp_path, monkeypatch)
    for job in plan["jobs"]:
        job["config"] = {"model": {"model": "fixed"}}
    rows[0]["endpoint_version"] = "changed-outcome-rule"
    save()
    cohorts = report.summarize(tmp_path)["m2_baseline"]["complete_cohorts"]
    assert len(cohorts) == 3
    assert all(c["outcome_endpoint_version"] == "study-outcome-v1" for c in cohorts)


def test_family_bootstrap_resamples_clusters_as_units():
    # Each possible resample is all A, one A + one B, or all B.
    # All-A mean is .5; all-B mean is .2, despite A's outcomes spanning 0..1.
    assert report._bootstrap({"A": [0, 1], "B": [.2, .2]}, 1000, 3) == [.2, .5]


@pytest.fixture
def trusted_report(tmp_path, monkeypatch):
    """Use real receipt/binding/evidence gates, with offline L1 boundaries only."""
    import hashlib
    from l3_sandbox import run
    from l3_sandbox.common import digest, read, write

    config = {"model": {"kind": "local"}, "budget": {"max_model_requests": 5}}
    task = {"task_id": "task", "family_id": "family", "family_verified": True, "split": "development"}
    job = {**task, "trial_id": "trial", "condition_id": "control", "replicate": 0,
           "config": config, "config_sha256": digest(config)}
    plan = {"experiment_id": "trusted", "purpose": "pilot", "design": "budget",
            "tasks": [task], "conditions": [{"condition_id": "control"}], "jobs": [job],
            "limits": {"max_replacements_per_slot": 2}, "analysis": {}}
    monkeypatch.setattr(report, "verify_plan", lambda *a, **k: plan)
    write(tmp_path / "plan.json", plan)
    episode = tmp_path / "runs/trial/attempt_000/episode"
    write(episode / "manifest.json", {"recording_mode": "live", "lineage": None,
          "task": {"task_id": "task"}, "config_sha256": digest(config)})
    write(episode / "raw/resolved_config.json", config)
    write(episode / "official_grade.json", {"y": 0})
    write(episode / "artifacts/inventory.json", {})
    for name in ("trajectory.jsonl", "state_changes.jsonl"):
        (episode / name).write_text("")
    sealed = {name: hashlib.sha256((episode / name).read_bytes()).hexdigest()
              for name in ("official_grade.json", "artifacts/inventory.json", "trajectory.jsonl", "state_changes.jsonl")}
    write(episode / "evidence_index.json", sealed)
    expected = hashlib.sha256((episode / "evidence_index.json").read_bytes()).hexdigest()
    def verifier(package, anchor):
        index = read(package / "evidence_index.json")
        return {"verified": all(hashlib.sha256((package / name).read_bytes()).hexdigest() == value
                                for name, value in index.items())}
    monkeypatch.setattr(run, "_verify_package", verifier)
    def derive(package, frozen_job, attempt):
        return {"study_valid": True, "study_outcome_y": read(package / "official_grade.json")["y"],
                "model_versions": ["offline-test-version"]}
    monkeypatch.setattr(run, "_summarize", derive)
    monkeypatch.setattr(run, "_native_run", lambda *a, **k: pytest.fail("Report must not execute a model"))
    row = {**{k: v for k, v in job.items() if k != "config"}, "attempt": 0,
           "binding": run._binding(plan, job), "accepted": True, "status": "completed",
           "expected_index_sha256": expected, **derive(episode, job, 0)}
    run._save_receipt(episode.parent / "receipt.json", row)
    write(tmp_path / "run_results.json", {"rows": [row]})
    return tmp_path, episode, row, job


def test_report_revalidates_actual_receipt_and_episode(trusted_report):
    directory, _, _, _ = trusted_report
    result = report.summarize(directory)
    assert result["nvalid"] == 1
    assert result["cells"][0]["mean"] == 0


@pytest.mark.parametrize("tamper", ["result_score", "receipt_digest", "result_and_receipt_score",
                                    "grade", "index", "recording_mode", "config", "omit_row"])
def test_report_rejects_mutable_projection_or_evidence_tamper(trusted_report, tamper):
    from l3_sandbox import run
    from l3_sandbox.common import read, write
    directory, episode, row, job = trusted_report
    if tamper in ("result_score", "result_and_receipt_score"):
        row["study_outcome_y"] = 1
        write(directory / "run_results.json", {"rows": [row]})
        if tamper == "result_and_receipt_score":
            run._save_receipt(episode.parent / "receipt.json", row)
    elif tamper == "receipt_digest":
        write(episode.parent / "receipt.json", {"row": row, "sha256": "0" * 64})
    elif tamper == "grade":
        write(episode / "official_grade.json", {"y": 1})
    elif tamper == "index":
        write(episode / "evidence_index.json", {})
    elif tamper == "recording_mode":
        manifest = read(episode / "manifest.json")
        write(episode / "manifest.json", {**manifest, "recording_mode": "test_fixture"})
    elif tamper == "config":
        write(episode / "raw/resolved_config.json", {"changed": True})
    elif tamper == "omit_row":
        write(directory / "run_results.json", {"rows": []})
    with pytest.raises(report.L3Error):
        report.summarize(directory)
    assert not (directory / "summary.json").exists()


def test_never_started_report_retains_missing_frozen_slots(tmp_path, monkeypatch):
    from l3_sandbox.common import write
    job = {"trial_id": "planned", "task_id": "t", "condition_id": "A", "replicate": 0}
    plan = {"experiment_id": "not-started", "purpose": "pilot", "design": "budget", "jobs": [job],
            "tasks": [{"task_id": "t"}], "conditions": [{"condition_id": "A"}], "analysis": {}}
    monkeypatch.setattr(report, "verify_plan", lambda *a, **k: plan)
    write(tmp_path / "plan.json", plan)
    result = report.summarize(tmp_path)
    assert result["nmissing"] == 1
    assert result["nvalid"] == result["nattempts"] == 0


def test_baseline_design_exports_m2_baseline_and_resolves_parent_excluded_p0(tmp_path, monkeypatch):
    plan, rows, save = experiment(tmp_path, monkeypatch, tasks=1, replicates=5)
    plan["design"] = "baseline"
    plan["conditions"] = [{"condition_id": "requests_035"}]
    plan["analysis"] = {"primary_comparison": ["requests_035"], "bootstrap_samples": 100, "seed": 7, "minimum_effect": 0.1}
    plan["jobs"] = [j for j in plan["jobs"] if j["condition_id"] == "control"]
    for idx, job in enumerate(plan["jobs"]):
        job["condition_id"] = "requests_035"
        job["trial_id"] = f"t0-requests_035-{idx}"
        job["config"] = {"model": {"endpoint": "https://example.invalid/v1", "temperature": 1}, "budget": {"max_model_requests": 35}}
    outcomes = [0.0, 0.2, 0.4, 0.6, 0.8]
    new_rows = []
    for idx, job in enumerate(plan["jobs"]):
        new_rows.append({
            **{k: v for k, v in job.items() if k != "config"},
            "attempt": 0,
            "accepted": True,
            "study_valid": True,
            "study_outcome_y": outcomes[idx],
            "endpoint_version": "study-outcome-v1",
            "model_versions": ["v1"],
            "delivery_exists": outcomes[idx] > 0,
            "legal_delivery": outcomes[idx] > 0,
            "dim1_pass": outcomes[idx] > 0,
            "known_total_tokens": 10,
            "unknown_usage_count": 0,
            "termination": "submitted",
            "model_requests": 5,
            "tool_calls": 4,
            "expected_index_sha256": f"{idx + 1:064x}",
        })
    rows[:] = new_rows
    import hashlib
    from l3_sandbox.common import digest, write
    cfg = {"model": {"endpoint": "https://example.invalid/v1", "temperature": 1}, "budget": {"max_model_requests": 35}}
    parent_ep = tmp_path / "parent_ep"
    write(parent_ep / "manifest.json", {"episode_id": "ep_t0_0", "task": {"task_id": "t0"}, "config_sha256": digest(cfg)})
    write(parent_ep / "raw/resolved_config.json", cfg)
    write(parent_ep / "evidence_index.json", {"files": []})
    parent_idx_sha = hashlib.sha256((parent_ep / "evidence_index.json").read_bytes()).hexdigest()
    rows[0]["expected_index_sha256"] = parent_idx_sha
    save()
    result = report.summarize(tmp_path)
    assert result["m2_baseline"]["exported"] is True
    assert (tmp_path / "m2_baseline.json").exists()
    cohorts = result["m2_baseline"]["complete_cohorts"]
    assert len(cohorts) == 1
    assert cohorts[0]["p0"] == pytest.approx(0.4)
    assert cohorts[0]["trials"][0]["p0_excluding_self"] == pytest.approx(0.5)
    resolved = report.resolve_baseline_p0(tmp_path, parent_ep)
    assert resolved["p0"] == pytest.approx(0.5)
    assert resolved["used_n"] == 4
    assert resolved["parent_in_cohort"] is True
    assert resolved["excluded_trial_id"] == "t0-requests_035-0"

