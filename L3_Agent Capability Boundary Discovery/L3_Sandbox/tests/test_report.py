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


@pytest.fixture
def verified_baseline(trusted_report, monkeypatch):
    import hashlib
    import shutil
    from l3_sandbox import run
    from l3_sandbox.common import digest, read, write
    from c3_sandbox.l1.evaluation.study import ENDPOINT_VERSION

    directory, episode, original_row, original_job = trusted_report
    plan = read(directory / "plan.json")
    plan["design"] = "baseline"
    plan["schema_version"] = __import__("l3_sandbox.plan", fromlist=["VERSION"]).VERSION
    jobs, rows = [], []
    def derive(package, job, attempt):
        return {"study_valid": True, "study_outcome_y": read(package / "official_grade.json")["y"],
                "model_versions": ["offline-test-version"], "endpoint_version": ENDPOINT_VERSION,
                "episode_id": read(package / "manifest.json")["episode_id"]}
    monkeypatch.setattr(run, "_summarize", derive)
    for i, y in enumerate([0, .4, .8]):
        job = {**original_job, "trial_id": "trial" if i == 0 else f"trial{i}", "replicate": i}
        package = directory / "runs" / job["trial_id"] / "attempt_000/episode"
        if i:
            shutil.copytree(episode, package)
        write(package / "manifest.json", {**read(package / "manifest.json"), "episode_id": f"ep{i}"})
        write(package / "official_grade.json", {"y": y})
        (package / "trajectory.jsonl").write_text(json.dumps({"kind": "model.parsed", "data": {"reported_model_version": "offline-test-version"}}) + "\n")
        index = {name: hashlib.sha256((package / name).read_bytes()).hexdigest()
                 for name in ("manifest.json", "raw/resolved_config.json", "official_grade.json", "trajectory.jsonl", "state_changes.jsonl", "artifacts/inventory.json")}
        write(package / "evidence_index.json", index)
        row = {**original_row, **{k: v for k, v in job.items() if k != "config"},
               "binding": run._binding(plan, job),
               "expected_index_sha256": hashlib.sha256((package / "evidence_index.json").read_bytes()).hexdigest(),
               **derive(package, job, 0)}
        run._save_receipt(package.parent / "receipt.json", row)
        jobs.append(job)
        rows.append(row)
    plan["jobs"] = jobs
    write(directory / "plan.json", plan)
    (directory / "plan.sha256").write_text(hashlib.sha256((directory / "plan.json").read_bytes()).hexdigest())
    write(directory / "run_results.json", {"rows": rows})
    from l3_sandbox.plan import verify_plan
    monkeypatch.setattr(report, "verify_plan", verify_plan)
    return directory, episode


def test_verified_baseline_resolves_parent_and_copy_without_writes(verified_baseline, tmp_path):
    import shutil
    directory, parent = verified_baseline
    report.summarize(directory)
    before = {str(p): p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    result = report.resolve_baseline_p0(directory, parent)
    assert result["p0"] == pytest.approx(.6)
    assert result["cohort_verified"] is True
    assert result["used_n"] == 2
    assert result["excluded_trial_id"] == "trial"
    copied = tmp_path / "copied-parent"
    shutil.copytree(parent, copied)
    assert report.resolve_baseline_p0(directory / "summary.json", copied) == result
    assert all(Path(p).read_bytes() == content for p, content in before.items())
    assert {str(p) for p in directory.rglob("*") if p.is_file()} == set(before) | {str(p) for p in copied.rglob("*") if p.is_file()}


@pytest.mark.parametrize("field,value", [("study_outcome_y", 1), ("expected_index_sha256", "f" * 64), ("episode_id", "invented")])
def test_cached_baseline_projection_cannot_change_p0(verified_baseline, field, value):
    from l3_sandbox.common import read, write
    directory, parent = verified_baseline
    report.summarize(directory)
    cached = read(directory / "summary.json")
    cached["m2_baseline"]["complete_cohorts"][0]["trials"][0][field] = value
    write(directory / "summary.json", cached)
    assert report.resolve_baseline_p0(directory / "summary.json", parent)["p0"] == pytest.approx(.6)
    assert read(directory / "summary.json") == cached  # Never repairs the source in place.


def test_linked_export_verified_and_unbound_rejected(verified_baseline, tmp_path):
    from l3_sandbox.common import read, write
    directory, parent = verified_baseline
    baseline = report.summarize(directory)["m2_baseline"]
    export = tmp_path / "portable.json"
    write(export, baseline)
    assert report.resolve_baseline_p0(export, parent)["p0"] == pytest.approx(.6)
    baseline.pop("source_binding")
    write(export, baseline)
    with pytest.raises(report.L3Error, match="Unverified baseline JSON"):
        report.resolve_baseline_p0(export, parent)


@pytest.mark.parametrize("tamper", ["value", "anchor", "endpoint", "plan_binding", "row"])
def test_linked_export_rejects_projection_tampering(verified_baseline, tmp_path, tamper):
    from l3_sandbox.common import write
    directory, parent = verified_baseline
    baseline = report.summarize(directory)["m2_baseline"]
    cohort = baseline["complete_cohorts"][0]
    if tamper == "value":
        cohort["trials"][1]["study_outcome_y"] = 1
    elif tamper == "anchor":
        cohort["trials"][0]["expected_index_sha256"] = "a" * 64
    elif tamper == "endpoint":
        cohort["outcome_endpoint_version"] = "stale"
    elif tamper == "row":
        baseline["rows"][0]["p0_excluding_self"] = .99
    else:
        baseline["source_binding"]["plan_sha256"] = "a" * 64
    export = tmp_path / "portable.json"
    write(export, baseline)
    with pytest.raises(report.L3Error, match="differs|mismatch"):
        report.resolve_baseline_p0(export, parent)


def test_current_endpoint_version_is_required(verified_baseline, monkeypatch):
    directory, parent = verified_baseline
    original = report._verified_rows
    def stale(*args):
        return [{**row, "endpoint_version": "stale"} for row in original(*args)]
    monkeypatch.setattr(report, "_verified_rows", stale)
    with pytest.raises(report.L3Error, match="outcome_endpoint_version"):
        report.resolve_baseline_p0(directory, parent)


@pytest.mark.parametrize("field", ["episode_id", "expected_index_sha256"])
def test_duplicate_baseline_anchors_are_not_exportable(verified_baseline, monkeypatch, field):
    directory, parent = verified_baseline
    original = report._verified_rows
    def duplicate(*args):
        rows = original(*args)
        rows[1][field] = rows[0][field]
        return rows
    monkeypatch.setattr(report, "_verified_rows", duplicate)
    with pytest.raises(report.L3Error, match="sealed episode anchors"):
        report.resolve_baseline_p0(directory, parent)


def test_l2_wrapper_matches_verified_l3_resolution(verified_baseline):
    l2_report = pytest.importorskip("l2_sandbox.report")
    directory, parent = verified_baseline
    assert l2_report.resolve_m2_baseline_p0(directory, parent) == report.resolve_baseline_p0(directory, parent)
