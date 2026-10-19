"""Bind experimental readiness to recorded measurement evidence, not approval intent."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .v3_execution import digest, file_digest, read_json, verify_seal


def human_review_policy_binding(
    audit: dict, audit_path: Path, *, required_scope: str = "G3_human_review",
) -> tuple[dict, list[str]]:
    """Validate an explicit user waiver for one scope and bind its source bytes.

    ``audit_path`` resolves relative authorization paths; callers must freeze and
    revalidate the returned binding before using the waiver.
    """
    try:
        return _human_review_policy_binding(audit, audit_path, required_scope)
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        return {"human_review_policy": audit.get("human_review_policy", "required")}, [
            f"human_review_waiver_authorization_invalid:{type(exc).__name__}"]


def _human_review_policy_binding(audit: dict, audit_path: Path, required_scope: str) -> tuple[dict, list[str]]:
    policy = audit.get("human_review_policy", "required")
    result = {"human_review_policy": policy, "human_review_status": "required"}
    if policy == "required":
        return result, []
    if policy != "waived_by_user":
        return result, ["human_review_policy_invalid"]
    result["human_review_status"] = "waived_by_user"
    evidence = audit.get("human_review_authorization")
    if not isinstance(evidence, dict) or not evidence.get("path") or not evidence.get("sha256"):
        return result, ["human_review_waiver_authorization_missing"]
    path = Path(evidence["path"])
    if not path.is_absolute():
        path = audit_path.parent / path
    path = path.resolve()
    result["human_review_authorization"] = {"path": str(path), "sha256": evidence["sha256"]}
    if not path.is_file() or file_digest(path) != evidence["sha256"]:
        return result, ["human_review_waiver_authorization_changed"]
    authorization = read_json(path)
    if (authorization.get("human_review_policy") != "waived_by_user"
            or authorization.get("authorization_source") != "direct_user_message"
            or authorization.get("exact_text") != "跳过人审核"
            or not isinstance(authorization.get("scope"), list)
            or required_scope not in authorization["scope"]):
        return result, ["human_review_waiver_authorization_invalid"]
    return result, []


def measurement_gate_binding(path: Path) -> tuple[dict, list[str]]:
    errors = []
    binding = {"path": str(path.resolve()), "sha256": None, "passed": False, "gates": {}}
    if not path.is_file():
        return binding, ["measurement_gate_audit_missing"]
    binding["sha256"] = file_digest(path)
    try:
        audit = read_json(path)
        if audit.get("schema") == "prospective_measurement_readiness_v2":
            return _prospective_measurement_binding_v2(audit, path, binding)
        if audit.get("schema") == "prospective_measurement_readiness_v1":
            return _prospective_measurement_binding(audit, path, binding)
        policy, policy_errors = human_review_policy_binding(audit, path)
        binding.update(policy)
        errors.extend(policy_errors)
        g1 = audit.get("V3_G1", {})
        g2 = audit.get("V3_G2_R19", {})
        g3 = audit.get("V3_G3_R20", {})
        g4 = audit.get("V3_G4_R21", {})
        g3_technical = (g3.get("sampled_checkpoint_count") == 20
                        and g3.get("missing_restored_directory_count") == 0
                        and g3.get("missing_scorer_exit_count") == 0)
        if policy["human_review_policy"] == "waived_by_user":
            g3_ready = (not policy_errors and g3_technical
                        and g3.get("restored_checkpoint_count") == 20
                        and g3.get("execution_metadata_verified") is True
                        and g3.get("source_provenance_verified") is True)
        else:
            g3_ready = (not policy_errors and g3_technical
                        and g3.get("human_review_completed_count") == 20
                        and g3.get("human_conclusions_verified_against_source") is True)
        passed = {
            "V3-G1": g1.get("byte_equivalence_verified") is True and g1.get("applicable_configuration_verified") is True,
            "V3-G2": g2.get("cp000000_all_pre_first_request_valid") is True
                and g2.get("completed_valid_pairs", 0) > 0
                and g2.get("ci95_completely_inside_pm_005") is True,
            "V3-G3": g3_ready,
            "V3-G4": g4.get("normal_and_p_requests_scanned", 0) > 0
                and g4.get("unverified_request_count") == 0
                and g4.get("leaked_request_count") == 0
                and g4.get("zero_diagnostic_leakage_verified_on_scanned") is True,
        }
        binding["gates"] = passed
        binding["passed"] = not errors and all(passed.values())
        errors.extend(f"measurement_gate_not_passed:{name}" for name, ok in passed.items() if not ok)
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        errors.append(f"measurement_gate_audit_invalid:{type(exc).__name__}")
    return binding, errors


def verify_measurement_gate_binding(plan: dict) -> None:
    """Called before production D1 admission and conditional D2 gate decisions.

    This validates the frozen readiness record. The audit producer remains
    responsible for validating the underlying packages and human review sources.
    """
    frozen = plan.get("measurement_gate_binding")
    if not isinstance(frozen, dict) or not frozen.get("path") or frozen.get("passed") is not True:
        raise ValueError("production execution requires verified measurement gates")
    current, errors = measurement_gate_binding(Path(frozen["path"]))
    if errors or current != frozen:
        raise ValueError("measurement gate evidence changed or remains incomplete")


_PROSPECTIVE_POLICY = "bounded-mechanism-contract-v1"
_DEV_TASKS = tuple(f"officeval_{task}" for task in ("003", "004", "010", "033", "050", "060", "070", "085", "090"))
_DEV_SLOTS = {f"{task}.r{repeat:02}" for task in _DEV_TASKS for repeat in range(5)}


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _slot_rows(rows: list[dict], expected: set[str]) -> dict[str, dict]:
    by_slot = {row["slot"]: row for row in rows}
    _require(len(rows) == len(by_slot) and set(by_slot) == expected, "slot_coverage")
    return by_slot


def _same_ref(report: dict, key: str, refs: dict, target: str) -> None:
    _require(report.get(key, {}).get("sha256") == refs[target]["sha256"], f"{key}_binding")


def _cohort_contract(plan: dict, cohort: dict, refs: dict) -> tuple[dict, int]:
    verify_seal(plan, "plan_sha256")
    verify_seal(cohort)
    _require(plan.get("schema_version") == "prospective-dev45-1", "collection_schema")
    _require(len(plan["slots"]) == 45 and set(plan["slots"]) == _DEV_SLOTS, "plan_slots")
    _require(plan["prospective_freeze_sha256"] == refs["prospective_freeze"]["sha256"], "freeze_binding")
    _require(cohort["plan_sha256"] == plan["plan_sha256"], "cohort_plan_binding")
    slots = cohort["slots"]
    _require(set(slots) == _DEV_SLOTS, "cohort_slots")
    for name, row in slots.items():
        task = name.split(".")[0]
        _require(row.get("study_valid") is True and row.get("status") == "completed"
                 and row.get("label") == name and row.get("task_id") == task
                 and row.get("config_sha256") == digest(plan["configs"][task])
                 and bool(row.get("index_file_sha256")) and bool(row.get("episode_id")), "cohort_slot_binding")
    requests = sum(row["usage"]["model_requests_including_parent"] for row in slots.values())
    _require(requests > 0, "empty_requests")
    return slots, requests


def _g1_contract(g1: dict, slots: dict, requests: int, refs: dict) -> None:
    for key, target in (("cohort", "cohort"), ("collection_plan", "collection_plan"), ("freeze", "prospective_freeze")):
        _same_ref(g1, key, refs, target)
    _require(g1.get("schema") == "prospective_recovery_census_v1"
             and g1.get("mechanical_census_complete") is True
             and g1.get("restore_exact") is True and g1.get("request_exact") is True
             and g1.get("actor_requests") == 0 and g1.get("request_count") == requests, "g1_census")
    rows = _slot_rows(g1["slots"], set(slots))
    checkpoints = compactions = 0
    for name, row in rows.items():
        source = slots[name]
        _require(row.get("package_index_sha256") == source["index_file_sha256"]
                 and row.get("episode_id") == source["episode_id"]
                 and row.get("config_sha256") == source["config_sha256"]
                 and row.get("request_count") == source["usage"]["model_requests_including_parent"], "g1_slot_binding")
        checks = row["checks"]
        _require(len(checks) == row["checkpoint_count"] and len({c["checkpoint_id"] for c in checks}) == len(checks), "g1_checkpoint_coverage")
        applicable = 0
        for check in checks:
            _require(check.get("restore_exact") is True, "g1_checkpoint_restore")
            if check.get("eligible") is True:
                _require(bool(check.get("request", {}).get("raw_sha256"))
                         and check["request"].get("size_bytes", 0) > 0, "g1_checkpoint_request")
                applicable += 1
            else:
                _require(check.get("eligible") is False
                         and check.get("ineligible_reasons") == ["request_budget_exhausted"]
                         and check.get("request") == {"status": "not_applicable_terminal"}, "g1_checkpoint_ineligible")
        _require(applicable == row["request_count"], "g1_request_coverage")
        ids = {c["checkpoint_id"] for c in checks}
        _require(all(c.get("request_status") == "applicable" and c["checkpoint_id"] in ids
                     and bool(c.get("event_sha256")) for c in row["compactions"]), "g1_compaction_binding")
        checkpoints += len(checks)
        compactions += len(row["compactions"])
    _require(checkpoints == g1["checkpoint_count"] and compactions == g1["compaction_count"]
             and compactions > 0 and g1.get("compression_coverage") == "observed", "g1_real_compression_required")


def _g2_contract(g2: dict, plan: dict, refs: dict, slots: dict, sidecar: dict | None = None) -> None:
    _require(plan.get("schema") == "prospective_tool_replay_v1", "g2_plan_schema")
    for key, target in (("cohort", "cohort"), ("collection_plan", "collection_plan"), ("freeze", "prospective_freeze")):
        _same_ref(plan, key, refs, target)
    _require(g2.get("schema") == "prospective_tool_replay_result_v1"
             and g2.get("status") == "complete" and g2.get("mechanism_gate_pass") is True
             and g2.get("actor_requests") == 0 and g2.get("analyst_requests") == 0
             and not g2.get("not_run") and not g2.get("error")
             and g2.get("plan_sha256") == refs["g2_plan"]["sha256"], "g2_incomplete")
    expected = {b["batch_id"]: b for b in plan["batches"]}
    actual = {b["batch_id"]: b for b in g2["batches"]}
    _require(expected and len(expected) == len(plan["batches"])
             and len(actual) == len(g2["batches"]) and set(actual) == set(expected), "g2_batch_coverage")
    indices = plan["package_indices"]
    _require(len(indices) == len(slots)
             and {Path(package).name: sha for package, sha in indices.items()}
             == {name: row["index_file_sha256"] for name, row in slots.items()}, "g2_package_indices")
    _require(plan.get("limits") == {"actor_http_requests": 0, "replays_per_batch": 2,
                                    "tool_calls": 6300, "wall_seconds": 21600, "workers": 1},
             "g2_frozen_limits")
    replacements = _g2_reclassifications(g2, plan, sidecar)
    attempts = 0
    applied = set()
    for key, batch in actual.items():
        selected = expected[key]
        _require(selected["slot"] in slots
                 and indices.get(selected["package"]) == slots[selected["slot"]]["index_file_sha256"]
                 and Path(selected["package"]).name == selected["slot"], "g2_cohort_slot")
        replays = batch["replays"]
        _require(bool(replays), "g2_replay_coverage")
        first_exact = all(call.get("exact") is True for call in replays[0]["calls"])
        _require(len(replays) == (1 if first_exact else 2), "g2_replay_coverage")
        for replay_number, replay in enumerate(replays, 1):
            _require(replay.get("status") == "complete" and replay.get("restore_exact") is True
                     and replay.get("request_exact") == selected["request_exact"]
                     and replay.get("cleanup", {}).get("confirmed") is True, "g2_restore_or_cleanup")
            calls = replay["calls"]
            attempts += len(calls)
            _require(len(calls) == len(selected["calls"])
                     and [c["call_id"] for c in calls] == [c["call"]["call_id"] for c in selected["calls"]], "g2_call_coverage")
            for ordinal, original in enumerate(calls):
                replacement_key = (key, replay_number, ordinal)
                call = replacements.get(replacement_key, original)
                if replacement_key in replacements:
                    applied.add(replacement_key)
                _require(call.get("unexplained_state_difference") is False
                         and call.get("unexplained_tool_difference") is False, "g2_unexplained_difference")
                if call.get("exact") is not True:
                    _require(bool(call.get("zip_timestamp_variance")) or call.get("ls_metadata_variance") is True,
                             "g2_undeclared_difference")
    _require(applied == set(replacements), "g2_unused_reclassification")
    runtime = g2.get("runtime_executions")
    feedback = g2.get("native_feedback_only")
    _require(type(runtime) is int and type(feedback) is int and runtime >= 0 and feedback >= 0
             and attempts == g2.get("tool_attempts") == runtime + feedback
             and attempts <= plan["limits"]["tool_calls"], "g2_attempt_accounting")


def _g4_contract(g4: dict, slots: dict, requests: int, refs: dict) -> None:
    _require(g4.get("schema") == "prospective-g4-post-audit-v1"
             and g4.get("g4_verified") is True and g4.get("actor_requests") == 0
             and g4.get("slot_count") == 45 and g4.get("request_count") == requests
             and g4.get("cohort_sha256") == refs["cohort"]["sha256"]
             and g4.get("freeze_sha256") == refs["prospective_freeze"]["sha256"], "g4_binding")
    rows = _slot_rows(g4["slots"], set(slots))
    for name, row in rows.items():
        source = slots[name]
        _require(row.get("status") == "verified" and row.get("index_sha256") == source["index_file_sha256"]
                 and row.get("episode_id") == source["episode_id"]
                 and row.get("request_count") == source["usage"]["model_requests_including_parent"], "g4_slot_binding")
    _require(sum(row["container_count"] for row in rows.values()) == g4["container_count"], "g4_container_coverage")


def _prospective_measurement_binding(audit: dict, path: Path, binding: dict) -> tuple[dict, list[str]]:
    """Fail closed on the explicitly versioned, limited recovery contract."""
    errors = []
    binding.update(schema="prospective_measurement_readiness_v1", policy=audit.get("policy"),
                   statistical_equivalence="not_established", outcome_distribution_equivalence="not_established",
                   execution_readiness=False, evidence_coverage={})
    try:
        _require(audit.get("policy") == _PROSPECTIVE_POLICY, "unsupported_policy")
        refs = audit["refs"]
        documents = {}
        bound = {}
        for key in ("collection_plan", "prospective_freeze", "cohort", "g1", "g2", "g2_plan", "g4",
                    "e1_manifest", "e1_verification", "authorization"):
            ref = refs[key]
            target = Path(ref["path"])
            if not target.is_absolute():
                target = path.parent / target
            target = target.resolve()
            _require(target.is_file() and file_digest(target) == ref["sha256"], f"reference_changed:{key}")
            documents[key] = read_json(target)
            bound[key] = {"path": str(target), "sha256": ref["sha256"]}
        if documents["g2"].get("reclassifications"):
            ref = refs["g2_reclassification"]
            target = Path(ref["path"])
            if not target.is_absolute():
                target = path.parent / target
            target = target.resolve()
            _require(target.is_file() and file_digest(target) == ref["sha256"],
                     "reference_changed:g2_reclassification")
            documents["g2_reclassification"] = read_json(target)
            bound["g2_reclassification"] = {"path": str(target), "sha256": ref["sha256"]}
        binding["refs"] = bound
        authorization = documents["authorization"]
        _require(authorization.get("authorization_source") == "direct_user_reply"
                 and authorization.get("exact_text") == "采用限定合同，披露变异并继续（推荐）"
                 and authorization.get("policy") == _PROSPECTIVE_POLICY
                 and authorization.get("allow_declared_runtime_variance") is True
                 and authorization.get("allow_unexplained_differences") is False
                 and "outcome-distribution equivalence" in authorization.get("scope", ""), "policy_authorization")
        policy, policy_errors = human_review_policy_binding(audit, path)
        binding.update(policy)
        errors.extend(policy_errors)
        _require(policy.get("human_review_status") == "waived_by_user" and not policy_errors, "human_review_policy")
        slots, requests = _cohort_contract(documents["collection_plan"], documents["cohort"], refs)
        validators = {
            "V3-G1": lambda: _g1_contract(documents["g1"], slots, requests, refs),
            "V3-G2": lambda: _g2_contract(documents["g2"], documents["g2_plan"], refs, slots,
                                             documents.get("g2_reclassification")),
            "V3-G3": lambda: _e1_contract(documents["e1_manifest"], documents["e1_verification"], slots, refs,
                                             documents["prospective_freeze"]),
            "V3-G4": lambda: _g4_contract(documents["g4"], slots, requests, refs),
        }
        for name, validate in validators.items():
            try:
                validate()
                binding["gates"][name] = True
            except (KeyError, IndexError, OSError, ValueError, TypeError, AttributeError) as exc:
                binding["gates"][name] = False
                errors.append(f"measurement_gate_not_passed:{name}:{exc}")
        binding["evidence_coverage"] = {"slots": len(slots), "requests": requests,
                                        "scope": _PROSPECTIVE_POLICY}
        binding["passed"] = not errors and all(binding["gates"].values())
        binding["execution_readiness"] = binding["passed"]
    except (KeyError, IndexError, OSError, ValueError, TypeError, AttributeError) as exc:
        errors.append(f"prospective_measurement_gate_invalid:{exc}")
    return binding, errors


def _e1_contract(manifest: dict, verification: dict, slots: dict, refs: dict, freeze: dict) -> None:
    _require(manifest.get("full_45_ready") is True and manifest.get("completed_count") == 45
             and manifest.get("actor_requests") == 0, "e1_manifest_incomplete")
    _same_ref(manifest, "cohort", refs, "cohort")
    _require(verification.get("schema") == "prospective_e1_independent_verification_v1"
             and verification.get("status") == "passed"
             and verification.get("cohort_sha256") == refs["cohort"]["sha256"]
             and verification.get("manifest_sha256") == refs["e1_manifest"]["sha256"]
             and verification.get("analysis_sha256") == manifest["analysis"]["sha256"]
             and verification.get("coverage_slots") == 45 and verification.get("full_45_ready") is True
             and verification.get("all_terminal_submission_scores_equal_Y") is True
             and verification.get("all_output_and_raw_hashes_valid") is True
             and verification.get("all_checkpoint_flags_revalidated") is True, "e1_independent_verification")
    _require(manifest["analysis"].get("files")
             and digest(manifest["analysis"]["files"]) == manifest["analysis"]["sha256"], "e1_analysis_code_binding")
    original = _slot_rows(manifest["slots"], set(slots))
    verified = _slot_rows(verification["slots"], set(slots))
    _slot_rows(manifest["completed"], set(slots))
    for name, row in verified.items():
        source = slots[name]
        binding = row["binding"]
        rubric = [sha for name, sha in freeze["bound_files"].items()
                  if name.endswith(f"/rubrics/{source['task_id']}.json")]
        _require(len(rubric) == 1 and row.get("rubric_sha256") == rubric[0], "e1_rubric_binding")
        _require(binding == original[name]["binding"]
                 and binding.get("cohort_sha256") == refs["cohort"]["sha256"]
                 and binding.get("parent_index_sha256") == source["index_file_sha256"]
                 and binding.get("manifest_sha256") == source["manifest_sha256"]
                 and binding.get("episode_id") == source["episode_id"]
                 and binding.get("config_sha256") == source["config_sha256"]
                 and binding.get("analysis_sha256") == verification["analysis_sha256"]
                 and binding.get("source_sha256") == original[name]["source"]["sha256"]
                 and binding.get("study_rule_version") == source["outcome"]["rule_version"], "e1_slot_binding")
        _require(row.get("Y") == source["outcome"]["study_outcome_y"]
                 and row.get("terminal_submission_score") == row["Y"]
                 and bool(row.get("receipt_sha256")) and bool(row.get("grading_receipt_sha256"))
                 and set(row.get("outputs", {})) == {"grades.jsonl", "facts.jsonl", "defects.json"}
                 and all(row["outputs"].values()) and row.get("actual_checkpoint_rows", 0) > 0,
                 "e1_terminal_or_outputs")


def _g2_reclassifications(g2: dict, plan: dict, sidecar: dict | None) -> dict:
    """Accept only the disclosed literal-input-ls diagnosis; retain original rows."""
    changes = g2.get("reclassifications", [])
    if not changes:
        return {}
    _require(isinstance(sidecar, dict) and sidecar.get("original_receipts_unchanged") is True
             and sidecar.get("changes") == changes, "g2_reclassification_sidecar")
    batches = {batch["batch_id"]: batch for batch in g2["batches"]}
    selected = {batch["batch_id"]: batch for batch in plan["batches"]}
    replacements = {}
    changed_fields = {"classification", "ls_metadata_variance", "unexplained_tool_difference"}
    for change in changes:
        batch_id = change["batch_id"]
        ref = change["original_comparison"]
        original_path = Path(ref["path"])
        directory = original_path.parent
        ordinal = int(directory.name.removeprefix("call_"))
        _require(directory.name == f"call_{ordinal:04d}" and original_path.name == "comparison.json",
                 "g2_reclassification_path")
        replay_number = next((n for n in (1, 2)
                              if directory.parent.name == f"{batch_id}_replay{n}"), None)
        _require(replay_number is not None and batch_id in batches and batch_id in selected,
                 "g2_reclassification_replay")
        _require(0 <= ordinal < len(selected[batch_id]["calls"])
                 and replay_number <= len(batches[batch_id]["replays"])
                 and ordinal < len(batches[batch_id]["replays"][replay_number - 1]["calls"]),
                 "g2_reclassification_position")
        original = batches[batch_id]["replays"][replay_number - 1]["calls"][ordinal]
        recorded = selected[batch_id]["calls"][ordinal]["call"]
        before, after = change["before"], change["after"]
        _require(recorded.get("name") == "bash"
                 and recorded.get("arguments") == {"command": "ls -la /workspace/input/"}
                 and change["call_id"] == original["call_id"] == recorded["call_id"]
                 and before == original, "g2_reclassification_original")
        _require(before.get("exact") is False and after.get("exact") is False
                 and before.get("classification") == "unexplained"
                 and before.get("ls_metadata_variance") is False
                 and before.get("unexplained_tool_difference") is True
                 and after.get("classification") == "declared_runtime_variance"
                 and after.get("ls_metadata_variance") is True
                 and after.get("unexplained_tool_difference") is False
                 and after.get("after_state_exact") is True
                 and after.get("unexplained_state_difference") is False
                 and after.get("zip_timestamp_variance") == []
                 and after.get("state_diff") == []
                 and after.get("expected_after_sha256") == after.get("actual_after_sha256")
                 and bool(after.get("expected_after_sha256"))
                 and {k: v for k, v in before.items() if k not in changed_fields}
                 == {k: v for k, v in after.items() if k not in changed_fields}, "g2_reclassification_delta")
        canonical_before = json.dumps(before, ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":"), allow_nan=False).encode() + b"\n"
        _require(ref["sha256"] == hashlib.sha256(canonical_before).hexdigest(),
                 "g2_reclassification_original_hash")
        raw = change["raw_files"]
        _require({"comparison.json", "result.json", "stdout.bin", "stderr.bin",
                  "native_validation.json", "delivered_observation.json"} <= set(raw)
                 and raw["comparison.json"] == ref
                 and change.get("reason") == "Already-authorized ls UID/GID/timestamp diagnosis; input directory added to literal command allowlist",
                 "g2_reclassification_raw_provenance")
        for name, evidence in raw.items():
            sha = evidence["sha256"]
            _require(Path(evidence["path"]) == directory / name and Path(name).name == name
                     and isinstance(sha, str) and len(sha) == 64
                     and all(c in "0123456789abcdef" for c in sha), "g2_reclassification_raw_reference")
        key = (batch_id, replay_number, ordinal)
        _require(key not in replacements, "g2_duplicate_reclassification")
        replacements[key] = after
    return replacements


_PROSPECTIVE_POLICY_V2 = "bounded-mechanism-contract-v2"


def _v2_references(audit: dict, path: Path) -> tuple[dict, dict]:
    documents, bound = {}, {}
    names = ("collection_plan", "prospective_freeze", "cohort", "g1", "g2", "g2_plan", "g4",
             "e1_manifest", "e1_verification", "authorization", "v2_contract", "v2_authorization",
             "g2_v2", "g2_forensics")
    for key in names:
        ref = audit["refs"][key]
        target = Path(ref["path"])
        if not target.is_absolute():
            target = path.parent / target
        target = target.resolve()
        _require(target.is_file() and file_digest(target) == ref["sha256"], f"reference_changed:{key}")
        documents[key] = read_json(target)
        bound[key] = {"path": str(target), "sha256": ref["sha256"]}
    return documents, bound


def _v2_contract(policy: dict) -> None:
    _require(policy.get("schema") == "bounded-mechanism-contract-policy-v2"
             and policy.get("policy") == _PROSPECTIVE_POLICY_V2
             and policy.get("derived_from_policy") == _PROSPECTIVE_POLICY
             and policy.get("post_hoc_forensic_extension") is True
             and policy.get("preserve_v1_original_decision") is True
             and policy.get("v1_decision") == "blocked_unexplained_difference"
             and policy.get("reuse_existing_evidence_only") is True
             and policy.get("requires_independent_raw_reaudit") is True
             and policy.get("inherit_v1_docx_zip_timestamp_exception_only") is True
             and policy.get("raw_tool_output_byte_equivalence") is False
             and policy.get("statistical_equivalence") == "not_established"
             and policy.get("outcome_distribution_equivalence") == "not_established"
             and policy.get("current_execution_scope") == "D0_freeze_and_D1_plan_only",
             "v2_contract_scope")
    for name in ("additional_tool_replays", "actor_requests", "analyst_requests"):
        _require(type(policy.get(name)) is int and policy[name] == 0, "v2_contract_execution_scope")
    _require(policy.get("exemptions") == {
        "ls_runtime_metadata_v2": {
            "allowed": ["uid", "gid", "mtime", "directory_total_allocated_blocks"],
            "exact": ["entry_count", "mode", "links", "logical_size", "name"],
            "requires_after_state_exact": True, "requires_other_observation_fields_exact": True,
            "forms": ["directory_ls_la", "single_file_ls_la", "cd_then_ls_la", "cp_then_ls_la"],
        },
        "python_object_repr_address_v1": {
            "allowed": ["hexadecimal_address_in_default_python_object_repr"],
            "requires_after_state_exact": True,
            "requires_full_stdout_stderr_and_other_observation_fields_exact_except_named_addresses": True,
        },
    }, "v2_contract_exemptions")


def _v2_authorization(authorization: dict, authorization_path: Path) -> dict:
    _require(authorization.get("authorization_source") == "direct_user_request_attachment",
             "v2_authorization_source")
    scope = authorization.get("adoption_scope", {})
    _require(scope.get("policy") == _PROSPECTIVE_POLICY_V2
             and scope.get("post_hoc_forensic_extension") is True
             and scope.get("execution") == "D0_freeze_and_D1_plan_only"
             and type(scope.get("actor_requests")) is int and scope["actor_requests"] == 0
             and type(scope.get("analyst_requests")) is int and scope["analyst_requests"] == 0,
             "v2_authorization_scope")
    ref = authorization["request_attachment"]
    source = Path(ref["path"])
    if not source.is_absolute():
        source = authorization_path.parent / source
    source = source.resolve()
    _require(source.is_file() and source.stat().st_size > 0
             and file_digest(source) == ref["sha256"], "v2_authorization_attachment_changed")
    return {"path": str(source), "sha256": ref["sha256"]}


def _g2_v2_structure(g2: dict, plan: dict, refs: dict, slots: dict) -> None:
    """Check original coverage/accounting without rewriting its blocked verdict."""
    _require(plan.get("schema") == "prospective_tool_replay_v1", "g2_plan_schema")
    for key, target in (("cohort", "cohort"), ("collection_plan", "collection_plan"), ("freeze", "prospective_freeze")):
        _same_ref(plan, key, refs, target)
    _require(g2.get("schema") == "prospective_tool_replay_result_v1"
             and g2.get("status") == "blocked_unexplained_difference"
             and g2.get("mechanism_gate_pass") is False
             and g2.get("actor_requests") == 0 and g2.get("analyst_requests") == 0
             and not g2.get("not_run") and not g2.get("error")
             and g2.get("plan_sha256") == refs["g2_plan"]["sha256"], "g2_original_binding")
    expected = {b["batch_id"]: b for b in plan["batches"]}
    actual = {b["batch_id"]: b for b in g2["batches"]}
    _require(len(expected) == len(plan["batches"]) == 67
             and len(actual) == len(g2["batches"]) and set(actual) == set(expected), "g2_batch_coverage")
    indices = plan["package_indices"]
    _require(len(indices) == len(slots)
             and {Path(package).name: sha for package, sha in indices.items()}
             == {name: row["index_file_sha256"] for name, row in slots.items()}, "g2_package_indices")
    _require(plan.get("limits") == {"actor_http_requests": 0, "replays_per_batch": 2,
                                    "tool_calls": 6300, "wall_seconds": 21600, "workers": 1}, "g2_frozen_limits")
    attempts = 0
    for key, batch in actual.items():
        selected = expected[key]
        _require(selected["slot"] in slots
                 and indices.get(selected["package"]) == slots[selected["slot"]]["index_file_sha256"]
                 and Path(selected["package"]).name == selected["slot"], "g2_cohort_slot")
        replays = batch["replays"]
        _require(bool(replays), "g2_replay_coverage")
        first_exact = all(call.get("exact") is True for call in replays[0]["calls"])
        _require(len(replays) == (1 if first_exact else 2), "g2_replay_coverage")
        for replay in replays:
            _require(replay.get("status") == "complete" and replay.get("restore_exact") is True
                     and replay.get("request_exact") == selected["request_exact"]
                     and replay.get("cleanup", {}).get("confirmed") is True, "g2_restore_or_cleanup")
            calls = replay["calls"]
            attempts += len(calls)
            _require(len(calls) == len(selected["calls"])
                     and [c["call_id"] for c in calls] == [c["call"]["call_id"] for c in selected["calls"]],
                     "g2_call_coverage")
    runtime, feedback = g2.get("runtime_executions"), g2.get("native_feedback_only")
    _require(type(runtime) is int and type(feedback) is int and runtime >= 0 and feedback >= 0
             and attempts == g2.get("tool_attempts") == runtime + feedback == 102, "g2_attempt_accounting")


def _prospective_measurement_binding_v2(audit: dict, path: Path, binding: dict) -> tuple[dict, list[str]]:
    """Adopt a source-bound post-hoc contract; never upgrade the original v1 record."""
    errors = []
    binding.update(schema="prospective_measurement_readiness_v2", policy=audit.get("policy"),
                   post_hoc_forensic_extension=True, statistical_equivalence="not_established",
                   outcome_distribution_equivalence="not_established", raw_tool_byte_equivalence=False,
                   measurement_readiness=False, execution_authorized=False, execution_readiness=False,
                   current_execution_scope="D0_freeze_and_D1_plan_only", evidence_coverage={})
    try:
        _require(audit.get("policy") == _PROSPECTIVE_POLICY_V2
                 and audit.get("post_hoc_forensic_extension") is True, "unsupported_v2_policy")
        for name in ("statistical_equivalence", "outcome_distribution_equivalence"):
            _require(audit.get(name, "not_established") == "not_established", "v2_equivalence_overclaim")
        documents, refs = _v2_references(audit, path)
        binding["refs"] = refs
        _v2_contract(documents["v2_contract"])
        binding["request_attachment"] = _v2_authorization(
            documents["v2_authorization"], Path(refs["v2_authorization"]["path"]))
        original_authorization = documents["authorization"]
        _require(original_authorization.get("authorization_source") == "direct_user_reply"
                 and original_authorization.get("exact_text") == "采用限定合同，披露变异并继续（推荐）"
                 and original_authorization.get("policy") == _PROSPECTIVE_POLICY
                 and original_authorization.get("allow_declared_runtime_variance") is True
                 and original_authorization.get("allow_unexplained_differences") is False
                 and "outcome-distribution equivalence" in original_authorization.get("scope", ""),
                 "original_policy_authorization")
        review, review_errors = human_review_policy_binding(audit, path)
        binding.update(review)
        errors.extend(review_errors)
        _require(review.get("human_review_status") == "waived_by_user" and not review_errors,
                 "human_review_policy")
        slots, requests = _cohort_contract(documents["collection_plan"], documents["cohort"], refs)

        def validate_g2():
            _g2_v2_structure(documents["g2"], documents["g2_plan"], refs, slots)
            # This verifier rereads and recomputes every classification from raw files;
            # report-level pass booleans are never an admission authority.
            import importlib.util
            verifier_path = Path(__file__).resolve().parents[2] / "scripts/reclassify_prospective_v2.py"
            spec = importlib.util.spec_from_file_location("_measurement_metadata_v2", verifier_path)
            _require(spec is not None and spec.loader is not None, "g2_v2_verifier_unavailable")
            verifier = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(verifier)
            proof = verifier.verify_reaudit(Path(refs["g2_v2"]["path"]), expected_refs=refs)
            binding["g2_v2_verifier"] = {"path": str(verifier_path), "sha256": file_digest(verifier_path)}
            _require(proof.get("tool_attempts") == 102 and proof.get("unexplained") == 0,
                     "g2_v2_recomputed_coverage")
            binding["g2_v2_recomputed"] = proof

        validators = {
            "V3-G1": lambda: _g1_contract(documents["g1"], slots, requests, refs),
            "V3-G2": validate_g2,
            "V3-G3": lambda: _e1_contract(documents["e1_manifest"], documents["e1_verification"], slots,
                                          refs, documents["prospective_freeze"]),
            "V3-G4": lambda: _g4_contract(documents["g4"], slots, requests, refs),
        }
        for name, validate in validators.items():
            try:
                validate()
                binding["gates"][name] = True
            except (KeyError, IndexError, ImportError, OSError, ValueError, TypeError, AttributeError) as exc:
                binding["gates"][name] = False
                errors.append(f"measurement_gate_not_passed:{name}:{exc}")
        binding["evidence_coverage"] = {"slots": len(slots), "requests": requests,
                                        "scope": _PROSPECTIVE_POLICY_V2}
        binding["passed"] = not errors and all(binding["gates"].values())
        binding["measurement_readiness"] = binding["passed"]
    except (KeyError, IndexError, OSError, ValueError, TypeError, AttributeError) as exc:
        errors.append(f"prospective_measurement_gate_v2_invalid:{exc}")
    return binding, errors
