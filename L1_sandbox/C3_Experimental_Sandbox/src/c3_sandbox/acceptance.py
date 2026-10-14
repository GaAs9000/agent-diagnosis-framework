"""Deterministic integration evidence for D1–D5; not an efficacy experiment."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import time

from .evidence._storage import encode
from .experiment_controller import QualifiedBranchExecutor, RegistryAuthorizer
from .experiment_scheduler import BranchSpec, ExecutionOutcome, ExperimentScheduler, ProtocolError
from .foundation_gate import source_identities
from .manifests import canonical_hash, validate_family_manifest, validate_system_manifest
from .probe_registry import ProbeRegistry
from .replay import ReplayQualification, STATE_CONTRACT_V1
from .trajectory import validate_trajectory

ROOT = Path(__file__).resolve().parents[2]


def synthetic_review(registry, contract, probe_id):
    """Fixture-only evidence; cannot authorize a real case."""
    record = {
        "task_identity": contract["task_identity"], "probe_identity": registry.identity(probe_id),
        "line": contract["line"],
        "source": {"kind": "synthetic_fixture", "signer_id": "deterministic-test",
                   "record_identity": canonical_hash({"fixture": contract["task_identity"]})},
        "findings": {
            "leakage": {"verdict": "no_task_answer_material", "evidence_identities": [canonical_hash({"synthetic_leakage_test": probe_id})]},
            "solvability": {"verdict": "necessary_resources_preserved", "evidence_identities": [canonical_hash({"synthetic_resource_test": probe_id})]},
            "reviewed_inputs": contract["available_inputs"], "reviewed_tools": contract["available_tools"],
        },
    }
    record["signature"] = {"scheme": "sha256-fixture", "value": canonical_hash(record)}
    return record


def prepare_fixture(root: Path):
    """Assemble actual D1–D4 objects for the broker rather than a boolean qualifier."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    registry = ProbeRegistry.from_file(ROOT / "configs/probes.v1.json")
    runtime = {
        "image_digest": "sha256:" + canonical_hash({"synthetic_runtime": "no-container-no-model-v1"}),
        "config": {"backend": "deterministic_in_process_fixture", "concurrency": 1},
        "model_config": {"provider": "none", "model": "no-model"},
        "tool_config": {"tools": ["fixture_reader"]}, "environment": {"LANG": "C.UTF-8"},
    }
    system_config = {"model": runtime["model_config"], "agent": {"id": "fixture-handler", "revision": "v1"},
                     "tools": runtime["tool_config"], "prompt": "Synthetic acceptance only",
                     "sampling": {"temperature": 0.1}, "runtime": runtime}
    system_hash = canonical_hash(system_config)
    systems = validate_system_manifest({"schema_version": "1.0", "systems": [
        {"system_id": "synthetic-system", "config": system_config, "config_sha256": system_hash}
    ]})
    (root / "system_manifest.json").write_bytes(encode(systems))
    replay = ReplayQualification(root / "checkpoint-store")
    reviews, cases, families = {}, {}, []
    for line, target, competitor in (("E2-S", "visibility.compact", "feedback.aggregate"),
                                      ("E2-F", "visibility.expanded", "feedback.structured")):
        case_id = "synthetic-" + line.lower()
        workspace = root / (case_id + "-original")
        workspace.mkdir()
        state = {"case_id": case_id, "original_completion": line == "E2-S", "value": 7}
        (workspace / "state.json").write_bytes(encode(state))
        (workspace / ".retained").write_text("hidden state retained\n")
        messages = [{"role": "system", "content": "Synthetic controller acceptance; no model inference."},
                    {"role": "user", "content": "Preserve and inspect the fixture state."}]
        checkpoint = replay.register_checkpoint(workspace, messages, system_hash, runtime, STATE_CONTRACT_V1, True)
        qualified = replay.qualify(checkpoint)
        if not qualified["eligible_for_intervention"]:
            raise AssertionError(qualified)
        task_hash = canonical_hash(state)
        panel = registry.experiment_panel(line, target, competitor)
        schedules = {phase: {"0": {"temperature": 0.1, "seed": seed}}
                     for phase, seed in (("exploration", 2026), ("confirmation", 2027))}
        contract = {
            "task_identity": task_hash, "is_synthetic": True, "line": line,
            "available_inputs": ["state.json"], "necessary_inputs": ["state.json"],
            "available_tools": ["fixture_reader"], "necessary_tools": ["fixture_reader"],
            "observation_capabilities": ["redundant_observation_views", "public_feedback_fields"],
            "sampling": schedules["exploration"]["0"], "sampling_by_phase": schedules,
            "budgets": {"tokens": 128, "tool_calls": 1}, "baseline_probe_id": "original",
            "control_probe_id": registry.get(target)["control_id"],
            "target_probe_id": target, "competing_probe_id": competitor,
        }
        specs = []
        for probe in panel:
            review = synthetic_review(registry, contract, probe["probe_id"])
            reviews[(case_id, probe["probe_id"])] = review
            if not registry.validate_for_case(probe["probe_id"], contract, review)["eligible"]:
                raise AssertionError("fixture review was not accepted")
            for phase in ("exploration", "confirmation"):
                specs.append(BranchSpec(case_id, task_hash, line, system_hash, checkpoint, checkpoint,
                                        probe["probe_id"], registry.identity(probe["probe_id"]), schedules[phase]["0"],
                                        0, qualified["runtime_identity"], canonical_hash({"fixture_grader": "v1"}), True, phase))
        flags = {"execution_valid": True, "target_resolved": line == "E2-S",
                 "regression_free": True, "task_completed": line == "E2-S"}
        trajectory = validate_trajectory({
            "schema_version": "1.0", "episode_id": case_id, "task_id": case_id,
            "task_family_id": "family-" + case_id, "system_id": "synthetic-system", "split": "development",
            "status": "success" if line == "E2-S" else "failure", "outcome": flags,
            "events": [{"event_id": "fixture-origin", "type": "synthetic_fixture_creation"}],
            "artifact_versions": [{"path": "state.json", "sha256": canonical_hash(state)}],
            "checkpoint_refs": [{"checkpoint_id": checkpoint, "checkpoint_identity": checkpoint,
                                 "qualification_report_ref": canonical_hash(qualified), "replay_qualified": True}],
            "provenance": {"source": "deterministic_acceptance_fixture", "version": "C3-E2-v1.0", "is_synthetic": True},
        })
        (root / (case_id + ".trajectory.json")).write_bytes(encode(trajectory))
        (root / (case_id + ".qualification.json")).write_bytes(encode(qualified))
        cases[case_id] = {"contract": contract, "branches": specs, "checkpoint": checkpoint,
                          "messages": messages, "workspace": workspace, "system_hash": system_hash, "runtime": runtime}
        families.append({"task_family_id": "family-" + case_id, "split": "development", "task_ids": [case_id],
                         "source_group_ids": ["source-" + case_id], "asset_hashes": {case_id + "/state.json": task_hash}})
    (root / "family_manifest.json").write_bytes(encode(validate_family_manifest({"schema_version": "1.0", "families": families})))
    (root / "probe_reviews.synthetic.json").write_bytes(encode({case + ":" + probe: review for (case, probe), review in reviews.items()}))
    return registry, replay, reviews, cases


def run_acceptance(output_dir: Path) -> dict:
    """Persist evidence of the six user-mandated invariants, including denials."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("acceptance output directory must be empty/new")
    begun = time.perf_counter()
    result = {"protocol": "C3-E2-v1.0", "is_synthetic": True, "llm_calls": 0,
              "docker_calls": 0, "concurrency": 1, "status": "error", "checks": {},
              "scope": "D1-D5 deterministic interface integration, not research efficacy"}
    checks, denials, executions = result["checks"], [], []

    def denied(label, operation):
        try:
            operation()
        except (ProtocolError, ValueError) as exc:
            denials.append({"check": label, "reason": str(exc)})
            return True
        raise AssertionError("expected denial: " + label)

    try:
        registry, replay, reviews, cases = prepare_fixture(root / "fixture")

        def independent_fixture(branch, workspace, messages, runtime):
            # Deliberately identical response across probes. This fixture claims no probe benefit.
            start = time.perf_counter()
            state = json.loads((workspace / "state.json").read_bytes())
            executions.append({"branch": branch.key, "line": branch.line, "probe_id": branch.probe_id,
                               "phase": branch.phase, "checkpoint": branch.checkpoint_id})
            completed = state["original_completion"] and state["value"] == 7
            scored = {"outcome": {"execution_valid": True, "target_resolved": completed,
                                   "regression_free": True, "task_completed": completed},
                      "grader": "independent deterministic fixture check"}
            return ExecutionOutcome(scored, {"input_tokens": 0, "output_tokens": 0, "gpu_seconds": 0,
                                            "environment_seconds": time.perf_counter() - start,
                                            "wall_seconds": time.perf_counter() - start})

        scheduler = ExperimentScheduler(root / "private" / "scheduler.sqlite", qualifier=replay.qualify,
                                        probe_authorizer=RegistryAuthorizer(registry, reviews),
                                        executor=QualifiedBranchExecutor(replay, root / "executions", independent_fixture), logical_price=1)
        for case_id, data in cases.items():
            scheduler.enroll_plan(case_id, case_contract=data["contract"], branches=data["branches"])
        a, b = scheduler.register_method("method-A", 16), scheduler.register_method("method-B", 16)
        shared_case = cases["synthetic-e2-s"]
        universe = [s.key for s in shared_case["branches"] if s.phase == "exploration"]
        forecasts = {key: 0.8 for key in universe}  # Independent Bernoulli probabilities, sum need not be 1.
        selected = universe[0]
        denied("query_without_commit", lambda: a.request_probe("synthetic-e2-s", 0))
        a.commit_prediction("synthetic-e2-s", 0, forecasts, selected)
        b.commit_prediction("synthetic-e2-s", 0, forecasts, selected)
        denied("immutable_round_prediction", lambda: a.commit_prediction("synthetic-e2-s", 0, forecasts, selected))
        denied("confirmation_before_exploration_ends", scheduler.finalize_confirmation)
        ra = a.request_probe("synthetic-e2-s", 0)
        denied("unpaid_result", lambda: a.reveal_result(ra))
        denied("cross_method_result", lambda: b.reveal_result(ra))
        a.charge_budget(ra)
        first = a.reveal_result(ra)
        checks["2_method_observation_isolation"] = len(a.observations()) == 1 and b.observations() == []
        rb = b.request_probe("synthetic-e2-s", 0)
        b.charge_budget(rb)
        second = b.reveal_result(rb)
        audit = scheduler.audit()
        checks["1_cache_one_execution_two_logical_charges"] = (
            first["result"] == second["result"] and len(executions) == 1
            and sorted(m["spent"] for m in audit["methods"]) == [1, 1]
            and len(audit["physical"]) == 1
        )
        checks["3_commit_before_query_and_immutable"] = all(
            any(d["check"] == name for d in denials)
            for name in ("query_without_commit", "immutable_round_prediction", "unpaid_result")
        )
        (root / "shared_cache_stage.json").write_bytes(encode({"physical_executions": len(executions),
                                                              "logical_spent": {m["id"]: m["spent"] for m in audit["methods"]}}))

        # Complete every exploration panel serially; one method's cache never edits another's history.
        for case_id, data in cases.items():
            branches = [s for s in data["branches"] if s.phase == "exploration"]
            keys = [s.key for s in branches]
            start_round = 1 if case_id == "synthetic-e2-s" else 0
            for round_index, branch in enumerate(branches[start_round:], start=start_round):
                a.commit_prediction(case_id, round_index, {k: 0.5 for k in keys}, branch.key)
                receipt = a.request_probe(case_id, round_index)
                a.charge_budget(receipt)
                a.reveal_result(receipt)
        checks["4_fresh_original_controls_both_lines"] = all(
            any(e["line"] == line and e["probe_id"] == "original" and e["phase"] == "exploration" for e in executions)
            for line in ("E2-S", "E2-F")
        )
        a.close_exploration()
        denied("confirmation_waits_for_all_methods", scheduler.finalize_confirmation)
        b.close_exploration()
        before = len(executions)
        confirmation = scheduler.finalize_confirmation()
        checks["6_confirmation_sealed_until_all_exploration_ends"] = (
            len(confirmation) == 8 and len(executions) == before + 8
            and not hasattr(a, "finalize_confirmation")
            and all(o["branch"] not in confirmation for o in a.observations() + b.observations())
        )

        # Unsupported state reaches the actual D3 gate, not a caller-supplied boolean stub.
        data = cases["synthetic-e2-f"]
        bad_contract = {**STATE_CONTRACT_V1, "persistent_processes": True}
        bad_cp = replay.register_checkpoint(data["workspace"], data["messages"], data["system_hash"], data["runtime"], bad_contract, True)
        bad_branches = [replace(s, case_id="unrestorable", checkpoint_id=bad_cp, checkpoint_identity=bad_cp) for s in data["branches"]]
        for spec in data["branches"]:
            reviews[("unrestorable", spec.probe_id)] = reviews[(spec.case_id, spec.probe_id)]
        no_calls = []
        bad = ExperimentScheduler(root / "private" / "unrestorable.sqlite", qualifier=replay.qualify,
                                  probe_authorizer=RegistryAuthorizer(registry, reviews),
                                  executor=lambda spec: no_calls.append(spec), logical_price=1)
        bad.enroll_plan("unrestorable", case_contract=data["contract"], branches=bad_branches)
        c = bad.register_method("method-C", 1)
        bad_keys = [s.key for s in bad_branches if s.phase == "exploration"]
        c.commit_prediction("unrestorable", 0, {k: 0.5 for k in bad_keys}, bad_keys[0])
        denied("unrestorable_checkpoint", lambda: c.request_probe("unrestorable", 0))
        checks["5_unrestorable_checkpoint_never_executes"] = no_calls == [] and bad.audit()["physical"] == []
        if not all(checks.values()) or len(checks) != 6:
            raise AssertionError(checks)
        (root / "private" / "audit.json").write_bytes(encode(scheduler.audit()))
        (root / "confirmation.synthetic.json").write_bytes(encode(confirmation))
        result.update(status="passed", physical_execution_count=len(executions),
                      logical_spent={m["id"]: m["spent"] for m in scheduler.audit()["methods"]},
                      confirmation_count=len(confirmation))
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        result["denials"] = denials
        result["physical_execution_trace"] = executions
        result["total_wall_seconds"] = time.perf_counter() - begun
        result["source_identities"] = source_identities()
        (root / "acceptance.json").write_bytes(encode(result))
    return result
