"""Adversarial D1–D5 integration using real replay and probe admission."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import stat

import jsonschema
import pytest

from c3_sandbox.acceptance import prepare_fixture, run_acceptance, synthetic_review
from c3_sandbox.experiment_controller import QualifiedBranchExecutor, RegistryAuthorizer
from c3_sandbox.experiment_scheduler import ExecutionOutcome, ExperimentScheduler, ProtocolError
from c3_sandbox.manifests import canonical_hash, validate_family_manifest, validate_system_manifest
from c3_sandbox.replay import STATE_CONTRACT_V1
from c3_sandbox.trajectory import validate_trajectory

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def integrated(tmp_path):
    registry, replay, reviews, cases = prepare_fixture(tmp_path / "fixture")
    calls = []

    def handler(branch, workspace, messages, runtime):
        case = cases[branch.case_id]
        # Inspect independently before any callback side effect. Exact JSON metadata
        # and hidden files are part of the state, not just rendered conversation text.
        assert messages == case["messages"]
        assert runtime == case["runtime"]
        assert (workspace / ".retained").read_bytes() == b"hidden state retained\n"
        state = json.loads((workspace / "state.json").read_bytes())
        assert state == {"case_id": branch.case_id, "original_completion": branch.line == "E2-S", "value": 7}
        calls.append(branch)
        completed = state["original_completion"]
        return ExecutionOutcome(
            {"outcome": dict(execution_valid=True, target_resolved=completed,
                             regression_free=True, task_completed=completed)},
            dict(input_tokens=0, output_tokens=0, gpu_seconds=None,
                 environment_seconds=0.0, wall_seconds=0.0),
        )

    scheduler = ExperimentScheduler(
        tmp_path / "broker.sqlite", qualifier=replay.qualify,
        probe_authorizer=RegistryAuthorizer(registry, reviews),
        executor=QualifiedBranchExecutor(replay, tmp_path / "runs", handler),
    )
    return dict(registry=registry, replay=replay, reviews=reviews, cases=cases,
                scheduler=scheduler, calls=calls, root=tmp_path)


def enroll(env, case_id="synthetic-e2-s"):
    case = env["cases"][case_id]
    env["scheduler"].enroll_plan(case_id, case_contract=case["contract"], branches=case["branches"])
    return [b for b in case["branches"] if b.phase == "exploration"]


def commit(client, panel, round_index=0, selected=0):
    # Bernoulli success forecasts are independent, so this intentionally sums > 1.
    return client.commit_prediction(panel[0].case_id, round_index,
                                    {b.key: .8 for b in panel}, panel[selected].key)


def acquire(client, panel):
    receipt = client.request_probe(panel[0].case_id, 0)
    client.charge_budget(receipt)
    return client.reveal_result(receipt)


def test_acceptance_persists_six_invariants_and_fresh_matched_controls(tmp_path):
    output = tmp_path / "acceptance"
    report = run_acceptance(output)
    assert report["status"] == "passed", report
    assert len(report["checks"]) == 6 and all(report["checks"].values())
    assert report["physical_execution_count"] == 16
    assert report["logical_spent"] == {"method-A": 8, "method-B": 1}
    assert report["confirmation_count"] == 8
    assert report["llm_calls"] == report["docker_calls"] == 0
    assert json.loads((output / "acceptance.json").read_bytes()) == report
    shared = json.loads((output / "shared_cache_stage.json").read_bytes())
    assert shared == {"physical_executions": 1, "logical_spent": {"method-A": 1, "method-B": 1}}
    for line in ("E2-S", "E2-F"):
        for phase in ("exploration", "confirmation"):
            rows = [r for r in report["physical_execution_trace"] if r["line"] == line and r["phase"] == phase]
            assert len(rows) == 4 and len({r["checkpoint"] for r in rows}) == 1
            assert sum(r["probe_id"] == "original" for r in rows) == 1
            assert sum(r["probe_id"].endswith(".matched") for r in rows) == 1


def test_fixture_trajectories_schema_and_native_refs_link_to_actual_replay(integrated):
    validator = jsonschema.Draft202012Validator(json.loads((ROOT / "schemas/trajectory_schema.json").read_bytes()))
    for case_id, case in integrated["cases"].items():
        record = json.loads((integrated["root"] / "fixture" / (case_id + ".trajectory.json")).read_bytes())
        validator.validate(record)
        assert validate_trajectory(record) == record
        reference = record["checkpoint_refs"][0]
        qualification = integrated["replay"].qualify(reference["checkpoint_id"])
        assert qualification["eligible_for_intervention"] is True
        assert reference["checkpoint_identity"] == case["checkpoint"]
        assert reference["qualification_report_ref"] == canonical_hash(qualification)
        bad = deepcopy(record)
        del bad["checkpoint_refs"][0]["qualification_report_ref"]
        assert not validator.is_valid(bad)
        with pytest.raises(ValueError):
            validate_trajectory(bad)


def test_cold_restore_ignores_later_origin_mutation_and_preserves_tool_metadata(integrated):
    case = integrated["cases"]["synthetic-e2-s"]
    workspace = case["workspace"]
    (workspace / "state.json").chmod(0o640)
    messages = case["messages"] + [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "fixture_reader", "arguments": '{"path":"state.json"}'}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "7", "audit_metadata": {"encoding": "exact", "items": [1, 2]}},
    ]
    checkpoint = integrated["replay"].register_checkpoint(workspace, messages, case["system_hash"], case["runtime"], STATE_CONTRACT_V1, True)
    case["checkpoint"], case["messages"] = checkpoint, messages
    case["branches"] = [replace(b, checkpoint_id=checkpoint, checkpoint_identity=checkpoint) for b in case["branches"]]
    (workspace / "state.json").write_text("origin was subsequently changed")
    (workspace / ".retained").unlink()
    panel = enroll(integrated)
    client = integrated["scheduler"].register_method("A", 1)
    commit(client, panel)
    acquire(client, panel)
    restored = integrated["root"] / "runs" / panel[0].key / "workspace"
    assert stat.S_IMODE((restored / "state.json").stat().st_mode) == 0o640
    assert len(integrated["calls"]) == 1


@pytest.mark.parametrize("mutation", ["messages", "file", "mode"])
def test_second_verification_stops_changed_restore_before_callback(integrated, monkeypatch, mutation):
    panel = enroll(integrated)
    client = integrated["scheduler"].register_method("A", 1)
    commit(client, panel)
    replay = integrated["replay"]
    original = replay.restore

    def changed_restore(checkpoint, destination):
        report = original(checkpoint, destination)
        assert report["valid"]
        if mutation == "messages":
            report["messages"][0]["injected_metadata"] = "changed"
        elif mutation == "file":
            (destination / "state.json").write_text("changed")
        else:
            (destination / ".retained").chmod(0o700)
        return report

    monkeypatch.setattr(replay, "restore", changed_restore)
    with pytest.raises(ProtocolError, match="indeterminate") as failure:
        client.request_probe(panel[0].case_id, 0)
    assert "restoration verification failed" in str(failure.value.__cause__)
    assert integrated["calls"] == []
    assert integrated["scheduler"].audit()["physical"][0]["state"] == "indeterminate"


@pytest.mark.parametrize("mutation", ["revoked", "stale", "checkpoint_record", "snapshot_object"])
def test_cached_branch_rechecks_actual_review_and_checkpoint(integrated, mutation):
    panel = enroll(integrated)
    scheduler = integrated["scheduler"]
    a, b = scheduler.register_method("A", 1), scheduler.register_method("B", 1)
    commit(a, panel)
    commit(b, panel)
    acquire(a, panel)
    before = scheduler.audit()
    if mutation == "revoked":
        del integrated["reviews"][(panel[0].case_id, panel[0].probe_id)]
    elif mutation == "stale":
        integrated["reviews"][(panel[0].case_id, panel[0].probe_id)]["task_identity"] = "0" * 64
    else:
        replay = integrated["replay"]
        record_path = replay.records / (panel[0].checkpoint_id + ".json")
        if mutation == "checkpoint_record":
            record_path.chmod(0o600)  # Explicit corruption by trusted store owner.
            record_path.write_text("{}")
        else:
            record = json.loads(record_path.read_bytes())
            manifest = json.loads((replay.workspace_store.manifests / (record["snapshot_id"] + ".json")).read_bytes())
            object_id = manifest["entries"]["state.json"]["sha256"]
            object_path = replay.workspace_store.objects / object_id
            object_path.chmod(0o600)
            object_path.write_bytes(b"changed")
    with pytest.raises(ProtocolError, match="probe not admitted|checkpoint ineligible"):
        b.request_probe(panel[0].case_id, 0)
    after = scheduler.audit()
    assert len(integrated["calls"]) == 1
    assert after["physical"] == before["physical"]
    assert after["receipts"] == before["receipts"]
    assert b.observations() == []
    assert [(m["spent"], m["reserved"]) for m in after["methods"]] == [(1, 0), (0, 0)]


def test_insufficient_budget_does_not_create_receipt_or_execution(integrated):
    panel = enroll(integrated)
    integrated["scheduler"].price = 2
    client = integrated["scheduler"].register_method("A", 1)
    commit(client, panel)
    with pytest.raises(ProtocolError, match="insufficient logical budget"):
        client.request_probe(panel[0].case_id, 0)
    audit = integrated["scheduler"].audit()
    assert integrated["calls"] == audit["receipts"] == audit["physical"] == []
    assert audit["methods"][0]["spent"] == audit["methods"][0]["reserved"] == 0


def test_method_isolation_and_prediction_immutable_in_real_pipeline(integrated):
    panel = enroll(integrated, "synthetic-e2-f")
    scheduler = integrated["scheduler"]
    a, b = scheduler.register_method("A", 2), scheduler.register_method("B", 2)
    with pytest.raises(ProtocolError, match="commit"):
        a.request_probe(panel[0].case_id, 0)
    probabilities = {p.key: .8 for p in panel}
    a.commit_prediction(panel[0].case_id, 0, probabilities, panel[0].key)
    probabilities[panel[0].key] = .1
    with pytest.raises(ProtocolError):
        commit(a, panel, selected=1)
    receipt = a.request_probe(panel[0].case_id, 0)
    with pytest.raises(ProtocolError):
        a.reveal_result(receipt)
    for operation in (b.charge_budget, b.reveal_result):
        with pytest.raises(ProtocolError):
            operation(receipt)
    a.charge_budget(receipt)
    observed = a.reveal_result(receipt)
    assert observed["result"]["outcome"]["task_completed"] is False
    assert a.reveal_result(receipt) == observed
    assert len(a.observations()) == 1 and b.observations() == []
    sealed = json.loads(scheduler.audit()["predictions"][0]["body"])
    assert sealed["probabilities"] == {p.key: .8 for p in panel}


def test_manifests_reject_changed_runtime_and_same_source_confirmation(integrated):
    fixture = integrated["root"] / "fixture"
    system = json.loads((fixture / "system_manifest.json").read_bytes())
    system["systems"][0]["config"]["runtime"]["environment"]["LANG"] = "changed"
    with pytest.raises(ValueError, match="does not match"):
        validate_system_manifest(system)
    family = json.loads((fixture / "family_manifest.json").read_bytes())
    family["families"][1]["split"] = "confirmation"
    family["families"][1]["source_group_ids"] = family["families"][0]["source_group_ids"]
    with pytest.raises(ValueError, match="source group leaks"):
        validate_family_manifest(family)


@pytest.mark.parametrize("mutation", ["overlapping_seed", "changed_temperature", "unregistered_seed"])
def test_stage_seed_namespace_and_frozen_sampling_gate(integrated, mutation):
    case = integrated["cases"]["synthetic-e2-s"]
    if mutation == "overlapping_seed":
        case["branches"] = [replace(b, sampling={**b.sampling, "seed": 2026}) for b in case["branches"]]
    elif mutation == "changed_temperature":
        case["branches"] = [replace(b, sampling={**b.sampling, "temperature": .9}) if b.phase == "confirmation" else b for b in case["branches"]]
    else:
        case["branches"] = [replace(b, sampling={**b.sampling, "seed": 999}) if b.phase == "exploration" else b for b in case["branches"]]
    with pytest.raises(ProtocolError, match="disjoint|non-seed|frozen case schedule"):
        enroll(integrated)
    assert integrated["calls"] == integrated["scheduler"].audit()["physical"] == []


def test_registered_plan_never_replaced_after_prediction(integrated):
    panel = enroll(integrated)
    scheduler = integrated["scheduler"]
    client = scheduler.register_method("A", 1)
    commit(client, panel)
    before = scheduler.audit()["predictions"]
    case = integrated["cases"][panel[0].case_id]
    with pytest.raises(ProtocolError, match="frozen"):
        scheduler.enroll_plan(panel[0].case_id, case_contract=case["contract"], branches=case["branches"])
    assert scheduler.audit()["predictions"] == before
    acquire(client, panel)
    with pytest.raises(ProtocolError, match="known branch"):
        commit(client, panel, round_index=1)
    assert len(scheduler.audit()["predictions"]) == 1


@pytest.mark.parametrize("mutation", ["renamed_baseline", "omitted_original", "omitted_matched", "reviewed_wrong_control"])
def test_semantic_original_and_matched_panel_cannot_be_relabelled(integrated, mutation):
    case_id = "synthetic-e2-s"
    case = integrated["cases"][case_id]
    contract, branches = case["contract"], case["branches"]
    if mutation == "renamed_baseline":
        contract["baseline_probe_id"] = contract["target_probe_id"]
        branches = [b for b in branches if b.probe_id != "original"]
    elif mutation == "omitted_original":
        branches = [b for b in branches if b.probe_id != "original"]
    elif mutation == "omitted_matched":
        branches = [b for b in branches if b.probe_id != contract["control_probe_id"]]
    else:
        wrong = "feedback.aggregate.matched"
        registry = integrated["registry"]
        integrated["reviews"][(case_id, wrong)] = synthetic_review(registry, contract, wrong)
        assert registry.validate_for_case(wrong, contract, integrated["reviews"][(case_id, wrong)])["eligible"]
        branches = [replace(b, probe_id=wrong, probe_identity=registry.identity(wrong))
                    if b.probe_id == contract["control_probe_id"] else b for b in branches]
        contract["control_probe_id"] = wrong
    with pytest.raises(ProtocolError, match="baseline|control|incomplete"):
        integrated["scheduler"].enroll_plan(case_id, case_contract=contract, branches=branches)
    with integrated["scheduler"]._db() as db:
        assert db.execute("SELECT COUNT(*) FROM plans").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM branches").fetchone()[0] == 0
    assert integrated["calls"] == integrated["scheduler"].audit()["physical"] == []


def test_confirmation_controller_waits_and_method_never_observes_confirmation(integrated):
    panel = enroll(integrated)
    scheduler = integrated["scheduler"]
    a, b = scheduler.register_method("A", 1), scheduler.register_method("B", 1)
    commit(a, panel)
    with pytest.raises(ProtocolError):
        scheduler.finalize_confirmation()
    receipt = a.request_probe(panel[0].case_id, 0)
    a.charge_budget(receipt)
    with pytest.raises(ProtocolError, match="pending"):
        a.close_exploration()
    a.reveal_result(receipt)
    a.close_exploration()
    with pytest.raises(ProtocolError):
        scheduler.finalize_confirmation()
    b.close_exploration()
    results = scheduler.finalize_confirmation()
    assert len(results) == 4
    assert len(integrated["calls"]) == 5
    assert scheduler.finalize_confirmation() == results
    assert len(integrated["calls"]) == 5
    assert b.observations() == [] and len(a.observations()) == 1
    assert not {o["branch"] for o in a.observations()} & results.keys()
    assert not hasattr(a, "finalize_confirmation")


@pytest.mark.parametrize("state", ["persistent_processes", "pending_tool_calls", "external_state"])
def test_unsupported_state_cannot_reach_callback(integrated, state):
    case = integrated["cases"]["synthetic-e2-s"]
    contract = deepcopy(STATE_CONTRACT_V1)
    contract[state] = {"persistent_processes": True, "pending_tool_calls": 1, "external_state": "service"}[state]
    checkpoint = integrated["replay"].register_checkpoint(
        case["workspace"], case["messages"], case["system_hash"], case["runtime"], contract, True,
    )
    case["branches"] = [replace(b, checkpoint_id=checkpoint, checkpoint_identity=checkpoint) for b in case["branches"]]
    panel = enroll(integrated)
    client = integrated["scheduler"].register_method("A", 1)
    commit(client, panel)
    with pytest.raises(ProtocolError, match="checkpoint ineligible"):
        client.request_probe(panel[0].case_id, 0)
    assert integrated["calls"] == integrated["scheduler"].audit()["physical"] == []
