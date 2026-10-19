import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("d0_v2_freezer", ROOT / "L2_Behavioral Event Localization/L2_Sandbox/scripts/freeze_d0_v2.py")
freezer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(freezer)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    def put(name, doc):
        path = tmp_path / name
        path.write_text(json.dumps(doc))
        return path
    authorization = put("authorization.json", {"authorization_source": "direct_user_reply"})
    contract = put("contract.json", {"policy": freezer.POLICY})
    refs = {name: {"path": str(path), "sha256": freezer.file_digest(path)} for name, path in [
        ("v2_authorization", authorization), ("v2_contract", contract)]}
    binding = dict(policy=freezer.POLICY, post_hoc_forensic_extension=True, passed=True,
        execution_readiness=False, execution_authorized=False, current_execution_scope="D0_freeze_and_D1_plan_only", request_attachment=refs["v2_authorization"], g2_v2_verifier=refs["v2_contract"],
        gates={f"V3-G{i}": True for i in range(1, 5)}, refs=refs,
        statistical_equivalence="not_established", outcome_distribution_equivalence="not_established")
    audit = put("audit.json", {})
    binding.update(path=str(audit), sha256=freezer.file_digest(audit))
    monkeypatch.setattr(freezer, "measurement_gate_binding", lambda path: (binding, []))
    plan = freezer.seal(dict(scientific_plan_ready=True, readiness_errors=[], mode="plan_only",
        execution_ready=False, execution_authorized=False, d0_candidate_parameters=dict(freezer.PARAMETERS),
        unit_schedules=[{} for _ in range(24)], r12_passing_harm_check_plan={"sampled_passing_units": [{} for _ in range(6)]},
        measurement_gate_binding=binding, prospective_binding={"inputs": {}}, task_ladder_message_specs={},
        stratified_sampling_frame={}, canonical_unit_census={}, execution_blockers=["observer", "caps"]), "plan_sha256")
    plan_path = put("plan.json", plan)
    gate_path = put("gate.json", {"binding": binding, "errors": []})
    monkeypatch.setattr(freezer, "_planner", lambda: SimpleNamespace(verify_prospective_plan_inputs=lambda p: None))
    return dict(plan_path=plan_path, gate_binding_path=gate_path,
        authorization_path=authorization, contract_path=contract)


def edit_plan(inputs, change):
    plan = freezer.read_json(inputs["plan_path"])
    change(plan)
    plan.pop("plan_sha256")
    inputs["plan_path"].write_text(json.dumps(freezer.seal(plan, "plan_sha256")))


def test_freeze_is_scientific_only_and_idempotent(inputs, tmp_path):
    output = tmp_path / "freeze.json"
    result = freezer.produce(output=output, **inputs)
    assert result["d0_frozen"] is True
    assert result["execution_authorized"] is result["execution_ready"] is False
    assert "approved_plan_sha256" not in result
    assert result["original_v1_gate_status"] == "preserved_blocked"
    assert freezer.produce(output=output, **inputs) == result
    assert freezer.verify_freeze(output) == result


@pytest.mark.parametrize("name,value", [("n_per_arm", 12), ("tau", .25), ("delta", .2), ("estimand", "family_population")])
def test_changed_scientific_parameter_rejected(inputs, name, value):
    edit_plan(inputs, lambda p: p["d0_candidate_parameters"].update({name: value}))
    with pytest.raises(ValueError, match="parameters differ"):
        freezer.build_freeze(**inputs)


def test_corrupted_plan_seal_rejected(inputs):
    plan = freezer.read_json(inputs["plan_path"])
    plan["scientific_plan_ready"] = False
    inputs["plan_path"].write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        freezer.build_freeze(**inputs)


def test_missing_passing_control_rejected(inputs):
    edit_plan(inputs, lambda p: p["r12_passing_harm_check_plan"]["sampled_passing_units"].pop())
    with pytest.raises(ValueError, match="exactly 24"):
        freezer.build_freeze(**inputs)


def test_gate_mismatch_rejected(inputs):
    document = freezer.read_json(inputs["gate_binding_path"])
    document["binding"]["gates"]["V3-G2"] = False
    inputs["gate_binding_path"].write_text(json.dumps(document))
    with pytest.raises(ValueError, match="binding differ"):
        freezer.build_freeze(**inputs)


def test_changed_authorization_bytes_rejected(inputs):
    inputs["authorization_path"].write_text("{}")
    with pytest.raises(ValueError, match="bound file changed"):
        freezer.build_freeze(**inputs)


def test_changed_input_after_freeze_rejected(inputs, tmp_path):
    output = tmp_path / "freeze.json"
    freezer.produce(output=output, **inputs)
    inputs["contract_path"].write_text("changed")
    with pytest.raises(ValueError, match="frozen input changed"):
        freezer.verify_freeze(output)


def test_different_freeze_cannot_overwrite(inputs, tmp_path):
    output = tmp_path / "freeze.json"
    freezer.produce(output=output, **inputs)
    original = output.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        freezer.produce(output=output, source_commit="different", **inputs)
    assert output.read_bytes() == original


def test_prospective_input_recheck_failure_propagates(inputs, monkeypatch):
    def fail(plan):
        raise ValueError("prospective source changed")
    monkeypatch.setattr(freezer, "_planner", lambda: SimpleNamespace(verify_prospective_plan_inputs=fail))
    with pytest.raises(ValueError, match="source changed"):
        freezer.build_freeze(**inputs)


@pytest.mark.parametrize("key,value", [("execution_readiness", True), ("execution_authorized", True), ("current_execution_scope", "execute")])
def test_gate_cannot_upgrade_execution_authority(inputs, key, value):
    gate = freezer.read_json(inputs["gate_binding_path"])
    gate["binding"][key] = value
    inputs["gate_binding_path"].write_text(json.dumps(gate))
    edit_plan(inputs, lambda p: p["measurement_gate_binding"].update({key: value}))
    with pytest.raises(ValueError, match="non-executable"):
        freezer.build_freeze(**inputs)


def test_raw_evidence_rechecked_during_build_and_verify(inputs, tmp_path, monkeypatch):
    output = tmp_path / "freeze.json"
    freezer.produce(output=output, **inputs)
    monkeypatch.setattr(freezer, "measurement_gate_binding", lambda path: ({}, ["raw stdout changed"]))
    with pytest.raises(ValueError, match="raw evidence recheck failed"):
        freezer.build_freeze(**inputs)
    with pytest.raises(ValueError, match="raw evidence recheck failed"):
        freezer.verify_freeze(output)
