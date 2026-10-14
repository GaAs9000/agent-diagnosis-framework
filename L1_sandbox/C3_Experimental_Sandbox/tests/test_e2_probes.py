from copy import deepcopy
import hashlib
import hmac
from pathlib import Path

import pytest

from c3_sandbox.evidence._storage import digest, encode
from c3_sandbox.probe_registry import ProbeRegistry, VERSION, _vocabulary


@pytest.fixture
def registry():
    return ProbeRegistry.from_file(Path(__file__).parents[1] / "configs/probes.v1.json")


def case(line="E2-F", synthetic=True):
    return {"task_identity": "a" * 64, "is_synthetic": synthetic, "line": line,
            "available_inputs": ["document"], "necessary_inputs": ["document"],
            "available_tools": ["editor"], "necessary_tools": ["editor"],
            "observation_capabilities": ["redundant_observation_views", "public_feedback_fields"],
            "sampling": {"temperature": 0.1, "seed": 2026}, "budgets": {"tokens": 1000}}


def review(registry, contract, probe="visibility.expanded", real=False):
    record = {"task_identity": contract["task_identity"], "probe_identity": registry.identity(probe),
              "line": contract["line"],
              "source": {"kind": "human" if real else "synthetic_fixture", "signer_id": "reviewer", "record_identity": "b" * 64},
              "findings": {"leakage": {"verdict": "no_task_answer_material", "evidence_identities": ["c" * 64]},
                           "solvability": {"verdict": "necessary_resources_preserved", "evidence_identities": ["d" * 64]},
                           "reviewed_inputs": contract["available_inputs"], "reviewed_tools": contract["available_tools"]}}
    body = encode(record)
    if real:
        contract["trusted_review_keys"] = {"reviewer": "test-only-trusted-review-key-00000000"}
        signature = hmac.new(contract["trusted_review_keys"]["reviewer"].encode(), body, hashlib.sha256).hexdigest()
    else:
        signature = digest(body)
    record["signature"] = {"scheme": "hmac-sha256" if real else "sha256-fixture", "value": signature}
    return record


@pytest.mark.parametrize("field,value", [("payload", {"script": "answer()"}), ("payload", {"mode": "expanded", "object_path": "answer.docx"}),
                                         ("payload", {"temperature": 0}), ("preconditions", []), ("preserve", [])])
def test_registry_rejects_freeform_and_modified_contract(field, value):
    document = {"version": VERSION, "probes": list(_vocabulary().values())}
    document["probes"][1][field] = value
    with pytest.raises(ValueError):
        ProbeRegistry(document)


def test_identity_frozen_and_returned_copy_is_safe(registry):
    old = registry.identity("visibility.expanded")
    registry.get("visibility.expanded")["payload"]["mode"] = "arbitrary"
    assert registry.identity("visibility.expanded") == old
    assert len(old) == 64


@pytest.mark.parametrize("line,target,competing", [("E2-S", "visibility.compact", "feedback.aggregate"),
                                                   ("E2-F", "visibility.expanded", "feedback.structured")])
def test_both_panels_have_fresh_original_and_matched_control(registry, line, target, competing):
    panel = registry.experiment_panel(line, target, competing)
    assert [p["role"] for p in panel] == ["original", "target", "matched", "competing"]
    assert all(p["requires_fresh_execution"] and p["checkpoint_policy"] == "same_checkpoint" for p in panel)
    assert panel[0]["payload"] == {}
    assert panel[2]["kind"] == "matched_control"
    assert panel[2]["payload"] == {"mode": "sham", "matched_mode": panel[1]["payload"]["mode"]}
    contract = case(line)
    for probe in panel:
        result = registry.validate_for_case(probe["probe_id"], contract, review(registry, contract, probe["probe_id"]))
        assert result["eligible"], result


@pytest.mark.parametrize("target,competing", [("visibility.expanded", "visibility.expanded"),
                                              ("visibility.expanded.matched", "feedback.structured"),
                                              ("visibility.compact", "feedback.structured")])
def test_invalid_panel(registry, target, competing):
    with pytest.raises(ValueError):
        registry.experiment_panel("E2-F", target, competing)


def test_missing_stale_or_unsubstantiated_review_denied(registry):
    contract = case()
    assert not registry.validate_for_case("visibility.expanded", contract, None)["eligible"]
    good = review(registry, contract)
    for key, value in [("task_identity", "e" * 64), ("probe_identity", "f" * 64), ("line", "E2-S"),
                       ("findings", {"leakage": True, "solvability": True})]:
        bad = deepcopy(good)
        bad[key] = value
        assert not registry.validate_for_case("visibility.expanded", contract, bad)["eligible"]
    good["findings"]["leakage"]["evidence_identities"] = []
    assert not registry.validate_for_case("visibility.expanded", contract, good)["eligible"]


@pytest.mark.parametrize("field,value", [("available_inputs", []), ("available_tools", []),
                                         ("observation_capabilities", []), ("sampling_override", {}),
                                         ("temperature_override", 0.5), ("removed_tools", ["editor"]),
                                         ("budget_override", {"tokens": 2000}), ("payload_override", {})])
def test_removed_resources_or_sampling_budget_shifts_denied(registry, field, value):
    contract = case()
    bound = review(registry, contract)
    contract[field] = value
    assert not registry.validate_for_case("visibility.expanded", contract, bound)["eligible"]


def test_real_requires_trusted_signed_provenance(registry):
    contract = case(synthetic=False)
    assert not registry.validate_for_case("visibility.expanded", contract, review(registry, contract))["eligible"]
    signed = review(registry, contract, real=True)
    assert registry.validate_for_case("visibility.expanded", contract, signed)["eligible"]
    signed["source"]["signer_id"] = "unknown"
    assert not registry.validate_for_case("visibility.expanded", contract, signed)["eligible"]


def test_signature_detects_finding_tampering(registry):
    contract = case(synthetic=False)
    signed = review(registry, contract, real=True)
    signed["findings"]["leakage"]["evidence_identities"] = ["e" * 64]
    assert not registry.validate_for_case("visibility.expanded", contract, signed)["eligible"]


def test_explicit_control_and_leakage_policy_are_frozen(registry):
    for probe in _vocabulary():
        spec = registry.get(probe)
        assert spec["control_id"] == spec["matched_control_id"]
        assert spec["leakage_policy"] == {
            "allowlist": "closed_generic_public_observation_presentation",
            "no_answer_material": True, "no_task_specific_patch": True,
            "no_target_object_paths": True, "per_task_review": "required_including_controls",
        }
    document = {"version": VERSION, "probes": list(_vocabulary().values())}
    document["probes"][1]["leakage_policy"]["no_answer_material"] = False
    with pytest.raises(ValueError):
        ProbeRegistry(document)


@pytest.mark.parametrize("malformed", [None, [], "case", 3])
def test_malformed_case_fails_closed(registry, malformed):
    assert not registry.validate_for_case("visibility.expanded", malformed, None)["eligible"]


@pytest.mark.parametrize("field,value", [("available_inputs", [["nested"]]), ("necessary_inputs", "input"),
                                         ("available_tools", [None]), ("necessary_tools", {}),
                                         ("observation_capabilities", [{}]), ("observation_capabilities", None),
                                         ("sampling", {"temperature": float("nan")}),
                                         ("sampling", {"temperature": 10 ** 500}),
                                         ("sampling", {"temperature": True}),
                                         ("budgets", {"tokens": float("inf")}),
                                         ("budgets", {"tokens": 10 ** 500}),
                                         ("budgets", {"tokens": False})])
def test_malformed_contract_fields_fail_closed(registry, field, value):
    contract = case()
    contract[field] = value
    assert not registry.validate_for_case("visibility.expanded", contract, None)["eligible"]


@pytest.mark.parametrize("field,value", [("source", None), ("source", []), ("findings", None),
                                         ("findings", True), ("signature", None), ("signature", []),
                                         ("probe_identity", []), ("task_identity", {}), ("line", [])])
def test_malformed_review_fields_fail_closed(registry, field, value):
    contract = case()
    malformed = review(registry, contract)
    malformed[field] = value
    assert not registry.validate_for_case("visibility.expanded", contract, malformed)["eligible"]
