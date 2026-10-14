from c3_sandbox.integrated import (
    ALL_TEXT, C3_STATE, OWN_TEXT, DeterministicPredictor,
    build_evidence_index, extract_object_changes, generate_report, select_probe,
)


def test_index_requires_explicit_join_and_keeps_unmatched():
    index = build_evidence_index(
        snapshots=[{"case_id": "a", "event_index": 1, "sha256": "x"}],
        dispatches=[{"case_id": "a", "event_index": 1, "dispatch_id": "d"}, {"case_id": "b", "event_index": 2}],
        scores=[{"case_id": "a", "event_index": 1, "official": 0.5}],
    )
    assert index.entries[0].status == "matched"
    assert any(item["key"] == ("b", "2") for item in index.unmatched)


def test_methods_are_sealed_and_same_budget():
    predictor = DeterministicPredictor(budget=2)
    evidence = {"text": "x", "state": {"changed": True}, "is_synthetic": True}
    records = [predictor.predict(evidence, case_id="a", candidate_universe=["yes", "no"], method=method)
               for method in (OWN_TEXT, ALL_TEXT, C3_STATE)]
    assert all(record["sealed"] and record["pre_confirmation"] for record in records)
    assert {record["budget"] for record in records} == {2}
    assert all(sum(record["probabilities"].values()) == 1 for record in records)


def test_selector_unknown_without_admissible_probe():
    assert select_probe({"candidate_probes": []})["status"] == "unknown"
    assert select_probe({"candidate_probes": ["original"]})["status"] == "selected"


def test_report_marks_invalid_unknown_and_costs_without_efficacy_claim():
    good = DeterministicPredictor().predict({}, case_id="a", candidate_universe=["yes"], method=OWN_TEXT)
    bad = dict(good, prediction_identity="tampered")
    report = generate_report([good, bad], [{"case_id": "a", "natural_success": True, "logical_cost": 2, "physical_cost": 3}])
    assert report["counts"] == {"invalid": 1, "unknown": 0, "matched": 1}
    assert report["logical_cost"] == 2 and report["physical_cost"] == 3
    assert report["scientific_efficacy_claim"] is False


def test_object_changes_require_explicit_dispatch_delta():
    rows = extract_object_changes([{"task_id": "t", "tool_call_id": "c",
                                    "fs_delta": [{"path": "out.docx", "before": None,
                                                  "after": {"sha256": "a"}}]}])
    assert rows[0]["object_path"] == "out.docx"
    assert rows[0]["event_id"] == "c"
    assert extract_object_changes([{"task_id": "t", "tool_call_id": "c"}]) == []
