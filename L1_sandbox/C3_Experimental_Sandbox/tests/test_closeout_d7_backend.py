import json
from c3_sandbox.integrated.d7_backend import (
    ALL_TEXT, C3_STATE, OWN_TEXT, DiagnosticBackend, build_evidence_view, select_condition
)


def response(probs):
    return json.dumps({"choices": [{"message": {"content": json.dumps({"probabilities": probs})}}],
                       "usage": {"prompt_tokens": 3, "completion_tokens": 2}}).encode()


def test_views_and_all_text_reference_content():
    e = {"text": "own", "state": {"x": 1}, "reference_episodes": [{"case_id": "c", "text": "reference episode"}]}
    assert "reference_episodes" not in build_evidence_view(e, method=OWN_TEXT, case_id="c")
    assert build_evidence_view(e, method=ALL_TEXT, case_id="c")["reference_episodes"][0]["text"] == "reference episode"
    assert "reference_episodes" not in build_evidence_view(e, method=C3_STATE, case_id="c")


def test_backend_records_raw_bytes_identity_usage_and_independent_probabilities(tmp_path):
    seen = []
    backend = DiagnosticBackend(model="local/test", transport=lambda raw: (seen.append(raw) or response({"a": .2, "b": .2})), persist_dir=tmp_path)
    result = backend.predict({"text": "real evidence", "reference_episodes": [{"case_id": "c", "text": "ep"}]},
                             case_id="c", candidate_universe=["a", "b"], method=ALL_TEXT,
                             evidence_ids=["ev1"], checkpoint_identity="chk1")
    assert result["status"] == "ok" and result["usage"]["prompt_tokens"] == 3
    assert result["request_raw"] and result["response_raw"]
    assert result["evidence_ids"] == ["ev1"] and result["checkpoint_identity"] == "chk1"
    assert (tmp_path / "c-ALL_TEXT.json").exists()
    assert json.loads(seen[0])["messages"][1]["content"].find("\"text\": \"ep\"") >= 0


def test_invalid_output_and_selector_policy_never_fallback():
    backend = DiagnosticBackend(model="m", transport=lambda _: b'{"choices": []}')
    bad = backend.predict({}, case_id="c", candidate_universe=["a"], method=OWN_TEXT)
    assert bad["status"] == "selector_invalid"
    assert select_condition([bad], policy={})["status"] == "selector_invalid"
    rows = [{"method": OWN_TEXT, "status": "ok", "probabilities": {"a": .01}},
            {"method": ALL_TEXT, "status": "ok", "probabilities": {"a": .02}}]
    assert select_condition(rows, policy={"min_probability": .5})["status"] == "abstain"
    assert select_condition(rows, policy={"min_probability": .5, "allow_exploratory": True})["status"] == "exploratory"


def test_selector_returns_candidate_condition_not_method_name():
    rows = [{"method": OWN_TEXT, "status": "ok", "probabilities": {"original": .1, "probe-a": .8}},
            {"method": ALL_TEXT, "status": "ok", "probabilities": {"original": .2, "probe-a": .7}}]
    selected = select_condition(rows, policy={"min_probability": .5})
    assert selected["status"] == "selected" and selected["condition"] == "probe-a"
