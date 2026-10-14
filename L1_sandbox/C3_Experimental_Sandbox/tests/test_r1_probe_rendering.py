from copy import deepcopy
import hashlib
import re

import pytest

from c3_sandbox.r1.probes import render_pair, render_probe


@pytest.fixture
def evidence():
    return {"inventory": {"file_count": 271, "extension_counts": {".docx": 3, "other": 268},
                          "total_bytes": 981713, "docx_count": 3, "paragraph_count": 487,
                          "table_count": 19, "section_count": 7, "header_count": 11,
                          "omml_count": 43, "xml_parse_error_count": 2},
            "feedback": {"exit_code": 1, "stdout_bytes": 231, "stderr_bytes": 137,
                         "files_changed": 5, "parse_error_count": 2}}


def tokenizer(text):
    # Content-sensitive deterministic tokenizer: whitespace-only equality would
    # not suffice. Production must supply the exact runtime tokenizer.
    return len(re.findall(r"[a-zA-Z]{1,4}|[0-9]{1,3}|[^\w\s]", text))


def test_original_has_no_added_message(evidence):
    result = render_probe("original", evidence, token_counter=tokenizer)
    assert result.rendered_text == ""
    assert result.payload == {}
    assert result.token_count == 0


@pytest.mark.parametrize("target", ["visibility.expanded", "visibility.compact",
                                   "feedback.structured", "feedback.aggregate"])
def test_targets_are_real_observations_and_controls_are_neutral(target, evidence):
    pair = render_pair(target, evidence, token_counter=tokenizer)
    actual, control = pair[target], pair[target + ".matched"]
    assert actual.rendered_text != control.rendered_text
    assert "Runtime-owned advisory" in actual.rendered_text
    assert "unreported" in control.rendered_text
    for value in ("981713", "487", "231", "137", "368", "271"):
        assert value not in control.rendered_text
    assert len(actual.rendered_text.encode()) == len(control.rendered_text.encode())
    assert tokenizer(actual.rendered_text) == tokenizer(control.rendered_text) == actual.token_count
    assert actual.matching_audit["tokenizer_verified"]
    assert not control.matching_audit["control_inertness_assumed"]
    assert actual.public_digest == control.public_digest
    assert actual.rendered_digest == hashlib.sha256(actual.rendered_text.encode()).hexdigest()


def test_visibility_preserves_counts_while_layout_changes(evidence):
    expanded = render_probe("visibility.expanded", evidence).rendered_text
    compact = render_probe("visibility.compact", evidence).rendered_text
    assert "Document structure totals\n" in expanded
    assert "Document structure totals" not in compact
    for key, value in evidence["inventory"].items():
        assert key in expanded and key in compact
        if isinstance(value, int):
            assert str(value) in expanded and str(value) in compact


def test_feedback_granularity_actually_changes(evidence):
    structured = render_probe("feedback.structured", evidence).rendered_text
    aggregate = render_probe("feedback.aggregate", evidence).rendered_text
    assert "stdout_bytes: 231" in structured
    assert "stderr_bytes: 137" in structured
    assert "output_bytes=368" in aggregate
    assert "stdout_bytes" not in aggregate and "stderr_bytes" not in aggregate


@pytest.mark.parametrize("location", ["root", "inventory", "feedback", "extensions"])
def test_extra_leak_channels_rejected(location, evidence):
    bad = deepcopy(evidence)
    if location == "root":
        bad["answer"] = "secret"
    elif location == "extensions":
        bad["inventory"]["extension_counts"]["/target/answer.docx"] = 1
    else:
        bad[location]["expected_values"] = {"answer": "secret"}
    with pytest.raises(ValueError):
        render_probe("original", bad)


@pytest.mark.parametrize("value", [True, -1, "answer", {"path": "/target"}, float("nan"), 2**64])
def test_nested_or_non_integer_inventory_rejected(value, evidence):
    evidence["inventory"]["paragraph_count"] = value
    with pytest.raises(ValueError):
        render_probe("visibility.expanded", evidence)


def test_no_feedback_is_explicit_and_not_invented(evidence):
    evidence["feedback"] = None
    assert "unreported" in render_probe("feedback.structured", evidence).rendered_text
    assert "unreported" in render_probe("feedback.aggregate", evidence).rendered_text


def test_rendering_is_deterministic_and_detached(evidence):
    first = render_pair("visibility.expanded", evidence)
    assert first == render_pair("visibility.expanded", evidence)
    evidence["inventory"]["paragraph_count"] = 99
    assert first["visibility.expanded"].public_evidence["inventory"]["paragraph_count"] == 487
    assert first["visibility.expanded"].token_count is None
    assert first["visibility.expanded"].matching_audit["token_counts_equal"] is None


def test_matching_failure_is_closed(evidence):
    # This intentionally pathological counter cannot assign equal counts to
    # observations with and without reported information.
    with pytest.raises(ValueError, match="unable to match"):
        render_pair("visibility.expanded", evidence,
                    token_counter=lambda text: 1 if "981713" in text else 2)


@pytest.mark.parametrize("target", ["original", "visibility.compact.matched", "diagnosis"])
def test_pair_requires_registered_target(target, evidence):
    with pytest.raises(ValueError):
        render_pair(target, evidence)


def test_invalid_token_counter_rejected(evidence):
    with pytest.raises(ValueError, match="token_counter"):
        render_probe("visibility.compact", evidence, token_counter=lambda text: True)


def test_digit_family_matches_when_old_padding_density_is_flat(evidence):
    from c3_sandbox.r1.probes import _PAD_HEADER

    def flat_letters_counter(text):
        # Models a vocabulary with one token per " x" pair or "nn" pair,
        # while digits have a different density (one token each).
        prefix, filler = text.split(_PAD_HEADER) if _PAD_HEADER in text else (text, "")
        return tokenizer(prefix) + filler.count(" x") + (filler.count("n") + 1) // 2 + filler.count("0")

    arms = [render_probe(key, evidence).rendered_text for key in
            ("visibility.expanded", "visibility.expanded.matched")]
    width = max(map(len, arms)) + len(_PAD_HEADER) + 1024
    old_counts = []
    for text in arms:
        prefix = text + _PAD_HEADER
        size = width - len(prefix)
        old_counts.append({flat_letters_counter(prefix + " x" * i + "n" * (size - 2 * i)) for i in range(513)})
    assert not old_counts[0] & old_counts[1]
    pair = render_pair("visibility.expanded", evidence, token_counter=flat_letters_counter)
    values = list(pair.values())
    assert values[0].token_count == values[1].token_count
    assert len(values[0].rendered_text.encode()) == len(values[1].rendered_text.encode())
    for value in values:
        assert value.matching_audit["padding_family"] == "digits"
        assert value.rendered_text.startswith(render_probe(value.probe_id, evidence).rendered_text + _PAD_HEADER)
        assert value.token_count == flat_letters_counter(value.rendered_text)
    assert pair == render_pair("visibility.expanded", evidence, token_counter=flat_letters_counter)


def test_padding_search_has_hard_tokenizer_call_bound(evidence):
    calls = []

    def incompatible(text):
        calls.append(text)
        return 1 if "981713" in text else 2

    with pytest.raises(ValueError, match="1026 candidates per arm"):
        render_pair("visibility.expanded", evidence, token_counter=incompatible)
    assert len(calls) == 2 + 2 * 1026
