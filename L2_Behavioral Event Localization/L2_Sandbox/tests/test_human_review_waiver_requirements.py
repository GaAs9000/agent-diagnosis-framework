"""A user waiver permits sourced classifications without inventing human review."""
from copy import deepcopy

import pytest

from l2_sandbox.io import L2Error, digest
from l2_sandbox.v05 import build_task_requirement_lists


AUTHORIZATION = {
    "human_review_policy": "waived_by_user", "authorization_source": "direct_user_message",
    "exact_text": "跳过人审核", "source": "direct human user message",
    "scope": ["requirement_catalog_human_review", "G3_human_review"],
}


def catalog():
    return {"officeval_070": {
        "保持原始内容": {"source_category": "explicit", "source_citation": "task.md:4"},
        "隐藏字号区间": {"source_category": "implicit", "source_citation": "verifier.py:20"},
        "保持间距且容差0.15cm": {
            "source_category": "ambiguous", "source_citation": "task.md:5; verifier.py:30",
            "visible_requirement": "保持间距", "exact_verifier_specification": "容差0.15cm",
        },
    }}


def waive(data):
    return build_task_requirement_lists(
        data, "officeval_070", human_review_policy="waived_by_user",
        review_authorization=AUTHORIZATION,
    )


def test_strict_default_remains_blocked_without_human_review():
    result = build_task_requirement_lists(catalog(), "officeval_070")
    assert not result["review_complete"]
    assert not result["execution_eligible"]
    assert result["validation_status"] == "blocked"
    assert result["rule_sources"] == {}
    assert result["review_authorization"] is None


def test_waiver_allows_sourced_task_wide_rules_preserving_human_status_and_separation():
    data = catalog()
    original = deepcopy(data)
    result = waive(data)
    assert result["execution_eligible"]
    assert result["validation_status"] == "user_waived_human_review"
    assert not result["review_complete"]
    assert len(result["human_review_missing"]) == 3
    assert len(result["provenance_errors"]) == 3
    assert result["explicit"] == ["保持原始内容", "保持间距"]
    assert set(result["implicit"]) == {"隐藏字号区间", "容差0.15cm"}
    assert "容差0.15cm" not in " ".join(result["explicit"])
    assert len(result["rule_sources"]) == 3
    assert result["review_authorization_sha256"] == digest(AUTHORIZATION)
    assert result["review_authorization"] == AUTHORIZATION
    assert data == original
    assert all("reviewer" not in info and "reviewed_at" not in info and "human_label" not in info
               for info in data["officeval_070"].values())


@pytest.mark.parametrize("authorization", [None, {}, {"exact_text": "跳过人审核"},
    {"exact_text": "跳过人审核", "source": " "}, {"exact_text": True, "source": "chat"}])
def test_waiver_requires_bound_direct_authorization(authorization):
    with pytest.raises(L2Error, match="authorization"):
        build_task_requirement_lists(catalog(), "officeval_070",
            human_review_policy="waived_by_user", review_authorization=authorization)


def test_unknown_policy_rejected():
    with pytest.raises(L2Error, match="human_review_policy"):
        build_task_requirement_lists(catalog(), "officeval_070", human_review_policy="auto")


@pytest.mark.parametrize("change", [{"scope": ["G3_human_review"]}, {"scope": "requirement_catalog_human_review"},
    {"authorization_source": "automatic"}, {"exact_text": "other text"}, {"human_review_policy": "required"}])
def test_waiver_rejects_wrong_scope_policy_or_instruction(change):
    authorization = {**AUTHORIZATION, **change}
    with pytest.raises(L2Error, match="authorization"):
        build_task_requirement_lists(catalog(), "officeval_070",
            human_review_policy="waived_by_user", review_authorization=authorization)


@pytest.mark.parametrize("citation", [None, "", "  ", True, 123, [{}], [" "],
    {"path": "source.py", "sha256": "invalid", "line_start": 1},
    {"path": "source.py", "sha256": "a" * 64},
    {"path": " ", "sha256": "a" * 64, "quote": "requirement"}])
def test_waiver_does_not_waive_source_provenance(citation):
    data = catalog()
    data["officeval_070"]["保持原始内容"]["source_citation"] = citation
    result = waive(data)
    assert not result["execution_eligible"]
    assert "保持原始内容" in result["execution_errors"]
    assert "保持原始内容" not in result["rule_sources"]


def test_structured_citation_requires_path_hash_and_source_location():
    data = catalog()
    data["officeval_070"]["保持原始内容"]["source_citation"] = [{
        "path": "source.py", "sha256": "a" * 64, "line_start": 5,
    }]
    assert waive(data)["execution_eligible"]


def test_lookup_aliases_are_not_injected_as_repair_instructions_or_hidden_weights():
    data = {'officeval_070': {
        '+5：表格没有空行': {'source_category': 'explicit', 'source_citation': 'task.md:1',
                           'visible_requirement': '删除空行'},
        '-1：正文出现缩进': {'source_category': 'ambiguous', 'source_citation': 'rubric.json:2',
                          'visible_requirement': '保持格式', 'exact_verifier_specification': '-1：正文出现缩进'},
        '正文出现缩进': {'source_category': 'ambiguous', 'source_citation': 'verifier.py:4',
                        'visible_requirement': '保持格式', 'exact_verifier_specification': '正文出现缩进',
                        'lookup_only': True},
    }}
    result = waive(data)
    assert result['execution_eligible']
    assert result['explicit'] == ['删除空行', '保持格式']
    assert result['implicit'] == ['-1：正文出现缩进']
    assert result['rule_sources']['正文出现缩进'] == 'ambiguous'


def test_unresolved_ambiguous_rule_never_promoted_to_explicit():
    data = catalog()
    del data["officeval_070"]["保持间距且容差0.15cm"]["visible_requirement"]
    result = waive(data)
    assert not result["execution_eligible"]
    assert result["unresolved_rules"] == ["保持间距且容差0.15cm"]
    assert result["explicit"] == ["保持原始内容"]
    assert "保持间距且容差0.15cm" not in result["rule_sources"]


def test_real_completed_review_remains_completed_under_either_policy():
    data = catalog()
    for info in data["officeval_070"].values():
        info.update(reviewer="actual reviewer", reviewed_at="2026-10-18")
    for result in (waive(data), build_task_requirement_lists(data, "officeval_070")):
        assert result["review_complete"] and result["execution_eligible"]
        assert result["validation_status"] == "human_reviewed"


def test_waiver_does_not_accept_unknown_source_category():
    data = catalog()
    data["officeval_070"]["保持间距且容差0.15cm"]["source_category"] = "automatic"
    result = waive(data)
    assert not result["execution_eligible"]
    assert "保持间距且容差0.15cm" not in result["rule_sources"]
