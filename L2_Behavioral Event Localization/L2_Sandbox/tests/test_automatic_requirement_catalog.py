"""Offline contract checks for the explicitly unreviewed DEV9 catalog."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/build_automatic_requirement_catalog.py'
spec = importlib.util.spec_from_file_location('automatic_catalog', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture(scope='module')
def catalog():
    return module.build_catalog()


def test_full_dev9_rubric_coverage_and_grade_aliases(catalog):
    assert set(catalog) == {'metadata', *module.DEV9}
    coverage = catalog['metadata']['coverage']
    assert sum(row['rubric_row_count'] for row in coverage.values()) == 188
    assert all(item['covered'] for row in coverage.values() for item in row['rubric_rows'])
    assert catalog['metadata']['grade_alias_validation_complete']
    assert all(not row['observed_grade_alias_validation']['uncovered'] for row in coverage.values())
    # Guard against a misleading vacuous pass if fixture locations change.
    assert sum(len(row['observed_grade_alias_validation']['observed_distinct_aliases']) for row in coverage.values()) > 100


def test_grade_records_cannot_choose_requirements(catalog):
    without_validation = module.build_catalog(validate_grades=False)
    assert without_validation['metadata']['construction_sha256'] == catalog['metadata']['construction_sha256']
    assert {t: without_validation[t] for t in module.DEV9} == {t: catalog[t] for t in module.DEV9}
    assert not catalog['metadata']['construction_uses_outcomes']


def test_unreviewed_stays_unreviewed_and_public_text_is_original(catalog):
    metadata = catalog['metadata']
    assert metadata['status'] == 'automatic_unreviewed'
    prior = metadata['prior_human_approvals']['record']
    assert prior['user_answer'] == '批准这10条的分类与拆分口径'
    assert len(prior['rows']) == 10
    for task in module.DEV9:
        original = json.loads((module.ROOT / module.L1 / f'benchmarks/officeval/data/tasks/{task}.json').read_text())['instruction'].strip()
        for rule, info in catalog[task].items():
            assert info['source_citation'] and info['public_instruction_citation']['quote'] == original
            if info['status'] == 'automatic_unreviewed':
                assert info['source_category'] == 'ambiguous'
                assert info['visible_requirement'] == original
                assert info['exact_verifier_specification'] == rule
                assert info['reviewer'] == info['reviewed_at'] == ''
                assert not info['review_complete']
            else:
                assert info['status'] == 'previously_human_approved'
                assert info['reviewer'] == prior['reviewer']
                assert info['reviewed_at'] == prior['recorded_at_utc']
                assert info['approval_reference']['sample_id'] in {r['sample_id'] for r in prior['rows']}
                assert info['review_complete']


def test_static_extraction_does_not_execute_verifier_code():
    source = '''
raise RuntimeError("must never execute")
DIMENSION2_RULES: list = [("check", 3, "公开图形尺寸必须1.00厘米")]
RULES = [("+5", malicious(), "隐藏评分要求为2.00厘米")]
self.item(1, "框线宽度0.5pt", True)
x = "+5 条件评分规则 ✓"
'''
    aliases = module.verifier_aliases(source)
    for expected in ('公开图形尺寸必须1.00厘米', '隐藏评分要求为2.00厘米', '框线宽度0.5pt', '条件评分规则'):
        assert expected in aliases
        assert all(c['line_start'] > 0 and c['quote'] for c in aliases[expected])


def test_catalog_is_execution_eligible_only_under_explicit_waiver(catalog):
    from l2_sandbox.v05 import build_task_requirement_lists
    waiver = {'human_review_policy': 'waived_by_user', 'authorization_source': 'direct_user_message',
              'exact_text': '跳过人审核', 'scope': ['requirement_catalog_human_review'],
              'source': 'direct user message in current conversation'}
    for task in module.DEV9:
        strict = build_task_requirement_lists(catalog, task)
        allowed = build_task_requirement_lists(catalog, task, human_review_policy='waived_by_user', review_authorization=waiver)
        assert not strict['execution_eligible']
        assert not strict['review_complete']
        assert allowed['execution_eligible'], allowed['execution_errors']
        assert not allowed['review_complete']
        original = json.loads((module.ROOT / module.L1 / f'benchmarks/officeval/data/tasks/{task}.json').read_text())['instruction'].strip()
        assert allowed['explicit'].count(original) == 1  # unreviewed whole instruction deduplicates
        assert allowed['implicit']


def test_prior_approvals_are_applied_only_at_exact_saved_identity(catalog):
    mappings = catalog['metadata']['prior_human_approvals']['identity_mappings']
    assert len(mappings) == 10
    assert all(row['mapped_rules'] and not row['unmapped'] for row in mappings)
    assert catalog['officeval_003']['文档表格中没有空白行']['source_category'] == 'explicit'
    info = catalog['officeval_003']['文档表格中所有横线：相邻两条横线之间的距离几乎相等，只允许有0.15cm的误差']
    assert info['source_category'] == 'ambiguous'
    assert info['visible_requirement'] == '横线样式和图片中的一样'
    assert '0.15cm' in info['exact_verifier_specification']
    assert catalog['metadata']['effective_review_counts']['human_approval_decisions'] == 10
