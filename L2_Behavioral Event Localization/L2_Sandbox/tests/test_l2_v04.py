"""Comprehensive unit and regression tests for L2 v0.4 (F23-F26)."""
from __future__ import annotations
from pathlib import Path
import pytest

from conftest import make_l1_episode
from l2_sandbox.analysts import validate_hypotheses, validate_origins
from l2_sandbox.analysts.llm import CALL2_SYSTEM_PROMPT, build_call1_input, build_call2_input
from l2_sandbox.analysts.rules import build_terminal_official_view, predict_origins_rules, propose_hypotheses_rules
from l2_sandbox.config import DELIVER_REMINDER, RULE_REMINDER_TEMPLATE
from l2_sandbox.defects import build_contrast, build_defect_table
from l2_sandbox.evidence.l1 import L1Package
from l2_sandbox.facts.build import FACT_FIELD_WHITELIST, build_facts
from l2_sandbox.grades import compute_item_deductions
from l2_sandbox.io import L2Error, read_json, read_jsonl
from l2_sandbox.predicates import RULE_VOCABULARY, eval_predicate_on_turn
from l2_sandbox.report import diagnose_episode, verify_episode
from l2_sandbox.score import score_origins, summarize_runs
from l2_sandbox.stats import clustered_bootstrap, fisher_exact_one_sided, holm_adjust, lower_drop, permutation_one_sided, wilson_bounds
from l2_sandbox.verify import build_arm_intervention, evaluate_intervention, evaluate_partition, evaluate_two_point, load_branch_facts


def test_f23_compute_item_deductions_identity():
    # 1. Positive + negative items + gate deduction sum to 1 - completion_rate
    raw = {
        'status': 'ok',
        'error': None,
        'dim1_pass': True,
        'max_score': 10.0,
        'total_score': 6.0,
        'dim2_items': [
            {'rule': 'r1', 'max_delta': 5.0, 'delta': 5.0},
            {'rule': 'r2', 'max_delta': 5.0, 'delta': 2.0},
            {'rule': 'r_neg', 'max_delta': 0.0, 'delta': -1.0},
        ],
    }
    ded, m = compute_item_deductions(raw, 0.6)
    assert m == 10.0
    assert abs(sum(ded.values()) - (1.0 - 0.6)) < 1e-12
    assert abs(ded['r1'] - 0.0) < 1e-12
    assert abs(ded['r2'] - 0.3) < 1e-12
    assert abs(ded['r_neg'] - 0.1) < 1e-12

    # 2. Failed dim1_pass allocates 1 - Y to dim1_gate
    raw_fail = {'status': 'error', 'error': 'missing', 'dim1_pass': False, 'dim2_items': []}
    ded_f, _ = compute_item_deductions(raw_fail, 0.0)
    assert ded_f == {'dim1_gate': 1.0}


def test_f24_defect_table_categories_and_preview_rendered_fact(tmp_path):
    parent_dir = make_l1_episode(tmp_path / 'parent', num_turns=4, turn_actions={1: 'modify_submission', 3: 'readback'})
    parent = L1Package(parent_dir)
    # Inject a soffice preview command at turn 3
    for e in parent.events:
        if e.get('kind') == 'tool.call' and e.get('data', {}).get('call_id', '').endswith('r000002:c0'):
            e['data']['arguments']['command'] = 'soffice --headless --convert-to pdf work/out.docx && pdftoppm -png out.pdf page'
    facts = build_facts(parent)
    assert eval_predicate_on_turn(RULE_VOCABULARY['preview_rendered'], facts, 3) is True
    assert 'stdout_preview' not in FACT_FIELD_WHITELIST['tool_exit']
    assert 'stdout_preview' in CALL2_SYSTEM_PROMPT  # Explicit prohibition note

    # Build synthetic grade_rows: item r1 is undelivered (done at k>=2), r2 is broken at turn 3 (done at k=1,2, lost at k=3), r3 is never
    grade_rows = [
        {
            'kind': 'terminal_submission',
            'episode_id': parent.episode_id,
            'task_id': parent.family_id,
            'k': 4,
            'score': 0.0,
            'dim1_pass': False,
            'deductions': {'r1': 0.4, 'r2': 0.3, 'r3': 0.3},
        },
        {'kind': 'checkpoint', 'k': 0, 'candidate_status': 'ok', 'candidate': ['work/out.docx'], 'score': 0.0, 'deductions': {'r1': 0.4, 'r2': 0.3, 'r3': 0.3}},
        {'kind': 'checkpoint', 'k': 1, 'candidate_status': 'ok', 'candidate': ['work/out.docx'], 'score': 0.3, 'deductions': {'r1': 0.4, 'r2': 0.0, 'r3': 0.3}},
        {'kind': 'checkpoint', 'k': 2, 'candidate_status': 'ok', 'candidate': ['work/out.docx'], 'score': 0.7, 'deductions': {'r1': 0.0, 'r2': 0.0, 'r3': 0.3}},
        {'kind': 'checkpoint', 'k': 3, 'candidate_status': 'ok', 'candidate': ['work/out.docx'], 'score': 0.4, 'deductions': {'r1': 0.0, 'r2': 0.3, 'r3': 0.3}},
        {'kind': 'checkpoint', 'k': 4, 'candidate_status': 'ok', 'candidate': ['work/out.docx'], 'score': 0.4, 'deductions': {'r1': 0.0, 'r2': 0.3, 'r3': 0.3}},
    ]
    dt = build_defect_table(parent, grade_rows, facts)
    by_id = {d['defect']: d for d in dt['defects']}
    assert by_id['r1']['kind'] == 'undelivered'
    assert by_id['r1']['turn'] == 2
    assert by_id['r2']['kind'] == 'broken'
    assert by_id['r2']['turn'] == 3
    assert by_id['r3']['kind'] == 'never'
    assert by_id['r3']['turn'] is None
    assert abs(dt['total_deduction'] - 1.0) < 1e-9


def test_f24_call1_and_call2_information_boundary(tmp_path):
    parent = L1Package(make_l1_episode(tmp_path / 'parent', num_turns=3))
    facts = build_facts(parent)
    term_view = build_terminal_official_view(parent, facts)
    c1 = build_call1_input(parent, term_view, facts, include_facts=True)
    assert 'defects_table' not in c1
    with pytest.raises(L2Error, match='Information boundary violation'):
        build_call1_input(parent, {'defects_table': {}}, facts, include_facts=True)
    with pytest.raises(L2Error, match='Information boundary violation'):
        build_call2_input(parent, term_view, {'outcomes': []}, facts, include_facts=True)


def test_f25_d8_96_branch_regression():
    d8_root = Path(__file__).resolve().parents[1] / 'validation' / 'd8_confirmatory_20261016'
    dev45_root = Path(__file__).resolve().parents[3] / 'L1_sandbox' / 'C3_Experimental_Sandbox' / 'runs' / 'deepseek_dev45_20261015'
    if not d8_root.is_dir() or not dev45_root.is_dir():
        pytest.skip('D8 confirmatory or dev45 artifacts not present')

    d8_070 = d8_root / 'officeval_070.requests_035.r04'
    d8_060 = d8_root / 'officeval_060.requests_035.r03'
    p_070 = L1Package(dev45_root / 'runs' / 'officeval_070.requests_035.r04' / 'attempt_000' / 'episode')

    out_070 = read_jsonl(d8_070 / 'outcomes.jsonl')
    tp_070 = evaluate_two_point(out_070, 28, 29)
    assert tp_070['confirmed'] is True
    assert abs(tp_070['lower_drop_95'] - 0.19057533508440194) < 1e-12
    assert abs(tp_070['p_value'] - 0.0001999600079984003) < 1e-12

    out_060 = read_jsonl(d8_060 / 'outcomes.jsonl')
    tp_060 = evaluate_two_point(out_060, 28, 29)
    assert tp_060['confirmed'] is False
    assert tp_060['lower_drop_95'] < 0.0

    # Group B H2 partition check on 070.r04 (0 branches rerun)
    parent_facts = build_facts(p_070)
    branch_facts = load_branch_facts(d8_070, out_070)
    b_hyps_doc = read_json(d8_070 / 'hypotheses' / 'llm_gemini-3.8-flash_8ca2e57d160d6772.json')
    part = evaluate_partition(b_hyps_doc['hypotheses'], parent_facts, branch_facts, out_070)
    assert part['confirmed_count'] == 1
    assert abs(part['hypotheses'][1]['p_value_holm'] - 0.04459108178364327) < 1e-12


def test_f25_intervention_arms_and_evaluation():
    deliv = build_arm_intervention('deliver', 29)
    assert deliv == {'type': 'user_message', 'at_request': 30, 'message': DELIVER_REMINDER}
    rule_intv = build_arm_intervention('rule', 29, requirement='Must use SimHei 16pt')
    assert rule_intv == {
        'type': 'user_message',
        'at_request': 30,
        'message': RULE_REMINDER_TEMPLATE.format(rule='Must use SimHei 16pt', requirement='Must use SimHei 16pt'),
    }

    outcomes = [
        *[{'trial': f'plain_k029_i{i:02d}', 'arm': 'plain', 'k': 29, 'valid': True, 'y': 0.1} for i in range(12)],
        *[{'trial': f'deliver_k029_i{i:02d}', 'arm': 'deliver', 'k': 29, 'valid': True, 'y': 0.8} for i in range(12)],
    ]
    res = evaluate_intervention(
        [{'verification': 'intervention', 'k': 29, 'treatment_arm': 'deliver', 'control_arm': 'plain', 'defect': 'total'}],
        outcomes,
    )
    assert res['confirmed_count'] == 1
    assert res['hypotheses'][0]['status'] == 'confirmed'
    assert res['hypotheses'][0]['evidence_level'] == 'E2'
    assert abs(res['hypotheses'][0]['effect'] - 0.7) < 1e-9
