"""L2 v0.3 comprehensive test suite covering the 5 required categories in Section 7.2:
1. Predicate parsing & evaluation (`all`, `any`, `not`, `{fact, where}`, non-docx behavior)
2. Statistical calibration under null & alternative hypotheses (curve & permutation/Fisher tests)
3. Value curve verdicts (`concentrated_turn`, `concentrated_interval`, `diffuse_loss`, `out_of_reach`, >5% invalid exclusion)
4. Information boundary enforcement (Analyst Call 1 & Call 2 inputs have zero branch-level results)
5. Idempotent continuation & CLI (`l2 run` and `l2 summarize`, reusing existing valid branches)
"""
from __future__ import annotations
import random
from pathlib import Path
import pytest

from conftest import make_l1_episode
from l2_sandbox.analysts.llm import (
    assert_no_branch_leakage,
    build_call1_input,
    build_call2_input,
)
from l2_sandbox.analysts.rules import propose_hypotheses_rules
from l2_sandbox.cli import main as cli_main
from l2_sandbox.config import load as load_config
from l2_sandbox.curve import build_curve_plan, execute_curve, infer_curve_from_samples
from l2_sandbox.evidence.l1 import L1Package
from l2_sandbox.facts.build import build_facts
from l2_sandbox.io import L2Error, read_json, read_jsonl, write_json
from l2_sandbox.predicates import (
    RULE_VOCABULARY,
    eval_predicate,
    eval_predicate_on_turn,
    validate_predicate,
)
from l2_sandbox.report import run_episode
from l2_sandbox.stats import permutation_one_sided


# ============================================================================
# 1. Predicate Evaluation & Turn-Indexed Facts (Section 2.1 & 2.2)
# ============================================================================

def test_facts_turn_indexing_tool_exit_ok_and_predicates(tmp_path):
    ep_dir = make_l1_episode(
        tmp_path / 'ep_docx',
        num_turns=6,
        is_docx=True,
        turn_actions={
            1: 'normal',
            2: 'anomaly',
            3: 'truncated_output',
            4: 'readback',
            5: 'modify_submission',
            6: 'finish',
        },
    )
    pkg = L1Package(ep_dir)
    facts = build_facts(pkg)

    # Turn 1: normal tool exit -> ok is True
    assert not eval_predicate_on_turn(RULE_VOCABULARY['tool_execution_anomaly'], facts, 1)
    t1_exits = [f for f in facts if f['turn'] == 1 and f['type'] == 'tool_exit']
    assert len(t1_exits) == 1 and t1_exits[0]['value']['ok'] is True

    # Turn 2: non-zero exit_code -> tool_exit.ok is False -> tool_execution_anomaly is True
    assert eval_predicate_on_turn(RULE_VOCABULARY['tool_execution_anomaly'], facts, 2)

    # Turn 3: exit_code=0 but output_complete=False -> tool_exit.ok is False
    assert eval_predicate_on_turn(RULE_VOCABULARY['tool_execution_anomaly'], facts, 3)

    # Turn 4: read submission without modifying submission/ -> submission_readback is True
    assert eval_predicate_on_turn(RULE_VOCABULARY['submission_readback'], facts, 4)
    assert not eval_predicate_on_turn(RULE_VOCABULARY['submission_modified'], facts, 4)

    # Turn 5: modified submission/out.docx and changed .docx structure
    assert eval_predicate_on_turn(RULE_VOCABULARY['submission_modified'], facts, 5)
    assert eval_predicate_on_turn(RULE_VOCABULARY['object_structure_changed'], facts, 5)
    assert not eval_predicate_on_turn(RULE_VOCABULARY['submission_readback'], facts, 5)

    # Turn 6: finished without tool call
    assert eval_predicate_on_turn(RULE_VOCABULARY['assistant_finished_turn'], facts, 6)

    # Branch resumed from cp000004 numbers its first turn as k + 1 = 5
    branch_dir = make_l1_episode(
        tmp_path / 'branch_k4',
        num_turns=6,
        resume_from_k=4,
        is_docx=True,
        turn_actions={5: 'readback', 6: 'finish'},
    )
    b_pkg = L1Package(branch_dir)
    assert b_pkg.start_turn == 4 and b_pkg.turns == 6
    b_facts = build_facts(b_pkg)
    assert min(f['turn'] for f in b_facts) == 5
    assert eval_predicate_on_turn(RULE_VOCABULARY['submission_readback'], b_facts, 5)


def test_non_docx_episode_naturally_omits_object_structure_change(tmp_path):
    ep_xlsx = make_l1_episode(
        tmp_path / 'ep_xlsx',
        num_turns=4,
        is_docx=False,
        turn_actions={1: 'normal', 2: 'modify_submission', 3: 'anomaly', 4: 'finish'},
    )
    facts = build_facts(L1Package(ep_xlsx))
    assert not any(f['type'] == 'object_structure_change' for f in facts)
    assert not eval_predicate_on_turn(RULE_VOCABULARY['object_structure_changed'], facts, 2)

    # Rule analyst on turn 2 proposes submission_modified, never object_structure_changed
    hyps = propose_hypotheses_rules(facts, {'verdict': 'concentrated_turn', 'turn': 2})
    assert len(hyps) == 1
    assert hyps[0]['predicate'] == RULE_VOCABULARY['submission_modified']


def test_predicate_dsl_combinators_and_validation(tmp_path):
    turn_facts = [
        {'turn': 1, 'type': 'tool_requested', 'value': {'name': 'run', 'arguments': {'command': 'python fix_table.py'}}},
        {'turn': 1, 'type': 'tool_exit', 'value': {'ok': False, 'exit_code': 2}},
    ]
    complex_pred = {
        'all': [
            {'any': [
                {'fact': 'tool_requested', 'where': {'arguments.command': '*fix_*.py'}},
                {'fact': 'file_transition'},
            ]},
            {'not': {'fact': 'tool_exit', 'where': {'ok': True}}},
        ]
    }
    assert eval_predicate(complex_pred, turn_facts) is True

    with pytest.raises(L2Error):
        validate_predicate({'unknown_op': []})
    with pytest.raises(L2Error):
        validate_predicate({'fact': ''})

    # Config inline secret check retained per Section 7.2
    bad_cfg = tmp_path / 'bad_cfg.json'
    write_json(bad_cfg, {'model': {'api_key': 'sk-secret-inline'}})
    with pytest.raises(L2Error, match='Inline secret'):
        load_config(bad_cfg)


# ============================================================================
# 2. Statistical Calibration under Null & Alternative Hypotheses (Section 3 & 4)
# ============================================================================

def test_statistical_calibration_under_null_and_alternative():
    rng = random.Random(20261014)
    T = 16
    grid = [0, 4, 8, 12, 16]
    N1, N2 = 12, 24
    n_trials = 350

    # (A) Uniform loss null: V(c_k) = 0.70 * (1 - k / 16)
    v_uniform = {k: 0.70 * (1.0 - k / float(T)) for k in range(T + 1)}
    interval_fp = 0
    turn_fp = 0
    for _ in range(n_trials):
        s1 = {k: [1.0 if rng.random() < v_uniform[k] else 0.0 for _ in range(N1)] for k in grid[:-1]}
        res = infer_curve_from_samples(
            T=T,
            Y=0.0,
            grid=grid,
            s1=s1,
            s2_runner_or_map=lambda s2_turns: {
                k: [1.0 if rng.random() < v_uniform[k] else 0.0 for _ in range(N2)] for k in s2_turns
            },
            alpha=0.05,
        )
        if res['verdict'] in ('concentrated_interval', 'concentrated_turn'):
            interval_fp += 1
        if res['verdict'] == 'concentrated_turn':
            turn_fp += 1

    print(f'curve calibration alpha=.05: interval FP={interval_fp}/{n_trials}, turn FP={turn_fp}/{n_trials}')
    assert interval_fp / n_trials <= 0.07, f'Uniform interval FP too high: {interval_fp / n_trials:.3f}'
    assert turn_fp / n_trials <= 0.02, f'Uniform turn FP too high: {turn_fp / n_trials:.3f}'

    # (B) Concentrated single-turn drop: Delta = 0.50 at turn j*
    turn_hits = 0
    for trial in range(n_trials):
        j_star = (trial % T) + 1
        other_drop = (0.70 - 0.50) / (T - 1)
        v_true = {0: 0.70}
        for k in range(1, T + 1):
            v_true[k] = max(0.0, v_true[k - 1] - (0.50 if k == j_star else other_drop))
        s1 = {k: [1.0 if rng.random() < v_true[k] else 0.0 for _ in range(N1)] for k in grid[:-1]}
        res = infer_curve_from_samples(
            T=T,
            Y=0.0,
            grid=grid,
            s1=s1,
            s2_runner_or_map=lambda s2_turns: {
                k: [1.0 if rng.random() < v_true[k] else 0.0 for _ in range(N2)] for k in s2_turns
            },
            alpha=0.05,
        )
        if res['verdict'] == 'concentrated_turn' and res['turn'] == j_star:
            turn_hits += 1

    print(f'curve alternative power={turn_hits}/{n_trials}')
    assert turn_hits / n_trials >= 0.65, f'Single-turn detection power too low: {turn_hits / n_trials:.3f}'

    # (C) Permutation / Fisher test false positive rate under null hypothesis (Y_P ~ Y_not_P)
    perm_fp = 0
    for _ in range(300):
        y_p = [1.0 if rng.random() < 0.5 else 0.0 for _ in range(12)]
        y_np = [1.0 if rng.random() < 0.5 else 0.0 for _ in range(12)]
        pval = permutation_one_sided(y_p, y_np, direction=-1.0)
        if pval <= 0.05 and (sum(y_p) / 12 - sum(y_np) / 12) < 0:
            perm_fp += 1
    print(f'Fisher null FP={perm_fp}/300')
    assert perm_fp / 300 <= 0.06


# ============================================================================
# 3. Value Curve Verdicts & Branch Outcome Policy (Section 3 & 3.1)
# ============================================================================

def test_curve_verdicts_and_branch_outcomes(tmp_path):
    p_scored = L1Package(make_l1_episode(tmp_path / 'b_ok', num_turns=2, terminal_case='scored', completion_rate=0.8))
    assert p_scored.outcome()['valid'] is True and p_scored.outcome()['y'] == pytest.approx(0.8)

    p_lhtb = L1Package(make_l1_episode(tmp_path / 'b_lhtb', num_turns=2, benchmark='lhtb', terminal_case='scored', passed=True))
    assert p_lhtb.outcome()['valid'] is True and p_lhtb.outcome()['y'] == 1.0

    p_empty = L1Package(make_l1_episode(tmp_path / 'b_empty', num_turns=2, terminal_case='empty_submission'))
    assert p_empty.outcome()['valid'] is True and p_empty.outcome()['y'] == 0.0

    p_non_ok = L1Package(make_l1_episode(tmp_path / 'b_non_ok', num_turns=2, terminal_case='non_ok_grade'))
    assert p_non_ok.outcome()['valid'] is True and p_non_ok.outcome()['y'] == 0.0

    p_crash = L1Package(make_l1_episode(tmp_path / 'b_crash', num_turns=2, terminal_case='scorer_crash'))
    assert p_crash.outcome()['valid'] is False and p_crash.outcome()['y'] is None

    p_exec_inv = L1Package(make_l1_episode(tmp_path / 'b_inv', num_turns=2, terminal_case='execution_invalid'))
    assert p_exec_inv.outcome()['valid'] is False and p_exec_inv.outcome()['y'] is None

    # 1) out_of_reach: V(c_0) = 0/12 -> upper bound < 0.25
    res_oor = infer_curve_from_samples(8, 0.0, [0, 4, 8], {0: [0.0] * 12, 4: [0.0] * 12}, {})
    assert res_oor['verdict'] == 'out_of_reach'

    # 2) concentrated_turn: drop from 1.0 at k=2 to 0.0 at k=3 (turn 3)
    res_turn = infer_curve_from_samples(
        8, 0.0, [0, 4, 8],
        {0: [1.0] * 12, 4: [0.0] * 12},
        {0: [1.0] * 24, 1: [1.0] * 24, 2: [1.0] * 24, 3: [0.0] * 24, 4: [0.0] * 24},
    )
    assert res_turn['verdict'] == 'concentrated_turn' and res_turn['turn'] == 3

    # 3) concentrated_interval: interval [0, 4] drops from 1.0 to 0.0 across 4 equal 0.25 steps
    res_int = infer_curve_from_samples(
        16, 0.0, [0, 4, 8, 12, 16],
        {0: [1.0] * 12, 4: [0.0] * 12, 8: [0.0] * 12, 12: [0.0] * 12},
        {
            0: [1.0] * 24,
            1: [1.0] * 18 + [0.0] * 6,
            2: [1.0] * 12 + [0.0] * 12,
            3: [1.0] * 6 + [0.0] * 18,
            4: [0.0] * 24,
        },
    )
    assert res_int['verdict'] == 'concentrated_interval' and res_int['interval'] == [0, 4] and res_int['turn'] is None

    # 4) diffuse_loss: V(c_0)=0.75, gradual decline across the trajectory
    res_diff = infer_curve_from_samples(
        8, 0.0, [0, 4, 8],
        {0: [1.0] * 9 + [0.0] * 3, 4: [1.0] * 5 + [0.0] * 7},
        {k: [1.0] * (18 - 2 * k) + [0.0] * (6 + 2 * k) for k in range(5)},
    )
    assert res_diff['verdict'] == 'diffuse_loss'


# ============================================================================
# 4. Information Boundary Enforcement (Section 2 & 5)
# ============================================================================

def test_analyst_information_boundary(tmp_path):
    parent_dir = make_l1_episode(tmp_path / 'parent_ib', num_turns=4, turn_actions={2: 'anomaly'})
    parent = L1Package(parent_dir)
    facts = build_facts(parent)

    c1_in = build_call1_input(parent, p0=0.6, facts=facts, include_facts=True)
    assert 'trajectory' in c1_in and 'facts' in c1_in
    assert 'branches' not in c1_in and 'outcomes' not in c1_in

    c1_raw = build_call1_input(parent, p0=0.6, facts=facts, include_facts=False)
    assert 'facts' not in c1_raw

    curve_mock = {
        'verdict': 'concentrated_turn',
        'turn': 2,
        'interval': [0, 4],
        'v_hat': {'0': 0.8, '1': 0.8, '2': 0.1, '3': 0.1, '4': 0.0},
        'turn_drops': {'1': 0.0, '2': 0.7, '3': 0.0, '4': 0.1},
        'outcomes': [{'trial': 'should_be_stripped', 'y': 1.0}],
    }
    c2_in = build_call2_input(parent, p0=0.6, curve_summary=curve_mock, facts=facts, include_facts=True)
    assert 'outcomes' not in c2_in['curve_summary']

    with pytest.raises(L2Error, match='Information boundary violation'):
        assert_no_branch_leakage({'safe': 1, 'nested': {'outcomes': []}})


# ============================================================================
# 5. Idempotent Continuation, Hypothesis Testing & CLI (`l2 run` / `l2 summarize`)
# ============================================================================

def test_idempotent_run_and_hypothesis_testing_end_to_end(tmp_path):
    parent_dir = make_l1_episode(
        tmp_path / 'parent_e2e',
        num_turns=4,
        benchmark='officeval',
        task_id='officeval_003',
        is_docx=True,
        turn_actions={1: 'normal', 2: 'anomaly', 3: 'modify_submission', 4: 'finish'},
        terminal_case='failed_zero',
    )
    out_dir = tmp_path / 'l2_out'

    # Step 1: Dry run without --execute (only writes facts.jsonl, values/, plan.json, report.md)
    dry_res = run_episode(parent_dir, output_dir=out_dir, analysts=['rules'], execute=False, p0=0.75, n1=8, n2=16, stride=2, max_concurrency=2)
    assert dry_res['executed'] is False
    assert (out_dir / 'facts.jsonl').is_file()
    assert (out_dir / 'plan.json').is_file()
    assert (out_dir / 'values/rules_1.json').is_file()
    assert not (out_dir / 'outcomes.jsonl').exists()

    # Step 2: Execute with mock resume_fn where drop is concentrated at turn 2 (between k=1 and k=2):
    # At k=0: all 16 branches succeed (V(c_0)=1.0)
    # At k=1 (resampling turn 2): 5 branches do 'anomaly' (y=0.0) and 11 do 'readback' (y=1.0), so V(c_1)=11/16=0.6875
    # At k=2: all 16 branches fail (V(c_2)=0.0), so d_1 = 0.3125 and d_2 = 0.6875 > d_1 -> argmax turn is j = 2!
    resume_calls = {'count': 0}

    def mock_resume(parent_pkg: L1Package, k: int, branch_dir: Path) -> None:
        resume_calls['count'] += 1
        trial_name = branch_dir.name
        slot = int(trial_name.split('-i')[1].split('-')[0])
        if k == 0:
            succ = True
            act = 'normal'
        elif k == 1:
            succ = (slot >= 5)
            act = 'readback' if succ else 'anomaly'
        else:
            succ = False
            act = 'normal'
        make_l1_episode(
            branch_dir,
            num_turns=4,
            resume_from_k=k,
            parent_pkg=parent_pkg,
            is_docx=True,
            turn_actions={k + 1: act},
            terminal_case='scored' if succ else 'failed_zero',
            completion_rate=1.0 if succ else 0.0,
        )

    def mock_llm_caller(sys_prompt: str, payload: dict) -> dict:
        assert_no_branch_leakage(payload)
        if 'curve_summary' not in payload:
            return {'values': [0.95, 0.70, 0.02, 0.01]}
        return {
            'hypotheses': [
                {
                    'turn': 2,
                    'predicate': RULE_VOCABULARY['tool_execution_anomaly'],
                    'effect': -0.8,
                    'rationale': 'Shell anomaly in turn 2 drops completion rate.',
                },
                {
                    'turn': 2,
                    'predicate': RULE_VOCABULARY['assistant_finished_turn'],
                    'effect': -0.5,
                    'rationale': 'Inconsistent hypothesis (did not happen on parent turn 2).',
                },
            ]
        }

    exec_res = run_episode(
        parent_dir,
        output_dir=out_dir,
        analysts=['rules', 'llm'],
        execute=True,
        p0=0.75,
        n1=8,
        n2=16,
        stride=2,
        max_concurrency=2,
        resume_fn=mock_resume,
        llm_caller=mock_llm_caller,
    )
    first_call_count = resume_calls['count']
    assert first_call_count > 0
    assert exec_res['curve']['verdict'] in ('concentrated_turn', 'concentrated_interval')
    assert exec_res['curve']['candidate_turn'] == 2

    scores = exec_res['scores']
    rules_ht = scores['analysts']['rules@1']['hypothesis_testing']
    assert rules_ht['confirmed_count'] == 1
    assert rules_ht['explanation_covered'] is True
    confirmed_h = rules_ht['hypotheses'][0]
    assert confirmed_h['status'] == 'confirmed'
    assert confirmed_h['frequency'] == pytest.approx(5.0 / 16.0)
    assert confirmed_h['effect'] == pytest.approx(-1.0)
    assert confirmed_h['explained_share'] == pytest.approx(1.0)

    # Check LLM hypotheses: first is confirmed, second is inconsistent (since parent turn 2 called a tool)
    llm_key = next(k for k in scores['analysts'] if k.startswith('llm:'))
    llm_hyps = scores['analysts'][llm_key]['hypothesis_testing']['hypotheses']
    assert llm_hyps[0]['status'] == 'confirmed'
    assert llm_hyps[1]['status'] == 'inconsistent'

    first_outcomes = read_jsonl(out_dir / 'outcomes.jsonl')
    # Step 3: Idempotent re-run! All existing valid branches in branches/<trial>/ must be skipped!
    rerun_res = run_episode(
        parent_dir,
        output_dir=out_dir,
        analysts=['rules', 'llm'],
        execute=True,
        p0=0.75,
        n1=8,
        n2=16,
        stride=2,
        max_concurrency=2,
        resume_fn=mock_resume,
        llm_caller=mock_llm_caller,
    )
    assert resume_calls['count'] == first_call_count, 'Idempotent re-run must not re-execute existing valid branches!'
    outcomes = read_jsonl(out_dir / 'outcomes.jsonl')
    assert outcomes == first_outcomes, 'Reuse must preserve the original attempt ledger and outcome bindings'
    assert rerun_res['scores']['analysts']['rules@1'] == scores['analysts']['rules@1']

    # Step 4: Test `l2 summarize` across 2 task families via CLI
    parent_dir2 = make_l1_episode(
        tmp_path / 'parent_fam2',
        num_turns=4,
        benchmark='officeval',
        task_id='officeval_033',
        is_docx=True,
        turn_actions={1: 'normal', 2: 'anomaly', 3: 'modify_submission', 4: 'finish'},
        terminal_case='failed_zero',
    )
    out_dir2 = tmp_path / 'l2_out_fam2'
    run_episode(
        parent_dir2,
        output_dir=out_dir2,
        analysts=['rules', 'llm'],
        execute=True,
        p0=0.75,
        n1=8,
        n2=16,
        stride=2,
        max_concurrency=2,
        resume_fn=mock_resume,
        llm_caller=mock_llm_caller,
    )
    summary_file = tmp_path / 'summary.json'
    rc = cli_main(['summarize', str(out_dir), str(out_dir2), '--output', str(summary_file)])
    assert rc == 0
    summary = read_json(summary_file)
    assert len(summary['included_episodes']) == 2
    assert summary['analysts']['rules@1']['skill_score']['family_count'] == 2
    assert summary['vocabulary_mantel_haenszel']['tool_execution_anomaly']['status'] == 'evaluated'


def test_invalid_branch_rerun_and_trajectory_exclusion(tmp_path):
    parent_dir = make_l1_episode(tmp_path / 'parent_inv', num_turns=4, terminal_case='failed_zero')
    parent = L1Package(parent_dir)
    plan = build_curve_plan(parent, n1=4, n2=4, stride=2, max_concurrency=1)

    attempts: dict[str, int] = {}

    def flaky_resume(parent_pkg: L1Package, k: int, branch_dir: Path) -> None:
        base = branch_dir.name.split('-r')[0]
        cnt = attempts.get(base, 0)
        attempts[base] = cnt + 1
        if cnt == 0 and k == 0 and base.endswith('i00'):
            make_l1_episode(branch_dir, num_turns=4, resume_from_k=k, parent_pkg=parent_pkg, terminal_case='execution_invalid')
        else:
            make_l1_episode(branch_dir, num_turns=4, resume_from_k=k, parent_pkg=parent_pkg, terminal_case='scored', completion_rate=0.5)

    out_flaky = tmp_path / 'l2_flaky'
    write_json(out_flaky / 'plan.json', plan)
    c_flaky = execute_curve(parent, out_flaky, plan, resume_fn=flaky_resume, max_concurrency=1)
    assert c_flaky['invalid_branches'] == 1
    assert (out_flaky / 'branches/k000-s1-i00-r1/manifest.json').is_file()

    def always_bad_resume(parent_pkg: L1Package, k: int, branch_dir: Path) -> None:
        make_l1_episode(branch_dir, num_turns=4, resume_from_k=k, parent_pkg=parent_pkg, terminal_case='scorer_crash')

    out_bad = tmp_path / 'l2_bad'
    write_json(out_bad / 'plan.json', plan)
    c_bad = execute_curve(parent, out_bad, plan, resume_fn=always_bad_resume, max_concurrency=1)
    assert c_bad['excluded_high_invalid_rate'] is True
