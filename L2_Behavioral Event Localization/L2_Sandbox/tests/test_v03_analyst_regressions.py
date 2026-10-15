from pathlib import Path
import pytest
from conftest import make_l1_episode
from l2_sandbox import report
from l2_sandbox.io import L2Error, read_json, write_json
from l2_sandbox.evidence.l1 import L1Package
from l2_sandbox.analysts.llm import predict_values_llm, propose_hypotheses_llm, analyst_tag
from l2_sandbox.analysts.rules import predict_values_reference


def fake_curve(parent, output, plan, **kwargs):
    curve = {'verdict': 'diffuse_loss', 'v_hat': {}, 'turn_drops': {}}
    write_json(output / 'curve.json', curve)
    return curve


def fake_score(*args, **kwargs):
    return {'analysts': {}, 'active': kwargs['active_analysts'], 'bindings': kwargs['expected_input_bindings']}


def test_dryrun_paid_calls_zero_and_rules_required(tmp_path):
    parent = make_l1_episode(tmp_path / 'parent', num_turns=2)
    output = tmp_path / 'out'
    calls = []
    def caller(*args):
        calls.append(args)
        return {'values': [0.5, 0.5]}
    for specs in [('llm',), ('llm_raw',)]:
        result = report.run_episode(parent, p0=0.5, output_dir=output, analysts=specs, llm_caller=caller)
        assert 'rules@1' in result['values']
        assert result['deferred_analysts']
    assert calls == []
    # Even a corrupt cache cannot authorize a call.
    tag = analyst_tag(report.load_config()['model'])
    write_json(output / 'values' / report._safe_filename(tag), {'values': [2]})
    report.run_episode(parent, p0=0.5, output_dir=output, analysts=('llm',), llm_caller=caller)
    assert calls == []


def test_config_and_reference_bindings_change(tmp_path, monkeypatch):
    monkeypatch.setattr(report, 'execute_curve', fake_curve)
    monkeypatch.setattr(report, 'score_episode', fake_score)
    parent = make_l1_episode(tmp_path / 'parent', num_turns=2)
    cfg = tmp_path / 'cfg.json'
    calls = []
    def caller(*args):
        calls.append(args)
        return {'values': [0.4, 0.3]}
    write_json(cfg, {'model': {'parameters': {'temperature': 0.1}}})
    first = report.run_episode(parent, p0=0.5, output_dir=tmp_path/'out', execute=True, analysts=(f'llm:{cfg}', 'reference'), reference_snapshot_scores={0: 0.2, 1: 0.1}, llm_caller=caller)
    second = report.run_episode(parent, p0=0.5, output_dir=tmp_path/'out', execute=True, analysts=(f'llm:{cfg}', 'reference'), reference_snapshot_scores={0: 0.8, 1: 0.7}, llm_caller=caller)
    assert len(calls) == 1
    assert first['scores']['bindings']['values']['reference@1'] != second['scores']['bindings']['values']['reference@1']
    write_json(cfg, {'model': {'parameters': {'temperature': 0.9}}})
    third = report.run_episode(parent, p0=0.5, output_dir=tmp_path/'out', execute=True, analysts=(f'llm:{cfg}',), llm_caller=caller)
    assert len(calls) == 2
    assert len([a for a in third['scores']['active'] if a.startswith('llm:')]) == 1
    assert not (tmp_path/'out/history').exists()


def test_output_overlap_and_plan_identity_reject_before_write(tmp_path):
    parent = make_l1_episode(tmp_path / 'parent', num_turns=2)
    for output in (parent, parent/'nested', tmp_path):
        with pytest.raises(L2Error):
            report.run_episode(parent, p0=0.5, output_dir=output)
    assert not (parent/'facts.jsonl').exists()
    output = tmp_path/'out'
    report.run_episode(parent, p0=0.5, output_dir=output)
    before = (output/'facts.jsonl').read_bytes()
    other = make_l1_episode(tmp_path/'other', num_turns=2)
    with pytest.raises(L2Error):
        report.run_episode(other, p0=0.5, output_dir=output)
    assert (output/'facts.jsonl').read_bytes() == before


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -0.1, 1.1, True, '0.2'])
def test_invalid_values_are_rejected(tmp_path, value):
    parent = L1Package(make_l1_episode(tmp_path/'parent', num_turns=2))
    with pytest.raises(L2Error):
        predict_values_llm(parent, 0.5, {}, llm_caller=lambda *args: {'values': [value, 0.5]})


@pytest.mark.parametrize('hyp', [None, {'turn': '1', 'predicate': {'fact': 'tool_exit'}, 'effect': -0.1, 'rationale': ''}, {'turn': 1, 'predicate': {'fact': 'tool_exit'}, 'effect': float('nan'), 'rationale': ''}, {'turn': 2, 'predicate': {'fact': 'tool_exit'}, 'effect': -0.1, 'rationale': ''}])
def test_invalid_hypotheses_are_rejected(tmp_path, hyp):
    parent = L1Package(make_l1_episode(tmp_path/'parent', num_turns=2))
    with pytest.raises(L2Error):
        propose_hypotheses_llm(parent, 0.5, {'verdict':'concentrated_turn', 'turn':1}, {}, llm_caller=lambda *args: {'hypotheses':[hyp]})


def test_reference_unavailable_without_scores():
    with pytest.raises(L2Error, match='unavailable'):
        predict_values_reference([], 2, 0.5, 0.0)


def test_call2_prompt_documents_fact_schema_and_rules_interval_fallback():
    from l2_sandbox.analysts.llm import CALL2_SYSTEM_PROMPT
    from l2_sandbox.analysts.rules import propose_hypotheses_rules
    from l2_sandbox.predicates import RULE_VOCABULARY

    # F12: Group B/C prompt explicitly lists the v0.3 fact types and key where fields
    for token in (
        'tool_requested', 'tool_exit', 'file_transition', 'object_structure_change',
        'model_output_text', 'protocol_error', 'execution_error', 'termination',
        'arguments.command', 'ok', 'path', 'submission/*',
    ):
        assert token in CALL2_SYSTEM_PROMPT

    # F13: On concentrated_interval, if candidate_turn has no vocabulary match on parent,
    # rules analyst falls back to the highest-drop turn in (a, b] with a match.
    facts = [
        {'turn': 1, 'type': 'tool_requested', 'value': {'name': 'run', 'arguments': {'command': 'ls'}}},
        {'turn': 1, 'type': 'tool_exit', 'value': {'ok': True, 'exit_code': 0}},
        {'turn': 2, 'type': 'tool_requested', 'value': {'name': 'run', 'arguments': {'command': 'python bad.py'}}},
        {'turn': 2, 'type': 'tool_exit', 'value': {'ok': False, 'exit_code': 1}},
        {'turn': 3, 'type': 'tool_requested', 'value': {'name': 'run', 'arguments': {'command': 'cat log.txt'}}},
        {'turn': 3, 'type': 'tool_exit', 'value': {'ok': True, 'exit_code': 0}},
    ]
    curve_interval = {
        'verdict': 'concentrated_interval',
        'turn': None,
        'interval': [0, 3],
        'candidate_turn': 3,
        'turn_drops': {'1': 0.05, '2': 0.25, '3': 0.30},
    }
    hyps = propose_hypotheses_rules(facts, curve_interval)
    assert len(hyps) == 1
    assert hyps[0]['turn'] == 2
    assert hyps[0]['predicate'] == RULE_VOCABULARY['tool_execution_anomaly']
    # On concentrated_turn, only the localized turn itself is inspected
    curve_turn = {**curve_interval, 'verdict': 'concentrated_turn', 'turn': 3}
    assert propose_hypotheses_rules(facts, curve_turn) == []

