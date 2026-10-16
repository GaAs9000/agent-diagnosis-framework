"""The shared analyst contract must cover fields produced by the real extractor (L2 v0.4)."""
import re

from conftest import docx_bytes, make_l1_episode
from l2_sandbox.analysts.llm import (
    CALL2_SYSTEM_PROMPT, predict_origins_llm, propose_hypotheses_llm,
)
from l2_sandbox.evidence.l1 import L1Package
from l2_sandbox.facts.build import DEFAULT_PROBE_CONFIG, build_facts
from l2_sandbox.facts.docx_probe import probe
from l2_sandbox.predicates import eval_predicate


def scalar_paths(value, prefix=''):
    for key, item in value.items():
        path = f'{prefix}.{key}' if prefix else key
        if isinstance(item, dict):
            yield from scalar_paths(item, path)
        elif not isinstance(item, list):
            yield path, item


def contract_sections():
    return dict(re.findall(r'^- `([^`]+)`: (.*)$', CALL2_SYSTEM_PROMPT, re.MULTILINE))


def test_real_generated_facts_have_documented_predicate_fields(tmp_path):
    parent = L1Package(make_l1_episode(
        tmp_path / 'parent', num_turns=2, turn_actions={1: 'modify_submission'},
    ))
    parent.events.extend([
        {'kind': 'tool.rejected', 'data': {'call_id': 'rejected', 'reason': 'protocol_error'}},
        {'kind': 'episode.error', 'data': {'error_type': 'RuntimeError', 'code': 'runtime_error', 'message': 'failed'}},
        {'kind': 'model.error', 'data': {'request_id': 'request', 'reason': 'wall_time_budget',
                                        'stop_phase': 'retry_backoff', 'transport_attempts': 2}},
        {'kind': 'episode.finished', 'data': {'status': {
            'termination': 'assistant_finished', 'protocol_validity': 'valid',
            'execution_validity': 'valid', 'evaluation_validity': 'not_run',
            'task_outcome': 'unknown', 'evidence_completeness': 'complete',
        }}},
    ])
    facts = build_facts(parent)
    sections = contract_sections()
    assert {fact['type'] for fact in facts}.issubset(set(sections))
    assert set(sections) == {
        'tool_requested', 'preview_rendered', 'tool_exit', 'file_transition', 'object_structure_change',
        'model_output_text', 'protocol_error', 'execution_error', 'termination',
    }
    for fact in facts:
        section = sections[fact['type']]
        if fact['type'] == 'execution_error':
            section += sections['protocol_error']
        for path, value in scalar_paths(fact['value']):
            assert f'`{path}`' in section, (fact['type'], path)
            assert eval_predicate({'fact': fact['type'], 'where': {path: value}}, [fact])
        for path in ('turn', 'type'):
            assert f'`{path}`' in CALL2_SYSTEM_PROMPT
            assert eval_predicate({'fact': fact['type'], 'where': {path: fact[path]}}, [fact])

    structure = next(f for f in facts if f['type'] == 'object_structure_change')
    for side, empty_row in [('before', True), ('after', False)]:
        parsed = probe(docx_bytes(empty_row=empty_row, black=not empty_row), DEFAULT_PROBE_CONFIG)
        assert structure['value'][f'{side}_summary'] == parsed['summary']
        for key, value in parsed['summary'].items():
            assert f'`{side}_summary.{key}`' in sections['object_structure_change']
            assert type(value) is (bool if key.endswith('judgements') else int)


def test_raw_and_facts_analysts_share_system_prompts_and_other_inputs(tmp_path):
    parent = L1Package(make_l1_episode(
        tmp_path / 'parent', num_turns=2, turn_actions={1: 'modify_submission'},
    ))
    facts = build_facts(parent)
    term_view = {
        'delivered_valid': False,
        'terminal_score_y': 0.0,
        'zero_reason': 'undelivered_or_invalid',
        'target_exts': ['.docx'],
        'target_defects': [{'defect': 'dim1_gate', 'max_delta': 1.0, 'deduction': None}],
    }
    defects_table = {
        'episode_id': parent.episode_id,
        'defects': [{'defect': 'dim1_gate', 'deduction': 1.0, 'kind': 'undelivered', 'turn': 1}],
    }
    contrast = {'episode_id': parent.episode_id, 'by_defect': {}}
    calls = []

    def caller(system, payload):
        calls.append((system, payload))
        return {
            'origins': [{'defect': 'dim1_gate', 'kind': 'undelivered', 'turn': 1, 'rationale': 'ok'}],
            'hypotheses': [],
        }

    for include_facts in (False, True):
        predict_origins_llm(parent, term_view, {}, facts, contrast, include_facts=include_facts, llm_caller=caller)
        propose_hypotheses_llm(parent, term_view, defects_table, {}, facts, contrast, include_facts=include_facts, llm_caller=caller)
    for raw, assisted in ((calls[0], calls[2]), (calls[1], calls[3])):
        assert raw[0] == assisted[0]
        assert 'facts' not in raw[1]
        assert 'contrast' not in raw[1]
        assert assisted[1]['facts'] == facts
        assert assisted[1]['contrast'] == contrast
        assert raw[1] == {key: value for key, value in assisted[1].items() if key not in ('facts', 'contrast')}
    assert calls[1][0] == CALL2_SYSTEM_PROMPT
