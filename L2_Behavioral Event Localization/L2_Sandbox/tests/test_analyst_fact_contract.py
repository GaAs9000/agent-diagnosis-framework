"""The shared analyst contract must cover fields produced by the real extractor."""
import re

from conftest import docx_bytes, make_l1_episode
from l2_sandbox.analysts.llm import (
    CALL2_SYSTEM_PROMPT, predict_values_llm, propose_hypotheses_llm,
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
    # The fixture supplies actual DOCX blobs, snapshots and tool results. Add
    # native error/finish shapes in memory to exercise the remaining emitters.
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
    assert {fact['type'] for fact in facts} == set(sections) == {
        'tool_requested', 'tool_exit', 'file_transition', 'object_structure_change',
        'model_output_text', 'protocol_error', 'execution_error', 'termination',
    }
    for fact in facts:
        section = sections[fact['type']]
        if fact['type'] == 'execution_error':
            section += sections['protocol_error']  # Explicitly shared error fields.
        for path, value in scalar_paths(fact['value']):
            assert f'`{path}`' in section, (fact['type'], path)
            # These are actual legal where paths, not unused probe metadata.
            assert eval_predicate({'fact': fact['type'], 'where': {path: value}}, [fact])
        for path in ('turn', 'type'):
            assert f'`{path}`' in CALL2_SYSTEM_PROMPT
            assert eval_predicate({'fact': fact['type'], 'where': {path: fact[path]}}, [fact])

    structure = next(f for f in facts if f['type'] == 'object_structure_change')
    for side, empty_row in [('before', True), ('after', False)]:
        parsed = probe(docx_bytes(empty_row=empty_row, black=not empty_row), DEFAULT_PROBE_CONFIG)
        assert structure['value'][f'{side}_summary'] == parsed['summary']
        # Derive the paths from the real probe, so a new summary key must also
        # be made available to the raw analyst, even if old key names remain.
        for key, value in parsed['summary'].items():
            assert f'`{side}_summary.{key}`' in sections['object_structure_change']
            assert type(value) is (bool if key.endswith('judgements') else int)


def test_raw_and_facts_analysts_share_system_prompts_and_other_inputs(tmp_path):
    parent = L1Package(make_l1_episode(
        tmp_path / 'parent', num_turns=2, turn_actions={1: 'modify_submission'},
    ))
    facts = build_facts(parent)
    curve = {'verdict': 'concentrated_turn', 'turn': 1, 'v_hat': {'0': 0.8, '1': 0.2}}
    calls = []

    def caller(system, payload):
        calls.append((system, payload))
        return {'values': [0.8, 0.2], 'hypotheses': []}

    for include_facts in (False, True):
        predict_values_llm(parent, 0.5, {}, facts, include_facts=include_facts, llm_caller=caller)
        propose_hypotheses_llm(parent, 0.5, curve, {}, facts, include_facts=include_facts, llm_caller=caller)
    for raw, assisted in ((calls[0], calls[2]), (calls[1], calls[3])):
        assert raw[0] == assisted[0]
        assert 'facts' not in raw[1]
        assert assisted[1]['facts'] == facts
        assert raw[1] == {key: value for key, value in assisted[1].items() if key != 'facts'}
    assert calls[1][0] == CALL2_SYSTEM_PROMPT
