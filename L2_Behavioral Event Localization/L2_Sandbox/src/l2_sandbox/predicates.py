"""Predicate DSL parser, validator, evaluator, and rule vocabulary (L2 v0.3 Section 2.2)."""
from __future__ import annotations
from fnmatch import fnmatchcase
from typing import Any
from .io import L2Error

RULE_VOCABULARY: dict[str, dict[str, Any]] = {
    'tool_execution_anomaly': {'fact': 'tool_exit', 'where': {'ok': False}},
    'submission_modified': {'fact': 'file_transition', 'where': {'path': 'submission/*'}},
    'assistant_finished_turn': {'not': {'fact': 'tool_requested'}},
    'submission_readback': {
        'all': [
            {'fact': 'tool_requested', 'where': {'name': 'run', 'arguments.command': '*submission*'}},
            {'not': {'fact': 'file_transition', 'where': {'path': 'submission/*'}}},
        ]
    },
    'object_structure_changed': {'fact': 'object_structure_change'},
}


def validate_predicate(pred: Any) -> dict[str, Any]:
    """Validates that `pred` conforms to the v0.3 grammar: `all`, `any`, `not`, or `{fact, where?}`."""
    if not isinstance(pred, dict) or not pred:
        raise L2Error('Predicate must be a non-empty dict')
    keys = set(pred.keys())
    if keys == {'all'} or keys == {'any'}:
        key = next(iter(keys))
        items = pred[key]
        if not isinstance(items, list) or not items:
            raise L2Error(f'Predicate "{key}" must be a non-empty list')
        for item in items:
            validate_predicate(item)
        return pred
    if keys == {'not'}:
        validate_predicate(pred['not'])
        return pred
    if keys <= {'fact', 'where'} and 'fact' in keys:
        if not isinstance(pred['fact'], str) or not pred['fact']:
            raise L2Error('Atom "fact" must be a non-empty string')
        where = pred.get('where')
        if where is not None:
            if not isinstance(where, dict):
                raise L2Error('Atom "where" must be a dict')
            for k, v in where.items():
                if not isinstance(k, str) or not k:
                    raise L2Error('Atom "where" key must be a non-empty string')
                if not isinstance(v, (str, int, float, bool)) and v is not None:
                    raise L2Error(f'Atom "where" value for "{k}" must be a scalar')
        return pred
    raise L2Error(f'Invalid predicate keys: {sorted(keys)}')


def _lookup_dot_path(obj: dict[str, Any], dotted_key: str) -> tuple[bool, Any]:
    cur: Any = obj
    for part in dotted_key.split('.'):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return False, None
    return True, cur


def _match_scalar(actual: Any, expected: Any) -> bool:
    if isinstance(expected, str):
        if not isinstance(actual, str):
            return False
        return fnmatchcase(actual, expected)
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(actual) is type(expected) and actual == expected
    return actual == expected


def _match_atom(atom: dict[str, Any], turn_facts: list[dict[str, Any]]) -> bool:
    fact_type = atom['fact']
    where = atom.get('where') or {}
    for f in turn_facts:
        if f.get('type') != fact_type:
            continue
        val = f.get('value') if isinstance(f.get('value'), dict) else {}
        matched = True
        for k, expected in where.items():
            found, actual = _lookup_dot_path(val, k)
            if not found:
                found, actual = _lookup_dot_path(f, k)
            if not found or not _match_scalar(actual, expected):
                matched = False
                break
        if matched:
            return True
    return False


def eval_predicate(pred: dict[str, Any], turn_facts: list[dict[str, Any]]) -> bool:
    """Evaluates a validated predicate over the facts of a single turn."""
    validate_predicate(pred)
    if 'all' in pred:
        return all(eval_predicate(sub, turn_facts) for sub in pred['all'])
    if 'any' in pred:
        return any(eval_predicate(sub, turn_facts) for sub in pred['any'])
    if 'not' in pred:
        return not eval_predicate(pred['not'], turn_facts)
    return _match_atom(pred, turn_facts)


def eval_predicate_on_turn(pred: dict[str, Any], all_facts: list[dict[str, Any]], turn: int) -> bool:
    """Filters `all_facts` to `turn` and evaluates `pred`."""
    turn_facts = [f for f in all_facts if f.get('turn') == turn]
    return eval_predicate(pred, turn_facts)
