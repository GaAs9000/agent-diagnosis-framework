"""L2 v0.3 Analysts (Rules, Reference, and LLM)."""
from __future__ import annotations
from typing import NotRequired, TypedDict


class Values(TypedDict):
    analyst: str
    analyst_group: NotRequired[str]
    values: list[float]


class Hypothesis(TypedDict):
    turn: int
    predicate: dict
    effect: float
    rationale: str


def probability(value, name='probability') -> float:
    import math
    from ..io import InvalidOutput
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise InvalidOutput(f'{name} must be a finite number in [0, 1]')
    return float(value)


def validate_values(values, turns):
    from ..io import InvalidOutput
    if not isinstance(values, list) or len(values) != turns:
        raise InvalidOutput(f'Expected exactly {turns} values')
    return [probability(v, 'value') for v in values]


def validate_hypotheses(items, curve):
    import copy
    import math
    from ..io import InvalidOutput
    from ..predicates import validate_predicate
    if not isinstance(items, list):
        raise InvalidOutput('Hypotheses must be a list')
    if curve.get('verdict') == 'concentrated_turn':
        allowed = {curve.get('turn')}
    else:
        interval = curve.get('interval')
        if not isinstance(interval, list) or len(interval) != 2 or any(type(t) is not int for t in interval):
            raise InvalidOutput('Missing localized interval')
        allowed = set(range(interval[0] + 1, interval[1] + 1))
    clean = []
    for item in items:
        if not isinstance(item, dict) or type(item.get('turn')) is not int or item['turn'] not in allowed:
            raise InvalidOutput('Hypothesis turn must be an integer in the localized region')
        effect = item.get('effect')
        if type(effect) not in (int, float) or not math.isfinite(effect) or effect >= 0:
            raise InvalidOutput('Hypothesis effect must be finite and negative')
        if not isinstance(item.get('rationale'), str):
            raise InvalidOutput('Hypothesis rationale must be a string')
        clean.append({'turn': item['turn'], 'predicate': copy.deepcopy(validate_predicate(item.get('predicate'))),
                      'effect': float(effect), 'rationale': item['rationale']})
    return clean
