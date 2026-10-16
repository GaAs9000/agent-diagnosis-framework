"""L2 v0.4 Analysts: Origin (Call 1) and Hypothesis (Call 2) types and validators (§5.1)."""
from __future__ import annotations
import copy
import math
from typing import Any, NotRequired, TypedDict

from ..io import InvalidOutput
from ..predicates import validate_predicate

VALID_ORIGIN_KINDS = ('broken', 'undelivered', 'never')


class Origin(TypedDict):
    analyst: str
    analyst_group: NotRequired[str]
    defect: str
    kind: str
    turn: int | None
    rationale: str


class Hypothesis(TypedDict):
    analyst: str
    analyst_group: NotRequired[str]
    defect: str
    turn: int
    predicate: dict[str, Any]
    effect: float
    rationale: str


def probability(value: Any, name: str = 'probability') -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise InvalidOutput(f'{name} must be a finite number in [0, 1]')
    return float(value)


def validate_origins(
    items: Any,
    expected_defects: list[str],
    num_turns: int,
    analyst: str,
    analyst_group: str | None = None,
) -> list[Origin]:
    if not isinstance(items, list):
        raise InvalidOutput('Origins must be a list')
    by_defect: dict[str, Origin] = {}
    for item in items:
        if not isinstance(item, dict):
            raise InvalidOutput('Origin entry must be a dict')
        defect = item.get('defect')
        if not isinstance(defect, str) or not defect:
            raise InvalidOutput('Origin defect must be a non-empty string')
        kind = item.get('kind')
        if kind not in VALID_ORIGIN_KINDS:
            raise InvalidOutput(f'Origin kind must be one of {VALID_ORIGIN_KINDS}, got {kind!r}')
        turn = item.get('turn')
        if kind == 'never':
            turn = None
        else:
            if type(turn) is not int or not (1 <= turn <= num_turns):
                raise InvalidOutput(f'Origin turn for {kind} must be an integer in [1, {num_turns}], got {turn!r}')
        rationale = str(item.get('rationale') or '')
        rec: Origin = {
            'analyst': analyst,
            'defect': defect,
            'kind': kind,
            'turn': turn,
            'rationale': rationale,
        }
        if analyst_group is not None:
            rec['analyst_group'] = analyst_group
        by_defect[defect] = rec

    out: list[Origin] = []
    for d_id in expected_defects:
        if d_id in by_defect:
            out.append(by_defect[d_id])
        else:
            rec_default: Origin = {
                'analyst': analyst,
                'defect': d_id,
                'kind': 'never',
                'turn': None,
                'rationale': 'Defaulted to never (omitted by analyst response).',
            }
            if analyst_group is not None:
                rec_default['analyst_group'] = analyst_group
            out.append(rec_default)
    return out


def validate_hypotheses(
    items: Any,
    num_turns: int,
    analyst: str,
    analyst_group: str | None = None,
    allowed_defects: set[str] | None = None,
) -> list[Hypothesis]:
    if not isinstance(items, list):
        raise InvalidOutput('Hypotheses must be a list')
    clean: list[Hypothesis] = []
    for item in items:
        if not isinstance(item, dict):
            raise InvalidOutput('Hypothesis entry must be a dict')
        defect = str(item.get('defect') or '')
        if not defect:
            raise InvalidOutput('Hypothesis defect must be a non-empty string')
        if allowed_defects is not None and defect not in allowed_defects:
            raise InvalidOutput(f'Hypothesis defect {defect!r} not in target defects')
        turn = item.get('turn')
        if type(turn) is not int or not (1 <= turn <= num_turns):
            raise InvalidOutput(f'Hypothesis turn must be an integer in [1, {num_turns}]')
        effect = item.get('effect')
        if type(effect) not in (int, float) or not math.isfinite(effect) or effect >= 0:
            raise InvalidOutput('Hypothesis effect must be finite and negative')
        rationale = item.get('rationale')
        if not isinstance(rationale, str):
            raise InvalidOutput('Hypothesis rationale must be a string')
        rec: Hypothesis = {
            'analyst': analyst,
            'defect': defect,
            'turn': turn,
            'predicate': copy.deepcopy(validate_predicate(item.get('predicate'))),
            'effect': float(effect),
            'rationale': rationale,
        }
        if analyst_group is not None:
            rec['analyst_group'] = analyst_group
        clean.append(rec)
    return clean
