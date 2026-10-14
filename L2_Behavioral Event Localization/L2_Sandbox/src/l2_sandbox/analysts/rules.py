"""Deterministic Rule Analyst (`rules@1`, Group A) and Reference Analyst (`reference@1`, Group R) (L2 v0.3 Section 2.3 & 5)."""
from __future__ import annotations
import copy
from typing import Any
from ..predicates import RULE_VOCABULARY, eval_predicate_on_turn
from . import Hypothesis, Values

RULES_ANALYST_ID = 'rules@1'
REFERENCE_ANALYST_ID = 'reference@1'


def linear_decay_values(num_turns: int, p0: float, terminal_y: float) -> list[float]:
    """Linear decay baseline from p_0 at k=0 toward Y at k=T, evaluated for k = 0 .. T-1."""
    if num_turns <= 0:
        return []
    p0_f = max(0.0, min(1.0, float(p0)))
    y_f = max(0.0, min(1.0, float(terminal_y)))
    return [
        max(0.0, min(1.0, p0_f - (k / float(num_turns)) * (p0_f - y_f)))
        for k in range(num_turns)
    ]


def predict_values_rules(
    facts: list[dict[str, Any]],
    num_turns: int,
    p0: float,
    terminal_y: float,
) -> Values:
    """Call 1 for `rules@1` (Section 2.3):
    First-anomaly step function: predicts p_0 before the first `tool_execution_anomaly` turn,
    and Y from that turn onward. If no anomaly occurs, predicts linear decay from p_0 to Y.
    """
    p0_f = max(0.0, min(1.0, float(p0)))
    y_f = max(0.0, min(1.0, float(terminal_y)))
    anomaly_pred = RULE_VOCABULARY['tool_execution_anomaly']

    first_anomaly_turn: int | None = None
    for t in range(1, num_turns + 1):
        if eval_predicate_on_turn(anomaly_pred, facts, t):
            first_anomaly_turn = t
            break

    if first_anomaly_turn is None:
        vals = linear_decay_values(num_turns, p0_f, y_f)
    else:
        # Checkpoint c_k is after k completed turns (turns 1..k).
        # Before turn `first_anomaly_turn` is executed (i.e. k < first_anomaly_turn), predict p_0;
        # once turn `first_anomaly_turn` has completed (k >= first_anomaly_turn), predict Y.
        vals = [p0_f if k < first_anomaly_turn else y_f for k in range(num_turns)]

    return {'analyst': RULES_ANALYST_ID, 'values': vals}


def propose_hypotheses_rules(
    facts: list[dict[str, Any]],
    curve_summary: dict[str, Any],
) -> list[Hypothesis]:
    """Call 2 for `rules@1` (Section 2.3):
    Only invoked when curve verdict is `concentrated_turn` or `concentrated_interval`.
    Proposes all vocabulary predicates that hold on the parent trajectory at the localized turn.
    """
    verdict = curve_summary.get('verdict')
    if verdict not in ('concentrated_turn', 'concentrated_interval'):
        return []

    target_turn = curve_summary.get('turn') or curve_summary.get('candidate_turn')
    if not isinstance(target_turn, int) or target_turn < 1:
        interval = curve_summary.get('interval')
        if isinstance(interval, list) and len(interval) == 2:
            target_turn = int(interval[0]) + 1
        else:
            return []

    hyps: list[Hypothesis] = []
    for name, pred in RULE_VOCABULARY.items():
        if eval_predicate_on_turn(pred, facts, target_turn):
            hyps.append({
                'turn': int(target_turn),
                'predicate': copy.deepcopy(pred),
                'effect': -0.25,
                'rationale': f'Rule vocabulary predicate `{name}` holds on parent turn {target_turn}.',
            })
    return hyps


def predict_values_reference(
    facts: list[dict[str, Any]],
    num_turns: int,
    p0: float,
    terminal_y: float,
    checkpoint_snapshot_scores: dict[int, float] | None = None,
) -> Values:
    """Call 1 for Reference Analyst `reference@1` (Group R, Section 5):
    Uses official scores of intermediate submission snapshots when available, otherwise rules@1.
    """
    if not checkpoint_snapshot_scores:
        base = predict_values_rules(facts, num_turns, p0, terminal_y)
        return {'analyst': REFERENCE_ANALYST_ID, 'values': base['values']}

    p0_f = max(0.0, min(1.0, float(p0)))
    vals = []
    for k in range(num_turns):
        if k in checkpoint_snapshot_scores:
            vals.append(max(0.0, min(1.0, float(checkpoint_snapshot_scores[k]))))
        else:
            vals.append(p0_f)
    return {'analyst': REFERENCE_ANALYST_ID, 'values': vals}
