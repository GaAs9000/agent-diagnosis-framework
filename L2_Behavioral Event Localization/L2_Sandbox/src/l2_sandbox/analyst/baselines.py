"""Value-curve prediction baselines (Section 5.2): constant, linear_decay, first_anomaly, rules, random_step."""
from __future__ import annotations
import hashlib
import random
from typing import Any, Sequence


def _parse_cp_turn(cp_id: str) -> int:
    return int(cp_id[2:])


def extract_terminal_outcome(reader, endpoint: str = 'official_completion_rate') -> float:
    """Extracts the parent episode's terminal value Y in [0, 1] according to Section 3.5 & 3.6."""
    from ..experiments.controller import outcome
    oc = outcome(reader, endpoint)
    if oc.get('endpoint_defined') and oc.get('value') is not None:
        return float(oc['value'])
    return 0.0


def constant_baseline(
    checkpoints: Sequence[str],
    *,
    baseline_rate: float = 0.5,
) -> dict[str, Any]:
    val = min(1.0, max(0.0, float(baseline_rate)))
    return {
        'baseline_kind': 'constant',
        'value_predictions': {cp: val for cp in checkpoints},
        'predicted_loss_intervals': [],
    }


def linear_decay_baseline(
    checkpoints: Sequence[str],
    num_turns: int,
    *,
    baseline_rate: float = 0.5,
    terminal_y: float = 0.0,
) -> dict[str, Any]:
    """Uniform-share null hypothesis baseline: linearly interpolates from baseline_rate at cp000000 to Y at T."""
    t_max = max(1, int(num_turns))
    v0 = min(1.0, max(0.0, float(baseline_rate)))
    y = min(1.0, max(0.0, float(terminal_y)))
    preds = {}
    for cp in checkpoints:
        k = _parse_cp_turn(cp)
        frac = min(1.0, max(0.0, k / float(t_max)))
        preds[cp] = round(v0 + frac * (y - v0), 6)
    cps = sorted(checkpoints, key=_parse_cp_turn)
    intervals = []
    if len(cps) >= 2:
        intervals.append({
            'start_checkpoint': cps[0],
            'end_checkpoint': cps[1],
            'turn': None,
            'confidence': round(max(0.0, (v0 - y) / max(1, len(cps) - 1)), 4),
        })
    return {
        'baseline_kind': 'linear_decay',
        'value_predictions': preds,
        'predicted_loss_intervals': intervals,
    }


def first_anomaly_baseline(
    checkpoints: Sequence[str],
    num_turns: int,
    first_anomaly_turn: int | None,
    *,
    baseline_rate: float = 0.5,
    terminal_y: float = 0.0,
) -> dict[str, Any]:
    """Steps from baseline_rate before first_anomaly_turn down to Y at and after first_anomaly_turn."""
    v0 = min(1.0, max(0.0, float(baseline_rate)))
    y = min(1.0, max(0.0, float(terminal_y)))
    preds = {}
    for cp in checkpoints:
        k = _parse_cp_turn(cp)
        if first_anomaly_turn is not None and k >= first_anomaly_turn:
            preds[cp] = y
        else:
            preds[cp] = v0
    cps = sorted(checkpoints, key=_parse_cp_turn)
    intervals = []
    if first_anomaly_turn is not None and len(cps) >= 2:
        for a_cp, b_cp in zip(cps[:-1], cps[1:]):
            a, b = _parse_cp_turn(a_cp), _parse_cp_turn(b_cp)
            if a < first_anomaly_turn <= b:
                intervals.append({
                    'start_checkpoint': a_cp,
                    'end_checkpoint': b_cp,
                    'turn': first_anomaly_turn,
                    'confidence': round(max(0.0, v0 - y), 4),
                })
                break
    return {
        'baseline_kind': 'first_anomaly',
        'first_anomaly_turn': first_anomaly_turn,
        'value_predictions': preds,
        'predicted_loss_intervals': intervals,
    }


def random_step_baseline(
    checkpoints: Sequence[str],
    num_turns: int,
    *,
    baseline_rate: float = 0.5,
    terminal_y: float = 0.0,
    seed: str = 'l2-random-baseline-0.2.0',
) -> dict[str, Any]:
    """Sanity-check baseline: steps down at a deterministically seeded random turn."""
    t_max = max(1, int(num_turns))
    digest_int = int(hashlib.sha256(seed.encode('utf-8')).hexdigest()[:16], 16)
    rng = random.Random(digest_int)
    step_turn = rng.randint(1, t_max)
    res = first_anomaly_baseline(
        checkpoints,
        t_max,
        step_turn,
        baseline_rate=baseline_rate,
        terminal_y=terminal_y,
    )
    res['baseline_kind'] = 'random_step'
    res['random_step_turn'] = step_turn
    return res


def rules_value_predictions(
    reader,
    turn_tags: list[dict],
    checkpoints: Sequence[str],
    *,
    baseline_rate: float = 0.5,
    endpoint: str = 'official_completion_rate',
) -> dict[str, Any]:
    """Deterministic rule-based value predictions over checkpoints."""
    num_turns = max(1, len(turn_tags) or reader.manifest.get('counts', {}).get('model_requests', 1))
    terminal_y = extract_terminal_outcome(reader, endpoint=endpoint)
    first_anom = next(
        (t['turn_index'] for t in turn_tags if 'tool_execution_anomaly' in t.get('tags', ())),
        None,
    )
    if first_anom is not None:
        base = first_anomaly_baseline(
            checkpoints,
            num_turns,
            first_anom,
            baseline_rate=baseline_rate,
            terminal_y=terminal_y,
        )
    else:
        base = linear_decay_baseline(
            checkpoints,
            num_turns,
            baseline_rate=baseline_rate,
            terminal_y=terminal_y,
        )
    return {
        'value_predictions': base['value_predictions'],
        'predicted_loss_intervals': base['predicted_loss_intervals'],
    }


def compute_all_baselines(
    reader,
    turn_tags: list[dict],
    checkpoints: Sequence[str],
    *,
    baseline_rate: float = 0.5,
    endpoint: str = 'official_completion_rate',
    seed: str = 'l2-0.2.0',
) -> dict[str, dict[str, Any]]:
    num_turns = max(1, len(turn_tags) or reader.manifest.get('counts', {}).get('model_requests', 1))
    terminal_y = extract_terminal_outcome(reader, endpoint=endpoint)
    first_anom = next(
        (t['turn_index'] for t in turn_tags if 'tool_execution_anomaly' in t.get('tags', ())),
        None,
    )
    return {
        'constant': constant_baseline(checkpoints, baseline_rate=baseline_rate),
        'linear_decay': linear_decay_baseline(
            checkpoints, num_turns, baseline_rate=baseline_rate, terminal_y=terminal_y
        ),
        'first_anomaly': first_anomaly_baseline(
            checkpoints, num_turns, first_anom, baseline_rate=baseline_rate, terminal_y=terminal_y
        ),
        'rules': rules_value_predictions(
            reader, turn_tags, checkpoints, baseline_rate=baseline_rate, endpoint=endpoint
        ),
        'random_step': random_step_baseline(
            checkpoints,
            num_turns,
            baseline_rate=baseline_rate,
            terminal_y=terminal_y,
            seed=f'{seed}:{reader.episode_id}',
        ),
    }
