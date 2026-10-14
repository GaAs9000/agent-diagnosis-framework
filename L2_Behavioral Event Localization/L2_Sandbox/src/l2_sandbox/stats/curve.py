"""Two-stage sparse value-curve selection, inference, and verdict computation."""
from __future__ import annotations
from typing import Any
from .intervals import STATS_POLICY_VERSION, wilson_interval, wilson_upper_bound, bounded_score_lower_bound
from .missing import apply_missing_imputation, missing_bounds

CURVE_VERDICTS = (
    'concentrated_turn',
    'concentrated_interval',
    'diffuse_loss',
    'out_of_reach',
    'not_identifiable',
)


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def derive_stage2_interval(
    grid_turns: list[int],
    stage1_defined: dict[int, list[float]],
    terminal_turn: int,
    terminal_value: float,
) -> dict[str, Any]:
    """Selects [a, b] on the coarse grid maximizing V1(c_a) - V1(c_b), breaking ties by earliest a."""
    full_grid = sorted(set(grid_turns) | {terminal_turn})
    if len(full_grid) < 2:
        raise ValueError('Grid must contain at least one checkpoint and terminal turn')

    def v1(k: int) -> float:
        if k == terminal_turn:
            return float(terminal_value)
        vals = stage1_defined.get(k, [])
        return _mean(vals)

    intervals = []
    best_pair = (full_grid[0], full_grid[1])
    best_drop = -1e9
    for a, b in zip(full_grid[:-1], full_grid[1:]):
        drop = v1(a) - v1(b)
        intervals.append({'start_turn': a, 'end_turn': b, 'v1_start': v1(a), 'v1_end': v1(b), 'drop': drop})
        if drop > best_drop + 1e-12:
            best_drop = drop
            best_pair = (a, b)

    a, b = best_pair
    stage2_turns = [k for k in range(a, b + 1) if k < terminal_turn]
    return {
        'selected_interval': {'start_turn': a, 'end_turn': b, 'stage1_drop': best_drop},
        'stage2_turns': stage2_turns,
        'stage2_checkpoints': [f'cp{k:06d}' for k in stage2_turns],
        'stage1_intervals': intervals,
        'v1_initial': v1(0),
    }


def _infer_once(
    stage1_vals: dict[int, list[float]],
    stage2_vals: dict[int, list[float]],
    a: int,
    b: int,
    terminal_turn: int,
    terminal_value: float,
    alpha: float,
    turn_test: str,
) -> dict[str, Any]:
    v0_samples = stage1_vals.get(0, [])
    v1_0 = _mean(v0_samples)
    v0_upper_95 = wilson_upper_bound(v0_samples, alpha=0.05)

    def v2_mean(k: int) -> float:
        return float(terminal_value) if k == terminal_turn else _mean(stage2_vals.get(k, []))

    def v2_target(k: int) -> list[float] | float:
        return float(terminal_value) if k == terminal_turn else stage2_vals.get(k, [])

    share = ((b - a) / float(terminal_turn)) * max(0.0, v1_0 - float(terminal_value))
    s2_a = stage2_vals.get(a, [])
    s2_b = v2_target(b)
    interval_drop = v2_mean(a) - v2_mean(b)
    interval_lb = bounded_score_lower_bound(s2_a, s2_b, alpha=alpha, method=turn_test)
    interval_ok = interval_lb > share

    turn_candidates = list(range(a + 1, b + 1))
    best_j = turn_candidates[0]
    best_turn_drop = -1e9
    turn_details = []
    for k in turn_candidates:
        d_k = v2_mean(k - 1) - v2_mean(k)
        lb_k = bounded_score_lower_bound(
            stage2_vals.get(k - 1, []),
            v2_target(k),
            alpha=alpha / max(1, b - a),
            method=turn_test,
        )
        turn_details.append({
            'turn': k,
            'from_checkpoint': f'cp{k-1:06d}',
            'to_checkpoint': 'terminal' if k == terminal_turn else f'cp{k:06d}',
            'drop': d_k,
            'lower_bound': lb_k,
        })
        if d_k > best_turn_drop + 1e-12:
            best_turn_drop = d_k
            best_j = k

    delta_turn = max(0.10, 2.0 * v1_0 / float(terminal_turn))
    best_turn_lb = bounded_score_lower_bound(
        stage2_vals.get(best_j - 1, []),
        v2_target(best_j),
        alpha=alpha / max(1, b - a),
        method=turn_test,
    )
    turn_ok = interval_ok and (best_turn_lb > delta_turn)

    if v0_upper_95 < 0.25:
        verdict = 'out_of_reach'
    elif turn_ok:
        verdict = 'concentrated_turn'
    elif interval_ok:
        verdict = 'concentrated_interval'
    elif v1_0 > float(terminal_value) + 0.05:
        verdict = 'diffuse_loss'
    else:
        verdict = 'out_of_reach'

    return {
        'verdict': verdict,
        'v1_initial': v1_0,
        'v0_upper_95': v0_upper_95,
        'uniform_share_threshold': share,
        'interval_drop': interval_drop,
        'interval_lower_bound': interval_lb,
        'interval_passed': interval_ok,
        'selected_turn': best_j,
        'selected_turn_drop': best_turn_drop,
        'selected_turn_lower_bound': best_turn_lb,
        'delta_turn_threshold': delta_turn,
        'turn_passed': turn_ok,
        'turn_details': turn_details,
    }


def evaluate_curve(
    grid_turns: list[int],
    stage1_samples: dict[int, list[dict[str, Any]]],
    stage2_samples: dict[int, list[dict[str, Any]]],
    terminal_turn: int,
    terminal_value: float,
    *,
    alpha: float = 0.05,
    turn_test: str = 'newcombe_one_sided_bonferroni',
) -> dict[str, Any]:
    """Computes full two-stage curve inference, missing-outcome bounds, and final verdict."""
    s1_defined = {
        k: [float(r['outcome']['value']) for r in rows if r.get('outcome', {}).get('endpoint_defined')]
        for k, rows in stage1_samples.items()
    }
    s2_defined = {
        k: [float(r['outcome']['value']) for r in rows if r.get('outcome', {}).get('endpoint_defined')]
        for k, rows in stage2_samples.items()
    }
    selection = derive_stage2_interval(grid_turns, s1_defined, terminal_turn, terminal_value)
    a = selection['selected_interval']['start_turn']
    b = selection['selected_interval']['end_turn']

    primary = _infer_once(s1_defined, s2_defined, a, b, terminal_turn, terminal_value, alpha, turn_test)
    m_info = missing_bounds(stage1_samples, stage2_samples)

    # Compute missing-as-0 and missing-as-1 bounds
    s1_zero = apply_missing_imputation(stage1_samples, 0.0)
    s2_zero = apply_missing_imputation(stage2_samples, 0.0)
    s1_one = apply_missing_imputation(stage1_samples, 1.0)
    s2_one = apply_missing_imputation(stage2_samples, 1.0)
    inf_zero = _infer_once(s1_zero, s2_zero, a, b, terminal_turn, terminal_value, alpha, turn_test)
    inf_one = _infer_once(s1_one, s2_one, a, b, terminal_turn, terminal_value, alpha, turn_test)

    # Sensitivity check: unscorable submissions treated as missing
    s2_unscorable_one = apply_missing_imputation(stage2_samples, 1.0, include_unscorable_as_missing=True)
    inf_unscorable_one = _infer_once(s1_defined, s2_unscorable_one, a, b, terminal_turn, terminal_value, alpha, turn_test)

    bounds_agree = (inf_zero['verdict'] == inf_one['verdict'] == primary['verdict'])
    identifiable = (
        bounds_agree
        and m_info['missing_rate'] <= 0.05
        and m_info['policy_identity_violations'] == 0
        and m_info['model_version_stable']
        and all(len(s2_defined.get(k, [])) > 0 for k in range(a, b + 1) if k < terminal_turn)
    )
    final_verdict = primary['verdict'] if identifiable else 'not_identifiable'

    checkpoint_estimates = {}
    for k in sorted(set(s1_defined) | set(s2_defined)):
        vals1 = s1_defined.get(k, [])
        vals2 = s2_defined.get(k, [])
        checkpoint_estimates[f'cp{k:06d}'] = {
            'turn': k,
            'stage1_n': len(vals1),
            'stage1_mean': _mean(vals1) if vals1 else None,
            'stage1_wilson_95': list(wilson_interval(vals1, 0.05)) if vals1 else None,
            'stage2_n': len(vals2),
            'stage2_mean': _mean(vals2) if vals2 else None,
            'stage2_wilson_95': list(wilson_interval(vals2, 0.05)) if vals2 else None,
        }

    return {
        'stats_policy_version': STATS_POLICY_VERSION,
        'evidence_basis': 'value_curve',
        'curve_verdict': final_verdict,
        'evidence_level': 'E1' if final_verdict in ('concentrated_turn', 'concentrated_interval') else 'E0',
        'granularity': (
            'event' if final_verdict == 'concentrated_turn'
            else 'interval' if final_verdict in ('concentrated_interval', 'diffuse_loss')
            else 'terminal'
        ),
        'selected_interval': selection['selected_interval'],
        'selected_turn': primary['selected_turn'] if final_verdict == 'concentrated_turn' else None,
        'candidate_turn_in_interval': primary['selected_turn'],
        'primary_inference': primary,
        'missing_summary': m_info,
        'missing_bounds': {
            'missing_as_0_verdict': inf_zero['verdict'],
            'missing_as_1_verdict': inf_one['verdict'],
            'bounds_preserve_verdict': bounds_agree,
            'unscorable_as_1_sensitivity_verdict': inf_unscorable_one['verdict'],
        },
        'checkpoint_estimates': checkpoint_estimates,
        'terminal_turn': terminal_turn,
        'terminal_value': terminal_value,
    }
