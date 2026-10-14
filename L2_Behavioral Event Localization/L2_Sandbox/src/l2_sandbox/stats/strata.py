"""Same-state behavioral stratification (E1+) and Mantel-Haenszel cross-trajectory pooling."""
from __future__ import annotations
import math
from statistics import NormalDist
from typing import Any, Sequence
from .intervals import STATS_POLICY_VERSION, newcombe_interval, holm_adjust

STRATA_TAGS = (
    'tool_execution_anomaly',
    'submission_modified',
    'assistant_finished_turn',
    'submission_readback',
    'object_structure_changed',
)
_NORM = NormalDist()


def _two_sided_p(vals1: list[float], vals0: list[float]) -> float:
    n1, n0 = len(vals1), len(vals0)
    if n1 == 0 or n0 == 0:
        return 1.0
    m1 = sum(vals1) / n1
    m0 = sum(vals0) / n0
    p_pool = (sum(vals1) + sum(vals0)) / (n1 + n0)
    var = p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n0)
    if var <= 1e-12:
        return 1.0 if abs(m1 - m0) <= 1e-12 else 0.0
    z = abs(m1 - m0) / math.sqrt(var)
    return max(0.0, min(1.0, 2.0 * (1.0 - _NORM.cdf(z))))


def evaluate_strata(
    branches: Sequence[dict[str, Any]],
    *,
    kinds: Sequence[str] = STRATA_TAGS,
    min_per_class: int = 4,
    is_docx_task: bool = True,
    actual_turn_tags: Sequence[str] = (),
    v_before: float | None = None,
    turn_drop: float | None = None,
) -> dict[str, Any]:
    """Evaluates E1+ same-state behavioral stratification at a single checkpoint c_{j-1}."""
    defined = [
        r for r in branches
        if r.get('outcome', {}).get('endpoint_defined') and r.get('outcome', {}).get('value') is not None
    ]
    rows = []
    tested_indices = []
    raw_ps = []

    for idx, tag in enumerate(kinds):
        if tag == 'object_structure_changed' and not is_docx_task:
            rows.append({
                'tag': tag,
                'status': 'not_applicable',
                'reason': 'object_structure_changed is only defined for .docx tasks',
                'n_in_class': 0,
                'n_out_of_class': 0,
                'delta_b': None,
                'ci_95': None,
                'p_value': None,
                'p_value_holm': None,
                'explained_drop_share': None,
                'evidence_level': None,
            })
            continue

        g1 = [float(r['outcome']['value']) for r in defined if tag in (r.get('first_turn_tags') or ())]
        g0 = [float(r['outcome']['value']) for r in defined if tag not in (r.get('first_turn_tags') or ())]
        if len(g1) < min_per_class or len(g0) < min_per_class:
            rows.append({
                'tag': tag,
                'status': 'not_identifiable',
                'reason': f'Requires at least {min_per_class} branches in both B and not-B',
                'n_in_class': len(g1),
                'n_out_of_class': len(g0),
                'delta_b': None,
                'ci_95': None,
                'p_value': None,
                'p_value_holm': None,
                'explained_drop_share': None,
                'evidence_level': None,
            })
            continue

        m1 = sum(g1) / len(g1)
        m0 = sum(g0) / len(g0)
        delta_b = m1 - m0
        lo, hi = newcombe_interval(g1, g0, alpha=0.05)
        pval = _two_sided_p(g1, g0)
        actual_in_b = tag in actual_turn_tags
        actual_mean = m1 if actual_in_b else m0
        explained = None
        if v_before is not None and turn_drop is not None and abs(turn_drop) > 1e-9:
            explained = (float(v_before) - actual_mean) / float(turn_drop)

        tested_indices.append(idx)
        raw_ps.append(pval)
        rows.append({
            'tag': tag,
            'status': 'evaluated',
            'n_in_class': len(g1),
            'n_out_of_class': len(g0),
            'mean_in_class': m1,
            'mean_out_of_class': m0,
            'delta_b': delta_b,
            'ci_95': [lo, hi],
            'p_value': pval,
            'p_value_holm': None,
            'actual_turn_in_class': actual_in_b,
            'explained_drop_share': explained,
            'evidence_level': 'E1+',
        })

    adj_ps = holm_adjust(raw_ps)
    for ti, adj_p in zip(tested_indices, adj_ps):
        rows[ti]['p_value_holm'] = adj_p

    # Direct head-to-head contrast: assistant_finished_turn vs submission_readback (auditspec claim vs verification)
    finish_only = [
        float(r['outcome']['value']) for r in defined
        if 'assistant_finished_turn' in (r.get('first_turn_tags') or ())
        and 'submission_readback' not in (r.get('first_turn_tags') or ())
    ]
    readback_only = [
        float(r['outcome']['value']) for r in defined
        if 'submission_readback' in (r.get('first_turn_tags') or ())
        and 'assistant_finished_turn' not in (r.get('first_turn_tags') or ())
    ]
    if len(finish_only) >= min_per_class and len(readback_only) >= min_per_class:
        m_fin = sum(finish_only) / len(finish_only)
        m_rb = sum(readback_only) / len(readback_only)
        lo, hi = newcombe_interval(finish_only, readback_only, alpha=0.05)
        claim_vs_readback = {
            'status': 'evaluated',
            'n_finish_without_readback': len(finish_only),
            'n_submission_readback': len(readback_only),
            'mean_finish_without_readback': m_fin,
            'mean_submission_readback': m_rb,
            'delta_finish_minus_readback': m_fin - m_rb,
            'ci_95': [lo, hi],
            'evidence_level': 'E1+',
        }
    else:
        claim_vs_readback = {
            'status': 'not_identifiable',
            'n_finish_without_readback': len(finish_only),
            'n_submission_readback': len(readback_only),
            'delta_finish_minus_readback': None,
            'ci_95': None,
            'evidence_level': None,
        }

    return {
        'stats_policy_version': STATS_POLICY_VERSION,
        'evidence_basis': 'resample_stratified',
        'evidence_level': 'E1+',
        'causal_upgrade_allowed': False,
        'min_per_class': min_per_class,
        'multiplicity': 'holm',
        'tags': rows,
        'finish_vs_readback_contrast': claim_vs_readback,
    }


def mantel_haenszel_pool(strata_tables: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pools 2x2 / bounded-score strata across trajectories stratified by task family.

    Each entry in strata_tables has:
      {'family_id': str, 'n1': int, 'n0': int, 'mean1': float, 'mean0': float}
    Computes the Mantel-Haenszel weighted risk/score difference and 95% CI.
    """
    valid = [t for t in strata_tables if t.get('n1', 0) > 0 and t.get('n0', 0) > 0]
    if not valid:
        return {'status': 'not_identifiable', 'pooled_delta': None, 'ci_95': None, 'strata_count': 0}
    num = 0.0
    den = 0.0
    var_num = 0.0
    for t in valid:
        n1 = float(t['n1'])
        n0 = float(t['n0'])
        nt = n1 + n0
        m1 = min(1.0, max(0.0, float(t['mean1'])))
        m0 = min(1.0, max(0.0, float(t['mean0'])))
        w = (n1 * n0) / nt
        num += w * (m1 - m0)
        den += w
        # Sato-Robins variance weight for MH risk difference
        v1 = max(0.01, m1 * (1.0 - m1))
        v0 = max(0.01, m0 * (1.0 - m0))
        var_num += (w * w) * (v1 / max(1.0, n1) + v0 / max(1.0, n0))
    pooled = num / den
    se = math.sqrt(var_num) / den if den > 0 else 0.0
    z = _NORM.inv_cdf(0.975)
    return {
        'status': 'evaluated',
        'evidence_level': 'E1+',
        'strata_count': len(valid),
        'family_count': len({t.get('family_id', f's_{i}') for i, t in enumerate(valid)}),
        'pooled_delta': pooled,
        'ci_95': [max(-1.0, pooled - z * se), min(1.0, pooled + z * se)],
    }
