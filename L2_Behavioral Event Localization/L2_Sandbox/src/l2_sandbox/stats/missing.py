"""Missing-outcome imputation bounds (all-0 vs all-1) for sensitivity analysis."""
from __future__ import annotations
import copy
from typing import Any


def apply_missing_imputation(
    stage_samples: dict[int, list[dict[str, Any]]],
    fill_value: float,
    *,
    include_unscorable_as_missing: bool = False,
) -> dict[int, list[float]]:
    """Extracts numeric outcome vectors per checkpoint k, imputing missing outcomes as fill_value."""
    out: dict[int, list[float]] = {}
    for k, rows in stage_samples.items():
        vec: list[float] = []
        for r in rows:
            oc = r.get('outcome', {})
            if include_unscorable_as_missing and oc.get('disposition') == 'unscorable_submission_zero':
                vec.append(float(fill_value))
            elif oc.get('endpoint_defined') and oc.get('value') is not None:
                vec.append(float(oc['value']))
            else:
                vec.append(float(fill_value))
        out[k] = vec
    return out


def missing_bounds(
    stage1_samples: dict[int, list[dict[str, Any]]],
    stage2_samples: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Counts total attempted, defined, missing, and unscorable-submission branches across stages."""
    all_rows = [r for rows in stage1_samples.values() for r in rows] + [
        r for rows in stage2_samples.values() for r in rows
    ]
    total = len(all_rows)
    defined = sum(1 for r in all_rows if r.get('outcome', {}).get('endpoint_defined'))
    missing = total - defined
    empty_sub = sum(1 for r in all_rows if r.get('outcome', {}).get('disposition') == 'empty_submission_zero')
    unscorable_sub = sum(1 for r in all_rows if r.get('outcome', {}).get('disposition') == 'unscorable_submission_zero')
    policy_violations = sum(1 for r in all_rows if not r.get('first_request_identical', True))
    model_versions = sorted({
        r.get('outcome', {}).get('model_version')
        for r in all_rows
        if r.get('outcome', {}).get('model_version')
    })
    return {
        'total_branches': total,
        'endpoint_defined_count': defined,
        'missing_count': missing,
        'missing_rate': (missing / total) if total > 0 else 0.0,
        'empty_submission_zero_count': empty_sub,
        'unscorable_submission_zero_count': unscorable_sub,
        'policy_identity_violations': policy_violations,
        'observed_model_versions': model_versions,
        'model_version_stable': len(model_versions) <= 1,
    }
