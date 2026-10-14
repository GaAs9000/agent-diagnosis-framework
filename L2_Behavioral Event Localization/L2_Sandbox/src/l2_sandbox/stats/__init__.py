"""Deterministic, recomputable statistical routines for L2 v0.2 value curves and stratification."""
from __future__ import annotations
from .intervals import (
    STATS_POLICY_VERSION,
    wilson_bounds,
    wilson_interval,
    wilson_upper_bound,
    newcombe_lower_bound,
    newcombe_interval,
    bounded_score_lower_bound,
    bounded_score_interval,
    holm_adjust,
)
from .missing import missing_bounds, apply_missing_imputation
from .curve import (
    CURVE_VERDICTS,
    derive_stage2_interval,
    evaluate_curve,
)
from .strata import (
    STRATA_TAGS,
    evaluate_strata,
    mantel_haenszel_pool,
)

__all__ = [
    'STATS_POLICY_VERSION',
    'CURVE_VERDICTS',
    'STRATA_TAGS',
    'wilson_bounds',
    'wilson_interval',
    'wilson_upper_bound',
    'newcombe_lower_bound',
    'newcombe_interval',
    'bounded_score_lower_bound',
    'bounded_score_interval',
    'holm_adjust',
    'missing_bounds',
    'apply_missing_imputation',
    'derive_stage2_interval',
    'evaluate_curve',
    'evaluate_strata',
    'mantel_haenszel_pool',
]
