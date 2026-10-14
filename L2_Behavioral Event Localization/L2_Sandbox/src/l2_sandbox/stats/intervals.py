"""Wilson, Newcombe, t/bootstrap intervals and Holm multiplicity correction."""
from __future__ import annotations
import hashlib
import math
import random
from statistics import NormalDist
from typing import Sequence

STATS_POLICY_VERSION = '0.2.1'
_NORM = NormalDist()


def _z(alpha: float) -> float:
    if not 0.0 < alpha < 0.5:
        raise ValueError(f'alpha must be in (0, 0.5), got {alpha}')
    return _NORM.inv_cdf(1.0 - alpha)


def wilson_bounds(mean_p: float, n: int, z: float) -> tuple[float, float]:
    """Wilson score bounds for a proportion or [0, 1] mean with sample size n."""
    if n <= 0:
        return 0.0, 1.0
    p = min(1.0, max(0.0, float(mean_p)))
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    rad = max(0.0, p * (1.0 - p) / n + (z * z) / (4.0 * n * n))
    half = (z / denom) * math.sqrt(rad)
    return max(0.0, center - half), min(1.0, center + half)


def wilson_interval(vals: Sequence[float | int | bool], alpha: float = 0.05) -> tuple[float, float]:
    if not vals:
        return 0.0, 1.0
    z = _z(alpha / 2.0)
    mean_p = sum(float(x) for x in vals) / len(vals)
    return wilson_bounds(mean_p, len(vals), z)


def wilson_upper_bound(vals: Sequence[float | int | bool], alpha: float = 0.05) -> float:
    if not vals:
        return 1.0
    z = _z(alpha)
    mean_p = sum(float(x) for x in vals) / len(vals)
    _, upper = wilson_bounds(mean_p, len(vals), z)
    return upper


def newcombe_lower_bound(
    vals1: Sequence[float | int | bool],
    vals2: Sequence[float | int | bool] | float,
    alpha: float = 0.05,
) -> float:
    """One-sided (1 - alpha) Newcombe hybrid score lower bound for E[vals1] - E[vals2].

    When vals2 is a scalar float/int (exact terminal grade Y), treats endpoint 2 as
    deterministic (variance 0), reducing to the one-sided Wilson lower bound minus Y.
    """
    if not vals1:
        return -1.0
    z = _z(alpha)
    n1 = len(vals1)
    p1 = sum(float(x) for x in vals1) / n1
    l1, _ = wilson_bounds(p1, n1, z)
    if isinstance(vals2, (int, float)) and not isinstance(vals2, bool):
        p2 = float(vals2)
        u2 = p2
    else:
        seq2 = list(vals2)
        if not seq2:
            return -1.0
        n2 = len(seq2)
        p2 = sum(float(x) for x in seq2) / n2
        _, u2 = wilson_bounds(p2, n2, z)
    return (p1 - p2) - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)


def newcombe_interval(
    vals1: Sequence[float | int | bool],
    vals2: Sequence[float | int | bool] | float,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Two-sided (1 - alpha) Newcombe interval for E[vals1] - E[vals2]."""
    if not vals1:
        return -1.0, 1.0
    z = _z(alpha / 2.0)
    n1 = len(vals1)
    p1 = sum(float(x) for x in vals1) / n1
    l1, u1 = wilson_bounds(p1, n1, z)
    if isinstance(vals2, (int, float)) and not isinstance(vals2, bool):
        p2 = float(vals2)
        l2 = u2 = p2
    else:
        seq2 = list(vals2)
        if not seq2:
            return -1.0, 1.0
        n2 = len(seq2)
        p2 = sum(float(x) for x in seq2) / n2
        l2, u2 = wilson_bounds(p2, n2, z)
    diff = p1 - p2
    lo = diff - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = diff + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return max(-1.0, lo), min(1.0, hi)


def bounded_score_lower_bound(
    vals1: Sequence[float | int | bool],
    vals2: Sequence[float | int | bool] | float,
    alpha: float = 0.05,
    *,
    method: str = 'newcombe_one_sided_bonferroni',
    seed: str = 'l2-stats-0.2.0',
    n_boot: int = 2000,
) -> float:
    """One-sided lower bound on E[vals1] - E[vals2] for bounded scores in [0, 1]."""
    if method.startswith('newcombe'):
        return newcombe_lower_bound(vals1, vals2, alpha=alpha)
    if not vals1:
        return -1.0
    v1 = [float(x) for x in vals1]
    is_terminal = isinstance(vals2, (int, float)) and not isinstance(vals2, bool)
    v2 = [float(vals2)] if is_terminal else [float(x) for x in vals2]
    if not v2:
        return -1.0
    # Deterministic bootstrap combined with Newcombe floor when sample variance is degenerate
    nc_lb = newcombe_lower_bound(v1, float(vals2) if is_terminal else v2, alpha=alpha)
    digest_int = int(hashlib.sha256(f'{seed}:{alpha}:{len(v1)}:{len(v2)}'.encode('utf-8')).hexdigest()[:16], 16)
    rng = random.Random(digest_int)
    n1 = len(v1)
    n2 = len(v2)
    diffs = []
    for _ in range(n_boot):
        m1 = sum(v1[rng.randrange(n1)] for _ in range(n1)) / n1
        m2 = v2[0] if is_terminal else sum(v2[rng.randrange(n2)] for _ in range(n2)) / n2
        diffs.append(m1 - m2)
    diffs.sort()
    idx = max(0, min(len(diffs) - 1, int(math.floor(alpha * len(diffs)))))
    boot_lb = diffs[idx]
    # If sample variance is zero (e.g., all 1s or all 0s), bootstrap collapses to point estimate;
    # take min(boot_lb, nc_lb) so zero-variance small samples remain properly conservative.
    return min(boot_lb, nc_lb)


def bounded_score_interval(
    vals1: Sequence[float | int | bool],
    vals2: Sequence[float | int | bool] | float,
    alpha: float = 0.05,
) -> tuple[float, float]:
    return newcombe_interval(vals1, vals2, alpha=alpha)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values preserving original order."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = min(1.0, float(p) * (m - rank))
        running_max = max(running_max, adj)
        adjusted[orig_idx] = running_max
    return adjusted
