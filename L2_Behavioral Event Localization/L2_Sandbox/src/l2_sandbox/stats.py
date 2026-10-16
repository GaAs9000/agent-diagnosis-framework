"""Statistical bounds, permutation & Fisher tests, Holm step-down, and cluster bootstrap (L2 v0.4 Section 6 & 7)."""
from __future__ import annotations
import itertools
import math
import random
from statistics import NormalDist
from typing import Any, Callable, Sequence
from .io import L2Error, digest

_NORM = NormalDist()


def _bounded_sample(vals: Sequence[float]) -> list[float]:
    if any(type(v) not in (int, float, bool) or not math.isfinite(v) or not 0 <= v <= 1 for v in vals):
        raise L2Error('Statistical outcomes must be finite numbers in [0, 1]')
    return sorted(float(v) for v in vals)


def _is_boolean_sample(vals: Sequence[float]) -> bool:
    return all(float(v) in (0.0, 1.0) for v in vals)


def wilson_bounds(vals: Sequence[float], alpha: float = 0.05, *, one_sided: bool = True) -> tuple[float, float]:
    """Wilson score interval for boolean/proportion data in [0, 1]."""
    vals = _bounded_sample(vals)
    n = len(vals)
    if n == 0:
        return 0.0, 1.0
    p = max(0.0, min(1.0, sum(float(x) for x in vals) / n))
    z = _NORM.inv_cdf(1.0 - (alpha if one_sided else alpha / 2.0))
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))
    return max(0.0, center - half), min(1.0, center + half)


def lower_drop(
    vals_before: Sequence[float],
    vals_after: Sequence[float] | float,
    alpha: float = 0.05,
    *,
    seed: str = 'l2-v03-drop',
) -> float:
    """Nominal one-sided lower bound for V(c_before) - V(c_after).

    - Uses Newcombe one-sided lower bound for boolean samples (or single-sample Wilson when
      `vals_after` is the deterministic terminal state Y).
    - For bounded continuous scores, takes the smaller of the percentile-bootstrap
      bound and the corresponding Wilson/Newcombe-style bound, for all samples.
    """
    vals_before = _bounded_sample(vals_before)
    n1 = len(vals_before)
    if n1 == 0:
        return -1.0

    # Right endpoint is deterministic terminal state c_T = Y (single-sample case)
    if isinstance(vals_after, (int, float)) and not isinstance(vals_after, bool):
        y_term = float(vals_after)
        if _is_boolean_sample(vals_before):
            lo1, _ = wilson_bounds(vals_before, alpha=alpha, one_sided=True)
            return lo1 - y_term
        rng = random.Random(int(digest({'seed': seed, 'b': sorted(round(float(x), 8) for x in vals_before), 'y': y_term, 'a': alpha})[:16], 16))
        n_boot = 2000
        means = sorted((sum(rng.choice(vals_before) for _ in range(n1)) / n1) - y_term for _ in range(n_boot))
        idx = min(n_boot - 1, max(0, int(math.floor(alpha * (n_boot - 1)))))
        lo_w, _ = wilson_bounds(vals_before, alpha=alpha, one_sided=True)
        return min(means[idx], lo_w - y_term)

    after_list = _bounded_sample(vals_after)  # type: ignore[arg-type]
    n2 = len(after_list)
    if n2 == 0:
        return -1.0

    if _is_boolean_sample(vals_before) and _is_boolean_sample(after_list):
        p1 = sum(float(x) for x in vals_before) / n1
        p2 = sum(float(x) for x in after_list) / n2
        l1, _ = wilson_bounds(vals_before, alpha=alpha, one_sided=True)
        _, u2 = wilson_bounds(after_list, alpha=alpha, one_sided=True)
        delta = math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
        return (p1 - p2) - delta

    rng = random.Random(
        int(digest({'seed': seed, 'b': sorted(round(float(x), 8) for x in vals_before), 'a': sorted(round(float(x), 8) for x in after_list), 'alpha': alpha})[:16], 16)
    )
    n_boot = 2000
    diffs = sorted(
        (sum(rng.choice(vals_before) for _ in range(n1)) / n1) - (sum(rng.choice(after_list) for _ in range(n2)) / n2)
        for _ in range(n_boot)
    )
    idx = min(n_boot - 1, max(0, int(math.floor(alpha * (n_boot - 1)))))
    p1 = sum(float(x) for x in vals_before) / n1
    p2 = sum(float(x) for x in after_list) / n2
    l1, _ = wilson_bounds(vals_before, alpha=alpha, one_sided=True)
    _, u2 = wilson_bounds(after_list, alpha=alpha, one_sided=True)
    newcombe_lb = (p1 - p2) - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    return min(diffs[idx], newcombe_lb)


def fisher_exact_one_sided(y_p: Sequence[float], y_not_p: Sequence[float], *, direction: float = -1.0) -> float:
    """One-sided Fisher exact test p-value for binary outcomes in P vs not-P.
    When direction < 0, tests H_1: E[Y | P] < E[Y | not P].
    """
    y_p, y_not_p = _bounded_sample(y_p), _bounded_sample(y_not_p)
    n1 = len(y_p)
    n2 = len(y_not_p)
    if n1 == 0 or n2 == 0:
        return 1.0
    if not _is_boolean_sample(y_p) or not _is_boolean_sample(y_not_p):
        raise L2Error('Fisher exact inference requires binary outcomes')
    k1 = sum(1 for x in y_p if float(x) >= 0.5)
    k2 = sum(1 for x in y_not_p if float(x) >= 0.5)
    total_k = k1 + k2
    total_n = n1 + n2
    denom = math.comb(total_n, n1)
    if denom == 0:
        return 1.0
    lo_x = max(0, total_k - n2)
    hi_x = min(n1, total_k)
    if direction <= 0:
        num = sum(math.comb(total_k, x) * math.comb(total_n - total_k, n1 - x) for x in range(lo_x, k1 + 1))
    else:
        num = sum(math.comb(total_k, x) * math.comb(total_n - total_k, n1 - x) for x in range(k1, hi_x + 1))
    return max(0.0, min(1.0, num / denom))


def permutation_one_sided(
    y_p: Sequence[float],
    y_not_p: Sequence[float],
    *,
    direction: float = -1.0,
    n_perm: int = 5000,
    seed: str = 'l2-v03-perm',
) -> float:
    """One-sided permutation test p-value (exact Fisher test when boolean, exact/MC permutation for scores)."""
    y_p, y_not_p = _bounded_sample(y_p), _bounded_sample(y_not_p)
    n1 = len(y_p)
    n2 = len(y_not_p)
    if n1 == 0 or n2 == 0:
        return 1.0
    if _is_boolean_sample(y_p) and _is_boolean_sample(y_not_p):
        return fisher_exact_one_sided(y_p, y_not_p, direction=direction)

    combined = sorted(float(x) for x in [*y_p, *y_not_p])
    total_n = n1 + n2
    total_sum = sum(combined)
    obs_diff = (sum(float(x) for x in y_p) / n1) - (sum(float(x) for x in y_not_p) / n2)

    n_comb = math.comb(total_n, n1)
    if n_comb <= 10000:
        extreme = 0
        for idxs in itertools.combinations(range(total_n), n1):
            s1 = sum(combined[i] for i in idxs)
            s2 = total_sum - s1
            diff = (s1 / n1) - (s2 / n2)
            if (direction <= 0 and diff <= obs_diff + 1e-12) or (direction > 0 and diff >= obs_diff - 1e-12):
                extreme += 1
        return extreme / n_comb

    rng = random.Random(int(digest({'seed': seed, 'p': sorted(round(float(x), 8) for x in y_p), 'np': sorted(round(float(x), 8) for x in y_not_p)})[:16], 16))
    extreme = 1
    pool = list(combined)
    for _ in range(n_perm):
        rng.shuffle(pool)
        s1 = sum(pool[:n1])
        s2 = total_sum - s1
        diff = (s1 / n1) - (s2 / n2)
        if (direction <= 0 and diff <= obs_diff + 1e-12) or (direction > 0 and diff >= obs_diff - 1e-12):
            extreme += 1
    return extreme / (n_perm + 1)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values."""
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in p_values):
        raise L2Error('Invalid hypothesis p-value')
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(float(p) for p in p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, pval) in enumerate(indexed):
        mult = m - rank
        running_max = max(running_max, min(1.0, pval * mult))
        adjusted[orig_idx] = running_max
    return adjusted


def clustered_bootstrap(
    cases: Sequence[dict[str, Any]],
    stat_fn: Callable[[Sequence[dict[str, Any]]], float | None],
    *,
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: str = 'l2-v03-cluster-boot',
) -> dict[str, Any]:
    """Family-clustered bootstrap confidence interval (Section 7)."""
    point = stat_fn(cases)
    by_family: dict[str, list[dict[str, Any]]] = {}
    for c in cases:
        fam = str(c.get('family_id') or 'unknown_family')
        by_family.setdefault(fam, []).append(c)
    fams = sorted(by_family.keys())
    if len(fams) < 2 or point is None:
        return {'estimate': point, 'ci_95': [None, None], 'family_count': len(fams)}

    rng = random.Random(int(digest({'seed': seed, 'fams': fams, 'n': len(cases)})[:16], 16))
    samples = []
    for _ in range(n_boot):
        drawn_fams = [rng.choice(fams) for _ in range(len(fams))]
        drawn_cases = [item for f in drawn_fams for item in by_family[f]]
        val = stat_fn(drawn_cases)
        if val is not None and math.isfinite(val):
            samples.append(float(val))
    samples.sort()
    if not samples:
        return {'estimate': point, 'ci_95': [None, None], 'family_count': len(fams)}
    lo_idx = max(0, min(len(samples) - 1, int(round((alpha / 2.0) * (len(samples) - 1)))))
    hi_idx = max(0, min(len(samples) - 1, int(round((1.0 - alpha / 2.0) * (len(samples) - 1)))))
    return {'estimate': point, 'ci_95': [samples[lo_idx], samples[hi_idx]], 'family_count': len(fams)}
