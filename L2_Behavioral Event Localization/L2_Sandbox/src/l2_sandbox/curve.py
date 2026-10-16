"""Two-stage sparse value curve planning, idempotent execution, and verdict inference (L2 v0.3 Section 3)."""
from __future__ import annotations
import concurrent.futures
import copy
import random
import inspect
import math
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable
from .evidence.l1 import L1Package
from .io import IntegrityError, L2Error, Unsupported, digest, read_json, read_jsonl, sha_file, validate_output_location, with_header, write_json, write_jsonl
from .stats import lower_drop, upper

VERDICTS = ('concentrated_turn', 'concentrated_interval', 'diffuse_loss', 'out_of_reach')


class TokenBucketRateLimiter:
    """Global thread-safe token bucket limiter for API requests/min and tokens/min."""

    def __init__(self, rpm: float | None = None, tpm: float | None = None):
        for value in (rpm, tpm):
            if value is not None and (not math.isfinite(float(value)) or float(value) <= 0):
                raise L2Error('Rate limits must be finite and positive')
        self.rpm = float(rpm) if rpm is not None else None
        self.tpm = float(tpm) if tpm is not None else None
        self._lock = threading.Lock()
        self._req_tokens = max(1.0, self.rpm) if self.rpm else 0.0
        self._tok_tokens = self.tpm if self.tpm else 0.0
        self._last = time.monotonic()

    def acquire(self, estimated_tokens: float = 1000.0, timeout: float | None = None) -> bool:
        estimate = float(estimated_tokens)
        if not math.isfinite(estimate) or estimate < 0:
            raise L2Error('Token estimate must be finite and nonnegative')
        if self.tpm is not None and estimate > self.tpm:
            return False
        deadline = time.monotonic() + max(0.0, float(timeout)) if timeout is not None else time.monotonic() + 60.0
        while True:
            with self._lock:
                now_ts = time.monotonic()
                elapsed = max(0.0, now_ts - self._last)
                self._last = now_ts
                if self.rpm is not None:
                    self._req_tokens = min(max(1.0, self.rpm), self._req_tokens + elapsed * self.rpm / 60.0)
                if self.tpm is not None:
                    self._tok_tokens = min(self.tpm, self._tok_tokens + elapsed * self.tpm / 60.0)
                if (self.rpm is None or self._req_tokens >= 1.0) and (self.tpm is None or self._tok_tokens >= estimate):
                    if self.rpm is not None:
                        self._req_tokens -= 1.0
                    if self.tpm is not None:
                        self._tok_tokens -= estimate
                    return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))

    def reconcile(self, estimate: float, actual: float | None) -> None:
        # Unknown usage retains the reservation; known usage replaces it, including debt.
        if self.tpm is not None and actual is not None:
            with self._lock:
                self._tok_tokens = min(self.tpm, self._tok_tokens + float(estimate) - float(actual))


def sampling_identity(plan: dict[str, Any]) -> dict[str, Any]:
    """Paid evidence depends on sampling inputs, never downstream inference code."""
    header = plan.get('input_sha256') or {}
    return {'parent_index_sha256': header.get('parent_index_sha256'),
            'actor_config_sha256': header.get('actor_config_sha256'),
            **{key: plan[key] for key in ('episode_id', 'turns', 'grid', 'n1', 'n2', 'stride', 'seed')}}


def ensure_plan_identity(output_dir: Path, plan: dict[str, Any]) -> None:
    existing = Path(output_dir) / 'plan.json'
    if existing.is_file() and sampling_identity(read_json(existing)) != sampling_identity(plan):
        raise IntegrityError('Existing output belongs to a different parent or sampling plan')
    if not existing.is_file() and Path(output_dir).exists():
        if any((Path(output_dir) / name).exists() for name in ('branches', 'outcomes.jsonl', 'curve.json')):
            raise IntegrityError('Existing execution artifacts lack plan identity')


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def build_curve_plan(
    parent: L1Package,
    *,
    n1: int = 12,
    n2: int = 24,
    stride: int = 4,
    alpha: float = 0.05,
    max_concurrency: int = 4,
    cost_per_branch_usd: float = 0.01,
) -> dict[str, Any]:
    """Builds the deterministic `plan.json` for a parent trajectory without executing branches (Section 3.2)."""
    if min(n1, n2, stride, max_concurrency) <= 0 or not 0 < alpha < 1:
        raise L2Error('Curve counts/stride/concurrency must be positive and alpha in (0,1)')
    if parent.start_turn != 0 or parent.manifest.get('lineage') or parent.turns < 1:
        raise L2Error('Curve parent must be a root trajectory with at least one turn')
    parameters = parent.json('raw/resolved_config.json').get('model', {}).get('parameters', {})
    temperature = parameters.get('temperature')
    if (type(temperature) not in (int, float) or temperature != 1.0
            or any(parameters.get(key) is not None for key in ('seed', 'random_seed', 'randomSeed'))):
        raise L2Error('Value curves require actor temperature=1.0 and no fixed sampling seed')
    T = parent.turns
    out = parent.outcome()
    if not out['valid'] or out['y'] is None:
        raise L2Error(f'Parent trajectory outcome is invalid ({out["reason"]}); cannot build value curve')
    Y = float(out['y'])
    # Every nonterminal point can be selected in Stage 2; filtering the grid changes the algorithm.
    for k in range(T):
        parent.validate_resume_checkpoint(k)
    grid_k = list(range(0, T, stride))
    grid = [*grid_k, T]

    stage1_branches = len(grid_k) * n1
    stage2_max_branches = min(T, stride + 1) * n2
    total_est_branches = stage1_branches + stage2_max_branches
    seed = digest({'episode_id': parent.episode_id, 'index_sha256': parent.index_sha256, 'n1': n1, 'n2': n2, 'stride': stride})[:16]

    return with_header(
        {'parent_index_sha256': parent.index_sha256, 'parent_content_sha256': parent.content_identity(),
         'actor_config_sha256': digest(parent.json('raw/resolved_config.json'))},
        {
            'episode_id': parent.episode_id,
            'family_id': parent.family_id,
            'turns': T,
            'terminal_outcome': Y,
            'grid': grid,
            'stage1_checkpoints': grid_k,
            'n1': n1,
            'n2': n2,
            'stride': stride,
            'alpha': alpha,
            'inference_method': {'nuisance_alpha_fraction': 0.2, 'drop_alpha_fraction': 0.8,
                'scope': 'per-comparison nuisance guard; no joint-error guarantee'},
            'max_concurrency': max_concurrency,
            'seed': seed,
            'estimated_branches': {
                'stage1': stage1_branches,
                'stage2_max': stage2_max_branches,
                'total_max': total_est_branches,
            },
            'estimated_cost_usd': round(total_est_branches * cost_per_branch_usd, 4),
        },
    )


def infer_curve_from_samples(
    T: int,
    Y: float,
    grid: list[int],
    s1: dict[int, list[float]],
    s2_runner_or_map: dict[int, list[float]] | Callable[[list[int]], dict[int, list[float]]],
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Infer the sparse curve using the retained D1 V0 upper-bound policy.

    The alpha split and bounds remain unchanged; calibration is conditional on
    tested scenarios and does not provide a general joint-error guarantee.
    """
    full_grid = sorted(set(grid) | {T})
    v0_upper = upper(s1.get(0, []), alpha)
    v1 = {k: _mean(vals) for k, vals in s1.items()} | {T: float(Y)}

    if v0_upper < 0.25:
        return {
            'verdict': 'out_of_reach',
            'turn': None,
            'interval': None,
            'v0_upper': v0_upper,
            'v1': {str(k): v for k, v in sorted(v1.items())},
            'v2': {},
            'v_hat': {str(k): v for k, v in sorted(v1.items())},
            'turn_drops': {},
        }

    pairs = list(zip(full_grid[:-1], full_grid[1:]))
    a, b = max(pairs, key=lambda ab: v1[ab[0]] - v1[ab[1]])
    s2_turns = [k for k in range(a, b + 1) if k < T]

    if callable(s2_runner_or_map):
        s2 = s2_runner_or_map(s2_turns)
    else:
        s2 = {k: s2_runner_or_map.get(k, []) for k in s2_turns}

    v2 = {k: _mean(vals) for k, vals in s2.items()} | {T: float(Y)}

    def _s2_target(k: int) -> list[float] | float:
        return float(Y) if k == T else s2.get(k, [])

    def _mean_drop(t_prev: int, t_curr: int) -> float:
        return v2[t_prev] - v2[t_curr]

    turn_drops = {t: _mean_drop(t - 1, t) for t in range(a + 1, b + 1)}
    j = max(range(a + 1, b + 1), key=lambda t: turn_drops[t])

    # V0 is estimated: a low point estimate otherwise makes the null loss threshold too small.
    # Fixed alpha allocation guards that nuisance separately from each drop comparison.
    nuisance_alpha, drop_alpha = 0.2 * alpha, 0.8 * alpha
    threshold_v0_upper = upper(s1.get(0, []), nuisance_alpha)
    nuisance_loss = max(0.0, threshold_v0_upper - float(Y))
    turn_thresh = max(0.10, 2.0 * nuisance_loss / float(T))
    turn_alpha = drop_alpha / max(1, b - a)
    turn_lb = lower_drop(s2.get(j - 1, []), _s2_target(j), turn_alpha)

    interval_thresh = ((b - a) / float(T)) * nuisance_loss
    interval_lb = lower_drop(s2.get(a, []), _s2_target(b), drop_alpha)

    v_hat = {str(k): v for k, v in sorted(v1.items())}
    for k, val in sorted(v2.items()):
        v_hat[str(k)] = val

    if turn_lb > turn_thresh:
        verdict = 'concentrated_turn'
        loc_turn = j
        loc_interval = [a, b]
    elif interval_lb > interval_thresh:
        verdict = 'concentrated_interval'
        loc_turn = None
        loc_interval = [a, b]
    else:
        verdict = 'diffuse_loss'
        loc_turn = None
        loc_interval = [a, b]

    return {
        'verdict': verdict,
        'turn': loc_turn,
        'interval': loc_interval,
        'candidate_turn': j,
        'v0_upper': v0_upper,
        'v1': {str(k): v for k, v in sorted(v1.items())},
        'v2': {str(k): v for k, v in sorted(v2.items())},
        'v_hat': v_hat,
        'turn_drops': {str(t): d for t, d in sorted(turn_drops.items())},
        'thresholds': {
            'threshold_v0_upper': threshold_v0_upper,
            'nuisance_alpha': nuisance_alpha,
            'drop_alpha': drop_alpha,
            'turn_alpha': turn_alpha,
            'error_scope': 'per-comparison nuisance guard; no joint-error guarantee',
            'turn_threshold': turn_thresh,
            'turn_lower_bound': turn_lb,
            'interval_threshold': interval_thresh,
            'interval_lower_bound': interval_lb,
        },
    }


def execute_curve(
    parent: L1Package,
    l2_dir: Path,
    plan: dict[str, Any],
    *,
    resume_fn: Callable[..., Any] | None = None,
    max_concurrency: int | None = None,
    cache_only: bool = False,
    max_invalid_rate: float = 0.05,
    rpm: float | None = None,
    tpm: float | None = None,
) -> dict[str, Any]:
    """Executes Stage 1 and Stage 2 branches idempotently under `l2_dir/branches/<trial>/` (Section 3.2)."""
    l2_dir = Path(l2_dir)
    validate_output_location(parent.root, l2_dir)
    # Re-read immutable input rather than trusting an older in-memory package instance.
    fresh_parent = L1Package(parent.root)
    expected = build_curve_plan(fresh_parent, n1=plan['n1'], n2=plan['n2'], stride=plan['stride'],
                                alpha=plan['alpha'], max_concurrency=plan['max_concurrency'])
    # Sampling identity survives changes to inference, accounting, and implementation.
    if sampling_identity(expected) != sampling_identity(plan):
        raise IntegrityError('Curve plan no longer matches parent/actor/sampling inputs')
    ensure_plan_identity(l2_dir, plan)
    parent = fresh_parent
    plan = {**plan, 'terminal_outcome': expected['terminal_outcome'],
            'stage1_checkpoints': expected['stage1_checkpoints'],
            'estimated_branches': expected['estimated_branches']}
    limiter = TokenBucketRateLimiter(rpm=rpm, tpm=tpm)
    if not 0 <= max_invalid_rate < 1:
        raise L2Error('Invalid-rate threshold must be in [0,1)')
    if resume_fn is None:
        try:
            from c3_sandbox.l1.core.orchestrator import run as l1_run
        except ImportError as exc:
            raise Unsupported('Install c3_sandbox L1 in this virtualenv or supply resume_fn') from exc
        if 'request_limiter' not in inspect.signature(l1_run).parameters:
            raise Unsupported('Installed L1 runner lacks per-request limiter support')
        cfg = parent.json('raw/resolved_config.json')

        def runner(parent_pkg: L1Package, k: int, branch_dir: Path, *, request_limiter) -> None:
            l1_run(copy.deepcopy(cfg), branch_dir, parent=parent_pkg.root,
                   checkpoint_id=f'cp{k:06d}', request_limiter=request_limiter)
    else:
        runner = resume_fn
    accepts_limiter = ('request_limiter' in inspect.signature(runner).parameters or
                       any(p.kind == inspect.Parameter.VAR_KEYWORD for p in inspect.signature(runner).parameters.values()))
    if (rpm is not None or tpm is not None) and not accepts_limiter:
        raise Unsupported('Configured limits require resume_fn(..., request_limiter=...)')

    T, Y = int(plan['turns']), float(plan['terminal_outcome'])
    n1, n2, alpha = int(plan['n1']), int(plan['n2']), float(plan['alpha'])
    concurrency = int(max_concurrency if max_concurrency is not None else plan.get('max_concurrency', 4))
    if concurrency < 1:
        raise L2Error('Concurrency must be positive')
    seed_str = str(plan['seed'])
    branches_root = l2_dir / 'branches'
    outcomes_path = l2_dir / 'outcomes.jsonl'
    rows = read_jsonl(outcomes_path) if outcomes_path.is_file() else []
    outcome_header = with_header({'sampling_sha256': digest(sampling_identity(plan))}, {})
    # Derived rows are rebuilt from each visited package under the current outcome policy.
    outcomes_by_trial = {}
    trial_pattern = re.compile(r'k([0-9]+)-(s[12])-i([0-9]+)(?:-r([1-9][0-9]*))?')
    previous = {row['trial']: row for row in rows}
    # Reproject retained attempts for accounting, without a global lineage gate or rehash.
    retained = set(previous) | {path.name for path in branches_root.iterdir()} if branches_root.exists() else set(previous)
    for name in sorted(retained):
        match = trial_pattern.fullmatch(name)
        if match is None:
            continue
        directory = branches_root / name
        try:
            if (directory / 'manifest.json').is_file():
                out = L1Package(directory).outcome()
            else:
                out = {'valid': False, 'y': None,
                       'reason': previous.get(name, {}).get('reason') or 'interrupted_attempt',
                       'termination': previous.get(name, {}).get('termination'),
                       'modelVersion': None}
        except (L2Error, KeyError, TypeError, ValueError) as exc:
            out = {'valid': False, 'y': None, 'reason': str(exc), 'termination': None, 'modelVersion': None}
        valid = bool(out['valid'] and out['y'] is not None)
        outcomes_by_trial[name] = {**outcome_header, 'trial': name, 'k': int(match[1]), 'stage': match[2],
            'valid': valid, 'y': float(out['y']) if valid else None, 'reason': out.get('reason'),
            'termination': out.get('termination'), 'modelVersion': out.get('modelVersion'),
            'reused': False, 'selected': False}
    write_json(l2_dir / 'plan.json', plan)
    branches_root.mkdir(parents=True, exist_ok=True)
    outcomes_lock = threading.Lock()

    def persist(row: dict) -> None:
        with outcomes_lock:
            outcomes_by_trial[row['trial']] = {**outcome_header, **row}
            ordered = sorted(outcomes_by_trial.values(), key=lambda r: (r['stage'], r['k'], r['trial']))
            write_jsonl(outcomes_path, ordered)

    maximum_valid = plan['estimated_branches']['total_max']

    def irrecoverable() -> bool:
        invalid = sum(not row['valid'] for row in outcomes_by_trial.values() if row.get('selected'))
        return invalid > max_invalid_rate * maximum_valid / (1 - max_invalid_rate)

    def _run_slot(stage: str, k: int, slot_idx: int) -> float | None:
        base_trial = f'k{k:03d}-{stage}-i{slot_idx:02d}'
        names = {name for name in previous if name == base_trial or name.startswith(base_trial + '-r')}
        names.update(path.name for path in branches_root.glob(base_trial + '*')
                     if trial_pattern.fullmatch(path.name) and
                     (path.name == base_trial or path.name.startswith(base_trial + '-r')))
        attempts = sorted(int(trial_pattern.fullmatch(name)[4] or 0) for name in names)
        next_attempt = 0
        while True:
            if attempts:
                attempt = attempts.pop(0)
                next_attempt = max(next_attempt, attempt + 1)
            else:
                if cache_only:
                    raise L2Error(f'Cache-only analysis requires missing sample {base_trial}; explicit sampling required')
                with outcomes_lock:
                    if irrecoverable():
                        return None
                attempt = next_attempt
                next_attempt += 1
            trial_name = base_trial if attempt == 0 else f'{base_trial}-r{attempt}'
            branch_dir = branches_root / trial_name
            if not branch_dir.resolve().is_relative_to(branches_root.resolve()):
                raise IntegrityError(f'Branch output redirects outside branches: {trial_name}')
            # Never rewrite a directory, even an interrupted attempt without a manifest.
            if branch_dir.exists():
                if (branch_dir / 'manifest.json').is_file():
                    b_pkg = L1Package(branch_dir)
                    b_pkg.validate_branch(parent, k)
                    b_out = b_pkg.outcome()
                else:
                    b_out = {'valid': False, 'y': None,
                             'reason': previous.get(trial_name, {}).get('reason') or 'interrupted_attempt',
                             'termination': previous.get(trial_name, {}).get('termination'),
                             'modelVersion': None}
            else:
                if trial_name in previous:
                    old = previous[trial_name]
                    if old.get('valid'):
                        raise IntegrityError(f'Cached valid branch is missing: {trial_name}')
                    persist({**old, 'termination': old.get('termination'), **outcome_header, 'selected': True})
                    continue
                try:
                    if accepts_limiter:
                        runner(parent, k, branch_dir, request_limiter=limiter)
                    else:
                        runner(parent, k, branch_dir)
                except Exception as exc:
                    b_out = {'valid': False, 'y': None, 'reason': str(exc), 'termination': None, 'modelVersion': None}
                else:
                    if not (branch_dir / 'manifest.json').is_file():
                        b_out = {'valid': False, 'y': None, 'reason': 'missing_manifest', 'termination': None, 'modelVersion': None}
                    else:
                        b_pkg = L1Package(branch_dir)
                        b_pkg.validate_branch(parent, k)
                        b_out = b_pkg.outcome()
            row = {'trial': trial_name, 'k': k, 'stage': stage,
                   'valid': bool(b_out['valid'] and b_out['y'] is not None),
                   'y': float(b_out['y']) if b_out['valid'] and b_out['y'] is not None else None,
                   'reason': b_out.get('reason'), 'termination': b_out.get('termination'),
                   'modelVersion': b_out.get('modelVersion'), 'reused': False, 'selected': True}
            persist(row)
            if row['valid']:
                return float(row['y'])
        return None

    def _run_stage_batch(stage: str, checkpoints_k: list[int], reps: int) -> dict[int, list[float]]:
        tasks = [(k, i) for i in range(reps) for k in checkpoints_k]
        rng = random.Random(int(digest({'seed': seed_str, 'stage': stage})[:16], 16))
        rng.shuffle(tasks)

        results_by_k: dict[int, list[float]] = {k: [] for k in checkpoints_k}
        if concurrency <= 1:
            for k, i in tasks:
                val = _run_slot(stage, k, i)
                if val is not None:
                    results_by_k[k].append(val)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                fut_map = {pool.submit(_run_slot, stage, k, i): k for k, i in tasks}
                for fut in concurrent.futures.as_completed(fut_map):
                    k = fut_map[fut]
                    val = fut.result()
                    if val is not None:
                        results_by_k[k].append(val)
        return results_by_k

    s1_k = [int(k) for k in plan['stage1_checkpoints']]
    s1_samples = _run_stage_batch('s1', s1_k, n1)

    s2_samples_holder: dict[int, list[float]] = {}

    def _s2_loader(s2_turns: list[int]) -> dict[int, list[float]]:
        nonlocal maximum_valid
        maximum_valid = len(s1_k) * n1 + len(s2_turns) * n2
        res = _run_stage_batch('s2', s2_turns, n2)
        s2_samples_holder.update(res)
        return res

    complete_stage1 = all(len(s1_samples[k]) == n1 for k in s1_k)
    if complete_stage1:
        inferred = infer_curve_from_samples(
            T=T, Y=Y, grid=[int(x) for x in plan['grid']], s1=s1_samples,
            s2_runner_or_map=_s2_loader, alpha=alpha,
        )
        complete = all(len(samples) == n2 for samples in s2_samples_holder.values())
    else:
        inferred = {'verdict': None, 'turn': None, 'interval': None, 'v_hat': {}, 'turn_drops': {}}
        complete = False
    if not complete:
        inferred.update(verdict=None, turn=None, interval=None, candidate_turn=None,
                        v_hat={}, v2={}, turn_drops={})
        inferred.pop('thresholds', None)
    if not s2_samples_holder:
        maximum_valid = len(s1_k) * n1
    outcomes_rows = sorted(outcomes_by_trial.values(), key=lambda r: (r['stage'], r['k'], r['trial']))
    write_jsonl(outcomes_path, outcomes_rows)

    total_attempted = len(outcomes_rows)
    invalid_count = sum(1 for r in outcomes_rows if not r['valid'])
    selected = [row for row in outcomes_rows if row.get('selected')]
    selected_invalid = sum(not row['valid'] for row in selected)
    invalid_rate = selected_invalid / len(selected) if selected else 0.0
    excluded = invalid_rate > max_invalid_rate

    valid_by_k: dict[int, list[dict[str, Any]]] = {}
    for row in selected:
        if row['valid']:
            valid_by_k.setdefault(int(row['k']), []).append(row)
    request_budget_rate_by_k = {
        str(k): sum(r.get('termination') == 'request_budget' for r in rows_k) / len(rows_k)
        for k, rows_k in sorted(valid_by_k.items()) if rows_k
    }
    terminations_by_k: dict[str, dict[str, int]] = {}
    for k, rows_k in sorted(valid_by_k.items()):
        counts: dict[str, int] = {}
        for r in rows_k:
            term = str(r.get('termination') or 'unknown')
            counts[term] = counts.get(term, 0) + 1
        terminations_by_k[str(k)] = dict(sorted(counts.items()))

    if excluded:
        inferred.update(verdict=None, turn=None, interval=None)
    curve_doc = with_header(
        {
            'plan_sha256': sha_file(Path(l2_dir) / 'plan.json') if (Path(l2_dir) / 'plan.json').is_file() else digest(plan),
            'outcomes_sha256': sha_file(outcomes_path),
        },
        {
            'episode_id': parent.episode_id,
            'family_id': parent.family_id,
            'turns': T,
            'terminal_outcome': Y,
            'total_attempted_branches': total_attempted,
            'invalid_branches': invalid_count,
            'invalid_rate': invalid_rate,
            'invalid_rate_scope': 'selected_attempts',
            'selected_attempted_branches': len(selected),
            'selected_invalid_branches': selected_invalid,
            'excluded_incomplete_samples': not complete,
            'sampling_status': 'complete' if complete else 'incomplete',
            'excluded_high_invalid_rate': excluded,
            'request_budget_rate_by_k': request_budget_rate_by_k,
            'terminations_by_k': terminations_by_k,
            **inferred,
        },
    )
    write_json(Path(l2_dir) / 'curve.json', curve_doc)
    return curve_doc
