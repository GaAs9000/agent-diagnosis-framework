"""Two-stage sparse value curve planning, idempotent execution, and verdict inference (L2 v0.3 Section 3)."""
from __future__ import annotations
import concurrent.futures
import copy
import random
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable
from .evidence.l1 import L1Package
from .io import L2Error, Unsupported, digest, read_json, sha_file, with_header, write_json, write_jsonl
from .stats import lower_drop, upper

VERDICTS = ('concentrated_turn', 'concentrated_interval', 'diffuse_loss', 'out_of_reach')


class TokenBucketRateLimiter:
    """Global thread-safe token bucket limiter for API requests/min and tokens/min."""

    def __init__(self, rpm: float | None = None, tpm: float | None = None):
        self.rpm = float(rpm) if rpm and rpm > 0 else None
        self.tpm = float(tpm) if tpm and tpm > 0 else None
        self._lock = threading.Lock()
        self._req_tokens = self.rpm if self.rpm else 0.0
        self._tok_tokens = self.tpm if self.tpm else 0.0
        self._last = time.monotonic()

    def acquire(self, estimated_tokens: float = 1000.0) -> None:
        if self.rpm is None and self.tpm is None:
            return
        while True:
            with self._lock:
                now_ts = time.monotonic()
                elapsed = max(0.0, now_ts - self._last)
                self._last = now_ts
                if self.rpm is not None:
                    self._req_tokens = min(self.rpm, self._req_tokens + elapsed * (self.rpm / 60.0))
                if self.tpm is not None:
                    self._tok_tokens = min(self.tpm, self._tok_tokens + elapsed * (self.tpm / 60.0))
                req_ok = self.rpm is None or self._req_tokens >= 1.0
                tok_ok = self.tpm is None or self._tok_tokens >= min(estimated_tokens, self.tpm)
                if req_ok and tok_ok:
                    if self.rpm is not None:
                        self._req_tokens -= 1.0
                    if self.tpm is not None:
                        self._tok_tokens -= min(estimated_tokens, self.tpm)
                    return
            time.sleep(0.02)


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
    T = parent.turns
    out = parent.outcome()
    if not out['valid'] or out['y'] is None:
        raise L2Error(f'Parent trajectory outcome is invalid ({out["reason"]}); cannot build value curve')
    Y = float(out['y'])
    cps = parent.checkpoints()
    grid_k = [k for k in range(0, T, stride) if k in cps and cps[k]['eligible']]
    if not grid_k:
        grid_k = [k for k in range(0, T, stride)]
    grid = [*grid_k, T]

    stage1_branches = len(grid_k) * n1
    stage2_max_branches = min(T, stride + 1) * n2
    total_est_branches = stage1_branches + stage2_max_branches
    seed = digest({'episode_id': parent.episode_id, 'index_sha256': parent.index_sha256, 'n1': n1, 'n2': n2, 'stride': stride})[:16]

    return with_header(
        {'parent_index_sha256': parent.index_sha256},
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
    """Implements the exact v0.3 Section 3 `curve()` algorithm."""
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

    turn_thresh = max(0.10, 2.0 * (v1[0] - float(Y)) / float(T))
    turn_lb = lower_drop(s2.get(j - 1, []), _s2_target(j), alpha / max(1, b - a))

    interval_thresh = ((b - a) / float(T)) * (v1[0] - float(Y))
    interval_lb = lower_drop(s2.get(a, []), _s2_target(b), alpha)

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
    max_retries_per_slot: int = 2,
    max_invalid_rate: float = 0.05,
    rpm: float | None = None,
    tpm: float | None = None,
) -> dict[str, Any]:
    """Executes Stage 1 and Stage 2 branches idempotently under `l2_dir/branches/<trial>/` (Section 3.2)."""
    if resume_fn is None:
        try:
            from c3_sandbox.l1.core.orchestrator import run as l1_run
        except ImportError as exc:
            raise Unsupported('Install c3_sandbox L1 in this virtualenv or supply resume_fn') from exc
        cfg = parent.json('raw/resolved_config.json')

        def _default_resume(parent_pkg: L1Package, k: int, branch_dir: Path) -> None:
            l1_run(copy.deepcopy(cfg), branch_dir, parent=parent_pkg.root, checkpoint_id=f'cp{k:06d}')

        runner = _default_resume
    else:
        runner = resume_fn

    T = int(plan['turns'])
    Y = float(plan['terminal_outcome'])
    n1 = int(plan['n1'])
    n2 = int(plan['n2'])
    alpha = float(plan['alpha'])
    concurrency = int(max_concurrency or plan.get('max_concurrency', 4))
    seed_str = str(plan.get('seed') or 'l2-v03-seed')

    branches_root = Path(l2_dir) / 'branches'
    branches_root.mkdir(parents=True, exist_ok=True)
    limiter = TokenBucketRateLimiter(rpm=rpm, tpm=tpm)
    outcomes_rows: list[dict[str, Any]] = []
    outcomes_lock = threading.Lock()

    def _run_slot(stage: str, k: int, slot_idx: int) -> float | None:
        base_trial = f'k{k:03d}-{stage}-i{slot_idx:02d}'
        for attempt in range(max_retries_per_slot + 1):
            trial_name = base_trial if attempt == 0 else f'{base_trial}-r{attempt}'
            branch_dir = branches_root / trial_name

            if (branch_dir / 'manifest.json').is_file():
                try:
                    existing_pkg = L1Package(branch_dir)
                    existing_out = existing_pkg.outcome()
                    if existing_out['valid'] and existing_out['y'] is not None:
                        row = {
                            'trial': trial_name,
                            'k': k,
                            'stage': stage,
                            'valid': True,
                            'y': float(existing_out['y']),
                            'modelVersion': existing_out['modelVersion'],
                            'reused': True,
                        }
                        with outcomes_lock:
                            outcomes_rows.append(row)
                        return float(existing_out['y'])
                except Exception:
                    pass

            if branch_dir.exists():
                shutil.rmtree(branch_dir)

            limiter.acquire()
            try:
                runner(parent, k, branch_dir)
                b_pkg = L1Package(branch_dir)
                b_out = b_pkg.outcome()
            except Exception as exc:
                b_out = {'valid': False, 'y': None, 'reason': str(exc), 'modelVersion': None}

            row = {
                'trial': trial_name,
                'k': k,
                'stage': stage,
                'valid': bool(b_out['valid'] and b_out['y'] is not None),
                'y': float(b_out['y']) if b_out['valid'] and b_out['y'] is not None else None,
                'modelVersion': b_out.get('modelVersion'),
                'reused': False,
            }
            with outcomes_lock:
                outcomes_rows.append(row)
            if row['valid'] and row['y'] is not None:
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
        res = _run_stage_batch('s2', s2_turns, n2)
        s2_samples_holder.update(res)
        return res

    inferred = infer_curve_from_samples(
        T=T,
        Y=Y,
        grid=[int(x) for x in plan['grid']],
        s1=s1_samples,
        s2_runner_or_map=_s2_loader,
        alpha=alpha,
    )

    outcomes_rows.sort(key=lambda r: (r['stage'], r['k'], r['trial']))
    outcomes_path = write_jsonl(Path(l2_dir) / 'outcomes.jsonl', outcomes_rows)

    total_attempted = len(outcomes_rows)
    invalid_count = sum(1 for r in outcomes_rows if not r['valid'])
    invalid_rate = (invalid_count / total_attempted) if total_attempted > 0 else 0.0
    excluded = invalid_rate > max_invalid_rate

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
            'excluded_high_invalid_rate': excluded,
            **inferred,
        },
    )
    write_json(Path(l2_dir) / 'curve.json', curve_doc)
    return curve_doc
