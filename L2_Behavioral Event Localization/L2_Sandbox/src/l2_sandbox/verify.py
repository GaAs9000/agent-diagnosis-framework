"""Targeted continuation verification: partition, intervention, and two-point tests (L2 v0.4 Section 6)."""
from __future__ import annotations
import concurrent.futures
import copy
import inspect
import math
import random
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Sequence
from .config import DELIVER_REMINDER, RULE_REMINDER_TEMPLATE
from .evidence.l1 import L1Package
from .facts.build import build_facts
from .grades import compute_item_deductions
from .io import IntegrityError, L2Error, Unsupported, digest, read_json, read_jsonl, sha_file, validate_output_location, with_header, write_json, write_jsonl
from .predicates import eval_predicate_on_turn, validate_predicate
from .stats import holm_adjust, lower_drop, permutation_one_sided, wilson_bounds


class TokenBucketRateLimiter:
    """Thread-safe token-bucket rate limiter over RPM (requests/min) and TPM (tokens/min)."""

    def __init__(
        self,
        rpm: float | None = None,
        tpm: float | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        for label, limit in (('rpm', rpm), ('tpm', tpm)):
            if limit is not None and (type(limit) not in (int, float) or not math.isfinite(limit) or limit <= 0):
                raise L2Error(f'{label} must be positive and finite')
        self.rpm = float(rpm) if rpm is not None else None
        self.tpm = float(tpm) if tpm is not None else None
        self._clock = clock
        self._sleep = sleeper
        self._lock = threading.Lock()
        now = self._clock()
        self._req_tokens = self.rpm if self.rpm is not None else 0.0
        self._tok_tokens = self.tpm if self.tpm is not None else 0.0
        self._last = now

    def _refill(self, now: float) -> None:
        elapsed = max(0.0, now - self._last)
        self._last = now
        if self.rpm is not None:
            self._req_tokens = min(self.rpm, self._req_tokens + elapsed * (self.rpm / 60.0))
        if self.tpm is not None:
            self._tok_tokens = min(self.tpm, self._tok_tokens + elapsed * (self.tpm / 60.0))

    def acquire(self, estimated_tokens: float | int = 1000, *, timeout: float | None = None) -> bool:
        if self.rpm is None and self.tpm is None:
            return True
        waited = 0.0
        while True:
            with self._lock:
                now = self._clock()
                self._refill(now)
                wait_req = 0.0
                wait_tok = 0.0
                if self.rpm is not None and self._req_tokens < 1.0:
                    wait_req = (1.0 - self._req_tokens) / (self.rpm / 60.0)
                if self.tpm is not None:
                    need = min(float(estimated_tokens or 0), self.tpm)
                    if self._tok_tokens < need:
                        wait_tok = (need - self._tok_tokens) / (self.tpm / 60.0)
                wait_s = max(wait_req, wait_tok)
                if wait_s <= 1e-9:
                    if self.rpm is not None:
                        self._req_tokens -= 1.0
                    if self.tpm is not None:
                        self._tok_tokens -= min(float(estimated_tokens or 0), self.tpm)
                    return True
                if timeout is not None and waited + wait_s > timeout:
                    return False
            self._sleep(wait_s)
            waited += wait_s

    def reconcile(self, estimated_tokens: float | int | None, actual_tokens: float | int | None) -> None:
        if self.tpm is None or estimated_tokens is None or actual_tokens is None:
            return
        with self._lock:
            now = self._clock()
            self._refill(now)
            diff = min(float(estimated_tokens), self.tpm) - float(actual_tokens)
            self._tok_tokens = max(0.0, min(self.tpm, self._tok_tokens + diff))


def build_arm_intervention(
    arm: str,
    k: int,
    *,
    requirement: str | None = None,
) -> dict[str, Any] | None:
    """Builds the L1 `agent.intervention` dict for a continuation arm at checkpoint `c_k` (Section 6.2)."""
    if arm == 'plain':
        return None
    if arm == 'deliver':
        return {
            'type': 'user_message',
            'at_request': int(k) + 1,
            'message': DELIVER_REMINDER,
        }
    if arm == 'rule' or arm.startswith('rule:'):
        req = requirement
        if req is None and ':' in arm:
            req = arm.split(':', 1)[1].strip()
        if not req:
            raise L2Error('Rule intervention arm requires a non-empty requirement string')
        return {
            'type': 'user_message',
            'at_request': int(k) + 1,
            'message': RULE_REMINDER_TEMPLATE.format(rule=req, requirement=req),
        }
    raise L2Error(f'Unknown continuation arm: {arm!r} (expected plain, deliver, or rule)')


def normalize_arm_specs(
    k: int,
    arms: Sequence[str | dict[str, Any]],
    n: int = 24,
) -> list[dict[str, Any]]:
    """Normalizes arm specs into canonical dicts: `{arm_id, arm_type, k, n, requirement, intervention}`."""
    specs: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for entry in arms:
        if isinstance(entry, str):
            raw_arm = entry.strip()
            arm_k = int(k)
            arm_n = int(n)
            req = None
            if raw_arm.startswith('rule:'):
                arm_type = 'rule'
                req = raw_arm.split(':', 1)[1].strip()
                arm_id = f"rule_{digest(req)[:8]}"
            else:
                arm_type = raw_arm
                arm_id = raw_arm
        elif isinstance(entry, dict):
            arm_type = str(entry.get('arm_type') or entry.get('arm') or 'plain').strip()
            if arm_type.startswith('rule:'):
                req = entry.get('requirement') or arm_type.split(':', 1)[1].strip()
                arm_type = 'rule'
            else:
                req = entry.get('requirement')
            arm_k = int(entry.get('k', k))
            arm_n = int(entry.get('n', n))
            arm_id = str(entry.get('arm_id') or (f"rule_{digest(req)[:8]}" if arm_type == 'rule' and req else arm_type))
        else:
            raise L2Error(f'Invalid arm specification: {entry!r}')

        if arm_k < 0:
            raise L2Error(f'Checkpoint k must be >= 0, got {arm_k}')
        if arm_n < 1:
            raise L2Error(f'Branch count n must be >= 1, got {arm_n}')
        intervention = build_arm_intervention(arm_type, arm_k, requirement=req)
        key = (arm_id, arm_k)
        if key not in seen:
            seen.add(key)
            specs.append({
                'arm_id': arm_id,
                'arm_type': arm_type,
                'k': arm_k,
                'n': arm_n,
                'requirement': req,
                'intervention': intervention,
            })
    return specs


def _extract_item_scores(b_pkg: L1Package) -> dict[str, float]:
    try:
        og = b_pkg.json('official_grade.json')
        raw = og.get('official_raw') if isinstance(og, dict) else None
        if not isinstance(raw, dict):
            return {}
        # Support SWE-bench Pro evaluations:
        if raw.get('test_config') and ('test_results' in raw or 'passed' in raw):
            passed_tests = set()
            tr = raw.get('test_results')
            if isinstance(tr, dict) and isinstance(tr.get('tests'), list):
                for t in tr['tests']:
                    if isinstance(t, dict) and t.get('name') and t.get('status') == 'PASSED':
                        passed_tests.add(t['name'])
            out: dict[str, float] = {}
            for key in ('fail_to_pass', 'pass_to_pass'):
                val = raw['test_config'].get(key)
                if isinstance(val, str):
                    import ast
                    try:
                        val = ast.literal_eval(val)
                    except Exception:
                        val = []
                if isinstance(val, list):
                    for t_name in val:
                        out[t_name] = 1.0 if t_name in passed_tests else 0.0
            return out
        if not raw.get('dim1_pass') or not isinstance(raw.get('dim2_items'), list):
            return {}
        out: dict[str, float] = {}
        for it in raw['dim2_items']:
            rule = str(it.get('rule') or '')
            max_d = float(it.get('max_delta') or 0.0)
            delta = float(it.get('delta') or 0.0)
            if rule and max_d > 0:
                out[rule] = max(0.0, min(1.0, delta / max_d))
        return out
    except Exception:
        return {}


def _extract_outcome_value(row: dict[str, Any], outcome: str = 'total') -> float | None:
    if not row.get('valid') or row.get('y') is None:
        return None
    if outcome in ('total', '', None):
        return float(row['y'])
    item_scores = row.get('item_scores') or {}
    if str(outcome) in item_scores:
        return float(item_scores[str(outcome)])
    return float(row['y'])


def resume_arms(
    parent: L1Package,
    l2_dir: Path,
    *,
    k: int = 0,
    arms: Sequence[str | dict[str, Any]] = ('plain',),
    n: int = 24,
    outcome: str = 'total',
    resume_fn: Callable[..., Any] | None = None,
    max_concurrency: int = 4,
    cache_only: bool = False,
    max_invalid_rate: float = 0.05,
    rpm: float | None = None,
    tpm: float | None = None,
    reuse_dirs: Sequence[Path] | None = None,
    seed: int = 20260414,
) -> dict[str, Any]:
    """Runs or reuses continuation arms at specified checkpoint(s) and writes `outcomes.jsonl` (Section 6 & 9.2)."""
    l2_dir = Path(l2_dir)
    validate_output_location(parent.root, l2_dir)
    fresh_parent = L1Package(parent.root)
    parent = fresh_parent
    specs = normalize_arm_specs(k, arms, n=n)
    for s in specs:
        if s['k'] >= parent.turns:
            raise L2Error(f"Checkpoint k={s['k']} out of range for parent with T={parent.turns} turns")

    limiter = TokenBucketRateLimiter(rpm=rpm, tpm=tpm)
    if not 0 <= max_invalid_rate < 1:
        raise L2Error('Invalid-rate threshold must be in [0,1)')
    if max_concurrency < 1:
        raise L2Error('Concurrency must be positive')

    if resume_fn is None:
        try:
            from c3_sandbox.l1.core.orchestrator import run as l1_run
        except ImportError as exc:
            raise Unsupported('Install c3_sandbox L1 in this virtualenv or supply resume_fn') from exc
        if 'request_limiter' not in inspect.signature(l1_run).parameters:
            raise Unsupported('Installed L1 runner lacks per-request limiter support')
        base_cfg = parent.json('raw/resolved_config.json')

        def runner(
            parent_pkg: L1Package,
            ck: int,
            branch_dir: Path,
            *,
            intervention: dict[str, Any] | None = None,
            request_limiter: Any = None,
        ) -> None:
            cfg = copy.deepcopy(base_cfg)
            if intervention is not None:
                cfg.setdefault('agent', {})['intervention'] = {
                    'at_request': int(intervention['at_request']),
                    'message': str(intervention['message']),
                }
            else:
                cfg.get('agent', {}).pop('intervention', None)
            l1_run(
                cfg,
                branch_dir,
                parent=parent_pkg.root,
                checkpoint_id=f'cp{ck:06d}',
                request_limiter=request_limiter,
            )
    else:
        runner = resume_fn

    sig_params = inspect.signature(runner).parameters
    accepts_limiter = (
        'request_limiter' in sig_params
        or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values())
    )
    accepts_intervention = (
        'intervention' in sig_params
        or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values())
    )

    branches_root = l2_dir / 'branches'
    branches_root.mkdir(parents=True, exist_ok=True)
    outcomes_path = l2_dir / 'outcomes.jsonl'
    existing_rows = read_jsonl(outcomes_path) if outcomes_path.is_file() else []
    outcomes_by_trial: dict[str, dict[str, Any]] = {r['trial']: r for r in existing_rows if 'trial' in r}
    outcomes_lock = threading.Lock()

    # Index reusable plain branches from reuse_dirs (e.g. d8_confirmatory_20261016) without mutating them.
    reusable_plain_by_k_slot: dict[tuple[int, int], Path] = {}
    search_roots = [l2_dir, *(Path(d) for d in (reuse_dirs or []))]
    legacy_pat = re.compile(r'^k([0-9]+)-(s[12])-i([0-9]+)$')
    v04_pat = re.compile(r'^([A-Za-z0-9_.-]+)_k([0-9]+)_i([0-9]+)$')
    for root_dir in search_roots:
        b_root = root_dir / 'branches'
        if not b_root.is_dir():
            continue
        for p in sorted(b_root.iterdir()):
            if not (p / 'manifest.json').is_file():
                continue
            m_leg = legacy_pat.fullmatch(p.name)
            if m_leg and m_leg[2] == 's2':
                ck, slot = int(m_leg[1]), int(m_leg[3])
                reusable_plain_by_k_slot.setdefault((ck, slot), p)
            m_v04 = v04_pat.fullmatch(p.name)
            if m_v04 and m_v04[1] == 'plain':
                ck, slot = int(m_v04[2]), int(m_v04[3])
                reusable_plain_by_k_slot.setdefault((ck, slot), p)

    header_inputs = {
        'parent_episode_id': parent.episode_id,
        'arms_sha256': digest(specs),
    }

    def persist(row: dict[str, Any]) -> None:
        with outcomes_lock:
            outcomes_by_trial[row['trial']] = with_header(header_inputs, row)
            ordered = sorted(
                outcomes_by_trial.values(),
                key=lambda r: (str(r.get('arm', 'plain')), int(r.get('k', 0)), str(r.get('trial', ''))),
            )
            write_jsonl(outcomes_path, ordered)

    maximum_valid = sum(s['n'] for s in specs)

    def irrecoverable() -> bool:
        invalid = sum(not r.get('valid') for r in outcomes_by_trial.values() if r.get('selected'))
        return invalid > max_invalid_rate * maximum_valid / (1.0 - max_invalid_rate)

    def _run_slot(spec: dict[str, Any], slot_idx: int) -> dict[str, Any] | None:
        arm_id = spec['arm_id']
        arm_type = spec['arm_type']
        arm_k = int(spec['k'])
        intervention = spec['intervention']
        base_trial = f'{arm_id}_k{arm_k:03d}_i{slot_idx:02d}'

        # Check if reusable plain branch exists in reuse_dirs or legacy s2 naming
        if arm_type == 'plain' and not (branches_root / base_trial).exists():
            reused_path = reusable_plain_by_k_slot.get((arm_k, slot_idx))
            if reused_path is not None and (reused_path / 'manifest.json').is_file():
                b_pkg = L1Package(reused_path)
                b_pkg.validate_branch(parent, arm_k, intervention=None)
                b_out = b_pkg.outcome()
                if b_out['valid'] and b_out['y'] is not None:
                    item_scores = _extract_item_scores(b_pkg)
                    row = {
                        'trial': base_trial,
                        'source_branch_dir': str(reused_path),
                        'arm': arm_id,
                        'arm_type': arm_type,
                        'k': arm_k,
                        'stage': 's2',
                        'valid': True,
                        'y': float(b_out['y']),
                        'item_scores': item_scores,
                        'reason': b_out.get('reason'),
                        'termination': b_out.get('termination'),
                        'modelVersion': b_out.get('modelVersion'),
                        'reused': True,
                        'selected': True,
                    }
                    persist(row)
                    return row

        attempt = 0
        while True:
            trial_name = base_trial if attempt == 0 else f'{base_trial}-r{attempt}'
            branch_dir = branches_root / trial_name
            if not branch_dir.resolve().is_relative_to(branches_root.resolve()):
                raise IntegrityError(f'Branch output redirects outside branches: {trial_name}')

            if branch_dir.exists():
                if (branch_dir / 'manifest.json').is_file():
                    b_pkg = L1Package(branch_dir)
                    b_pkg.validate_branch(parent, arm_k, intervention=intervention)
                    b_out = b_pkg.outcome()
                    item_scores = _extract_item_scores(b_pkg) if b_out['valid'] else {}
                    reused_flag = True
                else:
                    b_out = {
                        'valid': False,
                        'y': None,
                        'reason': outcomes_by_trial.get(trial_name, {}).get('reason') or 'interrupted_attempt',
                        'termination': outcomes_by_trial.get(trial_name, {}).get('termination'),
                        'modelVersion': None,
                    }
                    item_scores = {}
                    reused_flag = True
            else:
                if cache_only:
                    raise L2Error(f'Cache-only verification requires missing branch {base_trial}')
                with outcomes_lock:
                    if irrecoverable():
                        return None
                try:
                    kwargs: dict[str, Any] = {}
                    if accepts_limiter:
                        kwargs['request_limiter'] = limiter
                    if accepts_intervention:
                        kwargs['intervention'] = intervention
                    runner(parent, arm_k, branch_dir, **kwargs)
                except Exception as exc:
                    b_out = {'valid': False, 'y': None, 'reason': str(exc), 'termination': None, 'modelVersion': None}
                    item_scores = {}
                else:
                    if not (branch_dir / 'manifest.json').is_file():
                        b_out = {'valid': False, 'y': None, 'reason': 'missing_manifest', 'termination': None, 'modelVersion': None}
                        item_scores = {}
                    else:
                        b_pkg = L1Package(branch_dir)
                        b_pkg.validate_branch(parent, arm_k, intervention=intervention)
                        b_out = b_pkg.outcome()
                        item_scores = _extract_item_scores(b_pkg) if b_out['valid'] else {}
                reused_flag = False

            valid = bool(b_out['valid'] and b_out['y'] is not None)
            row = {
                'trial': trial_name,
                'source_branch_dir': str(branch_dir),
                'arm': arm_id,
                'arm_type': arm_type,
                'k': arm_k,
                'stage': 's2' if arm_type == 'plain' else arm_type,
                'valid': valid,
                'y': float(b_out['y']) if valid else None,
                'item_scores': item_scores,
                'reason': b_out.get('reason'),
                'termination': b_out.get('termination'),
                'modelVersion': b_out.get('modelVersion'),
                'reused': reused_flag,
                'selected': True,
            }
            persist(row)
            if valid:
                return row
            attempt += 1

    tasks = [(spec, i) for spec in specs for i in range(spec['n'])]
    rng = random.Random(seed)
    rng.shuffle(tasks)

    results_by_arm_k: dict[tuple[str, int], list[dict[str, Any]]] = {(s['arm_id'], s['k']): [] for s in specs}
    if max_concurrency <= 1:
        for spec, slot_idx in tasks:
            r = _run_slot(spec, slot_idx)
            if r is not None:
                results_by_arm_k[(spec['arm_id'], spec['k'])].append(r)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency) as pool:
            fut_map = {pool.submit(_run_slot, spec, slot_idx): (spec['arm_id'], spec['k']) for spec, slot_idx in tasks}
            for fut in concurrent.futures.as_completed(fut_map):
                key = fut_map[fut]
                r = fut.result()
                if r is not None:
                    results_by_arm_k[key].append(r)

    ordered_rows = sorted(
        outcomes_by_trial.values(),
        key=lambda r: (str(r.get('arm', 'plain')), int(r.get('k', 0)), str(r.get('trial', ''))),
    )
    write_jsonl(outcomes_path, ordered_rows)

    arm_summaries: dict[str, Any] = {}
    for spec in specs:
        key = (spec['arm_id'], spec['k'])
        rows = results_by_arm_k.get(key, [])
        vals = [_extract_outcome_value(r, outcome) for r in rows]
        clean_vals = [float(v) for v in vals if v is not None]
        mean_val = (sum(clean_vals) / len(clean_vals)) if clean_vals else None
        ci_lo, ci_hi = wilson_bounds(clean_vals, alpha=0.05, one_sided=False) if clean_vals else (None, None)
        arm_summaries[f"{spec['arm_id']}@k{spec['k']}"] = {
            'arm_id': spec['arm_id'],
            'arm_type': spec['arm_type'],
            'k': spec['k'],
            'target_n': spec['n'],
            'valid_n': len(clean_vals),
            'reused_n': sum(1 for r in rows if r.get('reused')),
            'executed_n': sum(1 for r in rows if not r.get('reused')),
            'outcome': outcome,
            'mean': mean_val,
            'wilson_95': [ci_lo, ci_hi],
            'values': clean_vals,
        }

    return {
        'episode_id': parent.episode_id,
        'family_id': parent.family_id,
        'outcomes_path': str(outcomes_path),
        'arms': arm_summaries,
        'outcomes': ordered_rows,
    }


def load_branch_facts(
    l2_dir: Path,
    outcomes: Sequence[dict[str, Any]],
    *,
    reuse_dirs: Sequence[Path] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Loads turn-level facts for valid branches in `outcomes`."""
    branch_facts: dict[str, list[dict[str, Any]]] = {}
    search_roots = [Path(l2_dir), *(Path(d) for d in (reuse_dirs or []))]
    for r in outcomes:
        if not r.get('valid'):
            continue
        trial = str(r['trial'])
        candidates: list[Path] = []
        if r.get('source_branch_dir'):
            candidates.append(Path(str(r['source_branch_dir'])))
        for root_dir in search_roots:
            candidates.append(root_dir / 'branches' / trial)
            # Also check legacy k<k>-s2-i<slot> naming when trial is plain_k<k>_i<slot>
            m = re.fullmatch(r'^plain_k([0-9]+)_i([0-9]+)$', trial)
            if m:
                candidates.append(root_dir / 'branches' / f'k{int(m[1]):03d}-s2-i{int(m[2]):02d}')
        for b_dir in candidates:
            if (b_dir / 'manifest.json').is_file():
                branch_facts[trial] = build_facts(L1Package(b_dir))
                break
    return branch_facts


def evaluate_partition(
    hypotheses: Sequence[dict[str, Any]],
    parent_facts: list[dict[str, Any]],
    branch_facts_by_trial: dict[str, list[dict[str, Any]]],
    outcomes: Sequence[dict[str, Any]],
    *,
    alpha: float = 0.05,
    min_per_group: int = 4,
    outcome: str = 'total',
) -> dict[str, Any]:
    """Evaluates partition hypotheses `(j, P, e)` against plain branches at `c_{j-1}` (Section 6.1)."""
    results: list[dict[str, Any]] = []
    testable_indices: list[int] = []
    raw_pvals: list[float] = []

    for idx, h in enumerate(hypotheses):
        if h.get('verification', 'partition') != 'partition':
            continue
        turn_j = int(h['turn'])
        pred = validate_predicate(h['predicate'])
        pred_effect = float(h.get('effect', -0.1))
        rationale = str(h.get('rationale') or '')
        defect_id = str(h.get('defect') or h.get('item_id') or outcome)

        # Step 1: Consistency on parent turn j
        if not eval_predicate_on_turn(pred, parent_facts, turn_j):
            results.append({
                'defect': defect_id,
                'item_id': defect_id,
                'turn': turn_j,
                'verification': 'partition',
                'predicate': pred,
                'predicted_effect': pred_effect,
                'rationale': rationale,
                'status': 'inconsistent',
                'finding_type': 'correlational',
                'n_p': 0,
                'n_not_p': 0,
                'frequency': None,
                'effect': None,
                'p_value': None,
                'p_value_holm': None,
                'evidence_level': 'E0',
            })
            continue

        # Step 2: Partition valid plain branches at k = j - 1
        k_prev = turn_j - 1
        plain_rows = [
            r for r in outcomes
            if (r.get('arm', 'plain') == 'plain' or r.get('stage') in ('s1', 's2'))
            and int(r.get('k', -1)) == k_prev
            and r.get('valid')
            and r.get('y') is not None
        ]
        y_p: list[float] = []
        y_not_p: list[float] = []
        missing_facts = any(
            not branch_facts_by_trial.get(r['trial'])
            or not any(f.get('turn') == turn_j for f in branch_facts_by_trial[r['trial']])
            for r in plain_rows
        )
        for r in plain_rows:
            b_facts = branch_facts_by_trial.get(r['trial'])
            if not b_facts or not any(f.get('turn') == turn_j for f in b_facts):
                continue
            val = _extract_outcome_value(r, outcome if outcome != 'item' else defect_id)
            if val is None:
                continue
            if eval_predicate_on_turn(pred, b_facts, turn_j):
                y_p.append(val)
            else:
                y_not_p.append(val)

        n_p = len(y_p)
        n_not_p = len(y_not_p)
        total_n = n_p + n_not_p
        f_hat = (n_p / total_n) if total_n > 0 else None

        if missing_facts or n_p < min_per_group or n_not_p < min_per_group:
            results.append({
                'defect': defect_id,
                'item_id': defect_id,
                'turn': turn_j,
                'verification': 'partition',
                'predicate': pred,
                'predicted_effect': pred_effect,
                'rationale': rationale,
                'status': 'untestable',
                'finding_type': 'correlational',
                'n_p': n_p,
                'n_not_p': n_not_p,
                'frequency': f_hat,
                'reason': 'missing_branch_facts' if missing_facts else 'insufficient_groups',
                'effect': None,
                'p_value': None,
                'p_value_holm': None,
                'evidence_level': 'E0',
            })
            continue

        mean_p = sum(y_p) / n_p
        mean_not_p = sum(y_not_p) / n_not_p
        delta_p = mean_p - mean_not_p
        pval = permutation_one_sided(y_p, y_not_p, direction=-1.0 if pred_effect <= 0 else 1.0)
        testable_indices.append(len(results))
        raw_pvals.append(pval)
        results.append({
            'defect': defect_id,
            'item_id': defect_id,
            'turn': turn_j,
            'verification': 'partition',
            'predicate': pred,
            'predicted_effect': pred_effect,
            'rationale': rationale,
            'status': 'pending',
            'finding_type': 'correlational',
            'n_p': n_p,
            'n_not_p': n_not_p,
            'mean_p': mean_p,
            'mean_not_p': mean_not_p,
            'frequency': f_hat,
            'effect': delta_p,
            'p_value': pval,
            'p_value_holm': None,
            'evidence_level': 'E0',
        })

    adj_pvals = holm_adjust(raw_pvals)
    for ti, adj_p in zip(testable_indices, adj_pvals):
        row = results[ti]
        row['p_value_holm'] = adj_p
        direction_ok = (row['effect'] < 0.0) if row['predicted_effect'] <= 0.0 else (row['effect'] > 0.0)
        if adj_p <= alpha and direction_ok:
            row['status'] = 'confirmed'
            row['evidence_level'] = 'E1+'
        else:
            row['status'] = 'not_confirmed'
            row['evidence_level'] = 'E0'

    n_proposed = len(results)
    n_confirmed = sum(1 for r in results if r['status'] == 'confirmed')
    return {
        'proposed_count': n_proposed,
        'confirmed_count': n_confirmed,
        'confirmation_rate': (n_confirmed / n_proposed) if n_proposed > 0 else None,
        'hypotheses': results,
    }


def evaluate_intervention(
    hypotheses: Sequence[dict[str, Any]],
    outcomes: Sequence[dict[str, Any]],
    *,
    alpha: float = 0.05,
    min_effect: float = 0.15,
    outcome: str = 'total',
) -> dict[str, Any]:
    """Evaluates intervention hypotheses `(k, treatment_arm, plain)` (Section 6.2)."""
    results: list[dict[str, Any]] = []
    testable_indices: list[int] = []
    raw_pvals: list[float] = []

    for h in hypotheses:
        if h.get('verification') != 'intervention':
            continue
        k = int(h.get('k', h.get('turn', 0)))
        treat_arm = str(h.get('treatment_arm') or 'deliver')
        control_arm = str(h.get('control_arm') or 'plain')
        defect_id = str(h.get('defect') or h.get('item_id') or outcome)
        rationale = str(h.get('rationale') or '')
        pred_effect = float(h.get('effect', min_effect))
        target_outcome = outcome if outcome != 'item' else defect_id

        y_treat = [
            val for r in outcomes
            if (r.get('arm') == treat_arm or r.get('arm_type') == treat_arm)
            and int(r.get('k', -1)) == k
            and (val := _extract_outcome_value(r, target_outcome)) is not None
        ]
        y_plain = [
            val for r in outcomes
            if (r.get('arm', 'plain') == control_arm or (control_arm == 'plain' and r.get('stage') in ('s1', 's2')))
            and int(r.get('k', -1)) == k
            and (val := _extract_outcome_value(r, target_outcome)) is not None
        ]

        if len(y_treat) == 0 or len(y_plain) == 0:
            results.append({
                'defect': defect_id,
                'item_id': defect_id,
                'k': k,
                'turn': int(h.get('turn', k)),
                'verification': 'intervention',
                'treatment_arm': treat_arm,
                'control_arm': control_arm,
                'predicted_effect': pred_effect,
                'rationale': rationale,
                'status': 'untestable',
                'finding_type': 'causal_intervention',
                'n_treatment': len(y_treat),
                'n_control': len(y_plain),
                'mean_treatment': (sum(y_treat) / len(y_treat)) if y_treat else None,
                'mean_control': (sum(y_plain) / len(y_plain)) if y_plain else None,
                'effect': None,
                'p_value': None,
                'p_value_holm': None,
                'evidence_level': 'E0',
            })
            continue

        mean_treat = sum(y_treat) / len(y_treat)
        mean_plain = sum(y_plain) / len(y_plain)
        delta_arm = mean_treat - mean_plain
        pval = permutation_one_sided(y_treat, y_plain, direction=1.0)
        testable_indices.append(len(results))
        raw_pvals.append(pval)
        results.append({
            'defect': defect_id,
            'item_id': defect_id,
            'k': k,
            'turn': int(h.get('turn', k)),
            'verification': 'intervention',
            'treatment_arm': treat_arm,
            'control_arm': control_arm,
            'predicted_effect': pred_effect,
            'rationale': rationale,
            'status': 'pending',
            'finding_type': 'causal_intervention',
            'n_treatment': len(y_treat),
            'n_control': len(y_plain),
            'mean_treatment': mean_treat,
            'mean_control': mean_plain,
            'effect': delta_arm,
            'p_value': pval,
            'p_value_holm': None,
            'evidence_level': 'E0',
        })

    adj_pvals = holm_adjust(raw_pvals)
    for ti, adj_p in zip(testable_indices, adj_pvals):
        row = results[ti]
        row['p_value_holm'] = adj_p
        if adj_p <= alpha and float(row['effect']) >= min_effect:
            row['status'] = 'confirmed'
            row['evidence_level'] = 'E2'
        else:
            row['status'] = 'not_confirmed'
            row['evidence_level'] = 'E0'

    n_proposed = len(results)
    n_confirmed = sum(1 for r in results if r['status'] == 'confirmed')
    return {
        'proposed_count': n_proposed,
        'confirmed_count': n_confirmed,
        'confirmation_rate': (n_confirmed / n_proposed) if n_proposed > 0 else None,
        'hypotheses': results,
    }


def evaluate_two_point(
    outcomes: Sequence[dict[str, Any]],
    k1: int,
    k2: int,
    *,
    alpha: float = 0.05,
    min_lower_drop: float = 0.10,
    outcome: str = 'total',
) -> dict[str, Any]:
    """Evaluates two-point continuation drop between `c_{k1}` and `c_{k2}` (Section 6.3)."""
    y_k1 = [
        val for r in outcomes
        if (r.get('arm', 'plain') == 'plain' or r.get('stage') in ('s1', 's2'))
        and int(r.get('k', -1)) == int(k1)
        and (val := _extract_outcome_value(r, outcome)) is not None
    ]
    y_k2 = [
        val for r in outcomes
        if (r.get('arm', 'plain') == 'plain' or r.get('stage') in ('s1', 's2'))
        and int(r.get('k', -1)) == int(k2)
        and (val := _extract_outcome_value(r, outcome)) is not None
    ]
    if not y_k1 or not y_k2:
        return {
            'k1': int(k1),
            'k2': int(k2),
            'turn': int(k2),
            'status': 'untestable',
            'n_k1': len(y_k1),
            'n_k2': len(y_k2),
            'v_k1': None,
            'v_k2': None,
            'drop': None,
            'lower_drop_95': None,
            'p_value': None,
            'confirmed': False,
        }
    v_k1 = sum(y_k1) / len(y_k1)
    v_k2 = sum(y_k2) / len(y_k2)
    drop = v_k1 - v_k2
    ld = lower_drop(y_k1, y_k2, alpha=alpha)
    pval = permutation_one_sided(y_k2, y_k1, direction=-1.0)
    confirmed = bool(ld >= min_lower_drop and pval <= alpha)
    return {
        'k1': int(k1),
        'k2': int(k2),
        'turn': int(k2),
        'status': 'confirmed' if confirmed else 'not_confirmed',
        'n_k1': len(y_k1),
        'n_k2': len(y_k2),
        'v_k1': v_k1,
        'v_k2': v_k2,
        'drop': drop,
        'lower_drop_95': ld,
        'p_value': pval,
        'confirmed': confirmed,
    }
