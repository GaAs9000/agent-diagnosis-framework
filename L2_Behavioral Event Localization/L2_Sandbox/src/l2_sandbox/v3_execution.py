"""Seals, bounded admission and native execution for the V3 development protocol.

Digests establish integrity, not external approval. Approval is a separate frozen
artifact with an exact plan digest; no draft or simulated artifact qualifies.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
import threading
from functools import wraps
from pathlib import Path
import time
from typing import Any


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(value: dict, field: str = 'artifact_sha256') -> dict:
    out = {k: v for k, v in value.items() if k != field}
    out[field] = digest(out)
    return out


def verify_seal(value: dict, field: str = 'artifact_sha256') -> None:
    if value.get(field) != digest({k: v for k, v in value.items() if k != field}):
        raise ValueError(f'invalid {field} seal')


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def approve(plan: dict, freeze: dict | None) -> dict:
    verify_seal(plan, 'plan_sha256')
    if not freeze or freeze.get('d0_frozen') is not True or freeze.get('approved_plan_sha256') != plan['plan_sha256']:
        raise ValueError('exact frozen approved_plan_sha256 required')
    if plan.get('execution_ready') is not True:
        raise ValueError('plan is not execution ready; review unresolved provenance first')
    caps = freeze.get('limits') or {}
    for key in ('max_requests', 'max_tokens', 'max_cost_usd', 'max_wall_seconds', 'max_invalid_attempts', 'max_tokens_per_request', 'usd_per_token_upper_bound'):
        val = caps.get(key)
        if type(val) not in (int, float) or not math.isfinite(val) or val < 0 or (val == 0 and key not in ('max_cost_usd', 'usd_per_token_upper_bound')):
            raise ValueError(f'finite frozen limit required (positive except API cost/price): {key}')
    concurrency = plan.get('execution_concurrency', 1)
    if type(concurrency) is not int or not 1 <= concurrency <= 3:
        raise ValueError('frozen execution_concurrency must be an integer from 1 to 3')
    if freeze.get('execution_concurrency', concurrency) != concurrency:
        raise ValueError('execution concurrency differs from frozen plan')
    budget = plan.get('budget_and_stop_conditions') or {}
    for limit, planned in (('max_requests', 'hard_actor_http_attempts'), ('max_wall_seconds', 'max_wall_seconds')):
        if planned in budget and caps[limit] > budget[planned]:
            raise ValueError(f'frozen {limit} exceeds approved plan hard cap')
    if caps['max_cost_usd'] == 0 or caps['usd_per_token_upper_bound'] == 0:
        actor = plan.get('actor_identity') or {}
        if plan.get('pricing_mode') != 'local_no_provider_charge' or actor.get('provider') != 'local' or not all(actor.get(k) for k in ('model', 'harness', 'endpoint')):
            raise ValueError('zero API cost/price requires frozen local provider identity and pricing_mode')
    if caps['max_tokens_per_request'] > caps['max_tokens']:
        raise ValueError('per-request token reservation exceeds total token cap')
    return caps


def _serialized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return call


class Admission:
    """Persist reservations before transport. Unknown/failed usage stays charged.

    The stage owns an exclusive process lock. Unsettled sends on restart stay
    charged and are never automatically retried by receipt consumers.
    """
    def __init__(self, path: Path, plan_hash: str, caps: dict):
        self._lock = threading.RLock()
        self._thread = threading.local()
        self.path, self.caps = path, caps
        self.state = read_json(path) if path.exists() else dict(plan_sha256=plan_hash, started_at=time.time(), requests=0, tokens=0, cost_usd=0.0, invalid_attempts=0, sends=[])
        if path.exists():
            verify_seal(self.state)
        self.state.pop('artifact_sha256', None)
        if self.state['plan_sha256'] != plan_hash:
            raise ValueError('ledger belongs to another plan')
        if self.state.get('limits', caps) != caps:
            raise ValueError('frozen limits differ from persisted admission ledger')
        self.state['limits'] = dict(caps)
        self.current = None
        self.save()

    @property
    def current(self):
        return getattr(self._thread, 'current', None)

    @current.setter
    def current(self, value):
        self._thread.current = value

    @_serialized
    def snapshot(self):
        return copy.deepcopy(self.state)

    @_serialized
    def close(self, reason):
        self.state.setdefault('stop_reason', reason)
        self.save()

    @_serialized
    def exhaustion_reason(self):
        if self.state.get('stop_reason'):
            return self.state['stop_reason']
        try:
            self.check_new_request()
        except RuntimeError as exc:
            return str(exc)
        return None

    @_serialized
    def save(self):
        write_json(self.path, seal(self.state))

    @_serialized
    def remaining_seconds(self):
        return max(0.0, self.caps["max_wall_seconds"] - (time.time() - self.state["started_at"]))

    @_serialized
    def check(self):
        if time.time() - self.state['started_at'] >= self.caps['max_wall_seconds']:
            raise RuntimeError('frozen wall-clock cap reached')
        if self.state['invalid_attempts'] >= self.caps['max_invalid_attempts']:
            raise RuntimeError('frozen invalid-attempt cap reached')

    @_serialized
    def check_new_request(self):
        if self.state.get('stop_reason'):
            raise RuntimeError(self.state['stop_reason'])
        self.check()
        if self.state['requests'] >= self.caps['max_requests']:
            raise RuntimeError('frozen HTTP-attempt cap reached')
        if self.state['tokens'] + self.caps['max_tokens_per_request'] > self.caps['max_tokens']:
            raise RuntimeError('frozen token cap reached')
        if self.state['cost_usd'] + self.caps['max_tokens_per_request'] * self.caps['usd_per_token_upper_bound'] > self.caps['max_cost_usd']:
            raise RuntimeError('frozen API-cost cap reached')

    @_serialized
    def acquire(self, estimate, timeout=None):
        if self.state.get('stop_reason'):
            return False
        self.check()
        tokens = max(int(math.ceil(estimate or 0)), int(self.caps['max_tokens_per_request']))
        cost = tokens * self.caps['usd_per_token_upper_bound']
        if (self.state['requests'] + 1 > self.caps['max_requests'] or self.state['tokens'] + tokens > self.caps['max_tokens'] or self.state['cost_usd'] + cost > self.caps['max_cost_usd']):
            return False
        if tokens > self.caps['max_tokens_per_request']:
            return False
        self.state['requests'] += 1
        self.state['tokens'] += tokens
        self.state['cost_usd'] += cost
        self.current = len(self.state['sends'])
        self.state['sends'].append(dict(tokens_reserved=tokens, cost_reserved=cost, status='inflight'))
        self.save()
        return True

    @_serialized
    def reconcile(self, estimate, actual):
        if self.current is None:
            raise RuntimeError('reconcile without admission')
        row = self.state['sends'][self.current]
        row.update(status='unknown_usage' if actual is None else 'settled', actual_tokens=actual)
        # Never refund requests or reservations; caps remain strict even with missing usage.
        self.current = None
        self.save()
        if actual is not None and actual > row['tokens_reserved']:
            self.state['tokens'] += actual - row['tokens_reserved']
            self.state['cost_usd'] += (actual - row['tokens_reserved']) * self.caps['usd_per_token_upper_bound']
            self.save()
            raise RuntimeError('transport exceeded frozen token reservation')
        self.check()

    @_serialized
    def invalid(self):
        self.state['invalid_attempts'] += 1
        self.save()


def identity(parent: Path, checkpoint_k: int) -> dict:
    from c3_sandbox.l1.package import verify_package
    report = verify_package(parent)
    if not report.get('verified'):
        raise ValueError('parent evidence verification failed')
    cp = parent / 'checkpoints' / f'cp{checkpoint_k:06d}.json'
    config = parent / 'raw/resolved_config.json'
    doc = read_json(cp)
    if (doc.get('qualification') or {}).get('eligible') is not True:
        raise ValueError('unqualified checkpoint')
    return dict(parent_episode_id=read_json(parent / 'manifest.json')['episode_id'], parent_index_sha256=report['index_sha256'], checkpoint_sha256=file_digest(cp), config_sha256=file_digest(config), manifest_sha256=file_digest(parent / 'manifest.json'))


def item_score(package: Path, unit: dict) -> float:
    from c3_sandbox.l1.evaluation.study import study_outcome
    out = study_outcome(package, verify_evidence=True)
    if not out.get('study_valid'):
        raise ValueError(f'invalid study evidence: {out.get("study_classification")}')
    if unit.get('unit_level') in ('whole_run', 'gate_or_no_artifact'):
        return float(out['study_outcome_y'])
    raw = read_json(package / 'official_grade.json')['official_raw']
    if raw.get('dim1_pass') is False:
        return 0.0
    rule = unit.get('defect_name') or unit.get('rule')
    matches = [r for r in raw.get('dim2_items', []) if r.get('rule') == rule]
    if len(matches) != 1:
        raise ValueError('official target rubric item missing or ambiguous')
    row = matches[0]
    maximum, actual = float(row['max_delta']), float(row['delta'])
    if maximum > 0:
        return max(0.0, min(1.0, actual / maximum))
    if maximum < 0:
        return max(0.0, min(1.0, 1 + actual / abs(maximum)))
    raise ValueError('unsupported zero-weight rubric endpoint')


def native_branch(unit: dict, branch: dict, message: dict, *, runs_dir: Path, output: Path, limiter: Admission) -> dict:
    """Production adapter: L1 owns restore, actor, native official scorer, sealing."""
    from c3_sandbox.l1.core.orchestrator import run
    from c3_sandbox.l1.package import verify_package
    parent = runs_dir / unit['episode_id']
    k = int(branch['checkpoint_k'])
    before = identity(parent, k)
    if before != unit['parent_identity']:
        raise ValueError('frozen parent/checkpoint/config identity changed')
    cp = read_json(parent / 'checkpoints' / f'cp{k:06d}.json')
    config = copy.deepcopy(read_json(parent / 'raw/resolved_config.json'))
    frozen_actor = unit.get('actor_identity')
    if frozen_actor:
        actual_actor = dict(model=config['model'].get('model'), harness=config['agent'].get('harness'), endpoint=config['model'].get('base_url'), provider='local')
        if actual_actor != frozen_actor:
            raise ValueError('native actor/endpoint differs from frozen pricing identity')
    used = int(cp['budget_used']['model_requests'])
    if config['budget']['max_model_requests'] - used != int(branch['remaining_budget']):
        raise ValueError('remaining budget differs from native inherited budget')
    text = message.get('message')
    if text:
        config['agent']['intervention'] = dict(at_request=used + 1, message=text)
    result = run(config, output, parent=parent, checkpoint_id=f'cp{k:06d}', request_limiter=limiter)
    cap_exceeded = None
    try:
        limiter.check()
    except RuntimeError as exc:
        cap_exceeded = str(exc)
    verified = verify_package(output)
    if not verified.get('verified') or not result.get('verification', {}).get('verified'):
        raise ValueError('branch evidence does not verify')
    if identity(parent, k) != before:
        raise ValueError('parent evidence changed during branch')
    try:
        score = item_score(output, unit)
    except ValueError as exc:
        reason = limiter.exhaustion_reason() if hasattr(limiter, 'exhaustion_reason') else None
        if not reason:
            raise
        return dict(valid_attempt=False, status='incomplete', cap_exceeded=reason, error=str(exc), package_path=str(output.resolve()), package_index_sha256=verified['index_sha256'], simulated_offline=False)

    manifest = read_json(output / 'manifest.json')
    if (manifest.get('status') or {}).get('termination') == 'wall_time_budget':
        cap_exceeded = cap_exceeded or 'frozen transport admission or wall-clock cap reached'
    return dict(valid_attempt=True, status='completed', outcome_score=score, branch_episode_id=manifest['episode_id'], package_path=str(output.resolve()), package_index_sha256=verified['index_sha256'], requests_used=int(manifest['counts']['model_requests']), cap_exceeded=cap_exceeded, official_grade_sha256=file_digest(output / 'official_grade.json'), parent_identity=before, simulated_offline=False)


def verify_branch(receipt: dict, *, expected: dict, verify_package_binding: bool = True) -> None:
    verify_seal(receipt)
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f'receipt binding mismatch: {key}')
    if receipt.get('status') != 'completed' or receipt.get('valid_attempt') is not True:
        raise ValueError('incomplete branch receipt; retained failure cannot be silently retried')
    if verify_package_binding and not receipt.get('simulated_offline'):
        from c3_sandbox.l1.package import verify_package
        root = Path(receipt['package_path'])
        report = verify_package(root, expected_index_sha256=receipt['package_index_sha256'])
        if not report.get('verified') or file_digest(root / 'official_grade.json') != receipt['official_grade_sha256']:
            raise ValueError('receipt package binding no longer verifies')
        manifest = read_json(root / 'manifest.json')
        lineage = manifest.get('lineage') or {}
        parent = receipt['parent_identity']
        for key in ('parent_episode_id', 'parent_index_sha256'):
            if lineage.get(key) != parent.get(key):
                raise ValueError('branch lineage differs from frozen parent')
        if lineage.get('parent_checkpoint_sha256') != parent.get('checkpoint_sha256') or lineage.get('parent_checkpoint_id') != f"cp{receipt['anchor_k']:06d}":
            raise ValueError('branch lineage differs from frozen checkpoint')
        message = (read_json(root / 'raw/resolved_config.json').get('agent',{}).get('intervention') or {}).get('message')
        message_hash = hashlib.sha256(message.encode()).hexdigest() if message else None
        if message_hash != receipt['message_sha256']:
            raise ValueError('branch intervention differs from frozen message')
        if item_score(root, receipt['endpoint']) != receipt['outcome_score']:
            raise ValueError('branch endpoint differs from official evidence')


@contextmanager
def exclusive(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('another stage writer owns this receipt directory') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
