"""Persistent OpenCode usage-equivalent admission, not a billed-cash estimate.

The batch runner must hold its exclusive external batch lock for this object's
lifetime. Unknown sends remain fully charged, including across process restarts.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time

from .v3_execution import read_json, seal, verify_seal

DEFAULT_CAPS = dict(max_requests=15000, max_wall_seconds=86400,
                    max_cost_usd=12.0, max_tokens=1_000_000_000,
                    max_tokens_per_request=33_562_624, max_invalid_attempts=1,
                    input_usd_per_million=0.30, cached_input_usd_per_million=0.006,
                    output_usd_per_million=1.20)
_OUTPUT = 8192
_INPUT_FLOOR = 1_048_576
_PATCH_LOCK = threading.RLock()
_PATCHES = {}


def _cost(prompt, completion, cached=0):
    return ((prompt - cached) * 0.30 + cached * 0.006 + completion * 1.20) / 1_000_000


def validate_budget_authorization(authorization, predecessor_sha256):
    """Validate the sealed grant; v2 removes only the usage-equivalent cost cap."""
    verify_seal(authorization)
    version = authorization.get('schema')
    no_cost_cap = version == 'opencode-d1-budget-authorization-v2'
    scope = ('cross_actor_D1_remove_usage_equivalent_cost_cap' if no_cost_cap
             else 'cross_actor_D1_budget_extension')
    if (version not in ('opencode-d1-budget-authorization-v1', 'opencode-d1-budget-authorization-v2')
            or authorization.get('execution_authorized') is not True
            or authorization.get('scope') != scope
            or authorization.get('source') != 'direct_user_message_20261019'
            or not isinstance(authorization.get('exact_user_text'), str)
            or not authorization['exact_user_text'].strip()
            or (no_cost_cap and authorization['exact_user_text'] != '$50 等值用量 全部放开就行')
            or authorization.get('predecessor_plan_sha256') != predecessor_sha256
            or authorization.get('cumulative_limits') != dict(
                max_requests=22000, max_tokens=1_000_000_000,
                max_cost_usd=None if no_cost_cap else 50.0)):
        raise ValueError('explicit predecessor-bound Option A budget authorization required')
    start, deadline, wall = (authorization.get(k) for k in
                            ('phase_started_at_epoch', 'deadline_epoch', 'max_wall_seconds'))
    if (any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
            for v in (start, deadline, wall)) or not 0 < deadline - start <= wall <= 86400):
        raise ValueError('budget authorization requires a finite phase deadline within 24 hours')
    return authorization


class Admission:
    """L1 request_limiter interface, with reservations persisted before HTTP."""

    def __init__(self, path: Path, plan_hash: str, caps: dict | None = None, *,
                 expected_model: str = 'deepseek-v4.1-flash',
                 absolute_deadline: float | None = None,
                 budget_authorization: dict | None = None):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._thread = threading.local()
        hard_caps = dict(DEFAULT_CAPS)
        if budget_authorization is not None:
            grant = validate_budget_authorization(
                budget_authorization, budget_authorization.get('predecessor_plan_sha256'))
            hard_caps.update(grant['cumulative_limits'])
            if absolute_deadline is None or absolute_deadline > grant['deadline_epoch']:
                raise ValueError('admission deadline exceeds budget authorization')
        self.caps = dict(DEFAULT_CAPS)
        for name, value in (caps or {}).items():
            if name not in DEFAULT_CAPS:
                raise ValueError('unknown admission cap')
            if name == 'max_cost_usd' and value is None and hard_caps[name] is None:
                self.caps[name] = None
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('admission caps must be positive finite numbers')
            if name.endswith('usd_per_million'):
                if value != DEFAULT_CAPS[name]:
                    raise ValueError('peak accounting rates are fixed')
            elif hard_caps[name] is not None and value > hard_caps[name]:
                raise ValueError('admission cap exceeds batch hard limit')
            if name in ('max_requests', 'max_tokens', 'max_tokens_per_request', 'max_invalid_attempts') and type(value) is not int:
                raise ValueError('count caps must be integers')
            self.caps[name] = value
        if not isinstance(expected_model, str) or not expected_model:
            raise ValueError('expected reported model required')
        if not isinstance(plan_hash, str) or len(plan_hash) != 64 or any(c not in '0123456789abcdef' for c in plan_hash):
            raise ValueError('plan SHA256 required')
        if absolute_deadline is not None:
            if (type(absolute_deadline) not in (int, float)
                    or not math.isfinite(absolute_deadline) or absolute_deadline <= 0):
                raise ValueError('absolute deadline must be positive and finite')
            absolute_deadline = float(absolute_deadline)
        if self.path.exists():
            self.state = read_json(self.path)
            verify_seal(self.state)
            self.state.pop('artifact_sha256')
            if (self.state['plan_sha256'] != plan_hash or self.state['limits'] != self.caps
                    or self.state['expected_model'] != expected_model
                    or self.state.get('absolute_deadline') != absolute_deadline):
                raise ValueError('persisted admission binding differs')
        else:
            self.state = dict(plan_sha256=plan_hash, limits=dict(self.caps),
                              expected_model=expected_model, started_at=time.time(),
                              absolute_deadline=absolute_deadline,
                              accounting_mode='usage_equivalent_peak_rates_not_billed_cash',
                              requests=0, tokens=0, cost_usd=0.0, invalid_attempts=0,
                              branch_invalid=0, fatal_invalid=0, failures=[], sends=[])
        if (absolute_deadline is not None
                and absolute_deadline > self.state['started_at'] + 86400):
            raise ValueError('absolute deadline exceeds batch hard wall limit')
        self._save()

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix='.' + self.path.name + '.', dir=self.path.parent)
        try:
            with os.fdopen(fd, 'w') as stream:
                json.dump(seal(self.state), stream, allow_nan=False, sort_keys=True)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.path)
        finally:
            Path(name).unlink(missing_ok=True)

    def _stop(self, reason):
        self.state.setdefault('stop_reason', reason)
        self._save()

    def snapshot(self):
        with self._lock:
            return copy.deepcopy(self.state)

    def close(self, reason='closed'):
        with self._lock:
            # Accept only bounded reason codes; never persist caller response text.
            self._stop(reason if reason in {'closed', 'completed', 'invalid_attempt', 'wall_cap'} else 'closed')

    def remaining_seconds(self):
        with self._lock:
            deadline = self.state['started_at'] + self.caps['max_wall_seconds']
            absolute_deadline = self.state.get('absolute_deadline')
            if absolute_deadline is not None:
                deadline = min(deadline, absolute_deadline)
            return max(0.0, deadline - time.time())

    def exhaustion_reason(self):
        with self._lock:
            if self.state.get('stop_reason'):
                return self.state['stop_reason']
            if self.remaining_seconds() <= 0:
                return 'wall_cap'
            if self.state['invalid_attempts'] >= self.caps['max_invalid_attempts']:
                return 'invalid_attempt'
            if self.state['requests'] >= self.caps['max_requests']:
                return 'http_attempt_cap'
            if self.state['tokens'] + _INPUT_FLOOR + _OUTPUT > self.caps['max_tokens']:
                return 'token_cap'
            if (self.caps['max_cost_usd'] is not None
                    and self.state['cost_usd'] + _cost(_INPUT_FLOOR, _OUTPUT) > self.caps['max_cost_usd']):
                return 'usage_equivalent_cost_cap'
            return None

    def check(self):
        with self._lock:
            reason = self.exhaustion_reason()
            if reason:
                raise RuntimeError(reason)

    def acquire(self, estimate, timeout=None):
        with self._lock:
            if getattr(self._thread, 'pending', None) is not None:
                raise RuntimeError('unsettled admission on thread')
            if self.exhaustion_reason() or (timeout is not None and timeout <= 0):
                return False
            if type(estimate) not in (int, float) or not math.isfinite(estimate) or estimate < 0:
                raise ValueError('finite nonnegative token estimate required')
            estimated = math.ceil(estimate)
            if estimated > self.caps['max_tokens_per_request']:
                return False
            prompt = max(estimated - _OUTPUT, _INPUT_FLOOR)
            tokens, cost = prompt + _OUTPUT, _cost(prompt, _OUTPUT)
            if (tokens > self.caps['max_tokens_per_request'] or self.state['tokens'] + tokens > self.caps['max_tokens']
                    or (self.caps['max_cost_usd'] is not None
                        and self.state['cost_usd'] + cost > self.caps['max_cost_usd'])):
                return False
            self.state['requests'] += 1
            self.state['tokens'] += tokens
            self.state['cost_usd'] += cost
            self._thread.pending = len(self.state['sends'])
            self.state['sends'].append(dict(status='inflight', prompt_reserved=prompt,
                                           completion_reserved=_OUTPUT, tokens_reserved=tokens,
                                           cost_reserved=cost))
            self._save()
            return True

    def observe(self, result):
        """Read full native JSON in memory; persist only numeric usage and codes."""
        with self._lock:
            index = getattr(self._thread, 'pending', None)
            if index is None:
                raise RuntimeError('HTTP without admission')
            row = self.state['sends'][index]
            row['http_status'] = result.status
            try:
                raw = json.loads(result.body)
            except (ValueError, UnicodeError, TypeError):
                raw = None
            if result.status in (401, 402, 403, 429):
                self._stop('provider_status_' + str(result.status))
            if result.status == 200 and not isinstance(raw, dict):
                self._stop('reported_model_mismatch')
            if isinstance(raw, dict):
                error = raw.get('error') or (raw if result.status != 200 else None)
                # Inspect provider errors in memory only. Never retain response text.
                if error and any(word in json.dumps(error).lower() for word in
                                 ('quota', 'credit', 'balance', 'billing', 'rate_limit', 'rate limit')):
                    self._stop('provider_quota')
                if result.status == 200 and raw.get('model') != self.state['expected_model']:
                    self._stop('reported_model_mismatch')
                usage = raw.get('usage')
                if isinstance(usage, dict):
                    prompt, completion = usage.get('prompt_tokens'), usage.get('completion_tokens')
                    details = usage.get('prompt_tokens_details', {})
                    cached = details.get('cached_tokens', 0) if isinstance(details, dict) else None
                    total = usage.get('total_tokens', prompt + completion if type(prompt) is int and type(completion) is int else None)
                    if (all(type(n) is int and n >= 0 for n in (prompt, completion, cached, total))
                            and cached <= prompt and total == prompt + completion and total > 0):
                        row['usage'] = dict(prompt_tokens=prompt, completion_tokens=completion,
                                            cached_tokens=cached, total_tokens=total)
                        if (prompt > row['prompt_reserved'] or completion > _OUTPUT
                                or total > row['tokens_reserved']
                                or _cost(prompt, completion, cached) > row['cost_reserved']):
                            self._stop('usage_exceeded_reservation')
                    else:
                        self._stop('invalid_reported_usage')
            self._save()

    def reconcile(self, estimate, actual):
        with self._lock:
            index = getattr(self._thread, 'pending', None)
            if index is None:
                raise RuntimeError('reconcile without admission')
            row = self.state['sends'][index]
            usage = row.get('usage')
            if usage:
                tokens = usage['total_tokens']
                cost = _cost(usage['prompt_tokens'], usage['completion_tokens'], usage['cached_tokens'])
                row.update(status='settled', actual_tokens=tokens, actual_cost_usd=cost)
                self.state['tokens'] += tokens - row['tokens_reserved']
                self.state['cost_usd'] += cost - row['cost_reserved']
                if (usage['prompt_tokens'] > row['prompt_reserved'] or usage['completion_tokens'] > _OUTPUT
                        or tokens > row['tokens_reserved'] or cost > row['cost_reserved']):
                    self._stop('usage_exceeded_reservation')
            else:
                row['status'] = 'unknown_usage'
            self._thread.pending = None
            self._save()

    def invalid(self, *, failure_scope='global', source='unclassified', phase='execution'):
        # Fixed codes only; exception/response text must never enter the ledger.
        if failure_scope not in ('branch', 'global'):
            raise ValueError('unknown failure scope')
        if source not in ('unclassified', 'model_protocol', 'model_transport', 'scorer', 'branch_wall_time',
                          'integrity_or_unclassified', 'global_admission'):
            raise ValueError('unknown failure source')
        if phase not in ('admission', 'execution', 'verification', 'scoring'):
            raise ValueError('unknown failure phase')
        with self._lock:
            existing_reason = self.exhaustion_reason()
            key = 'branch_invalid' if failure_scope == 'branch' else 'fatal_invalid'
            self.state[key] = self.state.get(key, 0) + 1
            self.state.setdefault('failures', []).append(dict(
                failure_scope=failure_scope, error_source=source, failure_phase=phase))
            if failure_scope == 'global':
                self.state['invalid_attempts'] += 1
                self._stop(existing_reason or 'invalid_attempt')
            else:
                self._save()


@contextmanager
def transport_guard(admission: Admission, native_http_cls=None):
    """Scope a shared class patch; concurrent contexts must share one admission."""
    if native_http_cls is None:
        from c3_sandbox.l1.models.http import NativeHTTP
        native_http_cls = NativeHTTP
    with _PATCH_LOCK:
        entry = _PATCHES.get(native_http_cls)
        if entry is not None:
            if entry['admission'] is not admission:
                raise RuntimeError('another batch owns native transport guard')
            entry['users'] += 1
        else:
            original = native_http_cls.post

            def guarded(client, url, body, timeout, *, gemini=False, extra_headers=None):
                with admission._lock:
                    if getattr(admission._thread, 'pending', None) is None:
                        raise RuntimeError('HTTP without admission')
                    row = admission.state['sends'][admission._thread.pending]
                    if row.get('transport_started'):
                        raise RuntimeError('one HTTP send per reservation')
                    # The last admitted request may legitimately exhaust a cap.
                    if admission.state.get('stop_reason') or admission.remaining_seconds() <= 0:
                        raise RuntimeError('batch transport stopped')
                    row['transport_started'] = True
                    admission._save()
                result = original(client, url, body, min(timeout, admission.remaining_seconds()),
                                  gemini=gemini, extra_headers=extra_headers)
                admission.observe(result)
                return result

            entry = dict(admission=admission, original=original, users=1)
            _PATCHES[native_http_cls] = entry
            native_http_cls.post = guarded
    try:
        yield admission
    finally:
        with _PATCH_LOCK:
            entry['users'] -= 1
            if entry['users'] == 0:
                native_http_cls.post = entry['original']
                del _PATCHES[native_http_cls]
