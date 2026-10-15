"""Persistent, per HTTP attempt admission shared by all native run workers."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import math
import os
from pathlib import Path
import threading
import time

from .common import L3Error, digest, read, write


class RequestLimiter:
    def __init__(self, directory: Path, *, rpm=None, tpm=None, max_http_attempts: int):
        for value in (rpm, tpm):
            if value is not None and (not math.isfinite(float(value)) or float(value) <= 0):
                raise L3Error('Rate limits must be finite and positive')
        if type(max_http_attempts) is not int or max_http_attempts < 1:
            raise L3Error('max_http_attempts must be a positive integer')
        self.directory = Path(directory)
        self.rpm, self.tpm, self.cap = rpm, tpm, max_http_attempts
        self.path = self.directory / 'http_admissions.json'
        self.lock_path = self.directory / '.http_admissions.lock'
        self._thread_lock = threading.Lock()
        self._local = threading.local()
        self.binding = digest({'rpm': rpm, 'tpm': tpm, 'max_http_attempts': max_http_attempts})
        with self._state() as state:
            pass

    @contextmanager
    def _state(self):
        with self._thread_lock:
            if self.directory.is_symlink() or self.path.is_symlink() or self.lock_path.is_symlink():
                raise L3Error('Symlink in limiter state')
            fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                if self.path.exists():
                    saved = read(self.path)
                    state = saved.get('state')
                    if not isinstance(state, dict) or digest(state) != saved.get('sha256'):
                        raise L3Error('Corrupt HTTP admission ledger')
                    if state.get('binding') != self.binding:
                        raise L3Error('HTTP admission limits changed')
                    records = state.get('admissions')
                    if (not isinstance(records, list) or state.get('count') != len(records)
                            or type(state.get('count')) is not int or state['count'] < 0):
                        raise L3Error('Invalid HTTP admission ledger')
                    for record in records:
                        if (not isinstance(record, dict) or not all(
                            isinstance(record.get(k), (int, float)) and math.isfinite(record[k])
                            and record[k] >= 0 for k in ('at', 'tokens'))):
                            raise L3Error('Invalid HTTP admission record')
                else:
                    state = {'binding': self.binding, 'count': 0, 'admissions': [], 'fatal_auth': False}
                state.setdefault('provider_stop', None)
                yield state
                write(self.path, {'state': state, 'sha256': digest(state)})
            finally:
                os.close(fd)

    def snapshot(self):
        with self._state() as state:
            return {k: state[k] for k in ('count', 'fatal_auth', 'provider_stop')}

    def check_send(self):
        with self._state() as state:
            if state['fatal_auth']:
                raise L3Error('Fatal provider authentication latch is set')
            if state['provider_stop']:
                raise L3Error('Provider stop latch is set: ' + state['provider_stop']['reason']
                              + '; resolve provider access/balance/quota/model identity before a deliberate new plan')

    def stop_provider(self, reason, status=None):
        # The first provider failure remains durable across process restarts.
        with self._state() as state:
            if state['provider_stop'] is None:
                state['provider_stop'] = {'reason': reason, 'status': status}

    def begin_episode(self):
        self._local.last_status = None
        self._local.provider_failure = None

    def finish_episode(self):
        # L1 owns retries for each logical request. Only after native execution
        # exits is its final 429 terminal; never count 429s across requests.
        reason = getattr(self._local, 'provider_failure', None)
        if reason is None and getattr(self._local, 'last_status', None) == 429:
            reason = 'quota_exhausted'
            self.stop_provider(reason, 429)
        return reason

    def observe(self, receipt):
        self._local.last_status = receipt.status
        metadata = getattr(receipt, 'metadata', None) or {}
        if metadata.get('model_identity_mismatch'):
            self._local.provider_failure = 'model_identity_mismatch'
            self.stop_provider('model_identity_mismatch', receipt.status)
        body = receipt.body.lower()
        if (receipt.status == 402 or (receipt.status is not None and receipt.status >= 400
                and any(marker in body for marker in (
                    b'insufficient balance', b'insufficient_balance',
                    b'payment required', b'payment_required', b'credit balance is too low')))):
            self._local.provider_failure = 'payment_required'
            self.stop_provider('payment_required', receipt.status)
        if receipt.status in (401, 403) or b'API_KEY_INVALID' in receipt.body:
            with self._state() as state:
                state['fatal_auth'] = True

    def acquire(self, estimated_tokens=1000.0, timeout=None):
        estimate = float(estimated_tokens)
        if not math.isfinite(estimate) or estimate < 0:
            raise L3Error('Token estimate must be finite and nonnegative')
        timeout = 60.0 if timeout is None else float(timeout)
        if not math.isfinite(timeout) or timeout < 0:
            raise L3Error('Admission timeout must be finite and nonnegative')
        deadline = time.monotonic() + timeout
        if self.tpm is not None and estimate > self.tpm:
            return False
        while True:
            with self._state() as state:
                if state['fatal_auth'] or state['provider_stop'] or state['count'] >= self.cap:
                    return False
                if time.monotonic() >= deadline:
                    return False
                recent = [x for x in state['admissions'] if x['at'] > time.time() - 60]
                if ((self.rpm is None or len(recent) + 1 <= self.rpm)
                        and (self.tpm is None or sum(x['tokens'] for x in recent) + estimate <= self.tpm)):
                    index = len(state['admissions'])
                    state['admissions'].append({'at': time.time(), 'tokens': estimate, 'usage_known': False})
                    state['count'] += 1
                    self._local.admission = index
                    return True
                delay = max(0.001, min((x['at'] + 60 - time.time() for x in recent), default=0.01))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(delay, remaining, 0.1))

    def reconcile(self, estimate, actual):
        index = getattr(self._local, 'admission', None)
        if index is None:
            raise L3Error('Reconciliation without an admission')
        if actual is not None and (not math.isfinite(float(actual)) or float(actual) < 0):
            raise L3Error('Invalid actual token usage')
        with self._state() as state:
            record = state['admissions'][index]
            if actual is not None:
                record['tokens'] = float(actual)
                record['usage_known'] = True
        self._local.admission = None
