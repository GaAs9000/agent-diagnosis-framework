"""Offline admission tests. Run only on the authorized AML CPU worker."""
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from l2_sandbox.opencode_admission import Admission, DEFAULT_CAPS, transport_guard

PLAN = 'a' * 64
MODEL = 'deepseek-v4.1-flash'


def response(status=200, *, model=MODEL, usage=None, error=None):
    data = dict(model=model)
    if usage is not None:
        data['usage'] = usage
    if error is not None:
        data['error'] = error
    return SimpleNamespace(status=status, body=json.dumps(data).encode())


def usage(prompt=100, completion=20, cached=10):
    return dict(prompt_tokens=prompt, completion_tokens=completion,
                total_tokens=prompt + completion,
                prompt_tokens_details=dict(cached_tokens=cached))


def test_reservation_and_known_positive_usage_refund(tmp_path):
    a = Admission(tmp_path / 'ledger.json', PLAN)
    assert a.acquire(50)
    initial = a.snapshot()
    assert initial['tokens'] == 1_048_576 + 8192
    assert initial['cost_usd'] == pytest.approx((1_048_576 * .30 + 8192 * 1.2) / 1e6)
    a.observe(response(usage=usage()))
    a.reconcile(50, 999999)  # Only validated raw usage controls accounting.
    assert a.snapshot()['tokens'] == 120
    assert a.snapshot()['cost_usd'] == pytest.approx((90 * .30 + 10 * .006 + 20 * 1.2) / 1e6)
    assert a.snapshot()['requests'] == 1


@pytest.mark.parametrize('actual', [None, 0, 50])
def test_missing_usage_never_refunded(tmp_path, actual):
    a = Admission(tmp_path / 'ledger.json', PLAN)
    assert a.acquire(5)
    before = a.snapshot()
    a.observe(response())
    a.reconcile(5, actual)
    assert a.snapshot()['tokens'] == before['tokens']
    assert a.snapshot()['cost_usd'] == before['cost_usd']
    assert a.snapshot()['sends'][0]['status'] == 'unknown_usage'


def test_restart_preserves_inflight_deadline_and_bindings(tmp_path):
    path = tmp_path / 'ledger.json'
    a = Admission(path, PLAN)
    assert a.acquire(5)
    before = a.snapshot()
    b = Admission(path, PLAN)
    assert b.snapshot() == before
    assert b.acquire(5)
    assert b.snapshot()['tokens'] == before['tokens'] * 2
    with pytest.raises(ValueError, match='binding'):
        Admission(path, 'b' * 64)
    with pytest.raises(ValueError, match='binding'):
        Admission(path, PLAN, dict(max_requests=10))


@pytest.mark.parametrize('status', [401, 402, 403, 429])
def test_status_latches_before_retry_and_redacts(tmp_path, status):
    a = Admission(tmp_path / 'ledger.json', PLAN)
    assert a.acquire(50)
    a.observe(response(status, error='secret response text'))
    a.reconcile(50, None)
    assert not a.acquire(50)
    assert a.exhaustion_reason() == f'provider_status_{status}'
    assert 'secret response text' not in a.path.read_text()


@pytest.mark.parametrize('result, reason', [
    (response(model='wrong'), 'reported_model_mismatch'),
    (response(500, error='secret quota exhausted'), 'provider_quota'),
    (response(usage=dict(prompt_tokens=-1, completion_tokens=1)), 'invalid_reported_usage'),
    (response(usage=dict(prompt_tokens=1, completion_tokens=1, total_tokens=1)), 'invalid_reported_usage'),
    (response(usage=usage(cached=101)), 'invalid_reported_usage'),
])
def test_bad_response_stops(tmp_path, result, reason):
    a = Admission(tmp_path / 'ledger.json', PLAN)
    assert a.acquire(50)
    a.observe(result)
    a.reconcile(50, None)
    assert a.exhaustion_reason() == reason
    assert not a.acquire(50)


def test_excess_usage_is_recorded_and_stops(tmp_path):
    a = Admission(tmp_path / 'ledger.json', PLAN)
    assert a.acquire(50)
    a.observe(response(usage=usage(prompt=2_000_000, completion=9000)))
    assert a.exhaustion_reason() == 'usage_exceeded_reservation'
    a.reconcile(50, 2_009_000)
    assert a.snapshot()['tokens'] == 2_009_000
    assert a.exhaustion_reason() == 'usage_exceeded_reservation'


def test_caps_and_first_invalid(tmp_path):
    a = Admission(tmp_path / 'ledger.json', PLAN, dict(max_requests=2))
    assert not a.acquire(33_562_625)
    for _ in range(2):
        assert a.acquire(5)
        a.reconcile(5, None)
    assert not a.acquire(5)
    assert a.snapshot()['requests'] == 2
    b = Admission(tmp_path / 'other.json', PLAN)
    b.invalid()
    assert not b.acquire(5)
    with pytest.raises(RuntimeError, match='invalid_attempt'):
        b.check()
    for name in ('max_requests', 'max_cost_usd', 'max_tokens', 'max_wall_seconds', 'max_invalid_attempts'):
        with pytest.raises(ValueError):
            Admission(tmp_path / 'bad.json', PLAN, {name: DEFAULT_CAPS[name] + 1})


def test_cost_reservations_stop_without_refunding_attempts(tmp_path):
    a = Admission(tmp_path / 'ledger.json', PLAN, dict(max_cost_usd=.4))
    assert a.acquire(100)
    a.reconcile(100, None)
    assert not a.acquire(100)
    assert a.exhaustion_reason() == 'usage_equivalent_cost_cap'


def test_expired_restart_cannot_reset_clock(tmp_path, monkeypatch):
    monkeypatch.setattr('l2_sandbox.opencode_admission.time.time', lambda: 1000)
    path = tmp_path / 'ledger.json'
    Admission(path, PLAN)
    monkeypatch.setattr('l2_sandbox.opencode_admission.time.time', lambda: 87400)
    a = Admission(path, PLAN)
    assert a.remaining_seconds() == 0
    assert not a.acquire(5)
    assert a.exhaustion_reason() == 'wall_cap'


def test_transport_guard_shared_restored_and_no_unreserved_send(tmp_path):
    class HTTP:
        calls = 0
        def post(self, url, body, timeout, *, gemini=False, extra_headers=None):
            HTTP.calls += 1
            return response(usage=usage())
    original = HTTP.post
    a = Admission(tmp_path / 'ledger.json', PLAN)
    with transport_guard(a, HTTP):
        with pytest.raises(RuntimeError, match='without admission'):
            HTTP().post('url', b'secret', 10)
        with transport_guard(a, HTTP):
            assert a.acquire(5)
            HTTP().post('url', b'secret', 10)
            with pytest.raises(RuntimeError, match='one HTTP'):
                HTTP().post('url', b'secret', 10)
            a.reconcile(5, 120)
        assert HTTP.post is not original
        other = Admission(tmp_path / 'other.json', PLAN)
        with pytest.raises(RuntimeError, match='another batch'):
            with transport_guard(other, HTTP):
                pass
    assert HTTP.post is original
    assert HTTP.calls == 1
    assert 'secret' not in a.path.read_text()


def test_concurrent_transport_is_thread_local_and_attempt_bounded(tmp_path):
    class HTTP:
        def post(self, url, body, timeout, **kwargs):
            return response(usage=usage(prompt=int(body), cached=0))
    a = Admission(tmp_path / 'ledger.json', PLAN, dict(max_requests=20))
    def send(i):
        if a.acquire(5):
            HTTP().post('url', str(i).encode(), 10)
            a.reconcile(5, i + 20)
            return True
        return False
    with transport_guard(a, HTTP), ThreadPoolExecutor(max_workers=4) as pool:
        accepted = list(pool.map(send, range(1, 31)))
    assert sum(accepted) == 20
    assert a.snapshot()['requests'] == 20
    assert a.snapshot()['tokens'] == sum(row['usage']['total_tokens'] for row in a.snapshot()['sends'])


def test_transport_error_keeps_reservation_and_patch_restores(tmp_path):
    class HTTP:
        def post(self, *args, **kwargs):
            raise OSError('sensitive transport text')
    original = HTTP.post
    a = Admission(tmp_path / 'ledger.json', PLAN)
    with pytest.raises(OSError):
        with transport_guard(a, HTTP):
            assert a.acquire(5)
            try:
                HTTP().post('url', b'', 10)
            finally:
                a.reconcile(5, None)
    assert HTTP.post is original
    assert a.snapshot()['sends'][0]['status'] == 'unknown_usage'
    assert 'sensitive' not in a.path.read_text()


def test_provider_stop_blocks_already_reserved_other_thread(tmp_path):
    import threading
    ready = threading.Event()
    stopped = threading.Event()
    class HTTP:
        calls = 0
        def post(self, *args, **kwargs):
            HTTP.calls += 1
            return response(429, error='private quota details')
    a = Admission(tmp_path / 'ledger.json', PLAN)
    def waiting_send():
        assert a.acquire(5)
        ready.set()
        assert stopped.wait(5)
        with pytest.raises(RuntimeError, match='batch transport stopped'):
            HTTP().post('url', b'', 10)
        a.reconcile(5, None)
    with transport_guard(a, HTTP), ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(waiting_send)
        assert ready.wait(5)
        assert a.acquire(5)
        HTTP().post('url', b'', 10)
        # Retry admission is denied immediately, even before reconciliation.
        assert a.exhaustion_reason() == 'provider_status_429'
        a.reconcile(5, None)
        assert not a.acquire(5)
        stopped.set()
        future.result(timeout=5)
    assert HTTP.calls == 1
    assert a.snapshot()['requests'] == 2


def test_tampered_ledger_rejected(tmp_path):
    path = tmp_path / 'ledger.json'
    Admission(path, PLAN)
    value = json.loads(path.read_text())
    value['started_at'] += 86400
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='seal'):
        Admission(path, PLAN)
