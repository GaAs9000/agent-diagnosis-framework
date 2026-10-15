import threading
from types import SimpleNamespace

import pytest

from l3_sandbox.common import L3Error, read, write
from l3_sandbox.limiter import RequestLimiter


def test_cap_persists_and_unknown_usage_retains_reservation(tmp_path):
    limiter = RequestLimiter(tmp_path, tpm=10, max_http_attempts=2)
    assert limiter.acquire(8, timeout=1)
    limiter.reconcile(8, None)
    assert not limiter.acquire(3, timeout=0.01)
    restarted = RequestLimiter(tmp_path, tpm=10, max_http_attempts=2)
    assert restarted.acquire(2, timeout=1)
    restarted.reconcile(2, 1)
    assert not restarted.acquire(0, timeout=1)
    assert restarted.snapshot()['count'] == 2


def test_actual_usage_replaces_estimate_and_deadline_prevents_admission(tmp_path):
    limiter = RequestLimiter(tmp_path, tpm=10, max_http_attempts=3)
    assert limiter.acquire(10, timeout=1)
    limiter.reconcile(10, 2)
    assert limiter.acquire(8, timeout=1)
    limiter.reconcile(8, None)
    assert not limiter.acquire(0, timeout=0)
    assert limiter.snapshot()['count'] == 2


def test_multiple_instances_cannot_race_persisted_cap(tmp_path):
    a = RequestLimiter(tmp_path, max_http_attempts=5)
    b = RequestLimiter(tmp_path, max_http_attempts=5)
    accepted = []
    def send(limiter):
        accepted.append(limiter.acquire(1, timeout=1))
    threads = [threading.Thread(target=send, args=(a if i % 2 else b,)) for i in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(accepted) == 5
    assert a.snapshot()['count'] == 5


@pytest.mark.parametrize('status,body', [(401, b''), (403, b''), (400, b'{"reason":"API_KEY_INVALID"}')])
def test_auth_latch_persists(status, body, tmp_path):
    limiter = RequestLimiter(tmp_path, max_http_attempts=10)
    limiter.observe(SimpleNamespace(status=status, body=body))
    restarted = RequestLimiter(tmp_path, max_http_attempts=10)
    assert not restarted.acquire(1, timeout=1)
    with pytest.raises(L3Error, match='authentication'):
        restarted.check_send()


def test_ledger_corruption_and_symlinks_fail_closed(tmp_path):
    RequestLimiter(tmp_path, max_http_attempts=2)
    value = read(tmp_path / 'http_admissions.json')
    value['state']['count'] = 0
    value['state']['fatal_auth'] = True
    write(tmp_path / 'http_admissions.json', value)
    with pytest.raises(L3Error, match='Corrupt'):
        RequestLimiter(tmp_path, max_http_attempts=2)
    (tmp_path / 'http_admissions.json').unlink()
    (tmp_path / 'http_admissions.json').symlink_to(tmp_path / 'missing')
    with pytest.raises(L3Error, match='Symlink'):
        RequestLimiter(tmp_path, max_http_attempts=2)


@pytest.mark.parametrize('status,body', [
    (402, b''), (400, b'{"error":"insufficient_balance"}'),
    (403, b'{"error":"credit balance is too low"}'),
])
def test_payment_latch_persists_and_blocks_send(status, body, tmp_path):
    limiter = RequestLimiter(tmp_path, max_http_attempts=10)
    limiter.begin_episode()
    limiter.observe(SimpleNamespace(status=status, body=body))
    assert limiter.finish_episode() == 'payment_required'
    restarted = RequestLimiter(tmp_path, max_http_attempts=10)
    assert restarted.snapshot()['provider_stop']['reason'] == 'payment_required'
    assert not restarted.acquire(1, timeout=1)
    with pytest.raises(L3Error):
        restarted.check_send()


def test_transient_429_never_latches_across_successful_requests(tmp_path):
    limiter = RequestLimiter(tmp_path, max_http_attempts=20)
    for _ in range(5):
        limiter.begin_episode()
        limiter.observe(SimpleNamespace(status=429, body=b'rate limit'))
        assert limiter.acquire(1, timeout=1)
        limiter.reconcile(1, None)
        limiter.observe(SimpleNamespace(status=200, body=b'{}'))
        assert limiter.finish_episode() is None
    assert limiter.snapshot()['provider_stop'] is None


def test_terminal_429_is_durable_and_episode_state_is_thread_local(tmp_path):
    limiter = RequestLimiter(tmp_path, max_http_attempts=10)
    limiter.begin_episode()
    limiter.observe(SimpleNamespace(status=200, body=b'{}'))
    def exhausted():
        limiter.begin_episode()
        for _ in range(4):
            assert limiter.acquire(1, timeout=1)
            limiter.observe(SimpleNamespace(status=429, body=b'rate limit'))
            limiter.reconcile(1, None)
            assert limiter.snapshot()['provider_stop'] is None
        assert limiter.finish_episode() == 'quota_exhausted'
    thread = threading.Thread(target=exhausted)
    thread.start()
    thread.join()
    assert limiter.finish_episode() is None
    restarted = RequestLimiter(tmp_path, max_http_attempts=10)
    assert restarted.snapshot()['count'] == 4
    assert not restarted.acquire(1, timeout=1)
    with pytest.raises(L3Error, match='quota_exhausted'):
        restarted.check_send()


def test_explicit_model_mismatch_latches_without_persisting_response_metadata(tmp_path):
    limiter = RequestLimiter(tmp_path, max_http_attempts=10)
    limiter.begin_episode()
    limiter.observe(SimpleNamespace(status=200, body=b'{}', metadata={
        'model_identity_mismatch': {'expected': 'a', 'observed': 'b'},
        'private': 'must not persist'}))
    assert limiter.finish_episode() == 'model_identity_mismatch'
    restarted = RequestLimiter(tmp_path, max_http_attempts=10)
    assert not restarted.acquire(1, timeout=1)
    assert restarted.snapshot()['provider_stop'] == {'reason': 'model_identity_mismatch', 'status': 200}
    assert 'must not persist' not in (tmp_path / 'http_admissions.json').read_text()


def test_reported_model_fingerprint_alone_does_not_latch(tmp_path):
    limiter = RequestLimiter(tmp_path, max_http_attempts=10)
    limiter.begin_episode()
    limiter.observe(SimpleNamespace(status=200, body=b'{}', metadata={
        'reported_model': 'model-version', 'system_fingerprint': 'new fingerprint'}))
    assert limiter.finish_episode() is None
    assert limiter.snapshot()['provider_stop'] is None
