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
