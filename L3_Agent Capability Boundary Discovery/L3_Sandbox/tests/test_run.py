import hashlib
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from l3_sandbox.common import L3Error, digest, read, write
from l3_sandbox import run
from l3_sandbox.run import _transport_guard as real_transport_guard


@pytest.fixture
def setup(tmp_path, monkeypatch):
    config = {'model': {'kind': 'local'}}
    job = {'trial_id': 'trial1', 'task_id': 'officeval_003', 'condition_id': 'A',
           'family_id': 'officeval_003', 'family_verified': True, 'split': 'validation',
           'replicate': 0, 'config': config, 'config_sha256': digest(config)}
    plan = {'experiment_id': 'test', 'jobs': [job], 'limits': {'max_concurrency': 2,
            'max_http_attempts': 10, 'max_replacements_per_slot': 2, 'rpm': None, 'tpm': None}}
    monkeypatch.setattr(run, 'verify_plan', lambda *a, **k: plan)
    monkeypatch.setattr(run, '_transport_guard', lambda limiter: nullcontext())
    monkeypatch.setattr(run, '_verify_package', lambda *a: {'verified': True})
    monkeypatch.setattr(run, '_summarize', lambda *a: {'study_valid': True, 'study_outcome_y': 0.0})
    calls = []
    def native(config, output, *, request_limiter):
        calls.append(output)
        assert request_limiter.acquire(1, timeout=1)
        request_limiter.reconcile(1, 1)
        output.mkdir()
        write(output / 'manifest.json', {'recording_mode': 'live', 'config_sha256': digest(config),
                                         'task': {'task_id': job['task_id']}, 'lineage': None})
        write(output / 'raw/resolved_config.json', config)
        for name in ('official_grade.json', 'artifacts/inventory.json', 'evidence_index.json'):
            write(output / name, {})
        for name in ('trajectory.jsonl', 'state_changes.jsonl'):
            (output / name).write_text('')
    monkeypatch.setattr(run, '_native_run', native)
    return tmp_path, plan, calls, native


def test_dry_run_never_calls_native_or_loads_credentials(setup, monkeypatch):
    directory, plan, calls, _ = setup
    monkeypatch.setattr(run, '_credentials', lambda _: pytest.fail('credentials loaded'))
    assert run.run_plan(directory)['workload'] == 1
    assert calls == []
    assert list(directory.iterdir()) == []


def test_healthy_zero_is_accepted_and_resume_has_no_new_requests(setup):
    directory, plan, calls, _ = setup
    first = run.run_plan(directory, execute=True)
    assert first['status'] == 'complete'
    assert first['rows'][0]['study_outcome_y'] == 0
    assert first['rows'][0]['accepted']
    again = run.run_plan(directory, execute=True)
    assert again['status'] == 'complete'
    assert len(calls) == 1
    assert again['http_admissions']['count'] == 1


def test_interrupted_directory_consumes_attempt_and_is_preserved(setup):
    directory, plan, calls, _ = setup
    interrupted = directory / 'runs/trial1/attempt_000'
    interrupted.mkdir(parents=True)
    (interrupted / 'sentinel').write_text('preserved')
    result = run.run_plan(directory, execute=True)
    assert result['status'] == 'complete'
    assert result['rows'][0]['status'] == 'interrupted'
    assert result['rows'][1]['replacement_of'] == 0
    assert calls[0].parent.name == 'attempt_001'
    assert (interrupted / 'sentinel').read_text() == 'preserved'


def test_invalid_runs_exhaust_finite_replacements(setup, monkeypatch):
    directory, plan, calls, _ = setup
    monkeypatch.setattr(run, '_summarize', lambda *a: {'study_valid': False, 'study_outcome_y': None})
    result = run.run_plan(directory, execute=True)
    assert result['status'] == 'incomplete'
    assert len(result['rows']) == len(calls) == 3
    run.run_plan(directory, execute=True)
    assert len(calls) == 3


def test_changed_accepted_evidence_fails_closed_without_spending(setup):
    directory, plan, calls, _ = setup
    run.run_plan(directory, execute=True)
    write(calls[0] / 'evidence_index.json', {'tampered': True})
    with pytest.raises(L3Error, match='retained progress'):
        run.run_plan(directory, execute=True)
    assert len(calls) == 1
    assert not read(directory / 'run_results.json')['rows'][0]['accepted']


def test_corrupt_receipt_fails_closed_and_records_error(setup):
    directory, plan, calls, _ = setup
    run.run_plan(directory, execute=True)
    write(calls[0].parent / 'receipt.json', {'corrupt': True})
    with pytest.raises(L3Error):
        run.run_plan(directory, execute=True)
    assert len(calls) == 1
    assert read(directory / 'run_results.json')['runner_errors']


def test_native_errors_retained_and_never_reported_complete(setup, monkeypatch):
    directory, plan, _, _ = setup
    def fail(*a, **k):
        raise RuntimeError('must not expose secret exception text')
    monkeypatch.setattr(run, '_native_run', fail)
    result = run.run_plan(directory, execute=True)
    assert result['status'] == 'incomplete'
    assert len(result['rows']) == 3
    assert all(row['runner_error'] == 'RuntimeError' for row in result['rows'])
    assert 'secret' not in (directory / 'run_results.json').read_text()


def test_cap_and_auth_block_replacements(setup, monkeypatch):
    directory, plan, calls, native = setup
    plan['limits']['max_http_attempts'] = 1
    monkeypatch.setattr(run, '_summarize', lambda *a: {'study_valid': False})
    assert run.run_plan(directory, execute=True)['status'] == 'blocked_http_cap'
    assert len(calls) == 1


def test_fixture_rejected_and_plan_lock_exclusive(setup, monkeypatch):
    directory, plan, calls, native = setup
    monkeypatch.setattr(run, '_summarize', lambda *a: {'study_valid': True, 'test_fixture': True})
    result = run.run_plan(directory, execute=True)
    assert result['status'] == 'incomplete'
    assert not any(r['accepted'] for r in result['rows'])
    with run._plan_lock(directory):
        with pytest.raises(L3Error, match='Another process'):
            run.run_plan(directory, execute=True)


def test_symlinks_rejected_before_any_native_call(setup):
    directory, plan, calls, _ = setup
    (directory / 'runs').symlink_to(directory / 'elsewhere')
    with pytest.raises(L3Error, match='Symlink'):
        run.run_plan(directory, execute=True)
    assert not calls


def test_missing_admission_ledger_cannot_reset_spending_cap(setup):
    directory, plan, calls, _ = setup
    run.run_plan(directory, execute=True)
    (directory / 'http_admissions.json').unlink()
    with pytest.raises(L3Error, match='ledger is missing'):
        run.run_plan(directory, execute=True)
    assert len(calls) == 1


def test_auth_guard_observes_real_receipt_before_retry_admission(tmp_path, monkeypatch):
    from c3_sandbox.l1.models.http import NativeHTTP
    from l3_sandbox.limiter import RequestLimiter
    limiter = RequestLimiter(tmp_path, max_http_attempts=10)
    sends = []
    def send(*a, **k):
        sends.append(True)
        return SimpleNamespace(status=403, body=b'forbidden')
    monkeypatch.setattr(NativeHTTP, 'post', send)
    # Retain real guard; this test does not use the fake-run fixture.
    with run._transport_guard(limiter):
        assert limiter.acquire(1, timeout=1)
        NativeHTTP.post(object(), 'url', b'', 1)
        limiter.reconcile(1, None)
        assert not limiter.acquire(1, timeout=1)
        with pytest.raises(L3Error, match='authentication'):
            NativeHTTP.post(object(), 'url', b'', 1)
    assert sends == [True]
    assert NativeHTTP.post is send


def test_one_worker_failure_retains_other_worker_progress(setup, monkeypatch):
    import copy
    directory, plan, calls, native = setup
    job = copy.deepcopy(plan['jobs'][0])
    job['trial_id'] = 'trial2'
    plan['jobs'].append(job)
    def selective(config, output, **kwargs):
        if output.parent.parent.name == 'trial1':
            raise RuntimeError('offline simulated failure')
        return native(config, output, **kwargs)
    monkeypatch.setattr(run, '_native_run', selective)
    result = run.run_plan(directory, execute=True)
    assert result['status'] == 'incomplete'
    assert len(result['rows']) == 4
    assert [r['trial_id'] for r in result['rows'] if r['accepted']] == ['trial2']
    assert read(directory / 'run_results.json') == result


def test_auth_block_persists_across_runner_resume(setup, monkeypatch):
    directory, plan, calls, native = setup
    def unauthorized(config, output, *, request_limiter):
        native(config, output, request_limiter=request_limiter)
        request_limiter.observe(SimpleNamespace(status=401, body=b''))
    monkeypatch.setattr(run, '_native_run', unauthorized)
    monkeypatch.setattr(run, '_summarize', lambda *a: {'study_valid': False})
    first = run.run_plan(directory, execute=True)
    assert first['status'] == 'blocked_auth'
    assert len(calls) == 1
    assert run.run_plan(directory, execute=True)['status'] == 'blocked_auth'
    assert len(calls) == 1


def test_credentials_only_load_named_literal_values_and_restore(tmp_path, monkeypatch):
    (tmp_path / 'PROJECT_MEMORY.md').write_text('test workspace')
    (tmp_path / '.env').write_text('UNRELATED=other\nTRIAL_KEY="offline-placeholder"\n')
    monkeypatch.setattr(run, '__file__', str(tmp_path / 'src/run.py'))
    monkeypatch.setenv('TRIAL_KEY', '')
    monkeypatch.delenv('UNRELATED', raising=False)
    import os
    plan = {'jobs': [{'config': {'model': {'api_key_env': 'TRIAL_KEY'}}}]}
    with run._credentials(plan):
        assert os.environ['TRIAL_KEY'] == 'offline-placeholder'
        assert 'UNRELATED' not in os.environ
    assert os.environ['TRIAL_KEY'] == ''


def test_resealed_tampered_receipt_outcome_rejected_without_native_calls(setup):
    directory, plan, calls, _ = setup
    run.run_plan(directory, execute=True)
    receipt = calls[0].parent / 'receipt.json'
    saved = read(receipt)
    saved['row']['study_outcome_y'] = 1.0
    saved['sha256'] = digest(saved['row'])
    write(receipt, saved)
    with pytest.raises(L3Error, match='retained progress'):
        run.run_plan(directory, execute=True)
    assert len(calls) == 1
    result = read(directory / 'run_results.json')
    assert result['status'] == 'incomplete'
    assert result['rows'][0]['reuse_invalid']
    assert not result['rows'][0]['accepted']


def test_resealed_receipt_identity_cannot_override_frozen_job(setup):
    directory, plan, calls, _ = setup
    run.run_plan(directory, execute=True)
    receipt = calls[0].parent / 'receipt.json'
    saved = read(receipt)
    saved['row']['replicate'] = 999
    saved['sha256'] = digest(saved['row'])
    write(receipt, saved)
    with pytest.raises(L3Error, match='retained progress'):
        run.run_plan(directory, execute=True)
    assert len(calls) == 1


def test_missing_ledger_with_orphan_episode_cannot_reset_cap(setup):
    directory, plan, calls, _ = setup
    orphan = directory / 'runs/trial1/attempt_000/episode'
    orphan.mkdir(parents=True)
    with pytest.raises(L3Error, match='ledger is missing'):
        run.run_plan(directory, execute=True)
    assert not calls


@pytest.mark.parametrize('accepted', [False, 0])
def test_resealed_inconsistent_acceptance_cannot_force_paid_replacement(setup, accepted):
    directory, plan, calls, _ = setup
    run.run_plan(directory, execute=True)
    receipt = calls[0].parent / 'receipt.json'
    saved = read(receipt)
    saved['row']['accepted'] = accepted
    saved['sha256'] = digest(saved['row'])
    write(receipt, saved)
    with pytest.raises(L3Error, match='retained progress'):
        run.run_plan(directory, execute=True)
    assert len(calls) == 1
    result = read(directory / 'run_results.json')
    assert result['http_admissions']['count'] == 1
    assert result['rows'][0]['status'] == 'receipt_corrupt'


@pytest.mark.parametrize('status,attempts', [(402, 1), (429, 4)])
def test_provider_failure_stops_replacements_and_queued_jobs_and_restart(setup, monkeypatch, status, attempts):
    import copy
    directory, plan, calls, native = setup
    plan['limits']['max_concurrency'] = 1
    for i in range(2, 46):
        job = copy.deepcopy(plan['jobs'][0])
        job['trial_id'] = f'trial{i}'
        plan['jobs'].append(job)
    def blocked(config, output, *, request_limiter):
        native(config, output, request_limiter=request_limiter)
        for i in range(attempts):
            if i:
                assert request_limiter.acquire(1, timeout=1)
                request_limiter.reconcile(1, None)
            request_limiter.observe(SimpleNamespace(status=status, body=b'provider blocked'))
    monkeypatch.setattr(run, '_native_run', blocked)
    first = run.run_plan(directory, execute=True)
    assert first['status'] == 'blocked_provider'
    assert first['http_admissions']['count'] == attempts
    assert len(first['rows']) == len(calls) == 1
    assert first['rows'][0]['status'] == 'blocked_provider'
    assert first['rows'][0]['study_valid'] is False
    assert not first['rows'][0]['accepted']
    again = run.run_plan(directory, execute=True)
    assert again['status'] == 'blocked_provider'
    assert len(calls) == 1
    assert len(list((directory / 'runs').glob('*/attempt_*'))) == 1


def test_provider_stop_does_not_invalidate_historical_healthy_receipt(setup):
    from l3_sandbox.limiter import RequestLimiter
    directory, plan, calls, _ = setup
    run.run_plan(directory, execute=True)
    limiter = RequestLimiter(directory, max_http_attempts=10)
    limiter.stop_provider('payment_required', 402)
    result = run.run_plan(directory, execute=True)
    assert result['rows'][0]['accepted']
    assert len(calls) == 1
    assert result['http_admissions']['provider_stop']['reason'] == 'payment_required'


def test_provider_failure_retained_when_native_raises_before_package(setup, monkeypatch):
    directory, plan, calls, _ = setup
    def fail(config, output, *, request_limiter):
        calls.append(output)
        request_limiter.observe(SimpleNamespace(status=429, body=b'quota'))
        raise RuntimeError('private provider text')
    monkeypatch.setattr(run, '_native_run', fail)
    result = run.run_plan(directory, execute=True)
    assert result['status'] == 'blocked_provider'
    assert len(calls) == 1
    assert result['rows'][0]['provider_stop'] == 'quota_exhausted'
    assert 'private provider text' not in (directory / 'run_results.json').read_text()


def test_real_chat_backend_wrong_model_stops_before_replacement_and_persists(setup, monkeypatch):
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from c3_sandbox.l1.models.opencode import OpenCodeChatBackend
    directory, plan, calls, _ = setup
    sends = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            sends.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            body = json.dumps({'model': 'unexpected-model', 'system_fingerprint': 'independent',
                               'usage': {'prompt_tokens': 3, 'completion_tokens': 2},
                               'choices': [{'message': {'role': 'assistant', 'content': 'done'},
                                            'finish_reason': 'stop'}]}).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = {'kind': 'local', 'base_url': f'http://127.0.0.1:{server.server_port}',
                  'model': 'requested-alias', 'expected_reported_model': 'approved-version',
                  'parameters': {}, 'max_response_bytes': 1024 * 1024}
        plan['jobs'][0]['config']['model'] = config
        plan['jobs'][0]['config_sha256'] = digest(plan['jobs'][0]['config'])
        monkeypatch.setattr(run, '_transport_guard', real_transport_guard)
        def native(config, output, *, request_limiter):
            calls.append(output)
            backend = OpenCodeChatBackend(config['model'])
            body = json.dumps(backend.request(backend.initial('system', 'prompt'), [])).encode()
            assert request_limiter.acquire(1, timeout=1)
            receipt = backend.send(body, 2)
            # Match the native runner: charge usage before protocol parsing.
            assert receipt.metadata['usage']['total_tokens'] == 5
            assert receipt.metadata['system_fingerprint'] == 'independent'
            request_limiter.reconcile(1, receipt.metadata['usage']['total_tokens'])
            backend.parse(receipt.body, 'request1')
        monkeypatch.setattr(run, '_native_run', native)
        first = run.run_plan(directory, execute=True)
        assert first['status'] == 'blocked_provider'
        assert len(first['rows']) == len(calls) == len(sends) == 1
        assert first['rows'][0]['provider_stop'] == 'model_identity_mismatch'
        assert first['rows'][0]['runner_error'] == 'ProtocolError'
        assert not first['rows'][0]['accepted']
        ledger = read(directory / 'http_admissions.json')['state']
        assert ledger['count'] == 1
        assert ledger['admissions'][0]['tokens'] == 5
        assert ledger['admissions'][0]['usage_known']
        assert ledger['provider_stop'] == {'reason': 'model_identity_mismatch', 'status': 200}
        again = run.run_plan(directory, execute=True)
        assert again['status'] == 'blocked_provider'
        assert len(calls) == len(sends) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
