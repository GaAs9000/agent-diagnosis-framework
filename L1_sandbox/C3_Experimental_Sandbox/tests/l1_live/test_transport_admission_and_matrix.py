from __future__ import annotations

import copy
import importlib.util
import json
from types import SimpleNamespace
import pytest
from c3_sandbox.l1.core import orchestrator
from c3_sandbox.l1.core.contracts import HTTPResult
from c3_sandbox.l1.storage import canonical, read_json
from c3_sandbox.l1.evidence.live import scan
from .conftest import ROOT, ScriptedGemini, gemini_call, gemini_done


class SpyLimiter:
    def __init__(self, admit=True, on_acquire=None):
        self.admit, self.on_acquire = admit, on_acquire
        self.acquired, self.reconciled = [], []

    def acquire(self, estimated_tokens, timeout=None):
        self.acquired.append((estimated_tokens, timeout))
        if self.on_acquire: self.on_acquire()
        return self.admit

    def reconcile(self, estimated_tokens, actual_tokens):
        self.reconciled.append((estimated_tokens, actual_tokens))


def retry_config(config):
    cfg = copy.deepcopy(config)
    cfg['model']['parameters']['maxOutputTokens'] = 32
    cfg['model']['retry'] = {'max_attempts': 3, 'backoff_seconds': [0.0], 'honor_retry_after': False}
    return cfg


def rate_limit():
    return HTTPResult(429, b'{"error":"rate limit"}', {'transport_error': None, 'body_truncated': False})


def test_limiter_covers_each_native_send_and_retry(tmp_path, config, components):
    cfg = retry_config(config)
    model = ScriptedGemini(cfg['model'], [rate_limit(), gemini_call(), gemini_done()])
    components['model'] = model
    limiter = SpyLimiter()
    out = tmp_path / 'limited'
    result = orchestrator.run(cfg, out, injected=components, request_limiter=limiter)
    assert result['verification']['verified']
    assert result['status']['termination'] == 'assistant_finished'
    assert len(model.bodies) == len(limiter.acquired) == 3
    estimates = [len(canonical(body)) + 32 for body in model.bodies]
    assert [e for e, _ in limiter.acquired] == estimates
    assert all(0 < timeout <= 60 for _, timeout in limiter.acquired)
    assert limiter.reconciled == list(zip(estimates, [None, 12.0, 7.0]))
    assert model.bodies[0] == model.bodies[1]
    assert (out / 'raw/model/r000000/attempt_01/response.bin').exists()
    assert (out / 'raw/model/r000000/attempt_02/response.bin').exists()
    baseline = tmp_path / 'baseline'
    orchestrator.run(cfg, baseline, injected=dict(components, model=ScriptedGemini(cfg['model'])))
    assert read_json(out / 'raw/model/r000000/request.json') == read_json(baseline / 'raw/model/r000000/request.json')
    assert read_json(out / 'raw/resolved_config.json') == read_json(baseline / 'raw/resolved_config.json')


@pytest.mark.parametrize('expired_after_admission', [False, True])
def test_admission_timeout_never_sends(tmp_path, config, components, monkeypatch, expired_after_admission):
    now = [100.0]
    monkeypatch.setattr(orchestrator, 'time', SimpleNamespace(monotonic=lambda: now[0], sleep=lambda seconds: None))
    limiter = SpyLimiter(admit=expired_after_admission,
                         on_acquire=(lambda: now.__setitem__(0, 161.0)) if expired_after_admission else None)
    result = orchestrator.run(config, tmp_path / 'timeout', injected=components, request_limiter=limiter)
    assert result['verification']['verified']
    assert result['status']['termination'] == 'wall_time_budget'
    assert result['status']['execution_validity'] == 'invalid'
    assert components['model'].bodies == []
    assert len(limiter.acquired) == 1
    assert limiter.reconciled == ([(limiter.acquired[0][0], 0.0)] if expired_after_admission else [])
    out = tmp_path / 'timeout'
    assert not (out / 'raw/model/r000000/response.bin').exists()
    error = [e for e in scan(out / 'trajectory.jsonl')['events'] if e['kind'] == 'model.error'][-1]
    assert error['data']['transport_attempts'] == 0
    assert error['data']['stop_phase'] == ('deadline_after_admission' if expired_after_admission else 'admission_timeout')


@pytest.mark.parametrize('retry_after', [False, True])
def test_retry_backoff_exhaustion_keeps_wall_budget(tmp_path, config, components, monkeypatch, retry_after, matrix_script):
    now = [100.0]
    def sleep(seconds): now[0] += seconds
    monkeypatch.setattr(orchestrator, 'time', SimpleNamespace(monotonic=lambda: now[0], sleep=sleep))
    cfg = retry_config(config)
    cfg['budget']['wall_time_seconds'] = 0.04
    cfg['model']['retry']['backoff_seconds'] = [0.0 if retry_after else 1.0]
    cfg['model']['retry']['honor_retry_after'] = retry_after
    error = rate_limit()
    if retry_after: error.metadata['headers'] = {'retry-after': '1'}
    model = ScriptedGemini(cfg['model'], [error, gemini_done()])
    components['model'] = model
    limiter = SpyLimiter()
    out = tmp_path / 'backoff'
    result = orchestrator.run(cfg, out, injected=components, request_limiter=limiter)
    assert result['verification']['verified']
    assert result['status']['termination'] == 'wall_time_budget'
    assert result['status']['execution_validity'] == 'invalid'
    assert len(model.bodies) == len(limiter.acquired) == 1
    assert limiter.reconciled == [(limiter.acquired[0][0], None)]
    assert (out / 'raw/model/r000000/attempt_01/response.bin').exists()
    assert not (out / 'raw/model/r000000/attempt_02/response.bin').exists()
    row = matrix_script.summarize_episode('fixture', 0, out)
    assert row['completion_rate'] is None
    assert row['score_eligible'] is False
    assert row['score_classification'] == 'invalid_execution'


@pytest.mark.parametrize('response,expected_validity', [
    (HTTPResult(429, b'{"error":"rate limit"}', {'transport_error': None}), 'invalid'),
    (HTTPResult(None, b'', {'transport_error': {'type': 'TimeoutError'}}), 'invalid'),
    (gemini_done(), 'unknown'),  # Successful test execution is never live-qualified.
])
def test_deadline_after_receipt_retains_transport_validity(tmp_path, config, components, monkeypatch,
                                                         response, expected_validity):
    now = [100.0]
    monkeypatch.setattr(orchestrator, 'time', SimpleNamespace(monotonic=lambda: now[0], sleep=lambda seconds: None))
    class DeadlineModel(ScriptedGemini):
        def send(self, body, timeout):
            receipt = super().send(body, timeout)
            now[0] = 161.0
            return receipt
    components['model'] = DeadlineModel(config['model'], [response])
    result = orchestrator.run(config, tmp_path / 'late_receipt', injected=components)
    assert result['verification']['verified']
    assert result['status']['termination'] == 'wall_time_budget'
    assert result['status']['execution_validity'] == expected_validity


@pytest.fixture
def matrix_script():
    spec = importlib.util.spec_from_file_location('matrix_admission_regression', ROOT / 'scripts/l1/run_matrix.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('execution,evaluation,projection,grade_status,files,expected,classification', [
    ('invalid', 'valid', 'valid', 'ok', ['deliverable'], None, 'invalid_execution'),
    ('valid', 'invalid', 'invalid', 'error', ['deliverable'], None, 'invalid_official_evaluation'),
    ('valid', 'invalid', 'invalid', 'error', [], None, 'invalid_official_evaluation'),
    ('valid', 'valid', 'invalid', 'error', ['deliverable'], None, 'invalid_official_evaluation'),
    ('valid', 'valid', 'valid', 'error', ['deliverable'], None, 'invalid_official_evaluation'),
    ('valid', 'valid', 'valid', 'ok', [], 0.0, 'official_zero'),
    ('valid', 'valid', 'valid', 'ok', ['damaged'], 0.0, 'official_zero'),
    ('valid', 'valid', 'valid', 'ok', ['deliverable'], 0.5, 'official_score'),
    ('valid', 'valid', 'valid', 'ok', ['deliverable'], None, 'undefined_official_endpoint'),
])
def test_matrix_validity_gates_actual_packages(tmp_path, matrix_script, execution, evaluation, projection,
                                               grade_status, files, expected, classification):
    (tmp_path / 'artifacts').mkdir()
    (tmp_path / 'manifest.json').write_text(json.dumps({'status': {
        'execution_validity': execution, 'evaluation_validity': evaluation}}))
    (tmp_path / 'artifacts/inventory.json').write_text(json.dumps({'files': files}))
    rate = expected if expected is not None else (None if classification == 'undefined_official_endpoint' else 0.8)
    (tmp_path / 'official_grade.json').write_text(json.dumps({
        'projection': {'evaluation_validity': projection, 'completion_rate': rate},
        'official_raw': {'status': grade_status, 'dim1_pass': bool(expected)}}))
    row = matrix_script.summarize_episode('fixture', 0, tmp_path)
    assert row['completion_rate'] == expected
    assert row['score_eligible'] == (expected is not None)
    assert row['score_classification'] == classification
    if expected is None: assert row['dim1'] is None


def test_matrix_missing_grade_is_undefined(tmp_path, matrix_script):
    (tmp_path / 'manifest.json').write_text(json.dumps({'status': {
        'execution_validity': 'valid', 'evaluation_validity': 'valid'}}))
    row = matrix_script.summarize_episode('fixture', 0, tmp_path)
    assert row['completion_rate'] is None
    assert row['score_eligible'] is False
