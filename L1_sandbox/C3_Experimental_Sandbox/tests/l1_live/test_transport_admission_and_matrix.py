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


def test_external_stage_deadline_bounds_model_and_tool_without_changing_config(tmp_path, config, components, monkeypatch):
    monkeypatch.setattr(orchestrator, 'time', SimpleNamespace(monotonic=lambda: 100.0, sleep=lambda seconds: None))
    observed_model, observed_tool = [], []
    model, runtime = components['model'], components['runtime']
    send, execute = model.send, runtime.execute
    def bounded_send(body, timeout):
        observed_model.append(timeout)
        return send(body, timeout)
    def bounded_execute(call, workspace, timeout):
        observed_tool.append(timeout)
        return execute(call, workspace, timeout)
    monkeypatch.setattr(model, 'send', bounded_send)
    monkeypatch.setattr(runtime, 'execute', bounded_execute)
    limiter = SpyLimiter()
    limiter.remaining_seconds = lambda: .25
    output = tmp_path / 'external-deadline'
    result = orchestrator.run(config, output, injected=components, request_limiter=limiter)
    assert result['verification']['verified']
    assert observed_model and observed_tool
    assert all(0 < timeout <= .25 for timeout in observed_model + observed_tool)
    assert read_json(output / 'raw/resolved_config.json')['budget']['wall_time_seconds'] == 60
    started = scan(output / 'trajectory.jsonl')['events'][0]
    assert started['data']['external_wall_seconds'] == .25


def test_expired_external_stage_deadline_sends_nothing(tmp_path, config, components):
    limiter = SpyLimiter()
    limiter.remaining_seconds = lambda: 0.0
    result = orchestrator.run(config, tmp_path / 'expired-stage', injected=components, request_limiter=limiter)
    assert result['verification']['verified']
    assert result['status']['termination'] == 'wall_time_budget'
    assert components['model'].bodies == []
    assert limiter.acquired == []


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
    assert row['score_classification'] == 'invalid_execution_or_evidence'


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


@pytest.mark.parametrize('case,expected', [
    ('invalid_execution', None), ('missing_grade', None), ('process_failure', None),
    ('malformed_grade', None), ('projection_mismatch', None),
    ('missing_delivery', 0.0), ('damaged_delivery', 0.0),
    ('valid_zero', 0.0), ('valid_score', 0.5), ('undefined_endpoint', None),
])
def test_matrix_validity_gates_actual_packages(tmp_path, matrix_script, case, expected):
    from c3_sandbox.l1.evaluation.officeval import interpret
    manifest = {'episode_id': 'matrix-fixture', 'recording_mode': 'live',
                'task': {'task_id': 'officeval_003', 'benchmark': 'officeval'},
                'status': {'execution_validity': 'valid', 'evaluation_validity': 'valid',
                           'evidence_completeness': 'complete'}}
    raw = {'id': '003', 'file_name': 'result.docx', 'status': 'ok', 'error': None,
           'dim1_pass': True, 'dim1_reason': '', 'dim2_items': [],
           'total_score': 5, 'max_score': 10}
    if case == 'invalid_execution': manifest['status']['execution_validity'] = 'invalid'
    if case in ('missing_delivery', 'damaged_delivery'):
        raw.update(status='error', error='FileNotFoundError' if case == 'missing_delivery' else 'BadZipFile',
                   dim1_pass=False, total_score=0)
        manifest['status']['evaluation_validity'] = 'invalid'
    if case == 'valid_zero': raw.update(total_score=0)
    if case == 'malformed_grade': raw.pop('id')
    if case == 'undefined_endpoint': raw.update(total_score=0, max_score=0)
    projection = interpret(raw, 'officeval_003')
    if case == 'projection_mismatch': projection['completion_rate'] = .8
    grade = {'official_raw': raw, 'projection': projection,
             'process': {'container_exit_code': 1 if case == 'process_failure' else 0}}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    if case != 'missing_grade':
        (tmp_path / 'official_grade.json').write_text(json.dumps(grade))
    worker = tmp_path / 'grading/worker-output/worker.json'
    worker.parent.mkdir(parents=True)
    worker.write_text(json.dumps({'official_returned': True, 'exception': None}))
    row = matrix_script.summarize_episode('officeval_003', 0, tmp_path)
    assert row['completion_rate'] == row['study_outcome_y'] == expected
    assert row['score_eligible'] == row['study_valid'] == (expected is not None)
    assert row['score_classification'] == row['study_classification']
    if expected is None:
        assert row['dim1'] is None
    elif case in ('missing_delivery', 'damaged_delivery'):
        assert row['evaluation_validity'] == row['native_evaluation_validity'] == 'invalid'
        assert row['zero_reason'] == 'missing_or_damaged_delivery'


def test_matrix_missing_grade_is_undefined(tmp_path, matrix_script):
    (tmp_path / 'manifest.json').write_text(json.dumps({'status': {
        'execution_validity': 'valid', 'evaluation_validity': 'valid'}}))
    row = matrix_script.summarize_episode('fixture', 0, tmp_path)
    assert row['completion_rate'] is None
    assert row['score_eligible'] is False
