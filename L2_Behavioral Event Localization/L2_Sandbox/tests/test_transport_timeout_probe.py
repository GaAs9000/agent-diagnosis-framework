"""Offline diagnostic contracts; all model and metrics I/O is replaced."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/probe_transport_timeout.py'
spec = importlib.util.spec_from_file_location('timeout_probe', SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def signed(plan):
    plan = copy.deepcopy(plan)
    plan.pop('plan_sha256', None)
    plan['plan_sha256'] = probe.hashlib.sha256(probe.canonical(plan)).hexdigest()
    return plan


@pytest.fixture
def setup(tmp_path):
    serial = tmp_path / 'serial'; serial.mkdir()
    probe.save(serial / 'supervisor_result.json', dict(exit_code=0, finished_at_utc='sealed'))
    probe.save(serial / 'admission.json', dict(requests=2, sends=[{}, {}]))
    body = dict(model='Qwen/Qwen3.6-27B', temperature=1, max_tokens=8192,
                messages=[dict(role='user', content='original')], stream=False, n=1)
    config = dict(model=dict(kind='local', model=body['model'], base_url='http://127.0.0.1:8001/v1',
                            timeout_seconds=300, parameters=dict(temperature=1, max_tokens=8192)))
    probe.save(tmp_path / 'request.json', body); probe.save(tmp_path / 'config.json', config)
    plan = signed(dict(request_path=str(tmp_path / 'request.json'), config_path=str(tmp_path / 'config.json'),
                       bound_files={str(tmp_path / 'request.json'): probe.sha(tmp_path / 'request.json'),
                                    str(tmp_path / 'config.json'): probe.sha(tmp_path / 'config.json')},
                       serial_dir=str(serial), output_dir=str(tmp_path / 'output'), old_http_attempts=93,
                       timeout_seconds=600, safety_seconds=645, max_http_attempts=1,
                       retry_attempts=0, aggregate_http_cap=3360, max_wait_seconds=21600,
                       poll_seconds=15, service_stop_epoch=10000))
    calls = []

    class Backend:
        def __init__(self, config):
            assert config['timeout_seconds'] == 300

        def send(self, data, timeout):
            calls.append((data, timeout))
            return SimpleNamespace(status=200, body=b'{"choices":[{"finish_reason":"tool_calls"}],"usage":{"completion_tokens":8192}}',
                                   metadata=dict(attempts=1, transport_error=None, body_truncated=False))

    def approval(p=plan):
        return dict(approved_plan_sha256=p['plan_sha256'], exact_answer=probe.ANSWER,
                    question_id=probe.QUESTION, internal_plan_reviewed=True)

    kwargs = dict(backend_factory=Backend, now=lambda: 1000,
                  metrics_reader=lambda: {'vllm:num_requests_running': 0, 'vllm:num_requests_waiting': 0})
    return plan, approval, calls, kwargs


def test_single_native_call_exact_body_and_raw_tools_never_executed(setup):
    plan, approval, calls, kwargs = setup
    original_config = Path(plan['config_path']).read_bytes()
    result = probe.execute(plan, approval(), **kwargs)
    assert calls == [(probe.canonical(probe.read(plan['request_path'])), 600)]
    assert result['status'] == 'transport_response' and result['tools_executed'] == 0
    assert result['finish_reasons'] == ['tool_calls']
    assert Path(plan['config_path']).read_bytes() == original_config
    assert (Path(plan['output_dir']) / 'response.bin').read_bytes().startswith(b'{"choices"')
    with pytest.raises(ValueError, match='already reserved'):
        probe.execute(plan, approval(), **kwargs)
    assert len(calls) == 1


@pytest.mark.parametrize('field,value', [('question_id', 'wrong'), ('exact_answer', 'yes'),
                                        ('internal_plan_reviewed', False), ('approved_plan_sha256', 'wrong')])
def test_exact_approval_required(setup, field, value):
    plan, approval, calls, kwargs = setup
    auth = approval(); auth[field] = value
    with pytest.raises(ValueError):
        probe.execute(plan, auth, **kwargs)
    assert not calls


def test_waits_for_full_supervisor_and_both_metrics_zero(setup):
    plan, approval, calls, kwargs = setup
    final = Path(plan['serial_dir']) / 'supervisor_result.json'; final.unlink()
    samples = iter([{'vllm:num_requests_running': 0, 'vllm:num_requests_waiting': 1},
                    {'vllm:num_requests_running': 0, 'vllm:num_requests_waiting': 0}])
    ticks = []

    def sleep(seconds):
        ticks.append(seconds)
        probe.save(final, dict(exit_code=0, finished_at_utc='sealed'))

    kwargs.update(sleep=sleep, metrics_reader=lambda: next(samples))
    probe.execute(plan, approval(), **kwargs)
    assert ticks == [15, 15] and len(calls) == 1


def test_unknown_metrics_do_not_send(setup):
    plan, approval, calls, kwargs = setup
    kwargs['metrics_reader'] = lambda: {'vllm:num_requests_running': 0}
    with pytest.raises(ValueError, match='unknown server'):
        probe.execute(plan, approval(), **kwargs)
    assert not calls


@pytest.mark.parametrize('remaining', [644, 0])
def test_service_deadline_prevents_send(setup, remaining):
    plan, approval, calls, kwargs = setup
    plan['service_stop_epoch'] = 1000 + remaining; plan = signed(plan)
    with pytest.raises(RuntimeError, match='deadline'):
        probe.execute(plan, approval(plan), **kwargs)
    assert not calls


def test_source_tamper_and_exhausted_budget_prevent_send(setup):
    plan, approval, calls, kwargs = setup
    ledger = Path(plan['serial_dir']) / 'admission.json'
    probe.save(ledger, dict(requests=3267, sends=[{}] * 3267))
    with pytest.raises(ValueError, match='budget exhausted'):
        probe.execute(plan, approval(), **kwargs)
    Path(plan['request_path']).write_text('{}')
    with pytest.raises(ValueError, match='bound input changed'):
        probe.execute(plan, approval(), **kwargs)
    assert not calls


def test_crash_consumes_authorization_without_retry(setup):
    plan, approval, calls, kwargs = setup

    class Broken:
        def __init__(self, config): pass
        def send(self, body, timeout):
            calls.append(1)
            raise TimeoutError('do not store exception text')

    kwargs['backend_factory'] = Broken
    with pytest.raises(TimeoutError):
        probe.execute(plan, approval(), **kwargs)
    assert probe.read(Path(plan['output_dir']) / 'result.json')['status'] == 'unknown_transport_exception'
    with pytest.raises(ValueError, match='already reserved'):
        probe.execute(plan, approval(), **kwargs)
    assert calls == [1]


def test_http_failure_is_preserved_without_retry(setup):
    plan, approval, calls, kwargs = setup

    class Failure:
        def __init__(self, config): pass
        def send(self, body, timeout):
            calls.append(1)
            return SimpleNamespace(status=503, body=b'not JSON', metadata=dict(attempts=1))

    kwargs['backend_factory'] = Failure
    result = probe.execute(plan, approval(), **kwargs)
    assert result['status'] == 'transport_failure' and result['raw_json_parse'].startswith('failed')
    assert calls == [1]


def test_wait_limit_without_supervisor_prevents_send(setup):
    plan, approval, calls, kwargs = setup
    (Path(plan['serial_dir']) / 'supervisor_result.json').unlink()
    clock = iter([1000, 22600])
    kwargs['now'] = lambda: next(clock)
    with pytest.raises(RuntimeError, match='deadline'):
        probe.execute(plan, approval(), **kwargs)
    assert not calls
