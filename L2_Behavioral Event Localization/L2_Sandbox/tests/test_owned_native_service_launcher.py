"""Inert launcher and birth receipts; no GPU subprocess or model request."""
import importlib.util
import io
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/start_owned_native_service.py'
spec = importlib.util.spec_from_file_location('owned_service_launcher_test', SCRIPT)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_default_is_inert_and_does_not_read_stdin(monkeypatch, capsys):
    monkeypatch.setattr(launcher, 'launch', lambda _: pytest.fail('unexpected launch'))
    monkeypatch.setattr(launcher.sys, 'stdin', None)
    launcher.main([])
    assert json.loads(capsys.readouterr().out)['status'] == 'not_executed'


@pytest.mark.parametrize('flags', [{}, {'phase_authorized': True},
                                  {'phase_authorized': True, 'phase_frozen': False}])
def test_unfrozen_phase_rejected_before_any_gpu_query(flags, monkeypatch):
    monkeypatch.setattr(launcher.subprocess, 'check_output', lambda *a, **k: pytest.fail('GPU queried'))
    with pytest.raises(ValueError, match='frozen phase'):
        launcher.launch(flags)


def test_execute_requires_and_passes_explicit_phase_payload(monkeypatch, capsys):
    payload = {'phase_authorized': True, 'phase_frozen': True}
    seen = []
    monkeypatch.setattr(launcher.sys, 'stdin', io.StringIO(json.dumps(payload)))
    monkeypatch.setattr(launcher, 'launch', lambda v: seen.append(v) or {'leader_start_ticks': 12345})
    launcher.main(['--execute'])
    assert seen == [payload]
    assert json.loads(capsys.readouterr().out)['leader_start_ticks'] == 12345


def test_process_birth_receipt_uses_stat_after_padded_title(tmp_path):
    proc = tmp_path / '123'
    proc.mkdir()
    fields = ['S', '1', '123'] + ['0'] * 16 + ['987654']
    (proc / 'stat').write_text('123 (Qwen title ) padded) ' + ' '.join(fields))
    assert launcher.leader_identity(123, tmp_path) == {'start_ticks': 987654}
    fields[2] = '456'
    (proc / 'stat').write_text('123 (timeout) ' + ' '.join(fields))
    with pytest.raises(ValueError, match='process group'):
        launcher.leader_identity(123, tmp_path)


def test_full_launch_records_real_birth_and_original_deadline_with_mocked_gpu(tmp_path, monkeypatch):
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'config.json').write_text('{}')
    monkeypatch.setattr(launcher, 'MODEL_ROOT', str(model))
    monkeypatch.setattr(launcher, 'SERVICE_ROOT', tmp_path / 'services')
    uuid = 'GPU-6d3f754b-b139-9354-0cbc-42bf22208180'
    def query(args, **_):
        assert args[0] == 'nvidia-smi'
        return f'4, {uuid}, 3, 0\n' if '--query-gpu=index,uuid,memory.used,utilization.gpu' in args else ''
    monkeypatch.setattr(launcher.subprocess, 'check_output', query)
    class Socket:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def connect_ex(self, address):
            assert address == ('127.0.0.1', 8001)
            return 1
    monkeypatch.setattr(launcher.socket, 'socket', Socket)
    monkeypatch.setattr(launcher.shutil, 'which', lambda name, **_: '/usr/local/cuda-13.0/bin/nvcc' if name == 'nvcc' else '/env/bin/' + name)
    monkeypatch.setattr(launcher.time, 'time', lambda: 1000.0)
    monkeypatch.setattr(launcher, 'leader_identity', lambda pid: {'start_ticks': 987654} if pid == 123 else pytest.fail('wrong leader'))
    calls = []
    class Process:
        pid = 123
        def __init__(self, args, **kwargs): calls.append((list(args), kwargs))
    monkeypatch.setattr(launcher.subprocess, 'Popen', Process)
    value = dict(phase_authorized=True, phase_frozen=True, plan_sha256='a' * 64,
                 local_start_epoch=1000.0, service_stop_epoch=22000.0,
                 clock_calibration=dict(local_sent_epoch=1899.9, local_received_epoch=1900.0, remote_epoch=900.0))
    receipt = launcher.launch(value)
    command, options = calls[0]
    assert options['start_new_session'] is True
    assert options['env']['CUDA_VISIBLE_DEVICES'] == '4'
    assert '/home/jiasheren2/envs/idea-n1/bin:' in options['env']['PATH']
    assert options['env']['CUDA_HOME'] == '/usr/local/cuda-13.0'
    assert command[:4] == ['timeout', '--signal=TERM', '--kill-after=30s', f"{receipt['timeout_seconds']}s"]
    assert receipt['remote_launch_epoch'] + receipt['timeout_seconds'] + 30 <= receipt['remote_hard_stop_epoch']
    assert receipt['hard_stop_epoch'] == value['local_start_epoch'] + 21600
    assert receipt['leader_start_ticks'] == 987654
    assert json.loads((Path(receipt['remote_directory']) / 'receipt.json').read_text()) == receipt
