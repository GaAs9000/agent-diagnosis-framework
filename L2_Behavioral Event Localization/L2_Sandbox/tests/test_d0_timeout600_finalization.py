"""Run only on aml-b CPU; local E1 Docker is faked in this test suite."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/finalize_d0_timeout600.py'
spec = importlib.util.spec_from_file_location('t2_finalization_test', SCRIPT)
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


def sealed(value):
    return {**value, 'artifact_sha256': hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()}


def test_driver_receipt_seal_preserves_invalid_state():
    row = sealed({'package': {'valid': False}, 'error': 'transport_error'})
    d.seal_check(row)
    row['package']['valid'] = True
    with pytest.raises(ValueError, match='seal mismatch'):
        d.seal_check(row)


def test_wait_is_bounded_and_does_not_network(tmp_path, monkeypatch):
    monkeypatch.setattr(d.time, 'time', lambda: 101)
    monkeypatch.setattr(d.time, 'sleep', lambda *a: pytest.fail('wait exceeded deadline'))
    assert d.wait_for(tmp_path / 'missing', 100) is False


def test_deadline_admits_no_more_processes(monkeypatch):
    monkeypatch.setattr(d.time, 'time', lambda: 100)
    monkeypatch.setattr(d.subprocess, 'Popen', lambda *a, **k: pytest.fail('process admitted'))
    with pytest.raises(TimeoutError):
        d.bounded(['actor-must-never-start'], 99)


def test_remote_mode_rejects_other_directory(tmp_path):
    with pytest.raises(ValueError, match='unexpected isolated'):
        d.remote_audits(tmp_path)


def test_source_drift_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(d, 'ROOT', tmp_path)
    path = tmp_path / 'frozen.py'
    path.write_text('original')
    binding = {'sources': {'frozen.py': d.sha(path)}}
    d.assert_binding(binding)
    path.write_text('changed')
    with pytest.raises(ValueError, match='pinned source changed'):
        d.assert_binding(binding)


def test_e1_rejects_full_cohort_scope_before_scoring(tmp_path, monkeypatch):
    phase = tmp_path / 'phase'; phase.mkdir()
    binding = tmp_path / 'binding.json'
    d.save(binding, {'sources': {}, 'phase': str(phase), 'run_root': str(tmp_path / 'runs')})
    prepare_e1_input(tmp_path, phase, binding)
    census = SimpleNamespace(preflight=lambda path: {'slots': [{'slot': d.LABEL, 'status': 'ready'}, {'slot': 'officeval_003.r00', 'status': 'ready'}], 'study_valid': 2}, execute=lambda *a, **k: pytest.fail('broader scoring admitted'))
    monkeypatch.setattr(d, 'load', lambda *a: census)
    with pytest.raises(ValueError, match='single new valid 050'):
        d.e1_only(binding)


def test_incremental_e1_exact_scope_and_single_execution(tmp_path, monkeypatch):
    phase = tmp_path / 'phase'; phase.mkdir()
    binding = tmp_path / 'binding.json'
    d.save(binding, {'sources': {}, 'phase': str(phase), 'run_root': str(tmp_path / 'runs')})
    calls = []
    def execute(plan, **kwargs):
        calls.append(kwargs)
        return {'completed_count': 1, 'full_45_ready': False}
    prepare_e1_input(tmp_path, phase, binding)
    census = SimpleNamespace(preflight=lambda path: {'slots': [{'slot': d.LABEL, 'status': 'ready'}], 'study_valid': 1}, execute=execute)
    monkeypatch.setattr(d, 'load', lambda *a: census)
    d.e1_only(binding)
    assert calls == [{'diag_dir': phase / 'e1/diagnoses', 'validation': phase / 'e1/validation'}]
    with pytest.raises(FileExistsError):
        d.e1_only(binding)
    assert len(calls) == 1


def test_snapshot_preserves_partial_and_invalid_and_timeout_metadata(tmp_path):
    phase = tmp_path / 'phase'; phase.mkdir()
    runs = tmp_path / 'runs'
    parent = runs / 'episodes' / d.LABEL
    d.save(parent / 'evidence_index.json', {})
    d.save(parent / 'raw/resolved_config.json', {'model': {'timeout_seconds': 600}})
    d.save(phase / f'receipt_{d.LABEL}.json', sealed({'package': {'valid': False, 'index_file_sha256': d.sha(parent / 'evidence_index.json')}, 'parent_timeout_seconds': None}))
    (runs / 'v3_g2_c0_resumes/officeval_003.r00_c0').mkdir(parents=True)
    snap = d.snapshot(phase, {'run_root': str(runs), 'parents': {d.LABEL: {'package': {'path': '/old/050'}}}})
    assert snap['new_packages'][f'episodes/{d.LABEL}']['driver_study_valid'] is False
    assert snap['new_packages'][f'episodes/{d.LABEL}']['effective_timeout_seconds'] == 600
    assert snap['new_packages']['v3_g2_c0_resumes/officeval_003.r00_c0']['sealed'] is False
    assert snap['g2']['receipt_count'] == 0
    assert snap['g2']['gate'] == 'undecided'


def test_partial_manifest_saved_even_when_remote_dispatch_fails(tmp_path, monkeypatch):
    phase = tmp_path / 'phase'; phase.mkdir()
    base = tmp_path / 'base'; monkeypatch.setattr(d, 'BASE', base); monkeypatch.setattr(d, 'ROOT', tmp_path)
    for name in ('build_d0_census.py', 'recover_d0_serial.py', 'finalize_d0_timeout600.py'):
        path = tmp_path / d.L2 / 'scripts' / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('frozen')
    d.save(base / 'prerequisite_resume/executed_plan.json', {'source_fingerprints': {}})
    d.save(base / 'serial_postprocess_recovery/manifest.json', {})
    d.save(phase / 'executed_plan.json', {'run_root': str(tmp_path / 'runs'), 'source_fingerprints': {}, 'parents': {}, 'service_stop_epoch': 0})
    monkeypatch.setattr(d, 'load', lambda *a: SimpleNamespace(source_inventory=lambda root: {}))
    monkeypatch.setattr(d.time, 'time', lambda: 200)
    monkeypatch.setattr(d.time, 'sleep', lambda *a: pytest.fail('unexpected wait'))
    monkeypatch.setattr(d, 'dispatch', lambda *a: (_ for _ in ()).throw(RuntimeError('server unavailable')))
    state = d.finalize(phase)
    assert state['status'] == 'partial_snapshot_supervisor_missing'
    assert state['remote_audits']['status'] == 'pending_failed_or_deadline'
    assert d.read(phase / 'postprocess/manifest.json')['gates_promoted'] is False
    with pytest.raises(FileExistsError):
        d.finalize(phase)


def prepare_e1_input(tmp_path, phase, binding):
    parent_index = tmp_path / 'runs/episodes' / d.LABEL / 'evidence_index.json'
    d.save(parent_index, {})
    receipt = phase / f'receipt_{d.LABEL}.json'
    d.save(receipt, {})
    d.save(binding.parent / 'e1_input.json', {'receipt_sha256': d.sha(receipt), 'index_sha256': d.sha(parent_index), 'owner': d.sha(binding)})


def test_e1_rejects_parent_resealed_after_admission(tmp_path, monkeypatch):
    phase = tmp_path / 'phase'; phase.mkdir()
    binding = tmp_path / 'binding.json'
    d.save(binding, {'sources': {}, 'phase': str(phase), 'run_root': str(tmp_path / 'runs')})
    prepare_e1_input(tmp_path, phase, binding)
    d.save(tmp_path / 'runs/episodes' / d.LABEL / 'evidence_index.json', {'modified': True})
    monkeypatch.setattr(d, 'load', lambda *a: pytest.fail('census admitted after reseal'))
    with pytest.raises(ValueError, match='parent changed after receipt admission'):
        d.e1_only(binding)


def test_e1_create_tracking_and_signal_unwinds_original_cleanup(tmp_path, monkeypatch):
    import signal
    import subprocess
    calls = []
    monkeypatch.setattr(d.subprocess, 'run', lambda args, *a, **k: calls.append(args))
    before = d.subprocess.run
    cleaned = []
    with pytest.raises(d.E1Terminated):
        with d.owned_e1_containers(tmp_path, 'a' * 64):
            try:
                d.subprocess.run(['docker', 'create', '--label', 'l1.managed=true', 'frozen-image'])
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            finally:
                cleaned.append(True)  # The original DockerRuntime finally can unwind.
    assert cleaned == [True]
    assert d.subprocess.run is before
    assert calls[0][:6] == ['docker', 'create', '--label', 'd0.e1.owner=' + 'a' * 64, '--cidfile', str(tmp_path / 'e1/container_ids/000000.cid')]


def test_owned_cleanup_removes_only_exact_owner_and_checks_empty(tmp_path, monkeypatch):
    import subprocess
    owned, unrelated = 'a' * 64, 'b' * 64
    live = {owned, unrelated}
    calls = []
    def docker(args, **kwargs):
        calls.append(args)
        if args[1] == 'ps':
            assert args[-1] == 'label=d0.e1.owner=' + 'c' * 64
            output = (owned + '\n').encode() if owned in live else b''
        else:
            assert args == ['docker', 'rm', '-f', owned]
            live.remove(owned)
            output = b''
        return subprocess.CompletedProcess(args, 0, stdout=output)
    monkeypatch.setattr(d.subprocess, 'run', docker)
    record = d.cleanup_e1(tmp_path, 'c' * 64, d.time.time() + 60)
    assert record['status'] == 'confirmed_empty'
    assert live == {unrelated}
    assert len(calls) == 3


def e1_controller_fixture(tmp_path, monkeypatch):
    phase = tmp_path / 'phase'; phase.mkdir()
    base = tmp_path / 'base'; monkeypatch.setattr(d, 'BASE', base); monkeypatch.setattr(d, 'ROOT', tmp_path)
    for name in ('build_d0_census.py', 'recover_d0_serial.py', 'finalize_d0_timeout600.py'):
        path = tmp_path / d.L2 / 'scripts' / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('frozen')
    d.save(base / 'prerequisite_resume/executed_plan.json', {'source_fingerprints': {}})
    d.save(base / 'serial_postprocess_recovery/manifest.json', {})
    parent = tmp_path / 'runs/episodes' / d.LABEL
    d.save(parent / 'evidence_index.json', {})
    d.save(parent / 'raw/resolved_config.json', {'model': {'timeout_seconds': 600}})
    d.save(phase / f'receipt_{d.LABEL}.json', sealed({'package': {'valid': True, 'index_file_sha256': d.sha(parent / 'evidence_index.json')}, 'parent_timeout_seconds': None}))
    d.save(phase / 'executed_plan.json', {'run_root': str(tmp_path / 'runs'), 'source_fingerprints': {}, 'parents': {}, 'service_stop_epoch': 1000})
    d.save(phase / 'supervisor_result.json', {'wall_timeout': True})
    monkeypatch.setattr(d, 'load', lambda *a: SimpleNamespace(source_inventory=lambda root: {}))
    monkeypatch.setattr(d, 'dispatch', lambda *a: {'status': 'pending_test_scope'})
    monkeypatch.setattr(d.time, 'time', lambda: 100)
    return phase


def test_e1_timeout_scoped_cleanup_leaves_no_owned_survivor(tmp_path, monkeypatch):
    import subprocess
    phase = e1_controller_fixture(tmp_path, monkeypatch)
    live = {'a' * 64, 'b' * 64}
    def timeout(args, deadline, seconds=900, **kwargs):
        assert deadline == 1025 and kwargs['termination_grace'] == 30
        raise subprocess.TimeoutExpired(args, seconds)
    def docker(args, **kwargs):
        if args[1] == 'ps':
            output = ('a' * 64 + '\n').encode() if 'a' * 64 in live else b''
        else:
            assert args == ['docker', 'rm', '-f', 'a' * 64]
            live.remove('a' * 64); output = b''
        return subprocess.CompletedProcess(args, 0, stdout=output)
    monkeypatch.setattr(d, 'bounded', timeout)
    monkeypatch.setattr(d.subprocess, 'run', docker)
    state = d.finalize(phase)
    assert state['e1']['status'] == 'failed_or_timed_out'
    assert state['e1']['container_cleanup']['status'] == 'confirmed_empty'
    assert live == {'b' * 64}
    assert d.read(phase / 'e1/cleanup.json')['owned_ids'] == ['a' * 64]


def test_cleanup_failure_never_records_e1_complete(tmp_path, monkeypatch):
    import subprocess
    phase = e1_controller_fixture(tmp_path, monkeypatch)
    def completed(args, *a, **kwargs):
        d.save(phase / 'e1/validation/manifest.json', {'completed_count': 1, 'full_45_ready': False})
    monkeypatch.setattr(d, 'bounded', completed)
    monkeypatch.setattr(d.subprocess, 'run', lambda args, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired(args, 20)))
    state = d.finalize(phase)
    assert state['e1']['status'] == 'cleanup_unconfirmed'
    assert state['e1']['container_cleanup']['status'] == 'unconfirmed'


def test_remote_receipt_separates_g1_g4_and_seal_scopes(tmp_path, monkeypatch):
    import subprocess
    monkeypatch.setattr(d, 'REMOTE', tmp_path)
    data = tmp_path / ('a' * 64); data.mkdir()
    parent = 'episodes/' + d.LABEL
    child = 'v3_g2_c0_resumes/officeval_003.r00_c0'
    invalid = 'v3_g2_c0_resumes/officeval_004.r00_c0'
    selected = {name: {'index_sha256': 'b' * 64} for name in (parent, child, invalid)}
    d.save(data / 'binding.json', {'sources': {}, 'old44_audits': {'sha256': 'old44'}})
    d.save(data / 'inventory.json', selected)
    monkeypatch.setattr(d, 'load', lambda *a: SimpleNamespace(verify_one=lambda p, h: {'study_outcome': {'study_valid': p.name != 'officeval_004.r00_c0'}}))
    commands = []
    def audit(args, **kwargs):
        commands.append(args)
        d.save(Path(args[args.index('--output') + 1]), {'gate_pass': False})
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(d.subprocess, 'run', audit)
    d.remote_audits(data)
    status = d.read(data / 'reports/status.json')
    assert status['per_audit_scopes'] == {'sealed_verification': sorted(selected), 'g1': [parent], 'g4': sorted([parent, child])}
    assert '--runs' in commands[0] and commands[0][commands[0].index('--runs') + 1] == str(data / 'runs/episodes')
    assert str(data / 'runs' / invalid) not in commands[1]
