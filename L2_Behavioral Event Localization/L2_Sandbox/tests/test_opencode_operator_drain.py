"""Operational closure tests; no actor, scorer, or Docker execution."""
import importlib.util
from pathlib import Path

import pytest

from l2_sandbox import opencode_d1_runtime as runtime
from l2_sandbox.v3_execution import file_digest, read_json, seal, write_json


@pytest.fixture
def closure_case(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / 'scripts/finalize_opencode_operator_drain.py'
    spec = importlib.util.spec_from_file_location('operator_drain_helper', script)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    monkeypatch.setattr(helper, '_alive', lambda pid: False)
    monkeypatch.setattr(runtime, '_verify_plan', lambda plan: None)
    phase = tmp_path / 'phase'
    output = phase / 'execution'
    (output / 'receipts').mkdir(parents=True)
    plan = seal(dict(limits={}, source_files={}, bound_files={}))
    write_json(phase / 'execution_plan.json', plan)
    write_json(phase / 'supervisor_started.json', dict(controller_pid=12345, started_at_epoch=10))
    write_json(phase / 'supervisor_result.json', dict(controller_pid=12345,
        started_at_epoch=10, finished_at_epoch=20, controller_exit_code=-2))
    drain = phase / 'operator_drain_request.json'
    write_json(drain, dict(controller_pid=12345))
    retained = dict(requests=10, tokens=100, cost_usd=1.0, invalid_attempts=2,
                    http_send_count=10, branch_invalid=0, fatal_invalid=2,
                    completed=0, infrastructure_invalid=0)
    write_json(phase / 'integrity_preflight.json', seal(dict(
        schema='opencode-d1-integrity-preflight-v1', plan_sha256=plan['artifact_sha256'],
        planned_jobs=2, retained=retained)))
    jobs = [dict(binding=dict(branch_id=str(i), plan_sha256=plan['artifact_sha256']),
                 path=output / 'receipts' / f'{i}.json', package=output / 'branches' / str(i))
            for i in range(2)]
    monkeypatch.setattr(runtime, '_scheduled_jobs', lambda plan, output: jobs)
    package = jobs[0]['package']
    (package / 'lineage').mkdir(parents=True)
    for filename in ('evidence_index.json', 'official_grade.json', 'lineage/actor_transition.json'):
        write_json(package / filename, {})
    write_json(jobs[0]['path'], seal(dict(jobs[0]['binding'], status='completed',
        valid_attempt=True, simulated_offline=False, outcome_score=0.5,
        package_path=str(package), package_index_sha256=file_digest(package / 'evidence_index.json'),
        official_grade_sha256=file_digest(package / 'official_grade.json'),
        model_transition_sha256=file_digest(package / 'lineage/actor_transition.json'))))
    write_json(output / 'admission.json', seal(dict(plan_sha256=plan['artifact_sha256'],
        limits={}, requests=1, tokens=12, cost_usd=0.1, invalid_attempts=0,
        sends=[dict(status='settled', transport_started=True)])))
    write_json(output / 'result.json', seal(dict(plan_sha256=plan['artifact_sha256'],
        status='incomplete', simulated_offline=False, d2_gate_open=False,
        completed_branches=0, execution_finished=False, process_finished=False)))
    return helper, phase, output, drain, jobs


def test_closure_census_preserves_originals_and_usage_and_is_idempotent(closure_case):
    helper, phase, output, drain, jobs = closure_case
    before = {n: (output / f'{n}.json').read_bytes() for n in ('result', 'admission')}
    admission = read_json(output / 'admission.json')
    manifest = helper.finalize(phase, drain)
    result = read_json(output / 'result.json')
    assert result['completed_branches'] == 1
    assert result['pending_branches'] == 1
    assert result['aggregate_requests'] == 11
    assert result['aggregate_http_send_count'] == 11
    assert result['process_finished'] is True
    assert result['status'] == 'incomplete'
    assert result['d2_gate_open'] is False
    closed = read_json(output / 'admission.json')
    assert {k: v for k, v in closed.items() if k not in ('artifact_sha256', 'stop_reason')} == {
        k: v for k, v in admission.items() if k != 'artifact_sha256'}
    assert closed['stop_reason'] == 'closed'
    for name in before:
        assert (output / 'operator_closure' / f'original_{name}.json').read_bytes() == before[name]
    hashes = {n: file_digest(output / f'{n}.json') for n in before}
    assert helper.finalize(phase, drain) == manifest
    assert hashes == {n: file_digest(output / f'{n}.json') for n in before}


@pytest.mark.parametrize('uncertain', ['live', 'inflight_send', 'inflight_receipt', 'orphan', 'missing_exit'])
def test_refuses_uncertain_drain_without_mutation(closure_case, uncertain, monkeypatch):
    helper, phase, output, drain, jobs = closure_case
    if uncertain == 'live':
        monkeypatch.setattr(helper, '_alive', lambda pid: True)
    elif uncertain == 'inflight_send':
        admission = read_json(output / 'admission.json')
        admission['sends'][0]['status'] = 'inflight'
        write_json(output / 'admission.json', seal(admission))
    elif uncertain == 'inflight_receipt':
        receipt = read_json(jobs[0]['path'])
        receipt.update(status='inflight', valid_attempt=False)
        write_json(jobs[0]['path'], seal(receipt))
    elif uncertain == 'orphan':
        (output / 'branches' / 'unbound').mkdir(parents=True)
    else:
        (phase / 'supervisor_result.json').unlink()
    before = {n: (output / f'{n}.json').read_bytes() for n in ('result', 'admission')}
    with pytest.raises((ValueError, FileNotFoundError)):
        helper.finalize(phase, drain)
    assert not (output / 'operator_closure').exists()
    assert all((output / f'{n}.json').read_bytes() == value for n, value in before.items())


def test_refuses_changed_receipt_on_idempotent_rerun(closure_case):
    helper, phase, output, drain, jobs = closure_case
    helper.finalize(phase, drain)
    receipt = read_json(jobs[0]['path'])
    receipt['outcome_score'] = 0.75
    write_json(jobs[0]['path'], seal(receipt))
    with pytest.raises(ValueError, match='reference changed'):
        helper.finalize(phase, drain)


def test_recovers_exact_prepared_replacement_after_interruption(closure_case):
    helper, phase, output, drain, jobs = closure_case
    manifest = helper.finalize(phase, drain)
    original = Path(manifest['operational_files']['result']['original']['path'])
    (output / 'result.json').write_bytes(original.read_bytes())
    assert helper.finalize(phase, drain) == manifest
    assert file_digest(output / 'result.json') == manifest['operational_files']['result']['final']['sha256']
