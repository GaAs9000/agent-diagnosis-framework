"""Offline report contract checks; run on the AML CPU worker only."""
import importlib.util
from pathlib import Path

import pytest

from l2_sandbox import opencode_d1_runtime as runtime
from l2_sandbox.opencode_finish_report import render_markdown, summarize, summarize_usage
from l2_sandbox.v3_execution import file_digest, read_json, seal, verify_seal, write_json


@pytest.fixture
def plan():
    def unit(i, arms):
        messages = {a: dict(eligible=True, message=None, message_sha256=None) for a in arms}
        return dict(unit_id=f'u{i}', task_id=f't{i}', anchor_k=2, remaining_budget=33,
            parent_identity=dict(parent_episode_id=f'ep_{i // 2}'),
            unit_level='rubric_item', defect_name=f'rule_{i}',
            eligible_arms=arms, arms=arms, arm_messages=messages,
            schedule=runtime.build_interleaved_arm_schedule(arms, 14, anchor_k=2, remaining_budget=33))
    failures = [unit(i, ['P', 'D', 'S', 'R_explicit', 'R_implicit'] +
                     (['R_single'] if i < 13 else [])) for i in range(24)]
    controls = [unit(i, ['P', 'D', 'S', 'R_explicit', 'R_single']) for i in range(24, 30)]
    science = seal(dict(unit_schedules=failures,
        r12_passing_harm_check_plan=dict(sampled_passing_units=controls),
        task_ladder_message_specs={u['task_id']: dict(ladder_messages=u['arm_messages'])
                                   for u in controls}), 'plan_sha256')
    return seal(dict(schema='cross-actor-d1-execution-v1', automatic_d2=False,
                     scientific_plan=science, model=dict(model=runtime.MODEL)))


def ledger(plan, sends=()):
    tokens = sum(row['actual_tokens'] if row['status'] == 'settled' else row['tokens_reserved']
                 for row in sends)
    cost = sum(row['actual_cost_usd'] if row['status'] == 'settled' else row['cost_reserved']
               for row in sends)
    return seal(dict(plan_sha256=plan['artifact_sha256'], started_at=1,
                     stop_reason='completed', sends=list(sends), requests=len(sends),
                     tokens=tokens, cost_usd=cost))


def receipt(plan, unit_id='u0', arm='P', replicate=1, score=.5, invalid=None):
    jobs = runtime._scheduled_jobs(plan, Path('/unused'))
    job = next(j for j in jobs if j['binding']['unit_id'] == unit_id
               and j['binding']['arm'] == arm and j['binding']['replicate'] == replicate)
    values = dict(job['binding'], simulated_offline=False,
                  status=invalid or 'completed', valid_attempt=invalid is None)
    if invalid is None:
        values['outcome_score'] = score
    return seal(values)


def test_census_preserves_missing_invalid_and_actual_cell_n(plan):
    receipts = [receipt(plan, score=0), receipt(plan, replicate=2, invalid='branch_invalid'),
                receipt(plan, replicate=3, invalid='infrastructure_invalid')]
    report = summarize(plan, receipts, [ledger(plan)])
    assert report['census'] == dict(planned=2282, valid=1, invalid_or_incomplete=2, unstarted=2279)
    assert report['cell_count'] == 163 and report['unit_count'] == 30
    cell = next(c for c in report['cells'] if c['unit_id'] == 'u0' and c['arm'] == 'P')
    assert (cell['planned'], cell['valid'], cell['invalid_or_incomplete'], cell['unstarted']) == (14, 1, 2, 11)
    assert cell['mean_target_endpoint_score'] == 0
    assert report['receipt_status_counts']['branch_invalid'] == 1
    assert report['automatic_d2'] is False and report['d2_gate_open'] is False
    assert report['formal_pvalue_gate_passed'] is False
    assert report['verification_scope'] == 'receipt_seals_and_plan_bindings'
    assert report['package_reverification_performed'] is False
    assert 'not whole-benchmark scores' in render_markdown(report)


def test_arm_baseline_matches_same_available_eligible_units(plan):
    receipts = [receipt(plan, 'u0', 'P', score=.2), receipt(plan, 'u0', 'D', score=.8),
                receipt(plan, 'u1', 'P', score=1), receipt(plan, 'u2', 'D', score=0),
                receipt(plan, 'u0', 'D', replicate=2, invalid='branch_invalid')]
    report = summarize(plan, receipts, [ledger(plan)])
    row = next(r for r in report['arm_summaries'] if r['arm'] == 'D' and not r['passing_control'])
    assert row['eligible_units'] == 24 and row['units_with_valid_scores'] == 2
    assert row['matched_units'] == 1
    assert row['matched_unit_arm_mean'] == .8 and row['matched_unit_baseline_mean'] == .2
    assert row['matched_unit_difference'] == pytest.approx(.6)
    assert row['matched_unit_details'][0]['unit_id'] == 'u0'


def test_usage_own_ledgers_unknown_http_codes_and_no_aggregate_double_count(plan):
    old = seal({**plan, 'phase': 'old'})
    settled = dict(status='settled', http_status=200, actual_tokens=100,
                   actual_cost_usd=.01, usage=dict(total_tokens=100), transport_started=True)
    unknown = dict(status='unknown_usage', tokens_reserved=500, cost_reserved=.05, transport_started=True)
    a = ledger(old, [settled, unknown])
    b = ledger(plan, [{**unknown, 'http_status': 503}])
    # Historical aggregate fields must have no effect on the own-row census.
    b = seal({**b, 'aggregate_tokens': 999999})
    usage = summarize_usage([a, b])
    assert usage['admitted_requests'] == 3
    assert usage['reported_settled_actual_tokens'] == 100
    assert usage['unknown_reserved_tokens'] == 1000
    assert usage['accounted_tokens_including_unknown_reservations'] == 1100
    assert usage['unknown_usage_http_status_counts'] == {'503': 1, 'no_http_status': 1}
    assert usage['http_status_counts'] == {'200': 1, '503': 1, 'no_http_status': 1}
    assert usage['billed_cash_usd'] is None and usage['unknown_usage_refunded'] is False
    with pytest.raises(ValueError, match='duplicate phase'):
        summarize_usage([a, a])
    with pytest.raises(ValueError, match='accounting differs'):
        summarize_usage([seal({**b, 'tokens': 0})])
    with pytest.raises(ValueError, match='unsettled'):
        summarize_usage([ledger(plan, [{**unknown, 'status': 'inflight'}])])


@pytest.mark.parametrize('change', [dict(unit_id='other'), dict(outcome_score=2),
                                  dict(simulated_offline=True), dict(valid_attempt=False)])
def test_resealed_wrong_receipt_rejected(plan, change):
    changed = seal({**receipt(plan), **change})
    with pytest.raises(ValueError):
        summarize(plan, [changed], [ledger(plan)])


def test_duplicate_and_unsealed_receipts_rejected(plan):
    row = receipt(plan)
    with pytest.raises(ValueError, match='duplicate'):
        summarize(plan, [row, row], [ledger(plan)])
    row['outcome_score'] = .9
    with pytest.raises(ValueError, match='seal'):
        summarize(plan, [row], [ledger(plan)])


def test_invalid_outcome_cannot_become_zero_score(plan):
    bad = seal({**receipt(plan, invalid='branch_invalid'), 'outcome_score': 0})
    with pytest.raises(ValueError, match='invalid receipt'):
        summarize(plan, [bad], [ledger(plan)])


def load_cli():
    path = Path(__file__).resolve().parents[1] / 'scripts/summarize_opencode_finish.py'
    spec = importlib.util.spec_from_file_location('finish_report_cli', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_portable_cli_walks_chain_seals_support_and_refuses_running(plan, tmp_path):
    cli = load_cli()
    remote = tmp_path / 'remote'
    previous = remote / 'previous'
    previous.mkdir(parents=True)
    old_plan = seal({**plan, 'phase': 'previous'})
    old_files = dict(plan=old_plan, admission=ledger(old_plan),
        result=seal(dict(plan_sha256=old_plan['artifact_sha256'], d2_gate_open=False)),
        supervisor=dict(controller_pid=11), supervisor_result=dict(controller_exit_code=0))
    references = {}
    for name, value in old_files.items():
        path = previous / (name + '.json')
        write_json(path, value)
        references[name] = dict(path='/logical/previous/' + path.name, sha256=file_digest(path))
    current = seal({**plan, 'predecessor': references,
                    'bound_files': {r['path']: r['sha256'] for r in references.values()}})
    phase = remote / 'current'
    write_json(phase / 'execution_plan.json', current)
    write_json(phase / 'execution/admission.json', ledger(current))
    result = seal(dict(plan_sha256=current['artifact_sha256'], d2_gate_open=False,
        simulated_offline=False, process_finished=True, planned_branches=2282,
        aggregate_requests=0, aggregate_tokens=0, aggregate_cost_usd=0, aggregate_http_send_count=0,
        completed_branches=1, invalid_or_incomplete_branches=0, pending_branches=2281))
    write_json(phase / 'execution/result.json', result)
    row = receipt(current)
    write_json(phase / 'execution/receipts' / (row['branch_id'] + '.json'), row)
    write_json(phase / 'integrity_preflight.json', seal(dict(
        schema='opencode-d1-integrity-preflight-v1', plan_sha256=current['artifact_sha256'], planned_jobs=2282)))
    write_json(phase / 'supervisor_started.json', dict(controller_pid=12, started_at_epoch=1, source_commit='abc'))
    ended = dict(controller_pid=12, controller_exit_code=0, finished_at_epoch=2)
    write_json(phase / 'supervisor_result.json', ended)
    report_dir = tmp_path / 'report'
    manifest = cli.finish(phase, report_dir, [('/logical', str(remote))])
    verify_seal(manifest)
    summary = read_json(report_dir / 'summary.json')
    verify_seal(summary)
    assert len(summary['usage']['phases']) == 2
    assert summary['source_commit'] == 'abc'
    for ref in manifest['input_snapshots']:
        assert file_digest(report_dir / ref['snapshot_path']) == ref['sha256']
    for name, sha in manifest['generated_files'].items():
        assert file_digest(report_dir / name) == sha
    # Original receipt remains byte-for-byte unchanged.
    assert read_json(phase / 'execution/receipts' / (row['branch_id'] + '.json')) == row
    with pytest.raises(FileExistsError):
        cli.finish(phase, report_dir, [('/logical', str(remote))])
    write_json(phase / 'execution/result.json', seal({**result, 'aggregate_tokens': 1}))
    with pytest.raises(ValueError, match='aggregate usage'):
        cli.finish(phase, tmp_path / 'wrong-usage', [('/logical', str(remote))])
    write_json(phase / 'execution/result.json', result)
    write_json(phase / 'supervisor_result.json', {**ended, 'controller_exit_code': None})
    with pytest.raises(ValueError, match='terminal supervisor'):
        cli.finish(phase, tmp_path / 'running-report', [('/logical', str(remote))])
