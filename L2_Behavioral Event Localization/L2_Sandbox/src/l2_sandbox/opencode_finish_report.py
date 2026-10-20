"""Read-only terminal census for cross-actor D1; no actor, scorer, or D2 calls.

The portable summary verifies receipt seals and scientific bindings. It does not
claim to reverify packages which were left on the execution host. The caller
supplies each phase's OWN admission ledger, never aggregate result counters.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path

from .opencode_d1_runtime import _scheduled_jobs
from .v3_execution import verify_branch, verify_seal


def _number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('finite nonnegative ' + label + ' required')
    return value


def _mean(values):
    return sum(values) / len(values) if values else None


def summarize_usage(ledgers):
    """Count own send rows once; unknown reservations remain charged."""
    seen, phases = set(), []
    totals = Counter()
    statuses, unknown_statuses, send_states = Counter(), Counter(), Counter()
    for ledger in ledgers:
        verify_seal(ledger)
        key = ledger['plan_sha256']
        if key in seen:
            raise ValueError('duplicate phase admission ledger')
        seen.add(key)
        if not ledger.get('stop_reason'):
            raise ValueError('closed phase admission required')
        rows = ledger['sends']
        if ledger['requests'] != len(rows):
            raise ValueError('own ledger request census differs')
        own = Counter(admitted_requests=len(rows))
        for row in rows:
            status = row['status']
            if status not in ('settled', 'unknown_usage'):
                raise ValueError('closed ledger contains unsettled or unknown admission send state')
            send_states[status] += 1
            http = row.get('http_status')
            if http is not None and (type(http) is not int or not 100 <= http <= 599):
                raise ValueError('invalid HTTP status')
            http_key = str(http) if http is not None else 'no_http_status'
            statuses[http_key] += 1
            own['transport_started_requests'] += bool(row.get('transport_started'))
            if status == 'settled':
                actual = _number(row.get('actual_tokens'), 'settled actual tokens')
                usage = row.get('usage')
                if usage is not None and usage.get('total_tokens') != actual:
                    raise ValueError('reported usage and settled actual tokens differ')
                own['reported_settled_actual_tokens'] += actual
                own['reported_settled_requests'] += 1
                own['settled_usage_equivalent_usd'] += _number(
                    row.get('actual_cost_usd'), 'settled usage equivalent cost')
            else:
                if row.get('actual_tokens') is not None:
                    raise ValueError('unsettled send cannot report settled actual tokens')
                own['unknown_reserved_tokens'] += _number(row.get('tokens_reserved'), 'reserved tokens')
                own['unknown_usage_requests'] += 1
                own['unknown_reserved_usage_equivalent_usd'] += _number(
                    row.get('cost_reserved'), 'reserved usage equivalent cost')
                unknown_statuses[http_key] += 1
        own['accounted_tokens_including_unknown_reservations'] = (
            own['reported_settled_actual_tokens'] + own['unknown_reserved_tokens'])
        own['accounted_usage_equivalent_usd'] = (
            own['settled_usage_equivalent_usd'] + own['unknown_reserved_usage_equivalent_usd'])
        if (ledger['tokens'] != own['accounted_tokens_including_unknown_reservations']
                or not math.isclose(ledger['cost_usd'], own['accounted_usage_equivalent_usd'],
                                    rel_tol=1e-9, abs_tol=1e-8)):
            raise ValueError('own ledger accounting differs from send rows')
        phases.append(dict(plan_sha256=key, admission_sha256=ledger['artifact_sha256'],
                           started_at_epoch=ledger.get('started_at'),
                           stop_reason=ledger['stop_reason'], **own))
        totals.update(own)
    # Keep explicit zero counters even when a whole category is absent.
    for key in ('admitted_requests', 'transport_started_requests', 'reported_settled_requests',
                'reported_settled_actual_tokens', 'unknown_usage_requests', 'unknown_reserved_tokens',
                'settled_usage_equivalent_usd', 'unknown_reserved_usage_equivalent_usd',
                'accounted_tokens_including_unknown_reservations', 'accounted_usage_equivalent_usd'):
        totals.setdefault(key, 0)
    return dict(**totals, phases=phases, http_status_counts=dict(sorted(statuses.items())),
                unknown_usage_http_status_counts=dict(sorted(unknown_statuses.items())),
                send_state_counts=dict(sorted(send_states.items())),
                accounting_mode='usage_equivalent_peak_rates_not_billed_cash',
                billed_cash_usd=None, unknown_usage_refunded=False)


def summarize(plan, receipts, ledgers):
    """Summarize sealed artifacts in memory; perform no package/model/scorer I/O.

    The CLI verifies completeness of the predecessor ledger chain before calling
    this function. Missing receipts are unstarted, never zero-score observations.
    """
    verify_seal(plan)
    verify_seal(plan['scientific_plan'], 'plan_sha256')
    if plan.get('schema') != 'cross-actor-d1-execution-v1' or plan.get('automatic_d2') is not False:
        raise ValueError('cross-actor D1 plan with automatic D2 disabled required')
    jobs = _scheduled_jobs(plan, Path('/portable-report-logical-output'))
    expected = {j['binding']['branch_id']: j for j in jobs}
    received = {}
    for receipt in receipts:
        verify_seal(receipt)
        key = receipt.get('branch_id')
        if key not in expected or key in received:
            raise ValueError('unknown or duplicate branch receipt')
        binding = expected[key]['binding']
        if any(receipt.get(k) != v for k, v in binding.items()):
            raise ValueError('receipt scientific binding differs')
        if receipt.get('simulated_offline') is not False:
            raise ValueError('native receipt required')
        if receipt.get('status') == 'completed':
            verify_branch(receipt, expected=binding, verify_package_binding=False)
            score = _number(receipt.get('outcome_score'), 'endpoint score')
            if score > 1:
                raise ValueError('endpoint score exceeds one')
        elif (receipt.get('status') not in ('branch_invalid', 'infrastructure_invalid', 'incomplete')
              or receipt.get('valid_attempt') is not False or 'outcome_score' in receipt):
            raise ValueError('invalid receipt status or outcome')
        received[key] = receipt
    grouped = defaultdict(list)
    for job in jobs:
        b = job['binding']
        grouped[(b['unit_id'], b['arm'])].append(job)
    cells, units = [], {}
    census = Counter(planned=len(jobs), valid=0, invalid_or_incomplete=0, unstarted=0)
    status_counts = Counter()
    for (unit_id, arm), group in sorted(grouped.items()):
        scores, counts = [], Counter(valid=0, invalid_or_incomplete=0, unstarted=0)
        statuses = Counter()
        for job in group:
            receipt = received.get(job['binding']['branch_id'])
            status = receipt['status'] if receipt else 'unstarted'
            statuses[status] += 1
            status_counts[status] += 1
            category = 'unstarted' if receipt is None else (
                'valid' if status == 'completed' else 'invalid_or_incomplete')
            counts[category] += 1
            census[category] += 1
            if category == 'valid':
                scores.append(receipt['outcome_score'])
        unit = group[0]['unit']
        binding = group[0]['binding']
        cell = dict(unit_id=unit_id, arm=arm, passing_control=binding['passing_control'],
                    planned=len(group), **counts, status_counts=dict(statuses),
                    mean_target_endpoint_score=_mean(scores), endpoint=binding['endpoint'])
        cells.append(cell)
        units.setdefault(unit_id, dict(unit_id=unit_id, task_id=unit['task_id'],
            passing_control=binding['passing_control'], parent_identity=binding['parent_identity'],
            cells=[]))['cells'].append(cell)
    for unit in units.values():
        unit['census'] = {key: sum(c[key] for c in unit['cells'])
                          for key in ('planned', 'valid', 'invalid_or_incomplete', 'unstarted')}
    arms = []
    for passing in (False, True):
        subset = [u for u in units.values() if u['passing_control'] is passing]
        names = sorted({c['arm'] for u in subset for c in u['cells']})
        for arm in names:
            eligible = [u for u in subset if any(c['arm'] == arm for c in u['cells'])]
            matched = []
            for unit in eligible:
                by_arm = {c['arm']: c for c in unit['cells']}
                target, baseline = by_arm[arm], by_arm.get('P')
                if (target['valid'] and baseline and baseline['valid']):
                    matched.append(dict(unit_id=unit['unit_id'],
                        arm_mean=target['mean_target_endpoint_score'],
                        baseline_mean=baseline['mean_target_endpoint_score']))
            available = [c for u in eligible for c in u['cells'] if c['arm'] == arm and c['valid']]
            arm_cells = [c for u in eligible for c in u['cells'] if c['arm'] == arm]
            arm_mean = _mean([m['arm_mean'] for m in matched])
            baseline_mean = _mean([m['baseline_mean'] for m in matched])
            arms.append(dict(arm=arm, passing_control=passing, eligible_units=len(eligible),
                units_with_valid_scores=len(available), matched_units=len(matched),
                available_unit_mean=_mean([c['mean_target_endpoint_score'] for c in available]),
                matched_unit_arm_mean=arm_mean, matched_unit_baseline_mean=baseline_mean,
                matched_unit_difference=(arm_mean-baseline_mean if matched else None),
                census={key: sum(c[key] for c in arm_cells)
                        for key in ('planned', 'valid', 'invalid_or_incomplete', 'unstarted')},
                matched_unit_details=matched))
    usage = summarize_usage(ledgers)
    if plan['artifact_sha256'] not in {p['plan_sha256'] for p in usage['phases']}:
        raise ValueError('current phase own admission ledger missing')
    return dict(schema='opencode-finish-summary-v1', plan_sha256=plan['artifact_sha256'],
        scientific_plan_sha256=plan['scientific_plan']['plan_sha256'],
        scientific_scope='cross_actor_exploratory', automatic_d2=False, d2_gate_open=False,
        formal_pvalue_gate_passed=False, verification_scope='receipt_seals_and_plan_bindings',
        package_reverification_performed=False, schedule_complete=census['unstarted'] == 0,
        all_planned_branches_valid=census['valid'] == len(jobs), census=dict(census),
        receipt_status_counts=dict(status_counts), unit_count=len(units), cell_count=len(cells),
        populations={name: dict(unit_count=sum(u['passing_control'] is passing for u in units.values()),
            unique_parent_episode_count=len({u['parent_identity'].get('parent_episode_id')
                for u in units.values() if u['passing_control'] is passing}),
            census={key: sum(u['census'][key] for u in units.values() if u['passing_control'] is passing)
                    for key in ('planned', 'valid', 'invalid_or_incomplete', 'unstarted')})
            for name, passing in (('failure', False), ('passing_control', True))},
        cells=cells, units=list(units.values()), arm_summaries=arms, usage=usage,
        receipt_seals={k: v['artifact_sha256'] for k, v in sorted(received.items())},
        limitations=[
            'Scores are frozen target rubric-item or gate endpoints, not whole-benchmark scores.',
            'Planned n=14 per cell does not imply 14 valid observations; consult each cell census.',
            'Arm comparisons use the same arm-eligible units with valid arm and P baseline means.',
            'Missing and invalid observations are excluded from score means and remain separately counted.',
            'Cross-actor exploratory results do not open D2 or establish a formal p-value gate.',
            'Passing controls share parent families; pooled harm comparisons are descriptive only.',
            'Unknown usage reservations remain charged in accounting; usage equivalents are not cash invoices.',
            'Package integrity was enforced by execution; portable reporting verifies receipt seals and bindings only.',
        ])


def render_markdown(report):
    c, usage = report['census'], report['usage']
    lines = ['# OpenCode finite-batch terminal report', '',
        f"Plan: `{report['plan_sha256']}`. Cross-actor exploratory; D2 remains closed.", '',
        f"Planned {c['planned']}; valid {c['valid']}; invalid/incomplete {c['invalid_or_incomplete']}; "
        f"unstarted {c['unstarted']}. Units {report['unit_count']}; cells {report['cell_count']}.", '',
        f"Reported settled actual tokens: {usage['reported_settled_actual_tokens']}; "
        f"unknown reserved tokens: {usage['unknown_reserved_tokens']} "
        f"({usage['unknown_usage_requests']} requests).", '',
        f"HTTP status counts: `{usage['http_status_counts']}`. "
        f"Unknown-usage HTTP statuses: `{usage['unknown_usage_http_status_counts']}`.", '',
        'Costs are usage equivalents, not billed cash. Unknown reservations are retained.', '',
        '| Population | Arm | Valid | Invalid/incomplete | Unstarted | Eligible units | Matched units | Arm mean | Matched P mean | Difference |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(value):
        return 'NA' if value is None else f'{value:.6f}'
    for row in report['arm_summaries']:
        lines.append(f"| {'passing' if row['passing_control'] else 'failure'} | {row['arm']} | "
            f"{row['census']['valid']} | {row['census']['invalid_or_incomplete']} | {row['census']['unstarted']} | "
            f"{row['eligible_units']} | {row['matched_units']} | {fmt(row['matched_unit_arm_mean'])} | "
            f"{fmt(row['matched_unit_baseline_mean'])} | {fmt(row['matched_unit_difference'])} |")
    lines += ['', 'All means above weight each matched unit equally. Per-cell valid n follows.', '',
              '| Unit | Arm | Planned | Valid | Invalid/incomplete | Unstarted | Target mean |',
              '|---|---|---:|---:|---:|---:|---:|']
    for row in report['cells']:
        lines.append(f"| {row['unit_id']} | {row['arm']} | {row['planned']} | {row['valid']} | "
            f"{row['invalid_or_incomplete']} | {row['unstarted']} | {fmt(row['mean_target_endpoint_score'])} |")
    lines += ['', *['- ' + note for note in report['limitations']], '']
    return '\n'.join(lines)
