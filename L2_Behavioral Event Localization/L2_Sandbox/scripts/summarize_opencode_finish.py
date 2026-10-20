"""Seal a terminal D1 census on an AML CPU worker, without actor/scorer calls.

Example (paths can be an rsynced snapshot):
  python scripts/summarize_opencode_finish.py --phase /snapshot/continuation_05 \
    --path-map /Users/jiashen/Desktop/问题诊断=/snapshot/workspace \
    --output /snapshot/terminal_report

Copy current phase execution_plan.json, integrity_preflight.json,
supervisor_started.json, supervisor_result.json, execution/{admission,result}.json
and ALL current execution/receipts/*.json. Also copy every predecessor plan and
its five bound plan/result/admission/supervisor/supervisor_result references at
their mapped paths. Old package trees are unnecessary for the portable scope.
The new support directory snapshots every consumed input; manifest.json seals
their byte hashes plus summary.json and report.md. Existing data is never edited.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil

from l2_sandbox.opencode_finish_report import render_markdown, summarize
from l2_sandbox.v3_execution import file_digest, read_json, seal, verify_seal, write_json


def finish(phase, output, path_maps=()):
    phase, output = Path(phase).resolve(), Path(output).resolve()
    mappings = sorted(((Path(a), Path(b)) for a, b in path_maps),
                      key=lambda pair: len(pair[0].parts), reverse=True)
    for logical, physical in mappings:
        if not logical.is_absolute() or not physical.is_absolute():
            raise ValueError('absolute path-map prefixes required')

    def mapped(name):
        path = Path(name)
        if not path.is_absolute() or '..' in path.parts:
            raise ValueError('absolute non-traversing input path required')
        for logical, physical in mappings:
            try:
                return physical / path.relative_to(logical)
            except ValueError:
                pass
        return path

    consumed = {}

    def consume(path, *, expected=None):
        path = Path(path)
        sha = file_digest(path)
        if expected is not None and sha != expected:
            raise ValueError('bound input hash differs: ' + str(path))
        if path in consumed and consumed[path] != sha:
            raise ValueError('input changed during report: ' + str(path))
        consumed[path] = sha
        return read_json(path)

    def bound(plan, ref):
        if plan.get('bound_files', {}).get(ref['path']) != ref['sha256']:
            raise ValueError('predecessor reference missing from bound_files')
        return consume(mapped(ref['path']), expected=ref['sha256'])

    plan = consume(phase / 'execution_plan.json')
    result = consume(phase / 'execution/result.json')
    ledger = consume(phase / 'execution/admission.json')
    started = consume(phase / 'supervisor_started.json')
    ended = consume(phase / 'supervisor_result.json')
    preflight = consume(phase / 'integrity_preflight.json')
    for artifact in (plan, result, ledger, preflight):
        verify_seal(artifact)
    if (type(ended.get('controller_exit_code')) is not int
            or ended.get('controller_pid') != started.get('controller_pid')
            or type(ended.get('finished_at_epoch')) not in (int, float)
            or type(started.get('started_at_epoch')) not in (int, float)
            or ended['finished_at_epoch'] < started['started_at_epoch']):
        raise ValueError('terminal supervisor exit evidence required')
    if (result.get('plan_sha256') != plan['artifact_sha256']
            or result.get('d2_gate_open') is not False
            or result.get('simulated_offline') is not False
            or result.get('process_finished') is not True):
        raise ValueError('terminal native result with D2 closed required')
    if (preflight.get('plan_sha256') != plan['artifact_sha256']
            or preflight.get('schema') != 'opencode-d1-integrity-preflight-v1'
            or preflight.get('planned_jobs') != 2282):
        raise ValueError('runtime integrity preflight binding differs')
    plans, ledgers = [plan], [ledger]
    node = plan
    while node.get('predecessor') is not None:
        if len(plans) >= 9:
            raise ValueError('predecessor chain depth limit exceeded')
        prior = node['predecessor']
        old = {key: bound(node, prior[key]) for key in (
            'plan', 'result', 'admission', 'supervisor', 'supervisor_result')}
        for key in ('plan', 'result', 'admission'):
            verify_seal(old[key])
        old_plan = old['plan']
        if old_plan['artifact_sha256'] in {p['artifact_sha256'] for p in plans}:
            raise ValueError('predecessor chain cycle')
        if (old_plan['scientific_plan'] != plan['scientific_plan']
                or old_plan.get('automatic_d2') is not False
                or old['result'].get('plan_sha256') != old_plan['artifact_sha256']
                or old['result'].get('d2_gate_open') is not False
                or old['admission'].get('plan_sha256') != old_plan['artifact_sha256']
                or type(old['supervisor_result'].get('controller_exit_code')) is not int):
            raise ValueError('predecessor plan/result/admission binding differs')
        plans.append(old_plan)
        ledgers.append(old['admission'])
        node = old_plan
    if ledger.get('plan_sha256') != plan['artifact_sha256']:
        raise ValueError('current own admission binding differs')
    receipt_paths = sorted((phase / 'execution/receipts').glob('*.json'))
    receipts = [consume(path) for path in receipt_paths]
    if any(p.stem != r.get('branch_id') for p, r in zip(receipt_paths, receipts)):
        raise ValueError('receipt filename binding differs')
    report = summarize(plan, receipts, ledgers)
    c = report['census']
    if any(result.get(key) != value for key, value in (
            ('planned_branches', c['planned']), ('completed_branches', c['valid']),
            ('invalid_or_incomplete_branches', c['invalid_or_incomplete']),
            ('pending_branches', c['unstarted']))):
        raise ValueError('terminal result census differs from receipts')
    usage = report['usage']
    if (any(result.get(key) != value for key, value in (
            ('aggregate_requests', usage['admitted_requests']),
            ('aggregate_tokens', usage['accounted_tokens_including_unknown_reservations']),
            ('aggregate_http_send_count', usage['transport_started_requests'])))
            or type(result.get('aggregate_cost_usd')) not in (int, float)
            or not math.isclose(result['aggregate_cost_usd'], usage['accounted_usage_equivalent_usd'],
                                rel_tol=1e-9, abs_tol=1e-8)):
        raise ValueError('terminal aggregate usage differs from own phase ledgers')
    report.update(created_at_utc=datetime.now(timezone.utc).isoformat(),
        terminal_supervisor=ended, runtime_integrity_preflight=preflight,
        source_commit=started.get('source_commit'),
        source_inventories=[dict(plan_sha256=p['artifact_sha256'],
            source_files=p.get('source_files', {}), historical_source_commit=p.get('historical_source_commit'))
            for p in plans],
        source_inventory_verification='expected frozen hashes retained; executable sources not reverified by portable reporter')
    report = seal(report)
    # Snapshot reporter sources too, so its own implementation is reviewable.
    import l2_sandbox.opencode_finish_report as reporting
    module_root = Path(reporting.__file__).resolve().parent
    for path in (Path(__file__).resolve(), Path(reporting.__file__).resolve(),
                 *(module_root / name for name in ('opencode_d1_runtime.py', 'v3_execution.py', 'v05.py'))):
        consumed[path] = file_digest(path)
    output.mkdir(parents=True, exist_ok=False)
    support = output / 'support'
    support.mkdir()
    refs = []
    for index, (path, sha) in enumerate(sorted(consumed.items(), key=lambda item: str(item[0]))):
        target = support / f'{index:05d}_{path.name}'
        shutil.copyfile(path, target)
        if file_digest(target) != sha or file_digest(path) != sha:
            raise ValueError('input changed while snapshotting: ' + str(path))
        refs.append(dict(source_path=str(path), snapshot_path=str(target.relative_to(output)), sha256=sha))
    if set(receipt_paths) != set((phase / 'execution/receipts').glob('*.json')):
        raise ValueError('receipt inventory changed while snapshotting')
    write_json(output / 'summary.json', report)
    (output / 'report.md').write_text(render_markdown(report))
    manifest = seal(dict(schema='opencode-terminal-report-manifest-v1',
        created_at_utc=report['created_at_utc'], plan_sha256=plan['artifact_sha256'],
        summary_sha256=report['artifact_sha256'], source_commit=started.get('source_commit'),
        verification_scope=report['verification_scope'], automatic_d2=False, d2_gate_open=False,
        input_snapshots=refs, expected_receipt_seals=report['receipt_seals'],
        generated_files={name: file_digest(output / name) for name in ('summary.json', 'report.md')}))
    write_json(output / 'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='new directory; existing paths are refused')
    parser.add_argument('--path-map', action='append', default=[], metavar='LOGICAL=PHYSICAL')
    args = parser.parse_args()
    mappings = []
    for value in args.path_map:
        if '=' not in value:
            parser.error('--path-map requires LOGICAL=PHYSICAL')
        mappings.append(value.split('=', 1))
    manifest = finish(args.phase, args.output, mappings)
    print(json.dumps(dict(manifest_sha256=manifest['artifact_sha256'], output=str(args.output))))


if __name__ == '__main__':
    main()
