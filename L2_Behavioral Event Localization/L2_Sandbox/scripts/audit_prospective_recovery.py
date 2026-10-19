#!/usr/bin/env python3
"""CPU-only full DEV45 restore/request census and preregistered tool replay plan.

No HTTP, model, Docker or recorded-code execution. The tool replay is a separate
bounded operation; writing its plan never certifies tool equivalence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile

L2 = Path(__file__).resolve().parents[1]
ROOT = L2.parents[1]
L1 = ROOT / 'L1_sandbox/C3_Experimental_Sandbox'
sys.path[:0] = [str(Path(__file__).parent), str(L1 / 'src'), str(L2 / 'src')]
from c3_sandbox.l1.storage import canonical, confined, digest, file_digest, read_json, utc_now
from c3_sandbox.l1.evidence.live import validate_snapshot
from l2_sandbox.v3_execution import verify_seal
from audit_g1_mechanical import DEV, bound_checkpoints, replay_workspace
from check_recovery_mechanism import events, referenced, reconstruct, verified, native_identity

ORIGINAL_ROOT = Path('/Users/jiashen/Desktop/问题诊断')
SLOTS = tuple(f'{task}.r{rep:02d}' for task in DEV for rep in range(5))
REPLAY_SLOTS = tuple(f'{task}.r{rep:02d}' for task in DEV for rep in range(2)) + (
    'officeval_060.r02', 'officeval_090.r02')
LIMITS = {'tool_calls': 6300, 'replays_per_batch': 2, 'workers': 1,
          'wall_seconds': 21600, 'actor_http_requests': 0}


def relocated(path):
    """Resolve preserved absolute collection paths in an explicit repository mirror."""
    path = Path(path)
    return ROOT / path.relative_to(ORIGINAL_ROOT) if path.is_relative_to(ORIGINAL_ROOT) else path


def checked_ref(ref):
    if not isinstance(ref, dict) or not Path(ref.get('path', '')).is_absolute():
        raise ValueError('absolute external hash reference required')
    path = relocated(ref['path'])
    if path.is_symlink() or not path.is_file() or file_digest(path) != ref.get('sha256'):
        raise ValueError('missing or changed external reference: ' + str(path))
    return path


def ref(path):
    return {'path': str(path.resolve()), 'sha256': file_digest(path)}


def one(items, label):
    if len(items) != 1:
        raise ValueError('expected exactly one ' + label)
    return items[0]


def compression_bindings(es, rows):
    """Bind each actual compaction to its first subsequent CP and request, once."""
    output = []
    for event in es:
        if event['kind'] != 'session.compacted':
            continue
        subsequent = [row for row in rows if row['checkpoint_event']['seq'] > event['seq']]
        if not subsequent:
            raise ValueError('session.compacted lacks subsequent actual checkpoint')
        row = min(subsequent, key=lambda r: r['checkpoint_event']['seq'])
        cp = row['checkpoint']
        if event['data'] not in cp['conversation'].get('_pi_compaction', []):
            raise ValueError('first post-compaction checkpoint omits compressed session state')
        request = row['next_request']
        output.append({'event_id': event['event_id'], 'event_sha256': event['sha256'],
                       'checkpoint_id': cp['checkpoint_id'],
                       'checkpoint_ref': row['checkpoint_event']['data']['checkpoint'],
                       'request_id': request['data']['request_id'] if request else None,
                       'request_status': 'applicable' if request else 'not_applicable_terminal'})
    return output


def complete_batch(root, es, row):
    """Return all original calls including native validation failures, in order."""
    request = row['next_request']
    if request is None:
        return None, 'no_following_model_request'
    rid = request['data']['request_id']
    calls = [e for e in es if e['kind'] == 'tool.call' and e['data']['request_id'] == rid]
    if not calls:
        return None, 'following_response_has_no_tool_calls'
    snapshots = {}
    after_by_result = {}
    for e in es:
        if e['kind'] == 'state.snapshot':
            snapshot = read_json(referenced(root, e['data']['snapshot']))
            snapshots[snapshot['snapshot_id']] = e['data']['snapshot']
            if e.get('links', {}).get('tool_result'):
                after_by_result[e['links']['tool_result']] = e['data']['snapshot']
    before = row['snapshot_ref']
    result_rows = []
    for ordinal, call in enumerate(calls):
        cid = call['data']['call_id']
        results = [e for e in es if e['kind'] == 'tool.result' and e['data']['call_id'] == cid]
        if not results:
            return None, 'incomplete_tool_batch:' + cid
        result = one(results, 'tool result')
        if result.get('links', {}).get('call') != call['event_id']:
            raise ValueError('tool result is not linked to its original call')
        body = read_json(referenced(root, result['data']['result']))
        starts = [e for e in es if e['kind'] == 'tool.started' and e['data']['call_id'] == cid]
        native_invalid = body.get('provenance', {}).get('pi_native_validation_error') is True
        if native_invalid:
            if starts:
                raise ValueError('native-invalid call unexpectedly executed')
        else:
            start = one(starts, 'tool start')
            if snapshots.get(start['data']['before_snapshot_id']) != before:
                raise ValueError('tool batch before-state differs from sequential preceding state')
        after = after_by_result.get(result['event_id'])
        if after is None:
            raise ValueError('tool result lacks linked post-state snapshot')
        before_state = read_json(referenced(root, before))
        after_state = read_json(referenced(root, after))
        for snapshot in (before_state, after_state):
            validate_snapshot(root, snapshot)
        if native_invalid and any(before_state[k] != after_state[k]
                                  for k in ('files', 'directories', 'total_bytes', 'content_sha256')):
            raise ValueError('native-invalid call changed workspace')
        for key in ('result', 'stdout', 'stderr', 'image'):
            if key in result['data']:
                referenced(root, result['data'][key])
        result_rows.append({'call': call['data'], 'call_event_sha256': call['sha256'],
                            'call_event_id': call['event_id'], 'batch_ordinal': ordinal,
                            'before_snapshot': before, 'after_snapshot': after,
                            'original_result': result['data'],
                            'original_observation': body['observation'],
                            'native_invalid_without_start': native_invalid})
        before = after
    return result_rows, None


def select_batches(root, es, rows, compressions, fixed_slot):
    candidates, reasons = [], []
    for row in rows:
        cp = row['checkpoint']
        calls, reason = complete_batch(root, es, row)
        if cp['pending_calls'] or cp['qualification'].get('eligible') is not True:
            reason = 'checkpoint_ineligible:' + ','.join(cp['qualification'].get('reason_codes', []))
        if reason:
            reasons.append({'checkpoint_id': cp['checkpoint_id'], 'reason': reason})
        else:
            candidates.append((row, calls))
    selected = {}
    def add(pair, role):
        row, calls = pair
        cid = row['checkpoint']['checkpoint_id']
        selected.setdefault(cid, {'row': row, 'calls': calls, 'roles': []})['roles'].append(role)
    if fixed_slot:
        cp0 = [(r, calls) for r, calls in candidates if r['checkpoint']['checkpoint_id'] == 'cp000000']
        if cp0:
            add(cp0[0], 'cp0')
        else:
            reasons.append({'role': 'cp0', 'reason': 'cp0_has_no_eligible_complete_tool_batch'})
        if candidates:
            add(candidates[len(candidates) // 2], 'middle_complete_tool_batch')
            add(candidates[-1], 'last_complete_tool_batch')
        else:
            reasons.append({'role': 'middle_and_last', 'reason': 'no_eligible_complete_tool_batch'})
    by_id = {r['checkpoint']['checkpoint_id']: (r, calls) for r, calls in candidates}
    for comp in compressions:
        if comp['checkpoint_id'] in by_id:
            add(by_id[comp['checkpoint_id']], 'first_after_compaction:' + comp['event_id'])
        else:
            reasons.append({'role': 'first_after_compaction', 'event_id': comp['event_id'],
                            'checkpoint_id': comp['checkpoint_id'],
                            'reason': 'first_post_compaction_has_no_eligible_complete_tool_batch'})
    return list(selected.values()), reasons


def verify_controller_archive(freeze):
    archives = [relocated(name) for name in freeze['bound_files']
                if Path(name).name == 'controller_source.tar']
    archive = one(archives, 'collection controller archive')
    with tarfile.open(archive, 'r:*') as stream:
        members = stream.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError('duplicate controller archive entries')
        for name, expected in freeze['source_files'].items():
            member = stream.getmember(name)
            if not member.isfile() or hashlib.sha256(stream.extractfile(member).read()).hexdigest() != expected:
                raise ValueError('collection controller archive differs: ' + name)
            if name.startswith('L1_sandbox/C3_Experimental_Sandbox/src/') and file_digest(ROOT / name) != expected:
                raise ValueError('actual imported L1 mechanics differ from collection controller: ' + name)


def load_collection(runs, cohort_path, batch_plan_path):
    cohort, plan = read_json(cohort_path), read_json(batch_plan_path)
    verify_seal(plan, 'plan_sha256')
    verify_seal(cohort)
    if cohort.get('plan_sha256') != plan.get('plan_sha256'):
        raise ValueError('cohort/batch plan hash binding differs')
    if set(cohort.get('slots', {})) != set(SLOTS) or set(plan.get('slots', [])) != set(SLOTS):
        raise ValueError('exact DEV9 x 5 external cohort required')
    if len(plan['slots']) != 45 or plan.get('parent') is not None or plan.get('attempt_limit_per_slot') != 1:
        raise ValueError('fresh one-attempt root cohort contract differs')
    freeze_path = checked_ref({'path': plan['prospective_freeze_path'], 'sha256': plan['prospective_freeze_sha256']})
    freeze = read_json(freeze_path)
    if (freeze.get('schema') != 'prospective-live-freeze-v1' or freeze.get('phase_authorized') is not True
            or set(freeze.get('task_allowed', [])) != set(DEV)):
        raise ValueError('prospective DEV9 collection authorization is absent')
    # Collection identity belongs to the original archive, while analysis helpers
    # have their own identity. Adding/changing L2 analysis cannot rewrite history.
    for name, expected in freeze['bound_files'].items():
        checked_ref({'path': name, 'sha256': expected})
    verify_controller_archive(freeze)
    if plan.get('expected_service_identity') != freeze.get('expected_service_identity'):
        raise ValueError('service runtime identity differs from collection freeze')
    review = read_json(checked_ref(freeze['mini_review']))
    auth = read_json(checked_ref(review['authorization']))
    if (review.get('approved') is not True or review.get('policy') != 'bounded-mechanism-contract-v1'
            or auth.get('policy') != review['policy'] or review.get('unexplained_differences_allowed') is not False):
        raise ValueError('limited mechanism contract is not authorized')
    return cohort, plan, freeze, freeze_path


def audit_and_plan(runs, cohort_path, batch_plan_path, *, temp_parent=None, replay_identity=None):
    runs = runs.resolve()
    cohort, collection, freeze, freeze_path = load_collection(runs, cohort_path, batch_plan_path)
    report = {'schema': 'prospective_recovery_census_v1', 'created_at': utc_now(),
              'collection_plan': ref(batch_plan_path), 'cohort': ref(cohort_path), 'freeze': ref(freeze_path),
              'analysis_identity': analysis_identity(), 'actor_requests': 0, 'docker_calls': 0,
              'official_scorer_calls': 0, 'tool_replay_status': 'NOT_RUN', 'd0_frozen': False,
              'scope': 'saved workspace byte/directory/mode and next-request raw-byte recovery; excludes unsaved metadata and provider state',
              'slots': []}
    batches, indices = [], {}
    for slot in SLOTS:
        external = cohort['slots'][slot]
        root = runs / slot
        if relocated(external['path']).resolve() != root:
            raise ValueError('cohort slot path differs: ' + slot)
        if external.get('study_valid') is not True or external.get('status') != 'completed':
            raise ValueError('cohort slot is not valid and complete: ' + slot)
        sha = external['index_file_sha256']
        verified(root, sha)
        if file_digest(root / 'manifest.json') != external['manifest_sha256']:
            raise ValueError('external manifest binding differs: ' + slot)
        es, config, manifest = events(root), read_json(root / 'raw/resolved_config.json'), read_json(root / 'manifest.json')
        task = slot.split('.')[0]
        if (digest(config) != freeze['task_config_sha256'][task] or digest(config) != external['config_sha256']
                or config != collection['configs'][task] or manifest.get('lineage')):
            raise ValueError('root/config identity differs: ' + slot)
        anchor_event = one([e for e in es if e['kind'] == 'provenance.bound'], 'prospective anchor')
        anchor = read_json(referenced(root, anchor_event['data']['anchor']))
        if (anchor.get('freeze', {}).get('sha256') != file_digest(freeze_path)
                or anchor.get('episode_id') != external['episode_id'] or anchor.get('config_sha256') != digest(config)):
            raise ValueError('episode collection provenance differs')
        binding = read_json(checked_ref(anchor['external_receipt']))
        if binding.get('episode_id') != external['episode_id'] or binding.get('freeze') != anchor.get('freeze'):
            raise ValueError('external episode provenance receipt differs')
        rows = bound_checkpoints(root, es)
        created = [e for e in es if e['kind'] == 'checkpoint.created']
        if len(rows) != len(created) or len(rows) != len(list((root / 'checkpoints').glob('cp*.json'))):
            raise ValueError('actual checkpoint/index/event census has uncovered entries')
        requests = [e for e in es if e['kind'] == 'model.request']
        if len(rows) > 36 or len(requests) > 35:
            raise ValueError('collection per-slot checkpoint/request cap exceeded')
        checks, covered = [], []
        for row in rows:
            cp = row['checkpoint']
            if cp['config_sha256'] != digest(config) or cp['pending_calls']:
                raise ValueError('checkpoint configuration or complete-boundary mismatch')
            replay_workspace(root, row['snapshot'], config, temp_parent)
            request = row['next_request']
            if request:
                prior = [r for r in rows if r['checkpoint_event']['seq'] < request['seq']]
                if max(prior, key=lambda r: r['checkpoint_event']['seq']) is not row:
                    raise ValueError('request maps to multiple checkpoints instead of immediate boundary')
                request_check = reconstruct(root, cp, request, config)
                covered.append(request['event_id'])
            else:
                request_check = {'status': 'not_applicable_terminal'}
            checks.append({'checkpoint_id': cp['checkpoint_id'],
                           'checkpoint_ref': row['checkpoint_event']['data']['checkpoint'],
                           'snapshot_ref': row['snapshot_ref'], 'restore_exact': True,
                           'request': request_check, 'eligible': cp['qualification'].get('eligible'),
                           'ineligible_reasons': cp['qualification'].get('reason_codes', [])})
        if sorted(covered) != sorted(e['event_id'] for e in requests):
            raise ValueError('actual next-request census has uncovered or duplicate requests')
        compressions = compression_bindings(es, rows)
        selections, exclusions = select_batches(root, es, rows, compressions, slot in REPLAY_SLOTS)
        for selection in selections:
            row = selection['row']; request = row['next_request']; cp = row['checkpoint']
            batches.append({'batch_id': slot + ':' + cp['checkpoint_id'], 'slot': slot,
                            'package': external['path'], 'checkpoint_id': cp['checkpoint_id'],
                            'checkpoint_ref': row['checkpoint_event']['data']['checkpoint'],
                            'snapshot_ref': row['snapshot_ref'], 'request_id': request['data']['request_id'],
                            'request_ref': request['data']['body'],
                            'request_exact': reconstruct(root, cp, request, config),
                            'roles': selection['roles'], 'calls': selection['calls']})
        verified(root, sha)
        indices[external['path']] = sha
        report['slots'].append({'slot': slot, 'package_index_sha256': sha, 'config_sha256': digest(config),
                                'episode_id': external['episode_id'], 'checks': checks,
                                'checkpoint_count': len(rows), 'request_count': len(requests),
                                'compactions': compressions, 'selection_exclusions': exclusions})
    report['checkpoint_count'] = sum(s['checkpoint_count'] for s in report['slots'])
    report['request_count'] = sum(s['request_count'] for s in report['slots'])
    report['compaction_count'] = sum(len(s['compactions']) for s in report['slots'])
    report['compression_coverage'] = 'observed' if report['compaction_count'] else 'not_observed'
    report['restore_exact'] = report['request_exact'] = True
    report['mechanical_census_complete'] = len(report['slots']) == 45
    report['live_gate_promoted'] = False
    calls = sum(len(b['calls']) for b in batches)
    if calls * LIMITS['replays_per_batch'] > LIMITS['tool_calls']:
        raise ValueError('preregistered tool plan worst-case replay budget exceeds cap')
    plan = {'schema': 'prospective_tool_replay_v1', 'created_at': utc_now(), 'limits': LIMITS,
            'collection_plan': ref(batch_plan_path), 'cohort': ref(cohort_path), 'freeze': ref(freeze_path),
            'selection_policy': 'DEV9 r00/r01 plus 060.r02 and 090.r02: cp0, middle and last eligible complete tool batch; all first checkpoints after every actual compaction; availability only, no score selection',
            'fixed_slots': list(REPLAY_SLOTS), 'package_indices': indices, 'batches': batches,
            'recorded_call_count': calls, 'maximum_replay_calls': calls * LIMITS['replays_per_batch'],
            'excluded_positions': {s['slot']: s['selection_exclusions'] for s in report['slots']},
            'compression_coverage': report['compression_coverage'], 'status': 'NOT_RUN',
            'analysis_identity': analysis_identity()}
    if replay_identity is not None:
        plan.update(replay_identity)
    return report, plan


def analysis_identity():
    paths = [Path(__file__).resolve(), Path(__file__).with_name('audit_g1_mechanical.py'),
             Path(__file__).with_name('check_recovery_mechanism.py')]
    return {str(path.relative_to(ROOT)): file_digest(path) for path in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--collection-plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New directory')
    parser.add_argument('--temp-parent', type=Path)
    parser.add_argument('--replay-executor', type=Path, help='Bind separately reviewed replay executor source identity')
    parser.add_argument('--replay-identity', type=Path, help='Pinned intended execution-host native identity receipt')
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.runs.resolve()):
        parser.error('output must be a new directory outside immutable runs')
    identity = None
    if args.replay_executor:
        import importlib.util
        spec = importlib.util.spec_from_file_location('prospective_replay_executor', args.replay_executor)
        executor = importlib.util.module_from_spec(spec); spec.loader.exec_module(executor)
        identity = {'source_identity': executor.source_identity(), 'native_identity': None if args.replay_identity else native_identity()}
    if args.replay_identity:
        if identity is None:
            parser.error('--replay-identity requires --replay-executor')
        intended = read_json(args.replay_identity)
        if not {'source_identity', 'native_identity'} <= set(intended) or intended['source_identity'] != identity['source_identity']:
            raise ValueError('intended execution identity/source differs from reviewed executor')
        identity = {**intended, 'execution_identity_receipt': ref(args.replay_identity)}
    args.output.mkdir(parents=True)
    try:
        report, plan = audit_and_plan(args.runs, args.cohort, args.collection_plan,
                                      temp_parent=args.temp_parent, replay_identity=identity)
        for name, value in [('g1_census.json', report), ('g2_replay_plan.json', plan)]:
            (args.output / name).write_bytes(canonical(value) + b'\n')
        print(json.dumps({k: report[k] for k in ('checkpoint_count', 'request_count', 'compaction_count',
                                               'compression_coverage', 'mechanical_census_complete')}))
    except Exception as exc:
        (args.output / 'failure.json').write_bytes(canonical({'status': 'failed', 'error': str(exc),
                                                              'error_type': type(exc).__name__}) + b'\n')
        raise


if __name__ == '__main__':
    main()
