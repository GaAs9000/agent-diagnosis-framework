#!/usr/bin/env python3
"""Replay frozen complete tool batches in Docker, without any model requests.

Default validates a plan. --execute requires its SHA and a fresh output directory.
An optional second replay diagnoses variance; it never replaces a first mismatch.
"""
from __future__ import annotations
import argparse
import copy
from contextlib import ExitStack, nullcontext
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_recovery_mechanism as mini

LIMITS = {'tool_calls': 6300, 'replays_per_batch': 2, 'workers': 1,
          'wall_seconds': 21600, 'actor_http_requests': 0}
SCHEMA = 'prospective_tool_replay_v1'


def source_identity():
    return {**mini.source_identity(), str(Path(__file__).resolve().relative_to(mini.ROOT)):
            mini.file_digest(Path(__file__))}


def one(items, label):
    if len(items) != 1:
        raise ValueError('expected exactly one ' + label)
    return items[0]


def select_batch(batch):
    """Reconstruct a complete recorded request/tool batch, preserving call order."""
    root = Path(batch['package'])
    es = mini.events(root)
    cp = mini.read_json(mini.referenced(root, batch['checkpoint_ref']))
    ce = one([e for e in es if e['kind'] == 'checkpoint.created'
              and e['data']['checkpoint'] == batch['checkpoint_ref']], 'checkpoint')
    req = one([e for e in es if e['kind'] == 'model.request'
               and e['data']['request_id'] == batch['request_id']], 'request')
    following = [e for e in es if e['kind'] == 'model.request' and e['seq'] > ce['seq']]
    if not following or following[0] != req or cp.get('pending_calls') or not cp['qualification']['eligible']:
        raise ValueError('not a qualified next-request checkpoint')
    config = mini.read_json(root / 'raw/resolved_config.json')
    request_exact = mini.reconstruct(root, cp, req, config)
    snapshots = [e for e in es if e['kind'] == 'state.snapshot']
    calls = [e for e in es if e['kind'] == 'tool.call'
             and e['data']['request_id'] == batch['request_id']]
    if not calls:
        raise ValueError('tool batch is empty')
    snapshot_event = one([e for e in snapshots if mini.read_json(mini.referenced(root, e['data']['snapshot']))['snapshot_id'] == cp['snapshot_id']], 'checkpoint state')
    before_snapshot = snapshot_event['data']['snapshot']
    prior_snapshot_id = cp['snapshot_id']
    rows = []
    for event in calls:
        call = event['data']; cid = call['call_id']
        result = one([e for e in es if e['kind'] == 'tool.result' and e['data']['call_id'] == cid], 'result')
        after = one([e for e in snapshots if e.get('links', {}).get('tool_result') == result['event_id']], 'after state')
        starts = [e for e in es if e['kind'] == 'tool.started' and e['data']['call_id'] == cid]
        validations = [e for e in es if e['kind'] == 'tool.validation' and e['data']['call_id'] == cid]
        validation = one(validations, 'native validation')['data']
        if len(starts) != int(validation['ok']):
            raise ValueError('native invalid/start contract differs')
        if starts and starts[0]['data']['before_snapshot_id'] != prior_snapshot_id:
            raise ValueError('sibling before-state chain differs')
        if not (event['seq'] < result['seq'] < after['seq']):
            raise ValueError('call/result/state event order differs')
        prior_snapshot_id = mini.read_json(mini.referenced(root, after['data']['snapshot']))['snapshot_id']
        observed = mini.read_json(mini.referenced(root, result['data']['result']))['observation']
        key = cid.split(':', 1)[1].replace(':', '_')
        delivered_path = mini.confined(root, 'observations/' + key + '.json')
        # These files are covered by verify_package's index before execution.
        delivered = mini.read_json(delivered_path)
        mini.validate_snapshot(root, mini.read_json(mini.referenced(root, after['data']['snapshot'])))
        for field in ('stdout', 'stderr', 'image'):
            if field in result['data']:
                mini.referenced(root, result['data'][field])
        rows.append({'call': call, 'call_event_sha256': event['sha256'],
                     'original_result': result['data'], 'original_observation': observed,
                     'original_delivered': delivered, 'after_snapshot': after['data']['snapshot'],
                     'native_validation_ok': validation['ok']})
    # A call with no result above fails selection rather than silently dropping a sibling.
    return {'checkpoint': cp, 'before_snapshot': before_snapshot, 'config': config, 'request_exact': request_exact, 'calls': rows}


def validate_plan(plan, *, validator=None):
    if plan.get('schema') != SCHEMA or plan.get('limits') != LIMITS:
        raise ValueError('frozen policy/limits mismatch')
    if plan.get('source_identity') != source_identity():
        raise ValueError('execution source differs from frozen plan')
    if plan.get('native_identity') != mini.native_identity(validator):
        raise ValueError('native validator differs from frozen plan')
    for path, sha in plan['package_indices'].items():
        mini.verified(Path(path), sha)
    batches = plan.get('batches', [])
    if not batches or len({b['batch_id'] for b in batches}) != len(batches):
        raise ValueError('batches absent or duplicate')
    keys = set(); total = 0
    for batch in batches:
        if (not batch['batch_id'] or Path(batch['batch_id']).name != batch['batch_id']
                or batch['batch_id'] in ('.', '..')):
            raise ValueError('unsafe batch id')
        if batch['package'] not in plan['package_indices']:
            raise ValueError('batch package unbound')
        key = (batch['package'], batch['request_id'])
        if key in keys:
            raise ValueError('duplicate selected request')
        keys.add(key)
        selected = select_batch(batch)
        if 'request_exact' in batch and batch['request_exact'] != selected['request_exact']:
            raise ValueError('selected request changed')
        if 'calls' in batch:
            frozen = [(c['call'], c['call_event_sha256']) for c in batch['calls']]
            actual = [(c['call'], c['call_event_sha256']) for c in selected['calls']]
            if frozen != actual:
                raise ValueError('frozen complete batch changed')
        total += len(selected['calls'])
    if total > LIMITS['tool_calls']:
        raise ValueError('first replay exceeds total tool cap')
    return total


def delivered_observation(observation, limit):
    delivered = copy.deepcopy(observation); cut = []
    for field in ('stdout', 'stderr', 'output'):
        if isinstance(delivered.get(field), str) and len(delivered[field]) > limit:
            delivered[field] = delivered[field][:limit]; cut.append(field)
    if cut:
        delivered['observation_truncation'] = {'fields': cut, 'limit_chars': limit, 'raw_retained': True}
    return delivered


def compare_call(root, recorded, actual, captured_root, captured, call_dir, config, *, persist=True):
    expected = mini.read_json(mini.referenced(root, recorded['after_snapshot']))
    original = recorded['original_result']; observed = recorded['original_observation']
    left = {key: mini.referenced(root, original[key]).read_bytes() for key in ('stdout', 'stderr')}
    image = mini.referenced(root, original['image']).read_bytes() if 'image' in original else None
    obs_exact = observed == actual.observation
    delivered = delivered_observation(actual.observation, config['recording']['observation_limit_chars'])
    delivered_exact = delivered == recorded['original_delivered']
    streams = actual.stdout == left['stdout'] and actual.stderr == left['stderr']
    image_exact = image == actual.image
    state_exact = mini.snapshot_equal(expected, captured)
    exact = obs_exact and delivered_exact and streams and image_exact and state_exact
    zip_variance = mini.zip_time_only(root, expected, captured_root, captured) if not state_exact else []
    # Only the already demonstrated literal ls command is eligible for metadata diagnosis.
    ls_variance = (recorded['call']['name'] == 'bash'
                   and recorded['call']['arguments'] in ({'command': 'ls -la /workspace/submission/'},
                                                          {'command': 'ls -la /workspace/input/'})
                   and state_exact and actual.stderr == left['stderr'] and image_exact
                   and actual.observation.get('exit_code') == observed.get('exit_code') == 0
                   and mini.ls_metadata_only(left['stdout'], actual.stdout)
                   and actual.observation == mini.format_pi_bash_observation(
                       actual.stdout, actual.stderr, 0, actual.observation.get('stop_reason'),
                       actual.observation.get('output_complete'), actual.observation.get('oom_killed'))
                   and {k: v for k, v in actual.observation.items() if k not in ('stdout', 'output')}
                       == {k: v for k, v in observed.items() if k not in ('stdout', 'output')}
                   and delivered_observation(observed, config['recording']['observation_limit_chars'])
                       == recorded['original_delivered'])
    declared = bool((zip_variance and obs_exact and delivered_exact and streams and image_exact) or ls_variance)
    if persist:
        mini.write_json(call_dir, 'delivered_observation.json', delivered)
    return {'call_id': recorded['call']['call_id'], 'exact': exact,
            'observation_exact': obs_exact, 'delivered_observation_exact': delivered_exact,
            'raw_streams_exact': streams, 'image_exact': image_exact, 'after_state_exact': state_exact,
            'classification': 'exact' if exact else 'declared_runtime_variance' if declared else 'unexplained',
            'zip_timestamp_variance': zip_variance, 'ls_metadata_variance': bool(ls_variance),
            'unexplained_state_difference': not state_exact and not zip_variance,
            'unexplained_tool_difference': not exact and not declared,
            'state_diff': mini.StateStore.diff(expected, captured),
            'expected_after_sha256': expected['content_sha256'], 'actual_after_sha256': captured['content_sha256']}


def cleanup_receipt(runtime_events):
    created = [e['data']['container_id'] for e in runtime_events if e['kind'] == 'runtime.container.created']
    removed = [e['data']['container_id'] for e in runtime_events
               if e['kind'] == 'runtime.container.removed' and e['data'].get('removed') is True]
    return {'created': created, 'removed': removed,
            'confirmed': len(created) == len(set(created)) and sorted(created) == sorted(removed)}


def run_batch(batch, output, deadline, *, expected_native_identity, admit,
              runtime_factory=mini.DockerRuntime, validator_factory=mini.NativePiValidator, shared_validator=None):
    selected = select_batch(batch); root = Path(batch['package']); config = selected['config']
    output.mkdir(); workspace = output / 'workspace'
    cp = selected['checkpoint']
    # Checkpoint snapshot is an indexed file reference, not a tool-intermediate reset.
    before = mini.read_json(mini.referenced(root, selected['before_snapshot']))
    mini.restore(root, before, workspace)
    restored = mini.capture(output / 'restore_capture', workspace, config['runtime'], 'restored')
    if not mini.snapshot_equal(before, restored):
        raise ValueError('restored batch state differs')
    runtime = runtime_factory(copy.deepcopy(config['runtime'])); runtime_events = []
    def event(kind, data):
        row = {'kind': kind, 'data': data}
        mini.write_json(output, f'runtime_event_{len(runtime_events):06d}.json', row)
        runtime_events.append(row)
    runtime.set_event_sink(event)
    rows = []
    report = {'batch_id': batch['batch_id'], 'restore_exact': True,
              'request_exact': selected['request_exact'], 'calls': rows, 'status': 'incomplete'}
    try:
        with (nullcontext(shared_validator) if shared_validator is not None else validator_factory()) as validator:
            identity = mini.native_identity(validator)
            if identity != expected_native_identity:
                raise ValueError('active bridge/native installation differs before batch')
            mini.write_json(output, 'native_identity.json', identity)
            provenance = runtime.preflight()
            if provenance['image_id'] != cp['qualification']['image_id']:
                raise ValueError('actual runtime image differs from checkpoint')
            mini.write_json(output, 'runtime_provenance.json', provenance)
            for ordinal, recorded in enumerate(selected['calls']):
                if time.monotonic() >= deadline:
                    raise TimeoutError('batch wall limit')
                call = recorded['call']; directory = output / f'call_{ordinal:04d}'; directory.mkdir()
                validation = validator.validate(call['name'], call['arguments'], call['native_id'] or call['call_id'],
                                                deadline_monotonic=deadline)
                mini.write_json(directory, 'native_validation.json', validation)
                if validation['ok'] != recorded['native_validation_ok']:
                    raise ValueError('native validation disagrees with recorded start/feedback')
                admit(batch['batch_id'], ordinal, bool(validation['ok']))
                if validation['ok']:
                    event('tool.started', {'call_id': call['call_id']})
                    execution = mini.NativeValidatedCall(call['call_id'], call['name'],
                                                         validation['validated_arguments'], call['native_id'])
                    actual = runtime.execute(execution, workspace,
                        min(config['runtime']['tool_timeout_seconds'], deadline - time.monotonic()))
                else:
                    actual = mini.Result(copy.deepcopy(validation['native_result']),
                                         provenance={'pi_native_validation_error': True})
                mini.write_json(directory, 'result.json', {'observation': actual.observation,
                                                           'provenance': actual.provenance})
                for key in ('stdout', 'stderr'):
                    mini.write_bytes(directory, key + '.bin', getattr(actual, key))
                if actual.image is not None:
                    mini.write_bytes(directory, 'image.bin', actual.image)
                captured_root = directory / 'capture'
                captured = mini.capture(captured_root, workspace, config['runtime'], 'after_replay')
                row = compare_call(root, recorded, actual, captured_root, captured, directory, config)
                rows.append(row); mini.write_json(directory, 'comparison.json', row)
                # Never reset before a sibling, including after a native-invalid call.
            report['status'] = 'complete'
    finally:
        report['cleanup'] = cleanup_receipt(runtime_events)
        mini.write_json(output, 'batch_result.json', report)
    if not report['cleanup']['confirmed']:
        raise ValueError('container cleanup unconfirmed')
    return report



def _ref(path):
    return {'path': str(Path(path).resolve()), 'sha256': mini.file_digest(Path(path))}


def _bound(ref):
    path = Path(ref['path'])
    if mini.file_digest(path) != ref['sha256']:
        raise ValueError('continuation evidence changed: ' + str(path))
    return path


def prior_inventory(root):
    """Bind all retained raw bytes, including partial workspaces, without aliases."""
    files = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('continuation symlink forbidden')
        if path.is_file():
            files[str(path.relative_to(root))] = mini.file_digest(path)
    return files


def build_continuation_plan(prior_plan_path, prior_result_path, started_receipt_path):
    """Create a new plan; never rewrite the failed frozen plan or its evidence."""
    old = mini.read_json(prior_plan_path); started = mini.read_json(started_receipt_path)
    original_sha = (mini.read_json(_bound(old['continuation']['started_receipt']))['plan_sha256']
                    if 'continuation' in old else mini.file_digest(prior_plan_path))
    if started['plan_sha256'] != original_sha:
        raise ValueError('original wall clock is not bound to prior plan')
    new = copy.deepcopy(old)
    new['source_identity'] = source_identity()
    new['replay_operation'] = {'validator_lifetime': 'shared-execution-v1'}
    new['continuation'] = {'prior_plan': _ref(prior_plan_path), 'prior_result': _ref(prior_result_path),
                           'started_receipt': _ref(started_receipt_path),
                           'prior_files': prior_inventory(Path(prior_result_path).parent)}
    validate_continuation(new)
    return new


def saved_replay(batch, replay, directory, native_identity):
    """Recheck retained raw artifacts; do not mutate their original classification."""
    if mini.read_json(directory / 'batch_result.json') != replay:
        raise ValueError('prior batch result differs from aggregate')
    events = [mini.read_json(p) for p in sorted(directory.glob('runtime_event_*.json'))]
    cleanup = cleanup_receipt(events)
    if replay['status'] != 'complete' or cleanup != replay['cleanup'] or not cleanup['confirmed']:
        raise ValueError('prior completed prefix cleanup unconfirmed')
    if mini.read_json(directory / 'native_identity.json') != native_identity:
        raise ValueError('prior active native bridge identity differs')
    selected = select_batch(batch); package = Path(batch['package'])
    if mini.read_json(directory / 'runtime_provenance.json')['image_id'] != selected['checkpoint']['qualification']['image_id']:
        raise ValueError('prior runtime image differs')
    expected_starts = [r['call']['call_id'] for r in selected['calls'] if r['native_validation_ok']]
    if [e['data']['call_id'] for e in events if e['kind'] == 'tool.started'] != expected_starts:
        raise ValueError('prior native invalid/start contract differs')
    if replay['request_exact'] != selected['request_exact'] or not replay['restore_exact']:
        raise ValueError('prior restore/request receipt differs')
    before = mini.read_json(mini.referenced(package, selected['before_snapshot']))
    restored = mini.read_json(directory / 'restore_capture/states/snapshots/s000000.json')
    mini.validate_snapshot(directory / 'restore_capture', restored)
    if not mini.snapshot_equal(before, restored):
        raise ValueError('prior restored snapshot differs')
    if len(replay['calls']) != len(selected['calls']):
        raise ValueError('prior batch missing calls')
    admissions = []; changes = []; effective = []
    for ordinal, (recorded, comparison) in enumerate(zip(selected['calls'], replay['calls'])):
        call_dir = directory / f'call_{ordinal:04d}'
        validation = mini.read_json(call_dir / 'native_validation.json')
        if validation['ok'] != recorded['native_validation_ok']:
            raise ValueError('prior native validation differs')
        admissions.append((batch['batch_id'], ordinal, bool(validation['ok'])))
        raw = mini.read_json(call_dir / 'result.json'); image_path = call_dir / 'image.bin'
        actual = mini.Result(raw['observation'], (call_dir / 'stdout.bin').read_bytes(),
                             (call_dir / 'stderr.bin').read_bytes(),
                             image_path.read_bytes() if image_path.exists() else None,
                             provenance=raw['provenance'])
        captured = mini.read_json(call_dir / 'capture/states/snapshots/s000000.json')
        mini.validate_snapshot(call_dir / 'capture', captured)
        # Diagnostic ZipInfo.date_time tuples persist as JSON arrays. Normalize
        # receipt representation only; raw outputs/snapshot byte tests above stay exact.
        recomputed = json.loads(mini.canonical(compare_call(
            package, recorded, actual, call_dir / 'capture', captured,
            call_dir, selected['config'], persist=False)))
        if mini.read_json(call_dir / 'comparison.json') != comparison:
            raise ValueError('prior comparison file differs from aggregate')
        if recomputed != comparison:
            changed_fields = {'classification', 'ls_metadata_variance', 'unexplained_tool_difference'}
            if (recorded['call'].get('name') != 'bash'
                    or recorded['call'].get('arguments') != {'command': 'ls -la /workspace/input/'}
                    or comparison['classification'] != 'unexplained'
                    or recomputed['classification'] != 'declared_runtime_variance'
                    or not recomputed['ls_metadata_variance']
                    or {k: v for k, v in comparison.items() if k not in changed_fields}
                       != {k: v for k, v in recomputed.items() if k not in changed_fields}):
                differences = {k: {'before': comparison.get(k), 'recomputed': recomputed.get(k)}
                               for k in set(comparison) | set(recomputed) if comparison.get(k) != recomputed.get(k)}
                raise ValueError('prior comparison cannot be independently reconstructed: '
                                 + batch['batch_id'] + ' ' + recorded['call']['call_id'] + ' '
                                 + json.dumps(differences, sort_keys=True))
            changes.append({'batch_id': batch['batch_id'], 'call_id': recorded['call']['call_id'],
                            'original_comparison': _ref(call_dir / 'comparison.json'),
                            'before': comparison, 'after': recomputed,
                            'raw_files': {p.name: _ref(p) for p in call_dir.iterdir() if p.is_file()},
                            'reason': 'Already-authorized ls UID/GID/timestamp diagnosis; input directory added to literal command allowlist'})
        effective.append(recomputed)
    return admissions, changes, effective


def validate_continuation(plan, _depth=0):
    c = plan.get('continuation')
    if c is None:
        return None
    if _depth >= 3:
        raise ValueError('at most three engineering continuations are authorized')
    old_path = _bound(c['prior_plan']); result_path = _bound(c['prior_result'])
    started = mini.read_json(_bound(c['started_receipt']))
    old = mini.read_json(old_path); previous = mini.read_json(result_path)
    ancestor = validate_continuation(old, _depth + 1)
    operation = plan.get('replay_operation')
    if operation is not None and operation != {'validator_lifetime': 'shared-execution-v1'}:
        raise ValueError('unknown replay operational change')
    # The third and last engineering repair removes repeated SSH handshakes.
    chain_links = 1; cursor = old
    while 'continuation' in cursor:
        chain_links += 1; cursor = mini.read_json(_bound(cursor['continuation']['prior_plan']))
        if chain_links > 3:
            raise ValueError('at most three engineering continuations are authorized')
    if chain_links == 3 and operation != {'validator_lifetime': 'shared-execution-v1'}:
        raise ValueError('third continuation requires the shared native validator')
    excluded = {'source_identity', 'execution_identity_receipt', 'continuation', 'replay_operation'}
    if ({k: v for k, v in old.items() if k not in excluded}
            != {k: v for k, v in plan.items() if k not in excluded}):
        raise ValueError('continuation changed scientific selection/config/native identity/limits')
    original_sha = (mini.read_json(_bound(old['continuation']['started_receipt']))['plan_sha256']
                    if ancestor else c['prior_plan']['sha256'])
    if (previous['plan_sha256'] != c['prior_plan']['sha256']
            or started['plan_sha256'] != original_sha or previous['status'] != 'blocked_error'):
        raise ValueError('continuation plan/result/original-clock binding differs')
    if ancestor and c['started_receipt'] != old['continuation']['started_receipt']:
        raise ValueError('continuation changed original clock receipt')
    error = previous.get('error', {})
    progress_bug = (error.get('type') == 'ValueError' and error.get('message', '').startswith(
                    "unsafe relative evidence path: 'progress_"))
    bridge_startup = (error == {'type': 'NativePiInfrastructureError',
                               'message': 'Native Pi bridge closed or exceeded response limit'})
    if not progress_bug and not bridge_startup:
        raise ValueError('continuation only repairs known filename or zero-call native-start failures')
    if bridge_startup and not ancestor:
        raise ValueError('native-start recovery follows the first engineering continuation')
    if chain_links == 3 and not bridge_startup:
        raise ValueError('third continuation only repairs the documented zero-call SSH startup failure')
    if previous['actor_requests'] != 0 or previous['analyst_requests'] != 0:
        raise ValueError('prior replay violated zero-model contract')
    root = result_path.parent
    if prior_inventory(root) != c['prior_files']:
        raise ValueError('prior raw evidence inventory changed')
    completed = previous['batches']
    if not completed or len(completed) >= len(plan['batches']):
        raise ValueError('expected nonempty incomplete prefix')
    carried = len(ancestor['result']['batches']) if ancestor else 0
    if ancestor and previous['batches'][:carried] != ancestor['result']['batches']:
        raise ValueError('carried original batch receipts changed')
    if bridge_startup:
        failed_batch = plan['batches'][len(completed)]
        failed_dir = root / (failed_batch['batch_id'] + '_replay1')
        failed = mini.read_json(failed_dir / 'batch_result.json')
        failed_events = [mini.read_json(p) for p in sorted(failed_dir.glob('runtime_event_*.json'))]
        if (failed['batch_id'] != failed_batch['batch_id'] or failed['status'] != 'incomplete'
                or failed['calls'] or failed['cleanup'] != cleanup_receipt([])
                or failed_events or list(failed_dir.glob('call_*'))
                or (root / (failed_batch['batch_id'] + '_replay2')).exists()):
            raise ValueError('native-start failure contains tool work or unconfirmed cleanup')
    count = 0; expected_admissions = []; dirs = {}; changes = []; effective = []
    for index, (item, batch) in enumerate(zip(completed, plan['batches'])):
        if item['batch_id'] != batch['batch_id'] or not 1 <= len(item['replays']) <= LIMITS['replays_per_batch']:
            raise ValueError('completed prefix order/replay count differs')
        for replay_index, replay in enumerate(item['replays']):
            key = batch['batch_id'] + f'_replay{replay_index + 1}'
            directory = Path(ancestor['directories'][key]) if index < carried else root / key
            dirs[key] = str(directory)
            admissions, reclassified, recomputed = saved_replay(batch, replay, directory, plan['native_identity'])
            if index >= carried:
                expected_admissions.extend(admissions)
            changes.extend(reclassified); effective.extend(recomputed)
            count += len(admissions)
    admissions = sorted((root / 'admissions').glob('*.json'))
    inherited_count = ancestor['result']['tool_attempts'] if ancestor else 0
    if len(admissions) != count - inherited_count or previous['tool_attempts'] != count:
        raise ValueError('prior attempts include unaccounted partial work')
    actual_admissions = [mini.read_json(p) for p in admissions]
    if [(a['batch_id'], a['ordinal'], a['native_valid_execution']) for a in actual_admissions] != expected_admissions:
        raise ValueError('prior admission order differs')
    execution_count = sum(bool(a['native_valid_execution']) for a in actual_admissions)
    if ancestor:
        execution_count += ancestor['result']['runtime_executions']
    if (previous['runtime_executions'] != execution_count
            or previous['native_feedback_only'] != count - execution_count):
        raise ValueError('prior admission type counts differ')
    started_at = started['started_at']
    if not isinstance(started_at, (float, int)) or started_at > time.time():
        raise ValueError('invalid original start clock')
    return {'result': previous, 'deadline_epoch': started_at + LIMITS['wall_seconds'],
            'directories': dirs, 'reclassifications': changes, 'effective_comparisons': effective}


def _execute(plan_path, output, expected_sha, *, runner=run_batch, shared_validator=None):
    if mini.file_digest(plan_path) != expected_sha:
        raise ValueError('frozen plan SHA mismatch')
    plan = mini.read_json(plan_path)
    mini.assert_output_safe(output.absolute(), [Path(p) for p in plan['package_indices']])
    if output.exists():
        raise ValueError('new output directory required')
    selected_calls = validate_plan(plan, validator=shared_validator)
    continuation = validate_continuation(plan)
    if continuation and time.time() >= continuation['deadline_epoch']:
        raise TimeoutError('original six-hour deadline exhausted')
    output.mkdir(parents=True)
    mini.write_json(output, 'started.json', {'plan_sha256': expected_sha, 'pid': os.getpid(), 'actor_requests': 0})
    report = {'schema': 'prospective_tool_replay_result_v1', 'plan_sha256': expected_sha,
              'actor_requests': 0, 'analyst_requests': 0, 'tool_attempts': 0,
              'selected_first_replay_calls': selected_calls, 'runtime_executions': 0, 'native_feedback_only': 0,
              'status': 'incomplete', 'batches': [], 'not_run': [], 'mechanism_gate_pass': False,
              'validator_lifetime': 'shared-execution-v1' if shared_validator is not None else 'injected-runner',
              'scope': 'bounded-mechanism-v1; recorded-state/request/tool contract only; no outcome-distribution equivalence',
              'excluded_result_envelope': ['call_id', 'provenance'],
              'exclusion_reason': 'call identity is bound separately; execution timestamps/container IDs are provenance, not model observations'}
    deadline = time.monotonic() + (min(LIMITS['wall_seconds'], continuation['deadline_epoch'] - time.time())
                                  if continuation else LIMITS['wall_seconds'])
    if continuation:
        prior = continuation['result']
        for key in ('batches', 'tool_attempts', 'runtime_executions', 'native_feedback_only'):
            report[key] = copy.deepcopy(prior[key])
        report['continuation'] = plan['continuation']
        report['carried_tool_attempts'] = prior['tool_attempts']
        report['absolute_deadline_epoch'] = continuation['deadline_epoch']
        report['reclassifications'] = continuation['reclassifications']
        mini.write_json(output, 'reclassification_sidecar.json', {
            'original_receipts_unchanged': True, 'changes': continuation['reclassifications']})
    admissions = output / 'admissions'; admissions.mkdir()
    def admit(batch_id, ordinal, executed):
        if report['tool_attempts'] >= LIMITS['tool_calls'] or time.monotonic() >= deadline:
            raise TimeoutError('total call or wall limit')
        number = report['tool_attempts'] + 1
        mini.write_json(admissions, f'{number:06d}.json', {'batch_id': batch_id, 'ordinal': ordinal,
                        'native_valid_execution': executed, 'monotonic': time.monotonic()})
        report['tool_attempts'] = number
        report['runtime_executions' if executed else 'native_feedback_only'] += 1
    try:
        with ExitStack() as stack:
            for backend in (mini.OpenAICompatibleBackend, mini.GeminiBackend):
                stack.enter_context(patch.object(backend, 'send', mini.no_actor_send))
            stack.enter_context(patch.object(socket, 'create_connection', mini.no_actor_send))
            for batch in plan['batches'][len(report['batches']):]:
                replays = []
                for replay in range(LIMITS['replays_per_batch']):
                    if replay and not any(c['classification'] != 'exact' for c in replays[0]['calls']):
                        break
                    borrowed = {'shared_validator': shared_validator} if shared_validator is not None else {}
                    row = runner(batch, output / (batch['batch_id'] + f'_replay{replay + 1}'), deadline,
                                 expected_native_identity=plan['native_identity'], admit=admit, **borrowed)
                    replays.append(row)
                report['batches'].append({'batch_id': batch['batch_id'], 'replays': replays})
                mini.write_json(output, 'progress_' + hashlib.sha256(batch['batch_id'].encode()).hexdigest() + '.json', report)
            report['status'] = 'complete'
            inherited = len(continuation['result']['batches']) if continuation else 0
            report['mechanism_gate_pass'] = all(
                not c['unexplained_state_difference'] and not c['unexplained_tool_difference']
                for c in (continuation['effective_comparisons'] if continuation else [])) and all(
                r['status'] == 'complete' and r['cleanup']['confirmed']
                and all(not c['unexplained_state_difference'] and not c['unexplained_tool_difference'] for c in r['calls'])
                for b in report['batches'][inherited:] for r in b['replays'])
            if not report['mechanism_gate_pass']:
                report['status'] = 'blocked_unexplained_difference'
    except Exception as exc:
        report['status'] = 'blocked_error'
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
    finally:
        completed = {b['batch_id'] for b in report['batches']}
        report['not_run'] = [b['batch_id'] for b in plan['batches'] if b['batch_id'] not in completed]
        for path, sha in plan['package_indices'].items():
            try:
                mini.verified(Path(path), sha)
            except Exception as exc:
                report['status'] = 'blocked_input_changed'; report['input_error'] = str(exc)
                report['mechanism_gate_pass'] = False
        mini.write_json(output, 'result.json', report)
    return report


def execute(plan_path, output, expected_sha, *, runner=run_batch, validator_factory=None):
    """Keep one execution-disabled bridge alive, with no automatic reconnection.

    Production uses one open/close for the entire replay. Injected unit-test
    runners may omit a validator; an explicit factory tests shared ownership.
    """
    if mini.file_digest(plan_path) != expected_sha:
        raise ValueError('frozen plan SHA mismatch')
    if output.exists():
        raise ValueError('new output directory required')
    if runner is not run_batch and validator_factory is None:
        return _execute(plan_path, output, expected_sha, runner=runner)
    plan = mini.read_json(plan_path)
    if plan.get('schema') != SCHEMA or plan.get('limits') != LIMITS or plan.get('source_identity') != source_identity():
        raise ValueError('frozen replay policy/source differs before native startup')
    continuation = validate_continuation(plan)
    if continuation and time.time() >= continuation['deadline_epoch']:
        raise TimeoutError('original six-hour deadline exhausted')
    factory = validator_factory or mini.NativePiValidator
    with ExitStack() as stack:
        for backend in (mini.OpenAICompatibleBackend, mini.GeminiBackend):
            stack.enter_context(patch.object(backend, 'send', mini.no_actor_send))
        stack.enter_context(patch.object(socket, 'create_connection', mini.no_actor_send))
        validator = stack.enter_context(factory())
        return _execute(plan_path, output, expected_sha, runner=runner, shared_validator=validator)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.execute:
        if mini.file_digest(args.plan) != args.plan_sha256:
            raise ValueError('frozen plan SHA mismatch')
        validate_continuation(mini.read_json(args.plan))
        print(json.dumps({'mode': 'validate_only', 'selected_calls': validate_plan(mini.read_json(args.plan)),
                          'actor_requests': 0, 'tool_attempts': 0}))
        return 0
    if not args.output:
        parser.error('--execute requires --output')
    def expired(*unused):
        raise TimeoutError('replay wall limit reached')
    alarm_seconds = LIMITS['wall_seconds']
    parsed_plan = mini.read_json(args.plan)
    if 'continuation' in parsed_plan:
        clock = mini.read_json(_bound(parsed_plan['continuation']['started_receipt']))
        remaining = clock['started_at'] + LIMITS['wall_seconds'] - time.time()
        if remaining <= 0:
            raise TimeoutError('original six-hour deadline exhausted')
        alarm_seconds = max(1, int(remaining))
    previous = signal.signal(signal.SIGALRM, expired); signal.alarm(alarm_seconds)
    try:
        result = execute(args.plan.resolve(), args.output, args.plan_sha256)
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)
    print(json.dumps({k: result[k] for k in ('status', 'tool_attempts', 'actor_requests', 'mechanism_gate_pass')}))
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
