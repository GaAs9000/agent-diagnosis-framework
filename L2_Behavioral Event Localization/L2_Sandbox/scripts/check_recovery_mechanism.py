#!/usr/bin/env python3
"""Bounded five-call P0 mechanism check. Default freezes a plan; no Actor requests.

--execute PLAN --output NEW_DIRECTORY --plan-sha256 SHA explicitly replays tools.
Recorded scripts execute only through the production isolated Docker runtime.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import re
import shutil
import sys
import time
import zipfile
from contextlib import ExitStack
from unittest.mock import patch

L2 = Path(__file__).resolve().parents[1]
ROOT = L2.parents[1]
L1 = ROOT / 'L1_sandbox/C3_Experimental_Sandbox'
sys.path[:0] = [str(L1 / 'src'), str(L2 / 'src')]
from c3_sandbox.l1.core.contracts import NativeValidatedCall, Result
from c3_sandbox.l1.evidence.live import validate_snapshot
from c3_sandbox.l1.models.native import OpenAICompatibleBackend, GeminiBackend
from c3_sandbox.l1.package import verify_package
from c3_sandbox.l1.recording.state import StateStore, restore
from c3_sandbox.l1.runtime.docker import DockerRuntime
from c3_sandbox.l1.runtime.pi_native_validation import NativePiValidator
from c3_sandbox.l1.runtime.tools import tools_for_config, format_pi_bash_observation
from c3_sandbox.l1.storage import canonical, confined, digest, file_digest, read_json, write_json, write_bytes

SPECS = (
    ('dev003_image', 'dev003', 'cp000000', 'r000000', 0),
    ('dev003_docx_read', 'dev003', 'cp000000', 'r000000', 1),
    ('context004_docx_read', 'context004', 'cp000000', 'r000021', 0),
    ('context004_docx_write', 'context004', 'cp000003', 'r000024', 0),
    ('context004_ls', 'context004', 'cp000007', 'r000028', 0),
)
FIELDS = ('files', 'directories', 'total_bytes', 'content_sha256')
LIMITS = {'tool_calls': 5, 'tool_timeout_seconds': 60, 'wall_seconds': 1800,
          'workers': 1, 'actor_http_requests': 0, 'replays_per_call': 1}


def events(root):
    return [json.loads(line) for line in (root / 'trajectory.jsonl').read_text().splitlines()]


def referenced(root, ref):
    path = confined(root, ref['path'])
    if file_digest(path) != ref['sha256']:
        raise ValueError('recorded reference changed: ' + ref['path'])
    return path


def verified(root, index_sha):
    if file_digest(root / 'evidence_index.json') != index_sha:
        raise ValueError('package index changed')
    result = verify_package(root, expected_index_sha256=index_sha)
    if not result['verified']:
        raise ValueError('package integrity failed: ' + repr(result['errors']))


def reconstruct(root, cp, request_event, config):
    # object.__new__ avoids HTTP client construction and credential resolution.
    cls = {'openai-compatible': OpenAICompatibleBackend, 'gemini': GeminiBackend}[config['model']['adapter']]
    backend = object.__new__(cls)
    backend.config = config['model']
    backend.native_pi_validation = config.get('agent', {}).get('validation_policy') == 'pi-native-0.73.1'
    rebuilt = canonical(backend.request(cp['conversation'], tools_for_config(config)))
    recorded = referenced(root, request_event['data']['body']).read_bytes()
    if rebuilt != recorded:
        raise ValueError('next request raw bytes differ')
    return {'raw_sha256': hashlib.sha256(recorded).hexdigest(), 'size_bytes': len(recorded)}


def select_case(root, label, cp_id, request_suffix, ordinal):
    es = events(root)
    def one(matches, what):
        if len(matches) != 1:
            raise ValueError('expected one ' + what)
        return matches[0]
    ce = one([e for e in es if e['kind'] == 'checkpoint.created'
              and e['data']['checkpoint']['path'] == 'checkpoints/' + cp_id + '.json'], 'checkpoint')
    cp = read_json(referenced(root, ce['data']['checkpoint']))
    req = one([e for e in es if e['kind'] == 'model.request'
               and e['data']['request_id'].endswith(':' + request_suffix)], 'request')
    following = [e for e in es if e['kind'] == 'model.request' and e['seq'] > ce['seq']]
    if not following or following[0] != req or cp['budget_used']['model_requests'] != int(request_suffix[1:]):
        raise ValueError('checkpoint/global request binding mismatch')
    if cp['pending_calls'] or not cp['qualification']['eligible']:
        raise ValueError('checkpoint is not a qualified complete tool boundary')
    if root.name == 'officeval_004.r02':
        # This child starts at global 21, not local checkpoint ordinal zero.
        manifest = read_json(root / 'manifest.json')
        if (not manifest.get('lineage', {}).get('context_overflow')
                or read_json(root / 'checkpoints/cp000000.json')['budget_used']['model_requests'] != 21):
            raise ValueError('context004 carried prefix binding missing')
    calls = [e for e in es if e['kind'] == 'tool.call' and e['data']['request_id'] == req['data']['request_id']]
    if ordinal >= len(calls):
        raise ValueError('fixed recorded tool is absent')
    call = calls[ordinal]
    cid = call['data']['call_id']
    start = one([e for e in es if e['kind'] == 'tool.started' and e['data']['call_id'] == cid], 'tool start')
    result = one([e for e in es if e['kind'] == 'tool.result' and e['data']['call_id'] == cid], 'tool result')
    snapshots = [e for e in es if e['kind'] == 'state.snapshot']
    before = one([e for e in snapshots if read_json(referenced(root, e['data']['snapshot']))['snapshot_id']
                  == start['data']['before_snapshot_id']], 'before snapshot')
    after = one([e for e in snapshots if e.get('links', {}).get('tool_result') == result['event_id']], 'after snapshot')
    config = read_json(root / 'raw/resolved_config.json')
    request = reconstruct(root, cp, req, config)
    for event in (before, after):
        validate_snapshot(root, read_json(referenced(root, event['data']['snapshot'])))
    for key in ('result', 'stdout', 'stderr', 'image'):
        if key in result['data']:
            referenced(root, result['data'][key])
    return {'case': label, 'package': str(root.resolve()), 'checkpoint_id': cp_id,
            'checkpoint_ref': ce['data']['checkpoint'], 'request_ref': req['data']['body'],
            'request_id': req['data']['request_id'], 'request_exact': request,
            'call': call['data'], 'call_event_sha256': call['sha256'], 'batch_ordinal': ordinal,
            'before_snapshot': before['data']['snapshot'], 'after_snapshot': after['data']['snapshot'],
            'original_result': result['data'],
            'original_observation': read_json(referenced(root, result['data']['result']))['observation'],
            'budget_used': cp['budget_used']}


def source_identity():
    paths = [Path(__file__).resolve(), *sorted((L1 / 'src/c3_sandbox/l1').rglob('*.py'))]
    paths += sorted((L1 / 'src/c3_sandbox/l1/runtime/pi_native').glob('*'))
    return {str(path.relative_to(ROOT)): file_digest(path) for path in paths if path.is_file()}


def native_identity(validator=None):
    """Bind the selected bridge and installed tree, including environment overrides."""
    if validator is None:
        with NativePiValidator() as actual:
            return native_identity(actual)
    dependency = validator.dependency_identity()
    if (dependency.get('version') != '0.73.1' or dependency.get('actor_requests') != 0
            or dependency.get('approved_modules_verified') is not True
            or not dependency.get('installed_tree_sha256')):
        raise ValueError('actual native validator lacks pinned installed identity')
    selected = validator.node_executable
    node = Path(shutil.which(selected) or selected).resolve(strict=True)
    bridge = validator.bridge_dir.resolve(strict=True)
    return {'dependency': dependency, 'node_path': str(node), 'node_sha256': file_digest(node),
            'bridge_path': str(bridge),
            'bridge_files': {p.name: file_digest(p) for p in sorted(bridge.iterdir()) if p.is_file()}}


def build_plan(dev003, context004):
    packages = {'dev003': dev003.resolve(), 'context004': context004.resolve()}
    indices = {str(p): file_digest(p / 'evidence_index.json') for p in packages.values()}
    for path, sha in indices.items():
        verified(Path(path), sha)
    cases = [select_case(packages[pkg], label, cp, req, ordinal)
             for label, pkg, cp, req, ordinal in SPECS]
    return {'schema': 'recovery_mechanism_mini_v1', 'limits': LIMITS,
            'native_identity': native_identity(),
            'package_indices': indices, 'source_identity': source_identity(), 'cases': cases,
            'scope': 'development_only; excludes provider hidden state and unsaved metadata; no live gate promotion'}


def no_actor_send(*args, **kwargs):
    raise AssertionError('Actor send is forbidden in the mechanism replay')


def wall_timeout(signum, frame):
    raise TimeoutError('mini 1800 second wall limit reached')


def assert_output_safe(output, roots):
    absolute = output.absolute()
    if absolute != output.resolve():
        raise ValueError('output symlink/alias is forbidden')
    for protected in [L1 / 'runs', *roots]:
        if output == protected or output.is_relative_to(protected):
            raise ValueError('output overlaps immutable evidence')


def validate_plan(plan):
    if plan.get('schema') != 'recovery_mechanism_mini_v1' or plan.get('limits') != LIMITS:
        raise ValueError('plan policy/limits mismatch')
    if plan.get('source_identity') != source_identity():
        raise ValueError('execution source differs from frozen plan')
    if plan.get('native_identity') != native_identity():
        raise ValueError('actual native validator differs from frozen plan')
    cases = plan.get('cases', [])
    if len(cases) != 5 or [c['case'] for c in cases] != [s[0] for s in SPECS]:
        raise ValueError('exact five fixed cases required')
    for path, sha in plan['package_indices'].items():
        verified(Path(path), sha)
    for case, (label, _, cp, req, ordinal) in zip(cases, SPECS):
        root = Path(case['package'])
        if str(root) not in plan['package_indices']:
            raise ValueError('unbound case package')
        if case != select_case(root, label, cp, req, ordinal):
            raise ValueError('case differs from recorded fixed selection')


def snapshot_equal(a, b):
    return all(a[field] == b[field] for field in FIELDS)


def capture(root, workspace, config, label):
    root.mkdir()
    return StateStore(root, config['max_workspace_bytes'], config['max_files']).capture(workspace, label)


def zip_time_only(root, expected, captured_root, actual):
    """Diagnose ZIP timestamp variance without changing either byte comparison."""
    changes = StateStore.diff(expected, actual)
    details = []
    if not changes:
        return []
    for change in changes:
        a, b = change['before'], change['after']
        if (change['change'] != 'modified' or not change['path'].endswith('.docx')
                or a['mode'] != b['mode'] or a['size_bytes'] != b['size_bytes']):
            return []
        try:
            left_bytes = confined(root, a['blob']).read_bytes()
            right_bytes = confined(captured_root, b['blob']).read_bytes()
            import io
            with zipfile.ZipFile(io.BytesIO(left_bytes)) as left, zipfile.ZipFile(io.BytesIO(right_bytes)) as right:
                if left.namelist() != right.namelist() or left.comment != right.comment:
                    return []
                stamps = []
                for li, ri in zip(left.infolist(), right.infolist()):
                    # Compare every persisted member attribute except the DOS timestamp.
                    attrs = ('filename', 'compress_type', 'comment', 'extra', 'create_system',
                             'create_version', 'extract_version', 'flag_bits', 'volume',
                             'internal_attr', 'external_attr', 'CRC', 'compress_size', 'file_size', 'header_offset')
                    if (any(getattr(li, k) != getattr(ri, k) for k in attrs)
                            or left.read(li) != right.read(ri)):
                        return []
                    if li.date_time != ri.date_time:
                        stamps.append({'member': li.filename, 'expected': li.date_time, 'actual': ri.date_time})
                if not stamps:
                    return []
                # Re-encode with recorded timestamps; this diagnoses byte-locality only.
                # It does not replace stored outputs or the exact comparison above.
                time_only = bytearray(right_bytes)
                for li, ri in zip(left.infolist(), right.infolist()):
                    time_only[ri.header_offset + 10:ri.header_offset + 14] = left_bytes[li.header_offset + 10:li.header_offset + 14]
                # Central headers have their own timestamps. Locate them by validated order.
                offset = right.start_dir
                for li, ri in zip(left.infolist(), right.infolist()):
                    if right_bytes[offset:offset + 4] != b'PK\x01\x02':
                        return []
                    import struct
                    name_len, extra_len, comment_len = struct.unpack_from('<HHH', right_bytes, offset + 28)
                    expected_time = (li.date_time[3] << 11) | (li.date_time[4] << 5) | (li.date_time[5] // 2)
                    expected_date = ((li.date_time[0] - 1980) << 9) | (li.date_time[1] << 5) | li.date_time[2]
                    time_only[offset + 12:offset + 16] = struct.pack('<HH', expected_time, expected_date)
                    offset += 46 + name_len + extra_len + comment_len
                if bytes(time_only) != left_bytes:
                    return []
                details.append({'path': change['path'], 'member_timestamp_changes': stamps,
                                'expected_sha256': a['sha256'], 'actual_sha256': b['sha256']})
        except (OSError, ValueError, zipfile.BadZipFile):
            return []
    return details


def ls_metadata_only(expected, actual):
    """Only allow ls UID/GID and timestamp columns; keep names/mode/size/count exact."""
    if expected == actual:
        return False
    try:
        left, right = expected.decode('utf-8').splitlines(), actual.decode('utf-8').splitlines()
    except UnicodeDecodeError:
        return False
    if len(left) != len(right) or not left or not re.fullmatch(r'total \d+', left[0]) or left[0] != right[0]:
        return False
    line_pattern = re.compile(r'(?P<mode>[-dlcbps][-rwxStTs]{9}[.+@]?)\s+(?P<links>\d+)\s+'
                              r'(?P<uid>\S+)\s+(?P<gid>\S+)\s+(?P<size>\d+)\s+'
                              r'(?P<month>\S+)\s+(?P<day>\d+)\s+(?P<when>[\d:]+)\s+(?P<name>.+)')
    for original, replay in zip(left[1:], right[1:]):
        a, b = line_pattern.fullmatch(original), line_pattern.fullmatch(replay)
        if not a or not b or any(a[k] != b[k] for k in ('mode', 'links', 'size', 'name')):
            return False
    return True


def run_case(case, output, deadline, *, expected_native_identity,
             runtime_factory=DockerRuntime, validator_factory=NativePiValidator):
    root = Path(case['package'])
    config = read_json(root / 'raw/resolved_config.json')
    before = read_json(referenced(root, case['before_snapshot']))
    after = read_json(referenced(root, case['after_snapshot']))
    case_dir = output / case['case']
    case_dir.mkdir()
    workspace = case_dir / 'workspace'
    restore(root, before, workspace)
    restored = capture(case_dir / 'before_capture', workspace, config['runtime'], 'restored')
    if not snapshot_equal(before, restored):
        raise ValueError('restored snapshot bytes/directories/modes differ')
    call = case['call']
    runtime = runtime_factory(copy.deepcopy(config['runtime']))
    runtime_events = []
    def runtime_event(kind, data):
        row = {'kind': kind, 'data': data}
        write_json(case_dir, f'runtime_event_{len(runtime_events):04d}.json', row)
        runtime_events.append(row)
    runtime.set_event_sink(runtime_event)
    with validator_factory() as validator:
        actual_identity = native_identity(validator)
        if actual_identity != expected_native_identity:
            raise ValueError('actual native validator differs before tool replay')
        write_json(case_dir, 'native_identity.json', actual_identity)
        provenance = runtime.preflight()
        cp = read_json(referenced(root, case['checkpoint_ref']))
        if provenance['image_id'] != cp['qualification']['image_id']:
            raise ValueError('actual runtime image differs from recorded checkpoint')
        write_json(case_dir, 'runtime_provenance.json', provenance)
        validation = validator.validate(call['name'], call['arguments'], call['native_id'] or call['call_id'], deadline_monotonic=deadline)
        write_json(case_dir, 'native_validation.json', validation)
        if time.monotonic() >= deadline:
            raise TimeoutError('mini total deadline before tool')
        if validation['ok']:
            execution = NativeValidatedCall(call['call_id'], call['name'], validation['validated_arguments'], call['native_id'])
            actual = runtime.execute(execution, workspace, min(60, deadline - time.monotonic()),
                                     supports_images=config['model']['supports_images'])
        else:
            actual = Result(copy.deepcopy(validation['native_result']), provenance={'pi_native_validation_error': True})
    write_json(case_dir, 'result.json', {'observation': actual.observation, 'provenance': actual.provenance})
    write_json(case_dir, 'runtime_events.json', runtime_events)
    write_bytes(case_dir, 'stdout.bin', actual.stdout)
    write_bytes(case_dir, 'stderr.bin', actual.stderr)
    if actual.image is not None:
        write_bytes(case_dir, 'image.bin', actual.image)
    captured = capture(case_dir / 'after_capture', workspace, config['runtime'], 'after_replay')
    original = case['original_result']
    observed = read_json(referenced(root, original['result']))['observation']
    observation_equal = actual.observation == observed
    streams_equal = all((case_dir / (key + '.bin')).read_bytes() == referenced(root, original[key]).read_bytes()
                        for key in ('stdout', 'stderr'))
    original_image = referenced(root, original['image']).read_bytes() if 'image' in original else None
    image_equal = actual.image == original_image
    state_equal = snapshot_equal(after, captured)
    exact = observation_equal and streams_equal and image_equal and state_equal
    zip_variance = (zip_time_only(root, after, case_dir / 'after_capture', captured)
                    if case['case'] == 'context004_docx_write' and not state_equal else [])
    # ls -la reports unsaved mtime/UID. Preserve all bytes; never convert this to exact.
    declared = (not exact and state_equal and case['case'] == 'context004_ls'
                and call['arguments'] == {'command': 'ls -la /workspace/submission/'}
                and actual.stderr == referenced(root, original['stderr']).read_bytes()
                and image_equal and actual.observation.get('exit_code') == observed.get('exit_code') == 0
                and ls_metadata_only(referenced(root, original['stdout']).read_bytes(), actual.stdout)
                and actual.observation == format_pi_bash_observation(
                    actual.stdout, actual.stderr, 0, actual.observation.get('stop_reason'),
                    actual.observation.get('output_complete'), actual.observation.get('oom_killed'))
                and {k: v for k, v in actual.observation.items() if k not in ('stdout', 'output')}
                    == {k: v for k, v in observed.items() if k not in ('stdout', 'output')})
    if zip_variance and observation_equal and streams_equal and image_equal:
        declared = True
    return {'case': case['case'], 'call_id': call['call_id'], 'restore_exact': True,
            'request_exact': True, 'deterministic_tool_exact': exact,
            'observation_exact': observation_equal, 'raw_streams_exact': streams_equal,
            'image_exact': image_equal, 'after_state_exact': state_equal,
            'declared_runtime_variance': declared,
            'variance_reason': ('zip member timestamps differ; byte differences retained' if zip_variance
                                else 'ls metadata contains unsaved mtime/UID; full diff retained') if declared else None,
            'zip_timestamp_variance': zip_variance,
            'unexplained_state_difference': not state_equal and not zip_variance,
            'unexplained_tool_difference': not exact and not declared,
            'before_snapshot_sha256': before['content_sha256'],
            'expected_after_snapshot_sha256': after['content_sha256'],
            'actual_after_snapshot_sha256': captured['content_sha256'],
            'state_diff': StateStore.diff(after, captured), 'native_validation_ok': validation['ok']}


def execute(plan_path, output, expected_sha, *, runner=run_case):
    if file_digest(plan_path) != expected_sha:
        raise ValueError('explicit frozen plan SHA mismatch')
    plan = read_json(plan_path)
    assert_output_safe(output.absolute(), [Path(p) for p in plan['package_indices']])
    if output.exists():
        raise ValueError('execute requires a new output directory')
    validate_plan(plan)  # Fail closed before any writes or tools.
    output.mkdir(parents=True)
    marker = os.open(output / 'started.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(marker, 'wb') as stream:
        stream.write(canonical({'plan_sha256': expected_sha, 'actor_requests': 0, 'pid': os.getpid()}))
        stream.flush(); os.fsync(stream.fileno())
    deadline = time.monotonic() + 1800
    report = {'schema': 'recovery_mechanism_mini_result_v1', 'plan_sha256': expected_sha,
              'actor_requests': 0, 'analyst_requests': 0, 'tool_attempts': 0,
              'cases': [], 'status': 'incomplete', 'live_gate_pass': False}
    try:
        with ExitStack() as stack:
            for backend in (OpenAICompatibleBackend, GeminiBackend):
                stack.enter_context(patch.object(backend, 'send', no_actor_send))
            for case in plan['cases']:
                if time.monotonic() >= deadline or report['tool_attempts'] >= 5:
                    raise TimeoutError('mini execution limit reached')
                report['tool_attempts'] += 1
                report['current_case'] = case['case']
                row = runner(case, output, deadline, expected_native_identity=plan['native_identity'])
                report['cases'].append(row)
                write_json(output, 'progress_' + case['case'] + '.json', report)
                if row['unexplained_state_difference'] or row['unexplained_tool_difference']:
                    report['status'] = 'blocked_unexplained_difference'
                    break
            else:
                report['status'] = 'review_runtime_variance' if any(
                    r['declared_runtime_variance'] for r in report['cases']) else 'mini_exact_with_limited_coverage'
    except Exception as exc:
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        report['status'] = 'blocked_error'
    finally:
        for path, sha in plan['package_indices'].items():
            try:
                verified(Path(path), sha)
            except Exception as exc:
                report['status'] = 'blocked_input_changed'; report['input_error'] = str(exc)
        write_json(output, 'result.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dev003', type=Path, default=L1 / 'runs/v3_qwen36_pi_dev45_20261017/episodes/officeval_003.r00')
    parser.add_argument('--context004', type=Path, default=L1 / 'runs/v3_qwen36_pi_context004_20261018/segments/officeval_004.r02')
    parser.add_argument('--plan', type=Path, help='New mini-plan output (default mode)')
    parser.add_argument('--execute', type=Path, metavar='PLAN')
    parser.add_argument('--plan-sha256')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.execute:
        if not args.plan_sha256 or not args.output or args.plan:
            parser.error('--execute requires --plan-sha256 and --output, without --plan')
        # An OS timer also bounds restore/validation/capture, not just tool admission.
        previous = signal.signal(signal.SIGALRM, wall_timeout)
        signal.alarm(1800)
        try:
            report = execute(args.execute.resolve(), args.output, args.plan_sha256)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
        print(json.dumps({'status': report['status'], 'tool_attempts': report['tool_attempts'], 'actor_requests': 0}))
        return 1 if report['status'].startswith('blocked') else 0
    if not args.plan:
        parser.error('--plan is required in default plan-only mode')
    plan = build_plan(args.dev003, args.context004)
    assert_output_safe(args.plan.absolute(), [args.dev003.resolve(), args.context004.resolve()])
    if args.plan.exists():
        raise ValueError('plan destination already exists')
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.plan.parent, args.plan.name, plan)
    print(json.dumps({'mode': 'plan_only', 'plan_sha256': file_digest(args.plan), 'cases': 5, 'actor_requests': 0}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
