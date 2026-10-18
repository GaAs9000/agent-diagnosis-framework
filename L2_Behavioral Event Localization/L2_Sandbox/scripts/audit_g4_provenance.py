#!/usr/bin/env python3
"""Constructive, offline DEV-only G4 audit. Never calls an Actor or grader.

Exact replay proves consistency with the *current* controller implementation.
It does not authenticate an unknown historical controller or certify image
contents. Those are separate, deliberately unverified trust-graph edges.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import sys

L2_ROOT = Path(__file__).resolve().parents[1]
ROOT = L2_ROOT.parents[1]
L1_ROOT = ROOT / 'L1_sandbox/C3_Experimental_Sandbox'
sys.path[:0] = [str(L1_ROOT / 'src'), str(L2_ROOT / 'src')]

from c3_sandbox.l1.core.contracts import L1Error, Result
from c3_sandbox.l1.core.orchestrator import _maybe_compact_pi_conversation
from c3_sandbox.l1.evaluation.study import study_outcome
from c3_sandbox.l1.models.native import OpenAICompatibleBackend
from c3_sandbox.l1.package import verify_package
from c3_sandbox.l1.runtime.tools import _detect_image_mime, format_pi_bash_observation, tools_for_config
from c3_sandbox.l1.storage import confined, digest, file_digest, read_json

DEV = tuple('officeval_' + n for n in ('003', '004', '010', '033', '050', '060', '070', '085', '090'))
DEFAULT_EPISODES = L1_ROOT / 'runs/v3_qwen36_pi_dev45_20261017/episodes'


def backend(config):
    if config['model']['adapter'] != 'openai-compatible':
        raise ValueError('Only the DEV OpenAI-compatible protocol is implemented')
    # Deliberately bypass HTTP construction. Only pure protocol methods are used.
    model = object.__new__(OpenAICompatibleBackend)
    model.config = config['model']
    model.native_pi_validation = config['agent'].get('validation_policy') == 'pi-native-0.73.1'
    return model


def initial_conversation(config, public):
    prompt = public['prompt'] + '\n\nInput files in /workspace/input:\n' + '\n'.join(a['name'] for a in public['assets'])
    return backend(config).initial(config['agent']['system_prompt'], prompt)


def compare(expected, actual):
    """Compare the entire JSON structure, including unknown/extra fields."""
    return {'status': 'verified' if expected == actual else 'mismatch',
            'expected_sha256': digest(expected), 'actual_sha256': digest(actual)}


def reconstruct_observation(call, raw, result_data, read):
    """Restore production dict insertion order lost by canonical evidence JSON.

    The backend embeds json.dumps(observation) as a *string*. Sorting that
    observation changes sent bytes, even though the parsed object is equal.
    Bash is rebuilt with the actual formatter and retained streams. Other
    supported tools use their exact current production literal field order;
    unrecognized fields/schema are refused rather than normalized away.
    """
    obs = raw['observation']
    if raw.get('provenance', {}).get('pi_native_validation_error') is True:
        if (not isinstance(obs, dict) or set(obs) != {'content', 'details', 'isError'}
                or obs['isError'] is not True or obs['details'] != {}
                or not isinstance(obs['content'], list) or not obs['content']
                or any(not isinstance(block, dict) or set(block) != {'type', 'text'}
                       or block['type'] != 'text' or not isinstance(block['text'], str)
                       for block in obs['content'])):
            raise ValueError('unsupported native validation error schema')
        return copy.deepcopy(obs)
    if call.name == 'bash':
        stdout = read(result_data['stdout'])
        stderr = read(result_data['stderr'])
        provenance = raw['provenance']
        rebuilt = format_pi_bash_observation(stdout, stderr, provenance['container_exit_code'],
                                             provenance['stop_reason'], provenance['output_complete'], provenance['oom_killed'])
        if rebuilt != obs:
            raise ValueError('bash observation differs from actual streams/runtime receipt')
        return rebuilt
    prefix = ('output', 'stdout', 'stderr')
    tail = ('exit_code', 'stop_reason', 'output_complete')
    candidates = [prefix + tail]  # common failed-tool observation
    if call.name == 'read':
        candidates += [prefix + ('path',) + tail, prefix + ('path', 'mime_type', 'size_bytes') + tail]
    elif call.name == 'write':
        candidates += [prefix + ('path', 'bytes_written') + tail]
    elif call.name == 'edit':
        candidates += [prefix + ('path', 'diff') + tail]
    elif call.name == 'read_image':
        candidates += [('path', 'mime_type', 'size_bytes', 'exit_code')]
    elif call.name == 'run':
        candidates += [('stdout', 'stderr', 'exit_code', 'stop_reason', 'output_complete', 'oom_killed')]
    matches = [order for order in candidates if set(order) == set(obs)]
    if len(matches) != 1:
        raise ValueError('unsupported or ambiguous production observation schema for ' + call.name)
    return {key: copy.deepcopy(obs[key]) for key in matches[0]}


def reconstruct_requests(config, public, events, read, *, parent_checkpoint=None):
    """Replay every history rewrite, not a prefix or newly appended delta.

    `read(ref)` returns indexed package bytes. Observations are independently
    rebuilt from actual tool results and the configured truncation transform.
    Unsupported injections and missing transitions stop verification; a later
    saved checkpoint is never silently adopted as trusted conversation state.
    """
    model = backend(config)
    conversation = initial_conversation(config, public)
    if config['agent'].get('intervention'):
        return [{'status': 'unverified', 'reason': 'intervention_not_supported_by_plain_DEV_G2_audit'}]
    if parent_checkpoint is not None:
        if parent_checkpoint.get('budget_used', {}).get('model_requests') != 0:
            requests = [e for e in events if e['kind'] == 'model.request']
            bindings = [e for e in events if e['kind'] == 'response.bound']
            if (not requests or requests[0]['data'].get('source') != 'saved_response'
                    or requests[0]['data'].get('http_attempts') != 0 or len(bindings) != 1):
                return [{'status': 'unverified', 'reason': 'only_plain_c0_G2_parent_supported'}]
            first = requests[0]['data']
            responses = [e for e in events if e['kind'] == 'model.response'
                         and e['data']['request_id'] == first['request_id']]
            binding = bindings[0]['data']
            if (len(responses) != 1 or read(binding['request']) != read(first['body'])
                    or read(binding['response']) != read(responses[0]['data']['body'])):
                return [{'status': 'mismatch', 'reason': 'saved_response_bytes_differ_from_bound_source'}]
            conversation = copy.deepcopy(parent_checkpoint['conversation'])
        if parent_checkpoint['conversation'] != conversation:
            return [{'status': 'mismatch', 'reason': 'parent_c0_not_public_initial_conversation'}]
    rows = []
    requests = [e for e in events if e['kind'] == 'model.request']
    by_request = {e['data']['request_id']: e for e in events if e['kind'] == 'model.response'}
    results = {e['data']['call_id']: e for e in events if e['kind'] == 'tool.result'}
    last_request_position = max((i for i, e in enumerate(events) if e['kind'] == 'model.request'), default=-1)
    compact_events = [e['data'] for e in events[:last_request_position] if e['kind'] == 'session.compacted']
    actual_compactions = []
    for number, event in enumerate(requests):
        rid = event['data']['request_id']
        actual = json.loads(read(event['data']['body']))
        expected = model.request(conversation, tools_for_config(config))
        row = {'request_id': rid, 'event_id': event['event_id'],
               'reason': 'full_messages_tools_parameters_reconstructed', **compare(expected, actual)}
        if parent_checkpoint is not None and parent_checkpoint.get('budget_used', {}).get('model_requests') != 0:
            row['history_scope'] = 'suffix_from_supplied_parent_checkpoint; historical_prefix_not_reconstructed'
            row['parent_conversation_sha256'] = digest(parent_checkpoint['conversation'])
        rows.append(row)
        if row['status'] != 'verified':
            row['reason'] = 'full_request_differs_from_constructed_input'
            rows.extend({'request_id': x['data']['request_id'], 'status': 'unverified',
                         'reason': 'earlier_history_mismatch'} for x in requests[number + 1:])
            break
        if number == len(requests) - 1:
            break  # No subsequent request depends on the final continuation.
        try:
            response = by_request[rid]
            reply = model.parse(read(response['data']['body']), rid)
            accepted = []
            sources = []
            for call in reply.calls:
                result_event = results[call.call_id]
                raw = json.loads(read(result_event['data']['result']))
                if raw['call_id'] != call.call_id:
                    raise ValueError('tool result call identity mismatch')
                delivered = reconstruct_observation(call, raw, result_event['data'], read)
                if raw.get('provenance', {}).get('pi_native_validation_error') is True:
                    validations = [e['data'] for e in events if e['kind'] == 'tool.validation'
                                   and e['data'].get('call_id') == call.call_id]
                    if (not model.native_pi_validation or len(validations) != 1
                            or not {'raw_arguments', 'validated_arguments', 'native_result', 'provenance', 'telemetry'} <= validations[0].keys()
                            or validations[0].get('ok') is not False
                            or validations[0].get('raw_arguments') != call.arguments
                            or validations[0].get('native_result') != delivered
                            or validations[0].get('provenance') != raw['provenance'].get('validation')):
                        raise ValueError('native validation error differs from validation event')
                limit = config['recording']['observation_limit_chars']
                cut = []
                for field in ('stdout', 'stderr', 'output'):
                    if isinstance(delivered.get(field), str) and len(delivered[field]) > limit:
                        delivered[field] = delivered[field][:limit]
                        cut.append(field)
                if cut:
                    delivered['observation_truncation'] = {'fields': cut, 'limit_chars': limit, 'raw_retained': True}
                image_ref = result_event['data'].get('image')
                image = read(image_ref) if image_ref else None
                mime = _detect_image_mime(image) if image else None
                if image and mime is None:
                    raise ValueError('unrecognized actual tool image')
                accepted.append((call, Result(delivered, image=image, image_mime=mime, provenance=copy.deepcopy(raw.get('provenance', {})))))
                sources.append({'call_id': call.call_id, 'result_sha256': digest(raw),
                                'delivered_observation_sha256': digest(delivered),
                                'image_sha256': image_ref.get('sha256') if image_ref else None})
            conversation = model.accept(conversation, reply, accepted)
            usage = reply.usage or {}
            tokens = usage.get('totalTokenCount', usage.get('total_tokens'))
            count = int(rid.rsplit(':r', 1)[1]) + 1
            conversation, receipt = _maybe_compact_pi_conversation(
                conversation, config['agent'].get('compaction'), tokens if type(tokens) is int else None, count)
            if receipt is not None:
                actual_compactions.append(receipt)
            row['continuation_sources'] = sources
        except (L1Error, ValueError, KeyError, TypeError) as exc:
            rows.extend({'request_id': x['data']['request_id'], 'status': 'unverified',
                         'reason': 'cannot_reconstruct_prior_transition: ' + str(exc)} for x in requests[number + 1:])
            break
    # Last-turn compaction may exist without a subsequent request; it does not
    # enter a sent body and is excluded from this exact compared prefix.
    # If replay stopped early its remaining receipts are not audited; request
    # rows already state why that suffix is unverified. For a complete replay,
    # extra as well as missing compaction events must be detected.
    complete = len(rows) == len(requests) and all(r['status'] == 'verified' for r in rows)
    compared = compact_events if complete else compact_events[:len(actual_compactions)]
    if actual_compactions != compared:
        rows.append({'status': 'mismatch', 'reason': 'compaction_receipt_differs_from_replayed_transform'})
    return rows


def source_identity():
    paths = [Path(__file__).resolve(), *(L1_ROOT / 'src/c3_sandbox/l1').rglob('*.py')]
    files = {str(p.relative_to(ROOT)): file_digest(p) for p in sorted(paths)}
    return {'sha256': digest(files), 'files': files, 'scope': 'current_audit_and_replay_code_only'}


def runtime_graph(events, runtime, *, episode_id=None):
    """Saved mount claims cannot establish unrecorded image/controller facts."""
    containers = [e['data'] for e in events if e['kind'] == 'runtime.container.created']
    unexpected = []
    unknown = []
    valid_episode = isinstance(episode_id, str) and re.fullmatch(r'ep_[0-9a-f]{32}', episode_id) is not None
    expected_source = '.l1-work-' + episode_id if valid_episode else None
    probe_id = runtime.get('environment_probe', {}).get('container_id')
    probe_count = 0
    actor_count = 0
    if expected_source is None:
        unknown.append({'reason': 'missing_or_invalid_episode_id'})
    for c in containers:
        cid = c.get('container_id')
        mounts = c.get('mounts')
        if not isinstance(cid, str) or not cid:
            unknown.append({'reason': 'missing_container_identity'})
        if not isinstance(mounts, list):
            unknown.append({'container_id': cid, 'reason': 'missing_or_invalid_mount_inventory'})
            continue
        # Production preflight runs its image probe with no workspace mount.
        # Only the exact recorded probe container is exempt from Actor mounts.
        if isinstance(probe_id, str) and probe_id and cid == probe_id:
            probe_count += 1
            if mounts:
                unexpected.append({'container_id': cid, 'reason': 'image_probe_has_unexpected_mounts', 'mounts': mounts})
            continue
        actor_count += 1
        if not mounts:
            unknown.append({'container_id': cid, 'reason': 'empty_actor_mount_inventory'})
            continue
        if len(mounts) != 1:
            unexpected.append({'container_id': cid, 'reason': 'extra_actor_mounts', 'mounts': mounts})
            continue
        mount = mounts[0]
        if not isinstance(mount, dict) or set(mount) != {'target', 'read_only', 'source_name'}:
            unknown.append({'container_id': cid, 'reason': 'incomplete_or_unknown_mount_receipt_schema'})
            continue
        if (mount['target'] != '/workspace' or mount['read_only'] is not False
                or (expected_source is not None and mount['source_name'] != expected_source)):
            unexpected.append({'container_id': cid, 'mount': mount})
    if not actor_count:
        unknown.append({'reason': 'no_actor_container_mount_receipts'})
    if probe_count > 1:
        unknown.append({'reason': 'duplicate_image_probe_identity'})
    return {
        'recorded_actor_mounts': {'status': 'mismatch' if unexpected else ('unverified' if unknown else 'verified'),
                                  'container_count': len(containers), 'unexpected': unexpected,
                                  'actor_container_count': actor_count, 'identified_probe_count': probe_count,
                                  'unverified_reasons': unknown,
                                  'expected_workspace_source_name': expected_source,
                                  'scope': 'recorded mount receipts; generic image files are not mounts'},
        'image_contents': {'status': 'unverified', 'image_id': runtime.get('image_id'),
                           'generic_grader_worker_sha256': runtime.get('python_environment', {}).get('grader_worker_sha256'),
                           'reason': 'no independently authenticated image filesystem exclusion inventory; generic worker may be accessible'},
        'historical_controller': {'status': 'unverified',
                                  'reason': 'saved package does not attest historical controller/tool implementation digests'},
        'external_input_exclusion': {'status': 'unverified',
                                     'reason': 'constructive request consistency is insufficient without historical controller and image trust edges'},
        'initial_system_and_public_task_origin': {'status': 'unverified',
                                                  'reason': 'saved config/public task are replay roots; no external historical approval/source anchor is supplied'},
    }


def audit_package(path):
    path = Path(path)
    public = read_json(path / 'benchmark/public_task.json')
    if public['task_id'] not in DEV:
        raise ValueError('G4 audit restricted to frozen DEV9; refusing other task content')
    integrity = verify_package(path)
    if not integrity.get('verified', False):
        return {'package': str(path), 'task_id': public['task_id'], 'status': 'mismatch',
                'reason': 'package_integrity_failed', 'integrity': integrity, 'requests': []}
    config = read_json(path / 'raw/resolved_config.json')
    events = [json.loads(line) for line in (path / 'trajectory.jsonl').read_text().splitlines()]

    def read(ref):
        target = confined(path, ref['path'])
        if not target.is_file() or target.is_symlink() or file_digest(target) != ref['sha256']:
            raise ValueError('referenced evidence bytes differ')
        return target.read_bytes()

    parent_path = path / 'lineage/parent_checkpoint.json'
    parent_cp = read_json(parent_path) if parent_path.exists() else None
    rows = reconstruct_requests(config, public, events, read, parent_checkpoint=parent_cp)
    for row in rows:
        row['assurance'] = 'exact_saved_input_consistency_with_current_code; historical_source_not_attested'
    graph = runtime_graph(events, read_json(path / 'runtime/provenance.json'),
                          episode_id=read_json(path / 'manifest.json')['episode_id'])
    cp_path = path / 'checkpoints/cp000000.json'
    if cp_path.exists():
        cp = read_json(cp_path)
        snap = read_json(path / ('states/snapshots/' + cp['snapshot_id'] + '.json'))
        actual = {f['path']: f['sha256'] for f in snap['files']}
        expected = {'input/' + a['name']: a['sha256'] for a in public['assets']}
        graph['initial_workspace_public_assets_only'] = compare(expected, actual)
    else:
        graph['initial_workspace_public_assets_only'] = {'status': 'unverified', 'reason': 'initial_checkpoint_missing'}
    mismatch = any(x['status'] == 'mismatch' for x in [*rows, *graph.values()])
    return {'package': str(path), 'task_id': public['task_id'], 'status': 'mismatch' if mismatch else 'unverified',
            'index_sha256': file_digest(path / 'evidence_index.json'), 'integrity': integrity,
            'requests': rows, 'source_graph': graph,
            'historical_controller_sha256': None,
            'reason': 'historical_execution_trust_edges_missing' if not mismatch else 'constructive_evidence_mismatch',
            'keyword_overlap_policy': 'not used as a leak verdict; model output and public images may independently contain rubric constants'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=Path, default=DEFAULT_EPISODES)
    parser.add_argument('--package', type=Path, action='append', help='Explicit DEV packages, including later plain G2 branches')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = args.package or [p for p in sorted(args.episodes.iterdir())
                             if p.name.split('.r', 1)[0] in DEV and (p / 'manifest.json').is_file()]
    rows = []
    skipped = []
    for path in paths:
        try:
            # Classification reads saved execution/grade evidence only; it does
            # not invoke the official scorer or inspect a new task family.
            if not args.package and not study_outcome(path).get('study_valid'):
                skipped.append({'package': str(path), 'reason': 'not_study_valid'})
                continue
            rows.append(audit_package(path))
        except (L1Error, OSError, ValueError, KeyError, TypeError) as exc:
            rows.append({'package': str(path), 'status': 'unverified', 'reason': str(exc), 'requests': []})
    report = {'schema': 'g4_constructive_provenance_v1', 'actor_requests': 0, 'scorer_calls': 0,
              'current_source_identity': source_identity(), 'historical_authenticity': 'not_attested',
              'gate_pass': bool(rows) and all(x['status'] == 'verified' for x in rows),
              'packages': rows, 'skipped': skipped}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
    print(json.dumps({'packages': len(rows), 'gate_pass': report['gate_pass'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
