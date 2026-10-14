"""Offline schema, byte-integrity, reference, chain and projection verification.

No imports from integrated, gateway, r1 agents, evaluators, or analyst modules.
A package may verify successfully AND be evidence-incomplete. These are separate.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import SCHEMA_VERSION
from .recorder import scan_journal
from .schema import validate
from .storage import (canonical, confined, digest, file_digest, read_json, regular_file,
                      resolve_ref, write_json)


def _files(root: Path) -> list[dict]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError('evidence root must be a real directory')
    records = []
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('symlink in evidence package')
        if path.is_dir(): continue
        regular_file(path)
        relative = path.relative_to(root).as_posix()
        if relative == 'evidence_index.json': continue
        records.append({'path': relative, 'sha256': file_digest(path), 'size_bytes': path.stat().st_size})
    return records


def seal_package(root: Path, events: list[dict]) -> str:
    manifest = read_json(root / 'manifest.json')
    index = {'schema_version': SCHEMA_VERSION, 'episode_id': manifest['episode_id'],
             'hash_algorithm': 'sha256', 'files': _files(root),
             'trajectory_head_sha256': events[-1]['event_sha256'] if events else None,
             'event_count': len(events), 'excludes': ['evidence_index.json'],
             'trust_scope': 'self-consistency only unless index digest is anchored outside this package; no historical authenticity attestation'}
    validate('evidence_index', index)
    write_json(root, 'evidence_index.json', index)
    return file_digest(root / 'evidence_index.json')


def _references(value: Any):
    if isinstance(value, dict):
        if set(value) == {'path', 'json_pointer'}:
            yield value
        else:
            for item in value.values(): yield from _references(item)
    elif isinstance(value, list):
        for item in value: yield from _references(item)


def verify_package(root: str | Path, *, expected_index_sha256: str | None = None) -> dict:
    root = Path(root)
    # Versioned live dispatch; historical imports retain the original P1 rules.
    if (root / 'manifest.json').is_file():
        try:
            if read_json(root / 'manifest.json').get('schema_version') == '0.2.0':
                from .evidence.live import verify
                return verify(root, expected_index_sha256=expected_index_sha256)
        except (OSError, ValueError):
            pass
    errors: list[dict] = []
    warnings: list[dict] = []
    manifest = None
    def error(code: str, detail: str):
        errors.append({'code': code, 'detail': detail})
    try:
        if root.is_symlink() or not root.is_dir():
            raise ValueError('evidence root must be a real directory')
        index = read_json(root / 'evidence_index.json')
        validate('evidence_index', index)
        index_hash = file_digest(root / 'evidence_index.json')
        if expected_index_sha256 and index_hash != expected_index_sha256:
            error('external_index_digest_mismatch', 'evidence_index.json differs from the supplied trusted digest')
        actual = _files(root)
        if actual != index['files']:
            error('file_inventory_or_digest_mismatch', 'Added, missing, altered, duplicated, or reordered file inventory entries')
        manifest = read_json(root / 'manifest.json')
        validate('manifest', manifest)
        if manifest['episode_id'] != index['episode_id']:
            error('episode_mismatch', 'Manifest and index disagree')
        components = {
            'official_grade': read_json(root / 'official_grade.json'),
            'checkpoints': read_json(root / 'checkpoints/index.json'),
            'artifact_inventory': read_json(root / 'artifacts/inventory.json'),
        }
        for kind, value in components.items():
            validate(kind, value)
            if value['episode_id'] != manifest['episode_id']:
                error('episode_mismatch', kind)
        scan = scan_journal(root / 'trajectory.jsonl')
        events = scan['events']
        if not scan['complete']:
            error('invalid_trajectory_journal', str(scan['errors']))
        states = []
        for line in (root / 'state_changes.jsonl').read_bytes().splitlines(keepends=True):
            if not line.endswith(b'\n'): raise ValueError('unterminated state_changes line')
            state = json.loads(line)
            validate('state_change', state)
            states.append(state)
        head = events[-1]['event_sha256'] if events else None
        if len(events) != index['event_count'] or head != index['trajectory_head_sha256']:
            error('trajectory_index_mismatch', 'Event count/head differs from index')
        normalized = [manifest, *components.values(), *events, *states]
        for record in normalized:
            for reference in _references(record):
                try: resolve_ref(root, reference)
                except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
                    error('unresolvable_reference', f'{reference}: {exc}')
        event_map = {event['event_id']: event for event in events}
        if len(event_map) != len(events): error('duplicate_event_id', 'Event IDs must be unique')
        source = read_json(root / 'raw/result.json')
        source_records, dispatches = source.get('raw_gemini_records') or [], source.get('tool_dispatch') or []
        model_requests, model_responses, calls, result_indices, deliveries = {}, {}, {}, [], set()
        for event in events:
            if event['episode_id'] != manifest['episode_id'] or event['event_id'] != f"{manifest['episode_id']}:e{event['seq']:06d}":
                error('event_identity_mismatch', event['event_id'])
            for key, event_id in event['links'].items():
                if key.endswith('_event_id'):
                    linked = event_map.get(event_id)
                    if linked is None or linked['seq'] >= event['seq']:
                        error('invalid_event_link', f"{event['event_id']} -> {event_id}")
            payload, kind = event['payload'], event['kind']
            if kind in ('model.request', 'model.response') or (kind == 'model.error' and 'response_ref' in payload):
                side = 'request' if kind == 'model.request' else 'response'
                call_id = payload['model_call_id']
                mapping = model_requests if side == 'request' else model_responses
                if call_id in mapping: error('duplicate_model_call', call_id)
                mapping[call_id] = event
                try:
                    i = int(call_id.rsplit(':m', 1)[1])
                    original = source_records[i]
                    raw = resolve_ref(root, payload[side + '_ref'])
                    if raw != (original.get(side + '_raw') or '').encode('utf-8'):
                        error('wire_projection_mismatch', call_id + ':' + side)
                    expected = original.get(side + '_identity')
                    if payload.get('source_identity_sha256') != expected:
                        error('wire_identity_projection_mismatch', call_id)
                    if side == 'response' and (payload.get('response_received') != bool(original.get('response_raw')) or payload.get('usage') != original.get('usage')):
                        error('usage_projection_mismatch', call_id)
                except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
                    error('invalid_model_projection', str(exc))
            elif kind == 'tool.call':
                call_id = payload['tool_call_id']
                if call_id in calls: error('duplicate_tool_call', call_id)
                calls[call_id] = event
                idx = payload['source_dispatch_index']
                if idx is not None:
                    d = dispatches[idx]
                    if payload.get('name') != d.get('name') or payload.get('arguments') != d.get('arguments') or payload.get('source_dispatch_id') != d.get('tool_call_id'):
                        error('tool_call_projection_mismatch', call_id)
                if payload['linkage'] != 'unmatched_recorded_dispatch':
                    fc = resolve_ref(root, event['source_refs'][0])
                    if fc.get('name') != payload.get('name') or fc.get('args', {}) != payload.get('arguments') or fc.get('id') != payload.get('native_tool_call_id'):
                        error('native_call_projection_mismatch', call_id)
            elif kind == 'tool.result':
                call_id = payload['tool_call_id']
                idx = payload['source_dispatch_index']
                result_indices.append(idx)
                if call_id not in calls: error('orphan_tool_result', call_id)
                elif calls[call_id]['payload']['source_dispatch_index'] != idx:
                    error('tool_result_link_mismatch', call_id)
                raw = resolve_ref(root, payload['raw_observation_ref'])
                if payload['raw_observation'] != raw or raw != dispatches[idx].get('raw_result'):
                    error('raw_observation_projection_mismatch', call_id)
            elif kind == 'observation.delivered':
                call_id = payload['tool_call_id']
                if call_id not in calls: error('unbound_delivered_observation', call_id)
                if call_id in deliveries: error('duplicate_delivered_observation', call_id)
                deliveries.add(call_id)
                if resolve_ref(root, payload['observation_ref']) != payload['delivered_observation']:
                    error('delivered_observation_projection_mismatch', call_id)
        if set(model_requests) != set(model_responses) or len(model_requests) != len(source_records):
            error('model_exchange_coverage_mismatch', 'Every archived model exchange must appear exactly once')
        if sorted(result_indices) != list(range(len(dispatches))):
            error('tool_dispatch_coverage_mismatch', 'Every recorded tool execution must appear exactly once')
        for call_id in model_responses:
            if call_id in model_requests and model_responses[call_id]['links'].get('request_event_id') != model_requests[call_id]['event_id']:
                error('request_response_pair_mismatch', call_id)
        for event in events:
            if event['kind'] == 'observation.delivered' and event['links'].get('delivered_in_model_call_id') not in model_requests:
                error('missing_delivery_request', event['event_id'])
        state_ids = set()
        for state in states:
            if state['change_id'] in state_ids: error('duplicate_state_id', state['change_id'])
            state_ids.add(state['change_id'])
            if state['episode_id'] != manifest['episode_id'] or state['event_id'] not in event_map:
                error('unbound_state_change', state['change_id'])
            if state['observation_type'] == 'recorded_step_delta':
                original = resolve_ref(root, state['source_refs'][0])
                if any(original.get(k) != state[k] for k in ('path', 'before', 'after')):
                    error('state_projection_mismatch', state['change_id'])
            for side in ('before', 'after'):
                reference = state[side + '_blob_ref']
                if reference:
                    actual_hash = file_digest(confined(root, reference['path']))
                    if actual_hash != (state[side] or {}).get('sha256'):
                        error('state_blob_digest_mismatch', state['change_id'])
        for artifact in components['artifact_inventory']['files']:
            path = confined(root, artifact['file']['path'])
            if file_digest(path) != artifact['file']['sha256'] or path.stat().st_size != artifact['file']['size_bytes'] or artifact['version_id'] != 'sha256:' + artifact['file']['sha256']:
                error('artifact_identity_mismatch', artifact['logical_path'])
        for phase in ('initial','final'):
            normalized = components['official_grade'][phase]
            original = source.get(phase + '_grade') or {}
            if normalized['official_raw'] != original.get('official_raw') or any(v != original.get(k) for k,v in normalized['normalization'].items()):
                error('official_grade_projection_mismatch', phase)
        from .projection_checks import check_projection
        errors.extend(check_projection(root, manifest, source, events, states, components['artifact_inventory']['files']))
        if manifest['task']['task_id'] != source.get('task_id') or manifest['agent']['requested_model'] != source.get('model'):
            error('source_identity_projection_mismatch', 'Task/model identity differs from raw result')
        expected_mode = 'synthetic_fixture' if source.get('is_synthetic') is True else 'legacy_import'
        if manifest['recording_mode'] != expected_mode:
            error('recording_mode_mismatch', 'An archived/fixture episode cannot be upgraded to live')
        if any(e['observed_at'] is not None or e['assurance'] == 'live_boundary' for e in events):
            error('fabricated_import_timing_or_assurance', 'This import profile has no live clock attestation')
        source_protocol = (source.get('agent_result') or {}).get('protocol_status')
        protocol_failed = source_protocol == 'invalid' or any(c['status'] == 'fail' for c in manifest['checks'] if c['name'].startswith(('wire:', 'continuation:', 'native_dispatch_linkage')))
        expected_protocol = 'invalid' if protocol_failed else 'valid' if source_protocol == 'valid' and source_records else 'unknown'
        expected_execution = 'invalid' if expected_protocol == 'invalid' else 'unknown'
        if manifest['status']['protocol_validity']['status'] != expected_protocol:
            error('protocol_validity_projection_mismatch', 'Protocol status disagrees with retained evidence')
        if manifest['status']['execution_validity']['status'] != expected_execution:
            error('unsupported_execution_qualification', 'An explicit protocol defect is invalid; otherwise the offline execution boundary remains unknown')
        original_final = source.get('final_grade') or {}
        grade_checks = []
        original_grade_path = root / 'raw/final-grade/grade.json'
        grade_checks.append(None if not original_grade_path.is_file() else read_json(original_grade_path) == original_final)
        for filename, key in [('process.json','process_sha256'),('stdout.txt','stdout_sha256'),('stderr.txt','stderr_sha256')]:
            path = root / 'raw/final-grade' / filename
            grade_checks.append(None if not path.is_file() or not original_final.get(key) else file_digest(path) == original_final[key])
        graded_artifact = original_final.get('artifact')
        if isinstance(graded_artifact, dict) and graded_artifact.get('name'):
            path = confined(root, 'artifacts/final/' + graded_artifact['name'])
            grade_checks.append(None if not path.is_file() else file_digest(path) == graded_artifact.get('sha256'))
        elif original_final.get('reason') != 'missing_deliverable':
            grade_checks.append(None)
        evaluation = ('invalid' if original_final.get('validity') in ('invalid','infrastructure_invalid') or False in grade_checks
                      else 'valid' if original_final.get('validity') == 'valid' and all(v is True for v in grade_checks) else 'unknown')
        if components['official_grade']['evaluation_validity']['status'] != evaluation or manifest['status']['evaluation_validity'] != components['official_grade']['evaluation_validity']:
            error('evaluation_validity_projection_mismatch', 'Evaluation validity differs from recomputed archive checks')
        completed = original_final.get('c3_operational_completed')
        outcome = 'unknown' if evaluation != 'valid' else 'success' if completed is True else 'failure' if completed is False else 'unknown'
        if manifest['status']['task_outcome']['status'] != outcome:
            error('task_outcome_projection_mismatch', 'Outcome differs from the retained completion policy')
        stop = (source.get('agent_result') or {}).get('stopped_reason')
        termination = {'max_turns':'budget_exhausted','assistant_finished':'assistant_finished','protocol_error':'protocol_error',
                       'tool_execution_error':'tool_execution_error','error':'provider_error','timeout':'interrupted'}.get(stop, 'unknown')
        if manifest['status']['termination'] != termination or manifest['status']['source_stopped_reason'] != stop:
            error('termination_projection_mismatch', 'Stop reason differs from source')
        kinds = [e['kind'] for e in events]
        counts = {'model_requests':kinds.count('model.request'), 'model_responses':kinds.count('model.response'),
                  'tool_calls':kinds.count('tool.call'),'tool_results':kinds.count('tool.result'),
                  'delivered_observations':kinds.count('observation.delivered'),'state_changes':len(states),
                  'checkpoints':len(components['checkpoints']['items']), 'events':len(events)}
        if counts != manifest['counts']: error('manifest_counts_mismatch', str(counts))
        for c in manifest['checks']:
            if c['status'] == 'fail': error('source_consistency_check_failed', c['name'])
        if manifest['status']['evidence_completeness'] == 'incomplete':
            warnings.append({'code':'historical_evidence_incomplete', 'detail':'Successful package verification does not imply complete execution evidence or resume qualification.'})
        if expected_index_sha256 is None:
            warnings.append({'code':'no_external_trust_anchor', 'detail':'The index digest was not compared with a separately trusted value.'})
        return {'verified':not errors,'episode_id':manifest['episode_id'],'index_sha256':index_hash,
                'expected_index_checked': expected_index_sha256 is not None,
                'errors':errors,'warnings':warnings,'counts':counts,
                'evidence_completeness':manifest['status']['evidence_completeness'],
                'execution_validity':manifest['status']['execution_validity']['status'],
                'cold_resume_qualified':manifest['capabilities']['cold_resume_qualified']}
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        error('invalid_package', str(exc))
        return {'verified':False,'errors':errors,'warnings':warnings}
