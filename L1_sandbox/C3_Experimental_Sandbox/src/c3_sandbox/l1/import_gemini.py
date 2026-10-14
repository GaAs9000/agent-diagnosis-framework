"""Loss-preserving migration of archived Gemini episodes, without running a model.

This is a source-evidence projection, not certification of past execution.
Absolute paths are retained only inside original raw files. Normalized references
are always package-relative. Missing snapshots never become synthetic checkpoints.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from . import SCHEMA_VERSION
from .recorder import EventJournal
from .schema import validate
from .storage import (canonical, confined, copy_file, digest, file_digest, read_json,
                      ref, safe_relative, utc_now, write_bytes, write_json)


def check(name: str, status: str, detail: str, refs: list | None = None) -> dict:
    return {'name': name, 'status': status, 'detail': detail, 'source_refs': refs or []}


def validity(status: str, scope: str, reasons: list[str]) -> dict:
    return {'status': status, 'scope': scope, 'reason_codes': reasons}


def _parse(raw: str | None) -> dict | None:
    try:
        value = json.loads(raw) if raw else None
        return value if isinstance(value, dict) else None
    except (ValueError, TypeError):
        return None


def _parts(response: dict | None) -> list:
    try:
        parts = response['candidates'][0]['content']['parts']
        return parts if isinstance(parts, list) else []
    except (TypeError, KeyError, IndexError):
        return []


def _tree(source: Path, output: Path, prefix: str) -> list[dict]:
    files = []
    if not source.exists():
        return files
    if source.is_symlink() or not source.is_dir():
        raise ValueError('unsafe source directory')
    for path in sorted(source.rglob('*')):
        if path.is_symlink():
            raise ValueError('symlink source entry')
        if path.is_file():
            files.append(copy_file(path, output, prefix + '/' + path.relative_to(source).as_posix()))
        elif not path.is_dir():
            raise ValueError('special source entry')
    return files


def _inventory(files: list[dict], role: str, prefix: str, episode_id: str) -> list[dict]:
    return [{'object_id': 'file:' + digest([episode_id, f['path'][len(prefix)+1:]])[:24],
             'version_id': 'sha256:' + f['sha256'], 'logical_path': f['path'][len(prefix)+1:],
             'role': role, 'file': f, 'original_mode': None} for f in files]


def _grade_phase(result: dict, phase: str, source: Path, out: Path) -> dict:
    original = result.get(phase.replace('-', '_'), {})
    source_pointer = '/' + phase.replace('-', '_')
    source_ref = ref('raw/result.json', source_pointer)
    checks = []
    grade_path = source / phase / 'grade.json'
    if grade_path.is_file():
        equal = read_json(grade_path) == original
        checks.append(check('archived_grade_matches_result', 'pass' if equal else 'fail',
                            'Comparison of retained grade.json and result.json; no grader rerun.',
                            [source_ref, ref('raw/' + phase + '/grade.json', '')]))
    else:
        checks.append(check('archived_grade_matches_result', 'unknown', 'Separate grade.json absent.', [source_ref]))
    for filename, key in [('process.json', 'process_sha256'), ('stdout.txt', 'stdout_sha256'), ('stderr.txt', 'stderr_sha256')]:
        path = source / phase / filename
        expected = original.get(key)
        status = 'unknown' if not path.is_file() or not expected else ('pass' if file_digest(path) == expected else 'fail')
        checks.append(check(filename + '_source_digest', status, 'Verify archived bytes against original recorded digest.',
                            [source_ref] + ([ref('raw/' + phase + '/' + filename)] if path.is_file() else [])))
    artifact = original.get('artifact')
    if isinstance(artifact, dict) and artifact.get('name'):
        root = source / ('public-input' if phase == 'initial-grade' else 'final-workspace')
        candidate = confined(root, artifact['name'])
        status = 'unknown' if not candidate.is_file() else ('pass' if file_digest(candidate) == artifact.get('sha256') else 'fail')
        checks.append(check('graded_artifact_digest', status, 'Bind grader-reported artifact hash to retained workspace bytes.', [source_ref]))
    elif original.get('reason') == 'missing_deliverable':
        checks.append(check('graded_artifact_digest', 'not_applicable', 'Grader records no designated deliverable.', [source_ref]))
    else:
        checks.append(check('graded_artifact_digest', 'unknown', 'No graded artifact identity.', [source_ref]))
    return {'source_ref': source_ref, 'official_raw': original.get('official_raw'),
            'normalization': {key: original.get(key) for key in ('validity', 'c3_operational_completed', 'outcome', 'reason', 'coverage_complete', 'unassessed_rules')},
            'checks': checks}


def import_episode(source: str | Path, output: str | Path, *, bundle: str | Path | None = None,
                   episode_id: str | None = None) -> dict:
    source, out = Path(source).resolve(), Path(output).absolute()
    result = read_json(source / 'result.json')
    if not isinstance(result, dict) or not isinstance(result.get('task_id'), str):
        raise ValueError('legacy result must identify its task')
    task_id = result['task_id']
    episode_id = episode_id or (source.parent.parent.name + '.' + task_id)
    # Validate ID before creating any paths.
    import re
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', episode_id):
        raise ValueError('invalid episode ID')
    if out.resolve().is_relative_to(source):
        raise ValueError('output must not be inside source evidence')
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    copy_file(source / 'result.json', out, 'raw/result.json')
    try:
        return _import(result, source, out, Path(bundle).resolve() if bundle else None, episode_id)
    except BaseException as exc:
        # Preserve already-written raw evidence and journal even on interruption.
        write_json(out, 'export_error.json', {'error_type': type(exc).__name__, 'message': str(exc),
                                             'partial_export_preserved': True, 'at': utc_now()})
        raise


def _import(result: dict, source: Path, out: Path, bundle: Path | None, episode_id: str) -> dict:
    base = {'schema_version': SCHEMA_VERSION, 'episode_id': episode_id}
    task_id = result['task_id']
    checks, coverage = [], []
    def cover(field: str, status: str, reason: str, count: int | None = None, refs: list | None = None):
        coverage.append({'field': field, 'status': status, 'count': count, 'reason': reason, 'source_refs': refs or []})

    for phase in ('initial-grade', 'final-grade'):
        _tree(source / phase, out, 'raw/' + phase)
    for folder in ('actor-runtime', 'tool-dispatch', 'checkpoints'):
        _tree(source / folder, out, 'raw/legacy-' + folder)
    inputs = _tree(source / 'public-input', out, 'artifacts/initial')
    finals = _tree(source / 'final-workspace', out, 'artifacts/final')
    artifacts = _inventory(inputs, 'initial', 'artifacts/initial', episode_id) + _inventory(finals, 'final', 'artifacts/final', episode_id)
    artifact_inventory = {**base, 'files': artifacts, 'scope': 'retained file bytes only; original directory metadata and modes unknown'}
    validate('artifact_inventory', artifact_inventory)
    write_json(out, 'artifacts/inventory.json', artifact_inventory)
    blobs = {a['file']['sha256']: ref(a['file']['path']) for a in artifacts}
    initial_by_path = {a['logical_path']: a for a in artifacts if a['role'] == 'initial'}
    final_by_path = {a['logical_path']: a for a in artifacts if a['role'] == 'final'}

    metadata, bm = {}, {}
    bundle_ref = task_ref = rubric_ref = verifier_ref = None
    if bundle is not None:
        bm = read_json(bundle / 'manifest.json')
        metadata = bm.get('tasks', {}).get(task_id, {})
        metadata = metadata.get('metadata', metadata)
        copy_file(bundle / 'manifest.json', out, 'benchmark/manifest.json')
        write_json(out, 'benchmark/task_metadata.json', metadata)
        bundle_ref, task_ref = ref('benchmark/manifest.json', ''), ref('benchmark/task_metadata.json', '')
        for key, target in [('task_path', 'benchmark/task.json'), ('rubric_path', 'benchmark/rubric.json'), ('verifier_path', 'benchmark/verifier.py')]:
            relative = metadata.get(key)
            if relative:
                path = confined(bundle, relative)
                if path.is_file():
                    copy_file(path, out, target)
                    if key == 'rubric_path': rubric_ref = ref(target, '')
                    if key == 'verifier_path': verifier_ref = ref(target)
        for name in ('LICENSE', 'NOTICE', 'requirements.txt', 'pyproject.toml'):
            path = bundle / 'source_metadata' / name
            if path.is_file(): copy_file(path, out, 'benchmark/source_metadata/' + name)
        for item in metadata.get('input_files', []):
            observed = initial_by_path.get(item['name'])
            status = 'unknown' if observed is None else ('pass' if observed['file']['sha256'] == item.get('sha256') else 'fail')
            checks.append(check('public_asset:' + item['name'], status, 'Retained input hash versus frozen benchmark manifest.', [task_ref]))
    cover('benchmark_definition', 'available' if metadata and rubric_ref and verifier_ref else 'partial',
          'Benchmark metadata and verifier source retained; Docker image availability is not established.', refs=[x for x in (bundle_ref, rubric_ref, verifier_ref) if x])
    lock_ref = None
    lock_path = source.parent.parent / 'config.lock'
    if lock_path.is_file():
        copy_file(lock_path, out, 'raw/config.lock')
        lock_ref = ref('raw/config.lock', '')
    cover('resolved_configuration', 'missing', 'config.lock records an identity, not the full resolved run configuration; not reconstructed from current configs.', refs=[lock_ref] if lock_ref else [])

    journal = EventJournal(out, episode_id)
    journal.append('episode.started', {'source_kind': 'legacy_gemini_result', 'is_synthetic_source_claim': result.get('is_synthetic'),
                                     'historical_authenticity': 'not_independently_attested'}, source_refs=[ref('raw/result.json', '')])
    records = result.get('raw_gemini_records') or []
    dispatches = result.get('tool_dispatch') or []
    requests, responses, native_refs = [], [], []
    for i, record in enumerate(records):
        for side in ('request', 'response'):
            value = record.get(side + '_raw')
            if value is not None and not isinstance(value, str):
                raise ValueError('wire record must preserve a string or null')
            raw = (value or '').encode('utf-8')
            path = f'raw/model/{i:06d}.{side}.json'
            write_bytes(out, path, raw)
            expected = record.get(side + '_identity')
            checks.append(check(f'wire:{i}:{side}:source_digest', 'unknown' if expected is None else ('pass' if hashlib.sha256(raw).hexdigest() == expected else 'fail'),
                                'Hash of UTF-8 source string against the original recorded identity.', [ref(path), ref('raw/result.json', f'/raw_gemini_records/{i}')]))
            (requests if side == 'request' else responses).append(_parse(value))
        native_refs.append(ref(f'raw/model/{i:06d}.request.json'))
    public_prompt_ref = None
    if requests and requests[0] is not None:
        public_prompt_ref = ref('raw/model/000000.request.json', '')
        journal.append('task.presented', {'native_request_ref': public_prompt_ref, 'input_inventory_ref': ref('artifacts/inventory.json', '')}, source_refs=[public_prompt_ref])
    cover('native_model_wire', 'available' if records else 'missing', 'Both sides of each archived exchange retained verbatim as UTF-8 strings, with original digest checks.', len(records), native_refs)
    cover('model_timestamps_and_latency', 'missing', 'Legacy Gemini records have no per-request clock or generation latency; import time is not execution time.')
    cover('model_cost_usd', 'unknown', 'Provider token usage retained; no historical tariff or billed cost evidence. Cost is not set to zero.')
    cover('provider_revision', 'partial', 'Provider modelVersion strings are retained, not treated as immutable model-weight digests.')

    consumed: set[int] = set()
    tool_event_ids, state_specs = {}, []
    delivered_count = 0
    tool_call_count = 0
    exact_bindings = 0
    cursor = 0
    for i, record in enumerate(records):
        model_call_id = f'{episode_id}:m{i:06d}'
        req_ref, res_ref = native_refs[i], ref(f'raw/model/{i:06d}.response.json')
        req_event = journal.append('model.request', {'model_call_id': model_call_id, 'request_ref': req_ref, 'source_identity_sha256': record.get('request_identity')},
                                   links={'model_call_id': model_call_id}, source_refs=[req_ref])
        res_event = journal.append('model.response' if record.get('response_raw') else 'model.error', {'model_call_id': model_call_id, 'response_ref': res_ref,
                                    'source_identity_sha256': record.get('response_identity'), 'usage': record.get('usage'),
                                    'selector_invalid': record.get('selector_invalid'), 'error': record.get('error'),
                                    'response_received': bool(record.get('response_raw'))},
                                   links={'model_call_id': model_call_id, 'request_event_id': req_event['event_id']}, source_refs=[res_ref])
        if record.get('response_raw') and (record.get('error') or responses[i] is None):
            journal.append('model.error', {'model_call_id': model_call_id, 'source_error': record.get('error'), 'response_parseable': responses[i] is not None},
                           links={'model_call_id': model_call_id, 'response_event_id': res_event['event_id']}, source_refs=[ref('raw/result.json', f'/raw_gemini_records/{i}')])
        parts = _parts(responses[i])
        function_calls = [(part_index, part['functionCall']) for part_index, part in enumerate(parts) if isinstance(part, dict) and isinstance(part.get('functionCall'), dict)]
        continuation = None
        if function_calls and i + 1 < len(records) and isinstance(requests[i], dict) and isinstance(requests[i+1], dict):
            prior = requests[i].get('contents', [])
            nxt = requests[i+1].get('contents', [])
            prefix_ok = (isinstance(prior, list) and isinstance(nxt, list) and len(nxt) > len(prior)+1 and
                         nxt[:len(prior)] == prior and nxt[len(prior)].get('parts') == parts)
            checks.append(check(f'continuation:{i}', 'pass' if prefix_ok else 'fail', 'Next request must preserve prior messages and exact native model parts.', [req_ref, native_refs[i+1]]))
            if prefix_ok:
                candidate = nxt[len(prior)+1].get('parts', [])
                if isinstance(candidate, list): continuation = (len(prior)+1, candidate)
        for call_index, (part_index, fc) in enumerate(function_calls):
            call_id = f'{episode_id}:t{tool_call_count:06d}'
            tool_call_count += 1
            dispatch_index = None
            if cursor < len(dispatches):
                candidate = dispatches[cursor]
                if candidate.get('name') == fc.get('name') and candidate.get('arguments') == fc.get('args', {}):
                    dispatch_index = cursor
                    consumed.add(cursor)
                    cursor += 1
                    exact_bindings += 1
            fc_ref = ref(res_ref['path'], f'/candidates/0/content/parts/{part_index}/functionCall')
            call_event = journal.append('tool.call', {'tool_call_id': call_id, 'name': fc.get('name'), 'arguments': fc.get('args', {}),
                            'native_tool_call_id': fc.get('id'), 'source_dispatch_index': dispatch_index,
                            'source_dispatch_id': dispatches[dispatch_index].get('tool_call_id') if dispatch_index is not None else None,
                            'linkage': 'sequence_and_exact_name_arguments' if dispatch_index is not None else 'unmatched_native_call'},
                            links={'model_call_id': model_call_id, 'response_event_id': res_event['event_id'], 'tool_call_id': call_id}, source_refs=[fc_ref])
            if dispatch_index is not None:
                d = dispatches[dispatch_index]
                observation_ref = ref('raw/result.json', f'/tool_dispatch/{dispatch_index}/raw_result')
                tool_event = journal.append('tool.result', {'tool_call_id': call_id, 'source_dispatch_index': dispatch_index,
                                'raw_observation': d.get('raw_result'), 'raw_observation_ref': observation_ref},
                                links={'model_call_id': model_call_id, 'tool_call_id': call_id, 'tool_call_event_id': call_event['event_id']}, source_refs=[observation_ref])
                tool_event_ids[dispatch_index] = tool_event['event_id']
            if continuation is not None and call_index < len(continuation[1]):
                response_part = continuation[1][call_index].get('functionResponse')
                if isinstance(response_part, dict) and response_part.get('name') == fc.get('name'):
                    observed_ref = ref(native_refs[i+1]['path'], f'/contents/{continuation[0]}/parts/{call_index}/functionResponse/response')
                    journal.append('observation.delivered', {'tool_call_id': call_id,
                                    'delivered_observation': response_part.get('response'), 'observation_ref': observed_ref},
                                    links={'tool_call_id': call_id, 'delivered_in_model_call_id': f'{episode_id}:m{i+1:06d}'}, source_refs=[observed_ref])
                    delivered_count += 1
    # An orphan is retained as observed, not silently discarded or paired by guessing.
    for i, d in enumerate(dispatches):
        if i in consumed: continue
        call_id = f'{episode_id}:orphan{i:06d}'
        call_event = journal.append('tool.call', {'tool_call_id': call_id, 'name': d.get('name'), 'arguments': d.get('arguments'),
                         'native_tool_call_id': None, 'source_dispatch_index': i, 'source_dispatch_id': d.get('tool_call_id'), 'linkage': 'unmatched_recorded_dispatch'},
                         links={'tool_call_id': call_id}, source_refs=[ref('raw/result.json', f'/tool_dispatch/{i}')])
        observation_ref = ref('raw/result.json', f'/tool_dispatch/{i}/raw_result')
        event = journal.append('tool.result', {'tool_call_id': call_id, 'source_dispatch_index': i, 'raw_observation': d.get('raw_result'), 'raw_observation_ref': observation_ref},
                               links={'tool_call_id': call_id, 'tool_call_event_id': call_event['event_id']}, source_refs=[observation_ref])
        tool_event_ids[i] = event['event_id']
        tool_call_count += 1
    checks.append(check('native_dispatch_linkage', 'pass' if exact_bindings == len(dispatches) == tool_call_count else 'fail',
                        'Legacy native call IDs differ from dispatcher IDs; binding uses serial order and exact name/arguments, not shared IDs.', [ref('raw/result.json', '/tool_dispatch')]))
    cover('raw_tool_observations', 'available' if dispatches else 'missing', 'Controller-recorded exit code/stdout/stderr/elapsed retained separately from model-visible payload.', len(dispatches))
    cover('model_visible_observations', 'available' if delivered_count == len(dispatches) else 'partial',
          'Only observations present in a subsequent native request are marked delivered. Last-turn results without a next request are not synthesized.', delivered_count)
    cover('call_id_linkage', 'partial', 'Different native and dispatcher IDs; exact content/ordering checks do not replace a shared live boundary ID.', exact_bindings)

    # Source state changes are not recomputed step snapshots: the portable archive
    # may contain hashes without the snapshot bytes needed to verify them.
    for i, d in enumerate(dispatches):
        for j, change in enumerate(d.get('fs_delta') or []):
            state_specs.append((change, tool_event_ids[i], 'recorded_step_delta',
                                [ref('raw/result.json', f'/tool_dispatch/{i}/fs_delta/{j}')]))
    state_event = journal.append('state.observed', {'scope': 'initial_to_final_retained_file_bytes', 'artifact_inventory_ref': ref('artifacts/inventory.json', '')}, assurance='recomputed', source_refs=[ref('artifacts/inventory.json', '')])
    for path in sorted(initial_by_path.keys() | final_by_path.keys()):
        before, after = initial_by_path.get(path), final_by_path.get(path)
        if before and after and before['version_id'] == after['version_id']: continue
        def entry(a):
            return {'kind': 'file', 'sha256': a['file']['sha256'], 'size_bytes': a['file']['size_bytes']} if a else None
        state_specs.append(({'path': path, 'before': entry(before), 'after': entry(after)}, state_event['event_id'], 'recomputed_endpoint_delta', [ref(a['file']['path']) for a in (before, after) if a]))
    changes = []
    for i, (c, event_id, kind, refs) in enumerate(state_specs):
        path = c['path']
        if path != '.': safe_relative(path)
        before, after = c.get('before'), c.get('after')
        scope = (after or before or {}).get('kind', 'unknown')
        row = {**base, 'change_id': f'{episode_id}:s{i:06d}', 'event_id': event_id,
               'object_id': 'file:' + digest([episode_id, path])[:24], 'path': path, 'scope': scope if scope in ('file','directory') else 'unknown',
               'observation_type': kind, 'before': before, 'after': after,
               'before_blob_ref': blobs.get((before or {}).get('sha256')), 'after_blob_ref': blobs.get((after or {}).get('sha256')),
               'verification': 'reported_without_retained_snapshots' if kind == 'recorded_step_delta' else 'recomputed_from_retained_bytes', 'source_refs': refs}
        validate('state_change', row)
        changes.append(row)
    write_bytes(out, 'state_changes.jsonl', b''.join(canonical(c) + b'\n' for c in changes))
    path_audit = []
    origin = str(result.get('output_dir', '')).rstrip('/')
    for i, dispatch in enumerate(dispatches):
        for key in ('before_path', 'after_path'):
            value = dispatch.get(key)
            relative = value[len(origin)+1:] if origin and isinstance(value, str) and value.startswith(origin + '/') else None
            present, archived_ref = False, None
            if relative:
                try:
                    candidate = confined(source, relative)
                    present = candidate.is_file()
                    if present and relative.startswith('tool-dispatch/'):
                        archived_ref = ref('raw/legacy-' + relative)
                except ValueError:
                    relative = None
            path_audit.append({'source_ref': ref('raw/result.json', f'/tool_dispatch/{i}/{key}'),
                               'original_path': value, 'source_relative_path': relative,
                               'retained_in_source': present, 'archived_ref': archived_ref})
    write_json(out, 'raw/source_path_audit.json', {'entries': path_audit})
    retained = sum(p['retained_in_source'] for p in path_audit)
    cover('step_snapshot_bytes', 'partial' if retained else 'missing',
          f'{len(path_audit)} referenced before/after paths; {retained} retained. Step deltas are reported, not independently reproduced. No snapshot qualification is performed.',
          retained, [ref('raw/source_path_audit.json', '')])
    cover('endpoint_file_bytes', 'available' if inputs and finals else 'partial', 'Initial and final retained bytes hashed in this import; endpoint differences do not locate the exact writing step.', len(artifacts))
    cover('object_level_state', 'missing', 'No L1 object-state extractor ran. Object changes or causal explanations are not fabricated.')

    phases = {p: _grade_phase(result, p + '-grade', source, out) for p in ('initial','final')}
    checks.extend(phases['initial']['checks'] + phases['final']['checks'])
    final_grade = result.get('final_grade') or {}
    evaluation_status = ('invalid' if final_grade.get('validity') in ('invalid', 'infrastructure_invalid') or any(c['status']=='fail' for c in phases['final']['checks'])
                         else 'valid' if final_grade.get('validity') == 'valid' and all(c['status'] in ('pass','not_applicable') for c in phases['final']['checks']) else 'unknown')
    evaluation_validity = validity(evaluation_status, 'archived_grader_result_and_retained_artifact_consistency; not a fresh grader run',
                                  [] if evaluation_status == 'valid' else ['archived_evaluation_invalid_or_unverified'])
    grade = {**base, 'evaluation_mode': 'imported_not_rerun', 'grader_image': final_grade.get('grader_image'), **phases, 'evaluation_validity': evaluation_validity}
    validate('official_grade', grade)
    write_json(out, 'official_grade.json', grade)
    journal.append('evaluation.completed', {'evaluation_mode': 'imported_not_rerun', 'official_grade_ref': ref('official_grade.json', '')}, source_refs=[ref('official_grade.json', '')])
    cover('official_evaluation', 'available' if evaluation_status == 'valid' else 'partial', 'Archived evaluator output, rule details, validity, coverage, stdout, stderr and process identity retained.', refs=[ref('official_grade.json', '')])

    cps = result.get('checkpoints') or []
    checkpoint_index = {**base, 'cold_resume_qualified': False,
                        'items': [{'checkpoint_id': str(cp.get('checkpoint_id', f'legacy-{i}')), 'qualification': 'unqualified',
                                   'source_ref': ref('raw/result.json', f'/checkpoints/{i}'),
                                   'reason_codes': ['legacy_checkpoint_not_qualified_by_l1']} for i, cp in enumerate(cps)],
                        'reason_codes': ['no_retained_qualified_checkpoint']}
    validate('checkpoints', checkpoint_index)
    write_json(out, 'checkpoints/index.json', checkpoint_index)
    cover('checkpoints', 'missing' if not cps else 'partial', 'No checkpoint is qualified by this offline importer. File presence alone never qualifies resume.', len(cps))
    actor_file = out / 'raw/legacy-actor-runtime/runtime_provenance.json'
    actor_ref = ref('raw/legacy-actor-runtime/runtime_provenance.json', '') if actor_file.is_file() else None
    cover('agent_runtime_provenance', 'partial' if actor_ref else 'missing', 'Agent-side runtime attestation absent or not independently verified; grader provenance is not substituted.', refs=[actor_ref] if actor_ref else [])
    versions = sorted({str(r['modelVersion']) for r in responses if r and r.get('modelVersion')})
    runtime = result.get('runtime') or {}
    gen_configs = [(r or {}).get('generationConfig') or {} for r in requests]
    seeds = [c.get('seed') for c in gen_configs]
    seed_present = any('seed' in c for c in gen_configs) if records else None
    cover('seed_application', 'available' if seed_present else 'not_applicable', 'Declared seed retained, but only native generationConfig establishes whether it was transmitted.', refs=native_refs)
    agent_result = result.get('agent_result') or {}
    stop = agent_result.get('stopped_reason')
    termination = {'max_turns':'budget_exhausted', 'assistant_finished':'assistant_finished', 'protocol_error':'protocol_error',
                   'tool_execution_error':'tool_execution_error', 'error':'provider_error', 'timeout':'interrupted'}.get(stop, 'unknown')
    recorded_protocol = agent_result.get('protocol_status')
    protocol_status = ('invalid' if recorded_protocol == 'invalid' or any(c['status']=='fail' for c in checks if c['name'].startswith(('wire:', 'continuation:', 'native_dispatch_linkage')))
                       else 'valid' if recorded_protocol == 'valid' and records else 'unknown')
    execution_status = 'invalid' if protocol_status == 'invalid' else 'unknown'
    completion = final_grade.get('c3_operational_completed')
    # Preserve the old explicit boolean contract, rather than invent partial credit.
    outcome = 'unknown' if evaluation_status != 'valid' else 'success' if completion is True else 'failure' if completion is False else 'unknown'
    journal.append('episode.finished', {'termination': termination, 'source_stopped_reason': stop, 'task_outcome': outcome,
                                       'execution_validity': execution_status, 'cold_resume_qualified': False}, source_refs=[ref('raw/result.json', '')])
    manifest = {**base, 'recording_mode': 'synthetic_fixture' if result.get('is_synthetic') is True else 'legacy_import',
        'imported_at': utc_now(), 'source_run_id': source.parent.parent.name,
        'task': {'benchmark':'OfficeVal', 'task_id':task_id, 'source_revision':bm.get('source_commit'), 'asset_dataset_revision':bm.get('asset_dataset_revision'),
                 'expected_output':metadata.get('expected_output'), 'bundle_manifest_ref':bundle_ref,'task_metadata_ref':task_ref,
                 'public_prompt_ref':public_prompt_ref,'rubric_ref':rubric_ref,'verifier_ref':verifier_ref},
        'agent':{'backend':'api','provider':result.get('provider'),'requested_model':result.get('model'), 'declared_fingerprint':runtime.get('model_fingerprint'),
                 'declared_fingerprint_verified':False,'response_model_versions':versions,'raw_runtime_ref':ref('raw/result.json','/runtime')},
        'configuration':{'declared_seed':result.get('seed'),'seed_in_native_requests':seed_present,'native_request_refs':native_refs,
                         'source_config_lock_ref':lock_ref,'resolved_config_verified':False,'declared_max_turns':(runtime.get('sampling') or {}).get('max_turns'), 'source_code_binding':'unverified'},
        'environment':{'agent_runtime_provenance_ref':actor_ref,'agent_image_digest':None,'grader_image_digest':final_grade.get('grader_image'),'isolation_verified':False},
        'status':{'termination':termination,'source_stopped_reason':stop,
                  'protocol_validity':validity(protocol_status,'archived native wire and dispatch consistency', [] if protocol_status=='valid' else ['protocol_not_verified']),
                  'execution_validity':validity(execution_status,'L1 complete execution boundary', ['recorded_protocol_invalid'] if execution_status == 'invalid' else ['historical_isolation_and_full_configuration_not_verified']),
                  'evaluation_validity':evaluation_validity,
                  'task_outcome':{'status':outcome,'policy':'preserved_legacy_c3_operational_completed; not new official success semantics', 'source_ref':ref('raw/result.json','/final_grade')},
                  'evidence_completeness':'incomplete'},
        'capabilities':{'offline_evidence_read':True,'tool_reexecution_qualified':False,'cold_resume_qualified':False,'deterministic_model_replay':False},
        'counts':{'model_requests':len(records),'model_responses':sum(bool(r.get('response_raw')) for r in records),'tool_calls':tool_call_count,'tool_results':len(dispatches),
                  'delivered_observations':delivered_count,'state_changes':len(changes),'checkpoints':len(cps),'events':len(journal.events)},
        'coverage':coverage,'checks':checks,
        'references':{k:ref(v) for k,v in {'raw_result':'raw/result.json','trajectory':'trajectory.jsonl','state_changes':'state_changes.jsonl',
                                         'artifacts':'artifacts/inventory.json','official_grade':'official_grade.json','checkpoints':'checkpoints/index.json'}.items()},
        'limitations':['Import is not a new agent run or grader execution.',
                       'Source is_synthetic=false is a source claim, not third-party proof of execution.',
                       'Hash chains and package digests detect alteration relative to an external trusted digest; they do not prove historical authenticity.',
                       'Event ordering is a normalized serial projection; absent historical timestamps remain null.',
                       'Source absolute paths remain verbatim in raw records; all normalized references are package-relative.',
                       'The package contains private task/model/tool content; no public-release redaction has been performed.',
                       'No object-level interpretation, capability inference, or diagnosis is produced.']}
    validate('manifest', manifest)
    write_json(out, 'manifest.json', manifest)
    from .package import seal_package, verify_package
    index_digest = seal_package(out, journal.events)
    report = verify_package(out, expected_index_sha256=index_digest)
    return {'episode_id':episode_id,'output':str(out),'index_sha256':index_digest,'counts':manifest['counts'],
            'status':manifest['status'],'verification':report}
