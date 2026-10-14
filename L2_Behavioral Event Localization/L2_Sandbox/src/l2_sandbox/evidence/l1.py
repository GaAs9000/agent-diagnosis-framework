"""Read-only L1 adapters, byte integrity and conservative semantic consistency.

The built-in v0.2 checks are explicitly named. They do not execute benchmark
verifiers, model code, plugins, Docker or commands from the episode. An installed
L1 verifier may additionally be required; it is not silently substituted.
"""
from __future__ import annotations
import copy
from pathlib import Path
from typing import Any
from ..io import (IntegrityError, Unsupported, bytes_read, canonical, digest, inventory,
                  loads, pointer, read_json, read_jsonl, relative, safe, sha_bytes, sha_file)


def refs(value):
    if isinstance(value, dict):
        if set(value) == {'path', 'sha256', 'size_bytes'}:
            yield value
        else:
            for item in value.values(): yield from refs(item)
    elif isinstance(value, list):
        for item in value: yield from refs(item)


def native_calls(body: dict, request_id: str) -> list[dict]:
    result = []
    if 'candidates' in body:
        candidates = body.get('candidates') or []
        parts = candidates[0].get('content', {}).get('parts', []) if candidates else []
        for i, part in enumerate(parts):
            if 'functionCall' not in part: continue
            call = part['functionCall']
            result.append({'request_id': request_id, 'call_id': f'{request_id}:c{i}',
                           'native_id': call.get('id'), 'name': call['name'], 'arguments': call.get('args', {})})
    elif 'choices' in body:
        choices = body.get('choices') or []
        message = choices[0].get('message', {}) if choices else {}
        for i, call in enumerate(message.get('tool_calls') or []):
            f = call['function']
            args = loads(f['arguments'])
            result.append({'request_id': request_id, 'call_id': f'{request_id}:c{i}',
                           'native_id': call.get('id'), 'name': f['name'], 'arguments': args})
    return result


def last_observation_turn(body: dict) -> list[dict]:
    if 'contents' in body: return body['contents'][-1:]
    messages = body.get('messages', [])
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get('role') == 'assistant': return messages[i + 1:]
    return messages


def observation_present(body: dict, observation: Any, native_id, name: str) -> bool:
    for item in last_observation_turn(body):
        for part in item.get('parts', []):
            f = part.get('functionResponse')
            if f and f.get('id') == native_id and f.get('name') == name and f.get('response') == observation:
                return True
        if item.get('role') == 'tool' and item.get('tool_call_id') == native_id:
            try:
                if loads(item['content']) == observation: return True
            except IntegrityError: pass
    return False


def diff_snapshots(before: dict, after: dict) -> list[dict]:
    result = []
    for field in ('files', 'directories'):
        b = {x['path']: x for x in before[field]}; a = {x['path']: x for x in after[field]}
        for p in sorted(set(a) | set(b)):
            if a.get(p) == b.get(p): continue
            change = ('directory_changed' if field == 'directories' else
                      'created' if p not in b else 'deleted' if p not in a else 'modified')
            result.append({'path': p, 'change': change, 'before': b.get(p), 'after': a.get(p)})
    return result


class L1Reader:
    def __init__(self, root: Path, *, expected_index: str | None = None, require_l1: bool = False):
        self.root = Path(root).absolute()
        safe(self.root, 'manifest.json')
        self.index = read_json(self.root / 'evidence_index.json')
        self.index_sha256 = sha_file(self.root / 'evidence_index.json')
        if expected_index and expected_index != self.index_sha256: raise IntegrityError('External source index SHA256 differs')
        self.manifest = read_json(self.root / 'manifest.json')
        self.version = self.manifest.get('schema_version')
        self.episode_id = self.manifest.get('episode_id')
        self.files = {}
        for row in self.index.get('files', []):
            if not isinstance(row, dict) or set(row) != {'path', 'sha256', 'size_bytes'}: raise IntegrityError('Bad source index entry')
            relative(row['path'])
            if row['path'] in self.files or row['path'] == 'evidence_index.json': raise IntegrityError('Duplicate/recursive source index')
            self.files[row['path']] = row
        self._cache: dict[str, Any] = {}
        self.events: list[dict] = []
        self.snapshots: dict[str, dict] = {}
        self.snapshot_paths: dict[str, str] = {}
        self.require_l1 = require_l1
        self.report = self._verify()
        if not self.report['verified']:
            raise IntegrityError('Source verification failed: ' + '; '.join(self.report['errors']))
        self.events_by_id = {e['event_id']: e for e in self.events}

    def read(self, path: str) -> bytes:
        if path == 'evidence_index.json':
            data = bytes_read(safe(self.root, path))
            if sha_bytes(data) != self.index_sha256: raise IntegrityError('Source index changed')
            return data
        if path not in self.files: raise IntegrityError('Unindexed source file: ' + path)
        data = bytes_read(safe(self.root, path))
        row = self.files[path]
        if len(data) != row['size_bytes'] or sha_bytes(data) != row['sha256']:
            raise IntegrityError('Source file content changed: ' + path)
        return data

    def json(self, path: str):
        # Revalidate bytes on every entry; parsed content is cached by pinned path.
        data = self.read(path)
        if path not in self._cache: self._cache[path] = loads(data)
        return copy.deepcopy(self._cache[path])

    def jsonl(self, path: str):
        raw = self.read(path)
        rows = []
        for line in raw.splitlines(keepends=True):
            if not line.endswith(b'\n') or not line.strip(): raise IntegrityError('Incomplete JSONL: ' + path)
            value = loads(line)
            if not isinstance(value, dict): raise IntegrityError('Invalid JSONL object')
            rows.append(value)
        return rows

    def _verify(self):
        checks = []; warnings = []; errors = []
        try:
            actual = inventory(self.root, {'evidence_index.json'})
            expected = sorted(self.files.values(), key=lambda x: x['path'])
            if actual != expected: raise IntegrityError('File set, size or digest differs from source index')
            checks.append('exact_file_inventory_sha256')
            if self.episode_id != self.index.get('episode_id'): raise IntegrityError('Episode identity differs')
            if self.version not in ('0.2.0', '0.1.0'): raise Unsupported('Unsupported L1 contract: ' + str(self.version))
            raw_events = self.jsonl('trajectory.jsonl')
            prev = None
            hash_key = 'sha256' if self.version == '0.2.0' else 'event_sha256'
            prev_key = 'previous_sha256' if self.version == '0.2.0' else 'prev_event_sha256'
            # P1 schema declares its previous-hash spelling; accept known explicit alternatives only.
            if self.version == '0.1.0' and raw_events:
                prev_key = next((k for k in ('prev_event_sha256', 'previous_event_sha256', 'previous_sha256') if k in raw_events[0]), prev_key)
            seen = {}
            for i, e in enumerate(raw_events):
                if e.get('seq') != i or e.get('episode_id') != self.episode_id or e.get('event_id') != f'{self.episode_id}:e{i:06d}':
                    raise IntegrityError('Event identity/order invalid')
                if e.get(prev_key) != prev or e.get(hash_key) != digest({k: v for k, v in e.items() if k != hash_key}):
                    raise IntegrityError('Event hash chain mismatch')
                for k, target in e.get('links', {}).items():
                    if self.version == '0.2.0' or k.endswith('_event_id'):
                        if target not in seen: raise IntegrityError('Forward or missing event link')
                seen[e['event_id']] = e
                prev = e[hash_key]
                if self.version == '0.2.0':
                    for r in refs(e):
                        if self.files.get(r['path']) != r: raise IntegrityError('Event file reference mismatch')
                self.events.append({**e, 'data': e.get('data', e.get('payload', {}))})
            if len(raw_events) != self.index.get('event_count') or prev != self.index.get('trajectory_head_sha256'):
                raise IntegrityError('Event count/head differs from index')
            checks.append('trajectory_chain_event_links')
            if self.version == '0.2.0':
                self._semantic_v02()
                checks += ['native_call_projection', 'raw_tool_observation', 'observation_delivery',
                           'snapshot_blobs_content', 'state_diff_recompute', 'artifact_snapshot_mapping', 'config_status_bindings']
            else:
                warnings += ['legacy_v01_limited_fact_mapping', 'legacy_capture_gaps_are_not_repaired']
            extra = None
            is_fixture = self.manifest.get('recording_mode') == 'test_fixture' or (self.manifest.get('capabilities') or {}).get('test_fixture')
            try:
                from c3_sandbox.l1.package import verify_package
            except ImportError:
                if self.require_l1 or self.version == '0.1.0':
                    raise Unsupported('Install existing L1 package for official v0.1 compatibility or required L1 verifier')
                warnings.append('upstream_l1_verifier_not_installed_builtin_checks_used')
            else:
                if not is_fixture or self.require_l1:
                    extra = verify_package(self.root, expected_index_sha256=self.index_sha256)
                    if not extra.get('verified'): raise IntegrityError('Installed L1 verifier rejected source: ' + str(extra.get('errors')))
                    checks.append('installed_l1_verifier')
                else:
                    warnings.append('fixture_source_internal_verification_used')
            return {'verified': True, 'schema_version': self.version, 'episode_id': self.episode_id,
                    'index_sha256': self.index_sha256, 'file_count': len(self.files), 'checks': checks,
                    'warnings': warnings, 'errors': [], 'installed_l1_verifier': extra,
                    'trust_scope': 'Self-consistency against source index; no independent historical execution attestation'}
        except (KeyError, TypeError, ValueError, OSError, IntegrityError, Unsupported) as exc:
            errors.append(type(exc).__name__ + ': ' + str(exc))
            return {'verified': False, 'checks': checks, 'warnings': warnings, 'errors': errors}

    def _semantic_v02(self):
        requests = {}; responses = {}; calls = {}; results = {}; delivered = set(); parsed = set()
        for e in self.events:
            d = e['data']; kind = e['kind']
            if kind == 'model.request':
                rid = d['request_id']
                if rid in requests: raise IntegrityError('Duplicate request')
                requests[rid] = self.json(d['body']['path'])
            elif kind == 'model.response':
                rid = d['request_id']
                if rid not in requests or rid in responses: raise IntegrityError('Response pairing invalid')
                try: responses[rid] = self.json(d['body']['path'])
                except IntegrityError: responses[rid] = None  # invalid native bytes remain recorded
            elif kind == 'model.parsed': parsed.add(d['request_id'])
            elif kind == 'tool.call':
                cid = d['call_id']
                if cid in calls or d['request_id'] not in responses: raise IntegrityError('Tool call pairing invalid')
                calls[cid] = d
            elif kind == 'tool.result':
                cid = d['call_id']
                if cid not in calls or cid in results: raise IntegrityError('Tool result pairing invalid')
                r = self.json(d['result']['path'])
                if r.get('call_id') != cid: raise IntegrityError('Result ID mismatch')
                for k in ('stdout', 'stderr'):
                    if k in r.get('observation', {}) and isinstance(r['observation'][k], str):
                        if self.read(d[k]['path']).decode('utf-8', 'replace') != r['observation'][k]:
                            raise IntegrityError('Tool output projection mismatch')
                results[cid] = d
            elif kind == 'observation.delivered':
                cid = d['call_id']
                if cid not in results or cid in delivered or d['request_id'] not in requests:
                    raise IntegrityError('Observation delivery pairing invalid')
                obs = self.json(d['observation']['path'])
                body = requests[d['request_id']]
                if not observation_present(body, obs, d.get('native_id'), calls[cid]['name']):
                    raise IntegrityError('Delivered Observation not present in actual next request')
                if 'image' in results[cid]:
                    import base64
                    encoded = base64.b64encode(self.read(results[cid]['image']['path'])).decode('ascii')
                    if encoded not in canonical(last_observation_turn(body)).decode('utf-8'):
                        raise IntegrityError('Delivered image missing in request')
                delivered.add(cid)
            elif kind == 'state.snapshot':
                s = self.json(d['snapshot']['path']); sid = s['snapshot_id']
                if sid in self.snapshots or sid != d['snapshot_id']: raise IntegrityError('Snapshot ID mismatch')
                names = set(); total = 0
                for f in s['files']:
                    relative(f['path'])
                    if f['path'] in names: raise IntegrityError('Duplicate snapshot path')
                    names.add(f['path']); total += f['size_bytes']
                    if self.files.get(f['blob'], {}).get('sha256') != f['sha256'] or self.files[f['blob']]['size_bytes'] != f['size_bytes']:
                        raise IntegrityError('Snapshot blob mismatch')
                for directory in s['directories']:
                    relative(directory['path'])
                    if directory['path'] in names: raise IntegrityError('Duplicate snapshot path')
                    names.add(directory['path'])
                if s['total_bytes'] != total or s['content_sha256'] != digest({'files': s['files'], 'directories': s['directories']}):
                    raise IntegrityError('Snapshot content digest mismatch')
                self.snapshots[sid] = s; self.snapshot_paths[sid] = d['snapshot']['path']
        for rid in parsed:
            if rid not in responses or not isinstance(responses[rid], dict): raise IntegrityError('Parsed reply has no valid native response')
            expected = native_calls(responses[rid], rid)
            observed = [d for d in calls.values() if d['request_id'] == rid]
            if expected != observed: raise IntegrityError('Native/normalized tool call projection mismatch')
        expected_changes = []
        for e in self.events:
            if e['kind'] == 'state.transition':
                d = e['data']
                b = self.snapshots[d['before']]; a = self.snapshots[d['after']]
                expected_changes.append({'episode_id': self.episode_id, 'event_id': e['event_id'],
                    'before_snapshot_id': d['before'], 'after_snapshot_id': d['after'], 'changes': diff_snapshots(b, a)})
        if self.jsonl('state_changes.jsonl') != expected_changes: raise IntegrityError('State change projection mismatch')
        cfg = self.json('raw/resolved_config.json')
        if digest(cfg) != self.manifest['config_sha256']: raise IntegrityError('Config digest mismatch')
        counts = {'events': len(self.events), 'model_requests': len(requests), 'model_responses': len(responses),
                  'tool_calls': len(calls), 'tool_results': len(results), 'observations_delivered': len(delivered), 'snapshots': len(self.snapshots)}
        if self.manifest['counts'] != counts: raise IntegrityError('Manifest count mismatch')
        ended = [e for e in self.events if e['kind'] == 'episode.finished']
        if ended and ended[-1]['data'].get('status') != self.manifest['status']: raise IntegrityError('Final status mismatch')
        artifacts = self.json('artifacts/inventory.json')
        sid = artifacts.get('snapshot_id')
        if sid is not None:
            expected = [{'path': x['path'][len('submission/'):], 'sha256': x['sha256'], 'size_bytes': x['size_bytes']}
                        for x in self.snapshots[sid]['files'] if x['path'].startswith('submission/')]
            if expected != artifacts['files']: raise IntegrityError('Artifact inventory not bound to final snapshot')
            for f in expected:
                if self.files.get('artifacts/submission/' + f['path']) != {**f, 'path': 'artifacts/submission/' + f['path']}:
                    raise IntegrityError('Artifact bytes differ from final snapshot')
        cp_index = self.json('checkpoints/index.json')
        for item in cp_index['checkpoints']:
            cp = self.json(item['path'])
            if cp['snapshot_id'] not in self.snapshots or cp['pending_calls'] or cp['config_sha256'] != self.manifest['config_sha256']:
                raise IntegrityError('Checkpoint state/config invalid')
            if cp['qualification']['eligible'] and cp['qualification'].get('image_id') != (self.manifest.get('runtime') or {}).get('image_id'):
                raise IntegrityError('Checkpoint image mismatch')
        grade = self.json('official_grade.json')
        if grade.get('projection'):
            for k in ('evaluation_validity', 'task_outcome'):
                if grade['projection'].get(k) != self.manifest['status'].get(k): raise IntegrityError('Grade status mismatch')
        if grade.get('official_raw_sha256'):
            p = 'grading/worker-output/official.json'
            if self.files[p]['sha256'] != grade['official_raw_sha256'] or self.json(p) != grade['official_raw']:
                raise IntegrityError('Official raw projection mismatch')

    def source_record(self) -> dict:
        return {'episode_id': self.episode_id, 'schema_version': self.version, 'index_sha256': self.index_sha256,
                'recording_mode': self.manifest.get('recording_mode', 'legacy_import'), 'verification': self.report}

    def verify_unchanged(self) -> bool:
        return sha_file(self.root / 'evidence_index.json') == self.index_sha256 and inventory(self.root, {'evidence_index.json'}) == sorted(self.files.values(), key=lambda x: x['path'])


def inspect(root: Path, **kwargs) -> dict:
    try:
        r = L1Reader(root, **kwargs)
        return {'verified': True, 'source': r.source_record(), 'task': r.manifest.get('task'),
                'status': r.manifest.get('status'), 'event_count': len(r.events),
                'scope': 'Controller inspection; not the restricted Analyst view'}
    except (IntegrityError, Unsupported, OSError, KeyError, TypeError) as e:
        return {'verified': False, 'error_type': type(e).__name__, 'error': str(e),
                'action': 'Use a complete decoded Evidence Package; no partial source is silently accepted'}
