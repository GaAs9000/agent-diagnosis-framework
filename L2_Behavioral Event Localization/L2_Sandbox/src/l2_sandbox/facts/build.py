"""Versioned deterministic fact generation; no LLM inference or official regrading."""
from __future__ import annotations
from pathlib import Path
from ..io import canonical, digest, loads, write_json
from ..evidence.l1 import L1Reader
from ..evidence.references import ReferenceStore
from .docx_probe import probe, VERSION as PROBE_VERSION

VERSION = 'facts-0.1.0'

class FactBuilder:
    def __init__(self, reader: L1Reader, output: Path, config: dict):
        self.reader = reader; self.output = output; self.config = config
        self.refs = ReferenceStore(reader, output)
        self.facts = []; self.objects = {}; self.events = []
        self.checkpoints = []; self.coverage = []
        self._by_id = {}; self.document_versions = {}; self.calls = {}

    def add(self, kind, value, refs, *, event_ids=None, origin='recorded', scope=None,
            availability='present', derivation=None, coverage=None):
        row = {'fact_type': kind, 'origin': origin, 'availability': availability, 'value': value,
               'source_ref_ids': list(dict.fromkeys(refs)), 'event_ids': event_ids or [], 'scope': scope or {},
               'derivation': derivation if origin == 'derived' else None,
               'coverage': coverage or {'basis': 'saved_source_records', 'does_not_prove': 'unique_cause_or_model_understanding'}}
        fid = 'fact_' + digest(row)[:24]
        if fid not in self._by_id:
            row = {'fact_id': fid, **row}; self.facts.append(row); self._by_id[fid] = row
        return fid

    def build(self):
        r = self.reader
        task = r.json('benchmark/public_task.json') if 'benchmark/public_task.json' in r.files else None
        if task:
            self.add('task_instruction', task, [self.refs.add('benchmark/public_task.json')])
        else:
            self.coverage.append({'area': 'public_task', 'status': 'unknown', 'reason': 'No public_task.json in legacy package'})
        if r.version == '0.1.0': return self._legacy(task)
        status = r.manifest['status']
        self.add('episode_status', status, [self.refs.add('manifest.json', json_pointer='/status')])
        self.add('execution_profile', {k: r.manifest.get(k) for k in ('recording_mode', 'capabilities')},
                 [self.refs.add('manifest.json', json_pointer='/recording_mode'), self.refs.add('manifest.json', json_pointer='/capabilities')])
        self.add('official_grade', r.json('official_grade.json'), [self.refs.add('official_grade.json')],
                 coverage={'basis': 'original_episode_official_result_visible', 'does_not_prove': 'rubric_success_or_validated_cause'})
        inventory = r.json('artifacts/inventory.json')
        self.add('submission_inventory', inventory, [self.refs.add('artifacts/inventory.json')])
        config = r.json('raw/resolved_config.json')
        self.add('execution_budget', config.get('budget'), [self.refs.add('raw/resolved_config.json', json_pointer='/budget')])
        delivered = {e['data']['call_id'] for e in r.events if e['kind'] == 'observation.delivered'}
        for e in r.events:
            kind = e['kind']; d = e['data']; eid = e['event_id']
            # Restricted event projection: no private code paths, parent results,
            # checkpoint conversation, raw configuration or arbitrary manifest data.
            projection = {'event_id': eid, 'seq': e['seq'], 'kind': kind, 'observed_at': e.get('observed_at'),
                          'links': e.get('links', {}), 'fact_ids': [], 'available_evidence_ref_ids': []}
            before = len(self.facts)
            if kind in ('model.request', 'model.response'):
                projection['available_evidence_ref_ids'] = [self.refs.declare(d['body']['path'], event_ids=[eid], as_json=kind=='model.request')]
            if kind == 'model.parsed' and d.get('text'):
                self.add('model_output_text', d, [self.refs.event(e)], event_ids=[eid],
                         coverage={'basis': 'recorded_model_text', 'does_not_prove': 'claimed_tool_action_happened_or_task_completed'})
            if kind == 'tool.call':
                fid = self.add('tool_requested', d, [self.refs.event(e)], event_ids=[eid],
                    coverage={'basis': 'model_native_tool_request', 'does_not_prove': 'each_shell_subcommand_executed'})
                self.calls[d['call_id']] = e
            elif kind == 'tool.result':
                raw = r.json(d['result']['path']); obs = raw.get('observation', {})
                rf = self.refs.add(d['result']['path'], json_pointer='/observation', event_ids=[eid])
                stdout_ref = self.refs.add(d['stdout']['path'], byte_range=(0, r.files[d['stdout']['path']]['size_bytes']), event_ids=[eid])
                stderr_ref = self.refs.add(d['stderr']['path'], byte_range=(0, r.files[d['stderr']['path']]['size_bytes']), event_ids=[eid])
                value = {'call_id': d['call_id'], 'name': self.calls.get(d['call_id'], {}).get('data', {}).get('name'),
                         'exit_code': obs.get('exit_code'), 'stop_reason': obs.get('stop_reason'),
                         'output_complete': obs.get('output_complete'), 'stdout_ref_id': stdout_ref, 'stderr_ref_id': stderr_ref,
                         'stderr_preview': obs.get('stderr', '')[:800], 'delivery_recorded': d['call_id'] in delivered}
                self.add('tool_exit', value, [rf, stdout_ref, stderr_ref, self.refs.event(e)], event_ids=[eid])
                if d['call_id'] not in delivered:
                    self.add('observation_not_recorded_delivered', {'call_id': d['call_id'], 'meaning': 'No later delivery event; not evidence of what model understood'},
                             [self.refs.event(e)], event_ids=[eid], origin='derived', derivation={'rule': VERSION, 'operation': 'delivery_event_set_difference'})
            elif kind == 'observation.delivered':
                ref = self.refs.add(d['observation']['path'], event_ids=[eid])
                self.add('observation_delivery', {'call_id': d['call_id'], 'request_id': d['request_id'], 'observation_ref_id': ref},
                         [self.refs.event(e), ref], event_ids=[eid], coverage={'basis': 'actual_following_request', 'does_not_prove': 'attention_or_understanding'})
            elif kind in ('episode.error', 'model.error', 'tool.error', 'tool.rejected', 'tool.skipped', 'state.error', 'evaluation.error'):
                self.add('protocol_error' if d.get('code') == 'protocol_error' or kind == 'tool.rejected' else 'execution_error',
                         {'kind': kind, **d}, [self.refs.event(e)], event_ids=[eid])
            elif kind == 'model.response':
                try:
                    body = r.json(d['body']['path'])
                    cand = body.get('candidates', body.get('choices', []))
                    if cand:
                        c = cand[0]; finish = c.get('finishReason', c.get('finish_reason'))
                        if finish not in (None, 'STOP', 'stop', 'tool_calls'):
                            key = '/candidates/0' if 'candidates' in body else '/choices/0'
                            self.add('native_protocol_signal', {'request_id': d['request_id'], 'finish_reason': finish,
                                     'finish_message': c.get('finishMessage'), 'source_observation_only': True},
                                     [self.refs.event(e), self.refs.add(d['body']['path'], json_pointer=key, event_ids=[eid])], event_ids=[eid])
                except Exception as exc:
                    from ..io import IntegrityError
                    if not isinstance(exc, IntegrityError): raise
                    self.coverage.append({'area': 'response_json', 'event_id': eid, 'status': 'invalid', 'reason': str(exc)})
            elif kind == 'episode.restored':
                self.add('restoration_recorded', d, [self.refs.event(e)], event_ids=[eid],
                         coverage={'basis': 'this_episode_restore_record', 'does_not_prove': 'parent_task_result_or_causal_improvement'})
            elif kind == 'episode.finished':
                self.add('termination', d, [self.refs.event(e)], event_ids=[eid])
            projection['fact_ids'] = [f['fact_id'] for f in self.facts[before:]]
            self.events.append(projection)
        changes = r.jsonl('state_changes.jsonl')
        for i, change in enumerate(changes):
            er = r.events_by_id[change['event_id']]
            for j, delta in enumerate(change['changes']):
                fid = self.add('file_transition', {**delta, 'before_snapshot_id': change['before_snapshot_id'],
                               'after_snapshot_id': change['after_snapshot_id'], 'tool_result_event_id': er.get('links', {}).get('tool_result')},
                      [self.refs.add('state_changes.jsonl', line=i, json_pointer=f'/changes/{j}', event_ids=[change['event_id']]),
                       self.refs.add(r.snapshot_paths[change['before_snapshot_id']]), self.refs.add(r.snapshot_paths[change['after_snapshot_id']])],
                      event_ids=[change['event_id']], scope={'file_path': delta['path']}, origin='derived',
                      derivation={'rule': VERSION, 'operation': 'reverified_file_boundary_diff'},
                      coverage={'basis': 'snapshot_boundary', 'does_not_prove': 'a_specific_line_inside_shell_command'})
                self.events[er['seq']]['fact_ids'].append(fid)
        if not any(c['changes'] for c in changes):
            self.add('no_recorded_file_change', {'transitions': len(changes), 'changed_entries': 0},
                     [self.refs.add('state_changes.jsonl', byte_range=(0, len(r.read('state_changes.jsonl'))))],
                     origin='derived', derivation={'rule': VERSION, 'operation': 'all_recorded_diffs_empty'},
                     coverage={'basis': 'captured_file_boundaries', 'does_not_prove': 'no_transient_writes_or_process_side_effects'})
        for item in r.json('checkpoints/index.json')['checkpoints']:
            cp = r.json(item['path'])
            # Do NOT give the Analyst the checkpoint native conversation, signatures,
            # images, parent prediction results or raw credentials configuration.
            safe_cp = {k: cp[k] for k in ('checkpoint_id', 'snapshot_id', 'qualification', 'budget_used', 'config_sha256')}
            safe_cp['checkpoint_sha256'] = r.files[item['path']]['sha256']
            safe_cp['source_path'] = item['path']
            self.checkpoints.append(safe_cp)
        if self.config['objects']['enabled']: self._objects()
        self.coverage.extend([
            {'area': 'analysis', 'status': 'retrospective', 'reason': 'Original outcome visible; intervention validation branches are not read'},
            {'area': 'object_scope', 'status': 'limited', 'reason': 'Version-scoped direct DOCX XML only; rendering, inherited styles and semantic identity unknown'},
            {'area': 'read_history', 'status': 'unknown', 'reason': 'No general inference of no-read-before-write from missing logs'},
            {'area': 'causal_status', 'status': 'untested', 'reason': 'No experiment executed by fact generation'},
            {'area': 'agent_images', 'status': 'unsupported', 'reason': 'Image bytes remain in source; this Analyst text view does not infer image content'},
        ])
        return self

    def _objects(self):
        r = self.reader; max_versions = self.config['objects']['max_docx_versions']
        locations = {}
        for sid, snap in r.snapshots.items():
            for f in snap['files']:
                if f['path'].lower().endswith('.docx'):
                    locations.setdefault(f['sha256'], []).append({'snapshot_id': sid, 'file_path': f['path'], 'blob': f['blob']})
        # Deterministic initial-to-final order, with declared cap rather than silent sampling.
        for n, (sha, locs) in enumerate(locations.items()):
            if n >= max_versions:
                self.coverage.append({'area': 'docx_versions', 'status': 'partial', 'reason': 'max_docx_versions',
                                      'not_inspected_sha256': sha}); continue
            blob = locs[0]['blob']; data = r.read(blob)
            parsed = probe(data, self.config['objects'])
            pid = parsed['probe_id']; outpath = f'derived/objects/{pid}.json'
            write_json(self.output, outpath, parsed); self.objects[pid] = parsed
            ref = self.refs.add(blob, binary=True)
            summary = {'probe_id': pid, 'file_sha256': sha, 'locations': locs,
                       'probe_status': parsed['status'], 'summary': parsed['summary'], 'unknowns': parsed['unknowns']}
            fid = self.add('object_property', summary, [ref, *[self.refs.add(r.snapshot_paths[loc['snapshot_id']]) for loc in locs]], origin='derived',
                           scope={'file_versions': locs}, availability=parsed['status'] if parsed['status'] in ('present', 'unsupported') else 'unknown',
                           derivation={'rule': PROBE_VERSION, 'output_path': outpath, 'output_sha256': digest(parsed),
                                       'config_sha256': parsed['config_sha256'], 'operation': 'read_only_docx_direct_xml'},
                           coverage={'basis': 'posthoc_parse_of_saved_file_bytes', 'does_not_prove': 'what_agent_saw_or_rendered_appearance'})
            self.document_versions[sha] = {'fact_id': fid, 'probe_id': pid, 'summary': parsed['summary']}
        for f in list(self.facts):
            if f['fact_type'] != 'file_transition': continue
            val = f['value']; b = val.get('before') or {}; a = val.get('after') or {}
            if b.get('sha256') in self.document_versions and a.get('sha256') in self.document_versions:
                left = self.document_versions[b['sha256']]; right = self.document_versions[a['sha256']]
                if left['summary'] == right['summary']: continue
                self.add('object_structure_change', {'file_path': val['path'], 'before_sha256': b['sha256'], 'after_sha256': a['sha256'],
                         'before_summary': left['summary'], 'after_summary': right['summary'], 'semantic_object_mapping': 'not_inferred',
                         'input_fact_ids': [f['fact_id'], left['fact_id'], right['fact_id']]},
                         f['source_ref_ids'], event_ids=f['event_ids'], origin='derived',
                         derivation={'rule': VERSION, 'operation': 'version_scoped_summary_comparison'},
                         coverage={'basis': 'whole_document_counts', 'does_not_prove': 'same_table_semantic_identity_or_specific_rule_satisfaction'})

    def _legacy(self, task):
        r = self.reader
        self.coverage += [{'area': 'legacy', 'status': 'limited', 'reason': 'v0.1 source verified by installed L1; no invented object snapshots or restore qualification'}]
        for e in r.events:
            before = len(self.facts)
            if e['kind'] in ('tool.call', 'tool.result', 'model.error', 'episode.finished', 'observation.delivered'):
                self.add('legacy_record', {'kind': e['kind'], 'payload': e['data']},
                         [self.refs.event(e, '/payload')], event_ids=[e['event_id']],
                         coverage={'basis': 'legacy_record_projection', 'does_not_prove': 'complete_live_capture'})
            self.events.append({'event_id': e['event_id'], 'seq': e['seq'], 'kind': e['kind'],
                                'observed_at': None, 'links': {}, 'fact_ids': [f['fact_id'] for f in self.facts[before:]]})
        return self
