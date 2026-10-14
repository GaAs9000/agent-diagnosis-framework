"""Check normalization coverage independently against retained native/source data."""
from __future__ import annotations
import json
from pathlib import Path
from .storage import digest, resolve_ref


def check_projection(root: Path, manifest: dict, source: dict, events: list[dict], states: list[dict], artifacts: list[dict]) -> list[dict]:
    errors = []
    def fail(code, detail): errors.append({'code':code,'detail':str(detail)})
    records = source.get('raw_gemini_records') or []
    native_calls = {}
    expected_deliveries = {}
    for i, record in enumerate(records):
        try:
            response = json.loads(record['response_raw'])
            parts = response['candidates'][0]['content']['parts']
            request = json.loads(record['request_raw'])
        except (ValueError, KeyError, IndexError, TypeError):
            continue
        ordinal = 0
        for j, part in enumerate(parts):
            fc = part.get('functionCall') if isinstance(part, dict) else None
            if not isinstance(fc, dict): continue
            pointer = f'/candidates/0/content/parts/{j}/functionCall'
            key = (f'raw/model/{i:06d}.response.json', pointer)
            native_calls[key] = (i, fc)
            if i + 1 < len(records):
                try:
                    nxt = json.loads(records[i+1]['request_raw'])
                    n = len(request['contents'])
                    item = nxt['contents'][n+1]['parts'][ordinal]['functionResponse']
                    if (nxt['contents'][:n] == request['contents'] and nxt['contents'][n]['parts'] == parts and item['name'] == fc.get('name')):
                        expected_deliveries[key] = (f'raw/model/{i+1:06d}.request.json', f'/contents/{n+1}/parts/{ordinal}/functionResponse/response', i+1)
                except (ValueError, KeyError, IndexError, TypeError):
                    pass
            ordinal += 1
    observed_native, call_keys, expected_delivery_ids = [], {}, {}
    for event in events:
        if event['kind'] != 'tool.call' or event['payload']['linkage'] == 'unmatched_recorded_dispatch': continue
        reference = event['source_refs'][0]
        key = (reference['path'], reference['json_pointer'])
        observed_native.append(key)
        call_id = event['payload']['tool_call_id']
        call_keys[call_id] = key
        if key not in native_calls:
            fail('unbound_native_call_reference',call_id)
        else:
            i = native_calls[key][0]
            if event['links'].get('model_call_id') != f"{manifest['episode_id']}:m{i:06d}":
                fail('native_call_model_link_mismatch',call_id)
        if key in expected_deliveries: expected_delivery_ids[call_id] = expected_deliveries[key]
    if sorted(observed_native) != sorted(native_calls):
        fail('native_call_coverage_mismatch','Each retained native functionCall must be projected exactly once')
    deliveries = [e for e in events if e['kind']=='observation.delivered']
    if sorted(e['payload']['tool_call_id'] for e in deliveries) != sorted(expected_delivery_ids):
        fail('delivery_coverage_mismatch','Delivered observations must match subsequent native requests exactly')
    for event in deliveries:
        call_id = event['payload']['tool_call_id']
        if call_id not in expected_delivery_ids: continue
        path, pointer, next_index = expected_delivery_ids[call_id]
        if event['payload']['observation_ref'] != {'path':path,'json_pointer':pointer} or event['links'].get('delivered_in_model_call_id') != f"{manifest['episode_id']}:m{next_index:06d}":
            fail('delivery_binding_mismatch',call_id)
    expected_steps = [f'/tool_dispatch/{i}/fs_delta/{j}' for i,d in enumerate(source.get('tool_dispatch') or []) for j,_ in enumerate(d.get('fs_delta') or [])]
    observed_steps = [s['source_refs'][0]['json_pointer'] for s in states if s['observation_type']=='recorded_step_delta']
    if sorted(expected_steps) != sorted(observed_steps):
        fail('step_delta_coverage_mismatch','Each archived fs_delta entry must appear exactly once')
    by_role = {'initial':{},'final':{}}
    seen = set()
    for artifact in artifacts:
        key = (artifact['role'],artifact['logical_path'])
        if key in seen: fail('duplicate_artifact_identity',key)
        seen.add(key)
        expected_object = 'file:' + digest([manifest['episode_id'], artifact['logical_path']])[:24]
        if artifact['object_id'] != expected_object or artifact['file']['path'] != f"artifacts/{artifact['role']}/{artifact['logical_path']}":
            fail('artifact_namespace_mismatch',key)
        by_role[artifact['role']][artifact['logical_path']] = {'kind':'file','sha256':artifact['file']['sha256'],'size_bytes':artifact['file']['size_bytes']}
    expected_endpoints = {}
    for path in by_role['initial'].keys() | by_role['final'].keys():
        before, after = by_role['initial'].get(path), by_role['final'].get(path)
        if before != after: expected_endpoints[path] = (before,after)
    observed_endpoints = [s for s in states if s['observation_type']=='recomputed_endpoint_delta']
    if sorted(s['path'] for s in observed_endpoints) != sorted(expected_endpoints):
        fail('endpoint_delta_coverage_mismatch','Endpoint diffs must match retained inventory')
    for state in states:
        if state['object_id'] != 'file:' + digest([manifest['episode_id'],state['path']])[:24]:
            fail('state_object_namespace_mismatch',state['change_id'])
    for state in observed_endpoints:
        if state['path'] in expected_endpoints and (state['before'],state['after']) != expected_endpoints[state['path']]:
            fail('endpoint_delta_projection_mismatch',state['change_id'])
    return errors
