#!/usr/bin/env python3
"""Offline G1 mechanics, never a live G1 certification. No model or grader calls.

Default: inventory only. --execute restores saved workspaces in temporary storage
and reconstructs request bodies using production helpers. Run on the approved
experiment host. Inputs are never changed. Missing genuine post-compaction
coverage remains untested; this script never opens the live G1 gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

L2_ROOT = Path(__file__).resolve().parents[1]
ROOT = L2_ROOT.parents[1]
L1_ROOT = ROOT / 'L1_sandbox/C3_Experimental_Sandbox'
sys.path[:0] = [str(L1_ROOT / 'src'), str(L2_ROOT / 'src')]
from c3_sandbox.l1.package import verify_package
from c3_sandbox.l1.recording.state import StateStore, restore
from c3_sandbox.l1.evidence.live import validate_snapshot
from c3_sandbox.l1.models.native import OpenAICompatibleBackend, GeminiBackend
from c3_sandbox.l1.runtime.tools import tools_for_config
from c3_sandbox.l1.storage import confined, digest, file_digest, read_json, utc_now
from l2_sandbox.evidence.l1 import L1Package

DEV = tuple('officeval_' + n for n in ('003', '004', '010', '033', '050', '060', '070', '085', '090'))
RUNS = L1_ROOT / 'runs/v3_qwen36_pi_dev45_20261017/episodes'


def bound_checkpoints(root, events):
    """Resolve each cp via checkpoint.created and its named state.snapshot ref."""
    states = {}
    for event in events:
        if event['kind'] == 'state.snapshot':
            ref = event['data']['snapshot']
            snapshot = read_json(confined(root, ref['path']))
            sid = snapshot['snapshot_id']
            if sid in states:
                raise ValueError('duplicate snapshot binding')
            if file_digest(confined(root, ref['path'])) != ref['sha256']:
                raise ValueError('snapshot reference digest mismatch')
            states[sid] = (event, ref, snapshot)
    result = []
    requests = [e for e in events if e['kind'] == 'model.request']
    for item in read_json(root / 'checkpoints/index.json')['checkpoints']:
        path = item['path']
        matches = [e for e in events if e['kind'] == 'checkpoint.created' and e['data']['checkpoint']['path'] == path]
        if len(matches) != 1:
            raise ValueError('checkpoint event binding missing or duplicated')
        event = matches[0]
        cp_path = confined(root, path)
        if file_digest(cp_path) != event['data']['checkpoint']['sha256']:
            raise ValueError('checkpoint reference digest mismatch')
        cp = read_json(cp_path)
        if cp['checkpoint_id'] != item['checkpoint_id']:
            raise ValueError('checkpoint identity mismatch')
        state_event, ref, snapshot = states[cp['snapshot_id']]
        preceding = [e for e in events if e['kind'] == 'state.snapshot' and e['seq'] < event['seq']]
        if not preceding or state_event['event_id'] != max(preceding, key=lambda e: e['seq'])['event_id']:
            raise ValueError('checkpoint must bind actual most recent captured snapshot')
        following = [e for e in requests if e['seq'] > event['seq']]
        nxt = min(following, key=lambda e: e['seq']) if following else None
        used = cp['budget_used']['model_requests']
        if used != len([e for e in requests if e['seq'] < event['seq']]):
            raise ValueError('checkpoint request counter mismatch (root episodes only)')
        compactions = [e for e in events if e['kind'] == 'session.compacted' and e['seq'] < event['seq']]
        saved_compactions = cp['conversation'].get('_pi_compaction', [])
        if [e['data'] for e in compactions] != saved_compactions:
            raise ValueError('saved compaction state differs from recorded events')
        result.append(dict(checkpoint=cp, checkpoint_path=path, snapshot=snapshot,
                           snapshot_ref=ref, checkpoint_event=event, next_request=nxt,
                           genuine_post_compaction=bool(compactions)))
    return result


def selected_checkpoints(rows):
    resumable = [r for r in rows if r['checkpoint']['qualification']['eligible'] and r['next_request']]
    if not resumable:
        raise ValueError('no qualified checkpoint with a recorded following request')
    selected = {}
    for name, row in (('c0', resumable[0]), ('middle', resumable[len(resumable)//2]), ('tail', resumable[-1])):
        selected.setdefault(row['checkpoint']['checkpoint_id'], {'row': row, 'roles': []})['roles'].append(name)
    post = [r for r in resumable if r['genuine_post_compaction']]
    if post:
        row = post[0]
        selected.setdefault(row['checkpoint']['checkpoint_id'], {'row': row, 'roles': []})['roles'].append('genuine_post_compaction')
    return list(selected.values())


def reconstruct_request(root, config, row):
    cp = row['checkpoint']
    cls = {'openai-compatible': OpenAICompatibleBackend, 'gemini': GeminiBackend}[config['model']['adapter']]
    body = cls(config['model']).request(cp['conversation'], tools_for_config(config))
    ref = row['next_request']['data']['body']
    recorded_path = confined(root, ref['path'])
    if file_digest(recorded_path) != ref['sha256']:
        raise ValueError('recorded request digest mismatch')
    if body != read_json(recorded_path):
        raise ValueError('production request reconstruction mismatch')
    return {'reconstructed_sha256': digest(body), 'recorded_body_ref': ref,
            'following_request_id': row['next_request']['data']['request_id']}


def replay_workspace(root, snapshot, config, temp_parent=None):
    validate_snapshot(root, snapshot)
    with tempfile.TemporaryDirectory(prefix='g1-mechanics-', dir=temp_parent) as directory:
        base = Path(directory)
        restore(root, snapshot, base / 'workspace')
        evidence = base / 'capture'
        evidence.mkdir()
        runtime = config['runtime']
        captured = StateStore(evidence, runtime['max_workspace_bytes'], runtime['max_files']).capture(base / 'workspace', 'g1_replay')
        for field in ('files', 'directories', 'total_bytes', 'content_sha256'):
            if captured[field] != snapshot[field]:
                raise ValueError('restored workspace hash/mode mismatch: ' + field)
    return {'files': snapshot['files'], 'directories': snapshot['directories'],
            'content_sha256': snapshot['content_sha256'], 'status': 'byte_and_mode_equal'}


def audit_package(root, temp_parent=None):
    index_sha = file_digest(root / 'evidence_index.json')
    verified = verify_package(root, expected_index_sha256=index_sha)
    if not verified['verified']:
        raise ValueError('package integrity verification failed: ' + repr(verified['errors']))
    pkg = L1Package(root)
    if pkg.manifest.get('lineage') or pkg.task_id not in DEV:
        raise ValueError('expected a DEV root package')
    if not pkg.outcome()['valid']:
        return {'slot': root.name, 'status': 'study_invalid_preserved', 'parent_index_sha256': index_sha}
    config = read_json(root / 'raw/resolved_config.json')
    rows = bound_checkpoints(root, pkg.events)
    checks = []
    for selection in selected_checkpoints(rows):
        row = selection['row']; cp = row['checkpoint']
        if cp['config_sha256'] != digest(config):
            raise ValueError('checkpoint resolved configuration mismatch')
        checks.append({'roles': selection['roles'], 'checkpoint_id': cp['checkpoint_id'],
                       'checkpoint_path': row['checkpoint_path'], 'snapshot_id': cp['snapshot_id'],
                       'snapshot_ref': row['snapshot_ref'], 'budget_used': cp['budget_used'],
                       'remaining_model_requests': config['budget']['max_model_requests'] - cp['budget_used']['model_requests'],
                       'pending_observations_sha256': digest(cp['pending_observations']),
                       'compaction_state': cp['conversation'].get('_pi_compaction', []),
                       'request': reconstruct_request(root, config, row),
                       'workspace': replay_workspace(root, row['snapshot'], config, temp_parent)})
    if file_digest(root / 'evidence_index.json') != index_sha:
        raise ValueError('parent changed during audit')
    # Repeat full integrity validation, rather than trusting an unchanged index file.
    if not verify_package(root, expected_index_sha256=index_sha)['verified']:
        raise ValueError('parent inventory changed during audit')
    return {'slot': root.name, 'status': 'mechanical_pass', 'parent_index_sha256': index_sha,
            'config_sha256': digest(config), 'model_config': config['model'],
            'budget_config': config['budget'], 'compaction_config': config['agent'].get('compaction'),
            'checkpoints_examined': len(rows), 'checks': checks,
            'budget_edge_coverage': 'tested_at_one_remaining' if any(c['remaining_model_requests'] == 1 for c in checks) else 'untested',
            'genuine_post_compaction': 'tested' if any('genuine_post_compaction' in x['roles'] for x in checks) else 'untested'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=RUNS)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--temp-parent', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    roots = [args.runs / f'{task}.r{rep:02d}' for task in DEV for rep in range(5)]
    report = {'schema': 'g1_offline_mechanics_v1', 'created_at': utc_now(),
              'actor_requests': 0, 'official_scorer_calls': 0, 'docker_calls': 0,
              'network_calls': 0, 'mode': 'execute' if args.execute else 'inventory_only',
              'applicable_configuration_verified': False, 'live_g1_gate_pass': False,
              'limitations': ['Offline reconstruction does not test live model seed/temperature behavior.',
                              'Missing genuine post-compaction checkpoints remain untested.',
                              'Self-consistency does not establish historical controller authenticity.'],
              'sources': {}, 'slots': []}
    source_paths = [Path(__file__).resolve(), L2_ROOT / 'src/l2_sandbox/evidence/l1.py',
                    L2_ROOT / 'src/l2_sandbox/io.py',
                    *sorted((L1_ROOT / 'src/c3_sandbox/l1').rglob('*.py'))]
    report['sources'] = {str(p.relative_to(ROOT)): file_digest(p) for p in source_paths}
    for root in roots:
        row = {'slot': root.name, 'status': 'unsealed_or_missing'}
        try:
            if (root / 'evidence_index.json').is_file():
                row = audit_package(root, args.temp_parent) if args.execute else {
                    'slot': root.name, 'status': 'not_executed',
                    'parent_index_sha256': file_digest(root / 'evidence_index.json')}
        except Exception as exc:
            row.update(status='mechanical_failure', error_type=type(exc).__name__, error=str(exc))
        report['slots'].append(row)
    report['mechanical_pass_count'] = sum(r['status'] == 'mechanical_pass' for r in report['slots'])
    report['genuine_post_compaction_parent_count'] = sum(r.get('genuine_post_compaction') == 'tested' for r in report['slots'])
    report['mechanical_failure_count'] = sum(r['status'] == 'mechanical_failure' for r in report['slots'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # An explicit new output prevents silently replacing a prior audit receipt.
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: report[k] for k in ('mode', 'mechanical_pass_count', 'mechanical_failure_count', 'live_g1_gate_pass')}))
    return int(report['mechanical_failure_count'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
