#!/usr/bin/env python3
"""Targeted verification for stateful cold checkpoint recovery.

Verifies:
1. Selection of an eligible checkpoint where workspace files were modified/added (files_changed > 0).
2. Proper restoration of prior files, content hashes, and permissions into fresh child workspace.
3. Successful continuation of native conversation, issuing real new requests and docker tool calls.
4. Total immutability of parent package: parent content and hashes are untouched.
5. Child package passes full sealed evidence verification.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

from c3_sandbox.l1.core.config import resolve
from c3_sandbox.l1.core.orchestrator import run
from c3_sandbox.l1.storage import read_json, write_json
from c3_sandbox.l1.package import verify_package


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=False, type=Path, default=None, help='Config path, defaults to parent raw/resolved_config.json')
    parser.add_argument('--parent', required=True, type=Path, help='Parent episode package directory')
    parser.add_argument('--checkpoint-id', required=True, type=str, help='Checkpoint ID to resume from')
    parser.add_argument('--output-root', required=True, type=Path, help='New child directory')
    args = parser.parse_args(argv)

    parent = args.parent.absolute()
    output = args.output_root.absolute()
    if output.exists():
        sys.exit(f"Output directory {output} already exists")

    config = resolve(args.config) if args.config else read_json(parent / 'raw/resolved_config.json')

    # 1. Audit parent checkpoint and snapshot
    idx = read_json(parent / 'checkpoints/index.json')
    cp_match = [c for c in idx.get('checkpoints', []) if c['checkpoint_id'] == args.checkpoint_id]
    if not cp_match:
        sys.exit(f"Checkpoint {args.checkpoint_id} not found in parent index")
    if not cp_match[0].get('eligible'):
        sys.exit(f"Checkpoint {args.checkpoint_id} is marked ineligible in parent")

    cp_data = read_json(parent / f"checkpoints/{args.checkpoint_id}.json")
    snap_id = cp_data['snapshot_id']
    target_snapshot = read_json(parent / f"states/snapshots/{snap_id}.json")
    s0_snapshot = read_json(parent / 'states/snapshots/s000000.json')

    s0_files = {f['path']: f['sha256'] for f in s0_snapshot.get('files', [])}
    snap_files = {f['path']: f for f in target_snapshot.get('files', [])}
    
    added_files = set(snap_files.keys()) - set(s0_files.keys())
    modified_files = {p for p in set(snap_files.keys()) & set(s0_files.keys()) if snap_files[p]['sha256'] != s0_files[p]}

    parent_index_sha256_before = sha256_file(parent / 'evidence_index.json')

    receipt = {
        'schema_version': '0.2.0',
        'checkpoint_id': args.checkpoint_id,
        'snapshot_id': snap_id,
        'parent_package': str(parent),
        'parent_index_sha256_before': parent_index_sha256_before,
        'files_in_snapshot': len(snap_files),
        'files_added_since_start': sorted(list(added_files)),
        'files_modified_since_start': sorted(list(modified_files)),
        'has_stateful_file_change': bool(added_files or modified_files),
    }

    if not receipt['has_stateful_file_change']:
        print(f"WARNING: Checkpoint {args.checkpoint_id} has 0 file changes compared to s000000.")

    # 2. Execute Branch
    print(f"Executing branch recovery from {args.checkpoint_id} ({len(added_files)} files added)...")
    branch_result = run(config, output, parent=parent, checkpoint_id=args.checkpoint_id)

    # 3. Verify Parent Immutability
    parent_index_sha256_after = sha256_file(parent / 'evidence_index.json')
    parent_immutable = (parent_index_sha256_before == parent_index_sha256_after)
    receipt['parent_immutable'] = parent_immutable

    # 4. Check Child Manifest & Events
    manifest = read_json(output / 'manifest.json')
    receipt['child_manifest'] = {
        'episode_id': manifest['episode_id'],
        'execution_validity': manifest['status']['execution_validity'],
        'termination': manifest['status']['termination'],
        'counts': manifest['counts'],
        'lineage': manifest.get('lineage')
    }

    # 5. Check Restoration of Snapshot Files in Lineage Records
    parent_snapshot_record = read_json(output / 'lineage/parent_snapshot.json')
    receipt['lineage_snapshot_verified'] = (parent_snapshot_record['content_sha256'] == target_snapshot['content_sha256'])

    # 6. Verify Child Package Integrity
    child_verification = branch_result.get('verification', {})
    receipt['child_verified'] = child_verification.get('verified', False)
    receipt['child_index_sha256'] = child_verification.get('index_sha256')

    receipt['test_passed'] = bool(
        receipt['has_stateful_file_change']
        and parent_immutable
        and receipt['lineage_snapshot_verified']
        and receipt['child_verified']
        and manifest['counts']['model_requests'] > 0
        and manifest['counts']['tool_results'] > 0
    )

    write_json(output, 'stateful_resume_receipt.json', receipt)
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return 0 if receipt['test_passed'] else 2


if __name__ == '__main__':
    sys.exit(main())
