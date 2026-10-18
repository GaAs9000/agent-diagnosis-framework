#!/usr/bin/env python3
"""Recheck saved G3 evidence under a user waiver; no model or scorer execution."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

SANDBOX = Path(__file__).resolve().parents[1]
ROOT = SANDBOX.parents[1]
sys.path[:0] = [str(SANDBOX / 'src'), str(ROOT / 'L1_sandbox/C3_Experimental_Sandbox/src')]
from l2_sandbox.evidence.l1 import L1Package
from l2_sandbox.v3_execution import file_digest, read_json, write_json
from l2_sandbox.measurement_gates import human_review_policy_binding, measurement_gate_binding

spec = importlib.util.spec_from_file_location('saved_g3', SANDBOX / 'scripts/prepare_g3_review.py')
g3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g3)


def checked_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('evidence path outside its root')
    return path


def verify_inventory(root: Path, rows: list[dict]) -> None:
    if not root.is_dir():
        raise ValueError('restored directory missing')
    if any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('restored directory contains symlinks')
    expected = {r['path']: (r['sha256'], r['size_bytes']) for r in rows}
    actual = {str(p.relative_to(root)): (file_digest(p), p.stat().st_size)
              for p in root.rglob('*') if p.is_file()}
    if len(expected) != len(rows) or actual != expected:
        raise ValueError('restored file inventory changed')


def verify_saved_g3(report_path: Path) -> dict:
    report = read_json(report_path)
    rows = report['rows']
    if len(rows) != 20 or {r['row_id'] for r in rows} != set(range(1, 21)):
        raise ValueError('exact frozen 20-point sample required')
    if len({(r['episode_id'], r['checkpoint_id']) for r in rows}) != 20:
        raise ValueError('duplicate G3 checkpoint')
    # Every grading byte recorded by the preparation is checked, including
    # process receipts and raw official output. This never reruns the grader.
    evidence_hashes = read_json(report_path.parent / 'evidence_hashes.json')
    if not evidence_hashes:
        raise ValueError('saved scoring evidence inventory is empty')
    verified_paths = set()
    for record in evidence_hashes:
        path = checked_path(report_path.parent, record['path'])
        if file_digest(path) != record['sha256'] or path.stat().st_size != record['size_bytes']:
            raise ValueError('saved scoring evidence changed')
        if path in verified_paths:
            raise ValueError('duplicate saved scoring evidence')
        verified_paths.add(path)
    packages = {}
    files = scores = 0
    for row in rows:
        if not row.get('candidate_evaluations'):
            raise ValueError('every checkpoint requires a saved official scorer invocation')
        snapshot_path = checked_path(ROOT, row['snapshot_path'])
        if file_digest(snapshot_path) != row['expected_snapshot_sha256']:
            raise ValueError('G3 snapshot identity changed')
        snapshot = read_json(snapshot_path)
        episode = row['episode_id']
        if episode not in packages:
            package = L1Package(ROOT / 'L1_sandbox/C3_Experimental_Sandbox/runs/v3_qwen36_pi_dev45_20261017/episodes' / episode)
            g3.verify_official_source(package)
            packages[episode] = package
        package = packages[episode]
        checkpoint = package.json(f"checkpoints/{row['checkpoint_id']}.json")
        if checkpoint['snapshot_id'] != snapshot['snapshot_id'] or snapshot['snapshot_id'] != row['snapshot_id']:
            raise ValueError('G3 checkpoint binding changed')
        g3.validate_snapshot(package.root, snapshot)
        inventory = read_json(checked_path(ROOT, row['restored_file_hashes']))
        if {r['path']: r['sha256'] for r in inventory} != {r['path']: r['sha256'] for r in snapshot['files']}:
            raise ValueError('restored files differ from the frozen snapshot')
        verify_inventory(checked_path(ROOT, row['restored_directory']), inventory)
        files += len(inventory)
        for evaluation in row['candidate_evaluations']:
            evidence = checked_path(ROOT, evaluation['evidence_directory'])
            required = ('grade.json', 'invocation.json', 'grading/process.json',
                        'grading/runtime.json', 'grading/stdout.bin', 'grading/stderr.bin',
                        'grading/worker-output/official.json')
            if any((evidence / name).resolve() not in verified_paths for name in required):
                raise ValueError('official scorer evidence is absent from the hash inventory')
            if read_json(evidence / 'grade.json') != evaluation['grade']:
                raise ValueError('official grade differs from the saved receipt')
            process = evaluation['grade']['process']
            if process.get('container_exit_code') != 0 or not process.get('output_complete'):
                raise ValueError('missing or failed official scorer execution')
            scores += 1
    return {'sampled_checkpoint_count': 20, 'restored_checkpoint_count': 20,
            'missing_restored_directory_count': 0, 'missing_scorer_exit_count': 0,
            'execution_metadata_verified': True, 'source_provenance_verified': True,
            'human_review_completed_count': sum(bool(r.get('reviewer') and r.get('human_conclusion')) for r in rows),
            'human_review_status': 'waived_by_user', 'visual_correctness_verified': False,
            'status': 'automatic_evidence_verified_human_review_waived',
            'restored_files_verified': files, 'saved_official_scorer_invocations_verified': scores,
            'report_path': str(report_path.resolve()), 'report_sha256': file_digest(report_path),
            'new_actor_requests': 0, 'new_scorer_invocations': 0,
            'official_grade_unknown_count': sum(r['candidate_status'] == 'official_grade_unknown' for r in rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authorization', type=Path, default=SANDBOX / 'configs/human_review_policy_20261018.json')
    parser.add_argument('--g3-report', type=Path, default=SANDBOX / 'validation/d0_20261018/g3_review/g3_20_actual_review.json')
    parser.add_argument('--base-audit', type=Path, default=SANDBOX / 'reports/v3_gates_g2_g3_g4_audit_20261017.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit = read_json(args.base_audit)
    audit.update(human_review_policy='waived_by_user', human_review_authorization={
        'path': str(args.authorization.resolve()), 'sha256': file_digest(args.authorization)})
    _, errors = human_review_policy_binding(audit, args.output)
    if errors:
        raise ValueError(';'.join(errors))
    audit['V3_G3_R20'] = verify_saved_g3(args.g3_report)
    audit['waiver_audit_provenance'] = {
        'base_audit_path': str(args.base_audit.resolve()), 'base_audit_sha256': file_digest(args.base_audit),
        'G1_G2_G4': 'historical evidence copied unchanged; no new PASS inferred',
        'scientific_efficacy': 'not_tested'}
    write_json(args.output, audit)
    binding, errors = measurement_gate_binding(args.output)
    print(json.dumps({'output': str(args.output), 'gates': binding['gates'], 'remaining_errors': errors}))


if __name__ == '__main__':
    main()
