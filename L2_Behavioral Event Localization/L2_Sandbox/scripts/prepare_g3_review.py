#!/usr/bin/env python3
"""Prepare exactly the frozen G3 20-point dossier, with zero model requests.

Default is read-only planning (Docker inspect/info only). --execute preserves
restored workspaces, staged candidates and every official grader invocation.
No cache, original episode, catalog or human-review fields are modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

SANDBOX = Path(__file__).resolve().parents[1]
ROOT = SANDBOX.parents[1]
sys.path[:0] = [str(SANDBOX / 'src'), str(ROOT / 'L1_sandbox/C3_Experimental_Sandbox/src')]
from l2_sandbox.evidence.l1 import L1Package
from l2_sandbox.grades import _office_targets, select_snapshot_candidates
from c3_sandbox.l1.core.contracts import EvaluationSpec
from c3_sandbox.l1.evaluation.evaluator import DockerEvaluator
from c3_sandbox.l1.evaluation.study import DELIVERY_ERRORS_V1_1
from c3_sandbox.l1.evidence.live import validate_snapshot
from c3_sandbox.l1.recording.state import restore

METADATA = SANDBOX / 'validation/d0_20261018/human_review/g3_20_metadata_review.json'
OUTPUT = SANDBOX / 'validation/d0_20261018/g3_review'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def checked_rows(metadata, root):
    rows = metadata['rows']
    if len(rows) != 20 or [r['row_id'] for r in rows] != list(range(1, 21)):
        raise ValueError('Expected exactly the frozen ordered 20 rows')
    if len({(r['episode_id'], r['checkpoint_id']) for r in rows}) != 20:
        raise ValueError('Duplicate checkpoint')
    for r in rows:
        p = (root / r['snapshot_path']).resolve()
        if not p.is_relative_to(root.resolve()) or sha(p) != r['expected_snapshot_sha256']:
            raise ValueError(f"Snapshot identity mismatch: row {r['row_id']}")
        source = root / r['source']['path']
        if sha(source) != r['source']['sha256']:
            raise ValueError('Frozen audit sheet changed')
        if read(p)['snapshot_id'] != r['snapshot_id']:
            raise ValueError('Snapshot ID mismatch')
    return rows


def candidate_inputs(pkg, row):
    cps = pkg.checkpoints()
    target = next((k for k, cp in cps.items() if cp['checkpoint_id'] == row['checkpoint_id']), None)
    if target is None or cps[target]['snapshot_id'] != row['snapshot_id']:
        raise ValueError('Checkpoint does not reference declared snapshot')
    previous, last = {}, {}
    for k, cp in sorted(cps.items()):
        if k > target:
            break
        current = {f['path']: f['sha256'] for f in pkg.json(f"states/snapshots/{cp['snapshot_id']}.json")['files']}
        for p, digest in current.items():
            if previous.get(p) != digest:
                last[p] = k
        previous = current
    initial = {f['sha256'] for f in pkg.json('states/snapshots/s000000.json')['files'] if f['path'].startswith('input/')}
    targets, exts = _office_targets(pkg)
    snapshot = pkg.json(f"states/snapshots/{row['snapshot_id']}.json")
    return snapshot, select_snapshot_candidates(snapshot, initial, targets, exts, last)


def verify_official_source(pkg):
    """Compare both directions so deleting an indexed dependency fails closed."""
    prefix = 'benchmark/source/'
    indexed = [f for f in pkg.json('evidence_index.json')['files'] if f['path'].startswith(prefix)]
    expected = {f['path'][len(prefix):]: (f['sha256'], f['size_bytes']) for f in indexed}
    source = pkg.root / 'benchmark/source'
    actual = {}
    for p in sorted(source.rglob('*')):
        if p.is_symlink():
            raise ValueError(f'Unsafe official scorer source: {p}')
        if p.is_file():
            actual[p.relative_to(source).as_posix()] = (sha(p), p.stat().st_size)
    if not expected or len(expected) != len(indexed) or actual != expected:
        raise ValueError(f'Official scorer source inventory or digest mismatch: {pkg.root}')
    return [{'path': path, 'sha256': digest, 'size_bytes': size} for path, (digest, size) in sorted(actual.items())]


def preflight(rows):
    packages, image_records, sources, source_inventories = {}, {}, {}, {}
    # Check every source inventory before any Docker command, including inspect.
    for row in rows:
        episode = row['episode_id']
        if episode not in packages:
            pkg = L1Package((ROOT / row['snapshot_path']).parents[2])
            packages[episode] = pkg
            source_inventories[episode] = verify_official_source(pkg)
    for row in rows:
        episode = row['episode_id']
        pkg = packages[episode]
        desc = pkg.json('benchmark/description.json')['evaluation']
        cfg = pkg.json('raw/resolved_config.json')['evaluation']
        if desc['kind'] != 'python-verifier' or not 0 < cfg['timeout_seconds'] <= 120:
            raise ValueError('Unsupported evaluator or unbounded timeout')
        frozen = pkg.json('grading/runtime.json')
        image = cfg['image']
        if not frozen.get('image_id'):
            raise ValueError('Missing frozen scorer image identity')
        if image not in image_records:
            result = subprocess.run(['docker', 'image', 'inspect', image], capture_output=True, check=True, timeout=30)
            inspected = json.loads(result.stdout)[0]
            image_records[image] = {k: inspected.get(k) for k in ('Id', 'RepoDigests', 'Architecture', 'Os')}
        if image_records[image]['Id'] != frozen['image_id']:
            raise ValueError('Local scorer image differs from frozen runtime')
        worker = ROOT / 'L1_sandbox/C3_Experimental_Sandbox/src/c3_sandbox/l1/evaluation/worker.py'
        if sha(worker) != frozen['python_environment']['grader_worker_sha256']:
            raise ValueError('Worker release differs from frozen runtime')
        source = pkg.root / 'benchmark/source'
        entry = source / desc['entrypoint']
        if not entry.is_file():
            raise ValueError('Missing official scorer entrypoint')
        sources[episode] = {'entrypoint': desc['entrypoint'], 'entrypoint_sha256': sha(entry), 'source_files': source_inventories[episode], 'frozen_runtime_sha256': sha(pkg.root / 'grading/runtime.json')}
        snapshot, candidates = candidate_inputs(pkg, row)
        validate_snapshot(pkg.root, snapshot)
    info = subprocess.run(['docker', 'info', '--format', '{{json .}}'], capture_output=True, check=True, timeout=30)
    server = json.loads(info.stdout)
    if server['OSType'] != 'linux':
        raise ValueError('Linux Docker engine required')
    return packages, {'status': 'ready', 'images': image_records, 'docker_server_version': server['ServerVersion'], 'scorer_sources': sources, 'row_count': 20, 'actor_requests': 0, 'concurrency': 1, 'timeout_seconds': 120}


def rank(record):
    c, g = record['candidate'], record['grade']
    score = (g.get('projection') or {}).get('completion_rate')
    if g.get('status') == 'infra_error' or not isinstance(score, (float, int)):
        return None
    return (round(score, 12), bool(c['in_submission']), c['last_written_turn'], -c['tier'], c['candidate_id'])


def summarize(row, point, candidates, evaluated):
    valid = [r for r in evaluated if rank(r) is not None]
    # With no numeric grade, retain the attempted candidate and actual result;
    # it is evidence, not a claimed maximum-scoring selection.
    selected = max(valid, key=rank) if valid else evaluated[0]
    result = {**row, 'restored_directory': str((point / 'restored_workspace').relative_to(ROOT)), 'restored_file_hashes': str((point / 'restored_file_hashes.json').relative_to(ROOT)), 'candidate_status': 'ok' if candidates and valid else 'no_candidate' if not candidates else 'official_grade_unknown', 'selection_status': 'maximum_valid_numeric_grade' if valid else 'unranked_unknown_grades', 'selected_candidate': selected['candidate'] if candidates else None, 'official_snapshot_grade': selected['grade'], 'scorer_exit_code': selected['grade']['process']['container_exit_code'], 'candidate_evaluations': evaluated, 'reviewer': '', 'reviewed_at': '', 'human_conclusion': '', 'missing_for_human_review': ['人工打开文档/图片核查后签署结论']}
    prior = row['prior_auto_snapshot_score']
    now = (result['official_snapshot_grade'] or {}).get('projection', {}).get('completion_rate')
    result['prior_score_comparison'] = {'prior': prior, 'actual_official': now, 'difference': now - float(prior) if prior != 'unknown' and isinstance(now, (int, float)) else None}
    result['l2_no_candidate_convention_score'] = 0.0 if not candidates else None
    error = (selected['grade'].get('official_raw') or {}).get('error') or ''
    matches = [marker for marker in DELIVERY_ERRORS_V1_1 if marker in error]
    result['delivery_evidence'] = {'category': 'no_artifact' if not candidates else 'known_delivery_error_marker' if matches else 'candidate_present', 'rule_version': 'v1.1', 'matched_markers': matches, 'study_outcome_evaluated': False, 'note': 'Marker evidence only; no synthetic episode or study endpoint projection created.'}
    write(point / 'review.json', result)
    return result


def write_sheet(results):
    lines = ['# G3 restored 20-point human review', '', 'All human fields intentionally blank. Official numeric grade remains unknown on official errors. No human PASS or study-endpoint classification is implied.', '', '| # | episode / checkpoint | selected paths | official score | exit | delivery evidence | restored directory | reviewer | reviewed_at | human_conclusion |', '|---|---|---|---|---|---|---|---|---|---|']
    for r in results:
        grade = r['official_snapshot_grade']
        score = grade['projection']['completion_rate']
        paths = (r.get('selected_candidate') or {}).get('paths') or []
        category = r['delivery_evidence']['category']
        if r['delivery_evidence']['matched_markers']:
            category += ': ' + str(grade['official_raw']['error'])
        lines.append(f"| {r['row_id']} | {r['episode_id']} / {r['checkpoint_id']} | {'; '.join(paths) or '(none)'} | {score if score is not None else 'unknown'} | {r['scorer_exit_code']} | {category} | [workspace]({Path(r['restored_directory']).relative_to(OUTPUT.relative_to(ROOT)).as_posix()}) |  |  |  |")
    (OUTPUT / 'G3_20_HUMAN_REVIEW.md').write_text('\n'.join(lines) + '\n')


def prepare_row(row, pkg, out, provenance):
    point = out / f"{row['row_id']:02d}_{row['episode_id']}_{row['checkpoint_id']}"
    point.mkdir()
    snapshot, candidates = candidate_inputs(pkg, row)
    workspace = point / 'restored_workspace'
    restore(pkg.root, snapshot, workspace)
    inventory = []
    for f in snapshot['files']:
        p = workspace / f['path']
        actual = sha(p)
        if actual != f['sha256'] or p.stat().st_size != f['size_bytes']:
            raise ValueError('Restored file mismatch')
        inventory.append({'path': f['path'], 'sha256': actual, 'size_bytes': p.stat().st_size})
    write(point / 'restored_file_hashes.json', inventory)
    write(point / 'candidates.json', candidates)
    desc = pkg.json('benchmark/description.json')['evaluation']
    spec = EvaluationSpec(desc['kind'], pkg.root / 'benchmark/source', desc['entrypoint'], pkg.task_id, desc.get('metadata') or {})
    cfg = dict(pkg.json('raw/resolved_config.json')['evaluation'])
    cfg['image'] = provenance['images'][cfg['image']]['Id']
    evaluated = []
    # Empty official submission is evaluated explicitly; it is not a fabricated grade.
    work = candidates or [{'candidate_id': 'empty_submission', 'paths': [], 'staged': [], 'in_submission': False, 'last_written_turn': 0, 'tier': 0}]
    for i, cand in enumerate(work):
        directory = point / f'candidate_{i:02d}'
        submission = directory / 'submission'
        submission.mkdir(parents=True)
        for name, rel, expected, _ in cand['staged']:
            dst = submission / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(workspace / rel, dst)
            if sha(dst) != expected:
                raise ValueError('Staged candidate hash mismatch')
            dst.chmod(0o644)
        for p in [submission, *submission.rglob('*')]:
            if p.is_dir():
                p.chmod(0o755)
        write(directory / 'invocation.json', {'image_id': cfg['image'], 'command': ['python', '-I', '/opt/l1/grade_worker.py', '--source', '/official', '--entrypoint', spec.entrypoint, '--submission', '/submission', '--output', '/grade'], 'source_root': str(spec.source_root), 'submission': str(submission), 'network': 'none', 'timeout_seconds': cfg['timeout_seconds']})
        grade = DockerEvaluator(cfg).evaluate(spec, submission, directory / 'grading')
        write(directory / 'grade.json', grade)
        evaluated.append({'candidate': cand, 'grade': grade, 'evidence_directory': str(directory.relative_to(ROOT))})
    return summarize(row, point, candidates, evaluated)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    rows = checked_rows(read(METADATA), ROOT)
    packages, provenance = preflight(rows)
    if not args.execute:
        print(json.dumps({'mode': 'plan_only', **provenance}, ensure_ascii=False, indent=2))
        return 0
    OUTPUT.mkdir(exist_ok=False)
    write(OUTPUT / 'preflight.json', provenance)
    write(OUTPUT / 'preparation_identity.json', {'script_sha256': sha(__file__), 'metadata_sha256': sha(METADATA), 'actor_requests': 0})
    results = []
    for row in rows:
        result = prepare_row(row, packages[row['episode_id']], OUTPUT, provenance)
        results.append(result)
        write(OUTPUT / 'g3_20_actual_review.json', {'status': 'prepared_pending_human_review' if len(results) == 20 else 'in_progress', 'actor_requests': 0, 'rows': results})
        print(f"row {row['row_id']}/20 {result['candidate_status']} scorer_exit={result['scorer_exit_code']}", flush=True)
    write_sheet(results)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
