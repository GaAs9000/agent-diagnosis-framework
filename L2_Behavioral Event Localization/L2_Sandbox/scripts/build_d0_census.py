#!/usr/bin/env python3
"""Plan or execute the sealed DEV-only E1 census; never invokes an Actor."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

L2_ROOT = Path(__file__).resolve().parents[1]
ROOT = L2_ROOT.parents[1]
L1_ROOT = ROOT / 'L1_sandbox/C3_Experimental_Sandbox'
sys.path[:0] = [str(L1_ROOT / 'src'), str(L2_ROOT / 'src')]

from c3_sandbox.l1.package import verify_package
from c3_sandbox.l1.evaluation.study import RULE_VERSION
from c3_sandbox.l1.storage import file_digest
from l2_sandbox import grades as grading
from l2_sandbox.defects import build_defect_table
from l2_sandbox.evidence.l1 import L1Package
from l2_sandbox.facts.build import build_facts
from l2_sandbox.io import L2Error, digest, read_json, write_json, write_jsonl

DEV = tuple(f'officeval_{n}' for n in ('003', '004', '010', '033', '050', '060', '070', '085', '090'))
RUNS = L1_ROOT / 'runs/v3_qwen36_pi_dev45_20261017/episodes'
DIAG = L2_ROOT / 'diagnoses/d0_20261018'
VALIDATION = L2_ROOT / 'validation/d0_20261018/census'
SCHEMA = 'd0_dev_census_v1'


def analysis_identity():
    """Bind the analysis and scorer code, excluding unrelated Actor controllers."""
    paths = set()
    for directory in ('facts', 'evidence', 'predicates'):
        paths.update((L2_ROOT / f'src/l2_sandbox/{directory}').rglob('*.py'))
    for rel in ('__init__.py', 'config.py', 'io.py', 'grades.py', 'defects.py', 'predicates.py'):
        path = L2_ROOT / 'src/l2_sandbox' / rel
        if path.is_file():
            paths.add(path)
    for directory in ('evaluation', 'recording', 'evidence', 'schema'):
        paths.update((L1_ROOT / f'src/c3_sandbox/l1/{directory}').rglob('*.py'))
    for rel in ('package.py', 'storage.py', 'core/contracts.py'):
        paths.add(L1_ROOT / 'src/c3_sandbox/l1' / rel)
    paths.add(Path(__file__).resolve())
    files = {str(p.relative_to(ROOT)): file_digest(p) for p in sorted(paths)}
    return {'sha256': digest(files), 'files': files, 'study_rule_version': RULE_VERSION}


def source_identity(pkg):
    paths = sorted((pkg.root / 'benchmark/source').rglob('*'))
    files = {str(p.relative_to(pkg.root)): file_digest(p) for p in paths if p.is_file()}
    if not files:
        raise ValueError('empty sealed official source')
    desc = pkg.json('benchmark/description.json')
    entry = 'benchmark/source/' + desc['evaluation']['entrypoint']
    if entry not in files or files[entry] != desc['provenance']['verifier_sha256']:
        raise ValueError('sealed verifier identity differs from benchmark provenance')
    return {'sha256': digest(files), 'entrypoint': entry, 'entrypoint_sha256': files[entry]}


def preflight(runs_dir=RUNS):
    """Read every expected slot and inspect existing images before any scoring."""
    analysis = analysis_identity()
    slots, images = [], {}
    for task in DEV:
        for rep in range(5):
            label = f'{task}.r{rep:02d}'
            path = runs_dir / label
            row = {'slot': label, 'task_id': task, 'parent': str(path)}
            slots.append(row)
            if not (path / 'manifest.json').is_file():
                row.update(status='unsealed_or_missing', study_valid=False)
                continue
            if not (path / 'evidence_index.json').is_file():
                row.update(status='unsealed_or_missing', study_valid=False)
                continue
            index = file_digest(path / 'evidence_index.json')
            seal = verify_package(path, expected_index_sha256=index)
            row.update(parent_index_sha256=index, seal=seal)
            if not seal.get('verified'):
                row.update(status='seal_invalid', study_valid=False)
                continue
            pkg = L1Package(path)
            if pkg.task_id != task or pkg.manifest.get('lineage'):
                raise ValueError(f'{label}: DEV root identity mismatch')
            outcome = pkg.outcome()
            row.update(outcome=outcome, study_valid=outcome['valid'], episode_id=pkg.episode_id)
            if not outcome['valid']:
                row['status'] = 'study_invalid_preserved'
                continue
            config = pkg.json('raw/resolved_config.json')
            if config['model']['model'] != 'Qwen/Qwen3.6-27B' or config['agent']['harness'] != 'pi':
                raise ValueError(f'{label}: Actor/harness identity mismatch')
            source = source_identity(pkg)
            original = pkg.json('official_grade.json')['runtime']
            image = config['evaluation']['image']
            image_id = original['image_id']
            worker = file_digest(L1_ROOT / 'src/c3_sandbox/l1/evaluation/worker.py')
            if original['python_environment']['grader_worker_sha256'] != worker:
                raise ValueError(f'{label}: frozen worker differs from current scorer')
            if image not in images:
                proc = subprocess.run(['docker', 'image', 'inspect', image], capture_output=True, check=True, timeout=30)
                inspected = json.loads(proc.stdout)[0]
                images[image] = {'image_id': inspected['Id'], 'os': inspected['Os'], 'worker_sha256': worker}
            if images[image]['image_id'] != image_id or images[image]['os'] != 'linux':
                raise ValueError(f'{label}: current image differs from sealed scorer image')
            row.update(status='ready', source=source, image=image, image_id=image_id)
            row['binding'] = {'schema': SCHEMA, 'parent_index_sha256': index,
                              'analysis_sha256': analysis['sha256'], 'source_sha256': source['sha256'],
                              'image_id': image_id, 'study_rule_version': RULE_VERSION}
    return {'schema': SCHEMA, 'analysis': analysis, 'images': images, 'slots': slots,
            'expected_slots': 45, 'study_valid': sum(r['study_valid'] for r in slots),
            'sealed_verified': sum(bool(r.get('seal', {}).get('verified')) for r in slots),
            'actor_requests': 0, 'max_concurrent_scorers': 1, 'cpus_per_scorer': 2,
            'scientific_efficacy': 'not_tested', 'd0_frozen': False}


def annotate_checkpoints(pkg, rows):
    checkpoints = pkg.checkpoints()
    for row in rows:
        if row['kind'] != 'checkpoint':
            continue
        cp = checkpoints.get(int(row['k']))
        row['actual_checkpoint_present'] = cp is not None
        row['synthetic_terminal_checkpoint'] = cp is None and int(row['k']) == pkg.turns
        row['cold_resume_eligible'] = False
        if cp is not None:
            try:
                pkg.validate_resume_checkpoint(int(row['k']))
                row['cold_resume_eligible'] = True
            except L2Error:
                pass
        row['anchor_eligible'] = row['cold_resume_eligible']
    return rows


def reusable(output, binding):
    receipt_path = output / 'receipt.json'
    if not receipt_path.is_file():
        return False
    receipt = read_json(receipt_path)
    if receipt.get('binding') != binding or receipt.get('status') != 'complete':
        return False
    files = receipt.get('outputs', {})
    return (set(files) == {'facts.jsonl', 'grades.jsonl', 'defects.json'} and all(
        (output / name).is_file() and file_digest(output / name) == sha for name, sha in files.items())
        and all(Path(path).is_file() and file_digest(Path(path)) == sha
                for path, sha in receipt.get('official_evidence_inventory', {}).items()))


@contextmanager
def preserve_grading_evidence(root, image_id):
    """Keep the existing scorer, retaining its temporary raw files before removal."""
    original = grading.DockerEvaluator
    invocations = []

    class PreservingEvaluator(original):
        def __init__(self, config):
            super().__init__({**config, 'image': image_id})

        def evaluate(self, spec, submission, output):
            invocation = root / f'invocation_{len(list(root.glob("invocation_*"))):06d}'
            invocation.mkdir(parents=True, exist_ok=False)
            start = time.monotonic()
            record = {'task_id': spec.task_id, 'entrypoint': spec.entrypoint,
                      'image_id': image_id, 'status': 'started'}
            invocations.append(record)
            write_json(invocation / 'invocation.json', record)
            try:
                result = super().evaluate(spec, submission, output)
                write_json(invocation / 'result.json', result)
                record.update(status='returned', official_raw_sha256=result.get('official_raw_sha256'),
                              projection=result.get('projection'))
                if result.get('status') == 'infra_error':
                    raise RuntimeError(f'official scorer infrastructure failure: {result.get("error")}')
                return result
            finally:
                if output.exists():
                    shutil.copytree(output, invocation / 'grading')
                record['elapsed_seconds'] = time.monotonic() - start
                write_json(invocation / 'invocation.json', record)

    grading.DockerEvaluator = PreservingEvaluator
    try:
        yield invocations
    finally:
        grading.DockerEvaluator = original


def execute(plan, diag_dir=DIAG, validation=VALIDATION):
    start = time.monotonic()
    completed = []
    for slot in plan['slots']:
        if slot['status'] != 'ready':
            continue
        output = diag_dir / slot['slot']
        if reusable(output, slot['binding']):
            completed.append({'slot': slot['slot'], 'reused': True})
            continue
        pkg = L1Package(Path(slot['parent']))
        before = verify_package(pkg.root, expected_index_sha256=slot['parent_index_sha256'])
        if not before['verified'] or analysis_identity()['sha256'] != plan['analysis']['sha256']:
            raise ValueError('parent or analysis changed after preflight')
        output.mkdir(parents=True, exist_ok=True)
        namespace = digest({k: v for k, v in slot['binding'].items() if k != 'parent_index_sha256'})
        cache = validation / 'cache' / namespace
        evidence = validation / 'official_evidence' / namespace
        prior = len(list((cache / 'grades').glob('*.json')))
        t0 = time.monotonic()
        facts = build_facts(pkg)
        write_jsonl(output / 'facts.jsonl', facts)
        with preserve_grading_evidence(evidence, slot['image_id']) as invocations:
            rows = annotate_checkpoints(pkg, grading.grade_episode_snapshots(pkg, cache_dir=cache))
        write_jsonl(output / 'grades.jsonl', rows)
        defects = build_defect_table(pkg, rows, facts, grades_path=output / 'grades.jsonl', facts_path=output / 'facts.jsonl')
        write_json(output / 'defects.json', defects)
        after = verify_package(pkg.root, expected_index_sha256=slot['parent_index_sha256'])
        if not after['verified'] or analysis_identity()['sha256'] != plan['analysis']['sha256']:
            raise ValueError('parent or analysis changed during grading')
        receipt = {'schema': SCHEMA, 'status': 'complete', 'binding': slot['binding'],
                   'parent': slot['parent'], 'source': slot['source'],
                   'official_evidence_dir': str(evidence), 'cache_dir': str(cache),
                   'official_evidence_inventory': {str(p): file_digest(p) for p in sorted(evidence.rglob('*')) if p.is_file()},
                   'official_scorer_invocations': len(invocations),
                   'cache_entries_before': prior, 'cache_entries_after': len(list((cache / 'grades').glob('*.json'))),
                   'cache_hits': None, 'cache_hits_note': 'existing helper does not expose cache-hit counts',
                   'actor_requests': 0, 'elapsed_seconds': time.monotonic() - t0,
                   'synthetic_terminal_rows': sum(bool(r.get('synthetic_terminal_checkpoint')) for r in rows),
                   'qualified_anchor_rows': sum(bool(r.get('anchor_eligible')) for r in rows),
                   'outputs': {name: file_digest(output / name) for name in ('facts.jsonl', 'grades.jsonl', 'defects.json')}}
        write_json(output / 'receipt.json', receipt)
        completed.append({'slot': slot['slot'], 'reused': False, 'official_scorer_invocations': len(invocations)})
        print(json.dumps(completed[-1]), flush=True)
    summary = {**plan, 'mode': 'executed', 'completed': completed,
               'completed_count': len(completed), 'full_45_ready': len(completed) == 45,
               'elapsed_seconds': time.monotonic() - start}
    write_json(validation / 'manifest.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    plan = preflight()
    write_json(VALIDATION / 'plan.json', plan)
    print(json.dumps({k: plan[k] for k in ('study_valid', 'sealed_verified', 'expected_slots', 'actor_requests')}), flush=True)
    if args.execute:
        execute(plan)


if __name__ == '__main__':
    main()
