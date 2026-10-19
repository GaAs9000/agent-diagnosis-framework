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
from l2_sandbox.v3_execution import verify_seal

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
    for rel in ('__init__.py', 'config.py', 'io.py', 'grades.py', 'defects.py', 'predicates.py', 'v3_execution.py'):
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


def cohort_identity(cohort_path):
    """The sealed batch ledger names every formal root, rather than a directory glob."""
    path = Path(cohort_path)
    value = read_json(path)
    verify_seal(value)
    expected = {f'{task}.r{rep:02d}' for task in DEV for rep in range(5)}
    records = value.get('slots', {})
    if not isinstance(records, dict) or set(records) != expected:
        raise ValueError('cohort must contain exactly DEV9 x 5 slots')
    for label, record in records.items():
        if (record.get('label') != label or record.get('task_id') != label.split('.')[0]
                or record.get('status') != 'completed' or record.get('study_valid') is not True):
            raise ValueError(f'{label}: cohort slot is not a valid completed root')
        for key in ('episode_id', 'index_file_sha256', 'manifest_sha256', 'config_sha256'):
            if not isinstance(record.get(key), str) or not record[key]:
                raise ValueError(f'{label}: cohort missing {key}')
    return {'path': str(path.resolve()), 'sha256': file_digest(path),
            'plan_sha256': value.get('plan_sha256'), 'slots': records}


def preflight(runs_dir=RUNS, cohort_path=None, *, inspect_images=True):
    """Read every expected slot and inspect existing images before any scoring."""
    runs_dir = Path(runs_dir)
    cohort = cohort_identity(cohort_path) if cohort_path is not None else None
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
            record = cohort['slots'][label] if cohort else None
            if record and (index != record['index_file_sha256']
                           or file_digest(path / 'manifest.json') != record['manifest_sha256']):
                raise ValueError(f'{label}: package differs from sealed cohort index/manifest')
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
            if record and (pkg.episode_id != record['episode_id']
                           or digest(config) != record['config_sha256']):
                raise ValueError(f'{label}: package differs from sealed cohort episode/config')
            if config['model']['model'] != 'Qwen/Qwen3.6-27B' or config['agent']['harness'] != 'pi':
                raise ValueError(f'{label}: Actor/harness identity mismatch')
            source = source_identity(pkg)
            original = pkg.json('official_grade.json')['runtime']
            image = config['evaluation']['image']
            image_id = original['image_id']
            worker = file_digest(L1_ROOT / 'src/c3_sandbox/l1/evaluation/worker.py')
            if original['python_environment']['grader_worker_sha256'] != worker:
                raise ValueError(f'{label}: frozen worker differs from current scorer')
            if inspect_images and image not in images:
                proc = subprocess.run(['docker', 'image', 'inspect', image], capture_output=True, check=True, timeout=30)
                inspected = json.loads(proc.stdout)[0]
                images[image] = {'image_id': inspected['Id'], 'os': inspected['Os'], 'worker_sha256': worker}
            if inspect_images and (images[image]['image_id'] != image_id or images[image]['os'] != 'linux'):
                raise ValueError(f'{label}: current image differs from sealed scorer image')
            row.update(status='ready', source=source, image=image, image_id=image_id)
            row['binding'] = {'schema': SCHEMA, 'parent_index_sha256': index,
                              'analysis_sha256': analysis['sha256'], 'source_sha256': source['sha256'],
                              'image_id': image_id, 'study_rule_version': RULE_VERSION}
            if cohort:
                row['binding'].update(cohort_sha256=cohort['sha256'], episode_id=pkg.episode_id,
                                      config_sha256=record['config_sha256'],
                                      manifest_sha256=record['manifest_sha256'])
    if cohort and any(row['status'] != 'ready' for row in slots):
        raise ValueError('sealed prospective cohort requires all 45 valid, verified roots')
    return {'schema': SCHEMA, 'analysis': analysis, 'images': images, 'slots': slots,
            'cohort': {k: v for k, v in cohort.items() if k != 'slots'} if cohort else None,
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


def grade_receipt(output, binding, validation):
    """Validate portable grading evidence after copying the batch to the server."""
    path = output / 'grading_receipt.json'
    if not path.is_file():
        return None
    receipt = read_json(path)
    if receipt.get('binding') != binding or receipt.get('status') != 'grades_complete':
        return None
    grades = output / 'grades.jsonl'
    if not grades.is_file() or file_digest(grades) != receipt.get('grades_sha256'):
        return None
    inventory = receipt.get('official_evidence_inventory_relative')
    if not isinstance(inventory, dict):
        return None
    count = receipt.get('official_scorer_invocations')
    if type(count) is not int or count < 0:
        return None
    if count == 0 and not inventory:
        if (receipt.get('zero_invocations_reason') != 'all_snapshot_grades_reused_from_bound_cache'
                or receipt.get('cache_entries_before', 0) <= 0):
            return None
    invocation_dirs = set()
    for name, sha in inventory.items():
        relative = Path(name)
        if (relative.is_absolute() or '..' in relative.parts
                or not relative.parts or relative.parts[0] != 'official_evidence'):
            raise ValueError('grading evidence path escapes validation directory')
        source = validation / relative
        if not source.is_file() or file_digest(source) != sha:
            return None
        if relative.name == 'invocation.json' and relative.parent.name.startswith('invocation_'):
            invocation_dirs.add(relative.parent)
    complete = 0
    for directory in invocation_dirs:
        result_name = str(directory / 'result.json')
        if (result_name in inventory
                and read_json(validation / directory / 'invocation.json').get('status') == 'returned'):
            complete += 1
    if count > complete:
        return None
    return receipt


def bind_frozen_plan(plan, frozen):
    """Relocation may change paths, but never the per-slot evidence bindings."""
    if (plan.get('analysis') != frozen.get('analysis')
            or (plan.get('cohort') or {}).get('sha256') != (frozen.get('cohort') or {}).get('sha256')):
        raise ValueError('frozen plan analysis/cohort differs')
    current = {row['slot']: row for row in plan['slots']}
    previous = {row['slot']: row for row in frozen['slots']}
    if set(current) != set(previous) or len(previous) != len(frozen['slots']):
        raise ValueError('frozen plan slot set differs')
    for label, row in current.items():
        if (row.get('binding') != previous[label].get('binding')
                or row.get('status') != previous[label].get('status')
                or row.get('source') != previous[label].get('source')):
            raise ValueError(f'{label}: frozen plan binding differs')
    return plan


def execute(plan, diag_dir=DIAG, validation=VALIDATION, *, stage='all'):
    if stage not in ('all', 'grades', 'analysis'):
        raise ValueError('unknown census stage')
    if plan.get('cohort') and stage != 'grades' and sys.platform == 'darwin':
        raise ValueError('prospective facts/defects must run on the lab server; use --stage grades here')
    diag_dir, validation = Path(diag_dir), Path(validation)
    start = time.monotonic()
    completed = []
    for slot in plan['slots']:
        if slot['status'] != 'ready':
            continue
        output = diag_dir / slot['slot']
        can_reuse = stage != 'grades' and reusable(output, slot['binding'])
        if can_reuse and plan.get('cohort'):
            graded = grade_receipt(output, slot['binding'], validation)
            final = read_json(output / 'receipt.json')
            expected_inventory = {str(validation / p): sha for p, sha in
                                  (graded or {}).get('official_evidence_inventory_relative', {}).items()}
            can_reuse = (graded is not None
                         and final.get('grading_receipt_sha256') == file_digest(output / 'grading_receipt.json')
                         and final.get('official_evidence_inventory') == expected_inventory)
        if can_reuse:
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
        t0 = time.monotonic()
        graded = grade_receipt(output, slot['binding'], validation)
        if graded is None:
            if stage == 'analysis':
                raise ValueError(f"{slot['slot']}: valid grading receipt/evidence required before analysis")
            prior = len(list((cache / 'grades').glob('*.json')))
            with preserve_grading_evidence(evidence, slot['image_id']) as invocations:
                rows = annotate_checkpoints(pkg, grading.grade_episode_snapshots(pkg, cache_dir=cache))
            write_jsonl(output / 'grades.jsonl', rows)
            graded = {'schema': SCHEMA, 'status': 'grades_complete', 'binding': slot['binding'],
                      'grades_sha256': file_digest(output / 'grades.jsonl'),
                      'official_scorer_invocations': len(invocations),
                      'cache_entries_before': prior,
                      'zero_invocations_reason': 'all_snapshot_grades_reused_from_bound_cache'
                          if not invocations and prior > 0 else None,
                      'cache_entries_after': len(list((cache / 'grades').glob('*.json'))),
                      'official_evidence_inventory_relative': {
                          str(p.relative_to(validation)): file_digest(p)
                          for p in sorted(evidence.rglob('*')) if p.is_file()},
                      'actor_requests': 0}
            write_json(output / 'grading_receipt.json', graded)
            if grade_receipt(output, slot['binding'], validation) is None:
                raise ValueError('grading evidence inventory does not support scorer invocation count')
        if stage == 'grades':
            after = verify_package(pkg.root, expected_index_sha256=slot['parent_index_sha256'])
            if not after['verified'] or analysis_identity()['sha256'] != plan['analysis']['sha256']:
                raise ValueError('parent or analysis changed during grading')
            completed.append({'slot': slot['slot'], 'status': 'grades_complete'})
            print(json.dumps(completed[-1]), flush=True)
            continue
        rows = [json.loads(line) for line in (output / 'grades.jsonl').read_text().splitlines() if line.strip()]
        facts = build_facts(pkg)
        write_jsonl(output / 'facts.jsonl', facts)
        defects = build_defect_table(pkg, rows, facts, grades_path=output / 'grades.jsonl', facts_path=output / 'facts.jsonl')
        write_json(output / 'defects.json', defects)
        after = verify_package(pkg.root, expected_index_sha256=slot['parent_index_sha256'])
        if not after['verified'] or analysis_identity()['sha256'] != plan['analysis']['sha256']:
            raise ValueError('parent or analysis changed during grading')
        receipt = {'schema': SCHEMA, 'status': 'complete', 'binding': slot['binding'],
                   'parent': slot['parent'], 'source': slot['source'],
                   'grading_receipt_sha256': file_digest(output / 'grading_receipt.json'),
                   'official_evidence_dir': str(evidence), 'cache_dir': str(cache),
                   'official_evidence_inventory': {
                       str(validation / p): sha for p, sha in graded['official_evidence_inventory_relative'].items()},
                   'official_scorer_invocations': graded['official_scorer_invocations'],
                   'cache_entries_before': graded['cache_entries_before'],
                   'cache_entries_after': graded['cache_entries_after'],
                   'cache_hits': None, 'cache_hits_note': 'existing helper does not expose cache-hit counts',
                   'actor_requests': 0, 'elapsed_seconds': time.monotonic() - t0,
                   'synthetic_terminal_rows': sum(bool(r.get('synthetic_terminal_checkpoint')) for r in rows),
                   'qualified_anchor_rows': sum(bool(r.get('anchor_eligible')) for r in rows),
                   'outputs': {name: file_digest(output / name) for name in ('facts.jsonl', 'grades.jsonl', 'defects.json')}}
        write_json(output / 'receipt.json', receipt)
        completed.append({'slot': slot['slot'], 'reused': False,
                          'official_scorer_invocations': graded['official_scorer_invocations']})
        print(json.dumps(completed[-1]), flush=True)
    summary = {**plan, 'mode': 'executed', 'stage': stage, 'completed': completed,
               'completed_count': len(completed), 'full_45_ready': stage != 'grades' and len(completed) == 45,
               'grades_45_ready': len(completed) == 45,
               'elapsed_seconds': time.monotonic() - start}
    write_json(validation / ('grading_manifest.json' if stage == 'grades' else 'manifest.json'), summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--plan-only', action='store_true')
    parser.add_argument('--runs', type=Path, help='sealed episode root (DEV9 x 5)')
    parser.add_argument('--cohort', type=Path, help='sealed prospective batch cohort.json')
    parser.add_argument('--diagnoses', '--output', type=Path, default=None,
                        help='per-slot facts/grades/defects output directory')
    parser.add_argument('--validation', type=Path, default=None, help='plans, raw scoring evidence and cache')
    parser.add_argument('--stage', choices=('all', 'grades', 'analysis'), default='all',
                        help='grades: Docker only; analysis: server facts/defects, no Docker')
    parser.add_argument('--plan', type=Path, help='frozen grading plan to verify after server relocation')
    args = parser.parse_args()
    validation = args.validation if args.validation is not None else VALIDATION
    diagnoses = args.diagnoses if args.diagnoses is not None else DIAG
    if args.cohort and (args.runs is None or args.diagnoses is None or args.validation is None):
        parser.error('--cohort requires explicit --runs, --diagnoses and --validation to isolate old outputs')
    if args.stage == 'analysis' and args.cohort and args.plan is None:
        parser.error('prospective --stage analysis requires --plan from the grading host')
    if args.runs is None and args.cohort is None and args.stage != 'analysis':
        plan = preflight()
    else:
        plan = preflight(args.runs if args.runs is not None else RUNS, args.cohort,
                         inspect_images=args.stage != 'analysis')
    if args.plan:
        bind_frozen_plan(plan, read_json(args.plan))
    write_json(validation / ('analysis_plan.json' if args.stage == 'analysis' else 'plan.json'), plan)
    print(json.dumps({k: plan[k] for k in ('study_valid', 'sealed_verified', 'expected_slots', 'actor_requests')}), flush=True)
    if args.execute:
        if args.runs is None and args.cohort is None and args.stage == 'all' and args.diagnoses is None and args.validation is None:
            execute(plan)
        else:
            execute(plan, diagnoses, validation, stage=args.stage)


if __name__ == '__main__':
    main()
