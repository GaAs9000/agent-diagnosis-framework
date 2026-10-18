#!/usr/bin/env python3
"""One-shot E1 for the exact three-segment 004 attempt; never calls an Actor.

Default plans only. ARM64 official grading is a separate, explicitly selected
local exception; facts and defects run only on the approved Linux CPU host.
Input paths in the frozen plan are repository relative for read-only mirrors.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

L2 = Path(__file__).resolve().parents[1]
ROOT = L2.parents[1]
sys.path[:0] = [str(ROOT / 'L1_sandbox/C3_Experimental_Sandbox/src'), str(L2 / 'src')]
from c3_sandbox.l1.package import verify_package
from c3_sandbox.l1.storage import digest, file_digest, read_json
from l2_sandbox.evidence.l1 import L1Package
from l2_sandbox import grades as grading
from l2_sandbox.facts.build import build_facts
from l2_sandbox.defects import build_defect_table
from l2_sandbox.io import L2Error


def load(name):
    spec = importlib.util.spec_from_file_location('chain_' + name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


census = load('build_d0_census')
owned = load('finalize_d0_timeout600')
PREFIX = 'L1_sandbox/C3_Experimental_Sandbox/runs/'
SEGMENTS = (
    (PREFIX + 'v3_qwen36_pi_dev45_20261017/failed_attempts/officeval_004.r02.d0_resume_1792257570352387000',
     'ep_94dde6631ebe4f038867e2a618fd0344', 0, 8),
    (PREFIX + 'v3_qwen36_pi_dev45_r22_20261018/segments/officeval_004.r02',
     'ep_bf1ec1b1deea4884b20c5a3b39d248b7', 7, 21),
    (PREFIX + 'v3_qwen36_pi_context004_20261018/segments/officeval_004.r02',
     'ep_cfdf809de29d4a6fa092e8f288f6e2a6', 21, 30),
)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write('\n')


def jsonl(path, rows):
    with path.open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def analysis_identity():
    identity = census.analysis_identity()
    files = dict(identity['files'])
    for path in (Path(__file__).resolve(), Path(census.__file__), Path(owned.__file__)):
        files[str(path.relative_to(ROOT))] = file_digest(path)
    return {'files': files, 'sha256': digest(files), 'study_rule_version': identity['study_rule_version'],
            'scope': 'analysis_and_official_scorer_only; independent_of_actor_phase_source'}


def snapshot(pkg, cp):
    return pkg.json('states/snapshots/' + cp['snapshot_id'] + '.json')


def workspace_identity(snap):
    return {key: snap[key] for key in ('files', 'directories', 'total_bytes', 'content_sha256')}


def make_plan(input_root=ROOT, max_seconds=1800):
    if not 1 <= max_seconds <= 21600:
        raise ValueError('grading deadline must be between 1 and 21600 seconds')
    packages, records = [], []
    worker = ROOT / 'L1_sandbox/C3_Experimental_Sandbox/src/c3_sandbox/l1/evaluation/worker.py'
    for relative, episode, offset, end in SEGMENTS:
        path = input_root / relative
        index = file_digest(path / 'evidence_index.json')
        seal = verify_package(path, expected_index_sha256=index)
        if not seal['verified']:
            raise ValueError('invalid sealed chain segment: ' + relative)
        pkg = L1Package(path)
        if pkg.episode_id != episode or pkg.task_id != 'officeval_004':
            raise ValueError('unexpected chain episode/task')
        cp0 = pkg.json('checkpoints/cp000000.json')
        if cp0['budget_used']['model_requests'] != offset or pkg.local_turns + offset != end:
            raise ValueError('chain request offset/count differs')
        requests = [event['data'] for event in pkg.events if event['kind'] == 'model.request']
        if [row['request_id'] for row in requests] != [f'{episode}:r{turn:06d}' for turn in range(offset, end)]:
            raise ValueError('chain request IDs are not the exact contiguous suffix')
        if packages:
            parent = packages[-1]
            lineage = pkg.manifest['lineage']
            parent_cp_path = 'checkpoints/' + lineage['parent_checkpoint_id'] + '.json'
            if (lineage['parent_episode_id'] != parent.episode_id
                    or lineage['parent_index_sha256'] != parent.index_sha256
                    or lineage['parent_checkpoint_sha256'] != file_digest(parent.root / parent_cp_path)
                    or pkg.json('lineage/parent_checkpoint.json') != parent.json(parent_cp_path)
                    or workspace_identity(snapshot(pkg, cp0)) != workspace_identity(snapshot(parent, parent.json(parent_cp_path)))):
                raise ValueError('parent lineage or workspace seam differs')
        source = census.source_identity(pkg)
        runtime = pkg.json('official_grade.json')['runtime']
        if runtime['python_environment']['grader_worker_sha256'] != file_digest(worker):
            raise ValueError('historical grader worker differs from analysis scorer')
        checkpoints = []
        for local, cp in pkg.checkpoints().items():
            checkpoints.append({'local_checkpoint_id': cp['checkpoint_id'], 'local_ordinal': local,
                                'request_boundary': cp['budget_used']['model_requests'],
                                'checkpoint_sha256': cp['sha256'], 'snapshot_id': cp['snapshot_id']})
        records.append({'path': relative, 'episode_id': episode, 'index_sha256': index,
                        'inventory_sha256': digest(read_json(path / 'evidence_index.json')['files']),
                        'request_offset': offset, 'request_end': end, 'source': source,
                        'image_id': runtime['image_id'], 'architecture': runtime['architecture'],
                        'checkpoints': checkpoints})
        packages.append(pkg)
    if len({(record['source']['sha256'], record['image_id'], record['architecture']) for record in records}) != 1:
        raise ValueError('chain official source/image identities differ')
    replay = [event['data'] for event in packages[1].events if event['kind'] == 'model.request'][0]
    if replay.get('source') != 'saved_response' or replay.get('http_attempts') != 0:
        raise ValueError('original eighth response must be consumed once without HTTP')
    if packages[-1].manifest['usage']['model_requests_including_parent'] != 30:
        raise ValueError('chain final budget differs from 30')
    final_projection = packages[-1].json('official_grade.json')['projection']
    if (final_projection.get('evaluation_validity') != 'valid'
            or final_projection.get('official_dimensions') != {'dim1_pass': True, 'max_score': 22, 'total_score': 8}):
        raise ValueError('latest official terminal must be the valid 8/22 endpoint')
    plan = {'schema': 'context004-e1-chain-v1', 'segments': records, 'final_request_boundary': 30,
            'analysis': analysis_identity(), 'max_seconds': max_seconds,
            'actor_requests': 0, 'analyst_requests': 0, 'gates_promoted': False,
            'historical_authenticity': 'not_attested; sealed_saved_consistency_only'}
    plan['plan_sha256'] = digest(plan)
    return plan


def validate_plan(plan, input_root):
    if digest({key: value for key, value in plan.items() if key != 'plan_sha256'}) != plan['plan_sha256']:
        raise ValueError('analysis plan seal mismatch')
    if make_plan(input_root, plan['max_seconds']) != plan:
        raise ValueError('frozen analysis source, scorer or chain inputs changed')


def provenance(record, row):
    row = copy.deepcopy(row)
    row['segment_provenance'] = {'path': record['path'], 'episode_id': record['episode_id'],
                                 'index_sha256': record['index_sha256'], 'request_offset': record['request_offset']}
    return row


def annotate_rows(pkg, record, rows):
    """Require corrected grading API; local cp identity never becomes global cp ID."""
    by_id = {cp['local_checkpoint_id']: cp for cp in record['checkpoints']}
    result = []
    for source in rows:
        row = provenance(record, source)
        if row['kind'] == 'checkpoint':
            cp = by_id.get(row.get('checkpoint_id'))
            if cp is None:
                if row.get('checkpoint_id') is not None or row['k'] != record['request_end']:
                    raise ValueError('unknown/synthetic checkpoint identity')
                row.update(actual_checkpoint_present=False, synthetic_terminal_checkpoint=True, anchor_eligible=False)
            else:
                if row['k'] != cp['request_boundary']:
                    raise ValueError('grading API returned local ordinal instead of request boundary')
                row.update(actual_checkpoint_present=True, synthetic_terminal_checkpoint=False, anchor_eligible=False)
                try:
                    pkg.validate_resume_checkpoint(cp['local_ordinal'])
                    row['anchor_eligible'] = True
                except L2Error:
                    pass
        result.append(row)
    return result


def merge_grades(segment_rows, final_boundary=30):
    checkpoints, superseded = {}, []
    terminals = []
    for rows in segment_rows:
        term = [row for row in rows if row['kind'] in ('terminal_submission', 'historical_segment_terminal')]
        if len(term) != 1:
            raise ValueError('each segment requires one terminal score')
        terminals.append(term[0])
        for row in rows:
            if row['kind'] == 'checkpoint':
                key = row['k']
                if key in checkpoints:
                    superseded.append(checkpoints[key])
                checkpoints[key] = row
    if sorted(checkpoints) != list(range(final_boundary + 1)):
        raise ValueError('full chain checkpoint boundaries must be contiguous 0..30')
    if terminals[-1]['k'] != final_boundary:
        raise ValueError('latest terminal does not match full chain')
    if terminals[-1]['kind'] != 'terminal_submission':
        raise ValueError('only latest segment supplies final terminal Y')
    return [terminals[-1], *[checkpoints[k] for k in sorted(checkpoints)]], [*terminals[:-1], *superseded]


def grade_segments(plan, input_root, output, grader=None):
    grader = grader or grading.grade_episode_snapshots
    cache = output / 'cache' / plan['analysis']['sha256']
    all_rows = []
    packages = [L1Package(input_root / record['path']) for record in plan['segments']]
    for number, (record, pkg) in enumerate(zip(plan['segments'], packages)):
        rows = annotate_rows(pkg, record, grader(pkg, cache_dir=cache, extra_runs=packages,
                                                verify_terminal_submission=number == 2))
        if number < 2:
            official = pkg.json('official_grade.json')
            for row in rows:
                if row['kind'] == 'terminal_submission':
                    row.update(kind='historical_segment_terminal',
                               score=official.get('projection', {}).get('completion_rate'),
                               historical_projection=official.get('projection'),
                               official_grade_ref={'path': record['path'] + '/official_grade.json',
                                                   'sha256': file_digest(pkg.root / 'official_grade.json')})
        jsonl(output / f'segment_{number}_grades.jsonl', rows)
        all_rows.append(rows)
    rows, history = merge_grades(all_rows)
    jsonl(output / 'grades.jsonl', rows)
    jsonl(output / 'superseded_grades.jsonl', history)


def artifact_receipt(output, names, plan):
    return {'plan_sha256': plan['plan_sha256'], 'actor_requests': 0, 'analyst_requests': 0,
            'files': {name: file_digest(output / name) for name in names}}


def input_inventory(plan, input_root):
    return {record['path']: {'index_sha256': file_digest(input_root / record['path'] / 'evidence_index.json'),
                             'files': read_json(input_root / record['path'] / 'evidence_index.json')['files']}
            for record in plan['segments']}


def grade_child(plan, input_root, output):
    if platform.system() != 'Darwin' or platform.machine() != 'arm64' or not (output / 'grade_once.lock').is_dir():
        raise ValueError('grade child requires its reserved local ARM64 grading stage')
    validate_plan(plan, input_root)
    before = input_inventory(plan, input_root)
    save(output / 'input_inventory_before.json', before)
    owner = digest({'plan': plan['plan_sha256'], 'output': str(output.resolve())})
    with owned.owned_e1_containers(output, owner), census.preserve_grading_evidence(output / 'official_evidence', plan['segments'][0]['image_id']):
        grade_segments(plan, input_root, output)
    validate_plan(plan, input_root)
    after = input_inventory(plan, input_root)
    if before != after:
        raise ValueError('input file inventory changed during official grading')
    save(output / 'input_inventory_after.json', after)
    names = ['grades.jsonl', 'superseded_grades.jsonl', 'input_inventory_before.json', 'input_inventory_after.json',
             *[f'segment_{n}_grades.jsonl' for n in range(3)]]
    names += [str(path.relative_to(output)) for path in sorted((output / 'official_evidence').rglob('*')) if path.is_file()]
    receipt = artifact_receipt(output, names, plan)
    receipt['container_owner'] = owner
    save(output / 'grade_receipt.json', receipt)


def grade_official(plan, plan_path, input_root, output):
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise ValueError('official ARM64 grading is restricted to the authorized local Mac exception')
    validate_plan(plan, input_root)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'grade_once.lock').mkdir()
    owner = digest({'plan': plan['plan_sha256'], 'output': str(output.resolve())})
    failure = None
    try:
        with (output / 'grade.log').open('xb') as log:
            owned.bounded([sys.executable, str(Path(__file__).resolve()), '--grade-child', '--plan', str(plan_path.resolve()),
                           '--input-root', str(input_root.resolve()), '--output', str(output.resolve())],
                          time.time() + plan['max_seconds'], plan['max_seconds'],
                          stdout=log, stderr=log, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '4'})
    except BaseException as exc:
        failure = exc
    finally:
        cleanup = owned.cleanup_e1(output, owner, time.time() + 60)
    if failure is not None:
        raise failure
    if cleanup['status'] != 'confirmed_empty':
        raise RuntimeError('official scorer container cleanup is unconfirmed')
    save(output / 'grading_complete.json', artifact_receipt(output, ['grade_receipt.json', 'e1/cleanup.json'], plan))


def merge_facts(plan, packages, builder=build_facts):
    kept, historical = [], []
    for number, (record, pkg) in enumerate(zip(plan['segments'], packages)):
        stop = plan['segments'][number + 1]['request_offset'] if number < 2 else 30
        for fact in builder(pkg):
            row = provenance(record, fact)
            target = kept if record['request_offset'] < row['turn'] <= stop else historical
            target.append(row)
    return kept, historical


def analyze_cpu(plan, input_root, grade_dir, output):
    if platform.system() != 'Linux' or Path.home() != Path('/home/jiasheren2'):
        raise ValueError('facts/defects analysis must run on the approved AML Linux CPU host')
    validate_plan(plan, input_root)
    receipt = read_json(grade_dir / 'grade_receipt.json')
    completed = read_json(grade_dir / 'grading_complete.json')
    if completed['plan_sha256'] != plan['plan_sha256'] or any(file_digest(grade_dir / name) != sha for name, sha in completed['files'].items()):
        raise ValueError('official grading completion receipt differs')
    if receipt['plan_sha256'] != plan['plan_sha256'] or any(file_digest(grade_dir / name) != sha for name, sha in receipt['files'].items()):
        raise ValueError('official grade artifacts differ from frozen receipt')
    cleanup = read_json(grade_dir / 'e1/cleanup.json')
    if cleanup['status'] != 'confirmed_empty' or cleanup['owner'] != receipt['container_owner']:
        raise ValueError('official scorer cleanup is not confirmed')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'analyze_once.lock').mkdir()
    packages = [L1Package(input_root / record['path']) for record in plan['segments']]
    facts, historical = merge_facts(plan, packages)
    jsonl(output / 'facts.jsonl', facts)
    jsonl(output / 'superseded_facts.jsonl', historical)
    rows = read_rows(grade_dir / 'grades.jsonl')
    # Latest package supplies final config/Y; merged rows/facts supply the full attempt.
    defects = build_defect_table(packages[-1], rows, facts,
                                 grades_path=grade_dir / 'grades.jsonl', facts_path=output / 'facts.jsonl')
    defects['chain_plan_sha256'] = plan['plan_sha256']
    save(output / 'defects.json', defects)
    validate_plan(plan, input_root)
    save(output / 'analysis_receipt.json', artifact_receipt(output, ['facts.jsonl', 'superseded_facts.jsonl', 'defects.json'], plan))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--plan-only', action='store_true')
    modes.add_argument('--grade-official', action='store_true')
    modes.add_argument('--analyze-cpu', action='store_true')
    modes.add_argument('--grade-child', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--input-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--grade-dir', type=Path)
    parser.add_argument('--max-seconds', type=int, default=1800)
    args = parser.parse_args()
    for relative, *_ in SEGMENTS:
        source = (args.input_root / relative).resolve()
        destination = args.output.resolve()
        if destination.is_relative_to(source) or source.is_relative_to(destination):
            raise ValueError('output must be separate from immutable input packages')
    if not (args.grade_official or args.analyze_cpu or args.grade_child):
        plan = make_plan(args.input_root, args.max_seconds)
        save(args.output / 'plan.json', plan)
        print(json.dumps({'mode': 'plan_only', 'plan_sha256': plan['plan_sha256'], 'actor_requests': 0}))
        return
    if args.plan is None:
        parser.error('execution requires an explicit frozen --plan')
    plan = read_json(args.plan)
    if args.grade_official:
        grade_official(plan, args.plan, args.input_root, args.output)
    elif args.grade_child:
        grade_child(plan, args.input_root, args.output)
    else:
        analyze_cpu(plan, args.input_root, args.grade_dir or args.output, args.output)


if __name__ == '__main__':
    main()
