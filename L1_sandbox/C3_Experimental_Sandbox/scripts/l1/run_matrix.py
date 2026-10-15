#!/usr/bin/env python3
"""Bounded-concurrency L1 baseline matrix runner.

Executes tasks x replicates using the existing L1 orchestrator.run() entrypoint
and writes summary.csv + summary.json for M3.1 task selection.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import csv
import json
import os
from pathlib import Path
import sys

for _p in [Path('.env'), Path(__file__).resolve().parents[2] / '.env']:
    if _p.is_file():
        for _line in _p.read_text().splitlines():
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _k, _v = _line.split('=', 1)
                _k, _v = _k.strip(), _v.strip().strip('"\'')
                if _k not in os.environ:
                    os.environ[_k] = _v

from c3_sandbox.l1.evaluation.study import study_outcome
from c3_sandbox.l1.core.config import resolve
from c3_sandbox.l1.core.orchestrator import run
from c3_sandbox.l1.evidence.live import scan
from c3_sandbox.l1.storage import read_json, write_json

COLUMNS = [
    'task_id',
    'replicate',
    'episode_id',
    'termination',
    'execution_validity',
    'evaluation_validity',
    'completion_rate',
    'dim1',
    'score_eligible',
    'score_classification',
    'model_requests',
    'eligible_checkpoints',
    'wall_seconds',
    'model_version',
    'retries',
    'config_sha256',
    'study_valid', 'study_outcome_y', 'zero_reason', 'study_classification',
    'endpoint_version', 'native_evaluation_validity',
    'native_projection_evaluation_validity', 'native_status', 'native_error',
]


def summarize_episode(task_id: str, replicate: int, package: Path) -> dict:
    manifest = read_json(package / 'manifest.json')
    grade = read_json(package / 'official_grade.json') if (package / 'official_grade.json').exists() else {}
    events = scan(package / 'trajectory.jsonl')['events'] if (package / 'trajectory.jsonl').exists() else []

    status = manifest.get('status', {})
    exec_valid = status.get('execution_validity')
    eval_valid = status.get('evaluation_validity')
    proj = grade.get('projection') or {}
    raw = grade.get('official_raw') if isinstance(grade.get('official_raw'), dict) else {}

    # One research denominator shared with L2. Native rubric validity remains
    # separately visible: a healthy missing delivery is a study zero.
    study = study_outcome(package)
    score_eligible = study['study_valid']
    completion_rate = study['study_outcome_y']
    score_classification = study['study_classification']
    dim1 = (raw.get('dim1_pass') if 'dim1_pass' in raw else
            (proj.get('official_dimensions') or {}).get('passed')) if score_eligible else None

    idx_path = package / 'checkpoints/index.json'
    eligible_cps = 0
    if idx_path.exists():
        cps = read_json(idx_path).get('checkpoints', [])
        eligible_cps = sum(1 for c in cps if c.get('eligible'))

    versions = [e['data'].get('reported_model_version') for e in events if e.get('kind') == 'model.parsed' and e['data'].get('reported_model_version')]
    retries = sum(1 for e in events if e.get('kind') == 'model.retry')

    return {
        'task_id': task_id,
        'replicate': replicate,
        'episode_id': manifest.get('episode_id'),
        'termination': status.get('termination'),
        'execution_validity': exec_valid,
        'evaluation_validity': eval_valid,
        'completion_rate': completion_rate,
        'dim1': dim1,
        'score_eligible': score_eligible,
        'score_classification': score_classification,
        'model_requests': (manifest.get('counts') or {}).get('model_requests', 0),
        'eligible_checkpoints': eligible_cps,
        'wall_seconds': round(float((manifest.get('usage') or {}).get('agent_phase_seconds', 0.0)), 3),
        'model_version': versions[-1] if versions else None,
        'retries': retries,
        'config_sha256': manifest.get('config_sha256'),
        **study,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--task', action='append', dest='tasks', help='Task ID (repeatable); defaults to config task_id')
    parser.add_argument('--replicates', type=int, default=5)
    parser.add_argument('--max-concurrency', type=int, default=4)
    parser.add_argument('--execute-model', action='store_true', help='Authorize paid model execution')
    args = parser.parse_args(argv)

    if not args.execute_model:
        parser.error('--execute-model is required to execute the baseline matrix')
    if args.replicates < 1 or args.max_concurrency < 1:
        parser.error('--replicates and --max-concurrency must be >= 1')

    base_config = resolve(args.config)
    tasks = args.tasks or [base_config['benchmark']['task_id']]
    root = args.output_root.absolute()
    root.mkdir(parents=True, exist_ok=False, mode=0o700)

    jobs = [(tid, rep) for rep in range(args.replicates) for tid in tasks]

    def _worker(job: tuple[str, int]) -> dict:
        tid, rep = job
        cfg = copy.deepcopy(base_config)
        cfg['benchmark']['task_id'] = tid
        ep_dir = root / f'{tid}_rep{rep:02d}'
        run(cfg, ep_dir)
        return summarize_episode(tid, rep, ep_dir)

    rows = []
    with ThreadPoolExecutor(max_workers=args.max_concurrency) as pool:
        future_map = {pool.submit(_worker, job): job for job in jobs}
        for fut in as_completed(future_map):
            rows.append(fut.result())

    rows.sort(key=lambda r: (r['task_id'], r['replicate']))
    csv_path = root / 'summary.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    write_json(root, 'summary.json', {'tasks': tasks, 'replicates': args.replicates, 'rows': rows})
    print(json.dumps({'output_root': str(root), 'total_runs': len(rows), 'summary_csv': str(csv_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
