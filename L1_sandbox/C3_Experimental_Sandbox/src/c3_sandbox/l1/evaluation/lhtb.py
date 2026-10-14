"""LHTB official verifier execution in isolated Docker container."""
from __future__ import annotations
import os
from pathlib import Path
from ..core.contracts import CapabilityError
from ..runtime.docker import DockerRuntime
from ..storage import read_json, write_bytes, write_json, file_digest, regular_file


def interpret_lhtb(raw: dict, task_id: str, policy: str = 'official_dimensions') -> dict:
    out = {
        'evaluation_validity': 'invalid',
        'task_outcome': 'unknown',
        'outcome_policy': policy,
        'official_dimensions': None,
        'completion_rate': None,
        'rubric_coverage': 'band_scored',
        'reason': 'malformed_official_result'
    }
    if not isinstance(raw, dict):
        return out

    reward = raw.get('reward', 0.0)
    raw_reward = raw.get('raw_reward', 0.0)
    band = int(raw_reward)

    rate = min(1.0, max(0.0, float(reward)))
    out.update(
        evaluation_validity='valid',
        reason='official_result_retained',
        completion_rate=rate,
        official_dimensions={
            'reward': float(reward),
            'raw_reward': float(raw_reward),
            'band': band,
            'passed': band >= 1
        }
    )
    if band >= 1:
        out['task_outcome'] = 'success'
    elif rate > 0:
        out['task_outcome'] = 'partial'
    else:
        out['task_outcome'] = 'failure'

    return out


class LHTBEvaluator:
    def __init__(self, config: dict):
        self.config = config

    def evaluate(self, spec, submission: Path, output: Path) -> dict:
        if spec.kind != 'lhtb-verifier':
            raise CapabilityError('LHTBEvaluator requires lhtb-verifier EvaluationSpec')

        output.mkdir(parents=True, exist_ok=False)
        image = spec.metadata.get('image', self.config.get('image', f'zli12321/lhtb-{spec.task_id}:20260615'))

        runtime = DockerRuntime({
            'image': image,
            'memory': self.config.get('memory', '2g'),
            'cpus': self.config.get('cpus', 2),
            'pids_limit': self.config.get('pids_limit', 128),
            'output_limit_bytes': self.config.get('output_limit_bytes', 4194304)
        })
        provenance = runtime.preflight()

        grade_dir = output / 'worker-output'
        grade_dir.mkdir(mode=0o777)
        if hasattr(os, 'getuid') and os.getuid() == 0:
            os.chown(grade_dir, 65532, 65532)

        # Ensure submission has moves.log even if empty
        moves_log = submission / 'moves.log'
        if not moves_log.exists():
            moves_log.write_text('')

        tests_dir = spec.source_root / 'tests'
        if not tests_dir.is_dir():
            tests_dir = spec.source_root

        command = [
            '/bin/bash', '-c',
            'SOKOBAN_MOVES_LOG=/submission/moves.log SOKOBAN_RENDER_VIDEO=0 python -m pytest -p no:cacheprovider -rA /tests/test_outputs.py; echo $? > /logs/verifier/pytest_exit.txt'
        ]

        mounts = [
            (submission, '/submission', True),
            (tests_dir, '/tests', True),
            (grade_dir, '/logs/verifier', False)
        ]

        timeout = self.config.get('timeout_seconds', 60)
        process = runtime.container(mounts, command, timeout, cwd='/tmp')

        write_bytes(output, 'stdout.bin', process.pop('stdout'))
        write_bytes(output, 'stderr.bin', process.pop('stderr'))
        write_json(output, 'process.json', process)
        write_json(output, 'runtime.json', provenance)

        # Extract score from reward.txt and raw_reward.txt
        reward = 0.0
        raw_reward = 0.0
        reward_path = grade_dir / 'reward.txt'
        raw_reward_path = grade_dir / 'raw_reward.txt'

        if reward_path.is_file():
            try:
                reward = float(reward_path.read_text().strip())
            except (ValueError, UnicodeError):
                reward = 0.0

        if raw_reward_path.is_file():
            try:
                raw_reward = float(raw_reward_path.read_text().strip())
            except (ValueError, UnicodeError):
                raw_reward = 0.0

        raw = {
            'task_id': spec.task_id,
            'reward': reward,
            'raw_reward': raw_reward,
            'band': int(raw_reward),
            'status': 'ok' if process['container_exit_code'] in (0, 1) else 'error'
        }

        raw_path = grade_dir / 'official.json'
        write_json(grade_dir, 'official.json', raw)

        projection = interpret_lhtb(raw, spec.task_id, self.config.get('outcome_policy', 'official_dimensions'))
        if process['stop_reason'] is not None:
            projection.update(
                evaluation_validity='invalid',
                task_outcome='unknown',
                reason=process['stop_reason']
            )

        return {
            'projection': projection,
            'official_raw': raw,
            'process': process,
            'runtime': provenance,
            'official_raw_sha256': file_digest(raw_path)
        }
