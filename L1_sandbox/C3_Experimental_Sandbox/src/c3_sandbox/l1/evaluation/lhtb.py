"""LHTB official verifier execution in isolated Docker container."""
from __future__ import annotations
import math
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from ..core.contracts import CapabilityError
from ..runtime.docker import DockerRuntime
from ..storage import write_bytes, write_json, file_digest


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def interpret_lhtb(raw: dict, task_id: str, policy: str = 'official_dimensions') -> dict:
    out = {
        'evaluation_validity': 'invalid', 'task_outcome': 'unknown',
        'outcome_policy': policy, 'official_dimensions': None,
        'completion_rate': None, 'rubric_coverage': 'band_scored',
        'reason': 'malformed_official_result'
    }
    if not isinstance(raw, dict) or raw.get('status') != 'ok' or raw.get('error'):
        return out
    reward, raw_reward = raw.get('reward'), raw.get('raw_reward')
    known_negative = (task_id == 'sokoban'
                      and raw.get('scoring_case') == 'official_no_moves'
                      and reward == 0 and raw_reward is None)
    if not _finite(reward) or not 0 <= reward <= 1:
        return out
    if not known_negative and (not _finite(raw_reward) or raw_reward < 0):
        return out
    band = 0 if known_negative else int(raw_reward)
    out.update(
        evaluation_validity='valid',
        reason='official_no_moves' if known_negative else 'official_result_retained',
        completion_rate=float(reward),
        official_dimensions={'reward': float(reward), 'raw_reward': raw_reward,
                             'band': band, 'passed': band >= 1},
        task_outcome='success' if band >= 1 else ('partial' if reward > 0 else 'failure')
    )
    return out


def _score_file(path):
    if not path.is_file():
        return None, None
    text = path.read_text(encoding='utf-8', errors='replace')
    try:
        value = float(text.strip())
    except ValueError:
        return None, text
    return (value if math.isfinite(value) else None), text


def _official_failure(path):
    """Recognize only the official test's expected benchmark assertion."""
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return None
    cases = list(root.iter('testcase'))
    if len(cases) != 1 or cases[0].get('name') != 'test_sokoban_progress':
        return None
    if list(root.iter('error')) or list(root.iter('skipped')):
        return None
    failures = list(root.iter('failure'))
    if len(failures) != 1:
        return None
    message = failures[0].get('message', '')
    if message.startswith('AssertionError: No moves found at /submission/moves.log.'):
        return 'official_no_moves'
    if message.startswith('AssertionError: Band 0 < pass threshold 1.'):
        return 'official_below_pass_band'
    return None


class LHTBEvaluator:
    def __init__(self, config: dict):
        self.config = config

    def evaluate(self, spec, submission: Path, output: Path) -> dict:
        if spec.kind != 'lhtb-verifier':
            raise CapabilityError('LHTBEvaluator requires lhtb-verifier EvaluationSpec')
        output.mkdir(parents=True, exist_ok=False)
        image = spec.metadata.get('image', self.config.get('image', f'zli12321/lhtb-{spec.task_id}:20260615'))
        runtime = DockerRuntime({
            'image': image, 'memory': self.config.get('memory', '2g'),
            'cpus': self.config.get('cpus', 2), 'pids_limit': self.config.get('pids_limit', 128),
            'output_limit_bytes': self.config.get('output_limit_bytes', 4194304)
        })
        provenance = runtime.preflight()
        grade_dir = output / 'worker-output'
        grade_dir.mkdir(mode=0o777)
        if hasattr(os, 'getuid') and os.getuid() == 0:
            os.chown(grade_dir, 65532, 65532)

        # The official scorer handles a missing moves.log itself. Preserve the
        # caller's submission exactly, and mount only an independent grading copy.
        grading_input = output / 'grading-input'
        shutil.copytree(submission, grading_input)
        moves_log = grading_input / 'moves.log'
        input_metadata = {
            'actor': 'evaluator_submission_copy',
            'moves_log_present': moves_log.is_file(),
            'moves_log_sha256': file_digest(moves_log) if moves_log.is_file() else None,
            'synthesized_files': []
        }
        write_json(output, 'grading-input.json', input_metadata)
        tests_dir = spec.source_root / 'tests'
        if not tests_dir.is_dir():
            tests_dir = spec.source_root
        command = ['/bin/bash', '-c',
                   'SOKOBAN_MOVES_LOG=/submission/moves.log SOKOBAN_RENDER_VIDEO=0 '
                   'python -m pytest -p no:cacheprovider -rA '
                   '--junitxml=/logs/verifier/pytest.xml /tests/test_outputs.py; '
                   'code=$?; echo "$code" > /logs/verifier/pytest_exit.txt; exit "$code"']
        process = runtime.container(
            [(grading_input, '/submission', True), (tests_dir, '/tests', True),
             (grade_dir, '/logs/verifier', False)], command,
            self.config.get('timeout_seconds', 60), cwd='/tmp')
        write_bytes(output, 'stdout.bin', process.pop('stdout'))
        write_bytes(output, 'stderr.bin', process.pop('stderr'))
        write_json(output, 'process.json', process)
        write_json(output, 'runtime.json', provenance)
        reward, reward_text = _score_file(grade_dir / 'reward.txt')
        raw_reward, raw_reward_text = _score_file(grade_dir / 'raw_reward.txt')
        exit_path = grade_dir / 'pytest_exit.txt'
        pytest_exit_text = exit_path.read_text(errors='replace') if exit_path.is_file() else None
        try:
            pytest_exit = int(pytest_exit_text.strip())
        except (ValueError, AttributeError):
            pytest_exit = None
        failure_case = _official_failure(grade_dir / 'pytest.xml') if pytest_exit == 1 else None
        case = failure_case or 'official_score'
        error = None
        if process.get('stop_reason') is not None:
            error = process['stop_reason']
        elif process.get('oom_killed') or process.get('output_complete') is False:
            error = 'incomplete_verifier_execution'
        elif pytest_exit not in (0, 1) or process.get('container_exit_code') != pytest_exit:
            error = 'invalid_pytest_exit'
        elif pytest_exit == 1 and failure_case is None:
            error = 'unexpected_official_test_failure'
        elif reward is None or not 0 <= reward <= 1:
            error = 'missing_or_malformed_official_reward'
        elif failure_case == 'official_no_moves':
            if (spec.task_id != 'sokoban' or reward != 0 or raw_reward_text is not None):
                error = 'inconsistent_official_no_moves_result'
        elif raw_reward is None or raw_reward < 0:
            error = 'missing_or_malformed_official_raw_reward'
        elif (pytest_exit == 0) != (int(raw_reward) >= 1):
            error = 'inconsistent_official_pass_result'
        raw = {
            'task_id': spec.task_id, 'reward': reward, 'raw_reward': raw_reward,
            'band': int(raw_reward) if raw_reward is not None and raw_reward >= 0 else (0 if failure_case == 'official_no_moves' else None),
            'status': 'error' if error else 'ok', 'error': error, 'scoring_case': case,
            'protocol': {'pytest_exit': pytest_exit, 'pytest_exit_text': pytest_exit_text,
                         'reward_text': reward_text, 'raw_reward_text': raw_reward_text},
            'source': {'entrypoint': 'tests/test_outputs.py',
                       'sha256': file_digest(tests_dir / 'test_outputs.py'),
                       'source_commit': spec.metadata.get('source_commit'),
                       'data_lock_status': spec.metadata.get('data_lock_status')},
            'grading_input': input_metadata
        }
        raw_path = write_json(grade_dir, 'official.json', raw)
        projection = interpret_lhtb(raw, spec.task_id, self.config.get('outcome_policy', 'official_dimensions'))
        if error:
            projection['reason'] = error
        result = {'projection': projection, 'official_raw': raw, 'process': process,
                  'runtime': provenance, 'official_raw_sha256': file_digest(raw_path)}
        if error:
            result.update(status='infra_error', error_kind='infrastructure', error=error)
        return result
