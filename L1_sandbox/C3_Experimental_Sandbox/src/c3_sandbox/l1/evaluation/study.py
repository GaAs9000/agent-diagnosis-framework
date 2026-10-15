"""Read-only research endpoints, separate from native rubric validity."""
from __future__ import annotations
import json
import math
from pathlib import Path
from .officeval import interpret
from .lhtb import interpret_lhtb, _official_failure

ENDPOINT_VERSION = 'study-outcome-v1'


def _read(path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError('expected object')
    return value


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def study_outcome(package: Path, *, verify_evidence: bool = False) -> dict:
    """Require actual grader health before counting any score or agent zero.

    Verification establishes self-consistency only; runners must separately bind
    the evidence index to their sealed expected trial identity.
    """
    package = Path(package)
    out = dict(study_valid=False, study_outcome_y=None, zero_reason=None,
               study_classification='missing_or_malformed_evidence', endpoint_version=ENDPOINT_VERSION,
               native_evaluation_validity=None, native_projection_evaluation_validity=None,
               native_status=None, native_error=None)
    def invalid(reason):
        out['study_classification'] = reason
        return out
    try:
        manifest = _read(package / 'manifest.json')
        status = manifest.get('status') or {}
        out['native_evaluation_validity'] = status.get('evaluation_validity')
        if (status.get('execution_validity') != 'valid'
                or status.get('evidence_completeness') != 'complete'
                or manifest.get('recording_mode') != 'live'):
            return invalid('invalid_execution_or_evidence')
        if verify_evidence:
            from ..package import verify_package
            if not verify_package(package).get('verified'):
                return invalid('evidence_verification_failed')
        grade = _read(package / 'official_grade.json')
        raw, projection = grade.get('official_raw'), grade.get('projection')
        if not isinstance(raw, dict) or not isinstance(projection, dict):
            return invalid('missing_official_grade')
        out.update(native_status=raw.get('status'), native_error=raw.get('error'),
                   native_projection_evaluation_validity=projection.get('evaluation_validity'))
        if grade.get('error_kind') == 'infrastructure' or grade.get('status') == 'infra_error':
            return invalid('infrastructure_failure')
        process = grade.get('process') or {}
        if (process.get('stop_reason') is not None or process.get('oom_killed')
                or process.get('output_complete') is False or 'container_exit_code' not in process):
            return invalid('unhealthy_grader_process')
        task = manifest.get('task') or {}
        task_id, benchmark = task.get('task_id'), task.get('benchmark')
        if not isinstance(task_id, str):
            return invalid('missing_task_identity')
        policy = projection.get('outcome_policy', 'official_dimensions')
        zero_reason = None
        if benchmark == 'officeval':
            worker = _read(package / 'grading/worker-output/worker.json')
            if (type(process['container_exit_code']) is not int or process['container_exit_code'] != 0
                    or worker.get('official_returned') is not True or worker.get('exception') is not None):
                return invalid('unhealthy_grader_worker')
            native = interpret(raw, task_id, policy)
            if native != projection:
                return invalid('raw_projection_mismatch')
            required = {'id', 'file_name', 'status', 'error', 'dim1_pass', 'dim1_reason', 'dim2_items', 'total_score', 'max_score'}
            if (not required.issubset(raw) or raw['id'] != task_id[-3:]
                    or not all(_finite(raw[k]) for k in ('total_score', 'max_score'))
                    or raw['max_score'] < 0 or raw['total_score'] > raw['max_score']
                    or type(raw['dim1_pass']) is not bool
                    or not isinstance(raw['file_name'], (str, list))
                    or not isinstance(raw['dim1_reason'], str) or not isinstance(raw['dim2_items'], list)):
                return invalid('malformed_official_grade')
            if native['evaluation_validity'] == 'valid':
                y = native['completion_rate']
                if not _finite(y) or not 0 <= y <= 1:
                    return invalid('undefined_official_endpoint')
                if y == 0:
                    zero_reason = 'official_zero' if raw['dim1_pass'] else 'illegal_delivery'
            else:
                error = raw.get('error')
                markers = ('TimeoutExpired', 'TimeoutError', 'Docker', 'Container', 'ModuleNotFoundError',
                           'ImportError', 'ConnectionError', 'MemoryError', 'PermissionError',
                           'SubprocessError', 'scorer_crash', 'infra_error')
                if (raw['status'] != 'error' or not isinstance(error, str) or not error
                        or raw['dim1_pass'] is not False or native['reason'] != 'official_error'
                        or any(marker in error for marker in markers)):
                    return invalid('unhealthy_or_ambiguous_official_error')
                # Accept only determinate missing/damaged delivery errors, never an arbitrary exception.
                delivery_errors = ('FileNotFoundError', 'BadZipFile', 'PackageNotFoundError',
                                   'InvalidFileException', 'XMLSyntaxError', 'ParseError',
                                   '未找到', '损坏', 'corrupt', 'not a zip file')
                if not any(marker in error for marker in delivery_errors):
                    return invalid('unhealthy_or_ambiguous_official_error')
                y, zero_reason = 0.0, 'missing_or_damaged_delivery'
        elif benchmark == 'lhtb' and task_id == 'sokoban':
            native = interpret_lhtb(raw, task_id, policy)
            if native != projection:
                return invalid('raw_projection_mismatch')
            code = (raw.get('protocol') or {}).get('pytest_exit')
            if (raw.get('task_id') != task_id or type(code) is not int or code not in (0, 1)
                    or type(process['container_exit_code']) is not int or process['container_exit_code'] != code
                    or native['evaluation_validity'] != 'valid'):
                return invalid('unhealthy_or_malformed_lhtb_grade')
            dimensions = native['official_dimensions']
            if type(raw.get('band')) is not int or raw['band'] != dimensions['band']:
                return invalid('inconsistent_official_band')
            passed = dimensions['passed']
            if code == 1:
                expected = _official_failure(package / 'grading/worker-output/pytest.xml')
                if expected is None or expected != raw.get('scoring_case') or passed:
                    return invalid('unexpected_official_test_failure')
            elif not passed or raw.get('scoring_case') != 'official_score':
                return invalid('inconsistent_official_pass_result')
            y = float(passed)
            if y == 0:
                zero_reason = raw.get('scoring_case')
        else:
            return invalid('unsupported_benchmark')
        out.update(study_valid=True, study_outcome_y=float(y), zero_reason=zero_reason,
                   study_classification='valid_zero' if y == 0 else 'valid_score')
        return out
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return out
