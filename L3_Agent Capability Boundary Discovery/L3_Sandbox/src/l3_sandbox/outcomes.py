"""Read-only trial summaries from sealed L1 evidence."""
from __future__ import annotations
import json
from pathlib import Path, PurePosixPath
from c3_sandbox.l1.evaluation.study import study_outcome


def _read(path):
    return json.loads(path.read_text())


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.is_file() else []


def _writes(package, events):
    """Use transition -> tool result -> call -> parsed response causal links."""
    by_id = {e['event_id']: e for e in events}
    parsed = {e.get('links', {}).get('response'): i for i, e in
              enumerate((e for e in events if e.get('kind') == 'model.parsed'), 1)}
    first, submission = None, None
    for state in _lines(package / 'state_changes.jsonl'):
        transition = by_id.get(state.get('event_id'), {})
        result = by_id.get(transition.get('links', {}).get('tool_result'), {})
        call = by_id.get(result.get('links', {}).get('call'), {})
        if transition.get('kind') != 'state.transition' or result.get('kind') != 'tool.result' or call.get('kind') != 'tool.call':
            continue
        turn = parsed.get(call.get('links', {}).get('response'))
        if turn is None:
            continue
        for change in state.get('changes', []):
            if change.get('change') not in ('created', 'modified') or not change.get('after'):
                continue
            path = PurePosixPath(change.get('path', ''))
            if path.is_absolute() or '..' in path.parts:
                continue
            first = turn if first is None else min(first, turn)
            if len(path.parts) > 1 and path.parts[0] == 'submission':
                submission = turn if submission is None else min(submission, turn)
    return first, submission


def _usage(package):
    """Count each HTTP attempt once, including failed attempts with usage.

    Top-level HTTP/response copies alias the final attempt and are not extra calls.
    Unknown usage is counted explicitly, never estimated from parsed successes.
    """
    receipts, total, unknown = [], 0, 0
    root = package / 'raw/model'
    if root.is_dir():
        for request in sorted(p for p in root.iterdir() if p.is_dir()):
            attempts = sorted(request.glob('attempt_*/http.json'))
            paths = attempts or ([request / 'http.json'] if (request / 'http.json').is_file() else [])
            for path in paths:
                receipt = _read(path)
                usage = None
                response = path.parent / 'response.bin'
                if response.is_file():
                    try:
                        body = _read(response)
                        usage = body.get('usageMetadata') or body.get('usage')
                    except (ValueError, UnicodeError, AttributeError):
                        pass
                tokens = (usage.get('totalTokenCount', usage.get('total_tokens'))
                          if isinstance(usage, dict) else None)
                if tokens is None and isinstance(usage, dict):
                    inp = usage.get('input_tokens', usage.get('prompt_tokens'))
                    comp = usage.get('output_tokens', usage.get('completion_tokens'))
                    if type(inp) is int and type(comp) is int and inp >= 0 and comp >= 0:
                        tokens = inp + comp
                if type(tokens) is int and tokens >= 0:
                    total += tokens
                else:
                    tokens = None
                    unknown += 1
                receipts.append({'path': path.relative_to(package).as_posix(),
                                 'request_id': request.name, 'known_total_tokens': tokens,
                                 'metadata': receipt})
    return receipts, total, unknown


def summarize_episode(package: Path, job: dict, attempt: int, *, verify_evidence: bool = True) -> dict:
    """Summarize an episode; external trial/index binding is the runner's gate."""
    package = Path(package)
    manifest = _read(package / 'manifest.json')
    grade = _read(package / 'official_grade.json') if (package / 'official_grade.json').is_file() else {}
    events = _lines(package / 'trajectory.jsonl')
    study = study_outcome(package, verify_evidence=verify_evidence)
    task, status = manifest.get('task') or {}, manifest.get('status') or {}
    if task.get('task_id') != job.get('task_id') or manifest.get('config_sha256') != job.get('config_sha256'):
        study.update(study_valid=False, study_outcome_y=None, zero_reason=None,
                     study_classification='trial_identity_mismatch')
    raw = grade.get('official_raw') or {}
    dimensions = (grade.get('projection') or {}).get('official_dimensions') or {}
    legal = raw.get('dim1_pass') if task.get('benchmark') == 'officeval' else dimensions.get('passed')
    first, submission = _writes(package, events)
    receipts, total, unknown = _usage(package)
    versions = sorted({e.get('data', {}).get('reported_model_version') for e in events
                       if e.get('kind') == 'model.parsed' and e.get('data', {}).get('reported_model_version')})
    counts = manifest.get('counts')
    if not isinstance(counts, dict):
        counts = {}
    # Counts are observations: absent or malformed values remain unknown.
    observed_counts = {key: value if type(value) is int and value >= 0 else None
                       for key in ('model_requests', 'tool_calls')
                       for value in (counts.get(key),)}
    delivery = package / 'artifacts/submission'
    return {**{k: job.get(k) for k in ('trial_id', 'task_id', 'family_id', 'family_verified', 'split',
                                      'condition_id', 'replicate', 'config_sha256', 'config')},
            **observed_counts, 'attempt': attempt, 'package_path': str(package.resolve()),
            'episode_id': manifest.get('episode_id'), 'benchmark': task.get('benchmark'),
            'manifest_config_sha256': manifest.get('config_sha256'),
            'termination': status.get('termination'), 'execution_validity': status.get('execution_validity'),
            'evaluation_validity': status.get('evaluation_validity'),
            'evidence_completeness': status.get('evidence_completeness'),
            'native_projection': grade.get('projection'), 'native_official_raw': grade.get('official_raw'),
            **study, 'delivery_exists': delivery.is_dir() and any(p.is_file() for p in delivery.rglob('*')),
            'legal_delivery': legal if type(legal) is bool else None,
            'first_persistent_write_turn': first, 'first_submission_write_turn': submission,
            'write_turn_basis': 'local model.parsed ordinal, starting at 1; linked tool transitions only',
            'model_versions': versions, 'http_receipts': receipts,
            'http_attempt_count': len(receipts), 'known_total_tokens': total,
            'unknown_usage_count': unknown, 'actual_cost': None}
