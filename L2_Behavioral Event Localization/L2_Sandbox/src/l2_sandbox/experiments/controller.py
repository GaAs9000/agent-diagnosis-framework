"""Explicitly authorized unchanged-resume controller, two-stage value-curve runner, and receipt verifier.

No arbitrary executor script, model command or analysis-authored patch is run.
Uses the user's installed L1 run(parent=..., checkpoint_id=...) implementation.
"""
from __future__ import annotations
import concurrent.futures
import copy
import inspect
import random
import threading
import time
from pathlib import Path
from ..io import (L2Error, Unsupported, IntegrityError, Journal, digest, now, read_json, read_jsonl,
                  sha_file, safe, write_json, write_jsonl, write_new)
from .plans import TEMPLATES
from ..stats.curve import derive_stage2_interval, evaluate_curve
from ..stats.strata import STRATA_TAGS, evaluate_strata
from ..stats.intervals import holm_adjust


class TokenBucketRateLimiter:
    """Thread-safe token bucket limiter for API requests/min and tokens/min."""

    def __init__(self, rpm: float | None = None, tpm: float | None = None):
        self.rpm = float(rpm) if rpm and rpm > 0 else None
        self.tpm = float(tpm) if tpm and tpm > 0 else None
        self._lock = threading.Lock()
        self._req_tokens = self.rpm if self.rpm else 0.0
        self._tok_tokens = self.tpm if self.tpm else 0.0
        self._last = time.monotonic()

    def acquire(self, estimated_tokens: float = 1000.0) -> None:
        if self.rpm is None and self.tpm is None:
            return
        while True:
            with self._lock:
                now_ts = time.monotonic()
                elapsed = max(0.0, now_ts - self._last)
                self._last = now_ts
                if self.rpm is not None:
                    self._req_tokens = min(self.rpm, self._req_tokens + elapsed * (self.rpm / 60.0))
                if self.tpm is not None:
                    self._tok_tokens = min(self.tpm, self._tok_tokens + elapsed * (self.tpm / 60.0))
                req_ok = self.rpm is None or self._req_tokens >= 1.0
                tok_ok = self.tpm is None or self._tok_tokens >= min(estimated_tokens, self.tpm)
                if req_ok and tok_ok:
                    if self.rpm is not None:
                        self._req_tokens -= 1.0
                    if self.tpm is not None:
                        self._tok_tokens -= min(estimated_tokens, self.tpm)
                    return
            time.sleep(0.02)


_INFRA_ERROR_MARKERS = (
    'TimeoutExpired', 'TimeoutError', 'Docker', 'Container', 'ModuleNotFoundError',
    'ImportError', 'ConnectionError', 'MemoryError', 'PermissionError', 'SubprocessError',
    'OSError: [Errno', 'scorer_crash', 'infra_error',
)
_MISSING_FILE_MARKERS = (
    'No such file', 'FileNotFoundError', 'missing submission', 'not found', 'no submission',
)


def _extract_model_version(reader) -> str | None:
    for e in reader.events:
        if e['kind'] == 'model.response':
            d = e['data']
            if isinstance(d.get('model_version'), str) and d['model_version']:
                return d['model_version']
            try:
                body = reader.json(d['body']['path'])
                if isinstance(body, dict):
                    mv = body.get('modelVersion') or body.get('model')
                    if isinstance(mv, str) and mv:
                        return mv
            except Exception:
                pass
    return (reader.manifest.get('model') or {}).get('model_version')


def _extract_dim2_hits(official_raw: dict | None) -> dict[str, float]:
    if not isinstance(official_raw, dict):
        return {}
    hits: dict[str, float] = {}
    items = official_raw.get('dim2_items') or official_raw.get('items') or official_raw.get('rubric_items')
    if isinstance(items, list):
        for idx, item in enumerate(items):
            if isinstance(item, dict):
                key = str(item.get('id') or item.get('item_id') or item.get('name') or f'item_{idx:03d}')
                if 'hit' in item and isinstance(item['hit'], (bool, int, float)):
                    hits[key] = 1.0 if item['hit'] else 0.0
                elif 'score' in item and 'max_score' in item and isinstance(item['max_score'], (int, float)) and item['max_score'] > 0:
                    hits[key] = max(0.0, min(1.0, float(item['score']) / float(item['max_score'])))
    elif isinstance(items, dict):
        for key, item in sorted(items.items()):
            if isinstance(item, bool):
                hits[str(key)] = 1.0 if item else 0.0
            elif isinstance(item, dict) and 'hit' in item:
                hits[str(key)] = 1.0 if item['hit'] else 0.0
    return hits


def check_first_request_identity(parent_reader, branch_reader, checkpoint_id: str) -> dict:
    """Verifies that the branch's first model request is byte-identical to the parent's request at the same turn."""
    cp = parent_reader.json(f'checkpoints/{checkpoint_id}.json')
    req_idx = int(cp['budget_used']['model_requests'])
    parent_reqs = [e for e in parent_reader.events if e['kind'] == 'model.request']
    branch_reqs = [e for e in branch_reader.events if e['kind'] == 'model.request']
    if not branch_reqs:
        return {'verified': False, 'reason': 'branch_missing_first_request', 'request_index': req_idx}
    b_event = branch_reqs[0]
    b_path = b_event['data']['body']['path']
    b_bytes = branch_reader.read(b_path)

    p_path = b_path if b_path in parent_reader.files else (
        parent_reqs[req_idx]['data']['body']['path'] if req_idx < len(parent_reqs) else None
    )
    if p_path is None or p_path not in parent_reader.files:
        return {'verified': False, 'reason': 'parent_request_missing_at_index', 'request_index': req_idx, 'branch_path': b_path}
    if b_path != p_path:
        return {'verified': False, 'reason': 'first_request_path_mismatch', 'request_index': req_idx, 'parent_path': p_path, 'branch_path': b_path}
    p_bytes = parent_reader.read(p_path)
    if b_bytes != p_bytes:
        return {'verified': False, 'reason': 'first_request_bytes_differ', 'request_index': req_idx, 'path': b_path}
    return {'verified': True, 'reason': None, 'request_index': req_idx, 'path': b_path, 'sha256': branch_reader.files[b_path]['sha256']}


def outcome(reader, endpoint, *, v02_policy: bool = False):
    """Evaluates branch outcome.

    Preserves exact v0.1 behavior for legacy string endpoints unless v02_policy=True
    or endpoint is a v0.2 dict / ('official_completion_rate', 'passed').
    """
    is_v02 = v02_policy or isinstance(endpoint, dict) or endpoint in ('official_completion_rate', 'passed')
    ep_name = endpoint['name'] if isinstance(endpoint, dict) else endpoint
    m = reader.manifest
    g = reader.json('official_grade.json')
    status = m['status']
    if not is_v02:
        record = {
            'execution_valid': status.get('execution_validity') == 'valid',
            'evaluation_available': status.get('evaluation_validity') == 'valid',
            'endpoint_defined': False,
            'value': None,
            'termination': status.get('termination'),
            'official_projection': g.get('projection'),
            'official_raw': g.get('official_raw'),
            'error_reason': None,
        }
        if not record['execution_valid']:
            record['error_reason'] = 'execution_invalid'
            return record
        if ep_name == 'submission_present':
            record.update(endpoint_defined=True, value=bool(reader.json('artifacts/inventory.json')['files']))
        elif not record['evaluation_available'] or not isinstance(g.get('official_raw'), dict):
            record['error_reason'] = 'official_evaluation_unavailable'
        elif ep_name == 'official_dim1':
            value = g['official_raw'].get('dim1_pass')
            if type(value) is bool:
                record.update(endpoint_defined=True, value=value)
            else:
                record['error_reason'] = 'dim1_undefined'
        elif ep_name == 'official_normalized_score':
            raw = g['official_raw']
            score = raw.get('total_score')
            maximum = raw.get('max_score')
            if type(score) in (int, float) and type(maximum) in (int, float) and maximum > 0:
                record.update(endpoint_defined=True, value=score / maximum)
            else:
                record['error_reason'] = 'score_denominator_undefined'
        return record

    # v0.2.1 Section 3.5 & 3.6 outcome table (reads L1 projection)
    exec_valid = status.get('execution_validity') == 'valid'
    eval_valid = status.get('evaluation_validity') == 'valid'
    inv_files = reader.json('artifacts/inventory.json').get('files') or []
    has_files = bool(inv_files)
    proj = g.get('projection') if isinstance(g.get('projection'), dict) else {}
    raw = g.get('official_raw') if isinstance(g.get('official_raw'), dict) else None

    record = {
        'execution_valid': exec_valid,
        'evaluation_available': eval_valid,
        'endpoint_defined': False,
        'value': None,
        'sensitivity_value': None,
        'outcome_case': None,
        'termination': status.get('termination'),
        'official_projection': g.get('projection'),
        'official_raw': g.get('official_raw'),
        'dim2_hits': _extract_dim2_hits(raw),
        'model_version': _extract_model_version(reader),
        'error_reason': None,
    }

    # Case 4 in table (row 4): Model transport / execution invalid -> missing
    if not exec_valid:
        record.update(outcome_case='infrastructure_invalid', error_reason='execution_invalid')
        return record

    if ep_name == 'submission_present':
        val = 1.0 if has_files else 0.0
        record.update(endpoint_defined=True, value=val, sensitivity_value=val, outcome_case='scored')
        return record

    # Row 2: execution_valid, but submission/ is empty -> valid outcome, score 0 (dim1 False)
    if not has_files:
        zero_val = False if ep_name == 'official_dim1' else 0.0
        record.update(
            endpoint_defined=True,
            value=zero_val,
            sensitivity_value=0.0,
            outcome_case='empty_submission',
            error_reason='empty_submission_scored_zero',
        )
        return record

    # If evaluation_validity != 'valid', distinguish Row 3 (corrupted/missing expected submission) vs Row 5 (scorer failure)
    if not eval_valid:
        err_text = str(g.get('error') or (raw or {}).get('error') or '')
        raw_status = (raw or {}).get('status') or g.get('status')
        is_infra = (
            raw is None
            or g.get('status') in ('infra_error', 'timeout')
            or g.get('error_kind') == 'infrastructure'
            or any(m in err_text for m in _INFRA_ERROR_MARKERS)
        )
        if is_infra:
            record.update(
                endpoint_defined=False,
                value=None,
                sensitivity_value=None,
                outcome_case='scorer_infrastructure_invalid',
                error_reason='scorer_infrastructure_error',
            )
            return record
        if any(m in err_text for m in _MISSING_FILE_MARKERS):
            zero_val = False if ep_name == 'official_dim1' else 0.0
            record.update(
                endpoint_defined=True,
                value=zero_val,
                sensitivity_value=0.0,
                outcome_case='empty_submission',
                error_reason='missing_expected_submission_scored_zero',
            )
            return record
        # Row 3: Scorer returned non-ok status on damaged/unparseable deliverable -> 0 in primary, missing in sensitivity
        if raw_status != 'ok':
            zero_val = False if ep_name == 'official_dim1' else 0.0
            record.update(
                endpoint_defined=True,
                value=zero_val,
                sensitivity_value=None,
                outcome_case='unscorable_submission',
                error_reason='unscorable_submission_scored_zero',
            )
            return record
        record.update(
            endpoint_defined=False,
            value=None,
            sensitivity_value=None,
            outcome_case='scorer_infrastructure_invalid',
            error_reason='official_evaluation_unavailable',
        )
        return record

    # Row 1: Normal valid execution & valid evaluation -> read strictly from L1 projection (Section 3.6)
    if not proj or raw is None:
        record.update(
            endpoint_defined=False,
            value=None,
            sensitivity_value=None,
            outcome_case='scorer_infrastructure_invalid',
            error_reason='official_projection_missing',
        )
        return record

    if ep_name in ('official_completion_rate', 'official_normalized_score'):
        cr = proj.get('completion_rate')
        if type(cr) in (int, float) and not isinstance(cr, bool):
            val = max(0.0, min(1.0, float(cr)))
            record.update(endpoint_defined=True, value=val, sensitivity_value=val, outcome_case='scored')
        else:
            record.update(outcome_case='scorer_infrastructure_invalid', error_reason='projection_completion_rate_undefined')
    elif ep_name == 'passed':
        dims = proj.get('official_dimensions') if isinstance(proj.get('official_dimensions'), dict) else {}
        passed = dims.get('passed') if 'passed' in dims else proj.get('passed')
        if type(passed) is bool:
            val = 1.0 if passed else 0.0
            record.update(endpoint_defined=True, value=val, sensitivity_value=val, outcome_case='scored')
        else:
            record.update(outcome_case='scorer_infrastructure_invalid', error_reason='projection_passed_undefined')
    elif ep_name == 'official_dim1':
        dims = proj.get('official_dimensions') if isinstance(proj.get('official_dimensions'), dict) else {}
        d1 = dims.get('dim1_pass') if 'dim1_pass' in dims else proj.get('dim1_pass')
        if type(d1) is bool:
            record.update(endpoint_defined=True, value=d1, sensitivity_value=1.0 if d1 else 0.0, outcome_case='scored')
        else:
            record.update(outcome_case='scorer_infrastructure_invalid', error_reason='projection_dim1_undefined')
    else:
        record.update(outcome_case='scorer_infrastructure_invalid', error_reason='unknown_endpoint')
    return record


def execute(plan_dir: Path, source: Path, output: Path, *, allow_execution=False, _test_runner=None):
    from ..package import verify, seal
    from ..evidence.l1 import L1Reader
    if not allow_execution: raise L2Error('Execution may spend real model/API budget; --allow-execution is required')
    v = verify(plan_dir)
    if not v['verified']: raise L2Error('Plan did not verify')
    plan = read_json(plan_dir / 'plan.json')
    if plan.get('package_kind') == 'value_curve_plan' or plan.get('template') == 'value_curve':
        return execute_curve(plan_dir, source, output, allow_execution=allow_execution, _test_runner=_test_runner)
    if plan['plan_state'] != 'frozen' or plan['template'] != 'resume_unchanged':
        raise Unsupported('Only a frozen resume_unchanged plan can be executed by this function')
    reader = L1Reader(source, expected_index=plan['source_index_sha256'])
    cp_path = f"checkpoints/{plan['checkpoint_id']}.json"
    if reader.files[cp_path]['sha256'] != plan['checkpoint_sha256']: raise IntegrityError('Checkpoint differs from frozen plan')
    cfg = reader.json('raw/resolved_config.json')
    if digest(cfg) != plan['config_sha256']: raise IntegrityError('Agent configuration differs')
    if _test_runner is None:
        try: from c3_sandbox.l1.core.orchestrator import run as runner
        except ImportError as exc: raise Unsupported('Install existing L1 0.2 in this virtualenv; no host fallback') from exc
    else: runner = _test_runner
    if not {'parent', 'checkpoint_id'} <= set(inspect.signature(runner).parameters): raise Unsupported('Installed L1 resume signature differs')
    output = Path(output).absolute()
    for p in (plan_dir, source):
        if output.resolve().is_relative_to(Path(p).resolve()) or Path(p).resolve().is_relative_to(output.resolve()):
            raise L2Error('Experiment output must be outside immutable inputs')
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    write_new(output, 'frozen_plan.json', (plan_dir / 'plan.json').read_bytes())
    write_json(output, 'authorization.json', {
        'authorized_at': now(), 'explicit_allow_execution': True, 'plan_index_sha256': v['index_sha256'],
        'plan_sha256': sha_file(plan_dir / 'plan.json'), 'mode': 'test_fixture' if _test_runner else 'live_l1_bridge',
        'authorized_branches': sum(x['replicates'] for x in plan['arms']), 'allowed_templates': ['resume_unchanged']})
    journal = Journal(output, 'experiment_trace.jsonl'); receipts = []
    journal.emit('experiment.started', {'plan_sha256': sha_file(plan_dir / 'plan.json'), 'source_index_sha256': reader.index_sha256})
    for i in range(plan['replicates_per_arm']):
        for arm in plan['arms']:
            name = f"branches/{arm['arm_id']}_{i:03d}"
            intent = {'trial_id': f"{arm['arm_id']}_{i:03d}", 'arm_id': arm['arm_id'], 'template': arm['template'],
                      'source_index_sha256': reader.index_sha256, 'checkpoint_sha256': plan['checkpoint_sha256'],
                      'checkpoint_id': plan['checkpoint_id'], 'replicate': i, 'branch_path': name}
            journal.emit('branch.intent', intent)
            try:
                runner(copy.deepcopy(cfg), output / name, parent=Path(source), checkpoint_id=plan['checkpoint_id'])
                branch = L1Reader(output / name)
                lineage = branch.manifest.get('lineage') or {}
                if lineage.get('parent_index_sha256') != reader.index_sha256 or lineage.get('parent_checkpoint_sha256') != plan['checkpoint_sha256']:
                    raise IntegrityError('New branch lineage differs from frozen plan')
                if branch.manifest['episode_id'] == reader.episode_id: raise IntegrityError('Resume did not create a new episode')
                row = {**intent, 'attempted': True, 'new_episode_id': branch.episode_id, 'branch_index_sha256': branch.index_sha256,
                       'branch_verified': True, 'counts': branch.manifest['counts'], 'outcome': outcome(branch, plan['endpoint']),
                       'controller_error': None, 'source_unchanged': reader.verify_unchanged()}
            except Exception as exc:
                row = {**intent, 'attempted': True, 'new_episode_id': None, 'branch_index_sha256': None, 'branch_verified': False, 'counts': None,
                       'outcome': {'execution_valid': False, 'evaluation_available': False, 'endpoint_defined': False, 'value': None,
                                   'error_reason': 'controller_or_branch_error'},
                       'controller_error': {'type': type(exc).__name__, 'message': str(exc)},
                       'source_unchanged': reader.verify_unchanged()}
            receipts.append(row); journal.emit('branch.receipt', row)
            if not row['source_unchanged']: raise IntegrityError('Parent source changed; experiment aborted with journal retained')
    write_jsonl(output, 'branch_receipts.jsonl', receipts)
    summary = summarize(receipts, plan, fixture=_test_runner is not None)
    write_json(output, 'results.json', summary); journal.emit('experiment.finished', summary)
    seal(output, 'experiment_execution')
    return {'output': str(output), 'results': summary, 'verification': verify(output)}


def summarize(receipts, plan, *, fixture=False):
    groups = {}
    for arm in plan['arms']:
        rows = [r for r in receipts if r['arm_id'] == arm['arm_id']]
        vals = [r['outcome']['value'] for r in rows if r['outcome']['endpoint_defined']]
        groups[arm['arm_id']] = {
            'attempted': len(rows), 'execution_valid': sum(r['outcome']['execution_valid'] for r in rows),
            'evaluation_available': sum(r['outcome']['evaluation_available'] for r in rows),
            'endpoint_defined': len(vals), 'endpoint_undefined': len(rows) - len(vals),
            'mean_on_defined_only': sum(vals) / len(vals) if vals else None,
            'errors': [r.get('controller_error') or r['outcome'].get('error_reason') for r in rows if not r['outcome']['endpoint_defined']]}
    return {'plan_state': 'completed' if all(r['branch_verified'] for r in receipts) else 'execution_invalid',
            'hypothesis_verdict': 'untested' if fixture else 'not_identifiable', 'groups': groups, 'endpoint': plan['endpoint'],
            'reason': 'Test fixtures are not real intervention evidence' if fixture else 'Only unchanged-resume observations; no intervention contrast is identified',
            'mode': 'test_fixture' if fixture else 'live_l1_bridge',
            'causal_effect': None, 'statistical_policy': 'descriptive_only_no_significance_or_ground_truth_label',
            'missing_outcomes_not_discarded': True}


def _active_receipts(receipts: list[dict], stage: str) -> list[dict]:
    """Returns active (non-superseded) receipts for a given stage ('stage1' or 'stage2')."""
    return [r for r in receipts if r.get('stage') == stage and not r.get('replaced_by')]


def derive_stage2(plan: dict, stage1_receipts: list[dict]) -> dict:
    """Deterministically derives the Stage 2 checkpoint interval from Stage 1 receipts."""
    T = int(plan['num_turns'])
    Y = float(plan['terminal_value'])
    grid_cps = plan['grid']['checkpoints']
    grid_turns = [int(cp[2:]) for cp in grid_cps]
    active = _active_receipts(stage1_receipts, 'stage1')
    stage1_defined: dict[int, list[float]] = {}
    for cp_id in grid_cps:
        k = int(cp_id[2:])
        stage1_defined[k] = [
            float(r['outcome']['value'])
            for r in active
            if r['checkpoint_id'] == cp_id and r['outcome']['endpoint_defined'] and r['outcome']['value'] is not None
        ]
    sel = derive_stage2_interval(grid_turns, stage1_defined, T, Y)
    a = int(sel['selected_interval']['start_turn'])
    b = int(sel['selected_interval']['end_turn'])
    return {
        'rule': plan['stage2']['rule'],
        'selected_interval': [a, b],
        'stage1_max_drop': sel['selected_interval']['stage1_drop'],
        'stage1_intervals': sel['stage1_intervals'],
        'checkpoints': sel['stage2_checkpoints'],
        'replicates': int(plan['stage2']['replicates']),
    }


def _schedule_rounds(checkpoints: list[str], replicates: int, plan_sha: str, stage: str) -> list[list[tuple[str, int]]]:
    """Block-randomizes checkpoint order within each replicate round using a deterministic seed."""
    rounds = []
    for rep in range(replicates):
        block = [(cp, rep) for cp in checkpoints]
        seed_int = int(digest({'plan_sha256': plan_sha, 'stage': stage, 'round': rep})[:16], 16)
        rng = random.Random(seed_int)
        rng.shuffle(block)
        rounds.append(block)
    return rounds


def execute_curve(
    plan_dir: Path,
    source: Path,
    output: Path,
    *,
    allow_execution: bool = False,
    max_concurrency: int | None = None,
    rpm: float | None = None,
    tpm: float | None = None,
    _test_runner=None,
) -> dict:
    """Executes a frozen two-stage value_curve_plan with bounded concurrency and rate limiting."""
    from ..package import verify, seal
    from ..evidence.l1 import L1Reader
    from ..facts.build import extract_turn_tags, is_docx_episode

    if not allow_execution:
        raise L2Error('Execution may spend real model/API budget; --allow-execution is required')
    v = verify(plan_dir)
    if not v['verified']:
        raise L2Error('Value curve plan did not verify: ' + str(v.get('errors')))
    plan = read_json(plan_dir / 'plan.json')
    if plan.get('package_kind') != 'value_curve_plan' or plan.get('plan_state') != 'frozen':
        raise Unsupported('Only a frozen value_curve_plan can be executed')

    reader = L1Reader(source, expected_index=plan['source_index_sha256'])
    cfg = reader.json('raw/resolved_config.json')
    if digest(cfg) != plan['config_sha256']:
        raise IntegrityError('Agent configuration differs from frozen value_curve_plan')

    for cp_id, expected_sha in plan.get('qualified_checkpoint_sha256', {}).items():
        cp_path = f'checkpoints/{cp_id}.json'
        if reader.files.get(cp_path, {}).get('sha256') != expected_sha:
            raise IntegrityError(f'Checkpoint {cp_id} differs from frozen plan')

    if _test_runner is None:
        try:
            from c3_sandbox.l1.core.orchestrator import run as runner
        except ImportError as exc:
            raise Unsupported('Install existing L1 0.2 in this virtualenv; no host fallback') from exc
    else:
        runner = _test_runner
    if not {'parent', 'checkpoint_id'} <= set(inspect.signature(runner).parameters):
        raise Unsupported('Installed L1 resume signature differs')

    concurrency = int(max_concurrency or plan['schedule'].get('max_concurrency', 4))
    if not 1 <= concurrency <= 16:
        raise L2Error('max_concurrency must be 1..16')
    max_replacements = int(plan['failure_policy'].get('max_replacements', 2))

    output = Path(output).absolute()
    for p in (plan_dir, source):
        if output.resolve().is_relative_to(Path(p).resolve()) or Path(p).resolve().is_relative_to(output.resolve()):
            raise L2Error('Experiment output must be outside immutable inputs')
    output.mkdir(parents=True, exist_ok=False, mode=0o700)

    plan_sha = sha_file(plan_dir / 'plan.json')
    write_new(output, 'frozen_plan.json', (plan_dir / 'plan.json').read_bytes())
    if (plan_dir / 'baselines.json').exists():
        write_new(output, 'baselines.json', (plan_dir / 'baselines.json').read_bytes())
    write_json(output, 'authorization.json', {
        'authorized_at': now(),
        'explicit_allow_execution': True,
        'plan_index_sha256': v['index_sha256'],
        'plan_sha256': plan_sha,
        'mode': 'test_fixture' if _test_runner else 'live_l1_bridge',
        'max_concurrency': concurrency,
        'allowed_templates': ['value_curve'],
    })

    limiter = TokenBucketRateLimiter(rpm=rpm, tpm=tpm)
    journal = Journal(output, 'experiment_trace.jsonl')
    journal_lock = threading.Lock()
    receipts: list[dict] = []
    parent_model_version = _extract_model_version(reader)
    parent_docx = is_docx_episode(reader)
    observed_versions_lock = threading.Lock()
    observed_model_versions: set[str] = {parent_model_version} if parent_model_version else set()

    journal.emit('experiment.started', {'plan_sha256': plan_sha, 'source_index_sha256': reader.index_sha256})

    def _run_single_trial(stage: str, cp_id: str, rep: int, attempt_idx: int = 0) -> list[dict]:
        suffix = f'_r{attempt_idx}' if attempt_idx > 0 else ''
        prefix = 's1' if stage == 'stage1' else 's2'
        trial_id = f'{prefix}_{cp_id}_{rep:03d}{suffix}'
        branch_rel = f'branches/{trial_id}'
        cp_sha = plan['qualified_checkpoint_sha256'][cp_id]
        turn_idx = int(cp_id[2:]) + 1
        intent = {
            'trial_id': trial_id,
            'stage': stage,
            'checkpoint_id': cp_id,
            'checkpoint_sha256': cp_sha,
            'replicate': rep,
            'replacement_attempt': attempt_idx,
            'source_index_sha256': reader.index_sha256,
            'branch_path': branch_rel,
        }
        with journal_lock:
            journal.emit('branch.intent', intent)
        limiter.acquire()
        try:
            runner(copy.deepcopy(cfg), output / branch_rel, parent=Path(source), checkpoint_id=cp_id)
            branch = L1Reader(output / branch_rel)
            lineage = branch.manifest.get('lineage') or {}
            if lineage.get('parent_index_sha256') != reader.index_sha256 or lineage.get('parent_checkpoint_sha256') != cp_sha:
                raise IntegrityError('New branch lineage differs from frozen plan')
            if branch.manifest['episode_id'] == reader.episode_id:
                raise IntegrityError('Resume did not create a new episode')

            id_check = check_first_request_identity(reader, branch, cp_id)
            if not id_check['verified']:
                raise IntegrityError('policy_identity_violation: ' + str(id_check['reason']))

            out = outcome(branch, plan['endpoint'], v02_policy=True)
            mv = out.get('model_version')
            version_drift = False
            if mv:
                with observed_versions_lock:
                    if observed_model_versions and mv not in observed_model_versions:
                        version_drift = True
                    observed_model_versions.add(mv)
            if version_drift:
                raise IntegrityError(f'model_version_drift: observed {mv} vs {sorted(observed_model_versions)}')

            b_turns = extract_turn_tags(branch)
            first_turn_entry = next((t for t in b_turns if t['turn_index'] == turn_idx), b_turns[0] if b_turns else None)
            first_tags = list(first_turn_entry['tags']) if first_turn_entry else []
            row = {
                **intent,
                'attempted': True,
                'new_episode_id': branch.episode_id,
                'branch_index_sha256': branch.index_sha256,
                'branch_verified': True,
                'first_request_identity': id_check,
                'first_request_identical': True,
                'first_turn_index': turn_idx,
                'first_turn_tags': first_tags,
                'is_docx_task': parent_docx,
                'model_version': mv,
                'counts': branch.manifest['counts'],
                'outcome': out,
                'controller_error': None,
                'replaced_by': None,
                'source_unchanged': reader.verify_unchanged(),
            }
        except Exception as exc:
            err_msg = str(exc)
            reason = (
                'policy_identity_violation' if 'policy_identity_violation' in err_msg else
                'model_version_drift' if 'model_version_drift' in err_msg else
                'controller_or_branch_error'
            )
            row = {
                **intent,
                'attempted': True,
                'new_episode_id': None,
                'branch_index_sha256': None,
                'branch_verified': False,
                'first_request_identity': {'verified': False, 'reason': reason},
                'first_request_identical': reason != 'policy_identity_violation',
                'first_turn_index': turn_idx,
                'first_turn_tags': [],
                'is_docx_task': parent_docx,
                'model_version': None,
                'counts': None,
                'outcome': {
                    'execution_valid': False,
                    'evaluation_available': False,
                    'endpoint_defined': False,
                    'value': None,
                    'sensitivity_value': None,
                    'outcome_case': 'infrastructure_invalid',
                    'termination': None,
                    'official_projection': None,
                    'official_raw': None,
                    'dim2_hits': {},
                    'model_version': None,
                    'error_reason': reason,
                },
                'controller_error': {'type': type(exc).__name__, 'message': err_msg},
                'replaced_by': None,
                'source_unchanged': reader.verify_unchanged(),
            }

        with journal_lock:
            journal.emit('branch.receipt', row)
        if not row['source_unchanged']:
            raise IntegrityError('Parent source changed; experiment aborted with journal retained')

        collected = [row]
        if (
            not row['outcome']['endpoint_defined']
            and row['outcome'].get('error_reason') not in ('policy_identity_violation', 'model_version_drift')
            and attempt_idx < max_replacements
        ):
            rep_rows = _run_single_trial(stage, cp_id, rep, attempt_idx + 1)
            row['replaced_by'] = rep_rows[0]['trial_id']
            collected.extend(rep_rows)
        return collected

    def _run_stage(stage: str, cps: list[str], reps: int) -> list[dict]:
        stage_rows: list[dict] = []
        rounds = _schedule_rounds(cps, reps, plan_sha, stage)
        for block in rounds:
            if concurrency == 1:
                for cp_id, rep in block:
                    stage_rows.extend(_run_single_trial(stage, cp_id, rep, 0))
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futs = [pool.submit(_run_single_trial, stage, cp_id, rep, 0) for cp_id, rep in block]
                    for fut in concurrent.futures.as_completed(futs):
                        stage_rows.extend(fut.result())
        return stage_rows

    s1_rows = _run_stage('stage1', plan['grid']['checkpoints'], int(plan['stage1']['replicates']))
    receipts.extend(s1_rows)

    stage2_plan = derive_stage2(plan, s1_rows)
    write_json(output, 'stage2_plan.json', stage2_plan)
    journal.emit('stage2.frozen', stage2_plan)

    s2_rows = _run_stage('stage2', stage2_plan['checkpoints'], int(stage2_plan['replicates']))
    receipts.extend(s2_rows)

    write_jsonl(output, 'branch_receipts.jsonl', receipts)
    summary = summarize_curve(receipts, plan, stage2_plan, fixture=_test_runner is not None)
    write_json(output, 'results.json', summary)
    journal.emit('experiment.finished', {'verdict': summary['verdict'], 'plan_state': summary['plan_state']})
    seal(output, 'value_curve_execution')
    return {'output': str(output), 'results': summary, 'verification': verify(output)}


def summarize_curve(receipts: list[dict], plan: dict, stage2_plan: dict, *, fixture: bool = False) -> dict:
    """Deterministically computes the value curve verdict, missing bounds, sensitivity, and E1+ strata."""
    T = int(plan['num_turns'])
    Y = float(plan['terminal_value'])
    alpha = float(plan['inference']['alpha'])
    turn_test = plan['inference'].get('turn_test', 'newcombe_one_sided_bonferroni')

    active_s1 = _active_receipts(receipts, 'stage1')
    active_s2 = _active_receipts(receipts, 'stage2')

    policy_violations = [
        r['trial_id'] for r in receipts
        if (r.get('outcome') or {}).get('error_reason') in ('policy_identity_violation', 'model_version_drift')
    ]
    model_versions = sorted({r.get('model_version') for r in receipts if r.get('model_version')})

    grid_turns = [int(cp[2:]) for cp in plan['grid']['checkpoints']]
    s1_samples: dict[int, list[dict]] = {
        int(cp_id[2:]): [r for r in active_s1 if r['checkpoint_id'] == cp_id]
        for cp_id in plan['grid']['checkpoints']
    }
    s2_samples: dict[int, list[dict]] = {
        int(cp_id[2:]): [r for r in active_s2 if r['checkpoint_id'] == cp_id]
        for cp_id in stage2_plan['checkpoints']
    }

    curve_eval = evaluate_curve(
        grid_turns=grid_turns,
        stage1_samples=s1_samples,
        stage2_samples=s2_samples,
        terminal_turn=T,
        terminal_value=Y,
        alpha=alpha,
        turn_test=turn_test,
    )

    if policy_violations or len(model_versions) > 1:
        verdict = 'not_identifiable'
        verdict_reason = 'policy_identity_or_model_version_violation'
    else:
        verdict = curve_eval['curve_verdict']
        verdict_reason = 'two_stage_sample_split_value_curve'

    active_all = active_s1 + active_s2
    case_counts = {
        'scored': sum(1 for r in active_all if r['outcome'].get('outcome_case') == 'scored'),
        'empty_submission': sum(1 for r in active_all if r['outcome'].get('outcome_case') == 'empty_submission'),
        'unscorable_submission': sum(1 for r in active_all if r['outcome'].get('outcome_case') == 'unscorable_submission'),
        'infrastructure_invalid': sum(1 for r in active_all if r['outcome'].get('outcome_case') == 'infrastructure_invalid'),
        'scorer_infrastructure_invalid': sum(1 for r in active_all if r['outcome'].get('outcome_case') == 'scorer_infrastructure_invalid'),
        'replacements_used': sum(1 for r in receipts if r.get('replacement_attempt', 0) > 0),
    }

    j_star = int(curve_eval['candidate_turn_in_interval'])
    cp_before = f'cp{j_star - 1:06d}'
    j_rows = [r for r in active_s2 if r['checkpoint_id'] == cp_before]
    is_docx = any(bool(r.get('is_docx_task', True)) for r in j_rows) if j_rows else True
    v_before = (curve_eval['checkpoint_estimates'].get(cp_before) or {}).get('stage2_mean')
    turn_drop = curve_eval['primary_inference'].get('selected_turn_drop')
    strata_summary = evaluate_strata(
        j_rows,
        kinds=plan['strata']['kinds'],
        min_per_class=int(plan['strata']['min_per_class']),
        is_docx_task=is_docx,
        v_before=v_before,
        turn_drop=turn_drop,
    )

    # Exploratory dim2 rubric item drops across Stage 2 interval [a, b]
    a_turn, b_turn = stage2_plan['selected_interval']
    cp_a = f'cp{a_turn:06d}'
    cp_b = f'cp{b_turn:06d}' if b_turn < T else None
    a_dim2 = [r['outcome'].get('dim2_hits') or {} for r in active_s2 if r['checkpoint_id'] == cp_a and r['outcome']['endpoint_defined']]
    b_dim2 = (
        [r['outcome'].get('dim2_hits') or {} for r in active_s2 if r['checkpoint_id'] == cp_b and r['outcome']['endpoint_defined']]
        if cp_b else []
    )
    dim2_keys = sorted({k for d in a_dim2 + b_dim2 for k in d})
    dim2_curves = {}
    dim2_pvals = []
    for key in dim2_keys:
        va = [d[key] for d in a_dim2 if key in d]
        vb = [d[key] for d in b_dim2 if key in d] if cp_b else [0.0]
        mean_a = sum(va) / len(va) if va else 0.0
        mean_b = sum(vb) / len(vb) if vb else 0.0
        drop = mean_a - mean_b
        dim2_curves[key] = {'mean_start': mean_a, 'mean_end': mean_b, 'drop': drop}
        dim2_pvals.append(max(1e-6, 1.0 - max(0.0, drop)))
    adj_p = holm_adjust(dim2_pvals) if dim2_pvals else []
    for key, p_adj in zip(dim2_keys, adj_p):
        dim2_curves[key]['p_value_holm'] = p_adj

    return {
        'plan_state': 'completed' if not policy_violations else 'execution_invalid',
        'verdict': verdict,
        'verdict_reason': verdict_reason,
        'evidence_basis': 'value_curve',
        'evidence_level': 'E1' if verdict in ('concentrated_turn', 'concentrated_interval') else 'descriptive',
        'mode': 'test_fixture' if fixture else 'live_l1_bridge',
        'endpoint': plan['endpoint'],
        'curve': {
            **curve_eval,
            'selected_interval_detail': curve_eval['selected_interval'],
            'selected_interval': [a_turn, b_turn],
            'argmax_turn': j_star,
        },
        'outcome_case_counts': case_counts,
        'policy_identity_violations': policy_violations,
        'observed_model_versions': model_versions,
        'strata': {
            'checkpoint_id': cp_before,
            'resampled_turn': j_star,
            **strata_summary,
        },
        'exploratory_dim2_item_curves': dim2_curves,
        'stats_policy_version': plan['stats_policy_version'],
        'missing_outcomes_not_discarded': True,
    }


def report_curve(analysis: Path, execution: Path, output: Path) -> dict:
    """Seals a verified value_curve_results package linking frozen diagnosis predictions and curve execution."""
    from ..package import verify, seal
    from ..evaluation.metrics import value_prediction_scores, curve_gold

    av = verify(analysis)
    ev = verify(execution)
    if not av['verified'] or not ev['verified']:
        raise L2Error('Verified diagnosis and value_curve_execution package required')
    plan = read_json(execution / 'frozen_plan.json')
    am = read_json(analysis / 'manifest.json')
    if plan['analysis_index_sha256'] != av['index_sha256'] or plan['analysis_id'] != am['analysis_id']:
        raise IntegrityError('Value curve execution does not belong to the frozen analysis')

    out = Path(output).absolute()
    if out.resolve().is_relative_to(analysis.resolve()) or out.resolve().is_relative_to(execution.resolve()):
        raise L2Error('Use separate curve report output directory')
    out.mkdir(parents=True, exist_ok=False, mode=0o700)

    for src in ('frozen_plan.json', 'stage2_plan.json', 'branch_receipts.jsonl', 'results.json'):
        write_new(out, src, (execution / src).read_bytes())
    if (execution / 'baselines.json').exists():
        write_new(out, 'baselines.json', (execution / 'baselines.json').read_bytes())

    diag = read_json(analysis / 'diagnosis.json')
    events = read_json(analysis / 'events.json')
    results = read_json(execution / 'results.json')
    receipts = read_jsonl(execution / 'branch_receipts.jsonl')
    baselines = read_json(execution / 'baselines.json') if (execution / 'baselines.json').exists() else {}

    gold = curve_gold(
        case_id=am['analysis_id'],
        curve_results=results,
        events=events,
    )
    eval_summary = value_prediction_scores(
        case_id=am['analysis_id'],
        family_id=str((am.get('task') or {}).get('task_id') or am['analysis_id']),
        diagnosis=diag,
        baselines=baselines,
        receipts=receipts,
        endpoint=plan['endpoint'],
    )
    write_json(out, 'validation_binding.json', {
        'analysis_index_sha256': av['index_sha256'],
        'execution_index_sha256': ev['index_sha256'],
        'imported_at': now(),
        'original_prediction_modified': False,
    })
    write_json(out, 'curve_gold.json', gold)
    write_json(out, 'value_evaluation.json', eval_summary)
    write_new(
        out,
        'curve_report.md',
        (
            f"# L2 两阶段价值曲线报告\n\n"
            f"- 曲线判定：`{results['verdict']}`（证据依据：`{results['evidence_basis']}`，等级：`{results['evidence_level']}`）\n"
            f"- 选中区间：`{results['curve']['selected_interval']}`，最大单步下跌回合：`{results['curve']['argmax_turn']}`\n"
            f"- 同状态行为分层（回合 {results['strata']['resampled_turn']}）：证据依据 `{results['strata']['evidence_basis']}`（E1+）\n"
        ).encode('utf-8'),
    )
    seal(out, 'value_curve_results')
    return verify(out)


def import_receipts(analysis: Path, execution: Path, output: Path):
    from ..package import verify, seal
    av = verify(analysis); ev = verify(execution)
    if not av['verified'] or not ev['verified']: raise L2Error('Verified diagnosis and controller receipt package required')
    plan = read_json(execution / 'frozen_plan.json')
    if plan.get('package_kind') == 'value_curve_plan' or plan.get('template') == 'value_curve':
        return report_curve(analysis, execution, output)
    am = read_json(analysis / 'manifest.json')
    if plan['analysis_index_sha256'] != av['index_sha256'] or plan['analysis_id'] != am['analysis_id']:
        raise IntegrityError('Experiment does not belong to the frozen analysis')
    out = Path(output).absolute()
    if out.resolve().is_relative_to(analysis.resolve()) or out.resolve().is_relative_to(execution.resolve()): raise L2Error('Use separate validation output')
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    for src, dst in [('frozen_plan.json', 'frozen_plan.json'), ('branch_receipts.jsonl', 'branch_receipts.jsonl'), ('results.json', 'results.json')]:
        write_new(out, dst, (execution / src).read_bytes())
    write_json(out, 'validation_binding.json', {'analysis_index_sha256': av['index_sha256'], 'execution_index_sha256': ev['index_sha256'],
        'imported_at': now(), 'original_prediction_modified': False, 'execution_package_required_for_branch_revalidation': True})
    write_json(out, 'statistical_analysis.json', {'hypothesis_verdict': 'not_identifiable', 'causal_effect': None,
        'reason': 'No implemented treatment arm; receipt import cannot upgrade a baseline to causal support'})
    write_new(out, 'validation_report.md', b'# L2 receipt validation\n\nBaseline continuation receipts imported. No causal intervention contrast is identified. Original predictions were not modified.\n')
    seal(out, 'experiment_validation')
    return verify(out)


def check_execution_or_validation(root, kind):
    from ..io import check_journal
    plan = read_json(root / 'frozen_plan.json')
    receipts = read_jsonl(root / 'branch_receipts.jsonl')
    results = read_json(root / 'results.json')

    if kind in ('value_curve_execution', 'value_curve_results'):
        if plan.get('template') != 'value_curve' or plan.get('plan_state') != 'frozen':
            raise IntegrityError('Unsupported value_curve plan state or template')
        stage2_plan = read_json(root / 'stage2_plan.json')
        expected_stage2 = derive_stage2(plan, receipts)
        if stage2_plan != expected_stage2:
            raise IntegrityError('Stage 2 plan differs from deterministic Stage 1 derivation')
        if len({r['trial_id'] for r in receipts}) != len(receipts):
            raise IntegrityError('Duplicate trial receipt in value curve package')
        active_s1 = _active_receipts(receipts, 'stage1')
        active_s2 = _active_receipts(receipts, 'stage2')
        expected_s1_count = len(plan['grid']['checkpoints']) * int(plan['stage1']['replicates'])
        expected_s2_count = len(stage2_plan['checkpoints']) * int(stage2_plan['replicates'])
        if len(active_s1) != expected_s1_count or len(active_s2) != expected_s2_count:
            raise IntegrityError('Active Stage 1 or Stage 2 receipt count differs from frozen plan')
        for r in receipts:
            cp_id = r['checkpoint_id']
            if r['source_index_sha256'] != plan['source_index_sha256'] or r['checkpoint_sha256'] != plan['qualified_checkpoint_sha256'].get(cp_id):
                raise IntegrityError('Curve receipt provenance differs from frozen plan')
        if kind == 'value_curve_execution':
            from ..evidence.l1 import L1Reader
            auth = read_json(root / 'authorization.json')
            trace = read_jsonl(root / 'experiment_trace.jsonl')
            check_journal(trace)
            if sha_file(root / 'frozen_plan.json') != auth['plan_sha256'] or not auth['explicit_allow_execution']:
                raise IntegrityError('Curve plan/authorization mismatch')
            trace_receipts = {e['data']['trial_id']: e['data'] for e in trace if e['kind'] == 'branch.receipt'}
            receipt_map = {r['trial_id']: r for r in receipts}
            if set(trace_receipts) != set(receipt_map):
                raise IntegrityError('Journal/receipt trial_id set mismatch')
            for tid, r in receipt_map.items():
                tr = dict(trace_receipts[tid])
                rc = dict(r)
                tr.pop('replaced_by', None)
                rc.pop('replaced_by', None)
                if tr != rc:
                    raise IntegrityError(f'Journal/receipt content mismatch for {tid}')
                if r['branch_verified']:
                    branch = L1Reader(safe(root, r['branch_path']), expected_index=r['branch_index_sha256'])
                    if outcome(branch, plan['endpoint'], v02_policy=True) != r['outcome']:
                        raise IntegrityError('Receipt outcome differs from branch recomputation')
                    if branch.manifest['episode_id'] != r['new_episode_id']:
                        raise IntegrityError('Receipt branch episode_id differs')
            fixture = auth['mode'] == 'test_fixture'
        else:
            fixture = results.get('mode') == 'test_fixture'
        expected = summarize_curve(receipts, plan, stage2_plan, fixture=fixture)
        if expected != results:
            raise IntegrityError('Value curve aggregate differs from deterministic stats recomputation')
        if results.get('evidence_basis') != 'value_curve':
            raise IntegrityError('Value curve results must declare evidence_basis=value_curve')
        if results.get('strata', {}).get('evidence_basis') != 'resample_stratified':
            raise IntegrityError('Strata results must declare evidence_basis=resample_stratified')
        return results['plan_state']

    if plan['template'] != 'resume_unchanged' or plan['plan_state'] != 'frozen': raise IntegrityError('Unsupported executed plan')
    if len(receipts) != sum(a['replicates'] for a in plan['arms']): raise IntegrityError('Missing declared trial receipts')
    if len({r['trial_id'] for r in receipts}) != len(receipts): raise IntegrityError('Duplicate trial receipt')
    for r in receipts:
        if r['source_index_sha256'] != plan['source_index_sha256'] or r['checkpoint_sha256'] != plan['checkpoint_sha256']:
            raise IntegrityError('Receipt provenance differs from plan')
    if kind == 'experiment_execution':
        from ..evidence.l1 import L1Reader
        auth = read_json(root / 'authorization.json'); trace = read_jsonl(root / 'experiment_trace.jsonl'); check_journal(trace)
        if sha_file(root / 'frozen_plan.json') != auth['plan_sha256'] or not auth['explicit_allow_execution']:
            raise IntegrityError('Plan/authorization mismatch')
        if [e['data'] for e in trace if e['kind'] == 'branch.receipt'] != receipts: raise IntegrityError('Journal/receipt mismatch')
        for r in receipts:
            if r['branch_verified']:
                branch = L1Reader(safe(root, r['branch_path']), expected_index=r['branch_index_sha256'])
                if outcome(branch, plan['endpoint']) != r['outcome']: raise IntegrityError('Receipt endpoint differs from branch')
                if branch.manifest['episode_id'] != r['new_episode_id']: raise IntegrityError('Receipt branch identity differs')
        expected = summarize(receipts, plan, fixture=auth['mode'] == 'test_fixture')
        if expected != results: raise IntegrityError('Experiment aggregate differs')
    else:
        if read_json(root / 'statistical_analysis.json')['hypothesis_verdict'] != 'not_identifiable':
            raise IntegrityError('Baseline import cannot claim a verified mechanism')
    if results['hypothesis_verdict'] not in ('untested', 'not_identifiable'): raise IntegrityError('Unsupported causal verdict')
    return results['plan_state']
