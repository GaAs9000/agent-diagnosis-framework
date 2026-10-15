"""Bounded fresh native runs with immutable attempt receipts and verified reuse."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import os
from pathlib import Path
import re
import threading

from .common import L3Error, digest, read, write
from .limiter import RequestLimiter
from .plan import verify_plan


# These lazy boundaries are monkeypatched only by offline tests.
def _native_run(config, output, *, request_limiter):
    from c3_sandbox.l1.core.orchestrator import run
    return run(config, output, request_limiter=request_limiter)


def _verify_package(package, expected):
    from c3_sandbox.l1.package import verify_package
    return verify_package(package, expected_index_sha256=expected)


def _summarize(package, job, attempt):
    from .outcomes import summarize_episode
    return summarize_episode(package, job, attempt, verify_evidence=True)


def _safe_tree(path):
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise L3Error('Symlink in execution path')
    if path.exists() and any(p.is_symlink() for p in path.rglob('*')):
        raise L3Error('Symlink in execution artifacts')


@contextmanager
def _plan_lock(directory):
    path = directory / '.run.lock'
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise L3Error('Another process is executing this plan') from exc
        yield
    finally:
        os.close(fd)


@contextmanager
def _credentials(plan):
    """Load only named credentials as literal dotenv values, without shell evaluation."""
    names = {job['config']['model'].get('api_key_env') for job in plan['jobs']} - {None, ''}
    if any(not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name) for name in names):
        raise L3Error('Invalid credential environment name')
    workspace = next((p for p in Path(__file__).resolve().parents if (p / 'PROJECT_MEMORY.md').is_file()), None)
    paths = [] if workspace is None else [workspace / '.env', workspace / 'L1_sandbox/C3_Experimental_Sandbox/.env']
    loaded = {}
    try:
        for name in sorted(names):
            if os.environ.get(name):
                continue
            value = None
            for path in paths:
                if not path.is_file() or path.is_symlink():
                    continue
                for line in path.read_text().splitlines():
                    match = re.fullmatch(r'\s*(?:export\s+)?' + re.escape(name) + r'\s*=\s*(.*?)\s*', line)
                    if match:
                        value = match[1]
                        if value[:1] in ('"', "'"):
                            if len(value) < 2 or value[-1] != value[0]:
                                raise L3Error('Malformed credential dotenv value')
                            value = value[1:-1]
                        else:
                            value = value.split(' #', 1)[0].strip()
            if not value or any(c in value for c in ('\n', '\r', '$', '`')):
                raise L3Error('Missing or unsupported credential value')
            loaded[name] = os.environ.get(name)
            os.environ[name] = value
        yield
    finally:
        for name, original in loaded.items():
            if original is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = original


_transport_scope_lock = threading.Lock()


@contextmanager
def _transport_guard(limiter):
    # L1's admission callback has no status argument. Observe the real HTTP receipt
    # here so a 401/403/error envelope latches before its next retry admission.
    from c3_sandbox.l1.models.http import NativeHTTP
    with _transport_scope_lock:
        original = NativeHTTP.post
        def guarded(instance, *args, **kwargs):
            limiter.check_send()
            result = original(instance, *args, **kwargs)
            limiter.observe(result)
            return result
        NativeHTTP.post = guarded
        try:
            yield
        finally:
            NativeHTTP.post = original


def _binding(plan, job):
    return {'experiment_id': plan['experiment_id'], 'job_sha256': digest(job),
            'config_sha256': job['config_sha256'], 'task_id': job['task_id']}


def _evidence(package, job, expected):
    _safe_tree(package)
    actual = hashlib.sha256((package / 'evidence_index.json').read_bytes()).hexdigest()
    if actual != expected:
        raise L3Error('Evidence index differs from attempt receipt')
    report = _verify_package(package, expected)
    if report.get('verified') is not True:
        raise L3Error('Evidence seal verification failed')
    manifest = read(package / 'manifest.json')
    task = manifest.get('task', {})
    if (manifest.get('recording_mode') != 'live'
            or manifest.get('config_sha256') != job['config_sha256']
            or task.get('task_id', task.get('id')) != job['task_id']
            or manifest.get('lineage')):
        raise L3Error('Episode is not a fresh live run of the frozen task/config')
    for name in ('trajectory.jsonl', 'state_changes.jsonl', 'official_grade.json',
                 'artifacts/inventory.json', 'raw/resolved_config.json'):
        if not (package / name).is_file():
            raise L3Error('Incomplete episode evidence')
    if digest(read(package / 'raw/resolved_config.json')) != job['config_sha256']:
        raise L3Error('Resolved episode config differs from frozen job')


def _save_receipt(path, row):
    write(path, {'row': row, 'sha256': digest(row)})


def _load_receipt(path, binding):
    saved = read(path)
    row = saved.get('row')
    if not isinstance(row, dict) or digest(row) != saved.get('sha256') or row.get('binding') != binding:
        raise L3Error('Corrupt or mismatched attempt receipt')
    if type(row.get('accepted')) is not bool or (not row['accepted'] and row.get('study_valid') is True):
        raise L3Error('Inconsistent attempt acceptance fields')
    return row


def _verify_summary(row, summary, job, attempt):
    """A receipt seal is insufficient: rederive each scientific/accounting field."""
    if not isinstance(summary, dict) or summary.get('study_valid') is not True:
        raise L3Error('Accepted outcome no longer qualifies')
    if summary.get('test_fixture') or summary.get('recording_mode') == 'test_fixture':
        raise L3Error('Fixture outcomes cannot qualify as production trials')
    expected_identity = {k: v for k, v in job.items() if k != 'config'}
    expected_identity['attempt'] = attempt
    if any(key not in row for key in expected_identity):
        raise L3Error('Receipt lacks frozen trial identity')
    if digest({key: row[key] for key in expected_identity}) != digest(expected_identity):
        raise L3Error('Receipt differs from frozen trial identity')
    if 'config' in row and digest(row['config']) != job['config_sha256']:
        raise L3Error('Receipt config differs from frozen job')
    if any(key not in row for key in summary):
        raise L3Error('Receipt lacks rederived outcome fields')
    if digest({key: row[key] for key in summary}) != digest(summary):
        raise L3Error('Receipt outcome differs from sealed episode')


def _run_job(directory, plan, job, limiter, publish):
    trial_id = job['trial_id']
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', trial_id) or trial_id in ('.', '..'):
        raise L3Error('Unsafe trial ID')
    root = directory / 'runs' / trial_id
    root.mkdir(parents=True, exist_ok=True)
    _safe_tree(root)
    binding = _binding(plan, job)
    existing = sorted(root.iterdir())
    if any(not p.is_dir() or not re.fullmatch(r'attempt_[0-9]{3}', p.name) for p in existing):
        raise L3Error('Unexpected attempt entry')
    if [p.name for p in existing] != [f'attempt_{i:03d}' for i in range(len(existing))]:
        raise L3Error('Attempt numbering gap')
    for index, attempt in enumerate(existing):
        receipt = attempt / 'receipt.json'
        if receipt.exists():
            try:
                row = _load_receipt(receipt, binding)
            except Exception:
                publish({**{k: v for k, v in job.items() if k != 'config'},
                         'attempt': index, 'binding': binding, 'accepted': False,
                         'study_valid': False, 'status': 'receipt_corrupt'})
                raise
            if row.get('status') == 'running':
                row = {**row, 'status': 'interrupted'}
            if row.get('accepted'):
                try:
                    _evidence(attempt / 'episode', job, row['expected_index_sha256'])
                    summary = _summarize(attempt / 'episode', job, index)
                    _verify_summary(row, summary, job, index)
                except Exception as exc:
                    row = {**row, 'accepted': False, 'reuse_invalid': True, 'runner_error': type(exc).__name__}
                    publish(row)
                    # A trusted accepted receipt whose evidence changed is an integrity
                    # failure, not permission to spend on a replacement.
                    raise L3Error('Accepted attempt evidence failed revalidation') from exc
                publish(row)
                return
            publish(row)
        else:
            publish({**job, 'config': None, 'attempt': index, 'binding': binding,
                     'accepted': False, 'status': 'interrupted', 'replacement_of': index - 1 if index else None})
    maximum = 1 + plan['limits'].get('max_replacements_per_slot', 2)
    for index in range(len(existing), maximum):
        state = limiter.snapshot()
        if state['fatal_auth'] or state['count'] >= limiter.cap:
            return
        attempt = root / f'attempt_{index:03d}'
        attempt.mkdir()  # Directory creation durably consumes this finite slot.
        row = {k: v for k, v in job.items() if k != 'config'}
        row.update(attempt=index, binding=binding, accepted=False, status='running',
                   replacement_of=index - 1 if index else None, episode=str(attempt / 'episode'))
        # Persist running receipt before invoking native execution; crash gaps still count.
        _save_receipt(attempt / 'receipt.json', row)
        publish(row)
        try:
            _native_run(copy.deepcopy(job['config']), attempt / 'episode', request_limiter=limiter)
            expected = hashlib.sha256((attempt / 'episode/evidence_index.json').read_bytes()).hexdigest()
            row['expected_index_sha256'] = expected
            _evidence(attempt / 'episode', job, expected)
            summary = _summarize(attempt / 'episode', job, index)
            if summary.get('test_fixture') or summary.get('recording_mode') == 'test_fixture':
                raise L3Error('Fixture outcomes cannot qualify as production trials')
            row.update(summary)
            row['accepted'] = summary.get('study_valid') is True
            if row['accepted']:
                _verify_summary(row, summary, job, index)
            row['status'] = 'completed' if row['accepted'] else 'execution_invalid'
        except Exception as exc:
            row.update(status='runner_error', runner_error=type(exc).__name__, accepted=False,
                       study_valid=False, study_outcome_y=None)
        _save_receipt(attempt / 'receipt.json', row)
        publish(row)
        if row['accepted']:
            return


def run_plan(directory: Path, *, execute: bool = False) -> dict:
    directory = Path(directory).absolute()
    _safe_tree(directory)
    plan = verify_plan(directory)
    if not execute:
        return {'status': 'planned', 'execute': False, 'experiment_id': plan['experiment_id'],
                'workload': len(plan['jobs']), 'limits': plan['limits']}
    with _plan_lock(directory):
        if not (directory / 'http_admissions.json').exists() and (
                (directory / 'run_results.json').exists() or
                any((directory / 'runs').glob('*/attempt_*/receipt.json')) or
                any((directory / 'runs').glob('*/attempt_*/episode'))):
            raise L3Error('HTTP admission ledger is missing from an existing execution')
        limiter = RequestLimiter(directory, rpm=plan['limits'].get('rpm'), tpm=plan['limits'].get('tpm'),
                                 max_http_attempts=plan['limits']['max_http_attempts'])
        rows, errors = {}, []
        result_lock = threading.Lock()
        def snapshot(status='running'):
            return {'experiment_id': plan['experiment_id'], 'status': status,
                    'rows': [rows[k] for k in sorted(rows)], 'runner_errors': list(errors),
                    'http_admissions': limiter.snapshot()}
        def publish(row):
            with result_lock:
                rows[(row['trial_id'], row['attempt'])] = row.copy()
                write(directory / 'run_results.json', snapshot())
        try:
            with _credentials(plan), _transport_guard(limiter), ThreadPoolExecutor(
                    max_workers=plan['limits']['max_concurrency']) as pool:
                futures = {pool.submit(_run_job, directory, plan, job, limiter, publish): job for job in plan['jobs']}
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        with result_lock:
                            errors.append({'trial_id': futures[future]['trial_id'], 'error': type(exc).__name__})
                            write(directory / 'run_results.json', snapshot())
        except Exception as exc:
            errors.append({'error': type(exc).__name__})
        accepted = {r['trial_id'] for r in rows.values() if r.get('accepted')}
        state = limiter.snapshot()
        status = ('complete' if len(accepted) == len(plan['jobs']) and not errors else
                  'blocked_auth' if state['fatal_auth'] else
                  'blocked_http_cap' if state['count'] >= limiter.cap else 'incomplete')
        result = snapshot(status)
        write(directory / 'run_results.json', result)
        if errors:
            raise L3Error('L3 runner failed; inspect run_results.json for retained progress')
        return result
