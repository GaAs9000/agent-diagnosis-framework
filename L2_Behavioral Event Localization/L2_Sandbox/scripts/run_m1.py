#!/usr/bin/env python3
"""Frozen M1 engineering pilot. Preparation is offline; paid execution is opt-in."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
L1 = ROOT / 'L1_sandbox/C3_Experimental_Sandbox'
L2 = Path(__file__).resolve().parents[1]
for source in (L1 / 'src', L2 / 'src'):
    sys.path.insert(0, str(source))
TASKS = ('officeval_003', 'officeval_090', 'officeval_060', 'sokoban')
BASELINE = 'M1 engineering placeholder p0=0.5; not an M2 cohort estimate or calibrated probability'


class GateError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(canonical(value) + b'\n')
    temporary.replace(path)


def image_id(image):
    result = subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}}', image],
                            check=True, capture_output=True, text=True).stdout.strip()
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', result):
        raise GateError('Invalid Docker image identity')
    return result


def sources():
    files = {Path(__file__).resolve()}
    for directory in (L1 / 'src/c3_sandbox/l1', L2 / 'src/l2_sandbox'):
        files.update(directory.rglob('*.py'))
    files.update(p for p in (L1 / 'docker/lhtb_sokoban_cold').rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts)
    return {str(p.resolve()): sha(p) for p in sorted(files)}


def tree_identity(root):
    """Caller submission/evidence identity includes contents, permissions and mtimes."""
    root = Path(root)
    rows = []
    for path in [root, *sorted(root.rglob('*'))]:
        if path.is_symlink():
            raise GateError('Symlink in immutable evidence/submission')
        stat = path.stat()
        rows.append([str(path.relative_to(root)), stat.st_mode, stat.st_mtime_ns,
                     sha(path) if path.is_file() else None])
    return hashlib.sha256(canonical(rows)).hexdigest()


def prepare(identifier):
    from c3_sandbox.l1.core.config import resolve
    from c3_sandbox.l1.core.orchestrator import build_execution_contract_prompt
    directory = L2 / 'validation' / ('m1_' + identifier)
    if directory.exists():
        raise GateError('Plan already exists; use a new id or verify the frozen plan')
    templates = {task: L1 / 'configs/l1' / (
        'lhtb_sokoban.gemini-35lite.t1.json' if task == 'sokoban' else 'officeval.gemini-35lite.t1.json')
        for task in TASKS}
    images = {tag: image_id(tag) for tag in (
        'l1-office:0.2.0', 'l1-lhtb-sokoban-cold:0.2.1', 'zli12321/lhtb-sokoban:20260615')}
    configs = {}
    locks = {}
    for task, template in templates.items():
        cfg = resolve(template)
        cfg['benchmark']['task_id'] = task
        cfg['runtime']['image'] = images['l1-lhtb-sokoban-cold:0.2.1' if task == 'sokoban' else 'l1-office:0.2.0']
        cfg['evaluation']['image'] = images['zli12321/lhtb-sokoban:20260615' if task == 'sokoban' else 'l1-office:0.2.0']
        cfg['agent']['system_prompt'] = build_execution_contract_prompt(cfg)
        if cfg['model']['parameters'].get('temperature') != 1.0 or any(
            key in cfg['model']['parameters'] for key in ('seed', 'random_seed', 'randomSeed')):
            raise GateError('M1 requires temperature 1 and no sampling seed')
        budget = cfg['budget']
        if (budget['max_model_requests'], budget['max_tool_calls']) != ((20, 30) if task == 'sokoban' else (35, 50)):
            raise GateError('Unexpected parent budget')
        lock = Path(cfg['benchmark']['data_lock'])
        locks[str(lock)] = sha(lock)
        configs[task] = cfg
    plan = {'id': identifier, 'git_base_head': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
             check=True, capture_output=True, text=True).stdout.strip(),
            'source_sha256': sources(), 'templates_sha256': {str(p): sha(p) for p in set(templates.values())},
            'locks_sha256': locks, 'images': images, 'configs': configs,
            'settings': {'n1': 2, 'n2': 2, 'stride': 4, 'alpha': .05, 'max_concurrency': 2,
                         'rpm': 30, 'tpm': 10_000_000, 'max_invalid_retries_per_slot': 2,
                         'unique_branch_bounds': {'officeval': 28, 'sokoban': 20},
                         'p0': .5, 'baseline_provenance': BASELINE, 'additional_regrades': 3,
                         'max_admitted_http_attempts': 2000,
                         'uncapped_worst_case_http_attempts': 41080,
                         'attempt_bound_basis': 'parents 3*35*4+20*4; branch slots (3*28*35+20*20)*3 attempts*4 HTTP retries; 3*35*4+20*4 success probes'}}
    directory.mkdir(parents=True)
    for task, cfg in configs.items():
        write(directory / 'configs' / (task + '.json'), cfg)
    plan['configs_sha256'] = {str(directory / 'configs' / (task + '.json')):
                             sha(directory / 'configs' / (task + '.json')) for task in TASKS}
    write(directory / 'frozen_plan.json', plan)
    (directory / 'frozen_plan.sha256').write_text(sha(directory / 'frozen_plan.json') + '\n')
    return directory


def verify_plan(directory):
    path = directory / 'frozen_plan.json'
    if sha(path) != (directory / 'frozen_plan.sha256').read_text().strip():
        raise GateError('Frozen plan hash mismatch')
    plan = read(path)
    if sources() != plan['source_sha256']:
        raise GateError('Frozen source inventory/hash changed')
    for field in ('templates_sha256', 'locks_sha256', 'configs_sha256'):
        for name, expected in plan[field].items():
            if not Path(name).is_file() or sha(name) != expected:
                raise GateError('Frozen input hash changed: ' + name)
    for task, config in plan['configs'].items():
        if read(directory / 'configs' / (task + '.json')) != config:
            raise GateError('Frozen resolved config changed')
    for tag, expected in plan['images'].items():
        if image_id(tag) != expected or image_id(expected) != expected:
            raise GateError('Frozen image changed: ' + tag)
    return plan


def load_credentials():
    # Never source shell code or print dotenv content. Only two approved keys load.
    for path in (ROOT / '.env', L1 / '.env'):
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            match = re.fullmatch(r'\s*(?:export\s+)?(GEMINI_API_KEY|GOOGLE_API_KEY)\s*=\s*(.*?)\s*', line)
            if match:
                key, value = match.groups()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                os.environ.setdefault(key, value)
    if not os.environ.get('GEMINI_API_KEY') and os.environ.get('GOOGLE_API_KEY'):
        os.environ['GEMINI_API_KEY'] = os.environ['GOOGLE_API_KEY']
    if not os.environ.get('GEMINI_API_KEY'):
        raise GateError('GEMINI_API_KEY/GOOGLE_API_KEY is unavailable')


def fatal_auth_receipt(roots):
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob('http.json'):
            if read(path).get('status_code') in (401, 403):
                return True
        for path in root.rglob('response.bin'):
            if b'API_KEY_INVALID' in path.read_bytes():
                return True
    return False


class AuthGuardLimiter:
    """Shared auth latch checked immediately before every new HTTP admission."""
    def __init__(self, limiter, roots, counter_path=None, cap=2000):
        self.limiter, self.roots = limiter, roots
        self.fatal = threading.Event()
        self.counter_path = counter_path
        self.cap = cap
        self.lock = threading.Lock()

    def check(self):
        if fatal_auth_receipt(self.roots):
            self.fatal.set()
        if self.fatal.is_set():
            raise GateError('Fatal authentication receipt; further model sends blocked')

    def acquire(self, estimated_tokens=1000., timeout=None):
        self.check()
        admitted = self.limiter.acquire(estimated_tokens, timeout=timeout)
        self.check()
        if admitted and self.counter_path is not None:
            import fcntl
            with self.lock, self.counter_path.with_suffix('.lock').open('a') as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                counter = read(self.counter_path) if self.counter_path.exists() else {'admitted': 0}
                if counter['admitted'] >= self.cap:
                    raise GateError('Frozen global HTTP admission cap exhausted')
                counter['admitted'] += 1
                counter['cap'] = self.cap
                write(self.counter_path, counter)
        return admitted

    def reconcile(self, estimated_tokens, actual_tokens):
        return self.limiter.reconcile(estimated_tokens, actual_tokens)


def failure_selected(pkg):
    """Use the package's canonical failure selection, including determinate zero outcomes."""
    return pkg.is_failed_trajectory()


def valid_parent(pkg):
    status = pkg.manifest.get('status', {})
    return (pkg.manifest.get('recording_mode') == 'live'
            and status.get('execution_validity') == 'valid'
            and status.get('evidence_completeness') == 'complete'
            and pkg.outcome()['valid'])


def grader_healthy(benchmark, task_id, grade, output):
    """Determinate agent failure is separate from native rubric projection validity."""
    raw = grade.get('official_raw')
    if raw is None or grade.get('error_kind') == 'infrastructure':
        return False
    projection = grade.get('projection') or {}
    if benchmark == 'lhtb':
        return projection.get('evaluation_validity') == 'valid'
    from c3_sandbox.l1.evaluation.officeval import interpret
    process = grade.get('process') or {}
    if process.get('container_exit_code') != 0 or process.get('stop_reason') is not None:
        return False
    worker_path = Path(output) / 'worker-output/worker.json'
    if not worker_path.is_file():
        return False
    worker = read(worker_path)
    if worker.get('official_returned') is not True or worker.get('exception') is not None:
        return False
    required = {'id', 'file_name', 'status', 'error', 'dim1_pass', 'dim1_reason',
                'dim2_items', 'total_score', 'max_score'}
    if not isinstance(raw, dict) or not required.issubset(raw) or raw.get('id') != task_id[-3:]:
        return False
    native = interpret(raw, task_id)
    if native != projection:
        return False
    if native.get('evaluation_validity') == 'valid':
        return True
    # Upstream evaluate returns this complete shape for missing/damaged submissions.
    infrastructure_markers = ('TimeoutExpired', 'TimeoutError', 'Docker', 'Container',
        'ModuleNotFoundError', 'ImportError', 'ConnectionError', 'MemoryError',
        'PermissionError', 'SubprocessError', 'scorer_crash', 'infra_error')
    if any(marker in str(raw.get('error') or '') for marker in infrastructure_markers):
        return False
    import math
    if any(type(raw.get(key)) not in (int, float) or not math.isfinite(raw[key])
           for key in ('total_score', 'max_score')):
        return False
    return (raw.get('status') == 'error' and isinstance(raw.get('error'), str)
            and bool(raw['error']) and raw.get('dim1_pass') is False
            and isinstance(raw.get('dim1_reason'), str) and isinstance(raw.get('dim2_items'), list)
            and native.get('reason') == 'official_error')


def substantive_grade(grade):
    ignored = {'provenance', 'runtime', 'process', 'process_result', 'elapsed_seconds',
               'duration_seconds', 'stdout', 'stderr', 'container_id', 'started_at', 'finished_at'}
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if k not in ignored}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value
    return clean({k: grade.get(k) for k in ('official_raw', 'projection')})


def regrade(pkg, config, directory):
    from c3_sandbox.l1.core.contracts import EvaluationSpec
    from c3_sandbox.l1.evaluation.registry import registry
    submission = pkg.root / 'artifacts/submission'
    before = tree_identity(submission)
    description = pkg.json('benchmark/description.json')
    e = description['evaluation']
    spec = EvaluationSpec(e['kind'], pkg.root / 'benchmark/source', e['entrypoint'], description['task_id'], e['metadata'])
    if pkg.benchmark == 'lhtb' and image_id(spec.metadata['image']) != config['evaluation']['image']:
        raise GateError('LHTB official grader image differs from frozen identity')
    evaluator = registry().create(config['evaluation']['adapter'], config['evaluation'])
    original_grade = pkg.json('official_grade.json')
    if not grader_healthy(pkg.benchmark, pkg.task_id, original_grade, pkg.root / 'grading'):
        raise GateError('Original official grader is missing, malformed or unhealthy')
    expected = substantive_grade(original_grade)
    receipts = []
    for index in range(3):
        target = directory / f'grade_{index + 1}'
        receipt_path = directory / f'receipt_{index + 1}.json'
        if target.exists():
            if not receipt_path.exists():
                raise GateError('Incomplete regrade; refusing automatic rerun')
            receipt = read(receipt_path)
            if receipt.get('output_identity') != tree_identity(target) or receipt.get('submission_identity') != before:
                raise GateError('Regrade receipt changed')
            grade = receipt['grade']
        else:
            grade = evaluator.evaluate(spec, submission, target)
            write(receipt_path, {'grade': grade, 'submission_identity': before,
                                 'output_identity': tree_identity(target)})
        receipts.append(grade)
        if tree_identity(submission) != before:
            raise GateError('Grader mutated caller submission')
    passed = all(grader_healthy(pkg.benchmark, pkg.task_id, g, directory / f'grade_{index + 1}')
                 and substantive_grade(g) == expected for index, g in enumerate(receipts))
    if not passed:
        raise GateError('Independent official regrades are invalid or disagree')
    return {'passed': passed, 'additional_evaluations': 3, 'submission_identity': before}


def verify_child(parent, child_dir, k, config):
    from l2_sandbox.evidence.l1 import L1Package
    from c3_sandbox.l1.models.registry import registry
    from c3_sandbox.l1.runtime.tools import TOOLS
    from c3_sandbox.l1.storage import canonical as native_canonical
    child = L1Package(child_dir)
    child.validate_branch(parent, k)
    checkpoint = parent.json(parent.validate_resume_checkpoint(k)['path'])
    image = checkpoint['qualification']['image_id']
    if image != config['runtime']['image'] or (child.manifest.get('runtime') or {}).get('image_id') != image:
        raise GateError('Child runtime/checkpoint image mismatch')
    model = registry().create(config['model']['adapter'], config['model'])
    tools = TOOLS if config['model']['supports_images'] else TOOLS[:1]
    expected = native_canonical(model.request(copy.deepcopy(checkpoint['conversation']), tools))
    requests = sorted((child.root / 'raw/model').glob('r*/request.json'))
    if not requests or requests[0].read_bytes() != expected:
        raise GateError('First resumed native request differs from checkpoint history')
    return child


def rate_summary(statuses):
    denominator = len(statuses)
    invalid = sum(s.get('execution_validity') != 'valid' or s.get('evaluation_validity') != 'valid' for s in statuses)
    wall = sum(s.get('termination') == 'wall_time_budget' for s in statuses)
    return {'denominator': denominator, 'invalid': invalid, 'wall_termination': wall,
            'invalid_rate': invalid / denominator if denominator else None,
            'wall_termination_rate': wall / denominator if denominator else None,
            'passed': denominator > 0 and invalid * 20 < denominator and wall * 100 < denominator}


def summarize(directory, plan):
    parent_root = L1 / 'runs' / ('m1_' + plan['id']) / 'parents'
    statuses = []
    native_invalid = 0
    def metric_status(trial):
        nonlocal native_invalid
        if not (trial / 'manifest.json').exists():
            return {}
        status = read(trial / 'manifest.json').get('status', {})
        if status.get('evaluation_validity') != 'valid':
            native_invalid += 1
            # Native official_error stays untouched on disk. M1 infrastructure
            # invalid rates use determinate L2 outcomes plus independent health.
            try:
                from l2_sandbox.evidence.l1 import L1Package
                pkg = L1Package(trial)
                if pkg.outcome()['valid'] and grader_healthy(pkg.benchmark, pkg.task_id,
                        pkg.json('official_grade.json'), trial / 'grading'):
                    return {**status, 'evaluation_validity': 'valid'}
            except Exception:
                pass
        return status
    for task in TASKS:
        parent = parent_root / task
        if parent.exists():
            statuses.append(metric_status(parent))
        task_dir = directory / task
        invalid_trials = set()
        recorded_trials = set()
        outcomes_path = task_dir / 'outcomes.jsonl'
        if outcomes_path.exists():
            for line in outcomes_path.read_text().splitlines():
                row = json.loads(line)
                recorded_trials.add(row['trial'])
                if not row.get('valid'):
                    invalid_trials.add(row['trial'])
        for name in ('branches', 'probes'):
            trial_root = task_dir / name
            actual_names = {trial.name for trial in trial_root.iterdir() if trial.is_dir()} if trial_root.exists() else set()
            names = actual_names | (recorded_trials if name == 'branches' else set())
            for trial_name in sorted(names):
                trial = trial_root / trial_name
                status = metric_status(trial)
                if trial_name in invalid_trials:
                    status = {**status, 'execution_validity': 'invalid'}
                statuses.append(status)
    summary = rate_summary(statuses)
    summary['rates_passed'] = summary.pop('passed')
    summary['native_evaluation_invalid_count'] = native_invalid
    summary['invalid_rate_basis'] = 'execution, determinate L2 outcome, healthy official grader and fidelity; native projection validity retained separately'
    progress_path = directory / 'progress.json'
    progress = read(progress_path) if progress_path.exists() else {}
    complete = all(progress.get(phase + ':' + task, {}).get('status') == 'complete'
                   for phase in ('parents', 'analysis') for task in TASKS)
    summary['all_phases_complete'] = complete
    summary['milestone_passed'] = complete and summary['rates_passed']
    summary['scope'] = 'M1 engineering only; partial until all eight task phases complete; no M2 calibration or causal claim'
    if (directory / 'http_admissions.json').exists():
        summary['http_admissions'] = read(directory / 'http_admissions.json')
    write(directory / 'summary.json', summary)
    return summary


def execute(directory, phase, selected=None):
    from c3_sandbox.l1.core.orchestrator import run
    from l2_sandbox.curve import TokenBucketRateLimiter, build_curve_plan
    from l2_sandbox.evidence.l1 import L1Package
    from l2_sandbox import report as report_module
    from l2_sandbox.facts.build import build_facts
    from l2_sandbox.analysts.rules import predict_values_rules
    plan = verify_plan(directory)  # All frozen inputs checked before credentials or any send.
    load_credentials()
    limiter = AuthGuardLimiter(TokenBucketRateLimiter(rpm=30, tpm=10_000_000),
        [directory, L1 / 'runs' / ('m1_' + plan['id'])],
        counter_path=directory / 'http_admissions.json', cap=plan['settings']['max_admitted_http_attempts'])
    limiter.check()
    progress_path = directory / 'progress.json'
    progress = read(progress_path) if progress_path.exists() else {}
    for task in ([selected] if selected else TASKS):
        verify_plan(directory)
        config = plan['configs'][task]
        parent_dir = L1 / 'runs' / ('m1_' + plan['id']) / 'parents' / task
        key = phase + ':' + task
        print(json.dumps({'event': 'task.start', 'phase': phase, 'task': task}), flush=True)
        progress[key] = {'status': 'running', 'started': time.time()}
        write(progress_path, progress)
        try:
            if phase == 'parents':
                if not parent_dir.exists():
                    run(copy.deepcopy(config), parent_dir, injected=None, request_limiter=limiter)
                limiter.check()
                pkg = L1Package(parent_dir)
                if pkg.json('raw/resolved_config.json') != config or (pkg.manifest.get('runtime') or {}).get('image_id') != config['runtime']['image']:
                    raise GateError('Existing parent actor/image differs from frozen plan')
                if not valid_parent(pkg):
                    raise GateError('Existing parent is incomplete/invalid; never automatically rerun')
                before = tree_identity(parent_dir)
                grade_gate = regrade(pkg, config, L1 / 'runs' / ('m1_' + plan['id']) / 'regrades' / task)
                if tree_identity(parent_dir) != before:
                    raise GateError('Regrading changed parent evidence')
                write(directory / task / 'parent_gate.json', {**grade_gate, 'parent_identity': before})
            else:
                pkg = L1Package(parent_dir)
                gate = read(directory / task / 'parent_gate.json')
                before = tree_identity(parent_dir)
                if not valid_parent(pkg) or not gate.get('passed') or gate.get('parent_identity') != before:
                    raise GateError('Analysis requires unchanged valid parent and three official regrades')
                task_dir = directory / task
                lock = threading.Lock()
                slots = set()
                bound = 20 if task == 'sokoban' else 28
                def resume(parent_pkg, k, branch_dir, *, request_limiter):
                    verify_plan(directory)
                    limiter.check()
                    # Curve trial names identify a slot independently of invalid retry suffix.
                    slot = re.sub(r'-r\d+$', '', branch_dir.name)
                    with lock:
                        slots.add(slot)
                        if len(slots) > bound:
                            raise GateError('Frozen unique branch bound exceeded')
                    if tree_identity(parent_dir) != before:
                        raise GateError('Parent changed before branch')
                    run(copy.deepcopy(config), branch_dir, injected=None, parent=parent_dir,
                        checkpoint_id=f'cp{k:06d}', request_limiter=limiter)
                    limiter.check()
                    verify_child(parent_pkg, branch_dir, k, config)
                    if tree_identity(parent_dir) != before:
                        raise GateError('Branch changed parent')
                if failure_selected(pkg):
                    curve_plan = build_curve_plan(pkg, n1=2, n2=2, stride=4, alpha=.05, max_concurrency=2)
                    if curve_plan['estimated_branches']['total_max'] > bound:
                        raise GateError('Curve exceeds frozen branch bound')
                    # Existing report defaults invent a per-branch dollar estimate.
                    # Override only this orchestration invocation's presentation field;
                    # execute_curve explicitly excludes it from actor/plan identity.
                    original_builder = report_module.build_curve_plan
                    def no_price_plan(*args, **kwargs):
                        result = original_builder(*args, **kwargs)
                        result['estimated_cost_usd'] = None
                        return result
                    report_module.build_curve_plan = no_price_plan
                    try:
                        report_module.run_episode(parent_dir, output_dir=task_dir, analysts=('rules',), execute=True,
                                    n1=2, n2=2, stride=4, alpha=.05, max_concurrency=2, rpm=30, tpm=10_000_000,
                                    p0=.5, baseline_provenance=BASELINE, resume_fn=resume)
                    finally:
                        report_module.build_curve_plan = original_builder
                    report_path = task_dir / 'report.md'
                    if report_path.exists():
                        text = re.sub(r'（预估成本：[^）]*）', '', report_path.read_text())
                        report_path.write_text('M1 engineering pilot; p0 is an uncalibrated placeholder.\n\n' + text)
                    limiter.check()
                    # Validate reused cached branches too; curve reuse never bypasses fidelity checks.
                    for child_path in (task_dir / 'branches').glob('*/manifest.json'):
                        child = L1Package(child_path.parent)
                        verify_child(pkg, child.root, child.start_turn, config)
                else:
                    facts = build_facts(pkg)
                    write(task_dir / 'facts.json', facts)
                    write(task_dir / 'values_rules.json', {'baseline_provenance': BASELINE,
                        **predict_values_rules(facts, pkg.turns, .5, float(pkg.outcome()['y']))})
                    eligible = []
                    for k in sorted(pkg.checkpoints()):
                        try:
                            pkg.validate_resume_checkpoint(k)
                            eligible.append(k)
                        except Exception:
                            continue
                    if not eligible:
                        raise GateError('No qualified checkpoint for runtime fidelity probe')
                    k = 1 if 1 in eligible else next((k for k in eligible if k > 0), 0)
                    probe = task_dir / 'probes' / f'cp{k:06d}'
                    if not probe.exists():
                        resume(pkg, k, probe, request_limiter=limiter)
                    verify_child(pkg, probe, k, config)
                    write(task_dir / 'probe.json', {'checkpoint': k, 'initial_checkpoint_fallback': k == 0,
                                                  'scope': 'runtime and first-request fidelity only; no causal diagnosis'})
                if tree_identity(parent_dir) != before:
                    raise GateError('Analysis changed parent evidence')
            progress[key].update(status='complete', finished=time.time())
            print(json.dumps({'event': 'task.end', 'phase': phase, 'task': task, 'status': 'complete'}), flush=True)
        except Exception as error:
            # Error text from providers may contain secrets; expose only the exception class.
            progress[key].update(status='blocked', error_type=type(error).__name__, finished=time.time())
            print(json.dumps({'event': 'task.end', 'phase': phase, 'task': task,
                              'status': 'blocked', 'error_type': type(error).__name__}), flush=True)
            raise
        finally:
            write(progress_path, progress)
            summarize(directory, plan)
    return summarize(directory, plan)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--id', required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--execute-model', action='store_true')
    parser.add_argument('--phase', choices=('parents', 'analysis'))
    parser.add_argument('--task', choices=TASKS)
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', args.id):
        parser.error('id must be a safe identifier')
    if args.prepare:
        if args.execute_model or args.phase or args.task:
            parser.error('prepare is separate from paid execution')
        print(str(prepare(args.id)))
        return 0
    if not args.execute_model or not args.phase:
        parser.error('execution requires --execute-model and --phase')
    directory = L2 / 'validation' / ('m1_' + args.id)
    execute(directory, args.phase, args.task)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({'status': 'blocked', 'error_type': type(exc).__name__}), file=sys.stderr)
        raise SystemExit(1)
