#!/usr/bin/env python3
"""Plan-only by default; a fresh DEV45 pool, never G2 or D1.

External observer contract (scripts/prospective_execution_observer.py):
validate_freeze(path, expected_sha256) -> verified freeze dict, and
ProspectiveExecutionObserver(path, expected_sha256, external_dir), implementing
the per-run L1 execution observer protocol. No observer module is synthesized.
The service owner must launch/clean up its own bounded GPU lease separately.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import threading
import time
import traceback

SCRIPT = Path(__file__).resolve()
L2 = SCRIPT.parents[1]
ROOT = L2.parents[1]
L1 = ROOT / 'L1_sandbox/C3_Experimental_Sandbox'
sys.path[:0] = [str(L1 / 'src'), str(L2 / 'src')]
from c3_sandbox.l1.core.config import resolve
from c3_sandbox.l1.core.orchestrator import build_execution_contract_prompt, run as l1_run
from c3_sandbox.l1.evaluation.study import study_outcome
from c3_sandbox.l1.evidence.live import verify
from l2_sandbox.v3_execution import Admission, digest, file_digest, read_json, seal, verify_seal, write_json

OBSERVER = SCRIPT.with_name('prospective_execution_observer.py')
CONFIG = L1 / 'configs/l1/officeval.qwen36.pi.t1.prospective-native.context-v2.timeout600.json'
TASKS = tuple('officeval_' + n for n in ('003', '004', '010', '033', '050', '060', '070', '085', '090'))
SLOTS = tuple(f'{task}.r{rep:02d}' for rep in range(5) for task in TASKS)
SCHEMA = 'prospective-dev45-1'
HTTP_CAP, LOGICAL_CAP, WALL, WINDOW = 6300, 1575, 21600, 645
_BOUND_HASHES = {}
_BOUND_LOCK = threading.RLock()


def regular(path):
    path = Path(path)
    if path.is_symlink() or any(p.is_symlink() for p in path.parents) or not path.is_file():
        raise ValueError('regular unaliased bound file required: ' + str(path))
    return path.resolve()


def bound_digest(path):
    """Hash large immutable proof bytes once, with per-use filesystem guards."""
    path = regular(path)
    def identity():
        stat = regular(path).stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    with _BOUND_LOCK:
        before = identity()
        cached = _BOUND_HASHES.get(str(path))
        if cached is not None and cached[0] == before:
            return cached[1]
        with path.open('rb') as stream:
            value = hashlib.file_digest(stream, 'sha256').hexdigest()
        if identity() != before:
            raise ValueError('bound proof changed while hashing')
        _BOUND_HASHES[str(path)] = (before, value)
        return value


def observer_module():
    spec = importlib.util.spec_from_file_location('prospective_dev45_observer', regular(OBSERVER))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_files():
    paths = [SCRIPT, OBSERVER, L2 / 'src/l2_sandbox/v3_execution.py',
             SCRIPT.with_name('supervise_prospective_dev45.py'),
             SCRIPT.with_name('supervise_r22_native_alignment.py'),
             SCRIPT.with_name('finalize_d0_timeout600.py'), SCRIPT.with_name('start_owned_native_service.py')]
    paths += [p for p in (L1 / 'src').rglob('*') if p.is_file()
              and p.suffix in ('.py', '.json', '.mjs', '.js') and '__pycache__' not in p.parts and 'node_modules' not in p.parts]
    return {str(regular(p).relative_to(ROOT)): file_digest(p) for p in sorted(paths)}


def task_configs(config_path):
    base = resolve(regular(config_path))
    model, agent, budget = base['model'], base['agent'], base['budget']
    if (model.get('kind') != 'local' or model.get('provider') != 'vllm-local'
            or model.get('model') != 'Qwen/Qwen3.6-27B' or model.get('base_url') != 'http://127.0.0.1:8001/v1'
            or model.get('timeout_seconds') != 600 or model.get('retry', {}).get('max_attempts') != 4
            or model.get('parameters', {}).get('temperature') != 1
            or model.get('parameters', {}).get('max_tokens') != 8192 or 'seed' in model.get('parameters', {})
            or model.get('parameters', {}).get('chat_template_kwargs', {}).get('enable_thinking') is not False
            or agent.get('harness') != 'pi' or agent.get('version') != '0.73.1'
            or agent.get('validation_policy') != 'pi-native-0.73.1'
            or agent.get('harness_revision') != 'l1-pi-native-errors-context-v2'
            or not agent.get('compaction', {}).get('enabled')
            or budget.get('max_model_requests') != 35 or budget.get('max_tool_calls') != 140
            or budget.get('wall_time_seconds') != 10800 or budget.get('max_total_tokens') is not None
            or base['runtime'].get('adapter') != 'docker-cold' or base['plugins'].get('load_installed') is not False):
        raise ValueError('chosen full Actor configuration violates frozen fresh DEV45 contract')
    values = {}
    for task in TASKS:
        cfg = copy.deepcopy(base)
        cfg['benchmark']['task_id'] = task
        cfg['agent']['system_prompt'] = build_execution_contract_prompt(cfg)
        values[task] = cfg
    return values


def lease(path, now):
    value = read_json(regular(path))
    keys = ('started_epoch', 'hard_stop_epoch', 'service_stop_epoch')
    if any(type(value.get(k)) not in (int, float) or not math.isfinite(value[k]) for k in keys):
        raise ValueError('finite service lease required')
    started, hard, stop = (value[k] for k in keys)
    if hard-started != WALL or not started <= now < stop or not WINDOW <= stop-now <= WALL or stop > hard-600:
        raise ValueError('original six-hour service lease required; no renewal')
    return {k: value[k] for k in keys}


def service_command(identity):
    """Exact frozen vLLM argv after the separately validated timeout prefix."""
    return ['nice', '-n', '10', identity['vllm']['executable'], 'serve', identity['model']['resolved_weight_path'],
        '--download-dir', '/home/jiasheren2/hf_cache', '--served-model-name', 'Qwen/Qwen3.6-27B', 'Qwen3.6-27B',
        '--port', '8001', '--host', '127.0.0.1', '--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_xml',
        '--reasoning-parser', 'qwen3', '--max-model-len', '131072', '--max-num-seqs', '64',
        '--gpu-memory-utilization', '0.85', '--dtype', 'bfloat16']


def validate_service_identity(identity, configs, bound_files):
    if not isinstance(identity, dict):
        raise ValueError('independently captured frozen service identity required')
    model, vllm = identity.get('model') or {}, identity.get('vllm') or {}
    cfg = configs[TASKS[0]]['model']
    def sha256_value(value):
        match = re.fullmatch(r'(?:sha256:)?([a-f0-9]{64})', value) if isinstance(value, str) else None
        if match is None:
            raise ValueError('strict SHA-256 service/config digest required')
        return match[1]
    if (model.get('reported_revision') != cfg.get('reported_revision') or not model.get('reported_revision')
            or any(sha256_value(model.get(key)) != sha256_value(cfg.get(key))
                   for key in ('weight_digest', 'tokenizer_digest'))):
        raise ValueError('frozen service model revision/weight/tokenizer identity differs')
    root = '/home/jiasheren2/hf_cache/models--Qwen--Qwen3.6-27B/snapshots/' + cfg['reported_revision']
    manifest = model.get('file_manifest') or {}
    if (model.get('resolved_weight_path') != root or not model.get('digest_algorithm')
            or type(model.get('validated_files')) is not int or model['validated_files'] < 1
            or not Path(manifest.get('path', '')).is_absolute()
            or not re.fullmatch('[a-f0-9]{64}', manifest.get('sha256', ''))):
        raise ValueError('actual resolved weight path/full file-manifest preparation proof required')
    if (vllm.get('executable') != '/home/jiasheren2/envs/idea-n1/bin/vllm' or not vllm.get('version')
            or not re.fullmatch('[a-f0-9]{64}', vllm.get('script_sha256', ''))
            or identity.get('command_after_timeout') != service_command(identity)):
        raise ValueError('frozen vLLM executable/version/script or exact parser/context launch arguments differ')
    reference = identity.get('identity_receipt') or {}
    path = reference.get('path')
    if (not path or bound_files.get(path) != reference.get('sha256')
            or bound_digest(path) != reference.get('sha256')):
        raise ValueError('independent server service identity receipt is not bound')
    actual = read_json(Path(path))
    if any(actual.get(key) != identity[key] for key in ('model', 'vllm', 'command_after_timeout')):
        raise ValueError('independent server model/vLLM/launch identity differs from freeze')
    return identity


def validated_prospective(path, expected, configs):
    path = regular(path)
    if file_digest(path) != expected:
        raise ValueError('prospective freeze bytes changed')
    value = read_json(path)
    if (value.get('phase_authorized') is not True or value.get('d0_frozen') is not False
            or value.get('provenance_candidate_verified') is not True
            or not value.get('actor_image_id', '').startswith('sha256:')
            or value.get('task_config_sha256') != {task: digest(cfg) for task, cfg in configs.items()}):
        raise ValueError('prospective provenance/full task configuration binding missing')
    mini = value.get('mechanism_mini') or {}
    bound = value.get('bound_files') or {}
    if mini.get('reviewed') is not True or mini.get('result_sha256') not in bound.values() or not bound:
        raise ValueError('reviewed actual mechanism mini result must be bound')
    for name, expected_hash in bound.items():
        if not Path(name).is_absolute() or bound_digest(name) != expected_hash:
            raise ValueError('prospective evidence file differs: ' + name)
    for key in ('provenance_verification', 'mini_review'):
        record = value.get(key) or {}
        if (not record.get('path') or bound.get(record['path']) != record.get('sha256')
                or not read_json(regular(record['path']))):
            raise ValueError('actual preparation verification and explicit mini policy review must be bound')
    # The separate verifier checks provenance and mini policy against artifacts;
    # booleans above are necessary declarations, never sufficient verification.
    checked = observer_module().validate_freeze(path, expected)
    if checked != value:
        raise ValueError('independent prospective freeze validation differs')
    sources = source_files()
    if any(value.get('source_files', {}).get(name) != sha for name, sha in sources.items()):
        raise ValueError('prospective freeze does not bind exact driver/L1/observer source')
    return value


def fresh_paths(run_root, output=None):
    run_root = Path(run_root).absolute()
    if (run_root.parent.resolve() != (L1 / 'runs').resolve()
            or not run_root.name.startswith('v3_qwen36_pi_prospective_dev45_')):
        raise ValueError('dedicated fresh prospective run root required; old pools forbidden')
    paths = [run_root]
    if output is not None:
        output = Path(output).absolute()
        allowed = (L2 / 'validation/a_plus_i_execute_20261018').resolve()
        if not output.resolve().is_relative_to(allowed) or output.resolve() == allowed:
            raise ValueError('fresh execution output must be inside new a_plus_i phase')
        paths.append(output)
    for path in paths:
        if path.exists() or path.is_symlink() or any(p.is_symlink() for p in path.parents):
            raise ValueError('restart/overwrite or aliased output forbidden: ' + str(path))
        for ancestor in path.parents:
            if (ancestor / 'evidence_index.json').exists() or (ancestor / 'attempt_started.json').exists():
                raise ValueError('cannot write inside an immutable prior package/attempt')


def build_plan(*, prospective_freeze, authorization, service_lease, run_root, config_path=CONFIG):
    configs = task_configs(config_path)
    frozen = validated_prospective(prospective_freeze, file_digest(regular(prospective_freeze)), configs)
    expected_service = validate_service_identity(frozen.get('expected_service_identity'), configs, frozen['bound_files'])
    auth = read_json(regular(authorization))
    if not auth.get('exact_text') or not auth.get('source'):
        raise ValueError('direct user authorization record required')
    original = lease(service_lease, time.time())
    fresh_paths(run_root)
    bound = dict(frozen['bound_files'])
    for path in (prospective_freeze, authorization, service_lease, config_path):
        bound[str(regular(path))] = file_digest(path)
    plan = dict(schema_version=SCHEMA, slots=list(SLOTS), concurrency=3, attempt_limit_per_slot=1,
        http_cap=HTTP_CAP, logical_cap=LOGICAL_CAP, tool_cap_per_slot=140, wall_seconds=WALL,
        request_timeout_seconds=600, send_window_seconds=WINDOW, parent=None, initial_budget=0,
        g2_actor_requests=0, d1_execution=False, d0_frozen=False, service_renewal_authorized=False,
        config_path=str(regular(config_path)), configs=configs,
        source_fingerprints={**frozen['source_files'], **source_files()}, bound_files=bound,
        prospective_freeze_path=str(regular(prospective_freeze)), prospective_freeze_sha256=file_digest(prospective_freeze),
        authorization_path=str(regular(authorization)), service_lease_path=str(regular(service_lease)),
        service_lease_origin=original, service_stop_epoch=original['service_stop_epoch'],
        actor_image_id=frozen['actor_image_id'], run_root=str(Path(run_root).resolve()), scientific_efficacy='not_tested')
    plan['expected_service_identity'] = copy.deepcopy(expected_service)
    plan = seal(plan, 'plan_sha256')
    assert_sources(plan)
    return plan


def assert_sources(plan):
    verify_seal(plan, 'plan_sha256')
    required = source_files()
    if any(plan['source_fingerprints'].get(name) != sha for name, sha in required.items()):
        raise ValueError('exact execution source changed')
    for name, expected in plan['source_fingerprints'].items():
        if name in required:
            continue
        path = ROOT / name
        if (Path(name).is_absolute() or '..' in Path(name).parts
                or file_digest(regular(path)) != expected):
            raise ValueError('additional frozen execution source changed')
    for name, expected in plan['bound_files'].items():
        if bound_digest(name) != expected:
            raise ValueError('bound configuration/provenance/authorization/lease changed')


def validate_plan(plan):
    assert_sources(plan)
    fixed = dict(schema_version=SCHEMA, slots=list(SLOTS), concurrency=3, attempt_limit_per_slot=1,
        http_cap=HTTP_CAP, logical_cap=LOGICAL_CAP, tool_cap_per_slot=140, wall_seconds=WALL,
        request_timeout_seconds=600, send_window_seconds=WINDOW, parent=None, initial_budget=0,
        g2_actor_requests=0, d1_execution=False, d0_frozen=False, service_renewal_authorized=False)
    if any(plan.get(k) != v for k, v in fixed.items()):
        raise ValueError('fresh DEV45 workload/hard limits differ')
    configs = task_configs(Path(plan['config_path']))
    if configs != plan['configs']:
        raise ValueError('chosen full task configurations changed')
    frozen = validated_prospective(plan['prospective_freeze_path'], plan['prospective_freeze_sha256'], configs)
    if frozen['actor_image_id'] != plan['actor_image_id']:
        raise ValueError('Actor image differs')
    identity = validate_service_identity(frozen.get('expected_service_identity'), configs, frozen['bound_files'])
    if identity != plan.get('expected_service_identity'):
        raise ValueError('frozen expected service identity differs from execution plan')
    if lease(plan['service_lease_path'], time.time()) != plan['service_lease_origin']:
        raise ValueError('service lease differs')


def validate_service(plan, service_path):
    service = read_json(regular(service_path))
    identity = validate_service_identity(plan.get('expected_service_identity'), plan['configs'], plan['bound_files'])
    if (service.get('identity_receipt_sha256') != identity['identity_receipt']['sha256']
            or not isinstance(service.get('command'), list) or service['command'][4:] != identity['command_after_timeout']):
        raise ValueError('actual service weight/vLLM/parser/context command or identity receipt differs')
    original = plan['service_lease_origin']
    if (any(service.get(k) != v for k, v in original.items())
            or service.get('owner_tag') != plan['plan_sha256'] or service.get('server_plan_hash') != plan['plan_sha256']
            or service.get('remote_host') != 'aml-b' or not service.get('leader_start_ticks')
            or service.get('gpu_uuid') != 'GPU-6d3f754b-b139-9354-0cbc-42bf22208180'):
        raise ValueError('actual bounded owned aml-b GPU service receipt differs')
    return service


class GuardedAdmission:
    def __init__(self, admission, plan, metadata, stop):
        self.admission, self.plan, self.metadata, self.stop = admission, plan, metadata, stop

    def remaining_seconds(self):
        return max(0, min(self.admission.remaining_seconds(), self.plan['service_stop_epoch']-time.time()))

    def acquire(self, estimate, timeout=None):
        if self.stop.is_set():
            return False
        try:
            assert_sources(self.plan)
            for path, expected in self.metadata.items():
                if file_digest(regular(path)) != expected:
                    raise ValueError('executed plan/service metadata changed')
            # L1 clamps timeout to its local wall budget; reserve the extra
            # safety margin against the absolute outer lease separately.
            if self.remaining_seconds() < WINDOW or (timeout is not None and timeout < 600):
                raise RuntimeError('insufficient 600-second send plus safety window')
            admitted = self.admission.acquire(estimate, timeout)
            if not admitted:
                self.stop.set()
            return admitted
        except Exception:
            self.stop.set()
            raise

    def reconcile(self, estimate, actual):
        return self.admission.reconcile(estimate, actual)


def slot_record(path, task, config):
    report = verify(path)
    if not report.get('verified'):
        raise ValueError('new package seal failed: ' + str(report))
    manifest = read_json(path / 'manifest.json')
    cp = read_json(path / 'checkpoints/cp000000.json')
    if (manifest.get('lineage') or cp.get('lineage') or manifest.get('config_sha256') != digest(config)
            or read_json(path / 'raw/resolved_config.json') != config
            or any(cp['budget_used'].get(k) != 0 for k in ('model_requests', 'tool_calls', 'known_total_tokens'))
            or cp.get('pending_calls') or cp.get('pending_observations') or cp['conversation'] is None):
        raise ValueError('fresh zero-budget root/config contract violated')
    if manifest['usage']['model_requests_including_parent'] > 35 or manifest['usage']['tool_calls_including_parent'] > 140:
        raise ValueError('per-slot request/tool budget exceeded')
    outcome = study_outcome(path, verify_evidence=True)
    return dict(path=str(path), task_id=task, episode_id=manifest['episode_id'],
        index_file_sha256=file_digest(path / 'evidence_index.json'), manifest_sha256=file_digest(path / 'manifest.json'),
        config_sha256=digest(config), study_valid=outcome.get('study_valid') is True, outcome=outcome,
        usage=manifest['usage'], e1_status='NOT_RUN')


def execute(plan, output, service_path):
    validate_plan(plan)
    validate_service(plan, service_path)
    output, run_root = Path(output), Path(plan['run_root'])
    fresh_paths(run_root, output)
    assert_sources(plan)  # Last check before the first write.
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'attempt_started.json', seal(dict(plan_sha256=plan['plan_sha256'], slots=list(SLOTS), attempt=1)))
    write_json(output / 'executed_plan.json', plan)
    run_root.mkdir(parents=True, exist_ok=False)
    reservation = 1024**3  # Conservative reservation unit, not observed tokens.
    caps = dict(max_requests=HTTP_CAP, max_tokens=HTTP_CAP*reservation, max_cost_usd=0,
        max_wall_seconds=max(1, plan['service_stop_epoch']-time.time()), max_invalid_attempts=46,
        max_tokens_per_request=reservation, usd_per_token_upper_bound=0)
    admission = Admission(output / 'admission.json', plan['plan_sha256'], caps)
    stop = threading.Event()
    metadata = {output / 'executed_plan.json':file_digest(output / 'executed_plan.json'),
                Path(service_path):file_digest(service_path)}
    observer_class = observer_module().ProspectiveExecutionObserver

    def slot(label):
        task = label.split('.')[0]
        path = run_root / 'episodes' / label
        row = dict(label=label, path=str(path), status='NOT_RUN', study_valid=False, error=None)
        try:
            limiter = GuardedAdmission(admission, plan, metadata, stop)
            if stop.is_set() or limiter.remaining_seconds() < WINDOW:
                row['error'] = 'stopped before slot admission'
                return row
            assert_sources(plan)
            observer = observer_class(Path(plan['prospective_freeze_path']), plan['prospective_freeze_sha256'],
                                      output / 'observer_receipts' / label)
            l1_run(copy.deepcopy(plan['configs'][task]), path, parent=None,
                   request_limiter=limiter, execution_observer=observer)
            row.update(slot_record(path, task, plan['configs'][task]), status='completed')
            if not row['study_valid']:
                admission.invalid()
        except Exception as exc:
            row.update(status='execution_invalid', error=dict(type=type(exc).__name__, message=str(exc),
                traceback=traceback.format_exc()))
            admission.invalid()
        finally:
            write_json(output / ('receipt_' + label + '.json'), seal(row))
        return row

    records = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(slot, label):label for label in SLOTS}
        for future in as_completed(futures):
            records[futures[future]] = future.result()
    integrity_error = None
    try:
        assert_sources(plan)
    except Exception as exc:
        integrity_error = dict(type=type(exc).__name__, message=str(exc))
    cohort = seal(dict(plan_sha256=plan['plan_sha256'], slots={k:records[k] for k in SLOTS},
                       d0_frozen=False, d1_execution=False, integrity_error=integrity_error))
    write_json(output / 'cohort.json', cohort)
    score_inputs = seal(dict(plan_sha256=plan['plan_sha256'], records=[records[k] for k in SLOTS],
        e1_status='NOT_RUN', g2_actor_requests=0, d1_execution=False, integrity_error=integrity_error))
    write_json(output / 'score_inputs.json', score_inputs)
    result = seal(dict(plan_sha256=plan['plan_sha256'], cohort_sha256=cohort['artifact_sha256'],
        all_45_study_valid=integrity_error is None and all(row['study_valid'] for row in records.values()), slots_terminal=len(records),
        admissions=admission.snapshot(), g2_actor_requests=0, d1_execution=False, d0_frozen=False,
        status='completed' if integrity_error is None and all(row['study_valid'] for row in records.values()) else 'incomplete_or_invalid',
        integrity_error=integrity_error, scientific_efficacy='not_tested'))
    write_json(output / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prospective-freeze', type=Path)
    parser.add_argument('--authorization', type=Path)
    parser.add_argument('--service-lease', type=Path)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--plan', type=Path, help='Read an already reviewed plan for execution')
    parser.add_argument('--output', type=Path, help='Fresh execution output; planning prints JSON only')
    parser.add_argument('--service-receipt', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        if not args.plan or not args.output or not args.service_receipt:
            parser.error('--execute requires --plan, --output and --service-receipt')
        value = execute(read_json(args.plan), args.output, args.service_receipt)
    else:
        if not all((args.prospective_freeze, args.authorization, args.service_lease, args.run_root)):
            parser.error('planning requires --prospective-freeze --authorization --service-lease --run-root')
        value = build_plan(prospective_freeze=args.prospective_freeze, authorization=args.authorization,
                           service_lease=args.service_lease, run_root=args.run_root, config_path=args.config)
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
