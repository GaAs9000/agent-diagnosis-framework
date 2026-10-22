#!/usr/bin/env python3
"""Acceptance-gap Exp1 block driver. Inert unless --execute; one sealed block per run.

Conditions (spec: configs/acceptance_gap_exp1_20261022.json):
  O  original task; hash-identical to the frozen DEV45 task configuration
  F  full grader specification: frozen D1 R_implicit requirement lists plus the
     deliverable filenames the verifiers look up
  T  self-written acceptance contract, checked before finishing (PSCI-style)
  V  render-and-inspect self-check before finishing (GDPval-style)

The condition text is inserted into agent.system_prompt between the Pi base
prompt and the sandbox execution contract. Pi compaction keeps messages[0:2]
(system prompt and task), so the text stays visible for the whole run. Nothing
else in the frozen DEV45 configuration changes.

Usage:
  --preview [--output P]                                       review material only
  --block N --authorization A --service-lease L --run-root R   sealed block plan
  --verify-plan P                                              re-derive and check a sealed plan
  --execute --plan P --output O --service-receipt S            run one block (via the supervisor)

The GPU service, its lease and container/service cleanup are owned by
scripts/launch_acceptance_gap_block.py and scripts/supervise_acceptance_gap_exp1.py.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re
import threading
import time
import traceback

SCRIPT = Path(__file__).resolve()
L2 = SCRIPT.parents[1]
ROOT = L2.parents[1]
L1 = ROOT / 'L1_sandbox/C3_Experimental_Sandbox'
BASE_DRIVER = SCRIPT.with_name('run_prospective_dev45.py')
OBSERVER = SCRIPT.with_name('prospective_execution_observer.py')
SUPERVISOR = SCRIPT.with_name('supervise_acceptance_gap_exp1.py')
LAUNCHER = SCRIPT.with_name('launch_acceptance_gap_block.py')
SPEC = L2 / 'configs/acceptance_gap_exp1_20261022.json'
PHASE_ROOT = L2 / 'validation/acceptance_gap_exp1_20261022'
RUN_PREFIX = 'acceptance_gap_exp1_20261022_b'
SCHEMA = 'acceptance-gap-exp1-block-1'
PREVIEW_SCHEMA = 'acceptance-gap-exp1-preview-1'
SPEC_SCHEMA = 'acceptance-gap-exp1-spec-v1'
EXPERIMENT_ID = 'acceptance_gap_exp1_20261022'
CONDITIONS = ('O', 'F', 'T', 'V')
CONTRACT_HEADER = '[SANDBOX EXECUTION CONTRACT & RUNTIME CONSTRAINTS]'
H_EXPLICIT = '请同时逐条核对以下任务明确要求：\n'
H_IMPLICIT = '\n\n请进一步逐条核对以下隐含格式与细则要求：\n'
GPU_UUID = 'GPU-6d3f754b-b139-9354-0cbc-42bf22208180'
RESERVATION_TOKENS = 1024 ** 3  # conservative per-send reservation unit, not observed usage


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The DEV45 driver validates the frozen pi and puts L1/L2 sources on sys.path.
base = _load(BASE_DRIVER, 'acceptance_gap_base_driver')
from c3_sandbox.l1.core.config import validate as l1_validate  # noqa: E402
from c3_sandbox.l1.core.orchestrator import build_execution_contract_prompt, run as l1_run  # noqa: E402
from l2_sandbox.v3_execution import Admission, digest, file_digest, read_json, seal, verify_seal, write_json  # noqa: E402

_OBSERVER_CACHE = {}


def observer_module():
    if 'module' not in _OBSERVER_CACHE:
        _OBSERVER_CACHE['module'] = _load(OBSERVER, 'acceptance_gap_observer_base')
    return _OBSERVER_CACHE['module']


# ---------------------------------------------------------------- specification

def load_spec(path=SPEC):
    path = base.regular(path)
    spec = read_json(path)
    tasks = spec.get('tasks') or {}
    primary, f_only = list(tasks.get('primary') or []), list(tasks.get('f_only') or [])
    blocks, limits, pi = spec.get('blocks') or {}, spec.get('limits_per_block') or {}, spec.get('pi') or {}
    slots = len(primary) * len(CONDITIONS) + len(f_only)
    try:
        base_config = base.regular(ROOT / pi.get('base_config', ''))
    except ValueError:
        base_config = None
    if (spec.get('schema') != SPEC_SCHEMA or spec.get('experiment_id') != EXPERIMENT_ID
            or tuple(spec.get('conditions') or ()) != CONDITIONS
            or sorted(primary + f_only) != sorted(base.TASKS) or len(set(primary + f_only)) != len(base.TASKS)
            or f_only != ['officeval_085']
            or blocks.get('count') != 5 or blocks.get('slots_per_block') != slots or not blocks.get('order_seed')
            or limits.get('concurrency') != 3 or limits.get('attempt_limit_per_slot') != 1
            or limits.get('max_model_requests_per_slot') != 35 or limits.get('max_tool_calls_per_slot') != 140
            or limits.get('logical_cap') != slots * 35 or limits.get('http_cap') != slots * 35 * 4
            or limits.get('lease_seconds') != base.WALL or limits.get('send_window_seconds') != base.WINDOW
            or limits.get('request_timeout_seconds') != 600
            or not isinstance(spec.get('condition_header'), str) or not spec['condition_header'].strip()
            or base_config is None or base_config != base.regular(base.CONFIG)
            or pi.get('base_config_sha256') != file_digest(base_config)):
        raise ValueError('unexpected or inconsistent Exp1 specification')
    return spec, file_digest(path)


def _message(value):
    return value.get('message') if isinstance(value, dict) else value


def f_texts(spec):
    """Full grader specification per task, from the frozen D1 R_implicit arm."""
    f = spec['conditions']['F']
    source = f['source']
    path = base.regular(ROOT / source['path'])
    if file_digest(path) != source['sha256']:
        raise ValueError('frozen F source (D1 plan) changed')
    ladder = read_json(path)['task_ladder_message_specs']
    if set(f['deliverable_names']) - set(base.TASKS):
        raise ValueError('unknown task in F deliverable names')
    out = {}
    for task in base.TASKS:
        messages = ladder[task]['ladder_messages']
        full, prefix = _message(messages['R_implicit']), _message(messages['S'])
        if not isinstance(full, str) or not isinstance(prefix, str) or not prefix or not full.startswith(prefix):
            raise ValueError('frozen R_implicit message is not nested under S: ' + task)
        tail = full[len(prefix):].lstrip('\n')
        if not tail.startswith(H_EXPLICIT) or tail.count(H_EXPLICIT) != 1 or tail.count(H_IMPLICIT) != 1:
            raise ValueError('unexpected R_implicit layout: ' + task)
        explicit, implicit = tail[len(H_EXPLICIT):].split(H_IMPLICIT)
        for note in f['annotation_removals']:
            explicit, implicit = explicit.replace(note, ''), implicit.replace(note, '')
        if '待人工' in explicit + implicit or not explicit.strip() or not implicit.strip():
            raise ValueError('F text keeps an internal review note or an empty list: ' + task)
        parts = [f['prefix'], f['explicit_heading'] + '\n' + explicit.strip(),
                 f['implicit_heading'] + '\n' + implicit.strip()]
        names = f['deliverable_names'].get(task)
        if names:
            verifier = base.regular(ROOT / names['verifier']['path'])
            if file_digest(verifier) != names['verifier']['sha256']:
                raise ValueError('verifier changed: ' + task)
            source_text = verifier.read_text(encoding='utf-8')
            if not names['names'] or any(n not in source_text or n not in names['text'] for n in names['names']):
                raise ValueError('deliverable names are not literally looked up by the verifier: ' + task)
            parts.append(names['text'])
        out[task] = '\n\n'.join(parts)
    return out


def condition_texts(spec):
    f = f_texts(spec)
    texts = {}
    for task in base.TASKS:
        row = {'O': None, 'F': f[task]}
        if task in spec['tasks']['primary']:
            row['T'] = spec['conditions']['T']['text']
            row['V'] = spec['conditions']['V']['text']
        texts[task] = row
    return texts


def condition_config(base_cfg, header, text):
    """Return a fixed point of L1 validation that differs from O only in agent.system_prompt."""
    cfg = copy.deepcopy(base_cfg)
    if text is not None:
        prompt = cfg['agent']['system_prompt']
        if prompt.count(CONTRACT_HEADER) != 1 or header in prompt or header in text:
            raise ValueError('unexpected frozen system prompt layout')
        head, _, tail = prompt.partition(CONTRACT_HEADER)
        cfg['agent']['system_prompt'] = head.rstrip() + '\n\n' + header + '\n' + text.strip()
        cfg['agent']['system_prompt'] = build_execution_contract_prompt(cfg)
        if not cfg['agent']['system_prompt'].endswith('\n\n' + CONTRACT_HEADER + tail):
            raise ValueError('condition text changed the sandbox execution contract')
    checked = l1_validate(copy.deepcopy(cfg))
    checked['agent']['system_prompt'] = build_execution_contract_prompt(checked)
    if checked != cfg:
        raise ValueError('condition configuration is not a fixed point of L1 validation')
    before, after = copy.deepcopy(base_cfg), copy.deepcopy(cfg)
    before['agent'].pop('system_prompt'), after['agent'].pop('system_prompt')
    if before != after:
        raise ValueError('condition changed more than agent.system_prompt')
    return cfg


def materialize(spec):
    base_configs = base.task_configs(base.CONFIG)
    texts = condition_texts(spec)
    configs = {task: {c: condition_config(base_configs[task], spec['condition_header'], text)
                      for c, text in row.items()} for task, row in texts.items()}
    return base_configs, texts, configs


def block_slots(spec, block):
    labels = [f'{task}.{c}.r{block:02d}' for task in spec['tasks']['primary'] for c in CONDITIONS]
    labels += [f'{task}.F.r{block:02d}' for task in spec['tasks']['f_only']]
    seed = int(hashlib.sha256(f"{spec['blocks']['order_seed']}:{block}".encode()).hexdigest(), 16)
    random.Random(seed).shuffle(labels)
    return labels


def parse_label(label):
    match = re.fullmatch(r'(officeval_\d{3})\.([OFTV])\.r(\d{2})', label)
    if not match:
        raise ValueError('malformed Exp1 slot label: ' + label)
    return match[1], match[2], int(match[3])


def review_record(spec, spec_sha, texts, configs):
    return dict(experiment_id=spec['experiment_id'], spec_sha256=spec_sha,
                base_config_sha256=spec['pi']['base_config_sha256'],
                condition_text_sha256={t: {c: (None if x is None else digest(x)) for c, x in row.items()}
                                       for t, row in texts.items()},
                config_sha256={t: {c: digest(cfg) for c, cfg in row.items()} for t, row in configs.items()},
                slots={str(b): block_slots(spec, b) for b in range(spec['blocks']['count'])},
                limits=spec['limits_per_block'])


def reference_hashes(spec):
    ref = spec['pi']['reference_prospective_freeze']
    path = base.regular(ROOT / ref['path'])
    if file_digest(path) != ref['sha256']:
        raise ValueError('reference DEV45 freeze changed')
    return read_json(path)['task_config_sha256']


def preview(spec_path=SPEC):
    spec, spec_sha = load_spec(spec_path)
    base_configs, texts, configs = materialize(spec)
    frozen = reference_hashes(spec)
    if any(digest(configs[t]['O']) != frozen[t] for t in base.TASKS):
        raise ValueError('O is not hash-identical to the frozen DEV45 task configuration')
    review = review_record(spec, spec_sha, texts, configs)
    return dict(schema=PREVIEW_SCHEMA, status='preview_not_executable', review=review,
                review_digest=digest(review), condition_texts=texts,
                system_prompts={t: {c: cfg['agent']['system_prompt'] for c, cfg in row.items()}
                                for t, row in configs.items()},
                character_counts={t: {c: len(x) for c, x in row.items() if x} for t, row in texts.items()},
                actor_requests=0, analyst_requests=0)


# ---------------------------------------------------------------- reference pi

def reference_freeze(spec):
    """DEV45 image/user/environment/native/service identity. Its controller-source lock
    is superseded by the Exp1 plan's own source fingerprints."""
    obs = observer_module()
    ref = spec['pi']['reference_prospective_freeze']
    path = base.regular(ROOT / ref['path'])
    if obs.sha(path) != ref['sha256']:
        raise ValueError('reference DEV45 freeze changed')
    frozen = obs.read(path)
    if (frozen.get('schema') != 'prospective-live-freeze-v1' or frozen.get('phase_authorized') is not True
            or frozen.get('internal_reviewed') is not True or frozen.get('provenance_candidate_verified') is not True
            or set(frozen.get('task_config_sha256', {})) != set(base.TASKS)
            or not re.fullmatch(r'sha256:[0-9a-f]{64}', frozen.get('actor_image_id', ''))
            or frozen.get('actor_user') != obs.host_user()):
        raise ValueError('reference DEV45 freeze identity/state differs')
    obs.env_dict(frozen.get('actor_image_env'))
    files = frozen.get('bound_files') or {}
    if not files or any(not Path(name).is_absolute() or obs.sha(name) != value for name, value in files.items()):
        raise ValueError('reference DEV45 evidence file changed')
    obs.validate_native_launch(frozen)
    image = obs.read(obs.bound_reference(frozen, frozen.get('actor_image_inspect')))
    if image.get('Id') != frozen['actor_image_id'] or image.get('Config', {}).get('Env') != frozen['actor_image_env']:
        raise ValueError('reference Actor image inspection differs')
    candidate_path = obs.bound_reference(frozen, frozen.get('candidate'))
    candidate = obs.read(candidate_path)
    verification_path = obs.bound_reference(frozen, frozen.get('provenance_verification'))
    verification = obs.read(verification_path)
    if (candidate.get('schema') != 'prospective-provenance-candidate-v1'
            or verification.get('preparation_links_validated') is not True
            or verification.get('candidate_sha256') != obs.sha(candidate_path)
            or verification.get('resolved_images', {}).get('actor') != frozen['actor_image_id']):
        raise ValueError('reference provenance candidate/verification differs')
    obs.validate_artifact_graph(frozen, candidate_path, candidate, verification_path)
    return frozen


def observer_class():
    module = observer_module()
    key = ('class', id(module))
    if key in _OBSERVER_CACHE:
        return _OBSERVER_CACHE[key]

    class Exp1Observer(module.ProspectiveExecutionObserver):
        """DEV45 container, native and isolation checks; the configuration is bound by the block plan."""

        def __init__(self, plan, label, external_dir):
            phase = (PHASE_ROOT / f"block_{plan['block']:02d}").absolute()
            external = Path(external_dir).absolute()
            if (external.exists() or external.is_symlink() or any(p.is_symlink() for p in external.parents)
                    or not external.is_relative_to(phase) or external == phase
                    or any((p / 'evidence_index.json').exists() or (p / 'trajectory.jsonl').exists()
                           for p in external.parents)):
                raise ValueError('external receipt directory must be new, unaliased and inside its Exp1 block phase')
            self.plan, self.label = plan, label
            self.freeze_path = base.regular(plan['reference_freeze_path'])
            self.expected_sha = plan['reference_freeze_sha256']
            self.external_dir = external
            self.frozen = reference_freeze(load_spec(Path(plan['spec_path']))[0])
            self.expected_config_sha256 = plan['config_sha256'][label]
            self.config = None
            self.workspace = None
            self.episode_id = None
            self.seen_containers = set()
            self.native_verified = False

        def guard(self):
            if module.sha(self.freeze_path) != self.expected_sha:
                raise ValueError('reference DEV45 freeze changed during execution')
            assert_sources(self.plan)

        def validate(self, config):
            self.guard()
            if self.config is not None or self.episode_id is not None:
                raise ValueError('observer is one episode only')
            task = config.get('benchmark', {}).get('task_id')
            if task != self.plan['slot_specs'][self.label]['task'] or digest(config) != self.expected_config_sha256:
                raise ValueError('final resolved task configuration differs from the sealed Exp1 block plan')
            runtime = config.get('runtime', {})
            if (runtime.get('adapter') != 'docker-cold' or runtime.get('memory') != '2g'
                    or runtime.get('cpus') != 2 or runtime.get('pids_limit') != 128
                    or runtime.get('read_only_rootfs', True) is not True
                    or runtime.get('tmpfs_exec', False) is not False
                    or runtime.get('options', {}).get('persist_app_repo', False) is not False
                    or runtime.get('user') not in (None, self.frozen['actor_user'])):
                raise ValueError('resolved runtime violates frozen isolation/resources')
            self.config = copy.deepcopy(config)

    _OBSERVER_CACHE[key] = Exp1Observer
    return Exp1Observer


# ---------------------------------------------------------------- plan

def source_files():
    paths = [SCRIPT, BASE_DRIVER, OBSERVER, SUPERVISOR, LAUNCHER, L2 / 'src/l2_sandbox/v3_execution.py',
             SCRIPT.with_name('supervise_r22_native_alignment.py'), SCRIPT.with_name('finalize_d0_timeout600.py'),
             SCRIPT.with_name('start_owned_native_service.py'), SCRIPT.with_name('build_prospective_provenance.py')]
    paths += [p for p in (L1 / 'src').rglob('*') if p.is_file() and p.suffix in ('.py', '.json', '.mjs', '.js')
              and '__pycache__' not in p.parts and 'node_modules' not in p.parts]
    return {str(base.regular(p).relative_to(ROOT)): base.bound_digest(p) for p in sorted(set(paths))}


def fixed_limits(spec):
    limits = spec['limits_per_block']
    return dict(concurrency=limits['concurrency'], attempt_limit_per_slot=1, http_cap=limits['http_cap'],
                logical_cap=limits['logical_cap'], request_cap_per_slot=35, tool_cap_per_slot=140,
                wall_seconds=base.WALL, request_timeout_seconds=600, send_window_seconds=base.WINDOW,
                max_invalid_attempts=spec['blocks']['slots_per_block'] + 1, parent=None, initial_budget=0,
                d1_execution=False, analyst_requests=0, service_renewal_authorized=False)


def check_authorization(path, spec, spec_sha, review_digest, block):
    auth = read_json(base.regular(path))
    if (not auth.get('exact_text') or not auth.get('source')
            or auth.get('experiment_id') != spec['experiment_id'] or auth.get('spec_sha256') != spec_sha
            or auth.get('approved_review_digest') != review_digest
            or block not in (auth.get('authorized_blocks') or [])):
        raise ValueError('direct user authorization for this Exp1 review digest and block is required')
    return auth


def fresh_paths(run_root, block, output=None):
    run_root = Path(run_root).absolute()
    if run_root.parent.resolve() != (L1 / 'runs').resolve() or run_root.name != f'{RUN_PREFIX}{block:02d}':
        raise ValueError('dedicated Exp1 block run root required')
    paths = [run_root]
    if output is not None:
        output = Path(output).absolute()
        phase = (PHASE_ROOT / f'block_{block:02d}').resolve()
        if not output.resolve().is_relative_to(phase) or output.resolve() == phase:
            raise ValueError('execution output must be inside its Exp1 block phase')
        paths.append(output)
    for path in paths:
        if path.exists() or path.is_symlink() or any(p.is_symlink() for p in path.parents):
            raise ValueError('restart/overwrite or aliased output forbidden: ' + str(path))
        for ancestor in path.parents:
            if (ancestor / 'evidence_index.json').exists() or (ancestor / 'attempt_started.json').exists():
                raise ValueError('cannot write inside an immutable prior package/attempt')


def assert_sources(plan):
    verify_seal(plan, 'plan_sha256')
    if plan['source_fingerprints'] != source_files():
        raise ValueError('exact Exp1 execution source changed')
    for name, expected in plan['bound_files'].items():
        if base.bound_digest(name) != expected:
            raise ValueError('bound configuration/provenance/authorization/lease changed: ' + name)


def build_block_plan(*, block, authorization, service_lease, run_root, spec_path=SPEC):
    spec, spec_sha = load_spec(spec_path)
    if type(block) is not int or not 0 <= block < spec['blocks']['count']:
        raise ValueError('block out of range')
    review_digest = preview(spec_path)['review_digest']
    base_configs, texts, configs = materialize(spec)
    auth = check_authorization(authorization, spec, spec_sha, review_digest, block)
    frozen = reference_freeze(spec)
    expected_service = base.validate_service_identity(frozen.get('expected_service_identity'),
                                                      base_configs, frozen['bound_files'])
    original = base.lease(service_lease, time.time())
    fresh_paths(run_root, block)
    labels = block_slots(spec, block)
    slot_specs, plan_configs, text_sha = {}, {}, {}
    for label in labels:
        task, condition, replicate = parse_label(label)
        slot_specs[label] = dict(task=task, condition=condition, replicate=replicate)
        plan_configs[label] = configs[task][condition]
        text = texts[task][condition]
        text_sha[label] = None if text is None else digest(text)
    reference_path = base.regular(ROOT / spec['pi']['reference_prospective_freeze']['path'])
    bound = dict(frozen['bound_files'])
    extra = [spec_path, ROOT / spec['conditions']['F']['source']['path'], authorization, service_lease,
             base.CONFIG, reference_path]
    extra += [ROOT / v['verifier']['path'] for v in spec['conditions']['F']['deliverable_names'].values()]
    for path in extra:
        bound[str(base.regular(path))] = file_digest(base.regular(path))
    plan = dict(schema_version=SCHEMA, experiment_id=spec['experiment_id'], block=block,
                spec_path=str(base.regular(spec_path)), spec_sha256=spec_sha, review_digest=review_digest,
                condition_header=spec['condition_header'], slots=labels, slot_specs=slot_specs,
                configs=plan_configs, config_sha256={k: digest(v) for k, v in plan_configs.items()},
                condition_text_sha256=text_sha, base_task_configs=base_configs, **fixed_limits(spec),
                source_fingerprints=source_files(), bound_files=bound,
                reference_freeze_path=str(reference_path),
                reference_freeze_sha256=spec['pi']['reference_prospective_freeze']['sha256'],
                actor_image_id=frozen['actor_image_id'], authorization_path=str(base.regular(authorization)),
                authorization_exact_text=auth['exact_text'], service_lease_path=str(base.regular(service_lease)),
                service_lease_origin=original, service_stop_epoch=original['service_stop_epoch'],
                run_root=str(Path(run_root).absolute()), expected_service_identity=copy.deepcopy(expected_service),
                scientific_efficacy='not_tested')
    plan = seal(plan, 'plan_sha256')
    assert_sources(plan)
    return plan


def validate_plan(plan):
    assert_sources(plan)
    spec, spec_sha = load_spec(Path(plan['spec_path']))
    block = plan.get('block')
    if (plan.get('schema_version') != SCHEMA or spec_sha != plan.get('spec_sha256')
            or plan.get('experiment_id') != spec['experiment_id']
            or type(block) is not int or not 0 <= block < spec['blocks']['count']):
        raise ValueError('Exp1 plan schema/spec/block binding differs')
    if any(plan.get(key) != value for key, value in fixed_limits(spec).items()):
        raise ValueError('Exp1 block workload/hard limits differ')
    base_configs, texts, configs = materialize(spec)
    review_digest = digest(review_record(spec, spec_sha, texts, configs))
    labels = block_slots(spec, block)
    if (plan.get('review_digest') != review_digest or plan.get('slots') != labels
            or set(plan.get('configs', {})) != set(labels) or set(plan.get('slot_specs', {})) != set(labels)
            or plan.get('base_task_configs') != base_configs
            or plan.get('condition_header') != spec['condition_header']):
        raise ValueError('sealed Exp1 block content differs from the reviewed specification')
    for label in labels:
        task, condition, replicate = parse_label(label)
        text = texts[task][condition]
        if (plan['slot_specs'][label] != dict(task=task, condition=condition, replicate=replicate)
                or replicate != block or plan['configs'][label] != configs[task][condition]
                or plan['config_sha256'].get(label) != digest(configs[task][condition])
                or plan['condition_text_sha256'].get(label) != (None if text is None else digest(text))):
            raise ValueError('Exp1 slot configuration differs: ' + label)
    check_authorization(plan['authorization_path'], spec, spec_sha, review_digest, block)
    frozen = reference_freeze(spec)
    if frozen['actor_image_id'] != plan['actor_image_id']:
        raise ValueError('Actor image differs')
    identity = base.validate_service_identity(frozen.get('expected_service_identity'),
                                              base_configs, frozen['bound_files'])
    if identity != plan.get('expected_service_identity'):
        raise ValueError('frozen expected service identity differs from the block plan')
    if base.lease(plan['service_lease_path'], time.time()) != plan['service_lease_origin']:
        raise ValueError('service lease differs')


def validate_service(plan, service_path):
    service = read_json(base.regular(service_path))
    identity = base.validate_service_identity(plan.get('expected_service_identity'),
                                              plan['base_task_configs'], plan['bound_files'])
    if (service.get('identity_receipt_sha256') != identity['identity_receipt']['sha256']
            or not isinstance(service.get('command'), list)
            or service['command'][4:] != identity['command_after_timeout']):
        raise ValueError('actual service weight/vLLM/parser/context command or identity receipt differs')
    original = plan['service_lease_origin']
    if (any(service.get(k) != v for k, v in original.items())
            or service.get('owner_tag') != plan['plan_sha256'] or service.get('server_plan_hash') != plan['plan_sha256']
            or service.get('remote_host') != 'aml-b' or not service.get('leader_start_ticks')
            or service.get('gpu_uuid') != GPU_UUID):
        raise ValueError('actual bounded owned aml-b GPU service receipt differs')
    return service


# ---------------------------------------------------------------- execution

class GuardedAdmission:
    """HTTP admission: lease/source problems stop the block; an episode whose own
    wall budget cannot cover a 600-second send ends as wall_time_budget only."""

    def __init__(self, admission, plan, metadata, stop):
        self.admission, self.plan, self.metadata, self.stop = admission, plan, metadata, stop

    def remaining_seconds(self):
        return max(0, min(self.admission.remaining_seconds(), self.plan['service_stop_epoch'] - time.time()))

    def acquire(self, estimate, timeout=None):
        if self.stop.is_set():
            return False
        try:
            assert_sources(self.plan)
            for path, expected in self.metadata.items():
                if file_digest(base.regular(path)) != expected:
                    raise ValueError('executed plan/service metadata changed')
            if self.remaining_seconds() < self.plan['send_window_seconds']:
                raise RuntimeError('insufficient lease time for a 600-second send plus safety window')
        except Exception:
            self.stop.set()
            raise
        if timeout is not None and timeout < self.plan['request_timeout_seconds']:
            return False
        try:
            admitted = self.admission.acquire(estimate, timeout)
        except Exception:
            self.stop.set()
            raise
        if not admitted:
            self.stop.set()
        return admitted

    def reconcile(self, estimate, actual):
        return self.admission.reconcile(estimate, actual)


def condition_record(path, plan, label):
    cfg = plan['configs'][label]
    body = read_json(Path(path) / 'raw/model/r000000/request.json')
    messages = body.get('messages') or []
    if (not messages or messages[0].get('role') != 'system'
            or messages[0].get('content') != cfg['agent']['system_prompt']):
        raise ValueError('first request system prompt differs from the sealed condition configuration')
    marked = plan['condition_header'] in messages[0]['content']
    if marked != (plan['slot_specs'][label]['condition'] != 'O'):
        raise ValueError('condition marker presence differs from the sealed condition')
    return dict(condition_text_sha256=plan['condition_text_sha256'][label],
                first_request_system_sha256=digest(messages[0]['content']))


def execute(plan, output, service_path):
    validate_plan(plan)
    validate_service(plan, service_path)
    output, run_root = Path(output), Path(plan['run_root'])
    fresh_paths(run_root, plan['block'], output)
    assert_sources(plan)  # last check before the first write
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'attempt_started.json', seal(dict(plan_sha256=plan['plan_sha256'], block=plan['block'],
                                                          slots=list(plan['slots']), attempt=1)))
    write_json(output / 'executed_plan.json', plan)
    run_root.mkdir(parents=True, exist_ok=False)
    caps = dict(max_requests=plan['http_cap'], max_tokens=plan['http_cap'] * RESERVATION_TOKENS, max_cost_usd=0,
                max_wall_seconds=max(1, plan['service_stop_epoch'] - time.time()),
                max_invalid_attempts=plan['max_invalid_attempts'], max_tokens_per_request=RESERVATION_TOKENS,
                usd_per_token_upper_bound=0)
    admission = Admission(output / 'admission.json', plan['plan_sha256'], caps)
    stop = threading.Event()
    metadata = {output / 'executed_plan.json': file_digest(output / 'executed_plan.json'),
                Path(service_path): file_digest(service_path)}
    observer_cls = observer_class()

    def slot(label):
        row_spec = plan['slot_specs'][label]
        path = run_root / 'episodes' / label
        row = dict(label=label, task_id=row_spec['task'], condition=row_spec['condition'],
                   replicate=row_spec['replicate'], path=str(path), status='NOT_RUN', study_valid=False, error=None)
        try:
            limiter = GuardedAdmission(admission, plan, metadata, stop)
            if stop.is_set() or limiter.remaining_seconds() < plan['send_window_seconds']:
                row['error'] = 'stopped before slot admission'
                return row
            assert_sources(plan)
            observer = observer_cls(plan, label, output / 'observer_receipts' / label)
            l1_run(copy.deepcopy(plan['configs'][label]), path, parent=None,
                   request_limiter=limiter, execution_observer=observer)
            record = base.slot_record(path, row_spec['task'], plan['configs'][label])
            record.update(condition_record(path, plan, label))
            row.update(record, status='completed')
            if not row['study_valid']:
                admission.invalid()
        except Exception as exc:
            row.update(status='execution_invalid', study_valid=False,
                       error=dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
            admission.invalid()
        finally:
            write_json(output / ('receipt_' + label + '.json'), seal(row))
        return row

    records = {}
    with ThreadPoolExecutor(max_workers=plan['concurrency']) as pool:
        futures = {pool.submit(slot, label): label for label in plan['slots']}
        for future in as_completed(futures):
            records[futures[future]] = future.result()
    integrity_error = None
    try:
        assert_sources(plan)
    except Exception as exc:
        integrity_error = dict(type=type(exc).__name__, message=str(exc))
    by_condition = {}
    for row in records.values():
        cell = by_condition.setdefault(row['condition'], dict(planned=0, completed=0, study_valid=0))
        cell['planned'] += 1
        cell['completed'] += row['status'] == 'completed'
        cell['study_valid'] += bool(row['study_valid'])
    ordered = [records[k] for k in plan['slots']]
    cohort = seal(dict(plan_sha256=plan['plan_sha256'], block=plan['block'], slots={k: records[k] for k in plan['slots']},
                       d1_execution=False, analyst_requests=0, integrity_error=integrity_error))
    write_json(output / 'cohort.json', cohort)
    score_inputs = seal(dict(plan_sha256=plan['plan_sha256'], block=plan['block'], records=ordered,
                             analysis='scripts/analyze_acceptance_gap_exp1.py (separate, zero model requests)',
                             integrity_error=integrity_error))
    write_json(output / 'score_inputs.json', score_inputs)
    complete = integrity_error is None and all(row['study_valid'] for row in ordered)
    result = seal(dict(plan_sha256=plan['plan_sha256'], block=plan['block'], cohort_sha256=cohort['artifact_sha256'],
                       all_study_valid=complete, slots_terminal=len(records),
                       study_valid_count=sum(bool(r['study_valid']) for r in ordered), by_condition=by_condition,
                       admissions=admission.snapshot(), analyst_requests=0, d1_execution=False,
                       status='completed' if complete else 'incomplete_or_invalid',
                       integrity_error=integrity_error, scientific_efficacy='not_tested'))
    write_json(output / 'result.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--spec', type=Path, default=SPEC)
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--block', type=int)
    parser.add_argument('--authorization', type=Path)
    parser.add_argument('--service-lease', type=Path)
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--plan-output', type=Path)
    parser.add_argument('--verify-plan', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--service-receipt', type=Path)
    args = parser.parse_args(argv)
    if args.execute:
        if not args.plan or not args.output or not args.service_receipt:
            parser.error('--execute requires --plan, --output and --service-receipt')
        value = execute(read_json(args.plan), args.output, args.service_receipt)
    elif args.verify_plan:
        validate_plan(read_json(args.verify_plan))
        value = dict(verified=True, plan_sha256=read_json(args.verify_plan)['plan_sha256'])
    elif args.preview:
        value = preview(args.spec)
        if args.output:
            if args.output.exists():
                parser.error('preview output exists; refusing to overwrite')
            write_json(args.output, value)
            value = dict(written=str(args.output), review_digest=value['review_digest'])
    else:
        if args.block is None or not all((args.authorization, args.service_lease, args.run_root)):
            parser.error('planning requires --block --authorization --service-lease --run-root')
        value = build_block_plan(block=args.block, authorization=args.authorization,
                                 service_lease=args.service_lease, run_root=args.run_root, spec_path=args.spec)
        if args.plan_output:
            if args.plan_output.exists():
                parser.error('plan output exists; refusing to overwrite')
            write_json(args.plan_output, value)
            value = dict(written=str(args.plan_output), plan_sha256=value['plan_sha256'])
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
