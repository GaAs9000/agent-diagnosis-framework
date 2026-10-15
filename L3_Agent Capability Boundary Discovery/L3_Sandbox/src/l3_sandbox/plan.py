"""Freeze task/condition matrices before any paid execution."""
from __future__ import annotations

import copy
from importlib import metadata
import math
from pathlib import Path
import random
import re
import subprocess

from .common import L3Error, canonical, digest, read, safe_path, sha_file, write
from .conditions import TRANSFORM_VERSION, placebo_context, structure_context

VERSION = 'l3-plan-0.1'
SPEC_KEYS = {'schema_version', 'experiment_id', 'purpose', 'design', 'base_config',
             'tasks', 'conditions', 'replicates', 'shared_budget', 'limits', 'analysis',
             'schedule_seed', 'exclude_task_ids'}


def _keys(value, allowed, required, name):
    if not isinstance(value, dict) or set(value) - set(allowed) or set(required) - set(value):
        raise L3Error('Invalid or unknown fields in ' + name)


def _positive(value, name, integer=True):
    types = (int,) if integer else (int, float)
    if type(value) not in types or not math.isfinite(value) or value <= 0:
        raise L3Error(name + ' must be finite and positive')


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,100}', value):
        raise L3Error('Unsafe or empty experiment/task/condition identifier')


def validate_spec(spec):
    _keys(spec, SPEC_KEYS, SPEC_KEYS - {'exclude_task_ids'}, 'study spec')
    if spec['schema_version'] != VERSION or spec['purpose'] not in ('pilot', 'validation'):
        raise L3Error('Unknown study version or purpose')
    if spec['design'] not in ('budget', 'structure_budget'):
        raise L3Error('Unknown experimental design')
    _identifier(spec['experiment_id'])
    if type(spec['schedule_seed']) is not int:
        raise L3Error('schedule_seed must be integer')
    _positive(spec['replicates'], 'replicates')
    tasks, conditions = spec['tasks'], spec['conditions']
    if not isinstance(tasks, list) or not tasks or not isinstance(conditions, list) or not conditions:
        raise L3Error('Nonempty tasks and conditions required')
    seen = set()
    for t in tasks:
        fields = {'task_id', 'family_id', 'family_verified', 'split', 'file_type', 'selection_reason'}
        _keys(t, fields, fields, 'task')
        _identifier(t['task_id']); _identifier(t['family_id'])
        if t['task_id'] in seen or t['task_id'] in spec.get('exclude_task_ids', []):
            raise L3Error('Duplicate or excluded development task')
        seen.add(t['task_id'])
        if type(t['family_verified']) is not bool or t['split'] not in ('development', 'heldout'):
            raise L3Error('Explicit family verification and split required')
        if not isinstance(t['selection_reason'], str) or not t['selection_reason'].strip():
            raise L3Error('Task selection rationale required')
        if t['split'] == 'heldout' and not t['family_verified']:
            raise L3Error('Unverified task families cannot be heldout')
        if spec['purpose'] == 'validation' and t['split'] != 'heldout':
            raise L3Error('Validation tasks must be heldout families')
    families = {}
    for t in tasks:
        state = (t['split'], t['family_verified'])
        if families.setdefault(t['family_id'], state) != state:
            raise L3Error('Same family cannot span development and heldout or verification states')
    seen = set(); axes = set()
    for c in conditions:
        _keys(c, {'condition_id', 'max_model_requests', 'context'},
              {'condition_id', 'max_model_requests', 'context'}, 'condition')
        _identifier(c['condition_id']); _positive(c['max_model_requests'], 'max_model_requests')
        axis = (c['max_model_requests'], c['context'])
        if c['condition_id'] in seen or axis in axes or c['context'] not in ('original', 'structure', 'placebo'):
            raise L3Error('Duplicate or unknown condition')
        seen.add(c['condition_id']); axes.add(axis)
    if spec['design'] == 'budget' and any(c['context'] != 'original' for c in conditions):
        raise L3Error('Budget design cannot mix input transforms')
    if spec['design'] == 'structure_budget':
        budgets = {c['max_model_requests'] for c in conditions}
        if len(budgets) != 2 or axes != {(b, k) for b in budgets for k in ('original', 'structure', 'placebo')}:
            raise L3Error('Structure design requires three contexts crossed with two budgets')
    budget = spec['shared_budget']
    _keys(budget, {'max_tool_calls', 'wall_time_seconds', 'max_total_tokens'},
          {'max_tool_calls', 'wall_time_seconds', 'max_total_tokens'}, 'shared_budget')
    _positive(budget['max_tool_calls'], 'max_tool_calls')
    _positive(budget['wall_time_seconds'], 'wall_time_seconds', False)
    if budget['max_total_tokens'] is not None:
        _positive(budget['max_total_tokens'], 'max_total_tokens')
    limits = spec['limits']
    fields = {'max_concurrency', 'rpm', 'tpm', 'max_http_attempts', 'max_replacements_per_slot'}
    _keys(limits, fields, fields, 'limits')
    for name in fields - {'max_replacements_per_slot'}:
        _positive(limits[name], name, name not in ('rpm', 'tpm'))
    n = limits['max_replacements_per_slot']
    if type(n) is not int or not 0 <= n <= 2:
        raise L3Error('Replacement cap must be integer 0..2')
    analysis = spec['analysis']
    fields = {'primary_comparison', 'minimum_effect', 'bootstrap_samples', 'seed'}
    _keys(analysis, fields, fields, 'analysis')
    pair = analysis['primary_comparison']
    if not isinstance(pair, list) or len(pair) != 2 or len(set(pair)) != 2 or not set(pair) <= seen:
        raise L3Error('Primary comparison must name treatment and control conditions')
    if type(analysis['minimum_effect']) not in (int, float) or not 0 < analysis['minimum_effect'] <= 1:
        raise L3Error('minimum_effect must be in (0,1]')
    _positive(analysis['bootstrap_samples'], 'bootstrap_samples')
    if type(analysis['seed']) is not int:
        raise L3Error('Bootstrap seed must be integer')
    return spec


def image_id(image):
    try:
        result = subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}}', image],
                                check=True, text=True, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        raise L3Error('A local frozen Docker image is required: ' + image) from exc
    value = result.stdout.strip()
    if not re.fullmatch(r'sha256:[a-f0-9]{64}', value):
        raise L3Error('Invalid Docker image identity')
    return value


def source_inventory():
    import c3_sandbox.l1
    roots = [Path(__file__).resolve().parent, Path(c3_sandbox.l1.__file__).resolve().parent]
    return {str(p): sha_file(p) for root in roots for p in sorted(root.rglob('*.py'))}


def load_native_task(config, task_id):
    from c3_sandbox.l1.benchmarks.officeval.adapter import NativeOfficeVal
    return NativeOfficeVal(config['benchmark']).load(task_id)


def create_plan(spec_path: Path, directory: Path) -> dict:
    from c3_sandbox.l1.core.config import resolve, validate
    from c3_sandbox.l1.core.orchestrator import build_execution_contract_prompt
    spec_path, directory = safe_path(spec_path), safe_path(directory)
    if directory.exists():
        raise L3Error('Plan destination exists; use another directory')
    spec = validate_spec(read(spec_path))
    template = Path(spec['base_config'])
    template = (spec_path.parent / template).resolve() if not template.is_absolute() else template
    base = resolve(template)
    if base['benchmark']['adapter'] != 'officeval-native' or base['plugins']['load_installed']:
        raise L3Error('First L3 implementation requires the locked native OfficeVal template')
    if base['model']['parameters'].get('temperature') != 1.0 or any(
        k in base['model']['parameters'] for k in ('seed', 'random_seed', 'randomSeed')):
        raise L3Error('L3 requires frozen T=1 native sampling without a model seed')
    images = {base[s]['image']: image_id(base[s]['image']) for s in ('runtime', 'evaluation')}
    inputs = {str(spec_path): sha_file(spec_path), str(template): sha_file(template)}
    lock = base['benchmark'].get('data_lock')
    if lock:
        inputs[str(Path(lock))] = sha_file(Path(lock))
    jobs = []; task_records = []
    structure_design = spec['design'] == 'structure_budget'
    for task_spec in spec['tasks']:
        task = load_native_task(base, task_spec['task_id'])
        for path in [*(a.path for a in task.assets), *task.private_files.values()]:
            path = Path(path).resolve()
            inputs[str(path)] = sha_file(path)
        structure = structure_context(task) if structure_design else ''
        task_records.append({**task_spec, 'public_prompt_sha256': digest(task.prompt),
            'public_assets': [{'name': a.name, 'sha256': a.sha256, 'size_bytes': a.size_bytes} for a in task.assets],
            'native_provenance': task.provenance})
        for replicate in range(spec['replicates']):
            block = []
            for condition in spec['conditions']:
                cfg = copy.deepcopy(base)
                cfg['benchmark']['task_id'] = task_spec['task_id']
                cfg['budget'].update(spec['shared_budget'])
                cfg['budget']['max_model_requests'] = condition['max_model_requests']
                for section in ('runtime', 'evaluation'):
                    cfg[section]['image'] = images[cfg[section]['image']]
                appendix = ''
                if structure_design:
                    context = condition['context']
                    appendix = structure if context == 'structure' else (
                        placebo_context(structure) if context == 'placebo' else '')
                    cfg['benchmark']['adapter'] = 'l3-officeval'
                    cfg['benchmark']['options'] = {'l3_context': context,
                        'l3_appendix': appendix, 'l3_transform_version': TRANSFORM_VERSION,
                        'l3_condition_id': condition['condition_id']}
                    cfg['plugins']['load_installed'] = True
                cfg['agent']['system_prompt'] = build_execution_contract_prompt(cfg)
                validate(cfg)
                trial_id = f"{task_spec['task_id']}.{condition['condition_id']}.r{replicate:02d}"
                block.append({**task_spec, **condition, 'trial_id': trial_id, 'replicate': replicate,
                    'config': cfg, 'config_sha256': digest(cfg), 'appendix_sha256': digest(appendix),
                    'appendix_characters': len(appendix), 'appendix_utf8_bytes': len(appendix.encode())})
            random.Random(f"{spec['schedule_seed']}:{task_spec['task_id']}:{replicate}").shuffle(block)
            jobs.extend(block)
    # Randomize whole task/replicate blocks as well as conditions within blocks.
    width = len(spec['conditions'])
    blocks = [jobs[i:i + width] for i in range(0, len(jobs), width)]
    random.Random(spec['schedule_seed']).shuffle(blocks)
    jobs = [job for block in blocks for job in block]
    logical = sum(j['config']['budget']['max_model_requests'] for j in jobs)
    attempts = sum(j['config']['budget']['max_model_requests'] *
                   j['config']['model'].get('retry', {}).get('max_attempts', 1) for j in jobs)
    plan = {'schema_version': VERSION, 'experiment_id': spec['experiment_id'],
        'purpose': spec['purpose'], 'design': spec['design'], 'spec_sha256': digest(spec),
        'replicates': spec['replicates'], 'tasks': task_records, 'conditions': spec['conditions'],
        'analysis': spec['analysis'], 'limits': spec['limits'], 'schedule_seed': spec['schedule_seed'],
        'jobs': jobs, 'source_sha256': source_inventory(), 'inputs_sha256': inputs,
        'images': images, 'transform_version': TRANSFORM_VERSION,
        'workload': {'trial_slots': len(jobs), 'logical_request_bound': logical,
            'http_attempt_bound_without_replacements': attempts,
            'http_attempt_bound_with_replacements': attempts * (1 + spec['limits']['max_replacements_per_slot']),
            'hard_http_admission_cap': spec['limits']['max_http_attempts'],
            'estimated_actual_cost': None,
            'cost_note': 'No calibrated per-request price or live usage; monetary cost unknown'},
        'interpretation': 'Exploratory pilot only' if spec['purpose'] == 'pilot' else 'Frozen heldout validation',
        'execution_status': 'unexecuted'}
    directory.mkdir(parents=True, mode=0o700)
    write(directory / 'plan.json', plan)
    (directory / 'plan.sha256').write_text(sha_file(directory / 'plan.json') + '\n')
    return plan


def verify_plan(directory: Path, *, verify_images: bool = True) -> dict:
    directory = safe_path(directory)
    plan = read(directory / 'plan.json')
    if sha_file(directory / 'plan.json') != (directory / 'plan.sha256').read_text().strip():
        raise L3Error('Frozen plan hash mismatch')
    if plan.get('schema_version') != VERSION:
        raise L3Error('Unknown plan version')
    if source_inventory() != plan['source_sha256']:
        raise L3Error('Source inventory changed; freeze a new plan')
    for name, expected in plan['inputs_sha256'].items():
        if sha_file(Path(name)) != expected:
            raise L3Error('Frozen input changed: ' + name)
    ids = set()
    for job in plan['jobs']:
        _identifier(job['trial_id'])
        if job['trial_id'] in ids or digest(job['config']) != job['config_sha256']:
            raise L3Error('Duplicate trial or changed config')
        ids.add(job['trial_id'])
    if verify_images:
        for expected in set(plan['images'].values()):
            if image_id(expected) != expected:
                raise L3Error('Frozen image unavailable')
        if plan['design'] == 'structure_budget':
            entries = metadata.entry_points(group='l1.benchmarks')
            if not any(e.name == 'l3-officeval' and e.value == 'l3_sandbox.conditions:ConditionOfficeVal'
                       for e in entries):
                raise L3Error('Install L3 package to register its public-input benchmark adapter')
    return plan
