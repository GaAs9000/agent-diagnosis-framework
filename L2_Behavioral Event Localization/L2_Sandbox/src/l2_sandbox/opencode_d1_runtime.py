"""Separately authorized cross-actor D1 execution; never advances the D2 gate.

The Mac controller owns the frozen ARM64 Docker environment. Injected adapters
are an offline test seam and their receipts cannot be reused by native execution.
"""
from __future__ import annotations

import copy
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import math
from pathlib import Path

from .opencode_admission import Admission, transport_guard
from .v05 import build_interleaved_arm_schedule
from .v3_execution import (digest, exclusive, file_digest, identity, item_score,
                           read_json, seal, verify_branch, verify_seal, write_json)

PARAMETERS = dict(n_per_arm=14, delta=.10, tau=.20, alpha=.05,
                  ceiling_threshold=.05, estimand='conditional_frozen_dev_pool',
                  target_failure_units=24, passing_harm_sample_size=6)
MODEL = 'deepseek-v4.1-flash'
_L2 = Path(__file__).resolve().parents[2]
_ROOT = _L2.parent.parent
CRITICAL_SOURCES = (
    Path(__file__).resolve(),
    Path(__file__).with_name('opencode_admission.py'),
    Path(__file__).with_name('v3_execution.py'),
    Path(__file__).with_name('v05.py'),
    Path(__file__).with_name('cross_actor_observer.py'),
    _L2 / 'scripts/prospective_execution_observer.py',
    _L2 / 'scripts/build_opencode_d1_execution.py',
    _ROOT / 'L1_sandbox/C3_Experimental_Sandbox/src/c3_sandbox/l1/core/orchestrator.py',
)


def _reference(ref):
    if not isinstance(ref, dict) or not isinstance(ref.get('path'), str):
        raise ValueError('absolute bound reference required')
    path = Path(ref['path'])
    if not path.is_absolute() or file_digest(path) != ref.get('sha256'):
        raise ValueError('bound reference changed')
    return path


def _verify_plan(plan):
    verify_seal(plan)
    if plan.get('schema') != 'cross-actor-d1-execution-v1':
        raise ValueError('separate cross-actor execution plan required')
    authorization = plan.get('authorization', {})
    if (authorization.get('execution_authorized') is not True
            or authorization.get('scope') != 'cross_actor_D1'
            or authorization.get('exact_user_text') != '开始吧，直接用opencode'):
        raise ValueError('explicit cross_actor_D1 user authorization required')
    if plan.get('execution_concurrency') != 3:
        raise ValueError('cross-actor concurrency must be three')
    if plan.get('automatic_d2') is not False:
        raise ValueError('automatic D2 must remain disabled')
    model = plan.get('model', {})
    if (model.get('adapter') != 'opencode-chat' or model.get('provider') != 'opencode-go'
            or model.get('model') != MODEL or model.get('timeout_seconds') != 600
            or model.get('base_url') != 'https://opencode.ai/inference/go/v1'
            or model.get('parameters', {}).get('temperature') != 1
            or model.get('parameters', {}).get('max_tokens') != 8192
            or model.get('options', {}).get('expected_reported_model') != MODEL):
        raise ValueError('frozen Go model configuration differs')
    for key in ('source_files', 'bound_files'):
        inventory = plan.get(key)
        if not isinstance(inventory, dict) or not inventory:
            raise ValueError('nonempty absolute file inventory required')
        for name, sha in inventory.items():
            _reference(dict(path=name, sha256=sha))
    for path in CRITICAL_SOURCES:
        if plan['source_files'].get(str(path)) != file_digest(path):
            raise ValueError('critical executable source missing from source_files')
    raw = read_json(_reference(plan.get('source_scientific_plan')))
    scientific = plan['scientific_plan']
    verify_seal(scientific, 'plan_sha256')
    if raw != scientific:
        raise ValueError('scientific plan differs from original raw reference')
    frozen = read_json(_reference(plan.get('source_scientific_freeze')))
    verify_seal(frozen, 'freeze_sha256')
    if (frozen.get('schema') != 'd0-scientific-input-freeze-v2'
            or frozen.get('d0_frozen') is not True
            or frozen.get('plan_sha256') != scientific['plan_sha256']
            or frozen.get('execution_authorized') is not False):
        raise ValueError('scientific freeze binding differs')
    params = scientific.get('d0_candidate_parameters', {})
    if any(params.get(k) != v for k, v in PARAMETERS.items()):
        raise ValueError('original 24+6/n14 scientific configuration required')
    if not Path(plan.get('runs_dir', '')).is_absolute():
        raise ValueError('absolute runs directory required')
    if plan.get('limits', {}).get('max_invalid_attempts') != 1:
        raise ValueError('one invalid attempt hard stop required')
    return scientific


def _jobs(plan, output):
    scientific = _verify_plan(plan)
    failures = scientific.get('unit_schedules', [])
    controls = scientific.get('r12_passing_harm_check_plan', {}).get('sampled_passing_units', [])
    if len(failures) != 24 or len(controls) != 6:
        raise ValueError('exactly 24 failure units and six controls required')
    schedules = [(copy.deepcopy(u), False) for u in failures]
    for control in controls:
        unit = copy.deepcopy(control)
        unit['eligible_arms'] = unit['arms']
        unit['arm_messages'] = scientific['task_ladder_message_specs'][unit['task_id']]['ladder_messages']
        unit['schedule'] = build_interleaved_arm_schedule(
            unit['arms'], 14, anchor_k=unit['anchor_k'], remaining_budget=unit['remaining_budget'])
        schedules.append((unit, True))
    groups, seen = [], set()
    for unit, passing in schedules:
        if unit['unit_id'] in seen:
            raise ValueError('duplicate unit identity')
        seen.add(unit['unit_id'])
        expected = build_interleaved_arm_schedule(unit['eligible_arms'], 14,
            anchor_k=unit['anchor_k'], remaining_budget=unit['remaining_budget'])
        if unit['schedule'] != expected:
            raise ValueError('original interleaved arm schedule changed')
        group = []
        for branch in unit['schedule']:
            message = unit['arm_messages'][branch['arm']]
            text = message.get('message')
            message_sha = hashlib.sha256(text.encode()).hexdigest() if text else None
            if message.get('eligible') is not True or message.get('message_sha256') != message_sha:
                raise ValueError('ineligible or altered message')
            binding = dict(plan_sha256=plan['artifact_sha256'], unit_id=unit['unit_id'],
                arm=branch['arm'], replicate=branch['replicate'], anchor_k=unit['anchor_k'],
                remaining_budget=unit['remaining_budget'], message_sha256=message_sha,
                parent_identity=unit['parent_identity'],
                endpoint=dict(unit_level=unit.get('unit_level', 'rubric_item'),
                              defect_name=unit.get('defect_name', unit.get('rule', ''))),
                passing_control=passing, target_model_sha256=digest(plan['model']))
            binding['branch_id'] = digest(binding)
            group.append(dict(unit=unit, branch=branch, message=message, binding=binding,
                path=output / 'receipts' / (binding['branch_id'] + '.json'),
                package=output / 'branches' / binding['branch_id']))
        groups.append(group)
    jobs = [g[i] for i in range(max(map(len, groups))) for g in groups if i < len(g)]
    if len(jobs) != 2282 or len({j['binding']['branch_id'] for j in jobs}) != 2282:
        raise ValueError('original 2282 unique branches required')
    return jobs


def _native_branch(unit, branch, message, *, output, limiter, plan):
    from c3_sandbox.l1.core.config import validate
    from c3_sandbox.l1.core.orchestrator import build_execution_contract_prompt, run
    from c3_sandbox.l1.package import verify_package
    from .cross_actor_observer import CrossActorExecutionObserver

    parent = Path(plan['runs_dir']) / unit['episode_id']
    k = int(branch['checkpoint_k'])
    checkpoint = parent / 'checkpoints' / f'cp{k:06d}.json'
    before = identity(parent, k)
    if before != unit['parent_identity']:
        raise ValueError('frozen parent identity changed')
    original = read_json(parent / 'raw/resolved_config.json')
    config = copy.deepcopy(original)
    config['model'] = copy.deepcopy(plan['model'])
    used = int(read_json(checkpoint)['budget_used']['model_requests'])
    if config['budget']['max_model_requests'] - used != branch['remaining_budget']:
        raise ValueError('inherited request budget differs')
    if message.get('message'):
        config['agent']['intervention'] = dict(at_request=used + 1, message=message['message'])
    config = copy.deepcopy(validate(config))
    config['agent']['system_prompt'] = build_execution_contract_prompt(config)
    base = copy.deepcopy(plan['observer_base'])
    if config['runtime']['image'] != base['actor_image_id']:
        raise ValueError('native parent image must remain pinned')
    transition = dict(schema_version='cross-actor-model-transition-v1',
        source_config_sha256=digest(original), target_config_sha256=digest(config),
        source_model=original['model'], target_model=config['model'],
        parent_index_sha256=before['parent_index_sha256'],
        parent_checkpoint_sha256=before['checkpoint_sha256'])
    parent_ref = dict(path=str(parent / 'evidence_index.json'), sha256=before['parent_index_sha256'])
    # Package index identity is its raw file SHA in the native package verifier.
    checkpoint_ref = dict(path=str(checkpoint), sha256=before['checkpoint_sha256'])
    bound = dict(plan['bound_files'])
    bound.update(base.get('bound_files', {}))
    for ref in (parent_ref, checkpoint_ref, base['wrapper'], base['actor_image_inspect']):
        _reference(ref)
        bound[ref['path']] = ref['sha256']
    contract = seal(dict(base, source_files=plan['source_files'], bound_files=bound,
        parent_index=parent_ref, checkpoint=checkpoint_ref, expected_config_sha256=digest(config)))
    observer = CrossActorExecutionObserver(contract,
        external_dir=output.parent.parent / 'provenance' / output.name)
    result = run(config, output, parent=parent, checkpoint_id=f'cp{k:06d}',
        request_limiter=limiter, actor_transition=transition, execution_observer=observer)
    if identity(parent, k) != before:
        raise ValueError('parent evidence changed during branch')
    verified = verify_package(output)
    if not verified.get('verified') or not result.get('verification', {}).get('verified'):
        raise ValueError('branch package verification failed')
    score = item_score(output, unit)
    manifest = read_json(output / 'manifest.json')
    if (manifest.get('status') or {}).get('termination') == 'wall_time_budget':
        raise RuntimeError('branch wall budget exhausted')
    return dict(valid_attempt=True, status='completed', outcome_score=score,
        branch_episode_id=manifest['episode_id'], package_path=str(output),
        package_index_sha256=verified['index_sha256'],
        official_grade_sha256=file_digest(output / 'official_grade.json'),
        model_transition_sha256=file_digest(output / 'lineage/actor_transition.json'),
        requests_used=int(manifest['counts']['model_requests']), simulated_offline=False)


def _verify_receipt(receipt, job, injected):
    verify_branch(receipt, expected=job['binding'], verify_package_binding=not injected)
    if receipt.get('simulated_offline') is not injected:
        raise ValueError('injected/native receipt modes differ')
    score = receipt.get('outcome_score')
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('invalid endpoint score')
    if not injected:
        package = Path(receipt['package_path'])
        if package != job['package'] or file_digest(package / 'lineage/actor_transition.json') != receipt['model_transition_sha256']:
            raise ValueError('model transition receipt binding differs')
        config = read_json(package / 'raw/resolved_config.json')
        if digest(config['model']) != job['binding']['target_model_sha256']:
            raise ValueError('receipt model differs from frozen target')


def execute(plan: dict, output: Path, branch_adapter=None):
    """Execute a sealed batch, one serial canary followed by at most three jobs.

    A branch_adapter accepts (unit, branch, message, *, output, limiter, plan).
    Its offline receipts remain explicitly simulated and cannot certify execution.
    """
    output = Path(output).absolute()
    # All scientific and executable hashes precede even ledger construction.
    jobs = _jobs(plan, output)
    injected = branch_adapter is not None
    with exclusive(output / 'stage.lock'):
        completed, missing = {}, []
        for job in jobs:
            if job['path'].exists():
                receipt = read_json(job['path'])
                _verify_receipt(receipt, job, injected)
                completed[job['binding']['branch_id']] = receipt
            elif job['package'].exists() or (output / 'provenance' / job['binding']['branch_id']).exists():
                raise ValueError('uncertain existing branch; automatic retry forbidden')
            else:
                missing.append(job)
        ledger = Admission(output / 'admission.json', plan['artifact_sha256'],
                           caps=plan['limits'], expected_model=MODEL)
        failures = []

        def progress():
            state = ledger.snapshot()
            result = seal(dict(schema='cross-actor-d1-execution-result-v1',
                plan_sha256=plan['artifact_sha256'],
                status=('simulated_completed' if injected else 'completed') if len(completed) == len(jobs) else 'incomplete',
                planned_branches=len(jobs), completed_branches=len(completed),
                invalid_or_incomplete_branches=len(failures), pending_branches=len(jobs)-len(completed)-len(failures),
                requests=state['requests'], tokens=state['tokens'], cost_usd=state['cost_usd'],
                invalid_attempts=state['invalid_attempts'], stop_reason=state.get('stop_reason'),
                http_send_count=sum(bool(s.get('transport_started')) for s in state['sends']),
                d2_gate_open=False, scientific_scope='cross_actor_exploratory',
                simulated_offline=injected))
            write_json(output / 'result.json', result)
            return result

        def perform(job):
            # Receipt reservation survives crashes even before L1 creates a package.
            write_json(job['path'], seal(dict(job['binding'], status='inflight',
                                            valid_attempt=False, simulated_offline=injected)))
            try:
                ledger.check()
                result = (branch_adapter or _native_branch)(job['unit'], job['branch'], job['message'],
                    output=job['package'], limiter=ledger, plan=plan)
                receipt = seal({**result, **job['binding'], 'simulated_offline': injected})
                _verify_receipt(receipt, job, injected)
                write_json(job['path'], receipt)
                return receipt
            except Exception as exc:
                if not ledger.exhaustion_reason():
                    ledger.invalid()
                failed = seal(dict(job['binding'], status='incomplete', valid_attempt=False,
                    error_type=type(exc).__name__, simulated_offline=injected,
                    attempt_package_path=str(job['package'])))
                write_json(job['path'], failed)
                return failed

        def collect(job, receipt):
            if receipt['status'] == 'completed':
                completed[job['binding']['branch_id']] = receipt
            else:
                failures.append(receipt)
            progress()

        progress()
        # Native transport patch is absent for an explicitly injected offline adapter.
        from contextlib import nullcontext
        with nullcontext() if injected else transport_guard(ledger):
            pending = iter(missing)
            if missing and not ledger.exhaustion_reason():
                canary = next(pending)
                collect(canary, perform(canary))
            with ThreadPoolExecutor(max_workers=3) as pool:
                active = {}
                exhausted = False
                while active or not exhausted:
                    while not exhausted and len(active) < 3 and not ledger.exhaustion_reason():
                        job = next(pending, None)
                        if job is None:
                            exhausted = True
                            break
                        active[pool.submit(perform, job)] = job
                    if ledger.exhaustion_reason():
                        exhausted = True
                    if not active:
                        break
                    done, _ = wait(active, return_when=FIRST_COMPLETED)
                    for future in done:
                        job = active.pop(future)
                        collect(job, future.result())
        if len(completed) == len(jobs):
            ledger.close('completed')
        return progress()
