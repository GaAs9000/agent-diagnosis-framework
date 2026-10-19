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

from .opencode_admission import Admission, transport_guard, validate_budget_authorization
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
    _ROOT / 'L1_sandbox/C3_Experimental_Sandbox/src/c3_sandbox/l1/models/opencode.py',
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
    if 'branch_storage_root' in plan:
        _branch_storage_root(plan)
    if plan.get('limits', {}).get('max_invalid_attempts') != 1:
        raise ValueError('one fatal invalid attempt hard stop required')
    return scientific


def _jobs(plan, output):
    _verify_plan(plan)
    return _scheduled_jobs(plan, output)


def _branch_storage_root(plan):
    name = plan.get('branch_storage_root')
    if not isinstance(name, str) or not Path(name).is_absolute():
        raise ValueError('absolute branch storage root required')
    path = Path(name)
    if '..' in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('branch storage root cannot traverse symlinks or parent components')
    return path


def _scheduled_jobs(plan, output):
    """Reconstruct original bindings without executing obsolete source validators."""
    package_root = _branch_storage_root(plan) if 'branch_storage_root' in plan else output / 'branches'
    scientific = plan['scientific_plan']
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
                package=package_root / binding['branch_id']))
        groups.append(group)
    jobs = [g[i] for i in range(max(map(len, groups))) for g in groups if i < len(g)]
    if len(jobs) != 2282 or len({j['binding']['branch_id'] for j in jobs}) != 2282:
        raise ValueError('original 2282 unique branches required')
    return jobs


class BranchFailure(RuntimeError):
    """Known branch-local failure, after evidence and source verification."""

    def __init__(self, source, phase='execution', *, evidence=None):
        if source not in ('model_protocol', 'model_transport', 'scorer', 'branch_wall_time'):
            raise ValueError('unknown branch failure source')
        if phase not in ('execution', 'scoring'):
            raise ValueError('unknown branch failure phase')
        super().__init__(source)
        self.source = source
        self.phase = phase
        self.evidence = evidence or {}


def _verified_branch_score(output, unit, manifest, verified, *, global_stop_reason=None):
    """Classify sealed branch evidence; unknown validity errors remain fatal."""
    from c3_sandbox.l1.evaluation.study import study_outcome
    status = manifest.get('status') or {}
    if (manifest.get('recording_mode') != 'live'
            or status.get('evidence_completeness') != 'complete'
            or (manifest.get('runtime') or {}).get('adapter') != 'docker-cold'):
        raise ValueError('branch provenance or evidence completeness differs')
    evidence = dict(package_path=str(output), package_index_sha256=verified['index_sha256'],
                    branch_episode_id=manifest['episode_id'],
                    termination=status.get('termination'))
    termination = status.get('termination')
    if termination == 'wall_time_budget':
        if global_stop_reason:
            raise RuntimeError('global admission budget exhausted')
        raise BranchFailure('branch_wall_time', evidence=evidence)
    if termination == 'protocol_error' and status.get('protocol_validity') == 'invalid':
        raise BranchFailure('model_protocol', evidence=evidence)
    if termination == 'model_transport_error' and status.get('execution_validity') == 'invalid':
        raise BranchFailure('model_transport', evidence=evidence)
    # Preserve valid study zeros, including recognized missing/damaged deliveries.
    outcome = study_outcome(output, verify_evidence=True)
    if not outcome.get('study_valid'):
        if outcome.get('study_classification') in {
                'infrastructure_failure', 'unhealthy_grader_process', 'unhealthy_grader_worker'}:
            raise BranchFailure('scorer', 'scoring', evidence=evidence)
        raise ValueError('unrecognized invalid study evidence')
    return item_score(output, unit)


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
    _verify_plan(plan)
    manifest = read_json(output / 'manifest.json')
    score = _verified_branch_score(output, unit, manifest, verified,
                                   global_stop_reason=limiter.exhaustion_reason())
    return dict(valid_attempt=True, status='completed', outcome_score=score,
        branch_episode_id=manifest['episode_id'], package_path=str(output),
        package_index_sha256=verified['index_sha256'],
        official_grade_sha256=file_digest(output / 'official_grade.json'),
        model_transition_sha256=file_digest(output / 'lineage/actor_transition.json'),
        requests_used=int(manifest['counts']['model_requests']), simulated_offline=False)


def _verify_receipt(receipt, job, injected):
    if 'predecessor_seed' in job:
        # The seed was independently reconstructed from bound original evidence
        # during this invocation, including package and no-op wire checks.
        verify_seal(receipt)
        if receipt != job['predecessor_seed']:
            raise ValueError('carried predecessor receipt differs from verified evidence')
        return
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


def _bound_reference(plan, reference):
    path = _reference(reference)
    if plan['bound_files'].get(str(path)) != reference['sha256']:
        raise ValueError('predecessor reference missing from bound_files')
    return path


def _logical_binding(binding):
    return {k: v for k, v in binding.items()
            if k not in ('plan_sha256', 'branch_id', 'target_model_sha256')}


def _wire_compatibility(receipt, old_model, new_model):
    """Prove the successor projection leaves every previously sent body intact."""
    from c3_sandbox.l1.models.opencode import project_tool_history
    package = Path(receipt['package_path'])
    requests = sorted((package / 'raw/model').glob('r*/request.json'))
    if not requests or len(requests) != receipt['requests_used']:
        raise ValueError('complete predecessor request inventory required')
    proof = []
    for path in requests:
        body = read_json(path)
        wire, audit = project_tool_history(body['messages'])
        projected = {**body, 'messages': wire}
        if (wire != body['messages'] or projected != body or audit['changed']
                or audit['source_messages_sha256'] != audit['wire_messages_sha256']
                or body.get('model') != old_model['model']):
            raise ValueError('predecessor wire history would change under successor projection')
        entry = dict(request=dict(path=str(path), sha256=file_digest(path)),
                     wire_body_sha256=digest(body), projection=audit)
        if new_model.get('options', {}).get('truncated_tool_call_policy') is not None:
            response_path = path.with_name('response.bin')
            response = read_json(response_path)
            choices = response.get('choices')
            if (not isinstance(choices, list) or len(choices) != 1
                    or choices[0].get('finish_reason') not in ('stop', 'tool_calls', 'length')):
                raise ValueError('complete predecessor raw response required')
            finish = choices[0]['finish_reason']
            if (old_model.get('options', {}).get('truncated_tool_call_policy') is None
                    and finish == 'length'):
                raise ValueError('predecessor length response would activate successor recovery')
            entry.update(response=dict(path=str(response_path), sha256=file_digest(response_path)),
                         finish_reason=finish)
        proof.append(entry)
    return dict(schema='cross-actor-predecessor-wire-compatibility-v1',
                policy='group_tool_results_before_images_v1',
                source_model_sha256=digest(old_model), target_model_sha256=digest(new_model),
                all_bodies_unchanged=True, requests=proof)


def _predecessor_seeds(plan, jobs, injected, *, _chain=()):
    """Read a closed predecessor without editing or rerunning any old attempt."""
    prior = plan.get('predecessor')
    if prior is None:
        if plan.get("budget_authorization") is not None:
            raise ValueError("budget authorization requires a predecessor")
        return dict(requests=0, tokens=0, cost_usd=0.0, http_send_count=0,
                    invalid_attempts=0, completed=0, infrastructure_invalid=0,
                    branch_invalid=0, fatal_invalid=0)
    node = (str(Path(prior['plan']['path']).resolve()), prior['plan']['sha256'])
    if (len(_chain) >= 8 or any(node[0] == p or node[1] == sha for p, sha in _chain)
            or node[1] == digest(plan)):
        raise ValueError('predecessor chain cycle or depth limit exceeded')
    old_plan = read_json(_bound_reference(plan, prior['plan']))
    old_result = read_json(_bound_reference(plan, prior['result']))
    old_admission = read_json(_bound_reference(plan, prior['admission']))
    supervisor = read_json(_bound_reference(plan, prior['supervisor']))
    supervisor_result = read_json(_bound_reference(plan, prior['supervisor_result']))
    if type(supervisor_result.get('controller_exit_code')) is not int:
        raise ValueError('predecessor controller must have exited before successor admission')
    for artifact in (old_plan, old_result, old_admission):
        verify_seal(artifact)
    if (old_plan.get('schema') != 'cross-actor-d1-execution-v1'
            or old_plan.get('authorization', {}).get('scope') != 'cross_actor_D1'
            or old_plan.get('authorization', {}).get('execution_authorized') is not True
            or old_plan['scientific_plan'] != plan['scientific_plan']
            or old_plan.get('runs_dir') != plan['runs_dir']
            or old_plan.get('observer_base') != plan.get('observer_base')
            or old_result.get('plan_sha256') != old_plan['artifact_sha256']
            or old_admission.get('plan_sha256') != old_plan['artifact_sha256']
            or old_admission.get('limits') != old_plan['limits']
            or old_result.get('status') != 'incomplete'
            or old_result.get('d2_gate_open') is not False
            or not old_admission.get('stop_reason')
            or old_result.get('simulated_offline') is not injected):
        raise ValueError('closed predecessor plan/result/admission binding differs')
    expected_model = copy.deepcopy(old_plan['model'])
    policies = dict(history_projection_policy='group_tool_results_before_images_v1',
                    truncated_tool_call_policy='output_limit_error_feedback_v1')
    old_options = expected_model.get('options', {})
    new_options = plan['model'].get('options', {})
    for key, supported in policies.items():
        if key in old_options and old_options[key] != supported:
            raise ValueError('unsupported predecessor internal policy')
        if key in new_options:
            if new_options[key] != supported:
                raise ValueError('unsupported successor internal policy')
            expected_model.setdefault('options', {})[key] = supported
    if expected_model != plan['model']:
        raise ValueError('successor may only add or retain supported internal policies')
    for count in ('requests', 'tokens', 'cost_usd'):
        used = old_admission.get(count)
        if (type(used) not in (int, float) or not math.isfinite(used) or used < 0
                or old_result.get(count) != used):
            raise ValueError('successor limits must carry all predecessor usage')
    prior_deadline = supervisor.get('deadline_epoch')
    if (type(prior_deadline) not in (int, float) or not math.isfinite(prior_deadline)
            or (old_plan.get('deadline_epoch') is not None and old_plan['deadline_epoch'] != prior_deadline)
            or prior_deadline > supervisor['started_at_epoch'] + old_plan['limits']['max_wall_seconds']):
        raise ValueError('invalid predecessor absolute deadline')
    grant = None
    if plan.get('budget_authorization') is not None:
        grant = read_json(_bound_reference(plan, plan['budget_authorization']))
        validate_budget_authorization(grant, old_plan['artifact_sha256'])
        if (plan.get('deadline_epoch') != grant['deadline_epoch']
                or plan['limits']['max_wall_seconds'] != grant['max_wall_seconds']
                or plan['model'] != old_plan['model']):
            raise ValueError('authorized extension must preserve the actor and finite phase deadline')
    else:
        deadline = supervisor.get('deadline_epoch')
        if (type(deadline) not in (int, float) or not math.isfinite(deadline)
                or plan.get('deadline_epoch') != deadline
                or (old_plan.get('deadline_epoch') is not None and old_plan['deadline_epoch'] != deadline)
                or deadline > supervisor['started_at_epoch'] + old_plan['limits']['max_wall_seconds']):
            raise ValueError('successor must preserve predecessor absolute deadline')
        for cap, count in (('max_requests', 'requests'), ('max_tokens', 'tokens'), ('max_cost_usd', 'cost_usd')):
            used = old_admission.get(count)
            if (type(used) not in (int, float) or not math.isfinite(used) or used < 0
                    or old_result.get(count) != used
                    or plan['limits'][cap] + used > old_plan['limits'][cap] + 1e-12):
                raise ValueError('successor limits must carry all predecessor usage')
        if not 0 < plan['limits']['max_wall_seconds'] <= old_plan['limits']['max_wall_seconds'] <= 86400:
            raise ValueError('successor wall budget exceeds original cap')

    directory = Path(prior['receipt_directory'])
    if not directory.is_absolute() or directory.is_symlink() or not directory.is_dir():
        raise ValueError('absolute predecessor receipt directory required')
    old_jobs = _scheduled_jobs(old_plan, directory.parent)
    carried = _predecessor_seeds(old_plan, old_jobs, injected, _chain=(*_chain, node))
    old_by_id = {j['binding']['branch_id']: j for j in old_jobs}
    current_by_key = {digest(_logical_binding(j['binding'])): j for j in jobs}
    if len(current_by_key) != len(jobs) or len(old_by_id) != len(old_jobs):
        raise ValueError('duplicate logical branch identity')
    counts = dict(completed=0, infrastructure_invalid=0)
    paths = sorted(directory.glob('*.json'))
    retained_ids = {p.stem for p in paths}
    retained_directories = {directory.parent / 'branches', directory.parent / 'provenance'}
    if 'branch_storage_root' in old_plan:
        storage = _branch_storage_root(old_plan)
        retained_directories.update((storage, storage.parent / 'provenance'))
    for retained in retained_directories:
        if retained.exists() and any(p.name not in retained_ids for p in retained.iterdir()):
            raise ValueError('uncertain predecessor branch without a bound receipt')
    for path in paths:
        ref = dict(path=str(path), sha256=file_digest(path))
        _bound_reference(plan, ref)
        receipt = read_json(path)
        verify_seal(receipt)
        old_job = old_by_id.get(receipt.get('branch_id'))
        if old_job is None or old_job['path'] != path:
            raise ValueError('unknown predecessor branch receipt')
        if any(receipt.get(k) != v for k, v in old_job['binding'].items()):
            raise ValueError('predecessor scientific binding differs')
        current = current_by_key.get(digest(_logical_binding(old_job['binding'])))
        if current is None or 'predecessor_seed' in current:
            raise ValueError('missing or duplicate successor logical branch')
        if 'predecessor_seed' in old_job:
            _verify_receipt(receipt, old_job, injected)
        if receipt.get('status') == 'completed':
            _verify_receipt(receipt, old_job, injected)
            proof = _wire_compatibility(receipt, old_plan['model'], plan['model'])
            seed = seal({**receipt, **current['binding'], 'predecessor_receipt_ref': ref,
                         'predecessor_target_model_sha256': receipt['target_model_sha256'],
                         'wire_compatibility_proof': proof})
            counts['completed'] += 1
        elif (receipt.get('status') in ('incomplete', 'infrastructure_invalid', 'branch_invalid')
              and receipt.get('valid_attempt') is False):
            if receipt.get('simulated_offline') is not injected:
                raise ValueError('predecessor attempt execution mode differs')
            if 'outcome_score' in receipt:
                raise ValueError('invalid predecessor attempt cannot have an outcome score')
            seed = seal(dict(current['binding'], predecessor_receipt_ref=ref,
                predecessor_target_model_sha256=receipt['target_model_sha256'],
                status=('branch_invalid' if receipt['status'] == 'branch_invalid'
                        or receipt.get('failure_scope') == 'branch' else 'infrastructure_invalid'),
                valid_attempt=False,
                simulated_offline=injected,
                attempt_package_path=receipt.get('attempt_package_path'),
                error_type=receipt.get('error_type'), no_resampling=True,
                **({key: receipt[key] for key in (
                    'failure_scope', 'error_source', 'failure_phase', 'global_stop_reason',
                    'testable', 'no_sample_replacement', 'package_path', 'package_index_sha256',
                    'branch_episode_id', 'termination') if key in receipt}
                   if receipt['status'] == 'branch_invalid' or receipt.get('failure_scope') == 'branch'
                   else {})))
            counts['infrastructure_invalid'] += 1
        else:
            raise ValueError('uncertain predecessor attempt cannot be resumed')
        current['predecessor_seed'] = seed
    if (counts['completed'] != old_result['completed_branches']
            or counts['infrastructure_invalid'] != old_result['invalid_or_incomplete_branches']
            or len(paths) != counts['completed'] + counts['infrastructure_invalid']
            or old_result['planned_branches'] != len(old_jobs)
            or old_result['pending_branches'] != len(old_jobs) - len(paths)):
        raise ValueError('predecessor receipt census differs from closed result')
    if any('predecessor_seed' in j and j['binding']['branch_id'] not in retained_ids for j in old_jobs):
        raise ValueError('missing carried predecessor receipt')
    if (not isinstance(old_admission.get('sends'), list)
            or any(s.get('status') == 'inflight' or s.get('state') == 'inflight'
                   for s in old_admission['sends'])):
        raise ValueError('uncertain predecessor admission sends')
    http_count = sum(bool(s.get('transport_started')) for s in old_admission['sends'])
    if old_result.get('http_send_count') != http_count:
        raise ValueError('predecessor HTTP accounting differs')
    own = dict(requests=old_admission['requests'], tokens=old_admission['tokens'],
               cost_usd=old_admission['cost_usd'], http_send_count=http_count,
               invalid_attempts=old_admission['invalid_attempts'],
               branch_invalid=old_admission.get('branch_invalid', 0),
               fatal_invalid=old_admission.get('fatal_invalid', old_admission['invalid_attempts']))
    aggregate = {key: value + carried[key] for key, value in own.items()}
    for key, value in aggregate.items():
        field = 'aggregate_' + key
        required = (old_plan.get('predecessor') is not None
                    and (key not in ('branch_invalid', 'fatal_invalid') or key in old_admission))
        if ((required or field in old_result)
                and (type(old_result.get(field)) not in (int, float)
                     or not math.isfinite(old_result[field])
                     or not math.isclose(old_result[field], value, rel_tol=1e-12, abs_tol=1e-12))):
            raise ValueError('predecessor aggregate accounting differs')
    for key in ('branch_invalid', 'fatal_invalid'):
        if key in old_result and old_result[key] != own[key]:
            raise ValueError('predecessor invalid accounting differs')
    if grant is not None:
        for cap, count in (('max_requests', 'requests'), ('max_tokens', 'tokens'), ('max_cost_usd', 'cost_usd')):
            remaining = grant['cumulative_limits'][cap] - aggregate[count]
            value = plan['limits'][cap]
            if (type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                    or not math.isclose(value, remaining, rel_tol=0, abs_tol=1e-12)):
                raise ValueError('authorized extension must subtract all ancestor usage exactly once')
    return dict(aggregate, **counts)



def _admission_authorization(plan):
    """Called only after the entire predecessor chain has been verified."""
    while plan.get('predecessor') is not None:
        if plan.get('budget_authorization') is not None:
            return read_json(_bound_reference(plan, plan['budget_authorization']))
        plan = read_json(_bound_reference(plan, plan['predecessor']['plan']))
    return None


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
        # Validate the entire predecessor census and every reusable package
        # before writing a successor receipt or constructing its admission.
        carried = _predecessor_seeds(plan, jobs, injected)
        completed, missing, failures = {}, [], []
        for job in jobs:
            if job['path'].exists():
                receipt = read_json(job['path'])
                _verify_receipt(receipt, job, injected)
                if receipt['status'] == 'completed':
                    completed[job['binding']['branch_id']] = receipt
                else:
                    failures.append(receipt)
            elif 'predecessor_seed' in job:
                if job['package'].exists() or any((root / 'provenance' / job['binding']['branch_id']).exists()
                        for root in (output, job['package'].parent.parent)):
                    raise ValueError('carried branch has an uncertain successor attempt')
                receipt = job['predecessor_seed']
                write_json(job['path'], receipt)
                if receipt['status'] == 'completed':
                    completed[job['binding']['branch_id']] = receipt
                else:
                    failures.append(receipt)
            elif job['package'].exists() or any((root / 'provenance' / job['binding']['branch_id']).exists()
                        for root in (output, job['package'].parent.parent)):
                raise ValueError('uncertain existing branch; automatic retry forbidden')
            else:
                missing.append(job)
        ledger = Admission(output / 'admission.json', plan['artifact_sha256'],
                           caps=plan['limits'], expected_model=MODEL,
                           absolute_deadline=plan.get('deadline_epoch'),
                           budget_authorization=_admission_authorization(plan))

        process_finished = False

        def progress():
            state = ledger.snapshot()
            result = seal(dict(schema='cross-actor-d1-execution-result-v1',
                plan_sha256=plan['artifact_sha256'],
                status=('simulated_completed' if injected else 'completed') if len(completed) == len(jobs) else 'incomplete',
                planned_branches=len(jobs), completed_branches=len(completed),
                invalid_or_incomplete_branches=len(failures), pending_branches=len(jobs)-len(completed)-len(failures),
                requests=state['requests'], tokens=state['tokens'], cost_usd=state['cost_usd'],
                invalid_attempts=state['invalid_attempts'], stop_reason=ledger.exhaustion_reason(),
                http_send_count=sum(bool(s.get('transport_started')) for s in state['sends']),
                current_phase_completed_branches=len(completed)-carried['completed'],
                carried_completed_branches=carried['completed'],
                carried_infrastructure_invalid_branches=carried['infrastructure_invalid'],
                aggregate_requests=state['requests']+carried['requests'],
                aggregate_tokens=state['tokens']+carried['tokens'],
                aggregate_cost_usd=state['cost_usd']+carried['cost_usd'],
                aggregate_http_send_count=sum(bool(s.get('transport_started')) for s in state['sends'])+carried['http_send_count'],
                aggregate_invalid_attempts=state['invalid_attempts']+carried['invalid_attempts'],
                aggregate_invalid_branches=len(failures),
                branch_invalid=state.get('branch_invalid', 0),
                fatal_invalid=state.get('fatal_invalid', state['invalid_attempts']),
                current_phase_branch_invalid=state.get('branch_invalid', 0),
                current_phase_fatal_invalid=state.get('fatal_invalid', state['invalid_attempts']),
                aggregate_branch_invalid=state.get('branch_invalid', 0)+carried.get('branch_invalid', 0),
                aggregate_fatal_invalid=state.get('fatal_invalid', state['invalid_attempts'])+carried.get('fatal_invalid', carried['invalid_attempts']),
                execution_finished=process_finished, process_finished=process_finished,
                schedule_complete=len(completed)+len(failures) == len(jobs),
                d2_gate_open=False, scientific_scope='cross_actor_exploratory',
                simulated_offline=injected))
            write_json(output / 'result.json', result)
            return result

        def perform(job):
            # Receipt reservation survives crashes even before L1 creates a package.
            write_json(job['path'], seal(dict(job['binding'], status='inflight',
                                            valid_attempt=False, simulated_offline=injected)))
            phase = 'admission'
            try:
                ledger.check()
                phase = 'execution'
                result = (branch_adapter or _native_branch)(job['unit'], job['branch'], job['message'],
                    output=job['package'], limiter=ledger, plan=plan)
                receipt = seal({**result, **job['binding'], 'simulated_offline': injected})
                phase = 'verification'
                _verify_receipt(receipt, job, injected)
                write_json(job['path'], receipt)
                return receipt
            except Exception as exc:
                reason = ledger.exhaustion_reason()
                local = isinstance(exc, BranchFailure) and not reason
                scope = 'branch' if local else 'global'
                source = exc.source if local else ('global_admission' if reason else 'integrity_or_unclassified')
                failure_phase = exc.phase if local else phase
                ledger.invalid(failure_scope=scope, source=source, phase=failure_phase)
                failed = seal(dict(job['binding'], status='incomplete', valid_attempt=False,
                    testable=False, no_resampling=True, no_sample_replacement=True,
                    failure_scope=scope, error_source=source, failure_phase=failure_phase,
                    global_stop_reason=ledger.exhaustion_reason(),
                    error_type=type(exc).__name__, simulated_offline=injected,
                    attempt_package_path=str(job['package']),
                    **(exc.evidence if isinstance(exc, BranchFailure) else {})))
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
        process_finished = True
        return progress()
