#!/usr/bin/env python3
"""Plan or execute the authorized R22 continuation and original twenty G2 slots.

Old evidence is read-only. Interrupted stages never restart automatically. The
cohort is a derived attempt-chain index; a suffix's cp0 is never an origin cp0.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
import math
import os
from pathlib import Path
import threading
import time

SCRIPT = Path(__file__).resolve()
spec = importlib.util.spec_from_file_location('r22_prior_helpers', SCRIPT.with_name('resume_d0_serial.py'))
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
p, c = s.prior, s.c
read_json, write_json, seal, verify_seal, file_digest = s.read_json, s.write_json, s.seal, s.verify_seal, s.file_digest
L1, L2, ROOT = p.L1, p.L2, p.ROOT
BASE = L2 / 'validation/d0_20261018'
OUTPUT = BASE / 'r22_native_alignment'
CONFIG = L1 / 'configs/l1/officeval.qwen36.pi.t1.native-validation.context-v2.timeout600.json'
RUN_ROOT = L1 / 'runs/v3_qwen36_pi_dev45_r22_20261018'
G2_ONLY_RUN_ROOT = L1 / 'runs/v3_qwen36_pi_dev45_r23_g2_20261018'
POLICY, REVISION = 'pi-native-0.73.1', 'l1-pi-native-errors-context-v2'
SELECTED = {
    'officeval_050.r04': ('runs/v3_qwen36_pi_dev45_t2_20261018/episodes/officeval_050.r04', 'ep_513d44f1f4a248dcadc8c7ba36901748', 9, 25),
    'officeval_004.r02': ('runs/v3_qwen36_pi_dev45_20261017/failed_attempts/officeval_004.r02.d0_resume_1792257570352387000', 'ep_94dde6631ebe4f038867e2a618fd0344', 7, 27),
    'officeval_010.r00': ('runs/v3_qwen36_pi_dev45_20261017/failed_attempts/officeval_010.r00.d0_resume_1792295803745985000', 'ep_3a4351548ebe45308d6b76e24b033f13', 23, 11),
}
DEMOTED = ('officeval_004.r02', 'officeval_010.r00')
EXCLUDED_SENSITIVITY = ('officeval_010.r04', 'officeval_050.r04', 'officeval_010.r00')
SLOTS = tuple(f'{task}.{rep}' for task in c.DEV_TASKS for rep in c.REPLICATES)
G2_SLOTS = [f'{task}.{rep}' for task, rep in c.G2_SLOTS]
WINDOW, WALL, OLD_HTTP = 645, 21600, 144
HTTP_CAPS = {'continuations': 380, 'g2': 2800}
LOGICAL_CAPS = {'continuations': 95, 'g2': 700}
PLAN_SCHEMA = 'r22-native-alignment-2'
G2_READINESS_POLICY = 'exact-frozen20-formal-endpoints-v1'
G2_ONLY_OLD_HTTP = 181
G2_ONLY_HTTP_CAPS = {'continuations': 0, 'g2': 2800}
G2_ONLY_LOGICAL_CAPS = {'continuations': 0, 'g2': 700}


def native_config(old):
    value = copy.deepcopy(old)
    if old['model']['timeout_seconds'] not in (300, 600):
        raise ValueError('unexpected parent timeout')
    value['model']['timeout_seconds'] = 600
    value['agent'].update(validation_policy=POLICY, harness_revision=REVISION)
    return value


def assert_delta(old, new):
    if native_config(old) != new:
        raise ValueError('configuration exceeds explicit timeout/native-policy delta')
    legacy = copy.deepcopy(old)
    legacy['model']['timeout_seconds'] = 300
    legacy['agent'].pop('validation_policy', None)
    legacy['agent'].pop('harness_revision', None)
    p.assert_config(legacy, legacy)


def dependency_identity():
    from c3_sandbox.l1.runtime.pi_native_validation import NativePiValidator
    with NativePiValidator() as validator:
        return validator.dependency_identity()


def source_files(config_path):
    paths = [SCRIPT, L2 / 'tests/test_r22_native_alignment.py', config_path, L1 / 'pyproject.toml', L2 / 'pyproject.toml',
             ROOT / 'scripts/run_v3_stage_d_dev45_and_g2.py', ROOT / 'scripts/v3_evaluate_gates_g2_g3_g4.py']
    for directory in (L1 / 'src', L2 / 'src'):
        paths.extend(f for f in directory.rglob('*') if f.is_file() and f.suffix in ('.py', '.json', '.mjs', '.js') and 'node_modules' not in f.parts and '__pycache__' not in f.parts)
    paths.extend(L2 / 'scripts' / name for name in ('resume_d0_serial.py', 'resume_d0_prerequisites.py',
        'audit_r22_native_validation.py', 'audit_g4_provenance.py', 'start_owned_native_service.py',
        'supervise_r22_native_alignment.py', 'finalize_d0_timeout600.py'))
    return {str(path.relative_to(ROOT)): file_digest(path) for path in sorted(set(paths))}


def assert_sources(plan):
    if source_files(Path(plan['config_path'])) != plan['source_fingerprints']:
        raise ValueError('frozen R22 execution source changed')
    for name, expected in plan['bound_files'].items():
        if file_digest(Path(name)) != expected:
            raise ValueError('frozen audit/authorization/predecessor changed')
    if dependency_identity() != plan['native_dependency_identity']:
        raise ValueError('native installed dependency identity changed')


def assert_originals(plan):
    for path, expected in plan['immutable_trees'].items():
        if s.tree_digest(Path(path)) != expected:
            raise ValueError('immutable original evidence changed: ' + path)


def assert_write_paths(plan, *paths):
    roots = [Path(path).resolve() for path in plan['immutable_trees']]
    if plan.get('predecessor_phase'):
        roots.append(Path(plan['predecessor_phase']).resolve())
    for path in paths:
        target = Path(path).resolve()
        if any(root in (target, *target.parents) for root in roots):
            raise ValueError('write path must be outside immutable predecessor phase and evidence roots: ' + str(path))


def checkpoint(path, k):
    from l2_sandbox.v3_execution import identity
    value = identity(path, k)  # Full seal + qualification; no study-valid shortcut.
    cp = read_json(path / 'checkpoints' / f'cp{k:06d}.json')
    cfg = read_json(path / 'raw/resolved_config.json')
    if cp['config_sha256'] != p.digest(cfg) or cp['budget_used']['model_requests'] != k:
        raise ValueError('source checkpoint config or actual request budget mismatch')
    if k == 0 and (cp['pending_calls'] or cp['pending_observations'] or any(cp['budget_used'].get(x, 0) for x in ('model_requests', 'tool_calls', 'known_total_tokens'))):
        raise ValueError('origin cp000000 is not before the first request')
    return {**value, 'path': str(path), 'checkpoint_id': f'cp{k:06d}', 'budget_used': cp['budget_used']}


def validate_audit(report):
    expected = set(SLOTS) - set(SELECTED)
    if (report.get('status') != 'pass' or report.get('passed_packages') != 42 or
            set(report.get('inventory', {})) != expected or
            any(report.get(key) != 0 for key in ('native_validation_failures', 'raw_differences', 'legacy_effective_differences')) or
            report.get('sources_unchanged') is not True or report.get('inventory_unchanged') is not True or
            report.get('native_identity_before') != report.get('native_identity_after') or
            not report.get('native_identity_before')):
        raise ValueError('R22 42-parent native audit failed or argument differences exist')
    if len(report.get('slots', [])) != 42 or {r['slot'] for r in report['slots']} != expected:
        raise ValueError('R22 audit slot coverage differs')
    for row in report['slots']:
        if row.get('status') != 'pass' or not row.get('full_seal_before_after') or row.get('index_sha256') != report['inventory'][row['slot']]:
            raise ValueError('R22 per-parent audit incomplete')
    for name, value in report['final_seal_verification'].items():
        if name not in expected or value.get('verified') is not True:
            raise ValueError('R22 final audit seal failed')
    if set(report['final_seal_verification']) != expected:
        raise ValueError('R22 final audit seal coverage differs')


def predecessor_bindings(base):
    paths = [base / 'timeout600_t2' / name for name in ('executed_plan.json', 'executed_freeze.json', 'result.json', 'admission.json', 'supervisor_result.json')]
    plan, freeze, result, ledger, supervisor = map(read_json, paths)
    verify_seal(plan, 'plan_sha256'); verify_seal(result); verify_seal(ledger)
    if (result.get('aggregate_http_attempts') != OLD_HTTP or plan.get('old_http_attempts') != 134 or result.get('g2') or ledger['requests'] != 10 or
            result['plan_sha256'] != plan['plan_sha256'] or ledger['plan_sha256'] != plan['plan_sha256'] or
            freeze['approved_plan_sha256'] != plan['plan_sha256'] or supervisor.get('exit_code') != 0 or supervisor.get('wall_timeout') or
            result['admission'] != {k: v for k, v in ledger.items() if k != 'artifact_sha256'}):
        raise ValueError('old 144 HTTP receipt identity or completion unknown')
    return {str(path): file_digest(path) for path in paths}


def reference_config(origin, config):
    """Accept only the unchanged baseline or its historical native-v1 delta."""
    legacy = origin['source_config']
    native_v1 = native_config(legacy)
    native_v1['agent']['harness_revision'] = 'l1-pi-native-errors-v1'
    if config != legacy and config != native_v1:
        raise ValueError('reference configuration exceeds baseline/native-v1 delta')
    return copy.deepcopy(config)


def g2_predecessor(cohort_path, phase):
    names = ('executed_plan.json', 'executed_freeze.json', 'result.json',
             'admission_continuations.json', 'admission_g2.json')
    paths = [phase / name for name in names]
    old, freeze, result, continuation, g2 = map(read_json, paths)
    cohort = read_json(cohort_path)
    verify_seal(old, 'plan_sha256')
    for value in (result, continuation, g2, cohort):
        verify_seal(value)
    if (old.get('schema_version') != 'r22-native-alignment-1' or old.get('selected') != list(SELECTED) or
            old.get('g2_slots') != G2_SLOTS or set(old.get('origins', {})) != set(SLOTS) or
            old.get('http_caps') != HTTP_CAPS or old.get('logical_caps') != LOGICAL_CAPS or
            old.get('old_http_attempts') != OLD_HTTP or old.get('aggregate_http_cap') != 3360 or
            old.get('aggregate_authorized_max') != 3324 or old.get('max_new_generation') != 63 or
            old.get('d0_frozen') is not False):
        raise ValueError('predecessor is not the original frozen R22 workload')
    if (freeze.get('approved_plan_sha256') != old['plan_sha256'] or
            any(freeze.get(key) is not True for key in ('phase_authorized', 'phase_frozen', 'internal_plan_reviewed')) or
            freeze.get('d0_frozen') is not False or not freeze.get('user_authorization_record', {}).get('exact_text') or
            not freeze.get('user_authorization_record', {}).get('source')):
        raise ValueError('predecessor freeze does not authorize the exact executed plan')
    if (result.get('plan_sha256') != old['plan_sha256'] or cohort.get('plan_sha256') != old['plan_sha256'] or
            result.get('aggregate_http_attempts') != G2_ONLY_OLD_HTTP or result.get('g2') != [] or
            result.get('all_45_study_valid') is not False or result.get('stop_reason') is not None or
            continuation.get('requests') != 37 or g2.get('requests') != 0 or
            len(continuation.get('sends', [])) != 37 or g2.get('sends') != []):
        raise ValueError('predecessor completed 37/0 HTTP and 181 aggregate receipt required')
    for name, ledger in (('continuations', continuation), ('g2', g2)):
        if (ledger.get('plan_sha256') != old['plan_sha256'] or
                result.get('admissions', {}).get(name) != {k: v for k, v in ledger.items() if k != 'artifact_sha256'}):
            raise ValueError('predecessor admission identity differs')
    receipts = result.get('continuations', [])
    if len(receipts) != 3 or [row['label'] for row in receipts] != list(SELECTED):
        raise ValueError('predecessor continuation coverage differs')
    if set(cohort.get('slots', {})) != set(SLOTS):
        raise ValueError('predecessor cohort must preserve all 45 slots')
    if cohort.get('demoted_controls') != {label: {**row, 'status': 'nonformal_control'} for label, row in old['demoted'].items()}:
        raise ValueError('predecessor demoted controls differ')
    for receipt in receipts:
        label = receipt['label']
        path = phase / f'receipt_continuations_{label}.json'
        value = read_json(path); verify_seal(value)
        if {k: v for k, v in value.items() if k != 'artifact_sha256'} != receipt:
            raise ValueError('predecessor continuation receipt differs')
        row = cohort['slots'][label]
        if (row['terminal_endpoint'] != receipt['package']['path'] or
                row['endpoint_index_sha256'] != receipt['package']['index_file_sha256'] or
                row['status'] != ('formal' if receipt['package']['valid'] else 'invalid_continuation')):
            raise ValueError('predecessor cohort differs from terminal receipt')
        paths.append(path)
    if (cohort['slots']['officeval_004.r02']['status'] != 'invalid_continuation' or
            any(row['status'] != 'formal' for label, row in cohort['slots'].items() if label != 'officeval_004.r02')):
        raise ValueError('predecessor invalid 004.r02 and 44 formal endpoints must be preserved')
    assert_originals(old)
    return old, cohort, {str(path.resolve()): file_digest(path) for path in [*paths, cohort_path]}


def build_g2_only_plan(*, service_stop_epoch, authorization_paths, config_path, run_root, cohort_path, predecessor_phase):
    old, cohort, bound = g2_predecessor(cohort_path, predecessor_phase)
    config = c.resolve_config(config_path)
    assert_delta(c.resolve_config(c.CONFIG_PATH), config)
    immutable = dict(old['immutable_trees'])
    immutable[old['run_root']] = s.tree_digest(Path(old['run_root']))
    references = {}
    for label, row in cohort['slots'].items():
        origin = old['origins'][label]
        if (row['attempt_id'] != origin['episode_id'] or row['origin_prefix'] != origin['path'] or
                row['origin_index_sha256'] != origin['index_sha256']):
            raise ValueError('predecessor cohort changed original attempt identity')
        endpoint = Path(row['terminal_endpoint'])
        references[label] = reference_config(origin, read_json(endpoint / 'raw/resolved_config.json'))
        for name in ('raw/resolved_config.json', 'evidence_index.json'):
            bound[str(endpoint / name)] = file_digest(endpoint / name)
        if file_digest(endpoint / 'evidence_index.json') != row['endpoint_index_sha256']:
            raise ValueError('predecessor terminal endpoint index changed')
        immutable[str(endpoint)] = s.tree_digest(endpoint)
    assert_write_paths(dict(immutable_trees=immutable, predecessor_phase=str(predecessor_phase)), run_root)
    if run_root.exists():
        raise ValueError('new G2-only root required; no overwrite or resume')
    for path in authorization_paths:
        bound[str(path.resolve())] = file_digest(path)
    plan = dict(schema_version=PLAN_SCHEMA, g2_readiness_policy=G2_READINESS_POLICY,
        mode='g2_only', phase='r22_existing_cohort_original_g2_context_v2', execution_ready=True,
        d0_frozen=False, service_stop_epoch=service_stop_epoch, service_safety_margin_seconds=300,
        service_renewal_authorized=False, wall_seconds=WALL, old_http_attempts=G2_ONLY_OLD_HTTP,
        http_caps=G2_ONLY_HTTP_CAPS, logical_caps=G2_ONLY_LOGICAL_CAPS,
        aggregate_http_cap=3360, aggregate_authorized_max=2981, max_new_generation=0,
        continuation_concurrency=0, g2_concurrency=3, attempt_limit_per_slot=1, selected=[], g2_slots=G2_SLOTS,
        config_path=str(config_path), native_config=config, origins=old['origins'], preserved=old['preserved'],
        demoted=old['demoted'], immutable_trees=immutable, run_root=str(run_root),
        source_fingerprints=source_files(config_path), bound_files=bound, native_dependency_identity=dependency_identity(),
        authorization_paths=[str(x.resolve()) for x in authorization_paths],
        predecessor_phase=str(predecessor_phase.resolve()), predecessor_plan_sha256=old['plan_sha256'],
        cohort_path=str(cohort_path.resolve()), frozen_cohort=cohort, reference_configs=references,
        source_profiles=dict(prefix='immutable original R22 v1 plan and cohort', continuation='none', g2='source_fingerprints'),
        known_selection_bias_slots=list(EXCLUDED_SENSITIVITY), pure_equivalence_eligible=False,
        g1_gate='historical_unverified', g2_gate='undecided', g4_gate='historical_unverified', scientific_efficacy='not_tested')
    validate_contract(seal(plan, 'plan_sha256'))
    assert_originals(plan); assert_sources(plan)
    return seal(plan, 'plan_sha256')


def build_plan(*, service_stop_epoch, authorization_paths, audit_path=OUTPUT / 'audit42/summary.json', config_path=CONFIG,
               run_root=None, base=BASE, cohort_path=None, predecessor_phase=None):
    if not math.isfinite(service_stop_epoch) or service_stop_epoch - time.time() < WINDOW or service_stop_epoch - time.time() > WALL:
        raise ValueError('fresh bounded service stop epoch required; no implicit renewal')
    if not authorization_paths:
        raise ValueError('exact user/Notion authorization artifacts required')
    if (cohort_path is None) != (predecessor_phase is None):
        raise ValueError('G2-only planning requires both cohort and predecessor phase')
    if run_root is None:
        run_root = G2_ONLY_RUN_ROOT if cohort_path is not None else RUN_ROOT
    if cohort_path is not None:
        return build_g2_only_plan(service_stop_epoch=service_stop_epoch, authorization_paths=authorization_paths,
            config_path=config_path, run_root=run_root, cohort_path=cohort_path, predecessor_phase=predecessor_phase)
    audit = read_json(audit_path); validate_audit(audit)
    if dependency_identity() != audit['native_identity_before']:
        raise ValueError('audit native dependencies differ from current execution dependencies')
    for name, expected in audit['sources'].items():
        if file_digest(ROOT / name) != expected:
            raise ValueError('audit executed source changed; repeat audit first')
    config = c.resolve_config(config_path)
    # Templates lack the task-specific, generated system prompt present in raw
    # episode configurations. Compare templates here; each trial uses its exact
    # original resolved configuration plus the three authorized changes.
    assert_delta(c.resolve_config(c.CONFIG_PATH), config)
    roots = {c.OUT_ROOT, L1 / 'runs/v3_qwen36_pi_dev45_t2_20261018'}
    immutable = {str(path): s.tree_digest(path) for path in roots}
    assert_write_paths(dict(immutable_trees=immutable), run_root)
    origins, preserved, demoted = {}, {}, {}
    from c3_sandbox.l1.core.orchestrator import _saved_response_source
    for label in SLOTS:
        path = L1 / SELECTED[label][0] if label in SELECTED else c.OUT_ROOT / 'episodes' / label
        row = s.identity(path)
        cfg = read_json(path / 'raw/resolved_config.json')
        manifest = read_json(path / 'manifest.json')
        if manifest.get('lineage') or cfg['benchmark']['task_id'] != label.rsplit('.', 1)[0]:
            raise ValueError('origin must be the matching original root attempt')
        preserved[label] = row
        origins[label] = dict(path=str(path), episode_id=manifest['episode_id'], index_sha256=row['package']['index_file_sha256'],
                              source_config=cfg, parent_timeout_seconds=cfg['model']['timeout_seconds'], cp0=checkpoint(path, 0))
        if label in SELECTED:
            _, eid, k, remaining = SELECTED[label]
            if manifest['episode_id'] != eid or row['package']['valid'] or manifest['counts']['model_requests'] != k + 1:
                raise ValueError('selected failed attempt identity or consumed budget differs')
            cp = checkpoint(path, k)
            response_id = f'r{k:06d}'
            saved = _saved_response_source(path, manifest, read_json(path / 'checkpoints' / f'cp{k:06d}.json'), response_id)
            if cfg['budget']['max_model_requests'] - (k + 1) != remaining:
                raise ValueError('selected continuation remaining request budget differs')
            origins[label].update(continuation_checkpoint=cp, resume_response_id=response_id, max_new_logical=remaining,
                response_sha256=__import__('hashlib').sha256(saved['response_bytes']).hexdigest())
        elif not row['package']['valid'] or row['package']['index_file_sha256'] != audit['inventory'][label]:
            raise ValueError('42-parent identity or study validity differs from audit')
    for label in DEMOTED:
        row = s.identity(c.OUT_ROOT / 'episodes' / label)
        if not row['package']['valid']:
            raise ValueError('demoted control must remain valid and immutable')
        demoted[label] = row
    if run_root.exists():
        raise ValueError('new R22 root already exists; no overwrite or resume')
    bound = predecessor_bindings(base)
    for path in [audit_path, *authorization_paths]:
        bound[str(path.resolve())] = file_digest(path)
    # Audit source profiles include the immutable archived controller used for old requests.
    for name in audit['sources']:
        bound[str(ROOT / name)] = file_digest(ROOT / name)
    for path in audit_path.parent.glob('*.json'):
        bound[str(path.resolve())] = file_digest(path)
    plan = dict(schema_version=PLAN_SCHEMA, g2_readiness_policy=G2_READINESS_POLICY, execution_ready=True, d0_frozen=False,
        phase='r22_continuations_then_original_g2', service_stop_epoch=service_stop_epoch,
        service_safety_margin_seconds=300, service_renewal_authorized=False, wall_seconds=WALL,
        old_http_attempts=OLD_HTTP, http_caps=HTTP_CAPS, logical_caps=LOGICAL_CAPS,
        aggregate_http_cap=3360, aggregate_authorized_max=3324, max_new_generation=63,
        continuation_concurrency=1, g2_concurrency=3, attempt_limit_per_slot=1,
        selected=list(SELECTED), g2_slots=G2_SLOTS, config_path=str(config_path), native_config=config,
        origins=origins, preserved=preserved, demoted=demoted, immutable_trees=immutable,
        run_root=str(run_root), source_fingerprints=source_files(config_path), bound_files=bound,
        native_dependency_identity=audit['native_identity_before'], audit_sha256=file_digest(audit_path),
        audit_executed_sources=audit['sources'], authorization_paths=[str(x.resolve()) for x in authorization_paths],
        source_profiles=dict(prefix='archived legacy controllers bound by raw packages and audit', continuation='source_fingerprints', g2='source_fingerprints'),
        known_selection_bias_slots=list(EXCLUDED_SENSITIVITY), pure_equivalence_eligible=False,
        g1_gate='historical_unverified', g2_gate='undecided', g4_gate='historical_unverified', scientific_efficacy='not_tested')
    assert_originals(plan); assert_sources(plan)
    return seal(plan, 'plan_sha256')


def validate_contract(plan):
    verify_seal(plan, 'plan_sha256')
    if plan.get('schema_version') != PLAN_SCHEMA or plan.get('g2_readiness_policy') != G2_READINESS_POLICY:
        raise ValueError('new R22 plan with explicit frozen20 readiness policy required')
    g2_only = plan.get('mode') == 'g2_only'
    if plan.get('mode', 'continuations_then_g2') not in ('continuations_then_g2', 'g2_only'):
        raise ValueError('unknown R22 execution mode')
    if (plan['selected'] != ([] if g2_only else list(SELECTED)) or plan['g2_slots'] != G2_SLOTS or
            set(plan['origins']) != set(SLOTS) or plan['http_caps'] != (G2_ONLY_HTTP_CAPS if g2_only else HTTP_CAPS) or
            plan['logical_caps'] != (G2_ONLY_LOGICAL_CAPS if g2_only else LOGICAL_CAPS) or
            plan['old_http_attempts'] != (G2_ONLY_OLD_HTTP if g2_only else OLD_HTTP) or plan['aggregate_http_cap'] != 3360 or
            plan['aggregate_authorized_max'] != (2981 if g2_only else 3324) or
            plan['max_new_generation'] != (0 if g2_only else 63) or plan['wall_seconds'] != WALL or
            plan['continuation_concurrency'] != (0 if g2_only else 1) or
            plan['g2_concurrency'] != 3 or plan['attempt_limit_per_slot'] != 1 or plan['d0_frozen'] is not False or
            plan['service_renewal_authorized'] is not False or set(plan['demoted']) != set(DEMOTED)):
        raise ValueError('R22 exact workload or hard limits differ')
    if g2_only:
        if (set(plan['reference_configs']) != set(SLOTS) or set(plan['frozen_cohort']['slots']) != set(SLOTS) or
                plan['frozen_cohort']['plan_sha256'] != plan['predecessor_plan_sha256'] or
                plan['pure_equivalence_eligible'] is not False):
            raise ValueError('G2-only frozen reference cohort differs')
        verify_seal(plan['frozen_cohort'])
        previous = read_json(Path(plan['predecessor_phase']) / 'executed_plan.json')
        verify_seal(previous, 'plan_sha256')
        if (previous['plan_sha256'] != plan['predecessor_plan_sha256'] or
                plan['frozen_cohort'] != read_json(Path(plan['cohort_path'])) or
                any(plan[key] != previous[key] for key in ('origins', 'preserved', 'demoted'))):
            raise ValueError('G2-only references differ from immutable predecessor')
        for label, config in plan['reference_configs'].items():
            reference_config(plan['origins'][label], config)
    if sum(row.get('max_new_logical', 0) for row in plan['origins'].values()) != 63:
        raise ValueError('R22 remaining logical budgets differ')
    for label, origin in plan['origins'].items():
        assert_delta(origin['source_config'], native_config(origin['source_config']))
        if origin['cp0']['checkpoint_id'] != 'cp000000':
            raise ValueError('G2 origin checkpoint must be actual cp000000')
        if label in SELECTED:
            path, eid, k, remaining = SELECTED[label]
            if (Path(origin['path']) != L1 / path or origin['episode_id'] != eid or
                    origin['resume_response_id'] != f'r{k:06d}' or origin['max_new_logical'] != remaining or
                    origin['continuation_checkpoint']['checkpoint_id'] != f'cp{k:06d}'):
                raise ValueError('selected origin or resume boundary changed')


class GuardedAdmission:
    def __init__(self, admission, plan, metadata, started, stop_event):
        self.admission, self.plan, self.metadata = admission, plan, metadata
        self.started, self.stop_event = started, stop_event

    def remaining_seconds(self):
        return max(0, min(WALL - (time.time() - self.started), self.plan['service_stop_epoch'] - time.time(), self.admission.remaining_seconds()))

    def acquire(self, estimate, timeout=None):
        if self.stop_event.is_set():
            return False
        assert_sources(self.plan)
        for path, expected in self.metadata.items():
            if file_digest(path) != expected:
                self.stop_event.set()
                raise ValueError('frozen executed plan/freeze changed')
        if self.remaining_seconds() < WINDOW or (timeout is not None and timeout < WINDOW):
            self.stop_event.set()
            raise RuntimeError('insufficient 600-second send plus safety window')
        return self.admission.acquire(estimate, timeout)

    def reconcile(self, estimate, actual):
        return self.admission.reconcile(estimate, actual)


def validate_lineage(origin, child, checkpoint_id, response=False):
    manifest = read_json(child / 'manifest.json')
    lineage = manifest.get('lineage') or {}
    cp = origin['continuation_checkpoint'] if response else origin['cp0']
    expected = dict(parent_episode_id=origin['episode_id'], parent_index_sha256=origin['index_sha256'],
                    parent_checkpoint_id=checkpoint_id, parent_checkpoint_sha256=cp['checkpoint_sha256'])
    if any(lineage.get(k) != v for k, v in expected.items()):
        raise ValueError('child lineage does not bind the real origin checkpoint')
    if response:
        use = lineage.get('response_use') or {}
        if (use.get('source_response_sha256') != origin['response_sha256'] or
                use.get('new_http_attempts_during_consume') != 0 or
                use != read_json(child / 'lineage/response_use.json')):
            raise ValueError('saved response consumption identity or zero-HTTP receipt differs')
    return lineage


def validate_cold_pair(origin, child):
    parent = Path(origin['path'])
    lineage = validate_lineage(origin, child, 'cp000000')
    before, after = [read_json(path / 'checkpoints/cp000000.json') for path in (parent, child)]
    for key in ('conversation', 'pending_calls', 'pending_observations', 'qualification'):
        if before[key] != after[key]:
            raise ValueError('G2 native checkpoint state mismatch: ' + key)
    for cp in (before, after):
        if cp['budget_used']['model_requests'] != 0 or cp['budget_used']['tool_calls'] != 0 or cp['budget_used']['known_total_tokens'] != 0:
            raise ValueError('G2 checkpoint contains used generation budget')
    snapshots = [read_json(path / 'states/snapshots' / (cp['snapshot_id'] + '.json')) for path, cp in ((parent, before), (child, after))]
    for key in ('content_sha256', 'files', 'directories', 'scope', 'total_bytes'):
        if snapshots[0][key] != snapshots[1][key]:
            raise ValueError('G2 restored workspace differs: ' + key)
    old, new = [read_json(path / 'raw/resolved_config.json') for path in (parent, child)]
    assert_delta(old, new)
    for name in ('benchmark/public_task.json',):
        if file_digest(parent / name) != file_digest(child / name):
            raise ValueError('G2 public task differs')
    first = 'raw/model/r000000/request.json'
    if file_digest(parent / first) != file_digest(child / first):
        raise ValueError('G2 actual first request bytes differ')
    return dict(lineage=lineage, actual_first_request_sha256=file_digest(child / first),
                source_first_request_sha256=file_digest(parent / first), native_state_equal=True,
                workspace_equal=True, runtime_qualification_equal=True, explicit_config_delta=True)


def score_record(label, path):
    from c3_sandbox.l1.evaluation.study import study_outcome
    outcome = study_outcome(path, verify_evidence=True)
    return dict(label=label, endpoint=str(path), endpoint_index_sha256=file_digest(path / 'evidence_index.json'),
                valid=outcome.get('study_valid') is True,
                score=outcome.get('study_outcome_y') if outcome.get('study_valid') is True else None)


def g2_readiness(plan, cohort):
    """Audit the exact frozen endpoints independently of DEV45 completeness.

    Eligibility permits the paired collection only; it does not pass G2 or D0.
    Common source and original-evidence guards apply even when all endpoints pass.
    """
    validate_contract(plan)
    common_errors = []
    try:
        assert_sources(plan)
        assert_originals(plan)
    except Exception as exc:
        common_errors.append(f'{type(exc).__name__}: {exc}')
    slots = {}
    for label in plan['g2_slots']:
        row = cohort.get(label)
        errors = []
        try:
            if not row or row.get('status') != 'formal':
                raise ValueError('required frozen slot lacks a formal terminal endpoint')
            origin = plan['origins'][label]
            if (row['attempt_id'] != origin['episode_id'] or row['origin_prefix'] != origin['path'] or
                    row['origin_index_sha256'] != origin['index_sha256']):
                raise ValueError('required endpoint origin identity differs')
            parent, endpoint = Path(origin['path']), Path(row['terminal_endpoint'])
            record = p.package_record(endpoint)
            if not record['valid'] or record['index_file_sha256'] != row['endpoint_index_sha256']:
                raise ValueError('required endpoint missing, invalid or index changed')
            if file_digest(parent / 'checkpoints/cp000000.json') != origin['cp0']['checkpoint_sha256']:
                raise ValueError('required origin cp000000 changed')
            config = read_json(endpoint / 'raw/resolved_config.json')
            expected = (plan['reference_configs'][label] if plan.get('mode') == 'g2_only' else
                        native_config(origin['source_config']) if label in SELECTED else origin['source_config'])
            if config != expected or config['benchmark']['task_id'] != label.rsplit('.', 1)[0]:
                raise ValueError('required endpoint task or runtime configuration differs')
            if file_digest(endpoint / 'benchmark/public_task.json') != file_digest(parent / 'benchmark/public_task.json'):
                raise ValueError('required endpoint public task differs')
            if label in SELECTED:
                if row['continuation_segment'] != str(endpoint) or endpoint == parent:
                    raise ValueError('required continuation endpoint differs')
                validate_lineage(origin, endpoint, origin['continuation_checkpoint']['checkpoint_id'], response=True)
            elif endpoint != parent or read_json(endpoint / 'manifest.json').get('lineage'):
                raise ValueError('required original endpoint replaced')
        except Exception as exc:
            errors.append(f'{type(exc).__name__}: {exc}')
        slots[label] = dict(ready=not errors, terminal_endpoint=row.get('terminal_endpoint') if row else None,
                            endpoint_index_sha256=row.get('endpoint_index_sha256') if row else None, errors=errors)
    return dict(schema_version='r22-g2-readiness-1', policy=G2_READINESS_POLICY, plan_sha256=plan['plan_sha256'],
                required_slots=list(plan['g2_slots']), n_required=len(slots), n_ready=sum(row['ready'] for row in slots.values()),
                common_harness_ready=not common_errors, common_errors=common_errors, slots=slots,
                eligible=not common_errors and all(row['ready'] for row in slots.values()),
                g2_gate='undecided', d0_frozen=False)


def score_inputs(plan, cohort, g2):
    primary = [score_record(label, Path(row['terminal_endpoint'])) for label, row in cohort.items()]
    alternatives = {label: score_record(label, Path(row['package']['path'])) for label, row in plan['demoted'].items()}
    return dict(schema_version='r22-score-inputs-1', plan_sha256=plan['plan_sha256'], primary=primary,
        sensitivity_a=[alternatives.get(row['label'], row) for row in primary],
        sensitivity_b=[row for row in primary if row['label'] not in EXCLUDED_SENSITIVITY],
        g2=[dict(label=row['label'], origin=row['origin'], endpoint=row['terminal_endpoint'],
                 endpoint_index_sha256=row['endpoint_index_sha256'], child=row['package']['path'],
                 comparison_valid=row['package']['valid'] and bool(row.get('structural_validation')),
                 endpoint_score=score_record(row['label'], Path(row['terminal_endpoint'])),
                 child_score=score_record(row['label'], Path(row['package']['path'])),
                 sensitivity_a_endpoint=alternatives.get(row['label'])) for row in g2],
        g1_gate='historical_unverified', g2_gate='undecided', g4_gate='historical_unverified', d0_frozen=False)


def summarize_cpu(inputs):
    def summary(rows):
        values = [r['score'] for r in rows if r['valid'] and r['score'] is not None]
        return dict(n_slots=len(rows), n_valid=len(values), n_invalid_or_unknown=len(rows)-len(values),
                    mean=sum(values)/len(values) if values else None)
    def paired(rows, alternate=False):
        values = []
        for row in rows:
            parent = (row.get('sensitivity_a_endpoint') or row['endpoint_score']) if alternate else row['endpoint_score']
            child = row['child_score']
            if row['comparison_valid'] and parent['valid'] and child['valid'] and parent['score'] is not None and child['score'] is not None:
                values.append((row['label'].rsplit('.', 1)[0], child['score'] - parent['score']))
        groups = {}
        for family, delta in values:
            groups.setdefault(family, []).append(delta)
        weights = {family: 1/len(groups) for family in groups} if groups else {}
        # Original independent-family bounded-difference Hoeffding formula (range 2).
        radius = math.sqrt(2 * math.log(40) * sum(w*w for w in weights.values())) if values else None
        mean = sum(sum(v)/len(v) for v in groups.values())/len(groups) if groups else None
        return dict(n_pairs=len(values), n_families=len(groups), mean_difference=mean,
            interval95=[max(-1, mean-radius), min(1, mean+radius)] if values else None,
            hoeffding_radius=radius, family_weights=weights, gate='undecided',
            comparison='joint restore/timeout/harness comparison; not pure equivalence')
    return dict(primary=summary(inputs['primary']), sensitivity_a=summary(inputs['sensitivity_a']),
        sensitivity_b=summary(inputs['sensitivity_b']), g2=paired(inputs['g2']),
        g2_sensitivity_a=paired(inputs['g2'], True),
        g2_sensitivity_b=paired([r for r in inputs['g2'] if r['label'] not in EXCLUDED_SENSITIVITY]),
        g1_gate='historical_unverified', g2_gate='undecided', g4_gate='historical_unverified', d0_frozen=False)


def execute(plan, freeze, output):
    validate_contract(plan)
    assert_write_paths(plan, Path(plan['run_root']), output)
    if (freeze.get('approved_plan_sha256') != plan['plan_sha256'] or freeze.get('phase_authorized') is not True or
            freeze.get('phase_frozen') is not True or freeze.get('internal_plan_reviewed') is not True or
            freeze.get('d0_frozen') is not False or not freeze.get('user_authorization_record', {}).get('exact_text') or
            not freeze.get('user_authorization_record', {}).get('source')):
        raise ValueError('reviewed R22 phase freeze and exact direct authorization required')
    assert_sources(plan); assert_originals(plan)
    if plan['service_stop_epoch'] - time.time() < WINDOW or plan['service_stop_epoch'] - time.time() > WALL:
        raise ValueError('insufficient bounded existing service lifetime')
    with p.exclusive(output / 'writer.lock'):
        if (output / 'result.json').exists():
            result = read_json(output / 'result.json'); verify_seal(result)
            if result['plan_sha256'] != plan['plan_sha256']:
                raise ValueError('result belongs to another plan')
            return result
        if (output / 'attempt_started.json').exists() or list(output.glob('admission_*.json')):
            raise ValueError('interrupted R22 stage; automatic restart forbidden')
        if Path(plan['run_root']).exists():
            raise ValueError('R22 output exists; no overwrite')
        with (output / 'attempt_started.json').open('xb') as handle:
            handle.write(json.dumps(dict(plan_sha256=plan['plan_sha256'], started_epoch=time.time())).encode())
            handle.flush(); os.fsync(handle.fileno())
        started = time.time()
        write_json(output / 'executed_plan.json', plan); write_json(output / 'executed_freeze.json', freeze)
        metadata = {path: file_digest(path) for path in (output / 'executed_plan.json', output / 'executed_freeze.json')}
        admissions = {phase: p.Admission(output / f'admission_{phase}.json', plan['plan_sha256'],
            {**p.LIMITS, 'max_requests': cap, 'max_tokens': cap*p.GIB, 'max_wall_seconds': WALL, 'max_invalid_attempts': 24})
            for phase, cap in plan['http_caps'].items()}
        stop_event = threading.Event()
        limiters = {phase: GuardedAdmission(admission, plan, metadata, started, stop_event) for phase, admission in admissions.items()}
        continuations, g2, stop_reason = [], [], None
        cohort = {label: dict(attempt_id=row['episode_id'], origin_prefix=row['path'], origin_index_sha256=row['index_sha256'],
            continuation_segment=None, terminal_endpoint=row['path'], endpoint_index_sha256=row['index_sha256'],
            status='pending_continuation' if label in SELECTED else 'formal') for label, row in plan['origins'].items()}
        if plan.get('mode') == 'g2_only':
            cohort = copy.deepcopy(plan['frozen_cohort']['slots'])
            write_json(output / 'cohort.json', seal(dict(plan_sha256=plan['plan_sha256'], slots=cohort,
                predecessor_plan_sha256=plan['predecessor_plan_sha256'],
                demoted_controls=plan['frozen_cohort']['demoted_controls'])))

        def trial(label, cold=False):
            phase = 'g2' if cold else 'continuations'
            if stop_event.is_set() or limiters[phase].remaining_seconds() < WINDOW:
                raise RuntimeError('stage stopped or request window exhausted')
            assert_sources(plan)
            origin = plan['origins'][label]
            parent = Path(origin['path'])
            cp = 'cp000000' if cold else origin['continuation_checkpoint']['checkpoint_id']
            path = Path(plan['run_root']) / ('g2' if cold else 'segments') / label
            if path.exists():
                raise ValueError('single trial output already exists')
            marker = output / f'attempt_{phase}_{label}.json'
            with marker.open('x') as handle:
                json.dump(dict(label=label, phase=phase, origin=origin, attempt=1), handle)
            kwargs = dict(parent=parent, checkpoint_id=cp, allow_pi_native_upgrade=True, request_limiter=limiters[phase])
            if not cold:
                kwargs['resume_response_id'] = origin['resume_response_id']
            error = None
            try:
                c.l1_run(native_config(origin['source_config']), path, **kwargs)
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
            record = p.package_record(path)
            row = dict(label=label, package=record, error=error, attempts=1, origin=origin['path'],
                origin_index_sha256=origin['index_sha256'], parent_timeout_seconds=origin['parent_timeout_seconds'],
                terminal_endpoint=cohort[label]['terminal_endpoint'], endpoint_index_sha256=cohort[label]['endpoint_index_sha256'])
            structural_error = None
            try:
                row['lineage'] = validate_lineage(origin, path, cp, response=not cold)
                if cold:
                    row['structural_validation'] = validate_cold_pair(origin, path)
                counts = read_json(path / 'manifest.json')['usage']
                used = counts['model_requests_including_parent'] - (0 if cold else SELECTED[label][2] + 1)
                if used < 0 or used > (35 if cold else origin['max_new_logical']):
                    raise ValueError('trial exceeded inherited logical budget')
                row['new_logical_requests'] = used
            except Exception as exc:
                structural_error = f'{type(exc).__name__}: {exc}'
                row['structural_error'] = structural_error
            if not record['valid']:
                admissions[phase].invalid()
            write_json(output / f'receipt_{phase}_{label}.json', seal(row))
            # Preserve an invalid receipt before propagating any structural mismatch.
            if structural_error:
                stop_event.set()
                for admission in admissions.values():
                    admission.close(structural_error)
                raise ValueError(structural_error)
            return row

        valid45, readiness = False, None
        try:
            for label in plan['selected']:
                row = trial(label); continuations.append(row)
                cohort[label].update(continuation_segment=row['package']['path'], terminal_endpoint=row['package']['path'],
                    endpoint_index_sha256=row['package']['index_file_sha256'], response_use=row['lineage']['response_use'],
                    status='formal' if row['package']['valid'] else 'invalid_continuation',
                    continuation_receipt=str(output / f'receipt_continuations_{label}.json'))
                assert_originals(plan)
                write_json(output / 'cohort.json', seal(dict(plan_sha256=plan['plan_sha256'], slots=cohort,
                    demoted_controls={label: {**row, 'status': 'nonformal_control'} for label, row in plan['demoted'].items()})))
            valid45 = len(cohort) == 45 and all(row['status'] == 'formal' and p.package_record(Path(row['terminal_endpoint']))['valid'] for row in cohort.values())
            readiness = g2_readiness(plan, cohort)
            write_json(output / 'g2_readiness.json', seal(readiness))
            if readiness['eligible']:
                with ThreadPoolExecutor(max_workers=3) as pool:
                    futures = [pool.submit(trial, label, True) for label in plan['g2_slots']]
                    for future in as_completed(futures):
                        try:
                            g2.append(future.result())
                        except Exception as exc:
                            stop_reason = stop_reason or f'{type(exc).__name__}: {exc}'
                            stop_event.set()
                            for admission in admissions.values():
                                admission.close(stop_reason)
                            for pending in futures:
                                pending.cancel()
                        write_json(output / 'progress.json', seal(dict(continuations=continuations, g2=g2, stop_reason=stop_reason)))
            write_json(output / 'score_inputs.json', seal(score_inputs(plan, cohort, g2)))
        except Exception as exc:
            stop_reason = f'{type(exc).__name__}: {exc}'; stop_event.set()
            for admission in admissions.values():
                admission.close(stop_reason)
        finally:
            assert_originals(plan)
        ledgers = {phase: admission.snapshot() for phase, admission in admissions.items()}
        total = plan['old_http_attempts'] + sum(row['requests'] for row in ledgers.values())
        result = seal(dict(plan_sha256=plan['plan_sha256'], continuations=continuations, g2=g2,
            all_45_study_valid=valid45, g2_readiness=readiness,
            g2_status='completed' if len(g2)==20 and all(r['package']['valid'] for r in g2) else 'incomplete',
            g2_gate='undecided', pure_equivalence_eligible=False, stop_reason=stop_reason, admissions=ledgers,
            aggregate_http_attempts=total, active_wall_seconds=time.time()-started, d0_frozen=False, scientific_efficacy='not_tested'))
        write_json(output / 'result.json', result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true'); parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--summarize-cpu', type=Path)
    parser.add_argument('--service-stop-epoch', type=float)
    parser.add_argument('--authorization', type=Path, action='append', default=[])
    parser.add_argument('--audit', type=Path, default=OUTPUT / 'audit42/summary.json')
    parser.add_argument('--cohort', type=Path)
    parser.add_argument('--predecessor-phase', type=Path)
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--plan', type=Path); parser.add_argument('--freeze', type=Path)
    args = parser.parse_args(); path = args.plan or args.output / 'plan.json'
    if args.summarize_cpu:
        inputs = read_json(args.summarize_cpu); verify_seal(inputs)
        write_json(args.output / 'sensitivity_summary.json', seal(summarize_cpu(inputs)))
    elif args.execute:
        if args.plan_only or not args.freeze:
            parser.error('--execute requires --freeze and cannot combine with --plan-only')
        result = execute(read_json(path), read_json(args.freeze), args.output)
        print(json.dumps(dict(g2_status=result['g2_status'], aggregate_http_attempts=result['aggregate_http_attempts'])))
    else:
        if args.service_stop_epoch is None:
            parser.error('planning requires --service-stop-epoch from the bounded fresh service lease')
        if args.predecessor_phase and any(args.predecessor_phase.resolve() in (target.resolve(), *target.resolve().parents)
                                         for target in (args.output, path)):
            parser.error('G2-only plan and output must be outside immutable predecessor phase')
        plan = build_plan(service_stop_epoch=args.service_stop_epoch, authorization_paths=args.authorization,
            audit_path=args.audit, config_path=args.config, cohort_path=args.cohort, predecessor_phase=args.predecessor_phase,
            run_root=args.run_root)
        assert_write_paths(plan, Path(plan['run_root']), args.output, path)
        write_json(path, plan)
        print(json.dumps(dict(plan=str(path), plan_sha256=plan['plan_sha256'],
            max_new_http=sum(plan['http_caps'].values()),
            max_new_continuation_requests=plan['max_new_generation'],
            max_g2_logical_requests=plan['logical_caps']['g2'],
            aggregate_authorized_max=plan['aggregate_authorized_max'])))


if __name__ == '__main__':
    main()
