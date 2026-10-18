#!/usr/bin/env python3
"""Independent, one-shot T2 collection; historical evidence is read-only."""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import types
import time

SCRIPT = Path(__file__).resolve()
spec = importlib.util.spec_from_file_location('t2_serial_helpers', SCRIPT.with_name('resume_d0_serial.py'))
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
p, c = s.prior, s.c
read_json, write_json, seal, verify_seal, file_digest = s.read_json, s.write_json, s.seal, s.verify_seal, s.file_digest
BASE = p.L2 / 'validation/d0_20261018'
OUTPUT = BASE / 'timeout600_t2'
PROTOCOL = p.L2 / 'configs/timeout_policy_t2_20261018.json'
CONFIG = p.L1 / 'configs/l1/officeval.qwen36.pi.t1.timeout600.json'
RUN_ROOT = p.L1 / 'runs/v3_qwen36_pi_dev45_t2_20261018'
LABEL = 'officeval_050.r04'
STOP = 1792314194.0
WINDOW = 645
OLD_HTTP = 134
NEW_HTTP = 2940


def timestamp(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()


def predecessor_bindings(base):
    """Bind final ledgers and supervisors, using measured intervals, never stale notes."""
    bindings, active = {}, 0.0
    for name, expected in [('prerequisite_resume', 93), ('serial_resume', 40)]:
        directory = base / name
        for filename in ('executed_plan.json', 'executed_freeze.json', 'result.json', 'admission.json', 'supervisor_result.json'):
            path = directory / filename
            bindings[str(path)] = file_digest(path)
        plan, result, ledger = [read_json(directory / x) for x in ('executed_plan.json', 'result.json', 'admission.json')]
        verify_seal(plan, 'plan_sha256'); verify_seal(result); verify_seal(ledger)
        if (ledger['requests'] != expected or len(ledger['sends']) != expected or
                result['admission'] != {k: v for k, v in ledger.items() if k != 'artifact_sha256'} or
                result['plan_sha256'] != plan['plan_sha256'] or ledger['plan_sha256'] != plan['plan_sha256'] or
                read_json(directory / 'executed_freeze.json')['approved_plan_sha256'] != plan['plan_sha256'] or result.get('g2')):
            raise ValueError('unexpected final predecessor budget or identity')
        supervisor = read_json(directory / 'supervisor_result.json')
        if supervisor['exit_code'] != 0 or supervisor.get('wall_timeout'):
            raise ValueError('predecessor supervisor incomplete')
        elapsed = timestamp(supervisor['finished_at_utc']) - timestamp(supervisor['started_at_utc'])
        if elapsed <= 0 or not math.isfinite(elapsed):
            raise ValueError('invalid predecessor active interval')
        active += elapsed
    probe = base / 'timeout_probe'
    for filename in ('executed_plan.json', 'authorization.json', 'attempt_started.json', 'result.json', 'http_metadata.json', 'supervisor_result.json'):
        bindings[str(probe / filename)] = file_digest(probe / filename)
    attempt, result, supervisor = [read_json(probe / x) for x in ('attempt_started.json', 'result.json', 'supervisor_result.json')]
    if (attempt['aggregate_including_probe'] != OLD_HTTP or result.get('http_attempts') != 1 or
            supervisor['exit_code'] != 0 or not supervisor['attempt_reserved'] or supervisor.get('hard_deadline_termination')):
        raise ValueError('probe budget or completion unknown')
    serial_result = read_json(base / 'serial_resume/result.json')
    probe_duration = read_json(probe / 'http_metadata.json')['duration_seconds']
    active = serial_result['active_wall_seconds'] + probe_duration
    if not math.isfinite(active) or active <= 0:
        raise ValueError('invalid prior aggregate active receipt')
    return bindings, active


def assert_sources(plan):
    for name, expected in plan['source_fingerprints'].items():
        if file_digest(p.ROOT / name) != expected:
            raise ValueError('frozen execution source/config/protocol changed')
    for name, expected in plan['predecessor_files'].items():
        if file_digest(Path(name)) != expected:
            raise ValueError('predecessor receipt changed')


def timeout_config(config, old):
    expected = copy.deepcopy(old)
    expected['model']['timeout_seconds'] = 600
    if config != expected:
        raise ValueError('T2 may change only model.timeout_seconds from 300 to 600')
    p.assert_config(old, old)


def build_plan(protocol_path=PROTOCOL, *, base=BASE, config_path=CONFIG, run_root=RUN_ROOT):
    protocol = read_json(protocol_path)
    validate_protocol(protocol)
    bindings, active = predecessor_bindings(base)
    old = read_json(base / 'serial_resume/executed_plan.json')
    config = c.resolve_config(config_path)
    timeout_config(config, old['frozen_base_config'])
    sources = dict(old['source_fingerprints'])
    for path in (SCRIPT, Path(__file__).with_name('resume_d0_serial.py'), protocol_path, config_path,
                 p.L2 / 'tests/test_d0_timeout600.py', SCRIPT.with_name('probe_transport_timeout.py'),
                 p.L2 / 'src/l2_sandbox/evidence/l1.py'):
        sources[str(path.resolve().relative_to(p.ROOT))] = file_digest(path)
    for path in sorted((p.L2 / 'src/l2_sandbox').rglob('*.py')):
        sources[str(path.relative_to(p.ROOT))] = file_digest(path)
    parents = {label: s.identity(Path(row['package']['path'])) for label, row in old['parents'].items()}
    if len(parents) != 45 or [k for k, row in parents.items() if not row['package']['valid']] != [LABEL]:
        raise ValueError('expected 44 valid originals and only invalid 050')
    # Bind the complete original run, including failed attempts, rather than only sealed indices.
    original_tree = s.tree_digest(c.OUT_ROOT)
    if run_root.exists():
        raise ValueError('independent T2 run root already exists; no overwrite or resume')
    caps = {**p.LIMITS, 'max_requests': NEW_HTTP, 'max_tokens': NEW_HTTP * p.GIB,
            'max_wall_seconds': math.floor(21600 - active), 'max_invalid_attempts': 21}
    if caps['max_wall_seconds'] < WINDOW:
        raise ValueError('aggregate six-hour active budget exhausted')
    plan = dict(schema_version='d0-timeout600-t2-1', execution_ready=True, d0_frozen=False,
        phase='independent_timeout600_t2', execution_concurrency=3, replacement_concurrency=1,
        g2_concurrency=3, replacements=[LABEL], attempt_limit_per_slot=1,
        timeout_seconds=600, request_safety_seconds=45, service_stop_epoch=STOP,
        aggregate_http_cap=3360, old_http_attempts=OLD_HTTP, old_active_seconds=active,
        aggregate_active_wall_cap=21600, limits=caps, pricing_mode=old['pricing_mode'],
        actor_identity=old['actor_identity'], frozen_base_config=config,
        frozen_parent_configs=old['frozen_parent_configs'], parents=parents,
        original_run_root=str(c.OUT_ROOT), original_tree_sha256=original_tree,
        run_root=str(run_root), protocol=protocol, protocol_path=str(protocol_path),
        source_fingerprints=sources, predecessor_files=bindings,
        g2_slots=old['g2_slots'], g2_checkpoint='cp000000',
        g2_purpose='joint cold restore plus timeout regime comparison; not pure equivalence',
        derivative=derivative_record(),
        g2_gate='undecided', scientific_efficacy='not_tested')
    assert_sources(plan)
    return seal(plan, 'plan_sha256')


def validate_protocol(protocol):
    expected = dict(schema_version='d0-timeout-policy-t2-1', t2_transport_policy_id='T-2-20261018-timeout600',
        parent_timeout_seconds=[300, 600], continuation_timeout_seconds=600, repair_label=LABEL,
        max_attempts_per_slot=1, replacement_concurrency=1, g2_concurrency=3,
        g2_checkpoint='cp000000', aggregate_http_cap=3360, prior_http_attempts=134,
        new_http_cap=2940, aggregate_active_wall_cap=21600, service_stop_epoch=STOP,
        original_parents_immutable=True, original_invalid_parent_immutable=True,
        pure_restore_equivalence_eligible=False, scientific_efficacy='not_tested', service_renewal_authorized=False)
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError('T2 protocol workload or evidence boundaries changed')
    for prefix in ('base', 'revised'):
        if file_digest(p.ROOT / protocol[prefix + '_config_path']) != protocol[prefix + '_config_sha256']:
            raise ValueError('protocol configuration hash differs')


def assert_originals(plan):
    s.assert_parents(plan['parents'])
    if s.tree_digest(Path(plan['original_run_root'])) != plan['original_tree_sha256']:
        raise ValueError('original batch tree changed')


def _t2_resume_configs_match(current, previous):
    if current == previous:
        return True
    expected = copy.deepcopy(previous)
    if expected.get('model', {}).get('timeout_seconds') != 300:
        return False
    expected['model']['timeout_seconds'] = 600
    return current == expected


def derivative_record(source_path=None):
    path = Path(source_path) if source_path is not None else p.L1 / 'src/c3_sandbox/l1/core/orchestrator.py'
    source = path.read_text()
    guard = 'if base_config!=prev_base:'
    if source.count(guard) != 1:
        raise ValueError('expected exact single original config guard')
    generated = source.replace(guard, 'if not _t2_resume_configs_match(base_config, prev_base):')
    return dict(version='T2-single-config-guard-v1', original_path=str(path),
        original_sha256=file_digest(path), generated_sha256=hashlib.sha256(generated.encode()).hexdigest(),
        old_guard=guard, new_guard='if not _t2_resume_configs_match(base_config, prev_base):')


def derivative_runner(plan, output):
    record = derivative_record()
    if record != plan['derivative']:
        raise ValueError('T2 derivative source changed')
    source = Path(record['original_path']).read_text().replace(record['old_guard'], record['new_guard'])
    generated_path = output / 'GENERATED_SOURCE.py'
    with generated_path.open('x') as handle:
        handle.write(source); handle.flush(); os.fsync(handle.fileno())
    write_json(output / 'source_transform.json', seal(record))
    module = types.ModuleType('c3_sandbox.l1.core.timeout600_t2_v1')
    module.__package__ = 'c3_sandbox.l1.core'
    module.__file__ = str(generated_path)
    module.__dict__['_t2_resume_configs_match'] = _t2_resume_configs_match
    exec(compile(source, str(generated_path), 'exec'), module.__dict__)

    def run(config, path, **kwargs):
        if kwargs.get('checkpoint_id') != 'cp000000' or kwargs.get('parent') is None:
            raise ValueError('T2 derivative is restricted to cp000000 cold continuation')
        if file_digest(generated_path) != record['generated_sha256']:
            raise ValueError('generated T2 source changed')
        return module.run(config, path, **kwargs)
    return run


class GuardedAdmission:
    def __init__(self, admission, plan, metadata, output=None, metrics_reader=None):
        self.admission, self.plan, self.metadata, self.output = admission, plan, metadata, output
        self.metrics_reader = metrics_reader
        if output is not None and metrics_reader is None:
            probe_spec = importlib.util.spec_from_file_location('t2_metrics', SCRIPT.with_name('probe_transport_timeout.py'))
            probe = importlib.util.module_from_spec(probe_spec); probe_spec.loader.exec_module(probe)
            self.metrics_reader = probe.metrics

    def remaining_seconds(self):
        return max(0, min(self.admission.remaining_seconds(), self.plan['service_stop_epoch'] - time.time()))

    def metrics(self):
        try:
            return dict(status='observed', values=self.metrics_reader())
        except Exception as exc:
            return dict(status='unknown', exception_type=type(exc).__name__)

    def acquire(self, estimate, timeout=None):
        assert_sources(self.plan)
        for path, expected in self.metadata.items():
            if file_digest(path) != expected:
                raise ValueError('T2 execution metadata changed')
        if self.remaining_seconds() < WINDOW or (timeout is not None and timeout < WINDOW):
            raise RuntimeError('insufficient 600-second request plus safety window')
        if self.plan['old_http_attempts'] + self.admission.snapshot()['requests'] >= 3360:
            raise RuntimeError('aggregate HTTP cap reached')
        admitted = self.admission.acquire(estimate, timeout)
        if admitted and self.output is not None:
            index = self.admission.current
            write_json(self.output / f'http_{index:06d}.json', seal(dict(attempt_index=index,
                admitted_epoch=time.time(), server_before=self.metrics(), actual_tokens=None,
                status='reserved_usage_unknown', timeout_policy_seconds=600)))
        return admitted

    def reconcile(self, estimate, actual):
        if self.output is not None:
            index = self.admission.current
            path = self.output / f'http_{index:06d}.json'
            row = read_json(path); row.pop('artifact_sha256', None)
            row.update(completed_epoch=time.time(), admission_to_reconcile_seconds=time.time() - row['admitted_epoch'],
                actual_tokens=actual, server_after=self.metrics(), status='unknown_usage' if actual is None else 'settled',
                detail_basis='native raw transport receipts retain prompt/completion tokens and HTTP latency')
            write_json(path, seal(row))
        return self.admission.reconcile(estimate, actual)


def validate_pair(parent, child):
    from l2_sandbox.evidence.l1 import L1Package
    before, after = L1Package(parent), L1Package(child)
    before.validate_resume_checkpoint(0)
    after.validate_resume_checkpoint(0)
    old, new = before.json('raw/resolved_config.json'), after.json('raw/resolved_config.json')
    if not _t2_resume_configs_match(new, old) or new['model']['timeout_seconds'] != 600:
        raise ValueError('pair config delta exceeds explicit T2 policy')
    if before.family_id != after.family_id or after.start_turn != 0:
        raise ValueError('pair task or start-turn mismatch')
    parent_cp, child_cp = [read_json(path / 'checkpoints/cp000000.json') for path in (parent, child)]
    for key in ('conversation', 'pending_calls', 'pending_observations', 'qualification'):
        if parent_cp[key] != child_cp[key]:
            raise ValueError('pair checkpoint state mismatch: ' + key)
    for key in ('model_requests', 'tool_calls', 'known_total_tokens'):
        if parent_cp['budget_used'][key] != 0 or child_cp['budget_used'][key] != 0:
            raise ValueError('pair cp000000 contains nonzero request/tool/token budget')
    snapshots = [read_json(path / 'states/snapshots' / (cp['snapshot_id'] + '.json'))
        for path, cp in ((parent, parent_cp), (child, child_cp))]
    for key in ('content_sha256', 'files', 'directories', 'scope', 'total_bytes'):
        if snapshots[0][key] != snapshots[1][key]:
            raise ValueError('pair workspace state mismatch: ' + key)
    return dict(sealed_parent=True, sealed_child=True, config_delta_valid=True,
        native_conversation_equal=True, pending_equal=True, workspace_equal=True,
        runtime_qualification_equal=True, task_equal=True, start_turn=0)


def validate_contract(plan):
    caps = plan['limits']
    if (plan['replacements'] != [LABEL] or plan['attempt_limit_per_slot'] != 1 or
        plan['execution_concurrency'] != 3 or plan['replacement_concurrency'] != 1 or plan['g2_concurrency'] != 3 or
        plan['g2_checkpoint'] != 'cp000000' or plan['g2_slots'] != [f'{t}.{r}' for t, r in c.G2_SLOTS] or
        plan['old_http_attempts'] != OLD_HTTP or plan['aggregate_http_cap'] != 3360 or
        plan['service_stop_epoch'] != STOP or plan['timeout_seconds'] != 600 or plan['request_safety_seconds'] != 45 or
        caps != {**p.LIMITS, 'max_requests': NEW_HTTP, 'max_tokens': NEW_HTTP * p.GIB,
            'max_wall_seconds': math.floor(21600 - plan['old_active_seconds']), 'max_invalid_attempts': 21}):
        raise ValueError('exact T2 workload, timeout and aggregate limits required')
    base = copy.deepcopy(plan['frozen_base_config']); base['model']['timeout_seconds'] = 300
    timeout_config(plan['frozen_base_config'], base)
    validate_protocol(plan['protocol'])
    if plan['derivative'] != derivative_record():
        raise ValueError('T2 derivative differs from frozen source')
    if Path(plan['run_root']).resolve() == Path(plan['original_run_root']).resolve():
        raise ValueError('T2 output must be independent')


def execute(plan, freeze, output):
    verify_seal(plan, 'plan_sha256')
    if (freeze.get('phase_authorized') is not True or freeze.get('phase_frozen') is not True or
            freeze.get('d0_frozen') is not False or freeze.get('internal_plan_reviewed') is not True):
        raise ValueError('reviewed independent T2 phase authorization required')
    auth = freeze.get('user_authorization_record') or {}
    if not auth.get('exact_text') or not auth.get('source'):
        raise ValueError('direct user authorization text and source required')
    caps = p.approve(plan, {**freeze, 'd0_frozen': True})
    validate_contract(plan)
    if caps != plan['limits']:
        raise ValueError('frozen T2 limits differ')
    assert_sources(plan)
    with p.exclusive(output / 'writer.lock'):
        if (output / 'result.json').exists():
            result = read_json(output / 'result.json'); verify_seal(result)
            if result['plan_sha256'] != plan['plan_sha256']:
                raise ValueError('existing T2 result belongs to another plan')
            return result
        if (output / 'attempt_started.json').exists() or (output / 'admission.json').exists():
            raise ValueError('interrupted T2 stage; automatic restart forbidden')
        assert_originals(plan)
        run_root = Path(plan['run_root'])
        if run_root.exists():
            raise ValueError('T2 run root exists; no overwrite')
        if STOP - time.time() < WINDOW:
            raise ValueError('insufficient existing service lifetime; no renewal')
        # Reserve the entire stage before creating any episode or sending any HTTP request.
        with (output / 'attempt_started.json').open('xb') as handle:
            handle.write(json.dumps(dict(plan_sha256=plan['plan_sha256'], started_epoch=time.time())).encode())
            handle.flush(); os.fsync(handle.fileno())
        write_json(output / 'executed_plan.json', plan); write_json(output / 'executed_freeze.json', freeze)
        admission = p.Admission(output / 'admission.json', plan['plan_sha256'], caps)
        metadata = {path: file_digest(path) for path in (output / 'executed_plan.json', output / 'executed_freeze.json')}
        cold_runner = derivative_runner(plan, output)
        metadata.update({path: file_digest(path) for path in (output / 'GENERATED_SOURCE.py', output / 'source_transform.json')})
        limiter = GuardedAdmission(admission, plan, metadata, output / 'telemetry')
        cohort = {k: row['package']['path'] for k, row in plan['parents'].items() if k != LABEL}
        replacements, g2, stop_reason = [], [], None

        def run_slot(label, cold=False):
            admission.check_new_request()
            if limiter.remaining_seconds() < WINDOW:
                raise RuntimeError('insufficient request window')
            assert_sources(plan)
            parent = Path(cohort[label]) if cold else None
            path = run_root / ('v3_g2_c0_resumes' if cold else 'episodes') / (f'{label}_c0' if cold else label)
            if path.exists():
                raise ValueError('attempt output exists; no overwrite')
            config = copy.deepcopy(plan['frozen_base_config'])
            config['benchmark']['task_id'] = label.rsplit('.', 1)[0]
            before = s.identity(parent) if cold else None
            parent_timeout = None
            if cold:
                config = read_json(parent / 'raw/resolved_config.json')
                parent_timeout = config['model']['timeout_seconds']
                expected = plan['frozen_parent_configs'][label] if label != LABEL else read_json(run_root / 'episodes' / LABEL / 'raw/resolved_config.json')
                if config != expected or parent_timeout not in (300, 600):
                    raise ValueError('parent configuration changed')
                config = copy.deepcopy(config); config['model']['timeout_seconds'] = 600
            write_json(output / f'attempt_{label}{"_c0" if cold else ""}.json', seal(dict(label=label, cold=cold, attempts=1)))
            error = None
            try:
                kwargs = dict(request_limiter=limiter)
                if cold:
                    kwargs.update(parent=parent, checkpoint_id='cp000000')
                (cold_runner if cold else c.l1_run)(config, path, **kwargs)
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
            package = p.package_record(path)
            if not package['valid']:
                admission.invalid()
            row = dict(label=label, package=package, attempts=1, error=error,
                parent_timeout_seconds=parent_timeout, continuation_timeout_seconds=600)
            if cold:
                s.assert_parents({label: before})
                first = Path('raw/model/r000000/request.json')
                row['parent_first_request_sha256'] = file_digest(parent / first)
                row['child_first_request_sha256'] = file_digest(path / first) if (path / first).is_file() else None
                row['byte_identical_first_request'] = row['parent_first_request_sha256'] == row['child_first_request_sha256']
                lineage = (read_json(path / 'manifest.json').get('lineage') or {}) if (path / 'manifest.json').exists() else {}
                row['lineage'] = lineage
                row['cold_parent_binding_valid'] = (lineage.get('parent_episode_id') == before['package']['info'].get('episode_id') and
                    lineage.get('parent_index_sha256') == before['package']['info'].get('index_sha256') and
                    lineage.get('parent_checkpoint_id') == 'cp000000' and
                    lineage.get('parent_checkpoint_sha256') == file_digest(parent / 'checkpoints/cp000000.json'))
                if row['byte_identical_first_request'] and row['cold_parent_binding_valid']:
                    try:
                        row['structural_validation'] = validate_pair(parent, path)
                    except Exception as exc:
                        row['structural_validation_error'] = f'{type(exc).__name__}: {exc}'
                if not (row['byte_identical_first_request'] and row['cold_parent_binding_valid'] and row.get('structural_validation')):
                    admission.close('G2 request, lineage or structural mismatch')
                row['comparison'] = 'joint_restore_timeout' if parent_timeout == 300 else 'same_timeout_restore'
            write_json(output / f'receipt_{label}{"_c0" if cold else ""}.json', seal(row))
            return row

        valid45 = False
        try:
            replacements.append(run_slot(LABEL))
            assert_originals(plan)
            if replacements[0]['package']['valid']:
                cohort[LABEL] = replacements[0]['package']['path']
                valid45 = all(p.package_record(Path(path))['valid'] for path in cohort.values()) and len(cohort) == 45
            write_json(output / 'mixed_cohort.json', seal(dict(plan_sha256=plan['plan_sha256'], parents=cohort,
                all_45_study_valid=valid45, timeout_regime='44 original 300s plus one independent 600s')))
            if valid45:
                cohort_before = {label: s.identity(Path(path)) for label, path in cohort.items()}
                with ThreadPoolExecutor(max_workers=3) as pool:
                    futures = [pool.submit(run_slot, label, True) for label in plan['g2_slots']]
                    for future in as_completed(futures):
                        try:
                            row = future.result(); g2.append(row)
                            if not row['byte_identical_first_request'] or not row['cold_parent_binding_valid'] or not row.get('structural_validation'):
                                raise ValueError('G2 actual request bytes or lineage mismatch')
                        except Exception as exc:
                            stop_reason = stop_reason or f'{type(exc).__name__}: {exc}'
                            admission.close(stop_reason)
                s.assert_parents(cohort_before)
        except Exception as exc:
            stop_reason = f'{type(exc).__name__}: {exc}'; admission.close(stop_reason)
        finally:
            assert_originals(plan)
        completed = len(g2) == 20 and all(row['package']['valid'] and row['byte_identical_first_request'] and row['cold_parent_binding_valid'] and row.get('structural_validation') for row in g2)
        result = seal(dict(plan_sha256=plan['plan_sha256'], replacements=replacements,
            all_45_study_valid=valid45, g2=g2, g2_status='completed' if completed else 'incomplete',
            g2_gate='undecided', comparison='joint cold restore plus timeout regime comparison; not pure equivalence',
            stop_reason=stop_reason, admission=admission.snapshot(),
            aggregate_http_attempts=OLD_HTTP + admission.snapshot()['requests'],
            aggregate_active_seconds=plan['old_active_seconds'] + time.time() - admission.state['started_at'],
            d0_frozen=False, scientific_efficacy='not_tested'))
        write_json(output / 'result.json', result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true'); parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--plan', type=Path); parser.add_argument('--freeze', type=Path)
    args = parser.parse_args(); path = args.plan or args.output / 'plan.json'
    if args.execute:
        if args.plan_only or not args.freeze:
            parser.error('--execute requires --freeze and cannot combine with --plan-only')
        result = execute(read_json(path), read_json(args.freeze), args.output)
        print(json.dumps(dict(g2_status=result['g2_status'], requests=result['admission']['requests'])))
    else:
        plan = build_plan(args.protocol); write_json(path, plan)
        print(json.dumps(dict(plan=str(path), plan_sha256=plan['plan_sha256'], max_new_http=NEW_HTTP)))


if __name__ == '__main__':
    main()
