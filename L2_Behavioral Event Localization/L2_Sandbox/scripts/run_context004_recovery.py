#!/usr/bin/env python3
"""Plan the one authorized 004 context recovery; execution requires a reviewed freeze."""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import threading
import time

SCRIPT = Path(__file__).resolve()
spec = importlib.util.spec_from_file_location('context004_r22', SCRIPT.with_name('run_r22_native_alignment.py'))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
p, c, s = r.p, r.c, r.s
ROOT, L1, L2 = r.ROOT, r.L1, r.L2
read_json, write_json, seal, verify_seal, file_digest = r.read_json, r.write_json, r.seal, r.verify_seal, r.file_digest
LABEL = 'officeval_004.r02'
PARENT = r.RUN_ROOT / 'segments' / LABEL
PARENT_INDEX = '44f52afac2434457382af8649d01a8e9afdf459dd70a48e9fcd0085006be2411'
CHECKPOINT, FAILED = 'cp000013', 'r000020'
OLD_HTTP, HTTP_CAP, LOGICAL_CAP, WALL, WINDOW = 578, 56, 14, 21600, 645
SCHEMA = 'context004-recovery-1'
OUTPUT = r.BASE / 'context004_recovery'
RUN_ROOT = L1 / 'runs/v3_qwen36_pi_context004_20261018'
G2_PHASE = r.BASE / 'r23_g2_original20'
assert_originals, assert_write_paths = r.assert_originals, r.assert_write_paths


def source_files(config_path):
    sources = r.source_files(config_path)
    for path in (SCRIPT, SCRIPT.with_name('supervise_context004_recovery.py'), L2 / 'tests/test_context004_recovery.py',
                 L2 / 'scripts/audit_g1_mechanical.py', L2 / 'tests/test_g1_mechanical_audit.py',
                 L2 / 'tests/test_g4_provenance.py', L1 / 'tests/l1_live/test_pi_context_overflow_resume.py'):
        sources[str(path.relative_to(ROOT))] = file_digest(path)
    return sources


def assert_sources(plan):
    if source_files(Path(plan['config_path'])) != plan['source_fingerprints']:
        raise ValueError('frozen context recovery source changed')
    for name, expected in plan['bound_files'].items():
        if file_digest(Path(name)) != expected:
            raise ValueError('frozen predecessor/authorization changed')
    if r.dependency_identity() != plan['native_dependency_identity']:
        raise ValueError('native installed dependency identity changed')


def validate_freeze(plan, freeze):
    if (freeze.get('approved_plan_sha256') != plan['plan_sha256'] or
            any(freeze.get(key) is not True for key in ('phase_authorized', 'phase_frozen', 'internal_plan_reviewed')) or
            freeze.get('d0_frozen') is not False or not freeze.get('user_authorization_record', {}).get('exact_text') or
            not freeze.get('user_authorization_record', {}).get('source')):
        raise ValueError('explicit reviewed context recovery freeze required')


def predecessor(cohort_path, phase, g2_phase):
    old, cohort, bound = r.g2_predecessor(cohort_path, phase)
    paths = [g2_phase / name for name in ('executed_plan.json', 'executed_freeze.json', 'result.json', 'admission_continuations.json', 'admission_g2.json', 'cohort.json')]
    gp, freeze, result, continuation, g2, gc = map(read_json, paths)
    r.validate_contract(gp); validate_freeze(gp, freeze)
    for value in (result, continuation, g2, gc):
        verify_seal(value)
    if (gp.get('mode') != 'g2_only' or gp.get('predecessor_plan_sha256') != old['plan_sha256'] or
            gp.get('frozen_cohort') != cohort or gc.get('slots') != cohort['slots'] or
            result.get('plan_sha256') != gp['plan_sha256'] or result.get('aggregate_http_attempts') != OLD_HTTP or
            result.get('g2_status') != 'completed' or result.get('stop_reason') is not None or
            result.get('continuations') != [] or continuation.get('requests') != 0 or g2.get('requests') != 397 or
            len(g2.get('sends', [])) != 397 or continuation.get('sends') != []):
        raise ValueError('completed unchanged 397 HTTP G2 predecessor required')
    for name, ledger in (('continuations', continuation), ('g2', g2)):
        if (ledger.get('plan_sha256') != gp['plan_sha256'] or
                result.get('admissions', {}).get(name) != {k:v for k,v in ledger.items() if k != 'artifact_sha256'}):
            raise ValueError('G2 predecessor admission differs')
    rows = result.get('g2', [])
    if len(rows) != 20 or {row['label'] for row in rows} != set(r.G2_SLOTS):
        raise ValueError('exact original twenty G2 pairs required')
    for row in rows:
        path = g2_phase / ('receipt_g2_' + row['label'] + '.json')
        receipt = read_json(path); verify_seal(receipt)
        package = p.package_record(Path(row['package']['path']))
        if ({k:v for k,v in receipt.items() if k != 'artifact_sha256'} != row or not row['package']['valid'] or
                not row.get('structural_validation') or package != row['package']):
            raise ValueError('sealed G2 receipt/package differs')
        if (row['terminal_endpoint'] != cohort['slots'][row['label']]['terminal_endpoint'] or
                row['endpoint_index_sha256'] != cohort['slots'][row['label']]['endpoint_index_sha256']):
            raise ValueError('G2 pair does not use frozen terminal endpoint')
        r.validate_cold_pair(gp['origins'][row['label']], Path(row['package']['path']))
        paths.append(path)
    bound.update({str(path.resolve()):file_digest(path) for path in paths})
    return old, cohort, gp, result, bound


def context_source(path):
    from c3_sandbox.l1.evidence.live import context_overflow_source, verify
    report = verify(path, expected_index_sha256=PARENT_INDEX)
    if not report['verified']:
        raise ValueError('context parent full seal failed: ' + str(report))
    proof = context_overflow_source(path, CHECKPOINT, FAILED)
    if (proof['manifest']['usage']['model_requests_including_parent'] != 21 or
            proof['config']['benchmark']['task_id'] != 'officeval_004' or
            proof['config']['agent']['harness_revision'] != 'l1-pi-native-errors-v1' or
            proof['config']['model']['timeout_seconds'] != 600 or
            proof['config']['budget']['max_model_requests'] != 35):
        raise ValueError('exact context parent budget/config differs')
    r.assert_delta(proof['config'], r.native_config(proof['config']))
    return proof


def service_lease(path, stop, now):
    value = read_json(path)
    keys = ('started_epoch', 'hard_stop_epoch', 'service_stop_epoch')
    if any(type(value.get(k)) not in (int, float) or not math.isfinite(value[k]) for k in keys):
        raise ValueError('finite original service lease required')
    started, hard, original_stop = (value[k] for k in keys)
    if (not started <= now < hard or hard-started != WALL or stop != original_stop or
            not WINDOW <= stop-now <= WALL or stop > hard-600):
        raise ValueError('existing six-hour service lease/deadline differs; no renewal')
    return {key:value[key] for key in keys}


def build_plan(*, service_stop_epoch, authorization_paths, service_lease_path, config_path=r.CONFIG, run_root=RUN_ROOT,
               cohort_path=r.OUTPUT / 'cohort.json', predecessor_phase=r.OUTPUT, g2_phase=G2_PHASE):
    if (not math.isfinite(service_stop_epoch) or not WINDOW <= service_stop_epoch-time.time() <= WALL or not authorization_paths):
        raise ValueError('bounded existing lease and authorization required; no renewal')
    lease = service_lease(service_lease_path, service_stop_epoch, time.time())
    old, cohort, gp, result, bound = predecessor(cohort_path, predecessor_phase, g2_phase)
    bound[str(service_lease_path.resolve())] = file_digest(service_lease_path)
    parent = Path(cohort['slots'][LABEL]['terminal_endpoint'])
    if parent != PARENT or cohort['slots'][LABEL]['endpoint_index_sha256'] != PARENT_INDEX:
        raise ValueError('exact failed 004 parent required')
    proof = context_source(parent)
    immutable = dict(old['immutable_trees'])
    for path in (Path(old['run_root']), Path(gp['run_root']), predecessor_phase, g2_phase):
        immutable[str(path.resolve())] = s.tree_digest(path)
    # All existing phase directories remain read-only, including older retries.
    for path in r.BASE.iterdir():
        if path.is_dir() and path.resolve() != OUTPUT.resolve():
            immutable[str(path.resolve())] = s.tree_digest(path)
    for path in authorization_paths:
        bound[str(path.resolve())] = file_digest(path)
    plan = dict(schema_version=SCHEMA, selected=[LABEL], g2_slots=[], old_http_attempts=OLD_HTTP,
        http_cap=HTTP_CAP, logical_cap=LOGICAL_CAP, aggregate_http_cap=3360, aggregate_authorized_max=634,
        wall_seconds=WALL, request_timeout_seconds=600, send_window_seconds=WINDOW, concurrency=1, attempt_limit_per_slot=1,
        service_stop_epoch=service_stop_epoch, service_lease_origin=lease,
        service_lease_path=str(service_lease_path.resolve()), service_renewal_authorized=False, d0_frozen=False,
        config_path=str(config_path), native_config=r.native_config(proof['config']),
        parent=str(parent), parent_index_sha256=PARENT_INDEX, parent_episode_id=proof['manifest']['episode_id'],
        checkpoint_id=CHECKPOINT, checkpoint_sha256=file_digest(parent / ('checkpoints/'+CHECKPOINT+'.json')),
        resume_context_overflow_id=FAILED, consumed_logical=21, first_new_request_id='r000021',
        context_binding=proof['binding'], frozen_cohort=cohort, frozen_g2=result['g2'], demoted=old['demoted'],
        predecessor_phase=str(predecessor_phase.resolve()), g2_phase=str(g2_phase.resolve()),
        immutable_trees=immutable, bound_files=bound, run_root=str(run_root.resolve()),
        source_fingerprints=source_files(config_path), native_dependency_identity=r.dependency_identity(),
        g1_gate='historical_unverified', g2_gate='undecided', g4_gate='historical_unverified', scientific_efficacy='not_tested')
    validate_contract(seal(plan, 'plan_sha256')); assert_sources(plan); assert_originals(plan)
    assert_write_paths(plan, run_root)
    if run_root.exists():
        raise ValueError('new context recovery root required')
    return seal(plan, 'plan_sha256')


def validate_contract(plan):
    verify_seal(plan, 'plan_sha256')
    expected = dict(schema_version=SCHEMA, selected=[LABEL], g2_slots=[], old_http_attempts=578,
        http_cap=56, logical_cap=14, aggregate_http_cap=3360, aggregate_authorized_max=634,
        wall_seconds=21600, request_timeout_seconds=600, send_window_seconds=645, concurrency=1,
        attempt_limit_per_slot=1, service_renewal_authorized=False, d0_frozen=False,
        parent=str(PARENT), parent_index_sha256=PARENT_INDEX, checkpoint_id=CHECKPOINT,
        resume_context_overflow_id=FAILED, consumed_logical=21, first_new_request_id='r000021')
    if any(plan.get(k) != v for k,v in expected.items()):
        raise ValueError('exact 004 recovery workload/hard limits differ')
    lease_path = Path(plan['service_lease_path'])
    if (service_lease(lease_path, plan['service_stop_epoch'], time.time()) != plan['service_lease_origin'] or
            plan['bound_files'].get(str(lease_path.resolve())) != file_digest(lease_path)):
        raise ValueError('original service lease is not bound to plan')
    proof = context_source(Path(plan['parent']))
    if (plan['native_config'] != r.native_config(proof['config']) or
            plan['context_binding'] != proof['binding'] or
            plan['parent_episode_id'] != proof['manifest']['episode_id'] or
            file_digest(Path(plan['parent']) / ('checkpoints/'+CHECKPOINT+'.json')) != plan['checkpoint_sha256']):
        raise ValueError('context source/config/checkpoint binding differs')
    if (read_json(Path(plan['predecessor_phase']) / 'cohort.json') != plan['frozen_cohort'] or
            read_json(Path(plan['g2_phase']) / 'result.json')['g2'] != plan['frozen_g2']):
        raise ValueError('frozen predecessor cohort/G2 pairs differ')
    verify_seal(plan['frozen_cohort'])
    if (set(plan['frozen_cohort']['slots']) != set(r.SLOTS) or
            plan['frozen_cohort']['slots'][LABEL]['status'] != 'invalid_continuation' or
            any(row['status'] != 'formal' for label,row in plan['frozen_cohort']['slots'].items() if label != LABEL) or
            len(plan['frozen_g2']) != 20 or {row['label'] for row in plan['frozen_g2']} != set(r.G2_SLOTS)):
        raise ValueError('frozen cohort/G2 selection differs')


class GuardedAdmission(r.GuardedAdmission):
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
            raise RuntimeError('insufficient 600-second send window')
        return self.admission.acquire(estimate, timeout)


def validate_child(plan, path):
    from c3_sandbox.l1.evidence.live import verify
    report = verify(path)
    if not report['verified']:
        raise ValueError('recovery child full seal/proof failed: ' + str(report))
    manifest = read_json(path / 'manifest.json'); lineage = manifest.get('lineage') or {}
    expected = dict(parent_episode_id=plan['parent_episode_id'], parent_index_sha256=plan['parent_index_sha256'],
        parent_checkpoint_id=CHECKPOINT, parent_checkpoint_sha256=plan['checkpoint_sha256'], context_overflow=plan['context_binding'])
    if any(lineage.get(k) != v for k,v in expected.items()):
        raise ValueError('recovery checkpoint/context proof lineage differs')
    if read_json(path / 'raw/resolved_config.json') != plan['native_config']:
        raise ValueError('recovery config differs')
    if file_digest(path / 'benchmark/public_task.json') != file_digest(Path(plan['parent']) / 'benchmark/public_task.json'):
        raise ValueError('recovery public task differs')
    used = manifest['usage']['model_requests_including_parent']
    if not 21 <= used <= 35 or manifest['counts']['model_requests'] != used-21:
        raise ValueError('recovery cumulative logical cap exceeded')
    return dict(lineage=lineage, new_logical_requests=used-21, cumulative_logical_requests=used)


def execute(plan, freeze, output):
    validate_contract(plan); validate_freeze(plan, freeze)
    assert_write_paths(plan, output, Path(plan['run_root']))
    assert_sources(plan); assert_originals(plan)
    if not WINDOW <= plan['service_stop_epoch']-time.time() <= WALL:
        raise ValueError('bounded service lifetime exhausted')
    with p.exclusive(output / 'writer.lock'):
        if (output / 'attempt_started.json').exists() or (output / 'result.json').exists() or (output / 'admission.json').exists() or Path(plan['run_root']).exists():
            raise ValueError('one attempt only; restart/overwrite forbidden')
        with (output / 'attempt_started.json').open('x') as handle:
            json.dump(dict(plan_sha256=plan['plan_sha256'], first_new_request_id='r000021', started_epoch=time.time()), handle)
            handle.flush(); os.fsync(handle.fileno())
        started = time.time()
        write_json(output / 'executed_plan.json', plan); write_json(output / 'executed_freeze.json', freeze)
        metadata = {path:file_digest(path) for path in (output / 'executed_plan.json', output / 'executed_freeze.json')}
        admission = p.Admission(output / 'admission.json', plan['plan_sha256'],
            {**p.LIMITS, 'max_requests':HTTP_CAP, 'max_tokens':HTTP_CAP*p.GIB, 'max_wall_seconds':WALL})
        limiter = GuardedAdmission(admission, plan, metadata, started, threading.Event())
        path = Path(plan['run_root']) / 'segments' / LABEL
        cohort = copy.deepcopy(plan['frozen_cohort'])
        error, structural_error, structural = None, None, None
        try:
            c.l1_run(plan['native_config'], path, parent=Path(plan['parent']), checkpoint_id=CHECKPOINT,
                resume_context_overflow_id=FAILED, allow_pi_native_upgrade=True, request_limiter=limiter)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        try:
            structural = validate_child(plan, path)
        except Exception as exc:
            structural_error = f'{type(exc).__name__}: {exc}'
        try:
            package = p.package_record(path)
        except Exception as exc:
            package = dict(path=str(path), valid=False, index_file_sha256=None, package_error=f'{type(exc).__name__}: {exc}')
        receipt = dict(label=LABEL, attempts=1, package=package, error=error, structural_error=structural_error,
            structural_validation=structural, parent=plan['parent'], first_new_request_id='r000021')
        write_json(output / ('receipt_context_'+LABEL+'.json'), seal(receipt))
        old_row = copy.deepcopy(cohort['slots'][LABEL])
        cohort['slots'][LABEL].update(previous_terminal=old_row, continuation_segment=str(path), terminal_endpoint=str(path),
            endpoint_index_sha256=package['index_file_sha256'], status='formal' if package['valid'] and structural else 'invalid_continuation',
            continuation_receipt=str(output / ('receipt_context_'+LABEL+'.json')))
        cohort.pop('artifact_sha256', None)
        cohort.update(plan_sha256=plan['plan_sha256'], predecessor_plan_sha256=plan['frozen_cohort']['plan_sha256'])
        write_json(output / 'cohort.json', seal(cohort))
        stop_reason = structural_error
        try:
            assert_originals(plan); assert_sources(plan)
        except Exception as exc:
            stop_reason = stop_reason or f'{type(exc).__name__}: {exc}'
        valid45 = False
        try:
            valid45 = not stop_reason and all(row['status']=='formal' and p.package_record(Path(row['terminal_endpoint']))['valid'] for row in cohort['slots'].values())
        except Exception as exc:
            stop_reason = stop_reason or f'{type(exc).__name__}: {exc}'
        if not stop_reason:
            try:
                write_json(output / 'score_inputs.json', seal(r.score_inputs(plan, cohort['slots'], plan['frozen_g2'])))
            except Exception as exc:
                stop_reason = f'{type(exc).__name__}: {exc}'
        ledger = admission.snapshot()
        result = seal(dict(plan_sha256=plan['plan_sha256'], recovery=receipt, g2=[], preserved_g2_pairs=20,
            g2_status='completed_predecessor_unchanged', all_45_study_valid=bool(valid45), stop_reason=stop_reason,
            admission=ledger, aggregate_http_attempts=OLD_HTTP+ledger['requests'], aggregate_authorized_max=634,
            active_wall_seconds=time.time()-started, g1_gate='historical_unverified', g2_gate='undecided',
            g4_gate='historical_unverified', d0_frozen=False, scientific_efficacy='not_tested'))
        write_json(output / 'result.json', result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true'); parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--summarize-cpu', type=Path); parser.add_argument('--service-stop-epoch', type=float)
    parser.add_argument('--service-lease', type=Path)
    parser.add_argument('--authorization', type=Path, action='append', default=[])
    parser.add_argument('--config', type=Path, default=r.CONFIG); parser.add_argument('--run-root', type=Path, default=RUN_ROOT)
    parser.add_argument('--cohort', type=Path, default=r.OUTPUT/'cohort.json')
    parser.add_argument('--predecessor-phase', type=Path, default=r.OUTPUT); parser.add_argument('--g2-phase', type=Path, default=G2_PHASE)
    parser.add_argument('--output', type=Path, default=OUTPUT); parser.add_argument('--plan', type=Path); parser.add_argument('--freeze', type=Path)
    args = parser.parse_args(); path = args.plan or args.output/'plan.json'
    if args.summarize_cpu:
        inputs = read_json(args.summarize_cpu); verify_seal(inputs)
        write_json(args.output/'sensitivity_summary.json', seal(r.summarize_cpu(inputs)))
    elif args.execute:
        if args.plan_only or not args.freeze:
            parser.error('--execute requires --freeze')
        print(json.dumps(execute(read_json(path), read_json(args.freeze), args.output)))
    else:
        if args.service_stop_epoch is None or args.service_lease is None:
            parser.error('planning requires existing --service-stop-epoch and --service-lease')
        plan = build_plan(service_stop_epoch=args.service_stop_epoch, authorization_paths=args.authorization,
            service_lease_path=args.service_lease,
            config_path=args.config, run_root=args.run_root, cohort_path=args.cohort,
            predecessor_phase=args.predecessor_phase, g2_phase=args.g2_phase)
        assert_write_paths(plan, path, args.output, args.run_root)
        if path.exists():
            raise ValueError('plan already exists; no overwrite')
        write_json(path, plan)
        print(json.dumps(dict(plan=str(path), plan_sha256=plan['plan_sha256'], max_new_http=56, max_new_logical=14)))


if __name__ == '__main__':
    main()
