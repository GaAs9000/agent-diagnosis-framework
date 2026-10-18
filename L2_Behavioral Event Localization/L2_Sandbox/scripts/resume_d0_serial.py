#!/usr/bin/env python3
"""Separately sealed, serial continuation of the remaining DEV and original G2 slots.

Planning only reads evidence. Execution requires a root-reviewed phase freeze;
neither this phase nor its predecessor constitutes a method-D0 freeze.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import importlib.util
import json
import math
from pathlib import Path
import shutil
import time

SCRIPT = Path(__file__).resolve()
spec = importlib.util.spec_from_file_location('d0_prior_resume_helpers', SCRIPT.with_name('resume_d0_prerequisites.py'))
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
c = prior.collector
read_json, write_json = prior.read_json, prior.write_json
seal, verify_seal, file_digest = prior.seal, prior.verify_seal, prior.file_digest
OUTPUT = prior.L2 / 'validation/d0_20261018/serial_resume'
OLD = prior.OUTPUT
REPLACEMENTS = ('officeval_010.r00', 'officeval_050.r04')
OLD_HTTP = 93
OLD_ACTIVE_SECONDS = 2058.032536
MAX_NEW_WALL = math.floor(21600 - OLD_ACTIVE_SECONDS)
SERVICE_STOP = datetime.fromisoformat('2026-10-18T17:08:14+08:00').timestamp() - 300


def old_bindings(directory):
    paths = {name: directory / name for name in ('executed_plan.json', 'executed_freeze.json', 'result.json', 'admission.json')}
    docs = {name: read_json(path) for name, path in paths.items()}
    verify_seal(docs['executed_plan.json'], 'plan_sha256')
    verify_seal(docs['result.json'])
    verify_seal(docs['admission.json'])
    old_plan, result, ledger = docs['executed_plan.json'], docs['result.json'], docs['admission.json']
    if (result['plan_sha256'] != old_plan['plan_sha256'] or
            docs['executed_freeze.json']['approved_plan_sha256'] != old_plan['plan_sha256'] or
            ledger['plan_sha256'] != old_plan['plan_sha256'] or
            result['admission']['requests'] != OLD_HTTP or ledger['requests'] != OLD_HTTP or
            result.get('g2') or result.get('all_45_study_valid') is not False):
        raise ValueError('unexpected predecessor identity, budget, or completion state')
    if result['admission'] != {k: v for k, v in ledger.items() if k != 'artifact_sha256'}:
        raise ValueError('predecessor result and final ledger differ')
    return old_plan, {str(path): file_digest(path) for path in paths.values()}


def assert_sources(plan):
    for relative, expected in plan['source_fingerprints'].items():
        if file_digest(prior.ROOT / relative) != expected:
            raise ValueError('frozen original or serial execution source changed')
    for path, expected in plan['predecessor_files'].items():
        if file_digest(Path(path)) != expected:
            raise ValueError('predecessor evidence changed')


def tree_digest(path):
    return prior.digest({str(p.relative_to(path)): file_digest(p) for p in path.rglob('*') if p.is_file()})


def identity(path):
    return dict(package=prior.package_record(path), tree_sha256=tree_digest(path))


def assert_parents(records):
    for label, before in records.items():
        if identity(Path(before['package']['path'])) != before:
            raise ValueError(f'parent evidence changed: {label}')


def build_plan(old_directory=OLD):
    old, bindings = old_bindings(old_directory)
    sources = dict(old['source_fingerprints'])
    sources[str(SCRIPT.relative_to(prior.ROOT))] = file_digest(SCRIPT)
    plan = dict(schema_version='d0-serial-continuation-1', d0_frozen=False,
        phase='remaining_dev2_then_original_g2', execution_concurrency=3,
        replacement_concurrency=1, g2_concurrency=3,
        source_fingerprints=sources, predecessor_files=bindings,
        predecessor_plan_sha256=old['plan_sha256'], old_http_attempts=OLD_HTTP,
        old_supervisor_active_seconds=OLD_ACTIVE_SECONDS,
        old_active_seconds_basis='prior supervisor measured active interval; root-reviewed operational receipt',
        aggregate_http_cap=3360, aggregate_active_wall_cap=21600,
        service_stop_epoch=SERVICE_STOP, service_safety_margin_seconds=300,
        pricing_mode=old['pricing_mode'], actor_identity=old['actor_identity'],
        frozen_base_config=old['frozen_base_config'], frozen_parent_configs=old['frozen_parent_configs'],
        replacements=list(REPLACEMENTS), attempt_limit_per_slot=1,
        g2_slots=list(old['g2_slots']), g2_checkpoint='cp000000',
        g2_purpose=old['g2_purpose'], scientific_efficacy='not_tested')
    assert_sources(plan)
    plan['parents'] = {label: identity(Path(row['path'])) for label, row in old['parents'].items()}
    invalid = sorted(label for label, row in plan['parents'].items() if not row['package']['valid'])
    if invalid != list(REPLACEMENTS) or len(plan['parents']) != 45:
        raise ValueError('expected precisely 43 valid DEV parents and the frozen two invalid slots')
    for label in plan['g2_slots']:
        if (c.OUT_ROOT / 'v3_g2_c0_resumes' / f'{label}_c0').exists():
            raise ValueError('unexpected prior G2 directory; review rather than overwrite')
    plan['execution_ready'] = True
    plan['limits'] = {**prior.LIMITS, 'max_requests': 3080,
        'max_tokens': 3080 * prior.GIB, 'max_wall_seconds': MAX_NEW_WALL,
        'max_invalid_attempts': 22}
    plan['budget_and_stop_conditions'] = dict(hard_actor_http_attempts=3080,
        max_wall_seconds=MAX_NEW_WALL, proof='(2 + 20) * 35 * 4 = 3080; old 93 + new <= 3173 <= 3360')
    return seal(plan, 'plan_sha256')


class GuardedAdmission:
    def __init__(self, admission, plan, metadata):
        self.admission, self.plan, self.metadata = admission, plan, metadata

    def remaining_seconds(self):
        return max(0, min(self.admission.remaining_seconds(), self.plan['service_stop_epoch'] - time.time()))

    def acquire(self, estimate, timeout=None):
        assert_sources(self.plan)
        for path, expected in self.metadata.items():
            if file_digest(path) != expected:
                raise ValueError('serial execution metadata changed')
        if self.remaining_seconds() <= 0:
            raise RuntimeError('serial active-wall or existing-service deadline reached')
        if self.plan['old_http_attempts'] + self.admission.snapshot()['requests'] >= self.plan['aggregate_http_cap']:
            raise RuntimeError('aggregate HTTP cap reached')
        return self.admission.acquire(estimate, timeout)

    def reconcile(self, estimate, actual):
        return self.admission.reconcile(estimate, actual)


def execute(plan, freeze, output):
    verify_seal(plan, 'plan_sha256')
    if (freeze.get('phase_authorized') is not True or freeze.get('phase_frozen') is not True or
            freeze.get('d0_frozen') is not False or freeze.get('internal_plan_reviewed') is not True):
        raise ValueError('reviewed serial phase authorization required; D0 remains unfrozen')
    auth = freeze.get('user_authorization_record') or {}
    if not auth.get('exact_text') or not auth.get('source'):
        raise ValueError('direct human authorization text and source required')
    caps = prior.approve(plan, {**freeze, 'd0_frozen': True})
    if (caps != plan['limits'] or caps['max_requests'] != 3080 or
            caps['max_wall_seconds'] != MAX_NEW_WALL or plan['execution_concurrency'] != 3 or
            plan.get('replacement_concurrency') != 1 or plan.get('g2_concurrency') != 3 or
            plan['replacements'] != list(REPLACEMENTS) or plan['old_http_attempts'] != OLD_HTTP or
            plan['aggregate_http_cap'] != 3360 or plan['service_stop_epoch'] != SERVICE_STOP):
        raise ValueError('exact serial workload and non-renewable limits required')
    assert_sources(plan)
    with prior.exclusive(output / 'writer.lock'):
        result_path = output / 'result.json'
        if result_path.exists():
            result = read_json(result_path)
            verify_seal(result)
            if result['plan_sha256'] != plan['plan_sha256']:
                raise ValueError('existing serial result belongs to another plan')
            return result
        if (output / 'admission.json').exists():
            raise ValueError('interrupted serial ledger requires review; automatic restart forbidden')
        assert_parents(plan['parents'])
        if time.time() >= plan['service_stop_epoch']:
            raise ValueError('existing service deadline reached; no implicit renewal')
        write_json(output / 'executed_plan.json', plan)
        write_json(output / 'executed_freeze.json', freeze)
        admission = prior.Admission(output / 'admission.json', plan['plan_sha256'], caps)
        # Persist even zero-send execution so a crash cannot reset its wall clock.
        admission.save()
        metadata = {p: file_digest(p) for p in (output / 'executed_plan.json', output / 'executed_freeze.json')}
        limiter = GuardedAdmission(admission, plan, metadata)
        protected = {k: v for k, v in plan['parents'].items() if v['package']['valid']}
        replacements, g2 = [], []
        stop_reason = None

        def run_slot(label, cold=False):
            if limiter.remaining_seconds() <= 0:
                raise RuntimeError('serial active-wall or existing-service deadline reached')
            admission.check_new_request()
            assert_sources(plan)
            parent = c.OUT_ROOT / 'episodes' / label
            path = c.OUT_ROOT / 'v3_g2_c0_resumes' / f'{label}_c0' if cold else parent
            marker = output / f'attempt_{label}{"_c0" if cold else ""}.json'
            if marker.exists() or (cold and path.exists()):
                raise ValueError('serial single attempt already exists')
            config = copy.deepcopy(plan['frozen_base_config'])
            config['benchmark']['task_id'] = label.rsplit('.', 1)[0]
            parent_before = identity(parent) if cold else None
            if cold:
                config = read_json(parent / 'raw/resolved_config.json')
                prior.assert_config(config, plan['frozen_parent_configs'][label])
            prior.assert_config(config, config)
            write_json(marker, seal(dict(label=label, cold=cold, attempt=1, parent=parent_before)))
            original_tree = tree_digest(path) if path.exists() else None
            mapping = prior.archive(path, label, output) if not cold else None
            if mapping and tree_digest(Path(mapping['archived_path'])) != original_tree:
                raise ValueError('prior failed package archive differs')
            error = None
            try:
                kwargs = dict(request_limiter=limiter)
                if cold:
                    kwargs.update(parent=parent, checkpoint_id='cp000000')
                c.l1_run(config, path, **kwargs)
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
            record = prior.package_record(path)
            if not record['valid']:
                admission.invalid()
            row = dict(label=label, package=record, attempts=1, error=error, archive=mapping)
            if cold:
                assert_parents({label: parent_before})
                first = 'raw/model/r000000/request.json'
                row['byte_identical_first_request'] = ((path / first).is_file() and
                    file_digest(parent / first) == file_digest(path / first))
                manifest = read_json(path / 'manifest.json') if (path / 'manifest.json').exists() else {}
                lineage = manifest.get('lineage') or {}
                info = parent_before['package']['info']
                row['lineage'] = lineage
                row['cold_parent_binding_valid'] = (lineage.get('parent_episode_id') == info.get('episode_id') and
                    lineage.get('parent_index_sha256') == info.get('index_sha256') and
                    lineage.get('parent_checkpoint_id') == 'cp000000' and
                    lineage.get('parent_checkpoint_sha256') == file_digest(parent / 'checkpoints/cp000000.json'))
                row['parent'] = parent_before
            write_json(output / f'receipt_{label}{"_c0" if cold else ""}.json', seal(row))
            return row

        valid45 = False
        try:
            for label in plan['replacements']:
                replacements.append(run_slot(label))
                assert_parents(protected)
            valid45 = all(prior.package_record(Path(row['package']['path']))['valid'] for row in plan['parents'].values())
            if valid45:
                all_parents = {k: identity(Path(v['package']['path'])) for k, v in plan['parents'].items()}
                # Preserve the original G2 runtime condition: three workers.
                # Each worker retains its own receipt; admission is shared.
                with ThreadPoolExecutor(max_workers=3) as pool:
                    futures = [pool.submit(run_slot, label, True) for label in plan['g2_slots']]
                    for future in futures:
                        try:
                            row = future.result()
                            g2.append(row)
                            if not row['byte_identical_first_request'] or not row['cold_parent_binding_valid']:
                                raise ValueError('cold replay request or checkpoint lineage mismatch; retained for review')
                        except Exception as exc:
                            stop_reason = stop_reason or f'{type(exc).__name__}: {exc}'
                            admission.close(stop_reason)
                assert_parents(all_parents)
        except Exception as exc:
            stop_reason = f'{type(exc).__name__}: {exc}'
            admission.close(stop_reason)
        finally:
            assert_parents(protected)
        completed = len(g2) == 20 and all(r['package']['valid'] and r['byte_identical_first_request'] and r['cold_parent_binding_valid'] for r in g2)
        result = seal(dict(plan_sha256=plan['plan_sha256'], replacements=replacements,
            all_45_study_valid=valid45, g2=g2, g2_status='completed' if completed else 'incomplete',
            stop_reason=stop_reason, admission=admission.snapshot(),
            aggregate_http_attempts=OLD_HTTP + admission.snapshot()['requests'],
            active_wall_seconds=OLD_ACTIVE_SECONDS + time.time() - admission.state['started_at'],
            scientific_efficacy='not_tested', d0_frozen=False))
        write_json(result_path, result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--old-output', type=Path, default=OLD)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--freeze', type=Path)
    args = parser.parse_args()
    path = args.plan or args.output / 'plan.json'
    if args.execute:
        if args.plan_only or not args.freeze:
            parser.error('--execute requires --freeze and cannot combine with --plan-only')
        result = execute(read_json(path), read_json(args.freeze), args.output)
        print(json.dumps(dict(g2_status=result['g2_status'], requests=result['admission']['requests'])))
    else:
        plan = build_plan(args.old_output)
        write_json(path, plan)
        print(json.dumps(dict(plan=str(path), plan_sha256=plan['plan_sha256'], valid_parents=43,
            replacement_concurrency=1, g2_concurrency=3, max_new_http=3080, max_new_active_seconds=MAX_NEW_WALL)))


if __name__ == '__main__':
    main()
