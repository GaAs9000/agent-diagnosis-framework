#!/usr/bin/env python3
"""Bounded, separately approved DEV45 repair and original G2 continuation.

Default planning performs no inference. Historical packages are moved intact;
canonical diagnoses/reports/progress are never rewritten by this driver.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
L1 = ROOT / 'L1_sandbox/C3_Experimental_Sandbox'
L2 = ROOT / 'L2_Behavioral Event Localization/L2_Sandbox'
sys.path[:0] = [str(L1 / 'src'), str(L2 / 'src')]
from l2_sandbox.v3_execution import (Admission, approve, digest, exclusive,
    file_digest, read_json, seal, verify_seal, write_json)

spec = importlib.util.spec_from_file_location('d0_original_collector', ROOT / 'scripts/run_v3_stage_d_dev45_and_g2.py')
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)

OUTPUT = L2 / 'validation/d0_20261018/prerequisite_resume'
REPLACEMENTS = ('officeval_004.r02', 'officeval_010.r00', 'officeval_010.r04', 'officeval_050.r04')
GIB = 1024 ** 3
LIMITS = dict(max_requests=3360, max_tokens=3360 * GIB, max_cost_usd=0,
              max_wall_seconds=21600, max_invalid_attempts=24,
              max_tokens_per_request=GIB, usd_per_token_upper_bound=0)


def source_fingerprints():
    paths = [Path(__file__), Path(collector.__file__), collector.CONFIG_PATH]
    paths += sorted((L1 / 'src').rglob('*.py'))
    paths += [L2 / 'src/l2_sandbox/v3_execution.py']
    return {str(p.relative_to(ROOT)): file_digest(p) for p in paths}


def assert_sources(plan):
    if source_fingerprints() != plan['source_fingerprints']:
        raise ValueError('frozen execution source/config changed')


def actor_identity(config):
    model, agent = config['model'], config['agent']
    if model.get('kind') != 'local' or model.get('provider') != 'vllm-local':
        raise ValueError('local provider required for zero API price')
    return dict(provider='local', model=model['model'], harness=agent['harness'], endpoint=model['base_url'])


def assert_config(config, frozen):
    if config != frozen:
        raise ValueError('configuration differs from frozen task configuration')
    if (actor_identity(config) != dict(provider='local', model='Qwen/Qwen3.6-27B',
            harness='pi', endpoint='http://127.0.0.1:8001/v1')
            or config['model']['parameters']['temperature'] != 1
            or config['model']['parameters']['max_tokens'] != 8192
            or 'seed' in config['model']['parameters']
            or config['model']['timeout_seconds'] != 300
            or config['model']['retry']['max_attempts'] != 4
            or config['budget']['max_model_requests'] != 35):
        raise ValueError('frozen Actor/budget contract mismatch')


def package_record(path):
    valid, info = collector.is_valid_episode(path)
    return dict(path=str(path), valid=valid, info=info,
                manifest_sha256=file_digest(path / 'manifest.json') if (path / 'manifest.json').exists() else None,
                index_file_sha256=file_digest(path / 'evidence_index.json') if (path / 'evidence_index.json').exists() else None)


def build_plan():
    config = collector.resolve_config(collector.CONFIG_PATH)
    assert_config(config, config)
    parents = {}
    parent_configs = {}
    for rep in collector.REPLICATES:
        for task in collector.DEV_TASKS:
            label = f'{task}.{rep}'
            parents[label] = package_record(collector.OUT_ROOT / 'episodes' / label)
            parent_config = read_json(Path(parents[label]['path']) / 'raw/resolved_config.json')
            expected = copy.deepcopy(config)
            expected['benchmark']['task_id'] = task
            if 'system_prompt' in parent_config['agent']:
                expected['agent']['system_prompt'] = parent_config['agent']['system_prompt']
            assert_config(parent_config, expected)
            parent_configs[label] = parent_config
    invalid = sorted(label for label, row in parents.items() if not row['valid'])
    ready = set(invalid).issubset(REPLACEMENTS)
    return seal(dict(schema_version='d0-prerequisite-resume-1', execution_ready=ready,
        d0_frozen=False, phase='previously_authorized_dev45_g2_prerequisites',
        execution_concurrency=3, pricing_mode='local_no_provider_charge',
        actor_identity=actor_identity(config), frozen_base_config=config,
        source_fingerprints=source_fingerprints(), parents=parents, frozen_parent_configs=parent_configs,
        replacements=list(REPLACEMENTS), attempt_limit_per_slot=1,
        g2_slots=[f'{t}.{r}' for t, r in collector.G2_SLOTS],
        g2_checkpoint='cp000000', g2_requires_all_45_study_valid=True,
        budget_and_stop_conditions=dict(hard_actor_http_attempts=3360,
            proof='4 * 1 * 35 * 4 + 20 * 1 * 35 * 4 = 3360', max_wall_seconds=21600),
        limits=LIMITS, reservation_unit='encoded-byte estimate upper bound; not measured tokens',
        old_parent_controller_hash='not recorded; current hashes do not retroactively qualify old parents',
        scientific_efficacy='not_tested', g2_purpose='previously authorized engineering and variance pilot; n20 cannot certify family Hoeffding bound within +/-0.05; no equivalence PASS computed', canonical_e1='skipped; existing derived outputs unchanged'), 'plan_sha256')


class GuardedAdmission:
    """Check frozen sources before every HTTP attempt, including transport retry."""
    def __init__(self, admission, plan, metadata=None):
        self.admission, self.plan = admission, plan
        self.metadata = metadata or {}

    def acquire(self, estimate, timeout=None):
        assert_sources(self.plan)
        for path, expected in self.metadata.items():
            if file_digest(path) != expected:
                raise ValueError('frozen execution metadata changed')
        return self.admission.acquire(estimate, timeout)

    def reconcile(self, estimate, actual):
        return self.admission.reconcile(estimate, actual)

    def remaining_seconds(self):
        return self.admission.remaining_seconds()


def assert_unchanged(records):
    for label, before in records.items():
        if package_record(Path(before['path'])) != before:
            raise ValueError(f'parent evidence changed: {label}')


def archive(path, label, output):
    if not path.exists():
        return None
    target = collector.OUT_ROOT / 'failed_attempts' / f'{label}.d0_resume_{time.time_ns()}'
    old = package_record(path)
    tree = {str(p.relative_to(path)): file_digest(p) for p in path.rglob('*') if p.is_file()}
    mapping = dict(old=old, archived_path=str(target), new_path=str(path), tree_sha256=digest(tree))
    write_json(output / f'archive_{label}.json', seal(mapping))
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(target))
    if {str(p.relative_to(target)): file_digest(p) for p in target.rglob('*') if p.is_file()} != tree:
        raise ValueError('archive package changed')
    return mapping


def cumulative_usage():
    rows = []
    for path in sorted(collector.OUT_ROOT.rglob('manifest.json')):
        doc = read_json(path)
        rows.append(dict(path=str(path), episode_id=doc.get('episode_id'),
            model_requests=doc.get('counts', {}).get('model_requests', 0),
            recorded_http_attempt_directories=sum(1 for p in (path.parent / 'raw/model').glob('r*/attempt_*') if p.is_dir())))
    return dict(packages=rows, model_requests=sum(row['model_requests'] for row in rows),
                recorded_http_attempt_directories=sum(row['recorded_http_attempt_directories'] for row in rows),
                note='manifest model requests and retained HTTP attempt directories include archived history; incomplete historical sends unknown; new HTTP attempts strictly charged by shared admission')


def execute(plan, freeze, output):
    verify_seal(plan, 'plan_sha256')
    if freeze.get('phase_authorized') is not True or freeze.get('phase_frozen') is not True or freeze.get('d0_frozen') is not False:
        raise ValueError('prerequisite phase authorization required; method D0 remains unfrozen')
    # Generic admission uses a historical field name. This transient adapter is
    # never persisted as the actual method-D0 approval.
    caps = approve(plan, {**freeze, 'd0_frozen': True})
    if caps != LIMITS or plan['limits'] != LIMITS:
        raise ValueError('exact prerequisite limits required')
    authorization = freeze.get('user_authorization_record')
    if not isinstance(authorization, dict) or not authorization.get('exact_text') or not authorization.get('source'):
        raise ValueError('exact user authorization text and source required')
    if freeze.get('internal_plan_reviewed') is not True:
        raise ValueError('root internal plan review required')
    assert_sources(plan)
    with exclusive(output / 'writer.lock'):
        receipt_path = output / 'result.json'
        if receipt_path.exists():
            result = read_json(receipt_path)
            verify_seal(result)
            if result['plan_sha256'] != plan['plan_sha256']:
                raise ValueError('existing result belongs to another plan')
            return result
        ledger_path = output / 'admission.json'
        if ledger_path.exists():
            raise ValueError('interrupted execution retained; review ledger before any restart')
        for label, before in plan['parents'].items():
            if package_record(Path(before['path'])) != before:
                raise ValueError(f'parent changed since planning: {label}')
        write_json(output / 'executed_plan.json', plan)
        write_json(output / 'executed_freeze.json', freeze)
        if collector.PROGRESS_JSON.exists():
            shutil.copy2(collector.PROGRESS_JSON, output / 'progress_before.json')
        write_json(output / 'usage_before.json', cumulative_usage())
        admission = Admission(ledger_path, plan['plan_sha256'], caps)
        metadata = {p: file_digest(p) for p in (output / 'executed_plan.json', output / 'executed_freeze.json')}
        limiter = GuardedAdmission(admission, plan, metadata)
        protected = {k: v for k, v in plan['parents'].items() if v['valid']}

        def run_slot(label, g2=False):
            parent = collector.OUT_ROOT / 'episodes' / label
            path = collector.OUT_ROOT / 'v3_g2_c0_resumes' / f'{label}_c0' if g2 else parent
            valid, _ = collector.is_valid_episode(path)
            if valid:
                if g2:
                    raise ValueError('unexpected existing valid G2; fresh cold binding review required')
                return dict(label=label, cached=True, package=package_record(path))
            marker = output / f'attempt_{label}{"_c0" if g2 else ""}.json'
            if marker.exists():
                raise ValueError('one authorized attempt already started')
            assert_sources(plan)
            admission.check_new_request()
            config = copy.deepcopy(plan['frozen_base_config'])
            config['benchmark']['task_id'] = label.rsplit('.', 1)[0]
            if g2:
                parent_before = package_record(parent)
                config_parent = read_json(parent / 'raw/resolved_config.json')
                assert_config(config_parent, plan['frozen_parent_configs'][label])
                config = config_parent
            assert_config(config, config)
            write_json(marker, seal(dict(label=label, g2=g2, attempt=1,
                parent=parent_before if g2 else None)))
            mapping = archive(path, label + ('_c0' if g2 else ''), output)
            error = None
            try:
                kwargs = dict(request_limiter=limiter)
                if g2:
                    kwargs.update(parent=parent, checkpoint_id='cp000000')
                collector.l1_run(config, path, **kwargs)
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
            record = package_record(path)
            if not record['valid']:
                admission.invalid()
            row = dict(label=label, cached=False, attempts=1, package=record, archive=mapping, error=error)
            if g2:
                assert_unchanged({label: parent_before})
                first = 'raw/model/r000000/request.json'
                row['parent'] = parent_before
                row['checkpoint_id'] = 'cp000000'
                row['byte_identical_first_request'] = ((path / first).exists() and
                    file_digest(parent / first) == file_digest(path / first))
                manifest = read_json(path / 'manifest.json') if (path / 'manifest.json').exists() else {}
                row['lineage'] = manifest.get('lineage')
                lineage = manifest.get('lineage') or {}
                row['cold_parent_binding_valid'] = (lineage.get('parent_episode_id') == parent_before['info'].get('episode_id')
                    and lineage.get('parent_index_sha256') == parent_before['info'].get('index_sha256')
                    and lineage.get('parent_checkpoint_id') == 'cp000000'
                    and lineage.get('parent_checkpoint_sha256') == file_digest(parent / 'checkpoints/cp000000.json'))
            write_json(output / f'receipt_{label}{"_c0" if g2 else ""}.json', seal(row))
            return row

        with ThreadPoolExecutor(max_workers=3) as pool:
            replacements = list(pool.map(run_slot, REPLACEMENTS))
        assert_unchanged(protected)
        valid45 = all(collector.is_valid_episode(collector.OUT_ROOT / 'episodes' / f'{t}.{r}')[0]
            for r in collector.REPLICATES for t in collector.DEV_TASKS)
        g2 = []
        if valid45:
            all_parents = {f'{t}.{r}': package_record(collector.OUT_ROOT / 'episodes' / f'{t}.{r}')
                for r in collector.REPLICATES for t in collector.DEV_TASKS}
            with ThreadPoolExecutor(max_workers=3) as pool:
                g2 = list(pool.map(lambda label: run_slot(label, True), plan['g2_slots']))
            assert_unchanged(all_parents)
        assert_unchanged(protected)
        result = seal(dict(plan_sha256=plan['plan_sha256'], replacements=replacements,
            all_45_study_valid=valid45, g2=g2, g2_status='blocked_partial_dev45' if not valid45 else
            ('completed' if all(r['package']['valid'] and r['cold_parent_binding_valid'] and r['byte_identical_first_request'] for r in g2) else 'partial_invalid'),
            admission=admission.snapshot(), usage_after=cumulative_usage(), scientific_efficacy='not_tested'))
        if collector.PROGRESS_JSON.exists():
            shutil.copy2(collector.PROGRESS_JSON, output / 'progress_after.json')
        write_json(receipt_path, result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--plan', type=Path, default=OUTPUT / 'plan.json')
    parser.add_argument('--freeze', type=Path)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.execute:
        if args.plan_only or args.freeze is None:
            parser.error('--execute requires --freeze and cannot combine with --plan-only')
        result = execute(read_json(args.plan), read_json(args.freeze), args.output)
        print(json.dumps(dict(g2_status=result['g2_status'], requests=result['admission']['requests'])))
    else:
        plan = build_plan()
        write_json(args.plan, plan)
        print(json.dumps(dict(plan=str(args.plan), plan_sha256=plan['plan_sha256'], execution_ready=plan['execution_ready'], valid_parents=sum(v['valid'] for v in plan['parents'].values()), max_http_attempts=3360)))


if __name__ == '__main__':
    main()
