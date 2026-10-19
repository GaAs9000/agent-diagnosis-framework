#!/usr/bin/env python3
"""Offline prospective G4: immutable precollection proof plus exact live receipts.

Runs only on Linux CPU. No Actor, grader, container, or service is launched.
Original path strings stay intact; --root-map OLD=RETAINED changes file I/O only.
The claim assumes the ordinary trusted runner, OS, Docker and HTTPS source.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PosixPath
import platform
import subprocess
import sys
import tarfile
import tempfile
import time

DEV = tuple('officeval_' + n for n in ('003', '004', '010', '033', '050', '060', '070', '085', '090'))
L2_REL = 'L2_Behavioral Event Localization/L2_Sandbox'
L1_REL = 'L1_sandbox/C3_Experimental_Sandbox'


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Portable:
    """Read retained files without rewriting signed/hash-bound JSON path fields."""
    def __init__(self, mappings):
        self.maps = sorted(((str(Path(a)), Path(b)) for a, b in mappings), key=lambda x: -len(x[0]))
        require(len({a for a, _ in self.maps}) == len(self.maps), 'duplicate root mapping')
        require(all(Path(a).is_absolute() and b.is_absolute() for a, b in self.maps), 'absolute roots required')

    def path(self, value):
        p = Path(value)
        require(p.is_absolute() and '..' not in p.parts, 'absolute confined evidence path required')
        for old, new in self.maps:
            if p == Path(old) or p.is_relative_to(old):
                p = new / p.relative_to(old)
                break
        require(not p.is_symlink() and not any(q.is_symlink() for q in p.parents), 'aliased evidence path')
        return p

    def ref(self, reference):
        p = self.path(reference['path'])
        require(sha(p) == reference['sha256'], 'external reference hash differs: ' + reference['path'])
        return p

    def path_class(self):
        portable = self
        class RetainedPath(PosixPath):
            # str() remains the original identity; filesystem functions use this.
            def __fspath__(self):
                return str(portable.path(str(self))) if self.is_absolute() else str(self)
        return RetainedPath


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def extract_controller(archive, controller, destination):
    expected = controller['source_files']
    with tarfile.open(archive) as stream:
        members = stream.getmembers()
        require(len({m.name for m in members}) == len(members), 'duplicate controller archive member')
        require({m.name for m in members} == set(expected), 'controller archive inventory differs')
        for member in members:
            relative = Path(member.name)
            require(member.isfile() and not relative.is_absolute() and '..' not in relative.parts,
                    'unsafe controller source member')
            content = stream.extractfile(member).read()
            require(hashlib.sha256(content).hexdigest() == expected[member.name], 'archived source hash differs')
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    require(hashlib.sha256(canonical(expected)).hexdigest() == controller['source_manifest_sha256'],
            'controller source manifest differs')


def preparation(frozen, portable, archive_root):
    """Replay original pure verifiers; never compare mutable present source files."""
    mapped = portable.path_class()
    observer = module('g4_frozen_observer', archive_root / L2_REL / 'scripts/prospective_execution_observer.py')
    builder = module('g4_frozen_builder', archive_root / L2_REL / 'scripts/build_prospective_provenance.py')
    observer.Path = builder.Path = mapped
    for name, expected in frozen['bound_files'].items():
        require(sha(portable.path(name)) == expected, 'precollection bound file differs: ' + name)
    candidate_path = mapped(frozen['candidate']['path'])
    candidate = observer.read(candidate_path)
    graph = observer.read(mapped(frozen['verified_artifacts']['path']))
    base = mapped(graph['evidence_root'])
    # Recompute image archive inventories and controller archive, not cached booleans.
    proof = builder.verify_candidate(candidate_path, base, mapped(graph['external_anchor']['path']),
                                     mapped(graph['reviewed_allowlist']['path']), time.monotonic() + 3600)
    require(proof['preparation_links_validated'] is True, 'preparation replay failed')
    # validate_artifact_graph otherwise reloads a builder with unmapped filesystem I/O.
    # Capture validation has just run above with original bytes and mapped I/O.
    for ref in candidate['stages']:
        path = base / ref['path']
        stage = observer.read(path)
        if stage['kind'] == 'image_capture':
            observer._CAPTURE_CACHE.add((str(path), observer.sha(path), observer.digest(stage['files'])))
    observer.validate_artifact_graph(frozen, candidate_path, candidate,
                                     mapped(frozen['provenance_verification']['path']))
    controller_stage = next(base / r['path'] for r in candidate['stages'] if r['kind'] == 'controller_and_inputs')
    public = observer.read(controller_stage.parent / 'public_inputs.json')
    require(frozen['schema'] == 'prospective-live-freeze-v1' and frozen['phase_authorized'] is True
            and frozen['internal_reviewed'] is True and frozen['provenance_candidate_verified'] is True
            and frozen['d0_frozen'] is False, 'prospective freeze state differs')
    require(set(frozen['task_allowed']) == set(DEV) and len(frozen['task_allowed']) == 9
            and set(frozen['task_config_sha256']) == set(DEV), 'DEV9 freeze differs')
    image = observer.read(mapped(frozen['actor_image_inspect']['path']))
    require(image['Id'] == frozen['actor_image_id'] and image['Config']['Env'] == frozen['actor_image_env'],
            'Actor image environment differs')
    return observer, public


def package_ref(root, reference):
    relative = Path(reference['path'])
    require(not relative.is_absolute() and '..' not in relative.parts, 'package reference escapes root')
    p = root / relative
    require(not p.is_symlink() and not any(q.is_symlink() for q in p.parents), 'aliased package reference')
    require(sha(p) == reference['sha256'], 'package reference hash differs')
    return p


def receipt_binding(frozen, freeze_sha, row, events, root, external, portable, observer, seen):
    """Exact external/internal receipt sets, identities, and every prestart boundary."""
    eid, task, config_sha = row['episode_id'], row['task_id'], row['config_sha256']
    anchors = [e for e in events if e['kind'] == 'provenance.bound']
    require(len(anchors) == 1, 'exactly one provenance anchor required')
    anchor = read(package_ref(root, anchors[0]['data']['anchor']))
    external_binding = portable.ref(anchor['external_receipt'])
    require(external_binding == external / 'episode_binding.json', 'foreign episode binding receipt')
    binding = read(external_binding)
    require(anchor == {**binding, 'external_receipt': anchor['external_receipt']}, 'internal/external anchor differs')
    require(binding['episode_id'] == eid and binding['task_id'] == task and binding['config_sha256'] == config_sha
            and binding['freeze']['sha256'] == freeze_sha
            and portable.path(binding['freeze']['path']) == portable.path(frozen['_freeze_path'])
            and portable.path(binding['external_dir']) == external
            and binding['workspace'] == str(Path(row['path']).parent / ('.l1-work-' + eid)), 'episode anchor identity differs')
    require(anchors[0]['seq'] < min(e['seq'] for e in events if e['kind'] == 'runtime.container.created'),
            'episode binding recorded after container creation')
    native = read(external / 'native_identity.json')
    require(native['schema'] == 'prospective-native-installed-v1' and native['episode_id'] == eid
            and native['config_sha256'] == config_sha and native['freeze_sha256'] == freeze_sha
            and native['before_container_or_actor_send'] is True
            and native['dependency_identity'] == frozen['native_execution']['dependency_identity'],
            'native installed identity differs')
    manifest = read(root / 'manifest.json')
    require(manifest['agent']['native_validation'] == native['dependency_identity'], 'manifest native identity differs')
    created = [e for e in events if e['kind'] == 'runtime.container.created']
    inspected = [e for e in events if e['kind'] == 'runtime.container.inspected']
    cids = [e['data']['container_id'] for e in created]
    require(len(cids) == len(set(cids)) and len(inspected) == len(created)
            and {e['data']['container_id'] for e in inspected} == set(cids), 'container receipt coverage differs')
    expected_names = {'episode_binding.json', 'native_identity.json'} | {cid + '.json' for cid in cids}
    require({p.name for p in external.iterdir()} == expected_names
            and all(p.is_file() and not p.is_symlink() for p in external.iterdir()), 'foreign/missing external receipt')
    internal = root / 'runtime/prospective/containers'
    require({p.name for p in internal.iterdir()} == {cid + '.json' for cid in cids}, 'foreign/missing internal inspection')
    checker = object.__new__(observer.ProspectiveExecutionObserver)
    checker.frozen, checker.workspace = frozen, Path(binding['workspace'])
    probes = 0
    for creation in created:
        cid = creation['data']['container_id']
        require(cid not in seen, 'container shared by foreign episode')
        seen.add(cid)
        event = next(e for e in inspected if e['data']['container_id'] == cid)
        require(creation['seq'] < event['seq'] and creation['data']['image_id'] == frozen['actor_image_id'],
                'container creation/inspection ordering or image differs')
        value = read(package_ref(root, event['data']['inspection']))
        ref = {k: value['receipt'][k] for k in ('path', 'sha256')}
        receipt_path = portable.ref(ref)
        require(receipt_path == external / (cid + '.json'), 'foreign container receipt')
        receipt = read(receipt_path)
        require(value['container_id'] == cid and value['inspection'] == receipt['inspection']
                and value['receipt'] == {**ref, 'episode_id': eid, 'container_id': cid, 'freeze_sha256': freeze_sha},
                'inspection internal/external differs')
        require(receipt['episode_id'] == eid and receipt['task_id'] == task
                and receipt['config_sha256'] == config_sha and receipt['freeze_sha256'] == freeze_sha
                and receipt['container_id'] == cid and receipt['before_start'] is True,
                'container receipt identity differs')
        role = checker.check_inspection(cid, receipt['inspection'])
        require(role == receipt['role'], 'inspection role differs')
        probes += role == 'environment_probe'
        expected_mounts = [] if role == 'environment_probe' else [
            {'target': '/workspace', 'read_only': False, 'source_name': '.l1-work-' + eid}]
        require(creation['data']['mounts'] == expected_mounts, 'created mount claim differs from actual inspection')
    require(probes == 1, 'exactly one fixed environment probe required')
    return len(created)


def audit_episode(frozen, freeze_sha, row, root, external, portable, observer, public, replay, seen):
    from c3_sandbox.l1.package import verify_package
    from c3_sandbox.l1.storage import digest
    require(sha(root / 'evidence_index.json') == row['index_file_sha256']
            and sha(root / 'manifest.json') == row['manifest_sha256'], 'cohort package anchor differs')
    checked = verify_package(root, expected_index_sha256=row['index_file_sha256'])
    require(checked['verified'] is True, 'sealed package failed: ' + str(checked.get('errors')))
    config = read(root / 'raw/resolved_config.json')
    manifest = read(root / 'manifest.json')
    require(digest(config) == row['config_sha256'] == frozen['task_config_sha256'][row['task_id']]
            and manifest['config_sha256'] == row['config_sha256'] and manifest['episode_id'] == row['episode_id']
            and not manifest.get('lineage'), 'frozen root configuration/identity differs')
    events = [json.loads(line) for line in (root / 'trajectory.jsonl').read_text().splitlines()]
    require(all(e['episode_id'] == row['episode_id'] for e in events), 'foreign event episode')
    count = receipt_binding(frozen, freeze_sha, row, events, root, external, portable, observer, seen)
    actual_public = read(root / 'benchmark/public_task.json')
    expected = public['tasks'][row['task_id']]
    expected_public = {'benchmark': 'officeval', 'task_id': row['task_id'], 'prompt': expected['prompt'],
                       'assets': [{**a, 'workspace_path': '/workspace/input/' + a['name']} for a in expected['assets']]}
    require(actual_public == expected_public, 'initial public prompt/assets differ from approved upstream lock')
    cp = read(root / 'checkpoints/cp000000.json')
    require(cp['budget_used']['model_requests'] == 0 and not cp.get('pending_calls'), 'nonroot initial checkpoint')
    initial = read(package_ref(root, next(e['data']['snapshot'] for e in events if e['kind'] == 'state.snapshot')))
    require(initial['label'] == 'initial' and {a['path']: a['sha256'] for a in initial['files']} == expected['workspace_files']
            and len(initial['files']) == len(expected['workspace_files'])
            and {d['path'] for d in initial['directories']} == {'input', 'submission'}, 'initial workspace contains nonpublic input')
    def recorded(reference):
        return package_ref(root, reference).read_bytes()
    reconstructed = replay.reconstruct_requests(config, actual_public, events, recorded)
    requests = [e for e in events if e['kind'] == 'model.request']
    require(len(reconstructed) == len(requests) and all(r['status'] == 'verified' for r in reconstructed),
            'full request history replay differs or incomplete')
    # Canonical raw body bytes, not merely JSON equivalence. The replay above
    # checks every production history transform against each preceding checkpoint.
    backend = replay.backend(config)
    from c3_sandbox.l1.runtime.tools import tools_for_config
    latest = None
    raw_count = 0
    for event in events:
        if event['kind'] == 'checkpoint.created':
            latest = read(package_ref(root, event['data']['checkpoint']))
        elif event['kind'] == 'model.request':
            require(latest is not None, 'request without preceding checkpoint')
            body = recorded(event['data']['body'])
            require(canonical(backend.request(latest['conversation'], tools_for_config(config))) == body,
                    'request raw bytes differ from reconstructed native checkpoint')
            raw_count += 1
    require(raw_count == len(requests), 'raw request coverage differs')
    return {'slot': row['label'], 'episode_id': row['episode_id'], 'index_sha256': row['index_file_sha256'],
            'request_count': len(requests), 'container_count': count, 'status': 'verified'}


def run(args, archive_root):
    portable = Portable([item.split('=', 1) for item in args.root_map])
    require(sha(args.freeze) == args.freeze_sha256 and sha(args.cohort) == args.cohort_sha256, 'input anchor SHA differs')
    frozen = read(args.freeze)
    frozen['_freeze_path'] = args.original_freeze
    graph = read(portable.ref(frozen['verified_artifacts']))
    candidate = read(portable.ref(frozen['candidate']))
    prep = next(r for r in candidate['stages'] if r['kind'] == 'controller_and_inputs')
    stage_path = portable.path(str(Path(graph['evidence_root']) / prep['path']))
    require(sha(stage_path) == prep['sha256'], 'controller stage anchor differs')
    stage = read(stage_path)
    controller_path = stage_path.parent / 'controller.json'
    require(sha(controller_path) == stage['files']['controller.json'], 'controller manifest anchor differs')
    controller = read(controller_path)
    for name, expected_hash in controller['source_files'].items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'unsafe archived source name')
        require(sha(archive_root / relative) == expected_hash
                and frozen['source_files'].get(name) == expected_hash, 'worker archived source differs')
    observer, public = preparation(frozen, portable, archive_root)
    replay = module('g4_frozen_replay', archive_root / L2_REL / 'scripts/audit_g4_provenance.py')
    cohort = read(args.cohort)
    plan = read(Path(args.batch).absolute().parent / 'plan.json')
    plan_digest = hashlib.sha256(canonical({k: v for k, v in plan.items() if k != 'plan_sha256'})).hexdigest()
    require(cohort['plan_sha256'] == plan_digest
            and plan['prospective_freeze_sha256'] == args.freeze_sha256, 'batch plan/freeze binding differs')
    service = read(Path(args.batch).absolute().parent / 'service.json')
    identity = frozen['expected_service_identity']
    identity_receipt = read(portable.ref(identity['identity_receipt']))
    require(all(identity_receipt[k] == identity[k] for k in ('model', 'vllm', 'command_after_timeout'))
            and service['identity_receipt_sha256'] == identity['identity_receipt']['sha256']
            and service['command'][4:] == identity['command_after_timeout']
            and service['server_plan_hash'] == plan_digest, 'actual service/model identity differs')
    for key in ('model', 'vllm'):
        portable.ref(identity[key]['file_manifest'])
    expected = {task + '.r' + str(i).zfill(2) for task in DEV for i in range(5)}
    require(set(cohort['slots']) == expected and set(plan['slots']) == expected
            and len(plan['slots']) == 45, 'exact DEV9 x 5 cohort required')
    external_root = Path(args.batch).absolute() / 'observer_receipts'
    require({p.name for p in external_root.iterdir()} == expected, 'foreign/missing observer slot')
    rows, seen, eids = [], set(), set()
    for slot in sorted(expected):
        row = cohort['slots'][slot]
        require(row['label'] == slot and row['task_id'] == slot.split('.')[0]
                and row['status'] == 'completed' and row['study_valid'] is True
                and row['episode_id'] not in eids, 'cohort slot identity/state differs')
        receipt = read(Path(args.batch).absolute() / ('receipt_' + slot + '.json'))
        require({k: v for k, v in receipt.items() if k != 'artifact_sha256'} == row, 'cohort/external slot receipt differs')
        require(receipt['artifact_sha256'] == hashlib.sha256(canonical(row)).hexdigest(), 'slot receipt seal differs')
        require(plan['configs'][row['task_id']] == read(portable.path(row['path']) / 'raw/resolved_config.json'),
                'batch plan resolved configuration differs')
        eids.add(row['episode_id'])
        try:
            rows.append(audit_episode(frozen, args.freeze_sha256, row, portable.path(row['path']),
                                      external_root / slot, portable, observer, public, replay, seen))
        except FileNotFoundError as exc:
            rows.append({'slot': slot, 'status': 'unverified', 'reason': str(exc)})
        except (ValueError, KeyError, TypeError) as exc:
            rows.append({'slot': slot, 'status': 'mismatch', 'reason': str(exc)})
    statuses = {r['status'] for r in rows}
    status = 'mismatch' if 'mismatch' in statuses else ('unverified' if 'unverified' in statuses else 'verified')
    return {'schema': 'prospective-g4-post-audit-v1', 'status': status, 'g4_verified': status == 'verified',
            'freeze_sha256': args.freeze_sha256, 'cohort_sha256': args.cohort_sha256,
            'source_commit': frozen['source_commit'], 'audit_source_sha256': sha(__file__),
            'root_maps': args.root_map, 'slots': rows, 'slot_count': len(rows),
            'request_count': sum(r.get('request_count', 0) for r in rows),
            'container_count': sum(r.get('container_count', 0) for r in rows),
            'claim': 'Approved public inputs, frozen controller/image, exact request histories and recorded prestart isolation boundaries agree under ordinary trusted runner/OS/Docker/HTTPS.',
            'excluded_claims': ['hardware attestation', 'provider or pretraining exclusion', 'universal upstream trust',
                                'full tool equivalence', 'outcome distribution equivalence'],
            'actor_requests': 0, 'grader_requests': 0, 'legacy_audit_modified': False, 'd0_frozen': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('freeze', 'freeze-sha256', 'original-freeze', 'cohort', 'cohort-sha256', 'batch', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--root-map', action='append', default=[])
    parser.add_argument('--archive-root', help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(platform.system() == 'Linux', 'formal G4 audit runs only on approved Linux CPU server')
    if args.archive_root:
        try:
            result = run(args, Path(args.archive_root))
        except FileNotFoundError as exc:
            result = {'schema': 'prospective-g4-post-audit-v1', 'status': 'unverified', 'g4_verified': False, 'reason': str(exc)}
        except (ValueError, KeyError, TypeError) as exc:
            result = {'schema': 'prospective-g4-post-audit-v1', 'status': 'mismatch', 'g4_verified': False, 'reason': str(exc)}
        Path(args.output, 'result.json').write_bytes(canonical(result) + b'\n')
        print(json.dumps({k: v for k, v in result.items() if k != 'slots'}))
        return 0 if result['g4_verified'] else 1
    output = Path(args.output).absolute()
    require(not output.exists(), 'output directory must be new')
    output.mkdir(parents=True)
    portable = Portable([item.split('=', 1) for item in args.root_map])
    require(sha(args.freeze) == args.freeze_sha256, 'freeze SHA differs')
    frozen = read(args.freeze)
    graph = read(portable.ref(frozen['verified_artifacts']))
    candidate = read(portable.ref(frozen['candidate']))
    prep = next(ref for ref in candidate['stages'] if ref['kind'] == 'controller_and_inputs')
    prep_path = portable.path(str(Path(graph['evidence_root']) / prep['path']))
    require(sha(prep_path) == prep['sha256'], 'preparation stage SHA differs')
    stage = read(prep_path)
    for name in ('controller.json', 'controller_source.tar'):
        require(sha(prep_path.parent / name) == stage['files'][name], 'controller evidence SHA differs')
    controller = read(prep_path.parent / 'controller.json')
    require(all(frozen['source_files'].get(k) == v for k, v in controller['source_files'].items()), 'controller differs from frozen sources')
    with tempfile.TemporaryDirectory(prefix='prospective-g4-source-') as temp:
        archive_root = Path(temp)
        extract_controller(prep_path.parent / 'controller_source.tar', controller, archive_root)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4',
                   OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
                   PYTHONPATH=str(archive_root / L1_REL / 'src') + ':' + str(archive_root / L2_REL / 'src'))
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--archive-root', str(archive_root)]
        return subprocess.run(command, env=env, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
