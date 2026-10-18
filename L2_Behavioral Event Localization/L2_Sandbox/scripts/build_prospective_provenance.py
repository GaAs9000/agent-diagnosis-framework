#!/usr/bin/env python3
"""Bounded, zero-Actor provenance preparation; never freezes an Actor pool.

Default writes a plan only. Capture, public download, image export, CPU archive
inventory, assembly and external-anchor verification are explicit one-shot stages.
Docker capture does not start a container or parse its archive on the Mac.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.parse
import urllib.request
import uuid

L2 = Path(__file__).resolve().parents[1]
ROOT = L2.parents[1]
ORIGIN_ROOT = Path('/Users/jiashen/Desktop/问题诊断')
L1 = ROOT / 'L1_sandbox/C3_Experimental_Sandbox'
sys.path[:0] = [str(L1 / 'src'), str(L2 / 'src')]
from c3_sandbox.l1.storage import digest
from c3_sandbox.l1.core.config import resolve as resolve_config

DEV = tuple('officeval_' + n for n in ('003', '004', '010', '033', '050', '060', '070', '085', '090'))
REPOSITORY = 'https://github.com/baidu-frontier-research/OmegaUse-OfficeVal.git'
DATASET = 'https://huggingface.co/datasets/baidu-frontier-research/OmegaUse-OfficeVal'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    with path.open('x', encoding='utf-8') as out:
        json.dump(value, out, ensure_ascii=False, sort_keys=True, indent=2)
        out.write('\n')
    path.chmod(0o600)


def remaining(deadline):
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError('provenance preparation deadline expired')
    return seconds


def sha(path, deadline=None):
    if Path(path).is_symlink():
        raise ValueError('evidence hashes require regular files, not symlinks')
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            if deadline is not None:
                remaining(deadline)
            value.update(block)
    return value.hexdigest()


def command(args, deadline, **kwargs):
    return subprocess.run(args, check=True, timeout=min(60, remaining(deadline)), **kwargs)


def assert_new_output(output):
    """Reject protected aliases and sealed ancestors before creating any file."""
    if output.exists() or output.is_symlink():
        raise FileExistsError('provenance stage requires a fresh output directory')
    destination = output.resolve()
    protected = (L1 / 'runs', L1 / 'src', L1 / 'tests', L1 / 'third_party', L1 / 'benchmarks',
                 L2 / 'src', L2 / 'scripts', L2 / 'tests', L2 / 'diagnoses', ROOT / 'scripts', ROOT / '.git')
    if any(destination.is_relative_to(path.resolve()) or path.resolve().is_relative_to(destination) for path in protected):
        raise ValueError('output overlaps a protected source, run or diagnosis tree')
    validation = (L2 / 'validation').resolve()
    allowed_phase = validation / 'a_plus_i_execute_20261018'
    if destination.is_relative_to(validation) and not destination.is_relative_to(allowed_phase):
        raise ValueError('old validation phases are immutable provenance inputs')
    for ancestor in destination.parents:
        if ((ancestor / 'once.lock').exists()
                or ((ancestor / 'manifest.json').is_file()
                    and ((ancestor / 'evidence_index.json').is_file() or (ancestor / 'trajectory.jsonl').is_file()
                         or (ancestor / 'started.json').is_file()))):
            raise ValueError('output is inside an immutable package or prior provenance stage')


def reserve(output, kind):
    assert_new_output(output)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    (output / 'once.lock').mkdir()
    write(output / 'started.json', {'kind': kind, 'actor_requests': 0, 'analyst_requests': 0})


def seal_stage(output, kind, payload, deadline):
    files = {str(path.relative_to(output)): sha(path, deadline)
             for path in sorted(output.rglob('*')) if path.is_file() and path.name != 'manifest.json'}
    manifest = {'schema': 'prospective-provenance-stage-v1', 'kind': kind, 'payload': payload,
                'files': files, 'actor_requests': 0, 'analyst_requests': 0,
                'assurance': 'captured_candidate; external_approval_and_runtime_links_required'}
    write(output / 'manifest.json', manifest)
    return manifest


def safe_path(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('evidence path escapes its root')
    lexical = root / relative
    if any(part.is_symlink() for part in (lexical, *lexical.parents) if part != root.parent):
        raise ValueError('symlink in evidence path')
    path = lexical.resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('evidence path escapes its root')
    return path


def configured_path(value, input_root):
    path = Path(value)
    if path.is_absolute():
        for origin in (ROOT, ORIGIN_ROOT):
            if path.is_relative_to(origin):
                path = path.relative_to(origin)
                break
        else:
            return path  # Explicit external upstream paths remain explicit.
    return safe_path(input_root, path)


def pinned_inputs(config, input_root, deadline):
    """Read only DEV9 task/rubric/verifier bytes, never enumerate held-out tasks."""
    benchmark = config['benchmark']
    for key in ('source_commit', 'dataset_revision'):
        value = benchmark[key]
        if len(value) != 40 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('immutable upstream Git revisions required')
    source = configured_path(benchmark['source_dir'], input_root)
    data = configured_path(benchmark['data_dir'], input_root)
    assets = configured_path(benchmark['assets_dir'], input_root)
    lock = read(configured_path(benchmark.get('data_lock', str(data / 'data.lock.json')), input_root))
    if lock['dataset_revision'] != benchmark['dataset_revision']:
        raise ValueError('dataset revision differs from lock')
    actual = command(['git', '-C', str(source), 'rev-parse', 'HEAD'], deadline, capture_output=True, text=True).stdout.strip()
    if actual != benchmark['source_commit']:
        raise ValueError('upstream source commit mismatch')
    tree = command(['git', '-C', str(source), 'rev-parse', 'HEAD^{tree}'], deadline, capture_output=True, text=True).stdout.strip()
    records, forbidden = {}, {}
    for task in DEV:
        task_path = data / 'tasks' / (task + '.json')
        rubric = data / 'rubrics' / (task + '.json')
        verifier = source / 'verifiers' / (task + '_verifier.py')
        raw = read(task_path)
        if raw['id'] != task or not isinstance(raw['instruction'], str):
            raise ValueError('invalid DEV task projection')
        expected = lock['tasks'][task]
        if sha(task_path, deadline) != expected['task'] or sha(rubric, deadline) != expected['rubric']:
            raise ValueError('DEV metadata differs from data lock')
        # Compare working verifier bytes to the pinned Git object, not only HEAD.
        git_bytes = command(['git', '-C', str(source), 'show', actual + ':verifiers/' + task + '_verifier.py'],
                            deadline, capture_output=True).stdout
        verifier_sha = sha(verifier, deadline)
        if hashlib.sha256(git_bytes).hexdigest() != verifier_sha:
            raise ValueError('DEV verifier working file differs from pinned Git object')
        public_assets = []
        for asset in raw['origin_files']:
            name = asset['dest']
            if PurePosixPath(name).name != name or name in ('', '.', '..') or '\\' in name:
                raise ValueError('unsafe or non-basename public asset')
            path = assets / task / name
            if not path.is_file():
                path = assets / 'task_files' / task / name
            actual_sha = sha(path, deadline)
            if actual_sha != expected['assets'][name]:
                raise ValueError('public asset differs from DEV lock')
            public_assets.append({'name': name, 'sha256': actual_sha, 'size_bytes': path.stat().st_size})
            forbidden[actual_sha] = {'task': task, 'kind': 'task_asset_must_not_be_baked_into_image'}
        if {a['name'] for a in public_assets} != set(expected['assets']):
            raise ValueError('public asset inventory differs from DEV lock')
        forbidden[expected['task']] = {'task': task, 'kind': 'task_metadata'}
        forbidden[expected['rubric']] = {'task': task, 'kind': 'private_rubric'}
        forbidden[verifier_sha] = {'task': task, 'kind': 'private_verifier'}
        records[task] = {'prompt': raw['instruction'], 'task_sha256': expected['task'],
                         'rubric_sha256': expected['rubric'], 'verifier_sha256': verifier_sha,
                         'assets': public_assets, 'workspace_files': {'input/' + a['name']: a['sha256'] for a in public_assets},
                         'recorded_origin': lock.get('origins', {}).get(task, 'unknown')}
    return {'source_repository': REPOSITORY, 'source_commit': actual, 'source_tree': tree,
            'dataset_repository': DATASET, 'dataset_revision': benchmark['dataset_revision'],
            'tasks': records, 'forbidden_image_file_sha256': forbidden,
            'origin_assurance': 'pinned_local_bytes; origin_labels_are_not_download_receipts'}


def plugin_inventory(config):
    entries = []
    for group in ('l1.models', 'l1.runtimes', 'l1.evaluators', 'l1.benchmarks'):
        for entry in importlib.metadata.entry_points(group=group):
            distribution = getattr(entry, 'dist', None)
            entries.append({'group': group, 'name': entry.name, 'value': entry.value,
                            'distribution': distribution.metadata.get('Name') if distribution else None,
                            'version': distribution.version if distribution else None})
    enabled = config.get('plugins', {}).get('load_installed', False)
    if enabled and entries:
        raise ValueError('enabled installed plugin entries require a separate reviewed implementation freeze; not loaded by this preparer')
    return {'load_installed': enabled, 'entries': entries,
            'policy': 'enabled installed entry-point groups must be empty; no unknown plugin code executed'}


def capture_controller(output, deadline, root=ROOT, config=None):
    plugins = plugin_inventory(config or {})
    files = {}
    paths = []
    for directory in (root / 'L1_sandbox/C3_Experimental_Sandbox/src', root / 'L2_Behavioral Event Localization/L2_Sandbox/src'):
        paths.extend(p for p in directory.rglob('*') if p.is_file() and not p.is_symlink()
                     and '__pycache__' not in p.parts and 'node_modules' not in p.parts
                     and p.suffix in ('.py', '.json', '.mjs', '.js'))
    paths.extend([Path(__file__).resolve(), L1 / 'pyproject.toml', L2 / 'pyproject.toml'])
    for directory in (L2 / 'scripts', root / 'scripts'):
        paths.extend(path for path in directory.rglob('*.py') if path.is_file() and not path.is_symlink())
    for path in sorted(set(paths)):
        files[str(path.relative_to(root))] = sha(path, deadline)
    with tarfile.open(output / 'controller_source.tar', 'w') as archive:
        for relative in sorted(files):
            remaining(deadline)
            archive.add(root / relative, arcname=relative, recursive=False)
    imports = {}
    for name in ('c3_sandbox.l1.core.orchestrator', 'c3_sandbox.l1.runtime.tools',
                 'c3_sandbox.l1.runtime.docker', 'c3_sandbox.l1.models.native', 'l2_sandbox.evidence.l1'):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(root):
            raise ValueError('actual controller import escaped frozen source root')
        imports[name] = {'path': str(path.relative_to(root)), 'sha256': sha(path, deadline)}
    from c3_sandbox.l1.runtime.pi_native_validation import NativePiValidator
    with NativePiValidator(timeout_seconds=min(30, remaining(deadline))) as native:
        pi = native.dependency_identity()
        schemas = native.native_tool_schemas()
        node = Path(native.node_executable).resolve()
    dependencies = {}
    for name, module in sorted(sys.modules.items()):
        value = getattr(module, '__file__', None)
        if value and Path(value).is_file():
            path = Path(value).resolve()
            dependencies[name] = {'path': str(path), 'sha256': sha(path, deadline)}
    if any(sha(root / relative, deadline) != expected for relative, expected in files.items()):
        raise ValueError('controller source changed during capture')
    return {'source_files': files, 'source_manifest_sha256': digest(files), 'imports': imports,
            'python': {'executable': sys.executable, 'executable_sha256': sha(Path(sys.executable).resolve(), deadline),
                       'version': sys.version, 'platform': platform.platform(),
                       'distributions': sorted((d.metadata.get('Name', ''), d.version) for d in importlib.metadata.distributions())},
            'actual_imported_python_modules': dependencies, 'plugin_entry_points': plugins,
            'native_pi': pi, 'native_tool_schemas': schemas,
            'node_executable': {'path': str(node), 'sha256': sha(node, deadline)},
            'environment_policy': 'actual approved launch environment must be independently observed before Actor sends'}


def prepare(config, config_path, input_root, output, deadline):
    reserve(output, 'controller_and_inputs')
    inputs = pinned_inputs(config, input_root, deadline)
    write(output / 'public_inputs.json', inputs)
    write(output / 'resolved_config.json', config)
    controller = capture_controller(output, deadline, config=config)
    write(output / 'controller.json', controller)
    return seal_stage(output, 'controller_and_inputs', {'config_sha256': sha(config_path, deadline),
                      'public_inputs_sha256': sha(output / 'public_inputs.json', deadline),
                      'controller_manifest_sha256': sha(output / 'controller.json', deadline)}, deadline)


def fetch_public(public_manifest, output, deadline, opener=urllib.request.urlopen):
    """Download only DEV9 public metadata/assets, once, into a new evidence tree."""
    reserve(output, 'public_downloads')
    inputs = read(public_manifest)
    initial_sha = sha(public_manifest, deadline)
    if tuple(sorted(inputs['tasks'])) != tuple(sorted(DEV)):
        raise ValueError('public acquisition is restricted to exact DEV9')
    revision = inputs['dataset_revision']
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('pinned dataset revision required')
    receipts = []
    for task in DEV:
        record = inputs['tasks'][task]
        entries = [('tasks/' + task + '.json', record['task_sha256'])]
        entries += [('task_files/' + task + '/' + urllib.parse.quote(a['name'], safe=''), a['sha256']) for a in record['assets']]
        for relative, expected in entries:
            url = DATASET + '/resolve/' + revision + '/' + relative
            destination = output / ('download_' + str(len(receipts)))
            size = 0
            with opener(urllib.request.Request(url, headers={'User-Agent': 'agent-diagnosis-provenance/1'}),
                        timeout=min(60, remaining(deadline))) as response, destination.open('xb') as stream:
                final = response.geturl()
                host = urllib.parse.urlparse(final).hostname or ''
                if not final.startswith('https://') or not (host == 'huggingface.co' or host.endswith('.huggingface.co')
                        or host == 'hf.co' or host.endswith('.hf.co')):
                    raise ValueError('unapproved upstream redirect origin')
                while True:
                    remaining(deadline)
                    block = response.read(1048576)
                    if not block:
                        break
                    size += len(block)
                    if size > 268435456:
                        raise ValueError('upstream public file exceeds 256MiB')
                    stream.write(block)
                headers = {key: response.headers.get(key) for key in ('ETag', 'X-Repo-Commit', 'Content-Length')}
            if not size or sha(destination, deadline) != expected:
                raise ValueError('downloaded public bytes differ from pinned local lock')
            receipts.append({'url': url, 'resolved_url': final, 'headers': headers,
                             'file': destination.name, 'sha256': expected, 'size_bytes': size})
    write(output / 'download_receipts.json', receipts)
    if sha(public_manifest, deadline) != initial_sha:
        raise ValueError('public input manifest changed during acquisition')
    return seal_stage(output, 'public_downloads', {'public_inputs_sha256': sha(public_manifest, deadline),
                      'dataset_revision': revision, 'transport_trust': 'HTTPS upstream origin plus externally approved immutable revision'}, deadline)


def inspect_image(image, output, deadline, runner=command):
    """Capture only: never docker-start, run a model, or parse a tar archive."""
    reserve(output, 'image_capture')
    inspected = json.loads(runner(['docker', 'image', 'inspect', image], deadline, capture_output=True).stdout)
    if len(inspected) != 1 or inspected[0].get('Os') != 'linux':
        raise ValueError('one immutable Linux image required')
    if inspected[0].get('Config', {}).get('Volumes'):
        raise ValueError('declared image volumes would make Docker export incomplete')
    image_id = inspected[0]['Id']
    if len(image_id) != 71 or not image_id.startswith('sha256:'):
        raise ValueError('invalid immutable image identity')
    write(output / 'image_inspect.json', inspected[0])
    owner = uuid.uuid4().hex
    write(output / 'ownership.json', {'owner': owner, 'image_id': image_id})
    cid = None
    try:
        result = runner(['docker', 'create', '--label', 'prospective.provenance.owner=' + owner,
                         '--network', 'none', '--read-only', '--entrypoint', '/bin/true', image_id],
                        deadline, capture_output=True)
        created_id = result.stdout.decode().strip()
        if len(created_id) != 64 or any(c not in '0123456789abcdef' for c in created_id):
            raise ValueError('invalid captured container identity')
        cid = created_id
        container = json.loads(runner(['docker', 'inspect', cid], deadline, capture_output=True).stdout)[0]
        if container['Image'] != image_id or container.get('State', {}).get('Running') is not False:
            raise ValueError('export container differs from inspected image or was started')
        write(output / 'export_container.json', container)
        with (output / 'rootfs.tar').open('xb') as stream:
            exported = runner(['docker', 'export', cid], deadline, stdout=stream)
        write(output / 'export_receipt.json', {'container_id': cid, 'image_id': image_id, 'owner': owner,
                    'exit_code': exported.returncode, 'command': ['docker', 'export', cid],
                    'archive_sha256': sha(output / 'rootfs.tar', deadline),
                    'archive_size_bytes': (output / 'rootfs.tar').stat().st_size})
    finally:
        cleanup_deadline = time.monotonic() + 30
        selector = ['docker', 'ps', '-aq', '--no-trunc', '--filter', 'label=prospective.provenance.owner=' + owner]
        ids = runner(selector, cleanup_deadline, capture_output=True).stdout.decode().split()
        if any(len(item) != 64 or any(c not in '0123456789abcdef' for c in item) for item in ids):
            raise ValueError('invalid owned container lookup; cleanup refused')
        if ids:
            runner(['docker', 'rm', '-f', *ids], cleanup_deadline, capture_output=True)
        remaining_ids = runner(selector, cleanup_deadline, capture_output=True).stdout.decode().split()
        write(output / 'cleanup.json', {'owner': owner, 'removed_container_ids': ids,
                                       'remaining_ids': remaining_ids,
                                       'status': 'confirmed_empty' if not remaining_ids else 'unconfirmed'})
        if remaining_ids:
            raise RuntimeError('image export container cleanup is unconfirmed')
    return seal_stage(output, 'image_capture', {'image_id': image_id, 'requested_image': image,
                      'rootfs_archive_sha256': sha(output / 'rootfs.tar', deadline),
                      'scope': 'actual image inspect and never-started container export; no exclusion verdict'}, deadline)


def validate_capture(path, stage, deadline):
    """Bind full raw Docker receipts; payload hashes alone cannot stand in for capture."""
    base = path.parent
    required = {'image_inspect.json', 'export_container.json', 'ownership.json', 'export_receipt.json',
                'cleanup.json', 'rootfs.tar'}
    if not required <= set(stage['files']):
        raise ValueError('actual image capture requires raw inspect/export/owner/cleanup evidence')
    image, container = read(base / 'image_inspect.json'), read(base / 'export_container.json')
    ownership, exported, cleanup = [read(base / name) for name in ('ownership.json', 'export_receipt.json', 'cleanup.json')]
    image_id, cid, owner = stage['payload']['image_id'], container.get('Id'), ownership.get('owner')
    if (not isinstance(cid, str) or len(cid) != 64 or any(c not in '0123456789abcdef' for c in cid)
            or not isinstance(owner, str) or len(owner) != 32 or any(c not in '0123456789abcdef' for c in owner)):
        raise ValueError('raw export container/owner identity is invalid')
    image_cfg, cfg, host, state = image.get('Config', {}), container.get('Config', {}), container.get('HostConfig', {}), container.get('State', {})
    labels = {**(image_cfg.get('Labels') or {}), 'prospective.provenance.owner': owner}
    if (image.get('Id') != image_id or container.get('Image') != image_id or ownership.get('image_id') != image_id
            or image.get('Os') != 'linux' or image_cfg.get('Volumes')
            or container.get('Mounts') != [] or host.get('Binds') not in (None, [])
            or host.get('NetworkMode') != 'none' or host.get('ReadonlyRootfs') is not True
            or host.get('Privileged') is not False or host.get('Devices') not in (None, [])
            or cfg.get('Labels') != labels or cfg.get('Image') != image_id
            or cfg.get('Entrypoint') != ['/bin/true'] or cfg.get('Env') != image_cfg.get('Env')
            or cfg.get('User') != image_cfg.get('User') or cfg.get('Volumes') not in (None, {})
            or state.get('Running') is not False or state.get('Status') != 'created'
            or not str(state.get('StartedAt', '')).startswith('0001-01-01T00:00:00')):
        raise ValueError('raw image export was not the exact unmounted never-started safe capture')
    if (exported.get('container_id') != cid or exported.get('image_id') != image_id or exported.get('owner') != owner
            or exported.get('exit_code') != 0 or exported.get('command') != ['docker', 'export', cid]
            or exported.get('archive_sha256') != stage['payload']['rootfs_archive_sha256']
            or exported.get('archive_sha256') != sha(base / 'rootfs.tar', deadline)
            or exported.get('archive_size_bytes') != (base / 'rootfs.tar').stat().st_size
            or cleanup.get('owner') != owner or cleanup.get('status') != 'confirmed_empty'
            or cleanup.get('removed_container_ids') != [cid] or cleanup.get('remaining_ids') != []):
        raise ValueError('raw export archive/completeness/cleanup binding differs')


def archive_inventory(archive_path, deadline, forbidden=None, max_files=1000000, max_bytes=107374182400):
    """Stream files, never extract; hardlinks resolve within this inventory only."""
    records, total = {}, 0
    with tarfile.open(archive_path, 'r|*') as archive:
        for member in archive:
            remaining(deadline)
            name = str(PurePosixPath(member.name))
            if name in ('', '.') and member.isdir():
                continue
            if name.startswith('/') or '..' in PurePosixPath(name).parts or '\\' in name or name in records:
                raise ValueError('unsafe or duplicate archive member')
            if len(records) >= max_files:
                raise ValueError('archive member limit exceeded')
            row = {'path': name, 'mode': member.mode, 'size_bytes': member.size}
            if member.isfile() and not member.issparse():
                total += member.size
                if total > max_bytes:
                    raise ValueError('archive expanded byte limit exceeded')
                value = hashlib.sha256()
                stream = archive.extractfile(member)
                for block in iter(lambda: stream.read(1048576), b''):
                    remaining(deadline)
                    value.update(block)
                row.update(kind='file', sha256=value.hexdigest())
            elif member.isdir():
                row['kind'] = 'directory'
            elif member.issym():
                row.update(kind='symlink', target=member.linkname)
            elif member.islnk():
                target = str(PurePosixPath(member.linkname))
                if target.startswith('/') or '..' in PurePosixPath(target).parts:
                    raise ValueError('unsafe archive hardlink')
                row.update(kind='hardlink', target=target)
            else:
                raise ValueError('special or sparse archive entries are unsupported')
            records[name] = row
    for row in records.values():
        target, visited = row, set()
        while target['kind'] == 'hardlink':
            if target['path'] in visited or target['target'] not in records:
                raise ValueError('cyclic or missing hardlink target')
            visited.add(target['path'])
            target = records[target['target']]
        if row['kind'] == 'hardlink':
            if target['kind'] != 'file':
                raise ValueError('hardlink must resolve to regular inventory file')
            row['sha256'] = target['sha256']
    matches = [{'path': row['path'], 'fingerprint': forbidden[row['sha256']]}
               for row in records.values() if row.get('sha256') in (forbidden or {})]
    return {'archive_sha256': sha(archive_path, deadline), 'member_count': len(records),
            'regular_file_bytes': total, 'entries': [records[name] for name in sorted(records)],
            'known_private_or_task_file_matches': matches,
            'exclusion_assurance': 'exact known DEV9 fingerprints only; full allowlist/build review still required'}


def inventory_stage(archive_path, capture_manifest, public_manifest, output, deadline):
    if platform.system() != 'Linux':
        raise ValueError('rootfs inventory must run on the approved Linux CPU server')
    reserve(output, 'image_inventory')
    capture = read(capture_manifest)
    validate_capture(capture_manifest, capture, deadline)
    inputs = read(public_manifest)
    inventory = archive_inventory(archive_path, deadline, inputs['forbidden_image_file_sha256'])
    if inventory['archive_sha256'] != capture['payload']['rootfs_archive_sha256']:
        raise ValueError('actual archive differs from captured image export')
    write(output / 'rootfs_inventory.json', inventory)
    return seal_stage(output, 'image_inventory', {'image_id': capture['payload']['image_id'],
                      'capture_manifest_sha256': sha(capture_manifest, deadline),
                      'public_inputs_sha256': sha(public_manifest, deadline),
                      'rootfs_inventory_sha256': sha(output / 'rootfs_inventory.json', deadline)}, deadline)


def assemble(artifacts, evidence_root, output, deadline):
    reserve(output, 'candidate')
    stages = []
    for path in artifacts:
        relative = path.resolve().relative_to(evidence_root.resolve())
        stage = read(path)
        stages.append({'path': str(relative), 'sha256': sha(path, deadline), 'kind': stage['kind']})
    candidate = {'schema': 'prospective-provenance-candidate-v1', 'stages': stages,
                 'actor_requests': 0, 'actor_pool_frozen': False, 'g4_verified': False,
                 'remaining_links': ['external candidate/source/input approval', 'reviewed full image content policy',
                                     'actual launcher/container observer bindings before and during execution'],
                 'claims_excluded': ['hardware attestation', 'model pretraining exclusion', 'malicious host/provider exclusion']}
    write(output / 'candidate.json', candidate)
    return seal_stage(output, 'candidate', {'candidate_sha256': sha(output / 'candidate.json', deadline)}, deadline)


def verify_candidate(candidate_path, evidence_root, anchor, allowlist, deadline):
    """External reviewed anchor is an explicit trust input, never generated here."""
    if platform.system() != 'Linux':
        raise ValueError('independent archive replay verification must run on the approved Linux CPU server')
    candidate = read(candidate_path)
    approval = read(anchor)
    if (anchor.resolve().is_relative_to(candidate_path.parent.resolve()) or approval.get('approved') is not True
            or approval.get('candidate_sha256') != sha(candidate_path, deadline)):
        raise ValueError('independent externally retained candidate approval required')
    stages, manifests = [], {}
    for ref in candidate['stages']:
        path = safe_path(evidence_root, ref['path'])
        if sha(path, deadline) != ref['sha256']:
            raise ValueError('bound stage manifest changed')
        stage = read(path)
        if stage['kind'] != ref['kind']:
            raise ValueError('stage kind differs from candidate')
        for relative, expected in stage['files'].items():
            if sha(safe_path(path.parent, relative), deadline) != expected:
                raise ValueError('bound stage evidence bytes changed')
        stages.append((path, stage))
        manifests[ref['sha256']] = stage
        if stage['kind'] == 'image_capture':
            validate_capture(path, stage, deadline)
    kinds = {stage['kind'] for _, stage in stages}
    required = {'controller_and_inputs', 'public_downloads', 'image_capture', 'image_inventory'}
    if not required <= kinds:
        raise ValueError('required prospective evidence stages missing')
    reviewed = read(allowlist)
    if approval.get('reviewed_allowlist_sha256') != sha(allowlist, deadline) or reviewed.get('approved') is not True:
        raise ValueError('external full-inventory review approval missing')
    inventories = [stage for _, stage in stages if stage['kind'] == 'image_inventory']
    captures = {stage['payload']['image_id']: stage for _, stage in stages if stage['kind'] == 'image_capture'}
    preparations = [(path, stage) for path, stage in stages if stage['kind'] == 'controller_and_inputs']
    if len(preparations) != 1:
        raise ValueError('one actual controller/input preparation is required')
    prep_path, _ = preparations[0]
    config = read(prep_path.parent / 'resolved_config.json')
    aliases = {stage['payload'].get('requested_image', image_id): image_id for image_id, stage in captures.items()}
    def resolve_image(requested):
        if requested in captures:
            return requested
        if requested not in aliases:
            raise ValueError('configured image tag lacks an actual immutable capture binding')
        return aliases[requested]
    actor_image = resolve_image(config['runtime']['image'])
    grader_image = resolve_image(config['evaluation']['image'])
    controller = read(prep_path.parent / 'controller.json')
    source_inventory = archive_inventory(prep_path.parent / 'controller_source.tar', deadline)
    archived_sources = {row['path']: row['sha256'] for row in source_inventory['entries'] if row['kind'] == 'file'}
    if (archived_sources != controller['source_files']
            or digest(archived_sources) != controller['source_manifest_sha256']
            or any(archived_sources.get(row['path']) != row['sha256'] for row in controller['imports'].values())):
        raise ValueError('actual controller imports differ from the retained source archive')
    public = read(prep_path.parent / 'public_inputs.json')
    if set(public['tasks']) != set(DEV):
        raise ValueError('prepared public lock must contain exact DEV9')
    expected_downloads = {}
    for task in DEV:
        record = public['tasks'][task]
        base = DATASET + '/resolve/' + public['dataset_revision'] + '/'
        expected_downloads[base + 'tasks/' + task + '.json'] = record['task_sha256']
        for asset in record['assets']:
            expected_downloads[base + 'task_files/' + task + '/' + urllib.parse.quote(asset['name'], safe='')] = asset['sha256']
    for path, stage in stages:
        if stage['kind'] == 'public_downloads':
            receipts = read(path.parent / 'download_receipts.json')
            if {r['url']: r['sha256'] for r in receipts} != expected_downloads or len(receipts) != len(expected_downloads):
                raise ValueError('upstream receipts do not cover exactly the frozen public files')
            for row in receipts:
                if sha(safe_path(path.parent, row['file']), deadline) != row['sha256']:
                    raise ValueError('download receipt differs from actual retained bytes')
    for inventory in inventories:
        payload = inventory['payload']
        if manifests.get(payload['capture_manifest_sha256']) is not captures.get(payload['image_id']):
            raise ValueError('inventory is not bound to an actual captured image')
        approved = reviewed.get('images', {}).get(payload['image_id'], {})
        if (approved.get('rootfs_inventory_sha256') != payload['rootfs_inventory_sha256']
                or not approved.get('content_policy') or approved.get('all_entries_reviewed') is not True):
            raise ValueError('image full inventory lacks matching reviewed content policy')
        path = next(path for path, stage in stages if stage is inventory)
        captured_path = next(path for path, stage in stages if stage is captures[payload['image_id']])
        recomputed = archive_inventory(captured_path.parent / 'rootfs.tar', deadline,
                                       public['forbidden_image_file_sha256'])
        if recomputed != read(path.parent / 'rootfs_inventory.json'):
            raise ValueError('rootfs inventory differs from independently replayed archive')
        if payload['image_id'] == actor_image and recomputed['known_private_or_task_file_matches']:
            raise ValueError('image contains forbidden known task/private files')
    reviewed_images = {stage['payload']['image_id'] for stage in inventories}
    if not {actor_image, grader_image} <= reviewed_images:
        raise ValueError('actor and grader images must resolve to explicit reviewed immutable IDs')
    public_sha = sha(prep_path.parent / 'public_inputs.json', deadline)
    if any(stage['payload'].get('public_inputs_sha256') != public_sha for _, stage in stages
           if stage['kind'] in ('public_downloads', 'image_inventory')):
        raise ValueError('download/image proof uses different public input lock')
    return {'preparation_links_validated': True, 'candidate_sha256': sha(candidate_path, deadline),
            'g4_verified': False, 'actor_pool_frozen': False, 'actor_requests': 0,
            'configured_images_unchanged': {'actor': config['runtime']['image'], 'grader': config['evaluation']['image']},
            'resolved_images': {'actor': actor_image, 'grader': grader_image},
            'trust_root': 'explicit reviewed external approval; trusted ordinary OS/runner/Docker/HTTPS',
            'remaining_link': 'actual launcher/runtime observer receipts for the future 45 runs'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    for name in ('plan-only', 'prepare', 'fetch-public', 'inspect-image', 'inventory-archive', 'assemble', 'verify-candidate'):
        modes.add_argument('--' + name, action='store_true')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--input-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=900)
    parser.add_argument('--public-manifest', type=Path)
    parser.add_argument('--image')
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--capture-manifest', type=Path)
    parser.add_argument('--artifact', type=Path, action='append', default=[])
    parser.add_argument('--evidence-root', type=Path)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--anchor', type=Path)
    parser.add_argument('--reviewed-allowlist', type=Path)
    args = parser.parse_args()
    requirements = ((args.prepare, ('config',)), (args.fetch_public, ('public_manifest',)),
                    (args.inspect_image, ('image',)),
                    (args.inventory_archive, ('archive', 'capture_manifest', 'public_manifest')),
                    (args.assemble, ('artifact',)),
                    (args.verify_candidate, ('candidate', 'anchor', 'reviewed_allowlist')))
    for active, required in requirements:
        if active and any(not getattr(args, name) for name in required):
            parser.error('selected stage requires ' + ', '.join('--' + name.replace('_', '-') for name in required))
    if not 1 <= args.max_seconds <= 21600:
        parser.error('max-seconds must be within 1..21600')
    deadline = time.monotonic() + args.max_seconds
    assert_new_output(args.output)
    for source in (L1 / 'src', L2 / 'src', ROOT / '.git', L1 / 'third_party', L1 / 'benchmarks'):
        if args.output.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(args.output.resolve()):
            raise ValueError('output must be separate from existing code and benchmark trees')
    # Inputs and existing source trees are never output destinations.
    for source in (args.config, args.archive, args.capture_manifest, args.public_manifest, args.candidate, args.anchor, *args.artifact):
        if source and source.resolve().is_relative_to(args.output.resolve()):
            raise ValueError('new output directory must not contain an existing input artifact')
    if args.prepare:
        config = resolve_config(args.config)
        for key in ('source_dir', 'data_dir', 'assets_dir'):
            source = configured_path(config['benchmark'][key], args.input_root).resolve()
            if args.output.resolve().is_relative_to(source) or source.is_relative_to(args.output.resolve()):
                raise ValueError('output must be separate from immutable benchmark inputs')
        result = prepare(config, args.config, args.input_root, args.output, deadline)
    elif args.fetch_public:
        result = fetch_public(args.public_manifest, args.output, deadline)
    elif args.inspect_image:
        result = inspect_image(args.image, args.output, deadline)
    elif args.inventory_archive:
        result = inventory_stage(args.archive, args.capture_manifest, args.public_manifest, args.output, deadline)
    elif args.assemble:
        result = assemble(args.artifact, args.evidence_root or args.output.parent, args.output, deadline)
    elif args.verify_candidate:
        result = verify_candidate(args.candidate, args.evidence_root or args.output.parent, args.anchor,
                                  args.reviewed_allowlist, deadline)
        reserve(args.output, 'verification')
        write(args.output / 'verification.json', result)
    else:
        reserve(args.output, 'plan_only')
        result = {'mode': 'plan_only', 'tasks': DEV, 'actor_requests': 0, 'actor_pool_frozen': False,
                  'config_sha256': sha(args.config, deadline) if args.config else None,
                  'explicit_stages': ['prepare', 'fetch-public', 'inspect-image', 'inventory-archive', 'assemble', 'verify-candidate'],
                  'g4_verified': False, 'image_policy': 'retain current image only after complete reviewed inventory; never auto-rebuild'}
        write(args.output / 'preparation_plan.json', result)
    print(json.dumps({'actor_requests': 0, 'g4_verified': False, 'output': str(args.output)}))
    return result


if __name__ == '__main__':
    main()
