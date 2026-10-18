"""CPU-only fake acquisition/export fixtures; no Docker/model/GPU execution."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile
import time

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/build_prospective_provenance.py'
spec = importlib.util.spec_from_file_location('prospective_provenance_test', SCRIPT)
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
IMAGE = 'sha256:' + 'a' * 64
OWNER = 'd' * 32
CID = 'c' * 64


def deadline():
    return time.monotonic() + 60


def tar(path, entries):
    with tarfile.open(path, 'w') as archive:
        for name, kind, value in entries:
            member = tarfile.TarInfo(name)
            member.mode = 0o640
            if kind == 'file':
                member.size = len(value)
                archive.addfile(member, io.BytesIO(value))
            else:
                member.type = {'symlink': tarfile.SYMTYPE, 'hardlink': tarfile.LNKTYPE,
                               'device': tarfile.CHRTYPE, 'directory': tarfile.DIRTYPE}[kind]
                member.linkname = value
                archive.addfile(member)


def public_inputs():
    content = {}
    records = {}
    for task in d.DEV:
        raw = ('public metadata ' + task).encode()
        asset = ('public input ' + task).encode()
        records[task] = {'task_sha256': hashlib.sha256(raw).hexdigest(),
                         'assets': [{'name': 'input.bin', 'sha256': hashlib.sha256(asset).hexdigest()}]}
        content['tasks/' + task + '.json'] = raw
        content['task_files/' + task + '/input.bin'] = asset
    return {'tasks': records, 'dataset_revision': 'b' * 40, 'forbidden_image_file_sha256': {}}, content


def fake_opener(content, calls):
    class Response(io.BytesIO):
        headers = {}
        def geturl(self):
            return self.url
    def opener(request, **kwargs):
        calls.append(request.full_url)
        relative = request.full_url.split('/' + 'b' * 40 + '/', 1)[1]
        result = Response(content[relative])
        result.url = request.full_url
        return result
    return opener


def capture_receipts(directory, owner=OWNER):
    image_config = {'Env': ['HOME=/tmp'], 'User': '65532', 'Labels': None}
    d.write(directory / 'image_inspect.json', {'Id': IMAGE, 'Os': 'linux', 'Config': image_config})
    d.write(directory / 'export_container.json', {'Id': CID, 'Image': IMAGE, 'Mounts': [],
        'Config': {**image_config, 'Image': IMAGE, 'Entrypoint': ['/bin/true'],
                   'Labels': {'prospective.provenance.owner': owner}},
        'HostConfig': {'NetworkMode': 'none', 'ReadonlyRootfs': True, 'Privileged': False, 'Binds': None, 'Devices': []},
        'State': {'Running': False, 'Status': 'created', 'StartedAt': '0001-01-01T00:00:00Z'}})
    d.write(directory / 'ownership.json', {'owner': owner, 'image_id': IMAGE})
    d.write(directory / 'export_receipt.json', {'container_id': CID, 'image_id': IMAGE, 'owner': owner,
        'command': ['docker', 'export', CID], 'exit_code': 0, 'archive_sha256': d.sha(directory / 'rootfs.tar'),
        'archive_size_bytes': (directory / 'rootfs.tar').stat().st_size})
    d.write(directory / 'cleanup.json', {'owner': owner, 'status': 'confirmed_empty', 'remaining_ids': [],
                                       'removed_container_ids': [CID]})


def test_stream_inventory_covers_bytes_modes_symlinks_and_hardlinks(tmp_path, monkeypatch):
    archive = tmp_path / 'rootfs.tar'
    tar(archive, [('opt/generic_worker.py', 'file', b'generic dispatcher'),
                  ('etc/link', 'symlink', '/usr/share/resource'), ('copy', 'hardlink', 'opt/generic_worker.py')])
    monkeypatch.setattr(tarfile.TarFile, 'extractall', lambda *_: pytest.fail('must never extract'))
    fingerprint = hashlib.sha256(b'generic dispatcher').hexdigest()
    result = d.archive_inventory(archive, deadline(), {fingerprint: {'kind': 'fixture-private'}})
    assert result['member_count'] == 3
    assert result['entries'][0]['sha256'] == fingerprint
    assert result['entries'][1]['target'] == '/usr/share/resource'
    assert result['entries'][2]['mode'] == 0o640
    assert len(result['known_private_or_task_file_matches']) == 2
    assert not (tmp_path / 'opt').exists()


@pytest.mark.parametrize('entries', [
    [('../escape', 'file', b'bad')], [('/absolute', 'file', b'bad')],
    [('device', 'device', '')], [('a', 'file', b'a'), ('a', 'file', b'a')],
    [('link', 'hardlink', '../escape')], [('a', 'hardlink', 'b'), ('b', 'hardlink', 'a')],
])
def test_unsafe_archive_is_refused_without_extraction(tmp_path, entries):
    archive = tmp_path / 'bad.tar'; tar(archive, entries)
    with pytest.raises(ValueError):
        d.archive_inventory(archive, deadline())


def test_archive_limits_and_deadline_are_enforced(tmp_path):
    archive = tmp_path / 'big.tar'; tar(archive, [('file', 'file', b'content')])
    with pytest.raises(ValueError, match='byte limit'):
        d.archive_inventory(archive, deadline(), max_bytes=1)
    with pytest.raises(TimeoutError):
        d.archive_inventory(archive, time.monotonic() - 1)


def test_docker_capture_inspects_exports_and_removes_exact_container_without_start(tmp_path, monkeypatch):
    calls = []
    removed = False
    def runner(args, limit, **kwargs):
        nonlocal removed
        calls.append(args)
        if args[:3] == ['docker', 'image', 'inspect']:
            result = json.dumps([{'Id': IMAGE, 'Os': 'linux', 'Config': {}}]).encode()
        elif args[:2] == ['docker', 'create']:
            result = ('c' * 64).encode()
        elif args[:2] == ['docker', 'inspect']:
            result = json.dumps([{'Image': IMAGE, 'State': {'Running': False}}]).encode()
        elif args[:2] == ['docker', 'export']:
            kwargs['stdout'].write(b'opaque archive; capture does not parse')
            result = b''
        elif args[:2] == ['docker', 'ps']:
            result = b'' if removed else ('c' * 64).encode()
        else:
            assert args == ['docker', 'rm', '-f', 'c' * 64]
            removed = True
            result = b''
        return subprocess.CompletedProcess(args, 0, stdout=result)
    monkeypatch.setattr(tarfile, 'open', lambda *_: pytest.fail('Mac capture must not parse tar'))
    result = d.inspect_image('mutable-tag', tmp_path / 'capture', deadline(), runner)
    assert result['payload']['image_id'] == IMAGE
    assert all(args[1] not in ('run', 'start') for args in calls)
    assert ['docker', 'rm', '-f', 'c' * 64] in calls
    assert d.read(tmp_path / 'capture/cleanup.json')['status'] == 'confirmed_empty'
    with pytest.raises(FileExistsError):
        d.inspect_image('mutable-tag', tmp_path / 'capture', deadline(), runner)


def test_fetch_only_exact_dev_public_files_and_retains_upstream_receipts(tmp_path):
    inputs, content = public_inputs(); manifest = tmp_path / 'public.json'; d.write(manifest, inputs)
    calls = []
    output = tmp_path / 'fetch'
    d.fetch_public(manifest, output, deadline(), fake_opener(content, calls))
    assert len(calls) == 18
    assert all('/rubrics/' not in url for url in calls)
    assert {r['sha256'] for r in d.read(output / 'download_receipts.json')} == {hashlib.sha256(v).hexdigest() for v in content.values()}
    with pytest.raises(FileExistsError):
        d.fetch_public(manifest, output, deadline(), fake_opener(content, calls))


def test_fetch_refuses_holdout_content_before_any_network_call(tmp_path):
    inputs, content = public_inputs(); inputs['tasks']['officeval_002'] = {}
    manifest = tmp_path / 'public.json'; d.write(manifest, inputs)
    calls = []
    with pytest.raises(ValueError, match='exact DEV9'):
        d.fetch_public(manifest, tmp_path / 'fetch', deadline(), fake_opener(content, calls))
    assert calls == []


def test_changed_public_download_fails_and_preserves_one_shot_marker(tmp_path):
    inputs, content = public_inputs(); content['tasks/officeval_003.json'] = b'changed'
    manifest = tmp_path / 'public.json'; d.write(manifest, inputs)
    with pytest.raises(ValueError, match='differ'):
        d.fetch_public(manifest, tmp_path / 'fetch', deadline(), fake_opener(content, []))
    assert (tmp_path / 'fetch/once.lock').is_dir()


@pytest.fixture
def candidate(tmp_path):
    evidence = tmp_path / 'evidence'; evidence.mkdir()
    prep = evidence / 'prep'; prep.mkdir()
    inputs, content = public_inputs()
    d.write(prep / 'public_inputs.json', inputs)
    d.write(prep / 'resolved_config.json', {'runtime': {'image': IMAGE}, 'evaluation': {'image': IMAGE}})
    sources = {'src/controller.py': hashlib.sha256(b'reviewed controller').hexdigest()}
    tar(prep / 'controller_source.tar', [('src/controller.py', 'file', b'reviewed controller')])
    d.write(prep / 'controller.json', {'source_files': sources, 'source_manifest_sha256': d.digest(sources),
                                      'imports': {'controller': {'path': 'src/controller.py', 'sha256': sources['src/controller.py']}}})
    d.seal_stage(prep, 'controller_and_inputs', {}, deadline())
    fetch = evidence / 'fetch'
    d.fetch_public(prep / 'public_inputs.json', fetch, deadline(), fake_opener(content, []))
    capture = evidence / 'capture'; capture.mkdir()
    tar(capture / 'rootfs.tar', [('opt/generic_worker.py', 'file', b'generic dispatcher')])
    capture_receipts(capture)
    d.seal_stage(capture, 'image_capture', {'image_id': IMAGE, 'rootfs_archive_sha256': d.sha(capture / 'rootfs.tar')}, deadline())
    inventory = evidence / 'inventory'; inventory.mkdir()
    rows = d.archive_inventory(capture / 'rootfs.tar', deadline(), inputs['forbidden_image_file_sha256'])
    d.write(inventory / 'rootfs_inventory.json', rows)
    d.seal_stage(inventory, 'image_inventory', {'image_id': IMAGE,
        'capture_manifest_sha256': d.sha(capture / 'manifest.json'),
        'public_inputs_sha256': d.sha(prep / 'public_inputs.json'),
        'rootfs_inventory_sha256': d.sha(inventory / 'rootfs_inventory.json')}, deadline())
    output = evidence / 'candidate'
    d.assemble([path / 'manifest.json' for path in (prep, fetch, capture, inventory)], evidence, output, deadline())
    policy = tmp_path / 'review.json'
    d.write(policy, {'approved': True, 'images': {IMAGE: {'rootfs_inventory_sha256': d.sha(inventory / 'rootfs_inventory.json'),
              'all_entries_reviewed': True, 'content_policy': 'reviewed dependencies plus generic dispatcher; no task assets'}}})
    anchor = tmp_path / 'external_anchor.json'
    d.write(anchor, {'approved': True, 'candidate_sha256': d.sha(output / 'candidate.json'),
                     'reviewed_allowlist_sha256': d.sha(policy)})
    return output / 'candidate.json', evidence, anchor, policy


def test_external_review_validates_preparation_but_never_promotes_g4(candidate):
    result = d.verify_candidate(*candidate, deadline())
    assert result['preparation_links_validated'] is True
    assert result['g4_verified'] is False and result['actor_pool_frozen'] is False
    assert 'runtime observer' in result['remaining_link']


def test_external_anchor_and_actual_bytes_are_required(candidate):
    path, root, anchor, policy = candidate
    (root / 'capture/rootfs.tar').write_bytes(b'changed image bytes')
    with pytest.raises(ValueError, match='evidence bytes'):
        d.verify_candidate(path, root, anchor, policy, deadline())


def test_self_written_approval_inside_candidate_is_not_external(candidate):
    path, root, anchor, policy = candidate
    internal = path.parent / 'anchor.json'; internal.write_bytes(anchor.read_bytes())
    with pytest.raises(ValueError, match='externally retained'):
        d.verify_candidate(path, root, internal, policy, deadline())


def test_default_cli_does_not_capture_fetch_or_inventory(tmp_path, monkeypatch):
    for name in ('prepare', 'fetch_public', 'inspect_image', 'inventory_stage'):
        monkeypatch.setattr(d, name, lambda *_: pytest.fail('default unexpectedly executed a capture stage'))
    monkeypatch.setattr('sys.argv', [str(SCRIPT), '--output', str(tmp_path / 'plan')])
    result = d.main()
    assert result['mode'] == 'plan_only' and result['g4_verified'] is False


def test_mirrored_config_paths_rebase_without_importing_server_identity(tmp_path):
    value = str(d.ORIGIN_ROOT / 'L1_sandbox/public')
    assert d.configured_path(value, tmp_path) == tmp_path / 'L1_sandbox/public'


def test_plugins_are_listed_but_unknown_enabled_plugins_are_not_loaded(monkeypatch):
    entry = type('Entry', (), {'name': 'external', 'value': 'unknown:factory', 'dist': None})()
    monkeypatch.setattr(d.importlib.metadata, 'entry_points', lambda **kwargs: [entry] if kwargs['group'] == 'l1.models' else [])
    disabled = d.plugin_inventory({'plugins': {'load_installed': False}})
    assert disabled['entries'][0]['value'] == 'unknown:factory'
    with pytest.raises(ValueError, match='separate reviewed'):
        d.plugin_inventory({'plugins': {'load_installed': True}})


def test_tag_configuration_is_preserved_and_bound_to_actual_capture(candidate):
    path, root, anchor, policy = candidate
    prep = root / 'prep'; capture = root / 'capture'; inventory = root / 'inventory'
    config = d.read(prep / 'resolved_config.json')
    config['runtime']['image'] = config['evaluation']['image'] = 'retained:tag'
    (prep / 'resolved_config.json').write_text(json.dumps(config))
    prep_manifest = d.read(prep / 'manifest.json')
    prep_manifest['files']['resolved_config.json'] = d.sha(prep / 'resolved_config.json')
    (prep / 'manifest.json').write_text(json.dumps(prep_manifest))
    captured = d.read(capture / 'manifest.json'); captured['payload']['requested_image'] = 'retained:tag'
    (capture / 'manifest.json').write_text(json.dumps(captured))
    inventoried = d.read(inventory / 'manifest.json')
    inventoried['payload']['capture_manifest_sha256'] = d.sha(capture / 'manifest.json')
    (inventory / 'manifest.json').write_text(json.dumps(inventoried))
    c = d.read(path)
    for ref in c['stages']:
        ref['sha256'] = d.sha(root / ref['path'])
    path.write_text(json.dumps(c))
    approved = d.read(anchor); approved['candidate_sha256'] = d.sha(path)
    anchor.write_text(json.dumps(approved))
    result = d.verify_candidate(path, root, anchor, policy, deadline())
    assert result['configured_images_unchanged']['actor'] == 'retained:tag'
    assert result['resolved_images']['actor'] == IMAGE


def test_hashes_and_review_cannot_substitute_for_actual_archive_replay(candidate):
    path, root, anchor, policy = candidate
    public_path = root / 'prep/public_inputs.json'
    public = d.read(public_path)
    public['forbidden_image_file_sha256'][hashlib.sha256(b'generic dispatcher').hexdigest()] = {'kind': 'fixture-private'}
    public_path.write_text(json.dumps(public))
    # Rebind every declared hash/approval, but leave the false empty match list.
    for name in ('prep', 'fetch', 'inventory'):
        manifest_path = root / name / 'manifest.json'
        manifest = d.read(manifest_path)
        if name == 'prep':
            manifest['files']['public_inputs.json'] = d.sha(public_path)
        else:
            manifest['payload']['public_inputs_sha256'] = d.sha(public_path)
        manifest_path.write_text(json.dumps(manifest))
    c = d.read(path)
    for ref in c['stages']:
        ref['sha256'] = d.sha(root / ref['path'])
    path.write_text(json.dumps(c))
    approved = d.read(anchor); approved['candidate_sha256'] = d.sha(path)
    anchor.write_text(json.dumps(approved))
    with pytest.raises(ValueError, match='independently replayed'):
        d.verify_candidate(path, root, anchor, policy, deadline())


def test_dev_locks_only_read_declared_development_metadata(tmp_path, monkeypatch):
    source, data, assets = [tmp_path / name for name in ('source', 'data', 'assets')]
    revision = 'b' * 40
    lock = {'dataset_revision': revision, 'tasks': {}, 'origins': {}}
    verifier_bytes = {}
    for task in d.DEV:
        for directory in (data / 'tasks', data / 'rubrics', source / 'verifiers', assets / task):
            directory.mkdir(parents=True, exist_ok=True)
        task_path = data / 'tasks' / (task + '.json')
        d.write(task_path, {'id': task, 'instruction': 'Exact public task', 'origin_files': [{'dest': 'input.bin'}]})
        rubric = data / 'rubrics' / (task + '.json'); rubric.write_bytes(b'private DEV rubric')
        verifier = source / 'verifiers' / (task + '_verifier.py'); verifier.write_bytes(b'official verifier')
        asset = assets / task / 'input.bin'; asset.write_bytes(b'public DEV input')
        verifier_bytes[task + '_verifier.py'] = verifier.read_bytes()
        lock['tasks'][task] = {'task': d.sha(task_path), 'rubric': d.sha(rubric), 'assets': {'input.bin': d.sha(asset)}}
    # A held-out lock entry exists; neither its task nor its content exists locally.
    lock['tasks']['officeval_002'] = {'must_not_read': True}
    d.write(data / 'data.lock.json', lock)
    def git(args, limit, **kwargs):
        if args[-1] == 'HEAD':
            stdout = 'a' * 40
        elif args[-1] == 'HEAD^{tree}':
            stdout = 'tree'
        else:
            stdout = verifier_bytes[args[-1].rsplit('/', 1)[1]]
        return subprocess.CompletedProcess(args, 0, stdout=stdout)
    monkeypatch.setattr(d, 'command', git)
    config = {'benchmark': {'source_commit': 'a' * 40, 'dataset_revision': revision,
              'source_dir': str(source), 'data_dir': str(data), 'assets_dir': str(assets)}}
    result = d.pinned_inputs(config, tmp_path, deadline())
    assert set(result['tasks']) == set(d.DEV)
    assert 'officeval_002' not in json.dumps(result)


@pytest.mark.parametrize('location', ['existing', 'sealed_child', 'stage_child', 'runs_alias', 'old_validation'])
def test_outputs_rejected_before_any_write(tmp_path, monkeypatch, location):
    l1, l2 = tmp_path / 'l1', tmp_path / 'l2'
    l1.mkdir(); l2.mkdir()
    monkeypatch.setattr(d, 'L1', l1); monkeypatch.setattr(d, 'L2', l2)
    if location == 'existing':
        output = tmp_path / 'existing'; output.mkdir()
    elif location == 'sealed_child':
        parent = tmp_path / 'sealed'; parent.mkdir()
        (parent / 'manifest.json').write_text('{}'); (parent / 'evidence_index.json').write_text('{}')
        output = parent / 'new'
    elif location == 'stage_child':
        parent = tmp_path / 'old_stage'; parent.mkdir(); (parent / 'once.lock').mkdir()
        output = parent / 'new'
    elif location == 'runs_alias':
        runs = l1 / 'runs'; runs.mkdir()
        alias = tmp_path / 'alias'; alias.symlink_to(runs, target_is_directory=True)
        output = alias / 'new'
    else:
        output = l2 / 'validation/old_phase/new'
    before = set(tmp_path.rglob('*'))
    with pytest.raises((ValueError, FileExistsError)):
        d.reserve(output, 'fixture')
    assert set(tmp_path.rglob('*')) == before


@pytest.mark.parametrize('change', ['missing_raw', 'image', 'mounted', 'started', 'label', 'cleanup', 'archive'])
def test_raw_image_capture_receipts_are_mandatory_and_consistent(tmp_path, change):
    tar(tmp_path / 'rootfs.tar', [('file', 'file', b'original')])
    capture_receipts(tmp_path)
    stage = d.seal_stage(tmp_path, 'image_capture', {'image_id': IMAGE,
                         'rootfs_archive_sha256': d.sha(tmp_path / 'rootfs.tar')}, deadline())
    if change == 'missing_raw':
        del stage['files']['image_inspect.json']
    elif change == 'archive':
        (tmp_path / 'rootfs.tar').write_bytes(b'wrong archive')
    elif change == 'cleanup':
        value = d.read(tmp_path / 'cleanup.json'); value['removed_container_ids'] = ['e' * 64]
        (tmp_path / 'cleanup.json').write_text(json.dumps(value))
    else:
        value = d.read(tmp_path / 'export_container.json')
        if change == 'image':
            value['Image'] = 'sha256:' + 'b' * 64
        elif change == 'mounted':
            value['Mounts'] = [{'Source': '/host/private'}]
        elif change == 'started':
            value['State']['StartedAt'] = '2026-10-18T00:00:00Z'
        else:
            value['Config']['Labels']['prospective.provenance.owner'] = 'wrong'
        (tmp_path / 'export_container.json').write_text(json.dumps(value))
    with pytest.raises(ValueError):
        d.validate_capture(tmp_path / 'manifest.json', stage, deadline())
