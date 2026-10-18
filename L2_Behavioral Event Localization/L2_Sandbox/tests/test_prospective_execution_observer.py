"""CPU-only actual-inspect fixtures; no Docker/model/network/GPU calls."""
import copy
import importlib.util
import os
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/prospective_execution_observer.py'
spec = importlib.util.spec_from_file_location('prospective_observer_test_target', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
IMAGE, CID = 'sha256:' + 'a' * 64, 'b' * 64


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(m.canonical(value) + b'\n')
    return {'path': str(path), 'sha256': m.sha(path)}


def proof_tree(tmp_path, source):
    base = tmp_path / 'proof'; base.mkdir()
    def stage(name, kind, files, payload):
        directory = base / name; directory.mkdir()
        for name, value in files.items():
            path = directory / name
            if isinstance(value, bytes): path.write_bytes(value)
            else: write(path, value)
        value = {'schema': 'prospective-provenance-stage-v1', 'kind': kind,
                 'files': {p.name: m.sha(p) for p in directory.iterdir()}, 'payload': payload}
        ref = write(directory / 'manifest.json', value)
        return ref, value
    public = {'dataset_repository': 'https://huggingface.co/datasets/test/dev', 'dataset_revision': 'a' * 40,
              'tasks': {task: {'task_sha256': m.digest({'task': task}), 'assets': []} for task in m.DEV}}
    controller = {'source_files': {'controller.py': m.sha(source)},
                  'source_manifest_sha256': m.digest({'controller.py': m.sha(source)})}
    prep, prep_json = stage('controller', 'controller_and_inputs', {
        'controller.json': controller, 'controller_source.tar': b'independently verified source archive',
        'public_inputs.json': public, 'resolved_config.json': {}}, {})
    public_sha = m.sha(base / 'controller/public_inputs.json')
    download_files, receipts = {}, []
    for i, task in enumerate(m.DEV):
        row = {'task': task}; filename = f'{i}.json'
        download_files[filename] = m.canonical(row)
        receipts.append({'url': public['dataset_repository'] + '/resolve/' + public['dataset_revision'] + '/tasks/' + task + '.json',
                         'file': filename, 'sha256': m.digest(row)})
    download_files['download_receipts.json'] = receipts
    downloaded, downloaded_json = stage('downloads', 'public_downloads', download_files,
                                       {'public_inputs_sha256': public_sha})
    rootfs = b'independently verified rootfs archive'
    rootfs_sha = m.hashlib.sha256(rootfs).hexdigest()
    owner = 'd' * 32
    image_json = {'Id': IMAGE, 'Os': 'linux', 'Config': {'Env': ['PATH=/bin', 'GPG_KEY=public-signing-key'],
                  'User': '65532', 'Labels': None}}
    container_json = {'Id': CID, 'Image': IMAGE, 'Mounts': [], 'HostConfig': {
        'Binds': None, 'NetworkMode': 'none', 'ReadonlyRootfs': True, 'Privileged': False, 'Devices': []},
        'Config': {'Image': IMAGE, 'Labels': {'prospective.provenance.owner': owner},
        'Entrypoint': ['/bin/true'], 'Env': image_json['Config']['Env'], 'User': '65532', 'Volumes': None},
        'State': {'Running': False, 'Status': 'created', 'StartedAt': '0001-01-01T00:00:00Z'}}
    captured, captured_json = stage('capture', 'image_capture', {'image_inspect.json': image_json,
        'export_container.json': container_json, 'ownership.json': {'owner': owner, 'image_id': IMAGE},
        'rootfs.tar': rootfs, 'export_receipt.json': {'container_id': CID, 'image_id': IMAGE, 'owner': owner,
            'exit_code': 0, 'command': ['docker', 'export', CID], 'archive_sha256': rootfs_sha,
            'archive_size_bytes': len(rootfs)}, 'cleanup.json': {'owner': owner, 'status': 'confirmed_empty',
            'removed_container_ids': [CID], 'remaining_ids': []}},
        {'image_id': IMAGE, 'rootfs_archive_sha256': rootfs_sha})
    inventory = {'archive_sha256': rootfs_sha, 'known_private_or_task_file_matches': [], 'entries': []}
    inventory_path = base / 'inventory/rootfs_inventory.json'
    inventory_sha = m.hashlib.sha256(m.canonical(inventory) + b'\n').hexdigest()
    inventoried, inventoried_json = stage('inventory', 'image_inventory', {'rootfs_inventory.json': inventory},
        {'image_id': IMAGE, 'capture_manifest_sha256': captured['sha256'],
         'public_inputs_sha256': public_sha, 'rootfs_inventory_sha256': inventory_sha})
    stage_records = [(prep, prep_json), (downloaded, downloaded_json), (captured, captured_json), (inventoried, inventoried_json)]
    refs = [{'path': str(Path(ref['path']).relative_to(base)), 'sha256': ref['sha256'], 'kind': value['kind']}
            for ref, value in stage_records]
    candidate = write(base / 'candidate/candidate.json', {'schema': 'prospective-provenance-candidate-v1',
                      'g4_verified': False, 'actor_pool_frozen': False, 'stages': refs})
    verification = write(base / 'verification.json', {'preparation_links_validated': True,
                         'candidate_sha256': candidate['sha256'], 'g4_verified': False,
                         'resolved_images': {'actor': IMAGE, 'grader': IMAGE}})
    allowlist = write(tmp_path / 'allowlist.json', {'approved': True, 'images': {
        IMAGE: {'rootfs_inventory_sha256': inventory_sha, 'all_entries_reviewed': True,
                'content_policy': 'reviewed generic Office runtime; excludes task-specific/private assets'}}})
    anchor = write(tmp_path / 'external_anchor.json', {'approved': True, 'candidate_sha256': candidate['sha256'],
                                                    'reviewed_allowlist_sha256': allowlist['sha256']})
    files = {r['path']: r['sha256'] for r in [candidate, verification, allowlist, anchor]}
    for ref, value in stage_records:
        files[ref['path']] = ref['sha256']
        files.update({str(Path(ref['path']).parent / name): value for name, value in value['files'].items()})
    graph = write(tmp_path / 'verified_artifacts.json', {'schema': 'prospective-verified-artifacts-v1',
        'candidate_sha256': candidate['sha256'], 'verification_sha256': verification['sha256'],
        'evidence_root': str(base), 'external_anchor': anchor, 'reviewed_allowlist': allowlist,
        'stage_manifests': [{'path': ref['path'], 'sha256': ref['sha256'], 'kind': value['kind']}
                            for ref, value in stage_records], 'files': files})
    image = {'path': str(base / 'capture/image_inspect.json'), 'sha256': captured_json['files']['image_inspect.json']}
    return candidate, verification, image, graph, files


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    source = tmp_path / 'controller.py'; source.write_text('reviewed source\n')
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    monkeypatch.setattr(m, 'host_user', lambda: '501:20')
    monkeypatch.setattr(m, 'operational_sources', lambda: {'controller.py': m.sha(source)})
    config = {'benchmark': {'task_id': m.DEV[0]}, 'runtime': {'adapter': 'docker-cold',
              'image': IMAGE, 'memory': '2g', 'cpus': 2, 'pids_limit': 128}}
    candidate, verification, image, graph, proof_files = proof_tree(tmp_path, source)
    node = tmp_path / 'native-node'; node.write_bytes(b'fixed CPU Node executable')
    bridge = tmp_path / 'native-bridge'; bridge.mkdir()
    (bridge / 'bridge.mjs').write_text('approved recorded-assistant bridge')
    monkeypatch.setenv('C3_PI_NODE', str(node))
    monkeypatch.setenv('C3_PI_NATIVE_DIR', str(bridge))
    installed = {'version': '0.73.1', 'actor_requests': 0, 'approved_modules_verified': True,
                 'installed_tree_sha256': 'e' * 64}
    native = {'node_path': str(node), 'node_sha256': m.sha(node), 'bridge_path': str(bridge),
              'bridge_files': {'bridge.mjs': m.sha(bridge / 'bridge.mjs')}, 'dependency_identity': installed}
    proof_files.update({str(node): m.sha(node), str(bridge / 'bridge.mjs'): m.sha(bridge / 'bridge.mjs')})
    rows = [{'case': str(i), 'restore_exact': True, 'request_exact': True,
             'unexplained_state_difference': False, 'unexplained_tool_difference': False,
             'deterministic_tool_exact': i < 3, 'declared_runtime_variance': i >= 3} for i in range(5)]
    mini = write(tmp_path / 'mini.json', {'schema': 'recovery_mechanism_mini_result_v1',
                  'actor_requests': 0, 'analyst_requests': 0, 'tool_attempts': 5,
                  'cases': rows, 'status': 'review_runtime_variance'})
    auth = write(tmp_path / 'authorization.json', {'authorization_source': 'direct_user_reply',
                  'exact_text': 'approved limited contract', 'policy': 'bounded-mechanism-contract-v1',
                  'allow_unexplained_differences': False})
    review = write(tmp_path / 'review.json', {'approved': True, 'policy': 'bounded-mechanism-contract-v1',
                    'result_sha256': mini['sha256'], 'authorization': auth,
                    'allow_declared_runtime_variance': True, 'unexplained_differences_allowed': False,
                    'full_tool_equivalence_claim': False, 'outcome_distribution_equivalence_claim': False})
    refs = [candidate, verification, mini, review, auth, image, graph]
    value = {'schema': 'prospective-live-freeze-v1', 'phase_authorized': True, 'internal_reviewed': True,
             'd0_frozen': False, 'provenance_candidate_verified': True,
             'task_allowed': list(m.DEV), 'task_config_sha256': {task: m.digest(config) for task in m.DEV},
             'actor_image_id': IMAGE, 'actor_image_env': ['PATH=/bin', 'GPG_KEY=public-signing-key'],
             'actor_user': '501:20', 'source_files': {'controller.py': m.sha(source)},
             'candidate': candidate, 'provenance_verification': verification, 'mini_review': review,
             'actor_image_inspect': image,
             'verified_artifacts': graph,
             'native_execution': native,
             'mechanism_mini': {'reviewed': True, 'result_path': mini['path'], 'result_sha256': mini['sha256']},
             'bound_files': {**proof_files, **{r['path']: r['sha256'] for r in refs}}}
    path = tmp_path / 'freeze.json'; write(path, value)
    return path, value, config


def ready(frozen, tmp_path, label='one'):
    path, _, config = frozen
    observer = m.ProspectiveExecutionObserver(path, m.sha(path), tmp_path / ('receipts_' + label))
    observer.validate(config)
    workspace = tmp_path / ('workspace_' + label); workspace.mkdir()
    observer.bind_workspace(workspace)
    binding = observer.bind_episode('ep_' + label, m.digest(config))
    observer.validate_native_identity(observer.frozen['native_execution']['dependency_identity'])
    return observer, binding


def inspection(observer, actor=True):
    mounts = [{'Type': 'bind', 'Source': str(observer.workspace), 'Destination': '/workspace',
               'RW': True, 'Propagation': 'rprivate'}] if actor else []
    cmd = ['-lc', 'printf recorded'] if actor else ['-I', '-c', m.PROBE_SCRIPT]
    entry = ['/bin/bash'] if actor else ['python']
    return {'Id': CID, 'Image': IMAGE, 'State': {'Status': 'created', 'Running': False,
            'StartedAt': '0001-01-01T00:00:00Z'}, 'Mounts': mounts,
            'Config': {'Env': ['PATH=/bin', 'GPG_KEY=public-signing-key'] + [k + '=' + v for k, v in m.RUNTIME_ENV.items()],
                       'User': '501:20', 'WorkingDir': '/workspace' if actor else '/tmp',
                       'Entrypoint': entry, 'Cmd': cmd},
            'Path': entry[0], 'Args': cmd, 'NetworkSettings': {'Networks': {'none': {}}, 'Ports': {}},
            'HostConfig': {'NetworkMode': 'none', 'ReadonlyRootfs': True, 'Privileged': False,
                'CapDrop': ['ALL'], 'SecurityOpt': ['no-new-privileges'], 'Binds': None, 'Devices': [],
                'Memory': 2 * 1024**3, 'NanoCpus': 2_000_000_000, 'PidsLimit': 128,
                'Tmpfs': {'/tmp': 'rw,nosuid,nodev,size=256m,mode=1777'},
                'Mounts': [{'Type': 'bind', 'Source': str(observer.workspace), 'Target': '/workspace',
                            'ReadOnly': False}] if actor else []}}


def test_valid_freeze_does_not_promote_preparation_to_live_g4(frozen):
    path, value, _ = frozen
    assert m.validate_freeze(path, m.sha(path)) == value
    verification = m.read(value['provenance_verification']['path'])
    assert verification['g4_verified'] is False


@pytest.mark.parametrize('mutation', ['source', 'config_keys', 'heldout', 'missing_review', 'missing_nested_auth',
                                      'candidate_changed', 'no_internal_review', 'd0_pass'])
def test_freeze_tampering_fails_closed(frozen, mutation):
    path, value, _ = frozen
    if mutation == 'source': (path.parent / 'controller.py').write_text('unreviewed')
    if mutation == 'config_keys': del value['task_config_sha256'][m.DEV[-1]]
    if mutation == 'heldout': value['task_allowed'].append('officeval_999')
    if mutation == 'missing_review': value['mini_review'] = None
    if mutation == 'missing_nested_auth': del value['bound_files'][str(path.parent / 'authorization.json')]
    if mutation == 'candidate_changed': Path(value['candidate']['path']).write_text('{}')
    if mutation == 'no_internal_review': value['internal_reviewed'] = False
    if mutation == 'd0_pass': value['d0_frozen'] = True
    write(path, value)
    with pytest.raises((ValueError, TypeError)):
        m.validate_freeze(path, m.sha(path))


def test_review_runtime_variance_string_is_not_approval(frozen):
    path, value, _ = frozen
    review_path = Path(value['mini_review']['path'])
    review = m.read(review_path); review['allow_declared_runtime_variance'] = False
    ref = write(review_path, review)
    value['mini_review'] = ref; value['bound_files'][ref['path']] = ref['sha256']; write(path, value)
    with pytest.raises(ValueError, match='explicit bounded mini contract'):
        m.validate_freeze(path, m.sha(path))


@pytest.mark.parametrize('actor', [True, False])
def test_actual_inspect_is_durable_full_and_pre_start(frozen, tmp_path, actor):
    observer, binding = ready(frozen, tmp_path)
    original = inspection(observer, actor)
    receipt = observer.observe_container(CID, original)
    saved = m.read(receipt['path'])
    assert saved['inspection'] == original
    assert saved['role'] == ('actor_tool' if actor else 'environment_probe')
    assert saved['before_start'] is True and saved['g4_verified'] is False
    assert receipt['sha256'] == m.sha(receipt['path'])
    assert binding['external_receipt']['sha256'] == m.sha(binding['external_receipt']['path'])
    with pytest.raises(ValueError, match='duplicate'):
        observer.observe_container(CID, original)


@pytest.mark.parametrize('field,value', [
    ('NetworkMode', 'host'), ('ReadonlyRootfs', False), ('Privileged', True), ('CapDrop', []),
    ('CapAdd', ['SYS_ADMIN']), ('SecurityOpt', ['seccomp=unconfined']), ('Devices', [{'PathOnHost': '/dev/x'}]),
    ('DeviceRequests', [{'Count': -1}]), ('Memory', 0), ('NanoCpus', 0), ('PidsLimit', -1),
    ('CpuQuota', 1), ('PortBindings', {'8000/tcp': []}), ('ExtraHosts', ['secret:host-gateway']),
    ('Binds', ['/secret:/leak']), ('PidMode', 'host'), ('Tmpfs', {'/tmp': 'exec'})])
def test_actual_hostconfig_violation_never_gets_receipt(frozen, tmp_path, field, value):
    observer, _ = ready(frozen, tmp_path)
    actual = inspection(observer); actual['HostConfig'][field] = value
    with pytest.raises(ValueError): observer.observe_container(CID, actual)
    assert not (observer.external_dir / (CID + '.json')).exists()


@pytest.mark.parametrize('mutation', ['mount_extra', 'mount_wrong', 'mount_ro', 'env_extra', 'env_duplicate',
                                      'user', 'image', 'running', 'started', 'probe_command', 'path', 'args'])
def test_inspection_mount_environment_identity_and_process_violations(frozen, tmp_path, mutation):
    observer, _ = ready(frozen, tmp_path)
    actual = inspection(observer, actor=mutation != 'probe_command')
    if mutation == 'mount_extra': actual['Mounts'].append(copy.deepcopy(actual['Mounts'][0]))
    if mutation == 'mount_wrong': actual['Mounts'][0]['Source'] = '/secret'
    if mutation == 'mount_ro': actual['Mounts'][0]['RW'] = False
    if mutation == 'env_extra': actual['Config']['Env'].append('GEMINI_API_KEY=secret')
    if mutation == 'env_duplicate': actual['Config']['Env'].append('HOME=/tmp')
    if mutation == 'user': actual['Config']['User'] = 'root'
    if mutation == 'image': actual['Image'] = 'sha256:' + 'c' * 64
    if mutation == 'running': actual['State']['Running'] = True
    if mutation == 'started': actual['State']['StartedAt'] = '2026-10-18T01:00:00Z'
    if mutation == 'probe_command': actual['Config']['Cmd'][-1] = 'print("not approved probe")'
    if mutation == 'path': actual['Path'] = '/bin/evil'
    if mutation == 'args': actual['Args'] = ['not-original']
    with pytest.raises(ValueError): observer.observe_container(CID, actual)
    assert not (observer.external_dir / (CID + '.json')).exists()


def test_source_drift_after_binding_blocks_container(frozen, tmp_path):
    observer, _ = ready(frozen, tmp_path)
    (tmp_path / 'controller.py').write_text('changed after bind')
    with pytest.raises(ValueError, match='source changed'):
        observer.observe_container(CID, inspection(observer))
    assert not (observer.external_dir / (CID + '.json')).exists()


def test_config_drift_rejected_before_any_external_writes(frozen, tmp_path):
    path, _, config = frozen
    observer = m.ProspectiveExecutionObserver(path, m.sha(path), tmp_path / 'receipts')
    changed = copy.deepcopy(config); changed['runtime']['cpus'] = 3
    with pytest.raises(ValueError, match='configuration differs'):
        observer.validate(changed)
    assert not observer.external_dir.exists()


def test_external_receipt_fsync_failure_is_not_success(frozen, tmp_path, monkeypatch):
    observer, _ = ready(frozen, tmp_path)
    def fail(fd): raise OSError('durability unavailable')
    monkeypatch.setattr(m.os, 'fsync', fail)
    with pytest.raises(OSError, match='durability'):
        observer.observe_container(CID, inspection(observer))
    assert CID not in observer.seen_containers


def test_run_isolation_and_foreign_workspace_rejected(frozen, tmp_path):
    first, _ = ready(frozen, tmp_path, 'first')
    second, _ = ready(frozen, tmp_path, 'second')
    foreign = inspection(first)
    with pytest.raises(ValueError, match='unique run workspace'):
        second.observe_container(CID, foreign)
    a = first.observe_container(CID, foreign)
    b = second.observe_container(CID, inspection(second))
    assert a['path'] != b['path'] and a['episode_id'] != b['episode_id']
    with pytest.raises(ValueError, match='one episode'):
        first.validate(frozen[2])


def test_external_directory_existing_or_alias_is_rejected(frozen, tmp_path):
    path, _, _ = frozen
    existing = tmp_path / 'existing'; existing.mkdir()
    with pytest.raises(ValueError, match='must be new'):
        m.ProspectiveExecutionObserver(path, m.sha(path), existing)
    alias = tmp_path / 'alias'; alias.symlink_to(existing, target_is_directory=True)
    with pytest.raises(ValueError, match='must be new'):
        m.ProspectiveExecutionObserver(path, m.sha(path), alias / 'new')


def test_actual_host_identity_must_match_frozen_actor_user(frozen, monkeypatch):
    path, _, _ = frozen
    monkeypatch.setattr(m, 'host_user', lambda: '1000:1000')
    with pytest.raises(ValueError, match='actual runtime host identity'):
        m.validate_freeze(path, m.sha(path))


def test_image_environment_cannot_be_declared_independently_of_capture(frozen):
    path, value, _ = frozen
    value['actor_image_env'].append('PRIVATE_CREDENTIAL=not-in-image')
    write(path, value)
    with pytest.raises(ValueError, match='actual frozen image inspection'):
        m.validate_freeze(path, m.sha(path))


def test_extra_network_attachment_is_rejected_even_networkmode_none(frozen, tmp_path):
    observer, _ = ready(frozen, tmp_path)
    actual = inspection(observer)
    actual['NetworkSettings']['Networks']['bridge'] = {}
    with pytest.raises(ValueError, match='network attachment'):
        observer.observe_container(CID, actual)
    assert not (observer.external_dir / (CID + '.json')).exists()


def refresh_ref(value, key, payload):
    ref = write(Path(value[key]['path']), payload)
    value[key] = ref
    value['bound_files'][ref['path']] = ref['sha256']
    return ref


def test_empty_candidate_cannot_pass_with_resealed_approval_flags(frozen):
    path, value, _ = frozen
    candidate = m.read(value['candidate']['path']); candidate['stages'] = []
    candidate_ref = refresh_ref(value, 'candidate', candidate)
    verification = m.read(value['provenance_verification']['path'])
    verification['candidate_sha256'] = candidate_ref['sha256']
    verification_ref = refresh_ref(value, 'provenance_verification', verification)
    graph = m.read(value['verified_artifacts']['path'])
    anchor = m.read(graph['external_anchor']['path']); anchor['candidate_sha256'] = candidate_ref['sha256']
    graph['external_anchor'] = write(Path(graph['external_anchor']['path']), anchor)
    for ref in (candidate_ref, verification_ref, graph['external_anchor']):
        value['bound_files'][ref['path']] = ref['sha256']
        graph['files'][ref['path']] = ref['sha256']
    graph['candidate_sha256'] = candidate_ref['sha256']
    graph['verification_sha256'] = verification_ref['sha256']
    refresh_ref(value, 'verified_artifacts', graph)
    write(path, value)
    with pytest.raises(ValueError, match='full controller/public/capture/inventory'):
        m.validate_freeze(path, m.sha(path))


@pytest.mark.parametrize('filename', ['proof/controller/controller.json', 'proof/controller/controller_source.tar',
    'proof/controller/public_inputs.json', 'proof/downloads/0.json', 'proof/capture/rootfs.tar',
    'proof/capture/export_receipt.json', 'proof/capture/cleanup.json', 'proof/inventory/rootfs_inventory.json'])
def test_nested_proof_tamper_resealed_outer_freeze_still_fails(frozen, filename):
    path, value, _ = frozen
    target = path.parent / filename
    target.write_bytes(target.read_bytes() + b' ')
    # A new self-declared outer hash cannot override a stage's original bytes.
    value['bound_files'][str(target)] = m.sha(target)
    write(path, value)
    with pytest.raises(ValueError, match='proof stage file differs'):
        m.validate_freeze(path, m.sha(path))


def test_exhaustive_manifest_must_not_omit_rootfs_or_raw_cleanup(frozen):
    path, value, _ = frozen
    graph = m.read(value['verified_artifacts']['path'])
    del graph['files'][str(path.parent / 'proof/capture/rootfs.tar')]
    refresh_ref(value, 'verified_artifacts', graph)
    write(path, value)
    with pytest.raises(ValueError, match='exhaustive artifact graph omits'):
        m.validate_freeze(path, m.sha(path))


def test_cached_hash_detects_same_length_change_even_mtime_restored(tmp_path):
    path = tmp_path / 'large-proof.bin'; path.write_bytes(b'unchanged-source')
    before = path.stat()
    old = m.sha(path)
    assert m.sha(path) == old  # Permitted unchanged trusted-OS stat cache.
    path.write_bytes(b'modified-source!')
    assert path.stat().st_size == before.st_size
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert m.sha(path) != old  # ctime differs even if mtime was restored.


def test_nested_proof_change_after_bind_blocks_before_container_start(frozen, tmp_path):
    observer, _ = ready(frozen, tmp_path)
    target = tmp_path / 'proof/capture/rootfs.tar'
    target.write_bytes(target.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='external evidence file changed'):
        observer.observe_container(CID, inspection(observer))
    assert not (observer.external_dir / (CID + '.json')).exists()


def test_selected_native_launch_override_blocks_before_any_child(frozen, tmp_path, monkeypatch):
    path, _, _ = frozen
    other = tmp_path / 'other-node'; other.write_bytes(b'other executable')
    monkeypatch.setenv('C3_PI_NODE', str(other))
    with pytest.raises(ValueError, match='selected native launch differs'):
        m.validate_freeze(path, m.sha(path))


def test_actual_native_installed_identity_mismatch_blocks_containers(frozen, tmp_path):
    path, _, config = frozen
    observer = m.ProspectiveExecutionObserver(path, m.sha(path), tmp_path / 'native_mismatch_receipts')
    observer.validate(config)
    workspace = tmp_path / 'native_workspace'; workspace.mkdir()
    observer.bind_workspace(workspace)
    observer.bind_episode('ep_native', m.digest(config))
    changed = copy.deepcopy(observer.frozen['native_execution']['dependency_identity'])
    changed['installed_tree_sha256'] = 'changed'
    with pytest.raises(ValueError, match='actual installed native validator differs'):
        observer.validate_native_identity(changed)
    assert not (observer.external_dir / 'native_identity.json').exists()
    with pytest.raises(ValueError, match='unbound'):
        observer.observe_container(CID, inspection(observer))


def test_native_binding_persists_actual_identity_prior_to_container(frozen, tmp_path):
    observer, _ = ready(frozen, tmp_path)
    saved = m.read(observer.external_dir / 'native_identity.json')
    assert saved['dependency_identity'] == observer.frozen['native_execution']['dependency_identity']
    assert saved['before_container_or_actor_send'] is True
    assert saved['episode_id'] == observer.episode_id
