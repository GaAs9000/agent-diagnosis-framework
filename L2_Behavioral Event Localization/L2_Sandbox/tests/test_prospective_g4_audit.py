"""Offline postcollection proof tests. No model, scorer, Docker, or GPU."""
import copy
import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
spec = importlib.util.spec_from_file_location('prospective_g4_audit_test', SCRIPTS / 'audit_g4_prospective.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
observer = m.module('g4_observer_fixture', SCRIPTS / 'prospective_execution_observer.py')
fixture_spec = importlib.util.spec_from_file_location('g4_inspect_fixture', Path(__file__).with_name('test_prospective_execution_observer.py'))
fixtures = importlib.util.module_from_spec(fixture_spec)
fixture_spec.loader.exec_module(fixtures)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(m.canonical(value) + b'\n')
    return {'path': str(path), 'sha256': m.sha(path)}


def test_portable_io_preserves_hash_bound_original_paths(tmp_path):
    old = Path('/original/research')
    retained = tmp_path / 'retained'; retained.mkdir()
    (retained / 'proof.json').write_text('{"path":"/original/research/x"}')
    portable = m.Portable([(str(old), str(retained))])
    p = portable.path_class()(old / 'proof.json')
    assert str(p) == '/original/research/proof.json'
    assert p.read_bytes() == (retained / 'proof.json').read_bytes()
    assert p.exists() and p.is_file()
    assert portable.ref({'path': str(p), 'sha256': m.sha(retained / 'proof.json')}) == retained / 'proof.json'
    with pytest.raises(ValueError, match='hash differs'):
        portable.ref({'path': str(p), 'sha256': 'f' * 64})


def test_portable_ref_rejects_symlink_and_escape(tmp_path):
    (tmp_path / 'target').write_text('data')
    (tmp_path / 'alias').symlink_to(tmp_path / 'target')
    portable = m.Portable([('/old', str(tmp_path))])
    for value in ('/old/alias', '/old/../target'):
        with pytest.raises(ValueError): portable.path(value)


def test_archived_controller_not_mutable_checkout_is_replay_root(tmp_path):
    content = b'old source\n'
    source_files = {'src/controller.py': hashlib.sha256(content).hexdigest()}
    controller = {'source_files': source_files, 'source_manifest_sha256': hashlib.sha256(m.canonical(source_files)).hexdigest()}
    archive = tmp_path / 'source.tar'
    with tarfile.open(archive, 'w') as stream:
        member = tarfile.TarInfo('src/controller.py'); member.size = len(content)
        stream.addfile(member, io.BytesIO(content))
    destination = tmp_path / 'archived'
    m.extract_controller(archive, controller, destination)
    assert (destination / 'src/controller.py').read_bytes() == content
    controller['source_files']['src/controller.py'] = '0' * 64
    with pytest.raises(ValueError, match='source hash'):
        m.extract_controller(archive, controller, destination)


@pytest.fixture
def receipt_tree(tmp_path):
    root = tmp_path / 'episodes/officeval_003.r00'; root.mkdir(parents=True)
    external = tmp_path / 'external/officeval_003.r00'; external.mkdir(parents=True)
    eid, fs, cs = 'ep_' + 'e' * 32, 'f' * 64, 'a' * 64
    freeze_path = tmp_path / 'freeze.json'; freeze_path.write_text('{}')
    frozen = {'actor_image_id': fixtures.IMAGE, 'actor_image_env': ['PATH=/bin', 'GPG_KEY=public-signing-key'],
              'actor_user': '501:20', '_freeze_path': str(freeze_path),
              'native_execution': {'dependency_identity': {'version': '0.73.1', 'installed_tree_sha256': 'c' * 64}}}
    row = {'episode_id': eid, 'task_id': 'officeval_003', 'config_sha256': cs, 'path': str(root)}
    binding = {'episode_id': eid, 'task_id': row['task_id'], 'config_sha256': cs,
               'freeze': {'path': str(freeze_path), 'sha256': fs}, 'external_dir': str(external),
               'workspace': str(root.parent / ('.l1-work-' + eid))}
    ref = write(external / 'episode_binding.json', binding)
    anchor = write(root / 'runtime/prospective/anchor.json', {**binding, 'external_receipt': ref})
    anchor['path'] = 'runtime/prospective/anchor.json'
    write(external / 'native_identity.json', {'schema': 'prospective-native-installed-v1',
          'episode_id': eid, 'config_sha256': cs, 'freeze_sha256': fs,
          'before_container_or_actor_send': True, 'dependency_identity': frozen['native_execution']['dependency_identity']})
    write(root / 'manifest.json', {'agent': {'native_validation': frozen['native_execution']['dependency_identity']}})
    events = [{'kind': 'provenance.bound', 'seq': 1, 'data': {'anchor': anchor}}]
    checker = object.__new__(observer.ProspectiveExecutionObserver)
    checker.frozen, checker.workspace = frozen, Path(binding['workspace'])
    for n, actor in enumerate((False, True)):
        cid = str(n + 1) * 64
        inspect = fixtures.inspection(checker, actor=actor); inspect['Id'] = cid
        value = {'schema': 'prospective-container-before-start-v1', 'episode_id': eid,
                 'task_id': row['task_id'], 'config_sha256': cs, 'freeze_sha256': fs,
                 'container_id': cid, 'before_start': True, 'role': 'actor_tool' if actor else 'environment_probe',
                 'inspection': inspect}
        ref = write(external / (cid + '.json'), value)
        internal = write(root / 'runtime/prospective/containers' / (cid + '.json'),
                         {'container_id': cid, 'inspection': inspect,
                          'receipt': {**ref, 'episode_id': eid, 'container_id': cid, 'freeze_sha256': fs}})
        internal['path'] = 'runtime/prospective/containers/' + cid + '.json'
        mounts = [] if not actor else [{'target': '/workspace', 'read_only': False, 'source_name': '.l1-work-' + eid}]
        events.extend([{'kind': 'runtime.container.created', 'seq': 2 + 2*n,
                        'data': {'container_id': cid, 'image_id': fixtures.IMAGE, 'mounts': mounts}},
                       {'kind': 'runtime.container.inspected', 'seq': 3 + 2*n,
                        'data': {'container_id': cid, 'inspection': internal}}])
    return frozen, fs, row, events, root, external, m.Portable([]), observer


def test_complete_exact_external_inspections_verify(receipt_tree):
    assert m.receipt_binding(*receipt_tree, set()) == 2


@pytest.mark.parametrize('mutation', ['native', 'foreign', 'missing', 'duplicate', 'cross_slot', 'changed_inspect', 'anchor'])
def test_receipt_boundary_tampering_fails_closed(receipt_tree, mutation):
    frozen, fs, row, events, root, external, portable, target = receipt_tree
    if mutation == 'native':
        p = external / 'native_identity.json'; value = m.read(p)
        value['dependency_identity']['installed_tree_sha256'] = '9' * 64; write(p, value)
    elif mutation == 'foreign': write(external / 'foreign.json', {})
    elif mutation == 'missing': (external / ('2' * 64 + '.json')).unlink()
    elif mutation == 'duplicate': events.append(copy.deepcopy(events[-1]))
    elif mutation == 'cross_slot': row['episode_id'] = 'ep_foreign'
    elif mutation == 'anchor':
        value = m.read(external / 'episode_binding.json'); value['task_id'] = 'officeval_999'
        write(external / 'episode_binding.json', value)
    else:
        p = external / ('2' * 64 + '.json'); value = m.read(p)
        value['inspection']['HostConfig']['Privileged'] = True; write(p, value)
    with pytest.raises((ValueError, FileNotFoundError)):
        m.receipt_binding(frozen, fs, row, events, root, external, portable, target, set())


@pytest.mark.parametrize('mutation', ['mount', 'env', 'image', 'network', 'user', 'probe'])
def test_frozen_pure_inspection_rejects_unsafe_actual_receipt(receipt_tree, mutation):
    frozen, _, row, _, root, external, _, target = receipt_tree
    checker = object.__new__(target.ProspectiveExecutionObserver)
    checker.frozen, checker.workspace = frozen, Path(root.parent / ('.l1-work-' + row['episode_id']))
    value = fixtures.inspection(checker, actor=mutation != 'probe')
    if mutation == 'mount': value['Mounts'][0]['Source'] = '/private/grader'
    if mutation == 'env': value['Config']['Env'].append('RUBRIC=secret')
    if mutation == 'image': value['Image'] = 'sha256:' + '0' * 64
    if mutation == 'network': value['HostConfig']['NetworkMode'] = 'host'
    if mutation == 'user': value['Config']['User'] = '0:0'
    if mutation == 'probe': value['Config']['Cmd'][-1] = 'print("extra")'
    with pytest.raises(ValueError): checker.check_inspection(fixtures.CID, value)

@pytest.mark.parametrize('name,kind', [('../escape.py', 'file'), ('link.py', 'symlink'), ('duplicate.py', 'duplicate')])
def test_controller_archive_refuses_unsafe_or_duplicate_members(tmp_path, name, kind):
    content = b'code'
    expected = {name: hashlib.sha256(content).hexdigest()}
    controller = {'source_files': expected, 'source_manifest_sha256': hashlib.sha256(m.canonical(expected)).hexdigest()}
    archive = tmp_path / 'source.tar'
    with tarfile.open(archive, 'w') as stream:
        for _ in range(2 if kind == 'duplicate' else 1):
            member = tarfile.TarInfo(name)
            if kind == 'symlink':
                member.type = tarfile.SYMTYPE; member.linkname = '/secret'
                stream.addfile(member)
            else:
                member.size = len(content); stream.addfile(member, io.BytesIO(content))
    with pytest.raises(ValueError):
        m.extract_controller(archive, controller, tmp_path / 'out')
