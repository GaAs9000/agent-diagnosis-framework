"""CPU-only branch contract tests; reuse the original actual-inspect fixture."""
import copy
import json
from pathlib import Path

import pytest

from l2_sandbox import cross_actor_observer as m
from l2_sandbox.v3_execution import digest, seal
from test_prospective_execution_observer import inspection, IMAGE, CID


@pytest.fixture
def branch(tmp_path):
    config = {'benchmark': {'task_id': 'officeval_003'},
              'actor': {'system_prompt': 'branch prompt', 'intervention': 'S'},
              'runtime': {'adapter': 'docker-cold', 'image': IMAGE,
                          'memory': '2g', 'cpus': 2, 'pids_limit': 128}}
    refs = {}
    for key in ('parent_index', 'checkpoint', 'wrapper', 'actor_image_inspect'):
        path = tmp_path / (key + '.json')
        payload = {'Id': IMAGE, 'Config': {'Env': ['PATH=/bin', 'GPG_KEY=public-signing-key']}} if key == 'actor_image_inspect' else {'key': key}
        path.write_text(json.dumps(payload))
        refs[key] = {'path': str(path), 'sha256': m._original.sha(path)}
    source = tmp_path / 'runner.py'; source.write_text('frozen runner')
    sources = [m._SCRIPT, Path(m.__file__).resolve(), source]
    contract = seal({**refs, 'source_files': {str(p): m._original.sha(p) for p in sources},
        'bound_files': {r['path']: r['sha256'] for r in refs.values()},
        'expected_config_sha256': digest(config), 'actor_image_id': IMAGE,
        'actor_image_env': ['PATH=/bin', 'GPG_KEY=public-signing-key'],
        'actor_user': m._original.host_user(),
        'native_dependency_identity': {'version': 'recorded-opencode', 'installed_tree_sha256': 'e' * 64}})
    return contract, config


def ready(branch, tmp_path):
    contract, config = branch
    observer = m.CrossActorExecutionObserver(contract, tmp_path / 'receipts')
    observer.validate(config)
    workspace = tmp_path / 'workspace'; workspace.mkdir()
    observer.bind_workspace(workspace)
    binding = observer.bind_episode('ep_branch', digest(config))
    observer.validate_native_identity(contract['native_dependency_identity'])
    return observer, binding


def actual(observer, actor=True):
    value = inspection(observer, actor)
    value['Config']['User'] = observer.frozen['actor_user']
    return value


@pytest.mark.parametrize('actor', [True, False])
def test_receipts_bind_full_config_contract_and_actual_inspection(branch, tmp_path, actor):
    observer, binding = ready(branch, tmp_path)
    value = actual(observer, actor)
    receipt = observer.observe_container(CID, value)
    saved = json.loads(Path(receipt['path']).read_text())
    assert saved['inspection'] == value and saved['before_start'] is True
    assert saved['contract_sha256'] == digest(branch[0])
    assert binding['config'] == branch[1]
    assert binding['contract_artifact_sha256'] == branch[0]['artifact_sha256']
    assert receipt['sha256'] == m._original.sha(receipt['path'])
    with pytest.raises(ValueError, match='duplicate'):
        observer.observe_container(CID, value)
    with pytest.raises(ValueError, match='one episode'):
        observer.validate(branch[1])


@pytest.mark.parametrize('key', ['parent_index', 'checkpoint', 'wrapper', 'actor_image_inspect'])
def test_reference_drift_blocks_before_receipt(branch, tmp_path, key):
    observer, _ = ready(branch, tmp_path)
    Path(branch[0][key]['path']).write_text('changed')
    with pytest.raises(ValueError, match='file changed'):
        observer.observe_container(CID, actual(observer))
    assert not (observer.external_dir / (CID + '.json')).exists()


def test_source_drift_and_missing_observer_inventory_fail(branch, tmp_path):
    contract, _ = branch
    missing = copy.deepcopy(contract); del missing['source_files'][str(m._SCRIPT)]
    with pytest.raises(ValueError, match='observer source'):
        m.CrossActorExecutionObserver(seal(missing), tmp_path / 'missing')
    observer, _ = ready(branch, tmp_path)
    (tmp_path / 'runner.py').write_text('changed runner')
    with pytest.raises(ValueError, match='source_files file changed'):
        observer.observe_container(CID, actual(observer))


def test_seal_and_complete_resolved_config_are_checked(branch, tmp_path):
    contract, config = branch
    tampered = copy.deepcopy(contract); tampered['actor_user'] = '1:1'
    with pytest.raises(ValueError, match='seal'):
        m.CrossActorExecutionObserver(tampered, tmp_path / 'invalid')
    observer = m.CrossActorExecutionObserver(contract, tmp_path / 'receipts')
    config['actor']['system_prompt'] += ' unapproved'
    with pytest.raises(ValueError, match='configuration differs'):
        observer.validate(config)
    assert not observer.external_dir.exists()


@pytest.mark.parametrize('mutation', ['mount', 'source_mount', 'network', 'env', 'user', 'image', 'running', 'resources'])
def test_original_isolation_checks_reject_actual_drift(branch, tmp_path, mutation):
    observer, _ = ready(branch, tmp_path)
    value = actual(observer)
    if mutation == 'mount': value['Mounts'].append(copy.deepcopy(value['Mounts'][0]))
    if mutation == 'source_mount': value['Mounts'][0]['Source'] = '/app/source'
    if mutation == 'network': value['HostConfig']['NetworkMode'] = 'host'
    if mutation == 'env': value['Config']['Env'].append('PRIVATE_KEY=unapproved')
    if mutation == 'user': value['Config']['User'] = '0:0'
    if mutation == 'image': value['Image'] = 'sha256:' + 'f' * 64
    if mutation == 'running': value['State']['Running'] = True
    if mutation == 'resources': value['HostConfig']['Memory'] = 0
    with pytest.raises(ValueError): observer.observe_container(CID, value)
    assert not (observer.external_dir / (CID + '.json')).exists()


def test_native_identity_requires_exact_installed_tree(branch, tmp_path):
    observer = m.CrossActorExecutionObserver(branch[0], tmp_path / 'receipts')
    observer.validate(branch[1])
    workspace = tmp_path / 'workspace'; workspace.mkdir()
    observer.bind_workspace(workspace); observer.bind_episode('ep_branch', digest(branch[1]))
    with pytest.raises(ValueError, match='native validator'):
        observer.validate_native_identity({'version': 'wrong'})
    with pytest.raises(ValueError, match='unbound'):
        observer.observe_container(CID, actual(observer))
    assert not (observer.external_dir / 'native_identity.json').exists()


def test_receipts_must_be_new_and_disjoint_from_workspace(branch, tmp_path):
    existing = tmp_path / 'existing'; existing.mkdir()
    with pytest.raises(ValueError, match='must be new'):
        m.CrossActorExecutionObserver(branch[0], existing)
    observer = m.CrossActorExecutionObserver(branch[0], tmp_path / 'workspace' / 'receipts')
    observer.validate(branch[1])
    workspace = tmp_path / 'workspace'; workspace.mkdir()
    with pytest.raises(ValueError, match='disjoint'):
        observer.bind_workspace(workspace)


def test_caller_mutation_does_not_change_frozen_contract(branch, tmp_path):
    observer = m.CrossActorExecutionObserver(branch[0], tmp_path / 'receipts')
    branch[0]['native_dependency_identity']['version'] = 'mutated caller object'
    observer.validate(branch[1])
