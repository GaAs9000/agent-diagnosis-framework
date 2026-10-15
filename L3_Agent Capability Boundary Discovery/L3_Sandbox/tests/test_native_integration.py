"""Optional installed-L1 integration checks; never send model requests."""
import copy
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from l3_sandbox.common import digest, read, sha_file
from l3_sandbox.conditions import HEADER, TRANSFORM_VERSION, placebo_context, structure_context


WORKSPACE = Path(__file__).resolve().parents[3]
L1 = WORKSPACE / 'L1_sandbox/C3_Experimental_Sandbox'


@pytest.fixture
def native_config():
    from c3_sandbox.l1.core.config import resolve
    path = L1 / 'configs/l1/officeval.gemini-35lite.t1.json'
    if not (L1 / 'benchmarks/officeval/data/data.lock.json').is_file():
        pytest.skip('Locked local OfficeVal assets unavailable')
    return resolve(path)


def _require_native_task_assets(config, task_id):
    """Skip optional local dependencies; leave validation failures to the adapter."""
    source = Path(config['source_dir'])
    if not source.is_dir():
        pytest.skip('Pinned official OfficeVal source checkout unavailable')
    revision = subprocess.run(
        ['git', '-C', str(source), 'rev-parse', 'HEAD'],
        capture_output=True, text=True, timeout=30)
    if revision.returncode or revision.stdout.strip() != config['source_commit']:
        pytest.skip('Pinned official OfficeVal source checkout unavailable or at wrong commit')

    data = Path(config['data_dir'])
    metadata = {}
    for kind in ('tasks', 'rubrics'):
        candidates = [data / kind / f'{task_id}.json',
                      data / 'data' / kind / f'{task_id}.json']
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            pytest.skip(f'Local OfficeVal {kind} asset unavailable: {task_id}')
        metadata[kind] = path
    if not (source / 'verifiers' / f'{task_id}_verifier.py').is_file():
        pytest.skip(f'Official OfficeVal verifier unavailable: {task_id}')
    # Malformed metadata, modified source, and mismatching asset hashes are real
    # failures, not missing-installation skips.
    for entry in read(metadata['tasks'])['origin_files']:
        name = entry['dest']
        assets = Path(config['assets_dir'])
        candidates = [assets / task_id / name, assets / 'task_files' / task_id / name]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            pytest.skip(f'Local OfficeVal input asset unavailable: {task_id}/{name}')
        with path.open('rb') as handle:
            if handle.read(100).startswith(b'version https://git-lfs.github.com/spec/v1'):
                pytest.skip(f'Local OfficeVal input is an unresolved LFS pointer: {task_id}/{name}')


@pytest.mark.parametrize('case', ['missing_source', 'wrong_commit', 'missing_input', 'malformed_metadata'])
def test_optional_native_dependencies_do_not_hide_metadata_errors(tmp_path, monkeypatch, case):
    source = tmp_path / 'source'
    config = {'source_dir': str(source), 'source_commit': 'a' * 40,
              'data_dir': str(tmp_path / 'data'), 'assets_dir': str(tmp_path / 'assets')}
    if case != 'missing_source':
        (source / 'verifiers').mkdir(parents=True)
        (source / 'verifiers/officeval_004_verifier.py').touch()
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout=('b' if case == 'wrong_commit' else 'a') * 40))
    for kind in ('tasks', 'rubrics'):
        folder = tmp_path / 'data' / kind
        folder.mkdir(parents=True)
        (folder / 'officeval_004.json').write_text(
            '{}' if case == 'malformed_metadata' else '{"origin_files": [{"dest": "input.docx"}]}')
    expected = KeyError if case == 'malformed_metadata' else pytest.skip.Exception
    with pytest.raises(expected):
        _require_native_task_assets(config, 'officeval_004')


@pytest.mark.parametrize('task_id', ['officeval_004', 'officeval_010', 'officeval_050', 'officeval_085', 'officeval_095'])
def test_installed_condition_adapter_preserves_native_task(native_config, task_id):
    from importlib.metadata import entry_points
    from c3_sandbox.l1.benchmarks.officeval.adapter import NativeOfficeVal
    from c3_sandbox.l1.benchmarks.registry import registry
    entries = entry_points(group='l1.benchmarks')
    if not any(e.name == 'l3-officeval' for e in entries):
        pytest.skip('Install L3 editable package to test its actual entry point')
    _require_native_task_assets(native_config['benchmark'], task_id)
    native = NativeOfficeVal(native_config['benchmark']).load(task_id)
    files = [a.path for a in native.assets] + list(native.private_files.values())
    before = {str(p): sha_file(p) for p in files}
    structure = structure_context(native)
    placebo = placebo_context(structure)
    assert len(placebo) == len(structure)
    # Character matching is the declared control; bytes and tokenizer lengths
    # can differ when native names include non-ASCII characters.
    assert len(placebo.encode()) <= len(structure.encode())
    for context, appendix in [('original', ''), ('structure', structure), ('placebo', placebo)]:
        config = copy.deepcopy(native_config['benchmark'])
        config.update(adapter='l3-officeval', task_id=task_id, options={
            'l3_context': context, 'l3_appendix': appendix,
            'l3_transform_version': TRANSFORM_VERSION, 'l3_condition_id': context})
        plugins = registry()
        plugins.load_installed()
        task = plugins.create('l3-officeval', config).load(task_id)
        assert task.task_id == native.task_id
        assert task.benchmark == native.benchmark
        assert task.assets == native.assets
        assert task.private_files == native.private_files
        assert task.evaluation == native.evaluation
        assert task.requirements == native.requirements
        assert task.prompt == native.prompt + (HEADER + appendix if appendix else '')
        assert task.provenance['l3_condition']['appendix_characters'] == len(appendix)
        assert task.provenance['l3_condition']['appendix_utf8_bytes'] == len(appendix.encode())
    assert {str(p): sha_file(p) for p in files} == before


def test_resolved_native_configuration_digest_matches_l1(native_config):
    from c3_sandbox.l1.core.config import validate
    from c3_sandbox.l1.storage import digest as native_digest
    validate(native_config)
    assert native_digest(native_config) == digest(native_config)


def test_real_package_verifier_signature_and_native_manifest():
    from l3_sandbox.run import _evidence, _verify_package
    package = L1 / 'runs/l1-accept-35lite-v2/episode'
    if not (package / 'manifest.json').is_file():
        pytest.skip('Frozen native L1 baseline unavailable')
    manifest = read(package / 'manifest.json')
    config = read(package / 'raw/resolved_config.json')
    assert manifest['task']['task_id'] == 'officeval_003'
    assert manifest['config_sha256'] == digest(config)
    expected = sha_file(package / 'evidence_index.json')
    assert _verify_package(package, expected)['verified'] is True
    _evidence(package, {'task_id': manifest['task']['task_id'],
                        'config_sha256': manifest['config_sha256']}, expected)
