"""Optional installed-L1 integration checks; never send model requests."""
import copy
from pathlib import Path

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


@pytest.mark.parametrize('task_id', ['officeval_004', 'officeval_010', 'officeval_050', 'officeval_085', 'officeval_095'])
def test_installed_condition_adapter_preserves_native_task(native_config, task_id):
    from importlib.metadata import entry_points
    from c3_sandbox.l1.benchmarks.officeval.adapter import NativeOfficeVal
    from c3_sandbox.l1.benchmarks.registry import registry
    entries = entry_points(group='l1.benchmarks')
    if not any(e.name == 'l3-officeval' for e in entries):
        pytest.skip('Install L3 editable package to test its actual entry point')
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
