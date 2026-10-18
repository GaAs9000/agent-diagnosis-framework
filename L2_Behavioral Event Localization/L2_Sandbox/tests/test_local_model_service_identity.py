import importlib.util
import json
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def identity(tmp_path, monkeypatch):
    capture = load('capture_local_model_identity')
    launcher = load('start_owned_native_service')
    root = tmp_path / 'revision'
    root.mkdir()
    blobs = tmp_path / 'blobs'
    blobs.mkdir()
    weights = {}
    for i in range(15):
        data = f'weight {i}'.encode()
        import hashlib
        blob = blobs / hashlib.sha256(data).hexdigest()
        blob.write_bytes(data)
        name = f'model-{i:05}.safetensors'
        (root / name).symlink_to(blob)
        weights[f'w{i}'] = name
    (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': weights}))
    (root / 'tokenizer.json').write_text('{}')
    executable = tmp_path / 'bin/vllm'
    executable.parent.mkdir()
    executable.write_text('#!/bin/python\n')
    runtime = dict(cli_shebang='#!/bin/python', resolved_python='/bin/python', python_version='fake', python_prefix='/fake', distribution_root='/fake/site', distribution_path='/fake/dist')
    model_files = [capture.capture_file(p, require_blob_hash=p.suffix == '.safetensors') for p in root.iterdir()]
    model_manifest = tmp_path / 'model_files.json'
    capture.write_json_exclusive(model_manifest, model_files)
    vllm_manifest = tmp_path / 'vllm_files.json'
    capture.write_json_exclusive(vllm_manifest, [capture.capture_file(executable)])
    monkeypatch.setattr(launcher, 'MODEL_ROOT', str(root))
    command = launcher.command_after_timeout()
    command[3] = str(executable)
    monkeypatch.setattr(launcher, 'command_after_timeout', lambda: command.copy())
    monkeypatch.setattr(launcher.subprocess, 'check_output', lambda *args, **kwargs: json.dumps(dict(version='0.30.0', **{k: v for k, v in runtime.items() if k != 'cli_shebang'})))
    record = {'model': {'resolved_weight_path': str(root), 'reported_revision': root.name,
                       'validated_files': len(model_files), 'weight_digest': capture.sha256(root / 'model.safetensors.index.json'),
                       'tokenizer_digest': capture.sha256(root / 'tokenizer.json'),
                       'file_manifest': {'path': str(model_manifest), 'sha256': capture.sha256(model_manifest)}},
              'vllm': {'executable': str(executable), 'version': '0.30.0', 'script_sha256': capture.sha256(executable), 'runtime': runtime,
                       'file_manifest': {'path': str(vllm_manifest), 'sha256': capture.sha256(vllm_manifest)}},
              'command_after_timeout': command.copy()}
    path = tmp_path / 'identity.json'
    capture.write_json_exclusive(path, record)
    reference = {'path': str(path), 'sha256': capture.sha256(path)}
    return capture, launcher, root, record, path, reference


def test_receipt_validates_without_model_or_gpu_import(identity):
    _, launcher, _, _, _, reference = identity
    assert launcher.verify_identity(reference) == reference['sha256']


def test_changed_weight_metadata_rejected_before_gpu_or_popen(identity, monkeypatch):
    _, launcher, root, _, _, reference = identity
    (root / 'model-00000.safetensors').resolve().write_bytes(b'changed weight')
    monkeypatch.setattr(launcher.subprocess, 'Popen', lambda *a, **kw: pytest.fail('Popen forbidden'))
    with pytest.raises(ValueError, match='file changed'):
        launcher.launch({'phase_authorized': True, 'phase_frozen': True, 'identity_receipt': reference})


@pytest.mark.parametrize('mutation', ['command', 'model', 'version', 'manifest', 'receipt'])
def test_mismatches_rejected(identity, mutation):
    capture, launcher, _, record, path, reference = identity
    if mutation == 'command':
        record['command_after_timeout'][-1] = 'float16'
    elif mutation == 'model':
        record['model']['reported_revision'] = 'other'
    elif mutation == 'version':
        record['vllm']['version'] = 'other'
    elif mutation == 'manifest':
        record['model']['file_manifest']['sha256'] = '0' * 64
    path.write_text(json.dumps(record))
    if mutation != 'receipt':
        reference['sha256'] = capture.sha256(path)
    with pytest.raises(ValueError):
        launcher.verify_identity(reference)


def test_blob_name_must_match_complete_bytes(tmp_path):
    capture = load('capture_local_model_identity')
    wrong = tmp_path / ('0' * 64)
    wrong.write_bytes(b'actual data')
    with pytest.raises(ValueError, match='blob name'):
        capture.capture_file(wrong, require_blob_hash=True)


def test_capture_detects_mutation_during_read(tmp_path, monkeypatch):
    capture = load('capture_local_model_identity')
    source = tmp_path / 'source'
    source.write_text('before')
    def changed(path):
        path.write_text('after longer')
        return '0' * 64
    monkeypatch.setattr(capture, 'sha256', changed)
    with pytest.raises(ValueError, match='changed during'):
        capture.capture_file(source)


def test_capture_cli_default_inert(capsys):
    capture = load('capture_local_model_identity')
    capture.main([])
    assert json.loads(capsys.readouterr().out)['status'] == 'not_executed'


def test_fixed_command_contains_frozen_model_and_generation_contract():
    launcher = load('start_owned_native_service')
    argv = launcher.command_after_timeout()
    assert argv[:6] == ['nice', '-n', '10', '/home/jiasheren2/envs/idea-n1/bin/vllm', 'serve', launcher.MODEL_ROOT]
    assert argv[argv.index('--max-model-len') + 1] == '131072'
    assert argv[argv.index('--tool-call-parser') + 1] == 'qwen3_xml'


def test_nested_hf_blob_uses_logical_lfs_digest_not_terminal_cas_name(tmp_path):
    capture = load('capture_local_model_identity')
    import hashlib
    data = b'full weight bytes'
    digest = hashlib.sha256(data).hexdigest()
    terminal = tmp_path / 'dedup-other-algorithm'
    terminal.write_bytes(data)
    logical = tmp_path / digest
    logical.symlink_to(terminal)
    snapshot = tmp_path / 'model.safetensors'
    snapshot.symlink_to(logical)
    record = capture.capture_file(snapshot, require_blob_hash=True)
    assert record['sha256'] == digest
    assert record['hf_blob_path'] == str(logical)
    assert record['realpath'] == str(terminal)


def test_added_model_configuration_rejected(identity):
    _, launcher, root, _, _, reference = identity
    (root / 'new_config.json').write_text('{}')
    with pytest.raises(ValueError, match='inventory mismatch'):
        launcher.verify_identity(reference)


def test_retargeted_symlink_chain_same_final_file_rejected(identity):
    _, launcher, root, _, _, reference = identity
    original = root / 'model-00000.safetensors'
    terminal = original.resolve()
    intermediate = root.parent / 'intermediate'
    intermediate.symlink_to(terminal)
    original.unlink()
    original.symlink_to(intermediate)
    with pytest.raises(ValueError, match='file changed'):
        launcher.verify_identity(reference)


def test_same_version_other_installation_rejected(identity):
    capture, launcher, _, record, path, reference = identity
    record['vllm']['runtime'] = dict(record['vllm']['runtime'], distribution_root='/other/site-packages')
    path.write_text(json.dumps(record))
    reference['sha256'] = capture.sha256(path)
    with pytest.raises(ValueError, match='runtime installation'):
        launcher.verify_identity(reference)


def test_capture_rejects_different_cli_interpreter(tmp_path):
    capture = load('capture_local_model_identity')
    executable = tmp_path / 'vllm'
    executable.write_text('#!/other/python\n')
    with pytest.raises(ValueError, match='shebang environment'):
        capture.runtime_identity(executable)
