"""Freeze failures must emit a blocked gate before any model request."""
import asyncio
import importlib.util
import json
from pathlib import Path


def test_changed_source_records_blocked_without_generation(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / 'scripts/run_r12_g0.py'
    spec = importlib.util.spec_from_file_location('r12_g0_freeze_fixture', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    source = tmp_path / 'fixture.py'
    source.write_text('before')
    manifest = {'system': {'provenance': {'scaffold_source_files': {'fixture.py': module.sha(source)}}}}
    root = tmp_path / 'run'
    (root / 'system_qualification').mkdir(parents=True)
    module.write(root / 'system_manifest.json', manifest)
    module.write(root / 'system_qualification/qualification_protocol.json', {'fixture': True})
    module.write(root / 'system_qualification/freeze_event.json', {
        'system_manifest_sha256': module.sha(root / 'system_manifest.json'),
        'qualification_protocol_sha256': module.sha(root / 'system_qualification/qualification_protocol.json')})
    source.write_text('after')
    called = []
    async def forbidden(*_args):
        called.append(True)
        raise AssertionError('model must not execute')
    monkeypatch.setattr(module, 'execute_qualification', forbidden)
    asyncio.run(module.run(root))
    gate = json.loads((root / 'system_qualification/gate.json').read_bytes())
    assert not called
    assert gate['status'] == 'BLOCKED_SYSTEM' and not gate['candidate_execution_admitted']
