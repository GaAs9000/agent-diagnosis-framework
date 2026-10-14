"""Freeze one new execution system, then perform exactly one qualification attempt."""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import secrets
import shutil
import traceback

from c3_sandbox.evidence._storage import encode
from c3_sandbox.r1.qualification import image_identity
from c3_sandbox.r1.r12_qualification import SYSTEM_PROMPT, qualification_inputs, execute_qualification, sha, write
from c3_sandbox.r1.r12_system import (SAMPLING, SEED_DOMAINS, MODEL_REPOSITORY, MODEL_REVISION,
    MODEL_PATH, TOOL_PARSER, TOOL_PARSER_SHA256, validated_system, system_identity, fingerprint_model)
from c3_sandbox.r1.r12_service import observe_service

ROOT = Path(__file__).resolve().parents[1]


def freeze(root):
    definition_path = root / 'system_definition.json'
    if definition_path.exists():
        raise FileExistsError('system is already frozen')
    tools = [{'type': 'function', 'function': {'name': 'workspace_python',
        'description': 'Execute Python as a standalone script in the isolated workspace. Use print() for observations. Installed: python-docx, lxml, PyMuPDF (pymupdf), pdfplumber, Pillow, openpyxl, python-pptx. Returns stdout, stderr and exit code.',
        'parameters': {'type': 'object', 'properties': {'code': {'type': 'string'}},
                       'required': ['code'], 'additionalProperties': False}}}]
    source_paths = sorted((ROOT / 'src/c3_sandbox').rglob('*.py'))
    source_paths += [Path(__file__).resolve(), ROOT / 'docker/Dockerfile.r12-base',
        ROOT / 'requirements.local-arm64.lock.txt', ROOT / 'requirements.model.local-arm64.lock.txt']
    code_lock = {str(path.relative_to(ROOT)): sha(path) for path in source_paths}
    system = {'schema_version': 'r12-system-v1', 'system_id': 'c3-r12-qwen3-coder30b-a3b-4bit-v1',
        'model': {'repository': MODEL_REPOSITORY, 'revision': MODEL_REVISION, 'path': MODEL_PATH,
                  'quantization': '4bit', 'tool_parser': TOOL_PARSER, 'tool_parser_sha256': TOOL_PARSER_SHA256},
        'sampling': SAMPLING, 'seed_domains': SEED_DOMAINS, 'system_prompt': SYSTEM_PROMPT,
        'tool_schemas': tools, 'agent_image': image_identity('c3-r12-agent:locked'),
        'server': {'host': '127.0.0.1', 'port': 18080, 'tokenizer_port': 18081,
                   'prompt_cache_size': 0, 'prompt_concurrency': 1, 'decode_concurrency': 1,
                   'packages_lock': 'requirements.model.local-arm64.lock.txt'},
        'provenance': {'scaffold_source_files': code_lock, 'model_choice_sha256': sha(root / 'setup/model_choice.json'),
                       'stage': 'qualification before any new candidate selection or execution', 'modality': 'text'}}
    system = validated_system(system)
    fingerprint = fingerprint_model(system)
    serving = observe_service(system, int((root / 'setup/model.pid').read_text()))
    qualification_inputs(root / 'system_qualification', str(100000 + secrets.randbelow(900000)))
    protocol_path = root / 'system_qualification/qualification_protocol.json'
    bundle = {'system': system, 'system_contract_sha256': system_identity(system), 'model_fingerprint': fingerprint,
        'serving_observation': serving,
        'qualification_protocol_sha256': sha(protocol_path), 'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'candidate_pool_frozen': False, 'qualification_failure_action': 'BLOCKED_SYSTEM; no alternate model or candidate calls'}
    write(definition_path, system)
    write(root / 'system_manifest.json', bundle)
    for relative in code_lock:
        target = root / 'system_qualification/source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    write(root / 'system_qualification/freeze_event.json', {
        'system_manifest_sha256': sha(root / 'system_manifest.json'),
        'qualification_protocol_sha256': sha(protocol_path), 'timestamp_utc': bundle['frozen_at_utc']})
    definition_path.chmod(0o444)
    (root / 'system_manifest.json').chmod(0o444)
    print(json.dumps({'system_frozen': system['system_id'], 'system_identity': system_identity(system)}))


async def run(root):
    try:
        manifest = json.loads((root / 'system_manifest.json').read_bytes())
        event = json.loads((root / 'system_qualification/freeze_event.json').read_bytes())
        if sha(root / 'system_manifest.json') != event['system_manifest_sha256']:
            raise ValueError('frozen system manifest changed')
        if sha(root / 'system_qualification/qualification_protocol.json') != event['qualification_protocol_sha256']:
            raise ValueError('qualification rules changed')
        for relative, digest in manifest['system']['provenance']['scaffold_source_files'].items():
            if sha(ROOT / relative) != digest:
                raise ValueError('frozen scaffold changed: ' + relative)
        result = await execute_qualification(root / 'system_qualification', manifest)
    except Exception as error:
        result = {'status': 'blocked_system', 'candidate_execution_admitted': False,
                  'error_type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc()}
    gate = {'gate': 'R12-G0', 'status': 'PASS' if result.get('status') == 'passed' else 'BLOCKED_SYSTEM',
        'candidate_execution_admitted': result.get('candidate_execution_admitted', False),
        'qualification_result': result, 'no_alternate_model': True}
    (root / 'system_qualification').mkdir(parents=True, exist_ok=True)
    write(root / 'system_qualification/gate.json', gate)
    print(json.dumps(gate, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze','run'))
    parser.add_argument('--run-root', type=Path, default=ROOT / 'runs/e2-r1-2')
    args = parser.parse_args()
    root = args.run_root.absolute()
    if args.action == 'freeze':
        freeze(root)
    else:
        asyncio.run(run(root))


if __name__ == '__main__':
    main()
