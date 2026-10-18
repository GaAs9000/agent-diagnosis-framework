#!/usr/bin/env python3
"""Capture local model and installed vLLM bytes on CPU; inert without --execute."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys

MODEL_ROOT = Path('/home/jiasheren2/hf_cache/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9')
VLLM_EXECUTABLE = Path('/home/jiasheren2/envs/idea-n1/bin/vllm')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def stat_identity(path):
    stat = Path(path).stat()
    return dict(size=stat.st_size, dev=stat.st_dev, ino=stat.st_ino,
                mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)


def symlink_chain(path):
    current = Path(path).absolute()
    chain = []
    seen = set()
    while current.is_symlink():
        if str(current) in seen:
            raise ValueError('cyclic identity symlink')
        seen.add(str(current))
        target = str(current.readlink())
        chain.append(dict(path=str(current), target=target))
        current = Path(os.path.abspath(current.parent / target))
    return chain


def capture_file(path, *, require_blob_hash=False):
    path = Path(path).absolute()
    realpath = str(path.resolve(strict=True))
    chain = symlink_chain(path)
    before = stat_identity(path)
    digest = sha256(path)
    after = stat_identity(path)
    if before != after or str(path.resolve(strict=True)) != realpath or symlink_chain(path) != chain:
        raise ValueError(f'file changed during capture: {path}')
    # HF's logical LFS blob may itself link into a deduplicated CAS with a
    # different naming algorithm. Compare the snapshot's immediate HF target.
    hf_blob = Path(os.path.abspath(path.parent / path.readlink())) if path.is_symlink() else path
    if require_blob_hash and hf_blob.name != digest:
        raise ValueError(f'HF shard blob name differs from full-byte SHA256: {path}')
    record = dict(path=str(path), realpath=realpath, symlink_chain=chain, sha256=digest, **after)
    if require_blob_hash:
        record['hf_blob_path'] = str(hf_blob)
    return record


def write_json_exclusive(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def runtime_identity(executable):
    executable = Path(executable)
    shebang = executable.open().readline().rstrip('\n')
    interpreter = shebang.removeprefix('#!')
    if (not shebang.startswith('#!/') or ' ' in interpreter or
            Path(interpreter).resolve() != Path(sys.executable).resolve() or
            Path(interpreter).parent.parent != Path(sys.prefix)):
        raise ValueError('capture Python must be the vLLM CLI shebang environment')
    distribution = importlib.metadata.distribution('vllm')
    root = Path(distribution.locate_file('')).resolve()
    if not root.is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError('vLLM distribution outside capture Python environment')
    return dict(capture_python=sys.executable, resolved_python=str(Path(sys.executable).resolve()),
                python_version=sys.version, python_prefix=sys.prefix,
                distribution_root=str(root), distribution_path=str(Path(distribution._path).resolve()),
                cli_shebang=shebang)


def capture(output, model_root=MODEL_ROOT, executable=VLLM_EXECUTABLE):
    model_root, executable, output = Path(model_root), Path(executable), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    runtime = runtime_identity(executable)
    index = model_root / 'model.safetensors.index.json'
    mapping = json.loads(index.read_text())['weight_map']
    shards = set(mapping.values())
    actual = {p.name for p in model_root.glob('*.safetensors')}
    if len(shards) != 15 or shards != actual or any(Path(s).name != s for s in shards):
        raise ValueError('weight index must reference exactly the 15 installed shards')
    model_files = [capture_file(p, require_blob_hash=p.name in shards)
                   for p in sorted(model_root.iterdir()) if p.is_file()]
    model_manifest = output / 'model_files.json'
    write_json_exclusive(model_manifest, model_files)
    distribution = importlib.metadata.distribution('vllm')
    installed = {Path(distribution.locate_file(p)).absolute()
                 for p in (distribution.files or [])
                 if '__pycache__' not in p.parts and p.suffix != '.pyc'}
    installed.update((executable.absolute(), Path(sys.executable).absolute()))
    missing = [str(p) for p in installed if not p.is_file()]
    if missing:
        raise ValueError(f'installed distribution files missing: {missing}')
    vllm_files = [capture_file(p) for p in sorted(installed)]
    vllm_manifest = output / 'vllm_files.json'
    write_json_exclusive(vllm_manifest, vllm_files)
    spec = importlib.util.spec_from_file_location('owned_launcher', Path(__file__).with_name('start_owned_native_service.py'))
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    identity = dict(model=dict(
        reported_revision=model_root.name, weight_digest=sha256(index),
        tokenizer_digest=sha256(model_root / 'tokenizer.json'),
        resolved_weight_path=str(model_root.resolve()),
        file_manifest=dict(path=str(model_manifest), sha256=sha256(model_manifest)),
        digest_algorithm='sha256-full-file-bytes; weight_digest=index-json; shard-hf-blob-sha256',
        validated_files=len(model_files)),
        vllm=dict(executable=str(executable), version=distribution.version,
                  script_sha256=sha256(executable),
                  runtime=runtime,
                  file_manifest=dict(path=str(vllm_manifest), sha256=sha256(vllm_manifest))),
        command_after_timeout=launcher.command_after_timeout())
    write_json_exclusive(output / 'identity.json', identity)
    return dict(path=str(output / 'identity.json'), sha256=sha256(output / 'identity.json'),
                model_files=len(model_files), vllm_files=len(vllm_files))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status='not_executed')))
        return
    if not args.output:
        parser.error('--output is required with --execute')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU capture requires CUDA_VISIBLE_DEVICES empty')
    print(json.dumps(capture(args.output)))


if __name__ == '__main__':
    main()
