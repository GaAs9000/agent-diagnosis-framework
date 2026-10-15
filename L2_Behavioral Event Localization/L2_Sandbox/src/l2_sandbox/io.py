"""Minimal JSON/JSONL I/O, canonical hashing, and input-digest provenance headers (L2 v0.3)."""
from __future__ import annotations
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from . import CODE_VERSION


class L2Error(RuntimeError):
    pass


class IntegrityError(L2Error):
    pass


class Unsupported(L2Error):
    pass


class InvalidOutput(L2Error):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def loads(data: bytes | str) -> Any:
    def _no_dups(pairs):
        d = {}
        for k, v in pairs:
            if k in d:
                raise IntegrityError(f'Duplicate JSON key: {k}')
            d[k] = v
        return d

    def _no_nan(token: str):
        raise IntegrityError(f'Non-finite JSON constant: {token}')

    try:
        return json.loads(data, object_pairs_hook=_no_dups, parse_constant=_no_nan)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise IntegrityError(f'Invalid JSON: {exc}') from exc


def read_json(path: Path) -> Any:
    return loads(Path(path).read_bytes())


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for idx, line in enumerate(Path(path).read_bytes().splitlines()):
        if not line.strip():
            continue
        obj = loads(line)
        if not isinstance(obj, dict):
            raise IntegrityError(f'JSONL line {idx + 1} is not an object')
        rows.append(obj)
    return rows


def atomic_write(path: Path, data: bytes) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + p.name + '.', dir=p.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
        os.replace(name, p)
    finally:
        Path(name).unlink(missing_ok=True)
    return p


def write_json(path: Path, value: Any) -> Path:
    return atomic_write(path, canonical(value) + b'\n')


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    return atomic_write(path, b''.join(canonical(r) + b'\n' for r in rows))


def with_header(input_sha256: dict[str, str] | str, payload: dict[str, Any]) -> dict[str, Any]:
    """Prepends input_sha256 and code_version to any L2 JSON artifact (Section 7.1)."""
    return {
        'input_sha256': input_sha256,
        'code_version': CODE_VERSION,
        **payload,
    }


def validate_output_location(source: Path, output: Path) -> None:
    """Keep generated artifacts disjoint from immutable input, including symlink aliases."""
    source, output = Path(source).resolve(), Path(output).absolute()
    resolved = output.resolve()
    if resolved.is_relative_to(source) or source.is_relative_to(resolved):
        raise IntegrityError('L2 output and L1 source must be disjoint')
    # Only owned write directories need containment; unrelated symlinks are harmless.
    for name in ('values', 'hypotheses', 'branches'):
        if not (output / name).resolve().is_relative_to(resolved):
            raise IntegrityError(f'L2 owned output directory redirects outside output: {name}')
