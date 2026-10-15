"""Small, strict persistence helpers; no model or L2 dependency."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile


class L3Error(RuntimeError):
    pass


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def safe_path(path: Path) -> Path:
    path = Path(path).absolute()
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise L3Error('Symlink paths are not accepted')
    return path


def sha_file(path: Path) -> str:
    path = safe_path(path)
    if not path.is_file():
        raise L3Error('Frozen input is not a regular file: ' + str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    def invalid(value):
        raise L3Error('Nonfinite JSON number: ' + value)
    return json.loads(safe_path(path).read_text(encoding='utf-8'), parse_constant=invalid)


def write(path: Path, value):
    path = safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(canonical(value) + b'\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
