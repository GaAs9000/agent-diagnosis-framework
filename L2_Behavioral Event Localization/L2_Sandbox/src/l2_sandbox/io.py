"""Strict JSON, bounded content reads, paths, new-only writes and append journals.

Trusted local controller/host is assumed. Hashes are not signatures. Rechecking
content detects mutations but is not a defense against a malicious kernel.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any

class L2Error(RuntimeError):
    pass
class IntegrityError(L2Error):
    pass
class AccessDenied(L2Error):
    pass
class Unsupported(L2Error):
    pass
class BudgetExceeded(L2Error):
    pass
class InvalidOutput(L2Error):
    pass

def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')

def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()

def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def now() -> str:
    return datetime.now(timezone.utc).isoformat()

def relative(name: str) -> str:
    if not isinstance(name, str) or not name or '\\' in name or '\x00' in name:
        raise IntegrityError('Invalid relative path')
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or str(p) != name or name == '.' or any(':' in v for v in p.parts):
        raise IntegrityError('Unsafe path: ' + name)
    return name

def safe(root: Path, name: str) -> Path:
    relative(name)
    root = Path(root).absolute()
    for p in (root, *root.parents):
        if p.is_symlink(): raise IntegrityError('Symlink root/parent rejected')
    p = root
    for part in PurePosixPath(name).parts:
        p /= part
        if p.is_symlink(): raise IntegrityError('Symlink rejected: ' + name)
    if not p.resolve().is_relative_to(root.resolve()): raise IntegrityError('Path escapes root')
    return p

def regular(p: Path):
    s = p.lstat()
    if not stat.S_ISREG(s.st_mode) or s.st_nlink != 1:
        raise IntegrityError('Only non-linked regular files allowed: ' + str(p))
    return s

def bytes_read(p: Path, limit: int = 64 * 1024 * 1024) -> bytes:
    regular(p)
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    with os.fdopen(os.open(p, flags), 'rb') as f:
        s = os.fstat(f.fileno())
        if not stat.S_ISREG(s.st_mode) or s.st_nlink != 1 or s.st_size > limit:
            raise IntegrityError('File type/size limit: ' + str(p))
        data = f.read(limit + 1)
        after = os.fstat(f.fileno())
    if len(data) > limit or len(data) != s.st_size or (s.st_mtime_ns, s.st_size) != (after.st_mtime_ns, after.st_size):
        raise IntegrityError('File changed or read limit exceeded')
    return data

def sha_file(p: Path) -> str:
    regular(p)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def loads(data: bytes | str):
    def pairs(items):
        d = {}
        for k, v in items:
            if k in d: raise IntegrityError('Duplicate JSON key: ' + k)
            d[k] = v
        return d
    def constant(s): raise IntegrityError('Non-finite JSON: ' + s)
    try:
        return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise IntegrityError('Invalid JSON: ' + str(exc)) from exc

def read_json(p: Path): return loads(bytes_read(p))

def read_jsonl(p: Path) -> list[dict]:
    result = []
    for i, line in enumerate(bytes_read(p).splitlines(keepends=True)):
        if not line.endswith(b'\n') or not line.strip(): raise IntegrityError(f'Invalid JSONL line {i + 1}')
        obj = loads(line)
        if not isinstance(obj, dict): raise IntegrityError('JSONL record must be object')
        result.append(obj)
    return result

def write_new(root: Path, name: str, data: bytes) -> Path:
    p = safe(root, name)
    p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(fd, 'wb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    return p

def write_json(root: Path, name: str, value) -> Path: return write_new(root, name, canonical(value) + b'\n')
def write_jsonl(root: Path, name: str, values) -> Path: return write_new(root, name, b''.join(canonical(v) + b'\n' for v in values))

def inventory(root: Path, excluded: set[str] | None = None) -> list[dict]:
    root = Path(root)
    safe(root, '_root_check')
    excluded = excluded or set()
    rows = []
    if not root.is_dir(): raise IntegrityError('Directory missing')
    for directory, dirs, files in os.walk(root, followlinks=False):
        for n in dirs:
            if (Path(directory) / n).is_symlink(): raise IntegrityError('Symlink directory')
        for n in files:
            p = Path(directory) / n
            name = p.relative_to(root).as_posix(); relative(name); s = regular(p)
            if name not in excluded: rows.append({'path': name, 'sha256': sha_file(p), 'size_bytes': s.st_size})
    return sorted(rows, key=lambda x: x['path'])

def pointer(value, path: str):
    if not path: return value
    if not isinstance(path, str) or not path.startswith('/'): raise IntegrityError('Invalid JSON Pointer')
    try:
        for key in path[1:].split('/'):
            key = key.replace('~1', '/').replace('~0', '~')
            if isinstance(value, list):
                if not key.isdigit() or str(int(key)) != key: raise KeyError(key)
                value = value[int(key)]
            else: value = value[key]
        return value
    except (KeyError, TypeError, IndexError, ValueError) as e:
        raise IntegrityError('Unresolvable JSON Pointer: ' + path) from e

class Journal:
    def __init__(self, root: Path, name: str = 'analyst_trace.jsonl'):
        self.path = write_new(root, name, b'')
        self.records: list[dict] = []
    def emit(self, kind: str, data: dict) -> dict:
        r = {'sequence': len(self.records), 'recorded_at': now(), 'kind': kind, 'data': data,
             'previous_sha256': self.records[-1]['sha256'] if self.records else None}
        r['sha256'] = digest(r)
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | getattr(os, 'O_NOFOLLOW', 0))
        with os.fdopen(fd, 'ab') as f:
            f.write(canonical(r) + b'\n'); f.flush(); os.fsync(f.fileno())
        self.records.append(r)
        return r

def check_journal(records):
    previous = None
    for i, r in enumerate(records):
        if r.get('sequence') != i or r.get('previous_sha256') != previous or r.get('sha256') != digest({k: v for k, v in r.items() if k != 'sha256'}):
            raise IntegrityError('Journal hash chain mismatch')
        previous = r['sha256']
    return previous
