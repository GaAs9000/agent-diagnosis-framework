"""Content addressed records with exclusive writes and integrity checks."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat


def encode(record) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def identity(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("invalid content identity")
    return value


def local_path(path) -> Path:
    """Normalize macOS system aliases while retaining user symlink checks."""
    path = Path(path).absolute()
    if ".." in path.parts:
        raise ValueError("parent traversal is not supported in evidence paths")
    for alias in ("/var", "/tmp", "/etc"):
        base = Path(alias)
        if base.is_symlink() and (path == base or base in path.parents):
            path = base.resolve() / path.relative_to(base)
    return path


def directory(path: Path):
    # Reject symlinks in every existing path component, including the store root.
    for part in (path.absolute(), *path.absolute().parents):
        if part.is_symlink():
            raise ValueError(f"symlink path component: {part}")
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise ValueError("store path is not a directory")


def read_regular(path: Path) -> bytes:
    for parent in path.parents:
        if parent.is_symlink():
            raise ValueError("stored path contains symlink")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"not a regular file: {path}")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            return handle.read()
    finally:
        os.close(fd)


def put(directory_path: Path, data: bytes, suffix="") -> str:
    key = digest(data)
    path = directory_path / (key + suffix)
    try:
        with path.open("xb") as handle:
            handle.write(data)
        path.chmod(0o400)
    except FileExistsError:
        if read_regular(path) != data:
            raise ValueError("existing content addressed object is corrupt")
    if digest(read_regular(path)) != key:
        raise ValueError("stored object failed readback verification")
    return key


def get(directory_path: Path, key: str, suffix="") -> bytes:
    data = read_regular(directory_path / (identity(key) + suffix))
    if digest(data) != key:
        raise ValueError("content addressed object was tampered with")
    return data
