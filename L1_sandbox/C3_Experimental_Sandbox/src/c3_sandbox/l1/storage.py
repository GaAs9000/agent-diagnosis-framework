"""Deterministic encoding and confined, exclusive evidence I/O."""
from __future__ import annotations

import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_relative(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("invalid relative evidence path")
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or str(p) != value or value == "." or ":" in p.parts[0]:
        raise ValueError(f"unsafe relative evidence path: {value!r}")
    return value


def confined(root: Path, relative: str) -> Path:
    relative = safe_relative(relative)
    if root.is_symlink():
        raise ValueError("symlink evidence root")
    root = root.resolve()
    path = root / relative
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink():
            raise ValueError("symlink evidence path")
    if not path.resolve().is_relative_to(root):
        raise ValueError("evidence path escapes root")
    return path


def regular_file(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError(f"not an independent regular evidence file: {path}")


def write_bytes(root: Path, relative: str, data: bytes) -> Path:
    path = confined(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    return path


def write_json(root: Path, relative: str, value: Any) -> Path:
    return write_bytes(root, relative, canonical(value) + b"\n")


def read_json(path: Path) -> Any:
    regular_file(path)
    def reject_constant(value: str):
        raise ValueError(f"non-finite JSON value: {value}")
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError(f"duplicate JSON key: {key}")
            obj[key] = value
        return obj
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant, object_pairs_hook=unique)


def copy_file(source: Path, root: Path, relative: str) -> dict:
    regular_file(source)
    target = write_bytes(root, relative, source.read_bytes())
    return {"path": relative, "sha256": file_digest(target), "size_bytes": target.stat().st_size}


def ref(path: str, pointer: str | None = None) -> dict:
    safe_relative(path)
    return {"path": path, "json_pointer": pointer}


def resolve_ref(root: Path, reference: dict) -> Any:
    path = confined(root, reference["path"])
    regular_file(path)
    pointer = reference.get("json_pointer")
    if pointer is None:
        return path.read_bytes()
    value = read_json(path)
    if pointer == "":
        return value
    if not pointer.startswith("/"):
        raise ValueError("JSON pointer must begin with /")
    for part in pointer[1:].split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value
