"""P0 filesystem-only snapshots. Not an OS isolation or live-state recovery layer."""
import json
import os
from pathlib import Path, PurePosixPath
import stat

from ..contracts import FILESYSTEM_SCOPE, RestoreReport
from ._storage import directory, get, put, read_regular, local_path


def _scan(root: Path) -> dict:
    if any(path.is_symlink() for path in (root, *root.parents)) or not root.is_dir():
        raise ValueError("workspace must be a real directory")
    entries = {".": {"kind": "directory", "mode": stat.S_IMODE(root.stat().st_mode)}}

    def visit(folder):
        for path in sorted(folder.iterdir()):
            info = path.lstat()
            key = path.relative_to(root).as_posix()
            if "\\" in key:
                raise ValueError(f"unsupported backslash in snapshot path: {key}")
            mode = stat.S_IMODE(info.st_mode)
            if stat.S_ISDIR(info.st_mode):
                entries[key] = {"kind": "directory", "mode": mode}
                visit(path)
            elif stat.S_ISREG(info.st_mode):
                data = read_regular(path)
                entries[key] = {"kind": "file", "mode": mode, "data": data}
            else:
                raise ValueError(f"unsupported symlink or special file: {key}")
    visit(root)
    return entries


class WorkspaceStore:
    def __init__(self, root):
        self.root = local_path(root)
        directory(self.root)
        self.objects = self.root / "objects"
        self.manifests = self.root / "snapshots"
        directory(self.objects)
        directory(self.manifests)

    def capture(self, workspace) -> str:
        workspace = local_path(workspace)
        if self.root == workspace or workspace in self.root.parents:
            raise ValueError("evidence store must be outside captured workspace")
        entries = _scan(workspace)
        for entry in entries.values():
            if entry["kind"] == "file":
                entry["sha256"] = put(self.objects, entry.pop("data"))
        from ._storage import encode
        return put(self.manifests, encode({"schema_version": 1, "scope": FILESYSTEM_SCOPE, "entries": entries}), ".json")

    def _manifest(self, snapshot_id):
        manifest = json.loads(get(self.manifests, snapshot_id, ".json"))
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or manifest.get("scope") != FILESYSTEM_SCOPE:
            raise ValueError("invalid snapshot manifest header")
        entries = manifest.get("entries")
        if not isinstance(entries, dict) or "." not in entries:
            raise ValueError("invalid snapshot entries")
        for name, entry in entries.items():
            if not isinstance(name, str) or not name:
                raise ValueError("invalid snapshot path")
            path = PurePosixPath(name)
            if name != "." and (path.is_absolute() or any(p in {".", "..", ""} for p in name.split("/")) or "\\" in name or "\x00" in name):
                raise ValueError("unsafe snapshot path")
            if not isinstance(entry, dict) or entry.get("kind") not in {"directory", "file"}:
                raise ValueError("invalid snapshot entry")
            mode = entry.get("mode")
            if type(mode) is not int or not 0 <= mode <= 0o7777:
                raise ValueError("invalid snapshot mode")
            expected = {"kind", "mode", "sha256"} if entry["kind"] == "file" else {"kind", "mode"}
            if set(entry) != expected:
                raise ValueError("invalid snapshot entry fields")
            if name == "." and entry["kind"] != "directory":
                raise ValueError("snapshot root must be a directory")
            if name != ".":
                parent = path.parent.as_posix()
                if parent not in entries or not isinstance(entries[parent], dict) or entries[parent].get("kind") != "directory":
                    raise ValueError("missing directory parent")
            if entry["kind"] == "file":
                get(self.objects, entry["sha256"])
        return entries

    def verify(self, snapshot_id, workspace) -> RestoreReport:
        try:
            expected = self._manifest(snapshot_id)
            actual = _scan(local_path(workspace))
            mismatches = []
            for name in sorted(set(expected) | set(actual)):
                if name not in actual:
                    mismatches.append(f"missing: {name}")
                elif name not in expected:
                    mismatches.append(f"unexpected: {name}")
                else:
                    item = actual[name].copy()
                    if item["kind"] == "file":
                        from ._storage import digest
                        item["sha256"] = digest(item.pop("data"))
                    if item != expected[name]:
                        mismatches.append(f"content/type/mode mismatch: {name}")
            return RestoreReport(not mismatches, mismatches)
        except (ValueError, OSError, TypeError, KeyError) as error:
            return RestoreReport(False, [f"verification failed: {error}"])

    def restore(self, snapshot_id, destination) -> RestoreReport:
        try:
            # Validate all paths and stored bytes before creating any destination content.
            entries = self._manifest(snapshot_id)
            destination = local_path(destination)
            for path in (destination, *destination.parents):
                if path.is_symlink():
                    raise ValueError("restore destination contains symlink")
            if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
                raise ValueError("restore destination must be new or empty")
            destination.mkdir(parents=True, exist_ok=True)
            for name, entry in sorted(entries.items(), key=lambda item: (len(PurePosixPath(item[0]).parts), item[0])):
                if name == ".":
                    continue
                path = destination / name
                if entry["kind"] == "directory":
                    path.mkdir()
                else:
                    with path.open("xb") as handle:
                        handle.write(get(self.objects, entry["sha256"]))
                    path.chmod(entry["mode"])
            # Apply directory modes last to permit creation under read-only directories.
            for name, entry in sorted(entries.items(), key=lambda item: len(PurePosixPath(item[0]).parts), reverse=True):
                if entry["kind"] == "directory":
                    (destination / name).chmod(entry["mode"])
            return self.verify(snapshot_id, destination)
        except (ValueError, OSError, TypeError, KeyError) as error:
            return RestoreReport(False, [f"restore failed: {error}"])
