"""Controller-owned, bounded aggregate inspection inside the actor runtime."""
from __future__ import annotations

import base64
import hashlib
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import stat
import zipfile
import zlib
import xml.etree.ElementTree as ET

from .docker_runtime import MAX_PATHS, MAX_WORKSPACE_BYTES, snapshot_workspace, validate_snapshot
from .probes import EXTENSIONS, INVENTORY_FIELDS, validate_public_evidence

MAX_ZIP_MEMBERS = 2048
MAX_XML_BYTES = 8 * 1024 * 1024
MAX_ZIP_BYTES = 32 * 1024 * 1024
MAX_ZIP_RATIO = 200
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def _document_counts(stream):
    """Return counts atomically, or one rejected-document parse error."""
    counts = dict.fromkeys(("paragraph_count", "table_count", "section_count", "header_count", "omml_count"), 0)
    try:
        with zipfile.ZipFile(stream) as archive:
            members = archive.infolist()
            if len(members) > MAX_ZIP_MEMBERS:
                raise ValueError("archive member limit")
            seen, total = set(), 0
            for member in members:
                name = member.filename
                if member.orig_filename != name:
                    raise ValueError("unsafe archive filename")
                clean = name[:-1] if member.is_dir() else name
                path = PurePosixPath(clean)
                mode = member.external_attr >> 16
                if (not clean or str(path) != clean or path.is_absolute() or ".." in path.parts
                        or "\\" in name or "\x00" in name or ":" in name or name in seen
                        or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR))
                        or member.flag_bits & 1):
                    raise ValueError("unsafe archive")
                seen.add(name)
                total += member.file_size
                if (total > MAX_ZIP_BYTES or member.file_size > MAX_ZIP_BYTES
                        or member.file_size > max(1, member.compress_size) * MAX_ZIP_RATIO):
                    raise ValueError("archive size limit")
            if "word/document.xml" not in seen:
                raise ValueError("missing document")
            for member in members:
                name = member.filename
                is_header = name.startswith("word/header") and name.endswith(".xml") and "/" not in name[5:]
                if name != "word/document.xml" and not is_header:
                    continue
                if member.file_size > MAX_XML_BYTES:
                    raise ValueError("XML part size limit")
                with archive.open(member) as xml_stream:
                    data = xml_stream.read(MAX_XML_BYTES + 1)
                # OOXML has no need for declarations or entities. Reject before parsing,
                # including UTF-16/32 representations, to prevent entity expansion.
                normalized = data.replace(b"\x00", b"").upper()
                if len(data) != member.file_size or b"<!DOCTYPE" in normalized or b"<!ENTITY" in normalized:
                    raise ValueError("unsafe XML")
                root = ET.fromstring(data)
                expected = "hdr" if is_header else "document"
                if root.tag != "{" + W_NS + "}" + expected:
                    raise ValueError("invalid document root")
                counts["header_count"] += int(is_header)
                for element in root.iter():
                    for key, tag in (("paragraph_count", "p"), ("table_count", "tbl"), ("section_count", "sectPr")):
                        counts[key] += int(element.tag == "{" + W_NS + "}" + tag)
                    counts["omml_count"] += int(element.tag == "{" + M_NS + "}oMath")
        return counts, 0
    except (ValueError, OSError, RuntimeError, zipfile.BadZipFile, ET.ParseError, EOFError, NotImplementedError, zlib.error):
        return dict.fromkeys(counts, 0), 1


def _inspect_workspace(root, before):
    inventory = dict.fromkeys(INVENTORY_FIELDS, 0)
    inventory["extension_counts"] = {}
    current, visited = {}, 0
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError("unsafe workspace root")

    def visit(directory):
        nonlocal visited
        for path in sorted(directory.iterdir()):
            visited += 1
            if visited > MAX_PATHS:
                raise ValueError("workspace path limit")
            info = path.lstat()
            name = path.relative_to(root).as_posix()
            mode = stat.S_IMODE(info.st_mode)
            if stat.S_ISDIR(info.st_mode):
                current[name] = ["directory", mode, None]
                visit(path)
            elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                inventory["file_count"] += 1
                inventory["total_bytes"] += info.st_size
                if inventory["total_bytes"] > MAX_WORKSPACE_BYTES:
                    raise ValueError("workspace byte limit")
                suffix = path.suffix.lower() or "none"
                suffix = suffix if suffix in EXTENSIONS else "other"
                inventory["extension_counts"][suffix] = inventory["extension_counts"].get(suffix, 0) + 1
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                with os.fdopen(fd, "rb") as stream:
                    opened = os.fstat(stream.fileno())
                    if (opened.st_ino, opened.st_dev, opened.st_nlink, opened.st_size) != (info.st_ino, info.st_dev, 1, info.st_size):
                        raise ValueError("workspace changed during inspection")
                    if suffix == ".docx":
                        inventory["docx_count"] += 1
                        counts, errors = _document_counts(stream)
                        for key, value in counts.items():
                            inventory[key] += value
                        inventory["xml_parse_error_count"] += errors
                    if before is not None:
                        stream.seek(0)
                        digest, size = hashlib.sha256(), 0
                        while chunk := stream.read(65536):
                            size += len(chunk)
                            if size > info.st_size:
                                raise ValueError("workspace changed during inspection")
                            digest.update(chunk)
                        if size != info.st_size:
                            raise ValueError("workspace changed during inspection")
                        current[name] = ["file", mode, digest.hexdigest()]
                    after = os.fstat(stream.fileno())
                    if (after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (info.st_size, info.st_mtime_ns, info.st_ctime_ns):
                        raise ValueError("workspace changed during inspection")
            else:
                raise ValueError("special workspace entry")
    visit(root)
    changed = None if before is None else sum(current.get(name) != before.get(name) and any(value is not None and value[0] == "file" for value in (current.get(name), before.get(name))) for name in current.keys() | before.keys())
    return inventory, changed


def _collector_source():
    constants = {name: globals()[name] for name in ("MAX_PATHS", "MAX_WORKSPACE_BYTES", "MAX_ZIP_MEMBERS", "MAX_XML_BYTES", "MAX_ZIP_BYTES", "MAX_ZIP_RATIO", "W_NS", "M_NS")}
    return ("import hashlib,json,os,stat,sys,zipfile,zlib\nfrom pathlib import Path,PurePosixPath\nimport xml.etree.ElementTree as ET\n"
            + "\n".join(f"{key}={value!r}" for key, value in constants.items())
            + f"\nEXTENSIONS={tuple(sorted(EXTENSIONS))!r}\nINVENTORY_FIELDS={INVENTORY_FIELDS!r}\n"
            + inspect.getsource(_document_counts) + "\n" + inspect.getsource(_inspect_workspace)
            + "\ninventory,changed=_inspect_workspace(Path('/workspace'),json.load(sys.stdin))\nprint(json.dumps({'inventory':inventory,'files_changed':changed},sort_keys=True))\n")


def collect_public_evidence(actor, *, checkpoint=None, preworkspace=None, last_tool_result=None):
    """Inspect once through controller exec_fixed; never execute a model command.

    checkpoint is a DockerRuntime workspace snapshot; preworkspace is an exported
    workspace directory. Supply at most one. Feedback requires both prior state
    and an actual complete raw runtime tool result; otherwise it is unreported.
    """
    if checkpoint is not None and preworkspace is not None:
        raise ValueError("supply only one prior workspace")
    snapshot = checkpoint if checkpoint is not None else (snapshot_workspace(Path(preworkspace)) if preworkspace is not None else None)
    before = None
    if snapshot is not None:
        validate_snapshot(snapshot)
        before = {entry["path"]: [entry["kind"], entry["mode"], hashlib.sha256(base64.b64decode(entry["data"], validate=True)).hexdigest() if entry["kind"] == "file" else None] for entry in snapshot["entries"]}
    feedback = None
    if last_tool_result is not None and before is not None:
        required = ("exit_code", "stdout_bytes", "stderr_bytes")
        if any(key not in last_tool_result or type(last_tool_result[key]) is not int for key in required):
            raise ValueError("raw runtime byte counts and exit code required")
        feedback = {key: last_tool_result[key] for key in required}
        feedback.update(files_changed=0, parse_error_count=0)
        # Validate signed exit status and bounded actual byte counts before execution.
        validate_public_evidence({"inventory": {key: {} if key == "extension_counts" else 0 for key in INVENTORY_FIELDS}, "feedback": feedback})
    result = actor.exec_fixed(["python", "-I", "-B", "-c", _collector_source()], input_bytes=json.dumps(before, sort_keys=True).encode())
    if result.get("exit_code") != 0 or result.get("error") or result.get("truncated") or result.get("stderr"):
        raise RuntimeError("public evidence collection failed")
    try:
        public = json.loads(result["stdout"])
        if type(public) is not dict or set(public) != {"inventory", "files_changed"}:
            raise ValueError("invalid collector response")
        if feedback is not None:
            feedback["files_changed"] = public["files_changed"]
            feedback["parse_error_count"] = public["inventory"]["xml_parse_error_count"]
        return validate_public_evidence({"inventory": public["inventory"], "feedback": feedback})
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("invalid public evidence response") from exc
