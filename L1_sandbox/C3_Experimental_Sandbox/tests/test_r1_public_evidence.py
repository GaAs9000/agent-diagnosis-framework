import io
import json
from pathlib import Path
import zipfile

import pytest

from c3_sandbox.r1.docker_runtime import snapshot_workspace
from c3_sandbox.r1.probes import validate_public_evidence
from c3_sandbox.r1.public_evidence import collect_public_evidence, W_NS, M_NS


class FakeActor:
    """Execute fixed collector source against a test directory; no Docker calls."""
    def __init__(self, root):
        self.root = root
        self.calls = []

    def exec_fixed(self, argv, input_bytes):
        import contextlib
        import sys
        self.calls.append((argv, input_bytes))
        assert argv[:4] == ["python", "-I", "-B", "-c"]
        source = argv[4].replace("Path('/workspace')", "Path(" + repr(str(self.root)) + ")")
        previous = sys.stdin
        output = io.StringIO()
        try:
            sys.stdin = io.StringIO(input_bytes.decode())
            with contextlib.redirect_stdout(output):
                exec(compile(source, "<fixed collector>", "exec"), {})
        except Exception:
            return {"exit_code": 1, "stdout": "", "stderr": "private rejected path", "truncated": False}
        finally:
            sys.stdin = previous
        return {"exit_code": 0, "stdout": output.getvalue(), "stderr": "", "truncated": False}


def docx(path, extra=None, xml=None):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", xml or f'<w:document xmlns:w="{W_NS}" xmlns:m="{M_NS}"><w:body><w:p><w:r><w:t>ANSWER_SECRET</w:t></w:r><m:oMath/></w:p><w:tbl><w:tr><w:tc><w:p/></w:tc></w:tr></w:tbl><w:sectPr/></w:body></w:document>')
        archive.writestr("word/header1.xml", f'<w:hdr xmlns:w="{W_NS}"><w:p/></w:hdr>')
        if extra:
            archive.writestr(*extra)


def test_real_counts_and_no_content_channel(tmp_path):
    docx(tmp_path / "RUBRIC_SECRET.docx")
    (tmp_path / "ANSWER_SECRET.sensitive_suffix").write_text("oracle answer")
    (tmp_path / "plain").write_bytes(b"123")
    before = snapshot_workspace(tmp_path)
    actor = FakeActor(tmp_path)
    result = collect_public_evidence(actor)
    assert result == validate_public_evidence(result)
    assert result["feedback"] is None
    assert result["inventory"] == {"file_count": 3, "extension_counts": {".docx": 1, "other": 1, "none": 1}, "total_bytes": sum(p.stat().st_size for p in tmp_path.iterdir()), "docx_count": 1, "paragraph_count": 3, "table_count": 1, "section_count": 1, "header_count": 1, "omml_count": 1, "xml_parse_error_count": 0}
    assert "SECRET" not in json.dumps(result)
    assert snapshot_workspace(tmp_path) == before
    assert len(actor.calls) == 1


def test_feedback_requires_observed_before_and_raw_result(tmp_path):
    (tmp_path / "modified.txt").write_text("before")
    (tmp_path / "removed.txt").write_text("deleted")
    before = snapshot_workspace(tmp_path)
    (tmp_path / "modified.txt").write_text("after")
    (tmp_path / "removed.txt").unlink()
    (tmp_path / "added.txt").write_text("new")
    raw = {"exit_code": -9, "stdout_bytes": 12000, "stderr_bytes": 33, "stdout": "truncated", "truncated": True}
    actor = FakeActor(tmp_path)
    result = collect_public_evidence(actor, checkpoint=before, last_tool_result=raw)
    assert result["feedback"] == {"exit_code": -9, "stdout_bytes": 12000, "stderr_bytes": 33, "files_changed": 3, "parse_error_count": 0}
    assert collect_public_evidence(actor, checkpoint=before)["feedback"] is None
    assert collect_public_evidence(actor, last_tool_result=raw)["feedback"] is None
    with pytest.raises(ValueError, match="raw runtime"):
        collect_public_evidence(actor, checkpoint=before, last_tool_result={"exit_code": 0, "stdout": "fake"})


@pytest.mark.parametrize("extra", [("../oracle.xml", "secret"), ("/absolute.xml", "secret"), ("word\\evil.xml", "secret"), ("huge.xml", "x" * (9 * 1024 * 1024)), ("bomb.xml", "x" * 30000)])
def test_unsafe_archive_counted_without_partial_counts(tmp_path, extra):
    docx(tmp_path / "secret.docx", extra=extra)
    counts = collect_public_evidence(FakeActor(tmp_path))["inventory"]
    assert counts["docx_count"] == 1
    assert counts["xml_parse_error_count"] == 1
    assert all(counts[key] == 0 for key in ("paragraph_count", "table_count", "section_count", "header_count", "omml_count"))


@pytest.mark.parametrize("xml", ['<!DOCTYPE foo [<!ENTITY x "secret">]><foo>&x;</foo>', "<malformed>", '<document/>'])
def test_invalid_xml_is_counted(tmp_path, xml):
    docx(tmp_path / "bad.docx", xml=xml)
    assert collect_public_evidence(FakeActor(tmp_path))["inventory"]["xml_parse_error_count"] == 1


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo"])
def test_special_workspace_entries_fail_closed(tmp_path, kind):
    import os
    (tmp_path / "regular").write_text("secret")
    if kind == "symlink":
        (tmp_path / "bad").symlink_to(tmp_path / "regular")
    elif kind == "hardlink":
        os.link(tmp_path / "regular", tmp_path / "bad")
    else:
        os.mkfifo(tmp_path / "bad")
    with pytest.raises(RuntimeError, match="^public evidence collection failed$"):
        collect_public_evidence(FakeActor(tmp_path))


def test_preworkspace_and_modes(tmp_path):
    old = tmp_path / "old"
    new = tmp_path / "new"
    old.mkdir()
    new.mkdir()
    (old / "file.txt").write_text("same")
    (new / "file.txt").write_text("same")
    (new / "file.txt").chmod(0o600)
    raw = {"exit_code": 0, "stdout_bytes": 0, "stderr_bytes": 0}
    result = collect_public_evidence(FakeActor(new), preworkspace=old, last_tool_result=raw)
    assert result["feedback"]["files_changed"] == 1
    with pytest.raises(ValueError, match="only one"):
        collect_public_evidence(FakeActor(new), checkpoint=snapshot_workspace(old), preworkspace=old)


@pytest.mark.parametrize("result", [
    {"exit_code": 0, "stdout": '{"inventory":{},"files_changed":0,"secret":"answer"}', "stderr": ""},
    {"exit_code": 0, "stdout": '{}', "stderr": "", "truncated": True},
    {"exit_code": 0, "stdout": '{}', "stderr": "private failure"},
])
def test_runtime_failure_and_unconstrained_response_rejected(result):
    class Actor:
        def exec_fixed(self, *args, **kwargs):
            return result
    with pytest.raises(RuntimeError):
        collect_public_evidence(Actor())


def test_large_embedded_font_does_not_trigger_xml_limit(tmp_path, monkeypatch):
    import random
    from c3_sandbox.r1 import public_evidence

    # Scaled bounds reproduce the real 12 MiB font / 8 MiB XML limit failure
    # without allocating a multi-megabyte asset in each test run.
    monkeypatch.setattr(public_evidence, "MAX_XML_BYTES", 1024)
    monkeypatch.setattr(public_evidence, "MAX_ZIP_BYTES", 8192)
    font = random.Random(42).randbytes(4096)
    docx(tmp_path / "font.docx", extra=("word/fonts/font5.odttf", font))
    result = collect_public_evidence(FakeActor(tmp_path))
    counts = result["inventory"]
    assert counts["xml_parse_error_count"] == 0
    assert (counts["paragraph_count"], counts["table_count"], counts["section_count"], counts["header_count"], counts["omml_count"]) == (3, 1, 1, 1, 1)
    assert "font5" not in json.dumps(result)


@pytest.mark.parametrize("part", ["word/document.xml", "word/header2.xml"])
def test_parsed_xml_parts_still_enforce_xml_limit(tmp_path, monkeypatch, part):
    from c3_sandbox.r1 import public_evidence

    monkeypatch.setattr(public_evidence, "MAX_XML_BYTES", 1024)
    monkeypatch.setattr(public_evidence, "MAX_ZIP_BYTES", 8192)
    name = "document" if part == "word/document.xml" else "hdr"
    xml = f'<w:{name} xmlns:w="{W_NS}"><w:p><w:r><w:t>' + "x" * 2048 + f'</w:t></w:r></w:p></w:{name}>'
    if part == "word/document.xml":
        docx(tmp_path / "large.docx", xml=xml)
    else:
        docx(tmp_path / "large.docx", extra=(part, xml))
    counts = collect_public_evidence(FakeActor(tmp_path))["inventory"]
    assert counts["xml_parse_error_count"] == 1
    assert counts["paragraph_count"] == 0


@pytest.mark.parametrize("violation", ["total", "ratio"])
def test_unparsed_font_still_enforces_archive_bounds(tmp_path, monkeypatch, violation):
    import random
    from c3_sandbox.r1 import public_evidence

    monkeypatch.setattr(public_evidence, "MAX_XML_BYTES", 1024)
    monkeypatch.setattr(public_evidence, "MAX_ZIP_BYTES", 2048 if violation == "total" else 8192)
    font = random.Random(42).randbytes(4096) if violation == "total" else b"x" * 4096
    if violation == "ratio":
        monkeypatch.setattr(public_evidence, "MAX_ZIP_RATIO", 50)
    docx(tmp_path / "invalid.docx", extra=("word/fonts/font5.odttf", font))
    counts = collect_public_evidence(FakeActor(tmp_path))["inventory"]
    assert counts["xml_parse_error_count"] == 1
    assert counts["paragraph_count"] == 0
