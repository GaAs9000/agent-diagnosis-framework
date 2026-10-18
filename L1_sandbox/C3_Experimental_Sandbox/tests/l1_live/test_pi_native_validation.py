"""CPU-only native Pi checks; no provider or native tool is executed."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import pytest

from c3_sandbox.l1.runtime.pi_native_validation import NativePiInfrastructureError, NativePiValidator


@pytest.fixture
def native():
    directory = os.environ.get("C3_PI_NATIVE_TEST_DIR")
    if not directory:
        pytest.skip("Set C3_PI_NATIVE_TEST_DIR to an npm-ci installed pinned bridge")
    with NativePiValidator(bridge_dir=directory) as validator:
        yield validator


@pytest.mark.parametrize("name,args,expected", [
    ("read", {"path": "a", "offset": "2", "limit": "5"}, {"path": "a", "offset": 2, "limit": 5}),
    ("bash", {"command": "exit 1", "timeout": "3"}, {"command": "exit 1", "timeout": 3}),
    ("write", {"path": "a", "content": "text", "extra": True}, {"path": "a", "content": "text", "extra": True}),
    ("read", {"path": "a", "extra": True}, {"path": "a", "extra": True}),
    ("bash", {"command": "exit 1", "extra": True}, {"command": "exit 1", "extra": True}),
    ("edit", {"path": "a", "oldText": "old", "newText": "new"}, {"path": "a", "edits": [{"oldText": "old", "newText": "new"}]}),
    ("edit", {"path": "a", "edits": "[{\"oldText\":\"old\",\"newText\":\"new\"}]"}, {"path": "a", "edits": [{"oldText": "old", "newText": "new"}]}),
    ("edit", {"path": "a", "edits": []}, {"path": "a", "edits": []}),
])
def test_native_preparation_and_coercion(native, name, args, expected):
    original = json.loads(json.dumps(args))
    result = native.validate(name, args)
    assert result["ok"] is True
    assert result["validated_arguments"] == expected
    assert result["raw_arguments"] == original == args
    assert result["native_result"] is None
    assert result["telemetry"] == {"actor_requests": 0, "native_loop_observed_responses": 1, "native_tool_executions": 0}


@pytest.mark.parametrize("args", [None, [], "bad", 42, {"path": {"path": "a"}},
    {"path": "a", "edits": [{"path": "b", "oldText": "x", "newText": "y"}]},
    {"path": "a", "edits": [{"oldText": "x"}]},
    {"path": "a", "edits": [], "extra": True}])
def test_error_text_matches_native_validator(native, args):
    # An independent native entrypoint supplies the expected message; Python never formats it.
    script = """
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, join } from 'node:path';
import { validateToolArguments } from '@mariozechner/pi-ai';
const root = dirname(fileURLToPath(import.meta.resolve('@mariozechner/pi-coding-agent')));
const { createEditTool } = await import(pathToFileURL(join(root,'core/tools/edit.js')));
const tool = createEditTool('/workspace');
let args = JSON.parse(process.argv[1]);
try {
  if (tool.prepareArguments) args = tool.prepareArguments(args);
  validateToolArguments(tool, {name:'edit', id:'validation', arguments:args});
  throw new Error('Test expected native rejection');
} catch(error) { process.stdout.write(JSON.stringify(error.message)); }
"""
    expected = json.loads(subprocess.check_output(
        [native.node_executable, "--input-type=module", "-e", script, json.dumps(args)],
        cwd=native.bridge_dir, timeout=15,
    ))
    result = native.validate("edit", args)
    assert result["ok"] is False
    assert result["validated_arguments"] is None
    assert result["native_result"]["content"] == [{"type": "text", "text": expected}]
    assert result["native_result"]["isError"] is True


def test_unknown_tool_and_same_batch_continuation(native):
    results = native.validate_batch([
        {"name": "missing", "arguments": {}, "id": "unknown"},
        {"name": "edit", "arguments": {"path": {"bad": "a"}}, "id": "bad"},
        {"name": "read", "arguments": {"path": "a", "offset": "2"}, "id": "good"},
    ])
    assert [result["ok"] for result in results] == [False, False, True]
    assert results[0]["native_result"]["content"] == [{"type": "text", "text": "Tool missing not found"}]
    assert results[2]["validated_arguments"]["offset"] == 2
    assert all(result["telemetry"]["native_tool_executions"] == 0 for result in results)


def test_schemas_and_dependency_identity(native):
    schemas = {item["name"]: item["parameters"] for item in native.native_tool_schemas()}
    assert set(schemas) == {"read", "bash", "edit", "write"}
    assert schemas["edit"]["additionalProperties"] is False
    assert "minItems" not in schemas["edit"]["properties"]["edits"]
    identity = native.dependency_identity()
    assert identity["commit"] == "781152fc24841dc54b22284514604048ebe5e2c9"
    assert all(package["version"] == "0.73.1" and package["integrity"].startswith("sha512-") for package in identity["packages"].values())
    assert len(identity["dependency_identity"]) == 64


def test_serialized_persistent_requests(native):
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda index: native.validate("read", {"path": str(index)}), range(12)))
    assert [result["validated_arguments"]["path"] for result in results] == [str(index) for index in range(12)]


def test_missing_assets_are_infrastructure_failure(tmp_path):
    with pytest.raises(NativePiInfrastructureError, match="Missing native Pi asset"):
        NativePiValidator(node_executable="/does/not/exist", bridge_dir=tmp_path).validate("read", {})


def test_timeout_closes_stuck_child(tmp_path):
    node = os.environ.get("C3_PI_NODE")
    if not node:
        pytest.skip("Set C3_PI_NODE to the server Node runtime")
    for filename in ("package.json", "package-lock.json", "provenance.json"):
        (tmp_path / filename).write_text("{}")
    (tmp_path / "bridge.mjs").write_text(
        "process.stdout.write(JSON.stringify({ready:true,provenance:{}})+'\\n');"
        "process.stdin.resume();setInterval(()=>{},1000);"
    )
    validator = NativePiValidator(node_executable=node, bridge_dir=tmp_path, timeout_seconds=.2)
    start = time.monotonic()
    with pytest.raises(NativePiInfrastructureError, match="timeout"):
        validator.validate("read", {})
    assert time.monotonic() - start < 3
    assert validator._process.poll() is not None


def test_oversized_request_fails_before_writing(native):
    with pytest.raises(NativePiInfrastructureError, match="byte limit"):
        native.validate("read", {"path": "x" * (1024 * 1024)})


def test_write_timeout_kills_child_that_never_reads(tmp_path):
    node = os.environ.get("C3_PI_NODE")
    if not node:
        pytest.skip("Set C3_PI_NODE to the server Node runtime")
    for filename in ("package.json", "package-lock.json", "provenance.json"):
        (tmp_path / filename).write_text("{}")
    (tmp_path / "bridge.mjs").write_text(
        "process.stdout.write(JSON.stringify({ready:true,provenance:{}})+'\\n');"
        "setInterval(()=>{},1000);"
    )
    validator = NativePiValidator(node_executable=node, bridge_dir=tmp_path, timeout_seconds=.2)
    start = time.monotonic()
    with pytest.raises(NativePiInfrastructureError, match="write timeout"):
        validator.validate("read", {"path": "x" * 300000})
    assert time.monotonic() - start < 3
    assert validator._process.poll() is not None


def test_absent_native_dependencies_are_fatal(tmp_path, native):
    import shutil
    for filename in ("bridge.mjs", "package.json", "package-lock.json", "provenance.json"):
        shutil.copyfile(native.bridge_dir / filename, tmp_path / filename)
    with pytest.raises(NativePiInfrastructureError):
        with NativePiValidator(node_executable=native.node_executable, bridge_dir=tmp_path) as missing:
            missing.validate("read", {"path": "a"})


@pytest.mark.parametrize("mutation", ["changed", "added", "removed"])
def test_approved_module_manifest_rejects_same_version_tampering(tmp_path, native, mutation):
    import shutil
    for filename in ("bridge.mjs", "package.json", "package-lock.json", "provenance.json", "approved-modules.json"):
        shutil.copyfile(native.bridge_dir / filename, tmp_path / filename)
    shutil.copytree(native.bridge_dir / "node_modules", tmp_path / "node_modules")
    package = tmp_path / "node_modules/@mariozechner/pi-agent-core"
    assert json.loads((package / "package.json").read_text())["version"] == "0.73.1"
    loop = package / "dist/agent-loop.js"
    if mutation == "changed":
        loop.write_text(loop.read_text() + "\nthrow new Error('tampered native module loaded');\n")
    elif mutation == "added":
        (package / "dist/unapproved.js").write_text("export const unapproved = true;\n")
    else:
        loop.unlink()
    with pytest.raises(NativePiInfrastructureError):
        NativePiValidator(node_executable=native.node_executable, bridge_dir=tmp_path).start()


def test_expired_deadline_never_starts_child(tmp_path):
    validator = NativePiValidator(bridge_dir=tmp_path)
    with pytest.raises(NativePiInfrastructureError, match="deadline"):
        validator.validate("read", {}, deadline_monotonic=time.monotonic() - 1)
    assert validator._process is None


def test_deadline_includes_waiting_for_lock(tmp_path):
    validator = NativePiValidator(bridge_dir=tmp_path)
    held = threading.Event()
    release = threading.Event()

    def hold():
        with validator._lock:
            held.set()
            release.wait(2)

    thread = threading.Thread(target=hold)
    thread.start()
    held.wait(1)
    start = time.monotonic()
    try:
        with pytest.raises(NativePiInfrastructureError, match="deadline"):
            validator.validate("read", {}, deadline_monotonic=start + .1)
        assert time.monotonic() - start < .4
        assert validator._process is None
    finally:
        release.set()
        thread.join(1)


def test_startup_and_reply_share_absolute_deadline(tmp_path):
    node = os.environ.get("C3_PI_NODE")
    if not node:
        pytest.skip("Set C3_PI_NODE to the server Node runtime")
    for filename in ("package.json", "package-lock.json", "provenance.json"):
        (tmp_path / filename).write_text("{}")
    (tmp_path / "bridge.mjs").write_text(
        "setTimeout(()=>process.stdout.write(JSON.stringify({ready:true,provenance:{}})+'\\n'),150);"
        "process.stdin.on('data',()=>setTimeout(()=>process.stdout.write(JSON.stringify({ok:true,provenance:{}})+'\\n'),150));"
    )
    validator = NativePiValidator(node_executable=node, bridge_dir=tmp_path, timeout_seconds=5)
    start = time.monotonic()
    with pytest.raises(NativePiInfrastructureError, match="timeout|deadline"):
        validator.validate("read", {}, deadline_monotonic=start + .25)
    assert time.monotonic() - start < .5
    assert validator._process.poll() is not None
