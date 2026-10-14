"""Read-only local serving-process identity checks on a trusted host.

Argv, listener ownership and disk source provenance are evidence of the configured
service, not attestation against a malicious host or modified in-memory process.
"""
from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
from importlib.metadata import distributions
from pathlib import Path

from ..manifests import canonical_hash
from .r12_system import ROOT, TOOL_PARSER_SHA256, validated_system, system_identity

MAX_OUTPUT_BYTES = 65536


def _read_command(argv: list[str]) -> str:
    result = subprocess.run(argv, capture_output=True, timeout=5, check=False,
                            env={"LC_ALL": "C", "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"})
    if result.returncode != 0 or len(result.stdout) > MAX_OUTPUT_BYTES:
        raise ValueError("serving process observation failed or exceeded bounds")
    return result.stdout.decode("utf-8", errors="strict")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _argv(command: str, model_path: Path) -> tuple[list[str], Path]:
    # macOS ps joins argv without reliably quoting paths containing spaces.
    # Protect only the two exact approved absolute paths before tokenization.
    candidates = [ROOT / ".model-venv/bin/python", ROOT / ".model-venv/bin/python3"]
    executable = next((p for p in candidates if command.startswith(str(p) + " ")), None)
    if executable is None:
        raise ValueError("generator must use the approved model Python environment")
    tail = command[len(str(executable)):].strip()
    if "__C3_APPROVED_MODEL_PATH__" in tail:
        raise ValueError("unexpected generator path marker")
    tail = tail.replace(str(model_path), "__C3_APPROVED_MODEL_PATH__")
    args = shlex.split(tail)
    if args[:1] == ["-u"]:
        args = args[1:]
    if args[:2] != ["-m", "mlx_lm.server"]:
        raise ValueError("generator must run the stock MLX server module")
    options = args[2:]
    expected = {"--model": "__C3_APPROVED_MODEL_PATH__", "--host": "127.0.0.1",
                "--port": "18080", "--temp": "0", "--max-tokens": "4096",
                "--prompt-cache-size": "0", "--prompt-concurrency": "1",
                "--decode-concurrency": "1"}
    optional = {"--log-level": {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"},
                "--top-k": {"1"}, "--min-p": {"0", "0.0"}, "--top-p": {"1", "1.0"}}
    seen = {}
    if len(options) % 2:
        raise ValueError("unexpected generator arguments")
    for flag, value in zip(options[::2], options[1::2]):
        if flag in seen or flag not in expected.keys() | optional.keys():
            raise ValueError("unexpected or duplicate generator option")
        if flag in optional and value not in optional[flag]:
            raise ValueError("generator sampling option differs from contract")
        seen[flag] = value
    if seen.get("--model") == model_path.relative_to(ROOT).as_posix():
        seen["--model"] = "__C3_APPROVED_MODEL_PATH__"
    if seen.get("--temp") == "0.0":
        seen["--temp"] = "0"
    if any(seen.get(flag) != value for flag, value in expected.items()):
        raise ValueError("generator arguments differ from frozen system")
    argv = [str(executable), *shlex.split(command[len(str(executable)):].strip().replace(
        str(model_path), "__C3_APPROVED_MODEL_PATH__"))]
    argv = [str(model_path) if arg == "__C3_APPROVED_MODEL_PATH__" else arg for arg in argv]
    return argv, executable


def _environment(executable: Path, system: dict) -> dict:
    environment = ROOT / ".model-venv"
    sites = sorted((environment / "lib").glob("python*/site-packages"))
    if len(sites) != 1:
        raise ValueError("model Python environment is ambiguous")
    site = sites[0]
    sources = {name: _hash(site / "mlx_lm" / name) for name in
               ("server.py", "tokenizer_utils.py", "tool_parsers/qwen3_coder.py")}
    if sources["tool_parsers/qwen3_coder.py"] != TOOL_PARSER_SHA256:
        raise ValueError("generator parser source differs from frozen parser")
    packages = {}
    for distribution in distributions(path=[str(site)]):
        name = distribution.metadata.get("Name")
        if name:
            key = name.lower().replace("_", "-")
            if key in packages:
                raise ValueError("model environment has duplicate distributions")
            packages[key] = distribution.version
    if not {"mlx", "mlx-lm", "transformers"} <= packages.keys():
        raise ValueError("model environment required packages are missing")
    return {"path": str(environment), "site_packages": str(site),
            "executable": str(executable), "resolved_executable": str(executable.resolve(strict=True)),
            "executable_sha256": _hash(executable),
            "pyvenv_cfg_sha256": _hash(environment / "pyvenv.cfg"),
            "packages": dict(sorted(packages.items())), "mlx_source_sha256": sources,
            "server_packages_lock_sha256": _hash(ROOT / system["server"]["packages_lock"])}


def observe_service(system: dict, pid: int) -> dict:
    """Observe the explicitly supplied initial PID, its start identity and unique listener."""
    system = validated_system(system)
    if type(pid) is not int or pid <= 0:
        raise ValueError("service pid must be a positive integer")
    process = _read_command(["/bin/ps", "-ww", "-p", str(pid), "-o", "pid=",
                             "-o", "lstart=", "-o", "command="]).strip()
    fields = process.split(None, 6)
    if len(fields) != 7 or fields[0] != str(pid):
        raise ValueError("serving PID is absent or ambiguous")
    start = " ".join(fields[1:6])
    model_path = (ROOT / system["model"]["path"]).resolve(strict=True)
    argv, executable = _argv(fields[6], model_path)
    cwd = _read_command(["/usr/sbin/lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"])
    directories = [line[1:] for line in cwd.splitlines() if line.startswith("n")]
    if directories != [str(ROOT)]:
        raise ValueError("generator working directory differs from approved model root")
    listeners = _read_command(["/usr/sbin/lsof", "-nP", "-iTCP:18080", "-sTCP:LISTEN", "-Fpcn"])
    records, owner = [], None
    for line in listeners.splitlines():
        if line.startswith("p"):
            owner = line[1:]
        elif line.startswith("n"):
            records.append((owner, line[1:]))
    if records != [(str(pid), "127.0.0.1:18080")]:
        raise ValueError("generator must exclusively own the pinned loopback listener")
    observed = {"pid": pid, "start_identity": start, "argv": argv, "cwd": str(ROOT),
                "listener": {"pid": pid, "address": "127.0.0.1", "port": 18080},
                "python_environment": _environment(executable, system), "model_path": str(model_path),
                "system_contract_sha256": system_identity(system),
                "assurance_scope": "trusted_host_process_argv_listener_and_disk_sources"}
    return {**observed, "sha256": canonical_hash(observed)}


def verify_service(system: dict, frozen_observation: dict) -> dict:
    """Re-observe the frozen PID, never consulting a mutable PID-file pointer."""
    frozen = json.loads(json.dumps(frozen_observation, allow_nan=False))
    if type(frozen) is not dict or frozen.get("sha256") != canonical_hash(
            {key: value for key, value in frozen.items() if key != "sha256"}):
        raise ValueError("invalid frozen serving observation")
    fresh = observe_service(system, frozen.get("pid"))
    if fresh != frozen:
        raise ValueError("serving process identity drifted after freeze")
    return fresh
