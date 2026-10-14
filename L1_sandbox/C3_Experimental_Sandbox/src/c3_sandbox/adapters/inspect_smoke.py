"""Explicitly synthetic, serial Inspect/Docker infrastructure smoke. No LLM calls.

Stable APIs verified against the inspect-ai 0.3.278 published wheel. The
scripted solver has no model tools and never receives the grader sandbox.
This is not a production model/tool access policy or a benchmark result.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import importlib.metadata
import json
import subprocess
import uuid
from pathlib import Path

INSPECT_VERSION = "0.3.278"
DOCKER_DIR = Path(__file__).resolve().parents[3] / "docker"



# Fixed controller helpers avoid Docker cp's inability to read this tmpfs setup.
# Isolated Python mode prevents importing Agent-written files from /workspace.
READ_ARTIFACT = """import os, stat, base64
fd = os.open('/workspace/output.json', os.O_RDONLY | os.O_NOFOLLOW)
with os.fdopen(fd, 'rb') as f:
    if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
        raise ValueError('artifact must be a regular file')
    raw = f.read(4097)
if not raw or len(raw) > 4096:
    raise ValueError('artifact size out of bounds')
print(base64.b64encode(raw).decode('ascii'))
"""
WRITE_ARTIFACT = """import os, sys, hashlib
raw = sys.stdin.buffer.read(4097)
if not raw or len(raw) > 4096:
    raise ValueError('artifact size out of bounds')
fd = os.open('/workspace/frozen-artifact.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
with os.fdopen(fd, 'wb') as f:
    f.write(raw)
print(hashlib.sha256(raw).hexdigest())
"""


async def freeze_artifact(agent) -> bytes:
    result = await agent.exec(['/usr/local/bin/python', '-I', '-c', READ_ARTIFACT], timeout=15, timeout_retry=False)
    if not result.success:
        raise RuntimeError(f"artifact capture failed (exit={result.returncode}): stdout={result.stdout!r}; stderr={result.stderr!r}")
    encoded = result.stdout.strip()
    if not encoded or len(encoded) > 5464:
        raise ValueError('invalid artifact transport length')
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError('invalid artifact transport base64') from exc
    if not raw or len(raw) > 4096 or base64.b64encode(raw).decode('ascii') != encoded:
        raise ValueError('invalid artifact transport bytes')
    return raw


async def deliver_artifact(grader, frozen: bytes) -> None:
    result = await grader.exec(['/usr/local/bin/python', '-I', '-c', WRITE_ARTIFACT], input=frozen, timeout=15, timeout_retry=False)
    if not result.success:
        raise RuntimeError(f"artifact delivery failed (exit={result.returncode}): stdout={result.stdout!r}; stderr={result.stderr!r}")
    if result.stdout.strip() != sha256(frozen):
        raise ValueError('grader artifact transport identity mismatch')


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def parse_grade(raw: str) -> dict:
    """Fail closed on absent, unknown or mistyped result fields."""
    try:
        result = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError("grader did not return valid JSON") from exc
    if not isinstance(result, dict) or set(result) != {"schema_version", "is_synthetic", "passed", "checks"}:
        raise ValueError("invalid grader result fields")
    if type(result["schema_version"]) is not int or result["schema_version"] != 1:
        raise ValueError("invalid grader schema version")
    if result["is_synthetic"] is not True or type(result["passed"]) is not bool:
        raise ValueError("invalid grader result types")
    checks = result["checks"]
    if not isinstance(checks, dict) or set(checks) != {"artifact_schema_valid", "value_matches"}:
        raise ValueError("invalid grader checks")
    if any(type(v) is not bool for v in checks.values()):
        raise ValueError("grader checks must be booleans")
    if checks["artifact_schema_valid"] is not True or result["passed"] != checks["value_matches"]:
        raise ValueError("inconsistent grader result")
    return result


def compose_config(run_id: str) -> dict:
    config = json.loads((DOCKER_DIR / "compose.yaml").read_text())
    for service in config["services"].values():
        service.pop("build")
        service["labels"] = {"c3.synthetic_run": run_id}
    return config


def docker(*args: str) -> str:
    result = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=60, check=True)
    return result.stdout


def verify_container(info: dict) -> None:
    """Check effective settings, not only the requested Compose settings."""
    host = info["HostConfig"]
    cfg = info["Config"]
    if cfg["User"] != "1000:1000" or host["ReadonlyRootfs"] is not True:
        raise ValueError("container user/rootfs policy violated")
    if host["NetworkMode"] != "none" or host["Privileged"] or host.get("CapAdd"):
        raise ValueError("container network/capability policy violated")
    if set(host["CapDrop"]) != {"ALL"} or "no-new-privileges:true" not in host["SecurityOpt"]:
        raise ValueError("container privilege policy violated")
    if host["NanoCpus"] != 2_000_000_000 or host["Memory"] != 4 * 1024**3 or host["PidsLimit"] != 128:
        raise ValueError("container resource limits violated")
    if host.get("Binds") or set(host["Tmpfs"]) != {"/workspace", "/tmp"}:
        raise ValueError("container mount policy violated")
    if any(m["Type"] != "tmpfs" or m["Destination"] not in {"/workspace", "/tmp"} for m in info["Mounts"]):
        raise ValueError("unexpected container mount")
    if cfg.get("WorkingDir") != "/workspace":
        raise ValueError("unexpected working directory")


def capture_containers(run_id: str) -> dict:
    ids = docker("ps", "-q", "--filter", f"label=c3.synthetic_run={run_id}").split()
    if len(ids) != 2:
        raise ValueError("expected exactly two own smoke containers")
    records = json.loads(docker("inspect", *ids))
    by_service = {}
    for record in records:
        verify_container(record)
        service = record["Config"]["Labels"]["com.docker.compose.service"]
        if service not in {"default", "grader"} or service in by_service:
            raise ValueError("unexpected or duplicate smoke service")
        by_service[service] = {"id": record["Id"], "image_id": record["Image"],
                               "config": record["Config"], "host_config": record["HostConfig"],
                               "mounts": record["Mounts"],
                               "config_sha256": sha256(json.dumps(record["Config"], sort_keys=True).encode()),
                               "host_config_sha256": sha256(json.dumps(record["HostConfig"], sort_keys=True).encode())}
    return by_service


def make_task(compose_path: Path, output_dir: Path, run_id: str, evidence: dict):
    """Build one synthetic Task using public Inspect APIs (optional dependency)."""
    from inspect_ai import Task
    from inspect_ai.dataset import Sample
    from inspect_ai.scorer import Score, scorer, accuracy
    from inspect_ai.solver import solver
    from inspect_ai.util import sandbox

    @solver
    def scripted_artifact():
        async def solve(state, generate):
            agent = sandbox("default")
            # Both hidden grader paths must be absent in this filesystem.
            probe = await agent.exec(["/usr/local/bin/python", "-I", "-c", "from pathlib import Path; import json; print(json.dumps({p:Path(p).exists() for p in ['/opt/grader/expected.json','/opt/grader/grader.py','/var/run/docker.sock']}))"], timeout=15, timeout_retry=False)
            if not probe.success:
                raise RuntimeError(f"agent boundary probe failed (exit={probe.returncode}): stdout={probe.stdout!r}; stderr={probe.stderr!r}")
            checks = json.loads(probe.stdout)
            if set(checks) != {'/opt/grader/expected.json', '/opt/grader/grader.py', '/var/run/docker.sock'} or any(v is not False for v in checks.values()):
                raise RuntimeError("grader or daemon socket is visible to agent")
            evidence["agent_paths_absent"] = checks
            execution = await agent.exec(["/usr/local/bin/python", "-I", "-c", "from pathlib import Path; import json; Path('/workspace/output.json').write_text(json.dumps({'schema_version':1,'value':3+4}))"], timeout=15, timeout_retry=False)
            if not execution.success:
                raise RuntimeError(f"synthetic artifact generation failed (exit={execution.returncode}): stdout={execution.stdout!r}; stderr={execution.stderr!r}")
            # Trusted controller freezes bytes before invoking any grader operation.
            frozen = await freeze_artifact(agent)
            if not isinstance(frozen, bytes) or len(frozen) > 4096:
                raise ValueError("invalid frozen artifact")
            evidence["artifact_sha256"] = sha256(frozen)
            (output_dir / "frozen-artifact.json").write_bytes(frozen)
            state.metadata["frozen_artifact"] = frozen.hex()
            evidence["containers"] = capture_containers(run_id)
            for service, record in evidence["containers"].items():
                if record["image_id"] != evidence["images"][service]["id"]:
                    raise ValueError("runtime image identity differs from frozen image")
            return state
        return solve

    @scorer(metrics=[accuracy()])
    def immutable_grade():
        async def score(state, target):
            frozen = bytes.fromhex(state.metadata["frozen_artifact"])
            if sha256(frozen) != evidence["artifact_sha256"]:
                raise ValueError("frozen artifact identity changed")
            grader = sandbox("grader")
            await deliver_artifact(grader, frozen)
            result = await grader.exec(["/usr/local/bin/python", "-I", "/opt/grader/grader.py", "/workspace/frozen-artifact.json"], timeout=15, timeout_retry=False)
            if not result.success:
                raise RuntimeError(f"immutable grader failed: {result.stderr}")
            grading = parse_grade(result.stdout)
            evidence["grading"] = grading
            return Score(value=1 if grading["passed"] else 0, metadata={"is_synthetic": True, "checks": grading["checks"], "artifact_sha256": evidence["artifact_sha256"]})
        return score

    return Task(name="c3_synthetic_infrastructure_smoke", dataset=[Sample(id="synthetic-1", input="Synthetic infrastructure only: write JSON containing the sum 3+4. No model inference.")],
                solver=scripted_artifact(), scorer=immutable_grade(), sandbox=("docker", str(compose_path)),
                metadata={"is_synthetic": True, "purpose": "infrastructure_smoke_only"})


def run_smoke(output_dir: Path) -> dict:
    """Run one serial infrastructure sample; persist manifest on success or failure.

    Images must be built beforehand. No provider credentials or external model
    are used; mockllm is only an Inspect-required model handle, never generated.
    """
    from inspect_ai import eval
    version = importlib.metadata.version("inspect-ai")
    if version != INSPECT_VERSION:
        raise RuntimeError(f"expected inspect-ai {INSPECT_VERSION}, found {version}")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "inspect-smoke.json"
    if manifest_path.exists():
        raise FileExistsError("use a new output directory to preserve smoke evidence")
    run_id = uuid.uuid4().hex
    config = compose_config(run_id)
    compose_path = output_dir / "compose.smoke.yaml"
    compose_bytes = (json.dumps(config, indent=2) + "\n").encode()
    compose_path.write_bytes(compose_bytes)
    evidence = {"is_synthetic": True, "purpose": "infrastructure_smoke_only", "status": "error", "passed": False,
                "run_id": run_id, "inspect_version": version, "model": "mockllm/model", "llm_calls": 0,
                "experiment_concurrency": 1, "compose_sha256": sha256(compose_bytes),
                "source_sha256": {p.name:sha256(p.read_bytes()) for p in [Path(__file__), DOCKER_DIR/'grader.py', DOCKER_DIR/'Dockerfile.agent', DOCKER_DIR/'Dockerfile.grader', DOCKER_DIR/'compose.yaml']}}
    try:
        images = {}
        for service, cfg in config["services"].items():
            identity = json.loads(docker("image", "inspect", cfg["image"]))[0]
            if identity["Architecture"] != "arm64" or identity["Os"] != "linux":
                raise ValueError("expected local linux/arm64 image")
            images[service] = {"id":identity["Id"], "repo_digests":identity.get("RepoDigests", []), "architecture":identity["Architecture"], "os":identity["Os"]}
            # Pin runtime to the actual image content, not a movable local tag.
            cfg["image"] = identity["Id"]
        compose_bytes = (json.dumps(config, indent=2) + "\n").encode()
        compose_path.write_bytes(compose_bytes)
        evidence["compose_sha256"] = sha256(compose_bytes)
        evidence["images"] = images
        logs = eval(make_task(compose_path, output_dir, run_id, evidence), model="mockllm/model", log_dir=str(output_dir/'inspect-logs'),
                    sandbox_prebuilt=True, sandbox_cleanup=True, max_samples=1, max_tasks=1, max_subprocesses=1, max_sandboxes=1,
                    retry_on_error=0, epochs=1, limit=1, fail_on_error=True, score_on_error=False, display="plain", checkpoint=False)
        evidence["inspect_log_paths"] = [log.location for log in logs]
        if len(logs) != 1 or logs[0].status != "success":
            raise RuntimeError("Inspect evaluation did not complete successfully")
        usage = logs[0].stats.model_usage if logs[0].stats else {}
        if usage:
            raise RuntimeError("unexpected model usage in scripted infrastructure smoke")
        if evidence.get("grading", {}).get("passed") is not True:
            raise RuntimeError("synthetic grader did not pass")
        evidence["status"] = "success"
        evidence["passed"] = True
    except Exception as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        evidence.setdefault("inspect_log_paths", [str(p) for p in (output_dir/'inspect-logs').glob('*.eval')])
    finally:
        try:
            leftovers = docker("ps", "-aq", "--filter", f"label=c3.synthetic_run={run_id}").split()
            if leftovers:
                docker("rm", "-f", *leftovers)
            evidence["own_containers_cleaned"] = True
        except Exception as exc:
            evidence["own_containers_cleaned"] = False
            evidence["cleanup_error"] = str(exc)
            evidence["status"] = "error"
            evidence["passed"] = False
        manifest_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    return evidence
