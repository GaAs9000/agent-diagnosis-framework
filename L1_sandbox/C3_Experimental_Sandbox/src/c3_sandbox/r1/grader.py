"""OfficeVal grading adapter that has no model-runtime dependencies.

The native Gemini path only needs the independent grader.  Keeping this small
module separate from :mod:`office_run` prevents importing the local
``inspect_ai`` agent stack when the selected generator is a remote native API.
The grader still uses the same pinned verifier image and normalization rules as
the original R1 runner.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

from ..evidence._storage import encode
from .docker_runtime import DockerRuntime
from .officeval import normalize_official


def _image(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise ValueError("immutable Docker image ID required")
    return value


def _name(value: str) -> str:
    if not isinstance(value, str) or value in ("", ".", "..") or Path(value).name != value or "\\" in value:
        raise ValueError("unsafe public filename")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value) -> None:
    with path.open("xb") as stream:
        stream.write(encode(value))


def grade_workspace(task_id: str, manifest: dict, workspace: Path,
                    grader_image: str, run_dir: Path) -> dict:
    """Grade one designated DOCX in an independent verifier container.

    This function intentionally performs no candidate or model imports.  It is
    safe to use from remote-model runners whose host environment does not have
    the local Inspect runtime installed.
    """
    _image(grader_image)
    task = manifest["tasks"][task_id]
    name = _name(task["expected_output"])
    if Path(name).suffix.lower() != ".docx":
        raise ValueError("designated output must be DOCX")
    run_dir = Path(run_dir).absolute()
    run_dir.mkdir(parents=True, exist_ok=False)
    staging = run_dir / "input"
    submission = staging / "submission"
    submission.mkdir(parents=True, mode=0o700)
    staging.chmod(0o700)
    candidate = Path(workspace) / name
    artifact = None
    rejected = None
    if candidate.is_symlink() or (candidate.exists() and
            (not candidate.is_file() or candidate.stat().st_nlink != 1)):
        rejected = "invalid_designated_deliverable"
    elif candidate.is_file():
        shutil.copyfile(candidate, submission / name)
        (submission / name).chmod(0o644)
        artifact = {"name": name, "sha256": _sha(submission / name),
                    "size_bytes": (submission / name).stat().st_size}
    with DockerRuntime(grader_image, run_dir / "runtime", is_grader=True) as grader:
        grader.seed_workspace(staging)
        process = grader.exec_fixed(["python", "-I", "/opt/grader/r1_grader.py",
                                    task_id, "/workspace/submission"])
    _write(run_dir / "process.json", process)
    (run_dir / "stdout.txt").write_text(process.get("stdout", ""))
    (run_dir / "stderr.txt").write_text(process.get("stderr", ""))
    parsed = None
    result = {"official_raw": None, "validity": "infrastructure_invalid",
              "c3_operational_completed": False, "outcome": "infrastructure_invalid",
              "reason": "invalid_grader_output", "coverage_complete": None,
              "unassessed_rules": []}
    if process.get("exit_code") == 0 and not process.get("error") and not process.get("truncated"):
        try:
            parsed = json.loads(process["stdout"])
            if isinstance(parsed, dict) and parsed.get("official_raw") is not None:
                normalized = normalize_official(parsed["official_raw"], task_id, task)
                if all(parsed.get(key) == value for key, value in normalized.items()):
                    result = normalized
            elif isinstance(parsed, dict) and parsed.get("validity") == "valid" and parsed.get("reason") in {
                    "missing_deliverable", "empty_deliverable", "invalid_deliverable_inventory"}:
                result.update(validity="valid", outcome="task_failed",
                              reason=parsed["reason"], coverage_complete=False)
            elif isinstance(parsed, dict) and parsed.get("validity") == "infrastructure_invalid":
                result["reason"] = str(parsed.get("reason", "invalid_grader_output"))
        except (ValueError, KeyError, TypeError):
            pass
    if rejected and result["validity"] == "valid":
        result.update(validity="valid", outcome="task_failed", reason=rejected,
                      c3_operational_completed=False, coverage_complete=False)
    result.update(grader_image=grader_image, artifact=artifact, raw_process=process,
                  grader_json=parsed, run_dir=str(run_dir),
                  process_sha256=_sha(run_dir / "process.json"),
                  stdout_sha256=_sha(run_dir / "stdout.txt"),
                  stderr_sha256=_sha(run_dir / "stderr.txt"))
    _write(run_dir / "grade.json", result)
    return result
