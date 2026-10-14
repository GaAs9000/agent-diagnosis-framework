"""Immutable OfficeVal grader. Invoke with Python -I; never execute candidate code."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

# Under -I only this immutable directory is added; /candidate never enters sys.path.
GRADER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(GRADER_ROOT))
from officeval import normalize_official  # noqa: E402

OFFICEVAL_ROOT = Path("/opt/officeval")


def grade(task_id: str, candidate: Path, officeval_root: Path = OFFICEVAL_ROOT) -> dict:
    raw = None
    def invalid(reason):
        return {"official_raw": raw, "validity": "infrastructure_invalid", "c3_operational_completed": False,
                "outcome": "infrastructure_invalid", "reason": reason,
                "coverage_complete": None, "unassessed_rules": []}
    try:
        manifest = json.loads((officeval_root / "manifest.json").read_text())
        task = manifest["tasks"][task_id]
        verifier = officeval_root / task["verifier_path"]
        if verifier.parent.resolve() != (officeval_root / "verifiers").resolve() or verifier.is_symlink():
            return invalid("invalid_verifier_path")
        if hashlib.sha256(verifier.read_bytes()).hexdigest() != manifest["files"][task["verifier_path"]]["sha256"]:
            return invalid("verifier_hash_mismatch")
        if not candidate.is_dir() or candidate.is_symlink():
            return invalid("candidate_directory_unavailable")
        entries = list(candidate.iterdir())
        # Only deliverables enter this mount, never reference PDFs/images or agent programs.
        if not entries:
            return {"official_raw": None, "validity": "valid", "c3_operational_completed": False,
                    "outcome": "task_failed", "reason": "missing_deliverable",
                    "coverage_complete": False, "unassessed_rules": []}
        if (len(entries) != 1 or entries[0].is_symlink() or not entries[0].is_file()
                or entries[0].suffix.lower() != ".docx" or entries[0].name.startswith("~$")
                or (task.get("exact_output_name") and entries[0].name != task["expected_output"])):
            return {"official_raw": None, "validity": "valid", "c3_operational_completed": False,
                    "outcome": "task_failed", "reason": "invalid_deliverable_inventory",
                    "coverage_complete": False, "unassessed_rules": []}
        if entries[0].stat().st_size == 0:
            return {"official_raw": None, "validity": "valid", "c3_operational_completed": False,
                    "outcome": "task_failed", "reason": "empty_deliverable",
                    "coverage_complete": False, "unassessed_rules": []}
        # Absolute import location and -I isolate official code from agent imports.
        spec = importlib.util.spec_from_file_location("official_" + task_id, verifier)
        if spec is None or spec.loader is None:
            return invalid("official_import_failed")
        module = importlib.util.module_from_spec(spec)
        # Match Python import semantics: dataclass annotation resolution reads
        # sys.modules[cls.__module__]. The scorer source remains unmodified.
        previous = sys.modules.get(spec.name)
        sys.modules[spec.name] = module
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                spec.loader.exec_module(module)
                raw = module.evaluate(str(candidate.resolve()))
        finally:
            if previous is None:
                sys.modules.pop(spec.name, None)
            else:
                sys.modules[spec.name] = previous
        return normalize_official(raw, task_id, task)
    except Exception as exc:
        return invalid("grader_exception:" + type(exc).__name__)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(json.dumps({"official_raw": None, "validity": "infrastructure_invalid",
                          "c3_operational_completed": False, "outcome": "infrastructure_invalid",
                          "reason": "expected_task_id_and_candidate_directory",
                          "coverage_complete": None, "unassessed_rules": []}))
        raise SystemExit(2)
    print(json.dumps(grade(sys.argv[1], Path(sys.argv[2])), ensure_ascii=False, allow_nan=False))
