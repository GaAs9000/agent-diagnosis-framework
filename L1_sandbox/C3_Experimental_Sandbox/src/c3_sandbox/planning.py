"""Readiness accounting, never fabricated execution results."""
from .config import validate_config
from .contracts import FILESYSTEM_SCOPE


def build_plan(config: dict) -> dict:
    validate_config(config)
    blockers = []
    model = config.get("model")
    if not isinstance(model, dict) or not all(model.get(key) for key in ("model_id", "version", "budget")):
        blockers.append("model identity, version and budget are not frozen")
    cases = config["cases"]
    if not cases or not all(isinstance(case, dict) and case.get("case_id") and case.get("status") == "frozen" for case in cases):
        blockers.append("case manifest is missing or not frozen")
    if config["probe_library_status"] != "frozen":
        blockers.append("probe library is not frozen after leakage review")
    for key in ("restore_acceptance_verified", "isolation_verified", "independent_scoring_verified", "confirmation_separation_verified"):
        if config.get(key) is not True:
            blockers.append(f"{key} is not established")
    if config["stage"] == "development":
        blockers.append("development stage cannot run actual experiments")
    if not config["experiment_execution_enabled"]:
        blockers.append("experiment execution is disabled")
    # D1-D5 has a deterministic interface runner; real runtime admission is separate.
    blockers.append("native Agent execution and real replay admission are not implemented")
    return {"schema_version": 1, "protocol": "C3-E2-v1.0", "stage": config["stage"], "platform": config["platform"], "concurrency": 1,
            "resources": dict(config["resources"]), "ready_for_experiments": not blockers,
            "readiness_blockers": blockers, "execution_status": "blocked", "snapshot_scope": FILESYSTEM_SCOPE}
