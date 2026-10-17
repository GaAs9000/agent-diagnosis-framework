"""Bind experimental readiness to recorded measurement evidence, not approval intent."""
from __future__ import annotations

from pathlib import Path

from .v3_execution import file_digest, read_json


def measurement_gate_binding(path: Path) -> tuple[dict, list[str]]:
    errors = []
    binding = {"path": str(path.resolve()), "sha256": None, "passed": False, "gates": {}}
    if not path.is_file():
        return binding, ["measurement_gate_audit_missing"]
    binding["sha256"] = file_digest(path)
    try:
        audit = read_json(path)
        g1 = audit.get("V3_G1", {})
        g2 = audit.get("V3_G2_R19", {})
        g3 = audit.get("V3_G3_R20", {})
        g4 = audit.get("V3_G4_R21", {})
        passed = {
            "V3-G1": g1.get("byte_equivalence_verified") is True and g1.get("applicable_configuration_verified") is True,
            "V3-G2": g2.get("cp000000_all_pre_first_request_valid") is True
                and g2.get("completed_valid_pairs", 0) > 0
                and g2.get("ci95_completely_inside_pm_005") is True,
            "V3-G3": g3.get("sampled_checkpoint_count") == 20
                and g3.get("human_review_completed_count") == 20
                and g3.get("missing_restored_directory_count") == 0
                and g3.get("missing_scorer_exit_count") == 0
                and g3.get("human_conclusions_verified_against_source") is True,
            "V3-G4": g4.get("normal_and_p_requests_scanned", 0) > 0
                and g4.get("unverified_request_count") == 0
                and g4.get("leaked_request_count") == 0
                and g4.get("zero_diagnostic_leakage_verified_on_scanned") is True,
        }
        binding["gates"] = passed
        binding["passed"] = all(passed.values())
        errors.extend(f"measurement_gate_not_passed:{name}" for name, ok in passed.items() if not ok)
    except (ValueError, TypeError, AttributeError) as exc:
        errors.append(f"measurement_gate_audit_invalid:{type(exc).__name__}")
    return binding, errors


def verify_measurement_gate_binding(plan: dict) -> None:
    """Called before production D1 admission and conditional D2 gate decisions.

    This validates the frozen readiness record. The audit producer remains
    responsible for validating the underlying packages and human review sources.
    """
    frozen = plan.get("measurement_gate_binding")
    if not isinstance(frozen, dict) or not frozen.get("path") or frozen.get("passed") is not True:
        raise ValueError("production execution requires verified measurement gates")
    current, errors = measurement_gate_binding(Path(frozen["path"]))
    if errors or current != frozen:
        raise ValueError("measurement gate evidence changed or remains incomplete")
