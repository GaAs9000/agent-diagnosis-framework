"""Evidence-bound engineering gate; never an automatic scientific pass."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
CHECKS = {
    "1_cache_one_execution_two_logical_charges", "2_method_observation_isolation",
    "3_commit_before_query_and_immutable", "4_fresh_original_controls_both_lines",
    "5_unrestorable_checkpoint_never_executes", "6_confirmation_sealed_until_all_exploration_ends",
}
SUITES = {"test_e2_data", "test_e2_probes", "test_e2_replay", "test_e2_scheduler", "test_e2_integration"}


def source_identities(root=ROOT):
    root = Path(root)
    files = list((root / "src/c3_sandbox").rglob("*.py")) + list((root / "tests").glob("*.py"))
    files += list((root / "schemas").glob("*.json"))
    files += [root / "configs/probes.v1.json", root / "pyproject.toml", root / "docs/C3_E2_PROTOCOL_v1_0.md"]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def verify_foundation(acceptance_path, junit_path):
    """Validate archived evidence against current code; trusted test producer assumed."""
    acceptance_path, junit_path = Path(acceptance_path), Path(junit_path)
    result = json.loads(acceptance_path.read_bytes())
    if result.get("status") != "passed" or result.get("is_synthetic") is not True:
        raise ValueError("successful explicitly synthetic integration evidence required")
    checks = result.get("checks")
    if not isinstance(checks, dict) or set(checks) != CHECKS or any(v is not True for v in checks.values()):
        raise ValueError("all six acceptance invariants must have executed and passed")
    if result.get("source_identities") != source_identities():
        raise ValueError("acceptance evidence is stale relative to code/schema/config/tests")
    xml = ET.fromstring(junit_path.read_bytes())
    cases = list(xml.iter("testcase"))
    covered = {c.get("classname", "").split(".")[-1] for c in cases}
    if not SUITES <= covered:
        raise ValueError("D1-D5 plus integration test suites must all be present")
    if not cases or any(c.find(tag) is not None for c in cases for tag in ("failure", "error", "skipped")):
        raise ValueError("tests must complete without failure/error/skip")
    for suite in xml.iter("testsuite"):
        if any(int(suite.get(name, "0")) for name in ("failures", "errors", "skipped")):
            raise ValueError("test suite did not pass completely")
    return {
        "protocol": "C3-E2-v1.0", "engineering_contracts_verified": ["D1", "D2", "D3", "D4", "D5"],
        "integration_invariants_passed": 6, "junit_testcase_count": len(cases),
        "acceptance_sha256": hashlib.sha256(acceptance_path.read_bytes()).hexdigest(),
        "junit_sha256": hashlib.sha256(junit_path.read_bytes()).hexdigest(),
        "source_identities": result["source_identities"],
        "may_develop_D6_D8": True, "formal_experiments_admitted": False,
        "real_replay_qualified": False, "real_probe_reviews_complete": False,
        "note": "Deterministic interface contracts only; not a result of scientific efficacy or hostile-host isolation.",
    }
