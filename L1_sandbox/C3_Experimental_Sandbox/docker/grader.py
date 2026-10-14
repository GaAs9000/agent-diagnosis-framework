"""Immutable synthetic checker; task artifacts can never supply a score."""
import json
import sys
from pathlib import Path


def grade_bytes(raw: bytes, expected: int) -> dict:
    try:
        artifact = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("invalid artifact JSON") from exc
    if not isinstance(artifact, dict) or set(artifact) != {"schema_version", "value"}:
        raise ValueError("artifact fields must be exactly schema_version and value")
    if type(artifact["schema_version"]) is not int or artifact["schema_version"] != 1:
        raise ValueError("invalid artifact version")
    if type(artifact["value"]) is not int:
        raise ValueError("artifact value must be an integer")
    return {"schema_version": 1, "is_synthetic": True, "passed": artifact["value"] == expected,
            "checks": {"artifact_schema_valid": True, "value_matches": artifact["value"] == expected}}


if __name__ == "__main__":
    try:
        result = grade_bytes(Path(sys.argv[1]).read_bytes(), json.loads(Path('/opt/grader/expected.json').read_bytes()))
        print(json.dumps(result, sort_keys=True))
    except (OSError, ValueError, IndexError) as exc:
        print(json.dumps({"error": str(exc), "is_synthetic": True}), file=sys.stderr)
        raise SystemExit(2)
