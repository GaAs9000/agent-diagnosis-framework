"""Validate the deliberately serial local configuration without starting work."""
import json
from pathlib import Path


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def validate_config(config: dict) -> dict:
    if not isinstance(config, dict):
        raise ValueError("configuration must be an object")
    if type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    if config.get("platform") != "linux/arm64":
        raise ValueError("local platform must be linux/arm64")
    if type(config.get("concurrency")) is not int or config["concurrency"] != 1:
        raise ValueError("concurrency must be exactly 1")
    if config.get("backend") != "inspect_docker":
        raise ValueError("backend must be inspect_docker")
    if config.get("stage") not in {"development", "p0", "p1"}:
        raise ValueError("stage must be development, p0 or p1")
    if type(config.get("experiment_execution_enabled")) is not bool:
        raise ValueError("experiment_execution_enabled must be explicit boolean")
    resources = config.get("resources")
    if not isinstance(resources, dict):
        raise ValueError("resources must be an object")
    bounds = {"cpus": (1, 2), "memory_mb": (256, 4096), "pids_limit": (16, 256), "tool_timeout_seconds": (1, 300)}
    for key, (lower, upper) in bounds.items():
        value = resources.get(key)
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"{key} must be an integer in [{lower}, {upper}]")
    if not isinstance(config.get("cases"), list):
        raise ValueError("cases must be a list")
    if config.get("probe_library_status") not in {"draft", "reviewed", "frozen"}:
        raise ValueError("probe_library_status must be draft, reviewed or frozen")
    return config


def load_config(path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle, object_pairs_hook=_unique, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite JSON: {value}")))
    return validate_config(config)
