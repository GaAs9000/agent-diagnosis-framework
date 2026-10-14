"""Finite JSON identities and split-safe family/system manifests (C3-E2 D2)."""
import hashlib
import json
import math
import re

SCHEMA_VERSION = "1.0"
SPLITS = {"development", "exploration", "confirmation", "transfer"}
CONFIG_FIELDS = {"model", "agent", "tools", "prompt", "sampling", "runtime"}


def _finite_json(value):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _finite_json(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _finite_json(item)
        return
    raise ValueError("expected finite JSON values and string object keys")


def _copy(value):
    _finite_json(value)
    return json.loads(json.dumps(value, allow_nan=False))


def canonical_hash(value) -> str:
    """SHA256 of sorted, compact, ASCII-escaped finite JSON (no normalization)."""
    _finite_json(value)
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _object(value, fields):
    if not isinstance(value, dict) or not fields <= value.keys():
        raise ValueError(f"object requires {sorted(fields)}")


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("expected a nonempty string")


def _hash(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("expected lowercase SHA256")


def _version(record):
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")


def system_identity(record) -> str:
    """Validate a system entry and return its config identity, excluding its label."""
    _finite_json(record)
    _object(record, {"system_id", "config", "config_sha256"})
    _text(record["system_id"])
    config = record["config"]
    _object(config, CONFIG_FIELDS)
    if set(config) != CONFIG_FIELDS:
        raise ValueError("config must specify exactly model, agent, tools, prompt, sampling, runtime")
    for name in CONFIG_FIELDS:
        value = config[name]
        if value is None or value == "" or value == {}:
            raise ValueError(f"config.{name} must explicitly describe the configuration")
    _hash(record["config_sha256"])
    identity = canonical_hash(config)
    if record["config_sha256"] != identity:
        raise ValueError("config_sha256 does not match full configuration")
    return identity


def validate_system_manifest(record: dict) -> dict:
    record = _copy(record)
    _object(record, {"schema_version", "systems"})
    _version(record)
    if not isinstance(record["systems"], list) or not record["systems"]:
        raise ValueError("systems must be a nonempty list")
    seen = set()
    for system in record["systems"]:
        system_identity(system)
        if system["system_id"] in seen:
            raise ValueError("duplicate system_id")
        seen.add(system["system_id"])
    return record


def validate_family_manifest(record: dict) -> dict:
    record = _copy(record)
    _object(record, {"schema_version", "families"})
    _version(record)
    if not isinstance(record["families"], list) or not record["families"]:
        raise ValueError("families must be a nonempty list")
    families, tasks, sources, assets, hash_splits = set(), set(), {}, {}, {}
    for family in record["families"]:
        _object(family, {"task_family_id", "split", "task_ids", "source_group_ids", "asset_hashes"})
        _text(family["task_family_id"])
        split = family["split"]
        if not isinstance(split, str) or split not in SPLITS:
            raise ValueError("unknown split")
        if family["task_family_id"] in families:
            raise ValueError("duplicate task_family_id (a family cannot cross splits)")
        families.add(family["task_family_id"])
        for name in ("task_ids", "source_group_ids"):
            values = family[name]
            if not isinstance(values, list) or not values:
                raise ValueError(f"{name} must be a nonempty list")
            for value in values:
                _text(value)
            if len(set(values)) != len(values):
                raise ValueError(f"duplicate {name}")
        for task in family["task_ids"]:
            if task in tasks:
                raise ValueError("duplicate task_id")
            tasks.add(task)
        for source in family["source_group_ids"]:
            if source in sources and sources[source] != split:
                raise ValueError("source group leaks across splits")
            sources[source] = split
        hashes = family["asset_hashes"]
        if not isinstance(hashes, dict) or not hashes:
            raise ValueError("asset_hashes must be a nonempty asset identity to SHA256 map")
        for asset, digest in hashes.items():
            _text(asset)
            _hash(digest)
            if asset in assets and assets[asset] != digest:
                raise ValueError("conflicting source asset hashes")
            if digest in hash_splits and hash_splits[digest] != split:
                raise ValueError("identical source asset leaks across splits")
            assets[asset], hash_splits[digest] = digest, split
    return record
