"""Frozen R1.2 execution system identities; no qualification or generation."""
from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path
import hashlib

from ..manifests import canonical_hash
from .tokenizer_client import TokenizerClient

ROOT = Path(__file__).resolve().parents[3]
MODEL_REPOSITORY = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"
MODEL_REVISION = "6e302ea604ad9ab206367e2c501d1571023e7b6d"
MODEL_PATH = "models/Qwen3-Coder-30B-A3B-Instruct-4bit"
TOOL_PARSER = "mlx_lm.tool_parsers.qwen3_coder.parse_tool_call"
TOOL_PARSER_SHA256 = "2d212e4e6f175178bd80b7db9963ccad613e29419f976e6d6b3272417f953722"
SAMPLING = {"temperature": 0, "seed": 2026, "top_k": 1, "min_p": 0,
            "max_tokens": 4096, "context_window": 32768, "max_turns": 20,
            "generation_timeout_seconds": 240, "max_retries": 0, "parallel_tool_calls": False}
SEED_DOMAINS = {"qualification": 2026, "original": 2026, "exploration": 2026,
                "confirmation": int.from_bytes(hashlib.sha256(
                    b"C3-E2-R1.2:2026:confirmation").digest()[:4], "big")}


def validated_system(system: dict) -> dict:
    """Validate and detach a finite JSON system contract, preserving extra identity fields."""
    canonical_hash(system)  # Reject non-JSON values, NaN, and non-string keys.
    if type(system) is not dict:
        raise ValueError("system must be an object")
    result = json.loads(json.dumps(system, allow_nan=False))
    required = {"schema_version", "system_id", "model", "sampling", "system_prompt",
                "tool_schemas", "agent_image", "server", "seed_domains"}
    if not required <= result.keys() or result["schema_version"] != "r12-system-v1":
        raise ValueError("incomplete r12-system-v1 contract")
    for key in ("system_id", "system_prompt", "agent_image"):
        if type(result[key]) is not str or not result[key].strip():
            raise ValueError(f"system requires nonempty {key}")
    expected_model = {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION,
                      "path": MODEL_PATH, "quantization": "4bit", "tool_parser": TOOL_PARSER,
                      "tool_parser_sha256": TOOL_PARSER_SHA256}
    for group, expected in (("model", expected_model), ("sampling", SAMPLING),
            ("server", {"host": "127.0.0.1", "port": 18080, "tokenizer_port": 18081,
                        "prompt_cache_size": 0, "prompt_concurrency": 1, "decode_concurrency": 1,
                        "packages_lock": "requirements.model.local-arm64.lock.txt"})):
        actual = result[group]
        if type(actual) is not dict:
            raise ValueError(f"system {group} must be an object")
        for key, value in expected.items():
            if (key not in actual or actual[key] != value
                    or (type(actual[key]) is bool) != (type(value) is bool)
                    or (type(value) is int and type(actual[key]) is not int)):
                raise ValueError(f"system {group}.{key} must equal {value!r}")
    if type(result["tool_schemas"]) is not list or not result["tool_schemas"]:
        raise ValueError("system requires tool_schemas")
    if (type(result["seed_domains"]) is not dict or result["seed_domains"] != SEED_DOMAINS
            or any(type(seed) is not int for seed in result["seed_domains"].values())):
        raise ValueError("seed_domains differs from the frozen phase derivation")
    return result


def system_identity(system: dict) -> str:
    return canonical_hash(validated_system(system))


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_model(system: dict, tokenizer: TokenizerClient | None = None) -> dict:
    """Hash all local model files and validate observed tokenizer/parser provenance."""
    system = validated_system(system)
    folder = ROOT / system["model"]["path"]
    # Disallow a redirected model root and external symlinks, including directory links.
    if folder.is_symlink() or folder.resolve(strict=True) != folder.absolute():
        raise ValueError("model root must not be redirected")
    artifacts = {}
    for path in sorted(folder.rglob("*")):
        relative = path.relative_to(folder)
        # HF local_dir acquisition state is mutable and is logged separately.
        if ".cache" in relative.parts or path.name.endswith(".lock"):
            continue
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(folder):
            raise ValueError("model symlink escapes approved root")
        if path.is_symlink() and path.is_dir():
            raise ValueError("model directory symlinks are unsupported")
        if path.is_file():
            artifacts[relative.as_posix()] = path
    if not artifacts or not any(name.endswith(".safetensors") for name in artifacts):
        raise ValueError("model weights are missing")
    for name, path in artifacts.items():
        if not name.endswith(".safetensors.index.json"):
            continue
        index = json.loads(path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map") if type(index) is dict else None
        if (type(weight_map) is not dict or not weight_map
                or not all(type(key) is str and key and type(shard) is str
                           and shard.endswith(".safetensors") for key, shard in weight_map.items())):
            raise ValueError("model shard index requires a nonempty weight_map")
        if any(shard not in artifacts for shard in weight_map.values()):
            raise ValueError("model shard index references a missing or excluded shard")
    files = {name: _file_hash(path) for name, path in artifacts.items()}
    model_config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    quantization = model_config.get("quantization", model_config.get("quantization_config"))
    if type(quantization) is not dict or type(quantization.get("bits")) is not int or quantization["bits"] != 4:
        raise ValueError("model config does not declare 4bit quantization")
    observed = (tokenizer or TokenizerClient()).text_details("fingerprint")["tokenizer_fingerprint"]
    if (observed.get("revision") != system["model"]["revision"]
            or observed.get("tool_parser") != TOOL_PARSER
            or observed.get("tool_parser_sha256") != TOOL_PARSER_SHA256):
        raise ValueError("observed tokenizer revision/parser differs from frozen system")
    tokenizer_identity = {key: value for key, value in observed.items() if key != "sha256"}
    # Tokenizer service historically hashes compact UTF-8 JSON rather than canonical ASCII JSON.
    expected = hashlib.sha256(json.dumps(tokenizer_identity, sort_keys=True,
                             separators=(",", ":")).encode()).hexdigest()
    if observed.get("sha256") != expected:
        raise ValueError("tokenizer fingerprint hash is invalid")
    for name, digest in observed.get("file_sha256", {}).items():
        if files.get(name) != digest:
            raise ValueError("tokenizer file differs from model file")
    descriptor = {"repository": system["model"]["repository"],
                  "revision": system["model"]["revision"],
                  "quantization": system["model"]["quantization"],
                  "quantization_config": quantization, "files": files,
                  "server_packages_lock": _file_hash(ROOT / system["server"]["packages_lock"]),
                  "inspect_version": version("inspect-ai"),
                  "tool_parser": observed["tool_parser"],
                  "tool_parser_sha256": observed["tool_parser_sha256"],
                  "tokenizer": observed, "system_contract_sha256": system_identity(system)}
    return {**descriptor, "sha256": canonical_hash(descriptor)}
