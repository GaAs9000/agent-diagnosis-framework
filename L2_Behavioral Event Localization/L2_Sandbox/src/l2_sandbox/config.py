"""Configuration loader with inline-secret rejection (L2 v0.3 Section 7.2)."""
from __future__ import annotations
import copy
import math
import re
from pathlib import Path
from typing import Any
from .io import L2Error, read_json

SECRET_KEYS = {'api_key', 'password', 'authorization', 'secret', 'access_token', 'token', 'private_key'}

DEFAULTS: dict[str, Any] = {
    'objects': {
        'enabled': True,
        'max_docx_versions': 64,
        'max_xml_bytes': 33554432,
        'max_zip_uncompressed_bytes': 134217728,
        'max_objects': 5000,
    },
    'curve': {
        'n1': 12,
        'n2': 24,
        'stride': 4,
        'alpha': 0.05,
        'max_concurrency': 4,
        'max_invalid_rate': 0.05,
        'max_retries_per_slot': 2,
        'cost_per_branch_usd': 0.01,
    },
    'model': {
        'adapter': 'gemini',
        'kind': 'api',
        'model': 'gemini-2.5-flash-lite-preview-06-17',
        'api_key_env': 'GEMINI_API_KEY',
        'base_url': 'https://generativelanguage.googleapis.com/v1beta',
        'parameters': {'temperature': 0.0, 'maxOutputTokens': 4096},
        'timeout_seconds': 120,
        'include_facts': True,
        'cost_per_call_usd': 0.002,
    },
}


def reject_inline_secrets(value: Any) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            if str(k).lower() in SECRET_KEYS:
                raise L2Error(f'Inline secret key forbidden in config: {k}')
            reject_inline_secrets(v)
    elif isinstance(value, list):
        for item in value:
            reject_inline_secrets(item)


def validate(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise L2Error('Configuration must be a JSON object')
    reject_inline_secrets(config)
    m = config.get('model') or {}
    env_name = m.get('api_key_env')
    if env_name is not None and (not isinstance(env_name, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', env_name)):
        raise L2Error('model.api_key_env must be a valid environment variable name')
    timeout = m.get('timeout_seconds', 120)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise L2Error('model.timeout_seconds must be a positive finite number')
    return config


def load(path: Path | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        raw = read_json(Path(path))
        if not isinstance(raw, dict):
            raise L2Error('Config file must contain a JSON object')
        reject_inline_secrets(raw)
        for k, v in raw.items():
            if k in cfg and isinstance(cfg[k], dict) and isinstance(v, dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return validate(cfg)
