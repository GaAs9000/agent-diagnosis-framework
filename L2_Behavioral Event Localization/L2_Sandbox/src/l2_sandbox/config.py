"""Only explicit retrospective single-Episode visibility is released."""
from __future__ import annotations
import copy
import math
import re
from pathlib import Path
from .io import L2Error, read_json

DEFAULTS = {
    'schema_version': 'l2-0.1.0', 'mode': 'rules', 'analysis_mode': 'retrospective',
    'source_policy': 'strict', 'require_l1_verifier': False, 'expected_source_index_sha256': None,
    'objects': {'enabled': True, 'max_docx_versions': 64, 'max_xml_bytes': 33554432,
                'max_zip_uncompressed_bytes': 134217728, 'max_objects': 5000},
    'budget': {'max_model_requests': 12, 'max_tool_calls': 40, 'max_output_repairs': 2,
               'wall_seconds': 600, 'max_tool_response_bytes': 48000, 'max_total_read_bytes': 1000000,
               'max_findings': 12},
    'model': {'adapter': 'gemini', 'kind': 'api', 'model': 'CONFIGURE_YOUR_MODEL',
              'api_key_env': 'GEMINI_API_KEY', 'base_url': 'https://generativelanguage.googleapis.com/v1beta',
              'parameters': {'temperature': 0, 'maxOutputTokens': 8192},
              'timeout_seconds': 90, 'max_response_bytes': 4194304, 'supports_images': False},
}

def validate(config: dict) -> dict:
    if not isinstance(config, dict) or set(config) != set(DEFAULTS): raise L2Error('Unknown/missing configuration fields')
    if config['schema_version'] != DEFAULTS['schema_version']: raise L2Error('Unsupported config version')
    if config['mode'] not in ('rules', 'analyst', 'structured-once'): raise L2Error('mode must be rules, analyst or structured-once')
    if config['analysis_mode'] != 'retrospective': raise L2Error('Prefix and privileged profiles are not released')
    if config['source_policy'] != 'strict': raise L2Error('Incomplete sources are inspectable but cannot enter strict analyze')
    if type(config['require_l1_verifier']) is not bool: raise L2Error('require_l1_verifier must be boolean')
    anchor = config['expected_source_index_sha256']
    if anchor is not None and (not isinstance(anchor, str) or not re.fullmatch('[0-9a-f]{64}', anchor)):
        raise L2Error('Invalid source index SHA256')
    for section in ('objects', 'budget', 'model'):
        if not isinstance(config[section], dict) or set(config[section]) != set(DEFAULTS[section]):
            raise L2Error('Unknown/missing config keys: ' + section)
    if type(config['objects']['enabled']) is not bool: raise L2Error('objects.enabled must be boolean')
    for section, fields in [('objects', set(config['objects']) - {'enabled'}), ('budget', set(config['budget']))]:
        for k in fields:
            x = config[section][k]
            if type(x) is not int or x < (0 if k == 'max_output_repairs' else 1): raise L2Error(f'{section}.{k}: nonnegative/positive integer required')
    m = config['model']
    if m['adapter'] not in ('gemini', 'openai-compatible') or m['kind'] not in ('api', 'local'):
        raise L2Error('Only supported L1 native backends are exposed')
    if not isinstance(m['parameters'], dict): raise L2Error('model.parameters must be object')
    if not isinstance(m['model'], str) or not m['model']: raise L2Error('Model name required')
    if m['supports_images'] is not False: raise L2Error('Analyst view currently provides text/object facts, not image rendering')
    if type(m['timeout_seconds']) not in (int, float) or not math.isfinite(m['timeout_seconds']) or m['timeout_seconds'] <= 0: raise L2Error('Invalid model timeout')
    if type(m['max_response_bytes']) is not int or m['max_response_bytes'] < 1: raise L2Error('Invalid response cap')
    if m['api_key_env'] is not None and (not isinstance(m['api_key_env'], str) or not re.fullmatch('[A-Za-z_][A-Za-z0-9_]*', m['api_key_env'])):
        raise L2Error('api_key_env must name a variable; never place credentials in JSON')
    def secrets(v):
        if isinstance(v, dict):
            for key, value in v.items():
                if key.lower() in {'api_key', 'password', 'authorization', 'secret', 'access_token'}: raise L2Error('Inline secrets forbidden')
                secrets(value)
        elif isinstance(v, list):
            for item in v: secrets(item)
    secrets(config)
    return config

def load(path: Path | None = None) -> dict:
    out = copy.deepcopy(DEFAULTS)
    given = read_json(path) if path else {}
    if not isinstance(given, dict): raise L2Error('Config must be object')
    for k, v in given.items():
        if k not in out: raise L2Error('Unknown config key: ' + k)
        if isinstance(out[k], dict):
            if not isinstance(v, dict): raise L2Error('Config section must be object')
            out[k].update(v)
        else: out[k] = v
    return validate(out)
