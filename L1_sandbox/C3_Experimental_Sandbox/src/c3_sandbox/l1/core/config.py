"""Strict resolved configuration. Relative paths are relative to the config file."""
from __future__ import annotations
import copy
import os
import re
from pathlib import Path
from ..storage import read_json, digest
from .contracts import ConfigurationError

DEFAULTS = {
 'schema_version': '0.2.0',
 'benchmark': {'adapter':'officeval-native', 'task_id':'officeval_003',
               'source_commit':'5f3da616ac6a0bc30ca35e9678f8f1eb255bd91a',
               'dataset_revision':'cd6ba6d8fb83b3fb551e24eebc20e1fb0bd154a5'},
 'model': {'adapter':'gemini','kind':'api', 'model':'gemini-2.5-flash',
           'base_url':'https://generativelanguage.googleapis.com/v1beta',
           'api_key_env':'GEMINI_API_KEY','parameters':{'temperature':0},
           'timeout_seconds':90, 'supports_images':True, 'max_response_bytes':16777216},
 'runtime': {'adapter':'docker-cold','image':'l1-office:0.2.0','memory':'2g','cpus':2,
             'pids_limit':128,'tool_timeout_seconds':60,'output_limit_bytes':4194304,
             'max_workspace_bytes':268435456, 'max_files':10000, 'max_image_bytes':8388608},
 'evaluation': {'adapter':'docker-verifier','image':'l1-office:0.2.0', 'timeout_seconds':120,
                'outcome_policy':'official_dimensions'},
 'budget': {'max_model_requests':20,'max_tool_calls':40,'wall_time_seconds':1200,
            'max_total_tokens':None},
 'recording': {'checkpoints':True,'objects':False,'observation_limit_chars':20000},
 'agent': {'name':'native-tool-agent','version':'0.2.0'},
 'plugins': {'load_installed':False},
}
ALLOWED = {k:set(v) if isinstance(v,dict) else None for k,v in DEFAULTS.items()}
ALLOWED['benchmark'] |= {'source_dir','data_dir','assets_dir','bundle_dir','data_lock', 'allow_unlocked_data'}
ALLOWED['model'] |= {'reported_revision','provider','tokenizer_digest','retry'}
ALLOWED['agent'] |= {'system_prompt', 'harness', 'intervention'}
for section in ('benchmark','model','runtime','evaluation'): ALLOWED[section].add('options')


def validate(config: dict) -> dict:
    from jsonschema import Draft202012Validator
    # Schema rejects typoed top-level and section keys; semantic checks below keep secrets out.
    properties = {}
    for key, allowed in ALLOWED.items():
        properties[key] = ({'type':'object','additionalProperties':False,
                            'properties':{name:{} for name in allowed}}
                           if allowed is not None else {'const':'0.2.0'})
    schema={'type':'object','additionalProperties':False,'required':list(DEFAULTS), 'properties':properties}
    errors=list(Draft202012Validator(schema).iter_errors(config))
    if errors: raise ConfigurationError(errors[0].message)
    for section, names in {'budget':['max_model_requests','max_tool_calls','wall_time_seconds'],
            'runtime':['cpus','pids_limit','tool_timeout_seconds','output_limit_bytes','max_workspace_bytes','max_files','max_image_bytes'],
            'model':['timeout_seconds','max_response_bytes'], 'evaluation':['timeout_seconds'],
            'recording':['observation_limit_chars']}.items():
        import math
        for name in names:
            x=config[section][name]
            if type(x) not in (int,float) or not math.isfinite(x) or x<=0:
                raise ConfigurationError(f'{section}.{name} must be finite and positive')
    for section,name in [('budget','max_model_requests'),('budget','max_tool_calls'),('runtime','pids_limit'),
                        ('runtime','output_limit_bytes'),('runtime','max_files'),('runtime','max_workspace_bytes'),
                        ('runtime','max_image_bytes'),('model','max_response_bytes'),('recording','observation_limit_chars')]:
        if type(config[section][name]) is not int: raise ConfigurationError(f'{section}.{name} must be integer')
    tokens=config['budget']['max_total_tokens']
    if tokens is not None and (type(tokens) is not int or tokens<1): raise ConfigurationError('max_total_tokens must be positive integer or null')
    if config['model']['kind'] not in ('api','local'): raise ConfigurationError('model.kind must be api or local')
    for s,k in [('recording','checkpoints'),('recording','objects'),('model','supports_images'),('plugins','load_installed')]:
        if type(config[s][k]) is not bool: raise ConfigurationError(f'{s}.{k} must be boolean')
    if config['recording']['objects']: raise ConfigurationError('object-level recording is not implemented; use files or a state plugin')
    if not isinstance(config['model']['parameters'],dict): raise ConfigurationError('model.parameters must be an object')
    if 'harness' in config['agent'] and config['agent']['harness'] is not None:
        if config['agent']['harness'] not in ('minimal', 'pi'):
            raise ConfigurationError('agent.harness must be minimal or pi')
    if 'intervention' in config['agent'] and config['agent']['intervention'] is not None:
        iv = config['agent']['intervention']
        if not isinstance(iv, dict) or not set(iv) <= {'at_request', 'message'}:
            raise ConfigurationError('agent.intervention must be object with at_request and message')
        if type(iv.get('at_request')) is not int or iv['at_request'] < 1:
            raise ConfigurationError('agent.intervention.at_request must be positive integer')
        if not isinstance(iv.get('message'), str) or not iv['message'].strip():
            raise ConfigurationError('agent.intervention.message must be nonempty string')
    if 'retry' in config['model'] and config['model']['retry'] is not None:
        import math
        retry=config['model']['retry']
        if not isinstance(retry,dict): raise ConfigurationError('model.retry must be an object')
        allowed_retry={'max_attempts','retry_on_status','retry_on_transport_error','backoff_seconds','honor_retry_after'}
        if not set(retry)<=allowed_retry: raise ConfigurationError('Unknown keys in model.retry')
        ma=retry.get('max_attempts')
        if type(ma) is not int or not 1<=ma<=16: raise ConfigurationError('model.retry.max_attempts must be integer 1..16')
        statuses=retry.get('retry_on_status',[429,500,502,503,504])
        if not isinstance(statuses,list) or any(type(s) is not int or not 100<=s<=599 for s in statuses):
            raise ConfigurationError('model.retry.retry_on_status must be list of HTTP status integers')
        if 'retry_on_transport_error' in retry and type(retry['retry_on_transport_error']) is not bool:
            raise ConfigurationError('model.retry.retry_on_transport_error must be boolean')
        if 'honor_retry_after' in retry and type(retry['honor_retry_after']) is not bool:
            raise ConfigurationError('model.retry.honor_retry_after must be boolean')
        backoffs=retry.get('backoff_seconds',[2,8,30])
        if not isinstance(backoffs,list) or any(type(b) not in (int,float) or not math.isfinite(b) or b<0 for b in backoffs):
            raise ConfigurationError('model.retry.backoff_seconds must be list of non-negative numbers')
    forbidden={'api_key','apikey','authorization','password','access_token','secret'}
    def check(value):
        if isinstance(value,dict):
            for k,v in value.items():
                if k.lower() in forbidden: raise ConfigurationError('Inline secrets forbidden; use api_key_env')
                check(v)
        elif isinstance(value,list):
            for v in value: check(v)
    check(config)
    for section in ('benchmark','model','runtime','evaluation'):
        if 'options' in config[section] and not isinstance(config[section]['options'],dict):
            raise ConfigurationError('plugin options must be an object')
    env=config['model']['api_key_env']
    if env is not None and (not isinstance(env,str) or not re.fullmatch('[A-Za-z_][A-Za-z0-9_]*',env)):
        raise ConfigurationError('api_key_env must name an environment variable or be null')
    if config['evaluation']['outcome_policy'] not in ('official_dimensions','full_score_v1'):
        raise ConfigurationError('Unknown outcome_policy')
    for section in ('benchmark','model','runtime','evaluation'):
        if not isinstance(config[section]['adapter'],str): raise ConfigurationError('adapter must be string')
    if not isinstance(config['model']['model'],str) or not config['model']['model']:
        raise ConfigurationError('model name required')
    return config


def resolve(path: Path) -> dict:
    path=Path(path).absolute()
    given=read_json(path)
    if not isinstance(given,dict): raise ConfigurationError('Config must be object')
    result=copy.deepcopy(DEFAULTS)
    for k,v in given.items():
        if k not in result: raise ConfigurationError(f'Unknown config section {k}')
        if isinstance(result[k],dict):
            if not isinstance(v,dict): raise ConfigurationError(f'{k} must be object')
            result[k].update(v)
        else: result[k]=v
    for k in ('source_dir','data_dir','assets_dir','bundle_dir','data_lock'):
        if k in result['benchmark']:
            raw=result['benchmark'][k]
            if not isinstance(raw,str) or not raw: raise ConfigurationError(f'{k} must be path string')
            p=Path(os.path.expanduser(raw))
            result['benchmark'][k]=str((path.parent/p).resolve() if not p.is_absolute() else p.resolve())
    return validate(result)
