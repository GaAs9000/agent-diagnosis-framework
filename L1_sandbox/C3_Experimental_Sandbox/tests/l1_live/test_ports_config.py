import copy
import json
import os
from pathlib import Path
import pytest
from c3_sandbox.l1.core.config import resolve,validate,DEFAULTS
from c3_sandbox.l1.core.registry import Registry
from c3_sandbox.l1.core.contracts import ConfigurationError
from c3_sandbox.l1.models.native import OpenAICompatibleBackend, GeminiBackend

@pytest.mark.parametrize('section,key,value',[
 ('budget','max_tool_calls',0),('budget','max_model_requests',True),('budget','wall_time_seconds',float('inf')),
 ('budget','max_total_tokens',-2),('runtime','pids_limit',2.2),('recording','objects',True),
 ('model','kind','fixture'),('model','api_key_env','KEY=secret'),('model','api_key','secret'),
 ('evaluation','outcome_policy','status_ok_is_success'),('recording','checkpoints','yes'),('budget','cost_limit',1)
])
def test_reject_invalid_config(config,section,key,value):
    config[section][key]=value
    with pytest.raises(ConfigurationError): validate(config)

def test_paths_relative_to_config(tmp_path):
    p=tmp_path/'folder/config.json';p.parent.mkdir();p.write_text(json.dumps({'benchmark':{'source_dir':'../src'}}))
    assert resolve(p)['benchmark']['source_dir']==str(tmp_path/'src')

def test_no_secrets_in_nested_parameters(config):
    config['model']['parameters']={'nested':{'authorization':'Bearer secret'}}
    with pytest.raises(ConfigurationError): validate(config)

def test_registry_duplicate_unknown():
    r=Registry('unit');r.register('a',dict)
    with pytest.raises(ConfigurationError):r.register('a',dict)
    with pytest.raises(ConfigurationError):r.create('b',{})
    assert r.create('a',{'x':1})=={'x':1}

@pytest.mark.parametrize('url,kind',[
 ('http://example.com/v1','api'),('http://127.0.0.1/v1','api'),
 ('https://username:password@example.com/v1','api'),('https://example.com/v1?api_key=x','api'),
 ('file:///tmp/model','local'),('https://example.com/#secret','api')
])
def test_endpoint_safety(config,url,kind):
    config['model'].update(base_url=url,kind=kind)
    with pytest.raises(ConfigurationError):GeminiBackend(config['model'])

def test_reserved_protocol_fields(config):
    config['model']['parameters']={'messages':[]}
    with pytest.raises(ConfigurationError):OpenAICompatibleBackend(config['model'])
