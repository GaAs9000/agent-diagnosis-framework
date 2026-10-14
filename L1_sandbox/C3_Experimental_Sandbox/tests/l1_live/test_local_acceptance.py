from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import pytest
from .conftest import ROOT

@pytest.fixture
def script():
    spec=importlib.util.spec_from_file_location('local_acceptance',ROOT/'scripts/l1/accept_local.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_preflight_does_not_execute_model(script,tmp_path,config,monkeypatch):
    monkeypatch.setattr(script,'resolve',lambda _:config)
    monkeypatch.setattr(script,'doctor',lambda _: {'ready':True,'model_endpoint_contacted':False})
    def forbidden(*a,**k):raise AssertionError('No inference authorized')
    monkeypatch.setattr(script,'run',forbidden)
    assert script.main(['--config','x','--output-root',str(tmp_path/'out')])==0
    r=json.loads((tmp_path/'out/acceptance.json').read_text())
    assert not r['qualification_passed'] and r['reason']=='preflight_only_no_model_execution'


def test_failed_doctor_keeps_receipt(script,tmp_path,config,monkeypatch):
    monkeypatch.setattr(script,'resolve',lambda _:config)
    monkeypatch.setattr(script,'doctor',lambda _: {'ready':False})
    assert script.main(['--config','x','--output-root',str(tmp_path/'out'),'--execute-model'])==2
    assert json.loads((tmp_path/'out/acceptance.json').read_text())['reason']=='prerequisites_not_ready'


def test_resume_requires_explicit_execution(script,tmp_path):
    with pytest.raises(SystemExit) as exc:
        script.main(['--config','x','--output-root',str(tmp_path/'out'),'--resume'])
    assert exc.value.code==2 and not (tmp_path/'out').exists()


def test_qualify_decouples_evaluation_from_engineering_and_resume(script,tmp_path):
    pkg = tmp_path / 'pkg'
    pkg.mkdir()
    (pkg / 'checkpoints').mkdir()
    # Mock manifest where evaluation failed because no submission was produced, but execution is valid
    manifest = {
        'schema_version': '0.2.0',
        'episode_id': 'ep_test123',
        'recording_mode': 'live',
        'task': {'provenance': {'adapter': 'officeval-native'}},
        'status': {
            'execution_validity': 'valid',
            'evaluation_validity': 'error',
            'evidence_completeness': 'complete',
            'task_outcome': 'unknown',
            'termination': 'request_budget',
        },
        'counts': {'model_requests': 35, 'tool_results': 35},
    }
    (pkg / 'manifest.json').write_text(json.dumps(manifest))
    (pkg / 'official_grade.json').write_text(json.dumps({'status': 'error', 'error': 'FileNotFoundError: submission'}))
    (pkg / 'checkpoints/index.json').write_text(json.dumps({
        'checkpoints': [
            {'checkpoint_id': 'cp000000', 'eligible': True},
            {'checkpoint_id': 'cp000001', 'eligible': True, 'path': 'checkpoints/cp000001.json'},
        ]
    }))
    result = {'verification': {'verified': True, 'index_sha256': 'abc123def'}}

    q = script.qualify(result, pkg)
    assert q['engineering_gate_passed'] is True
    assert q['official_evaluation']['evaluation_validity'] == 'error'
    assert q['official_evaluation']['status'] == 'error'
    assert q['checkpoint_eligibility']['can_resume'] is True
    assert q['checkpoint_eligibility']['eligible_checkpoint_count'] == 1


def test_qualify_extracts_official_evaluation_from_official_raw(script, tmp_path):
    # Regression test for acceptance.json field projection bug
    # Verifies both scoring success and missing artifact error are properly extracted from official_raw
    pkg = tmp_path / 'pkg_eval'
    pkg.mkdir()
    (pkg / 'checkpoints').mkdir()
    manifest = {
        'schema_version': '0.2.0',
        'episode_id': 'ep_eval_test',
        'recording_mode': 'live',
        'task': {'provenance': {'adapter': 'officeval-native'}},
        'status': {
            'execution_validity': 'valid',
            'evaluation_validity': 'valid',
            'evidence_completeness': 'complete',
            'task_outcome': 'unknown',
            'termination': 'assistant_finished',
        },
        'counts': {'model_requests': 10, 'tool_results': 10},
    }
    (pkg / 'manifest.json').write_text(json.dumps(manifest))
    (pkg / 'checkpoints/index.json').write_text(json.dumps({'checkpoints': []}))
    result = {'verification': {'verified': True, 'index_sha256': 'sha_dummy'}}

    # 1. Successful scoring wrapped in official_raw
    grade_ok = {
        'official_raw': {
            'dim1_pass': True,
            'total_score': 10,
            'max_score': 25,
            'status': 'ok',
            'error': None,
        },
        'projection': {'evaluation_validity': 'valid'},
    }
    (pkg / 'official_grade.json').write_text(json.dumps(grade_ok))
    q_ok = script.qualify(result, pkg)
    assert q_ok['official_evaluation']['status'] == 'ok'
    assert q_ok['official_evaluation']['dim1_pass'] is True
    assert q_ok['official_evaluation']['total_score'] == 10
    assert q_ok['official_evaluation']['max_score'] == 25
    assert q_ok['official_evaluation']['error'] is None

    # 2. Grader error wrapped in official_raw
    grade_err = {
        'official_raw': {
            'dim1_pass': False,
            'total_score': 0,
            'max_score': 0,
            'status': 'error',
            'error': '目录中未找到 .docx 文件：/submission',
        },
        'projection': {'evaluation_validity': 'invalid'},
    }
    (pkg / 'official_grade.json').write_text(json.dumps(grade_err))
    manifest['status']['evaluation_validity'] = 'invalid'
    (pkg / 'manifest.json').write_text(json.dumps(manifest))
    q_err = script.qualify(result, pkg)
    assert q_err['official_evaluation']['status'] == 'error'
    assert q_err['official_evaluation']['dim1_pass'] is False
    assert q_err['official_evaluation']['total_score'] == 0
    assert q_err['official_evaluation']['max_score'] == 0
    assert q_err['official_evaluation']['error'] == '目录中未找到 .docx 文件：/submission'

