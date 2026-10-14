"""Contract tests only. No Docker daemon or model calls."""
import importlib.util
import json
from pathlib import Path

import pytest

from c3_sandbox.adapters.inspect_smoke import DOCKER_DIR, compose_config, parse_grade, verify_container, make_task

spec = importlib.util.spec_from_file_location('synthetic_grader', DOCKER_DIR / 'grader.py')
grader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grader)


def test_strict_synthetic_grading():
    result = grader.grade_bytes(b'{"schema_version":1,"value":7}', 7)
    assert parse_grade(json.dumps(result))["passed"] is True
    wrong = grader.grade_bytes(b'{"schema_version":1,"value":8}', 7)
    assert parse_grade(json.dumps(wrong))["passed"] is False


@pytest.mark.parametrize('raw', [b'', b'null', b'[]', b'{', b'{"score":1}',
    b'{"schema_version":1,"value":true}', b'{"schema_version":true,"value":7}',
    b'{"schema_version":1,"value":7,"passed":true}'])
def test_artifact_cannot_supply_a_score(raw):
    with pytest.raises(ValueError):
        grader.grade_bytes(raw, 7)


@pytest.mark.parametrize('raw', ['', '{}', 'null', '{"passed":true}',
    '{"schema_version":1,"is_synthetic":true,"passed":1,"checks":{}}'])
def test_invalid_result_is_error(raw):
    with pytest.raises(ValueError):
        parse_grade(raw)


def test_inconsistent_result_is_error():
    result = grader.grade_bytes(b'{"schema_version":1,"value":7}', 7)
    result['checks']['value_matches'] = False
    with pytest.raises(ValueError):
        parse_grade(json.dumps(result))


def test_compose_is_separate_hardened_and_unmounted():
    services = compose_config('unit-run')['services']
    assert set(services) == {'default', 'grader'}
    assert services['default']['image'] != services['grader']['image']
    for cfg in services.values():
        assert cfg['platform'] == 'linux/arm64'
        assert cfg['user'] == '1000:1000'
        assert cfg['read_only'] is True
        assert cfg['network_mode'] == 'none'
        assert cfg['cap_drop'] == ['ALL']
        assert cfg['security_opt'] == ['no-new-privileges:true']
        assert (cfg['cpus'],cfg['mem_limit'],cfg['pids_limit']) == (2,'4g',128)
        assert {s.split(':')[0] for s in cfg['tmpfs']} == {'/workspace','/tmp'}
        assert 'volumes' not in cfg and 'build' not in cfg
        assert cfg['labels'] == {'c3.synthetic_run':'unit-run'}
        assert cfg['environment'] == {'PYTHONDONTWRITEBYTECODE':'1'}
    assert 'grader.py' not in (DOCKER_DIR/'Dockerfile.agent').read_text()
    assert 'ARG PYTHON_BASE=python:3.12-slim' in (DOCKER_DIR/'Dockerfile.agent').read_text()


def runtime_info():
    return {'Config':{'User':'1000:1000','WorkingDir':'/workspace'},'Mounts':[],
        'HostConfig':{'ReadonlyRootfs':True,'NetworkMode':'none','Privileged':False,
        'CapAdd':None,'CapDrop':['ALL'],'SecurityOpt':['no-new-privileges:true'],
        'NanoCpus':2_000_000_000,'Memory':4*1024**3,'PidsLimit':128,'Binds':None,
        'Tmpfs':{'/workspace':'rw','/tmp':'rw'}}}


def test_runtime_identity_policy():
    verify_container(runtime_info())


@pytest.mark.parametrize('change', [
    ('User','0'),('NetworkMode','bridge'),('ReadonlyRootfs',False),
    ('Privileged',True),('Memory',0),('Binds',['/host:/workspace'])])
def test_runtime_identity_rejects_weaker_settings(change):
    info = runtime_info()
    key,value=change
    info['Config' if key=='User' else 'HostConfig'][key]=value
    with pytest.raises(ValueError):
        verify_container(info)


def test_optional_inspect_task_public_api(tmp_path):
    pytest.importorskip('inspect_ai', reason='optional inspect-ai integration dependency absent')
    task = make_task(tmp_path/'compose.yaml', tmp_path, 'unit-run', {})
    assert task.name == 'c3_synthetic_infrastructure_smoke'
    assert len(task.dataset) == 1
    assert task.metadata['is_synthetic'] is True
    assert task.sandbox.type == 'docker'


class FakeTransport:
    def __init__(self, stdout='', success=True):
        from types import SimpleNamespace
        self.result = SimpleNamespace(stdout=stdout, stderr='transport diagnostic', success=success, returncode=0 if success else 2)
        self.calls = []

    async def exec(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        return self.result


def test_binary_capture_without_docker_cp():
    import asyncio
    import base64
    from c3_sandbox.adapters.inspect_smoke import freeze_artifact
    raw = b'\x00\xff\r\n{"value":7}'
    transport = FakeTransport(base64.b64encode(raw).decode()+'\n')
    assert asyncio.run(freeze_artifact(transport)) == raw
    cmd, kwargs = transport.calls[0]
    assert cmd[:3] == ['/usr/local/bin/python','-I','-c']
    assert kwargs['timeout_retry'] is False


@pytest.mark.parametrize('encoded', ['', 'not base64', 'YQ== junk', 'A'*5468])
def test_capture_rejects_invalid_transport(encoded):
    import asyncio
    from c3_sandbox.adapters.inspect_smoke import freeze_artifact
    with pytest.raises(ValueError):
        asyncio.run(freeze_artifact(FakeTransport(encoded)))


def test_binary_delivery_checks_content_identity():
    import asyncio
    from c3_sandbox.adapters.inspect_smoke import deliver_artifact, sha256
    raw=b'\x00\xff\r\n{"value":7}'
    transport=FakeTransport(sha256(raw)+'\n')
    asyncio.run(deliver_artifact(transport,raw))
    cmd,kwargs=transport.calls[0]
    assert kwargs['input'] is raw
    assert kwargs['timeout_retry'] is False
    with pytest.raises(ValueError):
        asyncio.run(deliver_artifact(FakeTransport('wrong hash'),raw))


def test_transport_failure_preserves_execution_diagnostic():
    import asyncio
    from c3_sandbox.adapters.inspect_smoke import freeze_artifact
    with pytest.raises(RuntimeError,match='transport diagnostic'):
        asyncio.run(freeze_artifact(FakeTransport(success=False)))
