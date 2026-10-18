"""Prospective observation ordering with fixture ports; no Docker/model calls."""
import copy
import json
from types import SimpleNamespace

import pytest

from c3_sandbox.l1.core.contracts import EvaluationSpec, Task
from c3_sandbox.l1.core.orchestrator import run
from c3_sandbox.l1.runtime import docker
from c3_sandbox.l1.storage import digest, read_json
from .conftest import FixtureEvaluator, ScriptedGemini, TrustedFixtureRuntime, gemini_done


CID = 'a' * 64
IMAGE = 'sha256:' + 'b' * 64


def fake_docker(monkeypatch):
    calls = []
    inspection = {'Id': CID, 'Image': IMAGE, 'State': {'Status': 'created', 'ExitCode': 0}}

    def subprocess_run(args, **kwargs):
        calls.append(tuple(args[:2]))
        return SimpleNamespace(returncode=0, stdout=CID.encode(), stderr=b'')

    def docker_json(*args):
        calls.append(tuple(args))
        if args[:2] == ('image', 'inspect'):
            return [{'Id': IMAGE, 'Os': 'linux', 'Architecture': 'arm64'}]
        if args[0] == 'info':
            return {'OSType': 'linux', 'ServerVersion': 'fixture'}
        return [copy.deepcopy(inspection)]

    def process(args, *other):
        calls.append(tuple(args[:2]))
        return {'stdout': json.dumps({'has_app_git': False}).encode(), 'stderr': b'',
                'exit_code': 0, 'output_complete': True}

    monkeypatch.setattr(docker.subprocess, 'run', subprocess_run)
    monkeypatch.setattr(docker, 'docker_json', docker_json)
    monkeypatch.setattr(docker, 'bounded_process', process)
    monkeypatch.setattr(docker.shutil, 'which', lambda name: '/fake/docker')
    runtime = docker.DockerRuntime(dict(image='fixture', memory='1g', cpus=1, pids_limit=32,
                                       output_limit_bytes=4096))
    runtime.provenance = {'image_id': IMAGE}
    return runtime, calls, inspection


def test_container_observed_before_start_and_default_has_no_extra_inspect(monkeypatch):
    runtime, calls, inspection = fake_docker(monkeypatch)
    seen = []

    def observer(cid, actual):
        calls.append(('observer', cid))
        seen.append((cid, actual))
        return {'external_receipt': 'fixture'}

    runtime.set_execution_observer(observer)
    runtime.container([], ['true'], 1)
    assert seen == [(CID, inspection)]
    assert calls.index(('inspect', CID)) < calls.index(('observer', CID)) < calls.index(('docker', 'start'))
    assert calls.count(('inspect', CID)) == 2
    assert calls[-1] == ('docker', 'rm')

    runtime.set_execution_observer(None)
    calls.clear()
    runtime.container([], ['true'], 1)
    assert calls.count(('inspect', CID)) == 1
    assert calls.index(('docker', 'start')) < calls.index(('inspect', CID))


def test_observer_failure_prevents_start_and_still_removes_container(monkeypatch):
    runtime, calls, _ = fake_docker(monkeypatch)

    def rejected(*args):
        raise RuntimeError('external persistence failed')

    runtime.set_execution_observer(rejected)
    with pytest.raises(RuntimeError, match='external persistence failed'):
        runtime.container([], ['true'], 1)
    assert ('docker', 'start') not in calls
    assert calls[-1] == ('docker', 'rm')
    assert runtime.is_quiescent


def test_environment_probe_uses_same_observer(monkeypatch):
    runtime, calls, _ = fake_docker(monkeypatch)
    runtime.provenance = None
    seen = []
    runtime.set_execution_observer(lambda cid, actual: seen.append(cid) or {'stored': True})
    runtime.preflight()
    assert seen == [CID]
    assert calls.count(('inspect', CID)) == 2


class Observer:
    def __init__(self, output, fail=None):
        self.output = output
        self.fail = fail
        self.calls = []

    def validate(self, config):
        assert not self.output.exists()
        assert 'system_prompt' in config['agent']
        self.calls.append('validate')
        if self.fail == 'validate':
            raise RuntimeError('validation failed')

    def bind_workspace(self, workspace):
        assert workspace.is_dir()
        self.workspace = workspace
        self.calls.append('workspace')

    def bind_episode(self, episode_id, config_sha256):
        self.calls.append('bind')
        self.episode_id = episode_id
        self.config_sha256 = config_sha256
        if self.fail == 'bind':
            raise RuntimeError('binding failed')
        return {'episode_id': episode_id, 'config_sha256': config_sha256, 'external': 'fixture'}

    def observe_container(self, cid, inspection):
        assert self.calls[:3] == ['validate', 'workspace', 'bind']
        assert (self.output / 'runtime/prospective/anchor.json').exists()
        self.calls.append('inspect')
        # The external writer cannot mutate the sealed actual inspection.
        inspection['Id'] = 'mutated'
        if self.fail == 'inspect':
            raise RuntimeError('observation failed')
        return {'external': 'persisted fixture'}


class ObservedRuntime(TrustedFixtureRuntime):
    def set_execution_observer(self, callback):
        self.callback = callback

    def preflight(self):
        if hasattr(self, 'callback'):
            self.callback(CID, {'Id': CID, 'State': {'Status': 'created'}})
        return super().preflight()


def fixture_components(config, tmp_path, runtime=None):
    class Benchmark:
        def load(self, task_id):
            return Task('fixture', task_id, 'Finish.', (), {}, {},
                        EvaluationSpec('fixture', tmp_path, 'fixture.py', task_id))

    return {'benchmark': Benchmark(), 'runtime': runtime or ObservedRuntime(),
            'model': ScriptedGemini(config['model'], [gemini_done()]), 'evaluation': FixtureEvaluator()}


def events(output):
    return [json.loads(line) for line in (output / 'trajectory.jsonl').read_text().splitlines()]


def test_binding_precedes_components_preflight_and_model_and_is_sealed(config, tmp_path):
    output = tmp_path / 'observed'
    observer = Observer(output)
    components = fixture_components(config, tmp_path)
    run(config, output, injected=components, execution_observer=observer)
    assert observer.calls == ['validate', 'workspace', 'bind', 'inspect']
    resolved = read_json(output / 'raw/resolved_config.json')
    assert observer.config_sha256 == digest(resolved)
    container = read_json(output / f'runtime/prospective/containers/{CID}.json')
    assert container['inspection']['Id'] == CID
    assert container['receipt'] == {'external': 'persisted fixture'}
    history = events(output)
    kinds = [event['kind'] for event in history]
    assert kinds.index('provenance.bound') < kinds.index('runtime.container.inspected') < kinds.index('model.request')
    indexed = read_json(output / 'evidence_index.json')
    assert 'runtime/prospective/anchor.json' in json.dumps(indexed)
    assert 'prospective' not in read_json(output / 'manifest.json')


def test_failed_validation_creates_no_output_or_components(config, tmp_path, monkeypatch):
    output = tmp_path / 'rejected'
    observer = Observer(output, 'validate')
    monkeypatch.setattr('c3_sandbox.l1.core.orchestrator.components', lambda c: pytest.fail('created components'))
    with pytest.raises(RuntimeError, match='validation failed'):
        run(config, output, execution_observer=observer)
    assert not output.exists()


def test_binding_exists_before_component_factory(config, tmp_path, monkeypatch):
    output = tmp_path / 'factory'
    observer = Observer(output)
    components = fixture_components(config, tmp_path)

    def factory(resolved):
        assert observer.calls == ['validate', 'workspace', 'bind']
        assert read_json(output / 'runtime/prospective/anchor.json')['config_sha256'] == digest(resolved)
        return components

    monkeypatch.setattr('c3_sandbox.l1.core.orchestrator.components', factory)
    run(config, output, execution_observer=observer)
    assert observer.calls[-1] == 'inspect'


@pytest.mark.parametrize('failure', ['bind', 'inspect'])
def test_failed_binding_or_receipt_never_sends_model(config, tmp_path, failure):
    output = tmp_path / failure
    components = fixture_components(config, tmp_path)
    result = run(config, output, injected=components, execution_observer=Observer(output, failure))
    assert components['model'].bodies == []
    assert result['status']['execution_validity'] == 'invalid'
    assert 'model.request' not in [event['kind'] for event in events(output)]


def test_no_observer_preserves_default_path(config, tmp_path):
    output = tmp_path / 'default'
    runtime = TrustedFixtureRuntime()  # Deliberately has no observer setter.
    run(config, output, injected=fixture_components(config, tmp_path, runtime))
    assert not (output / 'runtime/prospective').exists()
    assert not any(event['kind'] in ('provenance.bound', 'runtime.container.inspected') for event in events(output))


def test_two_runs_keep_callbacks_and_evidence_separate(config, tmp_path):
    first, second = tmp_path / 'one', tmp_path / 'two'
    a, b = Observer(first), Observer(second)
    run(config, first, injected=fixture_components(config, tmp_path), execution_observer=a)
    run(config, second, injected=fixture_components(config, tmp_path), execution_observer=b)
    assert a.episode_id != b.episode_id
    assert a.workspace != b.workspace
    assert read_json(first / 'runtime/prospective/anchor.json')['episode_id'] == a.episode_id
    assert read_json(second / 'runtime/prospective/anchor.json')['episode_id'] == b.episode_id


def test_native_identity_mismatch_stops_before_probe_or_model(config, tmp_path, monkeypatch):
    cfg=copy.deepcopy(config)
    cfg['agent'].update(harness='pi',version='0.73.1',validation_policy='pi-native-0.73.1',
                        harness_revision='l1-pi-native-errors-context-v2')
    calls=[]
    class Native:
        def start(self):calls.append('native_started')
        def dependency_identity(self):return {'actual_dependency':'fixture_wrong_hash'}
        def close(self):calls.append('native_closed')
    class RejectingObserver(Observer):
        def validate_native_identity(self,identity):
            calls.append('identity_checked')
            assert identity=={'actual_dependency':'fixture_wrong_hash'}
            raise RuntimeError('native dependency differs from prospective freeze')
    output=tmp_path/'native_rejected'
    components=fixture_components(cfg,tmp_path)
    monkeypatch.setattr('c3_sandbox.l1.runtime.pi_native_validation.NativePiValidator',Native)
    monkeypatch.setattr(components['runtime'],'preflight',lambda:pytest.fail('probe before native identity check'))
    result=run(cfg,output,injected=components,execution_observer=RejectingObserver(output))
    assert calls==['native_started','identity_checked','native_closed']
    assert components['model'].bodies==[]
    assert result['status']['execution_validity']=='invalid'
    assert not (output/'runtime/prospective/containers').exists()
    assert 'model.request' not in [event['kind'] for event in events(output)]
