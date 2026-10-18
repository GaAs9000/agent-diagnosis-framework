"""CPU transport fixtures exercise consumption; native validation uses real pinned JS."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from c3_sandbox.l1.core.contracts import EvidenceError, EvaluationSpec, Task
from c3_sandbox.l1.core.orchestrator import run
from c3_sandbox.l1.models.native import OpenAICompatibleBackend
from c3_sandbox.l1.runtime.pi_native_validation import NativePiValidator
from c3_sandbox.l1.storage import read_json
from .conftest import FixtureEvaluator, ScriptedGemini, gemini_call, gemini_done
from .test_pi_harness import PiFixtureRuntime


def events(root):
    return [json.loads(line) for line in (root/'trajectory.jsonl').read_text().splitlines()]


@pytest.fixture
def native_bridge(monkeypatch):
    directory = Path(os.environ.get('C3_PI_NATIVE_TEST_DIR', Path(__file__).parents[2]/'src/c3_sandbox/l1/runtime/pi_native'))
    if not (directory/'node_modules').is_dir():
        pytest.skip('Pinned native Node dependencies are required for integration test')
    import c3_sandbox.l1.runtime.pi_native_validation as module
    monkeypatch.setattr(module, 'NativePiValidator', lambda: NativePiValidator(bridge_dir=directory))


class EmptyBenchmark:
    def load(self, task_id):
        return Task('fixture', task_id, 'Write and edit a file.', (), {}, {},
                    EvaluationSpec('fixture', Path('/unused'), 'unused', task_id))


def pi_config(config, native=False):
    result = copy.deepcopy(config)
    result['agent'] = {'name':'pi-coding-agent', 'version':'0.73.1', 'harness':'pi'}
    result['model']['timeout_seconds'] = 600 if native else 300
    result['budget'].update(max_model_requests=5, max_tool_calls=8)
    if native:
        result['agent'].update(validation_policy='pi-native-0.73.1', harness_revision='l1-pi-native-errors-v1')
    return result


def fake_live_components(monkeypatch, model, runtime):
    # The production gate remains strict. Tests install isolated ports and exercise
    # the live lifecycle so that the saved-response gate itself is covered.
    import c3_sandbox.l1.core.orchestrator as orchestrator
    ports={'model':model,'runtime':runtime,'evaluation':FixtureEvaluator(),'benchmark':EmptyBenchmark()}
    monkeypatch.setattr(orchestrator, 'components', lambda _: ports)
    # Cold restoration constructs model/runtime/evaluation through their registry
    # ports rather than the fresh-episode component factory.
    from c3_sandbox.l1.models import registry as models
    from c3_sandbox.l1.runtime import registry as runtimes
    from c3_sandbox.l1.evaluation import registry as evaluators

    class FixtureRegistry:
        def __init__(self, value): self.value=value
        def load_installed(self): pass
        def create(self, name, settings): return self.value

    for module,key in ((models,'model'),(runtimes,'runtime'),(evaluators,'evaluation')):
        monkeypatch.setattr(module,'registry',lambda key=key:FixtureRegistry(ports[key]))


def make_parent(tmp_path, config, monkeypatch):
    cfg = pi_config(config)
    model = ScriptedGemini(cfg['model'], [
        gemini_call('write', {'path':'note.txt', 'content':'original'}, native_id='write-original'),
        gemini_call('edit', {'path':'note.txt', 'edits':'bad'}, native_id='original-error-id'),
    ])
    fake_live_components(monkeypatch, model, PiFixtureRuntime())
    parent = tmp_path/'parent'
    result = run(cfg, parent)
    assert result['status']['termination'] == 'protocol_error'
    assert result['verification']['verified'], result
    assert len(model.bodies) == 2
    return parent


def test_saved_response_consumes_without_resending_and_inherits_budget(tmp_path, config, monkeypatch, native_bridge):
    parent = make_parent(tmp_path, config, monkeypatch)
    original_index = (parent/'evidence_index.json').read_bytes()
    cfg = pi_config(config, native=True)
    model = ScriptedGemini(cfg['model'], [gemini_done()])
    fake_live_components(monkeypatch, model, PiFixtureRuntime())
    child = tmp_path/'child'
    result = run(cfg, child, parent=parent, checkpoint_id='cp000001',
                 resume_response_id='r000001', allow_pi_native_upgrade=True)
    assert result['status']['termination'] == 'assistant_finished', result
    assert result['verification']['verified'], result
    assert len(model.bodies) == 1  # only the next request went to the transport
    response = model.bodies[0]['contents'][-1]['parts'][0]['functionResponse']
    assert response['id'] == 'original-error-id'
    assert set(response['response']) == {'error'}
    manifest = read_json(child/'manifest.json')
    assert manifest['usage']['known_total_tokens_including_parent'] == 31
    assert manifest['usage']['model_requests_including_parent'] == 3
    assert manifest['usage']['tool_calls_including_parent'] == 2
    assert manifest['usage']['new_http_attempts'] == 1
    assert manifest['usage']['saved_response_consumptions'] == 1
    requests = [e for e in events(child) if e['kind'] == 'model.request']
    assert requests[0]['data']['request_id'].endswith(':r000001')
    assert requests[0]['data']['source'] == 'saved_response'
    assert requests[1]['data']['request_id'].endswith(':r000002')
    assert not any(e['kind'] == 'observation.delivered' and e['data']['native_id'] == 'write-original' for e in events(child))
    assert (parent/'evidence_index.json').read_bytes() == original_index
    for original, copied in [('evidence_index.json','source_evidence_index.json'), ('manifest.json','source_manifest.json'), ('trajectory.jsonl','source_trajectory.jsonl')]:
        assert (parent/original).read_bytes() == (child/'lineage'/copied).read_bytes()


@pytest.mark.parametrize('damage', ['missing', 'tampered'])
def test_invalid_parent_response_fails_before_send(tmp_path, config, monkeypatch, native_bridge, damage):
    parent = make_parent(tmp_path, config, monkeypatch)
    target = parent/'raw/model/r000001/response.bin'
    if damage == 'missing':
        target.unlink()
    else:
        target.write_bytes(target.read_bytes()+b' ')
    cfg = pi_config(config, native=True)
    model = ScriptedGemini(cfg['model'], [gemini_done()])
    fake_live_components(monkeypatch, model, PiFixtureRuntime())
    with pytest.raises(EvidenceError):
        run(cfg, tmp_path/'child', parent=parent, checkpoint_id='cp000001',
            resume_response_id='r000001', allow_pi_native_upgrade=True)
    assert model.bodies == []


def test_native_invalid_call_does_not_block_valid_call_in_same_batch(tmp_path, config, monkeypatch, native_bridge):
    cfg = pi_config(config, native=True)
    malformed = gemini_call('edit', {'path':'note.txt', 'edits':'bad'}, native_id='bad-edit')
    good = gemini_call('write', {'path':'note.txt', 'content':'saved'}, native_id='good-write')
    malformed['candidates'][0]['content']['parts'] += good['candidates'][0]['content']['parts']
    model = ScriptedGemini(cfg['model'], [malformed, gemini_done()])
    runtime = PiFixtureRuntime()
    fake_live_components(monkeypatch, model, runtime)
    result = run(cfg, tmp_path/'episode')
    assert result['status']['termination'] == 'assistant_finished'
    assert result['verification']['verified'], result
    replies = model.bodies[1]['contents'][-1]['parts']
    assert set(replies[0]['functionResponse']['response']) == {'error'}
    assert replies[1]['functionResponse']['response']['exit_code'] == 0
    assert read_json(tmp_path/'episode/manifest.json')['usage']['tool_calls_including_parent'] == 2
    assert len([e for e in events(tmp_path/'episode') if e['kind']=='tool.started']) == 1


def test_native_valid_json_nonobject_is_validation_input(config):
    model = OpenAICompatibleBackend(config['model'])
    body = json.dumps({'choices':[{'message':{'role':'assistant','tool_calls':[{
        'id':'bad-object','type':'function','function':{'name':'edit','arguments':'null'}}]},
        'finish_reason':'tool_calls'}]}).encode()
    from c3_sandbox.l1.core.contracts import ProtocolError
    with pytest.raises(ProtocolError):
        model.parse(body, 'r0')
    model.native_pi_validation = True
    assert model.parse(body, 'r0').calls[0].arguments is None
    body = body.replace(b'null', b'{bad')
    with pytest.raises(ProtocolError):
        model.parse(body, 'r0')


def test_checkpoint_zero_exact_native_upgrade_sends_fresh_origin_request(tmp_path, config, monkeypatch, native_bridge):
    parent = make_parent(tmp_path, config, monkeypatch)
    cfg = pi_config(config, native=True)
    model = ScriptedGemini(cfg['model'], [gemini_done()])
    fake_live_components(monkeypatch, model, PiFixtureRuntime())
    child = tmp_path/'child-zero'
    result = run(cfg, child, parent=parent, checkpoint_id='cp000000', allow_pi_native_upgrade=True)
    assert result['status']['termination'] == 'assistant_finished', result
    assert result['verification']['verified'], result
    assert len(model.bodies) == 1
    assert model.bodies[0] == read_json(parent/'raw/model/r000000/request.json')
    manifest = read_json(child/'manifest.json')
    assert manifest['usage']['saved_response_consumptions'] == 0
    assert manifest['usage']['new_http_attempts'] == 1
    assert manifest['usage']['model_requests_including_parent'] == 1
    assert manifest['usage']['known_total_tokens_including_parent'] == 7
    assert manifest['lineage']['config_upgrade']['mode'] == 'initial_checkpoint'
    assert not any(e['kind'] == 'response.consumed' for e in events(child))


@pytest.mark.parametrize('change', ['nonzero_checkpoint', 'unrelated_config'])
def test_native_upgrade_rejects_other_checkpoint_or_config_delta(tmp_path, config, monkeypatch, change):
    from c3_sandbox.l1.core.contracts import ConfigurationError
    parent = make_parent(tmp_path, config, monkeypatch)
    cfg = pi_config(config, native=True)
    cp = 'cp000001' if change == 'nonzero_checkpoint' else 'cp000000'
    if change == 'unrelated_config':
        cfg['budget']['max_model_requests'] += 1
    model = ScriptedGemini(cfg['model'], [gemini_done()])
    fake_live_components(monkeypatch, model, PiFixtureRuntime())
    with pytest.raises(ConfigurationError):
        run(cfg, tmp_path/'rejected', parent=parent, checkpoint_id=cp, allow_pi_native_upgrade=True)
    assert model.bodies == []


@pytest.mark.parametrize('validator_raises', [False, True])
def test_native_validation_deadline_skips_pending_batch_without_execution(tmp_path, config, monkeypatch, validator_raises):
    import c3_sandbox.l1.core.orchestrator as orchestrator
    import c3_sandbox.l1.runtime.pi_native_validation as validation_module
    clock = [100.0]
    monkeypatch.setattr(orchestrator.time, 'monotonic', lambda: clock[0])
    observed_deadlines = []

    class SlowValidator:
        def start(self): pass
        def close(self): pass
        def dependency_identity(self): return {'test':'slow-validator'}
        def validate(self, name, arguments, *, call_id, deadline_monotonic):
            observed_deadlines.append(deadline_monotonic)
            clock[0] = deadline_monotonic + 1
            if validator_raises:
                raise TimeoutError('native bridge reached deadline')
            return {'ok':True,'raw_arguments':arguments,'validated_arguments':arguments,
                    'native_result':None,'provenance':{'test':'slow-validator'}}

    monkeypatch.setattr(validation_module, 'NativePiValidator', SlowValidator)
    cfg = pi_config(config, native=True)
    first = gemini_call('write', {'path':'should-not-exist.txt','content':'mutated'}, native_id='late-first')
    second = gemini_call('write', {'path':'also-absent.txt','content':'mutated'}, native_id='late-second')
    first['candidates'][0]['content']['parts'] += second['candidates'][0]['content']['parts']
    model = ScriptedGemini(cfg['model'], [first])
    executed = []

    class NoLateRuntime(PiFixtureRuntime):
        def execute(self, call, workspace, timeout):
            executed.append(call)
            return super().execute(call, workspace, timeout)

    fake_live_components(monkeypatch, model, NoLateRuntime())
    output = tmp_path/'deadline-episode'
    result = run(cfg, output)
    assert result['status']['termination'] == 'wall_time_budget', result
    assert result['verification']['verified'], result
    assert observed_deadlines == [100.0 + cfg['budget']['wall_time_seconds']]
    assert executed == []
    recorded = events(output)
    assert not any(e['kind']=='tool.started' for e in recorded)
    skipped = [e for e in recorded if e['kind']=='tool.skipped']
    assert len(skipped) == 2
    assert all(e['data']['reason']=='wall_time_budget' for e in skipped)
    final = [e for e in recorded if e['kind']=='state.snapshot'][-1]
    snapshot = read_json(output/final['data']['snapshot']['path'])
    assert snapshot['files'] == []
    assert read_json(output/'manifest.json')['usage']['tool_calls_including_parent'] == 1
