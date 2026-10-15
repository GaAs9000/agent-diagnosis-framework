"""The L1 native request identity assumed by the lean L2 v0.3 reader."""
from __future__ import annotations

import copy
from pathlib import Path
import pytest

from c3_sandbox.l1.core.config import DEFAULTS
from c3_sandbox.l1.core.contracts import EvaluationSpec, Result, Task
from c3_sandbox.l1.core.orchestrator import run
from c3_sandbox.l1.runtime.tools import TOOLS
from c3_sandbox.l1.storage import canonical, read_json
from .conftest import FixtureEvaluator, ScriptedGemini, TrustedFixtureRuntime


class _LocalBenchmark:
    """Minimum port fixture: no retained bundle, upstream data or verifier."""

    def load(self, task_id):
        return Task(
            'officeval', task_id, 'Write a fixture deliverable. 保留原始会话。', (),
            {'adapter': 'test_fixture'}, {},
            EvaluationSpec('fixture', Path('.'), 'unused.py', task_id),
        )


class _LocalRuntime(TrustedFixtureRuntime):
    def execute(self, call, workspace, timeout):
        self.executed.append(call)
        (workspace / 'submission' / 'fixture.txt').write_text('fixture output', encoding='utf-8')
        stdout = '工具结果 ' + 'x' * 100
        return Result(
            {'stdout': stdout, 'stderr': '', 'exit_code': 0},
            stdout.encode('utf-8'),
            provenance={'test_fixture': True, 'quiescent': True},
        )


class _WireGemini(ScriptedGemini):
    def __init__(self, config):
        super().__init__(config)
        self.wire_bodies = []

    def send(self, body, timeout):
        # Capture bytes at the actual transport boundary, not reconstructed JSON.
        self.wire_bodies.append(bytes(body))
        return super().send(body, timeout)


@pytest.mark.parametrize('checkpoint_id', ['cp000000', 'cp000001'])
@pytest.mark.parametrize('supports_images', [False, True])
def test_self_contained_first_resumed_native_request(tmp_path, checkpoint_id, supports_images):
    cfg = copy.deepcopy(DEFAULTS)
    cfg['model'].update(api_key_env=None, supports_images=supports_images)
    cfg['model']['parameters'] = {'temperature': 1.0, 'topP': 0.83, 'maxOutputTokens': 127}
    cfg['budget'].update(max_model_requests=3, max_tool_calls=3, wall_time_seconds=60)
    cfg['recording']['observation_limit_chars'] = 20

    def ports():
        return {'benchmark': _LocalBenchmark(), 'model': _WireGemini(cfg['model']),
                'runtime': _LocalRuntime(), 'evaluation': FixtureEvaluator()}

    parent_ports = ports()
    parent = tmp_path / 'parent'
    result = run(cfg, parent, injected=parent_ports)
    assert result['verification']['verified'], result['verification']

    checkpoint = read_json(parent / f'checkpoints/{checkpoint_id}.json')
    request_number = checkpoint['budget_used']['model_requests']
    # Construct the oracle directly from persisted checkpoint history and the
    # declared tools/config, independently of GeminiBackend.request().
    expected = copy.deepcopy(checkpoint['conversation'])
    expected['tools'] = [{'functionDeclarations': copy.deepcopy(TOOLS if supports_images else TOOLS[:1])}]
    expected['generationConfig'] = copy.deepcopy(cfg['model']['parameters'])
    expected_bytes = canonical(expected)
    relative = f'raw/model/r{request_number:06d}/request.json'
    assert (parent / relative).read_bytes() == expected_bytes
    assert parent_ports['model'].wire_bodies[request_number] == expected_bytes

    child_ports = ports()
    child = tmp_path / 'child'
    result = run(cfg, child, injected=child_ports, parent=parent, checkpoint_id=checkpoint_id)
    assert result['verification']['verified'], result['verification']
    assert child_ports['model'].wire_bodies[0] == expected_bytes
    assert (child / relative).read_bytes() == expected_bytes
    assert read_json(child / 'lineage/parent_checkpoint.json') == checkpoint

    if request_number:
        parts = [part for message in expected['contents'] for part in message['parts']]
        assert any(part.get('thoughtSignature') == 'opaque-signature-retain-exactly' for part in parts)
        assert any(part.get('functionCall', {}).get('id') == 'native-call-1' for part in parts)
        assert any(part.get('functionResponse', {}).get('id') == 'native-call-1' for part in parts)
        assert checkpoint['pending_observations']
        assert (child / 'artifacts/submission/fixture.txt').read_text() == 'fixture output'


@pytest.mark.parametrize('checkpoint_id', ['cp000000', 'cp000001'])
def test_native_request_bytes_equal_parent_after_cold_resume(tmp_path, config, components, checkpoint_id):
    cfg = copy.deepcopy(config)
    cfg['model']['parameters']['temperature'] = 1.0
    parent_components = copy.deepcopy(components)
    parent_components['model'] = ScriptedGemini(cfg['model'])
    parent = tmp_path / 'parent'
    parent_result = run(cfg, parent, injected=parent_components)
    assert parent_result['verification']['verified'], parent_result['verification']

    child_components = copy.deepcopy(components)
    child_components['model'] = ScriptedGemini(cfg['model'])
    child = tmp_path / 'child'
    child_result = run(cfg, child, injected=child_components, parent=parent, checkpoint_id=checkpoint_id)
    assert child_result['verification']['verified'], child_result['verification']

    checkpoint = read_json(parent / f'checkpoints/{checkpoint_id}.json')
    request_number = checkpoint['budget_used']['model_requests']
    relative = f'raw/model/r{request_number:06d}/request.json'
    assert (child / relative).read_bytes() == (parent / relative).read_bytes()
    if request_number:
        native_history = checkpoint['conversation']['contents']
        assert any('thoughtSignature' in part for message in native_history for part in message.get('parts', []))
