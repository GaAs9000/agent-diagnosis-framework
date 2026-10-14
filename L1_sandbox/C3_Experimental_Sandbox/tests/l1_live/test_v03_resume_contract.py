"""The L1 native request identity assumed by the lean L2 v0.3 reader."""
from __future__ import annotations

import copy
import pytest

from c3_sandbox.l1.core.orchestrator import run
from c3_sandbox.l1.storage import read_json
from .conftest import ScriptedGemini


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
