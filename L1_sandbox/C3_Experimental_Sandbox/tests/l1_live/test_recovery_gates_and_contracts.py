from __future__ import annotations
import copy
import json
from pathlib import Path
import pytest
from c3_sandbox.l1.core.orchestrator import run, build_execution_contract_prompt
from c3_sandbox.l1.core.contracts import EvidenceError, ConfigurationError
from c3_sandbox.l1.evidence.live import verify, seal, scan
from c3_sandbox.l1.storage import read_json, canonical
from .conftest import ScriptedGemini, TrustedFixtureRuntime, FixtureEvaluator, gemini_call, gemini_done
import importlib.util
from .conftest import ROOT


def reseal(root: Path):
    idx_file = root / 'evidence_index.json'
    if idx_file.exists():
        idx_file.unlink()
    events = scan(root / 'trajectory.jsonl')['events']
    seal(root, events)


@pytest.fixture
def accept_script():
    spec = importlib.util.spec_from_file_location('local_acceptance', ROOT / 'scripts/l1/accept_local.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resume_succeeds_when_parent_evaluation_errored(tmp_path, config, components, accept_script):
    """Verifies that when parent evaluation fails (e.g. missing deliverable / grader exception),

    checkpoint recovery remains 100% eligible and functional (Gemini 3.8 Flash & Llama 8B scenario).
    """
    # Parent finishes with evaluation failure (e.g. missing deliverable / grader exception)
    components['evaluation'] = FixtureEvaluator(fail=True)
    parent = tmp_path / 'parent_eval_failed'
    r_parent = run(config, parent, injected=components)
    assert r_parent['verification']['verified']
    assert r_parent['status']['evaluation_validity'] == 'invalid'

    # 1. Verify qualification decoupling: evaluation error does not disqualify engineering gate or resume
    pkg = tmp_path / 'pkg_eval_error'
    pkg.mkdir()
    (pkg / 'checkpoints').mkdir()
    manifest = {
        'schema_version': '0.2.0',
        'episode_id': 'ep_parent_eval_err',
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
    (pkg / 'official_grade.json').write_text(json.dumps({
        'status': 'error',
        'error': 'FileNotFoundError: /workspace/submission/部门协调会记录 (1).docx'
    }))
    (pkg / 'checkpoints/index.json').write_text(json.dumps({
        'checkpoints': [
            {'checkpoint_id': 'cp000000', 'eligible': True},
            {'checkpoint_id': 'cp000001', 'eligible': True, 'path': 'checkpoints/cp000001.json'},
        ]
    }))
    r_mock = {'verification': {'verified': True, 'index_sha256': 'mock_sha256'}}

    q = accept_script.qualify(r_mock, pkg)
    assert q['engineering_gate_passed'] is True
    assert q['official_evaluation']['evaluation_validity'] == 'error'
    assert q['official_evaluation']['status'] == 'error'
    assert q['checkpoint_eligibility']['can_resume'] is True
    assert q['checkpoint_eligibility']['has_eligible_post_tool_checkpoint'] is True

    # 2. Execute real child cold resume from the genuine parent with failed evaluation
    cp = 'cp000001'
    child = tmp_path / 'child_resumed'
    model = ScriptedGemini(config['model'], [gemini_call(native_id='resume-call'), gemini_done()])
    components['model'] = model
    components['evaluation'] = FixtureEvaluator(outcome='partial')

    r_child = run(config, child, injected=components, parent=parent, checkpoint_id=cp)
    assert r_child['verification']['verified']
    assert r_child['status']['termination'] == 'assistant_finished'
    assert r_child['status']['evaluation_validity'] == 'valid'
    assert r_child['status']['task_outcome'] == 'partial'

    # Check child lineage and parent immutability
    child_manifest = read_json(child / 'manifest.json')
    assert child_manifest['lineage']['parent_checkpoint_id'] == cp
    assert child_manifest['counts']['model_requests'] == 2
    assert verify(parent)['verified'] is True


def test_resume_fails_closed_when_checkpoint_file_missing(tmp_path, config, components):
    """Verifies fail-closed behavior when requested checkpoint file does not exist."""
    parent = tmp_path / 'parent'
    run(config, parent, injected=components)

    child = tmp_path / 'child'
    components['model'] = ScriptedGemini(config['model'], [gemini_done()])

    # 1. Non-existent checkpoint ID
    with pytest.raises(EvidenceError, match='Checkpoint cp999999 not found'):
        run(config, child, injected=components, parent=parent, checkpoint_id='cp999999')

    # 2. Corrupted parent where checkpoint was deleted
    (parent / 'checkpoints/cp000001.json').unlink()
    child2 = tmp_path / 'child2'
    with pytest.raises(EvidenceError, match='Parent evidence does not verify|not found'):
        run(config, child2, injected=components, parent=parent, checkpoint_id='cp000001')


def test_resume_fails_closed_when_snapshot_hash_tampered(tmp_path, config, components):
    """Verifies fail-closed behavior when snapshot blob or metadata is tampered with."""
    parent = tmp_path / 'parent'
    run(config, parent, injected=components)

    # Tamper with the snapshot JSON content
    snap_path = parent / 'states/snapshots/s000001.json'
    snap_data = read_json(snap_path)
    snap_data['content_sha256'] = '0' * 64
    snap_path.write_bytes(canonical(snap_data) + b"\n")

    child = tmp_path / 'child'
    components['model'] = ScriptedGemini(config['model'], [gemini_done()])

    # Parent verification fails, so resume must be strictly rejected
    with pytest.raises(EvidenceError, match='Parent evidence does not verify'):
        run(config, child, injected=components, parent=parent, checkpoint_id='cp000001')


def test_resume_fails_closed_when_checkpoint_not_eligible(tmp_path, config, components):
    """Verifies fail-closed rejection when checkpoint is marked ineligible (e.g. request budget exhausted)."""
    # Force budget exhaustion on request 1 so the subsequent checkpoint is naturally ineligible
    config['budget']['max_model_requests'] = 1
    parent = tmp_path / 'parent'
    run(config, parent, injected=components)

    # Check that cp000001 has eligible == False due to budget exhaustion
    cp1_data = read_json(parent / 'checkpoints/cp000001.json')
    assert cp1_data['qualification']['eligible'] is False
    assert 'request_budget_exhausted' in cp1_data['qualification']['reason_codes']

    # Attempting to resume from cp000001 must fail-closed with EvidenceError
    child = tmp_path / 'child'
    components['model'] = ScriptedGemini(config['model'], [gemini_done()])

    with pytest.raises(EvidenceError, match='Checkpoint is not qualified'):
        run(config, child, injected=components, parent=parent, checkpoint_id='cp000001')


def test_execution_contract_budget_prompt_consistency(tmp_path, config, components):
    """Verifies that budget constraints are enforced as an Execution Contract:

    dynamically assembled from config['budget'], consistently recorded in resolved_config,
    and accurately dispatched to the model without fragile regex substitution.
    """
    config['budget']['max_model_requests'] = 27
    config['budget']['max_tool_calls'] = 42
    config['budget']['wall_time_seconds'] = 900
    config['agent']['system_prompt'] = "Custom user prompt: You are a professional document editing assistant."

    out = tmp_path / 'contract_test'
    run(config, out, injected=components)

    # Verify resolved config recorded on disk
    resolved = read_json(out / 'raw/resolved_config.json')
    prompt = resolved['agent']['system_prompt']

    assert "Custom user prompt: You are a professional document editing assistant." in prompt
    assert "[SANDBOX EXECUTION CONTRACT & RUNTIME CONSTRAINTS]" in prompt
    assert "- Model Request Budget: At most 27 model requests." in prompt
    assert "- Tool Execution Budget: At most 42 tool invocations." in prompt
    assert "- Wall Clock Limit: 900 seconds timeout." in prompt

    # Verify the actual request sent to the model matches resolved config identically
    model_req = read_json(out / 'raw/model/r000000/request.json')
    if 'systemInstruction' in model_req:
        actual_prompt = model_req['systemInstruction']['parts'][0]['text']
    else:
        actual_prompt = model_req['messages'][0]['content']
    assert actual_prompt == prompt
