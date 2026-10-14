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


def test_p0_1_system_prompt_preserves_guidance_and_legacy_idempotence():
    """Verifies P0-1: guidance after budget sentence is retained, function is idempotent,
    legacy truncated prompts stay identical on resume, and unformatted placeholders fail closed.
    """
    from c3_sandbox.l1.core.config import resolve
    cfg = resolve(ROOT / 'configs/l1/officeval.gemini-35lite.t1.json')
    p1 = build_execution_contract_prompt(cfg)
    assert 'Do not make dozens of exploratory shell calls.' in p1
    assert 'conclude your work promptly.' in p1
    assert 'You operate under a strict execution budget' not in p1

    # Idempotence
    cfg2 = copy.deepcopy(cfg)
    cfg2['agent']['system_prompt'] = p1
    assert build_execution_contract_prompt(cfg2) == p1

    # Legacy truncated prompt remains identical on resume
    legacy_cfg = copy.deepcopy(cfg)
    legacy_prompt = (
        "You are the tested agent in an execution-and-evidence sandbox. Complete the user's task.\n\n"
        "[SANDBOX EXECUTION CONTRACT & RUNTIME CONSTRAINTS]\n"
        "- Model Request Budget: At most 35 model requests.\n"
        "- Tool Execution Budget: At most 50 tool invocations.\n"
        "- Wall Clock Limit: 3600 seconds timeout.\n"
        "- Workspace Boundary: /workspace is persistent; create intermediate scripts and output files here.\n"
        "- Final Submission: Place ONLY final deliverables in /workspace/submission with required filenames.\n"
        "- Stateless Tools: Shell executions run in fresh, isolated Linux containers without network; env/tmp do not persist.\n"
        "- Autonomous Verification: No grader or test rubric is accessible. Conclude promptly once deliverables are verified."
    )
    legacy_cfg['agent']['system_prompt'] = legacy_prompt
    assert build_execution_contract_prompt(legacy_cfg) == legacy_prompt

    # Unformatted placeholder elsewhere in prompt raises ConfigurationError
    bad_cfg = copy.deepcopy(cfg)
    bad_cfg['agent']['system_prompt'] = "Custom prompt with leftover {max_model_requests} placeholder."
    with pytest.raises(ConfigurationError, match='Unformatted budget placeholder'):
        build_execution_contract_prompt(bad_cfg)


def test_p0_2_explicit_retry_429_then_200_and_4x_503(tmp_path, config, components):
    """Verifies P0-2: 429 -> 200 records 2 attempt receipts and 1 accepted response;
    4x 503 terminates with model_transport_error and retains 4 attempt receipts.
    """
    from c3_sandbox.l1.core.contracts import HTTPResult
    from c3_sandbox.l1.storage import utc_now

    cfg = copy.deepcopy(config)
    cfg['model']['retry'] = {
        'max_attempts': 4,
        'retry_on_status': [429, 500, 502, 503, 504],
        'retry_on_transport_error': True,
        'backoff_seconds': [0.0, 0.0, 0.0],
        'honor_retry_after': True,
    }

    r429 = HTTPResult(429, b'{"error":{"code":429,"message":"rate limit"}}', {
        'started_at': utc_now(), 'finished_at': utc_now(), 'duration_seconds': 0.01,
        'status_code': 429, 'headers': {'retry-after': '0'}, 'transport_error': None,
        'body_truncated': False, 'attempts': 1, 'auth_recorded': False
    })
    comp1 = copy.deepcopy(components)
    comp1['model'] = ScriptedGemini(cfg['model'], [r429, gemini_call(), gemini_done()])
    out1 = tmp_path / 'retry_ok'
    res1 = run(cfg, out1, injected=comp1)
    assert res1['verification']['verified'], res1['verification']
    m1 = read_json(out1 / 'manifest.json')
    assert m1['counts']['model_requests'] == 2
    assert m1['counts']['model_responses'] == 2
    assert (out1 / 'raw/model/r000000/attempt_01/response.bin').is_file()
    assert (out1 / 'raw/model/r000000/attempt_02/response.bin').is_file()
    assert not (out1 / 'raw/model/r000000/attempt_03/response.bin').exists()
    events1 = scan(out1 / 'trajectory.jsonl')['events']
    assert sum(1 for e in events1 if e['kind'] == 'model.retry') == 1

    r503 = HTTPResult(503, b'{"error":{"code":503,"message":"unavailable"}}', {
        'started_at': utc_now(), 'finished_at': utc_now(), 'duration_seconds': 0.01,
        'status_code': 503, 'headers': {}, 'transport_error': None,
        'body_truncated': False, 'attempts': 1, 'auth_recorded': False
    })
    comp2 = copy.deepcopy(components)
    comp2['model'] = ScriptedGemini(cfg['model'], [r503, r503, r503, r503])
    out2 = tmp_path / 'retry_exhausted'
    res2 = run(cfg, out2, injected=comp2)
    assert res2['verification']['verified'], res2['verification']
    assert res2['status']['termination'] == 'model_transport_error'
    for k in range(1, 5):
        assert (out2 / f'raw/model/r000000/attempt_{k:02d}/response.bin').is_file()
    events2 = scan(out2 / 'trajectory.jsonl')['events']
    assert sum(1 for e in events2 if e['kind'] == 'model.retry') == 3
