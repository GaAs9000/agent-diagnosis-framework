from __future__ import annotations
import copy
from pathlib import Path
import subprocess
import sys

import pytest

from c3_sandbox.l1.core.contracts import Call, ProtocolError, Result
from c3_sandbox.l1.core.orchestrator import PI_SYSTEM, build_execution_contract_prompt, run
from c3_sandbox.l1.evidence.live import verify
from c3_sandbox.l1.runtime.tools import (
    PI_TOOLS,
    active_harness,
    execute_pi_edit,
    execute_pi_read,
    execute_pi_write,
    format_pi_bash_observation,
    tools_for_config,
    validate_call,
)
from c3_sandbox.l1.storage import read_json
from .conftest import FixtureEvaluator, ScriptedGemini, gemini_call, gemini_done


class PiFixtureRuntime:
    """Executes real Pi workspace tools (read/write/edit) and fixed fixture bash inside workspace."""

    is_quiescent = True

    def preflight(self):
        return {'adapter': 'test_fixture', 'image_id': 'sha256:' + '0' * 64, 'is_test_double': True}

    def qualification(self):
        return {
            'eligible': True,
            'image_id': 'sha256:' + '0' * 64,
            'scopes': ['workspace_files', 'native_conversation'],
            'reason_codes': [],
            'test_only': True,
        }

    def execute(self, call, workspace, timeout):
        validate_call(call, harness='pi')
        if call.name == 'read':
            return execute_pi_read(call, workspace, max_image_bytes=8388608, supports_images=True)
        if call.name == 'write':
            return execute_pi_write(call, workspace)
        if call.name == 'edit':
            return execute_pi_edit(call, workspace)
        if call.name == 'bash':
            script = (
                "from pathlib import Path; import shutil; root=Path('.'); "
                "src=next((root/'input').glob('*.docx')); "
                "shutil.copyfile(src,root/'submission'/src.name); print('pi bash executed')"
            )
            p = subprocess.run([sys.executable, '-c', script], cwd=workspace, capture_output=True, check=True)
            obs = format_pi_bash_observation(p.stdout, p.stderr, 0, None, True, False)
            return Result(obs, p.stdout, p.stderr, provenance={'test_fixture': True, 'quiescent': True})
        raise AssertionError(f'Unexpected tool: {call.name}')


def test_pi_prompt_and_tool_surface_idempotent(config):
    pi_cfg = copy.deepcopy(config)
    pi_cfg['agent'] = {'name': 'pi-coding-agent', 'version': '0.73.1', 'harness': 'pi'}
    assert active_harness(pi_cfg) == 'pi'
    assert [t['name'] for t in tools_for_config(pi_cfg)] == ['read', 'bash', 'edit', 'write']
    p1 = build_execution_contract_prompt(pi_cfg)
    assert PI_SYSTEM in p1
    assert 'Current working directory: /workspace' in p1
    pi_cfg['agent']['system_prompt'] = p1
    p2 = build_execution_contract_prompt(pi_cfg)
    assert p1 == p2


def test_pi_workspace_tools_write_edit_read_and_errors(tmp_path):
    ws = tmp_path / 'workspace'
    ws.mkdir()

    # 1. write
    w_call = Call('r0:c0', 'write', {'path': 'scripts/solve.py', 'content': 'a = 1\nb = 2\nc = 3\n'}, 'id-w')
    validate_call(w_call, harness='pi')
    w_res = execute_pi_write(w_call, ws)
    assert w_res.observation['exit_code'] == 0
    assert (ws / 'scripts/solve.py').read_text() == 'a = 1\nb = 2\nc = 3\n'

    # 2. edit (multiple disjoint replacements)
    e_call = Call(
        'r1:c0',
        'edit',
        {
            'path': '/workspace/scripts/solve.py',
            'edits': [
                {'oldText': 'a = 1', 'newText': 'a = 10'},
                {'oldText': 'c = 3', 'newText': 'c = 30'},
            ],
        },
        'id-e',
    )
    validate_call(e_call, harness='pi')
    e_res = execute_pi_edit(e_call, ws)
    assert e_res.observation['exit_code'] == 0
    assert 'Successfully replaced 2 block(s)' in e_res.observation['output']
    assert (ws / 'scripts/solve.py').read_text() == 'a = 10\nb = 2\nc = 30\n'

    # 3. edit error (not found returns exit_code=1 without raising)
    bad_edit = Call(
        'r2:c0',
        'edit',
        {'path': 'scripts/solve.py', 'edits': [{'oldText': 'nonexistent', 'newText': 'x'}]},
        'id-bad',
    )
    bad_res = execute_pi_edit(bad_edit, ws)
    assert bad_res.observation['exit_code'] == 1
    assert 'Could not find the exact text' in bad_res.observation['stderr']

    # 4. read text with offset/limit
    r_call = Call('r3:c0', 'read', {'path': 'scripts/solve.py', 'offset': 2, 'limit': 1}, 'id-r')
    validate_call(r_call, harness='pi')
    r_res = execute_pi_read(r_call, ws, max_image_bytes=1024 * 1024)
    assert r_res.observation['exit_code'] == 0
    assert 'b = 2' in r_res.observation['output']
    assert 'Use offset=3 to continue' in r_res.observation['output']

    # 5. read PNG image
    png_bytes = b'\x89PNG\r\n\x1a\n' + b'\x00' * 32
    (ws / 'preview.png').write_bytes(png_bytes)
    img_call = Call('r4:c0', 'read', {'path': '/workspace/preview.png'}, 'id-img')
    img_res = execute_pi_read(img_call, ws, max_image_bytes=1024 * 1024, supports_images=True)
    assert img_res.observation['exit_code'] == 0
    assert img_res.image == png_bytes
    assert img_res.image_mime == 'image/png'

    # 6. Minimal tools rejected under Pi harness and vice versa
    with pytest.raises(ProtocolError):
        validate_call(Call('r5:c0', 'run', {'command': 'ls'}, 'id-run'), harness='pi')
    with pytest.raises(ProtocolError):
        validate_call(w_call, harness='minimal')


def test_pi_cold_resume_byte_identical_first_request_and_l2_facts(tmp_path, config, components):
    pi_cfg = copy.deepcopy(config)
    pi_cfg['agent'] = {'name': 'pi-coding-agent', 'version': '0.73.1', 'harness': 'pi'}
    pi_cfg['budget'].update(max_model_requests=6, max_tool_calls=6)
    pi_cfg['recording']['observation_limit_chars'] = 20000

    responses = [
        gemini_call('write', {'path': 'work.py', 'content': 'x = 1\ny = 2\n'}, native_id='call-1'),
        gemini_call('edit', {'path': 'work.py', 'edits': [{'oldText': 'x = 1', 'newText': 'x = 42'}]}, native_id='call-2'),
        gemini_call('read', {'path': '/workspace/submission/out.docx'}, native_id='call-3'),
        gemini_call('bash', {'command': 'python3 work.py'}, native_id='call-4'),
        gemini_done(),
    ]
    comp = {
        **components,
        'model': ScriptedGemini(pi_cfg['model'], responses),
        'runtime': PiFixtureRuntime(),
        'evaluation': FixtureEvaluator('success'),
    }
    parent_dir = tmp_path / 'pi_parent'
    res_parent = run(pi_cfg, parent_dir, injected=comp)
    assert res_parent['verification']['verified'], res_parent
    assert res_parent['status']['termination'] == 'assistant_finished'

    # Cold-resume from cp000002 (after write and edit, right before read)
    child_responses = [
        gemini_call('bash', {'command': 'python3 work.py'}, native_id='child-call-3'),
        gemini_done(),
    ]
    child_comp = {
        **components,
        'model': ScriptedGemini(pi_cfg['model'], child_responses),
        'runtime': PiFixtureRuntime(),
        'evaluation': FixtureEvaluator('success'),
    }
    child_dir = tmp_path / 'pi_child'
    res_child = run(pi_cfg, child_dir, injected=child_comp, parent=parent_dir, checkpoint_id='cp000002')
    assert res_child['verification']['verified'], res_child
    assert verify(parent_dir)['verified']
    assert verify(child_dir)['verified']

    # Verify byte-for-byte identity of the first resumed model request (r000002)
    parent_req2 = (parent_dir / 'raw/model/r000002/request.json').read_bytes()
    child_req2 = (child_dir / 'raw/model/r000002/request.json').read_bytes()
    assert parent_req2 == child_req2

    # Verify L2 fact extraction on Pi parent package
    from l2_sandbox.evidence.l1 import L1Package
    from l2_sandbox.facts.build import build_facts
    from l2_sandbox.predicates import RULE_VOCABULARY, eval_predicate_on_turn

    pkg = L1Package(parent_dir)
    facts = build_facts(pkg)
    req_names = [f['value']['name'] for f in facts if f['type'] == 'tool_requested']
    assert req_names == ['write', 'edit', 'read', 'bash']
    # Turn 3 called read on /workspace/submission/out.docx (which does not exist yet -> exit_code=1, no submission file change)
    assert eval_predicate_on_turn(RULE_VOCABULARY['submission_readback'], facts, 3)
    assert eval_predicate_on_turn(RULE_VOCABULARY['tool_execution_anomaly'], facts, 3)
    # Turn 4 wrote to submission/* via bash
    assert eval_predicate_on_turn(RULE_VOCABULARY['submission_modified'], facts, 4)


def test_intervention_cold_resume_byte_identical(tmp_path, config, components):
    pi_cfg = copy.deepcopy(config)
    pi_cfg['agent'] = {
        'name': 'pi-coding-agent',
        'version': '0.73.1',
        'harness': 'pi',
        'intervention': {
            'at_request': 2,
            'message': '[SYSTEM REMINDER] Conclude and place final deliverables in /workspace/submission.',
        },
    }
    pi_cfg['budget'].update(max_model_requests=4, max_tool_calls=4)
    responses = [
        gemini_call('write', {'path': 'a.txt', 'content': 'hello'}, native_id='c1'),
        gemini_done(),
    ]
    comp = {
        **components,
        'model': ScriptedGemini(pi_cfg['model'], responses),
        'runtime': PiFixtureRuntime(),
    }
    parent_dir = tmp_path / 'iv_parent'
    r1 = run(pi_cfg, parent_dir, injected=comp)
    assert r1['verification']['verified']

    child_comp = {
        **components,
        'model': ScriptedGemini(pi_cfg['model'], [gemini_done()]),
        'runtime': PiFixtureRuntime(),
    }
    child_dir = tmp_path / 'iv_child'
    r2 = run(pi_cfg, child_dir, injected=child_comp, parent=parent_dir, checkpoint_id='cp000001')
    assert r2['verification']['verified']
    assert (parent_dir / 'raw/model/r000001/request.json').read_bytes() == (
        child_dir / 'raw/model/r000001/request.json'
    ).read_bytes()
    assert b'[SYSTEM REMINDER]' in (child_dir / 'raw/model/r000001/request.json').read_bytes()
