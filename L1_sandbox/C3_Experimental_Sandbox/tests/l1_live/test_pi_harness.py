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


def test_f22_treatment_arm_cold_resume_from_plain_parent_and_at_request_28(tmp_path, config, components):
    from c3_sandbox.l1.core.contracts import ConfigurationError
    from l2_sandbox.evidence.l1 import L1Package

    # 1. Plain parent without intervention
    plain_cfg = copy.deepcopy(config)
    plain_cfg['agent'] = {'name': 'pi-coding-agent', 'version': '0.73.1', 'harness': 'pi'}
    plain_cfg['budget'].update(max_model_requests=4, max_tool_calls=4)
    parent_dir = tmp_path / 'plain_parent'
    comp = {
        **components,
        'model': ScriptedGemini(
            plain_cfg['model'],
            [
                gemini_call('write', {'path': 'a.txt', 'content': 'v1'}, native_id='c1'),
                gemini_call('write', {'path': 'b.txt', 'content': 'v2'}, native_id='c2'),
                gemini_done(),
            ],
        ),
        'runtime': PiFixtureRuntime(),
    }
    assert run(plain_cfg, parent_dir, injected=comp)['verification']['verified']

    # 2. Plain resume from cp000001 is byte-identical at r000001
    plain_child = tmp_path / 'plain_child'
    comp_plain = {
        **components,
        'model': ScriptedGemini(plain_cfg['model'], [gemini_done()]),
        'runtime': PiFixtureRuntime(),
    }
    assert run(plain_cfg, plain_child, injected=comp_plain, parent=parent_dir, checkpoint_id='cp000001')['verification']['verified']
    assert (parent_dir / 'raw/model/r000001/request.json').read_bytes() == (
        plain_child / 'raw/model/r000001/request.json'
    ).read_bytes()

    # 3. Treatment resume from cp000001 (k=1) with at_request=2 injects message before request 2 (r000001)
    iv = {'at_request': 2, 'message': '请现在就把目前最好的成果按任务要求保存为最终交付物，之后再决定是否继续改进。'}
    treat_cfg = copy.deepcopy(plain_cfg)
    treat_cfg['agent']['intervention'] = iv
    treat_child = tmp_path / 'treat_child'
    comp_treat = {
        **components,
        'model': ScriptedGemini(treat_cfg['model'], [gemini_done()]),
        'runtime': PiFixtureRuntime(),
    }
    assert run(treat_cfg, treat_child, injected=comp_treat, parent=parent_dir, checkpoint_id='cp000001')['verification']['verified']
    req_treat = read_json(treat_child / 'raw/model/r000001/request.json')
    assert iv['message'] in str(req_treat)

    parent_pkg = L1Package(parent_dir)
    L1Package(plain_child).validate_branch(parent_pkg, 1)
    L1Package(treat_child).validate_branch(parent_pkg, 1, intervention=iv)

    # Reject retroactive intervention at_request <= k
    bad_cfg = copy.deepcopy(plain_cfg)
    bad_cfg['agent']['intervention'] = {'at_request': 1, 'message': 'retroactive'}
    with pytest.raises(ConfigurationError):
        run(bad_cfg, tmp_path / 'bad_child', injected=comp_treat, parent=parent_dir, checkpoint_id='cp000001')

    # 4. Fresh run with at_request=28 injects only right before request 28 (r000027)
    fresh28_cfg = copy.deepcopy(plain_cfg)
    fresh28_cfg['budget'].update(max_model_requests=30, max_tool_calls=30)
    fresh28_cfg['agent']['intervention'] = {'at_request': 28, 'message': 'INTERVENTION_AT_28'}
    scripted28 = [gemini_call('write', {'path': 'step.txt', 'content': str(i)}, native_id=f's{i}') for i in range(27)] + [gemini_done()]
    fresh28_dir = tmp_path / 'fresh28'
    comp28 = {
        **components,
        'model': ScriptedGemini(fresh28_cfg['model'], scripted28),
        'runtime': PiFixtureRuntime(),
    }
    assert run(fresh28_cfg, fresh28_dir, injected=comp28)['verification']['verified']
    assert b'INTERVENTION_AT_28' not in (fresh28_dir / 'raw/model/r000026/request.json').read_bytes()
    assert b'INTERVENTION_AT_28' in (fresh28_dir / 'raw/model/r000027/request.json').read_bytes()


def test_pi_context_compaction_and_post_compaction_cold_resume(tmp_path, config, components):
    import json
    from c3_sandbox.l1.core.contracts import HTTPResult
    from c3_sandbox.l1.evidence.live import scan
    from c3_sandbox.l1.models.native import OpenAICompatibleBackend
    from c3_sandbox.l1.storage import canonical, utc_now

    def openai_call(name, args, native_id):
        return {
            'id': f'chatcmpl-{native_id}',
            'object': 'chat.completion',
            'model': 'Qwen/Qwen3.6-27B',
            'choices': [
                {
                    'index': 0,
                    'message': {
                        'role': 'assistant',
                        'content': 'Executing step ' + ('X' * 1200),
                        'tool_calls': [
                            {
                                'id': native_id,
                                'type': 'function',
                                'function': {'name': name, 'arguments': json.dumps(args)},
                            }
                        ],
                    },
                    'finish_reason': 'tool_calls',
                }
            ],
            'usage': {'prompt_tokens': 500, 'completion_tokens': 300, 'total_tokens': 800},
        }

    def openai_done():
        return {
            'id': 'chatcmpl-done',
            'object': 'chat.completion',
            'model': 'Qwen/Qwen3.6-27B',
            'choices': [
                {
                    'index': 0,
                    'message': {'role': 'assistant', 'content': 'Task complete.'},
                    'finish_reason': 'stop',
                }
            ],
            'usage': {'prompt_tokens': 400, 'completion_tokens': 10, 'total_tokens': 410},
        }

    class ScriptedOpenAI(OpenAICompatibleBackend):
        def __init__(self, cfg, responses):
            super().__init__(cfg)
            self.responses = list(responses)

        def send(self, body, timeout):
            val = self.responses.pop(0)
            return HTTPResult(
                200,
                canonical(val),
                {
                    'fixture': True,
                    'started_at': utc_now(),
                    'finished_at': utc_now(),
                    'transport_error': None,
                    'body_truncated': False,
                },
            )

    pi_cfg = copy.deepcopy(config)
    pi_cfg['model'].update(
        adapter='openai-compatible',
        kind='local',
        provider='vllm-local-aml-b',
        model='Qwen/Qwen3.6-27B',
        base_url='http://127.0.0.1:8001/v1',
        weight_digest='sha256:' + 'a' * 64,
        tokenizer_digest='sha256:' + 'b' * 64,
    )
    pi_cfg['agent'] = {
        'name': 'pi-coding-agent',
        'version': '0.73.1',
        'harness': 'pi',
        'compaction': {
            'enabled': True,
            'context_window_tokens': 2400,
            'reserve_tokens': 600,
            'keep_recent_tokens': 500,
        },
    }
    pi_cfg['budget'].update(max_model_requests=6, max_tool_calls=6)
    pi_cfg['recording']['observation_limit_chars'] = 20000

    responses = [
        openai_call('write', {'path': 'step1.txt', 'content': 'A' * 1500}, 'call-1'),
        openai_call('write', {'path': 'step2.txt', 'content': 'B' * 1500}, 'call-2'),
        openai_call('write', {'path': 'step3.txt', 'content': 'C' * 1500}, 'call-3'),
        openai_call('bash', {'command': 'echo done'}, 'call-4'),
        openai_done(),
    ]
    parent_dir = tmp_path / 'compact_parent'
    comp = {
        **components,
        'model': ScriptedOpenAI(pi_cfg['model'], responses),
        'runtime': PiFixtureRuntime(),
        'evaluation': FixtureEvaluator('success'),
    }
    res_parent = run(pi_cfg, parent_dir, injected=comp)
    assert res_parent['verification']['verified'], res_parent

    events = scan(parent_dir / 'trajectory.jsonl')['events']
    compacted_events = [e for e in events if e['kind'] == 'session.compacted']
    assert len(compacted_events) >= 1
    first_compact_step = compacted_events[0]['data']['at_request']
    cp_id = f'cp{first_compact_step:06d}'
    cp_doc = read_json(parent_dir / f'checkpoints/{cp_id}.json')
    assert '_pi_compaction' in cp_doc['conversation']
    assert any(
        '[PI CONTEXT COMPACTION SUMMARY' in str(m.get('content', ''))
        for m in cp_doc['conversation']['messages']
    )

    # Cold-resume from the post-compaction checkpoint and verify byte-identical first model request (V3-G1)
    remaining_steps = 5 - first_compact_step
    child_responses = [
        openai_call('bash', {'command': 'echo done'}, f'child-call-{i}')
        for i in range(max(0, remaining_steps - 1))
    ] + [openai_done()]
    child_dir = tmp_path / 'compact_child'
    child_comp = {
        **components,
        'model': ScriptedOpenAI(pi_cfg['model'], child_responses),
        'runtime': PiFixtureRuntime(),
        'evaluation': FixtureEvaluator('success'),
    }
    res_child = run(pi_cfg, child_dir, injected=child_comp, parent=parent_dir, checkpoint_id=cp_id)
    assert res_child['verification']['verified'], res_child
    req_rel = f'raw/model/r{first_compact_step:06d}/request.json'
    assert (parent_dir / req_rel).read_bytes() == (child_dir / req_rel).read_bytes()
    assert b'[PI CONTEXT COMPACTION SUMMARY' in (child_dir / req_rel).read_bytes()


