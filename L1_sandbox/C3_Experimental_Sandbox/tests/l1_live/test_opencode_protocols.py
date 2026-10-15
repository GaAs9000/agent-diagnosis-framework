import copy
import json
import pytest
from c3_sandbox.l1.core.contracts import ProtocolError, Result
from c3_sandbox.l1.models.opencode import (
    AnthropicMessagesBackend,
    OpenAIResponsesBackend,
    OpenCodeChatBackend,
    OpenCodeGeminiBackend,
)
from c3_sandbox.l1.storage import canonical


def _base_model_cfg(adapter, base_url='https://opencode.ai/inference/openai/v1', model='kimi-k2.6'):
    return {
        'adapter': adapter,
        'kind': 'api',
        'provider': 'opencode',
        'model': model,
        'api_key_env': 'OPENCODE_API_KEY',
        'base_url': base_url,
        'parameters': {'temperature': 1.0, 'max_tokens': 1024},
        'timeout_seconds': 60,
        'supports_images': True,
        'max_response_bytes': 16777216,
    }


def test_opencode_chat_toolless_and_tool_roundtrip():
    cfg = _base_model_cfg('opencode-chat')
    m = OpenCodeChatBackend(cfg)
    conv = m.initial('sys', 'hello')
    req_no_tools = m.request(conv, [])
    assert 'tools' not in req_no_tools and 'tool_choice' not in req_no_tools

    req_with_tools = m.request(conv, [{'name': 'run', 'description': 'run shell', 'parameters': {'type': 'object'}}])
    assert req_with_tools['tool_choice'] == 'auto'
    assert len(req_with_tools['tools']) == 1

    raw_call = {
        'choices': [{
            'message': {
                'role': 'assistant',
                'content': None,
                'tool_calls': [{'type': 'function', 'id': 'call_1', 'function': {'name': 'run', 'arguments': '{"command":"ls"}'}}],
            },
            'finish_reason': 'stop',  # OpenCode chat models may return 'stop' with tool_calls
        }],
        'usage': {'prompt_tokens': 10, 'completion_tokens': 5},
        'model': 'kimi-k2.6',
    }
    reply = m.parse(canonical(raw_call), 'r1')
    assert len(reply.calls) == 1
    assert reply.calls[0].native_id == 'call_1'
    assert reply.calls[0].arguments == {'command': 'ls'}
    assert reply.usage['total_tokens'] == 15

    conv2 = m.accept(conv, reply, [(reply.calls[0], Result({'stdout': 'ok'}))])
    assert conv2['messages'][-1]['role'] == 'tool'
    assert conv2['messages'][-1]['tool_call_id'] == 'call_1'


def test_openai_responses_roundtrip():
    cfg = _base_model_cfg('openai-responses', model='gpt-5.5')
    cfg['parameters'] = {'temperature': 1.0, 'max_output_tokens': 1024}
    m = OpenAIResponsesBackend(cfg)
    conv = m.initial('sys', 'solve task')
    req = m.request(conv, [{'name': 'run', 'description': 'shell', 'parameters': {'type': 'object'}}])
    assert req['instructions'] == 'sys'
    assert req['tools'][0]['name'] == 'run'

    raw_call = {
        'status': 'completed',
        'model': 'gpt-5.5',
        'output': [
            {'type': 'function_call', 'call_id': 'fc_1', 'name': 'run', 'arguments': '{"command":"pwd"}'},
        ],
        'usage': {'input_tokens': 20, 'output_tokens': 8},
    }
    reply = m.parse(canonical(raw_call), 'r1')
    assert len(reply.calls) == 1
    assert reply.calls[0].native_id == 'fc_1'
    assert reply.calls[0].arguments == {'command': 'pwd'}
    assert reply.usage['total_tokens'] == 28

    conv2 = m.accept(conv, reply, [(reply.calls[0], Result({'stdout': '/workspace'}))])
    assert conv2['input'][-1] == {
        'type': 'function_call_output',
        'call_id': 'fc_1',
        'output': '{"stdout": "/workspace"}',
    }

    raw_done = {
        'status': 'completed',
        'model': 'gpt-5.5',
        'output': [
            {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'all done'}]},
        ],
        'usage': {'total_tokens': 30},
    }
    reply_done = m.parse(canonical(raw_done), 'r2')
    assert reply_done.calls == ()
    assert reply_done.text == 'all done'


def test_anthropic_messages_roundtrip():
    cfg = _base_model_cfg('anthropic-messages', base_url='https://opencode.ai/inference/anthropic/v1', model='claude-sonnet-4-6')
    m = AnthropicMessagesBackend(cfg)
    conv = m.initial('sys', 'do task')
    req = m.request(conv, [{'name': 'run', 'description': 'shell', 'parameters': {'type': 'object'}}])
    assert req['max_tokens'] == 1024
    assert req['system'] == 'sys'
    assert req['tools'][0]['input_schema'] == {'type': 'object'}

    raw_call = {
        'role': 'assistant',
        'model': 'claude-sonnet-4-6',
        'stop_reason': 'tool_use',
        'content': [
            {'type': 'text', 'text': 'Inspecting workspace'},
            {'type': 'tool_use', 'id': 'toolu_01', 'name': 'run', 'input': {'command': 'ls -la'}},
        ],
        'usage': {'input_tokens': 50, 'output_tokens': 15},
    }
    reply = m.parse(canonical(raw_call), 'r1')
    assert len(reply.calls) == 1
    assert reply.calls[0].native_id == 'toolu_01'
    assert reply.calls[0].arguments == {'command': 'ls -la'}
    assert reply.text == 'Inspecting workspace'
    assert reply.usage['total_tokens'] == 65

    conv2 = m.accept(conv, reply, [(reply.calls[0], Result({'stdout': 'input submission'}))])
    last_msg = conv2['messages'][-1]
    assert last_msg['role'] == 'user'
    assert last_msg['content'][0]['type'] == 'tool_result'
    assert last_msg['content'][0]['tool_use_id'] == 'toolu_01'


def test_opencode_gemini_uses_bearer_auth(monkeypatch):
    cfg = _base_model_cfg('opencode-gemini', base_url='https://opencode.ai/inference/google/v1beta', model='gemini-3.1-pro')
    m = OpenCodeGeminiBackend(cfg)
    captured = {}

    def fake_post(url, body, timeout, *, gemini=False):
        captured['url'] = url
        captured['gemini'] = gemini
        return 'ok'

    monkeypatch.setattr(m.http, 'post', fake_post)
    assert m.send(b'{}', 10) == 'ok'
    assert captured['url'] == 'https://opencode.ai/inference/google/v1beta/models/gemini-3.1-pro:generateContent'
    assert captured['gemini'] is False
