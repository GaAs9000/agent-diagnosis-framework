"""OpenCode Inference API adapters (OpenAI Chat, OpenAI Responses, Anthropic Messages, Gemini Bearer).

Registered via `l1.models` entry-points so L1 core and existing `GeminiBackend` /
`OpenAICompatibleBackend` remain bit-for-bit untouched.
"""
from __future__ import annotations
import base64
import copy
import json
import re
import uuid
from urllib.parse import quote
from ..core.contracts import Call, Reply, ProtocolError, ConfigurationError, Result
from .http import NativeHTTP
from .native import GeminiBackend, parse_json, _arguments, tool_result_text, is_native_error


def _normalize_usage(usage):
    if not isinstance(usage, dict):
        return None
    out = copy.deepcopy(usage)
    if 'total_tokens' not in out and 'totalTokenCount' not in out:
        inp = out.get('input_tokens', out.get('prompt_tokens'))
        comp = out.get('output_tokens', out.get('completion_tokens'))
        if type(inp) is int and type(comp) is int and inp >= 0 and comp >= 0:
            out['total_tokens'] = inp + comp
    return out


def _attach_usage_metadata(http_res, expected_reported_model=None):
    if http_res.status is not None and 200 <= http_res.status < 300 and not http_res.metadata.get('transport_error'):
        try:
            doc = json.loads(http_res.body)
            if isinstance(doc, dict):
                for field in ('model', 'system_fingerprint'):
                    if isinstance(doc.get(field), str):
                        http_res.metadata['reported_model' if field == 'model' else field] = doc[field]
                if expected_reported_model is not None and doc.get('model') != expected_reported_model:
                    http_res.metadata['model_identity_mismatch'] = {
                        'expected': expected_reported_model, 'observed': doc.get('model'),
                    }
                norm = _normalize_usage(doc.get('usage') or doc.get('usageMetadata'))
                if norm is not None:
                    http_res.metadata['usage'] = norm
        except (ValueError, UnicodeError):
            pass
    return http_res


class OpenCodeChatBackend:
    """OpenCode OpenAI Chat Completions (`/inference/openai/v1/chat/completions`)."""

    def __init__(self, config):
        self.config = config
        self.http = NativeHTTP(config)
        expected = config.get('expected_reported_model')
        if expected is not None and (not isinstance(expected, str) or not expected.strip()):
            raise ConfigurationError('expected_reported_model must be a nonempty string')
        forbidden = {'model', 'messages', 'tools', 'stream', 'n', 'tool_choice', 'parallel_tool_calls'}
        if forbidden.intersection(config['parameters']):
            raise ConfigurationError('parameters override protocol-reserved fields')

    def identity(self):
        return {
            'adapter': 'opencode-chat',
            'kind': self.config['kind'],
            'provider': self.config.get('provider', 'opencode'),
            'requested_model': self.config['model'],
            'reported_revision': self.config.get('reported_revision'),
            'tokenizer_digest': self.config.get('tokenizer_digest'),
            'native_protocol': 'chat-completions',
            'weight_digest': self.config.get('weight_digest'),
        }

    def initial(self, system, prompt):
        return {
            '_opencode_session_id': uuid.uuid4().hex,
            'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': prompt}],
        }

    def request(self, conversation, tools):
        # Checkpointed conversation metadata survives a fresh backend on resume;
        # the session belongs to this conversation and is never a JSON API field.
        session_id = conversation.setdefault('_opencode_session_id', uuid.uuid4().hex)
        if not isinstance(session_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', session_id):
            raise ConfigurationError('Invalid OpenCode conversation session ID')
        self._session_id = session_id
        body = {
            'model': self.config['model'],
            'messages': copy.deepcopy(conversation['messages']),
            'stream': False,
            'n': 1,
            **copy.deepcopy(self.config['parameters']),
        }
        if tools:
            body['tools'] = [{'type': 'function', 'function': copy.deepcopy(t)} for t in tools]
            body['tool_choice'] = 'auto'
        return body

    def send(self, body, timeout):
        if not getattr(self, '_session_id', None):
            raise ConfigurationError('OpenCode chat request must bind a conversation before send')
        return _attach_usage_metadata(self.http.post(
            self.http.base + '/chat/completions', body, timeout,
            extra_headers={'x-opencode-session': self._session_id},
        ), self.config.get('expected_reported_model'))

    def parse(self, body, request_id):
        value = parse_json(body)
        expected = self.config.get('expected_reported_model')
        if expected is not None and value.get('model') != expected:
            raise ProtocolError('Reported model does not match expected_reported_model')
        choices = value.get('choices')
        if 'error' in value or not isinstance(choices, list) or len(choices) != 1:
            raise ProtocolError('Expected one chat choice')
        c = choices[0]
        message = c.get('message')
        if not isinstance(message, dict) or message.get('role') != 'assistant':
            raise ProtocolError('Missing assistant message')
        calls = []
        seen = set()
        raw_calls = message.get('tool_calls') or []
        if not isinstance(raw_calls, list):
            raise ProtocolError('tool_calls must be list')
        for index, tc in enumerate(raw_calls):
            if not isinstance(tc, dict) or tc.get('type') != 'function' or not isinstance(tc.get('function'), dict):
                raise ProtocolError('Invalid tool call')
            native_id = tc.get('id')
            fn = tc['function']
            if not isinstance(native_id, str) or not native_id or native_id in seen:
                raise ProtocolError('Missing/duplicate tool_call_id')
            seen.add(native_id)
            if not isinstance(fn.get('name'), str):
                raise ProtocolError('Invalid function name')
            raw_args = fn.get('arguments', '{}')
            args = (parse_json(raw_args.encode('utf-8'), require_object=not getattr(self, 'native_pi_validation', False)) if isinstance(raw_args, str)
                    else _arguments(raw_args, allow_any=getattr(self, 'native_pi_validation', False)))
            calls.append(Call(f'{request_id}:c{index}', fn['name'], args, native_id))
        finish = c.get('finish_reason')
        if finish not in ('stop', 'tool_calls'):
            raise ProtocolError('Non-complete chat choice: ' + str(finish))
        if not calls and finish == 'tool_calls':
            raise ProtocolError('tool_calls/finish_reason mismatch')
        return Reply(
            tuple(calls),
            copy.deepcopy(message),
            _normalize_usage(value.get('usage')),
            value.get('model'),
            finish,
            message.get('content') if isinstance(message.get('content'), str) else None,
        )

    def accept(self, conversation, reply, results):
        new = copy.deepcopy(conversation)
        new['messages'].append(copy.deepcopy(reply.native_message))
        images = []
        for call, result in results:
            new['messages'].append({
                'role': 'tool',
                'tool_call_id': call.native_id,
                'content': tool_result_text(result),
            })
            if result.image:
                images.append({
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': f'Image returned by tool call {call.native_id}'},
                        {'type': 'image_url', 'image_url': {
                            'url': f'data:{result.image_mime};base64,' + base64.b64encode(result.image).decode('ascii')
                        }},
                    ],
                })
        new['messages'].extend(images)
        return new


class OpenAIResponsesBackend:
    """OpenCode OpenAI Responses API (`/inference/openai/v1/responses`)."""

    def __init__(self, config):
        self.config = config
        self.http = NativeHTTP(config)
        forbidden = {'model', 'input', 'instructions', 'tools', 'stream', 'tool_choice'}
        if forbidden.intersection(config['parameters']):
            raise ConfigurationError('parameters override protocol-reserved fields')

    def identity(self):
        return {
            'adapter': 'openai-responses',
            'kind': self.config['kind'],
            'provider': self.config.get('provider', 'opencode'),
            'requested_model': self.config['model'],
            'reported_revision': self.config.get('reported_revision'),
            'tokenizer_digest': self.config.get('tokenizer_digest'),
            'native_protocol': 'responses',
            'weight_digest': self.config.get('weight_digest'),
        }

    def initial(self, system, prompt):
        return {
            'instructions': system,
            'input': [{'role': 'user', 'content': prompt}],
        }

    def request(self, conversation, tools):
        body = {
            'model': self.config['model'],
            'instructions': conversation['instructions'],
            'input': copy.deepcopy(conversation['input']),
            'stream': False,
            **copy.deepcopy(self.config['parameters']),
        }
        if tools:
            body['tools'] = [
                {
                    'type': 'function',
                    'name': t['name'],
                    'description': t.get('description', ''),
                    'parameters': copy.deepcopy(t.get('parameters', {'type': 'object', 'properties': {}})),
                }
                for t in tools
            ]
            body['tool_choice'] = 'auto'
        return body

    def send(self, body, timeout):
        return _attach_usage_metadata(self.http.post(self.http.base + '/responses', body, timeout))

    def parse(self, body, request_id):
        value = parse_json(body)
        if 'error' in value and value['error'] is not None:
            raise ProtocolError('Provider returned error envelope')
        status = value.get('status')
        if status not in (None, 'completed'):
            raise ProtocolError('Non-complete response status: ' + str(status))
        output = value.get('output')
        if not isinstance(output, list):
            raise ProtocolError('Expected output list in Responses payload')
        calls = []
        texts = []
        seen = set()
        for item in output:
            if not isinstance(item, dict):
                raise ProtocolError('Malformed output item in Responses payload')
            itype = item.get('type')
            if itype == 'function_call':
                name = item.get('name')
                native_id = item.get('call_id') or item.get('id')
                if not isinstance(name, str) or not name:
                    raise ProtocolError('Invalid function_call name')
                if not isinstance(native_id, str) or not native_id or native_id in seen:
                    raise ProtocolError('Missing or duplicate function_call call_id')
                seen.add(native_id)
                raw_args = item.get('arguments', '{}')
                args = (parse_json(raw_args.encode('utf-8'), require_object=not getattr(self, 'native_pi_validation', False)) if isinstance(raw_args, str)
                        else _arguments(raw_args, allow_any=getattr(self, 'native_pi_validation', False)))
                calls.append(Call(f'{request_id}:c{len(calls)}', name, args, native_id))
            elif itype == 'message':
                for part in item.get('content') or []:
                    if isinstance(part, dict) and part.get('type') in ('output_text', 'text') and isinstance(part.get('text'), str):
                        texts.append(part['text'])
        if not texts and isinstance(value.get('output_text'), str) and value['output_text']:
            texts.append(value['output_text'])
        return Reply(
            tuple(calls),
            {'output': copy.deepcopy(output)},
            _normalize_usage(value.get('usage')),
            value.get('model'),
            status or 'completed',
            '\n'.join(texts) or None,
        )

    def accept(self, conversation, reply, results):
        new = copy.deepcopy(conversation)
        for item in reply.native_message.get('output', []):
            new['input'].append(copy.deepcopy(item))
        for call, result in results:
            if not call.native_id:
                raise ProtocolError('Responses function_call_output requires call_id')
            new['input'].append({
                'type': 'function_call_output',
                'call_id': call.native_id,
                'output': tool_result_text(result),
            })
            if result.image:
                new['input'].append({
                    'role': 'user',
                    'content': [
                        {'type': 'input_text', 'text': f'Image returned by tool call {call.native_id}'},
                        {'type': 'input_image', 'image_url': f'data:{result.image_mime};base64,' + base64.b64encode(result.image).decode('ascii')},
                    ],
                })
        return new


class AnthropicMessagesBackend:
    """OpenCode Anthropic Messages API (`/inference/anthropic/v1/messages`)."""

    def __init__(self, config):
        self.config = config
        self.http = NativeHTTP(config)
        forbidden = {'model', 'messages', 'system', 'tools', 'stream', 'tool_choice'}
        if forbidden.intersection(config['parameters']):
            raise ConfigurationError('parameters override protocol-reserved fields')

    def identity(self):
        return {
            'adapter': 'anthropic-messages',
            'kind': self.config['kind'],
            'provider': self.config.get('provider', 'opencode'),
            'requested_model': self.config['model'],
            'reported_revision': self.config.get('reported_revision'),
            'tokenizer_digest': self.config.get('tokenizer_digest'),
            'native_protocol': 'anthropic-messages',
            'weight_digest': self.config.get('weight_digest'),
        }

    def initial(self, system, prompt):
        return {
            'system': system,
            'messages': [{'role': 'user', 'content': prompt}],
        }

    def request(self, conversation, tools):
        params = copy.deepcopy(self.config['parameters'])
        max_tokens = params.pop('max_tokens', params.pop('maxOutputTokens', 4096))
        if type(max_tokens) is not int or max_tokens < 1:
            raise ConfigurationError('anthropic-messages requires positive integer max_tokens')
        body = {
            'model': self.config['model'],
            'max_tokens': max_tokens,
            'system': conversation['system'],
            'messages': copy.deepcopy(conversation['messages']),
            **params,
        }
        if tools:
            body['tools'] = [
                {
                    'name': t['name'],
                    'description': t.get('description', ''),
                    'input_schema': copy.deepcopy(t.get('parameters', {'type': 'object', 'properties': {}})),
                }
                for t in tools
            ]
        return body

    def send(self, body, timeout):
        return _attach_usage_metadata(self.http.post(self.http.base + '/messages', body, timeout))

    def parse(self, body, request_id):
        value = parse_json(body)
        if value.get('type') == 'error' or 'error' in value:
            raise ProtocolError('Provider returned error envelope')
        content = value.get('content')
        if value.get('role', 'assistant') != 'assistant' or not isinstance(content, list):
            raise ProtocolError('Missing assistant content list in Anthropic message')
        calls = []
        texts = []
        seen = set()
        for block in content:
            if not isinstance(block, dict):
                raise ProtocolError('Malformed Anthropic content block')
            btype = block.get('type')
            if btype == 'tool_use':
                native_id = block.get('id')
                name = block.get('name')
                if not isinstance(native_id, str) or not native_id or native_id in seen:
                    raise ProtocolError('Invalid or duplicate tool_use id')
                if not isinstance(name, str) or not name:
                    raise ProtocolError('Invalid tool_use name')
                seen.add(native_id)
                args = _arguments(block.get('input', {}), allow_any=getattr(self, 'native_pi_validation', False))
                calls.append(Call(f'{request_id}:c{len(calls)}', name, args, native_id))
            elif btype == 'text' and isinstance(block.get('text'), str):
                texts.append(block['text'])
        stop_reason = value.get('stop_reason')
        if stop_reason not in (None, 'end_turn', 'tool_use', 'stop_sequence'):
            raise ProtocolError('Non-complete Anthropic stop_reason: ' + str(stop_reason))
        if not calls and stop_reason == 'tool_use':
            raise ProtocolError('tool_use stop_reason without tool_use blocks')
        return Reply(
            tuple(calls),
            {'role': 'assistant', 'content': copy.deepcopy(content)},
            _normalize_usage(value.get('usage')),
            value.get('model'),
            stop_reason,
            '\n'.join(texts) or None,
        )

    def accept(self, conversation, reply, results):
        new = copy.deepcopy(conversation)
        new['messages'].append(copy.deepcopy(reply.native_message))
        if results:
            blocks = []
            for call, result in results:
                if not call.native_id:
                    raise ProtocolError('Anthropic tool_result requires tool_use_id')
                obs_json = tool_result_text(result)
                if result.image:
                    content_val = [
                        {'type': 'text', 'text': obs_json},
                        {
                            'type': 'image',
                            'source': {
                                'type': 'base64',
                                'media_type': result.image_mime,
                                'data': base64.b64encode(result.image).decode('ascii'),
                            },
                        },
                    ]
                else:
                    content_val = obs_json
                blocks.append({
                    'type': 'tool_result',
                    'tool_use_id': call.native_id,
                    'content': content_val,
                    **({'is_error': True} if is_native_error(result) else {}),
                })
            new['messages'].append({'role': 'user', 'content': blocks})
        return new


class OpenCodeGeminiBackend(GeminiBackend):
    """OpenCode Gemini endpoint (`/inference/google/v1beta/models/<model>:generateContent`) using Bearer auth."""

    def identity(self):
        info = dict(super().identity())
        info['adapter'] = 'opencode-gemini'
        info['provider'] = self.config.get('provider', 'opencode')
        return info

    def send(self, body, timeout):
        model = self.config['model'].removeprefix('models/')
        return self.http.post(
            f'{self.http.base}/models/{quote(model, safe="")}:generateContent',
            body,
            timeout,
            gemini=False,
        )
