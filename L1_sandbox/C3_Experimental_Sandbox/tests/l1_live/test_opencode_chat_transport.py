"""OpenCode wire regressions: no external traffic or real credentials."""
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import pytest
from c3_sandbox.l1.core.contracts import ConfigurationError, ProtocolError, Result
from c3_sandbox.l1.models.http import NativeHTTP
from c3_sandbox.l1.models.opencode import OpenCodeChatBackend
from c3_sandbox.l1.storage import canonical


def response(model='deepseek-v4-pro'):
    return {
        'model': model, 'system_fingerprint': 'shared-routing-fingerprint',
        'choices': [{'message': {
            'role': 'assistant', 'content': None,
            'reasoning_content': 'Inspect both files before editing.',
            'tool_calls': [{'id': f'call_{i}', 'type': 'function',
                            'function': {'name': 'run', 'arguments': json.dumps({'command': f'cat {i}'})}}
                           for i in (1, 2)],
        }, 'finish_reason': 'tool_calls'}],
        'usage': {'prompt_tokens': 20, 'completion_tokens': 10,
                  'prompt_tokens_details': {'cached_tokens': 5}},
    }


@pytest.fixture
def wire_server():
    received = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append({'path': self.path, 'headers': dict(self.headers),
                             'body': json.loads(self.rfile.read(int(self.headers['Content-Length'])))})
            if self.path.endswith('/redirect'):
                self.send_response(307)
                self.send_header('Location', '/v1/chat/completions')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            payload = canonical(response())
            self.send_response(503 if len(received) == 1 else 200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1', received
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def config(base_url):
    return {'kind': 'local', 'provider': 'opencode', 'model': 'deepseek-v4-pro',
            'base_url': base_url, 'api_key_env': 'L1_TEST_OPENCODE_KEY',
            'parameters': {'temperature': 1, 'thinking': {'type': 'disabled'}},
            'max_response_bytes': 100000}


def test_chat_session_retry_resume_and_deepseek_native_history(wire_server, monkeypatch):
    url, received = wire_server
    monkeypatch.setenv('L1_TEST_OPENCODE_KEY', 'local-test-only')
    cfg = config(url)
    backend = OpenCodeChatBackend(cfg)
    conversation = backend.initial('system', 'inspect')
    session = conversation['_opencode_session_id']
    body = backend.request(conversation, [])
    assert set(body) == {'model', 'messages', 'stream', 'n', 'temperature', 'thinking'}
    assert body['thinking'] == {'type': 'disabled'}
    encoded = canonical(body)
    assert backend.send(encoded, 5).status == 503
    result = backend.send(encoded, 5)
    assert result.status == 200
    assert received[0]['body'] == received[1]['body'] == body
    assert result.metadata['opencode_session_id'] == session
    assert result.metadata['reported_model'] == 'deepseek-v4-pro'
    assert result.metadata['system_fingerprint'] == 'shared-routing-fingerprint'
    assert result.metadata['usage']['total_tokens'] == 30
    assert result.metadata['usage']['prompt_tokens_details'] == {'cached_tokens': 5}
    assert 'local-test-only' not in json.dumps(result.metadata)
    reply = backend.parse(result.body, 'r1')
    assert reply.model_version == 'deepseek-v4-pro'
    assert reply.usage == result.metadata['usage']
    updated = backend.accept(conversation, reply, [
        (reply.calls[0], Result({'stdout': 'one'}, image=b'first', image_mime='image/png')),
        (reply.calls[1], Result({'stdout': 'two'}, image=b'second', image_mime='image/png')),
    ])
    assert [m['role'] for m in updated['messages']] == ['system', 'user', 'assistant', 'tool', 'tool', 'user', 'user']
    assert updated['messages'][2] == response()['choices'][0]['message']
    assert [m['tool_call_id'] for m in updated['messages'][3:5]] == ['call_1', 'call_2']
    restored = json.loads(json.dumps(updated))
    resumed_backend = OpenCodeChatBackend(copy.deepcopy(cfg))
    replay = resumed_backend.request(restored, [])
    assert replay['messages'] == updated['messages']
    assert '_opencode_session_id' not in replay
    resumed_backend.send(canonical(replay), 5)
    assert received[-1]['body'] == replay
    assert {r['headers']['X-Opencode-Session'] for r in received} == {session}
    assert all(r['headers']['User-Agent'] == 'l1-evidence/0.2.0' for r in received)
    assert all(r['headers']['Authorization'] == 'Bearer local-test-only' for r in received)
    assert all(r['path'] == '/v1/chat/completions' for r in received)
    other = backend.initial('system', 'different episode')
    assert other['_opencode_session_id'] != session
    backend.send(canonical(backend.request(other, [])), 5)
    assert received[-1]['headers']['X-Opencode-Session'] == other['_opencode_session_id']
    backend.send(canonical(backend.request(restored, [])), 5)
    assert received[-1]['headers']['X-Opencode-Session'] == session


def test_model_identity_does_not_collapse_to_shared_fingerprint():
    backend = OpenCodeChatBackend(config('http://localhost/v1'))
    first = backend.parse(canonical(response('deepseek-revision-a')), 'r1')
    second = backend.parse(canonical(response('deepseek-revision-b')), 'r2')
    assert first.model_version == 'deepseek-revision-a'
    assert second.model_version == 'deepseek-revision-b'


@pytest.mark.parametrize('headers', [
    {'Authorization': 'override'}, {'x-goog-api-key': 'override'},
    {'Cookie': 'override'}, {'User-Agent': 'override'}, {'Host': 'override'},
    {'Content-Length': '0'}, {'x-opencode-session\r\nAuthorization': 'x'},
    {'x-opencode-session': 'x\r\nAuthorization: x'}, {'x-opencode-session': ''},
    {'x-opencode-session': 'x' * 129}, {'x-opencode-session': 123},
])
def test_extra_headers_reject_auth_overrides_and_injection(headers):
    transport = NativeHTTP(config('http://localhost/v1'))
    with pytest.raises(ConfigurationError, match='extra HTTP header'):
        transport.post('http://localhost/v1/chat/completions', b'{}', 1, extra_headers=headers)


def test_legacy_checkpoint_session_created_once_and_invalid_session_rejected():
    backend = OpenCodeChatBackend(config('http://localhost/v1'))
    legacy = {'messages': []}
    backend.request(legacy, [])
    session = legacy['_opencode_session_id']
    backend.request(legacy, [])
    assert legacy['_opencode_session_id'] == session
    with pytest.raises(ConfigurationError, match='session ID'):
        backend.request({'messages': [], '_opencode_session_id': 'x\r\nx'}, [])
    with pytest.raises(ConfigurationError, match='bind a conversation'):
        OpenCodeChatBackend(config('http://localhost/v1')).send(b'{}', 1)


def test_configured_model_identity_guard_in_wire_receipt_and_parse(wire_server, monkeypatch):
    url, _ = wire_server
    monkeypatch.setenv('L1_TEST_OPENCODE_KEY', 'local-test-only')
    cfg = config(url)
    cfg['expected_reported_model'] = 'expected-deepseek-revision'
    backend = OpenCodeChatBackend(cfg)
    payload = canonical(backend.request(backend.initial('s', 'u'), []))
    backend.send(payload, 5)  # fixture's initial 503
    result = backend.send(payload, 5)
    assert result.metadata['model_identity_mismatch'] == {
        'expected': 'expected-deepseek-revision', 'observed': 'deepseek-v4-pro'}
    assert result.metadata['usage']['total_tokens'] == 30
    with pytest.raises(ProtocolError, match='expected_reported_model'):
        backend.parse(result.body, 'r1')
    with pytest.raises(ProtocolError, match='expected_reported_model'):
        backend.parse(canonical({k: v for k, v in response().items() if k != 'model'}), 'r2')
    cfg['expected_reported_model'] = 'deepseek-v4-pro'
    matching = OpenCodeChatBackend(cfg)
    result = matching.send(canonical(matching.request(matching.initial('s', 'u'), [])), 5)
    assert 'model_identity_mismatch' not in result.metadata
    assert matching.parse(result.body, 'r3').model_version == 'deepseek-v4-pro'


def test_extra_session_header_preserves_response_cap_and_no_redirect(wire_server, monkeypatch):
    url, received = wire_server
    monkeypatch.setenv('L1_TEST_OPENCODE_KEY', 'local-test-only')
    cfg = config(url)
    cfg['max_response_bytes'] = 80
    transport = NativeHTTP(cfg)
    result = transport.post(url + '/chat/completions', b'{}', 5,
                            extra_headers={'x-opencode-session': 'test-session'})
    assert result.metadata['body_truncated'] and len(result.body) == 80
    assert result.metadata['attempts'] == 1 and len(received) == 1
    result = transport.post(url + '/redirect', b'{}', 5,
                            extra_headers={'x-opencode-session': 'test-session'})
    assert result.status == 307 and len(received) == 2
    assert result.metadata['opencode_session_id'] == 'test-session'
