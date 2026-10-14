"""G0: genuine native qualification plus clearly synthetic HTTP/real Docker fixtures."""
import asyncio
import base64
import copy
import hashlib
import json
from pathlib import Path

from c3_sandbox.evidence._storage import encode
from c3_sandbox.r1.docker_runtime import DockerRuntime
from c3_sandbox.r1.qualification import TOOL_SCHEMAS, image_identity
from c3_sandbox.r1.tool_protocol import run_protocol_fixture, validate_wire_response

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/e2-r1-1/tool_protocol_audit'


def write(path, value):
    with path.open('xb') as stream:
        stream.write(encode(value))


def response(calls=None, finish='stop', content='Synthetic fixture finished.'):
    return {'id': 'synthetic-protocol-response', 'object': 'chat.completion', 'created': 0,
        'model': 'default_model', 'choices': [{'index': 0, 'finish_reason': finish,
        'message': {'role': 'assistant', 'content': content, 'tool_calls': calls}}],
        'usage': {'prompt_tokens': 10, 'completion_tokens': 4, 'total_tokens': 14}}


async def main():
    live = OUT / 'live_qualification'
    qualification = json.loads((live / 'qualification.json').read_bytes())
    trace = (live / 'agent/model_calls.jsonl').read_bytes()
    assert hashlib.sha256(trace).hexdigest() == qualification['model_trace_sha256']
    records = [json.loads(line) for line in trace.splitlines()]
    requests = [r for r in records if r['type'] == 'request']
    wires = [r for r in records if r['type'] == 'wire_response']
    assert qualification['status'] == 'passed' and len(requests) == len(wires) == 2
    for request, wire in zip(requests, wires):
        assert request['identity_context']['model_fingerprint'] == qualification['model_fingerprint']
        assert request['identity_context']['tokenizer_fingerprint'] == qualification['tokenizer_fingerprint']
        decoded = json.loads(base64.b64decode(wire['response_body_base64']))
        assert validate_wire_response(decoded).status != 'protocol_invalid'
    call = {'id': 'synthetic-call-1', 'type': 'function', 'function': {
        'name': 'workspace_python', 'arguments': json.dumps({'code': "from pathlib import Path\nprint(Path('fixture_nonce.txt').read_text())"})}}
    wrong_json, missing_id = copy.deepcopy(call), copy.deepcopy(call)
    wrong_json['function']['arguments'] = '{"code":'
    missing_id.pop('id')
    cases = [
        ('valid_native', [response([call], 'tool_calls', ''), response()], 1),
        ('normal_text', [response()], 0),
        ('empty_tool_payload', [response([], 'tool_calls', '')], 0),
        ('invalid_json', [response([wrong_json], 'tool_calls', '')], 0),
        ('missing_tool_id', [response([missing_id], 'tool_calls', '')], 0),
        ('token_truncation', [response([call], 'length', '')], 0),
    ]
    folder = OUT / 'controlled_fixtures-v2'
    folder.mkdir()
    source = folder / 'input'
    source.mkdir(mode=0o700)
    (source / 'fixture_nonce.txt').write_text('public synthetic fixture value')
    checks = []
    with DockerRuntime(image_identity('c3-r1-agent:locked'), folder / 'runtime') as actor:
        actor.seed_workspace(source)
        for name, scripted, expected in cases:
            dispatched = []
            def invoke(code):
                result = actor.exec_python(code)
                dispatched.append({'code': code, 'actual_result': result})
                return result
            result = await run_protocol_fixture(scripted, folder / name,
                messages=[{'role': 'user', 'content': 'Synthetic protocol fixture; not a model task.'}],
                tools={'workspace_python': invoke}, tool_schemas=TOOL_SCHEMAS)
            write(folder / name / 'result.json', result)
            assert len(dispatched) == expected
            assert result['tokens']['input_tokens'] == 10 * len(scripted)
            assert result['tokens']['output_tokens'] == 4 * len(scripted)
            assert result.get('token_usage_unknown') is False
            if expected:
                assert dispatched[0]['actual_result']['stdout'].strip() == 'public synthetic fixture value'
                assert result['messages'][-2]['tool_call_id'] == call['id']
            elif name != 'normal_text':
                assert result['protocol_status'] == 'protocol_invalid'
            checks.append({'case': name, 'passed': True, 'actual_docker_dispatches': len(dispatched),
                           'is_synthetic_model_evidence': True})
        broken = [{'role': 'assistant', 'content': '', 'tool_calls': [{
            'id': 'expected', 'function': 'workspace_python', 'arguments': {'code': 'print(1)'}, 'type': 'function'}]},
            {'role': 'tool', 'content': 'synthetic result', 'tool_call_id': 'wrong', 'function': 'workspace_python'}]
        try:
            await run_protocol_fixture([response()], folder / 'mismatched_result_id', messages=broken,
                tools={'workspace_python': lambda code: actor.exec_python(code)}, tool_schemas=TOOL_SCHEMAS)
        except ValueError as exc:
            checks.append({'case': 'mismatched_result_id', 'passed': True, 'reason': str(exc),
                           'actual_docker_dispatches': 0, 'is_synthetic_model_evidence': True})
        else:
            raise AssertionError('mismatched native history accepted')
        equality = actor.verify_workspace(source)
        audit = actor.audit_boundary()
        assert equality['verified'] and audit['qualified']
    write(folder / 'boundary_audit.json', audit)
    write(OUT / 'gate-v2.json', {'gate': 'G0', 'status': 'PASS', 'controlled_cases': checks,
        'genuine_model_calls': qualification['model_calls'], 'genuine_native_tool_calls': qualification['native_tool_calls'],
        'genuine_trace_sha256': qualification['model_trace_sha256'], 'workspace_unchanged': equality,
        'scope': 'native API protocol and actual isolated dispatch; no OfficeVal outcome or G2/G3 claim'})
    write(OUT / 'accounting_erratum.json', {'supersedes_for_G0_release': 'gate.json',
        'historical_files_preserved': True, 'issue': 'missing-ID SDK conversion failure previously discarded synthetic wire usage',
        'fix': 'retain metered wire usage on conversion failure; unknown requests explicitly marked',
        'new_evidence': 'controlled_fixtures-v2', 'new_candidate_model_calls_before_fix': 0})
    print('G0 PASS: 7 controlled cases, 2 genuine model calls, 1 genuine native tool dispatch.')


if __name__ == '__main__':
    asyncio.run(main())
