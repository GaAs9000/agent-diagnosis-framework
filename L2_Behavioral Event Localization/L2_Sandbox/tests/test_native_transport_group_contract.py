"""Real loopback HTTP and report-to-summary grouping regression contracts (L2 v0.4)."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread

import pytest

from conftest import make_l1_episode
from l2_sandbox import report
from l2_sandbox.analysts.llm import _default_llm_transport
from l2_sandbox.config import load, validate
from l2_sandbox.io import L2Error
from l2_sandbox.score import summarize_runs


@contextmanager
def local_model():
    requests = []
    response = {'candidates': [{'content': {'role': 'model', 'parts': [{'text': '{"origins": []}'}]}, 'finishReason': 'STOP'}]}
    state = {'body': json.dumps(response).encode()}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(state['body'])))
            self.end_headers()
            self.wfile.write(state['body'])

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    cfg = load()['model']
    cfg.update(kind='local', base_url=f'http://127.0.0.1:{server.server_port}', api_key_env=None)
    try:
        yield cfg, requests, state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_loaded_default_reaches_native_http_and_rejects_truncation():
    with local_model() as (cfg, requests, state):
        assert cfg['max_response_bytes'] == 16777216
        assert _default_llm_transport(cfg, 'system', {'trajectory': []}) == {'origins': []}
        assert len(requests) == 1
        assert requests[0]['systemInstruction']['parts'][0]['text'] == 'system'
        cap = len(state['body']) + 1
        state['body'] += b' ' * 20
        with pytest.raises(L2Error, match='exceeded model.max_response_bytes'):
            _default_llm_transport({**cfg, 'max_response_bytes': cap}, 'system', {})
        assert len(requests) == 2


@pytest.mark.parametrize('maximum', [0, -1, True, False, 1.5, '1024', None, float('inf'), float('nan'), 67108865, 10**100])
def test_bad_response_limit_rejected_before_http(maximum):
    with local_model() as (cfg, requests, _):
        bad = {**cfg, 'max_response_bytes': maximum}
        with pytest.raises(L2Error, match='model.max_response_bytes'):
            validate({'model': bad})
        with pytest.raises(L2Error, match='model.max_response_bytes'):
            _default_llm_transport(bad, 'system', {})
        assert requests == []


def test_diagnose_groups_reach_real_scores_and_summary(tmp_path, monkeypatch):
    def fake_grades(pkg, **kwargs):
        return [
            {
                'kind': 'terminal_submission',
                'episode_id': pkg.episode_id,
                'task_id': pkg.family_id,
                'k': pkg.turns,
                'snapshot_id': f's{pkg.turns:06d}',
                'candidate_status': 'submission',
                'candidate': ['submission/out.docx'],
                'in_submission': True,
                'score': 0.0,
                'dim1_pass': False,
                'total_score': 0,
                'max_score': 10.0,
                'deductions': {'dim1_gate': 1.0},
            },
            *[
                {
                    'kind': 'checkpoint',
                    'episode_id': pkg.episode_id,
                    'task_id': pkg.family_id,
                    'k': k,
                    'checkpoint_id': f'cp{k:06d}',
                    'snapshot_id': f's{k:06d}',
                    'candidate_status': 'ok',
                    'candidate': ['submission/out.docx'],
                    'in_submission': True,
                    'last_written_turn': 1,
                    'score': 1.0 if k >= 2 else 0.0,
                    'dim1_pass': k >= 2,
                    'total_score': 10 if k >= 2 else 0,
                    'max_score': 10.0,
                    'deductions': {'dim1_gate': 0.0 if k >= 2 else 1.0},
                }
                for k in range(pkg.turns + 1)
            ],
        ]

    monkeypatch.setattr(report, 'grade_episode_snapshots', fake_grades)

    def caller(system, payload):
        is_c = 'facts' in payload
        return {
            'origins': [{
                'defect': 'dim1_gate',
                'kind': 'undelivered' if is_c else 'never',
                'turn': 2 if is_c else None,
                'rationale': 'test origin',
            }],
            'hypotheses': [{
                'defect': 'dim1_gate',
                'turn': 2,
                'predicate': {'fact': 'tool_requested', 'where': {'arguments.command': 'echo turn 2'}},
                'effect': -0.5,
                'rationale': 'Synthetic command partition',
            }],
        }

    outputs, families = [], []
    for i, task in enumerate(('officeval_003', 'officeval_033')):
        parent = make_l1_episode(tmp_path / f'parent{i}', num_turns=4, task_id=task, turn_actions={1: 'modify_submission'})
        output = tmp_path / f'out{i}'
        result = report.diagnose_episode(
            parent,
            output_dir=output,
            analysts=('llm', 'llm_raw'),
            execute=True,
            llm_caller=caller,
        )
        groups = {
            a['analyst_group']: a
            for a in result['scores']['analysts'].values()
            if a['analyst'].startswith('llm:')
        }
        assert set(groups) == {'B', 'C'}
        assert groups['C']['origin_localization']['kind_accuracy_unweighted'] == 1.0
        assert groups['B']['origin_localization']['kind_accuracy_unweighted'] == 0.0
        outputs.append(output)
        families.append(result['family_id'])

    summary = summarize_runs(outputs, heldout_families=families)
    assert 'C_minus_B' in summary['paired_comparisons']
    assert summary['paired_comparisons']['C_minus_B']['kind_accuracy_unweighted_diff']['estimate'] == 1.0


def test_opencode_adapters_in_default_llm_transport():
    with local_model() as (cfg, requests, state):
        state['body'] = json.dumps({
            'choices': [{'message': {'role': 'assistant', 'content': '{"origins": []}'}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 5, 'completion_tokens': 5},
        }).encode()
        chat_cfg = {**cfg, 'adapter': 'opencode-chat', 'model': 'kimi-k2.6', 'parameters': {'temperature': 0.0, 'max_tokens': 512}}
        assert _default_llm_transport(chat_cfg, 'sys', {'trajectory': []}) == {'origins': []}
        assert 'tools' not in requests[-1]

        state['body'] = json.dumps({
            'status': 'completed',
            'output': [{'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': '{"origins": []}'}]}],
            'usage': {'input_tokens': 4, 'output_tokens': 6},
        }).encode()
        resp_cfg = {**cfg, 'adapter': 'openai-responses', 'model': 'gpt-5.5', 'parameters': {'temperature': 0.0, 'max_output_tokens': 512}}
        assert _default_llm_transport(resp_cfg, 'sys', {'trajectory': []}) == {'origins': []}
        assert requests[-1]['instructions'] == 'sys'

        state['body'] = json.dumps({
            'role': 'assistant',
            'stop_reason': 'end_turn',
            'content': [{'type': 'text', 'text': '{"origins": []}'}],
            'usage': {'input_tokens': 8, 'output_tokens': 4},
        }).encode()
        anth_cfg = {**cfg, 'adapter': 'anthropic-messages', 'model': 'claude-sonnet-4-6', 'parameters': {'temperature': 0.0, 'max_tokens': 512}}
        assert _default_llm_transport(anth_cfg, 'sys', {'trajectory': []}) == {'origins': []}
        assert requests[-1]['system'] == 'sys'
        assert requests[-1]['max_tokens'] == 512
