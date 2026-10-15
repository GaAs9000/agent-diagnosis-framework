"""Real loopback HTTP and report-to-heldout grouping regression contracts."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread

import pytest

from conftest import make_l1_episode
from l2_sandbox import report
from l2_sandbox.analysts.llm import _default_llm_transport
from l2_sandbox.config import load, validate
from l2_sandbox.io import L2Error, digest, read_json, write_json, write_jsonl
from l2_sandbox.score import summarize_runs


@contextmanager
def local_model():
    requests = []
    response = {'candidates': [{'content': {'role': 'model', 'parts': [{'text': '{"values": [0.5, 0.5]}'}]}, 'finishReason': 'STOP'}]}
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
        assert _default_llm_transport(cfg, 'system', {'trajectory': []}) == {'values': [.5, .5]}
        assert len(requests) == 1
        assert requests[0]['systemInstruction']['parts'][0]['text'] == 'system'
        # A valid JSON prefix plus whitespace would parse successfully if the
        # transport silently accepted a bounded but truncated response.
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


def controlled_curve(parent, output, plan, **kwargs):
    """Offline branch evidence only; real report/scoring/summary paths remain active."""
    rows = []
    for i in range(8):
        name = f'branch-{i}'
        success = i >= 4
        make_l1_episode(output/'branches'/name, num_turns=4, resume_from_k=1,
                        parent_pkg=parent, turn_actions={2: 'readback' if success else 'normal'},
                        terminal_case='scored', completion_rate=float(success), passed=success)
        rows.append({'trial': name, 'stage': 's2', 'k': 1, 'valid': True, 'y': float(success)})
    write_jsonl(output/'outcomes.jsonl', rows)
    curve = {'verdict': 'concentrated_turn', 'turn': 2, 'v_hat': {'1': 1., '2': .5},
             'turn_drops': {'2': .5}}
    write_json(output/'curve.json', curve)
    return curve


def test_report_groups_reach_real_scores_and_heldout_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(report, 'execute_curve', controlled_curve)
    calls = []

    def caller(system, payload):
        calls.append(payload)
        # The response claims C even for raw mode. Membership must follow the
        # configured input mode rather than anything in the model output.
        return {'values': [.5]*4, 'analyst_group': 'C', 'hypotheses': [{
            'turn': 2, 'predicate': {'fact': 'tool_requested', 'where': {'arguments.command': 'echo turn 2'}},
            'effect': -.5, 'rationale': 'Synthetic command partition'}]}

    outputs, families = [], []
    for i, task in enumerate(('officeval_003', 'officeval_033')):
        parent = make_l1_episode(tmp_path/f'parent{i}', num_turns=4, task_id=task)
        output = tmp_path/f'out{i}'
        result = report.run_episode(parent, output_dir=output, analysts=('llm', 'llm_raw'),
                                    p0=1., execute=True, llm_caller=caller)
        groups = {a['analyst_group']: a for a in result['scores']['analysts'].values() if a['analyst'].startswith('llm:')}
        assert set(groups) == {'B', 'C'}
        assert groups['C']['hypothesis_testing']['explanation_covered'] is True
        assert result['scores']['analysts']['rules@1']['hypothesis_testing']['explanation_covered'] is False
        outputs.append(output)
        families.append(result['plan']['family_id'])

    assert len(set(families)) == 2
    summary = summarize_runs(outputs, heldout_families=families)
    assert summary['recommended_default_mode'] == 'hybrid'
    comparisons = summary['paired_holdout_comparisons']
    assert sorted(v['paired_episodes'] for v in comparisons.values()) == [0, 2]
    original_calls = len(calls)
    original_bindings = {}
    for output in outputs:
        for path in (output/'values').glob('llm*.json'):
            doc = read_json(path)
            original_bindings[path] = doc['input_sha256']
            # Legacy C lacks metadata; spoofed raw cache claims C.
            if doc['analyst_group'] == 'C':
                doc.pop('analyst_group')
            else:
                doc['analyst_group'] = 'C'
            write_json(path, doc)

    def never(*args):
        pytest.fail('trusted metadata migration must reuse cached calls')

    for i, output in enumerate(outputs):
        report.run_episode(tmp_path/f'parent{i}', output_dir=output, analysts=('llm', 'llm_raw'),
                           p0=1., execute=True, cache_only=True, llm_caller=never)
    assert len(calls) == original_calls
    for path, inputs in original_bindings.items():
        assert read_json(path)['input_sha256'] == inputs
    summary = summarize_runs(outputs, heldout_families=families)
    assert summary['recommended_default_mode'] == 'hybrid'
    assert sorted(v['paired_episodes'] for v in summary['paired_holdout_comparisons'].values()) == [0, 2]


def test_pre_response_limit_legacy_tag_and_group_migrate_without_recall(tmp_path, monkeypatch):
    from test_v03_analyst_regressions import fake_curve
    monkeypatch.setattr(report, 'execute_curve', fake_curve)
    parent = make_l1_episode(tmp_path/'parent', num_turns=2)
    output = tmp_path/'out'
    report.run_episode(parent, output_dir=output, analysts=('llm',), p0=.5, execute=True,
                       llm_caller=lambda *args: {'values': [.4, .3]})
    current = next((output/'values').glob('llm*.json'))
    original = read_json(current)
    cfg = load()
    cfg['model'].pop('max_response_bytes')
    tag_hash = digest({'c1': report.CALL1_SYSTEM_PROMPT, 'c2': report.CALL2_SYSTEM_PROMPT,
                       'mode': 'facts', 'model_config': cfg['model']})[:16]
    old_id = f"llm:{cfg['model']['model']}@{tag_hash}"
    spec = {'kind': 'llm', 'id': old_id, 'config': cfg, 'include_facts': True}
    doc = {**original, 'analyst': old_id, 'input_sha256': {
        **original['input_sha256'], 'analyst_config_sha256': digest(spec)}}
    doc['input_sha256'].pop('model_sha256')
    doc.pop('analyst_group')
    write_json(output/'values'/report._safe_filename(old_id), doc)
    current.unlink()

    def never(*args):
        pytest.fail('pre-default legacy metadata must not require a paid recall')

    result = report.run_episode(parent, output_dir=output, analysts=('llm',), p0=.5,
                                execute=True, cache_only=True, llm_caller=never)
    migrated = read_json(current)
    assert migrated['input_sha256'] == original['input_sha256']
    assert migrated['analyst_group'] == 'C'
    assert result['scores']['analysts'][migrated['analyst']]['analyst_group'] == 'C'
