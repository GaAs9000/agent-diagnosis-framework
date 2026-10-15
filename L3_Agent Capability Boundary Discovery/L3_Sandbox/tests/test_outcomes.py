import json
from l3_sandbox.outcomes import _writes, _usage


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def test_write_turns_require_tool_causal_chain(tmp_path):
    events = [{'event_id': 'p1', 'kind': 'model.parsed', 'links': {'response': 'r1'}},
              {'event_id': 'p2', 'kind': 'model.parsed', 'links': {'response': 'r2'}}]
    for i in (1, 2):
        events += [{'event_id': f'c{i}', 'kind': 'tool.call', 'links': {'response': f'r{i}'}},
                   {'event_id': f't{i}', 'kind': 'tool.result', 'links': {'call': f'c{i}'}},
                   {'event_id': f's{i}', 'kind': 'state.transition', 'links': {'tool_result': f't{i}'}}]
    states = [{'event_id': 'unlinked', 'changes': [{'path': 'submission/fake.docx', 'change': 'created', 'after': {'size_bytes': 3}}]}]
    for i, path in [(1, 'script.py'), (2, 'submission/result.docx')]:
        states.append({'event_id': f's{i}', 'changes': [{'path': path, 'change': 'created', 'after': {'size_bytes': 2}}]})
    (tmp_path / 'state_changes.jsonl').write_text('\n'.join(json.dumps(s) for s in states))
    assert _writes(tmp_path, events) == (1, 2)


def test_usage_counts_attempts_without_final_copy_duplication(tmp_path):
    request = tmp_path / 'raw/model/r000001'
    for sub, tokens in [('attempt_01', 7), ('attempt_02', None), ('', None)]:
        write(request / sub / 'http.json', {'status_code': 200 if tokens else 503})
        write(request / sub / 'response.bin', {'usageMetadata': {'totalTokenCount': tokens}} if tokens else {})
    receipts, total, unknown = _usage(tmp_path)
    assert len(receipts) == 2 and total == 7 and unknown == 1


def test_summary_preserves_delivery_quality_and_trial_identity(tmp_path, monkeypatch):
    import l3_sandbox.outcomes as module
    write(tmp_path / 'manifest.json', {'task': {'task_id': 'officeval_003', 'benchmark': 'officeval'},
                                      'config_sha256': 'sha', 'status': {'evaluation_validity': 'invalid'}})
    write(tmp_path / 'official_grade.json', {'official_raw': {'status': 'error', 'dim1_pass': False}})
    (tmp_path / 'artifacts/submission').mkdir(parents=True)
    (tmp_path / 'artifacts/submission/broken.docx').write_text('broken')
    monkeypatch.setattr(module, 'study_outcome', lambda *a, **kw: {'study_valid': True, 'study_outcome_y': 0.0})
    row = module.summarize_episode(tmp_path, {'task_id': 'officeval_003', 'config_sha256': 'sha'}, 1)
    assert row['delivery_exists'] is True and row['legal_delivery'] is False
    assert row['actual_cost'] is None and row['evaluation_validity'] == 'invalid'
    assert row['first_submission_write_turn'] is None
    row = module.summarize_episode(tmp_path, {'task_id': 'wrong', 'config_sha256': 'sha'}, 1)
    assert row['study_valid'] is False and row['study_outcome_y'] is None


def test_summary_counts_are_manifest_observations(tmp_path, monkeypatch):
    import l3_sandbox.outcomes as module
    monkeypatch.setattr(module, 'study_outcome', lambda *a, **kw: {'study_valid': False})
    for counts, expected in [({'model_requests': 29, 'tool_calls': 32}, (29, 32)),
                             ({'model_requests': 0, 'tool_calls': 0}, (0, 0)),
                             ({}, (None, None)), (None, (None, None)),
                             ({'model_requests': float('nan'), 'tool_calls': float('inf')}, (None, None)),
                             ({'model_requests': True, 'tool_calls': -1}, (None, None)),
                             ({'model_requests': '29', 'tool_calls': 3.0}, (None, None))]:
        write(tmp_path / 'manifest.json', {'counts': counts})
        row = module.summarize_episode(tmp_path, {}, 1, verify_evidence=False)
        assert (row['model_requests'], row['tool_calls']) == expected
