"""v0.2.1 test suite: two-stage value curves, calibration, integrity, 5-case outcomes, first-request identity, and E1+ strata."""
from __future__ import annotations
import copy
import random
import uuid
from pathlib import Path
import pytest

from conftest import docx_bytes
from l2_sandbox.config import load
from l2_sandbox.evidence.l1 import L1Reader, diff_snapshots
from l2_sandbox.evaluation.metrics import (
    curve_gold,
    evaluate_value_predictions_across_cases,
    localization_metrics,
    value_prediction_scores,
)
from l2_sandbox.experiments.controller import (
    check_first_request_identity,
    execute_curve,
    outcome,
    report_curve,
)
from l2_sandbox.experiments.plans import compile_curve_plan
from l2_sandbox.facts.build import extract_turn_tags
from l2_sandbox.io import (
    canonical,
    digest,
    inventory,
    now,
    read_json,
    read_jsonl,
    sha_bytes,
    write_json,
    write_jsonl,
    write_new,
)
from l2_sandbox.package import seal, verify
from l2_sandbox.pipeline import analyze
from l2_sandbox.stats.curve import evaluate_curve
from l2_sandbox.stats.strata import evaluate_strata, mantel_haenszel_pool


def make_curve_episode(
    root: Path,
    *,
    num_turns: int = 8,
    temperature: float = 1.0,
    recording_mode: str = 'live',
    benchmark: str = 'officeval',
    task_id: str = 'officeval_003',
    is_docx: bool = True,
    ineligible_cp: str | None = None,
    lineage: dict | None = None,
    resume_from_turn: int = 0,
    parent_conversations: dict[int, dict] | None = None,
    alter_first_request_byte: bool = False,
    first_turn_behavior: str = 'normal',
    terminal_case: str = 'failed_zero',
    model_version: str = 'gemini-2.5-flash-lite-preview-06-17',
) -> Path:
    """Constructs a self-consistent multi-turn v0.2.0 L1 episode for value-curve and strata testing."""
    root.mkdir(parents=True, exist_ok=False)
    eid = 'ep_' + uuid.uuid4().hex
    source_config = {
        'budget': {'max_model_requests': 35, 'max_tool_calls': 50, 'wall_time_seconds': 3600, 'max_total_tokens': None},
        'model': {
            'adapter': 'gemini',
            'kind': 'api',
            'model': 'SYNTHETIC_TEST_MODEL',
            'base_url': 'https://generativelanguage.googleapis.com/v1beta',
            'timeout_seconds': 300,
            'parameters': {'temperature': temperature, 'maxOutputTokens': 8192},
        },
        'agent': {'system_prompt': 'Fixture only'},
        'runtime': {'adapter': 'test-only'},
    }
    write_json(root, 'raw/resolved_config.json', source_config)
    ext = '.docx' if is_docx else '.xlsx'
    public = {
        'benchmark': benchmark,
        'task_id': task_id,
        'prompt': f'Process input/doc{ext} and write submission/out{ext}.',
        'assets': [{'name': f'doc{ext}'}],
    }
    write_json(root, 'benchmark/public_task.json', public)

    old_bytes = docx_bytes(empty_row=True, black=False) if is_docx else b'XLSX_INITIAL_BYTES'
    new_bytes = docx_bytes(empty_row=False, black=True) if is_docx else b'XLSX_MODIFIED_BYTES'

    def sf(path: str, data: bytes) -> dict:
        sha = sha_bytes(data)
        blob = 'states/blobs/' + sha
        if not (root / blob).exists():
            write_new(root, blob, data)
        return {'path': path, 'blob': blob, 'sha256': sha, 'size_bytes': len(data), 'mode': 0o644}

    dirs = [{'path': 'input', 'mode': 0o755}, {'path': 'submission', 'mode': 0o755}]
    snaps = []

    def make_snap(idx: int, files: list[dict]) -> dict:
        s = {
            'snapshot_id': f's{idx:06d}',
            'label': 'fixture',
            'scope': 'workspace_files_and_directories',
            'files': files,
            'directories': dirs,
            'total_bytes': sum(f['size_bytes'] for f in files),
            'content_sha256': digest({'files': files, 'directories': dirs}),
        }
        write_json(root, f"states/snapshots/{s['snapshot_id']}.json", s)
        snaps.append(s)
        return s

    def ref(path: str) -> dict:
        b = (root / path).read_bytes()
        return {'path': path, 'sha256': sha_bytes(b), 'size_bytes': len(b)}

    events = []

    def ev(kind: str, data: dict, links: dict | None = None) -> dict:
        x = {
            'schema_version': '0.2.0',
            'episode_id': eid,
            'event_id': f'{eid}:e{len(events):06d}',
            'seq': len(events),
            'kind': kind,
            'data': data,
            'links': links or {},
            'observed_at': now(),
            'previous_sha256': events[-1]['sha256'] if events else None,
        }
        x['sha256'] = digest(x)
        events.append(x)
        return x

    ev('episode.started', {'config': ref('raw/resolved_config.json')})
    if lineage:
        ev('episode.restored', lineage)

    cur_files = [sf(f'input/doc{ext}', old_bytes), sf(f'submission/out{ext}', old_bytes)]
    s0 = make_snap(0, cur_files)
    ev('state.snapshot', {'snapshot_id': s0['snapshot_id'], 'snapshot': ref(f"states/snapshots/{s0['snapshot_id']}.json")})

    if parent_conversations and resume_from_turn in parent_conversations:
        conversation = copy.deepcopy(parent_conversations[resume_from_turn])
    else:
        conversation = {
            'systemInstruction': {'parts': [{'text': 'Fixture only'}]},
            'contents': [{'role': 'user', 'parts': [{'text': public['prompt']}]}],
        }

    cp_index_rows = []
    state_changes = []
    saved_conversations: dict[int, dict] = {}
    tool_calls_count = 0
    tool_results_count = 0
    obs_delivered_count = 0
    model_req_count = 0
    model_resp_count = 0
    prev_call_info = None

    turns_to_run = range(resume_from_turn, num_turns)
    for k in turns_to_run:
        cp_id = f'cp{k:06d}'
        saved_conversations[k] = copy.deepcopy(conversation)
        is_elig = (cp_id != ineligible_cp)
        cp_obj = {
            'schema_version': '0.2.0',
            'episode_id': eid,
            'checkpoint_id': cp_id,
            'snapshot_id': snaps[-1]['snapshot_id'],
            'conversation': copy.deepcopy(conversation),
            'pending_calls': [],
            'pending_observations': [],
            'budget_used': {
                'model_requests': k,
                'tool_calls': k,
                'agent_seconds': k,
                'known_total_tokens': k * 10,
            },
            'qualification': {
                'eligible': is_elig,
                'image_id': 'sha256:' + 'a' * 64,
                'scopes': ['workspace_files', 'native_conversation'],
            },
            'config_sha256': digest(source_config),
        }
        write_json(root, f'checkpoints/{cp_id}.json', cp_obj)
        ev('checkpoint.created', {'checkpoint': ref(f'checkpoints/{cp_id}.json')})
        cp_index_rows.append({'checkpoint_id': cp_id, 'path': f'checkpoints/{cp_id}.json', 'eligible': is_elig})

        is_first_branch_turn = (k == resume_from_turn)
        req_body = copy.deepcopy(conversation)
        if is_first_branch_turn and alter_first_request_byte:
            req_body['tampered_byte'] = 'X'

        req_rel = f'raw/model/r{k:06d}/request.json'
        write_json(root, req_rel, req_body)
        rid = f'{eid}:r{k:06d}'
        req_ev = ev('model.request', {'request_id': rid, 'body': ref(req_rel)})
        model_req_count += 1

        if prev_call_info is not None:
            obs_path = f'observations/tool_{k - 1:06d}.json'
            write_json(root, obs_path, prev_call_info['obs'])
            ev(
                'observation.delivered',
                {
                    'call_id': prev_call_info['cid'],
                    'request_id': rid,
                    'native_id': prev_call_info['native_id'],
                    'observation': ref(obs_path),
                },
                {'request': req_ev['event_id'], 'result': prev_call_info['result_event_id']},
            )
            obs_delivered_count += 1
            prev_call_info = None

        behavior = first_turn_behavior if is_first_branch_turn else ('finish' if k == num_turns - 1 else 'normal')
        if behavior == 'finish':
            resp_body = {
                'candidates': [{'content': {'role': 'model', 'parts': [{'text': 'Task complete.'}]}, 'finishReason': 'STOP'}],
                'modelVersion': model_version,
            }
            resp_rel = f'raw/model/r{k:06d}/response.bin'
            http_rel = f'raw/model/r{k:06d}/http.json'
            write_json(root, resp_rel, resp_body)
            write_json(root, http_rel, {'status_code': 200, 'fixture': True})
            ev(
                'model.response',
                {'request_id': rid, 'body': ref(resp_rel), 'http': ref(http_rel), 'status_code': 200, 'model_version': model_version},
                {'request': req_ev['event_id']},
            )
            model_resp_count += 1
            ev('model.parsed', {'request_id': rid, 'text': 'Task complete.', 'finish_reason': 'STOP'})
            conversation['contents'].append(resp_body['candidates'][0]['content'])
            break

        cmd = (
            f'ls -la submission/out{ext}' if behavior == 'readback' else
            f'cp input/doc{ext} submission/out{ext}' if behavior in ('modify_submission', 'modify_docx_structure') else
            f'echo step {k}'
        )
        native_id = f'nat_{k:04d}'
        cid = f'{rid}:c0'
        call_Spec = {'name': 'run', 'args': {'command': cmd}, 'id': native_id}
        resp_body = {
            'candidates': [{'content': {'role': 'model', 'parts': [{'functionCall': call_Spec}]}, 'finishReason': 'STOP'}],
            'modelVersion': model_version,
        }
        resp_rel = f'raw/model/r{k:06d}/response.bin'
        http_rel = f'raw/model/r{k:06d}/http.json'
        write_json(root, resp_rel, resp_body)
        write_json(root, http_rel, {'status_code': 200, 'fixture': True})
        resp_ev = ev(
            'model.response',
            {'request_id': rid, 'body': ref(resp_rel), 'http': ref(http_rel), 'status_code': 200, 'model_version': model_version},
            {'request': req_ev['event_id']},
        )
        model_resp_count += 1
        ev('model.parsed', {'request_id': rid})
        tc_ev = ev(
            'tool.call',
            {'request_id': rid, 'call_id': cid, 'native_id': native_id, 'name': 'run', 'arguments': {'command': cmd}},
            {'response': resp_ev['event_id']},
        )
        tool_calls_count += 1

        exit_code = 1 if (behavior == 'anomaly' or (not lineage and k == 2)) else 0
        obs = {
            'exit_code': exit_code,
            'stdout': f'out {k}\n',
            'stderr': 'error\n' if exit_code else '',
            'output_complete': True,
            'stop_reason': None,
        }
        res_rel = f'raw/tools/r{k:06d}_result.json'
        out_rel = f'raw/tools/r{k:06d}_stdout.bin'
        err_rel = f'raw/tools/r{k:06d}_stderr.bin'
        write_json(root, res_rel, {'call_id': cid, 'observation': obs, 'provenance': {'test_fixture': True}})
        write_new(root, out_rel, obs['stdout'].encode())
        write_new(root, err_rel, obs['stderr'].encode())
        res_ev = ev(
            'tool.result',
            {'call_id': cid, 'result': ref(res_rel), 'stdout': ref(out_rel), 'stderr': ref(err_rel)},
            {'call': tc_ev['event_id']},
        )
        tool_results_count += 1

        before_snap = snaps[-1]
        if behavior in ('modify_submission', 'modify_docx_structure') or (not lineage and k == 4):
            next_files = [sf(f'input/doc{ext}', old_bytes), sf(f'submission/out{ext}', new_bytes)]
        elif terminal_case == 'empty_submission' and k == num_turns - 1:
            next_files = [sf(f'input/doc{ext}', old_bytes)]
        else:
            next_files = list(before_snap['files'])
        after_snap = make_snap(len(snaps), next_files)
        ev('state.snapshot', {'snapshot_id': after_snap['snapshot_id'], 'snapshot': ref(f"states/snapshots/{after_snap['snapshot_id']}.json")}, {'tool_result': res_ev['event_id']})
        tr_ev = ev('state.transition', {'before': before_snap['snapshot_id'], 'after': after_snap['snapshot_id']}, {'tool_result': res_ev['event_id']})
        state_changes.append({
            'episode_id': eid,
            'event_id': tr_ev['event_id'],
            'before_snapshot_id': before_snap['snapshot_id'],
            'after_snapshot_id': after_snap['snapshot_id'],
            'changes': diff_snapshots(before_snap, after_snap),
        })

        conversation['contents'] += [
            resp_body['candidates'][0]['content'],
            {'role': 'user', 'parts': [{'functionResponse': {'id': native_id, 'name': 'run', 'response': obs}}]},
        ]
        prev_call_info = {'cid': cid, 'native_id': native_id, 'obs': obs, 'result_event_id': res_ev['event_id']}

    write_jsonl(root, 'state_changes.jsonl', state_changes)
    write_json(root, 'checkpoints/index.json', {'checkpoints': cp_index_rows})

    final_snap = snaps[-1]
    sub_files = []
    for f in final_snap['files']:
        if f['path'].startswith('submission/'):
            rel_sub = f['path'][len('submission/'):]
            write_new(root, 'artifacts/submission/' + rel_sub, (root / f['blob']).read_bytes())
            sub_files.append({'path': rel_sub, 'sha256': f['sha256'], 'size_bytes': f['size_bytes']})
    write_json(root, 'artifacts/inventory.json', {'episode_id': eid, 'snapshot_id': final_snap['snapshot_id'], 'files': sub_files})

    exec_valid = 'valid'
    eval_valid = 'valid'
    if terminal_case == 'scored_success':
        rawgrade = {
            'status': 'ok', 'error': None, 'id': '003', 'dim1_pass': True,
            'dim2_items': [{'id': 'item_1', 'hit': True}, {'id': 'item_2', 'hit': True}],
            'total_score': 15, 'max_score': 15,
        }
        proj = {
            'evaluation_validity': 'valid', 'task_outcome': 'success',
            'completion_rate': 1.0, 'official_dimensions': {'dim1_pass': True, 'passed': True},
        }
    elif terminal_case == 'scored_negative_clipped':
        rawgrade = {'status': 'ok', 'error': None, 'id': '003', 'dim1_pass': True, 'total_score': -5, 'max_score': 15}
        proj = {
            'evaluation_validity': 'valid', 'task_outcome': 'failure',
            'completion_rate': 0.0, 'official_dimensions': {'dim1_pass': True, 'passed': False},
        }
    elif terminal_case == 'empty_submission':
        rawgrade = {'status': 'error', 'error': 'FileNotFoundError: No such file in submission', 'id': '003'}
        proj = {'evaluation_validity': 'invalid', 'task_outcome': 'unknown', 'completion_rate': None}
        eval_valid = 'invalid'
    elif terminal_case == 'unscorable_submission':
        rawgrade = {'status': 'parse_error', 'error': 'BadZipFile: corrupted docx archive', 'id': '003'}
        proj = {'evaluation_validity': 'invalid', 'task_outcome': 'unknown', 'completion_rate': None}
        eval_valid = 'invalid'
    elif terminal_case == 'scorer_crash':
        rawgrade = None
        proj = {'evaluation_validity': 'invalid', 'task_outcome': 'unknown'}
        eval_valid = 'invalid'
    elif terminal_case == 'transport_invalid':
        rawgrade = None
        proj = {'evaluation_validity': 'invalid', 'task_outcome': 'unknown'}
        exec_valid = 'invalid'
        eval_valid = 'invalid'
    else:  # 'failed_zero'
        rawgrade = {
            'status': 'ok', 'error': None, 'id': '003', 'dim1_pass': False,
            'dim2_items': [{'id': 'item_1', 'hit': False}, {'id': 'item_2', 'hit': False}],
            'total_score': 0, 'max_score': 15,
        }
        proj = {
            'evaluation_validity': 'valid', 'task_outcome': 'failure',
            'completion_rate': 0.0, 'official_dimensions': {'dim1_pass': False, 'passed': False},
        }

    write_json(root, 'official_grade.json', {'official_raw': rawgrade, 'projection': proj, 'status': 'error' if eval_valid != 'valid' else 'ok'})
    status = {
        'termination': 'model_transport_error' if exec_valid != 'valid' else 'assistant_finished',
        'execution_validity': exec_valid,
        'protocol_validity': 'valid',
        'evaluation_validity': eval_valid,
        'task_outcome': proj.get('task_outcome', 'unknown'),
        'evidence_completeness': 'complete',
    }
    ev('episode.finished', {'status': status})
    m = {
        'schema_version': '0.2.0',
        'episode_id': eid,
        'recording_mode': recording_mode,
        'created_at': now(),
        'finished_at': now(),
        'task': {'benchmark': benchmark, 'task_id': task_id, 'prompt': public['prompt']},
        'agent': {
            'identity': {'adapter': 'gemini', 'requested_model': 'SYNTHETIC_TEST_MODEL'},
            'model_configuration': {'adapter': 'gemini', 'model': 'SYNTHETIC_TEST_MODEL'},
        },
        'model': {'model_version': model_version},
        'coverage': [{'code': 'test_fixture', 'detail': 'synthetic curve fixture'}],
        'usage': {'all_requests_have_usage': True, 'known_total_tokens_including_parent': num_turns * 10},
        'status': status,
        'config_sha256': digest(source_config),
        'runtime': {'image_id': 'sha256:' + 'a' * 64},
        'capabilities': {'test_fixture': True},
        'counts': {
            'events': len(events),
            'model_requests': model_req_count,
            'model_responses': model_resp_count,
            'tool_calls': tool_calls_count,
            'tool_results': tool_results_count,
            'observations_delivered': obs_delivered_count,
            'snapshots': len(snaps),
        },
        'lineage': lineage,
    }
    write_json(root, 'manifest.json', m)
    write_jsonl(root, 'trajectory.jsonl', events)
    write_json(root, 'evidence_index.json', {
        'schema_version': '0.2.0',
        'episode_id': eid,
        'files': inventory(root),
        'event_count': len(events),
        'trajectory_head_sha256': events[-1]['sha256'],
        'trust_scope': 'SYNTHETIC FIXTURE',
    })
    assert L1Reader(root).report['verified']
    return root


def _overwrite_and_reseal(pkg_dir: Path, filename: str, content, kind: str, *, is_jsonl: bool = False):
    target = pkg_dir / filename
    if target.exists():
        target.unlink()
    if is_jsonl:
        write_jsonl(pkg_dir, filename, content)
    else:
        write_json(pkg_dir, filename, content)
    idx = pkg_dir / 'evidence_index.json'
    if idx.exists():
        idx.unlink()
    seal(pkg_dir, kind)


def test_five_case_outcome_table_and_l1_projection(tmp_path):
    """Verifies Section 3.5 (5-case table) and Section 3.6 (L1 projection endpoints & negative score clipping)."""
    ep_ok = L1Reader(make_curve_episode(tmp_path / 'ok', num_turns=2, terminal_case='scored_success'))
    res_ok = outcome(ep_ok, {'name': 'official_completion_rate', 'source': 'l1_projection'})
    assert res_ok['endpoint_defined'] is True and res_ok['value'] == 1.0 and res_ok['outcome_case'] == 'scored'
    assert res_ok['dim2_hits'] == {'item_1': 1.0, 'item_2': 1.0}

    ep_neg = L1Reader(make_curve_episode(tmp_path / 'neg', num_turns=2, terminal_case='scored_negative_clipped'))
    res_neg = outcome(ep_neg, 'official_completion_rate')
    assert res_neg['endpoint_defined'] is True and res_neg['value'] == 0.0

    res_lhtb = outcome(ep_ok, 'passed')
    assert res_lhtb['endpoint_defined'] is True and res_lhtb['value'] == 1.0
    res_lhtb_fail = outcome(ep_neg, 'passed')
    assert res_lhtb_fail['endpoint_defined'] is True and res_lhtb_fail['value'] == 0.0

    ep_empty = L1Reader(make_curve_episode(tmp_path / 'empty', num_turns=2, terminal_case='empty_submission'))
    res_empty = outcome(ep_empty, 'official_completion_rate')
    assert res_empty['endpoint_defined'] is True
    assert res_empty['value'] == 0.0
    assert res_empty['outcome_case'] == 'empty_submission'
    res_empty_dim1 = outcome(ep_empty, {'name': 'official_dim1', 'source': 'l1_projection'})
    assert res_empty_dim1['endpoint_defined'] is True and res_empty_dim1['value'] is False

    ep_corrupt = L1Reader(make_curve_episode(tmp_path / 'corrupt', num_turns=2, terminal_case='unscorable_submission'))
    res_corrupt = outcome(ep_corrupt, 'official_completion_rate')
    assert res_corrupt['endpoint_defined'] is True
    assert res_corrupt['value'] == 0.0
    assert res_corrupt['sensitivity_value'] is None
    assert res_corrupt['outcome_case'] == 'unscorable_submission'

    ep_trans = L1Reader(make_curve_episode(tmp_path / 'trans', num_turns=2, terminal_case='transport_invalid'))
    res_trans = outcome(ep_trans, 'official_completion_rate')
    assert res_trans['endpoint_defined'] is False and res_trans['value'] is None
    assert res_trans['outcome_case'] == 'infrastructure_invalid'

    ep_crash = L1Reader(make_curve_episode(tmp_path / 'crash', num_turns=2, terminal_case='scorer_crash'))
    res_crash = outcome(ep_crash, 'official_completion_rate')
    assert res_crash['endpoint_defined'] is False and res_crash['value'] is None
    assert res_crash['outcome_case'] == 'scorer_infrastructure_invalid'


def test_first_request_byte_identity_check(tmp_path):
    """Verifies Section 3.1 & 3.5: branch's first request must match parent's request at the same index byte-for-byte."""
    parent_dir = make_curve_episode(tmp_path / 'parent', num_turns=4)
    parent = L1Reader(parent_dir)
    parent_convs = {k: parent.json(f'checkpoints/cp{k:06d}.json')['conversation'] for k in range(4)}

    lineage = {
        'parent_episode_id': parent.episode_id,
        'parent_index_sha256': parent.index_sha256,
        'parent_checkpoint_id': 'cp000002',
        'parent_checkpoint_sha256': parent.files['checkpoints/cp000002.json']['sha256'],
    }
    good_branch = L1Reader(
        make_curve_episode(
            tmp_path / 'branch_good',
            num_turns=4,
            resume_from_turn=2,
            parent_conversations=parent_convs,
            lineage=lineage,
        )
    )
    ok_check = check_first_request_identity(parent, good_branch, 'cp000002')
    assert ok_check['verified'] is True and ok_check['request_index'] == 2

    bad_branch = L1Reader(
        make_curve_episode(
            tmp_path / 'branch_bad',
            num_turns=4,
            resume_from_turn=2,
            parent_conversations=parent_convs,
            alter_first_request_byte=True,
            lineage=lineage,
        )
    )
    bad_check = check_first_request_identity(parent, bad_branch, 'cp000002')
    assert bad_check['verified'] is False and bad_check['reason'] == 'first_request_bytes_differ'


def test_turn_tags_and_non_docx_applicability(tmp_path):
    """Verifies Section 4: positive/negative turn tags, non-docx applicability, and finish vs readback contrast."""
    ep_docx = L1Reader(make_curve_episode(tmp_path / 'docx_ep', num_turns=6, is_docx=True))
    tags_docx = extract_turn_tags(ep_docx)
    assert len(tags_docx) == 6
    assert 'tool_execution_anomaly' in tags_docx[2]['tags']
    assert 'tool_execution_anomaly' not in tags_docx[0]['tags']
    assert 'submission_modified' in tags_docx[4]['tags']
    assert 'object_structure_changed' in tags_docx[4]['tags']
    assert tags_docx[4]['object_structure_changed_status'] == 'present'
    assert 'assistant_finished_turn' in tags_docx[5]['tags']

    ep_rb = L1Reader(make_curve_episode(tmp_path / 'rb_ep', num_turns=2, is_docx=True, first_turn_behavior='readback'))
    tags_rb = extract_turn_tags(ep_rb)
    assert 'submission_readback' in tags_rb[0]['tags']
    assert 'submission_modified' not in tags_rb[0]['tags']

    ep_xlsx = L1Reader(make_curve_episode(tmp_path / 'xlsx_ep', num_turns=4, is_docx=False))
    tags_xlsx = extract_turn_tags(ep_xlsx)
    assert all(t['object_structure_changed_status'] == 'not_applicable' for t in tags_xlsx)
    assert all('object_structure_changed' not in t['tags'] for t in tags_xlsx)

    branches = []
    for i in range(6):
        branches.append({
            'first_turn_tags': ['assistant_finished_turn'],
            'outcome': {'endpoint_defined': True, 'value': 0.0 if i < 5 else 1.0},
        })
    for i in range(6):
        branches.append({
            'first_turn_tags': ['submission_readback'],
            'outcome': {'endpoint_defined': True, 'value': 1.0 if i < 5 else 0.0},
        })
    strata_res = evaluate_strata(branches, min_per_class=4, is_docx_task=False, v_before=0.5, turn_drop=0.4)
    obj_row = next(r for r in strata_res['tags'] if r['tag'] == 'object_structure_changed')
    assert obj_row['status'] == 'not_applicable'
    contrast = strata_res['finish_vs_readback_contrast']
    assert contrast['status'] == 'evaluated'
    assert contrast['delta_finish_minus_readback'] < -0.4

    mh = mantel_haenszel_pool([
        {'family_id': 'officeval_003', 'n1': 8, 'n0': 8, 'mean1': 0.125, 'mean0': 0.75},
        {'family_id': 'officeval_033', 'n1': 10, 'n0': 10, 'mean1': 0.20, 'mean0': 0.80},
    ])
    assert mh['status'] == 'evaluated' and mh['family_count'] == 2 and mh['pooled_delta'] < -0.4


def test_statistical_calibration_ci():
    """Verifies Section 3.4 & 6.3 calibration targets with fixed seed:
    - Uniform loss (T=16, V0=0.70, Y=0.0, N1=12, N2=24): interval FP <= 7%, turn FP <= 2%
    - Single-turn concentrated drop (Delta=0.50, N1=12, N2=24): turn accuracy >= 0.65
    """
    rng = random.Random(20261014)
    T = 16
    grid_turns = [0, 4, 8, 12]
    N1, N2 = 12, 24
    n_trials = 400

    v_uniform = {k: 0.70 * (1.0 - k / float(T)) for k in range(T + 1)}
    interval_fp = 0
    turn_fp = 0
    for _ in range(n_trials):
        s1 = {
            k: [{'outcome': {'endpoint_defined': True, 'value': 1.0 if rng.random() < v_uniform[k] else 0.0}} for _ in range(N1)]
            for k in grid_turns
        }
        means = {k: sum(r['outcome']['value'] for r in s1[k]) / N1 for k in grid_turns}
        means[T] = 0.0
        full_g = grid_turns + [T]
        best_ab = max(zip(full_g[:-1], full_g[1:]), key=lambda ab: means[ab[0]] - means[ab[1]])
        a, b = best_ab
        s2 = {
            k: [{'outcome': {'endpoint_defined': True, 'value': 1.0 if rng.random() < v_uniform[k] else 0.0}} for _ in range(N2)]
            for k in range(a, b + 1) if k < T
        }
        res = evaluate_curve(grid_turns, s1, s2, T, 0.0)
        if res['curve_verdict'] in ('concentrated_interval', 'concentrated_turn'):
            interval_fp += 1
        if res['curve_verdict'] == 'concentrated_turn':
            turn_fp += 1

    turn_hits = 0
    for trial in range(n_trials):
        j_star = (trial % T) + 1
        other_drop = (0.70 - 0.50) / (T - 1)
        v_true = {0: 0.70}
        for k in range(1, T + 1):
            drop_k = 0.50 if k == j_star else other_drop
            v_true[k] = max(0.0, v_true[k - 1] - drop_k)
        s1 = {
            k: [{'outcome': {'endpoint_defined': True, 'value': 1.0 if rng.random() < v_true[k] else 0.0}} for _ in range(N1)]
            for k in grid_turns
        }
        means = {k: sum(r['outcome']['value'] for r in s1[k]) / N1 for k in grid_turns}
        means[T] = 0.0
        full_g = grid_turns + [T]
        a, b = max(zip(full_g[:-1], full_g[1:]), key=lambda ab: means[ab[0]] - means[ab[1]])
        s2 = {
            k: [{'outcome': {'endpoint_defined': True, 'value': 1.0 if rng.random() < v_true[k] else 0.0}} for _ in range(N2)]
            for k in range(a, b + 1) if k < T
        }
        res = evaluate_curve(grid_turns, s1, s2, T, 0.0)
        if res['curve_verdict'] == 'concentrated_turn' and res['selected_turn'] == j_star:
            turn_hits += 1

    print(f'Calibration alpha=0.05 seed=20261014 trials={n_trials}: interval_fp={interval_fp}/{n_trials}, turn_fp={turn_fp}/{n_trials}, turn_hits={turn_hits}/{n_trials}')
    assert interval_fp / n_trials <= 0.07, f'Uniform interval FP too high: {interval_fp / n_trials:.3f}'
    assert turn_fp / n_trials <= 0.02, f'Uniform turn FP too high: {turn_fp / n_trials:.3f}'
    assert turn_hits / n_trials >= 0.65, f'Concentrated turn power too low: {turn_hits / n_trials:.3f}'


def test_end_to_end_synthetic_value_curve_and_integrity_gates(tmp_path):
    """Verifies Section 6.3: synthetic L1 value curve end-to-end + negative integrity gates."""
    cfg = load()
    parent_dir = make_curve_episode(tmp_path / 'parent', num_turns=8, temperature=1.0, recording_mode='live')
    parent_reader = L1Reader(parent_dir)
    parent_convs = {k: parent_reader.json(f'checkpoints/cp{k:06d}.json')['conversation'] for k in range(8)}

    analysis_dir = tmp_path / 'analysis'
    analyze(parent_dir, analysis_dir, cfg)

    blocked_vp = compile_curve_plan(
        analysis_dir, parent_dir, tmp_path / 'plan_blocked_vp',
        stride=4, stage1_replicates=4, stage2_replicates=8, require_value_predictions=True,
    )
    assert blocked_vp['plan']['plan_state'] == 'blocked'
    assert 'value_predictions_missing' in blocked_vp['plan']['blockers']

    t0_dir = make_curve_episode(tmp_path / 'parent_t0', num_turns=8, temperature=0.0, recording_mode='live')
    a_t0 = tmp_path / 'analysis_t0'
    analyze(t0_dir, a_t0, cfg)
    blocked_t0 = compile_curve_plan(a_t0, t0_dir, tmp_path / 'plan_t0', stride=4, stage1_replicates=4, stage2_replicates=8)
    assert blocked_t0['plan']['plan_state'] == 'blocked'
    assert 'degenerate_policy_sampling' in blocked_t0['plan']['blockers']

    inelig_dir = make_curve_episode(
        tmp_path / 'parent_inelig', num_turns=8, temperature=1.0, recording_mode='live', ineligible_cp='cp000004'
    )
    a_in = tmp_path / 'analysis_in'
    analyze(inelig_dir, a_in, cfg)
    blocked_in = compile_curve_plan(a_in, inelig_dir, tmp_path / 'plan_in', stride=4, stage1_replicates=4, stage2_replicates=8)
    assert blocked_in['plan']['plan_state'] == 'blocked'
    assert 'grid_checkpoint_ineligible' in blocked_in['plan']['blockers']

    plan_dir = tmp_path / 'curve_plan'
    compiled = compile_curve_plan(
        analysis_dir, parent_dir, plan_dir,
        stride=4, stage1_replicates=6, stage2_replicates=12, max_concurrency=2,
    )
    assert compiled['plan']['plan_state'] == 'frozen'
    assert verify(plan_dir)['verified']

    true_v = {'cp000000': 1.0, 'cp000001': 1.0, 'cp000002': 1.0, 'cp000003': 0.0, 'cp000004': 0.0}

    def synthetic_runner(run_cfg, out_path, *, parent, checkpoint_id):
        k = int(checkpoint_id[2:])
        p_prob = true_v.get(checkpoint_id, 0.0)
        seed = int(digest({'out': str(out_path), 'cp': checkpoint_id})[:12], 16)
        r_rng = random.Random(seed)
        succ = r_rng.random() < p_prob
        lineage = {
            'parent_episode_id': parent_reader.episode_id,
            'parent_index_sha256': parent_reader.index_sha256,
            'parent_checkpoint_id': checkpoint_id,
            'parent_checkpoint_sha256': parent_reader.files[f'checkpoints/{checkpoint_id}.json']['sha256'],
        }
        make_curve_episode(
            out_path,
            num_turns=8,
            resume_from_turn=k,
            parent_conversations=parent_convs,
            lineage=lineage,
            first_turn_behavior='readback' if succ else 'anomaly',
            terminal_case='scored_success' if succ else 'failed_zero',
        )
        return {'fixture': True}

    exec_dir = tmp_path / 'curve_exec'
    exec_res = execute_curve(
        plan_dir, parent_dir, exec_dir, allow_execution=True, max_concurrency=2, _test_runner=synthetic_runner
    )
    assert exec_res['verification']['verified'], exec_res
    assert exec_res['results']['verdict'] == 'concentrated_turn'
    assert exec_res['results']['curve']['selected_interval'] == [0, 4]
    assert exec_res['results']['curve']['argmax_turn'] == 3
    assert exec_res['results']['evidence_basis'] == 'value_curve'
    assert exec_res['results']['strata']['evidence_basis'] == 'resample_stratified'

    report_dir = tmp_path / 'curve_report'
    rep_ver = report_curve(analysis_dir, exec_dir, report_dir)
    assert rep_ver['verified'], rep_ver
    gold = read_json(report_dir / 'curve_gold.json')
    assert gold['verdict'] == 'concentrated_turn' and gold['localized_turn'] == 3 and gold['accepted_event_sets']
    val_eval = read_json(report_dir / 'value_evaluation.json')
    assert val_eval['scores']['linear_decay']['brier_or_mse'] is not None

    loc_score = localization_metrics(
        [{'case_id': gold['case_id'], 'event_sets': gold['accepted_event_sets'][:1]}],
        [gold],
        k=3,
    )
    assert loc_score['hit_rate'] == 1.0

    val_eval_fam2 = copy.deepcopy(val_eval)
    val_eval_fam2['case_id'] = 'case_fam2'
    val_eval_fam2['family_id'] = 'officeval_033'
    cluster_res = evaluate_value_predictions_across_cases([val_eval, val_eval_fam2], n_boot=200)
    assert cluster_res['status'] == 'incomplete_analyst_predictions' and cluster_res['family_count'] == 2
    assert val_eval['scores']['analyst']['brier_or_mse'] is None
    assert cluster_res['skill_score_vs_linear_decay'] is None

    # Gate 4: Out-of-order branch_receipts.jsonl MUST STILL PASS verification
    receipts = read_jsonl(exec_dir / 'branch_receipts.jsonl')
    _overwrite_and_reseal(exec_dir, 'branch_receipts.jsonl', list(reversed(receipts)), 'value_curve_execution', is_jsonl=True)
    assert verify(exec_dir)['verified'], 'Reordered branch_receipts.jsonl should verify cleanly'

    # Gate 5: Tampering with stage2_plan.json so it disagrees with Stage 1 derivation MUST FAIL
    s2_orig = read_json(exec_dir / 'stage2_plan.json')
    s2_bad = copy.deepcopy(s2_orig)
    s2_bad['selected_interval'] = [4, 8]
    _overwrite_and_reseal(exec_dir, 'stage2_plan.json', s2_bad, 'value_curve_execution')
    assert not verify(exec_dir)['verified']
    _overwrite_and_reseal(exec_dir, 'stage2_plan.json', s2_orig, 'value_curve_execution')

    # Gate 6: Tampering with results.json verdict so it disagrees with stats recomputation MUST FAIL
    res_orig = read_json(exec_dir / 'results.json')
    res_bad = copy.deepcopy(res_orig)
    res_bad['verdict'] = 'diffuse_loss'
    _overwrite_and_reseal(exec_dir, 'results.json', res_bad, 'value_curve_execution')
    assert not verify(exec_dir)['verified']
    _overwrite_and_reseal(exec_dir, 'results.json', res_orig, 'value_curve_execution')

    # Gate 7: Tampering with a receipt outcome value MUST FAIL
    bad_receipts = copy.deepcopy(receipts)
    bad_receipts[0]['outcome']['value'] = 0.12345
    _overwrite_and_reseal(exec_dir, 'branch_receipts.jsonl', bad_receipts, 'value_curve_execution', is_jsonl=True)
    assert not verify(exec_dir)['verified']
