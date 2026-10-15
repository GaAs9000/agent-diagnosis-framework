"""Synthetic L1 v0.2 package fixtures for L2 v0.3 tests."""
from __future__ import annotations
import copy
import io
import uuid
import zipfile
from pathlib import Path
from l2_sandbox.evidence.l1 import L1Package, diff_snapshots
from l2_sandbox.io import digest, sha_bytes, write_json, write_jsonl


def docx_bytes(*, empty_row: bool = True, black: bool = False) -> bytes:
    row = '<w:tr><w:tc><w:p><w:r><w:t>text</w:t></w:r></w:p></w:tc></w:tr>'
    if empty_row:
        row += '<w:tr><w:tc><w:p/></w:tc></w:tr>'
    borders = '<w:tblBorders><w:bottom w:val="single" w:color="000000" w:sz="8"/></w:tblBorders>' if black else ''
    xml = (
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f'<w:body><w:tbl><w:tblPr>{borders}</w:tblPr>{row}</w:tbl>'
        '<w:p><w:r><w:t>content</w:t></w:r></w:p></w:body></w:document>'
    )
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('word/document.xml', xml)
    return out.getvalue()


def make_l1_episode(
    root: Path,
    *,
    num_turns: int = 8,
    resume_from_k: int = 0,
    benchmark: str = 'officeval',
    task_id: str = 'officeval_003',
    is_docx: bool = True,
    turn_actions: dict[int, str] | None = None,
    terminal_case: str = 'failed_zero',
    completion_rate: float = 0.0,
    passed: bool = False,
    model_version: str = 'gemini-2.5-flash-lite-preview-06-17',
    parent_pkg=None,
    model_parameters: dict | None = None,
) -> Path:
    """Creates a minimal, self-consistent L1 episode directory (parent or resumed branch).

    `turn_actions` maps 1-based global turn index `t` to one of:
      - `'normal'`: exit_code=0 shell command without touching submission/
      - `'anomaly'`: exit_code=1 shell command (`tool_execution_anomaly`)
      - `'truncated_output'`: exit_code=0 but `output_complete=False` (`tool_exit.ok == False`)
      - `'readback'`: `ls submission/...` without modifying submission/ (`submission_readback`)
      - `'modify_submission'`: modifies `submission/out.*` (and changes `.docx` structure if `is_docx=True`)
      - `'finish'`: no tool call, finishes turn (`assistant_finished_turn`)
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    eid = 'ep_' + uuid.uuid4().hex[:12]
    turn_actions = turn_actions or {}
    if parent_pkg is not None:
        benchmark, task_id = parent_pkg.benchmark, parent_pkg.task_id

    if benchmark == 'lhtb' and task_id == 'officeval_003':
        task_id = 'sokoban'
    ext = '.docx' if is_docx else '.xlsx'
    old_data = docx_bytes(empty_row=True, black=False) if is_docx else b'XLSX_OLD'
    new_data = docx_bytes(empty_row=False, black=True) if is_docx else b'XLSX_NEW'

    def write_blob(rel_path: str, raw: bytes) -> dict:
        sha = sha_bytes(raw)
        blob = f'states/blobs/{sha}'
        blob_file = root / blob
        if not blob_file.exists():
            blob_file.parent.mkdir(parents=True, exist_ok=True)
            blob_file.write_bytes(raw)
        return {'path': rel_path, 'blob': blob, 'sha256': sha, 'size_bytes': len(raw), 'mode': 0o644}

    dirs = [{'path': 'input', 'mode': 0o755}, {'path': 'submission', 'mode': 0o755}]
    snaps: list[dict] = []

    def add_snap(files: list[dict]) -> dict:
        sid = f's{len(snaps):06d}'
        s = {
            'snapshot_id': sid,
            'files': files,
            'directories': dirs,
            'total_bytes': sum(f['size_bytes'] for f in files),
            'content_sha256': digest({'files': files, 'directories': dirs}),
        }
        write_json(root / f'states/snapshots/{sid}.json', s)
        snaps.append(s)
        return s

    events: list[dict] = []

    def emit(kind: str, data: dict) -> dict:
        ev = {
            'schema_version': '0.2.0',
            'episode_id': eid,
            'event_id': f'{eid}:e{len(events):06d}',
            'seq': len(events),
            'kind': kind,
            'data': data,
        }
        events.append(ev)
        return ev

    cfg = {
        'budget': {'max_model_requests': 35, 'max_tool_calls': 50, 'wall_time_seconds': 3600},
        'model': {'adapter': 'gemini', 'model': model_version, 'parameters': {'temperature': 1.0}},
    }
    if model_parameters is not None:
        cfg['model']['parameters'] = copy.deepcopy(model_parameters)
    if parent_pkg is not None:
        cfg = parent_pkg.json('raw/resolved_config.json')
    write_json(root / 'raw/resolved_config.json', cfg)
    write_json(
        root / 'benchmark/public_task.json',
        {'benchmark': benchmark, 'task_id': task_id, 'prompt': f'Edit input/doc{ext}', 'assets': [{'name': f'doc{ext}'}]},
    )

    lineage = (
        {'parent_checkpoint_id': f'cp{resume_from_k:06d}'}
        if resume_from_k > 0
        else None
    )
    if parent_pkg is not None:
        parent_cp = parent_pkg.checkpoints()[resume_from_k]
        lineage = {'parent_episode_id': parent_pkg.episode_id,
                   'parent_checkpoint_id': parent_cp['checkpoint_id'],
                   'parent_index_sha256': parent_pkg.index_sha256,
                   'parent_checkpoint_sha256': parent_cp['sha256']}
    if lineage:
        emit('episode.restored', lineage)

    cur_files = [write_blob(f'input/doc{ext}', old_data), write_blob(f'submission/out{ext}', old_data)]
    s0 = add_snap(cur_files)
    emit('state.snapshot', {'snapshot_id': s0['snapshot_id'], 'snapshot': {'path': f"states/snapshots/{s0['snapshot_id']}.json"}})

    cp_rows = []
    state_changes = []

    for k in range(resume_from_k, num_turns):
        turn = k + 1
        cp_id = f'cp{k:06d}'
        cp_obj = {
            'checkpoint_id': cp_id,
            'snapshot_id': snaps[-1]['snapshot_id'],
            'budget_used': {'model_requests': k, 'tool_calls': k, 'agent_seconds': k},
            'qualification': {'eligible': True},
            'config_sha256': digest(cfg),
        }
        write_json(root / f'checkpoints/{cp_id}.json', cp_obj)
        cp_rows.append({'checkpoint_id': cp_id, 'path': f'checkpoints/{cp_id}.json', 'eligible': True})

        rid = f'{eid}:r{k:06d}'
        emit('model.request', {'request_id': rid})
        emit('model.response', {'request_id': rid, 'model_version': model_version})

        action = turn_actions.get(turn, 'finish' if k == num_turns - 1 else 'normal')
        if action == 'finish':
            emit('model.parsed', {'request_id': rid, 'text': 'Done.', 'finish_reason': 'STOP'})
            continue

        emit('model.parsed', {'request_id': rid})
        cmd = (
            f'ls -la submission/out{ext}' if action == 'readback' else
            f'cp input/doc{ext} submission/out{ext}' if action == 'modify_submission' else
            f'echo turn {turn}'
        )
        cid = f'{rid}:c0'
        emit('tool.call', {'request_id': rid, 'call_id': cid, 'name': 'run', 'arguments': {'command': cmd}})

        exit_code = 1 if action == 'anomaly' else 0
        output_complete = False if action == 'truncated_output' else True
        obs = {
            'exit_code': exit_code,
            'stop_reason': None,
            'output_complete': output_complete,
            'stdout': f'stdout {turn}\n',
            'stderr': 'error\n' if exit_code else '',
        }
        res_rel = f'raw/tools/r{k:06d}.json'
        write_json(root / res_rel, {'call_id': cid, 'observation': obs})
        emit('tool.result', {'call_id': cid, 'result': {'path': res_rel}})

        before_snap = snaps[-1]
        if action == 'modify_submission':
            next_files = [write_blob(f'input/doc{ext}', old_data), write_blob(f'submission/out{ext}', new_data)]
        else:
            next_files = list(before_snap['files'])
        after_snap = add_snap(next_files)
        emit('state.snapshot', {'snapshot_id': after_snap['snapshot_id'], 'snapshot': {'path': f"states/snapshots/{after_snap['snapshot_id']}.json"}})
        tr_ev = emit('state.transition', {'before': before_snap['snapshot_id'], 'after': after_snap['snapshot_id']})
        state_changes.append({
            'episode_id': eid,
            'event_id': tr_ev['event_id'],
            'before_snapshot_id': before_snap['snapshot_id'],
            'after_snapshot_id': after_snap['snapshot_id'],
            'changes': diff_snapshots(before_snap, after_snap),
        })

    write_json(root / 'checkpoints/index.json', {'checkpoints': cp_rows})
    write_jsonl(root / 'state_changes.jsonl', state_changes)

    if terminal_case == 'empty_submission':
        write_json(root / 'artifacts/inventory.json', {'episode_id': eid, 'files': []})
    else:
        write_json(root / 'artifacts/inventory.json', {'episode_id': eid, 'files': [{'path': f'out{ext}', 'sha256': 'abc', 'size_bytes': 10}]})

    # Keep synthetic evidence structurally faithful to the native grader. The
    # shared endpoint must exercise the same health gates as production evidence.
    from c3_sandbox.l1.evaluation.officeval import interpret
    from c3_sandbox.l1.evaluation.lhtb import interpret_lhtb
    exec_valid = 'invalid' if terminal_case == 'execution_invalid' else 'valid'
    process = {'container_exit_code': 0, 'stop_reason': None}
    if benchmark == 'lhtb':
        band = 1 if passed else 0
        code = 0 if passed else 1
        raw = {'status': 'ok', 'error': None, 'task_id': task_id,
               'reward': float(passed), 'raw_reward': band, 'band': band,
               'scoring_case': 'official_score' if passed else 'official_below_pass_band',
               'protocol': {'pytest_exit': code}}
        proj = interpret_lhtb(raw, task_id)
        process['container_exit_code'] = code
        xml = root / 'grading/worker-output/pytest.xml'
        xml.parent.mkdir(parents=True, exist_ok=True)
        xml.write_text('<testsuite><testcase name="test_sokoban_progress">'
                       '<failure message="AssertionError: Band 0 &lt; pass threshold 1."/>'
                       '</testcase></testsuite>')
    else:
        is_scored = terminal_case == 'scored'
        error = ('FileNotFoundError: submission empty' if terminal_case == 'empty_submission'
                 else 'BadZipFile: damaged delivery' if terminal_case == 'non_ok_grade' else None)
        raw = {'id': task_id[-3:], 'file_name': f'out{ext}',
               'status': 'error' if error else 'ok', 'error': error,
               'dim1_pass': is_scored and completion_rate > 0, 'dim1_reason': '',
               'dim2_items': [], 'total_score': completion_rate * 10 if is_scored else 0,
               'max_score': 10}
        proj = interpret(raw, task_id)
        write_json(root / 'grading/worker-output/worker.json',
                   {'official_returned': True, 'exception': None})
    if terminal_case in ('scorer_crash', 'execution_invalid'):
        raw = None
        proj = {'evaluation_validity': 'invalid'}
        process['container_exit_code'] = 1
    eval_valid = proj['evaluation_validity']
    write_json(root / 'official_grade.json', {'official_raw': raw, 'projection': proj,
               'process': process, 'status': 'ok' if eval_valid == 'valid' else 'error'})
    manifest = {
        'schema_version': '0.2.0',
        'recording_mode': 'test_fixture',
        'episode_id': eid,
        'task': {'benchmark': benchmark, 'task_id': task_id},
        'model': {'model_version': model_version},
        'status': {
            'termination': 'assistant_finished' if exec_valid == 'valid' else 'model_transport_error',
            'execution_validity': exec_valid,
            'evidence_completeness': 'complete',
            'evaluation_validity': eval_valid,
        },
        'counts': {'model_requests': sum(1 for e in events if e['kind'] == 'model.request')},
        'lineage': lineage,
    }
    write_json(root / 'manifest.json', manifest)
    write_jsonl(root / 'trajectory.jsonl', events)
    from l2_sandbox.io import sha_file
    files = [{'path': path.relative_to(root).as_posix(), 'sha256': sha_file(path),
              'size_bytes': path.stat().st_size} for path in sorted(root.rglob('*'))
             if path.is_file() and path.name != 'evidence_index.json']
    write_json(root / 'evidence_index.json', {'schema_version': '0.2.0', 'episode_id': eid,
                                           'files': files, 'trust_scope': 'SYNTHETIC TEST FIXTURE'})
    return root
