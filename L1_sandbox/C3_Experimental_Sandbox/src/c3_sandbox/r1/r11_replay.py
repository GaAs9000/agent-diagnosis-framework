"""Fixed engineering coverage selection and fresh evidence-bound admission."""
from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path

from ..evidence._storage import encode
from ..manifests import canonical_hash
from ..replay import ReplayQualification, _message_reasons
from .admission import R1AdmissionAuthority, _audit
from .docker_runtime import DockerRuntime, validate_snapshot
from .replay_support import SYSTEM_PROMPT, _qualification, _sha, _write


def _load(path):
    return json.loads(Path(path).read_bytes())


def _boundary(reference, max_turns):
    if reference.get('source_boundary_qualified') is not True or reference.get('ineligible_reasons'):
        return None
    try:
        record = _load(reference['checkpoint_record_path'])
        messages = _load(reference['messages_path'])
        audit = _load(reference['source_audit_path'])
        if (record.get('registration_reasons') or ReplayQualification._input_reasons(record)
                or record.get('is_synthetic') is not False or not record.get('snapshot_id')
                or audit.get('qualified') is not True or messages != record['messages']
                or not messages or messages[-1].get('role') != 'tool' or _message_reasons(messages)):
            return None
        if canonical_hash(record) != reference['checkpoint_id']:
            return None
        remaining = max_turns - sum(m['role'] == 'assistant' for m in messages)
        if remaining <= 0:
            return None
        return remaining
    except (OSError, ValueError, KeyError, TypeError):
        return None


def select_checkpoint(baseline, kind, max_turns=10):
    """Select S's first qualified boundary or T's earliest observed later tool."""
    if kind not in ('S', 'T') or type(max_turns) is not int or max_turns <= 0:
        raise ValueError('kind S/T and positive max_turns required')
    agent = baseline.get('agent_result', {})
    if (baseline.get('execution_valid') is not True or baseline.get('is_synthetic') is not False
            or agent.get('protocol_status') != 'valid'):
        return None
    if kind == 'S' and baseline.get('task_completed') is not True:
        return None
    qualified = [(cp, remaining) for cp in baseline.get('checkpoints', [])
                 if (remaining := _boundary(cp, max_turns)) is not None]
    qualified.sort(key=lambda pair: pair[0]['event_index'])
    selected = None
    later_id = None
    if kind == 'S' and qualified:
        selected = qualified[0]
    if kind == 'T':
        dispatched_ids = {call['tool_call_id'] for call in agent.get('tool_dispatch', [])
                          if call.get('tool_call_id')}
        for event in agent.get('events', []):
            if event.get('type') != 'tool_result' or event.get('tool_call_id') not in dispatched_ids:
                continue
            # A second call within the same assistant batch is not a cold boundary.
            preceding = [cp for cp in baseline.get('checkpoints', [])
                         if cp['event_index'] < event['event_index']]
            if preceding:
                nearest = max(preceding, key=lambda cp: cp['event_index'])
                remaining = _boundary(nearest, max_turns)
                if remaining is not None and event.get('turn', -1) >= max_turns - remaining:
                    selected = (nearest, remaining)
                    later_id = event['tool_call_id']
                    break
    if selected is None:
        return None
    cp, remaining = selected
    return {'kind': kind, 'task_id': baseline['task_id'], 'checkpoint_id': cp['checkpoint_id'],
            'reference': copy.deepcopy(cp), 'remaining_turns': remaining,
            'reason': 'first qualified complete-tool boundary' if kind == 'S' else
                      'nearest complete-tool boundary before earliest subsequent actual model tool call',
            'subsequent_tool_call_id': later_id}


def select_engineering_candidates(results):
    chosen = {'S_candidate': None, 'T_candidate': None, 'decisions': []}
    for index, baseline in enumerate(results):
        selections = {kind: select_checkpoint(baseline, kind) for kind in ('S', 'T')}
        chosen['decisions'].append({'order': index, 'task_id': baseline['task_id'],
            'execution_valid': baseline.get('execution_valid'), 'task_completed': baseline.get('task_completed'),
            'outcome': baseline.get('outcome'), 'protocol_status': baseline.get('agent_result', {}).get('protocol_status'),
            'S_qualified': selections['S'] is not None, 'T_qualified': selections['T'] is not None,
            'selection_reasons': {kind: selection['reason'] if selection else 'coverage requirements not met'
                                  for kind, selection in selections.items()}})
        for kind, selection in selections.items():
            if chosen[kind + '_candidate'] is None and selection is not None:
                chosen[kind + '_candidate'] = selection
    return chosen


def admit_checkpoint(baseline, reference, root, model_trace):
    """Observe fresh Docker reconstruction, then issue a private admission receipt."""
    model = _qualification(baseline['model_qualification'])
    model_trace = Path(model_trace).absolute()
    if _sha(model_trace) != model.get('model_trace_sha256'):
        raise ValueError('raw model qualification trace hash mismatch')
    if _boundary(reference, baseline['runtime'].get('sampling', {}).get('max_turns', 10)) is None:
        raise ValueError('source checkpoint is not a qualified complete tool boundary')
    root = Path(root).absolute()
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    authority = R1AdmissionAuthority(root / 'private')
    replay = ReplayQualification(Path(baseline['replay_store']), real_admission=authority)
    record = replay._record(reference['checkpoint_id'])
    source = _load(reference['source_audit_path'])
    _audit(source, record['runtime']['image_digest'])
    _write(root / 'checkpoint_record.json', record)
    restored = replay.workspace_store.restore(record['snapshot_id'], root / 'staging')
    _write(root / 'host_restore.json', restored.to_dict())
    if not restored.valid:
        raise ValueError('host snapshot reconstruction failed')
    messages = _load(reference['messages_path'])
    _write(root / 'source_audit.json', source)
    with DockerRuntime(record['runtime']['image_digest'], root / 'runtime') as actor:
        actor.seed_workspace(root / 'staging')
        equality = actor.verify_workspace(root / 'staging')
        audit = actor.audit_boundary()
        _audit(audit, record['runtime']['image_digest'])
        env = actor._inspect()['Config']['Env']
        if not isinstance(env, list) or any(not isinstance(item, str) or '=' not in item for item in env):
            raise ValueError('exact runtime environment unavailable')
        actual = dict(item.split('=', 1) for item in env)
        if len(actual) != len(env):
            raise ValueError('duplicate runtime environment keys')
        runtime = copy.deepcopy(baseline['runtime'])
        runtime['environment'] = actual
        sampling = {'seed': 2026, 'temperature': 0, 'max_tokens': 2048, 'context_window': 16384,
                    'max_retries': 0, 'parallel_tool_calls': False, 'top_k': 1, 'min_p': 0, 'max_turns': 10}
        identity = canonical_hash({'model': runtime['model_config'], 'agent': 'NativeAgent-r1',
            'tools': runtime['tool_config'], 'system_prompt': SYSTEM_PROMPT, 'sampling': sampling, 'runtime': runtime})
        report = {**equality, 'messages_equal': encode(messages) == encode(record['messages']),
            'runtime_equal': encode(runtime) == encode(record['runtime']) and audit['effective_config_ok'],
            'system_equal': identity == record['system_identity'], 'observed_system_identity': identity,
            'message_identity': canonical_hash(messages), 'no_model_continuation_during_qualification': True,
            'scope': 'complete workspace bytes/paths/types/modes, paired native messages, image/config/environment'}
        actor.export_workspace(root / 'restored-workspace')
    for name, value in [('restore_audit', audit), ('restoration_report', report), ('model_qualification', model)]:
        _write(root / (name + '.json'), value)
    paths = [root / (name + '.json') for name in ('checkpoint_record', 'source_audit',
             'restore_audit', 'restoration_report', 'model_qualification')] + [model_trace]
    if _sha(model_trace) != model['model_trace_sha256']:
        raise ValueError('raw model qualification trace changed during reconstruction')
    receipt = authority.issue(reference['checkpoint_id'], record, source, audit, report, model, paths)
    verdict = replay.qualify(reference['checkpoint_id'])
    _write(root / 'qualification.json', {**verdict, 'receipt_identity': receipt})
    if verdict.get('eligible_for_intervention') is not True:
        raise ValueError('real admission failed')
    return {'qualification': verdict, 'receipt_identity': receipt, 'paths': [str(p) for p in paths],
            'replay': replay, 'authority': authority}


def snapshot_inventory(snapshot):
    validate_snapshot(snapshot)
    inventory = {'.': {'kind': 'directory', 'mode': snapshot['root_mode']}}
    for entry in snapshot['entries']:
        item = {key: entry[key] for key in ('kind', 'mode')}
        if entry['kind'] == 'file':
            data = base64.b64decode(entry['data'], validate=True)
            item.update(sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))
        inventory[entry['path']] = item
    return inventory


def filesystem_delta(before, after):
    before, after = snapshot_inventory(before), snapshot_inventory(after)
    return [{'path': path, 'before': before.get(path), 'after': after.get(path)}
            for path in sorted(before.keys() | after.keys()) if before.get(path) != after.get(path)]


class DispatchRecorder:
    """Bind each new SDK dispatch to full snapshots and independently derived delta."""
    def __init__(self, actor, root):
        self.actor, self.root = actor, Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.pending, self.records = {}, []

    def capture(self):
        try:
            return self.actor._snapshot(), None
        except (ValueError, RuntimeError) as error:
            if getattr(self.actor, 'invalidated', False) or 'timeout' in str(error).lower():
                raise
            if any(marker in str(error).lower() for marker in ('symlink', 'hardlink', 'special file', 'path limit', 'byte limit')):
                return None, str(error)
            raise

    async def __call__(self, event):
        call_id = event['tool_call_id']
        if event['stage'] == 'start':
            if call_id in self.pending:
                raise ValueError('duplicate tool dispatch ID')
            folder = self.root / f'call-{len(self.records):03d}'
            folder.mkdir()
            before, reason = self.capture()
            _write(folder / 'before.json', before)
            self.pending[call_id] = (folder, before, reason)
            return
        if event['stage'] != 'end' or call_id not in self.pending:
            raise ValueError('unpaired tool dispatch callback')
        folder, before, reason = self.pending.pop(call_id)
        after, after_reason = self.capture()
        _write(folder / 'after.json', after)
        arguments = event['arguments']
        code = arguments.get('code') if isinstance(arguments, dict) else None
        record = {**copy.deepcopy(event), 'code_sha256': hashlib.sha256(code.encode()).hexdigest() if isinstance(code, str) else None,
            'before_path': str(folder / 'before.json'), 'after_path': str(folder / 'after.json'),
            'fs_delta': filesystem_delta(before, after) if before is not None and after is not None else None,
            'snapshot_ineligible_reasons': [item for item in (reason, after_reason) if item],
            'evidence_kind': 'actual_model_tool_dispatch'}
        _write(folder / 'dispatch.json', record)
        self.records.append(record)
