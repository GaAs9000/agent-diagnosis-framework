"""Restricted trajectory view for LLM Analysts (L2 v0.3 Section 2 & 7.2).

Strict information boundary: accepts only the parent L1Package, p_0, and optional
curve summary. Never reads or exposes branch-level outcomes or branch facts.
"""
from __future__ import annotations
from typing import Any
from .l1 import L1Package


def build_trajectory_view(parent: L1Package) -> dict[str, Any]:
    """Constructs a structured turn-by-turn view of the parent trajectory for LLM Analysts."""
    pub = parent.json('benchmark/public_task.json') if (parent.root / 'benchmark/public_task.json').is_file() else {}
    out = parent.outcome()
    events = parent.events
    req_indices = [i for i, e in enumerate(events) if e.get('kind') == 'model.request']
    changes_by_event = {c['event_id']: c.get('changes', []) for c in parent.jsonl('state_changes.jsonl')}

    turns = []
    for idx, start_i in enumerate(req_indices):
        end_i = req_indices[idx + 1] if idx + 1 < len(req_indices) else len(events)
        turn_num = parent.start_turn + idx + 1
        slice_events = events[start_i:end_i]
        texts = []
        calls = []
        exits = []
        diffs = []
        for e in slice_events:
            kind = e.get('kind')
            d = e.get('data') or {}
            if kind == 'model.parsed' and d.get('text'):
                texts.append(str(d['text'])[:800])
            elif kind == 'tool.call':
                calls.append({'call_id': d.get('call_id'), 'name': d.get('name'), 'arguments': d.get('arguments')})
            elif kind == 'tool.result':
                res_path = (d.get('result') or {}).get('path')
                obs = (parent.json(res_path).get('observation') or {}) if res_path and (parent.root / res_path).is_file() else {}
                exits.append({
                    'call_id': d.get('call_id'),
                    'exit_code': obs.get('exit_code'),
                    'stop_reason': obs.get('stop_reason'),
                    'output_complete': obs.get('output_complete'),
                    'stdout_preview': str(obs.get('stdout') or '')[:400],
                    'stderr_preview': str(obs.get('stderr') or '')[:400],
                })
            elif kind == 'state.transition':
                for delta in changes_by_event.get(e['event_id'], []):
                    diffs.append({'path': delta.get('path'), 'change': delta.get('change')})
        turns.append({
            'turn': turn_num,
            'assistant_text': '\n'.join(texts) if texts else None,
            'tool_calls': calls,
            'tool_exits': exits,
            'file_changes': diffs,
        })

    return {
        'episode_id': parent.episode_id,
        'benchmark': parent.benchmark,
        'task_id': parent.task_id,
        'family_id': parent.family_id,
        'prompt': pub.get('prompt') or (parent.manifest.get('task') or {}).get('prompt'),
        'num_turns': parent.turns,
        'terminal_outcome': out['y'],
        'termination': (parent.manifest.get('status') or {}).get('termination'),
        'turns': turns,
    }
