"""Turn-indexed deterministic fact extraction `{turn, type, value}` (L2 v0.3 Section 2.1 & 7.2).

Changes from v0.1/v0.2:
1. Every fact carries `turn` (1-based model turn; on a resumed branch from cp{k}, starts at k + 1).
2. `tool_exit` includes boolean `ok` (`exit_code == 0 and stop_reason is None and output_complete is True`).
3. Strips reference/coverage/origin boilerplate so the same function runs identically on parent and branches.
"""
from __future__ import annotations
from typing import Any
from ..evidence.l1 import L1Package
from .docx_probe import probe

DEFAULT_PROBE_CONFIG = {
    'max_docx_versions': 64,
    'max_xml_bytes': 33554432,
    'max_zip_uncompressed_bytes': 134217728,
    'max_objects': 5000,
}

PREVIEW_BINARIES = ('soffice', 'libreoffice', 'pdftoppm')

FACT_FIELD_WHITELIST: dict[str, tuple[str, ...]] = {
    'tool_requested': (
        'name',
        'call_id',
        'request_id',
        'arguments.command',
        'arguments.path',
        'arguments.offset',
        'arguments.limit',
    ),
    'preview_rendered': ('name', 'command'),
    'tool_exit': ('call_id', 'name', 'exit_code', 'stop_reason', 'output_complete', 'ok', 'stderr_preview'),
    'file_transition': (
        'path',
        'change',
        'before.path',
        'after.path',
        'before.blob',
        'after.blob',
        'before.sha256',
        'after.sha256',
        'before.size_bytes',
        'after.size_bytes',
        'before.mode',
        'after.mode',
    ),
    'object_structure_change': (
        'file_path',
        'before_sha256',
        'after_sha256',
        'before_summary.paragraphs_recursive',
        'after_summary.paragraphs_recursive',
        'before_summary.tables_recursive',
        'after_summary.tables_recursive',
        'before_summary.top_level_tables',
        'after_summary.top_level_tables',
        'before_summary.rows_recursive',
        'after_summary.rows_recursive',
        'before_summary.textless_rows_without_drawing',
        'after_summary.textless_rows_without_drawing',
        'before_summary.textless_rows_are_not_rendered_empty_row_judgements',
        'after_summary.textless_rows_are_not_rendered_empty_row_judgements',
    ),
    'model_output_text': ('text', 'request_id'),
    'protocol_error': ('kind', 'code', 'message', 'reason', 'error_type', 'call_id', 'request_id', 'stop_phase', 'transport_attempts'),
    'execution_error': ('kind', 'code', 'message', 'reason', 'error_type', 'call_id', 'request_id', 'stop_phase', 'transport_attempts'),
    'termination': ('termination', 'protocol_validity', 'execution_validity', 'evaluation_validity', 'task_outcome', 'evidence_completeness'),
}


def format_fact_field_whitelist() -> str:
    return '\n'.join(
        f"- `{fact}`: " + ", ".join(f"`{f}`" for f in fields)
        for fact, fields in FACT_FIELD_WHITELIST.items()
    )


def build_facts(pkg: L1Package, probe_config: dict | None = None) -> list[dict[str, Any]]:
    """Extracts turn-indexed `{turn, type, value}` facts from any L1Package (parent or branch)."""
    cfg = {**DEFAULT_PROBE_CONFIG, **(probe_config or {})}
    events = pkg.events
    req_indices = [i for i, e in enumerate(events) if e.get('kind') == 'model.request']
    if not req_indices:
        return []

    # Pre-probe .docx blobs across snapshots so we can emit `object_structure_change` on transitions
    docx_summaries: dict[str, dict] = {}
    for snap in pkg.snapshots.values():
        for f in snap.get('files', []):
            path_str = str(f.get('path', ''))
            sha = f.get('sha256')
            blob = f.get('blob')
            if path_str.lower().endswith('.docx') and sha and blob and sha not in docx_summaries:
                try:
                    parsed = probe(pkg.read(blob), cfg)
                    if parsed.get('status') in ('present', 'partial') and parsed.get('summary'):
                        docx_summaries[sha] = parsed['summary']
                except Exception:
                    pass

    changes_by_event = {c['event_id']: c.get('changes', []) for c in pkg.jsonl('state_changes.jsonl')}
    calls_by_id: dict[str, dict] = {}
    for e in events:
        if e.get('kind') == 'tool.call':
            d = e.get('data') or {}
            if d.get('call_id'):
                calls_by_id[d['call_id']] = d

    facts: list[dict[str, Any]] = []
    for idx, start_i in enumerate(req_indices):
        end_i = req_indices[idx + 1] if idx + 1 < len(req_indices) else len(events)
        turn = pkg.start_turn + idx + 1
        slice_events = events[start_i:end_i]

        for e in slice_events:
            kind = e.get('kind')
            d = e.get('data') or {}
            if kind == 'model.parsed' and d.get('text'):
                facts.append({
                    'turn': turn,
                    'type': 'model_output_text',
                    'value': {'request_id': d.get('request_id'), 'text': d.get('text')},
                })
            elif kind == 'tool.call':
                args = d.get('arguments') or {}
                tool_name = d.get('name')
                facts.append({
                    'turn': turn,
                    'type': 'tool_requested',
                    'value': {
                        'call_id': d.get('call_id'),
                        'request_id': d.get('request_id'),
                        'name': tool_name,
                        'arguments': args,
                    },
                })
                if tool_name in ('run', 'bash') and isinstance(args, dict):
                    cmd = str(args.get('command') or '')
                    cmd_low = cmd.lower()
                    if any(b in cmd_low for b in PREVIEW_BINARIES):
                        facts.append({
                            'turn': turn,
                            'type': 'preview_rendered',
                            'value': {
                                'name': tool_name,
                                'command': cmd,
                            },
                        })
            elif kind == 'tool.result':
                res_path = (d.get('result') or {}).get('path')
                obs = (pkg.json(res_path).get('observation') or {}) if res_path and (pkg.root / res_path).is_file() else {}
                exit_code = obs.get('exit_code')
                stop_reason = obs.get('stop_reason')
                output_complete = obs.get('output_complete', True)
                is_ok = bool(exit_code == 0 and stop_reason is None and output_complete is True)
                call_meta = calls_by_id.get(d.get('call_id', ''), {})
                facts.append({
                    'turn': turn,
                    'type': 'tool_exit',
                    'value': {
                        'call_id': d.get('call_id'),
                        'name': call_meta.get('name'),
                        'exit_code': exit_code,
                        'stop_reason': stop_reason,
                        'output_complete': output_complete,
                        'ok': is_ok,
                        'stderr_preview': str(obs.get('stderr') or '')[:800],
                    },
                })
            elif kind in ('tool.error', 'tool.rejected', 'episode.error', 'model.error'):
                facts.append({
                    'turn': turn,
                    'type': 'protocol_error' if d.get('code') == 'protocol_error' or kind == 'tool.rejected' else 'execution_error',
                    'value': {'kind': kind, **d},
                })
            elif kind == 'state.transition':
                for delta in changes_by_event.get(e['event_id'], []):
                    facts.append({
                        'turn': turn,
                        'type': 'file_transition',
                        'value': {
                            'path': delta.get('path'),
                            'change': delta.get('change'),
                            'before': delta.get('before'),
                            'after': delta.get('after'),
                        },
                    })
                    b_sha = (delta.get('before') or {}).get('sha256')
                    a_sha = (delta.get('after') or {}).get('sha256')
                    if b_sha in docx_summaries and a_sha in docx_summaries:
                        if docx_summaries[b_sha] != docx_summaries[a_sha]:
                            facts.append({
                                'turn': turn,
                                'type': 'object_structure_change',
                                'value': {
                                    'file_path': delta.get('path'),
                                    'before_sha256': b_sha,
                                    'after_sha256': a_sha,
                                    'before_summary': docx_summaries[b_sha],
                                    'after_summary': docx_summaries[a_sha],
                                },
                            })
            elif kind == 'episode.finished':
                facts.append({
                    'turn': turn,
                    'type': 'termination',
                    'value': d.get('status') or d,
                })

    return facts
