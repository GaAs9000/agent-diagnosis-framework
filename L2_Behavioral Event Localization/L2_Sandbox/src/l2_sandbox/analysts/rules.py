"""Deterministic Rule Analyst (`rules@1`, Group A) (L2 v0.4 §5.1-5.2, F24)."""
from __future__ import annotations
import copy
from pathlib import Path
from typing import Any

from ..evidence.l1 import L1Package
from ..grades import (
    GATE_DEFECT_ID,
    NO_ARTIFACT_DEFECT_ID,
    OFFICE_EXTS,
    _office_targets,
    compute_item_deductions,
    extract_task_rubric,
)
from ..predicates import RULE_VOCABULARY, eval_predicate_on_turn
from . import Hypothesis, Origin

RULES_ANALYST_ID = 'rules@1'


def build_terminal_official_view(
    pkg: L1Package,
    facts: list[dict[str, Any]],
    *,
    extra_runs: list[L1Package] | None = None,
) -> dict[str, Any]:
    """L2 v0.4 §5.2 Call 1 terminal official grade view:
    - When delivered (`dim1_pass` is True): per-item deductions from terminal `official_grade.json`.
    - When not delivered (`dim1_pass` is False): `zero_reason` and each rubric item's max points (`max_delta`),
      or `no_artifact` when the trajectory never produced a surviving candidate file.
    Never reads checkpoint snapshot scores `s(k)` or `defects.json`.
    """
    og = pkg.json('official_grade.json') if (pkg.root / 'official_grade.json').is_file() else {}
    proj = og.get('projection') or {}
    raw = og.get('official_raw') if isinstance(og.get('official_raw'), dict) else {}
    y_val = float(proj['completion_rate']) if isinstance(proj.get('completion_rate'), (int, float)) else 0.0
    delivered_valid = bool(raw.get('status') == 'ok' and raw.get('dim1_pass') is True)
    _, target_exts = _office_targets(pkg)
    rubric = extract_task_rubric(pkg, extra_runs=extra_runs)

    if y_val >= 1.0 - 1e-12:
        return {
            'delivered_valid': True,
            'terminal_score_y': y_val,
            'zero_reason': None,
            'target_exts': sorted(target_exts),
            'target_defects': [],
        }

    if delivered_valid:
        deductions, max_score = compute_item_deductions(raw, y_val, rubric)
        pos_d = {k: v for k, v in deductions.items() if v > 1e-12}
        pos_sum = sum(pos_d.values())
        rem_target = max(0.0, 1.0 - y_val)
        scale = (rem_target / pos_sum) if (pos_sum > 0 and abs(pos_sum - rem_target) > 1e-9) else 1.0
        max_by_rule = {r: md for r, md in rubric}
        target_defects = [
            {
                'defect': rule_id,
                'max_delta': max_by_rule.get(rule_id),
                'deduction': round(d_val * scale, 12),
            }
            for rule_id, d_val in pos_d.items()
            if d_val * scale > 1e-12
        ]
        return {
            'delivered_valid': True,
            'terminal_score_y': y_val,
            'max_score': max_score,
            'zero_reason': None,
            'target_exts': sorted(target_exts),
            'target_defects': target_defects,
        }

    # Undelivered / invalid terminal submission: check from facts whether any candidate file exists at terminal
    s0 = pkg.snapshots.get('s000000') or pkg.json('states/snapshots/s000000.json')
    initial_input_hashes = {f['sha256'] for f in s0.get('files', []) if f['path'].startswith('input/')}
    active_office_files: set[str] = set()
    for f in facts:
        if f.get('type') != 'file_transition':
            continue
        val = f.get('value') or {}
        p = str(val.get('path') or '')
        if not p or p.startswith('input/') or Path(p).name.startswith('~$'):
            continue
        if Path(p).suffix.lower() not in target_exts:
            continue
        ch = val.get('change')
        after_sha = (val.get('after') or {}).get('sha256')
        if ch in ('created', 'modified'):
            if p.startswith('submission/') or after_sha not in initial_input_hashes:
                active_office_files.add(p)
            else:
                active_office_files.discard(p)
        elif ch == 'deleted':
            active_office_files.discard(p)

    zero_reason = proj.get('reason') or (raw.get('error') if isinstance(raw, dict) else None) or 'undelivered_or_invalid'
    if not active_office_files:
        return {
            'delivered_valid': False,
            'terminal_score_y': y_val,
            'zero_reason': zero_reason,
            'target_exts': sorted(target_exts),
            'target_defects': [{'defect': NO_ARTIFACT_DEFECT_ID, 'max_delta': 1.0, 'deduction': None}],
        }

    pos_rubric = [(r, md) for r, md in rubric if md > 0]
    if pos_rubric:
        target_defects = [{'defect': r, 'max_delta': md, 'deduction': None} for r, md in pos_rubric]
    else:
        target_defects = [{'defect': GATE_DEFECT_ID, 'max_delta': 1.0, 'deduction': None}]
    return {
        'delivered_valid': False,
        'terminal_score_y': y_val,
        'zero_reason': zero_reason,
        'target_exts': sorted(target_exts),
        'target_defects': target_defects,
    }


def predict_origins_rules(
    facts: list[dict[str, Any]],
    terminal_official_view: dict[str, Any],
) -> list[Origin]:
    """L2 v0.4 §5.1 Call 1 for `rules@1` (Group A):
    - When terminal has no valid delivery (`delivered_valid` is False), classify each defect as
      `undelivered` with `turn` = last turn that wrote a same-type file (or `never` if none).
    - When terminal has a valid delivery (`delivered_valid` is True), classify each defect as `never`.
    """
    target_defects = terminal_official_view.get('target_defects') or []
    if not target_defects:
        return []

    delivered_valid = bool(terminal_official_view.get('delivered_valid'))
    target_exts = {ext.lower() for ext in (terminal_official_view.get('target_exts') or OFFICE_EXTS)}

    if delivered_valid:
        return [
            {
                'analyst': RULES_ANALYST_ID,
                'analyst_group': 'A',
                'defect': str(d['defect']),
                'kind': 'never',
                'turn': None,
                'rationale': 'Terminal submission is valid; rule baseline classifies deducted items as never.',
            }
            for d in target_defects
        ]

    last_write_turn: int | None = None
    for f in facts:
        if f.get('type') != 'file_transition':
            continue
        val = f.get('value') or {}
        p = str(val.get('path') or '')
        if not p or p.startswith('input/') or Path(p).name.startswith('~$'):
            continue
        if Path(p).suffix.lower() in target_exts and val.get('change') in ('created', 'modified'):
            t = f.get('turn')
            if isinstance(t, int) and (last_write_turn is None or t > last_write_turn):
                last_write_turn = t

    out: list[Origin] = []
    for d in target_defects:
        d_id = str(d['defect'])
        if d_id == NO_ARTIFACT_DEFECT_ID or last_write_turn is None:
            out.append({
                'analyst': RULES_ANALYST_ID,
                'analyst_group': 'A',
                'defect': d_id,
                'kind': 'never',
                'turn': None,
                'rationale': 'No surviving candidate office file written; rule baseline classifies as never.',
            })
        else:
            out.append({
                'analyst': RULES_ANALYST_ID,
                'analyst_group': 'A',
                'defect': d_id,
                'kind': 'undelivered',
                'turn': last_write_turn,
                'rationale': f'Terminal submission invalid/missing; last same-type file write at turn {last_write_turn}.',
            })
    return out


def propose_hypotheses_rules(
    facts: list[dict[str, Any]],
    defects_table: dict[str, Any],
) -> list[Hypothesis]:
    """L2 v0.4 §5.1 Call 2 for `rules@1` (Group A):
    For each defect with an origin turn j >= 1, proposes all vocabulary predicates
    that hold on the parent trajectory at turn j.
    """
    hyps: list[Hypothesis] = []
    for d in defects_table.get('defects') or []:
        defect_id = str(d.get('defect') or '')
        target_turn = d.get('turn') if d.get('turn') is not None else d.get('two_point_turn')
        if not defect_id or not isinstance(target_turn, int) or isinstance(target_turn, bool) or target_turn < 1:
            continue
        for name, pred in RULE_VOCABULARY.items():
            if eval_predicate_on_turn(pred, facts, target_turn):
                hyps.append({
                    'analyst': RULES_ANALYST_ID,
                    'analyst_group': 'A',
                    'defect': defect_id,
                    'turn': int(target_turn),
                    'predicate': copy.deepcopy(pred),
                    'effect': -0.25,
                    'rationale': f'Rule vocabulary predicate `{name}` holds on parent turn {target_turn}.',
                })
    return hyps
