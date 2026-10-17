"""Defect table construction, uncorrected window summaries, and cross-run contrast (L2 v0.4 §4.1-4.6, F24).

Zero Actor model requests.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any

from .config import CODE_VERSION
from .evidence.l1 import L1Package
from .grades import GATE_DEFECT_ID, NO_ARTIFACT_DEFECT_ID, OFFICE_EXTS, _extract_dim1_pass, extract_task_rubric
from .io import L2Error, read_json, sha_file, write_json
from .predicates import RULE_VOCABULARY, eval_predicate_on_turn


def _is_same_target_activity(turn_facts: list[dict[str, Any]], target_paths: list[str], target_exts: set[str]) -> bool:
    """Check whether a turn in the uncorrected window is inspecting, previewing, or modifying the target file."""
    target_names = {Path(p).name for p in target_paths if p}
    target_stems = {Path(p).stem for p in target_paths if p}
    for f in turn_facts:
        ftype = f.get('type')
        val = f.get('value') or {}
        if ftype == 'preview_rendered':
            return True
        if ftype == 'file_transition':
            p = str(val.get('path') or '')
            if p in target_paths or Path(p).name in target_names or Path(p).suffix.lower() in target_exts:
                return True
            if p.startswith('submission/') or Path(p).suffix.lower() in {'.png', '.pdf', '.xml'}:
                return True
        if ftype == 'tool_requested':
            name = str(val.get('name') or '')
            args = val.get('arguments') or {}
            cmd = str(args.get('command') or '')
            path_arg = str(args.get('path') or '')
            combined = f'{cmd} {path_arg}'
            if any(tn and tn in combined for tn in target_names | target_stems):
                return True
            if any(ext in combined.lower() for ext in (*target_exts, '.png', '.pdf', 'slide', 'document.xml', 'submission', 'sokoban')):
                return True
            if name in ('read', 'edit', 'write', 'write_file', 'run', 'bash') and combined.strip():
                # Inspecting or scripting on the task's single office workflow
                return True
    return False


def summarize_uncorrected_window(
    pkg: L1Package,
    facts: list[dict[str, Any]],
    kind: str,
    origin_turn: int,
    target_paths: list[str],
    target_exts: set[str],
) -> dict[str, Any]:
    """L2 v0.4 §4.5: Summarize what happened in the uncorrected window [origin_turn + 1, T]."""
    T = pkg.turns
    resolved_cfg = pkg.json('raw/resolved_config.json')
    max_requests = int((resolved_cfg.get('budget') or {}).get('max_model_requests') or T)
    remaining_at_origin = max(0, max_requests - origin_turn)
    termination = pkg.termination() or 'unknown'

    window_turns = list(range(origin_turn + 1, T + 1))
    readback_turns = 0
    preview_turns = 0
    submission_write_turns = 0
    target_modified_turns = 0
    other_file_modified_turns = 0
    inspect_preview_or_same_turns = 0

    target_names = {Path(p).name for p in target_paths if p}

    for t in window_turns:
        tf = [f for f in facts if f.get('turn') == t]
        is_rb = eval_predicate_on_turn(RULE_VOCABULARY['submission_readback'], tf, t)
        is_pv = eval_predicate_on_turn(RULE_VOCABULARY['preview_rendered'], tf, t)
        is_sub_w = eval_predicate_on_turn(RULE_VOCABULARY['submission_modified'], tf, t)
        if is_rb:
            readback_turns += 1
        if is_pv:
            preview_turns += 1
        if is_sub_w:
            submission_write_turns += 1

        transitions = [
            str((f.get('value') or {}).get('path') or '')
            for f in tf
            if f.get('type') == 'file_transition'
        ]
        office_transitions = [
            p for p in transitions
            if Path(p).suffix.lower() in target_exts and not Path(p).name.startswith('~$')
        ]
        if any(p in target_paths or Path(p).name in target_names or p.startswith('submission/') for p in office_transitions):
            target_modified_turns += 1
        elif office_transitions:
            other_file_modified_turns += 1

        if is_rb or is_pv or is_sub_w or _is_same_target_activity(tf, target_paths, target_exts):
            inspect_preview_or_same_turns += 1

    n_win = len(window_turns)
    kind_cn = '没交付' if kind == 'undelivered' else ('从没做对' if kind == 'never' else '改坏')
    term_cn = '请求耗尽' if termination == 'request_budget' else ('主动结束' if termination == 'assistant_finished' else termination)
    summary_text = (
        f'{kind_cn}：窗口 {n_win} 回合（c{origin_turn}..c{T}），'
        f'预览 {preview_turns} 次，回读 {readback_turns} 次，'
        f'{submission_write_turns} 次写入 submission/，'
        f'{target_modified_turns} 次改目标文件，'
        f'来源时剩 {remaining_at_origin} 次请求，{term_cn}'
    )
    return {
        'start_turn': origin_turn + 1,
        'end_turn': T,
        'window_turns': n_win,
        'preview_rendered_turns': preview_turns,
        'submission_readback_turns': readback_turns,
        'submission_write_turns': submission_write_turns,
        'target_file_modified_turns': target_modified_turns,
        'other_file_modified_turns': other_file_modified_turns,
        'inspect_preview_or_same_file_turns': inspect_preview_or_same_turns,
        'inspect_preview_or_same_file_ratio': (inspect_preview_or_same_turns / n_win) if n_win > 0 else 1.0,
        'remaining_requests_at_origin': remaining_at_origin,
        'termination': termination,
        'summary_text': summary_text,
    }


def build_defect_table(
    pkg: L1Package,
    grade_rows: list[dict[str, Any]],
    facts: list[dict[str, Any]],
    *,
    grades_path: Path | None = None,
    facts_path: Path | None = None,
    extra_runs: list[L1Package] | None = None,
) -> dict[str, Any]:
    """L2 v0.4 §4.3-4.5: Construct `defects.json` from snapshot grades and facts."""
    if not grade_rows or grade_rows[0].get('kind') != 'terminal_submission':
        raise L2Error('grade_rows must start with terminal_submission row')

    term_row = grade_rows[0]
    cp_rows = sorted([r for r in grade_rows[1:] if r.get('kind') == 'checkpoint'], key=lambda r: int(r['k']))
    if not cp_rows:
        raise L2Error('grade_rows must include checkpoint rows k=0..T')

    by_k = {int(r['k']): r for r in cp_rows}
    T = max(by_k)
    final_cp = by_k[T]

    Y = float(term_row['score'])
    s_T = float(final_cp['score'])
    max_s = max(float(r['score']) for r in cp_rows)
    k_star = min(int(r['k']) for r in cp_rows if abs(float(r['score']) - max_s) <= 1e-12)
    delivery_gap_g = max(0.0, max_s - Y)
    undelivered_gap = max(0.0, s_T - Y)
    peak_to_terminal_loss = max(0.0, max_s - s_T)

    desc_path = pkg.root / 'benchmark/description.json'
    desc = read_json(desc_path) if desc_path.is_file() else {}
    ev_info = desc.get('evaluation') or {}
    is_lhtb = (
        pkg.manifest.get('benchmark') == 'lhtb'
        or desc.get('benchmark') == 'lhtb'
        or desc.get('task_id') == 'sokoban'
        or ev_info.get('kind') == 'lhtb-verifier'
        or pkg.family_id == 'sokoban'
    )
    is_swebench_pro = (
        pkg.manifest.get('benchmark') == 'swebench_pro'
        or desc.get('benchmark') == 'swebench_pro'
        or ev_info.get('kind') == 'swebench_pro-verifier'
        or pkg.benchmark == 'swebench_pro'
        or 'swebench_pro' in str(pkg.family_id)
        or 'instance_ansible' in str(pkg.task_id)
    )
    if is_lhtb:
        target_exts = {'.log'}
    elif is_swebench_pro:
        target_exts = {'.diff', '.patch'}
    else:
        target_exts = set(OFFICE_EXTS)
    defects: list[dict[str, Any]] = []

    if Y < 1.0 - 1e-12:
        if final_cp.get('candidate_status') == 'no_candidate':
            # §4.3: "终态没有任何候选时，整次运行只记一项 no_artifact，扣分 1-Y：
            # 某个检查点有过候选记改坏，回合取最后一次候选消失的回合；否则记从没做对。"
            had_cand_ks = [k for k in range(T) if by_k[k].get('candidate_status') == 'ok']
            if had_cand_ks:
                last_ok_k = max(had_cand_ks)
                origin_j = last_ok_k + 1
                win = summarize_uncorrected_window(
                    pkg, facts, 'broken', origin_j, by_k[last_ok_k].get('candidate') or [], target_exts
                )
                defects.append({
                    'defect': NO_ARTIFACT_DEFECT_ID,
                    'deduction': round(1.0 - Y, 12),
                    'kind': 'broken',
                    'turn': origin_j,
                    'window': win,
                    'evidence_level': 'E1',
                })
            else:
                defects.append({
                    'defect': NO_ARTIFACT_DEFECT_ID,
                    'deduction': round(1.0 - Y, 12),
                    'kind': 'never',
                    'turn': None,
                    'window': None,
                    'evidence_level': 'E1',
                })
        else:
            # Terminal has a candidate (final_cp['candidate_status'] == 'ok').
            # Equation (230): 1 - Y = (s(T) - Y) [undelivered] + sum_{i: d_i(T) > 0} d_i(T) [broken / never]
            d_T: dict[str, float] = dict(final_cp.get('deductions') or {})
            d_sub: dict[str, float] = dict(term_row.get('deductions') or {})
            rubric = extract_task_rubric(pkg, extra_runs=extra_runs)
            rubric_weights = {r: md for r, md in rubric if md > 0}

            # 1. Undelivered defects (when s(T) > Y):
            # Items done right on final_cp (d_i(T) == 0) that were not done right in terminal submission/
            if undelivered_gap > 1e-12:
                # Origin turn k for undelivered: earliest k such that for all m in [k..T], s(m) >= s(T)
                origin_k = T
                for k_cand in range(T, -1, -1):
                    if float(by_k[k_cand]['score']) >= s_T - 1e-12:
                        origin_k = k_cand
                    else:
                        break
                win_undeliv = summarize_uncorrected_window(
                    pkg, facts, 'undelivered', origin_k, final_cp.get('candidate') or [], target_exts
                )
                undeliv_items: list[tuple[str, float]] = []
                for rule_id, d_val in d_T.items():
                    if rule_id == GATE_DEFECT_ID:
                        continue
                    if abs(d_val) <= 1e-12:
                        # Done right on final_cp! Check if lost in terminal submission
                        sub_loss = d_sub.get(rule_id)
                        if sub_loss is None:
                            # Terminal submission failed dim1 so all positive items were lost
                            w = rubric_weights.get(rule_id, 0.0)
                        else:
                            w = max(0.0, sub_loss)
                        if w > 1e-12:
                            undeliv_items.append((rule_id, w))
                if not undeliv_items:
                    undeliv_items = [(GATE_DEFECT_ID, undelivered_gap)]
                w_sum = sum(w for _, w in undeliv_items)
                for rule_id, w in undeliv_items:
                    ded = undelivered_gap * (w / w_sum) if w_sum > 0 else undelivered_gap / len(undeliv_items)
                    defects.append({
                        'defect': rule_id,
                        'deduction': round(ded, 12),
                        'kind': 'undelivered',
                        'turn': origin_k,
                        'window': win_undeliv,
                        'evidence_level': 'E1',
                    })

            # 2. Broken / Never defects on final_cp (items with d_i(T) > 0):
            # If final_cp had no dim2 breakdown (only dim1_gate: 1.0) and task rubric is known,
            # decompose across positive rubric items so defect IDs align with Call 1 and peer runs.
            pos_d_T = {k: v for k, v in d_T.items() if v > 1e-12}
            if set(pos_d_T.keys()) == {GATE_DEFECT_ID} and rubric_weights:
                m_total = sum(rubric_weights.values())
                rem_loss = 1.0 - s_T
                pos_d_T = {r: rem_loss * (md / m_total) for r, md in rubric_weights.items()}

            pos_sum = sum(pos_d_T.values())
            rem_target = max(0.0, 1.0 - s_T)
            scale = (rem_target / pos_sum) if (pos_sum > 0 and abs(pos_sum - rem_target) > 1e-9) else 1.0

            for rule_id, raw_d in pos_d_T.items():
                ded = raw_d * scale
                if ded <= 1e-12:
                    continue
                # Check if rule_id was ever done right (d_i(k) == 0) at some checkpoint k < T
                ok_ks: list[int] = []
                for k in range(T):
                    r_k = by_k[k]
                    if r_k.get('candidate_status') != 'ok':
                        continue
                    ded_k = r_k.get('deductions') or {}
                    if rule_id in ded_k and abs(float(ded_k[rule_id])) <= 1e-12:
                        ok_ks.append(k)
                if ok_ks:
                    # §4.3: "j：最后一次由对变错的回合，即 d_i(j-1)=0 而 d_i(j)>0"
                    last_ok_k = max(ok_ks)
                    origin_j = last_ok_k + 1
                    win_broken = summarize_uncorrected_window(
                        pkg, facts, 'broken', origin_j, final_cp.get('candidate') or [], target_exts
                    )
                    defects.append({
                        'defect': rule_id,
                        'deduction': round(ded, 12),
                        'kind': 'broken',
                        'turn': origin_j,
                        'window': win_broken,
                        'evidence_level': 'E1',
                    })
                else:
                    defects.append({
                        'defect': rule_id,
                        'deduction': round(ded, 12),
                        'kind': 'never',
                        'turn': None,
                        'window': None,
                        'evidence_level': 'E1',
                    })

    total_deduction = sum(float(d['deduction']) for d in defects)
    by_kind = {
        k: round(sum(float(d['deduction']) for d in defects if d['kind'] == k), 12)
        for k in ('undelivered', 'broken', 'never')
    }
    return {
        'schema_version': CODE_VERSION,
        'code_version': CODE_VERSION,
        'episode_id': pkg.episode_id,
        'task_id': pkg.family_id,
        'episode_index_sha256': pkg.index_sha256,
        'grades_sha256': sha_file(grades_path) if grades_path and grades_path.is_file() else None,
        'facts_sha256': sha_file(facts_path) if facts_path and facts_path.is_file() else None,
        'terminal_score_y': Y,
        'terminal_snapshot_score_s_T': s_T,
        'max_snapshot_score': max_s,
        'k_star': k_star,
        'delivery_gap_g': round(delivery_gap_g, 12),
        'undelivered_gap': round(undelivered_gap, 12),
        'peak_to_terminal_loss': round(peak_to_terminal_loss, 12),
        'total_deduction': round(total_deduction, 12),
        'deduction_by_kind': by_kind,
        'defects': defects,
    }


def _summarize_run_for_contrast(pkg: L1Package, facts: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize a run's execution pattern and official terminal item results (never using s(k))."""
    T = pkg.turns
    resolved_cfg = pkg.json('raw/resolved_config.json')
    max_requests = int((resolved_cfg.get('budget') or {}).get('max_model_requests') or T)
    og = pkg.json('official_grade.json') if (pkg.root / 'official_grade.json').is_file() else {}
    proj = og.get('projection') or {}
    raw = og.get('official_raw') if isinstance(og.get('official_raw'), dict) else {}
    y = float(proj['completion_rate']) if isinstance(proj.get('completion_rate'), (int, float)) else 0.0
    dim1_pass = _extract_dim1_pass(raw, proj) and raw.get('status') == 'ok'

    sub_turns = [
        t for t in range(1, T + 1)
        if eval_predicate_on_turn(RULE_VOCABULARY['submission_modified'], facts, t)
    ]
    preview_turns = [
        t for t in range(1, T + 1)
        if eval_predicate_on_turn(RULE_VOCABULARY['preview_rendered'], facts, t)
    ]
    readback_turns = [
        t for t in range(1, T + 1)
        if eval_predicate_on_turn(RULE_VOCABULARY['submission_readback'], facts, t)
    ]
    last_sub_turn = max(sub_turns) if sub_turns else None
    rem_at_last_sub = (max_requests - last_sub_turn) if last_sub_turn is not None else None

    passed_rules: set[str] = set()
    if dim1_pass and isinstance(raw.get('dim2_items'), list):
        for it in raw['dim2_items']:
            max_d = max(0.0, float(it.get('max_delta', 0.0)))
            delta = float(it.get('delta', 0.0))
            if abs(max_d - delta) <= 1e-12:
                passed_rules.add(str(it['rule']))
    elif isinstance(raw.get('test_results'), dict) and isinstance(raw['test_results'].get('tests'), list):
        for t in raw['test_results']['tests']:
            if isinstance(t, dict) and t.get('name') and t.get('status') == 'PASSED':
                passed_rules.add(t['name'])

    return {
        'episode_id': pkg.episode_id,
        'terminal_score_y': y,
        'delivered_valid': dim1_pass,
        'turns_used': T,
        'max_model_requests': max_requests,
        'submission_write_turns': sub_turns,
        'last_submission_turn': last_sub_turn,
        'remaining_requests_at_last_submission': rem_at_last_sub,
        'preview_rendered_count': len(preview_turns),
        'submission_readback_count': len(readback_turns),
        'termination': pkg.termination(),
        'passed_rules': sorted(passed_rules),
    }


def build_contrast(
    pkg: L1Package,
    facts: list[dict[str, Any]],
    defect_ids: list[str],
    peer_runs: list[tuple[L1Package, list[dict[str, Any]]]],
    *,
    facts_path: Path | None = None,
) -> dict[str, Any]:
    """L2 v0.4 §4.6: Cross-run contrast on the same task (excluding own branches)."""
    valid_peers: list[dict[str, Any]] = []
    for peer_pkg, peer_facts in peer_runs:
        if peer_pkg.episode_id == pkg.episode_id:
            continue
        lineage = peer_pkg.manifest.get('lineage') or {}
        if lineage.get('parent_episode_id') == pkg.episode_id or lineage.get('parent_episode_id') is not None:
            # Never include branches in cross-run contrast
            continue
        if peer_pkg.family_id != pkg.family_id:
            continue
        valid_peers.append(_summarize_run_for_contrast(peer_pkg, peer_facts))

    self_summary = _summarize_run_for_contrast(pkg, facts)
    by_defect: dict[str, dict[str, Any]] = {}
    for d_id in defect_ids:
        if d_id == NO_ARTIFACT_DEFECT_ID:
            succeeded = [p for p in valid_peers if p['delivered_valid']]
        else:
            succeeded = [p for p in valid_peers if d_id in p['passed_rules']]
        failed = [p for p in valid_peers if p not in succeeded]
        by_defect[d_id] = {
            'defect': d_id,
            'peer_total': len(valid_peers),
            'peer_succeeded_count': len(succeeded),
            'peer_failed_count': len(failed),
            'succeeded_peers': [
                {
                    'episode_id': p['episode_id'],
                    'terminal_score_y': p['terminal_score_y'],
                    'turns_used': p['turns_used'],
                    'last_submission_turn': p['last_submission_turn'],
                    'remaining_requests_at_last_submission': p['remaining_requests_at_last_submission'],
                    'preview_rendered_count': p['preview_rendered_count'],
                    'termination': p['termination'],
                }
                for p in succeeded
            ],
        }

    return {
        'schema_version': CODE_VERSION,
        'code_version': CODE_VERSION,
        'episode_id': pkg.episode_id,
        'task_id': pkg.family_id,
        'episode_index_sha256': pkg.index_sha256,
        'facts_sha256': sha_file(facts_path) if facts_path and facts_path.is_file() else None,
        'self_execution_summary': {k: v for k, v in self_summary.items() if k != 'passed_rules'},
        'peer_count': len(valid_peers),
        'by_defect': by_defect,
    }


def write_defects(out_dir: Path, payload: dict[str, Any]) -> Path:
    path = out_dir / 'defects.json'
    write_json(path, payload)
    return path


def write_contrast(out_dir: Path, payload: dict[str, Any]) -> Path:
    path = out_dir / 'contrast.json'
    write_json(path, payload)
    return path
