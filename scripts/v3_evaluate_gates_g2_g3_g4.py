#!/usr/bin/env python3
"""Read-only V3 R19–R21 evidence audits. No execution or scoring is performed."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
L1_ROOT = REPO_ROOT / 'L1_sandbox' / 'C3_Experimental_Sandbox'
L2_ROOT = REPO_ROOT / 'L2_Behavioral Event Localization' / 'L2_Sandbox'
sys.path[:0] = [str(L1_ROOT / 'src'), str(L2_ROOT / 'src')]
from c3_sandbox.l1.evaluation.study import study_outcome
from l2_sandbox.evidence.l1 import L1Package
from l2_sandbox.io import digest, loads, sha_file
from l2_sandbox.v05 import V05_SCHEMA_VERSION

OUT_JSON = L2_ROOT / 'reports/v3_gates_g2_g3_g4_repaired_audit_20261017.json'
OUT_G3_SHEET_MD = L2_ROOT / 'docs/V3_G3_NEW_DEV45_MANUAL_AUDIT_SHEET_REPAIRED_20261017.md'
COHORT = 'v3_qwen36_pi_dev45_20261017'


def _load_json(path: Path) -> dict[str, Any]:
    value = loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f'Expected JSON object: {path}')
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [loads(line) for line in path.read_text().splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f'Expected JSONL objects: {path}')
    return rows


def _display(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def _verify_cp000000_before_first_request(ep_dir: Path) -> dict[str, Any]:
    try:
        pkg = L1Package(ep_dir)
        cp = pkg.validate_resume_checkpoint(0)
        meta = pkg.json(cp['path'])
        messages = (meta.get('conversation') or {}).get('messages')
        snap = pkg.json(f"states/snapshots/{cp['snapshot_id']}.json")
        budget = meta.get('budget_used') or {}
        valid = (meta.get('episode_id') == pkg.episode_id
                 and meta.get('checkpoint_id') == 'cp000000'
                 and meta.get('config_sha256') == pkg.manifest.get('config_sha256')
                 and budget.get('model_requests') == 0 and budget.get('tool_calls') == 0
                 and isinstance(messages, list) and bool(messages)
                 and all(isinstance(m, dict) and m.get('role') in ('system', 'user') for m in messages)
                 and not meta.get('pending_calls') and not meta.get('pending_observations')
                 and snap.get('label') == 'initial' and snap.get('snapshot_id') == cp['snapshot_id'])
        return {'exists': True, 'checkpoint_id': meta['checkpoint_id'],
                'snapshot_id': cp['snapshot_id'], 'checkpoint_sha256': cp['sha256'],
                'pre_first_request_valid': valid}
    except Exception as exc:
        return {'exists': (ep_dir / 'checkpoints/cp000000.json').exists(),
                'pre_first_request_valid': False, 'reason': str(exc)}


def _bound_pair(parent_dir: Path, child_dir: Path) -> dict[str, Any]:
    parent, child = L1Package(parent_dir), L1Package(child_dir)
    child.validate_branch(parent, 0)
    if parent.manifest.get('lineage') is not None:
        raise ValueError('fresh parent has lineage')
    if not parent.family_id or parent.family_id == 'unknown:unknown_task':
        raise ValueError('missing task identity')
    for key in ('task', 'agent'):
        if not parent.manifest.get(key) or child.manifest.get(key) != parent.manifest[key]:
            raise ValueError(f'{key} identity mismatch')
    for key in ('adapter', 'image_id', 'architecture', 'network', 'resource_limits'):
        expected = (parent.manifest.get('runtime') or {}).get(key)
        if expected is None or (child.manifest.get('runtime') or {}).get(key) != expected:
            raise ValueError(f'runtime {key} identity mismatch')
    cp0 = _verify_cp000000_before_first_request(parent_dir)
    if not cp0['pre_first_request_valid']:
        raise ValueError('parent checkpoint is not actual pre-request c0')
    # The branch must start from exactly the recorded workspace and native conversation.
    parent_cp = parent.json('checkpoints/cp000000.json')
    child_cp = child.json('checkpoints/cp000000.json')
    if any((child_cp.get('budget_used') or {}).get(k) != 0 for k in ('model_requests', 'tool_calls')):
        raise ValueError('child c0 budget is not zero')
    for key in ('conversation', 'pending_calls', 'pending_observations'):
        if child_cp.get(key) != parent_cp.get(key):
            raise ValueError(f'child c0 {key} mismatch')
    p_snap = parent.json(f"states/snapshots/{parent_cp['snapshot_id']}.json")
    c_snap = child.json(f"states/snapshots/{child_cp['snapshot_id']}.json")
    for key in ('files', 'directories', 'content_sha256'):
        if key not in p_snap or c_snap.get(key) != p_snap[key]:
            raise ValueError(f'child initial workspace {key} mismatch')
    p, c = study_outcome(parent_dir, verify_evidence=True), study_outcome(child_dir, verify_evidence=True)
    if not p.get('study_valid') or not c.get('study_valid'):
        raise ValueError('invalid outcome or evidence seal')
    yp, yc = p.get('study_outcome_y'), c.get('study_outcome_y')
    if any(type(y) not in (int, float) or not math.isfinite(y) or not 0 <= y <= 1 for y in (yp, yc)):
        raise ValueError('invalid bounded outcome')
    return {'run_label': parent_dir.name, 'task_id': parent.family_id,
            'parent_episode_id': parent.episode_id, 'child_episode_id': child.episode_id,
            'parent_index_sha256': parent.index_sha256, 'child_index_sha256': child.index_sha256,
            'checkpoint_sha256': cp0['checkpoint_sha256'], 'y_fresh': yp, 'y_c0_resume': yc,
            'diff_resume_minus_fresh': yc - yp, 'cp0_pre_first_request_valid': True}


def _cluster_interval(pairs: list[dict]) -> tuple[float | None, list[float] | None, int]:
    clusters: dict[str, list[float]] = defaultdict(list)
    for pair in pairs:
        clusters[pair['task_id']].append(pair['diff_resume_minus_fresh'])
    means = [sum(values) / len(values) for values in clusters.values()]
    if not means:
        return None, None, 0
    mean = sum(means) / len(means)
    # Independent task clusters; each cluster mean lies in [-1,1]. Unlike a percentile
    # bootstrap this finite-sample bound cannot collapse to [0,0] at identical outcomes.
    radius = math.sqrt(2 * math.log(2 / .05) / len(means))
    return mean, [max(-1., mean - radius), min(1., mean + radius)], len(means)


def evaluate_r19_g2(l1_runs_root: Path) -> dict[str, Any]:
    base = l1_runs_root / COHORT
    parents, children = base / 'episodes', base / 'v3_g2_c0_resumes'
    checks = [{'episode_id': p.name, **_verify_cp000000_before_first_request(p)}
              for p in sorted(parents.glob('*')) if (p / 'manifest.json').exists()]
    pairs, rejected = [], []
    seen_children, seen_parents = set(), set()
    for child in sorted(children.glob('*')):
        if not (child / 'manifest.json').exists():
            continue
        try:
            # Directory names are discovery hints only; all identities are verified below.
            parent = parents / child.name.removesuffix('_c0')
            pair = _bound_pair(parent, child)
            if pair['parent_episode_id'] in seen_parents or pair['child_episode_id'] in seen_children:
                raise ValueError('duplicate evidence identity')
            seen_parents.add(pair['parent_episode_id']); seen_children.add(pair['child_episode_id'])
            pairs.append(pair)
        except Exception as exc:
            rejected.append({'child': _display(child), 'reason': str(exc)})
    mean, ci, n_tasks = _cluster_interval(pairs)
    inside = ci is not None and ci[0] > -.05 and ci[1] < .05
    all_cp0 = bool(checks) and all(c['pre_first_request_valid'] for c in checks)
    return {'gate_id': 'V3-G2', 'work_item': 'R19', 'target_pairs': 20,
            'cp000000_pre_first_request_verified_count': sum(c['pre_first_request_valid'] for c in checks),
            'cp000000_all_pre_first_request_valid': all_cp0, 'cp0_checks': checks,
            'completed_valid_pairs': len(pairs), 'independent_task_clusters': n_tasks,
            'mean_diff_resume_minus_fresh': mean, 'ci95_diff_resume_minus_fresh': ci,
            'ci_method': 'equal_task_cluster_Hoeffding_bounded_minus1_plus1',
            'estimand': 'equal-task mean expected score difference; not outcome-distribution equivalence',
            'independence_assumption': 'independent sampled tasks; repeats within task are not independent',
            'equivalence_margin': .05, 'ci95_completely_inside_pm_005': inside,
            'status': 'pass' if len(pairs) >= 20 and all_cp0 and inside else 'undecided_pending_pairs_or_ci',
            'pairs': pairs, 'rejected_pairs': rejected}


def _checkpoint_record(ep: Path, row: dict) -> dict:
    cp_id = row.get('checkpoint_id') or row.get('checkpoint')
    if not isinstance(cp_id, str) or not re.fullmatch(r'cp\d{6}', cp_id):
        raise ValueError('missing actual checkpoint identity')
    cp_path = ep / 'checkpoints' / f'{cp_id}.json'
    cp = _load_json(cp_path)
    snap_id = cp.get('snapshot_id')
    if not isinstance(snap_id, str) or not re.fullmatch(r's\d{6}', snap_id):
        raise ValueError('missing actual snapshot identity')
    snap_path = ep / 'states/snapshots' / f'{snap_id}.json'
    snap = _load_json(snap_path)
    if snap.get('snapshot_id') != snap_id:
        raise ValueError('snapshot identity mismatch')
    return {'episode_id': ep.name, 'checkpoint_id': cp_id, 'k': int(cp_id[2:]),
            'checkpoint_path': _display(cp_path), 'checkpoint_sha256': sha_file(cp_path),
            'snapshot_id': snap_id, 'snapshot_path': _display(snap_path),
            'snapshot_sha256': sha_file(snap_path), 'snapshot_content_sha256': snap.get('content_sha256'),
            'grade_snapshot_matches_metadata': row.get('snapshot_id') == snap_id,
            'candidate': row.get('candidate'), 'candidate_status': row.get('candidate_status'),
            'auto_snapshot_score': row.get('score'),
            'restored_directory': row.get('restored_directory'),
            'scorer_exit_code': row.get('scorer_exit_code'),
            'reviewer': '', 'reviewed_at': '', 'human_conclusion': '', 'notes': ''}


def evaluate_r20_g3(l1_runs_root: Path, diag_root: Path, *, sheet_path: Path | None = None) -> dict[str, Any]:
    pool, errors = [], []
    matches = total = 0
    for ep in sorted((l1_runs_root / COHORT / 'episodes').glob('*')):
        if not (ep / 'manifest.json').exists():
            continue
        try:
            rows = _load_jsonl(diag_root / COHORT / ep.name / 'grades.jsonl')
            cps = [r for r in rows if r.get('kind') == 'checkpoint']
            st = study_outcome(ep)
            if cps and st.get('study_valid') and type(cps[-1].get('score')) in (float, int):
                total += 1
                matches += abs(cps[-1]['score'] - st['study_outcome_y']) <= 1e-6
            if not cps:
                cps = [{'checkpoint_id': p.stem} for p in sorted((ep / 'checkpoints').glob('cp*.json'))]
            for row in cps:
                try:
                    record = _checkpoint_record(ep, row)
                    grade_path = diag_root / COHORT / ep.name / 'grades.jsonl'
                    record['grade_source_path'] = _display(grade_path) if grade_path.exists() else None
                    record['grade_source_sha256'] = sha_file(grade_path) if grade_path.exists() else None
                    pool.append(record)
                except Exception as exc:
                    errors.append({'episode': ep.name, 'checkpoint': row.get('checkpoint_id'), 'error': str(exc)})
        except Exception as exc:
            errors.append({'episode': ep.name, 'error': str(exc)})
    sample = random.Random(20261017).sample(pool, min(20, len(pool)))
    sample.sort(key=lambda r: (r['episode_id'], r['k']))
    target = sheet_path or OUT_G3_SHEET_MD
    if target.exists():
        raise FileExistsError(f'Preserve existing human dossier; choose a new --output-dir: {target}')
    fields = ['episode_id', 'checkpoint_id', 'snapshot_id', 'snapshot_path', 'snapshot_sha256',
              'candidate', 'candidate_status', 'auto_snapshot_score', 'restored_directory',
              'scorer_exit_code', 'reviewer', 'reviewed_at', 'human_conclusion', 'notes']
    lines = ['# V3-G3 checkpoint dossier (R20)', '',
             'Pending human review. Unknown restored directories/scorer exits remain unknown; no scorer was run.', '',
             '| # | ' + ' | '.join(fields) + ' |', '|---|' + '|'.join('---' for _ in fields) + '|']
    for i, row in enumerate(sample, 1):
        values = ['unknown' if row[f] is None else str(row[f]).replace('|', '\\|').replace('\n', ' ') for f in fields]
        lines.append(f'| {i} | ' + ' | '.join(values) + ' |')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('\n'.join(lines) + '\n')
    return {'gate_id': 'V3-G3', 'work_item': 'R20',
            'auto_terminal_consistency': {'new_dev45_graded_episodes': total,
                'new_dev45_matched_episodes': matches, 'match_rate': matches / total if total else None},
            'manual_inspection_sheet_path': _display(target), 'checkpoint_pool_size': len(pool),
            'sampled_checkpoint_count': len(sample), 'sampled_checkpoints': sample,
            'missing_restored_directory_count': sum(r['restored_directory'] is None for r in sample),
            'missing_scorer_exit_count': sum(r['scorer_exit_code'] is None for r in sample),
            'identity_errors': errors, 'human_review_completed_count': 0,
            'status': ('pending_human_review_with_missing_execution_metadata'
                       if any(r['restored_directory'] is None or r['scorer_exit_code'] is None for r in sample)
                       else 'pending_human_review')}


def audit_request_boundary(payload: dict, *, public_messages: list[dict], private_corpus: list[str], _memo: dict | None = None) -> dict:
    """Check external instruction additions against public transcript and private corpus.

    This lexical audit is never a proof of semantic absence or authenticated tool history.
    Public task text is allowed even when it also appears in a private rubric.
    """
    messages = payload.get('messages') if isinstance(payload, dict) else None
    if not isinstance(messages, list) or not messages or any(not isinstance(m, dict) for m in messages):
        return {'status': 'unverified', 'reason': 'malformed_or_unsupported_messages', 'matches': []}
    if not public_messages or not private_corpus:
        return {'status': 'unverified', 'reason': 'missing_public_or_private_corpus', 'matches': []}
    public = {digest(m) for m in public_messages}
    matches, additions, external_private_matches = [], [], []
    for m in messages:
        if digest(m) in public:
            continue
        key = digest(m)
        if _memo is not None and key in _memo:
            overlaps = _memo[key]
        else:
            text = '\n'.join(_strings(m))
            # Actual private clauses/lines, including numeric/formula rules.
            overlaps = [item[:100] for item in private_corpus if len(item.strip()) >= 12 and item in text]
            if _memo is not None:
                _memo[key] = overlaps
        matches.extend(overlaps)
        if m.get('role') in ('system', 'user', 'developer'):
            additions.append(m.get('role'))
            external_private_matches.extend(overlaps)
        elif m.get('role') not in ('assistant', 'tool'):
            return {'status': 'unverified', 'reason': 'unsupported_message_role', 'matches': matches}
    return {'status': 'suspected_leakage' if external_private_matches else 'unverified',
            'reason': ('private_overlap_in_unapproved_external_instruction' if external_private_matches else
                       'external_instruction_or_tool_bridge_provenance_unverified' if additions else
                       'private_overlap_in_unverified_generated_history' if matches else
                       'lexical_scan_only_tool_and_assistant_provenance_not_established'),
            'matches': sorted(set(matches)), 'unapproved_instruction_roles': additions}


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value, *value.splitlines()]
    if isinstance(value, dict):
        return [s for k, v in value.items() for s in [str(k), *_strings(v)]]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def _public_messages(ep: Path) -> list[dict]:
    """Bind the public instruction to the public task, not an arbitrary c0 user message."""
    task = _load_json(ep / 'benchmark/public_task.json')
    cfg = _load_json(ep / 'raw/resolved_config.json')
    prompt = task['prompt'] + '\n\nInput files in /workspace/input:\n' + '\n'.join(a['name'] for a in task['assets'])
    expected = [{'role': 'system', 'content': cfg['agent']['system_prompt']},
                {'role': 'user', 'content': prompt}]
    actual = (_load_json(ep / 'checkpoints/cp000000.json').get('conversation') or {}).get('messages')
    if actual != expected:
        raise ValueError('c0 public transcript differs from public task and frozen harness config')
    return expected


def _private_corpus(ep: Path, diag: Path) -> list[str]:
    parts = []
    rubric = ep / 'benchmark/rubric.json'
    if rubric.exists():
        parts.extend(_strings(_load_json(rubric)))
    task = _load_json(ep / 'manifest.json').get('task') or {}
    entry = (task.get('evaluation') or {}).get('entrypoint')
    if isinstance(entry, str):
        source = ep / 'benchmark/source' / entry
        if source.is_file() and source.resolve().is_relative_to(ep.resolve()):
            parts.extend(source.read_text().splitlines())
    if diag.exists():
        for p in diag.glob('*.json'):
            parts.extend(_strings(_load_json(p)))
    return sorted({s.strip() for s in parts if len(s.strip()) >= 12})


def _cache_stats(diag_root: Path) -> dict:
    total = no_candidate = candidates = unknown = 0
    explicit_hits = explicit_misses = cache_unknown = 0
    elapsed, errors = [], []
    for cohort in (COHORT, 'swebench_pro_f27_live_20261017', 'lhtb_sokoban_qwen36'):
        for path in sorted((diag_root / cohort).glob('*/grades.jsonl')):
            try:
                rows = _load_jsonl(path)
            except Exception as exc:
                errors.append({'file': _display(path), 'error': str(exc)}); continue
            for row in rows:
                if 'grading_elapsed_seconds' in row:
                    t = row['grading_elapsed_seconds']
                    if type(t) in (int, float) and math.isfinite(t) and t >= 0:
                        elapsed.append({'file': _display(path), 'seconds': t})
                if row.get('kind') != 'checkpoint':
                    continue
                total += 1
                if row.get('candidate_status') == 'no_candidate' and row.get('candidate') == [] and row.get('candidates_evaluated') == 0:
                    no_candidate += 1
                elif isinstance(row.get('candidate'), list) and row['candidate']:
                    candidates += 1
                    hit = row.get('cache_hit')
                    if hit is True: explicit_hits += 1
                    elif hit is False: explicit_misses += 1
                    else: cache_unknown += 1
                else:
                    unknown += 1
    return {'total_checkpoints_graded_in_diag_dirs': total,
            'no_candidate_instant_zero_checkpoints': no_candidate,
            'nonempty_candidate_checkpoints': candidates, 'unknown_candidate_checkpoints': unknown,
            'explicit_cache_hit_checkpoints': explicit_hits, 'explicit_cache_miss_checkpoints': explicit_misses,
            'candidate_cache_status_unknown': cache_unknown,
            'unique_candidate_verifier_invocations': None, 'sha256_dedup_cache_hits': None,
            'effective_verifier_skip_rate': None,
            'reason_unknown': 'grade schema does not retain per-candidate cache keys/hits or invocation history',
            'grading_elapsed_seconds_observed_total': sum(r['seconds'] for r in elapsed) if elapsed else None,
            'grading_timing_records': elapsed, 'errors': errors}


def evaluate_r21_g4(l1_runs_root: Path, diag_root: Path) -> dict[str, Any]:
    requests = episodes = 0
    findings, errors, excluded = [], [], []
    cohorts = [('qwen36_pi_p1_pilot_20261017', 'episodes'),
               ('qwen36_pi_p1_pilot_20261017', 'v3_g1_cold_resume'),
               (COHORT, 'episodes'), (COHORT, 'v3_g2_c0_resumes')]
    for cohort, subdir in cohorts:
        for ep in sorted((l1_runs_root / cohort / subdir).glob('*')):
            if not (ep / 'manifest.json').exists(): continue
            try:
                cfg = _load_json(ep / 'raw/resolved_config.json')
                intervention = (cfg.get('agent') or {}).get('intervention')
                if intervention is not None:
                    excluded.append({'episode': _display(ep), 'reason': 'configured_intervention_outside_normal_or_P_scope'})
                    continue
            except Exception as exc:
                errors.append({'episode': ep.name, 'error': f'unknown_arm_configuration:{exc}'})
            episodes += 1
            reqs = sorted((ep / 'raw/model').glob('r*/request.json'))
            if not reqs:
                errors.append({'episode': ep.name, 'error': 'no_request_evidence'})
            try:
                public = _public_messages(ep)
                corpus = _private_corpus(ep, diag_root / cohort / ep.name)
            except Exception as exc:
                public, corpus = [], []
                errors.append({'episode': ep.name, 'error': str(exc)})
            message_memo: dict = {}
            for req in reqs:
                requests += 1
                try:
                    check = audit_request_boundary(_load_json(req), public_messages=public, private_corpus=corpus, _memo=message_memo)
                except Exception as exc:
                    check = {'status': 'unverified', 'reason': f'malformed_request:{exc}', 'matches': []}
                findings.append({'file': _display(req), **check})
    suspected = [r for r in findings if r['status'] == 'suspected_leakage']
    unverified = [r for r in findings if r['status'] == 'unverified']
    return {'gate_id': 'V3-G4', 'work_item': 'R21', 'episodes_scanned': episodes,
            'normal_and_p_requests_scanned': requests, 'leaked_request_count': None, 'suspected_leakage_request_count': len(suspected),
            'suspected_leakage_requests': suspected, 'unverified_request_count': len(unverified),
            'request_findings': findings, 'evidence_errors': errors, 'excluded_intervention_episodes': excluded,
            'zero_diagnostic_leakage_verified_on_scanned': False,
            'scan_scope': 'public c0 instruction allowlist versus task-private rubric/verifier/diagnosis text; lexical only',
            'snapshot_grading_cache_stats': _cache_stats(diag_root),
            'status': 'suspected_leakage_requires_review' if suspected else 'unverified_evidence_boundary'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    output = args.output_dir / OUT_JSON.name if args.output_dir else OUT_JSON
    sheet = args.output_dir / OUT_G3_SHEET_MD.name if args.output_dir else OUT_G3_SHEET_MD
    if output.exists() or sheet.exists():
        parser.error('Output already exists; choose a new output directory to preserve review history')
    runs, diagnostics = L1_ROOT / 'runs', L2_ROOT / 'diagnoses'
    payload = {'schema_version': V05_SCHEMA_VERSION, 'generated_at': datetime.now(timezone.utc).isoformat(),
               'source_script_sha256': sha_file(Path(__file__)),
               'V3_G2_R19': evaluate_r19_g2(runs),
               'V3_G3_R20': evaluate_r20_g3(runs, diagnostics, sheet_path=sheet),
               'V3_G4_R21': evaluate_r21_g4(runs, diagnostics)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    print(_display(output))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
