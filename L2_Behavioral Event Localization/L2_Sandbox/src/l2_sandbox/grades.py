"""Checkpoint snapshot grading s(k) and per-item deductions d_i(k) (L2 v0.4 §4.1-4.2, F23).

Uses L1's StateStore.restore and DockerEvaluator.evaluate + officeval.interpret.
Caches grading results by candidate content sha256 so repeated or identical
snapshots across checkpoints and runs require zero redundant Docker calls.
Zero Actor model requests.
"""
from __future__ import annotations
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any

try:
    from c3_sandbox.l1.core.contracts import EvaluationSpec
except ImportError:
    _l1_src = Path(__file__).resolve().parents[4] / 'L1_sandbox' / 'C3_Experimental_Sandbox' / 'src'
    if _l1_src.is_dir() and str(_l1_src) not in sys.path:
        sys.path.insert(0, str(_l1_src))
    from c3_sandbox.l1.core.contracts import EvaluationSpec
from c3_sandbox.l1.evidence.live import validate_snapshot
from c3_sandbox.l1.evaluation.evaluator import DockerEvaluator
from c3_sandbox.l1.recording.state import restore
from c3_sandbox.l1.storage import file_digest

from .config import CODE_VERSION
from .evidence.l1 import L1Package
from .io import L2Error, digest, read_json, write_json, write_jsonl

OFFICE_EXTS = {'.docx', '.pptx', '.xlsx', '.xlsm', '.xls'}
GATE_DEFECT_ID = 'dim1_gate'
NO_ARTIFACT_DEFECT_ID = 'no_artifact'


def _office_targets(pkg: L1Package) -> tuple[list[str], set[str]]:
    pub_path = pkg.root / 'benchmark/public_task.json'
    assets: list[str] = []
    if pub_path.is_file():
        pub = read_json(pub_path)
        for a in pub.get('assets') or []:
            name = a.get('name', '')
            if Path(name).suffix.lower() in OFFICE_EXTS:
                assets.append(name)
    exts = {Path(n).suffix.lower() for n in assets}
    prompt = str((pkg.manifest.get('task') or {}).get('prompt', '')).lower()
    if not exts:
        for ext in ('.docx', '.pptx', '.xlsx', '.xlsm'):
            if ext in prompt:
                exts.add(ext)
    if not exts:
        exts = set(OFFICE_EXTS)
    if '.xlsx' in exts or '.xls' in exts:
        exts |= {'.xlsx', '.xlsm', '.xls'}
    return assets, exts


def _matches_target(file_name: str, target_name: str, all_targets: list[str]) -> bool:
    if file_name == target_name:
        return True
    t_stem = Path(target_name).stem
    f_stem = Path(file_name).stem
    if t_stem and t_stem in f_stem:
        return True
    # Keyword matching for known multi-file patterns (e.g. 报告 vs 鉴定, 表1/2/3)
    for kw in ('报告', '鉴定', '1', '2', '3'):
        if kw in t_stem and kw in f_stem:
            # Make sure no other target also shares the same keyword unless identical
            others = [t for t in all_targets if t != target_name and kw in Path(t).stem]
            if not others:
                return True
    return False


def compute_item_deductions(
    official_raw: dict[str, Any] | None,
    completion_rate: float | None,
    fallback_rubric: list[ tuple[str, float] ] | None = None,
) -> tuple[dict[str, float], float]:
    """L2 v0.4 §4.1: Allocate 1 - completion_rate across dim2 items and dim1_gate.

    Returns (deductions_by_defect_id, max_score_M).
    Sum of deductions_by_defect_id.values() is identically equal to 1 - score.
    """
    score = float(completion_rate) if isinstance(completion_rate, (int, float)) else 0.0
    raw = official_raw if isinstance(official_raw, dict) else {}
    status_ok = raw.get('status') == 'ok' and raw.get('error') is None
    items = raw.get('dim2_items') if isinstance(raw.get('dim2_items'), list) and status_ok else []

    if items:
        m_pos = sum(max(0.0, float(it.get('max_delta', 0.0))) for it in items)
        max_score = m_pos if m_pos > 0 else float(raw.get('max_score') or 1.0)
        effective_total = score * max_score
        deductions: dict[str, float] = {}
        delta_sum = 0.0
        for it in items:
            rule = str(it['rule'])
            max_d = max(0.0, float(it.get('max_delta', 0.0)))
            delta = float(it.get('delta', 0.0))
            delta_sum += delta
            d_i = (max_d - delta) / max_score
            deductions[rule] = deductions.get(rule, 0.0) + d_i
        d_gate = (delta_sum - effective_total) / max_score
        if abs(d_gate) > 1e-12:
            deductions[GATE_DEFECT_ID] = d_gate
        return deductions, max_score

    # No dim2 breakdown (either status != 'ok' or verifier returned empty dim2_items):
    # §4.1: "评分没有 dim2 明细时，全部扣分记在门槛项。"
    max_score = float(raw.get('max_score') or 0.0)
    if max_score <= 0 and fallback_rubric:
        max_score = sum(max(0.0, md) for _, md in fallback_rubric)
    if max_score <= 0:
        max_score = 1.0
    return {GATE_DEFECT_ID: 1.0 - score}, max_score


def extract_task_rubric(pkg: L1Package, extra_runs: list[L1Package] | None = None) -> list[tuple[str, float]]:
    """Extract (rule_text, max_delta) list for the task from official_grade or peer runs."""
    for candidate_pkg in [pkg, *(extra_runs or [])]:
        if (candidate_pkg.root / 'official_grade.json').is_file():
            og = candidate_pkg.json('official_grade.json')
            raw = og.get('official_raw') if isinstance(og, dict) else None
            if isinstance(raw, dict) and isinstance(raw.get('dim2_items'), list) and raw['dim2_items']:
                return [(str(it['rule']), float(it['max_delta'])) for it in raw['dim2_items']]
    return []


def select_snapshot_candidates(
    snapshot: dict[str, Any],
    initial_input_hashes: set[str],
    office_targets: list[str],
    target_exts: set[str],
    last_written_turn: dict[str, int],
) -> list[dict[str, Any]]:
    """L2 v0.4 §4.2 step 2: Select candidate deliverable sets from a snapshot.

    Returns a list of candidate specs:
      {
        "candidate_id": str,
        "paths": [workspace_rel_path, ...],
        "staged": [(staged_filename, workspace_rel_path, sha256, blob_rel_path), ...],
        "in_submission": bool,
        "last_written_turn": int,
        "tier": int,   # 0: submission/, 1: same filename as target, 2: same extension
      }
    """
    files = snapshot.get('files') or []
    by_path = {f['path']: f for f in files}

    # All Agent-written Office files (excluding input/*; outside submission/, exclude unmodified copies of input/*)
    agent_files: list[dict[str, Any]] = []
    for f in files:
        p = f['path']
        if p.startswith('input/'):
            continue
        name = Path(p).name
        if name.startswith('~$'):
            continue
        ext = Path(p).suffix.lower()
        if ext not in target_exts:
            continue
        in_sub = p.startswith('submission/')
        if not in_sub and f['sha256'] in initial_input_hashes:
            continue
        agent_files.append(f)

    if not agent_files:
        return []

    candidates: list[dict[str, Any]] = []
    seen_staged_keys: set[tuple[tuple[str, str], ...]] = set()

    def _add_candidate(items: list[tuple[str, dict[str, Any]]], in_sub: bool, tier: int, label: str) -> None:
        if not items:
            return
        staged = sorted(
            [(staged_name, f['path'], f['sha256'], f['blob']) for staged_name, f in items],
            key=lambda x: x[0],
        )
        key = tuple((s[0], s[2]) for s in staged)
        if key in seen_staged_keys:
            return
        seen_staged_keys.add(key)
        lw = max((last_written_turn.get(f['path'], 0) for _, f in items), default=0)
        candidates.append({
            'candidate_id': label,
            'paths': [f['path'] for _, f in items],
            'staged': staged,
            'in_submission': in_sub,
            'last_written_turn': lw,
            'tier': tier,
        })

    # 1. submission/ directory as-is (if non-empty)
    sub_files = [f for f in agent_files if f['path'].startswith('submission/')]
    if sub_files:
        _add_candidate(
            [(f['path'][len('submission/'):], f) for f in sub_files],
            in_sub=True,
            tier=0,
            label='submission/',
        )

    # 2. Single-target vs Multi-target selection
    if len(office_targets) <= 1:
        target_name = office_targets[0] if office_targets else None
        # Order: submission/ target files -> Agent-written same-name files -> same-extension files
        ordered = sorted(
            agent_files,
            key=lambda f: (
                0 if f['path'].startswith('submission/') else (1 if Path(f['path']).name == target_name else 2),
                -last_written_turn.get(f['path'], 0),
                f['path'],
            ),
        )
        for f in ordered:
            in_sub = f['path'].startswith('submission/')
            tier = 0 if in_sub else (1 if Path(f['path']).name == target_name else 2)
            _add_candidate([(Path(f['path']).name, f)], in_sub=in_sub, tier=tier, label=f['path'])
    else:
        # Multi-target task (e.g. 004 with 2 docx files, 085 with 3 xlsx files):
        # Group by parent directory first (e.g., work/v2/, work/verify/), plus per-target best combination
        by_dir: dict[str, list[dict[str, Any]]] = {}
        for f in agent_files:
            parent_dir = str(Path(f['path']).parent)
            by_dir.setdefault(parent_dir, []).append(f)
        for d_name, d_files in sorted(by_dir.items()):
            if d_name == 'submission':
                continue
            matched_items: list[tuple[str, dict[str, Any]]] = []
            used_paths: set[str] = set()
            for t_name in office_targets:
                m = [
                    f for f in d_files
                    if f['path'] not in used_paths and _matches_target(Path(f['path']).name, t_name, office_targets)
                ]
                if m:
                    chosen = sorted(m, key=lambda x: (-last_written_turn.get(x['path'], 0), x['path']))[0]
                    matched_items.append((Path(chosen['path']).name, chosen))
                    used_paths.add(chosen['path'])
            if matched_items:
                tier = 1 if all(Path(f['path']).name in office_targets for _, f in matched_items) else 2
                _add_candidate(matched_items, in_sub=False, tier=tier, label=f'{d_name}/*')

        # Also combine per-target highest-priority candidates across workspace
        combined_items: list[tuple[str, dict[str, Any]]] = []
        used_paths = set()
        for t_name in office_targets:
            m = [
                f for f in agent_files
                if f['path'] not in used_paths and _matches_target(Path(f['path']).name, t_name, office_targets)
            ]
            if m:
                chosen = sorted(
                    m,
                    key=lambda f: (
                        0 if f['path'].startswith('submission/') else (1 if Path(f['path']).name == t_name else 2),
                        -last_written_turn.get(f['path'], 0),
                        f['path'],
                    ),
                )[0]
                combined_items.append((Path(chosen['path']).name, chosen))
                used_paths.add(chosen['path'])
        if combined_items:
            all_sub = all(f['path'].startswith('submission/') for _, f in combined_items)
            tier = 0 if all_sub else (1 if all(Path(f['path']).name in office_targets for _, f in combined_items) else 2)
            _add_candidate(combined_items, in_sub=all_sub, tier=tier, label='+'.join(f['path'] for _, f in combined_items))

    return candidates


def _evaluate_staged_candidate(
    pkg: L1Package,
    spec: EvaluationSpec,
    eval_cfg: dict[str, Any],
    snapshot: dict[str, Any],
    cand: dict[str, Any],
    cache_dir: Path | None,
    restored_workspace: Path | None = None,
) -> tuple[dict[str, Any], bool]:
    """Evaluate a single candidate (using content-hash cache if available).

    Returns (grade_dict, cache_hit).
    """
    cache_payload = {
        'schema': 'l2_grade_cache_v0.4',
        'task_id': spec.task_id,
        'entrypoint': spec.entrypoint,
        'outcome_policy': eval_cfg.get('outcome_policy', 'official_dimensions'),
        'image': eval_cfg.get('image'),
        'files': [{'name': name, 'sha256': sha} for name, _, sha, _ in cand['staged']],
    }
    key = digest(cache_payload)
    cache_file = (cache_dir / 'grades' / f'{key}.json') if cache_dir is not None else None
    if cache_file is not None and cache_file.is_file():
        cached = read_json(cache_file)
        if cached.get('cache_key') == key and isinstance(cached.get('grade'), dict):
            return cached['grade'], True

    with tempfile.TemporaryDirectory(prefix='l2-grade-') as tmp:
        tmp_root = Path(tmp)
        ws_dir = restored_workspace
        if ws_dir is None:
            ws_dir = tmp_root / 'restored_workspace'
            ws_dir.mkdir(mode=0o700)
            validate_snapshot(pkg.root, snapshot)
            restore(pkg.root, snapshot, ws_dir)

        sub_dir = tmp_root / 'submission'
        sub_dir.mkdir(parents=True, exist_ok=True)
        for staged_name, rel_path, expected_sha, _ in cand['staged']:
            src = ws_dir / rel_path
            dst = sub_dir / staged_name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            if file_digest(dst) != expected_sha:
                raise L2Error(f'Restored candidate sha256 mismatch for {rel_path}')
            dst.chmod(0o644)
        for p in [sub_dir, *sub_dir.rglob('*')]:
            if p.is_dir():
                p.chmod(0o755)

        evaluator = DockerEvaluator(eval_cfg)
        grade_out = tmp_root / 'grading'
        grade = evaluator.evaluate(spec, sub_dir, grade_out)

    stored_grade = {
        'projection': grade.get('projection'),
        'official_raw': grade.get('official_raw'),
        'official_raw_sha256': grade.get('official_raw_sha256'),
    }
    if cache_file is not None:
        write_json(cache_file, {'cache_key': key, 'request': cache_payload, 'grade': stored_grade})
    return stored_grade, False


def grade_episode_snapshots(
    pkg: L1Package,
    *,
    cache_dir: Path | None = None,
    extra_runs: list[L1Package] | None = None,
    verify_terminal_submission: bool = True,
) -> list[dict[str, Any]]:
    """Evaluate all checkpoints c_0 .. c_T (and terminal submission/) for an L1Package.

    Returns the rows for `grades.jsonl`:
      - Row 0: `kind == "terminal_submission"` (official terminal submission/ grade Y)
      - Rows 1..T+1: `kind == "checkpoint"` for k = 0 .. T (snapshot grade s(k) and d_i(k))
    """
    t0 = time.monotonic()
    resolved_cfg = pkg.json('raw/resolved_config.json')
    eval_cfg = resolved_cfg.get('evaluation') or {}
    desc = pkg.json('benchmark/description.json')
    ev_info = desc['evaluation']
    spec = EvaluationSpec(
        ev_info['kind'],
        pkg.root / 'benchmark/source',
        ev_info['entrypoint'],
        desc['task_id'],
        ev_info.get('metadata') or {},
    )
    office_targets, target_exts = _office_targets(pkg)
    fallback_rubric = extract_task_rubric(pkg, extra_runs=extra_runs)

    # Initial input hashes from s000000
    s0 = pkg.snapshots.get('s000000') or pkg.json('states/snapshots/s000000.json')
    initial_input_hashes = {f['sha256'] for f in s0.get('files', []) if f['path'].startswith('input/')}

    # Build ordered list of (k, checkpoint_id, snapshot_id) for k = 0 .. T
    cps = pkg.checkpoints()
    T = pkg.turns
    all_snapshots = sorted(
        (p.stem for p in (pkg.root / 'states/snapshots').glob('s*.json')),
        key=lambda s: int(s[1:]),
    )
    final_snap_id = all_snapshots[-1]
    checkpoint_steps: list[tuple[int, str, str]] = []
    for k in range(T + 1):
        if k in cps:
            checkpoint_steps.append((k, cps[k]['checkpoint_id'], cps[k]['snapshot_id']))
        elif k == T:
            checkpoint_steps.append((T, f'cp{T:06d}', final_snap_id))

    # Track last_written_turn_by_k[k][path] incrementally across k = 0 .. T
    running_last_written: dict[str, int] = {}
    last_written_by_k: dict[int, dict[str, int]] = {}
    prev_sha_by_path: dict[str, str] = {}
    snapshots_by_k: dict[int, dict[str, Any]] = {}
    for k, _, snap_id in checkpoint_steps:
        snap = pkg.snapshots.get(snap_id) or pkg.json(f'states/snapshots/{snap_id}.json')
        snapshots_by_k[k] = snap
        cur_sha_by_path: dict[str, str] = {}
        for f in snap.get('files', []):
            p, sha = f['path'], f['sha256']
            cur_sha_by_path[p] = sha
            if prev_sha_by_path.get(p) != sha:
                running_last_written[p] = k
        prev_sha_by_path = cur_sha_by_path
        last_written_by_k[k] = dict(running_last_written)

    # Row 0: Official terminal submission grade Y
    official_grade = pkg.json('official_grade.json')
    official_proj = official_grade.get('projection') or {}
    official_raw = official_grade.get('official_raw')
    y_val = float(official_proj['completion_rate']) if isinstance(official_proj.get('completion_rate'), (int, float)) else 0.0
    final_snap = snapshots_by_k[T]
    sub_files_at_T = [f for f in final_snap.get('files', []) if f['path'].startswith('submission/')]

    if verify_terminal_submission:
        sub_cand = {
            'candidate_id': 'submission/',
            'paths': [f['path'] for f in sub_files_at_T],
            'staged': sorted(
                [(f['path'][len('submission/'):], f['path'], f['sha256'], f['blob']) for f in sub_files_at_T],
                key=lambda x: x[0],
            ),
            'in_submission': True,
            'last_written_turn': max((last_written_by_k[T].get(f['path'], 0) for f in sub_files_at_T), default=0),
            'tier': 0,
        }
        re_grade, _ = _evaluate_staged_candidate(pkg, spec, eval_cfg, final_snap, sub_cand, cache_dir)
        re_proj = re_grade.get('projection') or {}
        re_y = float(re_proj['completion_rate']) if isinstance(re_proj.get('completion_rate'), (int, float)) else 0.0
        if abs(re_y - y_val) > 1e-9 or re_proj.get('official_dimensions') != official_proj.get('official_dimensions'):
            raise L2Error(
                f'Terminal submission/ re-evaluation mismatch on {pkg.episode_id}: '
                f're_evaluated={re_proj} vs official={official_proj}'
            )

    y_deductions, max_score_M = compute_item_deductions(official_raw, y_val, fallback_rubric)
    rows: list[dict[str, Any]] = [
        {
            'kind': 'terminal_submission',
            'episode_id': pkg.episode_id,
            'task_id': pkg.family_id,
            'k': T,
            'snapshot_id': final_snap_id,
            'candidate_status': 'submission' if sub_files_at_T else 'no_candidate',
            'candidate': [f['path'] for f in sub_files_at_T],
            'in_submission': bool(sub_files_at_T),
            'score': y_val,
            'dim1_pass': bool((official_raw or {}).get('dim1_pass')) if isinstance(official_raw, dict) else False,
            'total_score': (official_raw or {}).get('total_score') if isinstance(official_raw, dict) else 0,
            'max_score': max_score_M,
            'deductions': y_deductions,
            'episode_index_sha256': pkg.index_sha256,
            'code_version': CODE_VERSION,
        }
    ]

    # Rows for k = 0 .. T
    for k, cp_id, snap_id in checkpoint_steps:
        snap = snapshots_by_k[k]
        cands = select_snapshot_candidates(
            snap, initial_input_hashes, office_targets, target_exts, last_written_by_k[k]
        )
        if not cands:
            rows.append({
                'kind': 'checkpoint',
                'episode_id': pkg.episode_id,
                'task_id': pkg.family_id,
                'k': k,
                'checkpoint_id': cp_id,
                'snapshot_id': snap_id,
                'candidate_status': 'no_candidate',
                'candidate': [],
                'in_submission': False,
                'last_written_turn': None,
                'score': 0.0,
                'dim1_pass': False,
                'total_score': 0,
                'max_score': max_score_M,
                'deductions': {GATE_DEFECT_ID: 1.0},
                'candidates_evaluated': 0,
                'episode_index_sha256': pkg.index_sha256,
                'code_version': CODE_VERSION,
            })
            continue

        # Restore workspace once if any candidate at this snapshot is not yet cached
        best_tuple: tuple[Any, ...] | None = None
        best_record: dict[str, Any] | None = None
        with tempfile.TemporaryDirectory(prefix='l2-snap-') as tmp_ws:
            restored_ws: Path | None = None
            for cand in cands:
                # Check if we need to restore the snapshot first
                cache_payload = {
                    'schema': 'l2_grade_cache_v0.4',
                    'task_id': spec.task_id,
                    'entrypoint': spec.entrypoint,
                    'outcome_policy': eval_cfg.get('outcome_policy', 'official_dimensions'),
                    'image': eval_cfg.get('image'),
                    'files': [{'name': name, 'sha256': sha} for name, _, sha, _ in cand['staged']],
                }
                ckey = digest(cache_payload)
                cfile = (cache_dir / 'grades' / f'{ckey}.json') if cache_dir is not None else None
                if (cfile is None or not cfile.is_file()) and restored_ws is None:
                    restored_ws = Path(tmp_ws) / 'ws'
                    restored_ws.mkdir(mode=0o700)
                    validate_snapshot(pkg.root, snap)
                    restore(pkg.root, snap, restored_ws)

                grade, _ = _evaluate_staged_candidate(
                    pkg, spec, eval_cfg, snap, cand, cache_dir, restored_workspace=restored_ws
                )
                proj = grade.get('projection') or {}
                raw = grade.get('official_raw')
                s_val = float(proj['completion_rate']) if isinstance(proj.get('completion_rate'), (int, float)) else 0.0
                deductions, m_val = compute_item_deductions(raw, s_val, fallback_rubric)
                # §4.2 step 3: "逐个用 L1 官方评分器打分，取最高；并列时取 submission/ 里的，再取最近写入的。"
                rank_key = (
                    round(s_val, 12),
                    bool(cand['in_submission']),
                    int(cand['last_written_turn']),
                    -int(cand['tier']),
                    cand['candidate_id'],
                )
                if best_tuple is None or rank_key > best_tuple:
                    best_tuple = rank_key
                    best_record = {
                        'kind': 'checkpoint',
                        'episode_id': pkg.episode_id,
                        'task_id': pkg.family_id,
                        'k': k,
                        'checkpoint_id': cp_id,
                        'snapshot_id': snap_id,
                        'candidate_status': 'ok',
                        'candidate': cand['paths'],
                        'in_submission': bool(cand['in_submission']),
                        'last_written_turn': int(cand['last_written_turn']),
                        'score': s_val,
                        'dim1_pass': bool((raw or {}).get('dim1_pass')) if isinstance(raw, dict) else False,
                        'total_score': (raw or {}).get('total_score') if isinstance(raw, dict) else 0,
                        'max_score': m_val,
                        'deductions': deductions,
                        'candidates_evaluated': len(cands),
                        'episode_index_sha256': pkg.index_sha256,
                        'code_version': CODE_VERSION,
                    }
        assert best_record is not None
        rows.append(best_record)

    elapsed = time.monotonic() - t0
    rows[0]['grading_elapsed_seconds'] = round(elapsed, 4)
    return rows


def write_grades(out_dir: Path, rows: list[dict[str, Any]]) -> Path:
    path = out_dir / 'grades.jsonl'
    write_jsonl(path, rows)
    return path
