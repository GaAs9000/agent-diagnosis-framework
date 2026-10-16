"""End-to-end L2 v0.4 workflow (`l2 diagnose`, `l2 verify`, and Markdown report generator, Section 8 & 9.2)."""
from __future__ import annotations
import math
import re
from pathlib import Path
from typing import Any, Callable, Sequence
from .analysts import probability, validate_hypotheses, validate_origins
from .analysts.llm import (
    CALL1_SYSTEM_PROMPT,
    CALL2_SYSTEM_PROMPT,
    analyst_tag,
    build_call1_input,
    build_call2_input,
    predict_origins_llm,
    propose_hypotheses_llm,
    request_model_config,
)
from .analysts.rules import (
    RULES_ANALYST_ID,
    build_terminal_official_view,
    predict_origins_rules,
    propose_hypotheses_rules,
)
from .config import load as load_config
from .defects import build_contrast, build_defect_table, write_contrast, write_defects
from .evidence.l1 import L1Package
from .facts.build import build_facts
from .grades import grade_episode_snapshots, write_grades
from .io import (
    L2Error,
    atomic_write,
    digest,
    read_json,
    read_jsonl,
    validate_output_location,
    with_header,
    write_json,
    write_jsonl,
)
from .score import score_episode
from .verify import resume_arms

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / 'cache'


def _safe_filename(analyst_id: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', analyst_id) + '.json'


def discover_peer_runs(parent: L1Package, cohort_dir: Path | None) -> list[tuple[L1Package, list[dict[str, Any]]]]:
    """Finds same-task independent root runs under `cohort_dir` (excluding `parent` and any branches)."""
    if cohort_dir is None:
        return []
    c_root = Path(cohort_dir)
    if not c_root.is_dir():
        return []
    peers: list[tuple[L1Package, list[dict[str, Any]]]] = []
    seen_eps: set[str] = {parent.episode_id}
    for mf in sorted(c_root.rglob('manifest.json')):
        pkg_dir = mf.parent
        if 'branches' in pkg_dir.parts:
            continue
        try:
            doc = read_json(mf)
            ep_id = str(doc.get('episode_id') or '')
            if not ep_id or ep_id in seen_eps:
                continue
            lineage = doc.get('lineage') or {}
            if lineage.get('parent_episode_id') is not None:
                continue
            task_info = doc.get('task') or {}
            t_id = str(task_info.get('task_id') or '')
            b_name = str(task_info.get('benchmark') or 'officeval').lower()
            fam_id = f'{b_name}:{t_id}' if t_id else ''
            if fam_id and fam_id != parent.family_id:
                continue
            pkg = L1Package(pkg_dir)
            if pkg.family_id != parent.family_id:
                continue
            seen_eps.add(pkg.episode_id)
            peers.append((pkg, build_facts(pkg)))
        except Exception:
            continue
    return peers


def render_report_md(
    defects_doc: dict[str, Any],
    scores_doc: dict[str, Any] | None = None,
    verifications_doc: dict[str, Any] | None = None,
) -> str:
    """Renders L2 v0.4 Markdown report organized by defect, origin accuracy, and verification results."""
    ep_id = defects_doc.get('episode_id')
    fam_id = defects_doc.get('task_id') or defects_doc.get('family_id')
    y_val = defects_doc.get('terminal_score_y', defects_doc.get('terminal_outcome', 0.0))
    s_t = defects_doc.get('terminal_snapshot_score_s_T', y_val)
    max_s = defects_doc.get('max_snapshot_score', y_val)
    k_star = defects_doc.get('k_star', 0)
    g_val = defects_doc.get('delivery_gap_g', defects_doc.get('delivery_gap', 0.0))
    tot_ded = defects_doc.get('total_deduction', 0.0)
    by_kind = defects_doc.get('deduction_by_kind') or {}

    lines = [
        f"# L2 缺陷溯因诊断报告：`{ep_id}`",
        "",
        f"- **任务 (`task_id`)**：`{fam_id}`",
        f"- **终态提交得分 $Y$**：`{y_val:.4f}`，终态快照最高分 $s(T)$：`{s_t:.4f}`，全程峰值 $\\max_k s(k)$：`{max_s:.4f}`（首次达到于 `c_{k_star}`）",
        f"- **未留住差值 $g = \\max_k s(k) - Y$**：`{g_val:.4f}`，总扣分 $1 - Y$：`{tot_ded:.4f}`",
        f"- **按来源类别扣分拆分**：`undelivered={by_kind.get('undelivered', 0.0):.4f}`, `broken={by_kind.get('broken', 0.0):.4f}`, `never={by_kind.get('never', 0.0):.4f}`",
        "",
        "## 1. 缺陷表（E1 检查点快照真值）",
        "",
        "| 缺陷项 (`defect`) | 失分权重 $w_i$ | 来源类别 (`kind`) | 来源回合 (`turn`) | 未纠正窗口摘要 |",
        "| :--- | :---: | :---: | :---: | :--- |",
    ]

    defects = defects_doc.get('defects') or []
    if not defects:
        lines.append("| —（满分无缺陷） | `0.0000` | — | — | — |")
    else:
        for d in defects:
            win = d.get('window') or {}
            win_txt = win.get('summary_text') or '—'
            t_str = f"`{d['turn']}`" if d.get('turn') is not None else "—"
            lines.append(
                f"| `{d['defect']}` | `{float(d['deduction']):.4f}` | `{d['kind']}` | {t_str} | {win_txt} |"
            )

    if scores_doc and scores_doc.get('analysts'):
        lines.extend([
            "",
            "## 2. 分析器来源定位与机制检验评分",
            "",
            "| 分析器 | 组别 | 类别准确率（等权） | 类别准确率（加权） | 回合命中率 ($\\pm 1$) | 回合精确命中 | 提出假说数 | 证实数 (`confirmed`) | 缺陷覆盖率 |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for a_name, a_data in sorted(scores_doc['analysts'].items()):
            ol = a_data.get('origin_localization') or {}
            mt = a_data.get('mechanism_testing') or {}
            grp = a_data.get('analyst_group') or '—'
            ku = f"{ol['kind_accuracy_unweighted']:.2%}" if ol.get('kind_accuracy_unweighted') is not None else "—"
            kw = f"{ol['kind_accuracy_weighted']:.2%}" if ol.get('kind_accuracy_weighted') is not None else "—"
            tt1 = f"{ol['turn_accuracy_tol1']:.2%}" if ol.get('turn_accuracy_tol1') is not None else "—"
            tte = f"{ol['turn_accuracy_exact']:.2%}" if ol.get('turn_accuracy_exact') is not None else "—"
            cov = f"{mt['defect_coverage']:.2%}" if mt.get('defect_coverage') is not None else "—"
            lines.append(
                f"| `{a_name}` | `{grp}` | {ku} | {kw} | {tt1} | {tte} | {mt.get('proposed_count', 0)} | {mt.get('confirmed_count', 0)} | {cov} |"
            )

    tp_list = (verifications_doc or scores_doc or {}).get('two_point_comparisons') or []
    if tp_list:
        lines.extend([
            "",
            "## 3. 两点续跑比较（Two-Point Continuation Verification）",
            "",
            "| 比较区间 | $(n_{k_1}, n_{k_2})$ | $\\hat{V}(c_{k_1}) \\to \\hat{V}(c_{k_2})$ | 点估计跌落 $\\hat{d}$ | 95% 保守下界 `lower_drop` | 单侧置换 $p$ 值 | 结论 |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for tp in tp_list:
            v1 = f"{tp['v_k1']:.4f}" if tp.get('v_k1') is not None else "—"
            v2 = f"{tp['v_k2']:.4f}" if tp.get('v_k2') is not None else "—"
            dr = f"{tp['drop']:.4f}" if tp.get('drop') is not None else "—"
            ld = f"{tp['lower_drop_95']:.4f}" if tp.get('lower_drop_95') is not None else "—"
            pv = f"{tp['p_value']:.4f}" if tp.get('p_value') is not None else "—"
            lines.append(
                f"| `c_{tp['k1']} -> c_{tp['k2']}` | `({tp['n_k1']}, {tp['n_k2']})` | `{v1} -> {v2}` | `{dr}` | `{ld}` | `{pv}` | `{tp['status']}` |"
            )

    intv_list = ((verifications_doc or {}).get('direct_intervention') or {}).get('hypotheses') or []
    if intv_list:
        lines.extend([
            "",
            "## 4. 干预续跑检验（Causal Intervention Verification, E2）",
            "",
            "| 检查点 | 干预臂 vs 对照臂 | $(n_{\\text{treat}}, n_{\\text{ctrl}})$ | $\\bar{Y}_{\\text{treat}} \\text{ vs } \\bar{Y}_{\\text{ctrl}}$ | 干预提升 $\\hat{\\Delta}$ | 单侧置换 $p$ 值 | 结论 |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for iv in intv_list:
            mt = f"{iv['mean_treatment']:.4f}" if iv.get('mean_treatment') is not None else "—"
            mc = f"{iv['mean_control']:.4f}" if iv.get('mean_control') is not None else "—"
            ef = f"{iv['effect']:+.4f}" if iv.get('effect') is not None else "—"
            pv = f"{iv['p_value_holm']:.4f}" if iv.get('p_value_holm') is not None else "—"
            lines.append(
                f"| `c_{iv.get('k', iv.get('turn'))}` | `{iv['treatment_arm']} vs {iv['control_arm']}` | `({iv['n_treatment']}, {iv['n_control']})` | `{mt} vs {mc}` | `{ef}` | `{pv}` | `{iv['status']}` |"
            )

    return '\n'.join(lines) + '\n'


def resolve_m2_baseline_p0(
    source_path: Path,
    parent_dir: Path,
    *,
    condition_id: str | None = None,
) -> dict[str, Any]:
    """Optional L3 M2 baseline p_0 resolver."""
    try:
        from l3_sandbox.report import resolve_baseline_p0
        from l3_sandbox.common import L3Error
    except ImportError as exc:
        import sys
        _l3_src = Path(__file__).resolve().parents[4] / 'L3_Agent Capability Boundary Discovery' / 'L3_Sandbox' / 'src'
        if _l3_src.is_dir() and str(_l3_src) not in sys.path:
            sys.path.insert(0, str(_l3_src))
            try:
                from l3_sandbox.report import resolve_baseline_p0
                from l3_sandbox.common import L3Error
            except ImportError as exc2:
                raise L2Error('Verified --baseline requires L3; install the L3_Sandbox package, or use explicit --p0') from exc2
        else:
            raise L2Error('Verified --baseline requires L3; install the L3_Sandbox package, or use explicit --p0') from exc
    try:
        return resolve_baseline_p0(source_path, parent_dir, condition_id=condition_id)
    except (L3Error, OSError, ValueError, KeyError, TypeError) as exc:
        raise L2Error(str(exc)) from exc


def _parse_analyst_specs(analysts: Sequence[str]) -> list[dict[str, Any]]:
    parsed_specs = []
    for spec in ('rules', *analysts):
        s = spec.strip()
        if not s:
            continue
        if s in ('rules', RULES_ANALYST_ID):
            parsed_specs.append({'kind': 'rules', 'id': RULES_ANALYST_ID, 'group': 'A', 'config': None, 'include_facts': True})
        elif s.startswith('llm_raw:') or s == 'llm_raw':
            cfg_path = Path(s.split(':', 1)[1]) if ':' in s and s.split(':', 1)[1] else None
            cfg = load_config(cfg_path)
            a_id = analyst_tag(cfg['model'], include_facts=False)
            parsed_specs.append({'kind': 'llm', 'id': a_id, 'group': 'B', 'config': cfg, 'include_facts': False})
        elif s.startswith('llm:') or s == 'llm':
            cfg_path = Path(s.split(':', 1)[1]) if ':' in s and s.split(':', 1)[1] else None
            cfg = load_config(cfg_path)
            inc = bool(cfg['model'].get('include_facts', True))
            a_id = analyst_tag(cfg['model'], include_facts=inc)
            parsed_specs.append({'kind': 'llm', 'id': a_id, 'group': 'C' if inc else 'B', 'config': cfg, 'include_facts': inc})
        else:
            raise ValueError(f'Unknown analyst specification: {s}')
    return list({spec['id']: spec for spec in parsed_specs}.values())


def diagnose_episode(
    l1_run_dir: Path,
    *,
    output_dir: Path | None = None,
    analysts: Sequence[str] = ('rules',),
    execute: bool = False,
    cache_only: bool = False,
    cohort_dir: Path | None = None,
    p0: float | None = None,
    alpha: float = 0.05,
    llm_caller: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    cache_dir: Path | None = None,
    reuse_dirs: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Runs L2 v0.4 zero-continuation diagnosis (`l2 diagnose <l1_run>`).

    Produces under `l2_dir`:
      ├─ facts.jsonl
      ├─ grades.jsonl
      ├─ defects.json
      ├─ contrast.json
      ├─ origins/<analyst>.json
      ├─ hypotheses/<analyst>.json
      ├─ scores.json
      └─ report.md
    """
    parent = L1Package(Path(l1_run_dir))
    l2_dir = Path(output_dir) if output_dir is not None else (Path('l2') / parent.episode_id)
    if p0 is not None:
        probability(p0, 'p0')
    validate_output_location(parent.root, l2_dir)
    parsed_specs = _parse_analyst_specs(analysts)
    eff_cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR

    l2_dir.mkdir(parents=True, exist_ok=True)
    facts_path = l2_dir / 'facts.jsonl'
    facts = build_facts(parent)
    write_jsonl(facts_path, facts)

    peer_runs = discover_peer_runs(parent, cohort_dir)
    extra_pkgs = [p for p, _ in peer_runs]

    grades_path = l2_dir / 'grades.jsonl'
    grade_rows = grade_episode_snapshots(parent, cache_dir=eff_cache_dir, extra_runs=extra_pkgs)
    write_grades(l2_dir, grade_rows)

    defects_doc = build_defect_table(
        parent,
        grade_rows,
        facts,
        grades_path=grades_path,
        facts_path=facts_path,
        extra_runs=extra_pkgs,
    )
    # Check if existing outcomes in l2_dir or reuse_dirs confirm a two-point drop for unlocalized defects
    outcomes_cand = l2_dir / 'outcomes.jsonl'
    if not outcomes_cand.is_file() and reuse_dirs:
        for r_dir in reuse_dirs:
            c_out = Path(r_dir) / 'outcomes.jsonl'
            if c_out.is_file():
                outcomes_cand = c_out
                break
    if outcomes_cand.is_file():
        from .verify import evaluate_two_point
        out_rows = [r for r in read_jsonl(outcomes_cand) if r.get('selected', True)]
        plain_ks = sorted({
            int(r['k']) for r in out_rows
            if (r.get('arm', 'plain') == 'plain' or r.get('stage') in ('s1', 's2'))
            and r.get('valid') and isinstance(r.get('k'), int)
        })
        confirmed_tp_turns = [
            k2 for k1, k2 in zip(plain_ks, plain_ks[1:])
            if k2 >= 1 and evaluate_two_point(out_rows, k1=k1, k2=k2, alpha=alpha).get('confirmed')
        ]
        if confirmed_tp_turns:
            tp_turn = confirmed_tp_turns[0]
            for d in defects_doc.get('defects') or []:
                if d.get('turn') is None:
                    d['two_point_turn'] = tp_turn
    write_defects(l2_dir, defects_doc)

    defect_ids = [str(d['defect']) for d in (defects_doc.get('defects') or [])]
    contrast_doc = build_contrast(parent, facts, defect_ids, peer_runs, facts_path=facts_path)
    write_contrast(l2_dir, contrast_doc)

    terminal_official_view = build_terminal_official_view(parent, facts, extra_runs=extra_pkgs)
    expected_defects = [str(d['defect']) for d in (terminal_official_view.get('target_defects') or [])]

    origins_dir = l2_dir / 'origins'
    hyps_dir = l2_dir / 'hypotheses'
    origins_dir.mkdir(parents=True, exist_ok=True)
    hyps_dir.mkdir(parents=True, exist_ok=True)

    bindings: dict[str, dict[str, Any]] = {'origins': {}, 'hypotheses': {}}
    call1_outputs: dict[str, Any] = {}
    call2_outputs: dict[str, Any] = {}
    deferred: list[str] = []

    def binding(spec: dict[str, Any], call: int) -> dict[str, str]:
        if spec['kind'] == 'llm':
            payload = (
                build_call1_input(parent, terminal_official_view, facts, contrast_doc, include_facts=spec['include_facts'])
                if call == 1
                else build_call2_input(parent, terminal_official_view, defects_doc, facts, contrast_doc, include_facts=spec['include_facts'])
            )
            return {
                'payload_sha256': digest(payload),
                'model_sha256': digest(request_model_config(spec['config']['model'])),
                'prompt_sha256': digest(CALL1_SYSTEM_PROMPT if call == 1 else CALL2_SYSTEM_PROMPT),
            }
        payload = (
            {'facts': facts, 'terminal_official_view': terminal_official_view}
            if call == 1
            else {'facts': facts, 'defects_table': defects_doc}
        )
        return {'payload_sha256': digest(payload)}

    for spec in parsed_specs:
        fname = _safe_filename(spec['id'])
        o_path = origins_dir / fname
        in_sha1 = binding(spec, 1)

        o_doc = None
        if o_path.is_file():
            try:
                cand = read_json(o_path)
                if cand.get('episode_id') == parent.episode_id and cand.get('input_sha256') == in_sha1:
                    validate_origins(cand.get('origins'), expected_defects, parent.turns, spec['id'], analyst_group=spec['group'])
                    o_doc = cand
            except Exception:
                o_doc = None

        if o_doc is None and spec['kind'] == 'llm' and not execute and llm_caller is None:
            if cache_only:
                raise L2Error(f"Cache-only diagnosis requires missing LLM Call 1 output for {spec['id']}")
            deferred.append(spec['id'])
            continue

        if o_doc is None:
            if spec['kind'] == 'rules':
                origins = predict_origins_rules(facts, terminal_official_view)
            else:
                origins = predict_origins_llm(
                    parent,
                    terminal_official_view,
                    spec['config']['model'],
                    facts,
                    contrast_doc,
                    include_facts=spec['include_facts'],
                    cache_dir=eff_cache_dir,
                    llm_caller=llm_caller,
                )
            validate_origins(origins, expected_defects, parent.turns, spec['id'], analyst_group=spec['group'])
            o_doc = with_header(
                in_sha1,
                {
                    'analyst': spec['id'],
                    'analyst_group': spec['group'],
                    'episode_id': parent.episode_id,
                    'origins': origins,
                },
            )
            write_json(o_path, o_doc)

        bindings['origins'][spec['id']] = in_sha1
        call1_outputs[spec['id']] = o_doc

        # Call 2: Propose mechanism hypotheses for localized defects (turn >= 1)
        h_path = hyps_dir / fname
        in_sha2 = binding(spec, 2)
        h_doc = None
        if h_path.is_file():
            try:
                cand_h = read_json(h_path)
                if cand_h.get('episode_id') == parent.episode_id and cand_h.get('input_sha256') == in_sha2:
                    localized = {
                        str(d['defect'])
                        for d in (defects_doc.get('defects') or [])
                        if (
                            (isinstance(d.get('turn'), int) and not isinstance(d.get('turn'), bool) and d['turn'] >= 1)
                            or (isinstance(d.get('two_point_turn'), int) and not isinstance(d.get('two_point_turn'), bool) and d['two_point_turn'] >= 1)
                        )
                    }
                    validate_hypotheses(
                        cand_h.get('hypotheses'),
                        parent.turns,
                        spec['id'],
                        analyst_group=spec['group'],
                        allowed_defects=localized,
                    )
                    h_doc = cand_h
            except Exception:
                h_doc = None

        if h_doc is None:
            if spec['kind'] == 'rules':
                hyps = propose_hypotheses_rules(facts, defects_doc)
            else:
                hyps = propose_hypotheses_llm(
                    parent,
                    terminal_official_view,
                    defects_doc,
                    spec['config']['model'],
                    facts,
                    contrast_doc,
                    include_facts=spec['include_facts'],
                    cache_dir=eff_cache_dir,
                    llm_caller=llm_caller,
                )
            h_doc = with_header(
                in_sha2,
                {
                    'analyst': spec['id'],
                    'analyst_group': spec['group'],
                    'episode_id': parent.episode_id,
                    'hypotheses': hyps,
                },
            )
            write_json(h_path, h_doc)

        bindings['hypotheses'][spec['id']] = in_sha2
        call2_outputs[spec['id']] = h_doc

    scores_doc = score_episode(
        l2_dir,
        alpha=alpha,
        active_analysts=list(call1_outputs),
        expected_input_bindings=bindings,
        reuse_dirs=reuse_dirs,
    )
    verif_path = l2_dir / 'verifications.json'
    verif_doc = read_json(verif_path) if verif_path.is_file() else None
    rep_text = render_report_md(defects_doc, scores_doc, verif_doc)
    atomic_write(l2_dir / 'report.md', rep_text.encode('utf-8'))

    return {
        'episode_id': parent.episode_id,
        'family_id': parent.family_id,
        'l2_dir': str(l2_dir),
        'executed': bool(execute or not deferred),
        'deferred_analysts': deferred,
        'defects': defects_doc,
        'scores': scores_doc,
    }


def verify_episode(
    l1_run_dir: Path,
    *,
    output_dir: Path | None = None,
    mode: str = 'intervention',
    k: int = 0,
    arms: Sequence[str | dict[str, Any]] = ('plain',),
    n: int = 24,
    outcome: str = 'total',
    requirement: str | None = None,
    execute: bool = False,
    cache_only: bool = False,
    alpha: float = 0.05,
    max_concurrency: int = 4,
    rpm: float | None = None,
    tpm: float | None = None,
    reuse_dirs: Sequence[Path] | None = None,
    resume_fn: Callable[..., Any] | None = None,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Runs targeted continuation verification (`l2 verify <l1_run>`, Section 6 & 9.1)."""
    parent = L1Package(Path(l1_run_dir))
    l2_dir = Path(output_dir) if output_dir is not None else (Path('l2') / parent.episode_id)
    validate_output_location(parent.root, l2_dir)

    if not (l2_dir / 'defects.json').is_file() or not (l2_dir / 'facts.jsonl').is_file():
        diagnose_episode(l1_run_dir, output_dir=l2_dir, analysts=('rules',), execute=False, cache_dir=cache_dir)

    arm_specs: list[str | dict[str, Any]] = []
    two_point_pairs: list[tuple[int, int]] = []
    if mode == 'two_point':
        if k < 1:
            raise L2Error('two_point mode requires checkpoint k >= 1 (compares k-1 vs k)')
        arm_specs = [
            {'arm': 'plain', 'k': k - 1, 'n': n},
            {'arm': 'plain', 'k': k, 'n': n},
        ]
        two_point_pairs.append((k - 1, k))
    elif mode == 'partition':
        arm_specs = [{'arm': 'plain', 'k': k, 'n': n}]
    elif mode == 'intervention':
        raw_arms = list(arms) if arms else ['plain', 'deliver']
        has_plain = any(
            (a == 'plain' if isinstance(a, str) else a.get('arm') == 'plain')
            for a in raw_arms
        )
        if not has_plain:
            raw_arms.insert(0, 'plain')
        for a in raw_arms:
            if isinstance(a, str):
                arm_specs.append({'arm': a, 'k': k, 'n': n, 'requirement': requirement})
            else:
                spec_copy = dict(a)
                spec_copy.setdefault('k', k)
                spec_copy.setdefault('n', n)
                if requirement and not spec_copy.get('requirement'):
                    spec_copy['requirement'] = requirement
                arm_specs.append(spec_copy)
    else:
        raise L2Error(f'Unknown verification mode: {mode!r} (expected partition, intervention, or two_point)')

    if not execute and not cache_only:
        return {
            'episode_id': parent.episode_id,
            'family_id': parent.family_id,
            'l2_dir': str(l2_dir),
            'executed': False,
            'mode': mode,
            'k': k,
            'planned_arms': arm_specs,
        }

    arm_res = resume_arms(
        parent,
        l2_dir,
        k=k,
        arms=arm_specs,
        n=n,
        outcome=outcome,
        resume_fn=resume_fn,
        max_concurrency=max_concurrency,
        cache_only=cache_only,
        rpm=rpm,
        tpm=tpm,
        reuse_dirs=reuse_dirs,
    )

    # If mode == 'intervention', also ensure an explicit intervention hypothesis entry is evaluated
    # in verifications.json even when analysts only proposed partition predicates.
    scores_doc = score_episode(
        l2_dir,
        alpha=alpha,
        reuse_dirs=reuse_dirs,
        two_point_pairs=two_point_pairs or None,
    )
    verif_path = l2_dir / 'verifications.json'
    verif_doc = read_json(verif_path) if verif_path.is_file() else {}
    if mode == 'intervention':
        from .verify import evaluate_intervention
        treat_specs = [
            s for s in arm_specs
            if (s if isinstance(s, str) else s.get('arm', 'plain')) != 'plain'
        ]
        direct_hyps = [
            {
                'verification': 'intervention',
                'k': int(s['k']) if isinstance(s, dict) else k,
                'turn': int(s['k']) if isinstance(s, dict) else k,
                'treatment_arm': str(s['arm']).split(':', 1)[0] if isinstance(s, dict) else str(s).split(':', 1)[0],
                'control_arm': 'plain',
                'defect': outcome,
                'effect': 0.15,
                'rationale': f'Direct {mode} verification at c_{k}',
            }
            for s in treat_specs
        ]
        direct_intv = evaluate_intervention(direct_hyps, arm_res['outcomes'], alpha=alpha, outcome=outcome)
        verif_doc['direct_intervention'] = direct_intv
        write_json(verif_path, verif_doc)

    defects_doc = read_json(l2_dir / 'defects.json')
    rep_text = render_report_md(defects_doc, scores_doc, verif_doc)
    atomic_write(l2_dir / 'report.md', rep_text.encode('utf-8'))

    return {
        'episode_id': parent.episode_id,
        'family_id': parent.family_id,
        'l2_dir': str(l2_dir),
        'executed': True,
        'mode': mode,
        'arms': arm_res['arms'],
        'verifications': verif_doc,
        'scores': scores_doc,
    }


def run_episode(
    l1_run_dir: Path,
    *,
    output_dir: Path | None = None,
    analysts: Sequence[str] = ('rules',),
    execute: bool = False,
    cache_only: bool = False,
    p0: float | None = None,
    alpha: float = 0.05,
    cohort_dir: Path | None = None,
    llm_caller: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    cache_dir: Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Backward-compatible wrapper around `diagnose_episode`."""
    return diagnose_episode(
        l1_run_dir,
        output_dir=output_dir,
        analysts=analysts,
        execute=execute,
        cache_only=cache_only,
        cohort_dir=cohort_dir,
        p0=p0,
        alpha=alpha,
        llm_caller=llm_caller,
        cache_dir=cache_dir,
    )
