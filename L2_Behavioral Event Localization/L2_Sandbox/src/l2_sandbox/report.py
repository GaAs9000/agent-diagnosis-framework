"""End-to-end idempotent L2 v0.3 workflow (`l2 run`) and Markdown report generator (Section 7.1 & 7.2)."""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any, Callable, Sequence
from .analysts.llm import analyst_tag, predict_values_llm, propose_hypotheses_llm
from .analysts.rules import (
    REFERENCE_ANALYST_ID,
    RULES_ANALYST_ID,
    predict_values_reference,
    predict_values_rules,
    propose_hypotheses_rules,
)
from .config import load as load_config
from .curve import build_curve_plan, execute_curve
from .evidence.l1 import L1Package
from .facts.build import build_facts
from .io import digest, read_json, read_jsonl, sha_file, with_header, write_json, write_jsonl
from .score import score_episode


def _safe_filename(analyst_id: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', analyst_id) + '.json'


def render_report_md(
    plan: dict[str, Any],
    curve_doc: dict[str, Any] | None,
    scores_doc: dict[str, Any] | None,
) -> str:
    lines = [
        f"# L2 行为定位报告：`{plan['episode_id']}`",
        "",
        f"- **任务家族**：`{plan.get('family_id')}`",
        f"- **总回合数 $T$**：`{plan['turns']}`，终态结果 $Y$：`{plan['terminal_outcome']}`",
        f"- **采样网格**：`{plan['grid']}`（$N_1={plan['n1']}, N_2={plan['n2']}, \\text{{stride}}={plan['stride']}$）",
        f"- **预估最大分支数**：`{plan['estimated_branches']['total_max']}`（预估成本：`${plan['estimated_cost_usd']}`）",
    ]
    if curve_doc is None:
        lines.extend([
            "",
            "## 执行状态",
            "",
            "- 仅生成计划与第 1 次分析器价值预测（未带 `--execute`，未调用 L1 续跑）。",
        ])
        return '\n'.join(lines) + '\n'

    lines.extend([
        "",
        "## 价值曲线与定位判定",
        "",
        f"- **曲线判定**：`{curve_doc.get('verdict')}`",
        f"- **定位回合 / 区间**：回合 `{curve_doc.get('turn')}`，区间 `{curve_doc.get('interval')}`",
        f"- **分支统计**：尝试 `{curve_doc.get('total_attempted_branches')}`，无效 `{curve_doc.get('invalid_branches')}`（无效率 `{curve_doc.get('invalid_rate', 0.0):.2%}`，是否剔除：`{curve_doc.get('excluded_high_invalid_rate')}`）",
    ])

    if scores_doc and scores_doc.get('analysts'):
        lines.extend([
            "",
            "## 分析器评分与假说检验",
            "",
            "| 分析器 | Brier | 相对线性衰减技能分 | 提出假说数 | 证实数 (`confirmed`) | 解释覆盖 ($\\ge 0.5$) | 折合续跑次数 |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])
        for a_name, a_data in sorted(scores_doc['analysts'].items()):
            vp = a_data['value_prediction']
            ht = a_data['hypothesis_testing']
            cost = a_data['cost']
            b_str = f"{vp['brier']:.4f}" if vp.get('brier') is not None else "—"
            s_str = f"{vp['skill_score']:.4f}" if vp.get('skill_score') is not None else "—"
            lines.append(
                f"| `{a_name}` | {b_str} | {s_str} | {ht['proposed_count']} | {ht['confirmed_count']} | `{ht['explanation_covered']}` | {cost['equivalent_continuations']} |"
            )

    return '\n'.join(lines) + '\n'


def run_episode(
    l1_run_dir: Path,
    *,
    output_dir: Path | None = None,
    analysts: Sequence[str] = ('rules',),
    execute: bool = False,
    p0: float = 0.5,
    n1: int = 12,
    n2: int = 24,
    stride: int = 4,
    alpha: float = 0.05,
    max_concurrency: int = 4,
    resume_fn: Callable[..., Any] | None = None,
    llm_caller: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    reference_snapshot_scores: dict[int, float] | None = None,
) -> dict[str, Any]:
    """Runs the v0.3 L2 loop on a single L1 parent package (`l2 run <l1_run>`).

    Directory layout produced under `l2_dir`:
      ├─ facts.jsonl
      ├─ values/<analyst>.json
      ├─ plan.json
      ├─ branches/<trial>/       (when execute=True)
      ├─ outcomes.jsonl          (when execute=True)
      ├─ curve.json              (when execute=True)
      ├─ hypotheses/<analyst>.json (when execute=True & concentrated_*)
      ├─ scores.json             (when execute=True)
      └─ report.md
    """
    parent = L1Package(Path(l1_run_dir))
    l2_dir = Path(output_dir) if output_dir is not None else (Path('l2') / parent.episode_id)
    l2_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract turn-indexed parent facts
    facts = build_facts(parent)
    facts_path = write_jsonl(l2_dir / 'facts.jsonl', facts)
    facts_sha = sha_file(facts_path)

    # 2. Build curve plan (no branches run yet)
    plan = build_curve_plan(
        parent,
        n1=n1,
        n2=n2,
        stride=stride,
        alpha=alpha,
        max_concurrency=max_concurrency,
    )
    write_json(l2_dir / 'plan.json', plan)

    # Parse requested analysts (`rules`, `reference`, `llm:<config_path>`, `llm_raw:<config_path>`)
    parsed_specs = []
    for spec in analysts:
        s = spec.strip()
        if not s:
            continue
        if s in ('rules', RULES_ANALYST_ID):
            parsed_specs.append({'kind': 'rules', 'id': RULES_ANALYST_ID, 'config': None, 'include_facts': True})
        elif s in ('reference', REFERENCE_ANALYST_ID):
            parsed_specs.append({'kind': 'reference', 'id': REFERENCE_ANALYST_ID, 'config': None, 'include_facts': True})
        elif s.startswith('llm_raw:') or s == 'llm_raw':
            cfg_path = Path(s.split(':', 1)[1]) if ':' in s and s.split(':', 1)[1] else None
            cfg = load_config(cfg_path)
            a_id = analyst_tag(cfg['model'], include_facts=False)
            parsed_specs.append({'kind': 'llm', 'id': a_id, 'config': cfg, 'include_facts': False})
        elif s.startswith('llm:') or s == 'llm':
            cfg_path = Path(s.split(':', 1)[1]) if ':' in s and s.split(':', 1)[1] else None
            cfg = load_config(cfg_path)
            inc = bool(cfg['model'].get('include_facts', True))
            a_id = analyst_tag(cfg['model'], include_facts=inc)
            parsed_specs.append({'kind': 'llm', 'id': a_id, 'config': cfg, 'include_facts': inc})
        else:
            raise ValueError(f'Unknown analyst specification: {s}')

    # 3. Analyst Call 1: Value Predictions (cached by input sha256)
    values_dir = l2_dir / 'values'
    values_dir.mkdir(parents=True, exist_ok=True)
    call1_outputs = {}
    for spec in parsed_specs:
        fname = _safe_filename(spec['id'])
        v_path = values_dir / fname
        in_sha = {
            'parent_index_sha256': parent.index_sha256,
            'facts_sha256': facts_sha,
            'p0_sha256': digest(round(float(p0), 8)),
        }
        if v_path.is_file():
            existing = read_json(v_path)
            if existing.get('input_sha256') == in_sha:
                call1_outputs[spec['id']] = existing
                continue

        if spec['kind'] == 'rules':
            val_res = predict_values_rules(facts, parent.turns, p0, float(plan['terminal_outcome']))
        elif spec['kind'] == 'reference':
            val_res = predict_values_reference(
                facts, parent.turns, p0, float(plan['terminal_outcome']), reference_snapshot_scores
            )
        else:
            val_res = predict_values_llm(
                parent,
                p0,
                spec['config']['model'],
                facts,
                include_facts=spec['include_facts'],
                llm_caller=llm_caller,
            )
        doc = with_header(in_sha, dict(val_res))
        write_json(v_path, doc)
        call1_outputs[spec['id']] = doc

    if not execute:
        rep_text = render_report_md(plan, None, None)
        (l2_dir / 'report.md').write_text(rep_text, encoding='utf-8')
        return {
            'episode_id': parent.episode_id,
            'l2_dir': str(l2_dir),
            'executed': False,
            'plan': plan,
            'values': call1_outputs,
        }

    # 4. Execute two-stage value curve (idempotent branch reuse)
    curve_doc = execute_curve(
        parent,
        l2_dir,
        plan,
        resume_fn=resume_fn,
        max_concurrency=max_concurrency,
    )
    curve_sha = sha_file(l2_dir / 'curve.json')

    # 5. Analyst Call 2: Propose Predicate Hypotheses (only on concentrated_*)
    hyps_dir = l2_dir / 'hypotheses'
    hyps_dir.mkdir(parents=True, exist_ok=True)
    call2_outputs = {}
    if curve_doc.get('verdict') in ('concentrated_turn', 'concentrated_interval'):
        for spec in parsed_specs:
            fname = _safe_filename(spec['id'])
            h_path = hyps_dir / fname
            in_sha = {
                'parent_index_sha256': parent.index_sha256,
                'facts_sha256': facts_sha,
                'curve_sha256': curve_sha,
                'p0_sha256': digest(round(float(p0), 8)),
            }
            if h_path.is_file():
                existing = read_json(h_path)
                if existing.get('input_sha256') == in_sha:
                    call2_outputs[spec['id']] = existing
                    continue

            if spec['kind'] in ('rules', 'reference'):
                hyps = propose_hypotheses_rules(facts, curve_doc)
            else:
                # Hybrid mode optimization: when both rules and LLM are run in production,
                # Call 2 is always evaluated during benchmark runs or when rules haven't confirmed.
                hyps = propose_hypotheses_llm(
                    parent,
                    p0,
                    curve_doc,
                    spec['config']['model'],
                    facts,
                    include_facts=spec['include_facts'],
                    llm_caller=llm_caller,
                )
            doc = with_header(in_sha, {'analyst': spec['id'], 'hypotheses': hyps})
            write_json(h_path, doc)
            call2_outputs[spec['id']] = doc

    # 6. Score value predictions and test predicate hypotheses
    scores_doc = score_episode(l2_dir, p0=p0, alpha=alpha)
    rep_text = render_report_md(plan, curve_doc, scores_doc)
    (l2_dir / 'report.md').write_text(rep_text, encoding='utf-8')

    return {
        'episode_id': parent.episode_id,
        'l2_dir': str(l2_dir),
        'executed': True,
        'plan': plan,
        'curve': curve_doc,
        'scores': scores_doc,
    }
