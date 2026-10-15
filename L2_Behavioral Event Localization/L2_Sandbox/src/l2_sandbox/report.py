"""End-to-end idempotent L2 v0.3 workflow (`l2 run`) and Markdown report generator (Section 7.1 & 7.2)."""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any, Callable, Sequence
from .analysts import probability, validate_values, validate_hypotheses
from .analysts.llm import (analyst_tag, predict_values_llm, propose_hypotheses_llm, build_call1_input, build_call2_input, CALL1_SYSTEM_PROMPT, CALL2_SYSTEM_PROMPT, request_model_config)
from .analysts.rules import (
    REFERENCE_ANALYST_ID,
    RULES_ANALYST_ID,
    predict_values_reference,
    predict_values_rules,
    propose_hypotheses_rules,
)
from .config import load as load_config
from .curve import build_curve_plan, execute_curve, ensure_plan_identity
from .evidence.l1 import L1Package
from .facts.build import build_facts
from .io import atomic_write, validate_output_location, L2Error, digest, read_json, with_header, write_json, write_jsonl
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
            "- 仅生成计划与本地规则预测；外部分析器为 deferred，未调用模型或 L1 续跑。",
        ])
        return '\n'.join(lines) + '\n'

    lines.extend([
        "",
        "## 价值曲线与定位判定",
        "",
        f"- **曲线判定**：`{curve_doc.get('verdict')}`",
        f"- **定位回合 / 区间**：回合 `{curve_doc.get('turn')}`，区间 `{curve_doc.get('interval')}`",
        f"- **分支统计**：尝试 `{curve_doc.get('selected_attempted_branches', curve_doc.get('total_attempted_branches'))}`，无效 `{curve_doc.get('selected_invalid_branches', curve_doc.get('invalid_branches'))}`（无效率 `{curve_doc.get('invalid_rate', 0.0):.2%}`，是否剔除：`{curve_doc.get('excluded_high_invalid_rate')}`）",
    ])
    rb_rates = curve_doc.get('request_budget_rate_by_k') or {}
    if rb_rates:
        parts = [f"c_{k}: {rate:.1%}" for k, rate in sorted(rb_rates.items(), key=lambda kv: int(kv[0]))]
        lines.append(f"- **各检查点有效分支请求耗尽比例（`request_budget`）**：{', '.join(parts)}")

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


def resolve_m2_baseline_p0(
    source_path: Path,
    parent_dir: Path,
    *,
    condition_id: str | None = None,
) -> dict[str, Any]:
    """Use L3's single offline evidence verifier, only when baseline is requested."""
    try:
        from l3_sandbox.report import resolve_baseline_p0
        from l3_sandbox.common import L3Error
    except ImportError as exc:
        raise L2Error('Verified --baseline requires L3; install the L3_Sandbox package, or use explicit --p0') from exc
    try:
        return resolve_baseline_p0(source_path, parent_dir, condition_id=condition_id)
    except (L3Error, OSError, ValueError, KeyError, TypeError) as exc:
        raise L2Error(str(exc)) from exc


def run_episode(
    l1_run_dir: Path,
    *,
    output_dir: Path | None = None,
    analysts: Sequence[str] = ('rules',),
    execute: bool = False,
    cache_only: bool = False,
    p0: float,
    n1: int = 12,
    n2: int = 24,
    stride: int = 4,
    alpha: float = 0.05,
    max_concurrency: int = 4,
    rpm: float | None = None,
    tpm: float | None = None,
    baseline_provenance: str = "external supplied p0; cohort origin and parent exclusion unverified",
    cohort_verified: bool = False,
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
    probability(p0, 'p0')
    for label, limit in (('rpm', rpm), ('tpm', tpm)):
        if limit is not None:
            import math
            if type(limit) not in (int, float) or not math.isfinite(limit) or limit <= 0:
                raise L2Error(f'{label} must be positive and finite')
    validate_output_location(parent.root, l2_dir)
    plan = build_curve_plan(parent, n1=n1, n2=n2, stride=stride, alpha=alpha,
                            max_concurrency=max_concurrency)
    ensure_plan_identity(l2_dir, plan)
    facts = build_facts(parent)

    # Parse requested analysts (`rules`, `reference`, `llm:<config_path>`, `llm_raw:<config_path>`)
    parsed_specs = []
    for spec in ('rules', *analysts):
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

    # Validate all requested configurations before producing artifacts.
    parsed_specs = list({spec['id']: spec for spec in parsed_specs}.values())
    if any(spec['kind'] == 'reference' for spec in parsed_specs):
        predict_values_reference(facts, parent.turns, p0, float(plan['terminal_outcome']), reference_snapshot_scores)
    l2_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(l2_dir / 'facts.jsonl', facts)
    write_json(l2_dir / 'plan.json', plan)
    baseline = {'p0': p0, 'provenance': baseline_provenance, 'cohort_verified': bool(cohort_verified)}
    write_json(l2_dir / 'baseline.json', with_header(plan['input_sha256'], baseline))
    bindings = {'values': {}, 'hypotheses': {}}

    def binding(spec, call, curve=None):
        if spec['kind'] == 'llm':
            payload = (build_call1_input(parent, p0, facts, include_facts=spec['include_facts']) if call == 1 else
                       build_call2_input(parent, p0, curve, facts, include_facts=spec['include_facts']))
        else:
            payload = {'facts': facts, 'turns': parent.turns, 'p0': p0,
                       'terminal_y': plan['terminal_outcome']}
            if call == 2:
                # The constructed summary strips every branch-level field.
                payload['curve_summary'] = build_call2_input(parent, p0, curve, None, include_facts=False)['curve_summary']
            if spec['kind'] == 'reference':
                payload['reference_scores'] = {str(k): v for k, v in reference_snapshot_scores.items()}
        if spec['kind'] == 'llm':
            return {'payload_sha256': digest(payload),
                    'model_sha256': digest(request_model_config(spec['config']['model'])),
                    'prompt_sha256': digest(CALL1_SYSTEM_PROMPT if call == 1 else CALL2_SYSTEM_PROMPT)}
        return {'payload_sha256': digest(payload)}

    def cached(path, inputs, kind, spec, curve=None):
        # Older files may have a tag derived from unrelated config/code. Match actual inputs.
        candidates = [path, *sorted(path.parent.glob('*.json'))]
        cfg = spec['config']['model']
        from .config import DEFAULTS
        legacy_models = [cfg]
        if cfg.get('max_response_bytes') == DEFAULTS['model']['max_response_bytes']:
            # This execution-only default was absent from older full-config tags.
            legacy_models.append({k: v for k, v in cfg.items() if k != 'max_response_bytes'})
        legacy_config_hashes = {digest(spec)}
        for legacy_model in legacy_models:
            legacy_tag_hash = digest({'c1': CALL1_SYSTEM_PROMPT, 'c2': CALL2_SYSTEM_PROMPT,
                                      'mode': 'facts' if spec['include_facts'] else 'raw',
                                      'model_config': legacy_model})[:16]
            legacy_id = f"llm:{cfg.get('model') or 'unknown'}@{legacy_tag_hash}"
            legacy_config = {**spec['config'], 'model': legacy_model}
            legacy_config_hashes.add(digest({**spec, 'config': legacy_config}))
            legacy_config_hashes.add(digest({**spec, 'id': legacy_id, 'config': legacy_config}))
        for candidate in dict.fromkeys(candidates):
            if not candidate.is_file():
                continue
            try:
                doc = read_json(candidate)
                if not isinstance(doc, dict) or doc.get('episode_id') != parent.episode_id:
                    continue
                old = doc.get('input_sha256') or {}
                legacy_match = (old.get('payload_sha256') == inputs['payload_sha256'] and
                                old.get('prompt_sha256') == inputs['prompt_sha256'] and
                                old.get('analyst_config_sha256') in legacy_config_hashes)
                if old != inputs and not legacy_match:
                    continue
                if kind == 'values':
                    validate_values(doc.get('values'), parent.turns)
                else:
                    validate_hypotheses(doc.get('hypotheses'), curve)
                # Group membership comes from the trusted request mode, never the
                # model response, cache payload, or opaque analyst tag.
                group = 'C' if spec['include_facts'] else 'B'
                if (old != inputs or doc.get('analyst') != spec['id'] or candidate != path
                        or doc.get('analyst_group') != group):
                    doc = {**doc, 'input_sha256': inputs, 'analyst': spec['id'], 'analyst_group': group}
                    write_json(path, doc)
                return doc
            except (L2Error, ValueError, TypeError, KeyError):
                continue
        return None

    call1_outputs = {}
    deferred = []
    for spec in parsed_specs:
        v_path = l2_dir / 'values' / _safe_filename(spec['id'])
        inputs = binding(spec, 1)
        doc = cached(v_path, inputs, 'values', spec) if spec['kind'] == 'llm' else None
        if doc is None and spec['kind'] == 'llm' and cache_only:
            raise L2Error('Cache-only analysis requires a missing LLM prediction; explicit execution required')
        if doc is None and spec['kind'] == 'llm' and not execute:
            deferred.append(spec['id'])
            continue
        if doc is None:
            if spec['kind'] == 'rules':
                result = predict_values_rules(facts, parent.turns, p0, float(plan['terminal_outcome']))
            elif spec['kind'] == 'reference':
                result = predict_values_reference(facts, parent.turns, p0, float(plan['terminal_outcome']), reference_snapshot_scores)
            else:
                result = predict_values_llm(parent, p0, spec['config']['model'], facts,
                                            include_facts=spec['include_facts'], llm_caller=llm_caller)
            validate_values(result['values'], parent.turns)
            doc = with_header(inputs, {**result, 'episode_id': parent.episode_id})
            write_json(v_path, doc)
        bindings['values'][spec['id']] = inputs
        call1_outputs[spec['id']] = doc

    if not execute:
        rep_text = render_report_md(plan, None, None)
        atomic_write(l2_dir / 'report.md', rep_text.encode('utf-8'))
        return {
            'episode_id': parent.episode_id,
            'l2_dir': str(l2_dir),
            'executed': False,
            'plan': plan,
            'values': call1_outputs,
            'deferred_analysts': deferred,
            'baseline': baseline,
        }

    # 4. Execute two-stage value curve (idempotent branch reuse)
    curve_doc = execute_curve(
        parent,
        l2_dir,
        plan,
        resume_fn=resume_fn,
        rpm=rpm, tpm=tpm,
        max_concurrency=max_concurrency, cache_only=cache_only,
    )

    # 5. Analyst Call 2: Propose Predicate Hypotheses (only on concentrated_*)
    hyps_dir = l2_dir / 'hypotheses'
    hyps_dir.mkdir(parents=True, exist_ok=True)
    call2_outputs = {}
    if curve_doc.get('verdict') in ('concentrated_turn', 'concentrated_interval'):
        for spec in parsed_specs:
            fname = _safe_filename(spec['id'])
            h_path = hyps_dir / fname
            in_sha = binding(spec, 2, curve_doc)
            bindings['hypotheses'][spec['id']] = in_sha
            existing = cached(h_path, in_sha, 'hypotheses', spec, curve_doc) if spec['kind'] == 'llm' else None
            if existing is not None:
                call2_outputs[spec['id']] = existing
                continue

            if spec['kind'] == 'llm' and cache_only:
                raise L2Error('Cache-only analysis requires a missing LLM hypothesis; explicit execution required')
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
            validate_hypotheses(hyps, curve_doc)
            doc = with_header(in_sha, {'analyst': spec['id'], 'hypotheses': hyps,
                                      'episode_id': parent.episode_id})
            write_json(h_path, doc)
            call2_outputs[spec['id']] = doc

    # 6. Score value predictions and test predicate hypotheses
    scores_doc = score_episode(l2_dir, p0=p0, alpha=alpha, active_analysts=list(call1_outputs), expected_input_bindings=bindings)
    rep_text = render_report_md(plan, curve_doc, scores_doc)
    atomic_write(l2_dir / 'report.md', rep_text.encode('utf-8'))

    return {
        'episode_id': parent.episode_id,
        'l2_dir': str(l2_dir),
        'executed': True,
        'plan': plan,
        'curve': curve_doc,
        'scores': scores_doc,
    }
