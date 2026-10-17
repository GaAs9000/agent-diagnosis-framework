"""V3 ladder diagnosis: evidence, nested interventions, inference, and policy evaluation.

L1 produces execution evidence. This module computes offline observations and
counterfactual summaries; it never promotes engineering tests into study results.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Sequence

from .config import (
    DELIVER_REMINDER as DELIVER_INTERVENTION_MESSAGE,
    RULE_REMINDER_TEMPLATE as RULE_INTERVENTION_TEMPLATE,
)
from .io import L2Error, digest
from .stats import holm_adjust, lower_drop, permutation_one_sided, wilson_bounds

V05_SCHEMA_VERSION = "l2-0.5.3-d0-conditional-pool"

V3_PROPOSITIONS = (
    "V3-P1",  # Level-1 generic intervention (D / S) recoverability share >= 20%
    "V3-P2",  # Ladder decomposition across D, S, R_explicit, R_implicit, and single-item R localization
    "V3-P3",  # Cross-fitted diagnostic value ceiling >= 0.05 (prerequisite gate for V3-P4 / V3-P5)
    "V3-P4",  # Analyst arm-selection utility (C >= B > A > fixed best single arm)
    "V3-P5",  # Cross-family out-of-sample generalization on unexposed holdout families (Stage H)
)

DEFAULT_SELF_CHECK_DRAFT = (
    "在宣布结束前，请先重新通读原始任务指令，自行打开并检查 submission/ 下已生成的交付文件，"
    "核对是否存在遗漏部分、空表/空页、公式或代码报错、以及未完成的步骤；"
    "如有问题请先修正并重新保存后再结束。"
)

LADDER_ARM_LEVELS: dict[str, int] = {
    "P": 0,
    "D": 1,
    "S": 1,
    "R": 2,
    "R_single": 2,
    "R_explicit": 2,
    "R_all": 2,
    "R_implicit": 3,
}

OPERATIONAL_LABELS = (
    "broken",
    "undelivered",
    "delivery_responsive",
    "self_check_responsive",
    "explicit_rule_responsive",
    "implicit_rule_responsive",
    "rule_responsive",
    "unrecovered_under_protocol",
)

EVIDENCE_STATUSES = (
    "adjudicated",
    "plain_recoverable",
    "inconclusive",
    "ineligible",
    "not_applicable",
)

RULE_SOURCES = (
    "explicit",
    "implicit",
    "ambiguous",
    "gate_or_no_artifact",
)


def sha256_text(text: str) -> str:
    """Return hex SHA-256 digest of UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class E1Observation:
    """Zero-cost snapshot trajectory observation (V3 §3.1 Field 1)."""
    e1_class: str  # "broken" | "undelivered" | "never_succeeded_on_disk"
    candidate_substate: str
    k_star: int
    max_item_score: float
    terminal_item_score: float
    deduction: float
    source_turn: int | None = None
    first_candidate_write_turn: int | None = None
    deadlock_turn: int | None = None
    in_candidate_regression: bool = False
    independent_state_damage: str | None = None


@dataclass
class E2ResponseVector:
    """Counterfactual continuation response vector at anchor checkpoint k_int (V3 §3.1 Field 2 & §3.4)."""
    evaluated: bool = False
    anchor_k: int | None = None
    anchor_type: str | None = None
    remaining_budget_at_anchor: int | None = None
    arm_outcomes: dict[str, list[float]] = field(default_factory=dict)
    branch_identities: dict[str, list[str]] = field(default_factory=dict)
    arm_means: dict[str, float] = field(default_factory=dict)
    arm_lb95_abs: dict[str, float] = field(default_factory=dict)
    arm_ub95_abs: dict[str, float] = field(default_factory=dict)
    pairwise_deltas: dict[str, float] = field(default_factory=dict)
    pairwise_lb95: dict[str, float] = field(default_factory=dict)
    pairwise_ub95: dict[str, float] = field(default_factory=dict)
    pairwise_p_raw: dict[str, float] = field(default_factory=dict)
    pairwise_p_holm: dict[str, float] = field(default_factory=dict)
    inference: dict[str, Any] = field(default_factory=dict)
    ineligible_arms: list[str] = field(default_factory=list)
    # Backward-compatible convenience fields
    mean_P: float | None = None
    mean_D: float | None = None
    mean_S: float | None = None
    mean_R: float | None = None
    mean_R_explicit: float | None = None
    mean_R_implicit: float | None = None
    mean_R_all: float | None = None
    delta_P_vs_Y: float | None = None
    delta_D_vs_P: float | None = None
    delta_S_vs_P: float | None = None
    delta_R_vs_D: float | None = None
    delta_R_vs_P: float | None = None
    delta_R_single_vs_R_all: float | None = None
    lb95_P_vs_Y: float | None = None
    lb95_D_vs_P: float | None = None
    ub95_D_vs_P: float | None = None
    lb95_R_vs_D: float | None = None
    ub95_R_vs_D: float | None = None
    lb95_R_vs_P: float | None = None
    ub95_R_vs_P: float | None = None
    p_val_D_vs_P: float | None = None
    p_val_R_vs_D: float | None = None
    p_holm_D_vs_P: float | None = None
    p_holm_R_vs_D: float | None = None


@dataclass
class DiagnosticUnitV05:
    """Single V3 diagnostic unit (whole-run or single rubric item)."""
    unit_id: str
    episode_id: str
    task_id: str
    benchmark: str
    unit_level: str  # "gate_or_no_artifact" | "rubric_item" | "whole_run"
    defect_name: str
    deduction_weight: float
    rule_source: str  # "explicit" | "implicit" | "ambiguous" | "gate_or_no_artifact"
    e1_observation: E1Observation
    e2_response_vector: E2ResponseVector
    operational_label: str | None
    evidence_status: str  # "adjudicated" | "plain_recoverable" | "inconclusive" | "ineligible" | "not_applicable"
    early_anchor_k: int | None = None
    early_anchor_remaining_budget: int | None = None
    exhausted_budget: bool | None = None
    valid_delivery: bool | None = None
    stratum_id: str | None = None
    inclusion_prob: float = 1.0
    family_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def lower_bound_abs(
    vals: Sequence[float],
    alpha: float = 0.05,
    *,
    seed: str = "l2-v05-lb-abs",
) -> float:
    """One-sided (1 - alpha) lower confidence bound for E[V(arm)]."""
    return lower_drop(vals, 0.0, alpha=alpha, seed=seed)


def upper_drop(
    vals_treat: Sequence[float],
    vals_ctrl: Sequence[float] | float,
    alpha: float = 0.05,
    *,
    seed: str = "l2-v05-ub",
) -> float:
    """One-sided (1 - alpha) upper confidence bound for E[V(treat)] - E[V(ctrl)]."""
    if isinstance(vals_ctrl, (int, float)) and not isinstance(vals_ctrl, bool):
        y_term = float(vals_ctrl)
        _, hi_w = wilson_bounds(vals_treat, alpha=alpha, one_sided=True)
        if all(float(x) in (0.0, 1.0) for x in vals_treat):
            return hi_w - y_term
        n1 = len(vals_treat)
        if n1 == 0:
            return 1.0
        rng = random.Random(
            int(
                digest({
                    "seed": seed,
                    "t": sorted(round(float(x), 8) for x in vals_treat),
                    "y": y_term,
                    "a": alpha,
                })[:16],
                16,
            )
        )
        n_boot = 2000
        means = sorted((sum(rng.choice(vals_treat) for _ in range(n1)) / n1) - y_term for _ in range(n_boot))
        idx = min(n_boot - 1, max(0, int(math.ceil((1.0 - alpha) * (n_boot - 1)))))
        return max(means[idx], hi_w - y_term)
    return -lower_drop(vals_ctrl, vals_treat, alpha=alpha, seed=seed)


_NUMERIC_TOLERANCE_RE = re.compile(
    r"(?:只允许有[^，。；]*误差|"
    r"(?:约|大约|不少于|不超过|不大于|不小于|距离[^，。；]{0,12})?\s*"
    r"\d+(?:\.\d+)?\s*(?:[-\u2013\u2014~至到]\s*\d+(?:\.\d+)?)?\s*(?:cm|厘米|mm|毫米|pt|磅|%|倍行距|号|字符))"
)


def split_ambiguous_requirement(
    rule_item: str | dict[str, Any],
    info: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """R7 & R8 (V3 §3.3 & §3.4): Split an `ambiguous` rule into:
      1) `visible_requirement` (for `R_explicit`): qualitative / layout / content requirement visible from
         the task prompt or reference image, stripped of hidden verifier numeric tolerances;
      2) `exact_verifier_specification` (for `R_implicit`): exact numeric threshold / tolerance / formula
         enforced by `verifier.py`.
    """
    if isinstance(rule_item, dict):
        vis = str(rule_item.get("visible_requirement") or "").strip()
        ex = str(rule_item.get("exact_verifier_specification") or rule_item.get("rubric_text") or "").strip()
        if vis and ex:
            return vis, ex
        raw_text = str(rule_item.get("rubric_text") or rule_item.get("rule") or "").strip()
        info = rule_item
    else:
        raw_text = str(rule_item).strip()

    if info and isinstance(info, dict):
        vis_override = str(info.get("visible_requirement") or "").strip()
        ex_override = str(info.get("exact_verifier_specification") or "").strip()
        if vis_override and ex_override:
            return vis_override, ex_override

    raise L2Error("Ambiguous requirements require a reviewed visible_requirement and exact_verifier_specification split")


def _classify_officeval_rule_default(short_tid: str, defect_rule: str) -> tuple[str, str]:
    """Deterministic fallback rule-source classifier for whole-task OfficeVal rubrics."""
    if short_tid == "officeval_003":
        if "0.15cm的误差" in defect_rule:
            return "ambiguous", "image.png 展示等距横线，但 0.15cm 硬数值容差仅在评分器中定义"
        if "没有空白行" in defect_rule or "下方都出现横线" in defect_rule or "侯景澜" in defect_rule or "部门协调会" in defect_rule or "缩进" in defect_rule or "不可编辑" in defect_rule:
            return "explicit", "Prompt 明确要求放入表格加横线、删空行、其他格式不变"
        return "ambiguous", "视觉参考与评分细则容差混合"
    if short_tid == "officeval_004":
        if "实习单位评价" in defect_rule or "教学单位评价意见" in defect_rule:
            return "implicit", "参考图未在这两格标注宋体小四，但评分器要求宋体小四"
        if "学号" in defect_rule or "页面顶部没有出现黑体、20号" in defect_rule:
            return "ambiguous", "参考图展示表头与学号外观，未文字标注20号字号硬阈值"
        return "explicit", "参考图红字批注已逐项明确写出字体、字号、加粗、缩进与行距要求"
    if short_tid == "officeval_010":
        return "ambiguous", "Prompt 要求删第四行、字移框内做成可编辑，原图展示拓扑但评分器检查厘米级宽高与字号"
    if short_tid == "officeval_033":
        if "推荐客餐厅配套品表" in defect_rule or "定制价格表" in defect_rule or "门店订购记录页" in defect_rule:
            return "ambiguous", "Prompt 要求乘以1.1，未规定 .5 半整数四舍五入 vs 银行家舍入"
        return "explicit", "Prompt 明确要求所有零售价乘以 1.1 且其他格式不变"
    if short_tid == "officeval_050":
        if "Arial 或 Calibri" in defect_rule or "编号圆形" in defect_rule:
            return "implicit", "Prompt 未声明必须使用 Arial/Calibri 字体或第3阶段编号写02"
        return "ambiguous", "原 PPT 视觉可见，但评分器规定了精确厘米坐标框与字号区间"
    if short_tid == "officeval_060":
        if "0—0.3厘米" in defect_rule or "0%—22%" in defect_rule or "35%—100%" in defect_rule or "重叠面积" in defect_rule:
            return "ambiguous", "Prompt 要求重新排版符合比例无空白，评分器规定精确边距与百分比区间"
        return "explicit", "Prompt 明确要求第四页重新排版、符合页面比例且两边无大片空白"
    if short_tid == "officeval_070":
        if "10.5至11.5磅" in defect_rule or "0.9至1.0倍行距" in defect_rule or "7.2至8.2磅" in defect_rule or "0.3至0.5" in defect_rule:
            return "ambiguous", "Prompt 要求缩小字体间距调整布局，评分器规定精确磅值与边距区间"
        return "explicit", "Prompt 明确要求字完整放在板块区域框内不重叠、PPT尺寸不变"
    if short_tid == "officeval_085":
        if "18cm–21cm" in defect_rule or "10cm–13cm" in defect_rule:
            return "implicit", "Prompt 与参考图均未指定图表宽 18-21cm、高 10-13cm"
        return "explicit", "Prompt 明确要求画成图片上类似的 S11 图表并放置在对应表格里可编辑"
    if short_tid == "officeval_090":
        if "IFERROR" in defect_rule or "Sheet3及同结构子表" in defect_rule or "Sheet5及同结构子表" in defect_rule:
            return "implicit", "评分器强制要求特定跨表引用公式或 IFERROR 公式"
        return "explicit", "Prompt 明确要求将所有子表总重量合计与砼体积提取到汇总表"
    return "ambiguous", "视觉/结构要求与评分器细则容差混合"


def _default_officeval_rubrics_dir() -> Path | None:
    """Locate `benchmarks/officeval/data/rubrics` in the repository if available."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "L1_sandbox" / "C3_Experimental_Sandbox" / "benchmarks" / "officeval" / "data" / "rubrics"
        if cand.is_dir():
            return cand
    return None


def build_task_requirement_lists(
    task_rule_catalog: dict[str, Any],
    task_id: str,
    *,
    rubrics_dir: Path | None = None,
) -> dict[str, Any]:
    """R7 & R8 (V3 §3.4): Build full whole-task `explicit` and `implicit` requirement lists for `task_id`.

    - Extracts ALL rules belonging to `task_id` from the whole-task rubric catalog (never filtering by
      which items failed on a specific parent trajectory).
    - Supports direct `{task_id: {rule: info}}` dicts, `{task_id: [item_dicts]}` catalogs, AND full P0a
      audit JSON documents (`all_9_dev_tasks_rubric_citation_catalog`, `task_rule_catalog`, or `dev45`
      with automatic fallback to `benchmarks/officeval/data/rubrics/{short_id}.json` when a P0a audit
      document is passed).
    - For `ambiguous` rules, splits each rule into:
      * `visible_requirement` -> appended to `explicit` (so `R_explicit` includes all task/image-visible
        requirements without leaking hidden verifier numeric tolerances)
      * `exact_verifier_specification` -> appended to `implicit` (so `R_implicit` includes all exact
        verifier tolerances in addition to pure `implicit` rules).
    """
    short_id = task_id.split(":")[-1]
    full_id = f"officeval:{short_id}" if not task_id.startswith("officeval:") else task_id

    # Case 1: Direct task key in `task_rule_catalog`
    raw_entry = (
        task_rule_catalog.get(full_id)
        or task_rule_catalog.get(short_id)
        or task_rule_catalog.get(task_id)
    )
    is_p0a_wrapper = (
        raw_entry is None
        and isinstance(task_rule_catalog, dict)
        and any(
            k in task_rule_catalog
            for k in (
                "all_9_dev_tasks_rubric_citation_catalog",
                "task_rule_catalog",
                "dev45",
                "audit_id",
            )
        )
    )

    normalized_rules: list[tuple[str, dict[str, Any]]] = []
    seen_rules: set[str] = set()

    def _add_entries(source_obj: Any) -> None:
        if isinstance(source_obj, dict):
            for r_text, r_info in sorted(source_obj.items()):
                cleaned = str(r_text).strip()
                if cleaned and cleaned not in seen_rules:
                    seen_rules.add(cleaned)
                    normalized_rules.append((cleaned, dict(r_info) if isinstance(r_info, dict) else {}))
        elif isinstance(source_obj, list):
            for item in source_obj:
                if isinstance(item, dict):
                    cleaned = str(item.get("rubric_text") or item.get("rule") or "").strip()
                    if cleaned and cleaned not in seen_rules:
                        seen_rules.add(cleaned)
                        info_dict = dict(item)
                        if "source_category" not in info_dict and "rule_source" in info_dict:
                            info_dict["source_category"] = info_dict["rule_source"]
                        normalized_rules.append((cleaned, info_dict))
                elif isinstance(item, str) and item.strip():
                    cleaned = item.strip()
                    if cleaned not in seen_rules:
                        seen_rules.add(cleaned)
                        cat, rat = _classify_officeval_rule_default(short_id, cleaned)
                        normalized_rules.append((cleaned, {"source_category": cat, "rationale": rat}))

    if raw_entry is not None:
        _add_entries(raw_entry)
    elif is_p0a_wrapper:
        for cat_key in ("task_rule_catalog", "all_9_dev_tasks_rubric_citation_catalog"):
            sub_cat = task_rule_catalog.get(cat_key)
            if isinstance(sub_cat, dict):
                sub_entry = sub_cat.get(full_id) or sub_cat.get(short_id) or sub_cat.get(task_id)
                if sub_entry is not None:
                    _add_entries(sub_entry)
        dev_audit = (
            ((task_rule_catalog.get("dev45") or {}).get("delivered_item_wrong_rubric_source") or {}).get("task_rule_audit")
            or {}
        )
        if isinstance(dev_audit, dict):
            sub_audit = dev_audit.get(full_id) or dev_audit.get(short_id) or dev_audit.get(task_id)
            if sub_audit is not None:
                _add_entries(sub_audit)
        # Ensure whole-task coverage from `benchmarks/officeval/data/rubrics/{short_id}.json`
        r_dir = rubrics_dir or _default_officeval_rubrics_dir()
        if r_dir is not None:
            rubric_file = r_dir / f"{short_id}.json"
            if rubric_file.is_file():
                try:
                    r_doc = json.loads(rubric_file.read_text(encoding="utf-8"))
                    for r_text in (r_doc.get("rubrics") or {}).get("dim2") or []:
                        cleaned = str(r_text).strip()
                        if cleaned and cleaned not in seen_rules:
                            seen_rules.add(cleaned)
                            cat, rat = _classify_officeval_rule_default(short_id, cleaned)
                            normalized_rules.append((cleaned, {"source_category": cat, "rationale": rat}))
                except Exception:
                    pass

    explicit_only: list[str] = []
    implicit_only: list[str] = []
    ambiguous_reqs: list[str] = []
    ambiguous_visible: list[str] = []
    ambiguous_exact: list[str] = []
    explicit_reqs: list[str] = []
    implicit_reqs: list[str] = []

    unresolved_rules: list[str] = []
    provenance_errors: list[str] = []
    for cleaned, info in normalized_rules:
        if not all(info.get(k) for k in ("reviewer", "reviewed_at", "source_citation")):
            provenance_errors.append(cleaned)
        cat = str((info or {}).get("source_category") or (info or {}).get("rule_source") or "ambiguous")
        if cat == "explicit":
            explicit_only.append(cleaned)
            if cleaned not in explicit_reqs:
                explicit_reqs.append(cleaned)
        elif cat == "implicit":
            implicit_only.append(cleaned)
            if cleaned not in implicit_reqs:
                implicit_reqs.append(cleaned)
        else:
            ambiguous_reqs.append(cleaned)
            try:
                vis_req, exact_spec = split_ambiguous_requirement(cleaned, info)
            except L2Error:
                unresolved_rules.append(cleaned)
                continue
            ambiguous_visible.append(vis_req)
            ambiguous_exact.append(exact_spec)
            if vis_req not in explicit_reqs:
                explicit_reqs.append(vis_req)
            if exact_spec not in implicit_reqs:
                implicit_reqs.append(exact_spec)

    return {
        "task_id": short_id,
        "rule_sources": {rule: info.get("source_category", info.get("rule_source", "ambiguous"))
                         for rule, info in normalized_rules if rule not in provenance_errors and rule not in unresolved_rules},
        "review_complete": bool(normalized_rules) and not unresolved_rules and not provenance_errors,
        "unresolved_rules": unresolved_rules,
        "provenance_errors": provenance_errors,
        "explicit": explicit_reqs,
        "implicit": implicit_reqs,
        "ambiguous": ambiguous_reqs,
        "explicit_only": explicit_only,
        "implicit_only": implicit_only,
        "ambiguous_visible": ambiguous_visible,
        "ambiguous_exact": ambiguous_exact,
    }


def build_ladder_intervention_messages(
    *,
    explicit_requirements: Sequence[str] = (),
    implicit_requirements: Sequence[str] = (),
    ambiguous_requirements: Sequence[str | dict[str, Any]] = (),
    single_rule_requirement: str | None = None,
    deliver_text: str = DELIVER_INTERVENTION_MESSAGE,
    self_check_text: str = DEFAULT_SELF_CHECK_DRAFT,
) -> dict[str, dict[str, Any]]:
    """R7 & R8 (V3 §3.4): Construct the nested intervention ladder messages and their SHA-256 hashes.

    Enforces strict nesting along the information ladder (`D -> S -> R_explicit -> R_implicit`, and `S -> R_single`):
      - `P`: None (0 added user message)
      - `D`: `deliver_text`
      - `S`: `D.message` + "\\n\\n" + `self_check_text` (strictly prefixed by `D.message`)
      - `R_explicit`: `S.message` + "\\n\\n请同时逐条核对以下任务明确要求：\\n" + numbered visible/explicit requirements.
        If empty -> `eligible = False`, `evidence_status = "ineligible"`, `message = None` (NEVER falls back to `D` or `S`!).
      - `R_implicit`: `R_explicit.message` (or `S.message` when `R_explicit` is `ineligible`) +
        "\\n\\n请进一步逐条核对以下隐含格式与细则要求：\\n" + numbered implicit/exact-tolerance requirements.
        If empty -> `eligible = False`, `evidence_status = "ineligible"`, `message = None`.
      - `R_single`: `S.message` + "\\n\\n请重点核对并满足以下单项要求：" + `single_rule_requirement`.
        If empty -> `eligible = False`, `evidence_status = "ineligible"`, `message = None`.
    """
    d_clean = deliver_text.strip()
    s_raw = self_check_text.strip()
    if s_raw and d_clean:
        s_clean = s_raw if (s_raw.startswith(d_clean) and s_raw != d_clean) else f"{d_clean}\n\n{s_raw}"
    else:
        s_clean = ""

    exp_clean = [r.strip() for r in explicit_requirements if isinstance(r, str) and r.strip()]
    imp_clean = [r.strip() for r in implicit_requirements if isinstance(r, str) and r.strip()]

    for amb_item in ambiguous_requirements:
        vis_req, exact_spec = split_ambiguous_requirement(amb_item)
        if vis_req and vis_req not in exp_clean:
            exp_clean.append(vis_req)
        if exact_spec and exact_spec not in imp_clean:
            imp_clean.append(exact_spec)

    single_clean = (single_rule_requirement or "").strip()

    out: dict[str, dict[str, Any]] = {
        "P": {
            "arm": "P",
            "level": 0,
            "eligible": True,
            "evidence_status": "eligible",
            "message": None,
            "message_sha256": None,
            "prefix_arm": None,
        },
        "D": {
            "arm": "D",
            "level": 1,
            "eligible": bool(d_clean),
            "evidence_status": "eligible" if d_clean else "ineligible",
            "message": d_clean if d_clean else None,
            "message_sha256": sha256_text(d_clean) if d_clean else None,
            "prefix_arm": "P",
        },
        "S": {
            "arm": "S",
            "level": 1,
            "eligible": bool(s_clean),
            "evidence_status": "eligible" if s_clean else "ineligible",
            "message": s_clean if s_clean else None,
            "message_sha256": sha256_text(s_clean) if s_clean else None,
            "prefix_arm": "D",
        },
    }

    s_or_d_arm = "S"
    s_or_d_msg = out["S"]["message"] if out["S"]["eligible"] else None

    if exp_clean and s_or_d_msg:
        exp_block = "\n".join(f"({i}) {req}" for i, req in enumerate(exp_clean, start=1))
        r_exp_msg = f"{s_or_d_msg}\n\n请同时逐条核对以下任务明确要求：\n{exp_block}"
        out["R_explicit"] = {
            "arm": "R_explicit",
            "level": 2,
            "eligible": True,
            "evidence_status": "eligible",
            "message": r_exp_msg,
            "message_sha256": sha256_text(r_exp_msg),
            "prefix_arm": s_or_d_arm,
        }
    else:
        out["R_explicit"] = {
            "arm": "R_explicit",
            "level": 2,
            "eligible": False,
            "evidence_status": "ineligible",
            "message": None,
            "message_sha256": None,
            "prefix_arm": s_or_d_arm,
        }

    if imp_clean and s_or_d_msg:
        imp_parent_arm = "R_explicit" if out["R_explicit"]["eligible"] else s_or_d_arm
        base_prefix = out["R_explicit"]["message"] if out["R_explicit"]["eligible"] else s_or_d_msg
        imp_block = "\n".join(f"({i}) {req}" for i, req in enumerate(imp_clean, start=1))
        r_imp_msg = f"{base_prefix}\n\n请进一步逐条核对以下隐含格式与细则要求：\n{imp_block}"
        out["R_implicit"] = {
            "arm": "R_implicit",
            "level": 3,
            "eligible": True,
            "evidence_status": "eligible",
            "message": r_imp_msg,
            "message_sha256": sha256_text(r_imp_msg),
            "prefix_arm": imp_parent_arm,
        }
    else:
        out["R_implicit"] = {
            "arm": "R_implicit",
            "level": 3,
            "eligible": False,
            "evidence_status": "ineligible",
            "message": None,
            "message_sha256": None,
            "prefix_arm": "R_explicit",
        }

    if single_clean and s_or_d_msg:
        r_single_msg = f"{s_or_d_msg}\n\n请重点核对并满足以下单项要求：{single_clean}"
        out["R_single"] = {
            "arm": "R_single",
            "level": 2,
            "eligible": True,
            "evidence_status": "eligible",
            "message": r_single_msg,
            "message_sha256": sha256_text(r_single_msg),
            "prefix_arm": s_or_d_arm,
        }
    else:
        out["R_single"] = {
            "arm": "R_single",
            "level": 2,
            "eligible": False,
            "evidence_status": "ineligible",
            "message": None,
            "message_sha256": None,
            "prefix_arm": s_or_d_arm,
        }

    return out


def verify_nested_ladder_messages(ladder_spec: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """R7: Verify that every eligible arm in `ladder_spec` strictly begins with its protocol-mandated
    parent arm's text (`D -> S -> R_explicit -> R_implicit`, and `S -> R_single`), has a matching
    live SHA-256 digest (`sha256_text(message) == message_sha256`), and never silently duplicates a
    lower arm's text when a requirement list is empty.
    """
    errors: list[str] = []
    for arm in ("P", "D", "S", "R_explicit", "R_implicit", "R_single"):
        info = ladder_spec.get(arm)
        if not isinstance(info, dict):
            errors.append(f"Missing canonical arm {arm}")
        elif info.get("arm") != arm or info.get("level") != LADDER_ARM_LEVELS[arm]:
            errors.append(f"Arm {arm} has invalid canonical identity or level")
    if (ladder_spec.get("D") or {}).get("prefix_arm") != "P":
        errors.append("D must declare P as protocol parent")
    p_info = ladder_spec.get("P") or {}
    if p_info.get("eligible") is not True:
        errors.append("P must be eligible")
    if p_info.get("message") is not None or p_info.get("message_sha256") is not None:
        errors.append("P arm must have message=None and message_sha256=None")

    d_info = ladder_spec.get("D") or {}
    d_msg = d_info.get("message")
    if not d_info.get("eligible") or not isinstance(d_msg, str) or not d_msg.strip():
        errors.append("D arm must be eligible and have non-empty string message")
    elif d_info.get("message_sha256") != sha256_text(d_msg):
        errors.append("D arm message_sha256 does not match sha256_text(D.message)")

    s_info = ladder_spec.get("S") or {}
    s_eligible = bool(s_info.get("eligible"))
    s_or_d_expected = "S"

    expected_parents: dict[str, str] = {
        "S": "D",
        "R_explicit": s_or_d_expected,
        "R_implicit": "R_explicit" if bool((ladder_spec.get("R_explicit") or {}).get("eligible")) else s_or_d_expected,
        "R_single": s_or_d_expected,
        "R_all": s_or_d_expected,
    }

    for arm_name in ("S", "R_explicit", "R_implicit", "R_single", "R_all"):
        info = ladder_spec.get(arm_name)
        if not info:
            continue
        if not info.get("eligible"):
            if (
                info.get("message") is not None
                or info.get("message_sha256") is not None
                or info.get("evidence_status") != "ineligible"
            ):
                errors.append(
                    f"Ineligible arm {arm_name} must have message=None, message_sha256=None, and evidence_status='ineligible'"
                )
            continue

        msg = info.get("message")
        recorded_hash = info.get("message_sha256")
        if not isinstance(msg, str) or not msg.strip():
            errors.append(f"Eligible arm {arm_name} is missing non-empty string message")
            continue
        actual_hash = sha256_text(msg)
        if recorded_hash != actual_hash:
            errors.append(
                f"Arm {arm_name} message_sha256 mismatch (recorded={recorded_hash!r}, actual={actual_hash!r})"
            )

        req_parent = expected_parents[arm_name]
        declared_parent = info.get("prefix_arm")
        if declared_parent != req_parent:
            errors.append(
                f"Arm {arm_name} declared prefix_arm={declared_parent!r}, but protocol requires {req_parent!r}"
            )
        parent_msg = (ladder_spec.get(req_parent) or {}).get("message")
        if not isinstance(parent_msg, str) or not parent_msg:
            errors.append(f"Protocol parent {req_parent} of eligible arm {arm_name} is missing string message")
            continue
        if not msg.startswith(parent_msg):
            errors.append(f"Arm {arm_name} does not start with protocol parent {req_parent}")
        if msg == parent_msg:
            errors.append(f"Arm {arm_name} is identical to lower arm {req_parent} (empty fallback forbidden)")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "hashes": {
            k: sha256_text(v["message"]) if isinstance(v.get("message"), str) else None
            for k, v in ladder_spec.items()
        },
    }


def build_r_all_intervention_message(
    requirements: Sequence[str],
    *,
    deliver_text: str = DELIVER_INTERVENTION_MESSAGE,
) -> str | None:
    """Build the whole-run screening intervention prompt (`R-all`) with `D` as strict prefix.
    Returns `None` (`ineligible`) when `requirements` is empty (never falls back to `D`).
    """
    cleaned = [r.strip() for r in requirements if isinstance(r, str) and r.strip()]
    if not cleaned:
        return None
    d_clean = deliver_text.strip()
    joined = "\n".join(f"({idx}) {req}" for idx, req in enumerate(cleaned, start=1))
    return f"{d_clean}\n\n请同时逐条核对以下全部要求：\n{joined}"


def select_v05_intervention_anchors(
    checkpoint_rows: Sequence[dict[str, Any]],
    *,
    max_requests: int = 35,
    r_b: int = 6,
    item_id: str | None = None,
    termination: str | None = None,
    e1_class: str | None = None,
) -> dict[str, Any]:
    """Choose the latest eligible checkpoint using its recorded remaining requests.

    Voluntary finishes retain their actual remaining budget. Candidate scores do
    not change the primary anchor. Missing budget evidence is ineligible.
    """
    eligible_cps: list[dict[str, Any]] = []
    for r in checkpoint_rows:
        if r.get("kind") != "checkpoint":
            continue
        k = int(r.get("k", 0))
        raw_rem = r.get("remaining_requests")
        if raw_rem is None:
            continue
        rem = int(raw_rem)
        if rem >= r_b and bool(r.get("cold_resume_eligible", True)):
            eligible_cps.append({**r, "k": k, "remaining_requests": rem})

    if not eligible_cps:
        return {
            "eligible": False,
            "primary_anchor_k": None,
            "primary_anchor_type": None,
            "primary_remaining_budget": None,
            "voluntary_finish_anchor_k": None,
            "budget_tail_anchor_k": None,
            "first_candidate_write_k": None,
            "early_anchor_k": None,
            "early_remaining_budget": None,
        }

    eligible_cps.sort(key=lambda x: x["k"])
    cand_cps = [r for r in eligible_cps if r.get("candidate_status") == "ok"]

    def _score_at(r: dict[str, Any]) -> float:
        if e1_class == "never_succeeded_on_disk":
            return 0.0
        if item_id and isinstance(r.get("deductions"), dict):
            if not r.get("dim1_pass"):
                return 0.0
            ded = float(r["deductions"].get(item_id, 0.0))
            return 1.0 if ded <= 1e-9 else 0.0
        return float(r.get("score") or 0.0)

    tail_cp = eligible_cps[-1]
    is_voluntary = (termination == "assistant_finished")
    first_cand_k = int(cand_cps[0]["k"]) if cand_cps else None

    chosen = tail_cp
    anchor_type = "voluntary_pre_finish" if is_voluntary else (
        "budget_tail_with_candidate" if cand_cps else "budget_tail_no_candidate"
    )

    early_candidates = [
        r for r in eligible_cps
        if r["remaining_requests"] >= 2 * r_b and r["k"] < chosen["k"]
    ]
    early_chosen = early_candidates[-1] if early_candidates else None

    return {
        "eligible": True,
        "primary_anchor_k": int(chosen["k"]),
        "primary_anchor_type": anchor_type,
        "primary_remaining_budget": int(chosen["remaining_requests"]),
        "voluntary_finish_anchor_k": int(tail_cp["k"]) if is_voluntary else None,
        "budget_tail_anchor_k": int(tail_cp["k"]),
        "first_candidate_write_k": first_cand_k,
        "early_anchor_k": int(early_chosen["k"]) if early_chosen else None,
        "early_remaining_budget": int(early_chosen["remaining_requests"]) if early_chosen else None,
    }


def build_interleaved_arm_schedule(
    arms: Sequence[str],
    n_per_arm: int,
    *,
    anchor_k: int,
    remaining_budget: int,
    seed: str = "l2-v05-schedule",
) -> list[dict[str, Any]]:
    """Build an interleaved, randomized execution schedule across eligible arms at the exact same
    checkpoint `anchor_k` and exact same `remaining_budget` (V3 §3.2 Invariant).
    """
    if n_per_arm < 1:
        raise L2Error("n_per_arm must be >= 1")
    if not arms:
        raise L2Error("arms must be non-empty")
    schedule: list[dict[str, Any]] = []
    rng = random.Random(int(digest({"seed": seed, "arms": list(arms), "n": n_per_arm, "k": anchor_k})[:16], 16))
    for rep in range(n_per_arm):
        arm_order = list(arms)
        rng.shuffle(arm_order)
        for arm in arm_order:
            schedule.append({
                "arm": arm,
                "replicate": rep,
                "checkpoint_k": int(anchor_k),
                "at_request": int(anchor_k) + 1,
                "remaining_budget": int(remaining_budget),
            })
    return schedule


def _mean_bounds(vals: Sequence[float], error: float) -> tuple[float, float]:
    """Exact binomial inversion, or bounded Hoeffding bounds for independent [0,1] outcomes."""
    if not vals or not 0 < error < 1 or any(not 0 <= v <= 1 for v in vals):
        raise L2Error("Expected nonempty bounded outcomes and error probability in (0,1)")
    n = len(vals)
    mean = sum(vals) / n
    if not all(v in (0.0, 1.0) for v in vals):
        radius = math.sqrt(math.log(1 / error) / (2 * n))
        return max(0.0, mean - radius), min(1.0, mean + radius)
    k = int(sum(vals))
    def tail(prob: float, start: int, end: int) -> float:
        if prob <= 0:
            return float(start == 0)
        if prob >= 1:
            return float(end == n)
        logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                + i * math.log(prob) + (n - i) * math.log1p(-prob) for i in range(start, end + 1)]
        top = max(logs)
        return math.exp(top) * sum(math.exp(x - top) for x in logs)
    def invert(start: int, end: int, increasing: bool) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(44):
            mid = (lo + hi) / 2
            if (tail(mid, start, end) < error) == increasing:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    return (invert(k, n, True) if k else 0.0,
            invert(0, k, False) if k < n else 1.0)


def _difference_bounds(treat: Sequence[float], ctrl: Sequence[float], error: float) -> tuple[float, float]:
    lt, ut = _mean_bounds(treat, error / 2)
    lc, uc = _mean_bounds(ctrl, error / 2)
    return lt - uc, ut - lc


def _shifted_one_sided_p_value(vals_treat: Sequence[float], vals_ctrl: Sequence[float], *,
                               delta: float = 0.10, seed: str = "unused") -> float:
    """Invert conservative difference bounds for H0: mean(treat)-mean(ctrl) <= delta."""
    lo, hi = 0.0, 1.0
    for _ in range(32):
        mid = (lo + hi) / 2
        if _difference_bounds(vals_treat, vals_ctrl, mid)[0] > delta:
            hi = mid
        else:
            lo = mid
    return hi


def adjudicate_e2_outcomes(
    terminal_y: float,
    arm_outcomes: dict[str, Sequence[float]],
    *,
    anchor_k: int,
    anchor_type: str = "budget_tail_no_candidate",
    remaining_budget: int = 6,
    delta: float = 0.10,
    tau: float = 0.20,
    alpha: float = 0.05,
    ineligible_arms: Sequence[str] = (),
    primary_family_size: int | None = None,
    comparison_ids: Sequence[str] | None = None,
    branch_identities: dict[str, Sequence[str]] | None = None,
) -> tuple[E2ResponseVector, str | None, str]:
    """Adjudicate using simultaneous conservative bounds and Holm delta-null tests.

    Binary outcomes use exact binomial inversion; bounded continuous outcomes use
    Hoeffding. The primary family size and unique comparison IDs can be bound to
    a sealed plan; omitted values are explicitly exploratory. Interventions must
    exceed every lower level by delta. Absolute recovery uses tau separately.
    """
    clean_outcomes: dict[str, list[float]] = {
        k: [float(x) for x in v]
        for k, v in arm_outcomes.items()
        if v is not None and len(v) > 0
    }
    if set(clean_outcomes) - set(LADDER_ARM_LEVELS):
        raise L2Error("Unknown intervention arm in E2 outcomes")
    if "P" not in clean_outcomes:
        raise L2Error("E2 adjudication requires non-empty 'P' arm outcomes")
    if len(clean_outcomes) < 2:
        raise L2Error("E2 adjudication requires 'P' and at least one intervention arm")

    arm_means: dict[str, float] = {}
    arm_lb95_abs: dict[str, float] = {}
    arm_ub95_abs: dict[str, float] = {}

    for arm, vals in clean_outcomes.items():
        arm_means[arm] = round(sum(vals) / len(vals), 6)
        arm_lb95_abs[arm] = round(lower_bound_abs(vals, alpha=alpha, seed=f"v05-abs-lb-{arm}"), 6)
        arm_ub95_abs[arm] = round(upper_drop(vals, 0.0, alpha=alpha, seed=f"v05-abs-ub-{arm}"), 6)

    # Identify all ordered pairs (hi_arm, lo_arm) where level(hi) > level(lo)
    eval_arms = sorted(clean_outcomes.keys(), key=lambda a: (LADDER_ARM_LEVELS.get(a, 99), a))
    primary_arms = {"P", "D", "S", "R_explicit", "R_implicit"}
    comparison_pairs: list[tuple[str, str]] = [
        (hi, lo) for hi in eval_arms for lo in eval_arms
        if hi in primary_arms and lo in primary_arms
        and LADDER_ARM_LEVELS[hi] > LADDER_ARM_LEVELS[lo]
    ]
    if "D" in clean_outcomes and "S" in clean_outcomes:
        comparison_pairs.append(("S", "D"))
    primary_pairs = list(comparison_pairs)
    # Scorer-selected single-rule prompts are localization evidence, not ladder levels.
    for auxiliary in ("R", "R_single"):
        if auxiliary in clean_outcomes:
            comparison_pairs.extend((auxiliary, lo) for lo in ("P", "D") if lo in clean_outcomes)
    local_ids = [f"{hi}_vs_{lo}" for hi, lo in comparison_pairs]
    if comparison_ids is not None and (len(comparison_ids) != len(local_ids) or len(set(comparison_ids)) != len(local_ids)):
        raise L2Error("comparison_ids must uniquely identify every local comparison")
    family_size = primary_family_size if primary_family_size is not None else len(comparison_pairs)
    if family_size < len(comparison_pairs):
        raise L2Error("primary_family_size cannot be smaller than the observed comparison family")
    interval_error = alpha / (8 * max(1, family_size + len(clean_outcomes)))
    for arm, vals in clean_outcomes.items():
        lb, ub = _mean_bounds(vals, interval_error)
        arm_lb95_abs[arm], arm_ub95_abs[arm] = lb, ub

    pairwise_deltas: dict[str, float] = {}
    pairwise_lb95: dict[str, float] = {}
    pairwise_ub95: dict[str, float] = {}
    pairwise_p_raw: dict[str, float] = {}
    raw_p_list: list[float] = []
    decision_bounds: dict[str, tuple[float, float]] = {}

    for hi, lo in comparison_pairs:
        key = f"{hi}_vs_{lo}"
        v_hi = clean_outcomes[hi]
        v_lo = clean_outcomes[lo]
        diff = (sum(v_hi) / len(v_hi)) - (sum(v_lo) / len(v_lo))
        lb, ub = _difference_bounds(v_hi, v_lo, interval_error)
        p_raw = _shifted_one_sided_p_value(v_hi, v_lo, delta=delta, seed=f"v05-perm-{key}")
        decision_bounds[key] = (diff, lb)
        pairwise_deltas[key] = round(diff, 6)
        pairwise_lb95[key] = round(lb, 6)
        pairwise_ub95[key] = round(ub, 6)
        pairwise_p_raw[key] = round(p_raw, 6)
        raw_p_list.append(p_raw)

    holm_p_list = holm_adjust(raw_p_list + [1.0] * (family_size - len(raw_p_list)))[:len(raw_p_list)] if raw_p_list else []
    pairwise_p_holm: dict[str, float] = {
        f"{hi}_vs_{lo}": round(p_h, 6)
        for (hi, lo), p_h in zip(comparison_pairs, holm_p_list)
    }

    p_vals = clean_outcomes["P"]
    mean_p = arm_means["P"]
    delta_p_y = round(mean_p - float(terminal_y), 6)
    lb_p_y = arm_lb95_abs["P"] - float(terminal_y)

    r_key = "R" if "R" in clean_outcomes else ("R_single" if "R_single" in clean_outcomes else ("R_explicit" if "R_explicit" in clean_outcomes else None))
    mean_r = arm_means.get(r_key) if r_key else None
    mean_r_all = arm_means.get("R_all")
    delta_r_single_all = round(mean_r - mean_r_all, 6) if (mean_r is not None and mean_r_all is not None) else None

    vec = E2ResponseVector(
        evaluated=True,
        inference={"method": "binomial_exact_or_bounded_hoeffding_bonferroni_with_delta_null_holm",
                   "primary_family_size": family_size, "comparison_ids": list(comparison_ids or local_ids),
                   "scope": "frozen_family" if primary_family_size is not None else "exploratory_local",
                   "endpoint_error": interval_error},
        anchor_k=anchor_k,
        anchor_type=anchor_type,
        remaining_budget_at_anchor=remaining_budget,
        arm_outcomes=clean_outcomes,
        branch_identities={arm: list(ids) for arm, ids in (branch_identities or {}).items()},
        arm_means=arm_means,
        arm_lb95_abs=arm_lb95_abs,
        arm_ub95_abs=arm_ub95_abs,
        pairwise_deltas=pairwise_deltas,
        pairwise_lb95=pairwise_lb95,
        pairwise_ub95=pairwise_ub95,
        pairwise_p_raw=pairwise_p_raw,
        pairwise_p_holm=pairwise_p_holm,
        ineligible_arms=list(ineligible_arms),
        mean_P=mean_p,
        mean_D=arm_means.get("D"),
        mean_S=arm_means.get("S"),
        mean_R=mean_r,
        mean_R_explicit=arm_means.get("R_explicit"),
        mean_R_implicit=arm_means.get("R_implicit"),
        mean_R_all=mean_r_all,
        delta_P_vs_Y=delta_p_y,
        delta_D_vs_P=pairwise_deltas.get("D_vs_P"),
        delta_S_vs_P=pairwise_deltas.get("S_vs_P"),
        delta_R_vs_D=pairwise_deltas.get(f"{r_key}_vs_D") if r_key else None,
        delta_R_vs_P=pairwise_deltas.get(f"{r_key}_vs_P") if r_key else None,
        delta_R_single_vs_R_all=delta_r_single_all,
        lb95_P_vs_Y=lb_p_y,
        lb95_D_vs_P=pairwise_lb95.get("D_vs_P"),
        ub95_D_vs_P=pairwise_ub95.get("D_vs_P"),
        lb95_R_vs_D=pairwise_lb95.get(f"{r_key}_vs_D") if r_key else None,
        ub95_R_vs_D=pairwise_ub95.get(f"{r_key}_vs_D") if r_key else None,
        lb95_R_vs_P=pairwise_lb95.get(f"{r_key}_vs_P") if r_key else None,
        ub95_R_vs_P=pairwise_ub95.get(f"{r_key}_vs_P") if r_key else None,
        p_val_D_vs_P=pairwise_p_raw.get("D_vs_P"),
        p_val_R_vs_D=pairwise_p_raw.get(f"{r_key}_vs_D") if r_key else None,
        p_holm_D_vs_P=pairwise_p_holm.get("D_vs_P"),
        p_holm_R_vs_D=pairwise_p_holm.get(f"{r_key}_vs_D") if r_key else None,
    )

    # Step 1: Level 0 needs an absolute recovery lower bound strictly above tau.
    if arm_lb95_abs["P"] > tau or lb_p_y > tau:
        return vec, None, "plain_recoverable"

    # Step 2: Check ladder levels k = 1, 2, 3 in ascending order using incremental threshold `delta`
    label_map = {
        "D": "delivery_responsive",
        "S": "self_check_responsive",
        "R_explicit": "explicit_rule_responsive",
        "R": "rule_responsive",
        "R_single": "rule_responsive",
        "R_implicit": "implicit_rule_responsive",
    }

    for target_level in (1, 2, 3):
        candidates_at_level = [
            a for a in eval_arms
            if a in primary_arms and LADDER_ARM_LEVELS.get(a, 99) == target_level and a in label_map
        ]
        if not candidates_at_level:
            continue
        # Sort so that if both D and S pass at level 1, higher mean wins (tie-breaker D before S)
        candidates_at_level.sort(key=lambda a: (-arm_means[a], 0 if a == "D" else 1))
        for cand_arm in candidates_at_level:
            lower_arms = [
                lo for lo in eval_arms
                if lo in primary_arms and LADDER_ARM_LEVELS.get(lo, 99) < target_level
            ]
            if not lower_arms:
                continue
            beats_all_lower = True
            for lo in lower_arms:
                pair_key = f"{cand_arm}_vs_{lo}"
                if (
                    decision_bounds.get(pair_key, (-1.0, -1.0))[0] <= delta
                    or decision_bounds.get(pair_key, (-1.0, -1.0))[1] <= delta
                    or holm_p_list[local_ids.index(pair_key)] > alpha
                ):
                    beats_all_lower = False
                    break
            if beats_all_lower:
                return vec, label_map[cand_arm], "adjudicated"

    # Step 3: Unrecovered under protocol — requires ALL evaluated arms (including P) to have
    # absolute UB95 < tau AND all primary incremental upper bounds < delta.
    non_r_all = [a for a in eval_arms if a in primary_arms]
    all_abs_below_tau = all(arm_ub95_abs[a] < tau for a in non_r_all)
    all_increment_bounds_below_delta = all(
        pairwise_ub95[f"{hi}_vs_{lo}"] < delta for hi, lo in primary_pairs
    )
    if all_abs_below_tau and all_increment_bounds_below_delta:
        return vec, "unrecovered_under_protocol", "adjudicated"

    return vec, None, "inconclusive"


def build_v3_stratified_sampling_frame(
    units: Sequence[DiagnosticUnitV05],
    *,
    dataset_id: str,
    actor_model: str = "Qwen/Qwen3.6-27B",
    target_sample_per_stratum: int | dict[str, int] = 4,
    seed: str = "v3-d1-stratified-frame",
) -> dict[str, Any]:
    """R11 (V3 §3.4 & §5.2): Build the 3D stratified sampling frame on the NEW `dev45` only:
      `exhausted_budget (bool) x valid_delivery (bool) x rule_source (str)`

    Rejects legacy `deepseek` datasets to prevent cross-configuration contamination, and preserves
    each sampled unit's true checkpoint `anchor_k`, `anchor_type`, `remaining_budget`, `defect_name`,
    and `unit_level`.
    """
    if "deepseek" in dataset_id.lower() or "deepseek" in actor_model.lower():
        raise L2Error(
            f"R11 violation: sampling frame must only be built from the new Qwen/Qwen3.6-27B + pi dev45, "
            f"got dataset_id={dataset_id!r}, actor_model={actor_model!r}"
        )

    strata: dict[str, list[DiagnosticUnitV05]] = {}
    for u in units:
        if u.e1_observation.e1_class != "never_succeeded_on_disk":
            continue
        if u.evidence_status == "ineligible":
            continue
        exh = "exhausted" if bool(u.exhausted_budget) else "voluntary"
        vdel = "valid_delivery" if bool(u.valid_delivery) else "no_valid_delivery"
        rsrc = u.rule_source
        sid = f"{exh}__{vdel}__{rsrc}"
        u.stratum_id = sid
        strata.setdefault(sid, []).append(u)

    rng = random.Random(int(digest({"seed": seed, "dataset_id": dataset_id, "n": len(units)})[:16], 16))
    sampled_units: list[dict[str, Any]] = []
    stratum_summary: dict[str, dict[str, Any]] = {}

    for sid in sorted(strata.keys()):
        pool = sorted(strata[sid], key=lambda x: x.unit_id)
        N_h = len(pool)
        if isinstance(target_sample_per_stratum, dict):
            n_h = min(N_h, int(target_sample_per_stratum.get(sid, 4)))
        else:
            n_h = min(N_h, int(target_sample_per_stratum))
        if n_h <= 0:
            stratum_summary[sid] = {"population_count": N_h, "sampled_count": 0,
                "inclusion_prob": 0.0, "population_deduction": sum(x.deduction_weight for x in pool),
                "sampled_unit_ids": []}
            continue
        pi_h = n_h / N_h
        chosen = sorted(rng.sample(pool, k=n_h), key=lambda x: x.unit_id)
        for u in chosen:
            u.inclusion_prob = round(pi_h, 6)
            sampled_units.append({
                "unit_id": u.unit_id,
                "episode_id": u.episode_id,
                "task_id": u.task_id,
                "family_id": u.family_id,
                "unit_level": u.unit_level,
                "defect_name": u.defect_name,
                "stratum_id": sid,
                "exhausted_budget": bool(u.exhausted_budget),
                "valid_delivery": bool(u.valid_delivery),
                "rule_source": u.rule_source,
                "deduction_weight": u.deduction_weight,
                "stratum_population_size": N_h,
                "stratum_sample_size": n_h,
                "inclusion_prob": round(pi_h, 6),
                "anchor_k": u.e2_response_vector.anchor_k,
                "anchor_type": u.e2_response_vector.anchor_type,
                "remaining_budget": u.e2_response_vector.remaining_budget_at_anchor,
                "early_anchor_k": u.early_anchor_k,
                "early_anchor_remaining_budget": u.early_anchor_remaining_budget,
            })
        stratum_summary[sid] = {
            "population_count": N_h,
            "sampled_count": n_h,
            "inclusion_prob": round(pi_h, 6),
            "population_deduction": round(sum(x.deduction_weight for x in pool), 6),
            "sampled_unit_ids": [x.unit_id for x in chosen],
        }

    return {
        "schema_version": V05_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "actor_model": actor_model,
        "stratification_axes": ["exhausted_budget", "valid_delivery", "rule_source"],
        "total_eligible_never_units": sum(len(v) for v in strata.values()),
        "total_sampled_units": len(sampled_units),
        "strata": stratum_summary,
        "sampled_units": sampled_units,
    }


def evaluate_harm_on_passing_units(
    passing_rows: Sequence[dict[str, Any]],
    *,
    delta: float = 0.10,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """R12 (V3 §3.4): Evaluate potential harm (score degradation) of generic/explicit interventions
    (`D`, `S`, `R_explicit`) relative to `P` on checkpoints where the parent trajectory or target item
    already succeeded (`terminal_y == 1.0` or `item_score == 1.0`).

    Each entry in `passing_rows` contains:
      - `unit_id`: str
      - `arm_outcomes`: dict[str, Sequence[float]] (containing `"P"` and one or more of `"D"`, `"S"`, `"R_explicit"`)
    """
    unit_reports: list[dict[str, Any]] = []
    harm_counts_by_arm: dict[str, int] = {"D": 0, "S": 0, "R_explicit": 0}
    evaluated_counts_by_arm: dict[str, int] = {"D": 0, "S": 0, "R_explicit": 0}
    deltas_by_arm: dict[str, list[float]] = {"D": [], "S": [], "R_explicit": []}

    for row in passing_rows:
        uid = str(row.get("unit_id") or "passing_unit")
        outcomes = row.get("arm_outcomes") or {}
        p_vals = [float(x) for x in outcomes.get("P", [])]
        if not p_vals:
            continue
        mean_p = sum(p_vals) / len(p_vals)
        arm_harm: dict[str, Any] = {}
        for arm in ("D", "S", "R_explicit"):
            if arm not in outcomes or not outcomes[arm]:
                continue
            a_vals = [float(x) for x in outcomes[arm]]
            mean_a = sum(a_vals) / len(a_vals)
            diff_a_p = mean_a - mean_p
            # Drop from P to A is mean_p - mean_a; significant harm if LB95(P - A) > delta
            lb_drop_p_minus_a = lower_drop(p_vals, a_vals, alpha=alpha, seed=f"v05-harm-{uid}-{arm}")
            p_harm = permutation_one_sided(p_vals, a_vals, direction=1.0, seed=f"v05-harm-p-{uid}-{arm}")
            is_harmful = (mean_p - mean_a >= delta) and (lb_drop_p_minus_a > 0.0) and (p_harm <= alpha)
            evaluated_counts_by_arm[arm] += 1
            if is_harmful:
                harm_counts_by_arm[arm] += 1
            deltas_by_arm[arm].append(diff_a_p)
            arm_harm[arm] = {
                "mean_arm": round(mean_a, 6),
                "mean_P": round(mean_p, 6),
                "delta_arm_minus_P": round(diff_a_p, 6),
                "lb95_drop_P_minus_arm": round(lb_drop_p_minus_a, 6),
                "p_val_harm_one_sided": round(p_harm, 6),
                "significant_harm": is_harmful,
            }
        unit_reports.append({
            "unit_id": uid,
            "mean_P": round(mean_p, 6),
            "comparisons": arm_harm,
        })

    summary_by_arm: dict[str, Any] = {}
    for arm in ("D", "S", "R_explicit"):
        n_eval = evaluated_counts_by_arm[arm]
        d_list = deltas_by_arm[arm]
        summary_by_arm[arm] = {
            "evaluated_units": n_eval,
            "significant_harm_count": harm_counts_by_arm[arm],
            "significant_harm_rate": round(harm_counts_by_arm[arm] / n_eval, 6) if n_eval > 0 else 0.0,
            "mean_delta_arm_minus_P": round(sum(d_list) / len(d_list), 6) if d_list else None,
        }

    return {
        "schema_version": V05_SCHEMA_VERSION,
        "evaluated_unit_count": len(unit_reports),
        "summary_by_arm": summary_by_arm,
        "units": unit_reports,
    }


def compute_cross_fitted_diagnostic_value_ceiling(
    units: Sequence[DiagnosticUnitV05], *,
    candidate_arms: Sequence[str] = ("P", "D", "S", "R_explicit", "R_single"),
    fixed_arm_pool: Sequence[str] = ("P", "D", "S", "R_explicit"),
    ceiling_threshold: float = 0.05, alpha: float = 0.05,
    min_replicates_per_arm: int = 2, seed: str = "v3-d2-crossfit-ceiling",
    estimand: str = "family_population",
) -> dict[str, Any]:
    """Select both policies on one branch half, evaluate the other, then swap.

    Single-rule diagnosis is permitted only for explicit rules. Missing applicable
    outcomes invalidate a unit; they never become zero successes. Family-cluster
    percentile intervals are descriptive; sparse/constant clusters receive bounded
    Hoeffding intervals instead of spuriously degenerate bootstrap intervals.
    """
    if estimand not in {"family_population", "conditional_frozen_dev_pool"}:
        raise L2Error("Unknown diagnostic ceiling estimand")
    if estimand == "conditional_frozen_dev_pool":
        auxiliary = compute_cross_fitted_diagnostic_value_ceiling(
            units, candidate_arms=candidate_arms, fixed_arm_pool=fixed_arm_pool,
            ceiling_threshold=ceiling_threshold, alpha=alpha,
            min_replicates_per_arm=min_replicates_per_arm, seed=seed,
        )
        return _conditional_frozen_pool_ceiling(
            units, auxiliary, candidate_arms=candidate_arms,
            fixed_arm_pool=fixed_arm_pool, alpha=alpha,
            ceiling_threshold=ceiling_threshold,
        )
    if len({u.unit_id for u in units}) != len(units):
        raise L2Error("Duplicate diagnostic unit identities")
    if any(not 0 < u.inclusion_prob <= 1 or u.deduction_weight < 0 for u in units):
        raise L2Error("Invalid inclusion probability or deduction weight")
    allowed_diag = {"P", "D", "S", "R_explicit", "R_single"}
    allowed_fixed = {"P", "D", "S", "R_explicit"}
    if set(candidate_arms) - allowed_diag or set(fixed_arm_pool) - allowed_fixed:
        raise L2Error("Ceiling arm pools contain hidden or unclassified rule information")
    diag = list(dict.fromkeys(("P", *candidate_arms)))
    fixed = list(dict.fromkeys(("P", *fixed_arm_pool)))
    req = max(2, min_replicates_per_arm)
    eligible = []
    rejected = []
    available = {}
    for u in units:
        outcomes = u.e2_response_vector.arm_outcomes
        applicable = [arm for arm in dict.fromkeys(diag + fixed)
                      if arm not in u.e2_response_vector.ineligible_arms
                      and (arm != "R_single" or u.rule_source == "explicit")]
        # Globally absent optional arms are not part of this observed protocol.
        applicable = [arm for arm in applicable if arm == "P" or any(
            arm in x.e2_response_vector.arm_outcomes for x in units)]
        if (not u.e2_response_vector.evaluated or len(applicable) < 2
            or any(len(outcomes.get(arm, [])) < req for arm in applicable)):
            rejected.append(u.unit_id)
            continue
        eligible.append(u)
        available[u.unit_id] = applicable
    common_fixed = [arm for arm in fixed if eligible and all(arm in available[u.unit_id] for u in eligible)]
    if "P" not in common_fixed:
        common_fixed.insert(0, "P")
    missing_fixed_arms = [arm for arm in fixed if arm not in common_fixed]
    fixed_policy_pool_complete = not missing_fixed_arms
    rows = []
    folds = []
    def mean_half(u, arm, half):
        values = u.e2_response_vector.arm_outcomes[arm]
        mid = len(values) // 2
        part = values[:mid] if half == 0 else values[mid:]
        return sum(part) / len(part)
    for selection_half in (0, 1):
        def policy_score(arm, weighted):
            weights = [u.deduction_weight / u.inclusion_prob if weighted else 1.0 for u in eligible]
            denom = sum(weights)
            return sum(mean_half(u, arm, selection_half) * w for u, w in zip(eligible, weights)) / denom if denom else 0.0
        best_unw = max(common_fixed, key=lambda arm: policy_score(arm, False))
        best_w = max(common_fixed, key=lambda arm: policy_score(arm, True))
        folds.append({"fold": selection_half + 1, "selection_half": selection_half,
                      "evaluation_half": 1 - selection_half,
                      "best_fixed_arm_unweighted": best_unw, "best_fixed_arm_weighted": best_w})
        for u in eligible:
            choices = [arm for arm in diag if arm in available[u.unit_id]]
            chosen = max(choices, key=lambda arm: mean_half(u, arm, selection_half))
            ph = mean_half(u, "P", 1 - selection_half)
            gain = mean_half(u, chosen, 1 - selection_half) - ph
            fu = mean_half(u, best_unw, 1 - selection_half) - ph
            fw = mean_half(u, best_w, 1 - selection_half) - ph
            split_indices = {}
            for arm in available[u.unit_id]:
                n = len(u.e2_response_vector.arm_outcomes[arm]); mid = n // 2
                halves = [list(range(mid)), list(range(mid, n))]
                split_indices[arm] = {"selection": halves[selection_half], "evaluation": halves[1 - selection_half]}
            rows.append({"fold": selection_half + 1, "unit_id": u.unit_id, "task_id": u.task_id, "family_id": u.family_id,
                         "deduction_weight": u.deduction_weight / u.inclusion_prob, "selected_unit_arm": chosen,
                         "train_best_fixed_arm_unweighted": best_unw, "train_best_fixed_arm_weighted": best_w,
                         "oracle_eval_gain": gain, "fixed_unweighted_eval_gain": fu, "fixed_weighted_eval_gain": fw,
                         "ceiling_increment_unweighted": gain - fu, "ceiling_increment_weighted": gain - fw,
                         "branch_split_indices": split_indices})
    # Condition on each selection half separately. Shared fixed-arm selection then
    # remains fixed during evaluation; combine the two simultaneous intervals only
    # after calculating them. Family weights preserve parent/item dependence.
    def ratio(items, weighted):
        ws = [r["deduction_weight"] if weighted else 1.0 for r in items]
        key = "ceiling_increment_weighted" if weighted else "ceiling_increment_unweighted"
        den = sum(ws)
        return sum(r[key] * w for r, w in zip(items, ws)) / den if den else 0.0
    unw, weighted = ratio(rows, False), ratio(rows, True)
    families = sorted({u.family_id or "__unknown__" for u in eligible})
    direction_intervals = []
    for direction in (1, 2):
        items = [row for row in rows if row["fold"] == direction]
        record = {"fold": direction}
        for name, use_weights in (("unweighted", False), ("weighted", True)):
            family_weights = {}
            for row in items:
                family_key = row["family_id"] or "__unknown__"
                family_weights[family_key] = family_weights.get(family_key, 0.0) + (row["deduction_weight"] if use_weights else 1.0)
            denom = sum(family_weights.values())
            norm_sq = sum((w / denom)**2 for w in family_weights.values()) if denom else 1.0
            # alpha/2 directions, alpha/2 tails within each direction.
            radius = math.sqrt(2 * math.log(4 / alpha) * norm_sq)
            estimate = ratio(items, use_weights)
            record[name] = [max(-1.0, estimate - radius), min(1.0, estimate + radius)]
            record[f"{name}_family_weights"] = family_weights
        direction_intervals.append(record)
    ci_u = [sum(d["unweighted"][i] for d in direction_intervals) / 2 for i in (0, 1)]
    ci_w = [sum(d["weighted"][i] for d in direction_intervals) / 2 for i in (0, 1)]
    ci_method = "conditional_direction_family_weighted_hoeffding_bonferroni"
    return {"schema_version": V05_SCHEMA_VERSION, "proposition_id": "V3-P3",
            "eligible_unit_count": len(eligible), "ineligible_unit_ids": rejected,
            "min_replicates_per_arm_required": req, "fixed_arm_pool": common_fixed,
            "diagnostic_arm_pool": diag, "folds_disjoint": True, "branches_disjoint": True,
            "fixed_policy_pool_complete": fixed_policy_pool_complete,
            "missing_fixed_arms": missing_fixed_arms,
            "evidence_status": "inconclusive" if not fixed_policy_pool_complete else "evaluated",
            "split_unit": "continuation_branches_within_unit", "family_cluster_count": len(families),
            "unweighted_ceiling": unw, "deduction_weighted_ceiling": weighted,
            "unweighted_lb95": ci_u[0], "unweighted_ub95": ci_u[1],
            "deduction_weighted_lb95": ci_w[0], "deduction_weighted_ub95": ci_w[1],
            "interval_method": ci_method, "direction_intervals": direction_intervals,
            "best_case_equal_family_lower_bound_at_perfect_gain": 1 - math.sqrt(2 * math.log(4 / alpha) / max(1, len(families))),
            "threshold_attainable_with_observed_family_count": 1 - math.sqrt(2 * math.log(4 / alpha) / max(1, len(families))) > ceiling_threshold,
            "weighting": "deduction_weight/inclusion_prob", "ceiling_threshold": ceiling_threshold,
            "family_ids_complete": all(u.family_id for u in eligible),
            "passes_v3_p3_gate": fixed_policy_pool_complete and len(families) >= 2 and all(u.family_id for u in eligible) and ci_w[0] > ceiling_threshold,
            "folds": folds, "unit_evaluations": rows}


def _conditional_frozen_pool_ceiling(
    units: Sequence[DiagnosticUnitV05], auxiliary: dict[str, Any], *,
    candidate_arms: Sequence[str], fixed_arm_pool: Sequence[str],
    alpha: float, ceiling_threshold: float,
) -> dict[str, Any]:
    """Condition on frozen parents and selection outcomes; group physical draws.

    A package may score multiple rubric endpoints with arbitrary dependence. Only
    separate physical packages are independent conditional on fixed parents and
    the given direction's own selection-half outcomes. The endpoint coefficients combine
    before computing each package's worst-case range. Each direction is bounded
    separately; averaging simultaneous bounds needs no independence of folds.
    Reference: https://www.cs.mcgill.ca/~colt2009/papers/012.pdf, Theorem 11.
    """
    if not 0 < alpha < 1:
        raise L2Error("alpha must lie strictly between zero and one")
    requested = list(dict.fromkeys(("P", *candidate_arms, *fixed_arm_pool)))
    complete = bool(units) and not auxiliary["ineligible_unit_ids"]
    halves: dict[str, int] = {}
    identity_context: dict[str, tuple[str, str, str]] = {}
    endpoint_values: dict[tuple[str, tuple[str, str, str]], float] = {}
    for unit in units:
        vec = unit.e2_response_vector
        if (not unit.episode_id or not unit.task_id or not unit.family_id
                or not math.isfinite(unit.deduction_weight)
                or not math.isfinite(unit.inclusion_prob)):
            raise L2Error("Conditional pool requires complete finite parent metadata")
        applicable = [a for a in requested if a not in vec.ineligible_arms
                      and (a != "R_single" or unit.rule_source == "explicit")]
        if any(a not in applicable for a in fixed_arm_pool):
            complete = False
        for arm in applicable:
            values = vec.arm_outcomes.get(arm, [])
            if len(values) < auxiliary["min_replicates_per_arm_required"]:
                complete = False
            ids = vec.branch_identities.get(arm)
            if ids is None or len(ids) != len(values):
                raise L2Error("Conditional inference requires aligned verified physical branch identities")
            if len(set(ids)) != len(ids):
                raise L2Error("Duplicate physical branch within an arm")
            for index, (identity, value) in enumerate(zip(ids, values)):
                if not isinstance(identity, str) or re.fullmatch(r"[0-9a-f]{64}", identity) is None:
                    raise L2Error("Physical branch identity must be a package-index SHA256 digest")
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise L2Error("Conditional outcomes must be finite and bounded in [0, 1]")
                half = int(index >= len(values) // 2)
                if identity in halves and halves[identity] != half:
                    raise L2Error("Physical branch shared across selection and evaluation halves")
                halves[identity] = half
                context = (unit.episode_id, unit.task_id, arm)
                if identity in identity_context and identity_context[identity] != context:
                    raise L2Error("Inconsistent parent or arm for physical branch identity")
                identity_context[identity] = context
                endpoint = (identity, (unit.task_id, unit.unit_level, unit.defect_name))
                if endpoint in endpoint_values and endpoint_values[endpoint] != value:
                    raise L2Error("Inconsistent outcome for shared physical branch endpoint")
                endpoint_values[endpoint] = value
    if sum(u.deduction_weight / u.inclusion_prob for u in units) <= 0:
        complete = False
    eta = alpha / 8  # 2 directions x 2 tails x 2 concentration methods, per target.
    directions = []
    for fold in (1, 2):
        rows = [row for row in auxiliary["unit_evaluations"] if row["fold"] == fold]
        record: dict[str, Any] = {"fold": fold, "selection_half": fold - 1,
                                  "evaluation_half": 2 - fold}
        by_id = {unit.unit_id: unit for unit in units}
        for name, weighted in (("unweighted", False), ("weighted", True)):
            denominator = sum(row["deduction_weight"] if weighted else 1.0 for row in rows)
            coefficients: dict[tuple[str, tuple[str, str, str]], float] = {}
            for row in rows:
                unit = by_id[row["unit_id"]]
                weight = (row["deduction_weight"] if weighted else 1.0) / denominator if denominator else 0.0
                fixed_arm = row[f"train_best_fixed_arm_{name}"]
                if row["selected_unit_arm"] == fixed_arm:
                    continue
                for arm, sign in ((row["selected_unit_arm"], 1), (fixed_arm, -1)):
                    indices = row["branch_split_indices"][arm]["evaluation"]
                    for index in indices:
                        endpoint = (unit.e2_response_vector.branch_identities[arm][index],
                                    (unit.task_id, unit.unit_level, unit.defect_name))
                        coefficients[endpoint] = coefficients.get(endpoint, 0.0) + sign * weight / len(indices)
            # Exact cancellation follows from the selected policies, independent of Y.
            coefficients = {key: value for key, value in coefficients.items() if value != 0.0}
            groups: dict[str, dict[str, float]] = {}
            for endpoint, coefficient in coefficients.items():
                group = groups.setdefault(endpoint[0], {"lower": 0.0, "upper": 0.0, "contribution": 0.0})
                group["lower"] += min(0.0, coefficient)
                group["upper"] += max(0.0, coefficient)
                group["contribution"] += coefficient * endpoint_values[endpoint]
            estimate = sum(group["contribution"] for group in groups.values())
            ranges = [group["upper"] - group["lower"] for group in groups.values()]
            m = len(groups)
            hoeffding = math.sqrt(0.5 * math.log(1 / eta) * sum(r * r for r in ranges))
            eb = None
            if m >= 2:
                c = max(ranges)
                normalized = [(g["contribution"] - g["lower"]) / c for g in groups.values()]
                mean = sum(normalized) / m
                variance = sum((x - mean) ** 2 for x in normalized) / (m - 1)
                # Maurer & Pontil (2009), Theorem 11, applied to independent,
                # non-identically distributed [0,1] variables; additive term is
                # essential even when the observed variance is zero.
                log_term = math.log(2 / eta)
                eb = m * c * (math.sqrt(2 * variance * log_term / m)
                              + 7 * log_term / (3 * (m - 1)))
            radius = min(hoeffding, eb) if eb is not None else hoeffding
            record[name] = {
                "estimate": estimate, "lb95": max(-1.0, estimate - radius),
                "ub95": min(1.0, estimate + radius),
                "physical_group_count": m, "endpoint_coefficient_count": len(coefficients),
                "physical_range_sum_squares": sum(r * r for r in ranges),
                "hoeffding_radius": hoeffding, "empirical_bernstein_radius": eb,
                "interval_method": "exact_policy_identity" if not groups else "physical_group_hoeffding_or_maurer_pontil_2009_thm11",
            }
        directions.append(record)
    result = dict(auxiliary)
    result["family_population_auxiliary"] = auxiliary
    result["estimand"] = "conditional_frozen_dev_pool"
    result["conditional_pool_complete"] = complete
    result["conditional_ci_coverage"] = 1 - alpha
    result["conditional_ci_coverage_scope"] = "per_target_two_sided_across_both_directions"
    result["interval_method"] = "conditional_physical_group_concentration"
    result["direction_intervals"] = directions
    result["error_allocation"] = {
        "alpha_per_target": alpha, "eta_per_direction_tail_method": eta,
        "directions": 2, "tails": 2, "methods": 2,
        "targets_have_separate_coverage": True, "joint_weighted_unweighted_coverage_claimed": False,
    }
    for name, prefix in (("weighted", "deduction_weighted"), ("unweighted", "unweighted")):
        for endpoint in ("lb95", "ub95"):
            bound = sum(d[name][endpoint] for d in directions) / 2
            result[f"conditional_{prefix}_{endpoint}"] = bound if complete else (-1.0 if endpoint == "lb95" else 1.0)
            result[f"{prefix}_{endpoint}"] = result[f"conditional_{prefix}_{endpoint}"]
    result["evidence_status"] = "evaluated" if complete else "inconclusive"
    result["passes_v3_p3_gate"] = complete and result["conditional_deduction_weighted_lb95"] > ceiling_threshold
    result["physical_branch_count"] = len(halves)
    result["physical_branch_count_by_half"] = {str(half): sum(h == half for h in halves.values()) for half in (0, 1)}
    result["conditional_unweighted_ceiling"] = result["unweighted_ceiling"]
    result["conditional_deduction_weighted_ceiling"] = result["deduction_weighted_ceiling"]
    result["conditioning"] = "each_direction_conditions_on_frozen_parents_and_its_own_selection_half"
    result["conditional_target"] = "average_of_two_directional_conditional_policy_values"
    result["independence_unit"] = "physical_continuation_package"
    return result


def predict_majority_baseline_m(
    units: Sequence[DiagnosticUnitV05],
    *,
    default_majority_label: str = "unrecovered_under_protocol",
) -> list[dict[str, Any]]:
    """Baseline M (V3 §3.6): Zero-cost majority-class / fixed-arm predictor."""
    arm_map = {
        "delivery_responsive": "D",
        "self_check_responsive": "S",
        "explicit_rule_responsive": "R_explicit",
        "implicit_rule_responsive": "R_implicit",
        "rule_responsive": "R",
        "unrecovered_under_protocol": "none",
    }
    rec = arm_map.get(default_majority_label, "none")
    preds: list[dict[str, Any]] = []
    for u in units:
        preds.append({
            "unit_id": u.unit_id,
            "analyst_group": "M",
            "p_prob": 0.05,
            "d_prob": 0.80 if rec == "D" else 0.05,
            "s_prob": 0.80 if rec == "S" else 0.05,
            "r_explicit_prob": 0.80 if rec == "R_explicit" else 0.05,
            "r_implicit_prob": 0.80 if rec == "R_implicit" else 0.05,
            "r_prob": 0.80 if rec in ("R", "R_explicit", "R_implicit") else 0.05,
            "recommended_arm": rec,
            "predicted_label": default_majority_label,
            "confidence": 1.0,
            "abstain": False,
            "evidence_steps": [],
        })
    return preds


def predict_rule_analyst_a_v05(
    unit: DiagnosticUnitV05,
    facts: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Analyst Group A (V3 §3.6): Deterministic rule table on L2 facts (without seeing s(k) or LLM)."""
    submission_writes = [
        int(f.get("step", 0))
        for f in facts
        if f.get("type") == "submission_artifact_Step"
        or (f.get("type") == "workspace_file_op" and str((f.get("value") or {}).get("path", "")).startswith("submission/"))
    ]
    workspace_writes = [
        int(f.get("step", 0))
        for f in facts
        if f.get("type") == "workspace_file_op"
        and not str((f.get("value") or {}).get("path", "")).startswith("submission/")
    ]
    tool_anomalies = [
        int(f.get("step", 0))
        for f in facts
        if f.get("type") in ("tool_error", "non_zero_exit")
    ]

    if not submission_writes and workspace_writes:
        return {
            "unit_id": unit.unit_id,
            "analyst_group": "A",
            "p_prob": 0.10,
            "d_prob": 0.65,
            "s_prob": 0.45,
            "r_explicit_prob": 0.55,
            "r_implicit_prob": 0.55,
            "r_prob": 0.55,
            "recommended_arm": "D",
            "predicted_label": "delivery_responsive",
            "confidence": 0.65,
            "abstain": False,
            "evidence_steps": [workspace_writes[-1]],
        }
    if submission_writes and not tool_anomalies:
        rec_arm = "R_implicit" if unit.rule_source == "implicit" else "R_explicit"
        pred_lbl = "implicit_rule_responsive" if unit.rule_source == "implicit" else "explicit_rule_responsive"
        return {
            "unit_id": unit.unit_id,
            "analyst_group": "A",
            "p_prob": 0.05,
            "d_prob": 0.10,
            "s_prob": 0.30,
            "r_explicit_prob": 0.55 if rec_arm == "R_explicit" else 0.25,
            "r_implicit_prob": 0.55 if rec_arm == "R_implicit" else 0.35,
            "r_prob": 0.55,
            "recommended_arm": "P" if unit.rule_source == "implicit" else rec_arm,
            "predicted_attribution_arm": rec_arm,
            "predicted_label": pred_lbl,
            "confidence": 0.55,
            "abstain": unit.rule_source == "implicit",
            "evidence_steps": [submission_writes[-1]],
        }
    return {
        "unit_id": unit.unit_id,
        "analyst_group": "A",
        "p_prob": 0.05,
        "d_prob": 0.08,
        "s_prob": 0.10,
        "r_explicit_prob": 0.12,
        "r_implicit_prob": 0.15,
        "r_prob": 0.15,
        "recommended_arm": "none",
        "predicted_label": "unrecovered_under_protocol",
        "confidence": 0.60,
        "abstain": False,
        "evidence_steps": [],
    }


def evaluate_analysts_v05(
    units: Sequence[DiagnosticUnitV05],
    predictions: Sequence[dict[str, Any]],
    *,
    cost_per_unit_requests: float = 144.0,
) -> dict[str, Any]:
    """R13, R14 & R17 (V3 §3.6 & §3.7): Evaluate Analyst predictions (M / A / B / C) using
    cross-fitted diagnostic value ceiling (NEVER in-sample `max_possible_gain`), preserving signed
    actual gains (never clamping negative harm to 0.0) and reporting both full-sample and adjudicated
    coverage counts.
    """
    by_id = {u.unit_id: u for u in units}
    pred_by_id = {p["unit_id"]: p for p in predictions if p.get("unit_id") in by_id}

    all_matched_units = [u for u in units if u.unit_id in pred_by_id]
    inconclusive_units = [u for u in units if u.evidence_status == "inconclusive"]
    plain_recoverable_units = [u for u in units if u.evidence_status == "plain_recoverable"]
    not_applicable_units = [u for u in units if u.evidence_status in ("not_applicable", "ineligible")]

    adjudicated_e2 = [
        u for u in units
        if u.evidence_status == "adjudicated"
        and u.operational_label in (
            "delivery_responsive",
            "self_check_responsive",
            "explicit_rule_responsive",
            "implicit_rule_responsive",
            "rule_responsive",
            "unrecovered_under_protocol",
        )
        and u.unit_id in pred_by_id
    ]

    policy_unmeasured = len(units) - len(all_matched_units)
    inadmissible_policy_units: list[str] = []
    ranked_items: list[tuple[float, float, float]] = []
    for u in all_matched_units:
        p = pred_by_id[u.unit_id]
        abstain = bool(p.get("abstain", False))
        rec_arm = "none" if abstain else str(p.get("recommended_arm") or "none")
        allowed_policy_arms = {"none", "P", "D", "S", "R_explicit"}
        if u.rule_source == "explicit":
            allowed_policy_arms.add("R_single")
        if rec_arm not in allowed_policy_arms:
            inadmissible_policy_units.append(u.unit_id)
            continue
        prob_map = {
            "D": float(p.get("d_prob", 0.0)),
            "S": float(p.get("s_prob", p.get("d_prob", 0.0))),
            "R_explicit": float(p.get("r_explicit_prob", p.get("r_prob", 0.0))),
            "R_implicit": float(p.get("r_implicit_prob", p.get("r_prob", 0.0))),
            "R": float(p.get("r_prob", 0.0)),
            "R_single": float(p.get("r_prob", 0.0)),
        }
        if rec_arm in ("none", "P"):
            priority = 0.0
            gain_rate = 0.0
            req_cost = 0.0
        else:
            priority = prob_map.get(rec_arm, 0.0) * u.deduction_weight
            vec = u.e2_response_vector
            pair_k = f"{rec_arm}_vs_P"
            if pair_k in vec.pairwise_deltas:
                gain_rate = float(vec.pairwise_deltas[pair_k])
            elif rec_arm == "D" and vec.delta_D_vs_P is not None:
                gain_rate = float(vec.delta_D_vs_P)
            elif rec_arm == "S" and vec.delta_S_vs_P is not None:
                gain_rate = float(vec.delta_S_vs_P)
            elif rec_arm == "R" and "R" in vec.arm_outcomes and vec.delta_R_vs_P is not None:
                gain_rate = float(vec.delta_R_vs_P)
            elif rec_arm in vec.arm_means and "P" in vec.arm_means:
                gain_rate = vec.arm_means[rec_arm] - vec.arm_means["P"]
            else:
                policy_unmeasured += 1
                continue
            req_cost = float(p.get("verification_requests", cost_per_unit_requests))

        actual_gain = gain_rate * u.deduction_weight
        ranked_items.append((priority, actual_gain, req_cost))

    ranked_items.sort(key=lambda x: -x[0])
    total_active_requests = sum(c for _, _, c in ranked_items)
    total_actual_gain = sum(g for _, g, c in ranked_items)
    utility_per_100_req = (total_actual_gain / total_active_requests * 100.0) if total_active_requests > 0 else 0.0

    # Cross-fitted diagnostic value ceiling (replaces in-sample max_possible_gain)
    crossfit = compute_cross_fitted_diagnostic_value_ceiling(units)

    classes = (
        "delivery_responsive",
        "self_check_responsive",
        "explicit_rule_responsive",
        "implicit_rule_responsive",
        "rule_responsive",
        "unrecovered_under_protocol",
    )
    total_w = sum(u.deduction_weight for u in adjudicated_e2)
    correct_w = 0.0
    f1_per_class: dict[str, float] = {}

    def _infer_label(p_row: dict[str, Any]) -> str:
        if p_row.get("abstain", False):
            return "abstain"
        lbl = p_row.get("predicted_label")
        if lbl:
            return str(lbl)
        rec = str(p_row.get("recommended_arm") or "none")
        return {
            "D": "delivery_responsive",
            "S": "self_check_responsive",
            "R_explicit": "explicit_rule_responsive",
            "R_implicit": "implicit_rule_responsive",
            "R": "rule_responsive",
            "R_single": "rule_responsive",
        }.get(rec, "unrecovered_under_protocol")

    for u in adjudicated_e2:
        pred_lbl = _infer_label(pred_by_id[u.unit_id])
        if pred_lbl == u.operational_label:
            correct_w += u.deduction_weight

    for cls in classes:
        tp_w = sum(
            u.deduction_weight for u in adjudicated_e2
            if u.operational_label == cls and _infer_label(pred_by_id[u.unit_id]) == cls
        )
        fp_w = sum(
            u.deduction_weight for u in adjudicated_e2
            if u.operational_label != cls and _infer_label(pred_by_id[u.unit_id]) == cls
        )
        fn_w = sum(
            u.deduction_weight for u in adjudicated_e2
            if u.operational_label == cls and _infer_label(pred_by_id[u.unit_id]) != cls
        )
        denom = 2.0 * tp_w + fp_w + fn_w
        f1_per_class[cls] = (2.0 * tp_w / denom) if denom > 0 else 0.0

    present_classes = [c for c in classes if any(u.operational_label == c for u in adjudicated_e2)]
    weighted_macro_f1 = (
        sum(f1_per_class[c] for c in present_classes) / len(present_classes)
        if present_classes else 0.0
    )
    weighted_acc = (correct_w / total_w) if total_w > 0 else 0.0

    non_abstained = [u for u in adjudicated_e2 if not pred_by_id[u.unit_id].get("abstain", False)]
    coverage = (len(non_abstained) / len(units)) if units else 0.0
    adjudicated_selective_coverage = len(non_abstained) / len(adjudicated_e2) if adjudicated_e2 else 0.0
    all_non_abstained = [u for u in all_matched_units if not pred_by_id[u.unit_id].get("abstain", False)]
    full_sample_coverage = (len(all_non_abstained) / len(units)) if units else 0.0
    sel_w = sum(u.deduction_weight for u in non_abstained)
    sel_correct_w = sum(
        u.deduction_weight for u in non_abstained
        if _infer_label(pred_by_id[u.unit_id]) == u.operational_label
    )
    selective_weighted_acc = (sel_correct_w / sel_w) if sel_w > 0 else 0.0

    brier_terms: list[float] = []
    ece_pairs: list[tuple[float, float]] = []
    for u in adjudicated_e2:
        p = pred_by_id[u.unit_id]
        vec = u.e2_response_vector
        for prob_key, arm_name in (
            ("p_prob", "P"),
            ("d_prob", "D"),
            ("s_prob", "S"),
            ("r_explicit_prob", "R_explicit"),
            ("r_implicit_prob", "R_implicit"),
            ("r_prob", "R"),
        ):
            actual_mean = vec.arm_means.get(arm_name)
            if actual_mean is not None and prob_key in p:
                pred_p = max(0.0, min(1.0, float(p.get(prob_key, 0.0))))
                act_v = max(0.0, min(1.0, float(actual_mean)))
                brier_terms.append((pred_p - act_v) ** 2)
                ece_pairs.append((pred_p, act_v))

    brier_score = (sum(brier_terms) / len(brier_terms)) if brier_terms else None
    ece_score = None
    if ece_pairs:
        n_bins = 5
        total_pts = len(ece_pairs)
        ece_val = 0.0
        for b_idx in range(n_bins):
            lo = b_idx / n_bins
            hi = (b_idx + 1) / n_bins
            bucket = [
                (pr, ac) for pr, ac in ece_pairs
                if (lo <= pr < hi) or (b_idx == n_bins - 1 and pr == 1.0)
            ]
            if bucket:
                avg_p = sum(x[0] for x in bucket) / len(bucket)
                avg_a = sum(x[1] for x in bucket) / len(bucket)
                ece_val += (len(bucket) / total_pts) * abs(avg_p - avg_a)
        ece_score = round(ece_val, 6)

    broken_units = [
        u for u in units
        if u.operational_label == "broken"
        and u.e1_observation.source_turn is not None
        and u.unit_id in pred_by_id
    ]
    exact_hits = 0
    pm1_hits = 0
    for u in broken_units:
        steps = pred_by_id[u.unit_id].get("evidence_steps") or []
        true_t = int(u.e1_observation.source_turn)  # type: ignore[arg-type]
        if steps:
            s0 = int(steps[0])
            if s0 == true_t:
                exact_hits += 1
            if abs(s0 - true_t) <= 1:
                pm1_hits += 1

    unrec_units = [u for u in adjudicated_e2 if u.operational_label == "unrecovered_under_protocol"]
    false_loc_count = sum(
        1 for u in unrec_units
        if bool(pred_by_id[u.unit_id].get("evidence_steps")) and not pred_by_id[u.unit_id].get("abstain", False)
    )

    return {
        "schema_version": V05_SCHEMA_VERSION,
        "proposition_ids": list(V3_PROPOSITIONS),
        "total_input_unit_count": len(units),
        "matched_prediction_unit_count": len(all_matched_units),
        "missing_prediction_unit_count": len(units) - len(all_matched_units),
        "abstained_unit_count": sum(bool(p.get("abstain")) for p in pred_by_id.values()),
        "adjudicated_e2_unit_count": len(adjudicated_e2),
        "inconclusive_unit_count": len(inconclusive_units),
        "plain_recoverable_unit_count": len(plain_recoverable_units),
        "not_applicable_unit_count": len(not_applicable_units),
        "tier1_utility": {
            "utility_per_100_requests": round(utility_per_100_req, 6),
            "unmeasured_policy_unit_count": policy_unmeasured,
            "inadmissible_policy_unit_count": len(inadmissible_policy_units),
            "inadmissible_policy_unit_ids": inadmissible_policy_units,
            "measured_policy_unit_count": len(ranked_items),
            "primary_policy_evaluation_complete": not policy_unmeasured and not inadmissible_policy_units,
            "aggregate_scope": "observed_admissible_policy_units_only",
            "total_actual_deduction_recovered": round(total_actual_gain, 6),
            "total_verification_requests": round(total_active_requests, 2),
            "cross_fitted_ceiling": {
                "unweighted_ceiling": crossfit["unweighted_ceiling"],
                "deduction_weighted_ceiling": crossfit["deduction_weighted_ceiling"],
                "unweighted_lb95": crossfit.get("unweighted_lb95", 0.0),
                "deduction_weighted_lb95": crossfit.get("deduction_weighted_lb95", 0.0),
                "passes_v3_p3_gate": crossfit["passes_v3_p3_gate"],
            },
        },
        "tier2_classification": {
            "weighted_macro_f1": round(weighted_macro_f1, 6),
            "weighted_accuracy": round(weighted_acc, 6),
            "f1_by_class": {k: round(v, 6) for k, v in f1_per_class.items()},
            "coverage": round(coverage, 6),
            "adjudicated_selective_coverage": adjudicated_selective_coverage,
            "full_sample_coverage": round(full_sample_coverage, 6),
            "selective_weighted_accuracy": round(selective_weighted_acc, 6),
        },
        "tier2_attribution": {
            "adjudicated_unit_count": len(adjudicated_e2),
            "weighted_accuracy": (sum(u.deduction_weight for u in adjudicated_e2
                if pred_by_id[u.unit_id].get("predicted_label") == u.operational_label) / total_w) if total_w else None,
            "scope": "predicted_attribution_labels_including_hidden_rules_independent_of_policy_abstention",
        },
        "tier3_calibration": {
            "brier_score": round(brier_score, 6) if brier_score is not None else None,
            "ece": ece_score,
        },
        "tier4_localization_and_hallucination": {
            "broken_unit_count": len(broken_units),
            "broken_source_turn_exact_acc": round(exact_hits / len(broken_units), 6) if broken_units else None,
            "broken_source_turn_pm1_acc": round(pm1_hits / len(broken_units), 6) if broken_units else None,
            "unrecovered_unit_count": len(unrec_units),
            "false_localization_rate_on_unrecovered": round(false_loc_count / len(unrec_units), 6) if unrec_units else None,
        },
    }


def build_sokoban_v05_unit(
    episode_id: str,
    *,
    terminal_passed: bool,
    max_band: int,
    terminal_band: int,
    k_star: int,
    first_deadlock_turn: int | None = None,
) -> DiagnosticUnitV05:
    """V3 §3.6.2 & R4 Sokoban Benchmark Adaptation Boundary:
    - Marks `undelivered` and arm `D` as `not_applicable`.
    - Reports static box-corner deadlock separately as `independent_state_damage` ("corner_deadlock"),
      NEVER merging static corner deadlock into `broken` unless `max_band >= 1` regressed to `terminal_band == 0`.
    """
    y_val = 1.0 if terminal_passed else 0.0
    max_s = 1.0 if max_band >= 1 else 0.0
    ded = 1.0 - y_val

    if max_s > y_val:
        e1_cls = "broken"
        substate = "peak_regressed"
        op_label = "broken"
        ev_status = "adjudicated"
    else:
        e1_cls = "never_succeeded_on_disk"
        substate = "no_candidate" if k_star == 0 else "delivered_item_wrong"
        op_label = None
        ev_status = "not_applicable"

    return DiagnosticUnitV05(
        unit_id=f"{episode_id}:sokoban_pass",
        episode_id=episode_id,
        task_id="lhtb:sokoban",
        benchmark="lhtb",
        unit_level="rubric_item",
        defect_name="sokoban_official_pass_and_deadlock_free",
        deduction_weight=ded,
        rule_source="explicit",
        e1_observation=E1Observation(
            e1_class=e1_cls,
            candidate_substate=substate,
            k_star=k_star,
            max_item_score=max_s,
            terminal_item_score=y_val,
            deduction=ded,
            source_turn=first_deadlock_turn if max_s > y_val else None,
            deadlock_turn=first_deadlock_turn,
            independent_state_damage="corner_deadlock_detected" if first_deadlock_turn is not None else None,
        ),
        e2_response_vector=E2ResponseVector(evaluated=False),
        operational_label=op_label,
        evidence_status=ev_status,
    )


def verify_rule_prompt_non_leakage(
    rule_prompt: str,
    *,
    forbidden_tokens: Sequence[str] = (),
) -> bool:
    """Verify that a rule intervention prompt (`R*`) or normal request does not leak internal
    verifier variable names, function identifiers, or hidden test literals (V3 §3.3 & V3-G4).
    """
    lower_p = rule_prompt.lower()
    default_forbidden = (
        "def verify",
        "officeval_",
        "_verifier.py",
        "assert ",
        "pytest",
        "fail_to_pass",
        "pass_to_pass",
    )
    for tok in (*default_forbidden, *forbidden_tokens):
        if tok and tok.lower() in lower_p:
            return False
    return True


def verify_normal_request_zero_diagnostic_leakage(
    request_payload: dict[str, Any],
    *,
    forbidden_phrases: Sequence[str] = (),
    rubric_items: Sequence[str] = (),
    task_instruction: str | None = None,
) -> dict[str, Any]:
    """Scan diagnostic markers; a clean lexical scan cannot certify non-leakage.

    Full verification requires the independent input-boundary provenance gate.
    """
    if not isinstance(request_payload, dict):
        return {"zero_leakage": False, "zero_leakage_verified": False,
                "status": "invalid_request", "leaked_markers": ["<invalid_non_dict_request_payload>"]}
    try:
        raw_text = json.dumps(request_payload, ensure_ascii=False)
    except (TypeError, ValueError):
        return {"zero_leakage": False, "zero_leakage_verified": False,
                "status": "invalid_request", "leaked_markers": ["<invalid_json_payload>"]}
    default_leak_markers = (
        DELIVER_INTERVENTION_MESSAGE[:20],
        DEFAULT_SELF_CHECK_DRAFT[:20],
        "请同时逐条核对以下任务明确要求",
        "请进一步逐条核对以下隐含格式与细则要求",
        "请重点核对并满足以下单项要求",
        "请同时逐条核对以下全部要求",
        "_verifier.py",
        "def verify",
        "dim1_pass",
        "dim2_items",
        "official_grade.json",
        "defects.json",
        "grades.jsonl",
        "隐藏评分标准",
        "shape_count",
        "只允许有0.15cm的误差",
        "=IFERROR(G",
        "18cm–21cm",
        "10.5至11.5磅",
    )
    leaked: list[str] = []
    for m in (*default_leak_markers, *forbidden_phrases):
        if m and m in raw_text:
            leaked.append(m)

    instr_str = (task_instruction or "").strip()
    for r_item in rubric_items:
        r_clean = str(r_item).strip()
        if len(r_clean) >= 8 and r_clean not in instr_str and r_clean in raw_text:
            leaked.append(f"rubric_leak:{r_clean[:40]}")

    return {
        "zero_leakage": False if leaked else None,
        "zero_leakage_verified": False,
        "status": "suspected_leakage" if leaked else "clean_lexical_scan_only",
        "leaked_markers": leaked,
    }


def compute_ipw_recoverability_ratio(
    sampled_rows: Sequence[dict[str, Any]],
    *,
    total_never_deduction: float | None = None,
    alpha: float = 0.05,
    seed: str = "l2-v05-ipw",
) -> dict[str, Any]:
    """R11 (V3 §5.2 `V3-P1` & `V3-P2`): Horvitz-Thompson (HT) and Hajek inverse-probability
    weighted (IPW) estimators across the 3D stratified sampling frame with family-clustered bootstrap CIs.
    Includes Level 0 (`P` plain_recoverable), Level 1 (`D`, `S`, `D ∨ S`, and `level1_net_over_P`),
    and combined Level 0–1 (`P ∨ D ∨ S`).
    """
    valid = [r for r in sampled_rows if float(r.get("inclusion_prob", 0.0)) > 0.0]
    if not valid:
        return {
            "input_count": len(sampled_rows),
            "excluded_invalid_inclusion_count": len(sampled_rows),
            "sampled_count": 0,
            "interval_status": "no_valid_sample",
            "interval_estimand": "hajek_ratio",
            "family_cluster_ci95_level0_1_combined": [0.0, 1.0],
            "level0_1_combined_share_upper_bound_incl_inconclusive": 1.0,
            "ht_denom_never_deduction": 0.0,
            "plain_recoverable_share_lower_bound": 0.0,
            "delivery_responsive_share_lower_bound": 0.0,
            "self_check_responsive_share_lower_bound": 0.0,
            "level1_generic_share_lower_bound": 0.0,
            "level0_1_combined_share_lower_bound": 0.0,
            "delivery_responsive_share_upper_bound_incl_inconclusive": 1.0,
            "rule_responsive_share_lower_bound": 0.0,
            "explicit_rule_responsive_share_lower_bound": 0.0,
            "implicit_rule_responsive_share_lower_bound": 0.0,
            "inconclusive_share": 0.0,
            "hajek_shares": {},
            "ht_shares": {},
            "family_cluster_ci95_delivery": [0.0, 1.0],
            "family_cluster_ci95_level1_generic": [0.0, 1.0],
        }

    def _calc(rows: Sequence[dict[str, Any]], denom_override: float | None = None) -> dict[str, float]:
        ht_denom = sum(float(r["deduction"]) / float(r["inclusion_prob"]) for r in rows)
        denom = float(denom_override) if (denom_override is not None and denom_override > 0) else ht_denom
        if denom <= 0:
            return {
                "ht_denom": 0.0,
                "p_ht": 0.0,
                "d_ht": 0.0,
                "s_ht": 0.0,
                "lvl1_ht": 0.0,
                "lvl01_ht": 0.0,
                "d_ub_ht": 0.0,
                "r_ht": 0.0,
                "r_exp_ht": 0.0,
                "r_imp_ht": 0.0,
                "inc_ht": 0.0,
                "p_hajek": 0.0,
                "d_hajek": 0.0,
                "s_hajek": 0.0,
                "lvl1_hajek": 0.0,
                "lvl01_hajek": 0.0,
                "r_hajek": 0.0,
            }

        def _w_sum(pred) -> float:
            return sum(float(r["deduction"]) / float(r["inclusion_prob"]) for r in rows if pred(r))

        p_num = _w_sum(lambda r: bool(r.get("recovered_by_P") or r.get("plain_recoverable")))
        d_num = _w_sum(lambda r: bool(r.get("recovered_by_D")))
        s_num = _w_sum(lambda r: bool(r.get("recovered_by_S")))
        lvl1_num = _w_sum(lambda r: bool(r.get("recovered_by_D") or r.get("recovered_by_S")))
        lvl01_num = _w_sum(
            lambda r: bool(
                r.get("recovered_by_P")
                or r.get("plain_recoverable")
                or r.get("recovered_by_D")
                or r.get("recovered_by_S")
            )
        )
        r_exp_num = _w_sum(lambda r: bool(r.get("recovered_by_R_explicit")))
        r_imp_num = _w_sum(lambda r: bool(r.get("recovered_by_R_implicit")))
        r_any_num = _w_sum(
            lambda r: bool(
                r.get("recovered_by_R")
                or r.get("recovered_by_R_explicit")
                or r.get("recovered_by_R_implicit")
            )
        )
        inc_num = _w_sum(lambda r: bool(r.get("inconclusive")))

        return {
            "ht_denom": ht_denom,
            "p_ht": p_num / denom,
            "d_ht": d_num / denom,
            "s_ht": s_num / denom,
            "lvl1_ht": lvl1_num / denom,
            "lvl01_ht": lvl01_num / denom,
            "d_ub_ht": (d_num + inc_num) / denom,
            "r_ht": r_any_num / denom,
            "r_exp_ht": r_exp_num / denom,
            "r_imp_ht": r_imp_num / denom,
            "inc_ht": inc_num / denom,
            "p_hajek": p_num / ht_denom if ht_denom > 0 else 0.0,
            "d_hajek": d_num / ht_denom if ht_denom > 0 else 0.0,
            "s_hajek": s_num / ht_denom if ht_denom > 0 else 0.0,
            "lvl1_hajek": lvl1_num / ht_denom if ht_denom > 0 else 0.0,
            "lvl01_hajek": lvl01_num / ht_denom if ht_denom > 0 else 0.0,
            "r_hajek": r_any_num / ht_denom if ht_denom > 0 else 0.0,
        }

    metrics = _calc(valid, total_never_deduction)
    ht_denom = metrics["ht_denom"]
    d_lb = metrics["d_ht"]
    lvl1_lb = metrics["lvl1_ht"]

    by_fam: dict[str, list[dict[str, Any]]] = {}
    for r in valid:
        fid = str(r.get("family_id") or "__unknown__")
        by_fam.setdefault(fid, []).append(r)
    fams = sorted(by_fam.keys())

    if len(fams) <= 1:
        ci_d = [0.0, 1.0]
        ci_lvl1 = [0.0, 1.0]
        ci_lvl01 = [0.0, 1.0]
    else:
        rng = random.Random(int(digest({"seed": seed, "n": len(valid), "fams": fams})[:16], 16))
        boot_vals_d: list[float] = []
        boot_vals_lvl1: list[float] = []
        boot_vals_lvl01: list[float] = []
        n_boot = 2000
        for _ in range(n_boot):
            sampled_fams = [rng.choice(fams) for _ in range(len(fams))]
            boot_rows = [item for f in sampled_fams for item in by_fam[f]]
            b_m = _calc(boot_rows, None)
            boot_vals_d.append(b_m["d_hajek"])
            boot_vals_lvl1.append(b_m["lvl1_hajek"])
            boot_vals_lvl01.append(b_m["lvl01_hajek"])
        boot_vals_d.sort()
        boot_vals_lvl1.sort()
        boot_vals_lvl01.sort()
        lo_idx = max(0, int(math.floor((alpha / 2.0) * (n_boot - 1))))
        hi_idx = min(n_boot - 1, int(math.ceil((1.0 - alpha / 2.0) * (n_boot - 1))))
        ci_d = [round(boot_vals_d[lo_idx], 6), round(boot_vals_d[hi_idx], 6)]
        ci_lvl1 = [round(boot_vals_lvl1[lo_idx], 6), round(boot_vals_lvl1[hi_idx], 6)]
        ci_lvl01 = [round(boot_vals_lvl01[lo_idx], 6), round(boot_vals_lvl01[hi_idx], 6)]

    return {
        "input_count": len(sampled_rows),
        "excluded_invalid_inclusion_count": len(sampled_rows) - len(valid),
        "sampled_count": len(valid),
        "family_cluster_count": len(fams),
        "family_ids_complete": all(r.get("family_id") for r in valid),
        "ht_denom_never_deduction": round(ht_denom, 6),
        "population_never_deduction_used": round(float(total_never_deduction or ht_denom), 6),
        "plain_recoverable_share_lower_bound": round(metrics["p_ht"], 6),
        "delivery_responsive_share_lower_bound": round(metrics["d_ht"], 6),
        "self_check_responsive_share_lower_bound": round(metrics["s_ht"], 6),
        "level1_generic_share_lower_bound": round(metrics["lvl1_ht"], 6),
        "level0_1_combined_share_lower_bound": round(metrics["lvl01_ht"], 6),
        "delivery_responsive_share_upper_bound_incl_inconclusive": round(metrics["d_ub_ht"], 6),
        "rule_responsive_share_lower_bound": round(metrics["r_ht"], 6),
        "explicit_rule_responsive_share_lower_bound": round(metrics["r_exp_ht"], 6),
        "implicit_rule_responsive_share_lower_bound": round(metrics["r_imp_ht"], 6),
        "inconclusive_share": round(metrics["inc_ht"], 6),
        "ht_shares": {
            "P": round(metrics["p_ht"], 6),
            "D": round(metrics["d_ht"], 6),
            "S": round(metrics["s_ht"], 6),
            "level1_D_or_S": round(metrics["lvl1_ht"], 6),
            "level0_or_1_P_D_S": round(metrics["lvl01_ht"], 6),
            "R_explicit": round(metrics["r_exp_ht"], 6),
            "R_implicit": round(metrics["r_imp_ht"], 6),
            "R_any": round(metrics["r_ht"], 6),
        },
        "hajek_shares": {
            "P": round(metrics["p_hajek"], 6),
            "D": round(metrics["d_hajek"], 6),
            "S": round(metrics["s_hajek"], 6),
            "level1_D_or_S": round(metrics["lvl1_hajek"], 6),
            "level0_or_1_P_D_S": round(metrics["lvl01_hajek"], 6),
            "R_any": round(metrics["r_hajek"], 6),
        },
        "family_cluster_ci95_delivery": ci_d,
        "family_cluster_ci95_level1_generic": ci_lvl1,
        "family_cluster_ci95_level0_1_combined": ci_lvl01,
        "interval_estimand": "hajek_ratio_not_ht_population_ratio",
        "interval_status": "descriptive_family_bootstrap" if len(fams) > 1 else "insufficient_families",
        "level0_1_combined_share_upper_bound_incl_inconclusive": min(1.0, metrics["lvl01_ht"] + metrics["inc_ht"]),
    }
