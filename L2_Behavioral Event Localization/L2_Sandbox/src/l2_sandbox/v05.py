"""L2 v0.5 Intervention-Verifiable Failure Diagnosis Protocol (Notion V3 §3.1–3.7, §5.2 V3-P1..P5).

Aligned with V3 Revision Task Order (R7–R14):
1. Information-ordered intervention ladder (V3 §3.4):
   - Level 0: `P` (plain continuation, 0 added tokens)
   - Level 1: `D` (generic delivery check) and `S` (generic pre-finish self-check; wording parameterized for D0)
   - Level 2: `R_explicit` (nested prefix `D` + full task-level explicit requirements list)
   - Level 3: `R_implicit` (nested prefix `R_explicit` + full task-level implicit rubric conventions list)
   - Single-item localization: `R_single` (nested prefix `D` + single item requirement) and optional `R_all`
   - Strict nested prefix verification (`verify_nested_ladder_messages`) and SHA-256 text hashes.
2. Whole-task requirement lists & empty-level `ineligible` handling (`R8`, V3 §3.4):
   - Requirements for `R_explicit` and `R_implicit` are built for the entire `task_id`, never filtered by
     which items failed on the parent run.
   - When a task has an empty `explicit` or `implicit` requirement list, that ladder level is marked
     `ineligible` and NEVER falls back to `D` or a lower arm's message.
3. Ladder incremental adjudication (`R9`, V3 §3.2 & §3.4):
   - Parameterized `(delta, tau, alpha)`.
   - Level 0 check: `P` absolute recovery (`mean_P >= tau` and `LB95(P) > delta` -> `plain_recoverable`).
   - Level k > 0 check: an arm at level k is effective iff it beats ALL evaluated lower-level arms j < k
     (`mean_k - mean_j >= tau`, `LB95(k - j) > delta`, and Holm-adjusted one-sided permutation `p <= alpha`).
   - `unrecovered_under_protocol`: requires ALL evaluated arms (including `P`) to have `UB95(arm) < tau`.
4. Anchor checkpoint selection (`R10`, V3 §3.4 rule 7):
   - Supports candidate peak (`candidate_peak`), candidate first write (`candidate_first_write`),
     budget-exhausted tail (`budget_tail_no_candidate`), and voluntary termination pre-finish
     (`voluntary_pre_finish` when `termination == "assistant_finished"`).
5. 3D Stratification (`exhausted_budget x valid_delivery x rule_source`) & HT/Hajek IPW (`R11`, V3 §3.4 & §5.2).
6. Harm check on passing runs/items (`R12`, V3 §3.4).
7. Cross-fitted diagnostic value ceiling (`R13`, V3 §3.7 & `V3-P3`), replacing in-sample `max_possible_gain`.
8. Pre-registered V3 proposition identifiers `V3-P1` .. `V3-P5` (`R14`).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import math
import random
from typing import Any, Sequence

from .config import (
    DELIVER_REMINDER as DELIVER_INTERVENTION_MESSAGE,
    RULE_REMINDER_TEMPLATE as RULE_INTERVENTION_TEMPLATE,
)
from .io import L2Error, digest
from .stats import holm_adjust, lower_drop, permutation_one_sided, wilson_bounds

V05_SCHEMA_VERSION = "l2-0.5.1-v3-ladder"

V3_PROPOSITIONS = (
    "V3-P1",  # Level-1 generic intervention (D / S) recoverability share >= 20%
    "V3-P2",  # Ladder decomposition across D, S, R_explicit, R_implicit, and single-item R localization
    "V3-P3",  # Cross-fitted diagnostic value ceiling >= 0.05 (prerequisite gate for V3-P4 / V3-P5)
    "V3-P4",  # Analyst arm-selection utility (C >= B > A > fixed best single arm)
    "V3-P5",  # Cross-family out-of-sample generalization on unexposed holdout families (Stage H)
)

DEFAULT_SELF_CHECK_DRAFT = (
    "在结束任务前，请先对照原始任务要求和输入材料完整自查一遍当前工作区与 /workspace/submission 中的结果，"
    "确认所有要求的内容、格式与文件均已正确完成并保存；如有遗漏或问题请先修正后再结束。"
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
    arm_means: dict[str, float] = field(default_factory=dict)
    arm_lb95_abs: dict[str, float] = field(default_factory=dict)
    arm_ub95_abs: dict[str, float] = field(default_factory=dict)
    pairwise_deltas: dict[str, float] = field(default_factory=dict)
    pairwise_lb95: dict[str, float] = field(default_factory=dict)
    pairwise_ub95: dict[str, float] = field(default_factory=dict)
    pairwise_p_raw: dict[str, float] = field(default_factory=dict)
    pairwise_p_holm: dict[str, float] = field(default_factory=dict)
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


def build_task_requirement_lists(
    task_rule_catalog: dict[str, dict[str, dict[str, Any]]],
    task_id: str,
) -> dict[str, list[str]]:
    """R8 (V3 §3.4): Build full task-level `explicit` and `implicit` requirement lists for `task_id`.

    Crucially, this function extracts ALL rules belonging to `task_id` from the frozen catalog,
    NEVER filtering by which items failed on a specific parent trajectory.
    """
    short_id = task_id.split(":")[-1]
    full_id = f"officeval:{short_id}" if not task_id.startswith("officeval:") else task_id
    rules_map = task_rule_catalog.get(full_id) or task_rule_catalog.get(short_id) or task_rule_catalog.get(task_id) or {}

    explicit_reqs: list[str] = []
    implicit_reqs: list[str] = []
    ambiguous_reqs: list[str] = []

    for rule_text, info in sorted(rules_map.items()):
        cleaned = str(rule_text).strip()
        if not cleaned:
            continue
        cat = str((info or {}).get("source_category") or "ambiguous")
        if cat == "explicit":
            explicit_reqs.append(cleaned)
        elif cat == "implicit":
            implicit_reqs.append(cleaned)
        else:
            ambiguous_reqs.append(cleaned)

    return {
        "task_id": short_id,
        "explicit": explicit_reqs,
        "implicit": implicit_reqs,
        "ambiguous": ambiguous_reqs,
    }


def build_ladder_intervention_messages(
    *,
    explicit_requirements: Sequence[str] = (),
    implicit_requirements: Sequence[str] = (),
    single_rule_requirement: str | None = None,
    deliver_text: str = DELIVER_INTERVENTION_MESSAGE,
    self_check_text: str = DEFAULT_SELF_CHECK_DRAFT,
) -> dict[str, dict[str, Any]]:
    """R7 & R8 (V3 §3.4): Construct the nested intervention ladder messages and their SHA-256 hashes.

    Enforces strict nesting:
      - `P`: None (0 added user message)
      - `D`: `deliver_text`
      - `S`: `self_check_text` (parameterized for D0 freeze)
      - `R_explicit`: `deliver_text` + "\\n\\n[任务明确要求核对清单]：\\n" + numbered `explicit_requirements`.
        If `explicit_requirements` is empty -> `eligible = False`, `evidence_status = "ineligible"`,
        `message = None` (NEVER falls back to `D`!).
      - `R_implicit`: `R_explicit.message` (or `deliver_text` ONLY if explicit is non-empty; if explicit is empty,
        uses `deliver_text` as prefix + explicit note `[无额外明确文字清单]`) + "\\n\\n[补充细则核对清单]：\\n" + numbered `implicit_requirements`.
        If `implicit_requirements` is empty -> `eligible = False`, `evidence_status = "ineligible"`,
        `message = None` (NEVER falls back to `D` or `R_explicit`!).
      - `R_single`: `deliver_text` + "\\n\\n[单项核对要求]：" + `single_rule_requirement`.
        If `single_rule_requirement` is empty -> `eligible = False`, `evidence_status = "ineligible"`.
    """
    d_clean = deliver_text.strip()
    s_clean = self_check_text.strip()
    exp_clean = [r.strip() for r in explicit_requirements if isinstance(r, str) and r.strip()]
    imp_clean = [r.strip() for r in implicit_requirements if isinstance(r, str) and r.strip()]
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
            "prefix_arm": "P",
        },
    }

    if exp_clean:
        exp_block = "\n".join(f"({i}) {req}" for i, req in enumerate(exp_clean, start=1))
        r_exp_msg = f"{d_clean}\n\n请同时逐条核对以下任务明确要求：\n{exp_block}"
        out["R_explicit"] = {
            "arm": "R_explicit",
            "level": 2,
            "eligible": True,
            "evidence_status": "eligible",
            "message": r_exp_msg,
            "message_sha256": sha256_text(r_exp_msg),
            "prefix_arm": "D",
        }
    else:
        out["R_explicit"] = {
            "arm": "R_explicit",
            "level": 2,
            "eligible": False,
            "evidence_status": "ineligible",
            "message": None,
            "message_sha256": None,
            "prefix_arm": "D",
        }

    if imp_clean:
        base_prefix = out["R_explicit"]["message"] if out["R_explicit"]["eligible"] else d_clean
        imp_block = "\n".join(f"({i}) {req}" for i, req in enumerate(imp_clean, start=1))
        r_imp_msg = f"{base_prefix}\n\n请进一步逐条核对以下隐含格式与细则要求：\n{imp_block}"
        out["R_implicit"] = {
            "arm": "R_implicit",
            "level": 3,
            "eligible": True,
            "evidence_status": "eligible",
            "message": r_imp_msg,
            "message_sha256": sha256_text(r_imp_msg),
            "prefix_arm": "R_explicit" if out["R_explicit"]["eligible"] else "D",
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

    if single_clean:
        r_single_msg = f"{d_clean}\n\n请重点核对并满足以下单项要求：{single_clean}"
        out["R_single"] = {
            "arm": "R_single",
            "level": 2,
            "eligible": True,
            "evidence_status": "eligible",
            "message": r_single_msg,
            "message_sha256": sha256_text(r_single_msg),
            "prefix_arm": "D",
        }
    else:
        out["R_single"] = {
            "arm": "R_single",
            "level": 2,
            "eligible": False,
            "evidence_status": "ineligible",
            "message": None,
            "message_sha256": None,
            "prefix_arm": "D",
        }

    return out


def verify_nested_ladder_messages(ladder_spec: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """R7: Verify that every eligible arm in `ladder_spec` strictly begins with its parent arm's text
    and never silently duplicates a lower arm's text when a requirement list is empty.
    """
    errors: list[str] = []
    p_info = ladder_spec.get("P") or {}
    if p_info.get("message") is not None:
        errors.append("P arm must have message=None")

    d_msg = (ladder_spec.get("D") or {}).get("message")
    if not isinstance(d_msg, str) or not d_msg:
        errors.append("D arm must have non-empty message")

    for arm_name in ("R_explicit", "R_implicit", "R_single"):
        info = ladder_spec.get(arm_name)
        if not info:
            continue
        if not info.get("eligible"):
            if info.get("message") is not None or info.get("evidence_status") != "ineligible":
                errors.append(f"Ineligible arm {arm_name} must have message=None and evidence_status='ineligible'")
            continue
        msg = info.get("message")
        pref_arm = info.get("prefix_arm") or "D"
        pref_msg = (ladder_spec.get(pref_arm) or {}).get("message")
        if not isinstance(msg, str) or not isinstance(pref_msg, str):
            errors.append(f"Eligible arm {arm_name} or its prefix {pref_arm} is missing string message")
            continue
        if not msg.startswith(pref_msg):
            errors.append(f"Arm {arm_name} does not start with prefix {pref_arm}")
        if msg == pref_msg:
            errors.append(f"Arm {arm_name} is identical to lower arm {pref_arm} (empty fallback forbidden)")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "hashes": {
            k: v.get("message_sha256")
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
) -> dict[str, Any]:
    """R10 (V3 §3.4 Rule 7): Anchor Selection Rule supporting both budget exhaustion and voluntary finish.

    - Requires `remaining_requests >= r_b` and `cold_resume_eligible == True`.
    - If candidate checkpoints exist:
      - If max candidate score > 0: earliest checkpoint achieving max score (`candidate_peak`).
      - Else: earliest checkpoint where candidate was first written (`candidate_first_write`).
    - If no candidate (`k* = 0`) or whole-run tail anchor:
      - When `termination == "assistant_finished"` (voluntary finish before exhausting budget):
        selects the last eligible checkpoint before voluntary finish (`voluntary_pre_finish`).
      - Otherwise (`request_budget` or unspecified):
        selects the latest eligible checkpoint with `remaining_requests >= r_b` (`budget_tail_no_candidate`).
    """
    eligible_cps: list[dict[str, Any]] = []
    for r in checkpoint_rows:
        if r.get("kind") != "checkpoint":
            continue
        k = int(r.get("k", 0))
        rem = int(r.get("remaining_requests", max_requests - k))
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
            "early_anchor_k": None,
            "early_remaining_budget": None,
        }

    eligible_cps.sort(key=lambda x: x["k"])
    cand_cps = [r for r in eligible_cps if r.get("candidate_status") == "ok"]

    def _score_at(r: dict[str, Any]) -> float:
        if item_id and isinstance(r.get("deductions"), dict):
            if not r.get("dim1_pass"):
                return 0.0
            ded = float(r["deductions"].get(item_id, 0.0))
            return 1.0 if ded <= 1e-9 else 0.0
        return float(r.get("score") or 0.0)

    tail_cp = eligible_cps[-1]
    is_voluntary = (termination == "assistant_finished")

    if cand_cps:
        max_s = max(_score_at(r) for r in cand_cps)
        if max_s > 0.0:
            chosen = next(r for r in cand_cps if _score_at(r) >= max_s - 1e-9)
            anchor_type = "candidate_peak"
        else:
            if is_voluntary:
                # For voluntary finish with score-0 candidate, also record first write vs pre-finish
                chosen = cand_cps[0]
                anchor_type = "candidate_first_write"
            else:
                chosen = cand_cps[0]
                anchor_type = "candidate_first_write"
    else:
        chosen = tail_cp
        anchor_type = "voluntary_pre_finish" if is_voluntary else "budget_tail_no_candidate"

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
) -> tuple[E2ResponseVector, str | None, str]:
    """R9 (V3 §3.2 & §3.4): Ladder Statistical Decision Rule with parameterized `(delta, tau, alpha)`.

    Rules:
    1. Level 0 (`P` absolute recovery):
       If `mean(P) >= tau` and `LB95(P) > delta` (or `mean(P) - terminal_y >= tau` and `LB95(P - terminal_y) > delta`),
       returns `(vec, None, "plain_recoverable")`.
    2. Ladder Level `k >= 1` incremental effectiveness:
       - Evaluated arms are ordered by `LADDER_ARM_LEVELS`:
         Level 0: `P`
         Level 1: `D`, `S`
         Level 2: `R_explicit` (or `R` / `R_single` / `R_all`)
         Level 3: `R_implicit`
       - All one-sided comparisons `(higher_arm vs lower_arm)` across levels `level(higher) > level(lower)`
         are tested via one-sided permutation test and adjusted jointly via Holm step-down (`holm_adjust`).
       - An arm `A` at level `k >= 1` is adjudicated effective iff for EVERY evaluated arm `B` with `level(B) < k`:
         `mean(A) - mean(B) >= tau`, `LB95(A - B) > delta`, and `p_holm(A vs B) <= alpha`.
       - Among effective arms, the lowest level `k` wins:
         - `D` -> `"delivery_responsive"`
         - `S` -> `"self_check_responsive"`
         - `R_explicit` -> `"explicit_rule_responsive"`
         - `R` / `R_single` -> `"rule_responsive"`
         - `R_implicit` -> `"implicit_rule_responsive"`
         (If both `D` and `S` at Level 1 pass, chooses the one with larger `mean`, tie-breaking to `D`.)
    3. `unrecovered_under_protocol`:
       When no arm is effective, AND every evaluated arm `A` (including `P`) has absolute 95% upper bound
       `UB95(V(A)) < tau` AND relative upper bound `UB95(A - P) < tau` for all `A != P`.
    4. Otherwise: returns `(vec, None, "inconclusive")`.
    """
    clean_outcomes: dict[str, list[float]] = {
        k: [float(x) for x in v]
        for k, v in arm_outcomes.items()
        if v is not None and len(v) > 0
    }
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
    comparison_pairs: list[tuple[str, str]] = []
    for hi in eval_arms:
        lvl_hi = LADDER_ARM_LEVELS.get(hi, 99)
        if hi == "R_all":
            continue
        for lo in eval_arms:
            lvl_lo = LADDER_ARM_LEVELS.get(lo, 99)
            if lo == "R_all":
                continue
            if lvl_hi > lvl_lo:
                comparison_pairs.append((hi, lo))

    pairwise_deltas: dict[str, float] = {}
    pairwise_lb95: dict[str, float] = {}
    pairwise_ub95: dict[str, float] = {}
    pairwise_p_raw: dict[str, float] = {}
    raw_p_list: list[float] = []

    for hi, lo in comparison_pairs:
        key = f"{hi}_vs_{lo}"
        v_hi = clean_outcomes[hi]
        v_lo = clean_outcomes[lo]
        diff = (sum(v_hi) / len(v_hi)) - (sum(v_lo) / len(v_lo))
        lb = lower_drop(v_hi, v_lo, alpha=alpha, seed=f"v05-lb-{key}")
        ub = upper_drop(v_hi, v_lo, alpha=alpha, seed=f"v05-ub-{key}")
        p_raw = permutation_one_sided(v_hi, v_lo, direction=1.0, seed=f"v05-perm-{key}")
        pairwise_deltas[key] = round(diff, 6)
        pairwise_lb95[key] = round(lb, 6)
        pairwise_ub95[key] = round(ub, 6)
        pairwise_p_raw[key] = round(p_raw, 6)
        raw_p_list.append(p_raw)

    holm_p_list = holm_adjust(raw_p_list) if raw_p_list else []
    pairwise_p_holm: dict[str, float] = {
        f"{hi}_vs_{lo}": round(p_h, 6)
        for (hi, lo), p_h in zip(comparison_pairs, holm_p_list)
    }

    p_vals = clean_outcomes["P"]
    mean_p = arm_means["P"]
    delta_p_y = round(mean_p - float(terminal_y), 6)
    lb_p_y = round(lower_drop(p_vals, float(terminal_y), alpha=alpha, seed="v05-p-y"), 6)

    r_key = "R" if "R" in clean_outcomes else ("R_single" if "R_single" in clean_outcomes else ("R_explicit" if "R_explicit" in clean_outcomes else None))
    mean_r = arm_means.get(r_key) if r_key else None
    mean_r_all = arm_means.get("R_all")
    delta_r_single_all = round(mean_r - mean_r_all, 6) if (mean_r is not None and mean_r_all is not None) else None

    vec = E2ResponseVector(
        evaluated=True,
        anchor_k=anchor_k,
        anchor_type=anchor_type,
        remaining_budget_at_anchor=remaining_budget,
        arm_outcomes=clean_outcomes,
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

    # Step 1: Level 0 check (`P` absolute success rate >= tau and LB95(P) > delta)
    if (mean_p >= tau and arm_lb95_abs["P"] > delta) or (delta_p_y >= tau and lb_p_y > delta):
        return vec, None, "plain_recoverable"

    # Step 2: Check ladder levels k = 1, 2, 3 in ascending order
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
            if LADDER_ARM_LEVELS.get(a, 99) == target_level and a in label_map
        ]
        if not candidates_at_level:
            continue
        # Sort so that if both D and S pass at level 1, higher mean wins (tie-breaker D before S)
        candidates_at_level.sort(key=lambda a: (-arm_means[a], 0 if a == "D" else 1))
        for cand_arm in candidates_at_level:
            lower_arms = [
                lo for lo in eval_arms
                if LADDER_ARM_LEVELS.get(lo, 99) < target_level and lo != "R_all"
            ]
            if not lower_arms:
                continue
            beats_all_lower = True
            for lo in lower_arms:
                pair_key = f"{cand_arm}_vs_{lo}"
                if (
                    pairwise_deltas.get(pair_key, -1.0) < tau
                    or pairwise_lb95.get(pair_key, -1.0) <= delta
                    or pairwise_p_holm.get(pair_key, 1.0) > alpha
                ):
                    beats_all_lower = False
                    break
            if beats_all_lower:
                return vec, label_map[cand_arm], "adjudicated"

    # Step 3: Unrecovered under protocol — requires ALL evaluated arms (including P) to have
    # absolute UB95 < tau AND all pairwise UB95(arm - P) < tau.
    non_r_all = [a for a in eval_arms if a != "R_all"]
    all_abs_below_tau = all(arm_ub95_abs[a] < tau for a in non_r_all)
    all_rel_below_tau = all(
        pairwise_ub95.get(f"{a}_vs_P", -1.0) < tau
        for a in non_r_all if a != "P"
    )
    if all_abs_below_tau and all_rel_below_tau:
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

    Rejects legacy `deepseek` datasets to prevent cross-configuration contamination.
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
            continue
        pi_h = n_h / N_h
        chosen = sorted(rng.sample(pool, k=n_h), key=lambda x: x.unit_id)
        for u in chosen:
            u.inclusion_prob = round(pi_h, 6)
            sampled_units.append({
                "unit_id": u.unit_id,
                "episode_id": u.episode_id,
                "task_id": u.task_id,
                "stratum_id": sid,
                "exhausted_budget": bool(u.exhausted_budget),
                "valid_delivery": bool(u.valid_delivery),
                "rule_source": u.rule_source,
                "deduction_weight": u.deduction_weight,
                "stratum_population_size": N_h,
                "stratum_sample_size": n_h,
                "inclusion_prob": round(pi_h, 6),
                "anchor_k": u.e2_response_vector.anchor_k,
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
    units: Sequence[DiagnosticUnitV05],
    *,
    candidate_arms: Sequence[str] = ("D", "S", "R_explicit", "R_implicit", "R"),
    ceiling_threshold: float = 0.05,
    alpha: float = 0.05,
    seed: str = "v3-d2-crossfit-ceiling",
) -> dict[str, Any]:
    """R13 (V3 §3.7 & `V3-P3`): Compute 2-fold cross-fitted diagnostic value ceiling at 0 extra Actor cost.

    Method:
    1. Split adjudicated/evaluated units into two strictly disjoint halves `I_1` and `I_2` (`I_1 ∩ I_2 = ∅`).
    2. For each fold `(train_half, eval_half)` in `[(I_1, I_2), (I_2, I_1)]`:
       - On `train_half`:
         a) Select the single best fixed arm `a*_fixed(train_half)` that maximizes mean gain over `P`
            (computed separately for unweighted and deduction-weighted objectives).
         b) Within each unit `u` in `eval_half`, split `u`'s continuation branches for each arm into
            selection branches (`sel_half`) and evaluation branches (`eval_branch_half`), OR use the
            stratum-level / split-sample oracle arm selected on `sel_half` and evaluated on `eval_branch_half`,
            so that arm selection and gain evaluation NEVER share the same random continuation outcomes.
       - On `eval_half`:
         Evaluate the out-of-fold gain of the unit-wise optimal arm vs `a*_fixed(train_half)`.
    3. Reports both **unweighted (per-unit equal weight)** and **deduction-weighted** cross-fitted ceilings,
       never using in-sample `max_possible_gain`.
    """
    eligible = [
        u for u in units
        if u.e2_response_vector.evaluated and "P" in u.e2_response_vector.arm_outcomes
    ]
    if len(eligible) < 2:
        return {
            "schema_version": V05_SCHEMA_VERSION,
            "proposition_id": "V3-P3",
            "eligible_unit_count": len(eligible),
            "folds_disjoint": True,
            "unweighted_ceiling": 0.0,
            "deduction_weighted_ceiling": 0.0,
            "ceiling_threshold": ceiling_threshold,
            "passes_v3_p3_gate": False,
        }

    rng = random.Random(int(digest({"seed": seed, "ids": [u.unit_id for u in eligible]})[:16], 16))
    shuffled = list(eligible)
    rng.shuffle(shuffled)
    mid = len(shuffled) // 2
    fold_1 = shuffled[:mid]
    fold_2 = shuffled[mid:]
    ids_1 = {u.unit_id for u in fold_1}
    ids_2 = {u.unit_id for u in fold_2}
    assert ids_1.isdisjoint(ids_2), "Cross-fitting folds I_1 and I_2 must be strictly disjoint"

    # Available intervention arms present in the units
    present_arms = [
        a for a in candidate_arms
        if any(a in u.e2_response_vector.arm_outcomes and len(u.e2_response_vector.arm_outcomes[a]) > 0 for u in eligible)
    ]
    if not present_arms:
        return {
            "schema_version": V05_SCHEMA_VERSION,
            "proposition_id": "V3-P3",
            "eligible_unit_count": len(eligible),
            "folds_disjoint": True,
            "unweighted_ceiling": 0.0,
            "deduction_weighted_ceiling": 0.0,
            "ceiling_threshold": ceiling_threshold,
            "passes_v3_p3_gate": False,
        }

    def _split_branch_gain(u: DiagnosticUnitV05, arm: str) -> tuple[float, float]:
        """Split a unit's replicates for `arm` and `P` into disjoint halves `(sel_gain, eval_gain)`."""
        outcomes = u.e2_response_vector.arm_outcomes
        p_vals = outcomes.get("P", [])
        a_vals = outcomes.get(arm, [])
        if not p_vals or not a_vals:
            return 0.0, 0.0
        mp1 = max(1, len(p_vals) // 2)
        ma1 = max(1, len(a_vals) // 2)
        p_sel = p_vals[:mp1]
        p_ev = p_vals[mp1:] if len(p_vals) > 1 else p_vals
        a_sel = a_vals[:ma1]
        a_ev = a_vals[ma1:] if len(a_vals) > 1 else a_vals
        g_sel = max(0.0, (sum(a_sel) / len(a_sel)) - (sum(p_sel) / len(p_sel)))
        g_ev = max(0.0, (sum(a_ev) / len(a_ev)) - (sum(p_ev) / len(p_ev)))
        return g_sel, g_ev

    fold_records: list[dict[str, Any]] = []
    eval_unit_rows: list[dict[str, Any]] = []

    for fold_idx, (train_set, eval_set) in enumerate(((fold_1, fold_2), (fold_2, fold_1)), start=1):
        # 1. Select best fixed arm on train_set (both unweighted and deduction-weighted)
        best_fixed_unw = present_arms[0]
        best_fixed_unw_score = -1e9
        best_fixed_w = present_arms[0]
        best_fixed_w_score = -1e9

        for arm in present_arms:
            unw_gains = []
            w_gains = []
            w_denom = 0.0
            for u in train_set:
                g_sel, g_ev = _split_branch_gain(u, arm)
                g_full = 0.5 * (g_sel + g_ev)
                unw_gains.append(g_full)
                w_gains.append(g_full * u.deduction_weight)
                w_denom += u.deduction_weight
            mean_unw = sum(unw_gains) / len(unw_gains) if unw_gains else 0.0
            mean_w = sum(w_gains) / w_denom if w_denom > 0 else 0.0
            if mean_unw > best_fixed_unw_score:
                best_fixed_unw_score = mean_unw
                best_fixed_unw = arm
            if mean_w > best_fixed_w_score:
                best_fixed_w_score = mean_w
                best_fixed_w = arm

        # 2. Evaluate on disjoint eval_set using split-branch oracle vs train_set's fixed arm
        fold_unw_diffs: list[float] = []
        fold_w_num = 0.0
        fold_w_den = 0.0

        for u in eval_set:
            # Choose unit-best arm using ONLY `g_sel` (first half of branches), evaluate on `g_ev` (second half)
            best_u_arm = "none"
            best_u_sel_gain = 0.0
            eval_gains_by_arm: dict[str, float] = {"none": 0.0}
            for arm in present_arms:
                g_sel, g_ev = _split_branch_gain(u, arm)
                eval_gains_by_arm[arm] = g_ev
                if g_sel > best_u_sel_gain:
                    best_u_sel_gain = g_sel
                    best_u_arm = arm

            oracle_eval_gain = eval_gains_by_arm[best_u_arm]
            fixed_unw_eval_gain = eval_gains_by_arm.get(best_fixed_unw, 0.0)
            fixed_w_eval_gain = eval_gains_by_arm.get(best_fixed_w, 0.0)

            diff_unw = oracle_eval_gain - fixed_unw_eval_gain
            diff_w = oracle_eval_gain - fixed_w_eval_gain
            fold_unw_diffs.append(diff_unw)
            fold_w_num += diff_w * u.deduction_weight
            fold_w_den += u.deduction_weight

            eval_unit_rows.append({
                "fold": fold_idx,
                "unit_id": u.unit_id,
                "deduction_weight": u.deduction_weight,
                "selected_unit_arm": best_u_arm,
                "train_best_fixed_arm_unweighted": best_fixed_unw,
                "train_best_fixed_arm_weighted": best_fixed_w,
                "oracle_eval_gain": round(oracle_eval_gain, 6),
                "fixed_unweighted_eval_gain": round(fixed_unw_eval_gain, 6),
                "fixed_weighted_eval_gain": round(fixed_w_eval_gain, 6),
                "ceiling_increment_unweighted": round(diff_unw, 6),
                "ceiling_increment_weighted": round(diff_w, 6),
            })

        fold_records.append({
            "fold": fold_idx,
            "train_unit_ids": [u.unit_id for u in train_set],
            "eval_unit_ids": [u.unit_id for u in eval_set],
            "best_fixed_arm_unweighted": best_fixed_unw,
            "best_fixed_arm_weighted": best_fixed_w,
            "eval_unweighted_ceiling": round(sum(fold_unw_diffs) / len(fold_unw_diffs), 6) if fold_unw_diffs else 0.0,
            "eval_deduction_weighted_ceiling": round(fold_w_num / fold_w_den, 6) if fold_w_den > 0 else 0.0,
        })

    all_unw = [r["ceiling_increment_unweighted"] for r in eval_unit_rows]
    total_w = sum(r["deduction_weight"] for r in eval_unit_rows)
    unw_ceiling = sum(all_unw) / len(all_unw) if all_unw else 0.0
    w_ceiling = (
        sum(r["ceiling_increment_weighted"] * r["deduction_weight"] for r in eval_unit_rows) / total_w
        if total_w > 0 else 0.0
    )

    return {
        "schema_version": V05_SCHEMA_VERSION,
        "proposition_id": "V3-P3",
        "eligible_unit_count": len(eligible),
        "folds_disjoint": True,
        "unweighted_ceiling": round(unw_ceiling, 6),
        "deduction_weighted_ceiling": round(w_ceiling, 6),
        "ceiling_threshold": ceiling_threshold,
        "passes_v3_p3_gate": (unw_ceiling >= ceiling_threshold) or (w_ceiling >= ceiling_threshold),
        "folds": fold_records,
        "unit_evaluations": eval_unit_rows,
    }


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
            "recommended_arm": rec_arm,
            "predicted_label": pred_lbl,
            "confidence": 0.55,
            "abstain": False,
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
    """R13 & R14 (V3 §3.6 & §3.7): Evaluate Analyst predictions (M / A / B / C) using
    cross-fitted diagnostic value ceiling (NEVER in-sample `max_possible_gain`).
    """
    by_id = {u.unit_id: u for u in units}
    pred_by_id = {p["unit_id"]: p for p in predictions if p.get("unit_id") in by_id}

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

    ranked_items: list[tuple[float, float, float]] = []
    for u in adjudicated_e2:
        p = pred_by_id[u.unit_id]
        rec_arm = str(p.get("recommended_arm") or "none")
        prob_map = {
            "D": float(p.get("d_prob", 0.0)),
            "S": float(p.get("s_prob", p.get("d_prob", 0.0))),
            "R_explicit": float(p.get("r_explicit_prob", p.get("r_prob", 0.0))),
            "R_implicit": float(p.get("r_implicit_prob", p.get("r_prob", 0.0))),
            "R": float(p.get("r_prob", 0.0)),
            "R_single": float(p.get("r_prob", 0.0)),
        }
        if p.get("abstain") or rec_arm == "none":
            priority = 0.0
        else:
            priority = prob_map.get(rec_arm, 0.0) * u.deduction_weight

        vec = u.e2_response_vector
        if rec_arm == "none":
            gain_rate = 0.0
        else:
            pair_k = f"{rec_arm}_vs_P"
            if pair_k in vec.pairwise_deltas:
                gain_rate = max(0.0, float(vec.pairwise_deltas[pair_k]))
            elif rec_arm == "D" and vec.delta_D_vs_P is not None:
                gain_rate = max(0.0, float(vec.delta_D_vs_P))
            elif rec_arm in ("R", "R_single", "R_explicit", "R_implicit") and vec.delta_R_vs_P is not None:
                gain_rate = max(0.0, float(vec.delta_R_vs_P))
            else:
                gain_rate = 0.0

        actual_gain = gain_rate * u.deduction_weight
        req_cost = 0.0 if rec_arm == "none" else float(cost_per_unit_requests)
        ranked_items.append((priority, actual_gain, req_cost))

    ranked_items.sort(key=lambda x: -x[0])
    total_active_requests = sum(c for _, _, c in ranked_items)
    total_actual_gain = sum(g for _, g, c in ranked_items if c > 0)
    utility_per_100_req = (total_actual_gain / total_active_requests * 100.0) if total_active_requests > 0 else 0.0

    # Cross-fitted diagnostic value ceiling (replaces in-sample max_possible_gain)
    crossfit = compute_cross_fitted_diagnostic_value_ceiling(adjudicated_e2)

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
    coverage = (len(non_abstained) / len(adjudicated_e2)) if adjudicated_e2 else 0.0
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
        "adjudicated_e2_unit_count": len(adjudicated_e2),
        "tier1_utility": {
            "utility_per_100_requests": round(utility_per_100_req, 6),
            "total_actual_deduction_recovered": round(total_actual_gain, 6),
            "total_verification_requests": round(total_active_requests, 2),
            "cross_fitted_ceiling": {
                "unweighted_ceiling": crossfit["unweighted_ceiling"],
                "deduction_weighted_ceiling": crossfit["deduction_weighted_ceiling"],
                "passes_v3_p3_gate": crossfit["passes_v3_p3_gate"],
            },
        },
        "tier2_classification": {
            "weighted_macro_f1": round(weighted_macro_f1, 6),
            "weighted_accuracy": round(weighted_acc, 6),
            "f1_by_class": {k: round(v, 6) for k, v in f1_per_class.items()},
            "coverage": round(coverage, 6),
            "selective_weighted_accuracy": round(selective_weighted_acc, 6),
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
) -> dict[str, Any]:
    """R21 (`V3-G4`): Verify that a normal L1 request or `P` arm continuation request contains
    ZERO diagnostic leakage (no rule intervention prompts, no requirement lists, no verifier code).
    """
    raw_text = str(request_payload)
    default_leak_markers = (
        DELIVER_INTERVENTION_MESSAGE[:20],
        DEFAULT_SELF_CHECK_DRAFT[:20],
        "请同时逐条核对以下任务明确要求",
        "请进一步逐条核对以下隐含格式与细则要求",
        "请重点核对并满足以下单项要求",
        "_verifier.py",
        "def verify",
        "dim1_pass",
        "dim2_items",
    )
    leaked: list[str] = []
    for m in (*default_leak_markers, *forbidden_phrases):
        if m and m in raw_text:
            leaked.append(m)
    return {
        "zero_leakage": len(leaked) == 0,
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
    """
    valid = [r for r in sampled_rows if float(r.get("inclusion_prob", 0.0)) > 0.0]
    if not valid:
        return {
            "sampled_count": 0,
            "ht_denom_never_deduction": 0.0,
            "delivery_responsive_share_lower_bound": 0.0,
            "self_check_responsive_share_lower_bound": 0.0,
            "level1_generic_share_lower_bound": 0.0,
            "delivery_responsive_share_upper_bound_incl_inconclusive": 0.0,
            "rule_responsive_share_lower_bound": 0.0,
            "explicit_rule_responsive_share_lower_bound": 0.0,
            "implicit_rule_responsive_share_lower_bound": 0.0,
            "inconclusive_share": 0.0,
            "hajek_shares": {},
            "ht_shares": {},
            "family_cluster_ci95_delivery": [0.0, 0.0],
        }

    def _calc(rows: Sequence[dict[str, Any]], denom_override: float | None = None) -> dict[str, float]:
        ht_denom = sum(float(r["deduction"]) / float(r["inclusion_prob"]) for r in rows)
        denom = float(denom_override) if (denom_override is not None and denom_override > 0) else ht_denom
        if denom <= 0:
            return {
                "ht_denom": 0.0,
                "d_ht": 0.0,
                "s_ht": 0.0,
                "lvl1_ht": 0.0,
                "d_ub_ht": 0.0,
                "r_ht": 0.0,
                "r_exp_ht": 0.0,
                "r_imp_ht": 0.0,
                "inc_ht": 0.0,
                "d_hajek": 0.0,
                "s_hajek": 0.0,
                "lvl1_hajek": 0.0,
                "r_hajek": 0.0,
            }

        def _w_sum(pred) -> float:
            return sum(float(r["deduction"]) / float(r["inclusion_prob"]) for r in rows if pred(r))

        d_num = _w_sum(lambda r: bool(r.get("recovered_by_D")))
        s_num = _w_sum(lambda r: bool(r.get("recovered_by_S")))
        lvl1_num = _w_sum(lambda r: bool(r.get("recovered_by_D") or r.get("recovered_by_S")))
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
            "d_ht": d_num / denom,
            "s_ht": s_num / denom,
            "lvl1_ht": lvl1_num / denom,
            "d_ub_ht": (d_num + inc_num) / denom,
            "r_ht": r_any_num / denom,
            "r_exp_ht": r_exp_num / denom,
            "r_imp_ht": r_imp_num / denom,
            "inc_ht": inc_num / denom,
            "d_hajek": d_num / ht_denom if ht_denom > 0 else 0.0,
            "s_hajek": s_num / ht_denom if ht_denom > 0 else 0.0,
            "lvl1_hajek": lvl1_num / ht_denom if ht_denom > 0 else 0.0,
            "r_hajek": r_any_num / ht_denom if ht_denom > 0 else 0.0,
        }

    metrics = _calc(valid, total_never_deduction)
    ht_denom = metrics["ht_denom"]
    d_lb = metrics["d_ht"]

    by_fam: dict[str, list[dict[str, Any]]] = {}
    for r in valid:
        fid = str(r.get("family_id") or r.get("task_id") or "default")
        by_fam.setdefault(fid, []).append(r)
    fams = sorted(by_fam.keys())

    if len(fams) <= 1:
        ci_d = [round(d_lb, 6), round(d_lb, 6)]
    else:
        rng = random.Random(int(digest({"seed": seed, "n": len(valid), "fams": fams})[:16], 16))
        boot_vals: list[float] = []
        n_boot = 2000
        for _ in range(n_boot):
            sampled_fams = [rng.choice(fams) for _ in range(len(fams))]
            boot_rows = [item for f in sampled_fams for item in by_fam[f]]
            b_m = _calc(boot_rows, None)
            boot_vals.append(b_m["d_hajek"])
        boot_vals.sort()
        lo_idx = max(0, int(math.floor((alpha / 2.0) * (n_boot - 1))))
        hi_idx = min(n_boot - 1, int(math.ceil((1.0 - alpha / 2.0) * (n_boot - 1))))
        ci_d = [round(boot_vals[lo_idx], 6), round(boot_vals[hi_idx], 6)]

    return {
        "sampled_count": len(valid),
        "family_cluster_count": len(fams),
        "ht_denom_never_deduction": round(ht_denom, 6),
        "population_never_deduction_used": round(float(total_never_deduction or ht_denom), 6),
        "delivery_responsive_share_lower_bound": round(metrics["d_ht"], 6),
        "self_check_responsive_share_lower_bound": round(metrics["s_ht"], 6),
        "level1_generic_share_lower_bound": round(metrics["lvl1_ht"], 6),
        "delivery_responsive_share_upper_bound_incl_inconclusive": round(metrics["d_ub_ht"], 6),
        "rule_responsive_share_lower_bound": round(metrics["r_ht"], 6),
        "explicit_rule_responsive_share_lower_bound": round(metrics["r_exp_ht"], 6),
        "implicit_rule_responsive_share_lower_bound": round(metrics["r_imp_ht"], 6),
        "inconclusive_share": round(metrics["inc_ht"], 6),
        "ht_shares": {
            "D": round(metrics["d_ht"], 6),
            "S": round(metrics["s_ht"], 6),
            "level1_D_or_S": round(metrics["lvl1_ht"], 6),
            "R_explicit": round(metrics["r_exp_ht"], 6),
            "R_implicit": round(metrics["r_imp_ht"], 6),
            "R_any": round(metrics["r_ht"], 6),
        },
        "hajek_shares": {
            "D": round(metrics["d_hajek"], 6),
            "S": round(metrics["s_hajek"], 6),
            "level1_D_or_S": round(metrics["lvl1_hajek"], 6),
            "R_any": round(metrics["r_hajek"], 6),
        },
        "family_cluster_ci95_delivery": ci_d,
    }
