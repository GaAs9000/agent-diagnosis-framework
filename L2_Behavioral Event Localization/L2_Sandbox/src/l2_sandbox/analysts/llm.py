"""LLM Analyst (`llm:<model>@<prompt_sha>`, Groups B & C) with strict information boundary (L2 v0.3 Section 2.3 & 5)."""
from __future__ import annotations
import copy
from typing import Any, Callable
from ..evidence.l1 import L1Package
from ..evidence.view import build_trajectory_view
from ..io import L2Error, Unsupported, canonical, digest, loads
from ..predicates import validate_predicate
from . import Hypothesis, Values

CALL1_SYSTEM_PROMPT = """You are an L2 trajectory value predictor.
Given the parent trajectory, optional turn facts, and task baseline success rate p_0,
predict V(c_k) in [0, 1] for every checkpoint k = 0 .. T-1.
Output ONLY a JSON object: {"values": [v_0, v_1, ..., v_{T-1}]}."""

CALL2_SYSTEM_PROMPT = """You are an L2 behavioral localization analyst.
Given the parent trajectory, optional turn facts, task baseline p_0, and the value curve summary,
propose testable behavioral predicate hypotheses for the localized drop.
Each predicate must use only `all`, `any`, `not`, and `{"fact": "<type>", "where": {...}}`.
Output ONLY a JSON object: {"hypotheses": [{"turn": <int>, "predicate": <dict>, "effect": <negative float>, "rationale": "<str>"}]}."""

FORBIDDEN_BRANCH_KEYS = {'branches', 'outcomes', 'branch_receipts', 'branch_facts', 's1_samples', 's2_samples'}


def assert_no_branch_leakage(payload: Any) -> None:
    """Enforces the Section 2 information boundary: Analyst inputs never contain branch-level outcomes or actions."""
    if isinstance(payload, dict):
        for k, v in payload.items():
            if str(k) in FORBIDDEN_BRANCH_KEYS:
                raise L2Error(f'Information boundary violation: forbidden branch key `{k}` in Analyst input')
            assert_no_branch_leakage(v)
    elif isinstance(payload, list):
        for item in payload:
            assert_no_branch_leakage(item)


def build_call1_input(
    parent: L1Package,
    p0: float,
    facts: list[dict[str, Any]] | None = None,
    *,
    include_facts: bool = True,
) -> dict[str, Any]:
    """Constructs the Call 1 input for LLM Analysts (Group B when include_facts=False, Group C when True)."""
    view = build_trajectory_view(parent)
    payload: dict[str, Any] = {
        'task_baseline_p0': float(p0),
        'trajectory': view,
    }
    if include_facts and facts is not None:
        payload['facts'] = copy.deepcopy(facts)
    assert_no_branch_leakage(payload)
    return payload


def build_call2_input(
    parent: L1Package,
    p0: float,
    curve_summary: dict[str, Any],
    facts: list[dict[str, Any]] | None = None,
    *,
    include_facts: bool = True,
) -> dict[str, Any]:
    """Constructs the Call 2 input for LLM Analysts.

    `curve_summary` exposes only checkpoint-level V_hat, verdict, turn, and interval —
    never branch-level outcomes or branch facts.
    """
    safe_curve = {
        'verdict': curve_summary.get('verdict'),
        'turn': curve_summary.get('turn'),
        'interval': curve_summary.get('interval'),
        'candidate_turn': curve_summary.get('candidate_turn'),
        'v_hat': copy.deepcopy(curve_summary.get('v_hat') or {}),
        'turn_drops': copy.deepcopy(curve_summary.get('turn_drops') or {}),
    }
    payload = build_call1_input(parent, p0, facts, include_facts=include_facts)
    payload['curve_summary'] = safe_curve
    assert_no_branch_leakage(payload)
    return payload


def analyst_tag(model_cfg: dict[str, Any], *, include_facts: bool = True) -> str:
    model_name = str(model_cfg.get('model') or 'unknown')
    mode_tag = 'facts' if include_facts else 'raw'
    prompt_sha = digest({'c1': CALL1_SYSTEM_PROMPT, 'c2': CALL2_SYSTEM_PROMPT, 'mode': mode_tag})[:8]
    return f'llm:{model_name}@{prompt_sha}'


def _default_llm_transport(model_cfg: dict[str, Any], system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
    try:
        from c3_sandbox.l1.models.native import GeminiBackend, OpenAICompatibleBackend
    except ImportError as exc:
        raise Unsupported('Install c3_sandbox L1 for live LLM Analyst calls, or inject llm_caller in tests') from exc
    cls = GeminiBackend if model_cfg.get('adapter') == 'gemini' else OpenAICompatibleBackend
    backend = cls(copy.deepcopy(model_cfg))
    conv = backend.initial(system_prompt, canonical(user_payload).decode('utf-8'))
    req = backend.request(conv, [])
    resp = backend.send(canonical(req), timeout=float(model_cfg.get('timeout_seconds', 120)))
    parsed = backend.parse(resp.body, 'l2-analyst')
    if not parsed.text:
        raise L2Error('LLM Analyst returned empty text response')
    text = parsed.text.strip()
    if text.startswith('```'):
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith('```')]
        text = '\n'.join(lines).strip()
    return loads(text)


def predict_values_llm(
    parent: L1Package,
    p0: float,
    model_cfg: dict[str, Any],
    facts: list[dict[str, Any]] | None = None,
    *,
    include_facts: bool = True,
    llm_caller: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
) -> Values:
    """Call 1 for LLM Analyst: predicts V(c_k) for k = 0 .. T-1."""
    inp = build_call1_input(parent, p0, facts, include_facts=include_facts)
    caller = llm_caller or (lambda sys_p, u_inp: _default_llm_transport(model_cfg, sys_p, u_inp))
    raw = caller(CALL1_SYSTEM_PROMPT, inp)
    vals = raw.get('values')
    if not isinstance(vals, list) or len(vals) != parent.turns:
        raise L2Error(f'LLM Call 1 must return exactly {parent.turns} values')
    clean_vals = [max(0.0, min(1.0, float(v))) for v in vals]
    return {'analyst': analyst_tag(model_cfg, include_facts=include_facts), 'values': clean_vals}


def propose_hypotheses_llm(
    parent: L1Package,
    p0: float,
    curve_summary: dict[str, Any],
    model_cfg: dict[str, Any],
    facts: list[dict[str, Any]] | None = None,
    *,
    include_facts: bool = True,
    llm_caller: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
) -> list[Hypothesis]:
    """Call 2 for LLM Analyst: only invoked on `concentrated_*` verdicts."""
    verdict = curve_summary.get('verdict')
    if verdict not in ('concentrated_turn', 'concentrated_interval'):
        return []
    inp = build_call2_input(parent, p0, curve_summary, facts, include_facts=include_facts)
    caller = llm_caller or (lambda sys_p, u_inp: _default_llm_transport(model_cfg, sys_p, u_inp))
    raw = caller(CALL2_SYSTEM_PROMPT, inp)
    raw_hyps = raw.get('hypotheses')
    if not isinstance(raw_hyps, list):
        raise L2Error('LLM Call 2 must return a list in "hypotheses"')

    loc_turn = curve_summary.get('turn')
    loc_interval = curve_summary.get('interval')
    allowed_turns: set[int] = set()
    if verdict == 'concentrated_turn' and isinstance(loc_turn, int):
        allowed_turns.add(loc_turn)
    elif isinstance(loc_interval, list) and len(loc_interval) == 2:
        allowed_turns.update(range(int(loc_interval[0]) + 1, int(loc_interval[1]) + 1))

    hyps: list[Hypothesis] = []
    for item in raw_hyps:
        if not isinstance(item, dict):
            continue
        t = int(item['turn'])
        if allowed_turns and t not in allowed_turns:
            raise L2Error(f'Hypothesis turn {t} outside localized region {sorted(allowed_turns)}')
        pred = validate_predicate(item['predicate'])
        eff = float(item['effect'])
        if eff >= 0.0:
            raise L2Error('Hypothesis predicted effect E[Y|P] - E[Y|~P] must be negative')
        hyps.append({
            'turn': t,
            'predicate': copy.deepcopy(pred),
            'effect': eff,
            'rationale': str(item.get('rationale') or ''),
        })
    return hyps
