"""LLM Analyst (`llm:<model>@<prompt_sha>`, Groups B & C) with strict information boundary (L2 v0.4 §5.1-5.2, F24)."""
from __future__ import annotations
import copy
from pathlib import Path
from typing import Any, Callable

from ..evidence.l1 import L1Package
from ..evidence.view import build_trajectory_view
from ..facts.build import FACT_FIELD_WHITELIST, format_fact_field_whitelist
from ..io import L2Error, Unsupported, canonical, digest, loads, read_json, write_json
from . import Hypothesis, Origin, validate_hypotheses, validate_origins

CALL1_SYSTEM_PROMPT = """You are an L2 defect-origin analyst (Call 1).
Given the agent trajectory, the terminal official evaluation view, and (for Group C) turn-indexed facts and cross-run contrast:
For every target defect in `terminal_official_grade.target_defects`, determine its origin category (`kind`) and origin turn (`turn`):
- Candidate deliverable rule (L2 v0.4 §4.2): Unmodified initial template files in `input/*` at turn 0 do NOT count as candidate deliverables (`s(0) = 0`). A workspace candidate only exists after the agent creates or modifies a target Office file (`.docx`, `.pptx`, `.xlsx`) during turns `1..num_turns`.
- `kind`: one of `"broken"`, `"undelivered"`, or `"never"`:
  - `"broken"`: the rubric item was satisfied on a created/modified workspace candidate during the run and later overwritten/corrupted at turn `turn` (`1..num_turns`), remaining unsatisfied at the end.
  - `"undelivered"`: the rubric item is satisfied on a surviving created/modified workspace candidate starting from turn `turn` (`1..num_turns`) through the end of the episode, but that candidate was never copied/saved to `submission/` (or `submission/` has an older version). When `terminal_official_grade.delivered_valid` is `false` and the agent wrote a surviving candidate Office file outside `submission/` that satisfies the requirement (or preserves template items), classify as `"undelivered"` at the turn (`1..num_turns`) when that candidate file was created/modified.
  - `"never"`: the rubric item was never satisfied on any created/modified workspace candidate (`turn` must be `null`).
- `turn`: integer turn index in `1..num_turns` for `"broken"` or `"undelivered"` (NEVER `0`), or `null` for `"never"`.
IMPORTANT: Output Playbook:
1. You MUST include exactly one entry for EVERY `defect` listed in `terminal_official_grade.target_defects`, copying the `defect` string verbatim.
2. Output ONLY a valid JSON object (no markdown fences):
{"origins": [{"defect": "<exact defect id>", "kind": "broken|undelivered|never", "turn": <int in 1..num_turns or null>, "rationale": "<concise explanation>"}]}"""

CALL2_SYSTEM_PROMPT = (
    """You are an L2 behavioral mechanism analyst (Call 2).
Given the agent trajectory, the defect table (`defects_table`), and (for Group C) turn-indexed facts and cross-run contrast:
For localized defects in `defects_table.defects` that have an integer origin turn `j >= 1` (in `turn` or `two_point_turn`), propose 1 to 5 testable behavioral predicate hypotheses for turn `j`. (Do NOT propose hypotheses for defects where both `turn` and `two_point_turn` are `null` or `0`.)
Each predicate must use ONLY boolean combinators `{"all": [...]}`, `{"any": [...]}`, `{"not": <pred>}`, and atomic predicates `{"fact": "<type>", "where": {...}}` (string values in `where` support `*` wildcards; nested keys use dot notation like `arguments.command`; envelope fields `turn` and `type` are also accessible).

STRICT RULES (L2 v0.4 §5.1):
1. You may ONLY reference `fact` types and `where` fields from the exported field whitelist below (retrieved from `facts/build.py`):
"""
    + format_fact_field_whitelist()
    + """
   Note: `stdout_preview` does NOT exist on `tool_exit` facts and must never be used.
2. Predicates are evaluated across independent resampled branches at turn `j`: write predicates in terms of reproducible tool names (`name`), file paths (`submission/*`, `*.docx`, `*.pptx`, `*.xlsx`), or command structure (`*soffice*`, `*pdftoppm*`, `ok: false`). Do NOT bind to incidental variable names or exact prose strings that only appeared in the parent trajectory.
3. Every hypothesis `defect` MUST exactly match a defect ID from `defects_table.defects` that has `turn >= 1` or `two_point_turn >= 1`, and `effect` MUST be a negative float (e.g. `-0.25`).

Output ONLY a valid JSON object (no markdown fences):
{"hypotheses": [{"defect": "<exact localized defect id>", "turn": <int>, "predicate": <dict>, "effect": <negative float>, "rationale": "<concise explanation>"}]}"""
)

FORBIDDEN_BRANCH_KEYS = {
    'branches',
    'outcomes',
    'branch_receipts',
    'branch_facts',
    's1_samples',
    's2_samples',
}

FORBIDDEN_CALL1_KEYS = FORBIDDEN_BRANCH_KEYS | {
    'grades',
    'snapshot_scores',
    'defects_table',
    'terminal_snapshot_score_s_T',
    'max_snapshot_score',
    'delivery_gap_g',
    'undelivered_gap',
    'peak_to_terminal_loss',
    'deduction_by_kind',
}


def assert_no_forbidden_keys(payload: Any, forbidden: set[str], context: str) -> None:
    if isinstance(payload, dict):
        for k, v in payload.items():
            if str(k) in forbidden:
                raise L2Error(f'Information boundary violation in {context}: forbidden key `{k}`')
            assert_no_forbidden_keys(v, forbidden, context)
    elif isinstance(payload, list):
        for item in payload:
            assert_no_forbidden_keys(item, forbidden, context)


def assert_no_branch_leakage(payload: Any) -> None:
    assert_no_forbidden_keys(payload, FORBIDDEN_BRANCH_KEYS, 'Analyst input')


def build_call1_input(
    parent: L1Package,
    terminal_official_view: dict[str, Any],
    facts: list[dict[str, Any]] | None = None,
    contrast: dict[str, Any] | None = None,
    *,
    include_facts: bool = True,
) -> dict[str, Any]:
    """L2 v0.4 §5.2 Call 1 input for LLM Analysts (Group B when include_facts=False, Group C when True).

    Strictly excludes checkpoint snapshot scores s(k), defect table categories/origins/windows,
    and any branch-level data.
    """
    view = build_trajectory_view(parent)
    payload: dict[str, Any] = {
        'num_turns': parent.turns,
        'terminal_official_grade': copy.deepcopy(terminal_official_view),
        'trajectory': view,
    }
    if include_facts:
        if facts is not None:
            payload['facts'] = copy.deepcopy(facts)
        if contrast is not None:
            payload['contrast'] = copy.deepcopy(contrast)
    assert_no_forbidden_keys(payload, FORBIDDEN_CALL1_KEYS, 'Call 1 input')
    return payload


def build_call2_input(
    parent: L1Package,
    terminal_official_view: dict[str, Any],
    defects_table: dict[str, Any],
    facts: list[dict[str, Any]] | None = None,
    contrast: dict[str, Any] | None = None,
    *,
    include_facts: bool = True,
) -> dict[str, Any]:
    """L2 v0.4 §5.2 Call 2 input for LLM Analysts.

    Includes Call 1's inputs plus `defects_table` (E1 defect categories, origin turns, and windows),
    while strictly excluding any branch-level outcomes or branch facts.
    """
    payload = build_call1_input(
        parent, terminal_official_view, facts, contrast, include_facts=include_facts
    )
    payload['defects_table'] = copy.deepcopy(defects_table)
    assert_no_branch_leakage(payload)
    return payload


def request_model_config(model_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        key: model_cfg[key]
        for key in ('adapter', 'kind', 'model', 'base_url', 'parameters')
        if key in model_cfg
    }


def analyst_tag(model_cfg: dict[str, Any], *, include_facts: bool = True) -> str:
    model_name = str(model_cfg.get('model') or 'unknown')
    mode_tag = 'facts' if include_facts else 'raw'
    from ..config import reject_inline_secrets

    reject_inline_secrets(model_cfg)
    prompt_sha = digest({
        'c1': CALL1_SYSTEM_PROMPT,
        'c2': CALL2_SYSTEM_PROMPT,
        'whitelist': FACT_FIELD_WHITELIST,
        'mode': mode_tag,
        'model_config': request_model_config(model_cfg),
    })[:16]
    return f'llm:{model_name}@{prompt_sha}'


def _resolve_model_backend(model_cfg: dict[str, Any]):
    try:
        from c3_sandbox.l1.models.native import GeminiBackend, OpenAICompatibleBackend
        from c3_sandbox.l1.models.opencode import (
            AnthropicMessagesBackend,
            OpenAIResponsesBackend,
            OpenCodeChatBackend,
            OpenCodeGeminiBackend,
        )
    except ImportError as exc:
        raise Unsupported('Install c3_sandbox L1 for live LLM Analyst calls, or inject llm_caller in tests') from exc
    adapters = {
        'gemini': GeminiBackend,
        'openai-compatible': OpenAICompatibleBackend,
        'opencode-chat': OpenCodeChatBackend,
        'openai-responses': OpenAIResponsesBackend,
        'anthropic-messages': AnthropicMessagesBackend,
        'opencode-gemini': OpenCodeGeminiBackend,
    }
    adapter = model_cfg.get('adapter', 'gemini')
    if adapter not in adapters:
        raise L2Error(f'Unknown model adapter: {adapter}')
    return adapters[adapter](copy.deepcopy(model_cfg))


def _extract_json_obj(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith('```'):
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith('```')]
        text = '\n'.join(lines).strip()
    try:
        return loads(text)
    except Exception:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end > start:
            return loads(text[start : end + 1])
        raise


def _default_llm_transport(
    model_cfg: dict[str, Any],
    system_prompt: str,
    user_payload: dict[str, Any],
    *,
    cache_dir: Path | None = None,
    validator: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    import hashlib
    import json
    import time
    from ..config import DEFAULTS, validate

    model_cfg = copy.deepcopy(model_cfg)
    model_cfg.setdefault('max_response_bytes', DEFAULTS['model']['max_response_bytes'])
    validate({'model': model_cfg})

    cache_req = {
        'schema': 'l2_llm_analyst_cache_v0.4',
        'model_config': request_model_config(model_cfg),
        'system_prompt': system_prompt,
        'user_payload': user_payload,
    }
    ckey = digest(cache_req)
    cfile = (cache_dir / 'llm' / f'{ckey}.json') if cache_dir is not None else None
    if cfile is not None and cfile.is_file():
        cached = read_json(cfile)
        if cached.get('cache_key') == ckey and isinstance(cached.get('parsed'), dict):
            if validator is None:
                return cached['parsed']
            try:
                validator(cached['parsed'])
                return cached['parsed']
            except Exception:
                pass

    backend = _resolve_model_backend(model_cfg)
    last_err: Exception | None = None
    prompt_to_send = system_prompt
    max_attempts = 3 if validator is not None else 1
    for attempt in range(max_attempts):
        conv = backend.initial(prompt_to_send, canonical(user_payload).decode('utf-8'))
        req = backend.request(conv, [])
        if model_cfg.get('adapter') == 'openai-compatible' and not req.get('tools'):
            req.pop('tools', None)
            req.pop('tool_choice', None)
        req_bytes = canonical(req)
        t0 = time.time()
        try:
            resp = backend.send(req_bytes, timeout=float(model_cfg.get('timeout_seconds', 120)))
            elapsed_s = round(time.time() - t0, 4)
            if resp.metadata.get('body_truncated'):
                raise L2Error('LLM Analyst response exceeded model.max_response_bytes')
            parsed = backend.parse(resp.body, 'l2-analyst')
            if not parsed.text:
                raise L2Error('LLM Analyst returned empty text response')
            obj = _extract_json_obj(parsed.text)
            if validator is not None:
                validator(obj)
        except Exception as exc:
            if 'exceeded model.max_response_bytes' in str(exc) or max_attempts == 1:
                raise
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
            prompt_to_send = (
                system_prompt
                + f'\n\n[RETRY CORRECTION (attempt {attempt + 1})]: Your previous output failed validation with error: {exc}. '
                'Fix the violation and return ONLY the valid JSON object.'
            )
            continue

        raw_body_obj: Any = None
        try:
            raw_body_obj = json.loads(resp.body.decode('utf-8', errors='replace'))
        except Exception:
            raw_body_obj = None
        usage = parsed.usage or (
            raw_body_obj.get('usageMetadata') if isinstance(raw_body_obj, dict) else {}
        ) or {}
        if cfile is not None:
            write_json(
                cfile,
                {
                    'cache_key': ckey,
                    'parsed': obj,
                    'usage': usage,
                    'http_status': resp.status,
                    'elapsed_seconds': elapsed_s,
                    'request_sha256': digest(req),
                    'response_sha256': hashlib.sha256(resp.body).hexdigest(),
                    'reported_model_version': (
                        raw_body_obj.get('modelVersion') if isinstance(raw_body_obj, dict) else None
                    ),
                },
            )
        return obj
    assert last_err is not None
    raise last_err


def predict_origins_llm(
    parent: L1Package,
    terminal_official_view: dict[str, Any],
    model_cfg: dict[str, Any],
    facts: list[dict[str, Any]] | None = None,
    contrast: dict[str, Any] | None = None,
    *,
    include_facts: bool = True,
    cache_dir: Path | None = None,
    llm_caller: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
) -> list[Origin]:
    """L2 v0.4 §5.1 Call 1 for LLM Analysts (Group B when include_facts=False, Group C when True)."""
    target_defects = [str(d['defect']) for d in (terminal_official_view.get('target_defects') or [])]
    if not target_defects:
        return []
    inp = build_call1_input(
        parent, terminal_official_view, facts, contrast, include_facts=include_facts
    )
    tag = analyst_tag(model_cfg, include_facts=include_facts)
    group = 'C' if include_facts else 'B'
    if llm_caller is not None:
        raw = llm_caller(CALL1_SYSTEM_PROMPT, inp)
        return validate_origins(raw.get('origins'), target_defects, parent.turns, tag, analyst_group=group)
    raw = _default_llm_transport(
        model_cfg,
        CALL1_SYSTEM_PROMPT,
        inp,
        cache_dir=cache_dir,
        validator=lambda obj: validate_origins(obj.get('origins'), target_defects, parent.turns, tag, analyst_group=group),
    )
    return validate_origins(raw.get('origins'), target_defects, parent.turns, tag, analyst_group=group)


def propose_hypotheses_llm(
    parent: L1Package,
    terminal_official_view: dict[str, Any],
    defects_table: dict[str, Any],
    model_cfg: dict[str, Any],
    facts: list[dict[str, Any]] | None = None,
    contrast: dict[str, Any] | None = None,
    *,
    include_facts: bool = True,
    cache_dir: Path | None = None,
    llm_caller: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
) -> list[Hypothesis]:
    """L2 v0.4 §5.1 Call 2 for LLM Analysts."""
    localized = [
        str(d['defect'])
        for d in (defects_table.get('defects') or [])
        if (
            (isinstance(d.get('turn'), int) and not isinstance(d.get('turn'), bool) and d['turn'] >= 1)
            or (isinstance(d.get('two_point_turn'), int) and not isinstance(d.get('two_point_turn'), bool) and d['two_point_turn'] >= 1)
        )
    ]
    if not localized:
        return []
    inp = build_call2_input(
        parent, terminal_official_view, defects_table, facts, contrast, include_facts=include_facts
    )
    tag = analyst_tag(model_cfg, include_facts=include_facts)
    group = 'C' if include_facts else 'B'
    if llm_caller is not None:
        raw = llm_caller(CALL2_SYSTEM_PROMPT, inp)
        return validate_hypotheses(
            raw.get('hypotheses'),
            parent.turns,
            tag,
            analyst_group=group,
            allowed_defects=set(localized),
        )
    raw = _default_llm_transport(
        model_cfg,
        CALL2_SYSTEM_PROMPT,
        inp,
        cache_dir=cache_dir,
        validator=lambda obj: validate_hypotheses(
            obj.get('hypotheses'),
            parent.turns,
            tag,
            analyst_group=group,
            allowed_defects=set(localized),
        ),
    )
    return validate_hypotheses(
        raw.get('hypotheses'),
        parent.turns,
        tag,
        analyst_group=group,
        allowed_defects=set(localized),
    )
