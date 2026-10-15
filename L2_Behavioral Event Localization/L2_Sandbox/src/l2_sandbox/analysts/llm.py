"""LLM Analyst (`llm:<model>@<prompt_sha>`, Groups B & C) with strict information boundary (L2 v0.3 Section 2.3 & 5)."""
from __future__ import annotations
import copy
from typing import Any, Callable
from ..evidence.l1 import L1Package
from ..evidence.view import build_trajectory_view
from ..io import L2Error, Unsupported, canonical, digest, loads
from ..predicates import validate_predicate
from . import Hypothesis, Values, probability, validate_values, validate_hypotheses

CALL1_SYSTEM_PROMPT = """You are an L2 trajectory value predictor.
Given the parent trajectory, optional turn facts, and task baseline success rate p_0,
predict V(c_k) in [0, 1] for every checkpoint k = 0 .. T-1.
Output ONLY a JSON object: {"values": [v_0, v_1, ..., v_{T-1}]}."""

CALL2_SYSTEM_PROMPT = """You are an L2 behavioral localization analyst.
Given the parent trajectory, optional turn facts, task baseline p_0, and the value curve summary,
propose testable behavioral predicate hypotheses for the localized drop.
Each predicate must use only boolean combinators `{"all": [...]}`, `{"any": [...]}`, `{"not": <pred>}`, and atomic predicates `{"fact": "<type>", "where": {...}}` (string values in `where` support `*` wildcards; nested keys use dot notation like `arguments.command`).
The following generated fact contract is identical for raw-trajectory and facts-assisted analysts; it describes fields, not observed values. Facts have `turn` (1-based integer), `type` (fact type string), and `value` (object). In `where`, value fields use unprefixed paths below; `value.<path>` also works. Envelope `turn` and `type` are accessible. Match scalars (string, number, bool, null), not whole objects or lists. A missing path never matches, including when the expected value is null. Optional fields may be absent; nullable fields may contain null.
Valid `fact` types and their `where` fields are:
- `tool_requested`: `name`, `call_id`, `request_id` (string or null); `arguments` (tool-defined object, default {}), with scalar paths such as `arguments.command` (string) or `arguments.<arg>` as defined by the tool, including nested argument paths.
- `tool_exit`: `call_id`, `name` (string or null); `exit_code` (integer or null); `stop_reason` (string or null); `output_complete` (bool, defaults to true when absent in the observation); `ok` (bool, true exactly when exit_code == 0, stop_reason is null, and output_complete is true); `stderr_preview` (string, first 800 characters of stderr or empty string).
- `file_transition`: `path` (string or null; string matching supports patterns such as "submission/*"); `change` ("created", "modified", "deleted", or "directory_changed"); `before`, `after` (snapshot entry object or null for a missing side). File entry paths: `before.path`, `after.path`, `before.blob`, `after.blob`, `before.sha256`, `after.sha256` (strings); `before.size_bytes`, `after.size_bytes`, `before.mode`, `after.mode` (integers). Directory entries contain only path and mode. Additional snapshot entry fields, if present, retain their source names and scalar types.
- `object_structure_change`: `file_path`, `before_sha256`, `after_sha256` (strings); `before_summary`, `after_summary` (DOCX direct-XML summary objects). Integer counts: `before_summary.paragraphs_recursive`, `after_summary.paragraphs_recursive`, `before_summary.tables_recursive`, `after_summary.tables_recursive`, `before_summary.top_level_tables`, `after_summary.top_level_tables`, `before_summary.rows_recursive`, `after_summary.rows_recursive`, `before_summary.textless_rows_without_drawing`, `after_summary.textless_rows_without_drawing`. Boolean flags: `before_summary.textless_rows_are_not_rendered_empty_row_judgements`, `after_summary.textless_rows_are_not_rendered_empty_row_judgements` (always true). These count XML descendants (top_level_tables counts direct body tables); textless rows exclude drawing/pict/object content and are not rendered-empty-row judgements. Emitted only when both DOCX versions have supported nonempty summaries and those summaries differ; no rendered appearance or semantic object identity is inferred.
- `model_output_text`: `text` (nonempty string), `request_id` (string or null).
- `protocol_error`: `kind` (event kind string), plus the original event data without a fixed allowlist. Native optional fields: `code`, `message`, `reason`, `error_type`, `call_id`, `request_id`, `stop_phase` (strings), `transport_attempts` (integer). This type is selected when data.code == "protocol_error" or the event kind is "tool.rejected".
- `execution_error`: `kind` (event kind string), plus the same original-event fields and types as protocol_error; applies to tool.error, episode.error, or model.error events not selected as protocol_error. Original event data can contain additional scalar or nested fields; use their exact source paths and types. If original data contains kind, it overrides the event kind in both error types.
- `termination`: the episode.finished `status` object, or its event data when status is absent/empty. Native fields are `termination` (stop-reason string), `protocol_validity`, `execution_validity` ("valid", "invalid", "unknown"), `evaluation_validity` ("valid", "invalid", "not_run"), `task_outcome` ("success", "failure", "unknown"), `evidence_completeness` ("complete", "incomplete"). Additional original status/event fields retain their source names and types.
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
        'task_baseline_p0': probability(p0, 'p0'),
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


def request_model_config(model_cfg: dict[str, Any]) -> dict[str, Any]:
    """Identity of request content/routing, excluding accounting and execution controls."""
    return {key: model_cfg[key] for key in ('adapter', 'kind', 'model', 'base_url', 'parameters')
            if key in model_cfg}


def analyst_tag(model_cfg: dict[str, Any], *, include_facts: bool = True) -> str:
    model_name = str(model_cfg.get('model') or 'unknown')
    mode_tag = 'facts' if include_facts else 'raw'
    from ..config import reject_inline_secrets
    reject_inline_secrets(model_cfg)
    prompt_sha = digest({'c1': CALL1_SYSTEM_PROMPT, 'c2': CALL2_SYSTEM_PROMPT, 'mode': mode_tag, 'model_config': request_model_config(model_cfg)})[:16]
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


def _default_llm_transport(model_cfg: dict[str, Any], system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
    from ..config import DEFAULTS, validate
    model_cfg = copy.deepcopy(model_cfg)
    model_cfg.setdefault('max_response_bytes', DEFAULTS['model']['max_response_bytes'])
    validate({'model': model_cfg})
    backend = _resolve_model_backend(model_cfg)
    conv = backend.initial(system_prompt, canonical(user_payload).decode('utf-8'))
    req = backend.request(conv, [])
    if model_cfg.get('adapter') == 'openai-compatible' and not req.get('tools'):
        req.pop('tools', None)
        req.pop('tool_choice', None)
    resp = backend.send(canonical(req), timeout=float(model_cfg.get('timeout_seconds', 120)))
    if resp.metadata.get('body_truncated'):
        raise L2Error('LLM Analyst response exceeded model.max_response_bytes')
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
    vals = validate_values(raw.get('values'), parent.turns)
    return {'analyst': analyst_tag(model_cfg, include_facts=include_facts),
            'analyst_group': 'C' if include_facts else 'B', 'values': vals}


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
    return validate_hypotheses(raw.get('hypotheses'), curve_summary)
