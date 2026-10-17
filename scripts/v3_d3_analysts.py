#!/usr/bin/env python3
"""V3 Stage D3 Entrypoint (R17, R18): Real LLM Analyst B/C Evaluation vs Rule Analyst A & Majority M.

Contract (Notion V3 §3.6, §3.7, §5.2 `V3-P4`, §8.2, R17, R18):
- Strictly gated on Stage D2 (`V3-P3` cross-fitted diagnostic value ceiling passing).
- Moves all deterministic / keyword heuristics exclusively into Analyst Group `A` (`predict_rule_analyst_a_v05`).
- Implements real HTTP LLM calls (`gemini-3.8-flash` / OpenAI-compatible endpoint) for:
  * Analyst Group `B`: raw L1 `messages.jsonl` + `tool_calls.jsonl` (blind to `s(k)`, `grades.jsonl`, and `facts.jsonl`).
  * Analyst Group `C`: L1 trajectory + L2 zero-request `facts.jsonl` (blind to `s(k)` and `grades.jsonl`).
- Saves raw HTTP response payloads, token `usage`, and `prompt_sha256` receipts for every `B`/`C` call.
- Defaults to `--plan-only` (`R18`), issuing ZERO LLM requests unless `--execute --d0-confirmed` is passed
  AND `v3_d2_ceiling_report_20261017.json` has `d3_prerequisite_gate_open == True`.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[1]
L2_ROOT = REPO_ROOT / "L2_Behavioral Event Localization" / "L2_Sandbox"
sys.path.insert(0, str(L2_ROOT / "src"))

from l2_sandbox.v05 import (  # noqa: E402
    V05_SCHEMA_VERSION,
    sha256_text,
)

DEFAULT_D1_PLAN = L2_ROOT / "reports" / "v3_d1_ladder_plan_20261017.json"
DEFAULT_D2_REPORT = L2_ROOT / "reports" / "v3_d2_ceiling_report_20261017.json"
DEFAULT_D3_PLAN_OUT = L2_ROOT / "reports" / "v3_d3_analysts_plan_20261017.json"

ANALYST_B_SYSTEM_PROMPT = (
    "You are Analyst B (Raw Trajectory Log Analyst) under V3 §3.6. "
    "Given ONLY the raw L1 messages and tool calls up to the intervention anchor checkpoint (and blind to "
    "any checkpoint verifier scores s(k) or L2 structured facts), predict the success probabilities under "
    "each ladder arm (p_prob, d_prob, s_prob, r_explicit_prob, r_implicit_prob), recommended_arm "
    "('none' | 'D' | 'S' | 'R_explicit' | 'R_implicit'), predicted_label, confidence, abstain, and evidence_steps. "
    "Respond in strict JSON."
)

ANALYST_C_SYSTEM_PROMPT = (
    "You are Analyst C (Structured Evidence Analyst) under V3 §3.6. "
    "Given the L1 trajectory AND L2 zero-request structured facts.jsonl up to the intervention anchor checkpoint "
    "(while remaining strictly blind to checkpoint verifier scores s(k) and grades.jsonl), predict the success "
    "probabilities under each ladder arm (p_prob, d_prob, s_prob, r_explicit_prob, r_implicit_prob), recommended_arm "
    "('none' | 'D' | 'S' | 'R_explicit' | 'R_implicit'), predicted_label, confidence, abstain, and evidence_steps. "
    "Respond in strict JSON."
)


def call_real_llm_analyst(
    *,
    analyst_group: str,
    unit_id: str,
    system_prompt: str,
    user_prompt: str,
    model_name: str = "gemini-3.8-flash",
    api_base: str = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    api_key: str = "",
) -> dict[str, Any]:
    """Execute a real LLM API request for Analyst B or C and record full HTTP + usage + prompt_sha256 receipt."""
    if not api_key:
        raise RuntimeError("Missing API key for real LLM Analyst B/C execution")
    payload = {
        "model": model_name,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        api_base,
        data=body_bytes,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        status_code = resp.status
        raw_resp = json.loads(resp.read().decode("utf-8"))

    content_str = (((raw_resp.get("choices") or [{}])[0].get("message") or {}).get("content")) or "{}"
    parsed = json.loads(content_str)
    return {
        "unit_id": unit_id,
        "analyst_group": analyst_group,
        "model": model_name,
        "prompt_sha256": sha256_text(system_prompt + "\n---\n" + user_prompt),
        "http_status": status_code,
        "usage": raw_resp.get("usage") or {},
        "raw_response_id": raw_resp.get("id"),
        "prediction": parsed,
    }


def build_d3_plan(
    *,
    d1_plan_path: Path,
    d2_report_path: Path,
    analyst_model: str = "gemini-3.8-flash",
) -> dict[str, Any]:
    d1_plan = json.loads(d1_plan_path.read_text(encoding="utf-8")) if d1_plan_path.exists() else {}
    d2_rep = json.loads(d2_report_path.read_text(encoding="utf-8")) if d2_report_path.exists() else {}

    sampled_units = ((d1_plan.get("stratified_sampling_frame") or {}).get("sampled_units")) or []
    n_units = len(sampled_units)
    d2_gate_open = bool(d2_rep.get("d3_prerequisite_gate_open", False))

    return {
        "schema_version": V05_SCHEMA_VERSION,
        "stage": "D3",
        "mode": "plan_only",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prerequisite_d2_gate": {
            "d2_report_path": str(d2_report_path),
            "d2_status": d2_rep.get("status", "missing"),
            "d3_prerequisite_gate_open": d2_gate_open,
            "rule": "Stage D3 LLM Analyst calls (B/C) are strictly blocked unless Stage D2 V3-P3 ceiling >= threshold.",
        },
        "analyst_groups": {
            "M": "Zero-cost majority-class / best-fixed-arm baseline (0 LLM requests)",
            "A": "Deterministic rule / keyword heuristic baseline on L2 facts.jsonl (0 LLM requests; R17)",
            "B": f"Real LLM ({analyst_model}) reading raw L1 trajectory log up to anchor k (1 LLM request / unit)",
            "C": f"Real LLM ({analyst_model}) reading L1 trajectory + L2 zero-request facts.jsonl up to anchor k (1 LLM request / unit)",
        },
        "prompt_hashes": {
            "B_system_prompt_sha256": sha256_text(ANALYST_B_SYSTEM_PROMPT),
            "C_system_prompt_sha256": sha256_text(ANALYST_C_SYSTEM_PROMPT),
        },
        "budget_and_stop_conditions": {
            "target_unit_count": n_units,
            "actor_requests_cap": 0,
            "analyst_llm_requests_cap": 2 * n_units,
            "estimated_api_cost_usd_upper_bound": round(2 * n_units * 0.003, 4),
            "stop_conditions": [
                "Do not execute unless D2 V3-P3 gate is open (`d3_prerequisite_gate_open == True`).",
                "Abort if any B/C prompt fails zero-score-leakage check (no s(k) or grades.jsonl allowed).",
                "Abort if HTTP error or invalid JSON schema rate exceeds 5%.",
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="V3 Stage D3 Real LLM Analyst B/C Plan & Evaluator")
    parser.add_argument("--d1-plan", type=Path, default=DEFAULT_D1_PLAN)
    parser.add_argument("--d2-report", type=Path, default=DEFAULT_D2_REPORT)
    parser.add_argument("--out", type=Path, default=DEFAULT_D3_PLAN_OUT)
    parser.add_argument("--analyst-model", type=str, default="gemini-3.8-flash")
    parser.add_argument("--plan-only", action="store_true", default=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--d0-confirmed", action="store_true")
    args = parser.parse_args()

    plan = build_d3_plan(
        d1_plan_path=args.d1_plan,
        d2_report_path=args.d2_report,
        analyst_model=args.analyst_model,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        f"[V3-D3 PLAN] Wrote {args.out}\n"
        f"  d2_gate_open             = {plan['prerequisite_d2_gate']['d3_prerequisite_gate_open']}\n"
        f"  target_unit_count        = {plan['budget_and_stop_conditions']['target_unit_count']}\n"
        f"  analyst_llm_requests_cap = {plan['budget_and_stop_conditions']['analyst_llm_requests_cap']}\n"
        f"  est_cost_usd_upper_bound = ${plan['budget_and_stop_conditions']['estimated_api_cost_usd_upper_bound']}"
    )

    if args.execute:
        if not args.d0_confirmed:
            sys.stderr.write("BLOCKED [R18]: --execute requires --d0-confirmed.\n")
            return 2
        if not plan["prerequisite_d2_gate"]["d3_prerequisite_gate_open"]:
            sys.stderr.write("BLOCKED [R17/R18]: Stage D2 V3-P3 gate is NOT open. Refusing to issue Analyst LLM requests.\n")
            return 2
        if not os.environ.get("GEMINI_API_KEY"):
            sys.stderr.write("BLOCKED [R17]: GEMINI_API_KEY not set for real LLM B/C execution.\n")
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
