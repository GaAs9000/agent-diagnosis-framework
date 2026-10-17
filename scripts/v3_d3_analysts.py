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
import urllib.error

REPO_ROOT = Path(__file__).resolve().parents[1]
L2_ROOT = REPO_ROOT / "L2_Behavioral Event Localization" / "L2_Sandbox"
sys.path.insert(0, str(L2_ROOT / "src"))

from l2_sandbox.v05 import (  # noqa: E402
    V05_SCHEMA_VERSION,
    sha256_text,
    DiagnosticUnitV05, E1Observation, E2ResponseVector,
    predict_majority_baseline_m, predict_rule_analyst_a_v05, evaluate_analysts_v05, adjudicate_e2_outcomes,
)

L1_ROOT = REPO_ROOT / "L1_sandbox" / "C3_Experimental_Sandbox"
sys.path.insert(0, str(L1_ROOT / "src"))
from l2_sandbox.v3_execution import Admission, approve, digest, exclusive, file_digest, identity, read_json, seal, verify_seal, write_json

DEFAULT_D1_PLAN = L2_ROOT / "reports" / "v3_d1_ladder_plan_20261017.json"
DEFAULT_D2_REPORT = L2_ROOT / "reports" / "v3_d2_ceiling_report_20261017.json"
DEFAULT_D3_PLAN_OUT = L2_ROOT / "reports" / "v3_d3_analysts_plan_20261017.json"

ANALYST_B_SYSTEM_PROMPT = (
    "You are Analyst B (Raw Trajectory Log Analyst) under V3 §3.6. "
    "Given ONLY the raw L1 messages and tool calls from the parent run with the shared target and anchor summary (and blind to "
    "any checkpoint verifier scores s(k) or L2 structured facts), predict the success probabilities under "
    "each ladder arm (p_prob, d_prob, s_prob, r_explicit_prob, r_implicit_prob), recommended_arm "
    "('none' | 'P' | 'D' | 'S' | 'R_explicit' | 'R_single' only when explicitly deployable). R_implicit probabilities are attribution only; never recommend hidden specifications, predicted_label, confidence, abstain, and evidence_steps. "
    "Respond in strict JSON."
)

ANALYST_C_SYSTEM_PROMPT = (
    "You are Analyst C (Structured Evidence Analyst) under V3 §3.6. "
    "Given the L1 trajectory AND L2 zero-request structured facts.jsonl from the parent run with the shared target and anchor summary "
    "(while remaining strictly blind to checkpoint verifier scores s(k) and grades.jsonl), predict the success "
    "probabilities under each ladder arm (p_prob, d_prob, s_prob, r_explicit_prob, r_implicit_prob), recommended_arm "
    "('none' | 'P' | 'D' | 'S' | 'R_explicit' | 'R_single' only when explicitly deployable). R_implicit probabilities are attribution only; never recommend hidden specifications, predicted_label, confidence, abstain, and evidence_steps. "
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
    max_tokens: int = 1024,
    timeout_seconds: float = 90,
) -> dict[str, Any]:
    """Execute a real LLM API request for Analyst B or C and record full HTTP + usage + prompt_sha256 receipt."""
    if not api_key:
        raise RuntimeError("Missing API key for real LLM Analyst B/C execution")
    payload = {
        "model": model_name,
        "temperature": 1.0,
        "max_tokens": max_tokens,
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
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            status_code = resp.status
            raw_text = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        status_code = exc.code
        raw_text = exc.read().decode("utf-8", errors="replace")
    try:
        raw_resp = json.loads(raw_text)
        if not isinstance(raw_resp, dict):
            raw_resp = {"invalid_response": raw_resp}
    except ValueError:
        raw_resp = {"invalid_json_response": raw_text}
    content_str = (((raw_resp.get("choices") or [{}])[0].get("message") or {}).get("content")) or "{}"
    try:
        parsed = json.loads(content_str) if status_code == 200 else None
    except (ValueError, TypeError):
        parsed = None
    return {
        "unit_id": unit_id,
        "analyst_group": analyst_group,
        "model": model_name,
        "prompt_sha256": sha256_text(system_prompt + "\n---\n" + user_prompt),
        "http_status": status_code,
        "usage": raw_resp.get("usage") or {},
        "raw_response_id": raw_resp.get("id"),
        "raw_response": raw_resp,
        "request_payload": payload,
        "prediction": parsed,
    }


def _unit(row):
    return DiagnosticUnitV05(unit_id=row['unit_id'], episode_id=row['episode_id'], task_id=row['task_id'], benchmark='officeval', family_id=row.get('family_id'), unit_level=row.get('unit_level','rubric_item'), defect_name=row.get('defect_name',''), deduction_weight=row.get('deduction_weight',1.0), rule_source=row.get('rule_source','ambiguous'), e1_observation=E1Observation(**row.get('e1_observation', dict(e1_class='never_succeeded_on_disk', candidate_substate='delivered_item_wrong', k_star=row['anchor_k'], max_item_score=0.0, terminal_item_score=row.get('terminal_y',0.0), deduction=row.get('deduction_weight',1.0)))), e2_response_vector=E2ResponseVector(), operational_label=None, evidence_status='inconclusive')


def _analyst_input(unit, d1_plan, *, runs_dir=None, diag_dir=None):
    runs = runs_dir or Path(d1_plan['source_dataset']['runs_dir'])
    diagnoses = diag_dir or L2_ROOT / 'diagnoses/v3_qwen36_pi_dev45_20261017'
    parent = runs / unit['episode_id']
    if identity(parent, unit['anchor_k']) != unit['parent_identity']:
        raise ValueError('parent identity differs from D1 plan')
    cp = read_json(parent / 'checkpoints' / f"cp{unit['anchor_k']:06d}.json")
    raw_grade = read_json(parent / 'official_grade.json')['official_raw']
    manifest = read_json(parent / 'manifest.json')
    deployable = unit.get('deployable_arms') or [a for a in unit['eligible_arms'] if a in ('P','D','S','R_explicit') or (a=='R_single' and unit['rule_source']=='explicit')]
    common = dict(deployable_arms=deployable, attribution_only_arms=[a for a in unit['eligible_arms'] if a not in deployable], task_id=unit['task_id'], target=unit['defect_name'], terminal_parent_score=dict(total_score=raw_grade.get('total_score'), max_score=raw_grade.get('max_score'), dim1_pass=raw_grade.get('dim1_pass')), e1_observation=unit.get('e1_observation'), anchor_k=unit['anchor_k'], remaining_budget=unit['remaining_budget'], configuration=unit['parent_identity'], arm_definitions={a: unit['arm_messages'][a]['message'] for a in unit['eligible_arms']}, trajectory=cp['conversation'])
    parent_events = []
    for line in (parent / 'trajectory.jsonl').read_text().splitlines():
        event = json.loads(line)
        if event.get('kind') in ('model.parsed', 'tool.call', 'tool.result'):
            data = dict(event.get('data') or {})
            if event['kind'] == 'tool.result':
                relative = (data.get('result') or {}).get('path')
                if relative:
                    result_path = (parent / relative).resolve()
                    if not result_path.is_relative_to(parent.resolve()):
                        raise ValueError('unsafe observation path')
                    data['recorded_result'] = read_json(result_path)
            parent_events.append(dict(kind=event['kind'], data=data))
    common['parent_raw_events'] = parent_events
    # Derive C/A facts from the bound parent rather than trusting an external
    # diagnosis file whose parent index may belong to an older episode.
    from l2_sandbox.evidence.l1 import L1Package
    from l2_sandbox.facts.build import FACT_FIELD_WHITELIST, build_facts
    parent_facts = build_facts(L1Package(parent))
    facts = []
    allowed = set(FACT_FIELD_WHITELIST) - {'termination'}
    for fact in parent_facts:
        turn = fact.get('turn', fact.get('step'))
        if type(turn) is not int or fact.get('type') not in allowed:
            continue
        value = fact.get('value') or {}
        projected = {}
        for field in FACT_FIELD_WHITELIST[fact['type']]:
            src, dst = value, projected
            bits = field.split('.')
            for key in bits[:-1]:
                src = src.get(key, {}) if isinstance(src, dict) else {}
                dst = dst.setdefault(key, {})
            if isinstance(src, dict) and bits[-1] in src:
                dst[bits[-1]] = src[bits[-1]]
        facts.append(dict(turn=turn, step=turn, type=fact['type'], value=projected))
    return dict(unit={k:v for k,v in unit.items() if k not in ('schedule','arm_messages')}, parent_identity=unit['parent_identity'], B_prompt=json.dumps(common, ensure_ascii=False), C_prompt=json.dumps(dict(common, facts=facts), ensure_ascii=False), facts=facts, facts_sha256=digest(facts))


def _validate_prediction(pred, unit_id, group, deployable_arms):
    if not isinstance(pred, dict):
        raise ValueError('prediction must be object')
    for field in ('p_prob','d_prob','s_prob','r_explicit_prob','r_implicit_prob','confidence'):
        value = pred.get(field)
        if type(value) not in (int,float) or not 0 <= value <= 1:
            raise ValueError(f'invalid prediction probability {field}')
    if pred.get('recommended_arm') not in ('none', *deployable_arms) or type(pred.get('abstain')) is not bool:
        raise ValueError('invalid prediction action/abstain')
    if not isinstance(pred.get('predicted_label'), str) or not isinstance(pred.get('evidence_steps'), list):
        raise ValueError('invalid prediction label/evidence')
    return dict(pred, unit_id=unit_id, analyst_group=group)


def execute_d3_plan(plan, *, freeze, d1_plan, d2_report, outcomes_path, receipts_dir, results_out, api_key='', llm_call=None):
    caps = approve(plan, freeze)
    verify_seal(d1_plan, 'plan_sha256')
    verify_seal(d2_report)
    if plan['d1_plan_sha256'] != d1_plan['plan_sha256'] or plan['d2_report_sha256'] != d2_report['artifact_sha256'] or not d2_report.get('d3_prerequisite_gate_open'):
        raise ValueError('D3 prerequisite artifacts differ from approved plan')
    if plan.get('development_baseline'):
        majority, fixed = validate_development_baseline(plan['development_baseline'], [e['unit'] for e in plan['analyst_inputs']])
        if (majority, fixed) != (plan['majority_label'], plan['best_fixed_arm']):
            raise ValueError('baseline changed after freeze')
    with exclusive(receipts_dir / '.stage.lock'):
        if caps["max_requests"] < 2 * len(plan["analyst_inputs"]):
            raise ValueError("approved analyst request cap does not cover frozen plan")
        for group, system in (("B", ANALYST_B_SYSTEM_PROMPT), ("C", ANALYST_C_SYSTEM_PROMPT)):
            if group + "_system_prompt_sha256" in plan.get("prompt_hashes", {}) and plan["prompt_hashes"][group + "_system_prompt_sha256"] != sha256_text(system):
                raise ValueError("Analyst system prompt differs from frozen plan")
        ledger = Admission(receipts_dir / 'admission.json', plan['plan_sha256'], caps)
        call = llm_call or call_real_llm_analyst
        predictions = []
        for entry in plan['analyst_inputs']:
            unit = _unit(entry['unit'])
            baselines = {'M': predict_majority_baseline_m([unit], default_majority_label=plan['majority_label'])[0], 'A': predict_rule_analyst_a_v05(unit, entry['facts'])}
            baselines['M']['recommended_arm'] = plan.get('best_fixed_arm') or baselines['M']['recommended_arm']
            for group in ('M','A','B','C'):
                prompt = entry.get(group + '_prompt', '')
                system = ANALYST_B_SYSTEM_PROMPT if group == 'B' else ANALYST_C_SYSTEM_PROMPT
                binding = dict(plan_sha256=plan['plan_sha256'], unit_id=unit.unit_id, analyst_group=group, prompt_sha256=sha256_text(system + "\n---\n" + prompt) if group in ('B','C') else digest(baselines[group]))
                path = receipts_dir / (digest(binding) + '.json')
                if path.exists():
                    receipt = read_json(path)
                    verify_seal(receipt)
                    if any(receipt.get(k) != v for k,v in binding.items()) or receipt.get('status') != 'completed':
                        raise ValueError('prediction cache invalid/inflight/failed; no automatic retry')
                else:
                    ledger.check()
                    write_json(path, seal(dict(binding, status='inflight')))
                    try:
                        if group in baselines:
                            result = dict(prediction=baselines[group], usage={}, request_count=0)
                        else:
                            estimate = len(system.encode('utf-8')) + len(prompt.encode('utf-8')) + plan['max_output_tokens'] + 512
                            if not ledger.acquire(estimate):
                                raise RuntimeError('frozen analyst admission cap reached')
                            result = call(analyst_group=group, unit_id=unit.unit_id, system_prompt=system, user_prompt=prompt, model_name=plan['analyst_model'], api_key=api_key, max_tokens=plan['max_output_tokens'], timeout_seconds=min(90, caps['max_wall_seconds'] - (__import__('time').time()-ledger.state['started_at'])))
                            usage = result.get('usage') or {}
                            ledger.reconcile(caps['max_tokens_per_request'], usage.get('total_tokens'))
                        write_json(path, seal(dict(result, **binding, status='response_received')))
                        deployable = entry['unit'].get('deployable_arms') or [a for a in entry['unit']['eligible_arms'] if a in ('P','D','S','R_explicit') or (a=='R_single' and unit.rule_source=='explicit')]
                        pred = _validate_prediction(result['prediction'], unit.unit_id, group, deployable)
                        receipt = seal(dict(result, **binding, prediction=pred, status='completed'))
                        write_json(path, receipt)
                    except Exception as exc:
                        if ledger.current is not None:
                            ledger.reconcile(caps['max_tokens_per_request'], None)
                        ledger.invalid()
                        retained = read_json(path)
                        retained.pop('artifact_sha256', None)
                        write_json(path, seal(dict(retained, status='failed', error_type=type(exc).__name__, error=str(exc))))
                        raise
                deployable = entry['unit'].get('deployable_arms') or [a for a in entry['unit']['eligible_arms'] if a in ('P','D','S','R_explicit') or (a=='R_single' and unit.rule_source=='explicit')]
                predictions.append(_validate_prediction(receipt['prediction'], unit.unit_id, group, deployable))
        # Mandatory temporal boundary: write all predictions before reading validation outcomes.
        prediction_seal = seal(dict(plan_sha256=plan['plan_sha256'], predictions=predictions))
        prediction_path = receipts_dir / 'sealed_predictions.json'
        if prediction_path.exists() and read_json(prediction_path) != prediction_seal:
            raise ValueError('sealed prediction set changed')
        write_json(prediction_path, prediction_seal)
        outcomes = read_json(outcomes_path)
        verify_seal(outcomes)
        if outcomes['artifact_sha256'] != plan['d1_outcomes_sha256'] or outcomes.get('simulated_offline'):
            raise ValueError('D1 outcome identity/mode mismatch')
        # Revalidate the evidence path after predictions are sealed.
        from v3_d2_ceiling import evaluate_d2_from_d1_payload
        params = d1_plan['d0_candidate_parameters']
        check = evaluate_d2_from_d1_payload(outcomes, d1_plan_payload=d1_plan, ceiling_threshold=d2_report['d0_parameters']['ceiling_threshold'], delta=params['delta'], tau=params['tau'], alpha=params['alpha'], mode='execute')
        if not check['d1_provenance_audit']['plan_binding_verified']:
            raise ValueError('D1 branch provenance failed at evaluation')
        units = []
        for row in outcomes['evaluated_units']:
            u = _unit(row)
            vec, label, status = adjudicate_e2_outcomes(row['terminal_y'], row['arm_outcomes'], anchor_k=row['anchor_k'], remaining_budget=row['remaining_budget'], delta=params['delta'], tau=params['tau'], alpha=params['alpha'], primary_family_size=d1_plan['primary_test_family']['size'], comparison_ids=row['comparison_ids'])
            u.e2_response_vector, u.operational_label, u.evidence_status = vec, label, status
            units.append(u)
        for pred in predictions:
            row = next(r for r in outcomes['evaluated_units'] if r['unit_id'] == pred['unit_id'])
            chosen = 'P' if pred.get('abstain') or pred.get('recommended_arm') in ('none','P') else pred['recommended_arm']
            pred['verification_requests'] = sum(r.get('requests_used', 0) for r in row.get('branch_receipts', {}).get(chosen, []))
        result = seal(dict(stage='D3', status='completed', plan_sha256=plan['plan_sha256'], predictions_sha256=prediction_seal['artifact_sha256'], d1_outcomes_sha256=outcomes['artifact_sha256'], results={g:evaluate_analysts_v05(units, [p for p in predictions if p['analyst_group']==g]) for g in ('M','A','B','C')}, admission=ledger.state))
        write_json(results_out, result)
        return result


def validate_development_baseline(baseline, evaluated_units):
    """Recompute M from independently sealed production calibration evidence."""
    verify_seal(baseline)
    source_plan = read_json(Path(baseline['source_plan_path']))
    source = read_json(Path(baseline['source_outcomes_path']))
    verify_seal(source_plan, 'plan_sha256')
    verify_seal(source)
    if source_plan['plan_sha256'] != baseline['source_plan_sha256'] or source['artifact_sha256'] != baseline['source_outcomes_sha256']:
        raise ValueError('calibration source hash mismatch')
    if source.get('simulated_offline') or source.get('status') != 'completed':
        raise ValueError('calibration requires completed production evidence')
    from v3_d2_ceiling import evaluate_d2_from_d1_payload
    params = source_plan['d0_candidate_parameters']
    checked = evaluate_d2_from_d1_payload(source, d1_plan_payload=source_plan, delta=params['delta'], tau=params['tau'], alpha=params['alpha'], ceiling_threshold=params['ceiling_threshold'], min_replicates_per_arm=params['min_replicates_per_arm'], mode='execute')
    if not checked['d1_provenance_audit']['plan_binding_verified'] or checked['d1_provenance_audit']['rejected_unit_count']:
        raise ValueError('calibration D1 provenance does not verify')
    if baseline.get('split') != 'development' or baseline.get('actor_model') != 'Qwen/Qwen3.6-27B' or baseline.get('harness') != 'pi':
        raise ValueError('calibration must use matched development actor/harness')
    rows = source['evaluated_units']
    ids, episodes, families = {r['unit_id'] for r in rows}, {r['episode_id'] for r in rows}, {r['family_id'] for r in rows}
    pool = baseline['calibration_pool']
    if ids != set(baseline['training_unit_ids']) or episodes != set(pool['episode_ids']) or families != set(pool['family_ids']):
        raise ValueError('calibration pool differs from frozen source identities')
    if ids & {r['unit_id'] for r in evaluated_units} or episodes & {r['episode_id'] for r in evaluated_units} or families & {r['family_id'] for r in evaluated_units}:
        raise ValueError('calibration/evaluation units, parents and families must be disjoint')
    configs = set()
    for row in rows:
        parent = Path(source_plan['source_dataset']['runs_dir']) / row['episode_id']
        config = read_json(parent / 'raw/resolved_config.json')
        if config['model'].get('model') != baseline['actor_model'] or config['agent'].get('harness') != baseline['harness']:
            raise ValueError('actual calibration model/harness mismatch')
        configs.add(row['parent_identity']['config_sha256'])
    if configs != set(baseline['source_config_sha256s']):
        raise ValueError('calibration configuration digests mismatch')
    counts, weighted = {}, {}
    for row in rows:
        _, label, state = adjudicate_e2_outcomes(row['terminal_y'], row['arm_outcomes'], anchor_k=row['anchor_k'], remaining_budget=row['remaining_budget'], delta=params['delta'], tau=params['tau'], alpha=params['alpha'], primary_family_size=source_plan['primary_test_family']['size'], comparison_ids=row['comparison_ids'])
        label = label or (state if state == 'plain_recoverable' else None)
        if label is not None:
            counts[label] = counts.get(label,0) + 1
            weighted[label] = weighted.get(label,0.0) + row['deduction_weight']
    if not counts or counts != baseline['label_counts'] or weighted != baseline['deduction_weighted_label_counts']:
        raise ValueError('baseline label frequencies do not match independently recomputed calibration')
    common = set.intersection(*(set(r['arm_outcomes']) for r in rows)) & {'P','D','S','R_explicit'}
    total_weight = sum(r['deduction_weight'] for r in rows)
    scores = {a:sum((sum(r['arm_outcomes'][a])/len(r['arm_outcomes'][a]))*r['deduction_weight'] for r in rows)/total_weight for a in common}
    if scores != baseline['fixed_arm_development_scores']:
        raise ValueError('fixed policy scores do not match calibration')
    return max(sorted(weighted),key=weighted.get), max(sorted(scores),key=scores.get)


def build_d3_plan(
    *,
    d1_plan_path: Path,
    d2_report_path: Path,
    analyst_model: str = "gemini-3.8-flash",
    runs_dir: Path | None = None,
    diag_dir: Path | None = None,
    baseline_path: Path | None = None,
) -> dict[str, Any]:
    d1_plan = json.loads(d1_plan_path.read_text(encoding="utf-8")) if d1_plan_path.exists() else {}
    d2_rep = json.loads(d2_report_path.read_text(encoding="utf-8")) if d2_report_path.exists() else {}

    sampled_units = d1_plan.get("unit_schedules") or []
    n_units = len(sampled_units)
    d2_gate_open = bool(d2_rep.get("d3_prerequisite_gate_open", False))

    inputs, errors = [], []
    baseline_path = baseline_path or L2_ROOT / "configs/v3_d3_development_baseline.json"
    baseline, majority_label, best_fixed = {}, None, None
    try:
        baseline = read_json(baseline_path)
        majority_label, best_fixed = validate_development_baseline(baseline, sampled_units)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        errors.append(f"development_baseline:{exc}")
    try:
        verify_seal(d1_plan, "plan_sha256")
        verify_seal(d2_rep)
        if d2_rep.get("d1_plan_sha256") != d1_plan["plan_sha256"] or d2_rep.get("mode") != "execute" or d2_rep.get("simulated_offline"):
            raise ValueError("D2 frozen provenance is not production execution")
    except (ValueError, KeyError) as exc:
        errors.append(str(exc))
    for unit in sampled_units:
        if best_fixed is not None and best_fixed not in unit.get("deployable_arms", unit.get("eligible_arms", [])):
            errors.append(f"{unit['unit_id']}:frozen_best_fixed_arm_ineligible")
        try:
            inputs.append(_analyst_input(unit, d1_plan, runs_dir=runs_dir, diag_dir=diag_dir))
        except (ValueError, OSError, KeyError) as exc:
            errors.append(f"{unit['unit_id']}:{exc}")
    return seal({
        "schema_version": V05_SCHEMA_VERSION,
        "stage": "D3",
        "d1_plan_sha256": d1_plan.get("plan_sha256"),
        "d2_report_sha256": d2_rep.get("artifact_sha256"),
        "d1_outcomes_sha256": d2_rep.get("d1_outcomes_sha256"),
        "execution_ready": bool(inputs and not errors and d2_gate_open),
        "readiness_errors": errors,
        "analyst_model": analyst_model,
        "analyst_inputs": inputs,
        "majority_label": majority_label,
        "best_fixed_arm": best_fixed,
        "development_baseline": baseline,
        "input_scope": "all parent model.parsed/tool.call/tool.result events, checkpoint conversation and parent zero-request facts; no validation branch evidence",
        "max_output_tokens": 1024,
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
            "estimated_api_cost_usd_upper_bound": None,
            "pricing_status": "requires frozen conservative usd_per_token_upper_bound and token reservations",
            "stop_conditions": [
                "Do not execute unless D2 V3-P3 gate is open (`d3_prerequisite_gate_open == True`).",
                "Abort if any B/C prompt fails zero-score-leakage check (no s(k) or grades.jsonl allowed).",
                "Abort if HTTP error or invalid JSON schema rate exceeds 5%.",
            ],
        },
    }, "plan_sha256")


def main() -> int:
    parser = argparse.ArgumentParser(description="V3 Stage D3 Real LLM Analyst B/C Plan & Evaluator")
    parser.add_argument("--d1-plan", type=Path, default=DEFAULT_D1_PLAN)
    parser.add_argument("--d2-report", type=Path, default=DEFAULT_D2_REPORT)
    parser.add_argument("--out", type=Path, default=DEFAULT_D3_PLAN_OUT)
    parser.add_argument("--approved-plan", type=Path)
    parser.add_argument("--d0-freeze-file", type=Path)
    parser.add_argument("--d1-outcomes", type=Path, default=L2_ROOT / "reports/v3_d1_ladder_outcomes_20261017.json")
    parser.add_argument("--receipts-dir", type=Path, default=L2_ROOT / "validation/v3_d3_receipts")
    parser.add_argument("--results-out", type=Path, default=L2_ROOT / "reports/v3_d3_analysts_results_20261017.json")
    parser.add_argument("--development-baseline", type=Path)
    parser.add_argument("--analyst-model", type=str, default="gemini-3.8-flash")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--d0-confirmed", action="store_true")
    args = parser.parse_args()
    if args.plan_only and args.execute:
        parser.error("choose --plan-only or --execute")

    if args.execute and (not args.approved_plan or not args.d0_freeze_file):
        parser.error("execute requires --approved-plan and --d0-freeze-file")
    plan = read_json(args.approved_plan) if args.execute else build_d3_plan(
        d1_plan_path=args.d1_plan,
        d2_report_path=args.d2_report,
        analyst_model=args.analyst_model,
        baseline_path=args.development_baseline,
    )
    if not args.execute:
        write_json(args.out, plan)

    print(
        f"[V3-D3 PLAN] Wrote {args.out}\n"
        f"  d2_gate_open             = {plan['prerequisite_d2_gate']['d3_prerequisite_gate_open']}\n"
        f"  target_unit_count        = {plan['budget_and_stop_conditions']['target_unit_count']}\n"
        f"  analyst_llm_requests_cap = {plan['budget_and_stop_conditions']['analyst_llm_requests_cap']}\n"
        f"  pricing_status          = {plan['budget_and_stop_conditions']['pricing_status']}"
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
        try:
            execute_d3_plan(plan, freeze=read_json(args.d0_freeze_file), d1_plan=read_json(args.d1_plan), d2_report=read_json(args.d2_report), outcomes_path=args.d1_outcomes, receipts_dir=args.receipts_dir, results_out=args.results_out, api_key=os.environ["GEMINI_API_KEY"])
        except (ValueError, RuntimeError, OSError, KeyError) as exc:
            sys.stderr.write(f"BLOCKED/FAILED D3: {exc}\n")
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
