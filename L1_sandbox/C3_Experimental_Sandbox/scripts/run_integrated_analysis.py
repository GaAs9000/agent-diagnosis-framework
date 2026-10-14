#!/usr/bin/env python3
"""Produce the bounded D6/D7/D8 report from saved development evidence.

This command does not run a model, sandbox, scorer, or branch. It consumes
immutable prior development records and the current run's explicit target
coverage records; missing target or probe outcomes remain unknown.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from c3_sandbox.integrated import ALL_TEXT, C3_STATE, OWN_TEXT, build_evidence_index, extract_object_changes, generate_report, select_probe, seal_prediction
from c3_sandbox.integrated.d7_backend import DiagnosticBackend, select_condition
from c3_sandbox.integrated.plan import build_plan
from c3_sandbox.probe_registry import ProbeRegistry

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/integrated-delivery-001"
TASK_IDS = ("officeval_002", "officeval_003", "officeval_006", "officeval_012",
            "officeval_015", "officeval_018", "officeval_033", "officeval_037")


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256(); h.update(Path(path).read_bytes()); return h.hexdigest()


def saved_cases():
    rows = []
    sources = [
        ("reference_qwen3-4b", RUN.parent / "e2-r1/originals", ("officeval_002", "officeval_003")),
        ("reference_qwen3-4b", RUN.parent / "e2-r1-1/originals", ("officeval_012", "officeval_015", "officeval_018", "officeval_033", "officeval_037")),
    ]
    for system_id, base, task_ids in sources:
        for task_id in task_ids:
            path = base / task_id / "result.json"
            if not path.is_file():
                continue
            result = load(path)
            case_id = f"{system_id}:{task_id}"
            dispatches = []
            for index, dispatch in enumerate(result.get("tool_dispatch", [])):
                dispatch = dict(dispatch)
                dispatch.update(case_id=case_id, task_id=task_id,
                                event_id=dispatch.get("tool_call_id", f"dispatch-{index}"),
                                source_result=str(path))
                dispatches.append(dispatch)
            grade = result.get("final_grade") or {}
            score = {"case_id": case_id, "task_id": task_id, "event_id": "final",
                     "outcome": result.get("outcome"), "execution_valid": result.get("execution_valid"),
                     "task_completed": result.get("task_completed"),
                     "official": grade.get("official_raw"), "source_result": str(path)}
            rows.append({"case_id": case_id, "task_id": task_id, "system_id": system_id,
                         "result": result, "dispatches": dispatches, "score": score,
                         "source_result": str(path)})
    legacy = RUN.parent / "e2-legacy-import-001/ep_officeval_006_qwen3-4b_s2026.json"
    if legacy.is_file():
        result = load(legacy)
        case_id = "reference_qwen3-4b:officeval_006"
        dispatches = []
        for event in result.get("events", []):
            if event.get("event_type") not in {"agent_action", "tool_result"}:
                continue
            dispatches.append({"case_id": case_id, "task_id": "officeval_006",
                               "event_id": event.get("event_id"), "turn": event.get("turn_id"),
                               "function": event.get("tool_name"), "arguments": event.get("arguments"),
                               "raw_result": {"exit_code": event.get("exit_code"),
                                              "stdout": event.get("stdout", ""),
                                              "stderr": event.get("stderr", "")},
                               "fs_delta": [], "source_result": str(legacy)})
        score = {"case_id": case_id, "task_id": "officeval_006", "event_id": "final",
                 "outcome": None, "execution_valid": None, "task_completed": None,
                 "official": None, "source_result": str(legacy)}
        rows.append({"case_id": case_id, "task_id": "officeval_006", "system_id": system_id,
                     "result": result, "dispatches": dispatches, "score": score,
                     "source_result": str(legacy)})
    return rows


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-endpoint", default=None)
    parser.add_argument("--diagnostic-model", default="default_model")
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    cases = saved_cases()
    dispatches = [d for case in cases for d in case["dispatches"]]
    # Snapshot rows are intentionally explicit and use dispatch IDs. Endpoint
    # scores use a distinct "final" event, so the index reports that mismatch.
    snapshots = [{"case_id": d["case_id"], "event_id": d["event_id"],
                  "before": d.get("before_path"), "after": d.get("after_path"),
                  "fs_delta": d.get("fs_delta")} for d in dispatches]
    scores = [case["score"] for case in cases]
    index = build_evidence_index(snapshots=snapshots, dispatches=dispatches, scores=scores)
    changes = extract_object_changes(dispatches)

    registry = ProbeRegistry.from_file(ROOT / "configs/probes.v1.json")
    predictions = []
    methods = (OWN_TEXT, ALL_TEXT, C3_STATE)
    backend = (DiagnosticBackend(model=args.diagnostic_model, endpoint=args.diagnostic_endpoint,
                                 persist_dir=RUN / "raw_diagnostic")
               if args.diagnostic_endpoint else None)
    for case in cases:
        result = case["result"]
        state = {"object_changes": [x for x in changes if x["case_id"] == case["case_id"]],
                 "matched_evidence": len(index.by_case(case["case_id"]))}
        reference_episodes = [other["result"] for other in cases
                              if other["task_id"] == case["task_id"] and other["case_id"] != case["case_id"]]
        evidence = {"text": {"events": len(result.get("agent_result", {}).get("events", [])),
                              "protocol": result.get("agent_result", {}).get("protocol_status")},
                    "reference_episodes": reference_episodes,
                    "public_evidence": {"task_id": case["task_id"]},
                    "state": state, "snapshot": state, "dispatch": case["dispatches"],
                    "score": case["score"], "is_synthetic": False}
        line = "E2-S" if result.get("task_completed") is True else "E2-F"
        candidates = (["visibility.compact", "feedback.aggregate"] if line == "E2-S"
                      else ["visibility.expanded", "feedback.structured"])
        selected = select_probe({"candidate_probes": candidates, "case_contract": None}, registry=registry, line=line)
        for method in methods:
            if backend is None:
                prediction = {"case_id": case["case_id"], "method": method,
                              "candidate_universe": ["original", *candidates],
                              "probabilities": {}, "status": "selector_invalid",
                              "model": None, "request_raw": "", "response_raw": "",
                              "usage": None, "evidence_ids": [case["case_id"]],
                              "checkpoint_identity": case.get("checkpoint_identity"),
                              "error": "diagnostic endpoint not configured"}
            else:
                prediction = backend.predict(evidence, case_id=case["case_id"],
                                             candidate_universe=["original", *candidates], method=method,
                                             evidence_ids=[case["case_id"]],
                                             checkpoint_identity=case.get("checkpoint_identity"))
            prediction.pop("prediction_identity", None)
            prediction["selected_probe"] = selected
            prediction["task_id"] = case["task_id"]
            prediction["system_id"] = case["system_id"]
            predictions.append(seal_prediction(prediction))

    selector = select_condition(predictions, policy={"min_probability": .5}) if predictions else {"status": "selector_invalid"}
    report = generate_report(predictions, actuals=[], index=index)
    target_rows = load(RUN / "originals/target_results.json") if (RUN / "originals/target_results.json").is_file() else []
    target_calls = sum(row.get("new_model_calls", 0) for row in target_rows)
    valid_target = sum(1 for row in target_rows if row.get("execution_valid") is True)
    plan = build_plan(source_paths=[ROOT / "scripts/run_integrated_analysis.py"],
                      asset_paths=[RUN / "bundle/manifest.json"],
                      verifier_paths=[RUN / "bundle/manifest.json"], existing_runs=cases)
    final = {
        "schema_version": "c3-e2-integrated-report-v1",
        "software_delivery": {"status": "implemented", "protocol_recovery": "bounded_two_recoveries",
                               "d6": "implemented", "d7": "implemented", "d8": "implemented"},
        "runtime_coverage": {"target_model": "Qwen3-Coder-30B-A3B", "target_officeval_rows": len(target_rows),
                              "target_valid_officeval_rows": valid_target, "target_new_model_calls": target_calls,
                              "target_matrix_setup_failures": sum(1 for row in target_rows if row.get("status") == "infrastructure_invalid"),
                              "reference_saved_cases": len(cases), "new_probe_branches": 0},
        "natural_failure_side": {"status": "observed" if cases else "unknown", "cases": len(cases),
                                 "source_scope": "prior development evidence; not a new confirmation set"},
        "natural_success_side": {"status": "no_data", "complete_success_cases": sum(1 for c in cases if c["result"].get("task_completed") is True),
                                 "claim": "not substituted by synthetic tasks"},
        "d6_object_state": {"explicit_change_records": len(changes), "indexed_entries": len(index.entries),
                            "unmatched_evidence": len(index.unmatched), "alignment_policy": "explicit keys only"},
        "d7_method_comparison": {"methods": list(methods), "prediction_records": len(predictions),
                                 "valid_predictions": sum(1 for p in predictions if p.get("status") == "ok"),
                                 "diagnostic_backend": "configured" if backend else "not_configured",
                                 "selector_status": selector.get("status"),
                                 "equal_budget": True, "probe_results": "unknown_no_new_branches",
                                 "scientific_efficacy_claim": False},
        "d8_report": report,
        "costs": {"new_logical_probe_queries": 0, "new_physical_probe_executions": 0,
                  "saved_reference_model_calls": sum(c["result"].get("agent_result", {}).get("modelcall_count", 0) for c in cases),
                  "target_model_calls": target_calls, "gpu_seconds": None, "monetary_cost": None},
        "plan": plan.to_dict(), "limitations": [
            "No valid target-model OfficeVal trajectory was produced because the bounded matrix stopped after three infrastructure revisions.",
            "No exploration or confirmation probe branch was executed; prediction outcomes remain unknown.",
            "Reference evidence is prior development material and is not a held-out research comparison.",
            "Object alignment is limited to explicit filesystem delta keys and does not infer semantic identity.",
            "Scientific efficacy and C3-vs-baseline claims remain unestablished."],
        "evidence": {"saved_cases": [c["source_result"] for c in cases],
                     "target_results": str(RUN / "originals/target_results.json"),
                     "bundle_manifest": str(RUN / "bundle/manifest.json")},
    }
    (RUN / "integrated_report.json").write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n")
    (RUN / "d6_object_changes.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2) + "\n")
    (RUN / "d7_predictions.jsonl").write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in predictions) + "\n")
    (RUN / "d6_evidence_index.json").write_text(json.dumps(index.to_dict(), ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"cases": len(cases), "object_changes": len(changes), "predictions": len(predictions),
                      "target_rows": len(target_rows), "probe_results": "unknown_no_new_branches",
                      "diagnostic_backend": "configured" if backend else "not_configured",
                      "selector_status": selector.get("status"),
                      "scientific_efficacy_claim": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
