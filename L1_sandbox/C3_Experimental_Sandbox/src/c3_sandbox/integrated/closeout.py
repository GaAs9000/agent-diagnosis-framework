"""Single-pass post-original closeout pipeline.

The pipeline consumes original episode receipts, builds typed D6 evidence,
submits real D7 diagnostic requests for eligible cases, and records D5 branch
work as explicit NOT_RUN when no qualified checkpoint/branch executor exists.
It never substitutes fixture predictions or original scores for probe outcomes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from .d7_backend import ALL_TEXT, C3_STATE, OWN_TEXT, DiagnosticBackend, select_condition
from .delivery import build_relation_graph, compute_d8_metrics
from .experiment_card import CandidateExperiment, CompetingExplanation, ExperimentCard, ExperimentCardCompiler, outcome_not_run, outcome_selector_invalid


def _write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def _case_evidence(rows: list[dict]) -> list[dict]:
    cases = []
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        task = str(row.get("task_id", "")); grouped.setdefault(task, []).append(dict(row))
    for row in rows:
        task = str(row.get("task_id", "")); system = str(row.get("model_key", row.get("system_id", "unknown")))
        result = row.get("result", row)
        case_id = f"{system}:{task}"
        refs = []
        for other in grouped.get(task, []):
            if str(other.get("model_key", other.get("system_id", "unknown"))) != system:
                refs.append({"case_id": f"{other.get('model_key', other.get('system_id', 'unknown'))}:{task}",
                             "episode": other.get("result", other)})
        cases.append({"case_id": case_id, "task_id": task, "system_id": system,
                      "own_text": {"events": result.get("agent_result", {}).get("events", []),
                                   "dispatches": result.get("tool_dispatch", [])},
                      "reference_episodes": refs,
                      "state": {"checkpoints": result.get("checkpoints", []),
                                "dispatches": result.get("tool_dispatch", [])},
                      "score": result.get("final_grade", {}),
                      "checkpoint_identity": (result.get("checkpoints") or [{}])[0].get("checkpoint_id"),
                      "result": result})
    return cases


def run_closeout(rows: list[dict], config: Mapping[str, Any], output_dir: str | Path,
                 *, backend_factory: Callable[..., Any] | None = None,
                 branch_runner: Callable[[dict, str, dict], Mapping] | None = None,
                 gateway: Any | None = None) -> dict:
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    cases = _case_evidence(rows)
    graph = build_relation_graph(
        tasks=[{"task_id": c["task_id"]} for c in cases],
        episodes=[{"episode_id": c["case_id"], "task_id": c["task_id"], "system_identity": c["system_id"]} for c in cases],
        tool_calls=[{**call, "case_id": c["case_id"], "episode_id": c["case_id"], "call_id": call.get("tool_call_id", call.get("event_id"))}
                    for c in cases for call in c["result"].get("tool_dispatch", [])],
        observations=[], snapshots=[], artifact_versions=[], object_descriptors=[], grades=[], rules=[], reference_pairs=[])
    (root / "objects" / "relation_graph.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "objects" / "relation_graph.json").write_text(json.dumps(graph.to_dict(), ensure_ascii=False, indent=2) + "\n")

    predictions, receipts, branches, cards = [], [], [], []
    diagnostic = config.get("diagnostic", {})
    for case in cases:
        result = case["result"]
        card = ExperimentCard.create(
            phenomenon=f"{case['task_id']} execution requires explanation",
            evidence_references=[case["case_id"]],
            competing_explanations=[
                CompetingExplanation("information", "public information presentation affected the behavior", None),
                CompetingExplanation("operation", "the operation or object choice affected the behavior", None),
            ],
            candidate_experiments=[
                CandidateExperiment(f"{case['case_id']}:visibility", {"mode": "expanded"}, "visibility.expanded",
                                    {"checkpoint_id": case.get("checkpoint_identity") or "unavailable"},
                                    {"operator": "visibility.expanded.matched"}, {"task_completed": True},
                                    {"max_model_calls": 2}, {"line": "E2-F"}),
                CandidateExperiment(f"{case['case_id']}:feedback", {"mode": "structured"}, "feedback.structured",
                                    {"checkpoint_id": case.get("checkpoint_identity") or "unavailable"},
                                    {"operator": "feedback.structured.matched"}, {"task_completed": True},
                                    {"max_model_calls": 2}, {"line": "E2-F"}),
            ])
        if result.get("execution_valid") is not True:
            cards.append({"card": {"phenomenon": card.phenomenon, "card_id": card.card_id, "version": card.version,
                                   "status": card.status.value, "evidence_references": [case["case_id"]]},
                          "outcome": outcome_not_run(reason="original execution invalid")})
            branches.append({"case_id": case["case_id"], "status": "NOT_RUN", "reason": "original execution invalid"})
            continue
        try:
            compiled = ExperimentCardCompiler().compile(card, evidence={case["case_id"]: case},
                                                        system_identity=case["system_id"],
                                                        runtime_identity=case.get("result", {}).get("runtime", {}),
                                                        task_identity=case["task_id"], line="E2-F")
            compiled_card = card._version(status="compiled")
            cards.append({"card": {"phenomenon": compiled_card.phenomenon, "card_id": compiled_card.card_id,
                                    "parent_card_id": compiled_card.parent_card_id, "version": compiled_card.version,
                                    "status": compiled_card.status.value, "evidence_references": [case["case_id"]]},
                          "compiled_experiments": list(compiled)})
        except Exception as exc:
            cards.append({"card": {"phenomenon": card.phenomenon, "card_id": card.card_id, "version": card.version,
                                   "status": "review_required", "evidence_references": [case["case_id"]]},
                          "outcome": outcome_selector_invalid(reason=str(exc))})
            branches.append({"case_id": case["case_id"], "status": "NOT_RUN", "reason": "experiment card compile failed"})
            continue
        endpoint = diagnostic.get("endpoint") or config.get("services", {}).get("diagnostic_endpoint")
        model = diagnostic.get("model", "default_model")
        persist_dir = root / "raw_diagnostic" / case["case_id"].replace(":", "__")
        try:
            if backend_factory is not None:
                backend = backend_factory(model=model, endpoint=endpoint, persist_dir=persist_dir)
            elif gateway is not None and getattr(gateway, "diagnostician", None) is not None:
                backend = gateway.create_diagnostic_backend(persist_dir=persist_dir)
            elif config.get("models", {}).get("diagnostician"):
                from ..gateway.registry import BackendRegistry
                from ..gateway.diagnostician import GatewayDiagnosticBackend
                diag_model = BackendRegistry.create(config["models"]["diagnostician"])
                backend = GatewayDiagnosticBackend(diag_model, persist_dir=persist_dir)
            else:
                backend = DiagnosticBackend(model=model, endpoint=endpoint, persist_dir=persist_dir)
        except Exception as exc:
            branches.append({"case_id": case["case_id"], "status": "NOT_RUN", "reason": "diagnostic backend unavailable: " + str(exc)})
            continue
        candidates = ["original", "visibility.expanded", "feedback.structured"]
        for method in (OWN_TEXT, ALL_TEXT, C3_STATE):
            prediction = backend.predict(case, case_id=case["case_id"], candidate_universe=candidates,
                                         method=method, evidence_ids=[case["case_id"]],
                                         checkpoint_identity=case.get("checkpoint_identity"))
            predictions.append(prediction)
        selector = select_condition(predictions[-3:], policy=diagnostic.get("selection_policy", {"min_probability": .5}))
        receipts.append({"case_id": case["case_id"], "status": "NOT_RUN" if selector.get("status") != "selected" else "PENDING_D5",
                         "selector": selector, "logical_cost": 0, "physical_cost": 0})
        if branch_runner is None:
            branches.append({"case_id": case["case_id"], "status": "NOT_RUN", "reason": "no qualified D5 branch executor supplied"})
        else:
            for condition in candidates:
                try:
                    branch = dict(branch_runner(case, condition, selector)); branch.setdefault("case_id", case["case_id"]); branches.append(branch)
                except Exception as exc:
                    branches.append({"case_id": case["case_id"], "condition": condition, "status": "failed", "error": str(exc)})
    _write_jsonl(root / "predictions.jsonl", predictions)
    _write_jsonl(root / "experiment_cards.jsonl", cards)
    _write_jsonl(root / "query_receipts.jsonl", receipts)
    _write_jsonl(root / "branch_results.jsonl", branches)
    metric_rows = []
    for row in branches:
        if not isinstance(row, Mapping):
            continue
        metric_rows.append({**row, "branch_key": row.get("branch_key", row.get("case_id")),
                            "probability": row.get("probability", row.get("forecast_probability")),
                            "actual": row.get("actual", row.get("outcome"))})
    metrics = compute_d8_metrics(metric_rows)
    (root / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    mapping_rows = [{"case_id": p.get("case_id"), "forecast_id": p.get("prediction_id"),
                     "branch_key": p.get("branch_key"), "status": p.get("status")}
                    for p in predictions]
    mapping = {"schema_version": "forecast-branch-map-v1", "rows": mapping_rows, "branch_results": len(branches),
               "predictions": len(predictions), "unknown_outcomes": len(branches)}
    (root / "forecast_result_map.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n")
    return {"cases": len(cases), "cards": len(cards), "predictions": len(predictions), "receipts": len(receipts),
            "branches": len(branches), "graph": graph.to_dict(), "metrics": metrics}
