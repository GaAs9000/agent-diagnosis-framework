"""Conservative, deterministic D6/D7/D8 delivery layer.

Records are intentionally small and JSON compatible so callers can pass the
native R1 exports, R1.1 dispatch records, and official score records without
adapters.  Missing joins remain ``unmatched``; no alignment is inferred from
list order or fuzzy identifiers.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import math
from typing import Any, Iterable, Mapping

OWN_TEXT, ALL_TEXT, C3_STATE = "OWN_TEXT", "ALL_TEXT", "C3_STATE"
METHODS = (OWN_TEXT, ALL_TEXT, C3_STATE)


def _records(value: Any) -> list[dict]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        value = value.get("records", value.get("items", []))
    if not isinstance(value, list):
        raise TypeError("evidence input must be a list or {records: list}")
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _key(item: Mapping) -> tuple[str, str]:
    case = item.get("case_id", item.get("episode_id", item.get("task_id")))
    event = item.get("event_id", item.get("event_index", item.get("checkpoint_id", "")))
    return (str(case), str(event)) if case is not None else ("", str(event))


@dataclass(frozen=True)
class IndexedEvidence:
    key: tuple[str, str]
    snapshot: dict | None
    dispatch: dict | None
    score: dict | None
    status: str


@dataclass(frozen=True)
class EvidenceIndex:
    entries: tuple[IndexedEvidence, ...]
    unmatched: tuple[dict, ...]

    def by_case(self, case_id: str) -> tuple[IndexedEvidence, ...]:
        return tuple(item for item in self.entries if item.key[0] == str(case_id))

    def to_dict(self) -> dict:
        return {"entries": [asdict(item) for item in self.entries], "unmatched": list(self.unmatched)}


def build_evidence_index(*, snapshots: Any = None, dispatches: Any = None,
                         scores: Any = None) -> EvidenceIndex:
    """Join evidence only on an explicit case plus event/checkpoint key.

    Records lacking an explicit case key, or appearing in only one source,
    are retained as unmatched diagnostics rather than being aligned by order.
    """
    sources = {"snapshot": _records(snapshots), "dispatch": _records(dispatches), "score": _records(scores)}
    grouped: dict[tuple[str, str], dict[str, dict]] = {}
    unmatched: list[dict] = []
    for source, records in sources.items():
        for record in records:
            key = _key(record)
            if not key[0]:
                unmatched.append({"source": source, "record": record, "reason": "missing case_id"})
                continue
            grouped.setdefault(key, {})[source] = record
    entries = []
    for key in sorted(grouped):
        found = grouped[key]
        status = "matched" if len(found) == 3 else "unmatched"
        entries.append(IndexedEvidence(key, found.get("snapshot"), found.get("dispatch"), found.get("score"), status))
        if status == "unmatched":
            unmatched.append({"key": key, "sources": sorted(found), "reason": "incomplete explicit join"})
    return EvidenceIndex(tuple(entries), tuple(unmatched))


def extract_object_changes(dispatches: Any) -> list[dict]:
    """Extract only explicit filesystem deltas from dispatch evidence.

    A delta is an object-level record when the runtime supplied a concrete
    path and before/after metadata.  No list-order, filename similarity, or
    natural-language guess is used to align objects.
    """
    changes = []
    for dispatch in _records(dispatches):
        case_id = dispatch.get("case_id", dispatch.get("task_id"))
        event_id = dispatch.get("event_id", dispatch.get("tool_call_id", dispatch.get("event_index")))
        deltas = dispatch.get("fs_delta") or []
        if case_id is None or event_id is None or not isinstance(deltas, list):
            continue
        for delta in deltas:
            if not isinstance(delta, Mapping) or not isinstance(delta.get("path"), str):
                continue
            changes.append({
                "case_id": str(case_id), "event_id": str(event_id),
                "object_path": delta["path"], "before": delta.get("before"),
                "after": delta.get("after"),
                "evidence_identity": _identity({"dispatch": dispatch, "delta": delta}),
                "status": "explicit_change",
            })
    return changes


def _identity(record: Mapping) -> str:
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def seal_prediction(record: Mapping) -> dict:
    """Return an immutable JSON-compatible prediction envelope with digest."""
    body = dict(record)
    body.setdefault("schema_version", "d6-d8-prediction-v1")
    body["sealed"] = True
    body["prediction_identity"] = _identity({k: v for k, v in body.items() if k != "prediction_identity"})
    return body


class DeterministicPredictor:
    """Three fixed-budget, evidence-only methods; no model calls."""
    def __init__(self, budget: int = 1):
        if type(budget) is not int or budget < 0:
            raise ValueError("budget must be a nonnegative integer")
        self.budget = budget

    def predict(self, evidence: Mapping, *, case_id: str, candidate_universe: Iterable[str],
                method: str) -> dict:
        if method not in METHODS:
            raise ValueError("unknown method")
        universe = list(candidate_universe)
        if not universe or len(set(universe)) != len(universe):
            raise ValueError("candidate_universe must be unique and nonempty")
        # All methods receive the same bounded evidence count.  Their feature
        # views differ only in which saved fields are considered.
        fields = {OWN_TEXT: ("text", "own_text"), ALL_TEXT: ("text", "all_text", "public_evidence"), C3_STATE: ("state", "snapshot", "dispatch", "score")}[method]
        signal = sum(1 for field in fields if evidence.get(field) not in (None, "", [], {}))
        index = min(len(universe) - 1, signal + self.budget) % len(universe)
        probabilities = {candidate: (1.0 if i == index else 0.0) for i, candidate in enumerate(universe)}
        return seal_prediction({"case_id": case_id, "method_id": method, "budget": self.budget,
                                "candidate_universe": universe, "probabilities": probabilities,
                                "evidence_budget_used": min(self.budget, signal),
                                "pre_confirmation": True, "is_synthetic": bool(evidence.get("is_synthetic", False))})


def select_probe(bundle: Mapping, *, registry=None, line: str | None = None) -> dict:
    """Select one admissible fixed probe, or return an explicit unknown."""
    candidates = bundle.get("candidate_probes", bundle.get("probes", []))
    if not isinstance(candidates, list):
        candidates = []
    for probe in candidates:
        probe_id = probe if isinstance(probe, str) else probe.get("probe_id") if isinstance(probe, Mapping) else None
        if not probe_id:
            continue
        if registry is not None:
            try:
                spec = registry.get(probe_id)
            except (KeyError, ValueError):
                continue
            if line and line not in spec.get("lines", []):
                continue
            # When the bundle carries the normal R1 case/review contracts,
            # delegate the full closed-schema admissibility check to the
            # existing registry.  A bare probe list remains useful for
            # synthetic offline fixtures and is only vocabulary-checked.
            case_contract = bundle.get("case_contract")
            reviews = bundle.get("reviews", {})
            if case_contract is not None:
                review = reviews.get(probe_id) if isinstance(reviews, Mapping) else None
                try:
                    if not registry.validate_for_case(probe_id, case_contract, review).get("eligible"):
                        continue
                except (KeyError, TypeError, ValueError):
                    continue
        return {"status": "selected", "probe_id": probe_id, "line": line, "evidence_identity": _identity(bundle)}
    return {"status": "unknown", "probe_id": None, "line": line,
            "reason": "no admissible probe", "evidence_identity": _identity(bundle)}


def generate_report(predictions: Iterable[Mapping], actuals: Iterable[Mapping] = (), *, index: EvidenceIndex | None = None) -> dict:
    """Compare sealed predictions with available exploration/confirmation results."""
    actual_map = {str(item.get("case_id")): item for item in actuals if isinstance(item, Mapping) and item.get("case_id") is not None}
    rows, counts = [], {"invalid": 0, "unknown": 0, "matched": 0}
    for prediction in predictions:
        valid = (prediction.get("sealed") is True
                 and prediction.get("status") not in {"selector_invalid", "invalid"}
                 and prediction.get("prediction_identity") == _identity({k: v for k, v in prediction.items() if k != "prediction_identity"}))
        case_id = str(prediction.get("case_id", ""))
        actual = actual_map.get(case_id)
        if not valid:
            counts["invalid"] += 1
        elif actual is None:
            counts["unknown"] += 1
        else:
            counts["matched"] += 1
        rows.append({"case_id": case_id, "method_id": prediction.get("method_id"), "valid": valid,
                     "actual": actual, "status": "matched" if valid and actual is not None else "unknown"})
    logical = sum(float(item.get("logical_cost", 0) or 0) for item in actual_map.values())
    physical = sum(float(item.get("physical_cost", 0) or 0) for item in actual_map.values())
    natural = [item for item in actual_map.values() if isinstance(item.get("natural_success"), bool)]
    predicted_success = sum(1 for row in rows if row["valid"] and row["actual"] and row["actual"].get("natural_success") is True)
    gap = None if not natural else predicted_success / max(1, len(rows)) - sum(item["natural_success"] for item in natural) / len(natural)
    return {"schema_version": "d6-d8-report-v1", "rows": rows, "counts": counts,
            "logical_cost": logical, "physical_cost": physical,
            "natural_success_gap": gap, "natural_success_gap_status": "reported" if gap is not None else "unknown",
            "scientific_efficacy_claim": False,
            "index_status": None if index is None else {"entries": len(index.entries), "unmatched": len(index.unmatched)}}

# --- Closeout relation graph and branch-level D8 accounting -----------------
@dataclass(frozen=True)
class GraphNode:
    """A typed evidence node.  ``node_id`` is unique even for conflicts."""
    node_type: str
    node_id: str
    source: str
    data: dict


@dataclass(frozen=True)
class GraphRelation:
    relation_type: str
    source: str
    target: str
    data: dict


@dataclass(frozen=True)
class RelationGraph:
    nodes: tuple[GraphNode, ...]
    relations: tuple[GraphRelation, ...]
    unmatched: tuple[dict, ...]
    conflicts: tuple[dict, ...]

    def by_type(self, node_type: str) -> tuple[GraphNode, ...]:
        return tuple(n for n in self.nodes if n.node_type == node_type)

    def relations_of(self, relation_type: str) -> tuple[GraphRelation, ...]:
        return tuple(r for r in self.relations if r.relation_type == relation_type)

    def to_dict(self) -> dict:
        return {"schema_version": "typed-relation-graph-v1",
                "nodes": [asdict(n) for n in self.nodes],
                "relations": [asdict(r) for r in self.relations],
                "unmatched": list(self.unmatched), "conflicts": list(self.conflicts)}


_NODE_SPECS = {
    "tasks": ("Task", ("task_id", "id")),
    "episodes": ("Episode", ("episode_id", "case_id", "id")),
    "tool_calls": ("ToolCall", ("call_id", "tool_call_id", "event_id", "id")),
    "observations": ("Observation", ("observation_id", "observation_key", "id")),
    "snapshots": ("Snapshot", ("snapshot_id", "checkpoint_id", "id")),
    "artifact_versions": ("ArtifactVersion", ("artifact_version_id", "version_id", "content_hash", "sha256", "id")),
    "object_descriptors": ("ObjectDescriptor", ("object_id", "object_path", "path", "id")),
    "grades": ("Grade", ("grade_id", "score_id", "id")),
    "rules": ("Rule", ("rule_id", "rubric_id", "id")),
    "reference_pairs": ("ReferencePair", ("pair_id", "reference_pair_id", "id")),
    "same_task_episodes": ("SameTaskEpisodes", ("same_task_episodes_id", "group_id", "id")),
}


def _explicit_id(record: Mapping, candidates: tuple[str, ...]) -> str | None:
    for field in candidates:
        value = record.get(field)
        if value not in (None, ""):
            return str(value)
    return None


def _as_node_id(node_type: str, logical_id: str, ordinal: int = 0) -> str:
    base = f"{node_type}:{logical_id}"
    return base if ordinal == 0 else f"{base}#duplicate-{ordinal}"


def build_relation_graph(*, tasks: Any = None, episodes: Any = None,
                         tool_calls: Any = None, observations: Any = None,
                         snapshots: Any = None, artifact_versions: Any = None,
                         artifacts: Any = None, object_descriptors: Any = None,
                         grades: Any = None, rules: Any = None,
                         reference_pairs: Any = None, same_task_episodes: Any = None,
                         evidence: Mapping | None = None) -> RelationGraph:
    """Build an explicit typed evidence graph without positional joins.

    Every input row is retained.  Repeated ``(source, explicit key)`` rows are
    represented by distinct nodes and a ``conflict`` relation; no row is
    overwritten.  Rows without an explicit key are retained in ``unmatched``.
    """
    if evidence is not None:
        if not isinstance(evidence, Mapping):
            raise TypeError("evidence must be a mapping")
        # Accept either plural source names or their common singular aliases.
        values = dict(evidence)
        for name in _NODE_SPECS:
            if name not in values:
                singular = name[:-1] if name.endswith("s") else name
                values[name] = values.get(singular)
        tasks, episodes = values.get("tasks"), values.get("episodes")
        tool_calls, observations = values.get("tool_calls"), values.get("observations")
        snapshots = values.get("snapshots"); artifact_versions = values.get("artifact_versions") or values.get("artifacts")
        object_descriptors = values.get("object_descriptors"); grades, rules = values.get("grades"), values.get("rules")
        reference_pairs = values.get("reference_pairs")
        same_task_episodes = values.get("same_task_episodes")
    if artifact_versions is None:
        artifact_versions = artifacts
    if object_descriptors is None:
        object_descriptors = []
    raw = {"tasks": tasks, "episodes": episodes, "tool_calls": tool_calls,
           "observations": observations, "snapshots": snapshots,
           "artifact_versions": artifact_versions, "object_descriptors": object_descriptors,
           "grades": grades, "rules": rules, "reference_pairs": reference_pairs,
           "same_task_episodes": same_task_episodes}
    nodes: list[GraphNode] = []
    unmatched: list[dict] = []
    conflicts: list[dict] = []
    lookup: dict[tuple[str, str], list[str]] = {}
    for source, (node_type, candidates) in _NODE_SPECS.items():
        records = _records(raw[source])
        for ordinal, record in enumerate(records):
            logical_id = _explicit_id(record, candidates)
            if logical_id is None:
                logical_id = _identity(record)[:16]
                unmatched.append({"source": source, "record": record, "reason": "missing explicit key"})
            prior = lookup.setdefault((node_type, logical_id), [])
            node_id = _as_node_id(node_type, logical_id, len(prior))
            prior.append(node_id)
            node = GraphNode(node_type, node_id, source, record)
            nodes.append(node)
            if len(prior) > 1:
                conflict = {"source": source, "key": logical_id, "nodes": [prior[0], node_id], "reason": "duplicate explicit key"}
                conflicts.append(conflict)
                # Conflict is itself an explicit relation, preserving both rows.
                nodes_idx = [n for n in nodes if n.node_id in (prior[0], node_id)]
                _ = nodes_idx
    relations: list[GraphRelation] = []
    by_type: dict[str, list[GraphNode]] = {}
    for node in nodes:
        by_type.setdefault(node.node_type, []).append(node)
    def connect(rel: str, source_node: GraphNode, target_type: str, target_id: Any, data: Mapping | None = None):
        if target_id in (None, ""):
            unmatched.append({"source": source_node.source, "node_id": source_node.node_id, "relation": rel, "reason": "missing target key"})
            return
        targets = [n for n in by_type.get(target_type, []) if n.data.get("id") == target_id or any(str(n.data.get(k)) == str(target_id) for k in _NODE_SPECS.get(next((s for s,(t,_) in _NODE_SPECS.items() if t == target_type), ""), (target_type, ("id",)))[1])]
        if not targets:
            unmatched.append({"source": source_node.node_id, "relation": rel, "target": str(target_id), "reason": "unresolved target"})
            return
        for target in targets:
            relations.append(GraphRelation(rel, source_node.node_id, target.node_id, dict(data or {})))
    def value(record, *keys):
        for key in keys:
            if record.get(key) not in (None, ""):
                return record[key]
        return None
    # Explicit foreign-key edges.  A row can carry more than one edge.
    for n in nodes:
        d = n.data
        if n.node_type == "Episode":
            # Episode -> Task is retained for backward traversal; the canonical
            # closeout direction Task -> Episode is added below as well.
            connect("system_identity", n, "Task", value(d, "task_id", "task"), {"system_identity": value(d, "system_identity")})
        elif n.node_type == "ToolCall":
            connect("call_id", n, "Episode", value(d, "episode_id", "case_id"))
            connect("observation", n, "Observation", value(d, "observation_id", "observation_id"))
            connect("before_snapshot", n, "Snapshot", value(d, "before_snapshot_id", "before_snapshot", "before_checkpoint_id"))
            connect("after_snapshot", n, "Snapshot", value(d, "after_snapshot_id", "after_snapshot", "after_checkpoint_id"))
        elif n.node_type == "Observation":
            connect("observation", n, "ToolCall", value(d, "tool_call_id", "call_id"))
        elif n.node_type == "Snapshot":
            connect("artifact_version", n, "ArtifactVersion", value(d, "artifact_version_id", "version_id", "content_hash", "sha256"))
        elif n.node_type == "ArtifactVersion":
            connect("object_descriptor", n, "ObjectDescriptor", value(d, "object_id", "object_path", "path"))
        elif n.node_type == "Grade":
            connect("final_artifact_version", n, "ArtifactVersion", value(d, "final_artifact_version_id", "artifact_version_id", "content_hash", "sha256"))
            rule_ids = d.get("rule_ids", d.get("rules", d.get("rule_id")))
            if not isinstance(rule_ids, list): rule_ids = [rule_ids]
            for rid in rule_ids:
                connect("rule", n, "Rule", rid)
        elif n.node_type == "ReferencePair":
            ep_ids = d.get("episode_ids", d.get("episodes", []))
            if isinstance(ep_ids, (str, int)): ep_ids = [ep_ids]
            for eid in ep_ids or []:
                connect("reference_pair", n, "Episode", eid)
    # Canonical closeout directions.  These are explicit foreign-key joins,
    # never list-order joins; reverse links above keep older consumers working.
    for task in by_type.get("Task", []):
        for episode in by_type.get("Episode", []):
            episode_task = episode.data.get("task_id", episode.data.get("task", ""))
            task_id = _explicit_id(task.data, _NODE_SPECS["tasks"][1])
            if str(episode_task) == str(task_id):
                relations.append(GraphRelation("system_identity", task.node_id, episode.node_id, {"system_identity": episode.data.get("system_identity")}))
    for episode in by_type.get("Episode", []):
        for call in by_type.get("ToolCall", []):
            call_episode = call.data.get("episode_id", call.data.get("case_id", ""))
            episode_id = _explicit_id(episode.data, _NODE_SPECS["episodes"][1])
            if str(call_episode) == str(episode_id):
                relations.append(GraphRelation("call_id", episode.node_id, call.node_id, {"call_id": _explicit_id(call.data, _NODE_SPECS["tool_calls"][1])}))
    for call in by_type.get("ToolCall", []):
        for obs in by_type.get("Observation", []):
            if str(obs.data.get("tool_call_id", obs.data.get("call_id", ""))) == str(_explicit_id(call.data, _NODE_SPECS["tool_calls"][1])):
                relations.append(GraphRelation("observation", call.node_id, obs.node_id, {}))
    for group in by_type.get("SameTaskEpisodes", []):
        pair_id = value(group.data, "pair_id", "reference_pair_id")
        if pair_id is not None:
            connect("reference_pair", group, "ReferencePair", pair_id)
    for pair in by_type.get("ReferencePair", []):
        group_id = value(pair.data, "same_task_episodes_id", "group_id")
        if group_id is not None:
            connect("same_task_episodes", pair, "SameTaskEpisodes", group_id)
    for conflict in conflicts:
        relations.append(GraphRelation("conflict", conflict["nodes"][0], conflict["nodes"][1], {"source": conflict["source"], "key": conflict["key"]}))
    return RelationGraph(tuple(nodes), tuple(relations), tuple(unmatched), tuple(conflicts))

# Names used by early closeout drafts; retaining aliases makes the API additive.
build_typed_relation_graph = build_relation_graph
build_evidence_graph = build_relation_graph


def _branch_key(row: Mapping) -> str:
    return str(row.get("branch_key", row.get("case_id", row.get("episode_id", ""))))


def _probability(row: Mapping) -> tuple[float | None, Any]:
    actual = row.get("actual", row.get("outcome", row.get("label", row.get("y"))))
    p = row.get("probability", row.get("forecast", row.get("prediction", row.get("probabilities"))))
    if isinstance(p, Mapping):
        if actual in p: p = p[actual]
        elif "yes" in p and isinstance(actual, bool): p = p["yes"] if actual else 1 - float(p["yes"])
        else: return None, actual
    try:
        return (float(p), actual) if p is not None else (None, actual)
    except (TypeError, ValueError):
        return None, actual


def compute_d8_metrics(records: Any, *, clip: float = 1e-15) -> dict:
    """Compute D8 metrics grouped by branch, retaining every forecast row.

    A missing/invalid forecast is counted explicitly and never replaces another
    row with the same branch.  ``round``, ``condition``, and ``forecast_id``
    are carried into each branch's row ledger for auditability.
    """
    if clip <= 0 or clip >= .5: raise ValueError("clip must be between 0 and .5")
    rows = _records(records)
    branches: dict[str, dict] = {}
    for ordinal, row in enumerate(rows):
        branch = _branch_key(row) or f"unmatched-{ordinal}"
        bucket = branches.setdefault(branch, {"branch_key": branch, "rows": [], "brier": [], "log_loss": [], "logical_cost": 0.0, "physical_cost": 0.0, "invalid": 0, "unknown": 0})
        p, actual = _probability(row)
        valid_actual = isinstance(actual, bool) or isinstance(actual, (int, float)) and actual in (0, 1)
        item = {"forecast_id": row.get("forecast_id", row.get("prediction_id")), "round": row.get("round"), "condition": row.get("condition"), "actual": actual, "probability": p}
        if p is None or not math.isfinite(p) or not 0 <= p <= 1:
            bucket["invalid"] += 1; item["status"] = "invalid"
        elif not valid_actual:
            bucket["unknown"] += 1; item["status"] = "unknown"
        else:
            y = 1.0 if bool(actual) else 0.0
            q = min(1 - clip, max(clip, p))
            bucket["brier"].append((p - y) ** 2); bucket["log_loss"].append(-(y * math.log(q) + (1 - y) * math.log(1 - q))); item["status"] = "scored"
        bucket["logical_cost"] += float(row.get("logical_cost", 0) or 0)
        bucket["physical_cost"] += float(row.get("physical_cost", 0) or 0)
        bucket["rows"].append(item)
    output = {}
    for branch, bucket in branches.items():
        brier = bucket.pop("brier"); logs = bucket.pop("log_loss")
        bucket["brier"] = sum(brier) / len(brier) if brier else None
        bucket["clipped_log_loss"] = sum(logs) / len(logs) if logs else None
        # Short aliases are kept for consumers that call this metric LogLoss.
        bucket["log_loss"] = bucket["clipped_log_loss"]
        bucket["brier_score"] = bucket["brier"]
        bucket["scored"] = len(brier)
        output[branch] = bucket
    return {"schema_version": "d8-branch-metrics-v1", "branches": output, "branch_count": len(output), "row_count": len(rows), "invalid": sum(v["invalid"] for v in output.values()), "unknown": sum(v["unknown"] for v in output.values())}

compute_branch_metrics = compute_d8_metrics
calculate_d8_metrics = compute_d8_metrics
