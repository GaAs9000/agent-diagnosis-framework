from c3_sandbox.integrated.delivery import build_relation_graph, compute_d8_metrics


def test_relation_graph_preserves_duplicates_and_unmatched():
    graph = build_relation_graph(
        tasks=[{"task_id": "t1"}],
        episodes=[{"episode_id": "e1", "task_id": "t1", "system_identity": "sys"}],
        tool_calls=[{"call_id": "c1", "episode_id": "e1", "before_snapshot_id": "s0", "after_snapshot_id": "s1"}],
        snapshots=[{"snapshot_id": "s0", "content_hash": "h0"}, {"snapshot_id": "s1", "content_hash": "h1"}],
        artifact_versions=[{"content_hash": "h1", "object_path": "out.docx"}],
        object_descriptors=[{"object_path": "out.docx"}],
        grades=[{"grade_id": "g1", "final_artifact_version_id": "h1", "rule_id": "r1"}],
        rules=[{"rule_id": "r1"}],
        observations=[{"observation_id": "o1", "tool_call_id": "c1"}],
        reference_pairs=[{"pair_id": "p1", "episode_ids": ["e1", "missing"]}],
    )
    assert len(graph.by_type("Snapshot")) == 2
    assert {r.relation_type for r in graph.relations} >= {"system_identity", "call_id", "before_snapshot", "after_snapshot", "artifact_version", "object_descriptor", "final_artifact_version", "rule", "observation", "reference_pair"}
    assert any(x["reason"] == "unresolved target" for x in graph.unmatched)

    duplicate = build_relation_graph(tasks=[{"task_id": "t"}, {"task_id": "t", "extra": 1}])
    assert len(duplicate.by_type("Task")) == 2
    assert duplicate.conflicts and any(r.relation_type == "conflict" for r in duplicate.relations)


def test_d8_metrics_group_by_branch_without_last_case_overwrite():
    result = compute_d8_metrics([
        {"branch_key": "b1", "forecast_id": "f1", "round": 1, "condition": "A", "probability": .8, "actual": True, "logical_cost": 2, "physical_cost": 3},
        {"branch_key": "b1", "forecast_id": "f2", "round": 2, "condition": "A", "probability": .2, "actual": False},
        {"branch_key": "b2", "forecast_id": "f3", "probability": "bad", "actual": True},
        {"branch_key": "b2", "forecast_id": "f4", "probability": .5},
    ])
    assert result["branch_count"] == 2 and result["row_count"] == 4
    assert result["branches"]["b1"]["scored"] == 2
    assert result["branches"]["b1"]["logical_cost"] == 2
    assert result["branches"]["b1"]["physical_cost"] == 3
    assert result["branches"]["b2"]["invalid"] == 1 and result["branches"]["b2"]["unknown"] == 1
    assert len(result["branches"]["b1"]["rows"]) == 2
