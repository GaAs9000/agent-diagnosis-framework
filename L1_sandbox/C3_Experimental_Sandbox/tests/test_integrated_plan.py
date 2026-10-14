from c3_sandbox.integrated.plan import (
    MAX_ORIGINAL_EPISODES, TASK_IDS, analyze_saved_runs, build_plan, serial_execute,
)


def test_plan_freezes_fixed_order_and_explicit_unknown_state():
    plan = build_plan(task_ids=("officeval_037", "officeval_002", "unreviewed"))
    assert plan.task_ids == ("officeval_002", "officeval_037")
    assert plan.excluded_tasks == ("unreviewed",)
    analysis = analyze_saved_runs(plan, [])
    assert analysis["data_state"] == "no_data"


def test_serial_budget_is_sixteen_and_does_not_replace_tasks():
    plan = build_plan()
    rows = serial_execute(plan, lambda *_: {"status": "unused"}, dry_run=True)
    assert len(rows) == MAX_ORIGINAL_EPISODES
    assert [row["task_id"] for row in rows[::2]] == list(TASK_IDS)
    assert all(row["status"] == "planned" for row in rows)
