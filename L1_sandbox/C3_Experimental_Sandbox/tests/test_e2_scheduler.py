from dataclasses import asdict, replace
import multiprocessing
import time

import pytest

from c3_sandbox.experiment_scheduler import (
    BranchSpec, ExecutionOutcome, ExperimentScheduler, MethodClient, ProtocolError,
)

H = "a" * 64
COST = dict(input_tokens=0, output_tokens=0, gpu_seconds=None,
            environment_seconds=0.01, wall_seconds=0.01)


def specs():
    base = BranchSpec("case", H, "E2-S", H, "checkpoint", H, "original", H,
                      {"seed": 7, "temperature": 0}, 0, H, H, True, "exploration")
    return [replace(base, line=line, phase=phase, probe_id=probe,
                    sampling={"seed": 7 if phase == "exploration" else 8, "temperature": 0})
            for phase in ("exploration", "confirmation")
            for line in ("E2-S",) for probe in ("original", "control")]


def qualification(_):
    return dict(eligible_for_intervention=True, checkpoint_identity=H,
                system_identity=H, runtime_identity=H, is_synthetic=True, reasons=[])


def authorize(spec, contract):
    assert spec.probe_id in {contract["baseline_probe_id"], contract["control_probe_id"]}


def validate_test_panel(case_id, contract, panel):
    assert case_id == "case" or case_id == "other"
    if contract.get("baseline_probe_id") != "original" or contract.get("control_probe_id") != "control":
        raise ProtocolError("fixture panel roles mismatch")
    for phase in ("exploration", "confirmation"):
        for repeat in {b.repeat for b in panel if b.phase == phase}:
            if {b.probe_id for b in panel if b.phase == phase and b.repeat == repeat} != {"original", "control"}:
                raise ProtocolError("fixture panel roles mismatch")


authorize.validate_panel = validate_test_panel


def broker(tmp_path, *, executor=None, qualifier=qualification, probe_authorizer=authorize, price=2):
    calls = []
    def run(spec):
        calls.append(spec.key)
        return ExecutionOutcome({"answer": spec.probe_id}, COST)
    scheduler = ExperimentScheduler(tmp_path / "broker.db", qualifier=qualifier,
                                    probe_authorizer=probe_authorizer,
                                    executor=executor or run, logical_price=price)
    scheduler.enroll_plan("case", case_contract=dict(line="E2-S", baseline_probe_id="original", control_probe_id="control"), branches=specs())
    return scheduler, calls


def commit(client, rnd=0, selected=None):
    universe = [b.key for b in specs() if b.phase == "exploration"]
    return client.commit_prediction("case", rnd, {k: 1 / len(universe) for k in universe}, selected or universe[0])


def reveal(client, rnd=0):
    token = client.request_probe("case", rnd)
    client.charge_budget(token)
    return client.reveal_result(token)


def test_invariant_1_all_branch_dimensions_and_enrollment(tmp_path):
    b = specs()[0]
    changes = dict(case_id="other", task_identity="b" * 64, line="E2-F",
                   system_identity={"sha256": H, "full_config": {"temperature": 2}},
                   checkpoint_id="other", checkpoint_identity="b" * 64,
                   probe_id="other", probe_identity="b" * 64,
                   sampling={"seed": 8}, repeat=1, runtime_identity="b" * 64,
                   grader_identity="b" * 64, is_synthetic=False, phase="confirmation")
    assert set(changes) == set(asdict(b))
    assert len({b.key, *(replace(b, **{name: value}).key for name, value in changes.items())}) == len(changes) + 1
    with pytest.raises(ProtocolError):
        replace(b, system_identity="mutable-label")
    with pytest.raises(ProtocolError):
        replace(b, runtime_identity={"image_digest": "sha256:" + H})
    s, _ = broker(tmp_path)
    with pytest.raises(ProtocolError, match="share checkpoint"):
        s.enroll_plan("other", case_contract=dict(line="E2-S", baseline_probe_id="original", control_probe_id="control"),
                      branches=[replace(x, case_id="other", checkpoint_id="different" if x.probe_id == "control" else x.checkpoint_id) for x in specs()])


def test_invariant_2_capability_isolation_commit_before_query(tmp_path):
    s, calls = broker(tmp_path)
    a, b = s.register_method("A", 10), s.register_method("B", 10)
    with pytest.raises(ProtocolError, match="commit"):
        a.request_probe("case", 0)
    forged = MethodClient(s._dispatch, "forged")
    with pytest.raises(ProtocolError, match="capability"):
        forged.observations()
    commit(a)
    token = a.request_probe("case", 0)
    assert isinstance(token, str) and "answer" not in token
    for verb in (b.charge_budget, b.reveal_result, a.reveal_result):
        with pytest.raises(ProtocolError):
            verb(token)
    a.charge_budget(token)
    a.reveal_result(token)
    with pytest.raises(ProtocolError):
        commit(a)
    assert b.observations() == []
    with pytest.raises(ProtocolError):
        s.register_method("C", 10)
    with pytest.raises(ProtocolError):
        s.finalize_confirmation()
    assert len(calls) == 1


def test_invariant_3_shared_cache_equal_budget_idempotent(tmp_path):
    s, calls = broker(tmp_path)
    a, b = s.register_method("A", 2), s.register_method("B", 2)
    commit(a); commit(b)
    first = a.request_probe("case", 0)
    assert a.request_probe("case", 0) == first
    a.charge_budget(first); a.charge_budget(first)
    assert a.reveal_result(first) == a.reveal_result(first)
    assert reveal(b) == a.reveal_result(first)
    assert len(calls) == 1
    audit = s.audit()
    assert [(r["spent"], r["reserved"]) for r in audit["methods"]] == [(2, 0), (2, 0)]
    assert len(audit["observations"]) == 2
    assert len(audit["physical"]) == 1
    assert '"gpu_seconds":null' in audit["physical"][0]["cost"]
    pred = __import__("json").loads(audit["predictions"][0]["body"])
    assert pred["budget"] == dict(budget=2, spent=0, reserved=0)
    assert pred["budget_hash"]
    with pytest.raises(ProtocolError, match="known branch"):
        commit(a, 1)
    commit(a, 1, selected=specs()[1].key)
    with pytest.raises(ProtocolError, match="budget"):
        a.request_probe("case", 1)
    assert len(calls) == 1


def test_invariant_4_eligibility_before_cache_failed_restore_noexecute(tmp_path):
    state = dict(eligible=True, probe=True)
    def qualify(cp):
        return {**qualification(cp), "eligible_for_intervention": state["eligible"]}
    def auth(spec, contract):
        if not state["probe"]:
            raise ProtocolError("probe ineligible")
    auth.validate_panel = validate_test_panel
    s, calls = broker(tmp_path, qualifier=qualify, probe_authorizer=auth)
    a, b = s.register_method("A", 10), s.register_method("B", 10)
    commit(a); commit(b)
    state["eligible"] = False
    with pytest.raises(ProtocolError):
        a.request_probe("case", 0)
    assert calls == [] and s.audit()["receipts"] == []
    state["eligible"] = True
    reveal(a)
    state["probe"] = False
    with pytest.raises(ProtocolError, match="probe"):
        b.request_probe("case", 0)
    assert len(calls) == 1 and len(s.audit()["receipts"]) == 1
    state["probe"] = True
    s.qualifier = lambda cp: {**qualification(cp), "system_identity": "b" * 64}
    with pytest.raises(ProtocolError, match="mismatch"):
        b.request_probe("case", 0)


def test_invariant_5_reopen_sequential_round_confirmation_gate(tmp_path):
    s, calls = broker(tmp_path)
    a, b = s.register_method("A", 10), s.register_method("B", 10)
    commit(a)
    with pytest.raises(ProtocolError, match="revealed"):
        commit(a, 1)
    with pytest.raises(ProtocolError, match="pending"):
        a.close_exploration()
    receipt = a.request_probe("case", 0)
    reopened = ExperimentScheduler(tmp_path / "broker.db", qualifier=qualification,
                                  probe_authorizer=authorize,
                                  executor=lambda spec: ExecutionOutcome({"confirmation": True}, COST), logical_price=2)
    # A root RPC adapter retains the bearer; the reopened broker verifies it.
    persisted = reopened.resume_method(a._MethodClient__token)
    persisted.charge_budget(receipt)
    persisted.reveal_result(receipt)
    assert len(persisted.observations()) == 1
    persisted.close_exploration()
    with pytest.raises(ProtocolError):
        reopened.finalize_confirmation()
    b.close_exploration()
    results = reopened.finalize_confirmation()
    assert len(results) == 2
    assert len(reopened.audit()["physical"]) == 3
    assert len(reopened.finalize_confirmation()) == 2
    assert len(reopened.audit()["physical"]) == 3
    assert all(o["result"] == {"answer": "original"} for o in persisted.observations())
    assert not hasattr(persisted, "finalize_confirmation")
    assert not hasattr(persisted, "audit")


def test_invariant_6_uncertain_execution_no_duplicate_side_effect(tmp_path):
    calls = []
    def crash(spec):
        calls.append(spec.key)
        raise RuntimeError("restore/execution failed")
    s, _ = broker(tmp_path, executor=crash)
    a, b = s.register_method("A", 10), s.register_method("B", 10)
    commit(a); commit(b)
    with pytest.raises(ProtocolError, match="indeterminate"):
        a.request_probe("case", 0)
    token = a.request_probe("case", 0)
    a.charge_budget(token)
    with pytest.raises(ProtocolError, match="unavailable"):
        a.reveal_result(token)
    with pytest.raises(ProtocolError, match="retry"):
        b.request_probe("case", 0)
    assert len(calls) == 1
    assert s.audit()["physical"][0]["state"] == "indeterminate"
    assert "RuntimeError" in s.audit()["physical"][0]["error"]
    # Simulate durable crash marker left behind without a callback exception.
    with s._db() as db:
        db.execute("UPDATE outcomes SET state='running'")
    assert a.request_probe("case", 0) == token
    assert len(calls) == 1


def _parallel_worker(path, token, start, active, peak, invocations):
    def execute(spec):
        with active.get_lock():
            active.value += 1
            peak.value = max(peak.value, active.value)
            invocations.value += 1
        time.sleep(0.05)
        with active.get_lock():
            active.value -= 1
        return ExecutionOutcome({"answer": spec.probe_id}, COST)
    s = ExperimentScheduler(path, qualifier=qualification, probe_authorizer=authorize,
                            executor=execute, logical_price=2)
    start.wait()
    client = s.resume_method(token)
    receipt = client.request_probe("case", 0)
    client.charge_budget(receipt)
    client.reveal_result(receipt)


def test_shared_physical_executor_is_serial_across_processes(tmp_path):
    s, _ = broker(tmp_path)
    a, b = s.register_method("A", 2), s.register_method("B", 2)
    commit(a)
    commit(b, selected=specs()[1].key)
    ctx = multiprocessing.get_context("spawn")
    start = ctx.Event()
    active, peak, invocations = (ctx.Value("i", 0) for _ in range(3))
    children = [ctx.Process(target=_parallel_worker,
                           args=(s.path, client._MethodClient__token, start, active, peak, invocations))
                for client in (a, b)]
    for child in children:
        child.start()
    start.set()
    for child in children:
        child.join(10)
        assert child.exitcode == 0
    assert peak.value == 1 and invocations.value == 2
    assert [row["spent"] for row in s.audit()["methods"]] == [2, 2]


def test_independent_probabilities_and_corrupt_prediction_denied(tmp_path):
    s, calls = broker(tmp_path)
    a = s.register_method("A", 10)
    universe = [b.key for b in specs() if b.phase == "exploration"]
    a.commit_prediction("case", 0, {k: .8 for k in universe}, universe[0])
    with s._db() as db:
        row = db.execute("SELECT body FROM predictions").fetchone()
        body = __import__("json").loads(row[0])
        body["selected_branch"] = universe[1]
        db.execute("UPDATE predictions SET body=?", (__import__("json").dumps(body),))
    with pytest.raises(ProtocolError, match="hash"):
        a.request_probe("case", 0)
    assert calls == []


def test_e2_f_case_same_rules(tmp_path):
    s = ExperimentScheduler(tmp_path / "f.db", qualifier=qualification,
                            probe_authorizer=authorize,
                            executor=lambda spec: ExecutionOutcome({"line": spec.line}, COST))
    panel = [replace(b, line="E2-F") for b in specs()]
    s.enroll_plan("case", case_contract=dict(line="E2-F", baseline_probe_id="original", control_probe_id="control"), branches=panel)
    client = s.register_method("A", 1)
    keys = [b.key for b in panel if b.phase == "exploration"]
    client.commit_prediction("case", 0, {k: .8 for k in keys}, keys[0])
    assert reveal(client)["result"] == {"line": "E2-F"}


def test_seed_banks_and_repeat_panel_pairing(tmp_path):
    def enroll(panel):
        s = ExperimentScheduler(tmp_path / "seeds.db", qualifier=qualification,
                                probe_authorizer=authorize,
                                executor=lambda spec: ExecutionOutcome({}, COST))
        s.enroll_plan("case", case_contract=dict(line="E2-S", baseline_probe_id="original", control_probe_id="control"), branches=panel)
    with pytest.raises(ProtocolError, match="disjoint"):
        enroll([replace(b, sampling={"seed": 7, "temperature": 0}) for b in specs()])
    with pytest.raises(ProtocolError, match="distinct seeds"):
        enroll(specs() + [replace(b, repeat=1) for b in specs()])
    with pytest.raises(ProtocolError, match="phase/repeat"):
        enroll(specs() + [replace(specs()[0], repeat=1, sampling={"seed": 9, "temperature": 0})])
    with pytest.raises(ProtocolError, match="non-seed"):
        enroll([replace(b, sampling={**b.sampling, "temperature": .1 if b.phase == "confirmation" else 0}) for b in specs()])
    with pytest.raises(ProtocolError, match="integer seed"):
        enroll([replace(b, sampling={"seed": True, "temperature": 0}) for b in specs()])
    enroll(specs() + [replace(b, repeat=1, sampling={"seed": 9 if b.phase == "exploration" else 10, "temperature": 0}) for b in specs()])


def test_enrollment_requires_frozen_panel_validator_and_original_role(tmp_path):
    panel = specs()
    def absent(spec, contract):
        pass
    s = ExperimentScheduler(tmp_path / "roles.db", qualifier=qualification,
                            probe_authorizer=absent,
                            executor=lambda spec: ExecutionOutcome({}, COST))
    contract = dict(line="E2-S", baseline_probe_id="original", control_probe_id="control")
    with pytest.raises(ProtocolError, match="panel validator"):
        s.enroll_plan("case", case_contract=contract, branches=panel)
    absent.validate_panel = validate_test_panel
    with pytest.raises(ProtocolError, match="original"):
        s.enroll_plan("case", case_contract={**contract, "baseline_probe_id": "control"}, branches=panel)
    def reject(case, contract, panel):
        raise ProtocolError("frozen panel role denied")
    absent.validate_panel = reject
    with pytest.raises(ProtocolError, match="role denied"):
        s.enroll_plan("case", case_contract=contract, branches=panel)
    with s._db() as db:
        assert not db.execute("SELECT 1 FROM plans").fetchone()
        assert not db.execute("SELECT 1 FROM branches").fetchone()
