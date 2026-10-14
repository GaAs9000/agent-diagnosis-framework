"""Trusted D3/D4/D5 binding. This is not a hypothesis generator or live Agent."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time

from .evidence._storage import encode
from .experiment_scheduler import BranchSpec, ExecutionOutcome, ProtocolError
from .manifests import canonical_hash


class RegistryAuthorizer:
    """Recheck task-bound approval even for a previously executed cached branch."""

    def __init__(self, registry, reviews):
        self.registry = registry
        self.reviews = reviews  # Trusted controller may revoke an approval.

    def validate_panel(self, case_id, case, branches) -> None:
        if case.get("baseline_probe_id") != "original":
            raise ProtocolError("baseline must be registry original condition")
        try:
            expected = self.registry.experiment_panel(
                case["line"], case["target_probe_id"], case["competing_probe_id"],
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise ProtocolError("invalid frozen target/competing probe panel") from exc
        matched = next(p["probe_id"] for p in expected if p["role"] == "matched")
        if case.get("control_probe_id") != matched:
            raise ProtocolError("matched control must match the registered target")
        required = {p["probe_id"] for p in expected}
        groups = {(branch.phase, branch.repeat) for branch in branches}
        for phase, repeat in groups:
            panel = [b for b in branches if b.phase == phase and b.repeat == repeat]
            if not required.issubset({b.probe_id for b in panel}):
                raise ProtocolError("original/target/matched/competing panel is incomplete")
        for branch in branches:
            if branch.case_id != case_id:
                raise ProtocolError("panel case identity mismatch")
            self(branch, case)

    def __call__(self, branch: BranchSpec, case: dict) -> None:
        for field in ("task_identity", "line", "is_synthetic"):
            if case.get(field) != getattr(branch, field):
                raise ProtocolError(f"case/probe {field} mismatch")
        if branch.probe_identity != self.registry.identity(branch.probe_id):
            raise ProtocolError("probe content identity changed")
        schedules = case.get("sampling_by_phase", {})
        expected = schedules.get(branch.phase, {}).get(str(branch.repeat))
        if expected is None or encode(branch.sampling) != encode(expected):
            raise ProtocolError("branch sampling is not in the frozen case schedule")
        view = deepcopy(case)
        view["sampling"] = deepcopy(branch.sampling)
        review = self.reviews.get((branch.case_id, branch.probe_id))
        verdict = self.registry.validate_for_case(branch.probe_id, view, review)
        if not verdict["eligible"]:
            raise ProtocolError("probe not admitted: " + "; ".join(verdict["reasons"]))


class QualifiedBranchExecutor:
    """Restore and verify before invoking a trusted synthetic execution adapter.

    A real Agent adapter remains unavailable; a declaration cannot promote a
    fixture into scientific evidence. ``handler`` is trusted application code,
    never code supplied by a queried method or loaded from an artifact.
    """

    def __init__(self, replay, root: Path, handler):
        self.replay, self.root, self.handler = replay, Path(root), handler
        self.root.mkdir(parents=True, exist_ok=True)

    def __call__(self, branch: BranchSpec) -> ExecutionOutcome:
        if not branch.is_synthetic:
            raise ProtocolError("real runtime execution is not admitted")
        started = time.perf_counter()
        folder = self.root / branch.key
        folder.mkdir()  # A previous physical attempt is never overwritten/retried.
        workspace = folder / "workspace"
        restored = self.replay.restore(branch.checkpoint_id, workspace)
        (folder / "restore.json").write_bytes(encode(restored))
        if not restored.get("valid") or not restored.get("eligible_for_intervention"):
            raise ProtocolError("restore failed before branch handler")
        for field in ("checkpoint_identity", "system_identity", "runtime_identity", "is_synthetic"):
            if restored.get(field) != getattr(branch, field):
                raise ProtocolError(f"restored {field} differs from branch")
        verified = self.replay.verify_restoration(
            branch.checkpoint_id, workspace, restored["messages"],
            restored["system_identity"], restored["runtime"],
        )
        (folder / "verification.json").write_bytes(encode(verified))
        if not verified.get("valid") or not verified.get("eligible_for_intervention"):
            raise ProtocolError("restoration verification failed before branch handler")
        preparation_seconds = time.perf_counter() - started
        outcome = self.handler(branch, workspace, deepcopy(restored["messages"]), deepcopy(restored["runtime"]))
        if not isinstance(outcome, ExecutionOutcome) or not isinstance(outcome.result, dict):
            raise ProtocolError("independent scorer returned no structured outcome")
        flags = outcome.result.get("outcome")
        names = {"execution_valid", "target_resolved", "regression_free", "task_completed"}
        if (not isinstance(flags, dict) or set(flags) != names
                or any(v is not None and type(v) is not bool for v in flags.values())
                or type(flags["execution_valid"]) is not bool):
            raise ProtocolError("invalid independent scorer outcome fields")
        if flags["execution_valid"] is False and any(flags[k] is True for k in names - {"execution_valid"}):
            raise ProtocolError("invalid execution cannot claim verified success")
        cost = deepcopy(outcome.physical_cost)
        for name in ("environment_seconds", "wall_seconds"):
            if cost.get(name) is not None:
                cost[name] += preparation_seconds
        result = {
            **deepcopy(outcome.result), "is_synthetic": True, "branch_key": branch.key,
            "checkpoint_identity": branch.checkpoint_identity,
            "probe_identity": branch.probe_identity, "grader_identity": branch.grader_identity,
            "restoration_report_identity": canonical_hash(verified),
        }
        (folder / "result.json").write_bytes(encode(result))
        (folder / "cost.json").write_bytes(encode(cost))
        return ExecutionOutcome(result, cost)

