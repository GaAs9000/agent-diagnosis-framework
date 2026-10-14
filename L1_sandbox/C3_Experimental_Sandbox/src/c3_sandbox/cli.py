"""Local research tooling; live study execution is deliberately not a command yet."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def emit(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


@contextmanager
def serial_execution(lock_path: Path):
    """One C3 invocation may execute containers at a time on this checkout."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another C3 execution is active; local concurrency is fixed at 1.") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="C3 local serial infrastructure")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("validate", "plan"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/local_serial.json")
    smoke = sub.add_parser("smoke", help="Synthetic infrastructure check; no model API calls")
    smoke.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs/infrastructure")
    e2 = sub.add_parser("e2-acceptance", help="Run D1-D5 deterministic integration; no model calls")
    e2.add_argument("--output", type=Path, required=True)
    gate = sub.add_parser("e2-gate", help="Verify D1-D5 test evidence against current source")
    gate.add_argument("--acceptance", type=Path, required=True)
    gate.add_argument("--junit", type=Path, required=True)
    check = sub.add_parser("e2-validate", help="Validate a trajectory or frozen manifest")
    check.add_argument("kind", choices=["trajectory", "family", "system"])
    check.add_argument("record", type=Path)
    legacy = sub.add_parser("e2-import-legacy", help="Lossless development-only legacy import")
    legacy.add_argument("run_directory", type=Path)
    legacy.add_argument("--family-id", required=True)
    legacy.add_argument("--system-id", required=True)
    legacy.add_argument("--output", required=True, type=Path)
    snapshot = sub.add_parser("snapshot")
    actions = snapshot.add_subparsers(dest="action", required=True)
    for name in ("capture", "restore", "verify"):
        cmd = actions.add_parser(name)
        cmd.add_argument("--store", required=True, type=Path)
        cmd.add_argument("--workspace", required=True, type=Path)
        if name != "capture":
            cmd.add_argument("snapshot_id")
    prediction = sub.add_parser("prediction")
    actions = prediction.add_subparsers(dest="action", required=True)
    freeze = actions.add_parser("freeze")
    freeze.add_argument("record", type=Path)
    freeze.add_argument("--store", required=True, type=Path)
    verify = actions.add_parser("verify")
    verify.add_argument("prediction_id")
    verify.add_argument("--store", required=True, type=Path)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command in ("validate", "plan"):
            from .config import load_config
            from .planning import build_plan

            plan = build_plan(load_config(args.config))
            emit({"configuration_valid": True, **plan})
            return 0
        if args.command == "smoke":
            from .adapters.inspect_smoke import run_smoke

            with serial_execution(PROJECT_ROOT / "runs/.execution.lock"):
                result = run_smoke(args.output.resolve())
                emit(result)
            return 0 if result.get("passed", False) else 1
        if args.command == "e2-acceptance":
            from .acceptance import run_acceptance
            with serial_execution(PROJECT_ROOT / "runs/.execution.lock"):
                result = run_acceptance(args.output.resolve())
                emit(result)
            return 0 if result.get("status") == "passed" else 1
        if args.command == "e2-gate":
            from .foundation_gate import verify_foundation
            emit(verify_foundation(args.acceptance, args.junit))
            return 0
        if args.command == "e2-validate":
            from .manifests import canonical_hash, validate_family_manifest, validate_system_manifest
            from .trajectory import validate_trajectory
            validators = {"trajectory": validate_trajectory, "family": validate_family_manifest, "system": validate_system_manifest}
            record = validators[args.kind](json.loads(args.record.read_text(encoding="utf-8")))
            emit({"valid": True, "kind": args.kind, "identity": canonical_hash(record)})
            return 0
        if args.command == "e2-import-legacy":
            from .trajectory import import_legacy_episode
            from .evidence._storage import encode
            from .manifests import canonical_hash
            record = import_legacy_episode(args.run_directory, task_family_id=args.family_id, system_id=args.system_id)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("xb") as file:
                file.write(encode(record))
            emit({"imported": True, "identity": canonical_hash(record), "split": "development", "replay_qualified": False})
            return 0
        if args.command == "snapshot":
            from .evidence import WorkspaceStore

            store = WorkspaceStore(args.store)
            if args.action == "capture":
                emit({"snapshot_id": store.capture(args.workspace), "scope": "filesystem_only"})
                return 0
            if args.action == "restore":
                report = store.restore(args.snapshot_id, args.workspace)
            else:
                report = store.verify(args.snapshot_id, args.workspace)
            emit(report.to_dict())
            return 0 if report.valid else 1
        if args.command == "prediction":
            from .evidence import PredictionStore

            store = PredictionStore(args.store)
            if args.action == "freeze":
                record = json.loads(args.record.read_text(encoding="utf-8"))
                emit({"prediction_id": store.freeze(record)})
            else:
                emit({"integrity_valid": True, "record": store.read(args.prediction_id)})
            return 0
        raise ValueError("Unsupported command")
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        emit({"error": str(exc), "completed": False})
        return 2


if __name__ == "__main__":
    sys.exit(main())

