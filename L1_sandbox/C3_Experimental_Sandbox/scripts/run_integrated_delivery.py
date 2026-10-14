#!/usr/bin/env python3
"""Analyse or execute the frozen integrated delivery matrix.

Execution is deliberately dependency-injected: tests can provide ``runner`` and
``preflight`` hooks, while the CLI uses the real OfficeVal adapter only after a
single fail-closed preflight.
"""
from __future__ import annotations
import argparse, hashlib, inspect, json, sys, traceback
from pathlib import Path
from typing import Any, Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from c3_sandbox.integrated.plan import TASK_IDS, build_plan, analyze_saved_runs
from c3_sandbox.integrated.office_runner import build_task_bundle, run_episode_sync
from c3_sandbox.integrated.service_launcher import SerialServiceManager, ServiceLaunchError
from c3_sandbox.integrated.closeout import run_closeout

DEFAULT_CONFIG = ROOT / "configs/integrated_end_to_end.json"

def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("config must be an object")
    return value

def _resolve(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p

def preflight_config(config: Mapping[str, Any], *, root: Path = ROOT) -> dict:
    """Validate all task assets/verifiers and service identity exactly once."""
    tasks = tuple(config.get("task_ids", ()))
    if tasks != TASK_IDS:
        raise ValueError(f"config must contain canonical eight tasks in order: {TASK_IDS}")
    for key in ("seed", "budgets", "models", "identities", "paths", "services"):
        if key not in config:
            raise ValueError(f"missing config section: {key}")
    if config["seed"] != 2026 or config["budgets"].get("max_original_episodes") != 16 or config["budgets"].get("max_execution_fragments") != 100:
        raise ValueError("seed or fixed budgets do not match integrated contract")
    paths = config["paths"]
    bundle_root = _resolve(root, paths["bundle_root"])
    assets, verifiers = _resolve(root, paths["assets"]), _resolve(root, paths["verifiers"])
    bundle = build_task_bundle(bundle_root, assets, verifiers, pinned_manifest=config.get("identities"))
    if not config["identities"].get("grader") or not config["identities"].get("tokenizer") or not config["identities"].get("server"):
        raise ValueError("incomplete grader/tokenizer/server identity configuration")
    target_key = "executor" if "executor" in config["models"] else "target"
    for model_key in (target_key, "reference"):
        model = config["models"].get(model_key, {})
        name = model.get("name") or model.get("model")
        if not name:
            raise ValueError(f"missing {model_key} model name")
        if not model.get("agent_image"):
            raise ValueError(f"missing {model_key} model field: agent_image")

        prov = (model.get("provider") or model.get("engine") or "").lower()
        backend_type = (model.get("backend") or ("api" if prov in {"gemini", "google", "openai", "anthropic"} else ("saved_reference" if prov == "saved_reference" else "local"))).lower()
        is_api = backend_type == "api" or prov in {"gemini", "google", "openai", "anthropic"}
        is_saved_ref = backend_type == "saved_reference" or prov == "saved_reference"

        if not is_api and not is_saved_ref:
            if not model.get("fingerprint"):
                raise ValueError(f"missing {model_key} model field: fingerprint")
            if model.get("model_path"):
                model_path = _resolve(root, model["model_path"])
                if not model_path.is_dir():
                    raise ValueError(f"model path unavailable for {model_key}: {model_path}")
            if model_key in ("target", "executor"):
                contract_path_str = model.get("system_contract_path")
                if contract_path_str:
                    contract_path = _resolve(root, contract_path_str)
                    if not contract_path.is_file():
                        raise ValueError("target system contract unavailable")
                    contract = json.loads(contract_path.read_text())
                    if "system" not in contract:
                        contract = {"system": contract}
                    if model.get("model_path") and contract["system"]["model"]["path"] != model["model_path"]:
                        raise ValueError("target model path differs from system contract")
        elif is_saved_ref:
            qualification_path = _resolve(root, model.get("qualification_path", ""))
            if qualification_path.suffix and not qualification_path.is_file():
                raise ValueError("reference qualification receipt unavailable")
        elif is_api:
            import os
            if prov in {"gemini", "google"} and not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or model.get("api_key")):
                raise ValueError("Gemini API key is missing; set GEMINI_API_KEY or GOOGLE_API_KEY")
            elif prov in {"openai"} and not (os.environ.get("OPENAI_API_KEY") or model.get("api_key")):
                raise ValueError("OpenAI API key is missing; set OPENAI_API_KEY")
            elif prov in {"anthropic"} and not (os.environ.get("ANTHROPIC_API_KEY") or model.get("api_key")):
                raise ValueError("Anthropic API key is missing; set ANTHROPIC_API_KEY")

    server = config["identities"]["server"]
    local_provider_present = any(
        (config["models"][k].get("backend") == "local" or config["models"][k].get("provider") not in {"gemini", "saved_reference", "openai", "anthropic"})
        for k in (target_key, "reference") if k in config["models"]
    )
    if local_provider_present and server.get("python"):
        python_path = Path(server.get("python", ""))
        if not python_path.is_absolute() or not python_path.is_file():
            raise ValueError("approved service Python must be an existing absolute path")
    services = config["services"]
    if not services.get("model_endpoint") or not services.get("tokenizer_endpoint"):
        raise ValueError("model and tokenizer endpoints are required")
    if not config["identities"]["server"].get("auto_launch") and (type(services.get("model_pid")) is not int or services["model_pid"] <= 0):
        raise ValueError("live model_pid is required when auto_launch is disabled")
    return {"bundle": bundle, "bundle_root": str(bundle_root), "task_ids": list(tasks),
            "service_config": dict(services), "preflight_config": dict(config)}

def _default_runner(context: Mapping[str, Any], task_id: str, model_key: str) -> Mapping[str, Any]:
    cfg = context["config"]
    models_cfg = cfg.get("models", {})
    actual_key = "executor" if (model_key in {"target", "executor"} and "executor" in models_cfg) else model_key
    model = models_cfg.get(actual_key, models_cfg.get(model_key, {}))
    out = context["output_dir"] / model_key / task_id

    is_saved_ref = (model.get("backend") == "saved_reference") or (model.get("provider") == "saved_reference")
    if is_saved_ref:
        return {"status": "NOT_RUN", "reason": "reference evidence is preserved separately; no duplicate original run"}

    gateway = context.get("gateway")
    if gateway is None:
        try:
            from c3_sandbox.gateway import ModelGateway
            gateway = ModelGateway.from_config(cfg, root=ROOT)
            context["gateway"] = gateway
        except Exception:
            pass

    if gateway is not None:
        backend = gateway.executor if model_key in {"target", "executor"} else gateway.reference
        if isinstance(backend, SavedReferenceBackend) or getattr(backend, "backend_type", "") == "saved_reference":
            return {"status": "NOT_RUN", "reason": "reference evidence is preserved separately; no duplicate original run"}

        ok, reason = backend.check_availability()
        if not ok:
            if backend.backend_type == "local":
                return {"status": "NOT_RUN", "reason": f"local backend unverified / unavailable: {reason}"}
            raise RuntimeError(f"Backend unavailable for {model_key}: {reason}")

        agent_image = model.get("agent_image", "sha256:e336609b5a86ee91c28751e7cfd2a067db3ac4442d0263b9709de9b3a16ebef3")
        grader_image = cfg.get("identities", {}).get("grader", {}).get("image_id", "c3-integrated-grader:r2")

        if backend.backend_type == "local":
            manager = context.get("service_manager")
            if manager is not None:
                pid = manager.ensure(model_key)
                cfg["services"]["model_pid"] = pid

        return gateway.run_episode_sync(
            context["bundle"],
            task_id,
            agent_image=agent_image,
            grader_image=grader_image,
            output_dir=out,
            seed=cfg["seed"],
            max_turns=model.get("max_turns", 10),
            budget=cfg["budgets"]["max_execution_fragments"],
            backend_override=backend,
        )

    if model.get("provider") == "gemini":
        from c3_sandbox.integrated.gemini_episode import run_gemini_episode_sync
        return run_gemini_episode_sync(
            context["bundle"], task_id, agent_image=model["agent_image"],
            grader_image=cfg["identities"]["grader"]["image_id"], output_dir=out,
            model_fingerprint=model.get("name", "gemini"), tokenizer_fingerprint="gemini-native-wire-v1",
            seed=cfg["seed"], max_turns=model.get("max_turns", 10), budget=cfg["budgets"]["max_execution_fragments"])
    manager = context.get("service_manager")
    if manager is not None:
        pid = manager.ensure(model_key)
        cfg["services"]["model_pid"] = pid
    contract = None
    if model_key in ("target", "executor"):
        contract_data = json.loads(_resolve(ROOT, model["system_contract_path"]).read_text())
        contract = contract_data.get("system", contract_data)
        profile = {"model": model.get("repository", model.get("name")), "model_path": model["model_path"],
                   "model_revision": model.get("revision"), "tool_parser": model.get("tool_parser"),
                   "model_fingerprint": model["fingerprint"],
                   "tokenizer_fingerprint": model.get("tokenizer_fingerprint"),
                   "provider": "openai-api/mlx/default_model"}
    else:
        profile = json.loads(_resolve(ROOT, model["qualification_path"]).read_text())
    def guard():
        # A real target run must provide a live PID observed by the caller;
        # returning a synthetic observation is forbidden.
        pid = cfg["services"].get("model_pid")
        if type(pid) is not int or pid <= 0:
            raise RuntimeError("live model_pid is required for service identity guard")
        from c3_sandbox.r1.r12_service import observe_service
        return observe_service(contract, pid) if contract is not None else {"pid": pid, "model": model.get("name", model.get("model"))}
    return run_episode_sync(context["bundle"], task_id,
        agent_image=model["agent_image"], grader_image=cfg["identities"]["grader"]["image_id"],
        model_profile=profile, output_dir=out, model_fingerprint=model["fingerprint"],
        tokenizer_fingerprint=model.get("tokenizer_fingerprint", cfg.get("identities", {}).get("tokenizer", {}).get("fingerprints", {}).get(model_key)),
        seed=cfg["seed"], max_turns=model.get("max_turns", 10), budget=cfg["budgets"]["max_execution_fragments"],
        system_contract=contract, generation_guard=guard if contract is not None else None)

def execute_delivery(config: Mapping[str, Any], output_dir: Path, *, runner: Callable | None = None,
                     preflight: Callable | None = None, root: Path = ROOT,
                     gateway: Any | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    global_setup_error = None
    try:
        context = (preflight or preflight_config)(config) if preflight else preflight_config(config, root=root)
        if not isinstance(context, Mapping):
            raise ValueError("preflight hook must return a mapping")
        context = dict(context); context.update(config=config, output_dir=output_dir)
    except Exception as exc:
        manifest = {"schema_version": "integrated-run-manifest-v1", "status": "setup_failed", "error": f"{type(exc).__name__}: {exc}", "runs": []}
        (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        raise RuntimeError(manifest["error"]) from exc
    lock = {"schema_version": "integrated-config-lock-v1", "config_sha256": _sha(json.dumps(config, sort_keys=True, ensure_ascii=False).encode()), "task_ids": list(TASK_IDS), "seed": config["seed"]}
    (output_dir / "config.lock").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    if gateway is not None:
        context["gateway"] = gateway
    else:
        try:
            from c3_sandbox.gateway import ModelGateway
            context["gateway"] = ModelGateway.from_config(config, root=root)
        except Exception:
            context["gateway"] = None
    manager = None if runner is not None else SerialServiceManager(root=root, config=dict(config), output_dir=output_dir)
    context["service_manager"] = manager
    hook = runner or _default_runner
    rows = []
    target_key = "executor" if "executor" in config.get("models", {}) else "target"
    try:
        # Keep one model service warm for its complete fixed task slice.
        for model_key in (target_key, "reference"):
            for task_id in TASK_IDS:
                model_entry = config["models"][model_key]
                model_name = model_entry.get("name") or model_entry.get("model", "unknown")
                row = {"task_id": task_id, "model": model_name, "model_key": model_key}
                try:
                    try:
                        arity = len(inspect.signature(hook).parameters)
                    except (TypeError, ValueError):
                        arity = 3
                    result = hook(task_id, model_name) if arity <= 2 else hook(context, task_id, model_key)
                    if inspect.isawaitable(result):
                        import asyncio; result = asyncio.run(result)
                    if isinstance(result, Mapping): row.update(dict(result))
                    raw_status = str(row.get("status", "complete")).upper()
                    row["status"] = {"OK": "complete", "SUCCESS": "complete", "COMPLETE": "complete", "FAILED": "failed", "NOT_RUN": "NOT_RUN"}.get(raw_status, "failed")
                except Exception as exc:
                    if isinstance(exc, ServiceLaunchError):
                        row.update(status="setup_failed", error=f"{type(exc).__name__}: {exc}")
                        global_setup_error = row["error"]
                        rows.append(row)
                        remaining = [(mk, tid) for mk in (target_key, "reference") for tid in TASK_IDS
                                     if not any(r.get("model_key") == mk and r.get("task_id") == tid for r in rows)]
                        rows.extend({"task_id": tid, "model": config["models"][mk].get("name", config["models"][mk].get("model", "unknown")),
                                     "model_key": mk, "status": "NOT_RUN", "reason": global_setup_error}
                                    for mk, tid in remaining)
                        break
                    row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                if not global_setup_error:
                    rows.append(row)
            if global_setup_error:
                break
    finally:
        if manager is not None:
            manager.stop()
    manifest_status = "setup_failed" if global_setup_error else ("complete" if all(r["status"] == "complete" for r in rows) else "completed_with_failures")
    manifest = {"schema_version": "integrated-run-manifest-v1", "status": manifest_status, "runs": rows, "config_lock": lock}
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    closeout = run_closeout(rows, config, output_dir / "closeout", gateway=context.get("gateway"))
    manifest["closeout"] = {"cases": closeout["cases"], "cards": closeout.get("cards", 0), "predictions": closeout["predictions"],
                             "receipts": closeout["receipts"], "branches": closeout["branches"]}
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest

def main(argv=None, *, runner=None, preflight=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG); ap.add_argument("--runs", type=Path, action="append", default=[]); ap.add_argument("--manifest", type=Path, default=Path("integrated_delivery_freeze.json")); ap.add_argument("--output-dir", type=Path, default=Path("runs/integrated-delivery-execute")); ap.add_argument("--model-pid", type=int); ap.add_argument("--execute", action="store_true")
    args = ap.parse_args(argv); config = _load(args.config); runs = []
    if args.model_pid is not None:
        config.setdefault("services", {})["model_pid"] = args.model_pid
    for path in args.runs:
        if path.exists():
            value = json.loads(path.read_text()); runs.extend(value if isinstance(value, list) else [value])
    plan = build_plan(task_ids=config["task_ids"], target_model=config["models"]["target"]["name"], reference_model=config["models"]["reference"]["name"], seed=config["seed"], existing_runs=runs)
    if args.execute:
        try: result = execute_delivery(config, args.output_dir, runner=runner, preflight=preflight)
        except RuntimeError as exc: print(str(exc), file=sys.stderr); return 2
        print(json.dumps(result, indent=2, ensure_ascii=False)); return 0 if result["status"] == "complete" else 1
    result = plan.to_dict(); result["analysis"] = analyze_saved_runs(plan, runs); args.manifest.parent.mkdir(parents=True, exist_ok=True); args.manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); print(json.dumps(result, indent=2, sort_keys=True)); return 0

if __name__ == "__main__": raise SystemExit(main())
