"""Serial real OfficeVal baseline execution; grading never runs candidate code."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import shutil

from ..evidence._storage import encode
from ..manifests import canonical_hash
from ..replay import ReplayQualification, STATE_CONTRACT_V1
from .agent import NativeAgent
from .docker_runtime import DockerRuntime
from .officeval import normalize_official, task_public_view
from .qualification import MODEL_REPO, TOOL_SCHEMAS, delivered_tool_result
from .tokenizer_client import TokenizerClient

SUPPORTED_STATE = copy.deepcopy(STATE_CONTRACT_V1)
SYSTEM_PROMPT = (
    "Complete the user's document task using the native workspace_python tool. "
    "Use the installed document libraries to inspect and edit the supplied files. "
    "Save the deliverable with its designated original filename. Preserve existing "
    "content and formatting unless the task requires a change. Tool observations "
    "are actual execution results. Explain briefly when finished."
)


def _write(path: Path, value) -> None:
    with path.open("xb") as stream:
        stream.write(encode(value))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _image(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise ValueError("immutable Docker image ID required")
    return value


def _name(value: str) -> str:
    if not isinstance(value, str) or value in ("", ".", "..") or Path(value).name != value or "\\" in value:
        raise ValueError("unsafe public filename")
    return value


def grade_workspace(task_id: str, manifest: dict, workspace: Path,
                    grader_image: str, run_dir: Path) -> dict:
    """Grade only the designated DOCX in a new, independent container.

    Callers must close any actor before invoking this function. A missing original
    filename is delivered as an empty submission directory, a legitimate failure.
    """
    _image(grader_image)
    task = manifest["tasks"][task_id]
    name = _name(task["expected_output"])
    if Path(name).suffix.lower() != ".docx":
        raise ValueError("designated output must be DOCX")
    run_dir = Path(run_dir).absolute()
    run_dir.mkdir(parents=True, exist_ok=False)
    staging = run_dir / "input"
    submission = staging / "submission"
    submission.mkdir(parents=True, mode=0o700)
    staging.chmod(0o700)
    candidate = Path(workspace) / name
    artifact = None
    rejected = None
    if candidate.is_symlink() or (candidate.exists() and
            (not candidate.is_file() or candidate.stat().st_nlink != 1)):
        rejected = "invalid_designated_deliverable"
    elif candidate.is_file():
        shutil.copyfile(candidate, submission / name)
        (submission / name).chmod(0o644)
        artifact = {"name": name, "sha256": _sha(submission / name),
                    "size_bytes": (submission / name).stat().st_size}
    with DockerRuntime(grader_image, run_dir / "runtime", is_grader=True) as grader:
        grader.seed_workspace(staging)
        process = grader.exec_fixed(["python", "-I", "/opt/grader/r1_grader.py",
                                    task_id, "/workspace/submission"])
    _write(run_dir / "process.json", process)
    (run_dir / "stdout.txt").write_text(process.get("stdout", ""))
    (run_dir / "stderr.txt").write_text(process.get("stderr", ""))
    parsed = None
    result = {"official_raw": None, "validity": "infrastructure_invalid",
              "c3_operational_completed": False, "outcome": "infrastructure_invalid",
              "reason": "invalid_grader_output", "coverage_complete": None,
              "unassessed_rules": []}
    if process.get("exit_code") == 0 and not process.get("error") and not process.get("truncated"):
        try:
            parsed = json.loads(process["stdout"])
            # Only this local adapter executes on the host, never official verifier code.
            if isinstance(parsed, dict) and parsed.get("official_raw") is not None:
                normalized = normalize_official(parsed["official_raw"], task_id, task)
                if all(parsed.get(key) == value for key, value in normalized.items()):
                    result = normalized
            elif isinstance(parsed, dict) and parsed.get("validity") == "valid" and parsed.get("reason") in {
                    "missing_deliverable", "empty_deliverable", "invalid_deliverable_inventory"}:
                result.update(validity="valid", outcome="task_failed",
                              reason=parsed["reason"], coverage_complete=False)
            elif isinstance(parsed, dict) and parsed.get("validity") == "infrastructure_invalid":
                result["reason"] = str(parsed.get("reason", "invalid_grader_output"))
        except (ValueError, KeyError, TypeError):
            pass
    if rejected and result["validity"] == "valid":
        result.update(validity="valid", outcome="task_failed", reason=rejected,
                      c3_operational_completed=False, coverage_complete=False)
    result.update(grader_image=grader_image, artifact=artifact, raw_process=process,
                  grader_json=parsed, run_dir=str(run_dir),
                  process_sha256=_sha(run_dir / "process.json"),
                  stdout_sha256=_sha(run_dir / "stdout.txt"),
                  stderr_sha256=_sha(run_dir / "stderr.txt"))
    _write(run_dir / "grade.json", result)
    return result


def _qualification(receipt: dict) -> dict:
    if (receipt.get("status") != "passed" or receipt.get("is_synthetic") is not False
            or receipt.get("provider") != "openai-api/mlx/default_model"
            or receipt.get("model") != MODEL_REPO
            or not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("model_fingerprint", "")))):
        raise ValueError("matching passed real native model qualification required")
    for key in ("native_tool_calls", "prompt_tokens", "completion_tokens", "model_calls"):
        if type(receipt.get(key)) is not int or receipt[key] <= 0:
            raise ValueError("positive native qualification usage required")
    if (not receipt.get("tool_call_id") or receipt["tool_call_id"] != receipt.get("observation_tool_call_id")
            or not receipt.get("nonce") or not receipt["nonce"] == receipt.get("observed_nonce") == receipt.get("independent_nonce")):
        raise ValueError("native qualification observation mismatch")
    return copy.deepcopy(receipt)


def _rule_occurrences(items: list[dict]) -> dict:
    """Preserve verifier order within each repeated (label, weight) group."""
    counts, indexed = {}, {}
    for item in items:
        group = (item["rule"], item["max_delta"])
        ordinal = counts.get(group, 0)
        counts[group] = ordinal + 1
        indexed[(*group, ordinal)] = item["hit"]
    return indexed


def runtime_system_identity(runtime: dict, tool_config: dict, system_prompt: str) -> str:
    """Bind the active prompt, tools and runtime to a checkpoint identity."""
    return canonical_hash({"model": runtime["model_config"], "agent": "NativeAgent-r1",
                           "tools": tool_config, "system_prompt": system_prompt,
                           "sampling": runtime.get("sampling", {}), "runtime": runtime})


def _outcome(initial: dict, final: dict, agent: dict) -> dict:
    valid = initial["validity"] == final["validity"] == "valid" and agent["stopped_reason"] in {"assistant_finished", "max_turns"}
    completed = final["c3_operational_completed"] if valid else None
    raw = final.get("official_raw")
    targets = None
    if final["validity"] == "valid":
        targets = bool(raw and raw.get("dim1_pass") and
                       all(i["hit"] for i in raw["dim2_items"] if i["max_delta"] > 0))
    regression = None
    if initial["validity"] == final["validity"] == "valid":
        before, after = initial.get("official_raw"), final.get("official_raw")
        if (before and after and before.get("dim1_pass")
                and not initial.get("unassessed_rules") and not final.get("unassessed_rules")):
            if not after.get("dim1_pass"):
                regression = False
            else:
                prior = _rule_occurrences(before["dim2_items"])
                current = _rule_occurrences(after["dim2_items"])
                regression = all(current.get(key) is True for key, hit in prior.items()
                                 if key[1] > 0 and hit) and all(
                                     prior.get(key) is True for key, hit in current.items()
                                     if key[1] < 0 and hit)
    outcome = "infrastructure_invalid" if not valid else (
        "success" if completed is True else "partial" if completed is None or targets is True else "failure")
    if not valid:
        targets = regression = None
    return dict(execution_valid=valid, task_completed=completed, target_resolved=targets,
                regression_free=regression, outcome=outcome)


async def run_baseline(task_id: str, manifest: dict, agent_image: str,
                       grader_image: str, model_qualification: dict, output_dir: Path,
                       *, seed: int = 2026, max_turns: int = 10,
                       system_contract: dict | None = None,
                       generation_guard=None,
                       protocol_recovery_policy=None) -> dict:
    """Run one serial baseline, retaining real checkpoints without admitting them."""
    # Historical R1 callers retain the pinned Qwen3-4B qualification contract.
    # Integrated delivery may provide an explicit system contract/profile; that
    # path records the profile and still requires hashes from the caller, but it
    # does not reinterpret a failed historical qualification as a pass.
    receipt = (_qualification(model_qualification) if system_contract is None
               else copy.deepcopy(model_qualification))
    _image(agent_image)
    _image(grader_image)
    if seed != 2026 or type(max_turns) is not int or max_turns < 1:
        raise ValueError("R1 requires seed 2026 and positive max_turns")
    public = task_public_view(manifest, task_id)
    output_dir = Path(output_dir).absolute()
    output_dir.mkdir(parents=True, exist_ok=False)
    staging = output_dir / "public-input"
    staging.mkdir(mode=0o700)
    for entry in public["input_files"]:
        name = _name(entry["name"])
        bundle = Path(manifest["bundle_root"]).resolve()
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe frozen public input path")
        source = bundle / relative
        if not source.resolve().is_relative_to(bundle):
            raise ValueError("public input escapes frozen bundle")
        if source.is_symlink() or not source.is_file() or _sha(source) != entry["sha256"]:
            raise ValueError("frozen public input hash mismatch")
        shutil.copyfile(source, staging / name)
        (staging / name).chmod(0o644)
    if system_contract is not None:
        from .r12_system import validated_system
        system_contract = validated_system(system_contract)
        contract_max_turns = system_contract["sampling"]["max_turns"]
        if max_turns > contract_max_turns:
            raise ValueError("max_turns exceeds integrated system contract")
        sampling = copy.deepcopy(system_contract["sampling"])
        tool_schemas = copy.deepcopy(system_contract["tool_schemas"])
        system_prompt = system_contract["system_prompt"]
    else:
        sampling = {"seed": seed, "temperature": 0, "max_tokens": 2048,
                    "context_window": 16384, "max_retries": 0, "parallel_tool_calls": False,
                    "top_k": 1, "min_p": 0, "max_turns": max_turns}
        tool_schemas = copy.deepcopy(TOOL_SCHEMAS)
        system_prompt = SYSTEM_PROMPT
    runtime = {"image_digest": agent_image,
               "config": {"network": "none", "readonly_rootfs": True, "user": "1000:1000",
                          "cpus": 2, "memory_bytes": 4294967296,
                          "memory_swap_bytes": 4294967296, "pids_limit": 128,
                          "tmpfs": {"/workspace": "rw,nosuid,nodev,size=256m,uid=1000,gid=1000,mode=0700",
                                    "/tmp": "rw,nosuid,nodev,noexec,size=64m,uid=1000,gid=1000,mode=0700"},
                          "cap_drop": ["ALL"], "cap_add": [], "binds": [],
                          "ipc": "none", "pid_mode": "", "privileged": False,
                          "no_new_privileges": True, "workdir": "/workspace",
                          "entrypoint": "sleep", "command": ["infinity"]},
               "model_config": {key: receipt[key] for key in ("provider", "model", "model_fingerprint")},
               "tool_config": {"schemas": copy.deepcopy(tool_schemas), "timeout_seconds": 30,
                               "raw_output_limit_bytes": 1048576,
                               "delivered_stdout_chars": 6000, "delivered_stderr_chars": 2000,
                               "delivered_result": "r1.qualification.delivered_tool_result-v1"},
               "environment": {}}
    _write(output_dir / "model_qualification.json", receipt)
    initial = grade_workspace(task_id, manifest, staging, grader_image, output_dir / "initial-grade")
    replay = ReplayQualification(output_dir / "replay-store")
    checkpoints = []
    with DockerRuntime(agent_image, output_dir / "actor-runtime") as actor:
        actor.seed_workspace(staging)
        environment = actor._inspect()["Config"]["Env"]
        if not isinstance(environment, list) or any(not isinstance(item, str) or "=" not in item for item in environment):
            raise ValueError("exact image environment unavailable")
        runtime["environment"] = dict(item.split("=", 1) for item in environment)
        if len(runtime["environment"]) != len(environment):
            raise ValueError("duplicate image environment keys")
        runtime["sampling"] = sampling
        system_identity = runtime_system_identity(runtime, runtime["tool_config"], system_prompt)
        _write(output_dir / "runtime.json", runtime)
        from .r11_replay import DispatchRecorder
        dispatch = DispatchRecorder(actor, output_dir / "tool-dispatch")
        async def checkpoint(messages, event_index):
            # NativeAgent also invokes the callback after a final text response.
            if messages[-1]["role"] != "tool":
                return
            directory = output_dir / "checkpoints" / f"boundary-{len(checkpoints):03d}"
            directory.mkdir(parents=True)
            try:
                snapshot = actor.export_workspace(directory / "workspace")
                audit = actor.audit_boundary()
            except (ValueError, RuntimeError) as error:
                if getattr(actor, "invalidated", False) or "timeout" in str(error).lower():
                    raise
                if not any(marker in str(error).lower() for marker in
                           ("symlink", "hardlink", "special file", "path limit", "byte limit")):
                    raise
                reference = {"checkpoint_id": None, "event_index": event_index,
                             "source_boundary_qualified": False, "ineligible_reasons": [str(error)]}
                _write(directory / "reference.json", reference)
                checkpoints.append(reference)
                return
            _write(directory / "export.json", snapshot)
            _write(directory / "source_audit.json", audit)
            _write(directory / "messages.json", messages)
            _write(directory / "runtime.json", runtime)
            identifier = replay.register_checkpoint(directory / "workspace", messages,
                system_identity, runtime, SUPPORTED_STATE, is_synthetic=False)
            qualification = replay.qualify(identifier)
            record = replay._record(identifier)
            _write(directory / "checkpoint_record.json", record)
            reference = {"checkpoint_id": identifier, "event_index": event_index,
                         "workspace": str(directory / "workspace"),
                         "messages_path": str(directory / "messages.json"),
                         "source_audit_path": str(directory / "source_audit.json"),
                         "checkpoint_record_path": str(directory / "checkpoint_record.json"),
                         "qualification": qualification, "source_boundary_qualified": audit["qualified"],
                         "runtime_identity": record["runtime_identity"]}
            _write(directory / "reference.json", reference)
            checkpoints.append(reference)
        tokenizer = TokenizerClient()
        identity_context = {"model_fingerprint": receipt["model_fingerprint"],
                            "tokenizer_fingerprint": receipt.get("tokenizer_fingerprint"),
                            "runtime_identity": canonical_hash(runtime),
                            "system_identity": system_identity}
        if system_contract is not None:
            from .r12_system import system_identity as contract_identity
            identity_context["system_contract_sha256"] = contract_identity(system_contract)
        agent_config = {**sampling, "token_count": tokenizer.count_messages,
                        "tool_result_transform": delivered_tool_result,
                        "identity_context": identity_context,
                        "protocol_recovery_policy": protocol_recovery_policy or
                            ("bounded" if system_contract is not None else None)}
        if generation_guard is not None:
            agent_config["generation_guard"] = generation_guard
        if system_contract is None:
            agent = NativeAgent(agent_config, output_dir / "agent")
        else:
            agent = NativeAgent(agent_config, output_dir / "agent", system_contract=system_contract)
        agent_result = await agent.run([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": public["task_prompt"] + "\nDesignated deliverable: " + public["expected_output"]},
        ], {"workspace_python": actor.exec_python}, tool_schemas, max_turns=max_turns,
           checkpoint_callback=checkpoint, dispatch_callback=dispatch)
        actor.export_workspace(output_dir / "final-workspace")
    # Actor context has closed before this independent grader is constructed.
    final = grade_workspace(task_id, manifest, output_dir / "final-workspace", grader_image,
                            output_dir / "final-grade")
    result = {"task_id": task_id, "seed": seed, "is_synthetic": False,
              "output_dir": str(output_dir), "system_identity": system_identity,
              "runtime": runtime, "model_qualification": receipt,
              "initial_grade": initial, "final_grade": final, "agent_result": agent_result,
              "checkpoints": checkpoints, "replay_store": str(output_dir / "replay-store"),
              "final_workspace": str(output_dir / "final-workspace"), "tool_dispatch": dispatch.records,
              **_outcome(initial, final, agent_result)}
    _write(output_dir / "result.json", result)
    return result
