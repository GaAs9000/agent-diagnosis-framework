"""Controller-owned real cold continuation adapter for the serial scheduler."""
from __future__ import annotations

import asyncio
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from ..experiment_scheduler import BranchSpec, ExecutionOutcome, ProtocolError
from ..manifests import canonical_hash
from ..probe_registry import ProbeRegistry, VERSION, _vocabulary
from .admission import _audit
from .agent import NativeAgent
from .docker_runtime import DockerRuntime
from .office_run import _image, _outcome, _qualification, _sha, _write, grade_workspace
from .qualification import TOOL_SCHEMAS, delivered_tool_result, model_fingerprint
from .tokenizer_client import TokenizerClient


def _require(condition, reason):
    if not condition:
        raise ProtocolError(reason)


class RealBranchExecutor:
    """Execute exactly one physical attempt; return only scorer flags and hashes.

    The scheduler/controller must authorize the registered probe and freeze its
    sampling schedule before dispatch. This adapter independently binds all
    restored identities and rechecks real admission before native continuation.
    """

    def __init__(self, replay, root: Path, baseline: dict, manifest: dict,
                 agent_image: str, grader_image: str, rendered_probes: dict,
                 model_qualification: dict | None = None, expected_grader_identity: str | None = None,
                 *, system_contract: dict | None = None, model_profile: dict | None = None,
                 service_guard=None, generation_guard=None):
        self.replay, self.root = replay, Path(root).absolute()
        self.baseline, self.manifest = copy.deepcopy(baseline), copy.deepcopy(manifest)
        self.agent_image, self.grader_image = _image(agent_image), _image(grader_image)
        self.expected_grader_identity = expected_grader_identity
        self.service_guard = service_guard if service_guard is not None else generation_guard
        self.system_contract = None
        if system_contract is not None:
            from .r12_system import validated_system, system_identity
            self.system_contract = validated_system(system_contract)
            self.system_contract_identity = system_identity(self.system_contract)
            if self.agent_image != self.system_contract["agent_image"]:
                raise ProtocolError("agent image differs from system contract")
            self.qualification = copy.deepcopy(model_profile or model_qualification or {})
            required_profile = {"model_fingerprint", "tokenizer_fingerprint"}
            if not required_profile <= self.qualification.keys():
                raise ProtocolError("complete model profile required for system contract")
            if not callable(self.service_guard):
                raise ProtocolError("service guard must be callable")
            for key, expected in (("model", self.system_contract["model"]["repository"]),
                                  ("model_path", self.system_contract["model"]["path"]),
                                  ("model_revision", self.system_contract["model"]["revision"]),
                                  ("tool_parser", self.system_contract["model"]["tool_parser"]),
                                  ("tool_parser_sha256", self.system_contract["model"]["tool_parser_sha256"])):
                if key in self.qualification:
                    if self.qualification[key] != expected:
                        raise ProtocolError(f"model profile {key} differs from system contract")
            self.model_fingerprint = self.qualification["model_fingerprint"]
            self.model_path = self.system_contract["model"]["path"]
            self.model_revision = self.system_contract["model"]["revision"]
            self.tool_parser = self.system_contract["model"]["tool_parser"]
            self.tool_schemas = copy.deepcopy(self.system_contract["tool_schemas"])
            self.native_tool_schemas = self.tool_schemas
            self.system_prompt = self.system_contract["system_prompt"]
            self.contract_sampling = copy.deepcopy(self.system_contract["sampling"])
            self.protocol_recovery_policy = self.system_contract.get("protocol_recovery_policy", "bounded")
        else:
            self.qualification = _qualification(model_profile or model_qualification or {})
            self.tool_schemas = copy.deepcopy(TOOL_SCHEMAS)
            self.native_tool_schemas = self.tool_schemas
            self.system_prompt = None
            self.contract_sampling = None
            self.protocol_recovery_policy = None
        self.rendered_probes = {key: copy.deepcopy(value.to_dict() if hasattr(value, "to_dict") else value)
                                for key, value in rendered_probes.items()}
        self.registry = ProbeRegistry({"version": VERSION, "probes": list(_vocabulary().values())})
        self.root.mkdir(parents=True, exist_ok=True)

    def __call__(self, spec: BranchSpec) -> ExecutionOutcome:
        started = time.perf_counter()
        folder = self.root / spec.key
        folder.mkdir()  # Exclusive: neither a failed nor complete attempt is retried.
        _write(folder / "attempt.json", asdict(spec))
        try:
            return asyncio.run(self._run(spec, folder, started))
        except Exception as error:
            _write(folder / "failure.json", {"state": "indeterminate", "error_type": type(error).__name__,
                                            "message": str(error), "wall_seconds": time.perf_counter() - started})
            raise

    async def _run(self, spec, folder, started):
        _require(spec.is_synthetic is False and self.baseline.get("is_synthetic") is False,
                 "real branches require real baseline and spec")
        _require(self.baseline.get("execution_valid") is True,
                 "baseline execution must be valid before branching")
        completion = self.baseline.get("task_completed")
        _require(type(completion) is bool,
                 "baseline completion must be assessed before assigning a branch line")
        _require(spec.line == ("E2-S" if completion else "E2-F"),
                 "branch line differs from assessed baseline completion")
        task_id = self.baseline["task_id"]
        task_identity = self.baseline.get("task_identity", canonical_hash(self.manifest["tasks"][task_id]))
        _require(spec.task_identity == task_identity, "task identity mismatch")
        _require(spec.grader_identity == self.expected_grader_identity, "grader identity mismatch")
        _require(spec.system_identity == self.baseline["system_identity"], "baseline system mismatch")
        if self.system_contract is not None:
            declared = self.baseline.get("system_contract_sha256")
            if declared is None and isinstance(self.baseline.get("model_qualification"), dict):
                declared = self.baseline["model_qualification"].get("system_contract_sha256")
            _require(declared in (None, self.system_contract_identity), "system contract identity drift")
            _require(self.baseline.get("runtime", {}).get("model_config", {}).get("model") ==
                     self.system_contract["model"]["repository"], "model identity differs from system contract")
        references = [cp for cp in self.baseline["checkpoints"] if cp["checkpoint_id"] == spec.checkpoint_id]
        _require(len(references) == 1 and spec.checkpoint_identity == spec.checkpoint_id,
                 "checkpoint is not the designated baseline checkpoint")
        reference = references[0]
        _require(spec.probe_id in self.rendered_probes and
                 spec.probe_identity == self.registry.identity(spec.probe_id), "probe identity mismatch")
        probe = self.rendered_probes[spec.probe_id]
        text = probe["rendered_text"]
        _require(probe["probe_id"] == spec.probe_id and isinstance(text, str) and
                 hashlib.sha256(text.encode()).hexdigest() == probe["rendered_digest"], "rendered probe changed")
        _require((text == "") if spec.probe_id == "original" else
                 text.startswith("Runtime-owned advisory: public observation presentation.\n"),
                 "probe is not a runtime-owned advisory")
        _require(type(spec.sampling.get("seed")) is int, "branch seed required")
        allowed_sampling = {"seed", "temperature", "max_tokens", "context_window", "max_retries",
                            "parallel_tool_calls", "top_k", "min_p", "max_turns", "generation_timeout_seconds"}
        _require(set(spec.sampling) <= allowed_sampling, "unsupported branch sampling field")
        if self.contract_sampling is None:
            fixed = {"temperature": 0, "max_tokens": 2048, "context_window": 16384,
                     "max_retries": 0, "parallel_tool_calls": False, "top_k": 1, "min_p": 0,
                     "max_turns": 10, "generation_timeout_seconds": 120}
        else:
            fixed = self.contract_sampling
            _require(type(spec.sampling.get("seed")) is int and
                     spec.sampling["seed"] == fixed.get("seed"),
                     "branch seed differs from system contract")
        _require(all(key in fixed and type(spec.sampling[key]) is type(fixed[key]) and spec.sampling[key] == fixed[key]
                     for key in spec.sampling if key != "seed"), "branch changed frozen generation config")
        base_max_turns = fixed["max_turns"]
        qualification = self.replay.qualify(spec.checkpoint_id)
        _write(folder / "admission_before.json", qualification)
        _require(qualification.get("eligible_for_intervention") is True and
                 qualification.get("is_synthetic") is False, "real checkpoint admission unavailable")
        restored = self.replay.restore(spec.checkpoint_id, folder / "workspace")
        _write(folder / "restore.json", restored)
        _require(restored.get("valid") is True and restored.get("eligible_for_intervention") is True,
                 "real restore failed")
        _require(restored["system_identity"] == spec.system_identity and restored["checkpoint_identity"] == spec.checkpoint_identity,
                 "restored checkpoint/system differs from spec")
        runtime = restored["runtime"]
        _require(runtime == self.baseline["runtime"] and runtime["image_digest"] == self.agent_image,
                 "restored runtime differs from baseline image/config")
        _require(spec.runtime_identity in (restored["runtime_identity"], runtime), "spec runtime differs from checkpoint")
        expected_model = (self.system_contract["model"]["repository"]
                          if self.system_contract is not None else self.qualification["model"])
        expected_provider = self.qualification.get("provider", runtime["model_config"].get("provider"))
        _require(runtime["model_config"].get("provider") == expected_provider and
                 runtime["model_config"].get("model") == expected_model and
                 runtime["model_config"].get("model_fingerprint") == self.qualification["model_fingerprint"],
                 "qualified model differs from checkpoint")
        _require(runtime["tool_config"]["schemas"] == self.tool_schemas, "native tool schema differs from checkpoint")
        messages = restored["messages"]
        if self.system_contract is not None:
            _require(messages and messages[0].get("role") == "system" and
                     messages[0].get("content") == self.system_prompt,
                     "checkpoint system prompt differs from system contract")
        source_prefix = json.loads(Path(reference["messages_path"]).read_text())
        _require(messages == source_prefix, "source conversation prefix changed")
        verified = self.replay.verify_restoration(spec.checkpoint_id, folder / "workspace", messages,
                                                spec.system_identity, runtime)
        _write(folder / "verification.json", verified)
        _require(verified.get("valid") is True and verified.get("eligible_for_intervention") is True,
                 "exact cold restoration verification failed")
        _write(folder / "base_prefix.json", messages)
        _write(folder / "base_runtime.json", runtime)
        _write(folder / "probe.json", probe)
        if self.system_contract is None:
            fingerprint = model_fingerprint()
        else:
            from .r12_system import fingerprint_model
            try:
                fingerprint = fingerprint_model(self.system_contract)
            except TypeError:
                fingerprint = model_fingerprint()
        _write(folder / "current_model_fingerprint.json", fingerprint)
        _require(fingerprint["sha256"] == self.qualification["model_fingerprint"], "current disk/tokenizer model fingerprint changed")
        completed_turns = sum(message["role"] == "assistant" for message in messages)
        remaining_turns = base_max_turns - completed_turns
        _require(remaining_turns > 0, "checkpoint has exhausted frozen baseline turn budget")
        tokenizer = TokenizerClient()
        base_count = await tokenizer.count_messages_details(messages, self.tool_schemas)
        injected = copy.deepcopy(messages)
        if spec.probe_id != "original":
            injected.append({"role": "user", "content": text})
        injected_count = await tokenizer.count_messages_details(injected, self.tool_schemas)
        advisory_count = tokenizer.text_details(text)
        _require(probe.get("token_count") == advisory_count["token_count"], "rendered advisory tokenizer count changed")
        _write(folder / "token_counts.json", {"base": base_count, "injected": injected_count,
                                             "advisory": advisory_count})
        _write(folder / "injected_prefix.json", injected)
        # Keep the contract's request budget immutable; only the call count passed
        # to the continuation is reduced for this checkpoint.
        sampling = {**fixed, "seed": spec.sampling["seed"]}
        identity = canonical_hash({"base_runtime": runtime, "base_system": spec.system_identity,
                                   "probe_identity": spec.probe_identity, "rendered_digest": probe["rendered_digest"],
                                   "sampling": sampling})
        _write(folder / "branch_runtime.json", {"identity": identity, "runtime": runtime,
                                               "sampling": sampling, "completed_assistant_turns": completed_turns})
        source_audit = json.loads(Path(reference["source_audit_path"]).read_text())
        environment_seconds = time.perf_counter() - started
        setup_started = time.perf_counter()
        with DockerRuntime(self.agent_image, folder / "actor-runtime") as actor:
            actor.seed_workspace(folder / "workspace")
            seeded = actor.verify_workspace(folder / "workspace")
            _write(folder / "seed_verification.json", seeded)
            _require(seeded.get("verified") is True, "fresh runtime workspace verification failed")
            audit = actor.audit_boundary()
            _write(folder / "boundary_audit.json", audit)
            _audit(audit, self.agent_image)
            _require(audit["inspect"]["State"]["Pid"] != source_audit["inspect"]["State"]["Pid"],
                     "branch runtime is not distinct from source runtime")
            environment = actor._inspect()["Config"]["Env"]
            _require(isinstance(environment, list) and all(isinstance(item, str) and "=" in item for item in environment),
                     "runtime environment unavailable")
            actual_environment = dict(item.split("=", 1) for item in environment)
            _require(len(actual_environment) == len(environment) and actual_environment == runtime["environment"],
                     "fresh runtime environment differs from checkpoint")
            admission_again = self.replay.qualify(spec.checkpoint_id)
            _write(folder / "admission_before_model.json", admission_again)
            _require(admission_again.get("eligible_for_intervention") is True and admission_again.get("is_synthetic") is False,
                     "real admission revoked before model execution")
            environment_seconds += time.perf_counter() - setup_started
            def workspace_python(code):
                nonlocal environment_seconds
                tool_started = time.perf_counter()
                try:
                    return actor.exec_python(code)
                finally:
                    environment_seconds += time.perf_counter() - tool_started
            from .r11_replay import DispatchRecorder
            dispatch = DispatchRecorder(actor, folder / "tool-dispatch")
            agent_config = {**sampling, "token_count": tokenizer.count_messages,
                            "tool_result_transform": delivered_tool_result,
                            "protocol_recovery_policy": self.protocol_recovery_policy,
                            "identity_context": {"model_fingerprint": self.qualification["model_fingerprint"],
                                "tokenizer_fingerprint": self.qualification.get("tokenizer_fingerprint"),
                                "runtime_identity": restored["runtime_identity"],
                                "system_identity": spec.system_identity}}
            if self.service_guard is not None:
                agent_config["generation_guard"] = self.service_guard
            if self.system_contract is not None:
                agent_config["identity_context"]["system_contract_sha256"] = self.system_contract_identity
            if self.system_contract is None:
                agent = NativeAgent(agent_config, folder / "agent")
                schemas, run_messages = self.tool_schemas, injected
            else:
                from .r12_system import system_identity as contract_identity
                agent_config["system_contract_sha256"] = contract_identity(self.system_contract)
                agent_config["execution_phase"] = spec.phase
                agent = NativeAgent(agent_config, folder / "agent", system_contract=self.system_contract)
                schemas, run_messages = self.tool_schemas, [{"role": "system", "content": self.system_prompt}, *injected[1:]]
            agent_result = await agent.run(run_messages, {"workspace_python": workspace_python}, schemas,
                                           max_turns=remaining_turns, dispatch_callback=dispatch)
            finalization_started = time.perf_counter()
            export = actor.export_workspace(folder / "final-workspace")
            _write(folder / "final_export.json", export)
        grade = grade_workspace(task_id, self.manifest, folder / "final-workspace", self.grader_image, folder / "grade")
        environment_seconds += time.perf_counter() - finalization_started
        _write(folder / "agent_result.json", agent_result)
        _write(folder / "final_grade.json", grade)
        flags = _outcome(self.baseline["initial_grade"], grade, agent_result)
        if flags["execution_valid"] is not True:
            flags.update(execution_valid=False, target_resolved=None, regression_free=None, task_completed=None)
        names = ("execution_valid", "target_resolved", "regression_free", "task_completed")
        result = {"outcome": {name: flags[name] for name in names}, "is_synthetic": False,
                  "tool_dispatch_count": len(dispatch.records), "tool_dispatch_identity": canonical_hash(dispatch.records),
                  "branch_key": spec.key, "checkpoint_identity": spec.checkpoint_identity,
                  "probe_identity": spec.probe_identity, "grader_identity": spec.grader_identity,
                  "evidence_hashes": {"restoration": canonical_hash(verified), "runtime": identity,
                                      "agent_result": _sha(folder / "agent_result.json"),
                                      "final_grade": _sha(folder / "final_grade.json"),
                                      "base_prefix": _sha(folder / "base_prefix.json"),
                                      "injected_prefix": _sha(folder / "injected_prefix.json")}}
        raw = grade.get("official_raw")
        if isinstance(raw, dict) and type(raw.get("total_score")) is int:
            result["raw_score"] = raw["total_score"]
        cost = {"input_tokens": None if agent_result.get("token_usage_unknown") else agent_result["tokens"]["full_prompt_tokens"],
                "output_tokens": None if agent_result.get("token_usage_unknown") else agent_result["tokens"]["output_tokens"], "gpu_seconds": None,
                "environment_seconds": environment_seconds, "wall_seconds": time.perf_counter() - started}
        _write(folder / "cost_definition.json", {"environment_seconds": "controller preparation, tokenizer preparation, Docker setup/tools/export/close and grading; excludes agent model-wait loop", "gpu_seconds": "unmeasured"})
        _write(folder / "result.json", result)
        _write(folder / "cost.json", cost)
        return ExecutionOutcome(result, cost)
