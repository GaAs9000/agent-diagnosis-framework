"""Cold checkpoint integrity with optional controller-owned real admission.

This module does not attest process isolation or recover a live model session.
"""
import json
from pathlib import Path
import re
import tempfile

from .evidence import WorkspaceStore
from .evidence._storage import directory, digest, encode, get, put


STATE_CONTRACT_V1 = {
    "scope": "cold-tool-boundary-v1",
    "at_tool_boundary": True,
    "pending_tool_calls": 0,
    "persistent_processes": False,
    "external_state": "none",
    "caches": "none",
    "environment": "runtime.environment",
}


def _state_reasons(contract):
    if not isinstance(contract, dict):
        return ["missing state contract"]
    reasons = []
    if set(contract) != set(STATE_CONTRACT_V1):
        reasons.append("state contract fields must exactly match v1")
    for key, expected in STATE_CONTRACT_V1.items():
        value = contract.get(key)
        if type(value) is not type(expected) or value != expected:
            reasons.append(f"unsupported or missing state: {key}")
    return reasons


def _runtime_reasons(runtime):
    if not isinstance(runtime, dict):
        return ["missing runtime configuration"]
    reasons = []
    if not isinstance(runtime.get("image_digest"), str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", runtime["image_digest"]):
        reasons.append("missing immutable runtime image digest")
    for key in ("config", "model_config", "tool_config", "environment"):
        if not isinstance(runtime.get(key), dict):
            reasons.append(f"missing runtime identity component: {key}")
    environment = runtime.get("environment")
    if isinstance(environment, dict) and any(not isinstance(k, str) or not isinstance(v, str) for k, v in environment.items()):
        reasons.append("environment must contain exact string keys and values")
    return reasons


def _message_reasons(messages):
    if not isinstance(messages, list) or not messages:
        return ["missing exact conversation/tool prefix"]
    pending = set()
    reasons = []
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str) or message["role"] not in {"system", "developer", "user", "assistant", "tool"}:
            reasons.append("invalid conversation message")
            continue
        if "content" not in message and "tool_calls" not in message:
            reasons.append("missing message content/tool call")
        if "function_call" in message:
            reasons.append("legacy function call state is unsupported")
        if "tool_calls" in message and message["role"] != "assistant":
            reasons.append("tool calls require an assistant message")
        calls = message.get("tool_calls", [])
        if not isinstance(calls, list):
            reasons.append("invalid tool call prefix")
            continue
        for call in calls:
            call_id = call.get("id") if isinstance(call, dict) else None
            if not isinstance(call_id, str) or not call_id or call_id in pending:
                reasons.append("invalid/duplicate pending tool call identity")
            else:
                pending.add(call_id)
        if message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in pending:
                reasons.append("tool result lacks matching pending call")
            else:
                pending.remove(call_id)
    if pending:
        reasons.append("conversation contains unfinished tool calls")
    return reasons


class ReplayQualification:
    """Content addressed checkpoints; qualification is recomputed, never overridden.

    ``root`` may be a WorkspaceStore or its path. Records are persisted next to
    its objects, outside the captured workspace. Real checkpoints additionally
    require an evidence-bound R1 admission receipt from the trusted controller.
    """

    def __init__(self, root: Path | WorkspaceStore, real_admission=None):
        self.real_admission = real_admission
        self.workspace_store = root if isinstance(root, WorkspaceStore) else WorkspaceStore(root)
        self.records = self.workspace_store.root / "replay-checkpoints"
        directory(self.records)

    def register_checkpoint(self, workspace: Path, messages: list, system_identity: str,
                            runtime: dict, state_contract: dict, is_synthetic: bool) -> str:
        # Canonical serialization preserves all JSON fields, not merely text.
        inputs = {"messages": messages, "system_identity": system_identity,
                  "runtime": runtime, "state_contract": state_contract,
                  "is_synthetic": is_synthetic}
        data = encode(inputs)
        if json.loads(data) != inputs:
            raise ValueError("checkpoint inputs must be exact JSON values")
        reasons = self._input_reasons(inputs)
        snapshot_id = None
        try:
            snapshot_id = self.workspace_store.capture(workspace)
            report = self.workspace_store.verify(snapshot_id, workspace)
            if not report.valid:
                reasons.extend(report.mismatches)
        except (ValueError, OSError, TypeError, KeyError) as error:
            reasons.append(f"snapshot capture failed: {error}")
        record = {"schema_version": 1, **inputs, "snapshot_id": snapshot_id,
                  "message_identity": digest(encode(messages)),
                  "runtime_identity": digest(encode(runtime)),
                  "registration_reasons": reasons}
        return put(self.records, encode(record), ".json")

    @staticmethod
    def _input_reasons(record):
        reasons = _state_reasons(record.get("state_contract"))
        reasons += _runtime_reasons(record.get("runtime"))
        reasons += _message_reasons(record.get("messages"))
        if not isinstance(record.get("system_identity"), str) or not re.fullmatch(r"[0-9a-f]{64}", record["system_identity"]):
            reasons.append("system identity must be an immutable full-configuration SHA-256 (64 lowercase hexadecimal characters)")
        if type(record.get("is_synthetic")) is not bool:
            reasons.append("is_synthetic must be an explicit boolean")
        return reasons

    def _record(self, checkpoint_id):
        record = json.loads(get(self.records, checkpoint_id, ".json"))
        if not isinstance(record, dict) or record.get("schema_version") != 1:
            raise ValueError("invalid replay checkpoint schema")
        if digest(encode(record["messages"])) != record["message_identity"]:
            raise ValueError("message identity mismatch")
        if digest(encode(record["runtime"])) != record["runtime_identity"]:
            raise ValueError("runtime identity mismatch")
        return record

    def qualify(self, checkpoint_id: str) -> dict:
        result = {"eligible_for_intervention": False, "checkpoint_identity": checkpoint_id,
                  "system_identity": None, "runtime_identity": None,
                  "is_synthetic": None, "reasons": []}
        try:
            record = self._record(checkpoint_id)
            result.update(system_identity=record["system_identity"],
                          runtime_identity=record["runtime_identity"],
                          is_synthetic=record["is_synthetic"])
            reasons = self._input_reasons(record) + record["registration_reasons"]
            if record.get("is_synthetic") is not True:
                from .r1.admission import R1AdmissionAuthority
                if type(self.real_admission) is not R1AdmissionAuthority or not self.real_admission.verify(checkpoint_id, record):
                    reasons.append("real replay blocked: runtime isolation attestations and acceptance unavailable")
                else:
                    result["admission_receipt"] = self.real_admission.receipt_identity(checkpoint_id, record)
            if record["snapshot_id"] is None:
                reasons.append("missing supported filesystem snapshot")
            else:
                # Exercise the actual cold restoration and complete verification.
                with tempfile.TemporaryDirectory(prefix="c3-replay-check-") as folder:
                    restored = self.workspace_store.restore(record["snapshot_id"], Path(folder) / "workspace")
                    reasons.extend(restored.mismatches)
            result["reasons"] = list(dict.fromkeys(reasons))
            result["eligible_for_intervention"] = not result["reasons"]
        except (ValueError, OSError, TypeError, KeyError) as error:
            result["reasons"] = [f"checkpoint verification failed: {error}"]
        return result

    def restore(self, checkpoint_id: str, destination: Path) -> dict:
        result = self.qualify(checkpoint_id)
        result["valid"] = False
        if not result["eligible_for_intervention"]:
            return result
        try:
            record = self._record(checkpoint_id)
            report = self.workspace_store.restore(record["snapshot_id"], destination)
            result["workspace_report"] = report.to_dict()
            result["reasons"].extend(report.mismatches)
            result["valid"] = report.valid
            result["eligible_for_intervention"] = report.valid
            if report.valid:
                result.update(messages=record["messages"], runtime=record["runtime"],
                              state_contract=record["state_contract"])
        except (ValueError, OSError, TypeError, KeyError) as error:
            result["reasons"].append(f"restoration failed: {error}")
            result["eligible_for_intervention"] = False
        return result

    def verify_restoration(self, checkpoint_id: str, workspace: Path, messages: list,
                           system_identity: str, runtime: dict) -> dict:
        result = self.qualify(checkpoint_id)
        result["valid"] = False
        if not result["eligible_for_intervention"]:
            return result
        try:
            record = self._record(checkpoint_id)
            report = self.workspace_store.verify(record["snapshot_id"], workspace)
            result["workspace_report"] = report.to_dict()
            result["reasons"].extend(report.mismatches)
            if encode(messages) != encode(record["messages"]):
                result["reasons"].append("exact conversation/tool prefix mismatch")
            if system_identity != record["system_identity"]:
                result["reasons"].append("system identity mismatch")
            if encode(runtime) != encode(record["runtime"]):
                result["reasons"].append("runtime identity mismatch")
            result["valid"] = not result["reasons"]
            result["eligible_for_intervention"] = result["valid"]
        except (ValueError, OSError, TypeError, KeyError) as error:
            result["reasons"].append(f"restoration verification failed: {error}")
            result["eligible_for_intervention"] = False
        return result
