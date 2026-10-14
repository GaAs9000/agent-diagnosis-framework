"""Contract binding checks for the real branch adapter (no runtime execution)."""
from pathlib import Path

import pytest

from c3_sandbox.r1.branch_executor import RealBranchExecutor
from c3_sandbox.r1.r12_system import system_identity
from c3_sandbox.experiment_scheduler import ProtocolError
from test_r12_system import contract


def _executor(tmp_path, *, profile=None, system=None, **kwargs):
    system = system or contract()
    profile = profile or {"model_fingerprint": "f" * 64,
                          "tokenizer_fingerprint": {"sha256": "t" * 64},
                          "provider": "openai-api/mlx/default_model",
                          "model": system["model"]["repository"],
                          "model_path": system["model"]["path"],
                          "model_revision": system["model"]["revision"],
                          "tool_parser": system["model"]["tool_parser"]}
    return RealBranchExecutor(object(), Path(tmp_path), {}, {"tasks": {}},
                              system["agent_image"], "sha256:" + "b" * 64, {},
                              profile, "sha256:" + "c" * 64,
                              system_contract=system, **kwargs)


def test_contract_values_are_bound_without_legacy_defaults(tmp_path):
    system = contract()
    system["protocol_recovery_policy"] = {"enabled": True, "max_recoveries": 1}
    executor = _executor(tmp_path, system=system, service_guard=lambda: {})
    assert executor.tool_schemas == system["tool_schemas"]
    assert executor.system_prompt == system["system_prompt"]
    assert executor.contract_sampling == system["sampling"]
    assert executor.protocol_recovery_policy == system["protocol_recovery_policy"]
    assert executor.system_contract_identity == system_identity(system)


def test_contract_requires_complete_profile_before_branch_dispatch(tmp_path):
    with pytest.raises(ProtocolError, match="complete model profile"):
        _executor(tmp_path, profile={"model_fingerprint": "f" * 64})


def test_contract_rejects_noncallable_service_guard(tmp_path):
    with pytest.raises(ProtocolError, match="service guard"):
        _executor(tmp_path, service_guard="not-callable")
