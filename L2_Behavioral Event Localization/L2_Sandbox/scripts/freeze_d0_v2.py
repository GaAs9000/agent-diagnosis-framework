#!/usr/bin/env python3
"""Freeze reviewed scientific inputs without authorizing any D1 execution."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "L2_Behavioral Event Localization/L2_Sandbox/src"))
from l2_sandbox.v3_execution import file_digest, read_json, seal, verify_seal
from l2_sandbox.measurement_gates import measurement_gate_binding

POLICY = "bounded-mechanism-contract-v2"
PARAMETERS = {"n_per_arm": 14, "delta": .10, "tau": .20, "alpha": .05,
              "ceiling_threshold": .05, "estimand": "conditional_frozen_dev_pool",
              "target_failure_units": 24, "passing_harm_sample_size": 6}


def _planner():
    path = REPO_ROOT / "scripts/v3_d1_ladder.py"
    spec = importlib.util.spec_from_file_location("_d0_canonical_planner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _recheck_gate(binding):
    if not binding.get("path") or not binding.get("sha256"):
        raise ValueError("missing measurement audit file binding")
    path = Path(binding["path"])
    if file_digest(path) != binding["sha256"]:
        raise ValueError("measurement audit file changed")
    current, errors = measurement_gate_binding(path)
    if errors or current != binding:
        raise ValueError(f"measurement raw evidence recheck failed: {errors}")


def build_freeze(*, plan_path, gate_binding_path, authorization_path, contract_path, source_commit=None):
    plan = read_json(plan_path)
    verify_seal(plan, "plan_sha256")
    _planner().verify_prospective_plan_inputs(plan)
    if plan.get("scientific_plan_ready") is not True or plan.get("readiness_errors"):
        raise ValueError("scientific plan is not ready")
    if plan.get("mode") != "plan_only" or plan.get("execution_ready") is not False or plan.get("execution_authorized") is not False:
        raise ValueError("D0 scientific freeze requires a strictly non-executable plan")
    params = plan.get("d0_candidate_parameters", {})
    if any(params.get(k) != v for k, v in PARAMETERS.items()):
        raise ValueError("approved D0 parameters differ")
    if len(plan.get("unit_schedules", [])) != 24 or len(plan.get("r12_passing_harm_check_plan", {}).get("sampled_passing_units", [])) != 6:
        raise ValueError("D0 requires exactly 24 failure units and six controls")
    document = read_json(gate_binding_path)
    if document.get("errors") != []:
        raise ValueError("v2 gate has errors or missing explicit error list")
    binding = document.get("binding", {})
    if binding != plan.get("measurement_gate_binding"):
        raise ValueError("plan and supplied gate binding differ")
    if (binding.get("policy") != POLICY or binding.get("post_hoc_forensic_extension") is not True
            or binding.get("passed") is not True
            or binding.get("gates") != {f"V3-G{i}": True for i in range(1, 5)}):
        raise ValueError("all four v2 measurement gates must pass under the post-hoc contract")
    if (binding.get("execution_readiness") is not False
            or binding.get("execution_authorized") is not False
            or binding.get("current_execution_scope") != "D0_freeze_and_D1_plan_only"):
        raise ValueError("measurement contract must remain non-executable and plan-only")
    if any(binding.get(k) != "not_established" for k in ("statistical_equivalence", "outcome_distribution_equivalence")):
        raise ValueError("statistical or outcome equivalence overclaim")
    _recheck_gate(binding)
    path_mapping = plan["prospective_binding"].get("read_only_path_mapping", {})
    def physical(path):
        path = Path(path)
        logical = path_mapping.get("logical_root")
        if logical:
            try:
                return Path(path_mapping["physical_root"]) / path.relative_to(logical)
            except ValueError:
                pass
        return path
    refs = {}
    def bind(path, expected=None):
        path = physical(path)
        sha = file_digest(path)
        if expected is not None and sha != expected:
            raise ValueError(f"bound file changed: {path}")
        refs[str(path.resolve())] = sha
        return sha
    bind(binding["path"], binding["sha256"])
    for name, supplied in (("v2_authorization", authorization_path), ("v2_contract", contract_path)):
        reference = binding.get("refs", {}).get(name, {})
        if not reference.get("path") or physical(reference["path"]).resolve() != Path(supplied).resolve():
            raise ValueError(f"gate {name} file reference differs")
        bind(supplied, reference.get("sha256"))
    for name in ("request_attachment", "g2_v2_verifier"):
        reference = binding.get(name, {})
        if not reference.get("path") or not reference.get("sha256"):
            raise ValueError(f"missing mandatory gate source binding: {name}")
        bind(reference["path"], reference["sha256"])
    review = plan.get("requirements_review_policy_binding", {})
    if review.get("human_review_authorization"):
        reference = review["human_review_authorization"]
        bind(reference["path"], reference["sha256"])
    for ref in binding.get("refs", {}).values():
        if isinstance(ref, dict) and "path" in ref and "sha256" in ref:
            bind(ref["path"], ref["sha256"])
    for path in (plan_path, gate_binding_path, authorization_path, contract_path, Path(__file__)):
        bind(path)
    refs.update(plan["prospective_binding"]["inputs"])
    result = dict(schema="d0-scientific-input-freeze-v2", d0_frozen=True,
        freeze_scope="scientific_inputs_and_measurement_contract", policy=POLICY,
        post_hoc_forensic_extension=True, plan_sha256=plan["plan_sha256"],
        source_commit=source_commit, candidate_parameters=params,
        task_ladder_message_specs=plan["task_ladder_message_specs"],
        stratified_sampling_frame=plan["stratified_sampling_frame"],
        canonical_unit_census=plan["canonical_unit_census"],
        measurement_gate_binding=binding, bound_files=refs,
        execution_authorized=False, execution_ready=False,
        execution_blockers=plan.get("execution_blockers", []),
        original_v1_gate_status="preserved_blocked",
        statistical_equivalence="not_established", outcome_distribution_equivalence="not_established",
        actor_requests=0, analyst_requests=0, scorer_invocations=0, tool_calls=0)
    return seal(result, "freeze_sha256")


def verify_freeze(path):
    result = read_json(path)
    verify_seal(result, "freeze_sha256")
    if result.get("freeze_scope") != "scientific_inputs_and_measurement_contract" or result.get("execution_authorized") is not False or result.get("execution_ready") is not False or "approved_plan_sha256" in result:
        raise ValueError("scientific freeze scope or execution authority changed")
    for filename, expected in result["bound_files"].items():
        if file_digest(Path(filename)) != expected:
            raise ValueError(f"frozen input changed: {filename}")
    _recheck_gate(result["measurement_gate_binding"])
    return result


def produce(*, output, **kwargs):
    result = build_freeze(**kwargs)
    output = Path(output)
    if output.exists():
        if read_json(output) != result:
            raise ValueError("refusing to overwrite a different D0 freeze")
        verify_freeze(output)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2)
                stream.write("\n")
        except FileExistsError:
            if read_json(output) != result:
                raise ValueError("refusing to overwrite a different D0 freeze")
            verify_freeze(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path)
    for name in ("plan", "gate-binding", "authorization", "contract", "output"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    if args.verify:
        result = verify_freeze(args.verify)
    else:
        if any(getattr(args, name) is None for name in ("plan", "gate_binding", "authorization", "contract", "output")):
            parser.error("all five input/output paths are required")
        result = produce(plan_path=args.plan, gate_binding_path=args.gate_binding,
            authorization_path=args.authorization, contract_path=args.contract,
            output=args.output, source_commit=args.source_commit)
    print(result["freeze_sha256"])


if __name__ == "__main__":
    main()
