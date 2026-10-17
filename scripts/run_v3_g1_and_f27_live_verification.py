#!/usr/bin/env python3
"""Execute V3-G1 complete 4-position + D/R cold-resume verification and F27 live SWE-bench Pro verification."""

from __future__ import annotations

import concurrent.futures
import copy
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
L1_ROOT = REPO_ROOT / "L1_sandbox" / "C3_Experimental_Sandbox"
sys.path.insert(0, str(L1_ROOT / "src"))

from c3_sandbox.l1.core.config import resolve as resolve_config
from c3_sandbox.l1.core.orchestrator import run as l1_run
from c3_sandbox.l1.evidence.live import verify as l1_verify
from c3_sandbox.l1.storage import read_json


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json_obj(obj: Any) -> str:
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _verify_resume_pair(parent_dir: Path, child_dir: Path, cp_id: str, req_idx: int) -> dict[str, Any]:
    v_child = l1_verify(child_dir)
    req_name = f"r{req_idx:06d}"
    parent_req = parent_dir / "raw" / "model" / req_name / "request.json"
    child_req = child_dir / "raw" / "model" / req_name / "request.json"
    parent_sha = _sha256_file(parent_req)
    child_sha = _sha256_file(child_req)

    parent_cp = read_json(parent_dir / "checkpoints" / f"{cp_id}.json")
    parent_snap = read_json(parent_dir / "states" / "snapshots" / f"{parent_cp['snapshot_id']}.json")
    child_restored_snap = read_json(child_dir / "lineage" / "parent_snapshot.json")
    child_manifest = read_json(child_dir / "manifest.json")

    return {
        "parent_dir": str(parent_dir.relative_to(REPO_ROOT)),
        "child_dir": str(child_dir.relative_to(REPO_ROOT)),
        "checkpoint_id": cp_id,
        "resumed_request_id": req_name,
        "child_episode_id": child_manifest.get("episode_id"),
        "c3_l1_verified": bool(v_child.get("verified")),
        "parent_request_sha256": parent_sha,
        "child_request_sha256": child_sha,
        "byte_identical_first_request": parent_sha == child_sha,
        "parent_snapshot_content_sha256": parent_snap["content_sha256"],
        "child_restored_snapshot_content_sha256": child_restored_snap["content_sha256"],
        "workspace_state_identical": parent_snap["content_sha256"] == child_restored_snap["content_sha256"],
    }


def run_v3_g1_verification() -> dict[str, Any]:
    pilot_root = L1_ROOT / "runs" / "qwen36_pi_p1_pilot_20261017"
    resume_root = pilot_root / "v3_g1_cold_resume"
    resume_root.mkdir(parents=True, exist_ok=True)
    base_cfg_path = L1_ROOT / "configs" / "l1" / "officeval.qwen36.pi.t1.json"

    results: dict[str, Any] = {
        "positions": {},
        "treatment_arms": {},
    }

    # 1. Position (a): Pre-request c_0 (cp000000, model_requests=0) on officeval_090.r00
    print("[V3-G1] Position (a): Pre-request c_0 (cp000000) on officeval_090.r00...", flush=True)
    parent_c0 = pilot_root / "episodes" / "officeval_090.r00"
    child_c0 = resume_root / "officeval_090.r00_cp000000"
    if not (child_c0 / "manifest.json").exists() or not l1_verify(child_c0).get("verified"):
        if child_c0.exists():
            shutil.rmtree(child_c0)
        cfg_c0 = read_json(parent_c0 / "raw" / "resolved_config.json")
        l1_run(cfg_c0, child_c0, parent=parent_c0, checkpoint_id="cp000000")
    results["positions"]["c0_pre_request_cp000000"] = _verify_resume_pair(parent_c0, child_c0, "cp000000", 0)
    print(f"  -> c0 byte_identical={results['positions']['c0_pre_request_cp000000']['byte_identical_first_request']}", flush=True)

    # 2. Position (b): Mid-trajectory cp000010 on officeval_060.r00
    print("[V3-G1] Position (b): Mid-trajectory cp000010 on officeval_060.r00...", flush=True)
    parent_c10 = pilot_root / "episodes" / "officeval_060.r00"
    child_c10 = resume_root / "officeval_060.r00_cp000010"
    results["positions"]["mid_trajectory_cp000010"] = _verify_resume_pair(parent_c10, child_c10, "cp000010", 10)
    print(f"  -> cp000010 byte_identical={results['positions']['mid_trajectory_cp000010']['byte_identical_first_request']}", flush=True)

    # 3. Position (c) & (e): Tail checkpoint cp000014 on officeval_070.r01 + Arms P, D, R in parallel
    print("[V3-G1] Position (c) & (e): Tail checkpoint cp000014 (P, D, R) on officeval_070.r01...", flush=True)
    parent_tail = pilot_root / "episodes" / "officeval_070.r01"
    parent_tail_cfg = read_json(parent_tail / "raw" / "resolved_config.json")

    arm_d_msg = (
        "CRITICAL DELIVERY REMINDER: You are near the request budget limit. "
        "Immediately write or copy your best output file to the required submission path "
        "(`/workspace/submission/output.docx`) and verify it exists and is non-empty before stopping."
    )
    arm_r_msg = (
        "CRITICAL RUBRIC & DELIVERY REMINDER: Ensure `/workspace/submission/output.docx` satisfies "
        "all required document structure, table, and formatting rules, and save the final file now."
    )

    arm_specs = [
        ("P_plain", None, resume_root / "officeval_070.r01_cp000014_P"),
        ("D_deliver", arm_d_msg, resume_root / "officeval_070.r01_cp000014_D"),
        ("R_rule", arm_r_msg, resume_root / "officeval_070.r01_cp000014_R"),
    ]

    def _run_arm(spec: tuple[str, str | None, Path]) -> None:
        arm_name, msg, out_dir = spec
        if (out_dir / "manifest.json").exists() and l1_verify(out_dir).get("verified"):
            return
        if out_dir.exists():
            shutil.rmtree(out_dir)
        cfg = copy.deepcopy(parent_tail_cfg)
        if msg is not None:
            cfg["agent"]["intervention"] = {
                "at_request": 15,
                "message": msg,
            }
        l1_run(cfg, out_dir, parent=parent_tail, checkpoint_id="cp000014")

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(_run_arm, s) for s in arm_specs]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    child_tail_p = resume_root / "officeval_070.r01_cp000014_P"
    results["positions"]["tail_checkpoint_cp000014"] = _verify_resume_pair(parent_tail, child_tail_p, "cp000014", 14)
    print(f"  -> cp000014 (P) byte_identical={results['positions']['tail_checkpoint_cp000014']['byte_identical_first_request']}", flush=True)

    p_req_obj = read_json(child_tail_p / "raw" / "model" / "r000014" / "request.json")
    p_snap_obj = read_json(child_tail_p / "lineage" / "parent_snapshot.json")

    for arm_name, arm_msg, arm_dir in arm_specs[1:]:
        v_arm = l1_verify(arm_dir)
        arm_manifest = read_json(arm_dir / "manifest.json")
        arm_req_obj = read_json(arm_dir / "raw" / "model" / "r000014" / "request.json")
        arm_snap_obj = read_json(arm_dir / "lineage" / "parent_snapshot.json")

        prefix_identical = (
            arm_req_obj.get("model") == p_req_obj.get("model")
            and arm_req_obj.get("tools") == p_req_obj.get("tools")
            and arm_req_obj.get("parameters") == p_req_obj.get("parameters")
            and arm_req_obj.get("messages", [])[:-1] == p_req_obj.get("messages", [])
        )
        last_msg = arm_req_obj.get("messages", [])[-1]
        last_msg_matches = (
            isinstance(last_msg, dict)
            and last_msg.get("role") == "user"
            and last_msg.get("content") == arm_msg
        )
        snapshot_identical = arm_snap_obj.get("content_sha256") == p_snap_obj.get("content_sha256")

        results["treatment_arms"][arm_name] = {
            "parent_dir": str(parent_tail.relative_to(REPO_ROOT)),
            "child_dir": str(arm_dir.relative_to(REPO_ROOT)),
            "checkpoint_id": "cp000014",
            "resumed_request_id": "r000014",
            "child_episode_id": arm_manifest.get("episode_id"),
            "c3_l1_verified": bool(v_arm.get("verified")),
            "prefix_messages_sha256": _sha256_json_obj(arm_req_obj.get("messages", [])[:-1]),
            "plain_messages_sha256": _sha256_json_obj(p_req_obj.get("messages", [])),
            "prefix_and_params_identical_to_P": prefix_identical,
            "differs_only_by_single_user_message": prefix_identical and last_msg_matches,
            "pre_action_snapshot_identical_to_P": snapshot_identical,
            "injected_message_sha256": hashlib.sha256((arm_msg or "").encode("utf-8")).hexdigest(),
        }
        print(
            f"  -> {arm_name}: differs_only_by_single_user_message={prefix_identical and last_msg_matches}, "
            f"snapshot_identical={snapshot_identical}, verified={v_arm.get('verified')}",
            flush=True,
        )

    # 4. Position (d): Post-compaction checkpoint
    print("[V3-G1] Position (d): Live post-compaction parent + cold-resume...", flush=True)
    comp_parent_dir = resume_root / "officeval_060_compaction_parent"
    comp_child_dir = resume_root / "officeval_060_compaction_resume"
    if not (comp_parent_dir / "manifest.json").exists() or not l1_verify(comp_parent_dir).get("verified"):
        if comp_parent_dir.exists():
            shutil.rmtree(comp_parent_dir)
        cfg_comp = resolve_config(base_cfg_path)
        cfg_comp["benchmark"]["task_id"] = "officeval_060"
        cfg_comp["model"]["parameters"]["seed"] = 106099
        cfg_comp["budget"]["max_model_requests"] = 4
        cfg_comp["budget"]["max_tool_calls"] = 16
        cfg_comp["agent"]["compaction"] = {
            "enabled": True,
            "context_window_tokens": 1800,
            "reserve_tokens": 500,
            "keep_recent_tokens": 450,
        }
        l1_run(cfg_comp, comp_parent_dir)

    # Locate first checkpoint where _pi_compaction has >= 1 event and a subsequent request exists
    post_comp_cp_id = None
    post_comp_req_idx = None
    post_comp_events = None
    for cp_idx in [1, 2, 3]:
        cp_id = f"cp{cp_idx:06d}"
        cp_file = comp_parent_dir / "checkpoints" / f"{cp_id}.json"
        if cp_file.exists():
            cp_data = read_json(cp_file)
            pi_comp = (cp_data.get("conversation") or {}).get("_pi_compaction")
            comp_list = pi_comp if isinstance(pi_comp, list) else (pi_comp.get("events", []) if isinstance(pi_comp, dict) else [])
            req_idx = int(cp_data.get("budget_used", {}).get("model_requests", 0))
            if len(comp_list) >= 1 and (comp_parent_dir / "raw" / "model" / f"r{req_idx:06d}" / "request.json").exists():
                post_comp_cp_id = cp_id
                post_comp_req_idx = req_idx
                post_comp_events = comp_list
                break

    if post_comp_cp_id is None or post_comp_req_idx is None:
        raise RuntimeError("Compaction did not trigger before request 4 in officeval_060_compaction_parent")

    if not (comp_child_dir / "manifest.json").exists() or not l1_verify(comp_child_dir).get("verified"):
        if comp_child_dir.exists():
            shutil.rmtree(comp_child_dir)
        parent_comp_cfg = read_json(comp_parent_dir / "raw" / "resolved_config.json")
        l1_run(parent_comp_cfg, comp_child_dir, parent=comp_parent_dir, checkpoint_id=post_comp_cp_id)

    comp_pair = _verify_resume_pair(comp_parent_dir, comp_child_dir, post_comp_cp_id, post_comp_req_idx)
    comp_pair["compaction_count"] = len(post_comp_events or [])
    comp_pair["compaction_events"] = post_comp_events or []
    results["positions"]["post_compaction"] = comp_pair
    print(
        f"  -> post_compaction ({post_comp_cp_id} -> r{post_comp_req_idx:06d}): "
        f"byte_identical={comp_pair['byte_identical_first_request']}, count={comp_pair['compaction_count']}",
        flush=True,
    )

    return results


def run_f27_live_swebench_verification() -> dict[str, Any]:
    print("[F27] Running live SWE-bench Pro trajectory (task2_s42) on Qwen/Qwen3.6-27B with persistent /app container...", flush=True)
    cfg_path = L1_ROOT / "configs" / "l1" / "swebench_pro.qwen36.task2_s42.json"
    cfg = resolve_config(cfg_path)
    cfg["model"]["reported_revision"] = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
    cfg["model"]["weight_digest"] = "sha256:a8ad2c26fb707ff8c245806315b03e3b4b74595528492423af5dae0ce39b4d9b"
    cfg["model"]["tokenizer_digest"] = "sha256:5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
    cfg["budget"]["max_model_requests"] = 12
    cfg["budget"]["max_tool_calls"] = 18

    out_dir = L1_ROOT / "runs" / "swebench_pro_f27_live_20261017" / "task2_s42_f27"
    if not (out_dir / "manifest.json").exists() or not l1_verify(out_dir).get("verified"):
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        l1_run(cfg, out_dir)

    v_res = l1_verify(out_dir)
    manifest = read_json(out_dir / "manifest.json")
    official_grade = read_json(out_dir / "official_grade.json")

    # Inspect checkpoints and their snapshots for patch.diff and submission/patch.diff
    cp_patches: list[dict[str, Any]] = []
    cp_index = read_json(out_dir / "checkpoints" / "index.json")
    for cp_entry in cp_index.get("checkpoints", []):
        cp_id = cp_entry["checkpoint_id"]
        cp_data = read_json(out_dir / "checkpoints" / f"{cp_id}.json")
        snap_id = cp_data["snapshot_id"]
        snap_data = read_json(out_dir / "states" / "snapshots" / f"{snap_id}.json")
        files_by_path = {e["path"]: e for e in snap_data.get("files", [])}
        app_diff_entry = files_by_path.get("patch.diff") or files_by_path.get("_swebench_app_patch.diff")
        sub_diff_entry = files_by_path.get("submission/patch.diff")
        cp_patches.append({
            "checkpoint_id": cp_id,
            "snapshot_id": snap_id,
            "model_requests": cp_data.get("budget_used", {}).get("model_requests", 0),
            "has_swebench_app_patch": app_diff_entry is not None and int(app_diff_entry.get("size_bytes", 0)) > 0,
            "swebench_app_patch_bytes": int(app_diff_entry.get("size_bytes", 0)) if app_diff_entry else 0,
            "swebench_app_patch_sha256": app_diff_entry.get("sha256") if app_diff_entry else None,
            "has_submission_patch": sub_diff_entry is not None and int(sub_diff_entry.get("size_bytes", 0)) > 0,
            "submission_patch_bytes": int(sub_diff_entry.get("size_bytes", 0)) if sub_diff_entry else 0,
            "submission_patch_sha256": sub_diff_entry.get("sha256") if sub_diff_entry else None,
        })

    tests = official_grade.get("official_raw", {}).get("test_results", {}).get("tests", [])
    test_status_map = {t["name"]: t["status"] for t in tests if "name" in t and "status" in t}
    f2p_names = [
        "test/units/plugins/lookup/test_password.py::TestParseContent::test",
        "test/units/plugins/lookup/test_password.py::TestParseContent::test_with_salt",
        "test/units/plugins/lookup/test_password.py::TestParseContent::test_empty_password_file",
        "test/units/plugins/lookup/test_password.py::TestParseContent::test_with_salt_and_ident",
    ]
    f2p_passed = [n for n in f2p_names if test_status_map.get(n) == "PASSED"]

    return {
        "run_dir": str(out_dir.relative_to(REPO_ROOT)),
        "episode_id": manifest.get("episode_id"),
        "c3_l1_verified": bool(v_res.get("verified")),
        "status": manifest.get("status"),
        "checkpoint_count": len(cp_patches),
        "checkpoints_with_nonempty_app_patch": sum(1 for c in cp_patches if c["has_swebench_app_patch"]),
        "checkpoints_with_submission_patch": sum(1 for c in cp_patches if c["has_submission_patch"]),
        "fail_to_pass_total": len(f2p_names),
        "fail_to_pass_passed_count": len(f2p_passed),
        "fail_to_pass_passed_tests": f2p_passed,
        "total_tests_run": len(tests),
        "total_tests_passed": sum(1 for t in tests if t.get("status") == "PASSED"),
        "total_tests_failed": sum(1 for t in tests if t.get("status") != "PASSED"),
        "checkpoint_details": cp_patches,
    }


def main() -> None:
    t0 = time.time()
    v3_g1 = run_v3_g1_verification()
    f27 = run_f27_live_swebench_verification()
    payload = {
        "schema_version": "0.5.0",
        "generated_at": "2026-10-17T19:00:00+08:00",
        "elapsed_seconds": round(time.time() - t0, 2),
        "qwen36_frozen_identity": {
            "model": "Qwen/Qwen3.6-27B",
            "hf_snapshot_revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
            "weight_digest": "sha256:a8ad2c26fb707ff8c245806315b03e3b4b74595528492423af5dae0ce39b4d9b",
            "tokenizer_digest": "sha256:5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
            "config_digest": "sha256:69db4eb7196bc8190813231b3018ca05d8c2e3abc7b1af19d55c157af44a9d9c",
        },
        "v3_g1_verification": v3_g1,
        "f27_swebench_live_verification": f27,
    }
    out_json = (
        REPO_ROOT
        / "L2_Behavioral Event Localization"
        / "L2_Sandbox"
        / "docs"
        / "v3_g1_and_f27_live_verification.json"
    )
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] Wrote {out_json} in {payload['elapsed_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
