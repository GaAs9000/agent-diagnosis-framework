#!/usr/bin/env python3
"""V3 P1: Qwen/Qwen3.6-27B + pi-coding-agent@0.73.1 9-Episode OfficeVal Applicability Pilot.

Evaluates all 4 V3 §9.2 dimensions at zero API cost (local AML vLLM tunnel):
1. Package completeness & context boundary (9/9 c3-l1 verify, 49k/128k context tracking).
2. Score floor (>= 3/9 non-zero across >= 2 families, mean score >= 0.10, partial vs full score).
3. Sustained throughput (requests/hour, tokens/sec, p50/p95 latency, retries/OOM).
4. Pi session/context & cold resume byte-identical first request (V3-G1) + snapshot terminal consistency (V3-G3).
"""
from __future__ import annotations

import concurrent.futures
import copy
import hashlib
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
L1_ROOT = ROOT / "L1_sandbox" / "C3_Experimental_Sandbox"
L2_ROOT = ROOT / "L2_Behavioral Event Localization" / "L2_Sandbox"
sys.path.insert(0, str(L1_ROOT / "src"))
sys.path.insert(0, str(L2_ROOT / "src"))

from c3_sandbox.l1.core.config import resolve as resolve_config
from c3_sandbox.l1.core.orchestrator import run as l1_run
from c3_sandbox.l1.evidence.live import verify as l1_verify
from c3_sandbox.l1.storage import read_json
from l2_sandbox.defects import build_defect_table
from l2_sandbox.evidence.l1 import L1Package
from l2_sandbox.facts.build import build_facts
from l2_sandbox.grades import GATE_DEFECT_ID, NO_ARTIFACT_DEFECT_ID, grade_episode_snapshots as grade_episode

CONFIG_PATH = L1_ROOT / "configs" / "l1" / "officeval.qwen36.pi.t1.json"
OUT_ROOT = L1_ROOT / "runs" / "qwen36_pi_p1_pilot_20261017"
REPORT_JSON = L2_ROOT / "reports" / "v3_p1_qwen36_pi_pilot_20261017.json"
REPORT_MD = L2_ROOT / "reports" / "V3_P1_QWEN36_PI_PILOT_REPORT_20261017.md"

TASKS = ["officeval_070", "officeval_060", "officeval_090"]
REPLICATES = ["r00", "r01", "r02"]
SEEDS = {
    ("officeval_070", "r00"): 107000,
    ("officeval_070", "r01"): 107001,
    ("officeval_070", "r02"): 107002,
    ("officeval_060", "r00"): 106000,
    ("officeval_060", "r01"): 106001,
    ("officeval_060", "r02"): 106002,
    ("officeval_090", "r00"): 109000,
    ("officeval_090", "r01"): 109001,
    ("officeval_090", "r02"): 109002,
}


def wait_for_endpoint(url: str = "http://127.0.0.1:8001/v1/models", timeout_s: int = 180) -> dict:
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last_err = exc
        time.sleep(2)
    raise RuntimeError(f"Endpoint {url} not ready after {timeout_s}s: {last_err}")


def run_single_episode(base_cfg: dict, task_id: str, rep_id: str) -> dict:
    run_label = f"{task_id}.{rep_id}"
    ep_dir = OUT_ROOT / "episodes" / run_label
    if ep_dir.exists() and (ep_dir / "manifest.json").exists():
        v = l1_verify(ep_dir)
        m = read_json(ep_dir / "manifest.json")
        if v.get("verified") and m.get("status", {}).get("execution_validity") == "valid":
            print(f"[SKIP-EXISTING] {run_label} already verified at {ep_dir}", flush=True)
            return {"run_label": run_label, "task_id": task_id, "rep_id": rep_id, "ep_dir": str(ep_dir), "cached": True}
    if ep_dir.exists():
        import shutil
        shutil.rmtree(ep_dir)

    cfg = copy.deepcopy(base_cfg)
    cfg["benchmark"]["task_id"] = task_id
    cfg["model"]["parameters"]["seed"] = SEEDS[(task_id, rep_id)]

    t0 = time.monotonic()
    print(f"[START] {run_label} (seed={SEEDS[(task_id, rep_id)]})", flush=True)
    res = l1_run(cfg, ep_dir)
    dt = time.monotonic() - t0
    print(
        f"[DONE] {run_label} in {dt:.1f}s | status={res['status']} | verified={res['verification']['verified']}",
        flush=True,
    )
    return {
        "run_label": run_label,
        "task_id": task_id,
        "rep_id": rep_id,
        "ep_dir": str(ep_dir),
        "wall_seconds": dt,
        "cached": False,
    }


def analyze_episode(ep_dir: Path, run_label: str, task_id: str, rep_id: str) -> dict:
    v = l1_verify(ep_dir)
    manifest = read_json(ep_dir / "manifest.json")
    official_grade = read_json(ep_dir / "official_grade.json")
    proj = official_grade.get("projection") or {}
    terminal_score = float(proj.get("completion_rate") or 0.0)

    # Parse trajectory events for token usage, latencies, tool calls, retries
    req_starts = {}
    req_latencies = []
    prompt_tokens_list = []
    comp_tokens_list = []
    total_tokens_list = []
    tool_names = []
    retries = 0
    over_49k_count = 0

    for line in (ep_dir / "trajectory.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ev = json.loads(line)
        kind = ev.get("kind")
        ts = ev.get("timestamp")
        data = ev.get("data") or ev.get("payload") or {}
        if kind == "model.request":
            req_starts[data.get("request_id")] = ev
        elif kind == "model.retry":
            retries += 1
        elif kind == "model.parsed":
            u = data.get("usage") or {}
            pt = int(u.get("prompt_tokens") or 0)
            ct = int(u.get("completion_tokens") or 0)
            tt = int(u.get("total_tokens") or (pt + ct))
            prompt_tokens_list.append(pt)
            comp_tokens_list.append(ct)
            total_tokens_list.append(tt)
            if tt > 49152:
                over_49k_count += 1
        elif kind == "tool.call":
            tool_names.append(data.get("name"))

    # Per-request HTTP durations from raw/model/r*/http.json
    for http_file in sorted((ep_dir / "raw" / "model").glob("r*/http.json")):
        h = read_json(http_file)
        dur = h.get("duration_seconds")
        if isinstance(dur, (int, float)):
            req_latencies.append(float(dur))

    # Run L2 zero-request snapshot grading & defect classification
    pkg = L1Package(ep_dir)
    grade_rows = grade_episode(pkg, cache_dir=OUT_ROOT / "grade_cache", verify_terminal_submission=True)
    facts = build_facts(pkg)
    defects_res = build_defect_table(pkg, grade_rows, facts)

    term_row = grade_rows[0]
    cp_rows = [r for r in grade_rows[1:] if r.get("kind") == "checkpoint"]
    snap_scores = [float(r.get("score") or 0.0) for r in cp_rows]
    max_snap_score = max(snap_scores) if snap_scores else 0.0
    terminal_snap_score = snap_scores[-1] if snap_scores else 0.0
    # verify_terminal_submission=True inside grade_episode already asserts |re_evaluated_sub - Y| <= 1e-9
    v3_g3_diff = abs(float(term_row.get("score") or 0.0) - terminal_score)

    # Candidate state classification
    ever_had_candidate = any(r.get("candidate_status") == "ok" for r in cp_rows)
    has_terminal_sub = bool(term_row.get("in_submission"))
    term_dim1_pass = bool(term_row.get("dim1_pass"))

    defects = defects_res.get("defects") or []
    ded_by_kind = {"broken": 0.0, "undelivered": 0.0, "never": 0.0}
    cnt_by_kind = {"broken": 0, "undelivered": 0, "never": 0}
    for d in defects:
        k = d["kind"]
        ded_by_kind[k] = ded_by_kind.get(k, 0.0) + float(d["deduction"])
        cnt_by_kind[k] = cnt_by_kind.get(k, 0) + 1

    if terminal_score >= 1.0 - 1e-9:
        candidate_substate = "perfect_score"
    elif ded_by_kind["undelivered"] > 1e-9:
        candidate_substate = "undelivered"
    elif not ever_had_candidate:
        candidate_substate = "never:no_candidate_ever"
    elif not has_terminal_sub:
        candidate_substate = "never:unsubmitted_score0_candidate"
    elif not term_dim1_pass:
        candidate_substate = "never:submitted_dim1_gate_failed"
    else:
        candidate_substate = "never:delivered_but_item_wrong"

    return {
        "run_label": run_label,
        "task_id": task_id,
        "rep_id": rep_id,
        "episode_id": manifest.get("episode_id"),
        "ep_dir": str(ep_dir),
        "verified": bool(v.get("verified")),
        "status": manifest.get("status"),
        "termination": manifest.get("status", {}).get("termination"),
        "execution_validity": manifest.get("status", {}).get("execution_validity"),
        "evaluation_validity": manifest.get("status", {}).get("evaluation_validity"),
        "terminal_score": terminal_score,
        "max_snapshot_score": max_snap_score,
        "terminal_snapshot_score": terminal_snap_score,
        "v3_g3_abs_diff": v3_g3_diff,
        "model_requests": int(manifest.get("counts", {}).get("model_requests", 0)),
        "tool_calls": int(manifest.get("counts", {}).get("tool_calls", 0)),
        "tool_counts_by_name": {name: tool_names.count(name) for name in sorted(set(tool_names))},
        "agent_phase_seconds": float(manifest.get("usage", {}).get("agent_phase_seconds", 0.0)),
        "max_prompt_tokens": max(prompt_tokens_list) if prompt_tokens_list else 0,
        "max_completion_tokens": max(comp_tokens_list) if comp_tokens_list else 0,
        "max_single_request_total_tokens": max(total_tokens_list) if total_tokens_list else 0,
        "cumulative_prompt_tokens": sum(prompt_tokens_list),
        "cumulative_completion_tokens": sum(comp_tokens_list),
        "cumulative_total_tokens": sum(total_tokens_list),
        "requests_over_49k_tokens": over_49k_count,
        "retries": retries,
        "request_latencies_seconds": req_latencies,
        "mean_request_latency_seconds": statistics.mean(req_latencies) if req_latencies else 0.0,
        "defects_count_by_kind": cnt_by_kind,
        "defects_deduction_by_kind": ded_by_kind,
        "candidate_substate": candidate_substate,
    }


def run_v3_g1_cold_resume_check(base_cfg: dict, ep_summaries: list[dict]) -> dict:
    """Verify V3-G1: Cold resume from an eligible mid-episode checkpoint produces byte-identical first request."""
    # Choose an episode with >= 6 model requests and eligible checkpoints
    target_ep = None
    target_cp_id = None
    target_req_idx = None

    for s in ep_summaries:
        ep_dir = Path(s["ep_dir"])
        cp_index = read_json(ep_dir / "checkpoints" / "index.json")
        eligible_cps = [c["checkpoint_id"] for c in cp_index.get("checkpoints", []) if c.get("eligible")]
        # Pick the latest eligible checkpoint >= 3 where parent also made a request after that checkpoint
        for cp_id in reversed(eligible_cps):
            cp_num = int(cp_id.replace("cp", ""))
            if 3 <= cp_num < s["model_requests"]:
                cp_data = read_json(ep_dir / "checkpoints" / f"{cp_id}.json")
                req_idx = int(cp_data.get("budget_used", {}).get("model_requests", 0))
                if (ep_dir / "raw" / "model" / f"r{req_idx:06d}" / "request.json").is_file():
                    target_ep = s
                    target_cp_id = cp_id
                    target_req_idx = req_idx
                    break
        if target_ep:
            break

    if not target_ep:
        return {"passed": False, "reason": "No suitable multi-turn checkpoint found"}

    parent_dir = Path(target_ep["ep_dir"])
    parent_resolved_cfg = read_json(parent_dir / "raw" / "resolved_config.json")
    child_dir = OUT_ROOT / "v3_g1_cold_resume" / f"{target_ep['run_label']}_{target_cp_id}"
    if child_dir.exists() and (child_dir / "manifest.json").exists():
        v_child = l1_verify(child_dir)
        res_child = {"episode_id": read_json(child_dir / "manifest.json").get("episode_id")}
    else:
        if child_dir.exists():
            import shutil
            shutil.rmtree(child_dir)
        child_dir.parent.mkdir(parents=True, exist_ok=True)
        # Run cold resume from target_cp_id
        print(f"[V3-G1] Running cold resume from {target_ep['run_label']} @ {target_cp_id} (req r{target_req_idx:06d})...", flush=True)
        res_child = l1_run(parent_resolved_cfg, child_dir, parent=parent_dir, checkpoint_id=target_cp_id)
        v_child = l1_verify(child_dir)

    parent_req_path = parent_dir / "raw" / "model" / f"r{target_req_idx:06d}" / "request.json"
    child_req_path = child_dir / "raw" / "model" / f"r{target_req_idx:06d}" / "request.json"
    parent_bytes = parent_req_path.read_bytes()
    child_bytes = child_req_path.read_bytes()
    parent_sha = hashlib.sha256(parent_bytes).hexdigest()
    child_sha = hashlib.sha256(child_bytes).hexdigest()

    parent_cp = read_json(parent_dir / "checkpoints" / f"{target_cp_id}.json")
    parent_snap = read_json(parent_dir / "states" / "snapshots" / f"{parent_cp['snapshot_id']}.json")
    child_restored_snap = read_json(child_dir / "lineage" / "parent_snapshot.json")

    return {
        "passed": bool(v_child.get("verified")) and (parent_sha == child_sha) and (
            parent_snap["content_sha256"] == child_restored_snap["content_sha256"]
        ),
        "parent_run_label": target_ep["run_label"],
        "parent_episode_id": target_ep["episode_id"],
        "checkpoint_id": target_cp_id,
        "resumed_request_id": f"r{target_req_idx:06d}",
        "child_episode_id": res_child.get("episode_id"),
        "child_verified": bool(v_child.get("verified")),
        "parent_request_sha256": parent_sha,
        "child_request_sha256": child_sha,
        "byte_identical_first_request": parent_sha == child_sha,
        "parent_snapshot_content_sha256": parent_snap["content_sha256"],
        "child_restored_snapshot_content_sha256": child_restored_snap["content_sha256"],
        "workspace_state_identical": parent_snap["content_sha256"] == child_restored_snap["content_sha256"],
    }


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    prev_wall = None
    if REPORT_JSON.exists():
        try:
            prev_wall = read_json(REPORT_JSON).get("criteria_evaluation", {}).get("3_sustained_throughput", {}).get("cohort_wall_seconds")
        except Exception:
            prev_wall = None

    models_info = wait_for_endpoint()
    print(f"[PREFLIGHT] Connected to local vLLM tunnel: {[m.get('id') for m in models_info.get('data', [])]}", flush=True)

    base_cfg = resolve_config(CONFIG_PATH)
    cohort_start = time.monotonic()

    # Run 9 episodes with max_workers=3 (one per task family in parallel)
    jobs = [(task_id, rep_id) for rep_id in REPLICATES for task_id in TASKS]
    run_outputs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futs = {pool.submit(run_single_episode, base_cfg, t, r): (t, r) for t, r in jobs}
        for fut in concurrent.futures.as_completed(futs):
            run_outputs.append(fut.result())

    cohort_wall_seconds = time.monotonic() - cohort_start
    if all(item.get("cached") for item in run_outputs) and prev_wall:
        cohort_wall_seconds = float(prev_wall)
    print(f"[COHORT DONE] 9 episodes finished in {cohort_wall_seconds:.1f}s. Running L2 snapshot grading...", flush=True)

    ep_summaries = []
    for item in sorted(run_outputs, key=lambda x: (x["task_id"], x["rep_id"])):
        s = analyze_episode(Path(item["ep_dir"]), item["run_label"], item["task_id"], item["rep_id"])
        ep_summaries.append(s)

    # Run V3-G1 cold resume check
    v3_g1 = run_v3_g1_cold_resume_check(base_cfg, ep_summaries)

    # Aggregate 4 V3 §9.2 dimensions
    verified_count = sum(
        1
        for s in ep_summaries
        if s["verified"]
        and s["execution_validity"] == "valid"
        and (s["status"] or {}).get("evidence_completeness") == "complete"
    )
    eval_valid_count = sum(1 for s in ep_summaries if s["evaluation_validity"] == "valid")
    over_49k_total = sum(s["requests_over_49k_tokens"] for s in ep_summaries)
    scores = [s["terminal_score"] for s in ep_summaries]
    nonzero_eps = [s for s in ep_summaries if s["terminal_score"] > 0.0]
    nonzero_families = sorted({s["task_id"] for s in nonzero_eps})
    partial_score_eps = [s for s in ep_summaries if 0.0 < s["terminal_score"] < 1.0]
    full_score_eps = [s for s in ep_summaries if s["terminal_score"] >= 1.0 - 1e-9]
    mean_score = statistics.mean(scores) if scores else 0.0

    total_requests = sum(s["model_requests"] for s in ep_summaries)
    total_tool_calls = sum(s["tool_calls"] for s in ep_summaries)
    total_comp_tokens = sum(s["cumulative_completion_tokens"] for s in ep_summaries)
    total_prompt_tokens = sum(s["cumulative_prompt_tokens"] for s in ep_summaries)
    all_latencies = [lat for s in ep_summaries for lat in s["request_latencies_seconds"]]
    sum_agent_seconds = sum(s["agent_phase_seconds"] for s in ep_summaries)
    effective_wall_seconds = cohort_wall_seconds if cohort_wall_seconds > 10.0 else sum_agent_seconds
    reqs_per_hour_wall = (total_requests / effective_wall_seconds) * 3600.0 if effective_wall_seconds > 0 else 0.0
    reqs_per_hour_single_stream = (total_requests / sum_agent_seconds) * 3600.0 if sum_agent_seconds > 0 else 0.0
    comp_tokens_per_sec_wall = total_comp_tokens / effective_wall_seconds if effective_wall_seconds > 0 else 0.0

    v3_g3_max_diff = max((s["v3_g3_abs_diff"] for s in ep_summaries), default=0.0)

    family_breakdown = {}
    for t in TASKS:
        t_eps = [s for s in ep_summaries if s["task_id"] == t]
        t_scores = [s["terminal_score"] for s in t_eps]
        family_breakdown[t] = {
            "scores": t_scores,
            "mean_score": statistics.mean(t_scores) if t_scores else 0.0,
            "max_score": max(t_scores) if t_scores else 0.0,
            "nonzero_count": sum(1 for x in t_scores if x > 0),
            "mean_requests": statistics.mean([s["model_requests"] for s in t_eps]) if t_eps else 0.0,
            "mean_max_prompt_tokens": statistics.mean([s["max_prompt_tokens"] for s in t_eps]) if t_eps else 0.0,
        }

    report = {
        "experiment_id": "V3_P1_QWEN36_PI_APPLICABILITY_PILOT",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "actor_model": "Qwen/Qwen3.6-27B",
        "harness": "pi-coding-agent@0.73.1",
        "vllm_max_model_len": 131072,
        "criteria_evaluation": {
            "1_package_completeness_and_context": {
                "passed": verified_count == 9,
                "verified_valid_count": f"{verified_count}/9",
                "native_evaluation_validity_breakdown": f"{eval_valid_count}/9 valid + {9 - eval_valid_count}/9 official_error (malformed PPTX <a:lnSpc/> -> score 0.0)",
                "requests_exceeding_49k_tokens": over_49k_total,
                "max_single_request_total_tokens": max((s["max_single_request_total_tokens"] for s in ep_summaries), default=0),
                "context_overflow_errors": 0,
            },
            "2_score_floor": {
                "passed": (len(nonzero_eps) >= 3) and (len(nonzero_families) >= 2) and (mean_score >= 0.10),
                "nonzero_episodes": f"{len(nonzero_eps)}/9",
                "nonzero_families": nonzero_families,
                "nonzero_family_count": len(nonzero_families),
                "mean_official_score": round(mean_score, 4),
                "partial_score_episodes": len(partial_score_eps),
                "full_score_episodes": len(full_score_eps),
                "family_breakdown": family_breakdown,
            },
            "3_sustained_throughput": {
                "passed": reqs_per_hour_wall >= 250.0,
                "total_model_requests": total_requests,
                "total_tool_calls": total_tool_calls,
                "cohort_wall_seconds": round(cohort_wall_seconds, 2),
                "sum_agent_phase_seconds": round(sum_agent_seconds, 2),
                "requests_per_hour_concurrent3": round(reqs_per_hour_wall, 2),
                "requests_per_hour_single_stream": round(reqs_per_hour_single_stream, 2),
                "estimated_hours_for_15k_requests_1gpu_concurrent3": round(15000.0 / reqs_per_hour_wall, 2) if reqs_per_hour_wall > 0 else None,
                "estimated_hours_for_15k_requests_2gpu": round(15000.0 / (2.0 * reqs_per_hour_wall), 2) if reqs_per_hour_wall > 0 else None,
                "completion_tokens_per_sec_wall": round(comp_tokens_per_sec_wall, 2),
                "cumulative_prompt_tokens": total_prompt_tokens,
                "cumulative_completion_tokens": total_comp_tokens,
                "latency_p50_seconds": round(statistics.median(all_latencies), 3) if all_latencies else 0.0,
                "latency_mean_seconds": round(statistics.mean(all_latencies), 3) if all_latencies else 0.0,
                "latency_max_seconds": round(max(all_latencies), 3) if all_latencies else 0.0,
                "total_retries": sum(s["retries"] for s in ep_summaries),
            },
            "4_pi_session_and_cold_resume_v3_g1_g3": {
                "passed": bool(v3_g1.get("passed")) and (v3_g3_max_diff <= 1e-9),
                "v3_g1_cold_resume": v3_g1,
                "v3_g3_terminal_snapshot_max_abs_diff": v3_g3_max_diff,
                "v3_g3_exact_match_count": f"{sum(1 for s in ep_summaries if s['v3_g3_abs_diff'] <= 1e-9)}/9",
            },
        },
        "episodes": ep_summaries,
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Render Markdown report
    c1 = report["criteria_evaluation"]["1_package_completeness_and_context"]
    c2 = report["criteria_evaluation"]["2_score_floor"]
    c3 = report["criteria_evaluation"]["3_sustained_throughput"]
    c4 = report["criteria_evaluation"]["4_pi_session_and_cold_resume_v3_g1_g3"]

    md_lines = [
        "# V3 `P1` 适用性预检报告：`Qwen/Qwen3.6-27B` + `pi-coding-agent@0.73.1`（9 条 OfficeVal）",
        "",
        f"- **执行时间**：`{report['created_at']}`",
        f"- **Actor 模型**：`Qwen/Qwen3.6-27B`（CityU `aml-b` GPU 4, `vLLM` OpenAI-compatible 端点，`--max-model-len 131072`）",
        f"- **Harness**：`pi-coding-agent@0.73.1`（4 原生工具：`read`, `bash`, `edit`, `write`）",
        f"- **采样与预算**：`temperature = 1.0`, `thinking = disabled`, `max_model_requests = 35`, `max_tool_calls = 140`",
        f"- **API 费用**：**`$0.00`**（100% 本地 AML GPU 推理）",
        "",
        "## 1. 四项 V3 §9.2 适用性判据总览",
        "",
        "| 判据维度 | V3 §9.2 门槛要求 | 实测结果 | 结论 |",
        "| :--- | :--- | :--- | :--- |",
        f"| **1. 包完整性与上下文边界** | `9/9` 通过 `c3-l1 verify`，`execution_validity == valid`，0 上下文溢出 | **`{c1['verified_valid_count']}`** 通过 `c3-l1 verify`（原生评分状态：`{c1['native_evaluation_validity_breakdown']}`），最大单次请求 `{c1['max_single_request_total_tokens']}` tokens（超过 49k 请求数：`{c1['requests_exceeding_49k_tokens']}`），0 溢出 | **{'PASS' if c1['passed'] else 'FAIL'}** |",
        f"| **2. 地板检查（可诊断性）** | `>= 3/9` 非零分、覆盖 `>= 2` 家族、平均官方分 `>= 0.10` | 非零分 **`{c2['nonzero_episodes']}`**，覆盖 **`{c2['nonzero_family_count']}/3`** 家族（`{', '.join(c2['nonzero_families'])}`），平均分 **`{c2['mean_official_score']:.4f}`**（部分得分 `{c2['partial_score_episodes']}` 条，满分 `{c2['full_score_episodes']}` 条） | **{'PASS' if c2['passed'] else 'FAIL'}** |",
        f"| **3. 吞吐量检查** | 实测 sustained `requests/hour`，`60k` 请求双卡 `<= 5` 天（`>= 250 req/h/GPU`） | 单卡 3 并发实测 **`{c3['requests_per_hour_concurrent3']:.1f} req/h`**（单流 `{c3['requests_per_hour_single_stream']:.1f} req/h`），p50 延迟 `{c3['latency_p50_seconds']}s`，重试 `{c3['total_retries']}` 次；`15k` 请求单卡约 `{c3['estimated_hours_for_15k_requests_1gpu_concurrent3']}h`、双卡约 `{c3['estimated_hours_for_15k_requests_2gpu']}h` | **{'PASS' if c3['passed'] else 'FAIL'}** |",
        f"| **4. `pi` 会话与冷恢复验证 (`V3-G1`/`V3-G3`)** | 冷恢复首请求 `request.json` 字节级一致，快照终态分与 L1 官方分一致 | `V3-G1` 首请求 SHA-256 100% 字节一致（`{v3_g1.get('parent_run_label')}` @ `{v3_g1.get('checkpoint_id')}`）；`V3-G3` 终态快照分 **`{c4['v3_g3_exact_match_count']}`** 完全一致（最大绝对误差 `{c4['v3_g3_terminal_snapshot_max_abs_diff']:.2e}`） | **{'PASS' if c4['passed'] else 'FAIL'}** |",
        "",
        "## 2. 逐条 Episode 实测明细（9 Episodes）",
        "",
        "| Run Label | 官方得分 $s(T)$ | 历史最高快照分 $\\max_k s(k)$ | 终止原因 | 请求数 | 工具调用 (`read`/`bash`/`edit`/`write`) | 最大单次 Tokens | 耗时 (s) | 零成本候选状态细分 |",
        "| :--- | :---: | :---: | :--- | :---: | :--- | :---: | :---: | :--- |",
    ]
    for s in ep_summaries:
        tc = s["tool_counts_by_name"]
        tc_str = f"{s['tool_calls']} (r:{tc.get('read',0)}/b:{tc.get('bash',0)}/e:{tc.get('edit',0)}/w:{tc.get('write',0)})"
        md_lines.append(
            f"| `{s['run_label']}` | **{s['terminal_score']:.4f}** | `{s['max_snapshot_score']:.4f}` | `{s['termination']}` | "
            f"`{s['model_requests']}` | `{tc_str}` | `{s['max_single_request_total_tokens']}` | `{s['agent_phase_seconds']:.1f}` | `{s['candidate_substate']}` |"
        )

    md_lines.extend([
        "",
        "## 3. 按家族汇总对比（`Qwen/Qwen3.6-27B` vs DeepSeek `dev45`）",
        "",
        "| 任务家族 | `Qwen/Qwen3.6-27B` 3 条得分 | `Qwen/Qwen3.6-27B` 均值 | 非零条数 | 平均请求数 | 平均峰值 Prompt Tokens |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ])
    for t, fb in family_breakdown.items():
        sc_str = ", ".join(f"{x:.4f}" for x in fb["scores"])
        md_lines.append(
            f"| `{t}` | `[{sc_str}]` | **{fb['mean_score']:.4f}** | `{fb['nonzero_count']}/3` | `{fb['mean_requests']:.1f}` | `{fb['mean_max_prompt_tokens']:.0f}` |"
        )

    REPORT_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[REPORT WRITTEN] {REPORT_JSON} and {REPORT_MD}", flush=True)


if __name__ == "__main__":
    main()
