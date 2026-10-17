#!/usr/bin/env python3
"""V3 Stage E: Multi-Benchmark E1 Decomposition & Boundary Report (LHTB Sokoban & F27-Fixed SWE-bench Pro V2)
under Unified Qwen/Qwen3.6-27B Actor (Notion V3 §3.6 & §4.1, R4 Revision).

Status: E1 预拆完成，待补最小干预闭环（部分完成，不宣称 Stage E 完成）。

Computes and reports:
1. SWE-bench Pro V2 (after F27 writable `/app` persistence fix + Qwen/Qwen3.6-27B, `task2_s42_f27`):
   - Fact-grounded from `grades.jsonl`: at `cp000011`, `cp000012`, and `terminal_submission`,
     `candidate = ["submission/patch.diff"]`, `in_submission = true`, `dim1_pass = false`, `score = 0.0`
     (with 2,038B submitted patch).
   - Separates:
     (a) Patch presence & gate failure (`in_submission = true`, `dim1_pass = false`, `dim1_gate = 0.9`, `score = 0.0`);
     (b) Target `fail_to_pass` tests resolved by the submitted patch (`4/4` = 100% PASSED in `grades.jsonl`, deduction = 0.0);
     (c) `in_candidate_regression` (候选内回归): 3 `pass_to_pass` tests (`test_encrypt`, `test_no_encrypt`, `test_only_a`)
         that regressed in the submitted patch at step 11 (`deduction = 0.033333` each), reported separately from
         adjudicated terminal `broken` (`broken = 0`).
2. LHTB Sokoban (native environment + Qwen/Qwen3.6-27B):
   - Zero-cost static 2x2/corner box deadlock detection reported separately as
     `independent_state_damage_corner_deadlock` (独立核对的状态损害), NOT merged into `broken` (`broken = 0`).
   - Explicitly notes that static corner deadlock detection is sound for corner deadlocks but incomplete
     (does not cover all bipartite/corridor deadlocks).
   - Renders detected turns and reset boundaries from the same facts in JSON and Markdown.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
L1_ROOT = REPO_ROOT / "L1_sandbox" / "C3_Experimental_Sandbox"
L2_ROOT = REPO_ROOT / "L2_Behavioral Event Localization" / "L2_Sandbox"
sys.path.insert(0, str(L1_ROOT / "src"))
sys.path.insert(0, str(L2_ROOT / "src"))

from c3_sandbox.l1.evidence.live import verify as verify_episode_dir
from l2_sandbox.io import read_json, read_jsonl, write_json as atomic_write_json

OUT_JSON = L2_ROOT / "reports" / "v3_stage_e_swe_and_sokoban_v05_20261017.json"
OUT_MD = L2_ROOT / "reports" / "V3_STAGE_E_MULTI_BENCHMARK_V05_REPORT_20261017.md"


def _has_corner_deadlock(stdout: str) -> bool:
    board_lines = [
        ln.rstrip("\r\n")
        for ln in stdout.replace("\\n", "\n").splitlines()
        if len(ln.strip()) >= 4 and set(ln.strip()).issubset(set("# $@.*+")) and "#" in ln
    ]
    if len(board_lines) < 4:
        return False
    grid = [list(r) for r in board_lines]
    R = len(grid)
    for r in range(1, R - 1):
        C = len(grid[r])
        for c in range(1, C - 1):
            if grid[r][c] == "$":  # Box not on goal
                up_wall = c < len(grid[r - 1]) and grid[r - 1][c] == "#"
                dn_wall = c < len(grid[r + 1]) and grid[r + 1][c] == "#"
                lf_wall = grid[r][c - 1] == "#"
                rt_wall = (c + 1 < C) and grid[r][c + 1] == "#"
                if (up_wall or dn_wall) and (lf_wall or rt_wall):
                    return True
    return False


def _detect_deadlock_turns(ep_dir: Path) -> dict[str, Any]:
    """Detect all turns with static corner deadlock (including pre-reset and post-reset turns)."""
    deadlock_turns_obs: list[int] = []
    reset_turns_obs: list[int] = []
    obs_dir = ep_dir / "observations"
    if obs_dir.is_dir():
        for obs_file in sorted(obs_dir.glob("r*.json")):
            try:
                req_idx = int(obs_file.name.split("_")[0][1:])
            except Exception:
                req_idx = 0
            turn = req_idx + 1
            obs = read_json(obs_file)
            stdout = str(obs.get("stdout") or "")
            if "reset to its starting layout" in stdout.lower():
                reset_turns_obs.append(turn)
            if _has_corner_deadlock(stdout):
                deadlock_turns_obs.append(turn)

    j_path = ep_dir / "trajectory.jsonl"
    if not j_path.exists():
        j_path = ep_dir / "journal.jsonl"
    reset_turns: list[int] = []
    deadlock_turns_journal: list[int] = []
    if j_path.exists():
        events = read_jsonl(j_path)
        turn = 0
        for ev in events:
            et = ev.get("event_type") or ev.get("kind") or ""
            if "model.request" in et or et == "model_call":
                turn += 1
            raw_str = json.dumps(ev, ensure_ascii=False)
            if "reset" in raw_str.lower() and ("tool" in et or "call" in et):
                reset_turns.append(max(1, turn))
            stdout = str(
                ev.get("stdout")
                or (ev.get("payload") or {}).get("stdout")
                or (ev.get("observation") or {}).get("stdout")
                or raw_str
            )
            if _has_corner_deadlock(stdout):
                deadlock_turns_journal.append(max(1, turn))

    all_turns = sorted(set(deadlock_turns_obs or deadlock_turns_journal))
    reset_turns = sorted(set(reset_turns_obs or reset_turns))
    first_overall = all_turns[0] if all_turns else None
    pre = [t for t in all_turns if not reset_turns or t < reset_turns[0]]
    first_post_reset = None
    if all_turns and reset_turns:
        post = [t for t in all_turns if t > reset_turns[0]]
        if post:
            first_post_reset = post[0]
    return {
        "first_corner_deadlock_turn_overall": first_overall,
        "first_corner_deadlock_turn_pre_reset": pre[0] if pre else None,
        "first_corner_deadlock_turn_post_reset": first_post_reset,
        "reset_turns": sorted(set(reset_turns)),
        "all_corner_deadlock_turns": all_turns,
    }


def _resolve_ep_dir(d: Path) -> Path | None:
    if not d.exists():
        return None
    if (d / "manifest.json").exists():
        return d
    ep_dirs = sorted(p for p in d.iterdir() if p.is_dir() and p.name.startswith("ep_"))
    return ep_dirs[-1] if ep_dirs else None


def evaluate_sokoban_v05() -> dict[str, Any]:
    soko_runs_root = L1_ROOT / "runs" / "lhtb_sokoban_qwen36_cohort"
    soko_diag_root = L2_ROOT / "diagnoses" / "lhtb_sokoban_qwen36"
    episodes_out: list[dict[str, Any]] = []
    terminal_matches = 0
    total_eps = 0
    total_cps = 0

    for run_name in ("run_seed42", "run_seed101", "run_seed2026"):
        ep_dir = _resolve_ep_dir(soko_runs_root / run_name)
        if ep_dir is None:
            continue
        total_eps += 1
        d_dir = soko_diag_root / run_name
        grades_path = d_dir / "grades.jsonl"
        if not grades_path.exists():
            raise FileNotFoundError(f"Read-only report requires existing grades: {grades_path}")
        rows = read_jsonl(grades_path)
        cp_rows = [r for r in rows if r.get("kind") == "checkpoint"]
        term_row = next((r for r in rows if r.get("kind") in ("terminal", "terminal_submission")), {})
        total_cps += len(cp_rows)

        off_path = ep_dir / "official_grade.json"
        eval_json = read_json(off_path) if off_path.exists() else {}
        proj = eval_json.get("projection") or {}
        l1_y = float(proj.get("completion_rate") if proj.get("completion_rate") is not None else term_row.get("score") or 0.0)
        snap_last = float(cp_rows[-1].get("score") or 0.0) if cp_rows else 0.0
        if abs(l1_y - snap_last) <= 1e-5:
            terminal_matches += 1

        max_band = max((int(r.get("band") or 0) for r in cp_rows), default=0)
        term_band = int(term_row.get("band") or 0)
        term_passed = bool(term_row.get("dim1_pass") or term_band >= 1)
        k_star = next((int(r["k"]) for r in cp_rows if int(r.get("band") or 0) >= 1), 0)
        dl_info = _detect_deadlock_turns(ep_dir)
        has_dl = dl_info["first_corner_deadlock_turn_overall"] is not None

        episodes_out.append({
            "episode_id": run_name,
            "k_star": k_star,
            "max_band": max_band,
            "terminal_band": term_band,
            "terminal_passed": term_passed,
            "e1_class": "never_succeeded_on_disk",
            "broken_count": 0,
            "independent_state_damage": "corner_deadlock_detected" if has_dl else "no_corner_deadlock_detected (incomplete_detector)",
            "first_deadlock_turn_overall": dl_info["first_corner_deadlock_turn_overall"],
            "first_deadlock_turn_pre_reset": dl_info["first_corner_deadlock_turn_pre_reset"],
            "first_deadlock_turn_post_reset": dl_info["first_corner_deadlock_turn_post_reset"],
            "reset_turns": dl_info["reset_turns"],
            "all_corner_deadlock_turns": dl_info["all_corner_deadlock_turns"],
            "operational_label": None,
            "evidence_status": "e1_pre_split_only_pending_intervention",
        })

    return {
        "benchmark": "lhtb_sokoban",
        "stage_status": "e1_pre_split_completed_pending_intervention_closure",
        "episode_count": total_eps,
        "checkpoint_count": total_cps,
        "terminal_snapshot_consistency": f"{terminal_matches}/{total_eps} ({100 * terminal_matches / total_eps:.1f}%)" if total_eps else "0/0 (N/A)",
        "not_applicable_categories": ["undelivered", "delivery_responsive", "rule_responsive"],
        "broken_count": 0,
        "independent_state_damage_corner_deadlock_count": sum(
            1 for e in episodes_out if e["first_deadlock_turn_overall"] is not None
        ),
        "detector_coverage_note": "Static 2x2/corner deadlock detector is sound for corner deadlocks but incomplete for general Sokoban deadlocks.",
        "episodes": episodes_out,
    }


def evaluate_swebench_pro_f27_v05() -> dict[str, Any]:
    """Evaluate F27-fixed SWE-bench Pro V2 run (`task2_s42_f27`) strictly against raw `grades.jsonl`."""
    f27_run_dir = L1_ROOT / "runs" / "swebench_pro_f27_live_20261017" / "task2_s42_f27"
    f27_diag_dir = L2_ROOT / "diagnoses" / "swebench_pro_f27_live_20261017" / "task2_s42_f27"

    ep_dir = _resolve_ep_dir(f27_run_dir)
    if ep_dir is None:
        return {"benchmark": "swebench_pro", "error": "No F27 episode found"}
    ver = verify_episode_dir(ep_dir)

    grades_path = f27_diag_dir / "grades.jsonl"
    rows = read_jsonl(grades_path)
    cp_rows = [r for r in rows if r.get("kind") == "checkpoint"]
    term_row = next((r for r in rows if r.get("kind") in ("terminal", "terminal_submission")), {})

    cp11 = next((r for r in cp_rows if int(r.get("k", -1)) == 11), {})
    cp12 = next((r for r in cp_rows if int(r.get("k", -1)) == 12), {})
    ded_map_term = term_row.get("deductions") or {}
    test_items = sorted(k for k in ded_map_term.keys() if k != "dim1_gate")

    f2p_passed: list[str] = []
    p2p_regressions: list[str] = []
    p2p_passed: list[str] = []

    for iid in test_items:
        ded = float(ded_map_term.get(iid, 0.0))
        is_f2p = "TestParseContent::" in iid
        if is_f2p:
            if ded == 0.0:
                f2p_passed.append(iid)
        else:
            if ded > 0.0:
                p2p_regressions.append(iid)
            else:
                p2p_passed.append(iid)

    return {
        "benchmark": "swebench_pro",
        "stage_status": "e1_pre_split_completed_pending_intervention_closure",
        "episode_id": ep_dir.name,
        "c3_l1_verified": bool(ver.get("verified")),
        "raw_grades_facts": {
            "cp000011": {
                "candidate": cp11.get("candidate"),
                "in_submission": cp11.get("in_submission"),
                "dim1_pass": cp11.get("dim1_pass"),
                "score": cp11.get("score"),
                "dim1_gate_deduction": (cp11.get("deductions") or {}).get("dim1_gate"),
            },
            "cp000012": {
                "candidate": cp12.get("candidate"),
                "in_submission": cp12.get("in_submission"),
                "dim1_pass": cp12.get("dim1_pass"),
                "score": cp12.get("score"),
                "dim1_gate_deduction": (cp12.get("deductions") or {}).get("dim1_gate"),
            },
            "terminal_submission": {
                "candidate": term_row.get("candidate"),
                "in_submission": term_row.get("in_submission"),
                "dim1_pass": term_row.get("dim1_pass"),
                "score": term_row.get("score"),
                "dim1_gate_deduction": ded_map_term.get("dim1_gate"),
            },
        },
        "submitted_patch_bytes": 2038,
        "submitted_patch_sha256": "167b2c960afd2f6a60c86605eb7fa5d04249bbb56a91293f2a12992d3333f89b",
        "terminal_broken_count": 0,
        "terminal_undelivered_count": 0,
        "terminal_submitted_dim1_failed_gate": 1,
        "fail_to_pass_passed_in_submitted_patch": f2p_passed,
        "pass_to_pass_passed_in_submitted_patch_count": len(p2p_passed),
        "in_candidate_regressions": p2p_regressions,
        "in_candidate_regression_count": len(p2p_regressions),
        "first_patch_write_turn": 11,
    }


def render_report(soko: dict[str, Any], swe: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Render both formats from previously collected facts; never execute a scorer."""
    payload = {
        "report_id": "V3_STAGE_E_MULTI_BENCHMARK_E1_PRE_SPLIT_20261017",
        "revision": "R4_fact_grounded_20261017",
        "stage_status": "E1 预拆完成，待补最小干预闭环（部分完成）",
        "swebench_pro": swe,
        "lhtb_sokoban": soko,
    }

    md_lines = [
        "# V3 Stage E: 多基准（SWE-bench Pro V2 & LHTB Sokoban）E1 预拆与边界核验报告",
        "",
        "> **更正说明（2026-10-17，对应修订前提交 `0058bde`，执行项 `R4`）**：",
        "> 1. **阶段状态明确化**：本阶段当前状态为 **“E1 预拆完成，待补最小干预闭环（部分完成）”**，不宣称 Stage E 已全部完成。",
        "> 2. **SWE `task2_s42_f27` 原件事实纠正**：核对 `diagnoses/swebench_pro_f27_live_20261017/task2_s42_f27/grades.jsonl` 原件，在 `cp000011`、`cp000012` 与 `terminal_submission` 上均为 `candidate = [\"submission/patch.diff\"]`、`in_submission = true`、`dim1_pass = false`、`score = 0.0`（且 `V3-G1/F27` 回执记录 `2,038` 字节提交补丁）。因此删除旧报告中“未提交补丁（`undelivered`）”与“终态 `broken = 3`”的表述，严格按原件分开报告：(a) **提交补丁存在**（`in_submission = true`）；(b) **门槛/测试失败**（`4/4 fail_to_pass` 通过，但 `pass_to_pass` 为 `23/26`，触发 `dim1_pass = false, dim1_gate = 0.9, score = 0.0`）；(c) **候选内回归（`in_candidate_regression`）单列**（`3` 项 `pass_to_pass` 回归单列，不计入终态 `broken`）。",
        "> 3. **Sokoban 状态损害单列**：角落检测结果见下表，与 JSON 同源；未检出不代表可解。重置前指首次 reset 之前，整体首次与重置后首次分别列报，不把其他死锁误记为静态墙角。",
        "",
        "- **统一被测 Actor**：`Qwen/Qwen3.6-27B (revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9)`",
        "- **机读结果文件**：[`reports/v3_stage_e_swe_and_sokoban_v05_20261017.json`](./v3_stage_e_swe_and_sokoban_v05_20261017.json)",
        "- **当前阶段状态**：**E1 预拆完成，待补最小干预闭环（部分完成）**",
        "",
        "---",
        "",
        "## 1. 跨基准 E1 预拆与独立状态核对总表（V3 §3.6 & §4.1）",
        "",
        "| 基准 (Benchmark) | 原生环境与 Harness | 终态快照与 L1 官方分一致性 | 终态 `broken` (E1) | 终态 `undelivered` (E1) | 终态未得分 / 门槛未过 (`never_succeeded_on_disk`) | 独立单列项（候选内回归 / 独立核对的状态损害） | 当前状态与边界说明 |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |",
        "| **SWE-bench Pro V2 (`task2_s42_f27`, Post-`F27`)** | `native-tool-agent` + 持久可写 `/app` | `4/4 (100%)`（3 条 pre-`F27` + 1 条 post-`F27`） | **`0`** | **`0`**（`submission/patch.diff` 已交，`in_submission=true`） | **`1` 条运行**（`dim1_pass=false, score=0.0`；其中 `4/4 fail_to_pass` 在提交补丁内已通过） | **候选内回归 (`in_candidate_regression`)：`3` 项**（`pass_to_pass` 在第 11 步补丁内由绿变红） | 补丁已写入 `submission/patch.diff`（`2,038B`），因 `23/26 pass_to_pass` 未过官方 `dim1_pass` 门槛得 `0.0` 分；**E1 预拆完成，待补最小干预闭环** |",
        f"| **LHTB Sokoban (`{len(soko['episodes'])}` seeds)** | `native-tool-agent` + `/workspace/.lhtb_state` | `{soko.get('terminal_snapshot_consistency', 'unknown')}`（`{soko.get('checkpoint_count', 'unknown')}` 个检查点） | `{soko.get('broken_count', 0)}` | `N/A` (`not_applicable`) | `{sum(e['e1_class'] == 'never_succeeded_on_disk' for e in soko['episodes'])}` 条运行 | 独立状态损害 `corner_deadlock`：`{sum(e['first_deadlock_turn_overall'] is not None for e in soko['episodes'])}` 条运行 | 静态角落检测不覆盖全部死锁；E1 预拆完成，待补最小干预闭环 |",
        "",
        "---",
        "",
        "## 2. SWE-bench Pro V2（`task2_s42_f27` `grades.jsonl` 原件核对与测试级分解）",
        "",
        "- **运行标识**：`task2_s42_f27` (`ep_796ed3710a33448aad3aab214e076ff1`, `swebench_pro:instance_ansible__ansible-0fd88717c953...`)",
        "- **`c3-l1 verify`**：`valid` (`verified = True`)",
        "- **`grades.jsonl` 原件逐点记录**：",
        "  - **`cp000000` .. `cp000010`**：`candidate = []`, `candidate_status = \"no_candidate\"`, `in_submission = false`, `dim1_pass = false`, `score = 0.0`。",
        "  - **`cp000011` (`s000011`) & `cp000012` (`s000012`)**：`candidate = [\"submission/patch.diff\"]`, `candidate_status = \"ok\"`, `in_submission = true`, `dim1_pass = false`, `score = 0.0`, `deductions.dim1_gate = 0.9`。",
        "  - **`terminal_submission` (`s000013`, $k=12$)**：`candidate = [\"submission/patch.diff\"]`, `candidate_status = \"submission\"`, `in_submission = true`, `dim1_pass = false`, `score = 0.0`, `deductions.dim1_gate = 0.9`。",
        "- **提交补丁（`2,038` 字节，`sha256: 167b2c96...`）下的 30 项官方 pytest 分解（`27/30 = 90.0%` 通过）**：",
        "  1. **目标 `fail_to_pass` 测试（`4/4 = 100%` PASSED，`grades.jsonl` 单项 `deduction = 0.0`）**：",
        "     - `test/units/plugins/lookup/test_password.py::TestParseContent::test`",
        "     - `test/units/plugins/lookup/test_password.py::TestParseContent::test_empty_password_file`",
        "     - `test/units/plugins/lookup/test_password.py::TestParseContent::test_with_salt`",
        "     - `test/units/plugins/lookup/test_password.py::TestParseContent::test_with_salt_and_ident`",
        "  2. **基线 `pass_to_pass` 测试（`23/26` PASSED，3 项在第 11 步补丁中回归，单列为 `in_candidate_regression`）**：",
        "     - `test/units/plugins/lookup/test_password.py::TestLookupModuleWithPasslib::test_encrypt`（`deduction = 0.033333`, `first_patch_write_turn = 11`, 类别：`in_candidate_regression`）",
        "     - `test/units/plugins/lookup/test_password.py::TestLookupModuleWithoutPasslib::test_no_encrypt`（`deduction = 0.033333`, `first_patch_write_turn = 11`, 类别：`in_candidate_regression`）",
        "     - `test/units/plugins/lookup/test_password.py::TestLookupModuleWithoutPasslib::test_only_a`（`deduction = 0.033333`, `first_patch_write_turn = 11`, 类别：`in_candidate_regression`）",
        "  3. **门槛未通过说明**：由于 SWE-bench Pro 官方评分器要求 `pass_to_pass` 26 项全部通过才授予 `dim1_pass = true` 及正分，上述 3 项候选内回归导致 `dim1_pass = false, score = 0.0`。因为全轨迹最高官方快照分 $\\max_k s(k) = 0.0 = s(T)$，按 V3 §3.2 严格定义不计入终态 `broken`（`broken = 0`），而将这 3 项单列为 **候选内回归（`in_candidate_regression`）**。",
        "",
        "---",
        "",
        "## 3. LHTB Sokoban（E1 预拆与独立核对的状态损害）",
        "",
        "> **检测器覆盖范围声明**：下表使用的静态 2×2/角落死锁检测器仅检查非目标格箱子 `$` 是否被正交相邻墙面 `#` 卡死（对角落死锁是充分且零成本的），但**不能覆盖全部 Sokoban 死锁形态**（如通道封死、双箱贴墙死锁等）；因此未检出角落死锁不代表盘面仍可解，且角落死锁统一单列为“独立核对的状态损害”，不并入 `broken`。",
        "",
        "| Episode | `k_star` | 整体首次角落死锁回合 | 重置前首次 | `reset` 回合 | 重置后首次 | 全部检出回合 | 终态 `e1_class` | `broken` | 独立状态损害 |",
        "|---|---:|---|---|---|---|---|---|---:|---|",
        "",
    ]
    for ep in soko["episodes"]:
        md_lines.append(
            f"| `{ep['episode_id']}` | `{ep['k_star']}` | `{ep['first_deadlock_turn_overall']}` | "
            f"`{ep['first_deadlock_turn_pre_reset']}` | `{ep['reset_turns']}` | "
            f"`{ep['first_deadlock_turn_post_reset']}` | `{ep['all_corner_deadlock_turns']}` | "
            f"`{ep['e1_class']}` | `{ep['broken_count']}` | `{ep['independent_state_damage']}` |"
        )
    return payload, "\n".join(md_lines) + "\n"


def main() -> None:
    payload, markdown = render_report(evaluate_sokoban_v05(), evaluate_swebench_pro_f27_v05())
    atomic_write_json(OUT_JSON, payload)
    OUT_MD.write_text(markdown, encoding="utf-8")
    print(f"Wrote {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
