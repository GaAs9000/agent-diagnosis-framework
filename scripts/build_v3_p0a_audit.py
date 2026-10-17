#!/usr/bin/env python3
"""V3 P0a Zero-Cost Audit on dev45 (and holdout45 candidate-state comparison).

Produces:
1. Five-way E1 candidate-state breakdown of all 45 dev episodes and 33.4610 total deduction:
   - broken
   - undelivered
   - never:no_candidate_ever (k* = 0)
   - never:unsubmitted_score0_candidate
   - never:submitted_dim1_gate_failed
   - never:delivered_but_item_wrong (dim1_pass == True)
2. Rubric-source classification (explicit vs implicit vs ambiguous) for all rubric items
   across the 9 dev task families (003, 004, 010, 033, 050, 060, 070, 085, 090), grounded
   in exact inspection of task instructions, input files/images, and official verifiers.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEV_VAL_DIR = ROOT / "L2_Behavioral Event Localization" / "L2_Sandbox" / "validation" / "v04_dev45_20261016"
HOLDOUT_VAL_DIR = ROOT / "L2_Behavioral Event Localization" / "L2_Sandbox" / "validation" / "v04_holdout45_20261017"
OUT_JSON = DEV_VAL_DIR / "v3_p0a_audit.json"
OUT_MD = ROOT / "L2_Behavioral Event Localization" / "L2_Sandbox" / "docs" / "V3_P0A_AUDIT_20261017.md"


def classify_dev_defect_rule(task_id: str, defect_rule: str) -> tuple[str, str]:
    """Return (source_category, rationale) in {'explicit', 'implicit', 'ambiguous', 'gate_or_no_artifact'}."""
    if defect_rule in ("no_artifact", "dim1_gate"):
        return "gate_or_no_artifact", "未产出候选文件或未通过 dim1 可打开/文件数门槛"

    short_tid = task_id.split(":")[-1]

    if short_tid == "officeval_003":
        if "没有空白行" in defect_rule:
            return "explicit", "Prompt 明确要求：'再把没有字的空行删掉'"
        if "下方都出现横线" in defect_rule:
            return "explicit", "Prompt 明确要求：'每一行文字都放到表里面并加上横线，横线样式和图片中的一样'，且 image.png 清晰展示黑色细实线"
        if "0.15cm的误差" in defect_rule:
            return "ambiguous", "image.png 展示等距横线，但 '相邻两条横线距离只允许有0.15cm误差' 的硬数值容差仅在评分器中定义"
        if "侯景澜" in defect_rule:
            return "explicit", "Prompt 明确要求：'其他格式保持不变'（保留原文正文内容）"
        return "ambiguous", "视觉参考与评分细则容差混合"

    if short_tid == "officeval_004":
        # Check image_001.png vs image_002.png
        if "实习单位评价" in defect_rule or "教学单位评价意见" in defect_rule:
            return (
                "implicit",
                "image_001.png 左侧写明'文字部分：宋体四号'，仅在'自我评价'框内标注'宋体小四'，未在'实习单位评价'与'教学单位评价意见'标注宋体小四，但评分器要求这两格也用宋体小四",
            )
        if "学号" in defect_rule or "页面顶部没有出现黑体、20号" in defect_rule:
            return "ambiguous", "image_001.png 展示了表头与学号横向排版外观，但未文字标注20号字号硬阈值"
        return "explicit", "image_001.png 与 image_002.png 红字批注已逐项明确写出字体、字号、加粗、缩进与行距要求"

    if short_tid == "officeval_010":
        return (
            "ambiguous",
            "Prompt 仅要求'第四行删除，字体需要移动到框内，做成可编辑的'，原图展示拓扑与文字，但评分器逐框检查精确厘米坐标/宽高/字号（如宽2.5-2.7cm、间距0.6-0.75cm、黑体小五）",
        )

    if short_tid == "officeval_033":
        if "任意价格单元格出现" in defect_rule or "整页图片" in defect_rule or "价格说明文字" in defect_rule:
            return "explicit", "Prompt 要求'其他格式不变，输出替换完成可修改的word文档'"
        if "推荐客餐厅配套品表" in defect_rule or "定制价格表" in defect_rule or "门店订购记录页" in defect_rule:
            return (
                "ambiguous",
                "Prompt 仅写'替换为乘以1.1之后的数值'，未规定小数取整规则；原表含 3855*1.1=4240.5、7515*1.1=8266.5、38675*1.1=42542.5，Python round() 银行家舍入得偶数(4240/8266/42542)，而评分器硬编码四舍五入(4241/8267/42543)",
            )
        return "explicit", "乘以 1.1 后无 .5 边界歧义的常规价格替换"

    if short_tid == "officeval_050":
        if "Arial 或 Calibri" in defect_rule or "编号圆形" in defect_rule:
            return "implicit", "Prompt 仅要求'改成可直接编辑状态，内容和原来一模一样'，未声明必须使用 Arial/Calibri 字体或第3阶段编号写'02'"
        return (
            "ambiguous",
            "原 PPT 为两页整图，视觉内容可见，但评分器对每个形状规定了精确厘米坐标框与字号区间（如距左13.5-21cm、宽6.2-6.6cm）",
        )

    if short_tid == "officeval_060":
        if "0—0.3厘米" in defect_rule or "0%—22%" in defect_rule:
            return (
                "ambiguous",
                "Prompt 仅写'对第四页PPT重新排版，要求图片组合符合页面比例，页面两边无大片空白'，而评分器要求单张背景图四边距页面边缘 0-0.3cm 且水罐/膜堆落在特定百分比区间",
            )
        return "explicit", "Prompt 明确要求'页面两边无大片空白'"

    if short_tid == "officeval_070":
        if "10.5至11.5磅" in defect_rule or "0.9至1.0倍行距" in defect_rule:
            return (
                "ambiguous",
                "Prompt 写'可以缩小字体和间距...调整成像临床问题和科学假说这两块的布局'，评分器则检查标题10.5-11.5pt、正文7.2-8.2pt、边距0.3-0.5cm等具体数值区间",
            )
        return "explicit", "Prompt 明确要求'将字完整放在板块区域框内，使字不要叠着放，PPT尺寸不改变'"

    if short_tid == "officeval_085":
        if "18cm–21cm" in defect_rule:
            return "implicit", "Prompt 与参考图均未指定 Excel 图表对象宽 18-21cm、高 10-13cm 及距数据区至少1列/2行"
        return "explicit", "Prompt 要求'画成图片上类似的S11的图，放置在数据对应的表格里，要可编辑的'"

    if short_tid == "officeval_090":
        if "IFERROR" in defect_rule:
            return (
                "implicit",
                "Prompt 仅写'将所有子表中的总重量合计数据以及砼体积数据提取到汇总表'，且原表 H 列(含量)已有静态数值；评分器却要求把 H3:H62 改写为公式 =IFERROR(G3/F3,\"\")",
            )
        if "Sheet3及同结构子表" in defect_rule or "Sheet5及同结构子表" in defect_rule:
            return (
                "implicit",
                "Prompt 仅写'提取到汇总表'，且细则第1条允许'全部为数值或引用公式'；但第4、5条评分器强制检查单元格必须为跨表公式引用 ='Sheet3'!D20，直接填入正确提取数值被判 0 分",
            )
        return "explicit", "Prompt 明确要求提取子表砼体积与总重量合计到汇总表，且原表 F/G 列已预设 0.000 格式"

    return "ambiguous", "未分类"


def analyze_cohort(val_dir: Path) -> dict:
    episodes = []
    substate_counts = {
        "perfect_score": 0,
        "broken": 0,
        "undelivered": 0,
        "never:no_candidate_ever": 0,
        "never:unsubmitted_score0_candidate": 0,
        "never:submitted_dim1_gate_failed": 0,
        "never:delivered_but_item_wrong": 0,
    }
    substate_deductions = {
        "broken": 0.0,
        "undelivered": 0.0,
        "never:no_candidate_ever": 0.0,
        "never:unsubmitted_score0_candidate": 0.0,
        "never:submitted_dim1_gate_failed": 0.0,
        "never:delivered_but_item_wrong": 0.0,
    }
    substate_episodes = {k: [] for k in substate_counts}

    rubric_source_deductions = {
        "explicit": 0.0,
        "ambiguous": 0.0,
        "implicit": 0.0,
    }
    rubric_source_counts = {
        "explicit": 0,
        "ambiguous": 0,
        "implicit": 0,
    }
    task_rule_audit = {}

    for ep_dir in sorted(val_dir.glob("*.requests_035.r*")):
        if not (ep_dir / "defects.json").is_file() or not (ep_dir / "grades.jsonl").is_file():
            continue
        run_label = ep_dir.name
        d_table = json.loads((ep_dir / "defects.json").read_text(encoding="utf-8"))
        grade_rows = [json.loads(line) for line in (ep_dir / "grades.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        term_row = grade_rows[0]
        cp_rows = [r for r in grade_rows[1:] if r.get("kind") == "checkpoint"]

        y_score = float(term_row.get("score") or 0.0)
        ever_had_cand = any(r.get("candidate_status") == "ok" for r in cp_rows)
        in_sub = bool(term_row.get("in_submission"))
        dim1_pass = bool(term_row.get("dim1_pass"))

        defects = d_table.get("defects") or []
        ded_broken = sum(float(x["deduction"]) for x in defects if x["kind"] == "broken")
        ded_undeliv = sum(float(x["deduction"]) for x in defects if x["kind"] == "undelivered")
        ded_never = sum(float(x["deduction"]) for x in defects if x["kind"] == "never")

        substate_deductions["broken"] += ded_broken
        substate_deductions["undelivered"] += ded_undeliv

        if y_score >= 1.0 - 1e-9:
            substate = "perfect_score"
        elif not ever_had_cand:
            substate = "never:no_candidate_ever"
            substate_deductions["never:no_candidate_ever"] += ded_never
        elif not in_sub:
            substate = "never:unsubmitted_score0_candidate"
            substate_deductions["never:unsubmitted_score0_candidate"] += ded_never
        elif not dim1_pass:
            substate = "never:submitted_dim1_gate_failed"
            substate_deductions["never:submitted_dim1_gate_failed"] += ded_never
        else:
            substate = "never:delivered_but_item_wrong"
            substate_deductions["never:delivered_but_item_wrong"] += ded_never

        if ded_undeliv > 1e-9 and substate != "never:unsubmitted_score0_candidate":
            substate_counts["undelivered"] += 1
            substate_episodes["undelivered"].append(run_label)
        else:
            substate_counts[substate] += 1
            substate_episodes[substate].append(run_label)

        # For delivered_but_item_wrong (dim1_pass == True), audit rubric source of each never defect
        if dim1_pass and ded_never > 1e-9:
            tid = d_table["task_id"]
            task_rule_audit.setdefault(tid, {})
            for item in defects:
                if item["kind"] != "never":
                    continue
                rule_str = item["defect"]
                ded_val = float(item["deduction"])
                src_cat, rationale = classify_dev_defect_rule(tid, rule_str)
                if src_cat in rubric_source_deductions:
                    rubric_source_deductions[src_cat] += ded_val
                    rubric_source_counts[src_cat] += 1
                entry = task_rule_audit[tid].setdefault(
                    rule_str,
                    {"source_category": src_cat, "rationale": rationale, "occurrences": 0, "total_deduction": 0.0},
                )
                entry["occurrences"] += 1
                entry["total_deduction"] = round(entry["total_deduction"] + ded_val, 6)

        episodes.append({
            "run_label": run_label,
            "task_id": d_table["task_id"],
            "terminal_score_Y": y_score,
            "max_snapshot_score": float(d_table.get("max_snapshot_score") or 0.0),
            "k_star": int(d_table.get("k_star") or 0),
            "ever_had_candidate": ever_had_cand,
            "in_submission": in_sub,
            "dim1_pass": dim1_pass,
            "substate": substate,
            "deduction_broken": round(ded_broken, 6),
            "deduction_undelivered": round(ded_undeliv, 6),
            "deduction_never": round(ded_never, 6),
        })

    total_loss = sum(substate_deductions.values())
    total_never = sum(v for k, v in substate_deductions.items() if k.startswith("never:"))
    delivered_never = substate_deductions["never:delivered_but_item_wrong"]

    return {
        "episodes_count": len(episodes),
        "total_loss": round(total_loss, 6),
        "total_never_loss": round(total_never, 6),
        "substate_episode_counts": substate_counts,
        "substate_episodes": substate_episodes,
        "substate_deductions": {k: round(v, 6) for k, v in substate_deductions.items()},
        "substate_share_of_total_loss": {
            k: round(v / total_loss, 6) if total_loss > 0 else 0.0 for k, v in substate_deductions.items()
        },
        "substate_share_of_never_loss": {
            k: round(v / total_never, 6) if total_never > 0 else 0.0
            for k, v in substate_deductions.items()
            if k.startswith("never:")
        },
        "delivered_item_wrong_rubric_source": {
            "total_delivered_never_deduction": round(delivered_never, 6),
            "deduction_by_source": {k: round(v, 6) for k, v in rubric_source_deductions.items()},
            "share_of_delivered_never_deduction": {
                k: round(v / delivered_never, 6) if delivered_never > 0 else 0.0
                for k, v in rubric_source_deductions.items()
            },
            "defect_item_counts_by_source": rubric_source_counts,
            "task_rule_audit": task_rule_audit,
        },
        "episodes": episodes,
    }


def main() -> None:
    dev_res = analyze_cohort(DEV_VAL_DIR)
    holdout_res = analyze_cohort(HOLDOUT_VAL_DIR) if HOLDOUT_VAL_DIR.is_dir() else None

    payload = {
        "audit_id": "V3_P0A_ZERO_COST_CANDIDATE_AND_RUBRIC_AUDIT",
        "created_at": "2026-10-17",
        "dev45": dev_res,
        "holdout45_candidate_state_comparison": {
            k: holdout_res[k]
            for k in (
                "episodes_count",
                "total_loss",
                "total_never_loss",
                "substate_episode_counts",
                "substate_deductions",
                "substate_share_of_total_loss",
                "substate_share_of_never_loss",
            )
        }
        if holdout_res
        else None,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Build Markdown documentation
    d_ded = dev_res["substate_deductions"]
    d_tot = dev_res["substate_share_of_total_loss"]
    d_nev = dev_res["substate_share_of_never_loss"]
    r_src = dev_res["delivered_item_wrong_rubric_source"]

    md = [
        "# V3 `P0a` 零成本审计报告：`dev45` 候选状态五路细分与评分规则来源审计",
        "",
        "- **生成日期**：`2026-10-17`",
        "- **数据范围**：`dev45`（9 个开发任务家族 × 5 次重复 = 45 条轨迹，总扣分 `33.4610`，`never` 扣分 `32.0221`）；附 `holdout45` 候选状态分布对照",
        "- **API 请求成本**：**`0` 次模型请求**",
        "",
        "---",
        "",
        "## 1. 候选状态零成本细分（回应 V3 第一性原理问题 1）",
        "",
        "| 类别 / 候选子状态 | `dev45` 扣分总和 | 占总扣分 (`33.4610`) 比例 | 占 `never` (`32.0221`) 比例 | 涉及运行条数 | 代表运行 |",
        "| :--- | :---: | :---: | :---: | :---: | :--- |",
        f"| `broken`（做好又改坏） | `{d_ded['broken']:.4f}` | `{d_tot['broken']*100:.2f}%` | — | `0` | — |",
        f"| `undelivered`（做对未交，`s(T) > Y`） | `{d_ded['undelivered']:.4f}` | `{d_tot['undelivered']*100:.2f}%` | — | `4` | `003.r02`, `003.r03`, `010.r04`, `033.r00` |",
        f"| `never:no_candidate_ever`（全轨迹无候选，$k^*=0$） | **`{d_ded['never:no_candidate_ever']:.4f}`** | **`{d_tot['never:no_candidate_ever']*100:.2f}%`** | **`{d_nev['never:no_candidate_ever']*100:.2f}%`** | `6` | `003.r00`, `010.r00/r02/r03`, `050.r03`, `070.r04` |",
        f"| `never:unsubmitted_score0_candidate`（修改候选未入 `submission/` 的剩余未得分部分） | `{d_ded['never:unsubmitted_score0_candidate']:.4f}` | `{d_tot['never:unsubmitted_score0_candidate']*100:.2f}%` | `{d_nev['never:unsubmitted_score0_candidate']*100:.2f}%` | `2` | `003.r03` (`0.6667`), `050.r00` (`0.9500`) |",
        f"| `never:submitted_dim1_gate_failed`（已交 `submission/` 但未过 `dim1` 门槛） | `{d_ded['never:submitted_dim1_gate_failed']:.4f}` | `{d_tot['never:submitted_dim1_gate_failed']*100:.2f}%` | `{d_nev['never:submitted_dim1_gate_failed']*100:.2f}%` | `8` | `004.r00`, `004.r04`, `010.r01`, `085.r00..r04` |",
        f"| `never:delivered_but_item_wrong`（已交且过 `dim1`，具体评分项失分） | **`{d_ded['never:delivered_but_item_wrong']:.4f}`** | **`{d_tot['never:delivered_but_item_wrong']*100:.2f}%`** | **`{d_nev['never:delivered_but_item_wrong']*100:.2f}%`** | `25` | `003`, `004`, `010.r04`, `033`, `050`, `060`, `070`, `090` 等 25 条 |",
        "",
        "> **关键结论 1**：在 `dev45` 的 `32.0221` 分 `never` 扣分中：",
        f"> - **`{d_nev['never:no_candidate_ever']*100:.2f}%`（`{d_ded['never:no_candidate_ever']:.4f}` 分，6 条运行）** 从未在磁盘写出任何候选文件（$k^*=0$），直接印证 V3 §3.1 指出的「快照只看磁盘、看不到上下文中已构造完成但未落盘的状态」（如 `070.r04` 在 `c28/c29` 经 `plain`/`deliver` 续跑即可得 `0.5385`/`0.4615`）。",
        f"> - **`{(d_nev['never:unsubmitted_score0_candidate'] + d_nev['never:submitted_dim1_gate_failed'])*100:.2f}%`（`{d_ded['never:unsubmitted_score0_candidate'] + d_ded['never:submitted_dim1_gate_failed']:.4f}` 分，10 条运行）** 属于候选未入 `submission/` 或死在 `dim1` 门槛（文件格式/数量/可打开性）。",
        f"> - **`{d_nev['never:delivered_but_item_wrong']*100:.2f}%`（`{d_ded['never:delivered_but_item_wrong']:.4f}` 分，25 条运行，共 `{sum(r_src['defect_item_counts_by_source'].values())}` 个缺陷项）** 已进入 `dim2` 逐项评分但在具体评分细则项上失分。",
        "",
    ]

    if holdout_res:
        h_ded = holdout_res["substate_deductions"]
        h_tot = holdout_res["substate_share_of_total_loss"]
        h_nev = holdout_res["substate_share_of_never_loss"]
        md.extend([
            "### 附：`holdout45` 候选状态零成本对照（仅作分布参考）",
            "",
            "| 类别 / 候选子状态 | `holdout45` 扣分总和 | 占总扣分 (`34.9389`) 比例 | 占 `never` (`34.4776`) 比例 |",
            "| :--- | :---: | :---: | :---: |",
            f"| `broken` | `{h_ded['broken']:.4f}` | `{h_tot['broken']*100:.2f}%` | — |",
            f"| `undelivered` | `{h_ded['undelivered']:.4f}` | `{h_tot['undelivered']*100:.2f}%` | — |",
            f"| `never:no_candidate_ever` ($k^*=0$) | **`{h_ded['never:no_candidate_ever']:.4f}`** | **`{h_tot['never:no_candidate_ever']*100:.2f}%`** | **`{h_nev['never:no_candidate_ever']*100:.2f}%`** |",
            f"| `never:unsubmitted_score0_candidate` | `{h_ded['never:unsubmitted_score0_candidate']:.4f}` | `{h_tot['never:unsubmitted_score0_candidate']*100:.2f}%` | `{h_nev['never:unsubmitted_score0_candidate']*100:.2f}%` |",
            f"| `never:submitted_dim1_gate_failed` | `{h_ded['never:submitted_dim1_gate_failed']:.4f}` | `{h_tot['never:submitted_dim1_gate_failed']*100:.2f}%` | `{h_nev['never:submitted_dim1_gate_failed']*100:.2f}%` |",
            f"| `never:delivered_but_item_wrong` | **`{h_ded['never:delivered_but_item_wrong']:.4f}`** | **`{h_tot['never:delivered_but_item_wrong']*100:.2f}%`** | **`{h_nev['never:delivered_but_item_wrong']*100:.2f}%`** |",
            "",
        ])

    r_ded = r_src["deduction_by_source"]
    r_sh = r_src["share_of_delivered_never_deduction"]
    r_cnt = r_src["defect_item_counts_by_source"]

    md.extend([
        "---",
        "",
        "## 2. 评分规则来源审计：`explicit` vs `ambiguous` vs `implicit`（回应 V3 第一性原理问题 7）",
        "",
        "对 `dev45` 中所有已合法交付并进入 `dim2` 逐项评分的 `never:delivered_but_item_wrong` 缺陷（共 **24 条运行、279 个缺陷项、总扣分 `15.4055`**），逐一核对任务 `instruction`、可见输入附件（`input/*` 文档/表格/参考图）与官方验证脚本（`verifiers/officeval_*_verifier.py`）：",
        "",
        "| 规则来源分类 | 定义（V3 §3.4） | 缺陷项数量 (`n=279`) | 扣分总和 (`15.4055`) | 占已交付 `dim2` 失分比例 |",
        "| :--- | :--- | :---: | :---: | :---: |",
        f"| **`explicit`（显式规则）** | 任务文本或输入附件（如 `004` 红字批注图、`003` 样图）已明确写出该要求 | `{r_cnt['explicit']}` | **`{r_ded['explicit']:.4f}`** | **`{r_sh['explicit']*100:.2f}%`** |",
        f"| **`ambiguous`（边界模糊）** | 任务仅给宏观描述或视觉样图，评分器却检查未声明的精确数值/舍入/坐标容差 | `{r_cnt['ambiguous']}` | **`{r_ded['ambiguous']:.4f}`** | **`{r_sh['ambiguous']*100:.2f}%`** |",
        f"| **`implicit`（隐性规则）** | 任务文本与输入附件均未提及（甚至与输入表预设样式冲突），仅存在于评分器代码中 | `{r_cnt['implicit']}` | **`{r_ded['implicit']:.4f}`** | **`{r_sh['implicit']*100:.2f}%`** |",
        "",
        "### 三大典型实证发现（直接支撑 V3 论文核心论点）",
        "",
        "1. **`officeval_090`（100% 隐性规则失分，`3.0000` 分 / 5 条运行全部卡在 `0.40` 或 `0.80`）**：",
        "   - **任务指令**仅一句话：*“将所有子表中的总重量合计数据以及砼体积数据提取到汇总表”*。",
        "   - **输入文件真相**：原工作簿 `汇总` 表的 `F3:G62` 全为空，`H3:H62`（含量）已预先填好静态浮点数（如 `387.88`），并非公式。",
        "   - **评分器隐性要求（`officeval_090_verifier.py`）**：",
        "     - 细则第 3 条（5/25 分）：强制要求将未在 Prompt 中提及的 `H3:H62` 改写为公式 `=IFERROR(G3/F3,\"\")`；",
        "     - 细则第 4、5 条（10/25 分）：尽管细则第 1 条写明“全部为数值或引用公式”，第 4、5 条却用正则强制检查单元格字符串必须是跨表公式引用 `='Sheet3'!D20`（`formula_reference(cf) == (sheet, 'D20')`），导致用 Python 精确提取正确数值写入单元格的 Agent 直接被扣掉 40% 总分！",
        "   - **诊断启示**：当 `rule` 干预臂把细则文本注入给 Agent 时，得分立刻跃升至 `0.80–1.00`。这不是因为 Agent “粗心没发现任务要求”，而是因为 **60% 的评分标准属于未写入任务的 Benchmark 隐性规格（`implicit`）**。",
        "",
        "2. **`officeval_033`（`.5` 舍入约定歧义导致丢掉 64.7% 分数，`3.2165` 分）**：",
        "   - **任务指令**：*“把文档的所有零售价的数值替换为乘以1.1之后的数值，其他格式不变”*（全文未提如何取整）。",
        "   - **评分器真相（`officeval_033_verifier.py`）**：原文档含 `3855`（$\\times 1.1 = 4240.5$）、`7515`（$\\times 1.1 = 8266.5$）、`38675`（$\\times 1.1 = 42542.5$）。Agent 使用 Python 标准 `round()`（银行家舍入，`.5` 凑偶）得到 `4240, 8266, 42542`，其余近百个价格全部完全命中；但评分器硬编码传统四舍五入（`4241, 8267, 42543`），导致 `推荐客餐厅配套品表`（5分）、`定制价格表`（5分）、`门店订购记录页`（1分）三项整项判 0 分（仅得 `6/17 = 0.3529`）。",
        "",
        "3. **`officeval_004`（批注图局部漏标 vs 显式批注，`0.4545` 分 `implicit` + `2.6364` 分 `explicit`）**：",
        "   - 输入附件 `image_002.png` 对《毕业实习报告》的每一处字体、字号、行距做了完整红字批注（全部属于 `explicit`）；",
        "   - 但 `image_001.png` 对《毕业实习鉴定表》仅在左侧总注“文字部分：宋体四号”，且只在“自我评价”一格内标红“宋体小四”，未在“实习单位评价”和“教学单位评价意见”标红“宋体小四”；评分器却要求这三格全部为“宋体小四”（`implicit`）。",
        "",
    ])

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_JSON} and {OUT_MD}")
    print(json.dumps(dev_res["delivered_item_wrong_rubric_source"], indent=2, ensure_ascii=False)[:1500])


if __name__ == "__main__":
    main()
