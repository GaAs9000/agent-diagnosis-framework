# V3 阶段 D0 冻结候选清单与待用户拍板选项（2026-10-17）

## 当前审阅要求（Codex 返修后，覆盖下文历史候选）

**D0 包尚未就绪，不能直接确认下文旧候选后启动实验。**已完成的是 [离线工程验收](R1_R21_REPAIR_ACCEPTANCE_20261017.md)，263 项通过；以下条件仍须补齐：

1. 完整新开发集和绑定原件的快照评分。当前草案只扫描到 36 条有效运行，部分缺评分或恢复资格，不能当完整 dev45。
2. 整任务要求的真人来源审核，尤其 ambiguous 的可见要求与评分器精确规格拆分。自动建议和 63 行空表不等于审核完成。
3. D0 冻结每臂消息／哈希、锚点、入样概率、样本量和主要检验族。S 必须保留 D 前缀；独立 S 文本只是附加句，不是完整处理消息。
4. δ 是增量门槛，τ 是绝对有用恢复水平；不再要求增量达到 τ。“未恢复”需要全臂绝对上界低于 τ 且主要增量上界低于 δ。默认 .10/.20/.05 仍是草案参数。
5. D2 的估计目标和精度：当前保守家族界在九家族时理论最好下界约 .0132，不能越过 .05。有限开发池条件收益与跨家族总体收益是不同目标，已交 Antigravity 推敲，未擅自改变方法。
6. 真实且独立的开发校准基线、留出名单、费用／请求／token／墙钟上限和停止条件。真实 API 账单未因准入估算而获得核对。

`reports/v3_d1_ladder_plan_20261017.json` 已更新为未就绪草案，默认每臂 24 条；当前 6 个缺陷单元和 4 个伤害单元的草案为 1128 分支、最多 17952 次请求。要求审核与完整抽样后还会变化，这不是完整实验的已批准上限。D2 为 plan-only，门禁关闭；D3 已发布文件仅为不含原始轨迹的摘要，完整本机草案在 `validation/r_repair_codex_20261017/`，不能用摘要执行。

执行入口只接受与明确批准记录完全匹配的冻结计划；修复阶段没有新增实验请求。先补齐上述可审阅输入，再冻结并申请执行。

## 历史候选（b159963 时点，仅保留讨论记录）

下文原有预算档、τ 的旧解释、S 文本哈希及“候选完成”表述均不是当前执行口径；上面的返修记录覆盖它们。旧分配方案不自动获得批准。

- **对标文档**：[V3 草案｜干预可核对的失败诊断与第一篇论文规划](https://app.notion.com/p/761837e695624632829c375a30331aa3) §3.2–3.7、§5.2、§8.2–8.4；[V3 修订任务单｜给编码 agent](https://app.notion.com/p/db13afd19d9a4e09bc979475157a32d2) §5。
- **状态**：**D0 候选备选完成，等待用户确认冻结（未确认前不执行 D1 阶梯实跑、不发起任何新增 Actor/Analyst 请求、不触碰留出家族）**。

---

## 一、需要用户拍板的 3 组关键决策

### 决策 1：总请求预算上限与可接受运行墙钟时间（基于 P1 实测吞吐 `352.3 req/h` @ 3 并发）

| 档位选项 | 适用实验规模设计 | 预计总 Actor 请求上限 | 3 并发预计墙钟时间 | 6 并发预计墙钟时间（若扩容 vLLM） | 备注 |
|---|---|---:|---:|---:|---|
| **选项 B1（轻量快速档：10,000 req）** | `D1` 抽样 9–12 个缺陷单元 × 每臂 `n=12` + `R12` 正确样本伤害检查 4 单元；`H` 留出验证 9 家族 | `≤ 10,000` | `~28.4 h`（1.18 天） | `~15–18 h` | 最适合先跑通 `D1 -> D2` 检验 `V3-P3` 价值上限门槛 |
| **选项 B2（标准均衡档：20,000 req）** | `D1` 抽样 15–18 个缺陷单元 × 每臂 `n=12`（边界单元扩至 `n=24`）+ `R12` 伤害检查 6 单元；`H` 留出验证 10 家族 | `≤ 20,000` | `~56.8 h`（2.37 天） | `~30–34 h` | 兼顾功效与周期的标准配置 |
| **选项 B3（高功效全量档：40,000 req）** | `D1` 抽样 24+ 单元 × 每臂 `n=24` + 全套 `H` 留出阶梯复验 | `≤ 40,000` | `~113.5 h`（4.73 天） | `~60 h` | 需多日连续占用 `aml-b` 双卡 |

---

### 决策 2：阶段 H 留出家族名单（19 个留出家族中，10 个完全未暴露 vs 9 个曾被旧 DeepSeek holdout45 使用）

根据 `L1_sandbox/C3_Experimental_Sandbox/benchmarks/officeval/family_split.json` 与历史运行记录核查：

#### 2.1 完全未暴露的 10 个留出家族（推荐作为 Stage H 主验证集，零历史接触）

| # | 留出家族 ID (`family_id`) | 包含任务 ID (`task_id`) | 应用分层 (`application_stratum`) | 建议 Stage H 代表任务 |
|---:|---|---|---|---|
| 1 | `bilingual-contract-layout` | `officeval_034` | `word` | `officeval_034` |
| 2 | `image-to-word-table` | `officeval_029` | `other` | `officeval_029` |
| 3 | `office-embedded-links` | `officeval_059`, `officeval_078` | `other` | `officeval_059` |
| 4 | `pdf-contents-page` | `officeval_017` | `other` | `officeval_017` |
| 5 | `pdf-exam-layout` | `officeval_027` | `other` | `officeval_027` |
| 6 | `pdf-resume-edit` | `officeval_032` | `other` | `officeval_032` |
| 7 | `ppt-merge-restructure` | `officeval_068` | `ppt` | `officeval_068` |
| 8 | `ppt-multimedia-animation` | `officeval_042, 044, 045, 048, 061, 062, 065, 067, 076, 077` | `ppt` | `officeval_042` |
| 9 | `ppt-template-restyle` | `officeval_075` | `ppt` | `officeval_075` |
| 10 | `ppt-translation` | `officeval_079` | `ppt` | `officeval_079` |

#### 2.2 曾在旧 `deepseek_holdout45_20261017` 中运行过的 9 个留出家族（未在 `Qwen3.6 + pi` 下跑过，但已暴露过评分细则分布）

| # | 留出家族 ID (`family_id`) | 旧 `deepseek_holdout45` 已跑任务 | 同家族其余任务 |
|---:|---|---|---|
| 1 | `excel-editable-shapes` | `officeval_093` | - |
| 2 | `excel-interactive-form-macro` | `officeval_082` | `officeval_084` |
| 3 | `financial-extraction-to-word` | `officeval_036` | - |
| 4 | `pdf-exam-to-word` | `officeval_006` | `officeval_031, 037, 038` |
| 5 | `pdf-translation-to-word` | `officeval_035` | - |
| 6 | `ppt-composite-edits` | `officeval_041` | `officeval_063, 071, 080` |
| 7 | `ppt-data-chart` | `officeval_066` | - |
| 8 | `ppt-label-annotation` | `officeval_058` | - |
| 9 | `word-answer-card` | `officeval_015` | - |

- **备选方案 H-A（严格主线）**：Stage H **仅使用 2.1 的 10 个完全未暴露家族**（每个家族取 1 个代表任务 × 5 次重复 = 50 条轨迹，或取其中 9 个家族 × 5 次重复 = 45 条轨迹）。
- **备选方案 H-B（分层报告）**：以 2.1 的 10 个完全未暴露家族作为主表，若需覆盖 Excel 留出族则将 2.2 中未跑过的同族兄弟任务（如 `officeval_084`）作为次级敏感性附录，严格分开报告。

---

### 决策 3：统计裁决阈值（`δ, τ`）、`V3-P3` 价值上限门槛、`V3-P1` 比例阈值与第 1 级臂 `S` 措辞定稿

#### 3.1 统计与预言阈值候选

| 参数/预言 | V3 草案当前默认值 | 保守备选 | 宽松备选 | 含义说明 |
|---|---|---|---|---|
| `δ`（单侧 95% 置信下界门槛） | **`0.10`** | `0.15` | `0.05` | 要求高阶臂相对全部低阶臂的增量 95% 下界 `LB95 > δ` |
| `τ`（点估计最小效应量 & 未恢复上界门槛） | **`0.20`** | `0.25` | `0.15` | 要求均值增量 `≥ τ`；且仅当所有评估臂（含 `P`）绝对上界 `UB95 < τ` 时才判为 `unrecovered_under_protocol` |
| `V3-P3` 交叉拟合诊断价值上限门槛 | **`0.05`** | `0.08` | `0.03` | 两折交叉拟合下，按单元选臂相对固定单臂的平均额外挽回率门槛（不过关则不跑 `D3` Analyst 评测） |
| `V3-P1` 第 1 级通用干预可救回比例门槛 | **`20%` (`0.20`)** | `25%` (`0.25`) | `15%` (`0.15`) | 第 1 级通用干预（`D` 或 `S`）在 `never_succeeded_on_disk` 扣分中的 HT/Hájek 加权挽回占比下界 |

#### 3.2 第 1 级通用自查臂 `S`（结束前自查）措辞候选（不含任何具体任务规则或评分器代码）

- **候选 S-1（当前代码默认草案，`sha256: 7f05c813022a99c3be0ccca6c7df75f514f41f5e76898912d6028a27bc49fc24`）**：
  > `在结束任务前，请先对照原始任务要求和输入材料完整自查一遍当前工作区与 /workspace/submission 中的结果，确认所有要求的内容、格式与文件均已正确完成并保存；如有遗漏或问题请先修正后再结束。`
- **候选 S-2（先嵌套 `D` 原文前缀 + 自查指令，严格满足 `D ⊂ S` 字符级前缀嵌套）**：
  > `请停止探索，在剩余步骤内把当前已完成的结果整理并写入 /workspace/submission 目录下的目标交付文件中。\n\n在最终结束前，请对照原始任务说明自查一遍交付文件是否完整可读、是否遗漏了题目明确要求的步骤；如有遗漏请修正后再结束。`
- **候选 S-3（极简自查提示）**：
  > `请不要立即结束：先打开并检查你准备交付的文件，对照原始用户指令自查是否有遗漏或格式错误，修正并保存到 /workspace/submission 后再结束。`

---

## 二、D0 一次交付核对表（对应 `scripts/v3_d1_ladder.py --plan-only` 产出）

1. **各级处理臂消息模板与 SHA-256 哈希**：已生成于 `L2_Behavioral Event Localization/L2_Sandbox/reports/v3_d1_ladder_plan_20261017.json` 的 `task_ladder_message_specs` 字段（含 `P`, `D`, `S`, `R_explicit`, `R_implicit`, `R_single` 及嵌套校验 `nesting_check.valid == True`）。
2. **整任务要求清单及空级 `ineligible` 标记**：按整任务生成（不按单次失败项筛选）；当某任务的 `explicit` 或 `implicit` 列表为空时，显式标记 `eligible: False, evidence_status: "ineligible", message: None`，严禁回退为 `D`。
3. **人工抽查表**：已在 `docs/V3_P0A_AUDIT_20261017.md §2.2` 生成 63 行人工复核抽样表（12 `explicit` + 6 `implicit` + 45 `ambiguous`），留白待人工签字。
4. **锚点规则（`R10`）**：同时覆盖 `budget_tail_no_candidate`（最晚仍余 `r_b=6` 请求的合格点）与 `voluntary_pre_finish`（`termination == "assistant_finished"` 时主动结束前最后合格点）。
5. **三维分层抽样框与抽样概率（`R11`）**：仅在新 `v3_qwen36_pi_dev45_20261017` 上按 `耗尽请求 × 合法交付 × 规则来源` 构建，记录每层总体数 `N_h`、样本数 `n_h` 与包含概率 `inclusion_prob = n_h / N_h`。
6. **停止条件与预算硬上限（`R18`）**：已写入 `v3_d1_ladder_plan_20261017.json` 与 `v3_d3_analysts_plan_20261017.json`。
