# 跳过人审：执行口径与验证（2026-10-18）

用户直接指示「跳过人审核」。本次取消要求清单与 G3 的真人签署前提，改用自动来源／执行证据核验。机器授权记录为 `configs/human_review_policy_20261018.json`；既有十项批准保留，其余条目维持 `automatic_unreviewed`，不补审阅人、日期或真人结论。

## 当前行为

- `build_task_requirement_lists` 将实际 `review_complete` 与 `execution_eligible` 分开。明确的用户豁免可免除人审；来源引用、有效分类和可见／精确规定拆分仍须齐备。没有豁免时保留原严格默认。
- D1 CLI 默认采用这份授权与新的自动清单／测量审计。计划保存授权路径及内容哈希，实际执行前重新核对。用户豁免不改变请求预算、统计阈值、留出隔离或 G1／G2／G4。
- 自动清单从九个开发任务的完整公开指令、188 条原始 rubric 与静态评分器描述构建，未按本次失败项筛选。当前 150 个实际评分别名全部覆盖。十项原批准只在精确样本文字或指针＋原文件哈希＋原引文一致时继承，共映射到 27 个别名；其余分类保守为 ambiguous，明确未经人审、未检查原图。
- 评分别名仅用于结果匹配，不能把去掉扣分符号的描述注入成修复命令。提示采用原始有正负分符号的 rubric；显式侧只使用公开任务原文或已批准的公开引文，不夹带评分权重。
- G3 复核已有 20 个快照的实际恢复文件、来源、22 份官方评分及进程原件；不再要求真人签字。97 个恢复文件核对通过。五个有候选但官方评分未知的点仍未知，自动核对不等于目视正确或官方评分全部有效。没有目录、没有评分调用或证据哈希为空时拒绝通过。

## 当前真实状态

上一批有界补采已于 **2026-10-18 01:53 HKT** 结束。四槽各尝试一次，两条转有效：`004.r02` 为有效得分、`010.r04` 为有效零；`010.r00` 仍为模型协议错误，`050.r04` 仍为传输错误。当前 **43/45 有效**，原定 20 对 G2 因槽位未齐而未启动。没有隐式追加重跑。

新增两条有效父轨迹的 E1 普查已补齐，复用其余 41 份输出，仅新增四次本地官方快照评分。本次政策修改没有新 Actor／Analyst 调用。

新 D1 草案中九个任务的要求均 `execution_eligible=true`，实际 `review_complete=false`。仅剩四项阻断：

1. `development_cohort_not_complete_45`。
2. `measurement_gate_not_passed:V3-G1`。
3. `measurement_gate_not_passed:V3-G2`。
4. `measurement_gate_not_passed:V3-G4`。

人审不再阻断。正式 D1 尚未启动，完整 D0 仍未冻结。

## 可复现入口

在仓库根目录执行；这些入口不调用 Actor／Analyst，自动审计复查已有评分原件，不新增评分：

```bash
B=L1_sandbox/C3_Experimental_Sandbox/.venv/bin
L2='L2_Behavioral Event Localization/L2_Sandbox'
$B/python "$L2/scripts/build_automatic_requirement_catalog.py"
$B/python "$L2/scripts/build_automatic_review_audit.py" --output "$L2/validation/d0_human_waiver_20261018/measurement_audit.json"
$B/python scripts/v3_d1_ladder.py --plan-only
```

实际审核状态、清单、快照审计和计划分别保存，历史十项签署与既有封签包不被改写。工程验收和文件哈希索引见同名 JSON；自动分类仍不构成真人金标准或诊断效果证据。
