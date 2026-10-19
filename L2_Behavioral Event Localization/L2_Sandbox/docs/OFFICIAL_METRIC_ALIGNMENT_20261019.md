# OfficeVal 官方指标对齐

对齐本批实际固定的 OfficeVal 源版本 `5f3da616ac6a0bc30ca35e9678f8f1eb255bd91a`。本次只新增报告投影，不调用模型、工具或评分器，不修改封存轨迹、官方原始返回值或 L2 的 0–1 研究端点。

## 主指标

采用官方字段 **`completion_rate`（完成度）**。计算及显示与官方一致：

```text
status == "ok" 且 dim1_pass == false：0.0%
否则：max(0, total_score) / max_score × 100%，显示一位小数
max_score 等字段无法支持计算时：留空，不伪装成有效得分
```

官方逐任务 JSON 是权威结果；`summary.csv` 和 `details.csv` 是这些结果的表格投影。规则可包含负向扣分，不能将完成度改算为“做对条数 / 规则条数”。依据：[官方结果格式](https://github.com/baidu-frontier-research/OmegaUse-OfficeVal/blob/5f3da616ac6a0bc30ca35e9678f8f1eb255bd91a/docs/result-format.md)、[官方 CSV 计算](https://github.com/baidu-frontier-research/OmegaUse-OfficeVal/blob/5f3da616ac6a0bc30ca35e9678f8f1eb255bd91a/core/csv_reporter.py#L61-L73)。

本研究内部保持 0–1 的未舍入数值用于统计计算；这是与官方百分比相同的量纲换算。跨运行主汇总标为“本研究对 45 次 official completion_rate 的等权平均”，先对未舍入值取平均，再显示百分比，不对已舍入 CSV 字符串取平均。

当前九任务开发子集 × 五次重复的均值为 **22.30%**。这是本研究的汇总方式；固定官方 evaluator 的 `summary.json` 汇集任务结果及运行元数据，没有另行规定全局均值、软件类型加权总分或满分成功率。依据：[官方汇总](https://github.com/baidu-frontier-research/OmegaUse-OfficeVal/blob/5f3da616ac6a0bc30ca35e9678f8f1eb255bd91a/core/result_store.py#L181-L189)。

## 零分与错误

缺交付物是正常的业务零分，必须计入分母。原始 verifier 可能将明确的缺文件返回为 `error`；官方 `build_system_result` 会将这类情况转为 `status="ok"`、`dim1_pass=false`、`total_score=0`，再由官方 CSV 输出 `0.0%`。其他解析或基础设施错误保持错误，不冒充业务零分。依据：[官方原生后处理](https://github.com/baidu-frontier-research/OmegaUse-OfficeVal/blob/5f3da616ac6a0bc30ca35e9678f8f1eb255bd91a/core/result_store.py#L106-L164)。

本批 085 的五条原始 verifier 返回明确的缺必需文档错误；官方对齐投影执行上述原生后处理后记正常零分。原始返回和派生结果分开保存，转换字段在绑定记录中逐条披露。该对齐不改变本批 45 条研究端点，也不删除这五条。

## 辅助指标与范围

- `dim1_pass` 及其比例只表示通过交付可用性门槛。本批为 40/45。
- 非零得分数 25/45、满分数 3/45，均为本研究的辅助诊断，不命名为官方 `success_rate`。
- `status="ok"` 表示评分正常完成，不能当作任务成功。
- 上游另有 `agent_success_rate`，统计原生 agent runner 的完成状态，不能替代交付物评分，也不适用于直接给本批 Pi 轨迹贴“官方成功率”。依据：[官方 agent runner 汇总](https://github.com/baidu-frontier-research/OmegaUse-OfficeVal/blob/5f3da616ac6a0bc30ca35e9678f8f1eb255bd91a/officeval_runner/src/reporting.py#L49-L67)。

本批使用官方 normal-mode verifier、我们的 L1/Pi 执行环境及固定 Linux 镜像。官方函数后处理和 CSV 投影不表示重新执行了完整官方批量 runner，也不将九任务开发子集冒充完整 100 任务榜单或上游原生 agent 配置。

## 可重跑导出

使用 `scripts/export_officeval_official.py`，输入封签 cohort 和新 DEV45 根目录，输出到独立新目录。在 aml-b CPU 执行；不需要 GPU、API key 或模型服务。五次重复各生成一个官方格式 job，保留每次九个不同任务编号，避免把重复任务混成同一编号覆盖。

原始 verifier JSON、官方后处理编号 JSON、官方 `summary.csv` / `details.csv`、输入绑定及导出回执均另存。实际导出目录与验证结果记录在同名 JSON 中。新池 E1、恢复机制与完整无泄漏后审计仍分别保留其尚未执行状态。

2026-10-19 已在 aml-b CPU 完成实际导出和独立核验：45 行官方 CSV、45 份字节相同的原始返回副本、45 包导出后封签复验均通过；五条缺交付物转换为正常零分，分母保持 45。另验四类完成度边界及五次错误调用拒绝（包括防覆盖、禁止写入证据目录、非开发任务和无效槽），全部通过。未新增 Actor、Analyst 或评分器调用。

提交的五份官方格式主表：[r00](./official_dev45_20261019/r00/summary.csv)、[r01](./official_dev45_20261019/r01/summary.csv)、[r02](./official_dev45_20261019/r02/summary.csv)、[r03](./official_dev45_20261019/r03/summary.csv)、[r04](./official_dev45_20261019/r04/summary.csv)。完整原始结果、标准化结果、逐规则表及绑定记录保留在本机 `validation/a_plus_i_execute_20261018/official_officeval_export_20261019/`。
