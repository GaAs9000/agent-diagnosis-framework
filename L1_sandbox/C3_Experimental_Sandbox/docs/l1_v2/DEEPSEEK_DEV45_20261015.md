# DeepSeek V4.1 Flash：L1 开发侧 45 条轨迹交付

9 个开发家族各选 1 个代表任务，每任务 5 次独立初态，共 **45/45 个有效完整槽位**。2026-10-15 香港时间 20:25:38 开始，22:24:38 执行结束；全部 47 份原始尝试通过独立封签、模型身份、评分、资产、镜像及账本核验。

## 结果

| 指标 | 结果 |
|---|---:|
| 非零完成度 | 26/45 |
| 满分 | 3/45 |
| 零分 | 19/45 |
| 未形成合法交付的零分 | 16/45 |
| 合法交付但官方零分 | 3/45 |
| 有文件产出 | 37/45 |
| 合法交付 | 29/45 |
| 平均官方完成度 | 0.2575327940 |
| 请求预算耗尽 / 主动结束 | 24 / 21 |

平均完成度是连续评分的均值，不是成功率。未交付与质量零分保留；无效尝试单列，不算作有效零分。

| 任务 | 开发家族 | 有效数 | 平均完成度 | 非零 / 满分 |
|---|---|---:|---:|---:|
| officeval_003 | word-table-structure | 5/5 | 0.1200 | 2 / 0 |
| officeval_004 | word-template-formatting | 5/5 | 0.3455 | 3 / 0 |
| officeval_010 | word-diagram-edit | 5/5 | 0.0000 | 0 / 0 |
| officeval_033 | word-price-computation | 5/5 | 0.2235 | 4 / 0 |
| officeval_050 | image-to-editable-ppt | 5/5 | 0.0400 | 4 / 0 |
| officeval_060 | ppt-static-relayout | 5/5 | 0.5273 | 4 / 1 |
| officeval_070 | ppt-diagram-edits | 5/5 | 0.6615 | 4 / 2 |
| officeval_085 | excel-chart-creation | 5/5 | 0.0000 | 0 / 0 |
| officeval_090 | office-extraction-aggregation | 5/5 | 0.4000 | 5 / 0 |

## 执行与用量

OpenCode Go，`deepseek-v4.1-flash`；T=1、thinking disabled、无 seed、最大输出 8192。每槽 35 次模型请求、140 次工具调用、10800 秒；并发 2；RPM 30、TPM 1000 万；HTTP 准入硬上限 6300。此预算尚未经 DeepSeek 预算对照校准。

全部 47 份尝试合计 1308 次逻辑模型请求、1308 次 HTTP 尝试、1401 次工具调用。1307 次 HTTP200，1 次 TimeoutError 用量未知；已知 23,684,382 tokens，其中 prompt 23,232,664、completion 451,718、已知缓存 prompt 21,840,891（缓存是 prompt 的子集，不另加总）。实际费用未知。

协议预检单列：3 次 HTTP（2 次成功、1 次早期地区限制 HTTP400），已知 1170 tokens，1 次用量未知，不计入 45 条。

`officeval_050.requests_035.r03` 的前两次尝试分别为图像文件不存在导致基础设施终止、请求准入等待超时；第三次有效，完成度 0.05。原件全部保留，没有按得分选择替换。

## 验收与封存

独立验收核对了 45 个唯一有效槽位、9×5 完整 cohort、47 个独立 episode/session、全部证据封签、官方评分原件、所请求及返回的模型身份、任务资产、镜像、冻结源码及 HTTP 账本。55 份源码/配置快照逐文件哈希一致；95 份旧 pilot 元数据及索引哈希未变。

初版验收脚本把已冻结的镜像 digest 当作 tag 查询，随后空均值遮蔽了诊断。保留首轮失败回执；v2 只修正 tag/digest 解析与空均值诊断，未删除语义门禁。v2 全量验收通过，errors=[]；两条警告对应保留的无效尝试。

执行代码：`3fe7f89b0b20b21eac983679e4db3f909194de86`。计划 SHA-256：`368ed77f80e584dabdca4c5d8815fa39e88292ddafeddcafa39bf3c8d13fe421`。独立验收 SHA-256：`5cd286126b9a30a51fa42aef942bd51c4fca7ff43286d47c67cef8c496576b71`。

本机运行目录 `L1_sandbox/C3_Experimental_Sandbox/runs/deepseek_dev45_20261015/` 包含原始请求/响应、HTTP 回执、工具输出、工作区状态、检查点、交付件、官方评分、manifest/index、冻结计划与源码、`summary.json`、`m2_baseline.json` 和 `verification.json`。ZIP 与外层逐文件哈希目录及归档 SHA-256 在交付目录中；不将密钥、题库或轨迹提交 Git。

## 后续边界

已导出 9 个同配置完整 cohort 的 p₀，并为每条父轨迹记录排除自身后的其余四次均值。仅适用于相同模型、任务、配置和研究终点；不混用 Gemini pilot。尚未采集 19 个留出家族，未启动 L2 Analyst、M3/M4 或实验 B。下一步是按已定设计在开发侧选择合格失败轨迹进入 L2；Notion F20 仍待修。

这些是开发侧执行证据和描述性评分，不能证明 L2 定位准确率、因果机制或稳定能力边界。

## 请求计数补充

1308 次逻辑请求和 1308 次 HTTP 数值相同，计数对象不同：1307 个请求目录实际发送过 HTTP，其中一次多发 1 次重试；另有一次逻辑请求在准入阶段超时，未发送 HTTP。两次无效执行消耗的 34 次 HTTP、284,908 已知 tokens 已计入总数。不能把未知请求的账本预留 3,986,025 tokens 当作实际消费。

完整账务明细保存在本机 `validation/deepseek_dev45_preflight_20261015/postrun_accounting_detail_20261015.json`。

## 本机交付位置

- 原始批次：`/Users/jiashen/Desktop/问题诊断/L1_sandbox/C3_Experimental_Sandbox/runs/deepseek_dev45_20261015`。
- 完整 ZIP：`/Users/jiashen/Desktop/问题诊断/L1_sandbox/deliverables_by_model/06_deepseek-v4.1-flash_dev45_20261015/deepseek_dev45_20261015.zip`（4,608,820,159 bytes）。
- ZIP SHA-256：`a9f55f6aab03de01d0c3861be5c3b5a011cd279dd303174f6a965833c6c43930`。
- 逐文件目录：`/Users/jiashen/Desktop/问题诊断/L1_sandbox/deliverables_by_model/06_deepseek-v4.1-flash_dev45_20261015/deepseek_dev45_20261015.files.json`（SHA-256 `06bd004dea04a862b218b9ad347871575a9bb7de9e3938d096b3d8479abf5d0c`）。
- 全部 45,598 个批次文件解压后逐一通过大小和 SHA-256 校验；原始批次未变。
- 归档回执：`/Users/jiashen/Desktop/问题诊断/L1_sandbox/deliverables_by_model/06_deepseek-v4.1-flash_dev45_20261015/deepseek_dev45_20261015.archive.json`。
