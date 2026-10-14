# L1 P1 重塑与实测审计 · 2026-10-12

## 交付判定

**已交付：独立 L1 证据核心、Evidence Package 0.1.0 的 Gemini 历史导入剖面、离线校验／读取命令及 8 份真实来源归档的规范化证据包。**

**未交付：完整的新实时 L1 运行器、API／本地双后端现场验收、逐步状态全量保留、对象级采集、冷恢复或分支续跑。**

本轮未发起新的 Gemini 任务，也未运行 Docker 评分器。归档的 `is_synthetic=false` 是来源声明；本轮验证的是来源一致性和可核验导出，而非第三方证明历史执行发生。

## 1. 输入基线

原始压缩包：`C3_Experimental_Sandbox_2026-10-12.zip`。

SHA-256：`39fff1b54e7ee7074002cdcfcd5cae8f0a18069a1f3c7407a3b19154c6a7a9ae`。

包内共 462 个文件，解压后条目总字节数 165865124。核查源码、固定 OfficeVal bundle 和 `runs/gemini-real-001/target/*/result.json`，没有把打包说明中的结论直接当作验收结果。

## 2. 已实施代码

新增 `src/c3_sandbox/l1/`：端口合同、实际使用的 JSON Schema、安全存储、追加式 EventJournal、Gemini 导入器、包索引与语义校验、独立 CLI。
新增 `tests/l1/`。更新 README 和 pyproject 的 L1 可选依赖、命令与 Schema 打包声明。
旧 README 原文保留；旧运行器与评分器源码未改写。

## 3. 8 条轨迹逐项核对

| 任务 | 模型请求 | 工具执行 | 可核实送达 | 终止原因 | 缺失前后快照文件 | 旧评分规则覆盖完整 |
|---|---:|---:|---:|---|---:|---|
| officeval_002 | 20 | 20 | 19 | budget_exhausted | 40 | true |
| officeval_003 | 20 | 20 | 19 | budget_exhausted | 40 | false |
| officeval_006 | 20 | 20 | 19 | budget_exhausted | 40 | true |
| officeval_012 | 20 | 20 | 19 | budget_exhausted | 40 | true |
| officeval_015 | 14 | 13 | 13 | assistant_finished | 26 | false |
| officeval_018 | 20 | 20 | 19 | budget_exhausted | 40 | true |
| officeval_033 | 20 | 20 | 19 | budget_exhausted | 40 | true |
| officeval_037 | 20 | 20 | 19 | budget_exhausted | 40 | false |

总计 **154 个请求、154 个响应、153 次工具执行、146 次后续请求中的 Observation、800 个规范化事件**。
源记录有 6 条逐步文件差异，本轮从端点重算得到 6 条文件差异，共 12 条状态证据；它们不能被当作 12 次不同操作。

8 份包的 Schema、索引、事件链、相对引用、原生请求／响应字节、调用配对、输出来源、状态映射和留存评分映射均通过离线校验。
全部旧 completion 布尔值为 false，按原政策保留为 failure，没有重新定义“部分成功”来美化结果。评分一致性通过不代表全部规则都被官方实现覆盖：003、015、037 的原记录 coverage_complete 为 false。

## 4. 决定当前资格的证据缺口

**缺失 306 个逐步快照文件。**153 条 dispatch 各引用 before_path 与 after_path，但归档没有这些文件。旧 fs_delta 仍然保留，却不能由前后快照重新核算。本批不能恢复任意历史步骤。

**8 条 Gemini 记录的 checkpoints 均为空。**不能借用其他运行的资格凭证，把本批 Gemini 包标成可恢复。每份包 cold_resume_qualified 均为 false。

**缺少 Agent 侧 runtime provenance。**评分器环境证据不能替代 Agent 执行环境证据。8 条旧记录中的 execution_valid=true 只作为 raw 原值保留，L1 的完整执行资格设为 unknown，而不是直接继承。

**声明 seed 不等于实际发送 seed。**runtime.sampling.seed=2026，但这 154 个原生请求中没有 generationConfig.seed。原生请求是配置应用的依据；不能据声明值声称确定性复现。

**没有完整 resolved config。**现有 config.lock 保存配置摘要、seed 和 task_ids，不能单靠摘要还原完整执行设置。本轮不从当前配置模板反推历史设置。

**缺少模型请求时间、生成延迟、实际计费及不可变模型指纹。**保留 provider token usage 和 modelVersion 标签；不把别名当权重摘要，不把费用填为 0。

**7 个末次工具结果没有下一请求中的送达凭证。**raw 结果仍在，但没有人工补造 observation.delivered。

以上缺口均写入 manifest.coverage 或 raw/source_path_audit.json，不会被 package verified 覆盖。

## 5. 发现并在新合同中规避的旧耦合

`integrated/gemini_episode.py::_episode_outcome` 和 `gateway/gateway.py::_episode_outcome` 把初态与终态评分有效性、循环终止和任务完成折叠在一起。新包将协议、执行、评分、任务结果、证据完整性分开。

旧 `gateway/__init__.py` 和 `integrated/__init__.py` 会导入研究角色或分析模块。新 L1 入口不经过它们。它们仍保留供历史兼容，不等于实时主链路已完成迁移。

`r1/r11_replay.py::DispatchRecorder` 的 before／after 是工作区快照来源。只保留终态文件与 fs_delta 不等于保留完整状态证据。

## 6. 测试与运行环境

新增 L1 测试：**70 passed**，覆盖原始字节保留、状态分离、原生／旧调用 ID、原始输出／可见输出、缺少下一请求、无响应错误、显式协议错误、无效评分、缺失快照、检查点降级、孤立调用、篡改与重新封装、语义覆盖、移动包、路径逃逸、软链接、尾部中断和无旧 Agent 依赖等。

旧全量测试原样尝试（允许收集错误继续）：**426 passed、42 failed、2 skipped、37 errors**；另有 11 个子用例通过。
这 37 个错误包含 12 个模块收集错误。典型阻塞是缺少 `inspect_ai`、未打包的外部 OfficeVal Git 仓库和部分历史运行凭证。没有把这些失败改为通过或静默忽略。

因此，原打包说明中的“640 项全部通过”“新环境开箱即用”**不作为这次交付验收结论**。这也不是声称 42 个失败都已做出逐项根因判定；完整 traceback 保留供复核。

此外，已从实际构建的 wheel 读取 Schema 并独立核验全部 8 份包，未加载旧 Agent 或分析模块。

当前环境为 Python 3.13.5，jsonschema 4.26.0，pytest 9.0.2；没有可执行 Docker 命令，未安装 Inspect 或 MLX。L1 离线链路独立于这些依赖。

## 7. 文件与报告

代码：`L1_P1_Refactor_2026-10-12.zip`，作为原目录的叠加补丁，不含模型权重。
数据：`L1_Gemini_Evidence_2026-10-12.zip`，8 个自包含 episode；不需要旧 Mac 绝对路径。

验证原件在 `docs/l1/validation/`：L1 pytest 日志与 JUnit、旧测试日志与 JUnit、汇总数据及独立包核验结果。批次包内 `episodes/import_report.json` 保存每个 evidence_index.json 的摘要。

## 8. 阶段结论

完成的是用户指定起点：**先冻结证据合同，以已有 Gemini 轨迹逐字段验证能否准确导出**。导出成功，但历史证据本身不完整；因此本轮既不能宣布 L1 完工，也不能通过重命名文件弥补运行时采集缺口。

下一次最小闭环应是：现有一个 OfficeVal 任务 → 新独立 API 入口 → 原生工具调用 → 容器真实执行 → 当场落盘的完整证据 → 独立评分。不要先加任务或恢复研究角色。
