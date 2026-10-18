# R22：Pi 原生错误回填与同尝试续接（2026-10-18）

## 协议与版本

用户已批准 R22a–f，计划在授权上限内无需再次确认。新执行配置为
`officeval.qwen36.pi.t1.native-validation.timeout600.json`，冻结
`validation_policy=pi-native-0.73.1`、`harness_revision=l1-pi-native-errors-v1`。
模型、权重、采样参数、35 次请求与 140 次工具预算保持既定口径；HTTP 等待窗口为 T-2 批准的 600 秒。

原生依赖固定为 Pi 0.73.1（commit `781152fc24841dc54b22284514604048ebe5e2c9`）。
桥接调用真实工具工厂和原生 `runAgentLoop` 的预处理、类型转换、校验及错误路径，
只用已记录的 assistant 响应驱动本地验证；禁止工具执行与网络 fetch。
验证在 aml-b CPU 运行，Mac 仅控制原先冻结的 ARM64 Docker 工具/官方评分环境。
安装前后依赖清单、锁文件及 941 个关键模块的批准哈希均受约束。
这里验证的是工具参数边界，不宣称整个 L1 等同于完整 Pi 原生 harness。

参数错误返回原生文本（含 Received arguments）和错误标记，继续同批其余调用及下一轮。
不修改模型参数、不添加纠错提示、不新增违约次数上限；违约调用计入工具预算。
解析失败、传输故障、原生校验器故障仍保留为基础设施/协议无效。
新增事实记录原始参数、原生错误、finish_reason、completion tokens 和后续可比较调用的纠正状态。
无法比较的后续调用记 unknown，不推断已经纠正。

## 不换样本的恢复边界

每个续接包保留旧尝试前缀和新段的独立封签，通过派生 cohort 连接。
直接消费已收到的错误响应，记 `response.consumed` 和零 HTTP 的 saved-response 事件，
回填错误后才发送下一次新请求。旧原始包不改写，也不把续接段的 cp0 当作原始任务 cp0。

| 槽 | 已消费请求数 | 保存响应 / 恢复检查点 | 最多新增 Actor 请求 |
|---|---:|---|---:|
| 050.r04（T-2） | 10 | r000009 / cp000009 | 25 |
| 004.r02（原始） | 8 | r000007 / cp000007 | 27 |
| 010.r00（指定补采尝试） | 24 | r000023 / cp000023 | 11 |

实际剩余合计 **63**，低于获批 **95** 次；HTTP（含重试）上限仍为 **380**。
004.r02 与 010.r00 现有替代样本仅在 cohort 元数据中降级为非正式对照，不删原件。
三个冷恢复点均已核对：响应完整、错误调用未开始执行、检查点为该响应之前、终态工作区与快照一致。

## 零模型复验与工程验证

42 条保留轨迹：**42/42 pass，767 次工具参数校验，744 条原生循环桩响应，
0 校验失败、0 原始参数差异、0 转换后有效参数差异**。
三个续接点之前另有 **42 次工具校验 / 39 条循环桩响应**，同样零差异；
三个预期错误响应另行验证。全部旧封签、清单、执行源码和依赖在复验前后不变。
Actor、Analyst、工具执行、评分调用均为 **0**。

审计摘要 SHA-256：`8eeebc1d48a68e73d5b2634863147625c74add9aa4352ca5f05ae4a5630eed6b`。
原始摘要、逐调用详情与失败点证明保存于
`validation/d0_20261018/r22_native_alignment/{audit42,plan_inputs}/`；均写入执行计划的绑定哈希。

最终在 aml-b CPU 分进程完成当前 L1 `tests/l1` 与 `tests/l1_live` 全量回归
（**401 passed / 87 skipped，35.27 秒**）及 L2 `tests` 全量回归
（**480 passed / 8 skipped，22.73 秒**）；监督器 17 项回归包含在 L2 总数内。
未把仓库历史 R1/Inspect 套件计入当前 L1 验收；这些旧套件在该环境缺 Inspect 依赖。
另验证 wheel 中桥接器、批准清单、锁文件及 Python helper 共 6 个资产与源码逐字节一致。
这些是工程证据，不能解释为模型表现或诊断有效性。

## 有界执行与结果边界

顺序：3 槽各续接一次 → 45 条逻辑尝试链均有效才运行原定 20 槽 G2 → CPU 敏感性汇总。
G2 从原尝试真实 cp000000 冷恢复，3 并发，600 秒，同一新 harness；不新增槽、不自动重开批次。
G2 最坏 2,800 HTTP；加恢复 380、此前 144，累计最多 **3,324 < 3,360**。
GPU 单卡、独立最长 6 小时租约；请求发送前需留出至少 645 秒，阶段不得续租。
监督器绑定计划与源码，记录唯一所属 Docker 容器，阶段收尾释放本批容器和 GPU 服务。
资源清理失败必须留 unconfirmed，不冒称释放成功。

敏感性 A：改用 004.r02、010.r00 的降级对照；
敏感性 B：剔除已知早期换样槽 010.r04、050.r04、010.r00。
两者均不覆盖正式口径、不删除旧证据。
G2 结果只汇报实际配对及原有区间；`pure_equivalence_eligible=false`，
历史 G1/G4 完整科学门禁仍未核定，`d0_frozen=false`，`scientific_efficacy=not_tested`。
本页记录工程协议和开跑条件，实际执行状态以封签 result/receipt/ledger 为准。

## 原生代码依据

- [agent-loop：prepare / validate / error ToolResult](https://github.com/earendil-works/pi/blob/781152fc24841dc54b22284514604048ebe5e2c9/packages/agent/src/agent-loop.ts#L529-L579)
- [validation：转换与错误文本](https://github.com/earendil-works/pi/blob/781152fc24841dc54b22284514604048ebe5e2c9/packages/ai/src/utils/validation.ts#L292-L323)
- [edit schema](https://github.com/earendil-works/pi/blob/781152fc24841dc54b22284514604048ebe5e2c9/packages/coding-agent/src/core/tools/edit.ts#L31-L50)
