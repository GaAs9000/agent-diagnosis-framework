# L1 重塑路线：保留旧代码，先独立证据边界

## 已实施的切割

新增 `c3_sandbox.l1` 与独立命令 `c3-l1`。旧执行器、DockerRuntime、OfficeVal 评分器、gateway 和历史模块均未删除或改写。
L1 入口不经过旧 integrated 初始化，不构建额外角色，也不依赖 Inspect、MLX 或任何分析模型。

新 `contracts.py` 给出 PublicTask、ExecutionBudget、BackendIdentity、ModelRequest、ModelReply、ToolInvocation、ToolExecution、CheckpointQualification，以及 BenchmarkAdapter、ModelBackend、IsolatedSandbox 端口。
**它们目前是接线合同，而不是 API／本地两条生产链路已经通过同一执行验收的证明。**

## 源码位置与复用方向

| 组件 | 当前候选来源 | 本轮结果 |
|---|---|---|
| Benchmark Adapter | r1/officeval.py；现有 frozen bundle | 已读取固定任务、核对资产并导出评分材料；新实时适配器待接线 |
| Model Backend | gateway/backend.py；api_backend.py；local_backend.py | 旧实现保留；L1 独立端口已定义，未迁移实时循环 |
| Isolated Sandbox | r1/docker_runtime.py | 保留；本环境没有 Docker，未重新验收 |
| Trajectory Recorder | 新 l1/recorder.py | 排他、追加、fsync、哈希链；中断尾部可检查；尚未嵌入旧模型循环 |
| State Recorder | 旧 tool_dispatch；包内初态／终态 | 报告级逐步差异和重新计算的端点差异分开；独立对象采集待实现 |
| Independent Evaluator | r1/grader.py | 保留真实评分入口；本轮只核对留存评分，不重新运行 |
| Evidence Package | 新 l1/import_gemini.py、package.py、projection_checks.py | 8 条历史记录批量导出与离线核验完成 |

## 下一次实施：先贯通一个新的 API episode

新增独立 `c3-l1 run --config ...`，只包含 benchmark、agent、backend、sandbox、recording、evaluation 和 budget，不再复用多角色研究入口。

模型调用前先持久化实际请求及身份；收到响应后保存原生响应，原生工具调用进入唯一 tool_call_id 的受控调度。执行前后采状态，原始工具输出与模型可见输出分别持久化，再构造下一次原生请求。
最后在 Agent 不可接触的评分环境中评价成品；无论模型完成、预算耗尽、工具失败或评分器错误，统一进入证据关闭流程。

不要把 unit-test transport 当真实模型。API 路径不得因 Metal 检查失败被阻断；本地后端明确报不可用，不偷换 API、不借用 saved_reference、不伪称真实运行。

首轮 live 验收只需现有一个任务，但必须具有新请求、新容器执行、原始 stdout／stderr、真正的前后快照、官方评分过程和统一包；最终任务失败也可以通过 L1 的执行证据验收。

## 分阶段门槛

**P1：**当前 Evidence Package／导入剖面已实测；端口合同存在。实时 Episode 配置与两个后端的合同一致性仍待接线验证。

**P2：**API 和本地分别独立执行真实工具调用。生产代码读取模型原生调用，不从普通文本中假装执行。记录协议差异，不强求内部消息协议相同。

**P3：**模型和工具的所有终止路径均归档。逐步快照必须能重新计算差异，事件 ID 贯通请求、调用、输出和版本；再接入明确版本的对象状态提取器。不能用成功任务或高分掩盖证据缺口。

**P4：**检查点绑定精确原生对话前缀、环境／模型／工具身份和已应用配置。冻结后台写入，资格检查明确包含与排除的状态；恢复后必须发生新模型请求与真实工具调用。未支持的外部状态、进程内存等必须拒绝资格。文件还原不等于 Agent 恢复。

**P5：**对 live 的成功、失败、部分完成、预算终止、协议无效、评分失效及中断分支执行统一导出验收。当前 8 条历史包通过，不能替代这些 live 分支的验收。

## 当前不做

不增加任务数，不优化被测模型得分，不添加解释模型，不产出错误归因／能力画像，不删除旧研究模块来假装已经完成运行时解耦，不把缺失证据补成合成证据。
