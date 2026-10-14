# C3-E2 Integrated Closeout

日期：2026-10-11。此次修改只收尾集成软件，不改变历史运行和研究结论。

## 软件状态

`implementation_ready=true`。唯一入口 `scripts/run_integrated_delivery.py --execute` 现在会：

1. 一次性核验固定八题、资产、验证器、模型合同、镜像、tokenizer 和绝对路径服务配置；
2. 写入 `config.lock` 与 `run_manifest.json`；
3. 串行启动目标/参考模型服务，逐任务调用真实 `run_episode`；
4. 对服务身份失败做一次全局 `setup_failed`，其余计划项写成 `NOT_RUN`；
5. 只有真实 runner 返回的记录才进入完成、失败或未知状态。

`RealBranchExecutor` 现在可以接收 R12 完整系统合同，派生模型、解析器、提示词、工具 schema、采样上限和协议恢复策略，并以检查点剩余请求数续跑。旧 R1 调用方式保持兼容。

D6 使用 typed relation graph；重复 key 产生 conflict relation，终态评分通过 final artifact/rule 关系连接，不要求和工具事件共享 ID。D8 按 `branch_key` 保留每一条 forecast，计算逐分支 Brier、clipped LogLoss、逻辑成本、物理成本、invalid 和 unknown。D7 正式路径使用 `DiagnosticBackend`，保存原始 request/response、usage、模型身份、checkpoint 和 evidence IDs；`FixturePredictor` 只保留为显式测试基线。

实验卡工作流已接入 closeout：每个案例记录现象、证据引用、竞争解释、候选实验、算子编译状态和版本链；未运行案例写 `NOT_RUN`，不把服务失败替换成预测或分支结果。注册算子之外的自由脚本会被拒绝。

## 本机真实入口验证

本机实际运行：`runs/integrated-real-001/`。

入口完成 preflight 和配置锁定后，启动 MLX 服务失败：当前命令环境没有可用 Metal device，模型和 tokenizer 端口没有打开。因此目标首任务为 `setup_failed`，其余 15 条计划项为 `NOT_RUN`，没有模型请求、工具调用或 Docker 分支。该结果不被解释为 Agent 任务失败，也没有回退到 fixture。

## 验证

全套纯 Python 回归：**615 passed、11 subtests passed、35 warnings**。测试覆盖真实入口的注入 runner、全局 preflight、状态归一化、D7 原始字节和无 fallback、D6 冲突/未匹配关系、D8 多 branch 计分、closeout NOT_RUN 输出及 R12 分支合同。

本轮没有获得 live path verified，也没有改变自然成功侧、科学效果或 C3 胜负结论。重新运行需要在可见 Metal 的本机环境中执行同一 config 和入口；不得把本次 setup failure 改写成业务结果。
