# 本轮工程验证记录

日期：2026-10-10（Asia/Hong_Kong）。本记录只覆盖新建的 `C3_Experimental_Sandbox`，不是历史实验复验，也不是 C3 方法有效性结论。

## 已确认

- 本地 Python 3.12.14 隔离环境，Inspect AI 0.3.278；实际依赖见 `requirements.local-arm64.lock.txt`。
- ARM64 Docker 镜像已按固定官方 Python 基础摘要构建；镜像 ID 见 `configs/build_identity.json`。
- `.venv/bin/python -m pytest -q`：**45 passed, 11 subtests passed**。
- 配置计划可读取，部署并发为 1；真实实验仍为未就绪，缺失项逐条显示。
- `.venv/bin/c3 smoke --output runs/infrastructure-003`：**工程检查成功**。
- Inspect 记录 `is_synthetic=true`、`llm_calls=0`、`experiment_concurrency=1`；没有调用真实模型。
- 核对了实际容器的非 root 用户、只读根目录、网络关闭、capabilities、禁止新增权限、CPU/内存/进程上限及挂载集合。
- 任务容器中不存在 grader 程序、grader 参考文件和 Docker socket。
- 成品通过受限固定命令读取为字节；宿主冻结并计算 SHA-256，再传至独立 grader，接收侧回报相同摘要。
- 独立评分成功；本次创建的两个容器已清理。

主证据：[运行清单](../runs/infrastructure-003/inspect-smoke.json)、[冻结成品](../runs/infrastructure-003/frozen-artifact.json)、[Inspect 日志目录](../runs/infrastructure-003/inspect-logs/)。

## 测试覆盖与独立审查

合同测试覆盖：非法配置/并发、文件内容与集合/权限变化、隐藏和二进制文件、空目录、路径越界及软链接拒绝、存储损坏、预测候选缺失/概率非法/记录修改、评分字段缺失或类型错误、冻结成品传输与摘要、CLI 串行锁释放。

独立代码审查发现两个快照路径问题，均已修复并加入回归测试：含 `..` 的路径可能绕过证据目录排除；包含反斜杠的 POSIX 文件名曾在 capture 与 restore 阶段接受范围不一致。

## 保留的失败记录

1. `infrastructure-001`：受限宿主文件访问阻止 Inspect 创建标准应用数据目录；未执行模型，记录仍保留。
2. `infrastructure-002`：容器内工具执行成功，但 Inspect 的 Docker copy 路径未读到 tmpfs 中的成品。适配层改为固定、限长、严格校验的字节传输，并显式禁用工具超时自动重试。
3. `infrastructure-003`：修正后在真实 Docker 中贯通，评分与清理成功。

失败记录未被覆盖；最终成功不表示此前失败从未发生。

## 尚未验收

- 真实 OfficeVal 任务、官方 verifier 与新终点的独立接入。
- 任意 Agent 工具调用的权限隔离及完整对抗性沙盒审计。
- 运行中的进程/服务状态、环境变量、精确模型上下文与 KV cache 的恢复。
- 文件快照模块与真实模型分支续跑的完整连接；当前文件恢复是在合同测试中验证。
- 对象语义对齐、探针泄漏审查、主动实验策略、封存确认集访问控制及跨任务预测。

因此，此处的成功只支持“选定开源底座可以在本机按此配置运行合成工程流程”。
