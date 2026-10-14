# C3-E2: Outcome-Contrastive Sandbox

本地串行的 C3 实验结构。采用 **Inspect AI + Docker Compose** 复用模型/评测/容器基础设施，C3 负责研究对象、证据、预测冻结和干预协议。

当前工程状态为 **PARTIAL_VERIFIED**。R1.2 的单独资格检查以 **BLOCKED_SYSTEM** 结束，但新的集成交付协议把 Agent 任务错误与后端开发分开：协议恢复、D6 对象证据、D7 封存预测和 D8 缺口报告已经实现。固定八任务的目标模型矩阵在三次实现修订后以 0 次模型请求结束；已有开发失败证据被用于离线分析，目标自然轨迹和成功侧仍缺失。详见 [集成交付报告](docs/INTEGRATED_DELIVERY_REPORT.md) 与 [R1.2 验收报告](runs/e2-r1-2/R1_2_ACCEPTANCE.md)。

此前 R1.1 的 7 个冻结任务已各执行一次；012 在真实恢复后产生并执行了 2 次新工具调用，两次均报缩进错误。R1.2 的文件写入发生在独立资格任务中，没有补齐 OfficeVal 恢复后成功修改交付物的覆盖。原有 [R1.1 验收报告](docs/E2_R1_1_ACCEPTANCE.md) 与 [复现说明](docs/E2_R1_1_REPRODUCE.md) 保留。

此前 OfficeVal 002 的失败侧四个探索分支及四个封存确认分支保留于 [R1 报告](docs/E2_R1_ACCEPTANCE.md)。本轮没有将历史记录替换成新结果，也没有将工程检查解释为方法收益。

本轮接口验收与复现入口见 [D1–D5 验收说明](docs/E2_D1_D5_ACCEPTANCE.md)。旧 Docker 贯通记录仍保留在 [初版验证记录](docs/VALIDATION.md)。

## 结构

```text
C3_Experimental_Sandbox/
├── configs/local_serial.json       # 本机 ARM64、并发 1、研究就绪条件
├── docker/                        # 任务与评分的独立镜像、Compose
├── src/c3_sandbox/
│   ├── contracts.py               # 状态转换、探针及扩展接口合同
│   ├── config.py                  # 配置校验
│   ├── planning.py                # 计划与尚未满足的条件
│   ├── cli.py                     # 本地入口及进程级串行锁
│   ├── trajectory.py              # D1 统一轨迹与无损历史导入
│   ├── manifests.py               # D2 来源划分与完整配置身份
│   ├── replay.py                  # D3 文件/消息/配置恢复资格
│   ├── probe_registry.py          # D4 固定探针、匹配关系及逐任务审核
│   ├── experiment_scheduler.py    # D5 封存、缓存、权限、两本账
│   ├── experiment_controller.py   # 恢复/审核/调度的可信连接
│   ├── acceptance.py              # 六项确定性集成验收
│   ├── foundation_gate.py         # 绑定源码及测试证据的工程门禁
│   ├── evidence/                  # 文件快照、恢复核验、预测记录
│   └── adapters/inspect_smoke.py   # Inspect + Docker 工程贯通检查
├── schemas/trajectory_schema.json # 独立 JSON Schema
├── tests/                         # 合同、完整性和拒绝错误输入的测试
├── docs/ARCHITECTURE.md            # 完整目标结构、开源取舍、复用边界
├── docs/VALIDATION.md              # 本轮实际验证及局限
└── runs/                          # 每次工程运行独立留档，不作为研究结果
```

## 当前边界

| 功能 | 当前状态 |
|---|---|
| 配置检查、串行计划 | 实现；不完整的研究配置会列出阻塞项 |
| 普通文件/目录快照与恢复 | 实现；含隐藏文件、二进制、空目录和权限；拒绝软链接/特殊文件 |
| 预测概率与候选全集校验、记录完整性 | 实现；文件摘要可检测修改，不提供第三方时间戳证明 |
| Inspect 与两个 Docker 环境的工程检查 | 实现；实际执行结果见验证记录 |
| D1 统一轨迹与 D2 Manifest | 已实现验证；12 条历史原始记录无损导入，未知结论保持未知 |
| D3 文件＋精确消息前缀＋配置绑定恢复 | 012 新检查点通过真实恢复，产生 2 次新工具执行及关联 observation；恢复后成功完成工具操作、成功侧仍未覆盖 |
| D4 非泄露探针注册与匹配对照 | E2-F 公开结构/反馈观测已真实呈现；目标/匹配对照完整输入均为 2,071 Token；E2-S 尚无运行覆盖 |
| D5 预测封存、预算、缓存与确认门禁 | 已实现并经集成测试；私有控制器与窄客户端 API，不声称 Python 对象能隔离恶意宿主代码 |
| OfficeVal 原生运行与官方评分器 | R1.1 冻结 7 个新候选并全部执行；5 个有效失败、1 个协议无效、1 个评分无效；官方完整成功为 0 |
| D6 对象对齐、D7 自动假设/选择、D8 统计报告 | 本轮未开发 |

## 本地使用

需要 Python 3.12+ 和可用的 Docker/Compose。项目依赖放在本目录的 `.venv`，不改全局 Python 或 Docker Desktop 配置。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[inspect,test]'
.venv/bin/c3 plan --config configs/local_serial.json
.venv/bin/python -m pytest -q
```

D1–D5 接口验收（串行、无模型调用）：

```bash
.venv/bin/python -m pytest -q --junitxml=runs/e2-tests.xml
.venv/bin/c3 e2-acceptance --output runs/e2-acceptance-new
.venv/bin/c3 e2-gate --acceptance runs/e2-acceptance-new/acceptance.json --junit runs/e2-tests.xml
```

`e2-gate` 要求六项验收全部通过、D1–D5 与集成测试均出现且没有失败/跳过，并核对当前源码、Schema、测试和协议摘要。它只开放后续模块开发资格；`formal_experiments_admitted` 仍为 false。测试产物生产者和宿主操作者属于可信边界，不提供第三方不可伪造认证。

`configuration_valid=true` 只表示格式与部署约束合格。`ready_for_experiments=false` 表示尚缺模型、案例或研究验收，不应解释成工程命令失败，更不能解释成研究已通过。

初版 Docker 工程检查见 [docker/README.md](docker/README.md)，它不调用模型。R1 真实模型入口另见 [复现步骤](docs/E2_R1_REPRODUCE.md)，使用本地开源模型，无付费 API；不属于正式研究评测。

文件快照操作：

```bash
.venv/bin/c3 snapshot capture --store runs/snapshots --workspace /absolute/path/to/workspace
.venv/bin/c3 snapshot restore SNAPSHOT_ID --store runs/snapshots --workspace /absolute/path/to/empty-restore-dir
.venv/bin/c3 snapshot verify SNAPSHOT_ID --store runs/snapshots --workspace /absolute/path/to/empty-restore-dir
```

恢复只接受新建或空目录。调用方应在完成工具执行且没有后台写入时冻结状态；正在变化的工作区不满足 P0 合同。恢复核验失败必须在将来的真实分支控制器中阻止模型续跑。

## 与旧项目及研究协议的关系

[最小验证协议](../Agent_Latent_Capability_Audit_Kit/decisions/C3_MINIMAL_VALIDATION_PROTOCOL_v0_1.md)定义科学问题。旧项目和原始产物保留，新项目不导入旧宿主执行器、脚本式分支修复或冒充 LLM 的规则基线。

[架构决议](docs/ARCHITECTURE.md)解释 Inspect、Harbor 和 OpenHands 的取舍。先复用执行基础，再分别实现 C3 的研究模块；未完成模块不产生占位的诊断分数。
