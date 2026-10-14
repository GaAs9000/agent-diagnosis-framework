# L1 · Agent Execution & Evidence Sandbox

> **L1 = Benchmark-driven Agent Execution & Evidence Sandbox**
>
> 复用现有 Benchmark，让不同 Agent 在可控环境中完成真实任务，并自动产出完整、可信、可比较、可重放的执行证据。

当前交付：**L1 P1 证据合同与 Gemini 历史导入剖面，版本 0.1.0**。
这不是完整 L1 执行系统的最终验收。新的 API／本地运行器、对象状态采集和冷恢复尚未接入本轮入口。

## 本轮实际可用

`c3_sandbox.l1` 是独立入口，不导入旧 integrated、gateway、分析模块、Inspect 或 MLX。
它能够把现有 Gemini 结果导出为统一证据包，保留原始字节，验证事件链、关联、文件摘要、评分原文及标准化字段。
执行失败、预算耗尽、评分无效、缺失证据相互独立；工具输出不是模型回复。

8 条归档 Gemini 轨迹已导出并校验；**154 个请求、154 个响应、153 次工具执行、146 个可在后续请求中核实的 Observation**。
这些是历史记录，不是本轮新发起的模型调用。8 份包均标记为 `incomplete`，没有冷恢复资格。

## 安装与测试

需要 Python 3.12 或更新版本。本轮离线功能不需要 Docker、模型权重、API Key 或 Metal。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[l1,test]'
python -m pytest -q tests/l1
```

当前新增 L1 测试：70 项通过。旧项目全量测试在本次环境下未通过，不能引用原归档的“640 项全过”作为当前验收。
环境限制和原始日志见 `docs/l1/validation/` 与 `docs/l1/AUDIT.md`。

## 从现有归档导出

输出目录必须是新目录；不会覆盖历史证据。

```bash
c3-l1 import-gemini-run \
  --run runs/gemini-real-001 \
  --bundle runs/integrated-delivery-001/bundle \
  --output runs/l1-gemini-import-001

c3-l1 verify runs/l1-gemini-import-001/officeval_003
c3-l1 inspect runs/l1-gemini-import-001/officeval_003
c3-l1 events runs/l1-gemini-import-001/officeval_003 --kind tool.result --limit 1
```

也可用 `PYTHONPATH=src python -m c3_sandbox.l1` 代替 `c3-l1`。
`verify --expected-index-sha256 <独立保存的摘要>` 才会核对外部给定的索引摘要；无外部摘要时明确报告信任边界。
`inspect` 和 `events` 只读取并核实证据，不调用模型或重跑工具。

本轮**没有** `c3-l1 run` 或 `c3-l1 restore`。不要把历史导出器当成新运行器或恢复器。

## 合同与交付边界

- `src/c3_sandbox/l1/evidence.schema.json`：实际运行使用的 JSON Schema。
- `docs/l1/CONTRACT.md`：字段、证据等级、状态与版本规则。
- `docs/l1/FIELD_MAP.md`：旧记录到新字段的来源映射。
- `docs/l1/MIGRATION.md`：不推倒旧代码的接线顺序及后续验收门槛。
- `docs/l1/AUDIT.md`：本次逐项实测与未完成项。

旧代码、配置、工具和评分器均保留；旧 README 原文归档到 `docs/l1/LEGACY_README.md`，其中历史状态不作为当前 L1 状态。
