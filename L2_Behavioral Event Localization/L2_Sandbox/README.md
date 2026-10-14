# L2 · 关键行为事件定位与干预预测

**L2 v0.1.0｜2026-10-13｜独立可安装工程，不覆盖 L1。**

输入 L1 Evidence Package，输出有原始引用、对象版本、事件位置、竞争解释和未验证预测的 Diagnosis Package。默认只读、无模型调用；明确授权后可用现有 L1 原生模型后端运行受限 Analyst。实验执行另行授权，不能把模型解释或历史重试冒充反事实真值。

> 本交付实做：代码、离线测试、九份真实 L1 记录上的规则/对象分析、可独立校验的示例、API/本地 Analyst 接入代码、冻结计划、受限续跑与回执接口。
>
> **未在本环境调用真实 Analyst 模型或执行 Docker 实验**。模型循环以明确的测试替身验证；信息注入模板保持 `blocked`，不得称为已经实现的局部因果验证。研究准确率与 M4 未开展。最新实测数量见 `docs/RELEASE_REPORT.md`。

## 1. 安装：两种方式

### 仅离线分析与读取

Python 3.11+（与 L1 共用时遵循 L1 的 Python 3.12+要求）：

```bash
cd L2_Sandbox
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python scripts/verify_examples.py examples
```

依赖仅 `jsonschema`、`lxml`；不需要 Docker、API Key、Metal、模型权重或 Inspect。

### 运行真实 Analyst / 连接 L1 续跑

**推荐在你已有 L1 的虚拟环境中安装本包**。假设两者同级解压：

```bash
source ../C3_Experimental_Sandbox/.venv/bin/activate
python -m pip install -e '.[test]'
```

本包安装名为 `l2-evidence-analyst`，代码位于独立 `l2_sandbox` 命名空间，不写入 `c3_sandbox/l1`。实时模型层通过薄适配器导入你现有的 `c3_sandbox.l1.models.native`，不是重新维护 Gemini 或 Chat Completions 协议实现。已有 L1 后端缺失/接口不兼容时，`doctor` 报错，**不会自动降级成规则或模拟模型并声称成功**。

## 2. 先看随包真实证据分析结果

`examples/officeval003/` 包含九份独立、**未调用 LLM 的规则基线** Diagnosis Package：

```bash
c3-l2 verify examples/officeval003/analysis_004
cat examples/officeval003/analysis_004/report.md
```

`analysis_004` 对应原 Pro 恢复分支中的协议异常。它可以引用原始响应的异常终止信号，不把它解释成“模型缺少文档理解能力”。输入映射在 `examples/officeval003/batch_receipt.json` 与 `examples/README.md`。

九份记录属于 **同一个 OfficeVal003 任务家族**，不是九个独立任务或有因果标签的数据集。随包保存的是被引用的源字节子集及原索引，不重复分发完整 L1 仓库或完整源 Episode；可以离线校验诊断引用，但不能拿诊断包执行 L1 恢复。

## 3. 分析你本机的一条 L1 记录

```bash
# SOURCE 应是含 manifest.json / evidence_index.json 的一条 Episode 目录。
c3-l2 inspect /path/to/deliverables_by_model/01_gemini-3.5-flash-lite/episode

c3-l2 analyze \
  --source /path/to/deliverables_by_model/01_gemini-3.5-flash-lite/episode \
  --config configs/rules.json \
  --output runs/l2-rules-001

c3-l2 verify runs/l2-rules-001
```

默认严格校验文件清单、原始字节、事件链、工具调用投影、Observation 送达、状态与成品引用。有当前 L1 校验器时另行调用；没有时使用本包明确列出的内置 0.2 检查，不声称调用过上游校验器。设置 `require_l1_verifier=true` 可以强制双重检查。

旧 L1 0.1 兼容需要已安装的 L1 校验器；只映射现有历史记录，缺失快照、对象与恢复资格不补造。不完整源包可以 `inspect` 查看错误，但本版不开放不完整输入的正常 `analyze`。

## 4. 运行真正的只读 Analyst

先修改 `configs/analyst.gemini.json` 中的模型名，使其与你账号实际可用的模型一致。示例故意不假定某个模型当前可用。

```bash
export GEMINI_API_KEY='在本机设置，不要写入 JSON 或仓库'

c3-l2 doctor --config configs/analyst.gemini.json

c3-l2 analyze \
  --source /path/to/original/episode \
  --config configs/analyst.gemini.json \
  --output runs/l2-analyst-001 \
  --allow-model

c3-l2 verify runs/l2-analyst-001
```

`--allow-model` 授权 Analyst 的推理费用，**不授权启动被测 Agent 的实验分支**。

本地推理服务使用 `configs/analyst.local.json`；修改 localhost 地址和实际 served model ID。它需要已启动且支持原生工具调用的服务，不自动下载、加载或启动本地权重。

当前 Analyst 只看文本和有限对象事实，不直接渲染/理解图片；原被测 Agent 的图片读取与送达事实仍保留。不能据此声称分析了图像参照的全部视觉要求。

### Analyst 能调用的工具

`get_episode_summary`、`list_events`、`get_requirement_evidence`、`read_evidence`、`get_state_diff`、`inspect_object_version`、`list_qualified_checkpoints`，以及最终 `submit_diagnosis`。

原始模型请求/响应可以从事件的注册引用读取；大型原文按需物化，实际检索与截断有记录。没有任意路径读取、shell、网页抓取、读取兄弟分支结果或私有 Verifier 源码的工具。

最终输出必须满足 Schema 和引用检查。没有读过的事实 ID、虚构事件/版本、`verdict=supported` 等越权输出会被拒绝；允许有限次结构修复，所有尝试原样留存。**结构与引用正确不保证自然语言结论必然被证据蕴含**，这一点单列为未认证。

## 5. 查看单条事实和它的原始依据

从 `facts.jsonl` / `report.md` 中复制一个 fact ID：

```bash
c3-l2 show-fact runs/l2-analyst-001 --fact fact_填写实际ID
```

输出包括源索引摘要、原文件 SHA-256、JSON Pointer / JSONL 行 / 字节范围和选中值。对象探针输出的是后处理的直接 XML 属性，不冒充原 Agent 当时看到的 Observation；样式继承、渲染、跨版本语义身份和“没读先写”保持明确未知。

## 6. 冻结计划：和执行分开

```bash
c3-l2 plan \
  --analysis runs/l2-analyst-001 \
  --source /path/to/original/episode \
  --checkpoint cp000030 \
  --template resume_unchanged \
  --replicates 3 \
  --endpoint submission_present \
  --output runs/l2-plan-001
```

`resume_unchanged` 可以调用现有 L1，建立新基线续跑。原配置与累计预算不重置，终态评分错误不直接否定合格检查点。**单组基线没有干预对照，不能推断某假说已支持。**

`provide_public_state_summary` 和 `generic_reminder` 可以编译为待实施计划，但在当前 L1 仅支持原样恢复的接口下固定为 `blocked`。本包不篡改检查点、不伪造历史 Observation，也不开放任意修复代码。它们需要独立、有资格证据的 L1 信息注入接口后再实现。

```bash
# 额外且明确的授权；会调用被测模型并可能产生多次 API 费用。
c3-l2 execute \
  --plan runs/l2-plan-001 \
  --source /path/to/original/episode \
  --output runs/l2-experiment-001 \
  --allow-execution

# 只导入本控制器产生的可核验回执；不信任任意手写“成功”表格。
c3-l2 validate \
  --analysis runs/l2-analyst-001 \
  --receipts runs/l2-experiment-001 \
  --output runs/l2-validation-001
```

所有尝试和不可定义终点分开计数，不按是否有好分数删除失败。当前结果只给描述性统计，`hypothesis_verdict=not_identifiable` 或测试中的 `untested`，不产出伪造因果真值。

## 7. 本机 M1/M2 验收

```bash
# 不带 --allow-model 时，真实 Analyst 配置只做预检，不发生模型推理。
python scripts/accept_local.py \
  --source /path/to/original/episode \
  --config configs/analyst.gemini.json \
  --output runs/l2-accept-001 \
  --allow-model
```

回执分开报告 `M1_passed`、`M2_live_analyst_passed`，不把无模型规则基线当作真正 Analyst 实测，也不把 M2 当作因果验证或研究有效性证明。

## 8. 可移植打包与批量处理

```bash
c3-l2 pack runs/l2-analyst-001 --output runs/l2-analyst-001.zip

# 完整 L1 ZIP 解压到新目录；只有已知缺少 UTF-8 标志时才显式开启修复。
c3-l2 unpack /path/to/deliverables_by_model.zip --output runs/l1-input
# c3-l2 unpack ... --repair-utf8-names

c3-l2 batch \
  --source /path/to/deliverables_by_model \
  --config configs/rules.json \
  --output runs/l2-batch-001
```

`pack` 会默认解压到临时新目录，再做校验。输出目录必须新建且在源树之外，不覆盖原 L1、既有分析或预测。已有包篡改、丢文件、错误引用不会被静默修复。

## 9. 文档入口

- `docs/RELEASE_REPORT.md`：实际验收和未执行项目。
- `docs/IMPLEMENTATION.md`：模块、合同、设计差异与阶段范围。
- `docs/SECURITY_AND_EVIDENCE.md`：权限、失败、来源与保证边界。
- `docs/RESEARCH_PROTOCOL.md`：基线比较、任务家族隔离和指标。
- `docs/DESIGN_INPUT_v0.1.md`：原始设计稿，原样保留；其中待实现项不自动变成已完成声明。
- `validation/`：实际测试、九份源记录回归、P1 兼容与打包记录。
