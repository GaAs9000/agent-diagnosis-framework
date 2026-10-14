# L2 与 L1 接口对接与配置指南

本文档定义 L2 (`l2-evidence-analyst`) 与 L1 (`c3-experimental-sandbox`) 之间的接口契约、依赖绑定与协同调用规范。

---

## 1. 运行时与依赖环境对接 (Runtime Integration)

L2 采用轻量非侵入式设计（仅依赖 `jsonschema` 与 `lxml`），但在执行真实分析与受控实验时，复用 L1 的底层基础设施。

### 推荐运行环境
直接在 L1 的主虚拟环境中以可编辑模式（`-e`）安装 L2：
```bash
# 激活 L1 虚拟环境
source /Users/jiashen/Desktop/问题诊断/L1_sandbox/C3_Experimental_Sandbox/.venv/bin/activate

# 安装 L2
cd "/Users/jiashen/Desktop/问题诊断/L2_Behavioral Event Localization/L2_Sandbox"
pip install -e '.[test]'
```

---

## 2. 核心对接接口 (Core Interfaces)

### 接口 A：证据包读取与校验接口 (`L1Reader` & `verify_package`)
- **功能**：L2 解析 L1 产出的证据包（`manifest.json`, `trajectory.jsonl`, `states/`, `checkpoints/`）。
- **契约**：
  - 自动调用 L1 官方封签校验器 `c3_sandbox.l1.package.verify_package`；
  - 校验全量文件哈希、轨迹哈希链、快照内容与评分投影；
  - 合成测试夹具（`test_fixture`）执行内建语义一致性校验，隔绝外部依赖。
- **CLI 检验**：
  ```bash
  c3-l2 inspect <path_to_l1_episode> --require-l1-verifier
  ```

### 接口 B：Analyst 模型后端适配 (`backend.py`)
- **功能**：L2 Analyst Agent 调用 LLM 生成假说与预测时，直接复用 L1 原生网络传输与认证模块（`c3_sandbox.l1.models.native`）。
- **配置对应表**：
  - Gemini API：[`configs/analyst.gemini-35lite.json`](./analyst.gemini-35lite.json) / [`configs/analyst.gemini-38flash.json`](./analyst.gemini-38flash.json)
  - 本地 / AML 服务器：[`configs/analyst.qwen32b.json`](./analyst.qwen32b.json)
- **健康检查**：
  ```bash
  c3-l2 doctor --config configs/analyst.gemini-35lite.json
  ```

### 接口 C：检查点资格与受限实验计划 (`plans.py`)
- **功能**：针对选定的 L1 Checkpoint（如 `cp000001`）生成不可变冻结实验计划（`frozen_plan.json`）。
- **契约**：
  - 严格校验父检查点 SHA256 与 `config_sha256`；
  - 严格继承父任务剩余预算（不重置预算）；
  - 首选受控模板：`resume_unchanged`（对照续跑，测量不干预基线分布）。
- **CLI 执行**：
  ```bash
  c3-l2 plan --analysis <analysis_dir> --source <l1_source_dir> --checkpoint <cp_id> --output <plan_dir>
  ```

### 接口 D：受限分支执行与回执接口 (`controller.py`)
- **功能**：通过 L1 编排器（`c3_sandbox.l1.core.orchestrator.run`）从指定检查点拉起真实的冷恢复分支。
- **契约**：
  - 必须显式携带 `--allow-execution` 授权，避免隐式消耗真实预算；
  - 实际执行完全由 L1 容器承担，L2 仅记录意图流并导入客观结果回执（`ExperimentReceipt`）；
  - 严禁 Analyst 自行篡改或捏造执行结果。
- **CLI 执行**：
  ```bash
  c3-l2 execute --plan <plan_dir> --source <l1_source_dir> --output <experiment_output_dir> --allow-execution
  ```
