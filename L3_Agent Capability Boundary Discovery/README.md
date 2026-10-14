# L3 · Agent Capability Boundary Discovery（跨任务能力边界发现）

本目录为三层研究架构中的 **L3 层（Capability Boundary Discovery）**，负责在 **L1（Execution Sandbox）** 客观执行证据与 **L2（Behavioral Event Localization）** 单任务行为诊断包的基础上，跨任务、跨模型、跨基准提炼并验证 Agent 的条件性能力边界。

---

## 1. 核心定位与职责

| 维度 | 说明 |
| :--- | :--- |
| **核心问题** | Agent 在什么任务条件、环境约束与交互模式下容易成功或失败？ |
| **上游输入** | L2 产出的结构化 `Diagnosis Package`（含 `FactRecord`、`EventLocalization`、`Hypothesis`、`ExperimentReceipt`）与 L1 密封执行证据 |
| **核心机制** | 跨任务模式聚合（Cross-task Analysis） $\rightarrow$ 能力边界假说生成（Boundary Hypothesis） $\rightarrow$ 独立任务与受控分支验证（Counterfactual & Out-of-Task Validation） |
| **最终产物** | **Capability Boundary Report（能力边界与适用条件报告）** |

---

## 2. 核心设计铁律

1. **严禁由单次成败直接推断稳定能力缺陷**：单条轨迹中的异常仅构成 L2 局部事实或候选解释；唯有跨任务、跨受控条件稳定复现的规律，方可进入 L3 能力边界结论。
2. **观察事实、候选假说、验证结论三者解耦**：
   - **Cross-task Pattern**：跨任务聚类的客观失败/成功模式与环境条件关联；
   - **Boundary Hypothesis**：关于触发条件（Task Complexity / Tool Interface / State Horizon / Feedback Sparsity）的可证伪假说；
   - **Validation Receipt**：通过独立新任务或 L1 检查点受控分支实验获得的检验回执。

---

## 3. 模块结构规划

- `schemas/`：跨任务模式聚类、能力边界假说与验证报告的数据合同定义
- `src/`：L3 跨任务聚合器、条件归因分析器与验证实验编排器
- `configs/`：跨基准（B1 OfficeVal / B2 LHTB / B3 SWE-bench Pro V2）聚合分析配置
- `tests/`：L3 合同与聚合逻辑单元测试
