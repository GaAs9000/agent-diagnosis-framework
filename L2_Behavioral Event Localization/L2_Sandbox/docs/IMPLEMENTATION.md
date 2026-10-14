# L2 0.1 实现与设计合同

## 模块

```text
src/l2_sandbox/
  evidence/l1.py          L1 0.2 内置校验、可选现有 L1 校验、0.1 有限兼容
  evidence/references.py  原索引绑定的 JSON/JSONL/字节引用、按需复制原始请求响应
  evidence/view.py        单 Episode 白名单视图和分页/字节预算
  facts/build.py         确定性事实、事件、版本与覆盖
  facts/docx_probe.py    安全的直接 OOXML 探针
  analyst/backend.py    复用已有 L1 原生 ModelBackend 的薄适配层
  analyst/run.py        单 Analyst 请求、工具检索、有限输出修复
  analyst/rules.py      明确标注的非 LLM 规则基线
  validation.py         Schema、引用、事件/版本定位和 untested 边界
  package.py            诊断封口、事实投影/对象重解析/原始模型记录核验
  experiments/plans.py  冻结计划与能力阻塞
  experiments/controller.py  显式授权的原样续跑、回执及描述性导入
  evaluation/metrics.py 外部真值驱动的基础指标与任务家族划分
  portable.py           UTF-8 打包/解包、默认解压回验
  cli.py                单一命令入口
```

这是独立安装包，没有按设计建议写入 `src/c3_sandbox/l2`，避免覆盖现有 L1 命名空间。模型协议仍复用 `c3_sandbox.l1.models.native`；没有悄悄实现第二套 API 协议。产出包的实例可以脱离 L1 代码检查引用；真实模型与恢复入口需要已有 L1。

## 输入与事实

仅发布 `retrospective`、`strict` profile。原 Episode 评分可见，但不允许访问兄弟分支、私有 Verifier 源码或 Rubric 源文件。`prefix` 和 `privileged` 明确拒绝，不提供未通过防泄漏验收的伪前缀分析。

L1 0.2 输入内置检查：文件清单/摘要、事件链/顺序/链接、规范化工具调用与原生响应一致、stdout/stderr 与 Observation 一致、送达内容出现在实际后续请求、快照文件/目录/内容摘要、重算差异、成品映射、配置与终态、检查点映像绑定。它不是官方评分语义的重新实现，也不执行源包的 Python Verifier。已安装 L1 校验器时另外调用；没有时报告确切检查集合。0.1 必须经过已安装 L1 校验器，只有限映射历史记录。

FactRecord 保存 `origin=recorded/derived`、原始引用、事件 ID、可用性、派生规则和覆盖范围。派生的对象状态包括解析器标识、配置摘要、输出摘要、源文件字节摘要以及所有对应快照中的文件路径。原始模型自述使用 `model_output_text`，不升级为真实工具动作。

对象探针检查整个 `word/document.xml` 中的表格/段落、递归行数量、没有文本且没有绘图元素的行、直接表格/单元格边框、段落直接间距等。只提供直接 XML 属性；空文本不等于视觉空行，缺少直接属性不等于继承样式没有属性。跨版本只比较文档级结构计数，不以表格编号伪造语义身份。DTD/entity、重复 ZIP 成员、加密、超限输入拒绝或标 unsupported。

## 诊断与模型运行

输出合同由 `schemas/contracts.json` 分发，包含 FactRecord、EventLocalization、Hypothesis、Diagnosis、Manifest 和 Index 定义。通过 `submit_diagnosis(diagnosis_json)` 提交，支持事件、事件集合、区间和终态粒度。引用不存在、事实未完整读过、事件无绑定支持、声明的文件版本不在支持事实中、无实验就输出 supported 等情况均拒绝。

模型记录包含请求前落盘、原生响应、元数据、全部工具请求/返回、失败尝试、结构修复、最终接受记录。Verifier 将接受的诊断反查至原始模型响应，不只检查输出 JSON 的形状。源索引固定，分析结束后核验源文件未改变。

自然语言的“证据是否蕴含结论”不能由 Schema 证明。程序显式记录 semantic_entailment 未认证；不把字面引用通过、第二模型同意或规则基线标签当作因果真值。

## 单次结构化基线

`structured-once` 仅提供初始结构化视图和按预算可容纳的完整事实记录，不开放检索工具；遗漏数量记录在案，因结构修复可能增加请求，因此名称指不再检索新证据，不保证只有一次 HTTP 请求。

当前提供 rules、structured-once、analyst 三种模式。**没有交付可直接用于论文的完整等信息 raw-log 基线与独立评测结果**；不同方法可读范围、实际字节和遗漏应在试验中审计，不因为它们从同一文件出发就自动声称等信息。

## 实验范围

`resume_unchanged` 桥接已安装 L1 的 `run(config, output, parent=..., checkpoint_id=...)`，不重置配置或预算。终态评分无效不阻止一个本来符合资格的检查点。授权范围固定分支数，全部失败保留。

信息注入、通用提醒的真实注入都需要 L1 审计接口；本版只生成 blocked 计划。没有猴子补丁、未记录的对话改写、任意 Agent 代码修复或假工具 Observation。受限基线续跑并不具备干预对比，验证导入明确输出 not_identifiable，而非 supported。

`execute` 在没有现有 L1 恢复实现、未授权或模板 blocked 时拒绝调用。跨机器恢复对 L1 镜像、状态和协议的要求由 L1 负责；本包不声称接管其执行隔离或恢复资格认证。

## 阶段解释

| 设计门槛 | 本版实现 | 本次验证范围 |
|---|---|---|
| M1 | 读取、事实、对象、覆盖、离线报告 | 九份实际源记录及单元/负向测试 |
| M2 | 原生后端桥接、只读循环、引用验证、失败留存 | 确定性模型端口测试；真实 Analyst 待本机凭证 |
| M3 | 冻结计划、原样续跑、结果导入 | 控制器测试替身；无真实对照干预，信息模板 blocked |
| M4 | 基础指标/任务家族工具和试验说明 | 无独立任务研究评测，无准确率/创新性主张 |

因此本版名称是 **只读定位与假说系统，附受限实验接口**，不是“已经通过因果验证的完整研究系统”。
