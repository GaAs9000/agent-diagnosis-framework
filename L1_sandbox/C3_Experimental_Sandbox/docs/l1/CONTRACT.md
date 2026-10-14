# L1 Evidence Package 合同 0.1.0

## 1. 当前剖面与非目标

本合同以现有 Gemini 归档的可验证导入为第一验收剖面。规范包含将来的 `live` 记录模式和 `partial` 任务状态，但**当前导出器和语义校验器只接受历史导入／显式合成测试剖面**，不声明实时后端或恢复资格。
所有“完整”“可信”结论都必须与证据范围绑定。归档自称 `is_synthetic=false` 不等于第三方执行证明。

L1 不产出错误原因、能力判断、预测或修复建议。文件哈希变化不等于文档语义变化。

## 2. 包结构

```text
episode/
  manifest.json
  trajectory.jsonl
  state_changes.jsonl
  official_grade.json
  evidence_index.json
  checkpoints/index.json
  artifacts/inventory.json
  artifacts/initial/...
  artifacts/final/...
  benchmark/manifest.json
  benchmark/task_metadata.json
  benchmark/task.json
  benchmark/rubric.json
  benchmark/verifier.py
  benchmark/source_metadata/...
  raw/result.json
  raw/config.lock
  raw/model/000000.request.json
  raw/model/000000.response.json
  raw/initial-grade/...
  raw/final-grade/...
  raw/source_path_audit.json
```

可选源文件不存在时不伪造占位内容；相应引用为 null，并在 coverage 中说明。没有合格检查点时保留空 index 和原因，而不是制造可恢复目录。导出中途报错会留下已写入证据和 `export_error.json`，不成为合格完整包。

## 3. Manifest：六个彼此独立的维度

| 字段 | 含义 | 本次 Gemini 导入 |
|---|---|---|
| `status.termination` | 为什么结束请求循环 | 预算用尽／助手结束 |
| `status.protocol_validity` | 留存原生请求、工具调用和响应协议是否一致 | valid |
| `status.execution_validity` | 是否具有完整、合格的执行边界证据 | unknown |
| `status.evaluation_validity` | 留存评分结果与过程、成品是否一致 | valid，范围限归档核对 |
| `status.task_outcome` | 在明确评分政策下的任务结果 | 保留旧 completion 布尔值，全部 failure |
| `status.evidence_completeness` | 应采字段是否齐备 | incomplete |

显式协议错误令 execution_validity 为 invalid；环境证据缺失不能把已知错误“洗成未知”。反过来，环境证据不足也不能被 task success 或 package verified 覆盖。

官方原始输出保存在 `official_grade.*.official_raw`。`c3_operational_completed` 明确标注为**旧适配层规则**，不是重新定义官方成功。没有新建“命中一条规则就 partial”的政策。规则覆盖不足与评分进程失效也不是同一概念。

## 4. 事件与真实 Observation

每行包含 schema_version、episode_id、event_id、seq、kind、observed_at、assurance、source_refs、links、payload、前一事件摘要及本事件摘要。

event_id 由 episode_id 与单调序号组成；model_call_id 和 tool_call_id 分别标识一次模型请求和一次工具调用。
历史导入保持逻辑串行顺序，不伪造执行时钟。`observed_at=null` 不使用导入时间替代；manifest.imported_at 仅表示迁移时间。

`model.request/response` 指向原生内容字节文件；请求失败且没有响应字节时输出 `model.error`，不增加 model_responses。
`tool.call` 保留原生 ID、旧调度器 ID、名称和参数，另建 L1 关联 ID。旧两种 ID 不同，只能声明“顺序＋名称参数精确匹配”，不能谎称原来已有共享 ID。
`tool.result` 保留执行器记录的原始 stdout、stderr、exit_code、elapsed 等；`observation.delivered` 则仅记录**下一次实际请求中出现的**工具返回内容。两者不能互相覆盖。

预算最后一步返回结果、但没有下一次请求时：保存 tool.result，不生成 observation.delivered。即使知道旧包装函数如何截断输出，也不事后伪造“模型已收到”的记录。

## 5. 状态与对象身份

当前只处理文件字节，不解析 Office 对象。
object_id 以 episode_id 和逻辑路径确定，保证同一 episode 内初态／终态对应；它不是跨任务的语义对象 ID。文件重命名按路径删除和新增处理，不推断对象移动。
version_id 是文件内容 sha256。相同内容可共享版本摘要；每个版本是否有实际字节，由 blob 引用确定。

`recorded_step_delta`：导入旧 fs_delta；缺少前后快照时，verification 为 `reported_without_retained_snapshots`。
`recomputed_endpoint_delta`：用包内初态和终态字节重新计算；只能证明两个端点的差异，不证明中间某一步。

两种记录不得相加当作操作次数。本批 6 条旧逐步文件差异和 6 条端点差异构成 12 条状态证据记录，不是 12 次独立文件修改。
目录权限、空目录、硬链接、进程内存、后台进程、外部服务状态均不在当前文件导入的可恢复保证中。原始解压后的权限也不当作历史运行权限。

## 6. 证据完整性与信任

规范化引用统一为 `{path, json_pointer}`，path 相对包根目录。raw 原件中的旧绝对路径照原文保留，但绝不自动跟随去读取宿主路径。
所有导出写入排他创建；拒绝逃逸路径、符号链接及非独立普通文件。
事件摘要使用 UTF-8 JSON、键排序、紧凑分隔符、不允许 NaN，排除 event_sha256 字段后计算。索引覆盖所有文件，唯独排除索引自身；索引摘要另存到批次报告或外部收据。

校验器检查 Schema、文件清单、摘要、事件链、引用、模型请求覆盖、原生调用与执行绑定、Observation 送达引用、逐步与端点差异覆盖、成品版本、评分映射与状态政策。

**哈希证明相对某个已信任摘要的内容一致性，不证明过去真的调用了服务，也不能阻止可信宿主重写整份证据和全部摘要。**
生产边界需要可信控制器；未来运行时应把 Recorder 放在 Agent 不可写位置。评分器私有材料不应进入 Agent 可见任务目录。

文件权限默认私有，raw 中仍可能含任务内容、提示词、工具输出及历史路径。本批不是脱敏后的公开数据集；未随包附带运行所需 API Key。

## 7. Coverage 与“未知”

coverage 每项包含 field、status、count、reason、source_refs。status 为 available、missing、unknown、partial 或 not_applicable。
已知不存在的证据与未核验的声明分开记录；成本不填 0，模型别名不当权重指纹，声明 seed 不当实际下发参数。

raw 不可获得的字段不能通过模型、模板或估算补齐。用户后来补充证据时应生成新的包版本和摘要，不原地改写已有 sealed package。

## 8. 重放能力与后续兼容

`offline_evidence_read` 仅指验证并按事件读取历史。
`tool_reexecution_qualified` 需要环境和输入等资格；它不是读取 stdout。
`cold_resume_qualified` 需要环境状态、精确原生对话前缀、配置、模型／工具身份及可恢复范围；它不是有个文件夹就成立。
`deterministic_model_replay` 不因设置 seed 自动成立。

本剖面后三项恒为 false；没有恢复 API。实时接线、对象级状态和通过资格的检查点需要后续实现与版本化验收。
