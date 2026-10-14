# Gemini 旧字段 → L1 字段映射

所有旧 result.json 原件均保存在 raw/result.json；未列字段仍在原件中，不会因为规范化器暂不理解就删除。

| 旧来源 | L1 位置 | 处理与限制 |
|---|---|---|
| task_id；bundle/manifest.json | manifest.task；benchmark/ | 任务、输入摘要、评分材料分开保留 |
| 首次 request_raw 的 contents／systemInstruction／tools | task.presented；raw/model/ | 使用实际下发内容，不从当前代码常量重建 prompt |
| runtime；provider；model | agent；configuration | 声明信息，不能冒充已验证执行指纹 |
| raw_gemini_records[*].request_raw | model.request；独立原生字节文件 | UTF-8 源字符串重新编码后与原摘要核对 |
| raw_gemini_records[*].response_raw | model.response 或 model.error | 空响应不伪装成模型回复 |
| response 原生 functionCall | tool.call | 保留 provider ID、part 来源、名称和参数 |
| tool_dispatch[*] | tool.call／tool.result | 使用串行顺序和精确名称参数关联；旧 ID 保留 |
| tool_dispatch[*].raw_result | tool.result.raw_observation | 原始输出，不用截断的模型可见版本覆盖 |
| 下一 request_raw 中 functionResponse.response | observation.delivered | 必须真实出现在留存请求；不使用包装器事后生成 |
| usage；usageMetadata | model.response.usage；原生响应 | 保留 provider 字段，不反推费用 |
| fs_delta | state_changes：recorded_step_delta | 旧报告值，不是快照重算结果 |
| public-input／final-workspace | artifacts；recomputed_endpoint_delta | 重新哈希实际保留字节，只说明端点 |
| before_path／after_path | raw/source_path_audit.json | 映射归档内相对路径并记录是否存在，不跟随任意旧宿主路径 |
| initial_grade／final_grade；评分目录 | official_grade；raw/*-grade | 保存旧官方原文、适配层判定与进程文件 |
| agent_result.stopped_reason | status.termination | max_turns → budget_exhausted；不因此强制任务失败 |
| execution_valid；outcome | raw/result.json；L1 分维状态 | 保留原值，但不直接继承旧“执行有效”结论 |
| checkpoints | checkpoints/index.json | 只列源记录，没有 L1 资格即 unqualified |
| config.lock | raw/config.lock；configuration | 摘要不是完整 resolved config；本批不能据此恢复配置 |

当前无法无损补出的字段：模型请求历史时间、模型生成延迟、历史实际费用、不可变权重版本、完整已应用配置、Agent 侧环境身份证据、缺失的逐步快照，以及对象级状态。

当前 7 个预算终止 episode 的最后一次工具结果没有下一次模型请求作为送达凭证。这 7 个结果仍完整保存，不能“补发”到历史轨迹。
