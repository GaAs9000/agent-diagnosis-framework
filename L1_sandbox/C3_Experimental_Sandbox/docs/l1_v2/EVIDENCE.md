# Evidence Package 0.2.0

## 目录与关联

```text
episode/
  manifest.json                  # 身份、task、配置摘要、六维状态、能力和缺失项
  evidence_index.json            # 全文件清单、摘要、事件链根
  trajectory.jsonl               # 顺序事件，hash/previous_hash/event_id/links
  state_changes.jsonl            # 关联前后快照与工具结果的差异
  raw/resolved_config.json       # 全部解析后的非密钥配置
  raw/model/rXXXXXX/              # request.json / response.bin / http.json
  raw/tools/.../                 # result.json / stdout.bin / stderr.bin / 可选 image.bin
  raw/final_conversation.json
  observations/                  # 真实送入后续请求的内容（可能截断）
  runtime/provenance.json         # 实际镜像 ID、实际镜像内包版本、资源限制
  benchmark/                     # 公有任务与私有原始评分源码/元数据
  states/blobs/<sha256>           # 原始文件内容，跨同包快照去重
  states/snapshots/               # 路径、模式、文件摘要、空目录、内容版本
  checkpoints/index.json
  checkpoints/cpXXXXXX.json      # 文件快照+原生对话+资格+累计预算
  artifacts/submission/          # 冻结的最终提交
  artifacts/inventory.json
  grading/                       # 实际评分 stdout/stderr/process/runtime/worker-output
  official_grade.json            # 原始评分及明确标注的投影
  lineage/                       # 分支时父检查点/快照和父摘要引用
```

某些目录在尚未达到相应阶段的失败运行中不存在；标准主清单仍能描述未执行、缺失与失败。不可采集的费用/服务端内部状态不能补造。证据包默认私有权限；原始 prompt、任务文档、模型输出可能含敏感信息，应审查后分享。

## 六维状态不合并

- `termination` 是为何停止，例如 `assistant_finished`、`request_budget`、`model_transport_error`、`protocol_error`、`unsupported_capability`。
- `protocol_validity` 是已记录协议是否可解释，独立于评分。
- `execution_validity` 是执行边界是否满足条件；fixture 固定为 `unknown`，不能升级为模型实测。
- `evaluation_validity` 是评分输入/过程/结果是否有效；评分正常结束不等于任务成功。
- `task_outcome` 按声明政策记录；官方未给统一成功规则时可以是 `unknown`，不能为了表格好看伪造标签。
- `evidence_completeness` 针对**声明的采集 profile**；文件 profile 完整不表示具备对象级状态、进程快照或服务端计费。达到捕获上限等额外丢失标 `incomplete`。

任务失败不是系统失败。`verify=true` 也不是隔离执行真实性的独立证明。保留的 P1 历史包仍为 0.1.0，其缺失状态不会因为用新校验器打开就变完整。

## 原始结果和送达 Observation

`tool.result` 记录执行器实际得到的 stdout/stderr 字节、退出码和工具 Observation。`observations/*.json` 是可能为上下文预算截断后的模型可见值。原始字节不会被覆盖。

只有后续真实 HTTP 请求携带相应原生工具返回、且端点返回 2xx 后，才写 `observation.delivered`。这证明请求体被端点接受，不证明模型注意到了内容，也不宣称能获知服务端内部推理。最后一步未发后续请求时，保留 result 和未送达状态，不生成虚构请求。

## 关联、差异与时序

事件 ID 在 Episode 内唯一；调用关联到模型请求/响应，结果关联到调用，差异关联前后状态以及相关工具结果。文件级变化可用于回答“此工具完成前后文件如何改变”，但不能把内容变化自动解释成根本原因。State Recorder 不解析 Excel 单元格公式或 Word 对象语义；当前 `recording.objects=true` 会显式报错。

## 校验到底证明什么

包括 Schema、文件清单/长度/SHA-256、事件哈希链、相对路径、引用覆盖、原生响应和规范化调用一致、实际请求中 Observation 与图片对应、文件版本与重算 diff、最终成品与终态、官方原始评分与投影一致、检查点配置和对话边界等。

`c3-l1 verify episode --expected-index-sha256 <可信摘要>` 可以把自洽校验锚定到外部保管的摘要。没有外部可信摘要/签名时，攻击者重写所有文件并重新封口仍可能通过自洽检查；没有硬件证明或外部运行收据时，哈希不能单独证明“曾真实执行”。本工具不声称解决恶意控制器伪造全部历史的问题。

## 冷检查点和分支

检查点写在工具批次已处理且没有待执行调用的边界；可以包含尚待下一请求送达的工具返回。资格检查包括 Runtime 的恢复范围、静止状态、镜像标识、请求/时间/Token 预算。测试替身 checkpoint 仅有测试资格，生产 CLI 不接受它作为真实恢复依据。

恢复读取并核验父包，复制冻结 Benchmark、从内容寻址存储恢复文件/普通权限/空目录、恢复原生对话并重新校验内容摘要。新 Episode 记录父包索引摘要和检查点摘要，执行新请求，不覆盖父轨迹。父记录中的送达引用保留父范围，不假造新分支工具结果。

只承诺文件和对话范围；包含 setuid/setgid/sticky 特殊权限的状态不具备本 profile 的恢复资格；初始工作区清除宿主目录继承的 setgid，避免错误承诺。

不承诺 UID/GID 身份、内存/进程、`/tmp`、shell 环境变量、模型服务端会话或相同生成。不可恢复的情况保留记录和理由，不伪装恢复成功。当前禁止恢复时修改配置；请求、工具、Token、时间预算累计，不通过新分支无限重置预算。

控制器在正常异常/KeyboardInterrupt 下尽量封口。在 SIGKILL、磁盘写满、宿主崩溃等情形下，未必能生成最终清单；日志可读前缀仍保留。`journal` 不是自动修复/续跑器。
