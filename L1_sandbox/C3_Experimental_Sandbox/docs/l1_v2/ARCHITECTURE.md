# L1 0.2 模块与扩展合同

## 两条独立接入链，一个执行生命周期

```text
模型配置 → Model Registry → ModelBackend ──────────┐
                                                   │
Benchmark Registry → BenchmarkAdapter → Task ──────┤
                                                   ▼
                                         Episode Orchestrator
                                          │              │
                                SandboxRuntime       Evaluator
                                          │              │
                                    实际工具执行      独立官方评分
                                          └──────┬───────┘
                                                 ▼
                                       Recorder / StateStore
                                                 ▼
                                           Evidence Package
```

这里 Sandbox 指整个执行与证据平台；Docker 是当前隔离运行时的一种实现。模型/Benchmark/Runtime 的组合不是无条件笛卡尔积：接口统一后仍必须满足工具、图片、操作系统、状态可恢复性等能力要求。不支持的组合拒绝运行，不做无痕降级。

## 源码布局与唯一责任

```text
src/c3_sandbox/l1/
  core/          # ports, config, registry, orchestrator, doctor
  models/        # 原生 Gemini / OpenAI-compatible, HTTP 传输
  benchmarks/    # 注册表、OfficeVal native/bundle、源码与数据获取
  runtime/       # Docker、run/read_image 工具契约
  recording/     # 文件状态、差异、恢复
  evaluation/    # 评分器生命周期、镜像侧 worker、官方投影
  evidence/      # 0.2 schema、追加日志、封口、独立校验
  cli.py         # 独立命令
  contracts.py / recorder.py / import_gemini.py / ...  # 保留的 P1 历史兼容代码
```

P1 根目录 `contracts.py` 是历史接口；**新现场接口的唯一入口是 `core/contracts.py`**。根目录 `SCHEMA_VERSION=0.1.0` 专用于 P1 导入，新现场合同 `core.contracts.VERSION=0.2.0`。文件位置不同不意味着任意互换。

## 实际可用接口

`BenchmarkAdapter.list_tasks()` / `.load(task_id) -> Task`：任务返回公有 prompt/assets，私有 verifier/rubric 文件及 `EvaluationSpec`。不调用模型、不执行评分。`Task.provenance` 保存版本与来源；能力信息放 `requirements`。

`ModelBackend.identity / initial / request / send / parse / accept`：构造并保留原生协议，发送真实 HTTP、提取规范化调用、将真实工具 Observation 回填原生对话。原始响应永远保留。文本中的“Action/Observation”不视作原生工具调用。后台不自动重试，以免隐去额外请求、费用和不可重复副作用。

`SandboxRuntime.preflight / execute / qualification`：实际环境凭证、工具执行与冷恢复资格。当前 Runtime 另暴露 `is_quiescent` 与可选 `set_event_sink`；异构 Runtime 要保证状态采集时无写入者，并负责报告资格。无法证明静止的状态不能评分或恢复。

`Evaluator.evaluate(spec, submission, output)`：在独立边界调用 Benchmark 提供的评分入口，保留原始字节、进程结果和评分投影。不改官方规则，也不把 C3 operational 标签冒充官方成功标签。

`StateStore.capture / diff` 与 `restore`：基于内容寻址的文件级状态。`Journal.emit` 先落盘 fsync，再对外执行；`seal/verify` 生成和校验可独立读取的证据包。Recorder 不依赖任何分析模型。

## 注册表和插件

`c3-l1 registry` 只显示已有实现，不把 B2/B3 路线图列成可用模块。内置是：

- models：`gemini`、`openai-compatible`。
- benchmarks：`officeval-native`、`officeval-bundle`。
- runtimes：`docker-cold`。
- evaluators：`docker-verifier`。

新增模块可以显式注册工厂，或者在安装包中使用 entry-point 组 `l1.models` / `l1.benchmarks` / `l1.runtimes` / `l1.evaluators`。工厂接收相应配置字典。只有 `plugins.load_installed=true` 才加载外部 entry points，因为安装的插件是可执行的可信代码，不是安全数据。

核心配置拒绝未知字段。插件自定义参数放到所属段的 `options` 对象，工厂调用前展开；与标准参数冲突会报错。复杂选项与能力验证由插件负责。标准配置中的保留字段不要改作其他语义。

**扩展边界**：0.2 内置工具集仍是 `run/read_image`，不是任意工具插件系统；异构状态仍须新增 Collector，不能靠修改 `requirements` 自动获得浏览器/VM/数据库状态。未知模型协议的独立校验会明确提示没有原生投影校验，不能默认为已有同等保证。因此新增 Benchmark 的第一目标是不动模型/证据核心；只有引入新状态类型或工具协议时，才有受控的合同演进。

## 设计中避免的耦合

运行器只处理 Task、EvaluationSpec、原生请求和事件；OfficeVal 字段解释在 Adapter/Evaluator。目录名 `/workspace/submission` 是当前文件任务执行 profile，不是声称所有 Benchmark 都有 Office 文档。评分保留原始结构，不强制所有基准压缩为 0–1 单值。

旧 C3 模块继续存在以免破坏已有使用者，但本轮 L1 不将旧诊断/分析模块注入主链路。迁移是新增独立路径，不是宣布所有旧全量测试与旧功能已重构完成。
