# E2-R1.1 复现说明

本轮运行已结束，状态为 `PARTIAL_VERIFIED`。以下区分核验已有证据与重新发起模型执行。不能覆盖 `runs/e2-r1-1`，也不能把复跑结果并入本轮七次原始执行的分母。

## 核验现有证据

在 `C3_Experimental_Sandbox` 根目录执行，无需启动模型或 Docker：

```bash
.venv/bin/python scripts/verify_r11_evidence.py
.venv/bin/python -m pytest -q
```

保存新核验结果时使用尚不存在的文件名：

```bash
.venv/bin/python scripts/verify_r11_evidence.py --output runs/e2-r1-1/verification/recheck.json
```

核验器会检查冻结源码、日志链、完整执行分母、首次 S/T 选择、SDK 原始响应、Token、实际调度、文件差分、恢复签名、评分及隔离证据。核验不产生新的模型输出，也不把 G3 的缺失变成通过。

## 固定执行配置

本机依赖位于 `.venv` 和 `.model-venv`，分别由两份 `requirements.*.lock.txt` 记录。模型文件位于 `models/Qwen3-4B-Instruct-2507-4bit`。不可变镜像 ID、模型／tokenizer 指纹、工具接口、提示、预算、种子和代码清单全部在 `candidate_order.json`；执行时源码原件在 `source_snapshot/`。

本轮模型服务启动参数：

```bash
.model-venv/bin/python -m mlx_lm.server --model models/Qwen3-4B-Instruct-2507-4bit --host 127.0.0.1 --port 18080 --prompt-cache-size 0 --decode-concurrency 1 --prompt-concurrency 1 --max-tokens 2048 --temp 0 --log-level DEBUG
.model-venv/bin/python src/c3_sandbox/r1/tokenizer_server.py --model-path models/Qwen3-4B-Instruct-2507-4bit
```

两个服务需各自启动并记录实际 PID；它们目前已经关闭。Agent 容器没有网络、宿主挂载或隐藏评分文件。原始运行、工具续跑和评分按串行方式执行，模型客户端不继承系统代理。

## 新运行的阶段入口

`scripts/run_r11.py` 接受 `--run-root`。新的根目录必须先准备真实 G0 资格、来源审查、评分器资格、服务观察和冻结资产。入口会重新核对它们的哈希，不接受只填写 `PASS` 的占位文件。

准备组件是 `r1.qualification.qualify_model`、`r1.tool_protocol.run_protocol_fixture`、`r1.officeval_selection.freeze_selection` 和 `r1.office_run.grade_workspace`。来源审查、评分器导入预检与镜像核对须在模型候选运行之前完成。`scripts/audit_r11_sources.py`、`run_r11_protocol_audit.py` 和 `audit_r11_runtime.py` 保留了本次确切调用，但路径固定于本轮证据目录，不能直接再次运行覆盖原件；新实验需要为这些准备步骤配置新的输出根目录。

前置材料齐全后，三个阶段依次为：

```bash
.venv/bin/python scripts/run_r11.py freeze --run-root /absolute/path/to/new-run
.venv/bin/python scripts/run_r11.py originals --run-root /absolute/path/to/new-run
.venv/bin/python scripts/run_r11.py coverage --run-root /absolute/path/to/new-run
```

`freeze` 发布候选顺序、配置和源码快照；`originals` 逐任务执行一次并记录所有结果；`coverage` 验证结果和选择封存，执行首次 T 的 Original 续跑，有 S 时才运行其八个新分支。缺少 S 会写入未运行记录。T 未发生新调用则标记 `tool_continuation_not_observed`，不会自动改 Seed、探针或任务。

这些入口可在当前工作区调用。迁移到其他机器仍需配置相同依赖、资产及本地模型服务；它们不是一个已经验证的跨平台一键安装包。本轮七个任务今后只可用于明确标注的开发或复现，不得作为未见研究确认集。

## 错误和预算规则

原始调用和物理分支目录采用独占创建。已经开始的尝试不会自动重跑。遇到中断或 `indeterminate`，保留该目录和日志；先核对哪些请求已经发生，再在新协议中决定处理方式。

SDK 解析失败保存原始响应和已计量 Token。Python 工具报错保留退出码、stderr 和 observation。评分器缺失规则、异常或未评估项目不会被视为成功。真实原始成功但 fresh Original 失败，也不能改写或归因给探针。

探索和确认使用固定主种子派生的不同域；temperature=0 不构成独立随机样本保证。确认结果只由控制器读取。HMAC 密钥、调度数据库及私有观察不得挂载到 Agent 容器中。
