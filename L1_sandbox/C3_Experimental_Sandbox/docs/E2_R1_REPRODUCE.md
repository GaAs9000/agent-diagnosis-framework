# R1 本机串行复现

这些入口执行开发集工程验收，不开发 D6–D8，不扩任务或模型。现有 `runs/e2-r1` 是不可覆盖的证据目录；新运行必须使用新目录。失败不会自动重试，也不会为了凑成功案例选择其他 Seed。

本机已具备 `.venv`、`.model-venv`、固定模型文件、三个 Docker 镜像和冻结 OfficeVal 包。依赖分别记录于 `requirements.local-arm64.lock.txt` 和 `requirements.model.local-arm64.lock.txt`；镜像构建定义在 `docker/Dockerfile.r1-*`。首次部署到其他机器仍需安装这些锁定依赖并取得同一模型修订和 OfficeVal 资产，不应把本机路径直接当成可移植安装包。

在项目根目录启动两个服务，各自占一个终端：

```bash
.model-venv/bin/mlx_lm.server --model models/Qwen3-4B-Instruct-2507-4bit --host 127.0.0.1 --port 18080 --prompt-cache-size 0 --decode-concurrency 1 --prompt-concurrency 1 --max-tokens 2048 --temp 0 --log-level DEBUG
.model-venv/bin/python src/c3_sandbox/r1/tokenizer_server.py --model-path models/Qwen3-4B-Instruct-2507-4bit
```

记录模型服务的实际 PID 到一个只包含数字的文件，供 `observe_r1_server.py --pid-file` 使用。进程核验只读取该 PID 的参数和监听端口，不修改系统代理或全局配置。所有模型 HTTP 请求使用显式本地客户端；Agent 容器没有网络。

依次运行以下步骤。`runs/e2-r1-reproduce` 和 `runs/r1-qualification-reproduce` 应为新的目录名：

```bash
.venv/bin/python -m c3_sandbox.r1.qualification runs/r1-qualification-reproduce
.venv/bin/python scripts/run_r1_originals.py --run-root runs/e2-r1-reproduce --qualification-dir runs/r1-qualification-reproduce
.venv/bin/python scripts/observe_r1_server.py --run-root runs/e2-r1-reproduce --pid-file /absolute/path/to/model.pid
.venv/bin/python scripts/run_r1_restore.py --run-root runs/e2-r1-reproduce
.venv/bin/python scripts/audit_r1_originals.py --run-root runs/e2-r1-reproduce
.venv/bin/python scripts/prepare_r1_panel.py --run-root runs/e2-r1-reproduce --output-name probe-audit-v2
.venv/bin/python scripts/run_r1_scheduler.py --run-root runs/e2-r1-reproduce
```

入口当前固定使用 003、002 两个开发任务，分支使用 002 的首个完整工具边界。若该边界不能恢复、没有合格原生调用、原始结果不符合 E2-F，或真实匹配失败，运行必须停止；不得把不合格状态改成 `is_synthetic=true` 继续。

现有证据可以在不启动模型或 Docker 的情况下核验：

```bash
.venv/bin/python scripts/verify_r1_evidence.py
.venv/bin/python -m pytest -q
```

核验器针对本轮保存证据的字段和案例设计，未来更换任务或协议时要独立更新，不能套用本轮结果。实际 Docker 负例入口为 `c3_sandbox.r1.runtime_negatives.run_runtime_negatives(agent_image, output_dir)`；它会顺序启动四个临时环境，不调用模型。

每个模型请求保存完整消息、工具 Schema、SDK 请求／响应、归一化事件和 Token 用量；每次分支保存独立评分器输出。`scheduler.sqlite`、带私有标记的控制器审计和 HMAC 密钥不得挂载进 Agent 或交给方法客户端。对外报告使用不含这些凭据的统计与哈希。

复现结束应停止自己启动的两个服务。不要重写旧记录；若工程错误导致分支进入 `indeterminate`，保留失败目录并在新、明确版本的运行计划下重新验收。
