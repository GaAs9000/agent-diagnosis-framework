# 本地执行、隔离与验收

## 支持范围

控制器 Python 3.12+，工具执行需要 Linux Docker 容器；macOS 可使用本地 Docker Desktop 的 Linux 容器环境。本轮没有做特定 macOS/Apple Silicon/Windows 实机资格认证。推荐先在你的本机完整运行验收脚本。

仅容器工具无网络；控制器可以访问模型 API，Benchmark 下载也在控制器侧显式进行。没有 `docker` 或镜像时立即拒绝，不切到宿主执行模型命令。容器采用非 root UID、只读 rootfs、`--cap-drop ALL`、no-new-privileges、内存/CPU/PID 限制、临时 `/tmp`。这**不等于宣称 Docker daemon 自身 rootless**，也不等于虚拟机级隔离。不要用高价值宿主机运行不信任的可执行插件/Verifier。

每次 shell 工具使用全新容器；只有 `/workspace` 挂载保留。不保留后台进程、临时目录、工作目录变更、shell export 等状态。系统 prompt 向 Agent 告知该执行模型。`read_image` 是受路径和大小限制的可信只读文件读取器，发送真实图片字节。

评分容器只读挂载冻结源码和冻结最终提交，单独挂载评分输出目录；不读取 Agent 未提交的任意工作区文件。评分 image 内 worker 的摘要必须与本工程版本相同，否则要求重建镜像。

## 边界与资源

默认每次工具 stdout+stderr 捕获上限 4 MiB；超限终止工具并保留前缀，标记不完整，不假装保留了超限后的全部字节。模型原始响应也有上限。发送给下一轮模型的文本可进一步按配置截断，但原始工具字节不因此覆盖。

文件状态默认最多 256 MiB / 10,000 文件，防止无限快照内存占用；**这是采集上限，不是写入时磁盘硬配额**。恶意/失控命令在超时前仍可能写满所挂磁盘。正式执行建议使用专用磁盘/卷及宿主监控；不把当前限制称作完整磁盘资源隔离。

拒绝工作区的符号链接、硬链接、FIFO、socket、设备文件和路径逃逸。出现不能安全采集的状态时不使用上一份快照冒充终态评分；保留错误与已有证据，停止相应阶段。

默认无自动 HTTP 重试。Token 预算按响应 usage 进行后验阈值检查，不能在未知未来响应前保证精确 Token/金额上限；金额与账单保持 null。请求数、工具数、时间预算是不同维度。实际生成延迟来自控制器时钟，不解释成模型内部计算时间。

## 密钥与证据

配置只保存 `api_key_env` 名称，不允许直接放 API Key/Token/Authorization 字段。传输认证头不进入请求 JSON 或日志；真实测试验证了测试密钥不进入证据。运行输出仍可能因模型正文/用户 prompt/任务文件而含秘密：不要把原始证据包当作已自动脱敏的公开数据。

`http://` 仅允许声明为 local 的 loopback endpoint；远程服务使用 HTTPS。拒绝 URL 用户名密码/查询字符串/重定向，避免认证意外外传。当前 local 模型需要已启动的服务，不自动管理权重/模型服务进程。

## 中断与清理

已创建容器在日志写入 container ID 后启动；正常结束/超时/异常路径尝试清理并记 receipt。控制器被 SIGKILL 或宿主崩溃可能遗留容器，可人工查看：

```bash
docker ps -a --filter label=l1.managed=true
```

先用 `c3-l1 journal <episode>` 核对具体 container ID 和任务，再对确认不再需要的容器处理。不要自动批量删除并发任务的容器或工作区。未封口包只能称日志前缀，不能过度承诺崩溃恢复。

## 验收层次

1. 离线 tests：使用明确的测试替身验证接口、异常、证据、状态与恢复。不是模型测评。
2. 本地真实 Docker：`L1_DOCKER_TESTS=1 python -m pytest tests/l1_live/test_http_runtime.py -m docker -q`。显式启用后缺 Docker 会失败，不会悄悄跳过。
3. 原生真实单任务：`scripts/l1/accept_local.py ... --execute-model`，或显式设置 `L1_LIVE_CONFIG` 运行 `-m live_api` 测试。需要你的模型可用配置、密钥和资产；可产生费用。
4. 真实恢复：上述脚本增加 `--resume`。必须新模型请求＋新真实工具结果；只输出“已完成”不满足此额外门槛。
5. 第二后端：将配置换为本地服务，独立跑上述门槛。API 成功不能代替本地后端资格。

CLI `run`：0 表示封口校验及执行/评分没有 invalid，不要求任务高分；2 表示环境/执行/评分无效；3 表示证据校验失败。自动化应读取 JSON 的具体字段而不只看退出码。验收脚本也区分 preflight-only 与 `qualification_passed=true`。

B2/B3、对象状态、VM/浏览器运行时、热进程快照、确定性模型重跑、价格自动查询和账单核销均未在本版提供。
