# 本地合成工程检查

此处只有一个明确标记为 synthetic 的微型 JSON 任务，用于核对 Inspect、Docker、成品传递和独立评分是否连通。没有 LLM 请求；不能将成功率用于任何 C3 研究主张。

任务与评分使用两个不同镜像。Inspect 会为同一样本创建两个环境，工具执行与评分按顺序进行；两个环境可以同时处于空闲存活状态，不代表并发实验。最大样本、任务和执行并发均为 1，CLI 另有本项目进程锁。

从项目目录构建，第一条 build 成功后才运行第二条：

```bash
mkdir -p runs/buildx
export BUILDX_CONFIG="$PWD/runs/buildx"
export PYTHON_BASE='python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1'
docker pull --platform linux/arm64 "$PYTHON_BASE"
docker compose -f docker/compose.yaml build default
docker compose -f docker/compose.yaml build grader
.venv/bin/c3 smoke --output runs/my-first-smoke
```

`PYTHON_BASE` 摘要是本轮实际拉取的官方 Python 多平台镜像摘要，运行固定为 ARM64。运行前再读取本地任务/评分镜像内容 ID，生成的 Compose 文件使用该 ID，防止运行期间标签变化。

每次输出目录应是新的。`inspect-smoke.json` 留下配置/源码摘要、有效容器设置、镜像身份、冻结成品摘要、评分、模型调用数量和清理结果；Inspect 日志保存单次执行细节。错误执行也保留记录，不覆盖成功或失败历史。

Inspect 使用其标准用户应用数据/缓存目录。受限执行环境可能需要允许该目录的本地写入；项目不修改 Docker daemon、用户组、镜像仓库凭据或全局配置。

两容器均为非 root 用户、只读根文件系统、无网络、移除 capabilities、禁止新增权限，只提供独立工作区和临时文件系统。控制器读取实际容器配置核验这些设置，检查任务环境没有 grader 文件及 Docker socket。该有限检查并不等于全面抗逃逸审计。
