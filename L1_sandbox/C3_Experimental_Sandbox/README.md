# L1 · Benchmark-driven Agent Execution & Evidence Sandbox

**版本 0.2.0｜2026-10-12｜完整合并工程，不是 P1 叠加补丁。**

给定一个 Benchmark 任务和一个 Agent 配置，现场执行并产出原始请求、真实工具结果、模型实际收到的 Observation、文件版本、成品、官方评分和可核验证据。本路径不引入 Analyst，不推断失败原因或模型能力。

> 本包已实现模块化主链路与本地验收入口。本次交付环境没有 Docker，没有运行真实模型；因此没有宣称 Docker＋真实 API/本地权重的端到端验收通过。详见 `docs/l1_v2/RELEASE_REPORT.md` 和随包原始测试日志。`tests/l1_live` 中包含明确标注的测试替身，不是模型实测成绩。

## 工程中有什么

| 模块 | 当前实现 |
|---|---|
| Benchmark | `officeval-native`：固定源码/数据版本、原始任务加载、资产准备、原始单任务 Verifier；`officeval-bundle`：显式兼容随包八任务，不能冒充 native |
| Model | Gemini 原生协议；OpenAI-compatible Chat Completions 协议，可连接 API 或本地推理服务 |
| Runtime | `docker-cold`，逐工具新建 Linux 容器，仅 `/workspace` 持久化；禁止宿主机执行降级 |
| Orchestration | 独立 `run` / `resume` / `doctor`，预算、终止、异常留存；不依赖旧 integrated/gateway |
| Recording | 原始 HTTP body、工具 stdout/stderr、送达凭证、文件/目录状态与内容寻址存储、冷检查点 |
| Evaluation | 独立容器调用官方 `evaluate(directory)`；保留原始分数与独立有效性判断 |
| Evidence | 新现场合同 0.2.0，仍可读 P1 历史合同 0.1.0；哈希、事件链、原生投影和引用校验 |

完整保留原项目源码、测试、八任务 Bundle、原始 Gemini 历史记录；不包含完整上游 OfficeVal Git 仓库/全部数据、模型权重、虚拟环境或已构建 Docker 镜像。B2/B3 尚未实现。旧 `c3` 入口仅为兼容保留，历史文档中的验收结论不自动适用于本版本。

## 1. 安装和离线验证

解压到**新目录**，不要覆盖仍在运行的旧工程。Python 3.12+，推荐使用独立虚拟环境：

```bash
cd C3_Experimental_Sandbox
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[l1,test]'
python -m pytest tests/l1 tests/l1_live -q
c3-l1 registry
c3-l1 verify examples/l1_evidence/fixture_episode
```

示例 `fixture_episode` / `fixture_branch` 是确定性测试替身；`missing_docker` 是环境不满足时真实产生的失败回执，均不是成功的模型实测。

## 2. 原生准备 OfficeVal

先检查 `configs/l1/officeval.gemini.json`。相对路径以**配置文件所在目录**为基准，而非当前工作目录。

```bash
c3-l1 benchmark fetch --config configs/l1/officeval.gemini.json --task officeval_003
c3-l1 benchmark list --config configs/l1/officeval.gemini.json
c3-l1 benchmark show --config configs/l1/officeval.gemini.json
```

默认固定版本继承自原交付：

- 源码 Commit：`5f3da616ac6a0bc30ca35e9678f8f1eb255bd91a`
- 数据 Revision：`cd6ba6d8fb83b3fb551e24eebc20e1fb0bd154a5`

源码与任务数据分别管理。`fetch` 从公开上游取得指定任务所需的任务 JSON、Rubric、输入资产，并建立哈希锁；不会回退到 `main`，不会重置已有源码目录。**本次环境未能联网取得这两个版本，旧数据 Revision 的可下载性未重新验明。**下载失败时先解决固定版本来源，不要把 Bundle 或新数据悄悄改名为 native。

已经有完整的源码和资产时，修改配置路径，使源码是对应版本的干净 Git checkout，目录满足：

```text
source_dir/verifiers/officeval_003_verifier.py
 data_dir/tasks/officeval_003.json
 data_dir/rubrics/officeval_003.json
assets_dir/officeval_003/<原始输入文件>
```

然后显式接受本地数据并锁定：

```bash
c3-l1 benchmark lock-local --config configs/l1/officeval.gemini.json --task officeval_003
```

`lock-local` 证明后续读取与接受时的本地文件一致，**不独立证明这些数据来自所声明的上游 Revision**。已存在锁与文件不一致时拒绝覆盖。详见 `docs/l1_v2/OFFICEVAL.md`。

## 3. 构建隔离环境并运行

需要本机可用的 Linux Docker 容器环境。API 在控制器侧请求；模型密钥不挂载到工具容器。

```bash
bash scripts/l1/build_image.sh
export GEMINI_API_KEY='在本机填写，不要写入配置或仓库'
c3-l1 doctor --config configs/l1/officeval.gemini.json
c3-l1 run --config configs/l1/officeval.gemini.json --output runs/l1-native-001
c3-l1 verify runs/l1-native-001
c3-l1 inspect runs/l1-native-001
c3-l1 events runs/l1-native-001 --kind tool.result --limit 10
```

配置中的模型名只是可修改的示例，需要与你的服务账号可用模型一致。`doctor` 不请求模型，但会启动临时容器探测镜像环境；绿色预检不等于模型工具能力或评分已验收。

**本地模型**使用 `configs/l1/officeval.local.json`，将地址和 `model` 改为已启动的、支持原生工具调用的 OpenAI-compatible 服务。当前不是自动下载/加载 MLX 权重；API 路径不依赖 Metal。本地服务默认不发送图片，可显式改能力配置。

## 4. 有收据的本机验收

不付费调用模型的预检：

```bash
python scripts/l1/accept_local.py --config configs/l1/officeval.gemini.json \
  --output-root runs/l1-preflight-001
```

明确授权真实模型调用：

```bash
python scripts/l1/accept_local.py --config configs/l1/officeval.gemini.json \
  --output-root runs/l1-accept-001 --execute-model
```

增加 `--resume` 会尝试从合格的工具执行后检查点建立新分支，**可能产生额外 API 费用**。验收要求新请求、新真实工具结果、评分与证据有效，不要求任务高分。模型恢复后直接答复而不调用工具时，不满足“真实工具续跑”这一额外门槛，不能伪造工具操作来刷验收。

## 5. 冷恢复与证据复核

```bash
# 先查看 checkpoints/index.json，选择 eligible=true 的检查点。
c3-l1 resume --package runs/l1-native-001 --checkpoint cp000001 \
  --output runs/l1-native-001-branch
c3-l1 verify runs/l1-native-001-branch
```

只恢复已声明范围：文件内容/路径/普通权限、目录、原生对话。使用同一已解析配置、同一 Agent 镜像 ID，预算累计；不恢复进程内存、`/tmp`、打开的文件描述符或服务端隐藏状态，不保证未来生成完全相同。每条分支新建 Episode，不覆盖父包。

强制杀死控制器后，可能只剩未封口日志：

```bash
c3-l1 journal runs/l1-native-001 --limit 20
```

该命令只读取可验证前缀，不把不完整日志认证为完整运行；不提供自动崩溃恢复。

## 文档入口

- `docs/l1_v2/ARCHITECTURE.md`：模块边界、实现接口、插件扩展与限制。
- `docs/l1_v2/EVIDENCE.md`：0.2 合同、状态含义、可信边界、检查点。
- `docs/l1_v2/OFFICEVAL.md`：原生来源、官方评分边界、兼容模式。
- `docs/l1_v2/OPERATIONS.md`：资源/密钥/异常/本机验收。
- `docs/l1_v2/RELEASE_REPORT.md`：实测结果与未验收项。
- `docs/l1_v2/SOURCES.md`：上游依据。
- `docs/l1/`：保留的 P1 证据合同与迁移说明；`docs/l1_v2/P1_README.md` 保留上一版本首页。
