# L1 0.2 模块化完整工程交付报告

交付日期：2026-10-12。工程根目录：`C3_Experimental_Sandbox/`。

## 1. 交付结论

本版是**原项目＋P1＋新模块化现场主链路的完整合并工程**，不再要求用户叠加上一份补丁。不把工程交付说成全部真实环境验收完成：本次环境没有 Docker，没有发起真实模型推理，没有重新下载固定版本 OfficeVal 源码/数据。因此，G1 原生任务＋Docker＋真实模型闭环、G3 本地权重服务实测、G4 真实模型冷恢复仍需本机验收。

不是空目录设计：`c3-l1 run / resume / doctor / benchmark / verify / inspect / events / journal` 已实现；生产路径只有 Docker，无宿主机执行降级。测试替身只在测试代码或显式 Python 注入中使用，不注册为生产模型或 Runtime。

## 2. 本轮实际修改

- `core`：0.2 端口、严格配置、注册表、Episode 生命周期、预算/终止、冷恢复、预检。
- `models`：Gemini 原生调用；OpenAI-compatible Chat API/本地服务；原始请求响应、工具返回与图片字节、签名保留。
- `benchmarks/officeval`：原生 Git/数据双版本管理、下载/本地锁、任务发现/加载、私有评分资产冻结；Bundle 兼容另命名。
- `runtime`：逐工具新 Docker 容器、实际镜像探测与版本、CPU/内存/PID/网络/权限约束、超时/输出上限、清理事件。
- `recording`：文件/目录内容寻址、版本差异、冷恢复；不包含对象语义或进程内存。
- `evaluation`：独立评分容器、镜像 worker 摘要检查、原始 Verifier 调用、官方结果与 L1 投影分离。
- `evidence`：现场 0.2 合同、fsync 追加日志、完整索引、原生投影/状态/成品/评分/检查点校验；P1 0.1 兼容。
- `scripts/l1`：镜像构建、本机预检和显式授权真实模型/恢复验收；配置与中文操作文档。

已有旧 `r1`、`gateway`、`integrated` 源码没有被整体重写。L1 新路径不导入 Analyst。保留旧源码不等于旧依赖已经齐备，见旧测试结果。

## 3. 本次实际验收

| 验收项 | 实际结果 | 范围与凭证 |
|---|---|---|
| L1 自动化测试 | **179 passed，2 skipped**，26.74s | `validation/l1-tests.txt`、`l1-tests.xml`；70 项原 P1＋109 项新增通过 |
| HTTP 协议 | 通过 | 真实 loopback HTTP 请求/响应，对端是确定性测试服务，不是 Gemini/本地权重模型 |
| 文件与恢复 | 通过 | 固定测试编写的子进程复制文件、原始字节记录、冷恢复后新测试请求和新工具执行；不是容器资格 |
| 原生源读取 | 通过 | 临时 fixture Git repo＋随包真实任务定义/输入/Verifier，测试 clean commit/锁/资产错误；不是新下载的 upstream checkout |
| 官方评分入口 | 通过 | 原始 OfficeVal003 Verifier 在宿主 Python 子进程实际运行；`validation/official-verifier-host-smoke`；无 Agent、无 Docker、非官方 batch runner |
| P1 八历史包 | **8/8 verified** | `validation/p1-backcompat.json`；原缺失未被“补齐”或升级 |
| 构建 wheel | 通过 | `validation/wheel-build.txt` |
| 独立 wheel 检查 | **11/11 evidence packages verified**，两份 Schema 可读 | `validation/wheel-check.json`；从解开的实际 wheel 导入，未加载 integrated/gateway/Inspect/MLX |
| 解压整包再测试 | **179 passed，2 skipped**，28.84s | `validation/release-extracted-tests.txt`；在独立解压目录运行 |
| Docker 实测 | **未执行** | 没有 Docker 命令；两项跳过测试之一是显式 opt-in Docker 测试 |
| 真实模型＋Docker | **未执行** | 未配置本次模型运行凭证；另一项显式 opt-in live API 测试未启用 |
| 固定版本上游下载 | **未通过实地验证** | shell DNS 无法访问 GitHub；不自动换 main，不声称随包完整上游 |

命令：

```bash
python -m pytest tests/l1 tests/l1_live -q \
  --junitxml=docs/l1_v2/validation/l1-tests.xml
```

### 旧测试也重新执行了

```bash
python -m pytest tests --ignore=tests/l1 --ignore=tests/l1_live \
  --continue-on-collection-errors -q \
  --junitxml=docs/l1_v2/validation/legacy-tests.xml
```

实际结果：**426 passed、42 failed、2 skipped、37 errors，另有 11 subtests passed**，7.86s。原始 stdout 和退出码分别保留在 `validation/legacy-tests.txt` / `legacy-exit-code.txt`。可见问题包含 `inspect_ai` 缺失、旧工程外部绝对/相对路径下 OfficeVal Git 仓库缺失、旧回执缺失等；没有把所有失败未经逐项分析一概归因于环境，也没有沿用原打包说明中“640 全通过”的结论。

上述 179 项 L1 通过不代表旧全量项目全绿。

## 4. 随包三个例子

| 目录 | 记录内容 | 资格 |
|---|---|---|
| `examples/l1_evidence/fixture_episode` | 2 次测试请求、1 次固定真实子进程工具结果、1 条可核实送达、3 个文件快照 | `test_fixture`，execution unknown |
| `examples/l1_evidence/fixture_branch` | 从 cp000001 恢复后新建 Episode，2 次新测试请求、1 次新固定工具结果 | 仅测试恢复资格，不可用于生产 CLI resume |
| `examples/l1_evidence/missing_docker` | 真实生产入口运行到 Docker 前置检查失败并封口，5 事件、0 模型请求、0 工具结果 | execution invalid，evaluation not_run，evidence incomplete |

三个包均可独立校验。`fixture_grade` 是明确合成评分，不能称为模型的官方成绩。另有独立的原始 Verifier 宿主 smoke 回执，两类证据没有混合标签。

## 5. 交付前发现并修复的问题

在 `/mnt/data` 共享目录实际生成分支示例时，宿主 setgid 继承导致原快照带特殊权限位，恢复 profile 仅恢复普通权限后摘要不同。原临时目录测试没有暴露该情形。

已修复初始工作区继承权限问题；包含不支持特殊权限位的状态明确不具备恢复资格，直接恢复也拒绝；增加共享 setgid 目录、资格拒绝、恢复拒绝三项回归测试。重新生成分支后真实发生新的测试请求和固定工具子进程，父包未覆盖。没有仅以“包 verify 通过”就声称恢复成功。

## 6. 本轮承诺与不承诺

**已交付实现**：模块化源码、两类模型传输协议、OfficeVal native/bundle 适配、Docker 和独立评分主链路、文件证据与冷检查点、单一 CLI、安装/配置/测试/本机验收文档、示例及实际日志。

**尚未实测或不在范围**：本机 Docker 镜像构建和运行、真实 API/本地模型推理、完整原生端到端和真实模型续跑；完整 OfficeVal 数据/源码随包、另外两个 Benchmark、自动启动本地权重/MLX、对象级状态、Windows COM、浏览器/VM、热进程快照、确定性 LLM 输出重放、实际金额计费、任意工具插件注册、官方完整批处理评分。

当前具备文件型任务的可扩展端口，不把任意 `Model × Benchmark × Runtime` 组合宣称天然可用。不支持的能力必须拒绝或显式降级标注。

## 7. Benchmark 与评分可信边界

随包已有八任务：002、003、006、012、015、018、033、037。八任务 Bundle 与 Gemini 历史原始记录原样保留；缺失的历史 306 个工具前后快照没有找回，也没有凭空重建。

官方 `status=ok` 不等于成功；默认保存官方维度和分数，通用成功标签不充分时留 unknown。可选满分映射是 L1 命名政策。当前调用上游单任务 Verifier，不等价于完整官方 batch runner；Linux profile 拒绝九个硬 COM 任务，其余任务也不自动宣称 Rubric 全覆盖。

哈希链与索引在无外部可信摘要时提供内容自洽校验，不能独立证明整个历史未被恶意控制器伪造。断电/SIGKILL/磁盘写满可能留下未封口前缀，不承诺自动崩溃恢复。输出/快照上限也不是磁盘写入硬配额。

## 8. 本机下一道验收

按根 README：安装 → 原生源码和数据准备 → Docker 镜像构建 → doctor → `accept_local.py --execute-model`。

必须使用 `officeval-native`，出现新请求/新真实工具结果，官方评分有效，证据可独立校验；不以高分作为交付门槛。之后用本地推理服务重复，再执行显式 `--resume` 验证真实续跑。不要在这道门槛前扩展 B2/B3。
