# Exp1 可执行计划：验收标准缺口的判定实验（2026-10-22）

> 状态：计划已可执行，尚未授权、尚未运行。本文与配套脚本只做了离线验证，没有发出任何 Actor、Analyst 或评分请求，也没有启动 GPU。
>
> 方向依据：[RESEARCH_DIRECTION_UNWRITTEN_ACCEPTANCE_20261022.md](./RESEARCH_DIRECTION_UNWRITTEN_ACCEPTANCE_20261022.md) 第 8 节。

## 1. 要回答的问题

在冻结的 DEV45 配置（Qwen3.6-27B + L1 Pi harness context-v2）下，回答两件事：

1. 把评分细则直接告诉 Agent，成绩会不会大幅提高？如果会，说明"写不出来的验收标准"确实是瓶颈。
2. 现有的两类做法——先自己写验收清单再对照检查（PSCI 式），以及渲染成图片后自查（GDPval 式）——能补上多少差距？

只有"给出细则后提升很大，而现有做法补不上"同时成立，才值得开发验收编译方法。

## 2. 实验设计

### 2.1 四个条件

| 条件 | 额外给 Agent 的内容 | 作用 |
|---|---|---|
| O | 无（与 DEV45 冻结配置哈希完全相同） | 同期对照 |
| F | 该任务的完整评分规格：冻结的 D1 `R_implicit` 要求清单（任务里写明的要求，加评分细则），以及 verifier 实际查找的交付文件名 | 信息上限 |
| T | 开工前根据任务和输入材料写一份验收清单，保存为 `/workspace/acceptance_contract.md`，结束前逐条对照检查并修复 | PSCI 式文字合同基线 |
| V | 结束前把交付文件转成 PDF 再转成 PNG，用 read 工具查看，对照任务和参考材料检查并修复 | GDPval 式渲染自检基线 |

- 条件文字放在系统提示里：插在 Pi 基础提示和沙箱执行契约之间，开头是固定标记 `[补充说明]`。Pi 压缩上下文时会保留前两条消息（系统提示和任务），所以这段文字整轮都可见；如果放在用户消息里，会被压缩摘要掉。
- 除 `agent.system_prompt` 外，F、T、V 的配置与 O 完全一致。脚本会逐条检查：每个配置都是 L1 校验的不动点，执行契约那一段没有变化。
- 每条运行的第一个请求还要再核对一次：实际发出的系统提示必须与封存配置完全相同，并且 F/T/V 带标记、O 不带标记。
- F 文本的来源：逐字使用冻结的 D1 `R_implicit` 要求清单，去掉了开头的 D/S 流程句和 003 的一条内部注记"（是否可由原图支持待人工看图）"。003、004、085 另外写明交付文件名，脚本会核对这些文件名确实出现在对应 verifier 的源码里。
- 条件文字和完整系统提示见 [preview.json](./acceptance_gap_exp1_20261022/preview.json)，`review_digest` 为 `1ca1856f3fcd59dbb656ad7ba2861e8d481af62e5ef06b99f7e4c855ebacce90`。

### 2.2 任务、重复与顺序

- 主分析用 8 个开发任务：003、004、010、033、050、060、070、090，每个跑 O/F/T/V 四个条件。
- 085 只跑 F。它的 verifier 硬性要求三个题面和官方 runner 都没给出的文件名，O/T/V 在那里必然为 0。085 的 F 单独报告，不进主对比。
- 共 5 个区块，每个区块包含每个格子各一次重复：8 × 4 + 1 = 33 条运行；5 个区块共 165 条。区块内的顺序用固定种子打乱，并写入封存的计划。
- O 在同期重新跑，而不是直接复用 DEV45 的 45 条。原因有三：DEV45 之后 L1 改了 3 个文件；GPU 服务是新启动的；同期随机交错可以排除时间漂移。DEV45 的 O 只用来做漂移检查。

### 2.3 不变的部分

模型权重、vLLM 启动参数、Pi harness 版本、工具、镜像、每条运行的预算（35 次请求、140 次工具调用、10,800 秒）、T=1、无 seed、官方评分器，全部与 DEV45 相同。不使用留出家族、Analyst、续跑分支。

### 2.4 预算

| 项目 | 每个区块 | 5 个区块合计 |
|---|---:|---:|
| 运行数 | 33 | 165 |
| 逻辑请求上限 | 1,155 | 5,775 |
| HTTP 尝试上限（含重试） | 4,620 | 23,100 |
| 预计请求数 | 约 700–920 | 约 3,500–4,600 |
| 预计 GPU 时间（aml-b GPU4，3 并发，按 DEV45 实测约 282 次/小时） | 约 2.5–3.3 小时，最坏约 4.1 小时 | 约 13–17 小时，最坏约 21 小时 |
| 预计本地磁盘 | 约 1.8 GB | 约 9 GB |

- 每个区块单独申请一次 6 小时租约，结束即释放，不续租、不自动重启。
- 每个槽位只尝试一次；无效运行如实保留，不补跑、不换样本。
- 本地模型，没有 API 费用；占用的是共享 AML 服务器的 1 张 GPU。

## 3. 预先登记的分析与决策

分析脚本：`scripts/analyze_acceptance_gap_exp1.py`，不发任何模型请求。阈值都写在冻结的 spec 里，脚本只负责读取。

- 终点：study-outcome-v1 的 `study_outcome_y`，即官方 completion_rate（0–1）。每条有效运行都从封存的 L1 包重新推导一次，必须与区块收据一致。
- 估计：对 8 个任务等权；先算每格均值，再做任务内差值，最后对任务求平均。区间用两阶段 bootstrap（先抽任务，再在格内抽运行），10,000 次，种子 20261022，95% 区间。
- 决策规则：

| 结论 | 条件 |
|---|---|
| 中止（无效果） | 跑完至少 3 个区块后，F−O 的 95% 上界 < 0.10 |
| 不成立：信息不是瓶颈 | 5 个区块后，F−O < 0.15，或其 95% 下界 ≤ 0 |
| 不成立：现有做法已够 | F−O 成立，且 max(T−O, V−O) ≥ ⅔·(F−O) |
| 部分成立：交用户决定 | F−O 成立，且 ⅓·(F−O) ≤ max(T−O, V−O) < ⅔·(F−O) |
| 进入方法开发 | F−O 成立，且 max(T−O, V−O) < ⅓·(F−O) |
| 数据不足 | 5 个区块后，主分析 8 个任务中有 O 或 F 格少于 4 条有效运行 |

- F−O 成立的判据用点估计加 bootstrap 下界；T、V 的比例规则用点估计，T 和 V 中取点估计较大者。
- 另报四项描述性结果，不参与决策：
  - 同期 O 与 DEV45 O 的漂移，差值超过 0.10 时标记；
  - 对 O 下大多做对的评分项（O 通过率 ≥ 0.6），F/T/V 是否把它们改坏；
  - 按信息来源（展示的、默认的、写明的、交付门槛）拆分的扣分；
  - 各条件的平均请求数。
- 运营层面：一个区块里如果超过 20% 的运行无效，下一个区块开跑前先停下诊断，不改冻结条件继续跑。

## 4. 文件

| 文件 | 作用 |
|---|---|
| `configs/acceptance_gap_exp1_20261022.json` | 冻结 spec：条件、任务、区块、上限、分析阈值、决策规则（SHA-256 `f5aee3b48c8a2d469a6543d20d32554349208a05a9149382f1791b2521661335`） |
| `scripts/run_acceptance_gap_exp1.py` | 驱动器：生成预览、生成封存的区块计划、校验计划、执行一个区块（需 `--execute`） |
| `scripts/supervise_acceptance_gap_exp1.py` | 监督器：复用 R22/DEV45 的 GPU 租约监督器，负责容器归属标签和服务释放 |
| `scripts/launch_acceptance_gap_block.py` | 操作助手：prepare / start-service / launch / status |
| `scripts/analyze_acceptance_gap_exp1.py` | 预先登记的分析 |
| `tests/test_acceptance_gap_exp1.py` | 36 项测试，全部使用替身，不连任何服务 |
| `docs/acceptance_gap_exp1_20261022/preview.json` | 待审阅的条件文字、完整系统提示、5 个区块的顺序 |

复用而未修改的组件：`run_prospective_dev45.py`（冻结 π 校验、服务身份、租约、单槽封签记录）、`prospective_execution_observer.py`（容器隔离检查、原生校验器身份）、`supervise_r22_native_alignment.py`、`finalize_d0_timeout600.py`、`start_owned_native_service.py`。

## 5. 已验证与未验证

已验证（离线，本机）：

- 新增 36 项测试全部通过；与复用组件相关的既有测试一起跑，共 176 项通过。
- L2 全套 1,299 项通过、1 项跳过。另有 5 项失败，都在 `test_prospective_provenance.py`，原因是这些测试只能在 Linux 服务器上运行（脚本里有平台检查）；本次没有改动这些文件。
- 在真实冻结输入上：
  - 参考 π 校验通过：DEV45 冻结记录绑定的 131 个文件无变化，原生校验器启动身份一致，Actor 镜像检查一致；
  - O 的 9 个任务配置与 DEV45 冻结哈希相同；
  - 用临时授权和租约生成区块 0 的封存计划并重新校验，都通过；改动配置后重新封签的计划会被拒绝。临时文件已删除。

未验证（要到真正运行时才知道）：

- aml-b 上的服务能否启动、隧道是否可用、远端启动脚本与本地是否一致（start-service 会先核对哈希再启动）。
- F/T/V 的实际效果、运行时长、无效率。

已知偏差：

- DEV45 冻结记录里的控制器源码锁已经失效。DEV45 之后改过 3 个执行路径文件：`orchestrator.py` 的跨模型和 OpenCode 输出截断分支，只对 opencode-chat 生效；`opencode.py` 在本配置下不使用；`pi_native_validation.py` 修了出错时的管道清理。本计划改为对当前源码重新取指纹，并用漂移检查衡量实际影响。
- 单条运行的墙钟预算剩余不足一次 600 秒请求时，DEV45 驱动会停掉整批；本驱动只结束这一条，记为 `wall_time_budget`。DEV45 中从未触发过。
- F 文本来自自动构建、未经人工审核的要求清单（用户已豁免人工审核）。它只用来测信息上限，不代表可部署的方法。

## 6. 需要你确认的内容

1. 同意这个设计和上述决策规则，或者指出要改的地方。改动后 spec 的哈希和 `review_digest` 都会变，需要重新生成预览。
2. 同意按区块使用 aml-b GPU4。每个区块一次 6 小时租约，最多 5 次。
3. 确认后，我按下面的模板写授权记录，`exact_text` 填你的原话：

```json
{
  "exact_text": "<你的原话>",
  "source": "direct user message in Kiro session, 2026-10-22",
  "experiment_id": "acceptance_gap_exp1_20261022",
  "spec_sha256": "f5aee3b48c8a2d469a6543d20d32554349208a05a9149382f1791b2521661335",
  "approved_review_digest": "1ca1856f3fcd59dbb656ad7ba2861e8d481af62e5ef06b99f7e4c855ebacce90",
  "authorized_blocks": [0, 1, 2, 3, 4]
}
```

保存位置：`L2_Behavioral Event Localization/L2_Sandbox/validation/acceptance_gap_exp1_20261022/authorization.json`（该目录被 Git 忽略）。

## 7. 运行步骤（每个区块）

以下命令都在仓库根目录执行，`N` 为区块号（0–4）。

```bash
PY=L1_sandbox/C3_Experimental_Sandbox/.venv/bin/python
S="L2_Behavioral Event Localization/L2_Sandbox/scripts"
AUTH="L2_Behavioral Event Localization/L2_Sandbox/validation/acceptance_gap_exp1_20261022/authorization.json"

# 0. 前置检查：GPU4 空闲，jiasheren2 总共不超过 2 张 GPU、64 个 CPU 线程；本地剩余空间至少 10 GB
ssh aml-b nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
df -h .

# 1. 准备：写租约（6 小时计时从这里开始）、生成封存计划、写区块批准记录
$PY "$S/launch_acceptance_gap_block.py" prepare --block N --authorization "$AUTH"

# 2. 在 aml-b GPU4 上启动冻结的 vLLM 服务（会先核对远端启动脚本的哈希）
$PY "$S/launch_acceptance_gap_block.py" start-service --block N --execute

# 3. 建立本地隧道 127.0.0.1:8001 → aml-b:8001
bash scripts/start_tunnel.sh

# 4. 等服务就绪、预检，然后在后台启动监督器
$PY "$S/launch_acceptance_gap_block.py" launch --block N --execute

# 5. 查看进度
$PY "$S/launch_acceptance_gap_block.py" status --block N
```

- 第 1–4 步要连续完成：6 小时租约从 prepare 开始计时，start-service 和 launch 都会检查剩余时间。
- 监督器在区块结束或租约到期时自动清理本区块的容器并释放 GPU 服务；结果在 `validation/acceptance_gap_exp1_20261022/block_NN/` 下。
- 第 4 步如果因为隧道没起来而超时，服务不会被关掉：修好隧道后重新执行第 4 步即可。预检失败则会立即释放服务。
- 跑完 3 个区块后先做一次中期分析，检查中止条件；5 个区块全部跑完后做最终分析：

```bash
$PY "$S/analyze_acceptance_gap_exp1.py" \
  --blocks "L2_Behavioral Event Localization/L2_Sandbox/validation/acceptance_gap_exp1_20261022/block_00" \
           "L2_Behavioral Event Localization/L2_Sandbox/validation/acceptance_gap_exp1_20261022/block_01" \
           "L2_Behavioral Event Localization/L2_Sandbox/validation/acceptance_gap_exp1_20261022/block_02" \
  --output "L2_Behavioral Event Localization/L2_Sandbox/validation/acceptance_gap_exp1_20261022/analysis_interim"
```

## 8. 结果之后

- 进入方法开发：在开发任务上实现验收编译（样例测量、惯例库、收尾门控、回归保护）。10 个未暴露的留出家族继续保持不碰。
- 部分成立：由你决定是否继续。继续的话，方法必须胜过 T 和 V 中较好的那个。
- 不成立：如实报告。对这个 π，第一篇不走"写不出来的验收标准"这条线，回到方向文档重新选择。
