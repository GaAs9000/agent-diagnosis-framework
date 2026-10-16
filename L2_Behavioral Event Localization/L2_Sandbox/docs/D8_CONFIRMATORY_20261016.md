# D8 方案 ② 定向确证与 Call 2 机制假说验证报告（2026-10-16）

- **批次 ID**：`d8_confirmatory_20261016`
- **产物目录**：[`validation/d8_confirmatory_20261016/`](../validation/d8_confirmatory_20261016/)
- **冻结计划**：[`plan.json`](../validation/d8_confirmatory_20261016/plan.json)（`plan_sha256 = a98d54410a98bd5432713075c8495c572e08890861f336777546bba8d3d00dfc`）
- **汇总与账本**：[`summary.json`](../validation/d8_confirmatory_20261016/summary.json)、[`accounting.json`](../validation/d8_confirmatory_20261016/accounting.json)、[`analyst_http_receipts.jsonl`](../validation/d8_confirmatory_20261016/analyst_http_receipts.jsonl)
- **被测主模型（Actor）**：`opencode/deepseek-v4.1-flash`（35 请求 / 140 工具 / 10800 秒，与 `dev45` 基线完全同配置）
- **分析器模型（Analyst）**：`rules@1`（A 组）、`llm_raw:gemini-3.8-flash@8ca2e57d160d6772`（B 组）、`llm:gemini-3.8-flash@7860654c47132cd1`（C 组）

---

## 1. 实验目的与预注册边界（Scope & Pre-Specification）

在开发侧小规模试跑 `m3_dev_mini_20261015`（`N1=4, N2=8`）中，`officeval_070.requests_035.r04` 与 `officeval_060.requests_035.r03` 在 `turn = 29`（即检查点 `c28 -> c29`）均观测到约 `+0.38` 的点估计下跌，但受限于 `N2=8` 的样本量与多重比较保守下界，全轨迹扫描均判为 `diffuse_loss`，从而未触发 M4 的 **Call 2（定位后机制假说与谓词分组检验）**。

根据 `D8` 决议（先执行方案 ②，再决定方案 ①），本批次执行**预指定单点双样本确证检验（Pre-specified Single Two-Sample Confirmatory Test）+ Call 2 端到端评估**：

1. **零旧分支复用（Zero Pilot Reuse）**：在 `officeval_070.requests_035.r04` 与 `officeval_060.requests_035.r03` 的 `c28`（`k=28`）与 `c29`（`k=29`）各全新独立采样 **`N = 24`** 条续跑分支（每条轨迹 `48` 条新分支，两轨迹共 **`96` 条新分支**，`reused = false`）。
2. **预指定单侧检验规则**：检验单一预指定比较 `V(c28) - V(c29)`，采用单侧 $\alpha = 0.05$（95% 置信下界 `lower_drop = min(bootstrap_lb_95, newcombe_lb_95)`），预指定确证阈值为 **`0.10`**。若 `lower_drop > 0.10`，确证 `concentrated_turn`（`turn = 29`），否则判为 `diffuse_loss`。
3. **Call 1 冻结复用与 Call 2 在线触发**：直接绑定 `m3_dev_mini_20261015` 在看真值前已冻结的 Call 1 预测文件（0 次 Call 1 重调）；仅对确证为 `concentrated_turn` 的轨迹在线触发 A/B/C 三组分析器的 Call 2 机制假说生成，并在 `c28` 的 `24` 条新分支上按第 29 轮行为谓词 $P(b)$ 与 $\neg P(b)$ 做双样本分组检验（Holm 校正 $\alpha = 0.05$，且要求 $n_P \ge 4, n_{\neg P} \ge 4$）。
4. **严格口径声明（`development_only`）**：候选轮次 `turn = 29` 由开发侧小跑 `m3_dev_mini_20261015` 提名；尽管确证检验完全使用全新独立采样的 `24 + 24` 条分支（消除了同批数据的胜者诅咒），本实验仍属于**开发侧定向确证实验**，不替代全轨迹盲扫的 Stage 1 + Stage 2 正式定位功效声明，也不替代留出侧（Holdout）最终结论。

---

## 2. 执行与计费账本（Execution & Accounting）

全部 `96` 条新分支均一次跑通并通过 `c3_sandbox.l1.package.verify_package` 完整性校验（`0` 条无效尝试，`invalid_rate = 0.0`）：

| 维度 | `officeval_070.requests_035.r04` | `officeval_060.requests_035.r03` | 合计 |
| :--- | :---: | :---: | :---: |
| 计划 / 实际有效新分支数 | `48 / 48`（`c28: 24, c29: 24`） | `48 / 48`（`c28: 24, c29: 24`） | **`96 / 96`**（`0` invalid） |
| 逻辑模型请求数（上限 / 实际） | `312 / 312`（`24×7 + 24×6`） | `312 / 312`（`24×7 + 24×6`） | **`624 / 624`** |
| Actor HTTP 尝试数 / 状态码 | `312`（`200: 312`，0 重试） | `312`（`200: 312`，0 重试） | **`624`**（100% HTTP 200） |
| Actor Prompt Tokens | `8,316,200` | `7,623,590` | **`15,939,790`** |
| Actor Completion Tokens | `149,355` | `111,203` | **`260,558`** |
| Actor Known Total Tokens | `8,465,555` | `7,734,793` | **`16,200,348`** |
| 分支终止原因（`request_budget` 占比） | `c28: 100% (24/24)`，`c29: 100% (24/24)` | `c28: 100% (24/24)`，`c29: 100% (24/24)` | **100% (`96/96`)** |
| Analyst Call 2 触发与 HTTP 账本 | 触发 B/C 组各 1 次落盘收据（共 `86,337` tokens）* | 未触发（确证为 `diffuse_loss`） | **2 条落盘收据 + 1 次修补前请求*** |

> *注：在首次触发 `070.r04` B 组 Call 2 时，审计包装器 `make_auditable_llm_caller` 因对 `resp.body`（`bytes`）直接调用 JSON `digest()` 抛出 `TypeError`，修复为 `hashlib.sha256(resp.body).hexdigest()` 后重跑成功并落盘 2 条完整收据（B 组 `26,388` tokens，HTTP 200，`18.29s`；C 组 `59,949` tokens，HTTP 200，`17.64s`）。该次未落盘的修补前 HTTP 调用已在 [`accounting.json`](../validation/d8_confirmatory_20261016/accounting.json) 中显式入账（`unpersisted_prior_analyst_http_attempts = 1`）。

---

## 3. 定向确证检验结果（`c28` vs `c29`，`N = 24` 全新分支）

### 3.1 核心对比总表

| 轨迹 Slot | 基线 $p_0$ | 小跑 `N2=8` $\hat{V}(c_{28}) \to \hat{V}(c_{29})$（$\hat{d}_{29}$） | 确证 `N=24` $\hat{V}_2(c_{28}) \to \hat{V}_2(c_{29})$ | 确证跌落 $\hat{d}_{29}$ | Bootstrap 95% 下界 | Newcombe 95% 下界 | 主口径 `lower_drop`（阈值 `0.10`） | 单侧置换检验 $p$ 值 | 确证结论 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [`officeval_070.requests_035.r04`](../validation/d8_confirmatory_20261016/officeval_070.requests_035.r04/curve.json) | `0.8269` | `0.5000 -> 0.1154`（`+0.3846`） | **`0.5288 -> 0.1186`** | **`+0.4103`** | `0.2628` | `0.1906` | **`0.1906 > 0.10`** | **`0.0002`** | **`concentrated_turn` (`turn = 29`)** |
| [`officeval_060.requests_035.r03`](../validation/d8_confirmatory_20261016/officeval_060.requests_035.r03/curve.json) | `0.6591` | `0.3750 -> 0.0000`（`+0.3750`） | **`0.2917 -> 0.3144`** | **`-0.0227`** | `-0.2500` | `-0.2320` | **`-0.2500 ≤ 0.10`** | `0.6237` | **`diffuse_loss` (`turn = null`)** |

### 3.2 分支得分分布与统计诊断

1. **`officeval_070.requests_035.r04`（真实单轮集中崩塌得到强确证）**：
   - `c28`（24 条新分支）：均值 **`0.528846`**，95% 双侧 Wilson 区间 `[0.3376, 0.7120]`。其中 `18/24`（75.0%）取得正分：`3` 条 `1.0000`、`8` 条 `0.7692`、`7` 条 `0.5385`、`6` 条 `0.0000`。
   - `c29`（24 条新分支）：均值 **`0.118590`**，95% 双侧 Wilson 区间 `[0.0398, 0.3036]`。其中仅 `4/24`（16.7%）取得正分：`3` 条 `0.7692`、`1` 条 `0.5385`、`20` 条 `0.0000`。
   - **高度可复现性**：24+24 全新独立分支测得的点估计（`0.5288 -> 0.1186`，$\hat{d}_{29} = +0.4103$）与小跑 8+8 分支（`0.5000 -> 0.1154`，$\hat{d}_{29} = +0.3846`）几乎完全一致；主口径下界 `0.1906` 显著高于 `0.10` 阈值，单侧置换检验 $p = 0.0002$。
2. **`officeval_060.requests_035.r03`（成功识别并剔除 `N2=8` 小样本胜者诅咒）**：
   - `c28`（24 条新分支）：均值 **`0.291667`**（`7/24` 得 `1.0000`，`17/24` 得 `0.0000`）。
   - `c29`（24 条新分支）：均值 **`0.314394`**（`7/24` 得 `1.0000`，`1/24` 得 `0.5455`，`16/24` 得 `0.0000`）。
   - **方法学意义**：在真实续跑成功率约为 `0.30` 的检查点上，小跑 `N2=8` 抽到 `0/8` 全零的概率为 $(1-0.30)^8 \approx 5.76\%$。小跑中 `060.r03` 在 `c29` 恰好抽到 `0/8`（并在 `c30` 反弹至 `0.1364`），形成了表观 `+0.3750` 的假性尖峰；而 `N=24` 全新独立确证彻底消除了这一小样本均值回归假象，证明 `060.r03` 在 `turn = 29` 并未发生能力崩塌（其真正跌落发生在更晚的 `c29 -> c32` 区间，小跑中 `c31 = 0.0682`、`c32 = 0.0000`）。这直接印证了在候选轮次上用 `N=24` 独立样本做确证的必要性。

---

## 4. Call 1 定位指标更新与 Call 2 机制假说验证（`070.r04`）

### 4.1 `070.r04` 确证 `turn = 29` 后的 Call 1 定位评估

直接复用 `m3_dev_mini_20261015` 冻结的 Call 1 预测（参见 [`070.r04/scores.json`](../validation/d8_confirmatory_20261016/officeval_070.requests_035.r04/scores.json)）：

| 分析器组别 | Call 1 预测 `top_drop_turns` | `exact_hit`（真值 `turn=29`） | `hit_at_1` | `hit_at_3` | 小跑全曲线 `value_spearman` / `drop_spearman` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A 组（`rules@1`）** | `[]` | `0` | `0.0` | `0.0` | `0.0000` / `0.0000` |
| **B 组（`llm_raw:gemini-3.8-flash`）** | **`[29, 32, 31]`** | **`1`** | **`1.0`** | **`1.0`** | **`0.8668`** / **`0.4648`**（确证两点相关性 `1.0000`） |
| **C 组（`llm:gemini-3.8-flash`）** | `[31, 32, 26]` | `0` | `0.0` | `0.0` | `0.8501` / `0.3401`（确证两点相关性 `1.0000`） |

- **B 组在仅看父轨迹、未见任何续跑真值的情况下，将 `turn = 29` 排在 Top-1 跌落轮次首位（预测跌落 `0.25`）**，并在独立确证后实现 `exact_hit = 1`、`hit_at_1 = 1.0`。

### 4.2 Call 2 机制假说与谓词分组检验全景对比（`070.r04`，`c28` 24 条新分支）

在 `070.r04` 确证 `turn = 29`（$\hat{d}_{29} = +0.4103$）后，系统对 A/B/C 三组执行 Call 2 机制假说生成与 `c28`（$N=24$）分支分组验证：

| 分析器组别 | 假说序号 | 候选谓词 $P(b)$ 摘要（作用轮次 `turn = 29`） | 父轨迹一致性 | 分组样本量 $(n_P, n_{\neg P})$ | 组内均值 $(\bar{Y}_P, \bar{Y}_{\neg P})$ | 效应量 $\hat{\Delta}_P = \bar{Y}_P - \bar{Y}_{\neg P}$ | 单侧置换 $p$ 值（Holm 校正后） | 解释份额 `explained_share` | 判定状态与证据等级 |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A 组（`rules@1`）** | — | 无（5 项固定规则词表在父轨迹 `turn=29` 均不成立） | — | — | — | — | — | `0.0000` | `0` 提出 / `0` 确证（`explanation_covered = false`） |
| **B 组（`llm_raw`）** | H1 | `tool_requested(name="run", command="*darkcols*")` | `true` | `(0, 24)` | `(null, 0.5288)` | `null` | `null` | `null` | `untestable`（`insufficient_groups`） |
| **B 组（`llm_raw`）** | **H2** | **`tool_requested(name="run", command="*out/pg-1.png*4498848*")`** | **`true`** | **`(5, 19)`** | **`(0.2615, 0.5992)`** | **`-0.3377`** | **`0.0446` (`0.0446`)** | **`0.6516`** | **`confirmed` (`E1+`, `explanation_covered = true`)** |
| **B 组（`llm_raw`）** | H3 | `tool_exit(ok=true, stdout_preview="*[] None*")` | `false`* | `(0, 24)` | `(null, 0.5288)` | `null` | `null` | `null` | `inconsistent`（`parent_predicate_false`） |
| **C 组（`llm_struct`）** | H1 | `tool_requested(name="run", command="*darkcols*")` | `true` | `(0, 24)` | `(null, 0.5288)` | `null` | `null` | `null` | `untestable`（`insufficient_groups`） |
| **C 组（`llm_struct`）** | **H2** | **`tool_requested(run, "*open('out/pg-1.png')*") AND tool_requested(run, "*import numpy as np*")`** | **`true`** | **`(7, 17)`** | **`(0.4396, 0.5656)`** | **`-0.1261`** | `0.2162` (`0.2162`) | `0.2176` | `not_confirmed`（方向正确但未达显著） |
| **C 组（`llm_struct`）** | H3 | `model_output_text(text="*measure the actual rendered pixel positions*")` | `true` | `(0, 24)` | `(null, 0.5288)` | `null` | `null` | `null` | `untestable`（`insufficient_groups`） |

### 4.3 机制解释的实质发现与工程启示

1. **为什么规则基线（A 组）在 `070.r04` 完全失明**：
   - 父轨迹在第 29 轮执行的是一段语法完全合法、返回码 `exit_code = 0` 的内联 Python 脚本（扫描 `out/pg-1.png` 的像素列并与硬编码的 EMU 坐标 `4498848` 比对）。因此 `tool_error`、`nonzero_exit`、`Gate-1/2`、`repeated_identical_call`、`Premature finish` 五条规则谓词全部为 `false`。这展示了纯语法/错误码规则无法捕捉**“在预算尾声陷入无进展的像素测量死循环”**这一语义级行为退化。
2. **B 组 H2 获 `E1+` 确证且覆盖主跌落的 65.2%（`explained_share = 0.6516`）**：
   - 从 `c28` 分叉出的 24 条新分支中，有 $n_P = 5$ 条分支在第 29 轮重复了父轨迹的机制模式（调用 `run` 对 `out/pg-1.png` 与坐标 `4498848` 做逆向像素测算），其最终平均得分仅为 **`0.2615`**；而另外 $n_{\neg P} = 19$ 条直接进入 PPTX 形状修改或采用其他路径的分支，平均得分高达 **`0.5992`**。
   - 两组差异 $\hat{\Delta}_P = -0.3377$（单侧置换检验 $p = 0.0446 \le 0.05$），解释了 `c28 -> c29` 总跌落（`+0.4103`）中的 **`65.16%`**（超过 `0.50` 门槛，`explanation_covered = true`）。
3. **C 组 H2 的粒度对比**：
   - C 组 H2 提出了更宽泛的谓词（在 `run` 中同时包含 `*open('out/pg-1.png')*` 与 `*import numpy as np*`），在 24 条分支上实现了更均衡的分组 $(n_P = 7, n_{\neg P} = 17)$，且方向同样为负（$\hat{\Delta}_P = -0.1261$），但因为部分成功分支在修改完 PPTX 后也会用 `numpy` 打开 `out/pg-1.png` 做快速视觉验收，把“事后验收看图”与“修改前逆向像素测算”混在同一谓词中稀释了效应量（$p = 0.2162$）。
4. **进入正式 M4 前的两项明确工程改进点**：
   - **Call 2 提示词显式列出 `turn_facts` 字段白名单**：B 组 H3 使用了 `tool_exit.where.stdout_preview`，但 L2 的 `turn_facts` 中 `tool_exit` 仅暴露 `ok, exit_code, timed_out, error, stderr_preview`（不含 `stdout_preview`），导致父轨迹自检判为 `inconsistent`。在正式跑 M4 前，应在 Call 2 提示词中明确写出每个 `fact` 支持的字段名。
   - **Call 2 提示词禁止匹配父轨迹偶然变量名或原话长短语**：B 组 H1 与 C 组 H1 匹配了父轨迹临时取的 Python 变量名 `*darkcols*`，C 组 H3 匹配了父轨迹思维链原话 `*measure the actual rendered pixel positions*`，在独立重采样的兄弟分支上必然出现 $n_P = 0$（`untestable: insufficient_groups`）。提示词应明确要求分析器使用**跨分支可复现的工具/文件名/命令结构谓词**，避免绑定偶发变量名。

---

## 5. 结论与下一步衔接（按用户指令暂停）

1. **`D8` 方案 ② 已圆满闭环**：
   - 在 `officeval_070.requests_035.r04` 上，以 `24 + 24` 条全新分支确证了 `turn = 29` 的单轮集中价值崩塌（$\hat{d}_{29} = +0.4103$，`lower_drop = 0.1906 > 0.10`，$p = 0.0002$），并首次在真实数据上跑通且确证了 Call 2 的 `E1+` 机制假说（B 组 `explained_share = 0.6516`）。
   - 在 `officeval_060.requests_035.r03` 上，以 `24 + 24` 条全新分支证明小跑 `N2=8` 在 `c29` 的 `0.0000` 属于小样本抽样波动（确证均值为 `0.2917 -> 0.3144`），验证了 `N=24` 确证阶段的防误报能力。
2. **当前状态**：遵照用户指令，完成 `D8` 方案 ② 的实验、账本核对与文档/Notion 同步后**立即暂停**，暂不启动 `D8` 方案 ①（正式全轨迹 `N1=12, N2=24, stride=4` 扩展运行）。
