# V3 `P1` 适用性预检与 `V3-G1` / `F27` 实测报告：`Qwen/Qwen3.6-27B` + `pi-coding-agent@0.73.1`

> **更正说明（2026-10-17，对应修订前提交 `0058bde`，执行项 `R1`）**：
> 1. **门禁状态分项标注（去掉“G1–G4 全通过”）**：本报告仅包含 `P1` 9 条适用性实跑、`V3-G1` 四位置冷恢复首包字节级回执及 `P1` 9 条终态快照与 L1 官方分自动比对；**不宣称 `V3-G1..G4` 全通过**（`V3-G2` $c_0$ 20 对等价性、`V3-G3` 新 `dev45` 20 点人工核查、`V3-G4` 正常请求无诊断泄漏与缓存计时独立门禁均单列验收）。
> 2. **吞吐口径更正**：删除原“吞吐满足全套预算（PASS）”的定性结论，改为只报实测吞吐 **`352.3 req/h`（3 并发）／`121.2 req/h`（单流）**，并在 §1.1 给出 `10,000`、`20,000`、`40,000`、`60,000` 次总请求数下的预计墙钟时间表，供用户决定总请求预算与可接受墙钟上限。
> 3. **自愿结束（`assistant_finished`）与 `S` 臂依据明确化**：明确写出本批 **`9/9` 条运行全部为 `assistant_finished`（每条仅用 `7–18 / 35` 次请求），`0/9` 耗尽请求预算**；且 §4.1 的“尾段检查点 `cp000014`”明确注明来自一条总长 `16` 次请求的主动结束运行（`officeval_070.r01`），并非接近 35 请求预算上限的尾段。这是 V3 在干预阶梯第 1 级新增 **`S` 臂（结束前自查提示）** 及主动结束锚点规则的直接实证依据。
> 4. **`study.py` 规则版本（`v1.0` vs `v1.1`）并列报告**：对触发 `AttributeError: 'NoneType' object has no attribute 'val'` 的 `officeval_070` 运行（本批 `qwen36_pi_p1_pilot_20261017` 中为 `officeval_070.r02`，早期单跑探针中曾记为 `070.r00`），在 §1 与 §2 中并列列出 `study.py` 旧规则（`rule_version="v1.0"`：`unhealthy_or_ambiguous_official_error`, `study_valid=False`, `Y=None`）与新规则（`rule_version="v1.1"`，提交 `b8d9a39`：`valid_zero`, `study_valid=True`, `zero_reason="missing_or_damaged_delivery"`, `Y=0.0`）下的状态与均值。
> 5. **SWE `F27` 原件事实纠正**：按 `grades.jsonl` 原件如实记录 `task2_s42_f27` 在 `cp11 / cp12 / terminal` 均已有 `candidate=["submission/patch.diff"]`、`in_submission=true`，因 `pass_to_pass` `23/26`（3 项候选内回归）未满足官方 `dim1_pass` 全通过门槛而记 `dim1_pass=false, score=0.0`。

- **执行时间**：`2026-10-17T09:42:47Z`（主 `P1` 9 条队列）/ `2026-10-17T19:05:04+08:00`（`V3-G1` 四位置恢复 + `P/D/R` 前缀核验 + `F27` 实跑）
- **Actor 模型**：`Qwen/Qwen3.6-27B`（CityU `aml-b` GPU 4, `vLLM` OpenAI-compatible 端点，`--max-model-len 131072`）
- **冻结权重与配置摘要（`V3-G4` 前提项）**：
  - HuggingFace Snapshot Revision: `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`
  - `weight_digest` (`model.safetensors.index.json`): `sha256:a8ad2c26fb707ff8c245806315b03e3b4b74595528492423af5dae0ce39b4d9b`
  - `tokenizer_digest` (`tokenizer.json`): `sha256:5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`
  - `config_digest` (`config.json`): `sha256:69db4eb7196bc8190813231b3018ca05d8c2e3abc7b1af19d55c157af44a9d9c`
- **Harness**：`pi-coding-agent@0.73.1`（4 原生工具：`read`, `bash`, `edit`, `write`；确定性上下文压缩契约 `context_window_tokens=131072, reserve_tokens=16384, keep_recent_tokens=20000`）
- **采样与预算**：`temperature = 1.0`, `thinking = disabled`, `max_model_requests = 35`, `max_tool_calls = 140`
- **API 费用**：**`$0.00`**（100% 本地 AML GPU 推理）

---

## 1. V3 §9.2 适用性判据总览（分项状态）

| 判据维度 | V3 §9.2 检查内容 | 实测结果 | 状态 |
| :--- | :--- | :--- | :--- |
| **1. 包完整性与 `study.py` 版本边界** | `9/9` 通过 `c3-l1 verify`，`execution_validity == valid`，0 上下文溢出；并列报告 `study.py` `v1.0` 与 `v1.1` | **`9/9` 通过 `c3-l1 verify`**；按 **`study.py v1.1`** 为 **`9/9` `study_valid == True`**（`070.r02` / 早期探针 `070.r00` 的畸形 OpenXML `<a:lnSpc/>` `AttributeError` 记 `valid_zero`），按 **`study.py v1.0`** 为 **`8/9` `study_valid == True` + `1/9` `unhealthy_or_ambiguous_official_error`**；最大单次请求 `78,326` tokens（超 49k 请求数：`11`），0 溢出 | **`v1.1`: `9/9` 有效 / `v1.0`: `8/9` 有效** |
| **2. 地板检查（可诊断性）** | `>= 3/9` 非零分、覆盖 `>= 2` 家族、平均官方分 `>= 0.10` | 非零分 **`6/9`**，覆盖 **`2/3`** 家族（`officeval_060, officeval_090`）；平均分按 `v1.1`（$N=9$）为 **`0.3657`**，按 `v1.0`（$N=8$）为 **`0.4114`**（部分得分 `5` 条，满分 `1` 条） | **自动核验通过** |
| **3. 实测吞吐量（不含预设达标判断）** | 实测单流与并发 `requests/hour`，并给出不同总请求预算下的预计墙钟时间表 | **3 并发实测 `352.3 req/h`**，**单流实测 `121.2 req/h`**（p50 延迟 `13.101s`，重试 `0` 次；不同总请求预算的墙钟预计见 §1.1） | **仅报实测（待用户定总预算）** |
| **4. 终止方式与 `S` 臂实证依据** | 统计 `assistant_finished` vs `request_budget` 比例 | **`9/9`（`100%`）均为 `assistant_finished` 主动结束**，每条仅用 `7–18 / 35` 次请求，**`0/9` 耗尽请求预算**；直接支撑 V3 第 1 级新增 **`S` 臂（结束前自查）** | **实测确认** |
| **5. `pi` 会话与冷恢复验证 (`V3-G1` 及 P1 终态自动比对)** | 4 类检查点位置冷恢复首包 `request.json` 字节级一致；干预臂与 `P` 仅差单条追加用户消息 | **`V3-G1` 4/4 检查点位置**（`c0` / 中段 `cp000010` / 16 请求运行尾段 `cp000014` / 压缩后 `cp000003`）首请求 SHA-256 100% 字节一致；`D`/`R` 臂与 `P` 臂前缀及快照 100% 一致、仅差 `messages[-1]`；P1 9 条终态快照分与 L1 官方分自动比对 `9/9` 一致（注：不替代 `V3-G2`、`V3-G3` 人工核查与 `V3-G4` 无泄漏核验） | **`V3-G1` 自动核验通过（G2/G3/G4 另列）** |

### 1.1 不同总请求数下的预计墙钟时间表

按本批实测吞吐 **`352.3 req/h`（3 并发）** 与 **`121.2 req/h`（单流）** 折算：

| 计划总 Actor 请求数 | 3 并发预计墙钟时间 (`352.3 req/h`) | 单流预计墙钟时间 (`121.2 req/h`) |
| ---: | ---: | ---: |
| **`10,000` 次请求** | **`28.38 小时`**（约 `1.18 天`） | `82.51 小时`（约 `3.44 天`） |
| **`20,000` 次请求** | **`56.77 小时`**（约 `2.37 天`） | `165.02 小时`（约 `6.88 天`） |
| **`40,000` 次请求** | **`113.54 小时`**（约 `4.73 天`） | `330.03 小时`（约 `13.75 天`） |
| **`60,000` 次请求** | **`170.31 小时`**（约 `7.10 天`） | `495.05 小时`（约 `20.63 天`） |

---

## 2. 逐条 Episode 实测明细（9 Episodes，并列 `study.py v1.0` 与 `v1.1`）

> **关于 `9/9 assistant_finished` 的关键观察**：9 条运行全部在 `7–18` 次请求内由模型自己结束（`assistant_finished`），**没有任何一条把 35 次请求用完（`0/9` `request_budget`）**。因此，当模型交出缺项或损坏文档时，并非因为“没时间写”，而是“写完没有自查就提前收工”或“不知具体要求”。这是 V3 在第 1 级加入 **`S` 臂（结束前自查，与 `D` 同为不带任务专有信息的通用提示）** 的直接依据。

| Run Label | Episode ID | 官方得分 $s(T)$ (`v1.1`) | 历史最高快照分 $\max_k s(k)$ | 终止原因 | 请求数 (`/35`) | 工具调用 (`read`/`bash`/`edit`/`write`) | 最大单次 Tokens | 耗时 (s) | `study.py v1.0` 旧规则状态 | `study.py v1.1` 新规则状态 |
| :--- | :--- | :---: | :---: | :--- | :---: | :--- | :---: | :---: | :--- | :--- |
| `officeval_060.r00` | `ep_f0034d3c7c574e8384f232b9b46c078b` | **0.5455** | `0.5455` | `assistant_finished` | `11` | `10 (r:2/b:8/e:0/w:0)` | `36821` | `266.0` | `valid_score` (`Y=0.5455`) | `valid_score` (`Y=0.5455`) |
| `officeval_060.r01` | `ep_3ce6a89251be4b9eb1e3ab865b62b305` | **0.5455** | `0.5455` | `assistant_finished` | `11` | `11 (r:1/b:10/e:0/w:0)` | `31435` | `573.3` | `valid_score` (`Y=0.5455`) | `valid_score` (`Y=0.5455`) |
| `officeval_060.r02` | `ep_50ec2289e5a64dc7a0dc2346251db66f` | **1.0000** | `1.0000` | `assistant_finished` | `11` | `13 (r:5/b:7/e:0/w:1)` | `35702` | `487.6` | `valid_score` (`Y=1.0000`) | `valid_score` (`Y=1.0000`) |
| `officeval_070.r00` | `ep_2a67a56e65144a89b8d47451bd860163` | **0.0000** | `0.0000` | `assistant_finished` | `13` | `12 (r:0/b:12/e:0/w:0)` | `41440` | `420.2` | `valid_zero` (`official_zero`, `Y=0.0`) | `valid_zero` (`official_zero`, `Y=0.0`) |
| `officeval_070.r01` | `ep_cb46f1200f0047a298a2276a4a9d8420` | **0.0000** | `0.0000` | `assistant_finished` | `16` | `15 (r:0/b:15/e:0/w:0)` | `40871` | `401.4` | `valid_zero` (`official_zero`, `Y=0.0`) | `valid_zero` (`official_zero`, `Y=0.0`) |
| `officeval_070.r02` *(注：早期单跑探针中亦对应 `070.r00` `ep_a26d6a98...`)* | `ep_8756455129224a78a522a9fc734b86a8` | **0.0000** | `0.0000` | `assistant_finished` | `18` | `17 (r:0/b:15/e:0/w:2)` | `40985` | `555.3` | **`unhealthy_or_ambiguous_official_error` (`study_valid=False, Y=None`)** | **`valid_zero` (`missing_or_damaged_delivery, Y=0.0`)** |
| `officeval_090.r00` | `ep_6f01390425814b2189f4e22e95bf8966` | **0.4000** | `0.4000` | `assistant_finished` | `7` | `6 (r:0/b:6/e:0/w:0)` | `67866` | `144.3` | `valid_score` (`Y=0.4000`) | `valid_score` (`Y=0.4000`) |
| `officeval_090.r01` | `ep_b4546d232bc5498b88c0f24cb4ec22eb` | **0.4000** | `0.4000` | `assistant_finished` | `8` | `7 (r:0/b:7/e:0/w:0)` | `46458` | `113.5` | `valid_score` (`Y=0.4000`) | `valid_score` (`Y=0.4000`) |
| `officeval_090.r02` | `ep_5ff025ff91574077a4ff067e1c264981` | **0.4000** | `0.4000` | `assistant_finished` | `12` | `11 (r:0/b:11/e:0/w:0)` | `78326` | `217.6` | `valid_score` (`Y=0.4000`) | `valid_score` (`Y=0.4000`) |

---

## 3. 按家族汇总对比（`study.py v1.0` vs `v1.1`）

| 任务家族 | `Qwen/Qwen3.6-27B` 3 条得分 (`v1.1`) | `v1.1` 均值 ($N=9$) | `v1.0` 均值 ($N=8$) | 非零条数 | 平均请求数 | 耗尽预算条数 (`request_budget`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `officeval_070` | `[0.0000, 0.0000, 0.0000]` | **0.0000** (`3/3` valid) | **0.0000** (`2/3` valid, `1` invalid `AttributeError`) | `0/3` | `15.7` | `0/3` |
| `officeval_060` | `[0.5455, 0.5455, 1.0000]` | **0.6970** (`3/3` valid) | **0.6970** (`3/3` valid) | `3/3` | `11.0` | `0/3` |
| `officeval_090` | `[0.4000, 0.4000, 0.4000]` | **0.4000** (`3/3` valid) | **0.4000** (`3/3` valid) | `3/3` | `9.0` | `0/3` |
| **总计 / 总体均值** | — | **`0.3657`** (`9/9` valid) | **`0.4114`** (`8/9` valid) | **`6/9`** | **`11.9`** | **`0/9` (`9/9 assistant_finished`)** |

---

## 4. `V3-G1` 四类检查点位置 + `P/D/R` 三臂冷恢复核验明细

机读回执：[`docs/v3_g1_and_f27_live_verification.json`](../docs/v3_g1_and_f27_live_verification.json)

### 4.1 四类检查点位置（`P` 原样恢复首包字节一致性 + 工作区快照一致性）

| 检查点位置类型 | 父轨迹 | 检查点 ID | 恢复首请求 | 父请求 `request.json` SHA-256 | 恢复首请求 `request.json` SHA-256 | 首包字节级一致 | 工作区快照 SHA-256 一致 | `c3-l1 verify` |
| :--- | :--- | :---: | :---: | :--- | :--- | :---: | :---: | :---: |
| **(a) 初始请求前 $c_0$** | `officeval_090.r00`（总 `7` 次请求） | `cp000000` | `r000000` | `73bb594cdf12c4e9...ec98aadb3` | `73bb594cdf12c4e9...ec98aadb3` | **True** | **True** (`b595b5a1...`) | **PASS** |
| **(b) 中段多步后** | `officeval_060.r00`（总 `11` 次请求） | `cp000010` | `r000010` | `c76ee2a29d56719e...1c8064110` | `c76ee2a29d56719e...1c8064110` | **True** | **True** (`30a41cf5...`) | **PASS** |
| **(c) 主动结束前尾段检查点（来自总 `16` 请求运行，非 35 请求预算上限尾段）** | `officeval_070.r01`（总 `16` 次请求 `assistant_finished`） | `cp000014` | `r000014` | `eb5a04476e469c49...9905ab6aa` | `eb5a04476e469c49...9905ab6aa` | **True** | **True** (`aad84fc0...`) | **PASS** |
| **(d) 上下文压缩后 (`session.compacted`)** | `officeval_060_compaction_parent` | `cp000003` (`count=1`) | `r000003` | `044364f4b018c0d2...e378c9bc9` | `044364f4b018c0d2...e378c9bc9` | **True** | **True** (`4a4935a7...`) | **PASS** |

### 4.2 干预臂 `D`（交付提醒）与 `R`（规则提示）相对于 `P`（原样恢复）的差分核验（@ `officeval_070.r01` `cp000014` -> `r000014`）

| 干预臂 | 子轨迹 Episode ID | `messages[:-1]` 前缀 SHA-256（对照 `P` 臂 `messages`） | `model` / `tools` / `parameters` / `messages[:-1]` 与 `P` 完全一致 | 仅差末尾单条 `role="user"` 干预消息 (`messages[-1]`) | 动作前工作区快照与 `P` 完全一致 (`aad84fc0...`) | `c3-l1 verify` |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **`P_plain`** | `ep_ad96a33ff2a54585afe6bdd964135615` | `57392102e46be2d9647cd0d555cf7b224ba448ec512b57b1fe8cca936c1464f6` | Baseline | Baseline | **True** | **PASS** |
| **`D_deliver`** | `ep_49826e4e7a72443f9f02cab7ac7a9330` | `57392102e46be2d9647cd0d555cf7b224ba448ec512b57b1fe8cca936c1464f6` | **True** | **True** (`ba2d8f74...`) | **True** | **PASS** |
| **`R_rule`** | `ep_d72ca6d5d2884fd9862bcc0dbcd016d7` | `57392102e46be2d9647cd0d555cf7b224ba448ec512b57b1fe8cca936c1464f6` | **True** | **True** (`13d1af69...`) | **True** | **PASS** |

---

## 5. `F27` SWE-bench Pro V2 持久容器真实 `Qwen/Qwen3.6-27B` 实跑核验

- **核验任务与随机种子**：`instance_ansible__ansible-0fd88717c953b92ed8a50495d55e630eb5d59166-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5` (`task2_s42_f27`, `seed=42`, `max_model_requests=12`)
- **轨迹路径**：`L1_sandbox/C3_Experimental_Sandbox/runs/swebench_pro_f27_live_20261017/task2_s42_f27` (`ep_796ed3710a33448aad3aab214e076ff1`)
- **修复前后对比（同一任务、同一模型、同一 `seed=42`）**：
  - **`F27` 修复前（`swebench_pro_qwen36_cohort/run_task2_s42`）**：模型在第 7 步尝试修改 `/app/lib/ansible/plugins/lookup/password.py` 时因 `--read-only` 根文件系统与每步 `docker run --rm` 临时容器而失败，全轨迹补丁大小为 `0` 字节，`fail_to_pass` 4 个测试 `0/4` 通过。
  - **`F27` 修复后（`swebench_pro_f27_live_20261017/task2_s42_f27`）**：模型在请求 `r000010` 直接改写 `/app/lib/ansible/plugins/lookup/password.py` 并返回 `exit_code=0`，持久容器自动将 `2,038` 字节的 `/app` `git diff`（`sha256: 167b2c960afd2f6a60c86605eb7fa5d04249bbb56a91293f2a12992d3333f89b`）同步至工作区快照 `s000011`（`cp000011`）、`s000012`（`cp000012`）与终态的 `patch.diff` 及 `submission/patch.diff`。
  - **原件评分与事实记录（`grades.jsonl`）**：在 `cp000011`、`cp000012` 及 `terminal_submission` 上，`grades.jsonl` 均为 `candidate_paths = ["submission/patch.diff"]`、`in_submission = true`、`dim1_pass = false`、`score = 0.0`。具体测试分解为：**4 个 `fail_to_pass` 目标测试全部通过（`4/4` = 100% PASSED：`TestParseContent::test`, `test_with_salt`, `test_empty_password_file`, `test_with_salt_and_ident`）**，但 **`pass_to_pass` 集为 `23/26` 通过（出现 3 个回归：`test_encrypt`, `test_no_encrypt`, `test_only_a`）**，未达官方 `pass_to_pass` 100% 门槛（`dim1_pass = false`），故终态官方总分为 `0.0`；这 3 个回归单列为**候选内回归（`in_candidate_regression`）**。`c3-l1 verify` 100% 通过。
