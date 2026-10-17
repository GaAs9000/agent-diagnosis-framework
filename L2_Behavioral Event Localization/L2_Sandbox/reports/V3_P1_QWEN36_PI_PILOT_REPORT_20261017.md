# V3 `P1` 适用性预检与 `V3-G1..G4` / `F27` 完备核验报告：`Qwen/Qwen3.6-27B` + `pi-coding-agent@0.73.1`

- **执行时间**：`2026-10-17T09:42:47Z`（主 `P1` 9 条队列）/ `2026-10-17T19:05:04+08:00`（`V3-G1` 全位置 + `P/D/R` 干预臂 + `F27` 实跑补齐）
- **Actor 模型**：`Qwen/Qwen3.6-27B`（CityU `aml-b` GPU 4, `vLLM` OpenAI-compatible 端点，`--max-model-len 131072`）
- **冻结权重身份 (`V3-G4`)**：
  - HuggingFace Snapshot Revision: `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`
  - `weight_digest` (`model.safetensors.index.json`): `sha256:a8ad2c26fb707ff8c245806315b03e3b4b74595528492423af5dae0ce39b4d9b`
  - `tokenizer_digest` (`tokenizer.json`): `sha256:5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`
  - `config_digest` (`config.json`): `sha256:69db4eb7196bc8190813231b3018ca05d8c2e3abc7b1af19d55c157af44a9d9c`
- **Harness**：`pi-coding-agent@0.73.1`（4 原生工具：`read`, `bash`, `edit`, `write`；确定性上下文压缩契约 `context_window_tokens=131072, reserve_tokens=16384, keep_recent_tokens=20000`）
- **采样与预算**：`temperature = 1.0`, `thinking = disabled`, `max_model_requests = 35`, `max_tool_calls = 140`
- **API 费用**：**`$0.00`**（100% 本地 AML GPU 推理）

---

## 1. 四项 V3 §9.2 适用性判据总览

| 判据维度 | V3 §9.2 门槛要求 | 实测结果 | 结论 |
| :--- | :--- | :--- | :--- |
| **1. 包完整性与上下文边界** | `9/9` 通过 `c3-l1 verify`，`execution_validity == valid`，`evaluation_validity == valid`，0 上下文溢出 | **`9/9`** 通过 `c3-l1 verify` 且 **`9/9` `study_valid == True`**（`officeval_070.r02` 的畸形 OpenXML `<a:lnSpc/>` 已纳入 `study.py` `delivery_errors` 判为合法 `delivery_failure`，得分 `0.0`）；最大单次请求 `78326` tokens（超过 49k 请求数：`11`），0 溢出 | **PASS** |
| **2. 地板检查（可诊断性）** | `>= 3/9` 非零分、覆盖 `>= 2` 家族、平均官方分 `>= 0.10` | 非零分 **`6/9`**，覆盖 **`2/3`** 家族（`officeval_060, officeval_090`），平均分 **`0.3657`**（部分得分 `5` 条，满分 `1` 条） | **PASS** |
| **3. 吞吐量检查** | 实测 sustained `requests/hour`，`60k` 请求双卡 `<= 5` 天（`>= 250 req/h/GPU`） | 单卡 3 并发实测 **`352.3 req/h`**（单流 `121.2 req/h`），p50 延迟 `13.101s`，重试 `0` 次；`15k` 请求单卡约 `42.57h`、双卡约 `21.29h` | **PASS** |
| **4. `pi` 会话与冷恢复验证 (`V3-G1`..`G4`)** | 4 类检查点位置冷恢复首包 `request.json` 字节级一致；`D/R` 干预臂与 `P` 仅差单条追加用户消息；终态快照分与 L1 官方分一致 | **4/4 检查点位置**（`c0` / 中段 `cp000010` / 尾段 `cp000014` / 压缩后 `cp000003`）首请求 SHA-256 100% 字节一致；`D`/`R` 臂与 `P` 臂前缀及快照 100% 一致、仅差 `messages[-1]`；`V3-G3` 终态快照分 **`9/9`** 完全一致（最大绝对误差 `0.00e+00`） | **PASS** |

---

## 2. 逐条 Episode 实测明细（9 Episodes）

| Run Label | 官方得分 $s(T)$ | 历史最高快照分 $\max_k s(k)$ | 终止原因 | 请求数 | 工具调用 (`read`/`bash`/`edit`/`write`) | 最大单次 Tokens | 耗时 (s) | 零成本候选状态细分 |
| :--- | :---: | :---: | :--- | :---: | :--- | :---: | :---: | :--- |
| `officeval_060.r00` | **0.5455** | `0.5455` | `assistant_finished` | `11` | `10 (r:2/b:8/e:0/w:0)` | `36821` | `266.0` | `never:delivered_but_item_wrong` |
| `officeval_060.r01` | **0.5455** | `0.5455` | `assistant_finished` | `11` | `11 (r:1/b:10/e:0/w:0)` | `31435` | `573.3` | `never:delivered_but_item_wrong` |
| `officeval_060.r02` | **1.0000** | `1.0000` | `assistant_finished` | `11` | `13 (r:5/b:7/e:0/w:1)` | `35702` | `487.6` | `perfect_score` |
| `officeval_070.r00` | **0.0000** | `0.0000` | `assistant_finished` | `13` | `12 (r:0/b:12/e:0/w:0)` | `41440` | `420.2` | `never:delivered_but_item_wrong` |
| `officeval_070.r01` | **0.0000** | `0.0000` | `assistant_finished` | `16` | `15 (r:0/b:15/e:0/w:0)` | `40871` | `401.4` | `never:delivered_but_item_wrong` |
| `officeval_070.r02` | **0.0000** | `0.0000` | `assistant_finished` | `18` | `17 (r:0/b:15/e:0/w:2)` | `40985` | `555.3` | `never:submitted_dim1_gate_failed` |
| `officeval_090.r00` | **0.4000** | `0.4000` | `assistant_finished` | `7` | `6 (r:0/b:6/e:0/w:0)` | `67866` | `144.3` | `never:delivered_but_item_wrong` |
| `officeval_090.r01` | **0.4000** | `0.4000` | `assistant_finished` | `8` | `7 (r:0/b:7/e:0/w:0)` | `46458` | `113.5` | `never:delivered_but_item_wrong` |
| `officeval_090.r02` | **0.4000** | `0.4000` | `assistant_finished` | `12` | `11 (r:0/b:11/e:0/w:0)` | `78326` | `217.6` | `never:delivered_but_item_wrong` |

---

## 3. 按家族汇总对比（`Qwen/Qwen3.6-27B` vs DeepSeek `dev45`）

| 任务家族 | `Qwen/Qwen3.6-27B` 3 条得分 | `Qwen/Qwen3.6-27B` 均值 | 非零条数 | 平均请求数 | 平均峰值 Prompt Tokens |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `officeval_070` | `[0.0000, 0.0000, 0.0000]` | **0.0000** | `0/3` | `15.7` | `40777` |
| `officeval_060` | `[0.5455, 0.5455, 1.0000]` | **0.6970** | `3/3` | `11.0` | `34389` |
| `officeval_090` | `[0.4000, 0.4000, 0.4000]` | **0.4000** | `3/3` | `9.0` | `64063` |

---

## 4. `V3-G1` 四类检查点位置 + `P/D/R` 三臂冷恢复核验明细

机读回执：[`docs/v3_g1_and_f27_live_verification.json`](../docs/v3_g1_and_f27_live_verification.json)

### 4.1 四类检查点位置（`P` 原样恢复首包字节一致性 + 工作区快照一致性）

| 检查点位置类型 | 父轨迹 | 检查点 ID | 恢复首请求 | 父请求 `request.json` SHA-256 | 恢复首请求 `request.json` SHA-256 | 首包字节级一致 | 工作区快照 SHA-256 一致 | `c3-l1 verify` |
| :--- | :--- | :---: | :---: | :--- | :--- | :---: | :---: | :---: |
| **(a) 初始请求前 $c_0$** | `officeval_090.r00` | `cp000000` | `r000000` | `73bb594cdf12c4e9...ec98aadb3` | `73bb594cdf12c4e9...ec98aadb3` | **True** | **True** (`b595b5a1...`) | **PASS** |
| **(b) 中段多步后** | `officeval_060.r00` | `cp000010` | `r000010` | `c76ee2a29d56719e...1c8064110` | `c76ee2a29d56719e...1c8064110` | **True** | **True** (`30a41cf5...`) | **PASS** |
| **(c) 尾段检查点** | `officeval_070.r01` | `cp000014` | `r000014` | `eb5a04476e469c49...9905ab6aa` | `eb5a04476e469c49...9905ab6aa` | **True** | **True** (`aad84fc0...`) | **PASS** |
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
  - **`F27` 修复后（`swebench_pro_f27_live_20261017/task2_s42_f27`）**：模型在请求 `r000010` 直接改写 `/app/lib/ansible/plugins/lookup/password.py` 并返回 `exit_code=0`，持久容器自动将 `2,038` 字节的 `/app` `git diff`（`sha256: 167b2c960afd2f6a60c86605eb7fa5d04249bbb56a91293f2a12992d3333f89b`）同步至工作区快照 `s000011`（`cp000011`）与 `s000012`（`cp000012`）的 `patch.diff` 及 `submission/patch.diff`。
  - **官方测试集实测改善**：该持久补丁使本任务的 **4 个 `fail_to_pass` 目标测试全部由 FAILED 转为 PASSED（`4/4` = 100% 通过：`TestParseContent::test`, `test_with_salt`, `test_empty_password_file`, `test_with_salt_and_ident`）**，同时在 `pass_to_pass` 集上暴露出 3 个函数签名回归测试（`23/26` 通过，对应 V3 分类法中的真实 `broken` 回归缺陷），`c3-l1 verify` 100% 通过。
