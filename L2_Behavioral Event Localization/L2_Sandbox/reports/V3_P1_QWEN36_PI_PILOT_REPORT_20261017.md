# V3 `P1` 适用性预检报告：`Qwen/Qwen3.6-27B` + `pi-coding-agent@0.73.1`（9 条 OfficeVal）

- **执行时间**：`2026-10-17T09:42:47Z`
- **Actor 模型**：`Qwen/Qwen3.6-27B`（CityU `aml-b` GPU 4, `vLLM` OpenAI-compatible 端点，`--max-model-len 131072`）
- **Harness**：`pi-coding-agent@0.73.1`（4 原生工具：`read`, `bash`, `edit`, `write`）
- **采样与预算**：`temperature = 1.0`, `thinking = disabled`, `max_model_requests = 35`, `max_tool_calls = 140`
- **API 费用**：**`$0.00`**（100% 本地 AML GPU 推理）

## 1. 四项 V3 §9.2 适用性判据总览

| 判据维度 | V3 §9.2 门槛要求 | 实测结果 | 结论 |
| :--- | :--- | :--- | :--- |
| **1. 包完整性与上下文边界** | `9/9` 通过 `c3-l1 verify`，`execution_validity == valid`，0 上下文溢出 | **`9/9`** 通过 `c3-l1 verify`（原生评分状态：`8/9 valid + 1/9 official_error (malformed PPTX <a:lnSpc/> -> score 0.0)`），最大单次请求 `78326` tokens（超过 49k 请求数：`11`），0 溢出 | **PASS** |
| **2. 地板检查（可诊断性）** | `>= 3/9` 非零分、覆盖 `>= 2` 家族、平均官方分 `>= 0.10` | 非零分 **`6/9`**，覆盖 **`2/3`** 家族（`officeval_060, officeval_090`），平均分 **`0.3657`**（部分得分 `5` 条，满分 `1` 条） | **PASS** |
| **3. 吞吐量检查** | 实测 sustained `requests/hour`，`60k` 请求双卡 `<= 5` 天（`>= 250 req/h/GPU`） | 单卡 3 并发实测 **`352.3 req/h`**（单流 `121.2 req/h`），p50 延迟 `13.101s`，重试 `0` 次；`15k` 请求单卡约 `42.57h`、双卡约 `21.29h` | **PASS** |
| **4. `pi` 会话与冷恢复验证 (`V3-G1`/`V3-G3`)** | 冷恢复首请求 `request.json` 字节级一致，快照终态分与 L1 官方分一致 | `V3-G1` 首请求 SHA-256 100% 字节一致（`officeval_060.r00` @ `cp000010`）；`V3-G3` 终态快照分 **`9/9`** 完全一致（最大绝对误差 `0.00e+00`） | **PASS** |

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

## 3. 按家族汇总对比（`Qwen/Qwen3.6-27B` vs DeepSeek `dev45`）

| 任务家族 | `Qwen/Qwen3.6-27B` 3 条得分 | `Qwen/Qwen3.6-27B` 均值 | 非零条数 | 平均请求数 | 平均峰值 Prompt Tokens |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `officeval_070` | `[0.0000, 0.0000, 0.0000]` | **0.0000** | `0/3` | `15.7` | `40777` |
| `officeval_060` | `[0.5455, 0.5455, 1.0000]` | **0.6970** | `3/3` | `11.0` | `34389` |
| `officeval_090` | `[0.4000, 0.4000, 0.4000]` | **0.4000** | `3/3` | `9.0` | `64063` |
