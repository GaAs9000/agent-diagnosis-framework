# V3 Stage E: 多基准（SWE-bench Pro V2 & LHTB Sokoban）E1 预拆与边界核验报告

> **更正说明（2026-10-17，对应修订前提交 `0058bde`，执行项 `R4`）**：
> 1. **阶段状态明确化**：本阶段当前状态为 **“E1 预拆完成，待补最小干预闭环（部分完成）”**，不宣称 Stage E 已全部完成。
> 2. **SWE `task2_s42_f27` 原件事实纠正**：核对 `diagnoses/swebench_pro_f27_live_20261017/task2_s42_f27/grades.jsonl` 原件，在 `cp000011`、`cp000012` 与 `terminal_submission` 上均为 `candidate = ["submission/patch.diff"]`、`in_submission = true`、`dim1_pass = false`、`score = 0.0`（且 `V3-G1/F27` 回执记录 `2,038` 字节提交补丁）。因此删除旧报告中“未提交补丁（`undelivered`）”与“终态 `broken = 3`”的表述，严格按原件分开报告：(a) **提交补丁存在**（`in_submission = true`）；(b) **门槛/测试失败**（`4/4 fail_to_pass` 通过，但 `pass_to_pass` 为 `23/26`，触发 `dim1_pass = false, dim1_gate = 0.9, score = 0.0`）；(c) **候选内回归（`in_candidate_regression`）单列**（`3` 项 `pass_to_pass` 回归单列，不计入终态 `broken`）。
> 3. **Sokoban 状态损害单列与死锁轮次统一**：将静态 2×2/角落死锁检测结果单列为 **“独立核对的状态损害（`independent_state_damage_corner_deadlock`）”**，不并入 `broken`（`broken = 0`），并注明静态角落死锁检测不能覆盖全部死锁；同时统一 `run_seed42` 的死锁回合记录（**重置前第 9 回合 `r000008_c0.json` 首次出现角落死锁，第 10 回合 `reset` 后在第 15 回合 `r000014_c0.json` 再次出现角落死锁**）。

- **统一被测 Actor**：`Qwen/Qwen3.6-27B (revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9)`
- **机读结果文件**：[`reports/v3_stage_e_swe_and_sokoban_v05_20261017.json`](./v3_stage_e_swe_and_sokoban_v05_20261017.json)
- **当前阶段状态**：**E1 预拆完成，待补最小干预闭环（部分完成）**

---

## 1. 跨基准 E1 预拆与独立状态核对总表（V3 §3.6 & §4.1）

| 基准 (Benchmark) | 原生环境与 Harness | 终态快照与 L1 官方分一致性 | 终态 `broken` (E1) | 终态 `undelivered` (E1) | 终态未得分 / 门槛未过 (`never_succeeded_on_disk`) | 独立单列项（候选内回归 / 独立核对的状态损害） | 当前状态与边界说明 |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **SWE-bench Pro V2 (`task2_s42_f27`, Post-`F27`)** | `native-tool-agent` + 持久可写 `/app` | `4/4 (100%)`（3 条 pre-`F27` + 1 条 post-`F27`） | **`0`** | **`0`**（`submission/patch.diff` 已交，`in_submission=true`） | **`1` 条运行**（`dim1_pass=false, score=0.0`；其中 `4/4 fail_to_pass` 在提交补丁内已通过） | **候选内回归 (`in_candidate_regression`)：`3` 项**（`pass_to_pass` 在第 11 步补丁内由绿变红） | 补丁已写入 `submission/patch.diff`（`2,038B`），因 `23/26 pass_to_pass` 未过官方 `dim1_pass` 门槛得 `0.0` 分；**E1 预拆完成，待补最小干预闭环** |
| **LHTB Sokoban (`3` seeds)** | `native-tool-agent` + `/workspace/.lhtb_state` | `3/3 (100%)`（`63` 个检查点） | **`0`** | `N/A` (`not_applicable`) | **`3` 条运行**（全程 `band = 0`） | **独立核对的状态损害 (`corner_deadlock`)：`2` 条运行**（`run_seed42`、`run_seed2026`） | 无独立交付通道（`undelivered = N/A`）；静态角落死锁检测单列，且不覆盖全部死锁类型；**E1 预拆完成，待补最小干预闭环** |

---

## 2. SWE-bench Pro V2（`task2_s42_f27` `grades.jsonl` 原件核对与测试级分解）

- **运行标识**：`task2_s42_f27` (`ep_796ed3710a33448aad3aab214e076ff1`, `swebench_pro:instance_ansible__ansible-0fd88717c953...`)
- **`c3-l1 verify`**：`valid` (`verified = True`)
- **`grades.jsonl` 原件逐点记录**：
  - **`cp000000` .. `cp000010`**：`candidate = []`, `candidate_status = "no_candidate"`, `in_submission = false`, `dim1_pass = false`, `score = 0.0`。
  - **`cp000011` (`s000011`) & `cp000012` (`s000012`)**：`candidate = ["submission/patch.diff"]`, `candidate_status = "ok"`, `in_submission = true`, `dim1_pass = false`, `score = 0.0`, `deductions.dim1_gate = 0.9`。
  - **`terminal_submission` (`s000013`, $k=12$)**：`candidate = ["submission/patch.diff"]`, `candidate_status = "submission"`, `in_submission = true`, `dim1_pass = false`, `score = 0.0`, `deductions.dim1_gate = 0.9`。
- **提交补丁（`2,038` 字节，`sha256: 167b2c96...`）下的 30 项官方 pytest 分解（`27/30 = 90.0%` 通过）**：
  1. **目标 `fail_to_pass` 测试（`4/4 = 100%` PASSED，`grades.jsonl` 单项 `deduction = 0.0`）**：
     - `test/units/plugins/lookup/test_password.py::TestParseContent::test`
     - `test/units/plugins/lookup/test_password.py::TestParseContent::test_empty_password_file`
     - `test/units/plugins/lookup/test_password.py::TestParseContent::test_with_salt`
     - `test/units/plugins/lookup/test_password.py::TestParseContent::test_with_salt_and_ident`
  2. **基线 `pass_to_pass` 测试（`23/26` PASSED，3 项在第 11 步补丁中回归，单列为 `in_candidate_regression`）**：
     - `test/units/plugins/lookup/test_password.py::TestLookupModuleWithPasslib::test_encrypt`（`deduction = 0.033333`, `first_patch_write_turn = 11`, 类别：`in_candidate_regression`）
     - `test/units/plugins/lookup/test_password.py::TestLookupModuleWithoutPasslib::test_no_encrypt`（`deduction = 0.033333`, `first_patch_write_turn = 11`, 类别：`in_candidate_regression`）
     - `test/units/plugins/lookup/test_password.py::TestLookupModuleWithoutPasslib::test_only_a`（`deduction = 0.033333`, `first_patch_write_turn = 11`, 类别：`in_candidate_regression`）
  3. **门槛未通过说明**：由于 SWE-bench Pro 官方评分器要求 `pass_to_pass` 26 项全部通过才授予 `dim1_pass = true` 及正分，上述 3 项候选内回归导致 `dim1_pass = false, score = 0.0`。因为全轨迹最高官方快照分 $\max_k s(k) = 0.0 = s(T)$，按 V3 §3.2 严格定义不计入终态 `broken`（`broken = 0`），而将这 3 项单列为 **候选内回归（`in_candidate_regression`）**。

---

## 3. LHTB Sokoban（E1 预拆与独立核对的状态损害）

> **检测器覆盖范围声明**：下表使用的静态 2×2/角落死锁检测器仅检查非目标格箱子 `$` 是否被正交相邻墙面 `#` 卡死（对角落死锁是充分且零成本的），但**不能覆盖全部 Sokoban 死锁形态**（如通道封死、双箱贴墙死锁等）；因此未检出角落死锁不代表盘面仍可解，且角落死锁统一单列为“独立核对的状态损害”，不并入 `broken`。

| Episode | `k_star` | 重置前首次角落死锁回合 | `reset` 回合 | 重置后首次角落死锁回合 | 终态 `e1_class` | 终态 `broken` | 独立核对的状态损害 (`independent_state_damage`) | 当前状态 |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: | :--- | :--- |
| `run_seed42` | `0` | **第 `9` 回合** (`r000008_c0.json`) | 第 `10` 回合 | **第 `15` 回合** (`r000014_c0.json`) | `never_succeeded_on_disk` | `0` | **`corner_deadlock_detected`**（重置前第 9 回合、重置后第 15 回合） | `E1 预拆完成，待补最小干预闭环` |
| `run_seed101` | `0` | `None` | `[8]` | `None` | `never_succeeded_on_disk` | `0` | `no_corner_deadlock_detected`（静态角落检测未触发，不排除其他类型死锁） | `E1 预拆完成，待补最小干预闭环` |
| `run_seed2026` | `0` | **第 `4` 回合** (`r000003_c0.json`) | `[9]` | **第 `12` 回合** (`r000011_c0.json`) | `never_succeeded_on_disk` | `0` | **`corner_deadlock_detected`**（重置前第 4 回合、重置后第 12 回合） | `E1 预拆完成，待补最小干预闭环` |
