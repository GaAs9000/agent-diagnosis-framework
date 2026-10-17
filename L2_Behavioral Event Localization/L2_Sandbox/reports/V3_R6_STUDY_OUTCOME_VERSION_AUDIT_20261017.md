# R6 研究终点规则版本化与新旧敏感性双算报告（2026-10-17）

- **生成时间**：`2026-10-17T14:25:27.435827+00:00`
- **支持规则版本**：`('v1.0', 'v1.1')`（当前默认 `RULE_VERSION = 'v1.1'`）
- **`v1.1` 新增条目**：`["AttributeError: 'NoneType' object has no attribute 'val'"]`
- **原始官方评分文件是否改动**：`False`（所有 `official_grade.json` 与 `manifest.json` 原件保持只读，不改原始官方分）

## 1. 规则版本定义对比

| 规则版本 | 涵盖为 `valid_zero` (`missing_or_damaged_delivery`) 的损坏交付异常 | 触发背景 |
|---|---|---|
| `v1.0` | `FileNotFoundError`, `BadZipFile`, `PackageNotFoundError`, `InvalidFileException`, `XMLSyntaxError`, `ParseError`, `'未找到'`, `'不存在'`, `'无法打开'`, `'不是 .docx'`, `'损坏'`, `'corrupt'`, `'not a zip file'` | 初始归档/空文件损坏交付白名单 |
| `v1.1` | 在 `v1.0` 基础上显式新增：OpenXML 属性缺失异常 `"AttributeError: 'NoneType' object has no attribute 'val'"` | `P1` 探针 `officeval_070.r02`（以及早期单条探针 `070.r00`）：产出的 `.docx` 缺少必要 XML `.val` 属性导致评测器抛 `AttributeError`，实为损坏交付而非评测容器故障 |

## 2. 各批次 `v1.0` vs `v1.1` 并排敏感性汇总表

| 批次 (`cohort_id`) | 扫描包总数 | `v1.0` 有效数 / 有效零分数 / 无效数 | `v1.0` 有效均分 | `v1.1` 有效数 / 有效零分数 / 无效数 | `v1.1` 有效均分 | 受影响包数 (`v1.0 != v1.1`) |
|---|---:|---:|---:|---:|---:|---:|
| `qwen36_pi_p1_pilot_20261017/episodes` | 9 | 8 / 2 / 1 | `0.4114` | 9 / 3 / 0 | `0.3657` | **1** |
| `qwen36_pi_p1_pilot_20261017/v3_g1_cold_resume` | 7 | 7 / 5 / 0 | `0.1351` | 7 / 5 / 0 | `0.1351` | **0** |
| `v3_qwen36_pi_dev45_20261017/episodes` | 31 | 29 / 13 / 2 | `0.2104` | 29 / 13 / 2 | `0.2104` | **0** |
| `v3_qwen36_pi_dev45_20261017/failed_attempts` | 10 | 0 / 0 / 10 | `N/A` | 0 / 0 / 10 | `N/A` | **0** |
| `v3_qwen36_pi_dev45_20261017/v3_g2_c0_resumes` | 0 | 0 / 0 / 0 | `N/A` | 0 / 0 / 0 | `N/A` | **0** |
| `deepseek_dev45_20261015/runs` | 0 | 0 / 0 / 0 | `N/A` | 0 / 0 / 0 | `N/A` | **0** |
| `deepseek_holdout45_20261017/runs` | 0 | 0 / 0 / 0 | `N/A` | 0 / 0 / 0 | `N/A` | **0** |
| `swebench_pro_f27_live_20261017` | 1 | 1 / 1 / 0 | `0.0000` | 1 / 1 / 0 | `0.0000` | **0** |
| `lhtb_sokoban_qwen36_cohort` | 3 | 3 / 3 / 0 | `0.0000` | 3 / 3 / 0 | `0.0000` | **0** |

## 3. 受影响运行（`v1.0 != v1.1`）逐条明细清单

| 批次 | 运行/尝试 ID | `official_grade.json` 原始 `error` | `v1.0` 判定 (`classification / valid / score`) | `v1.1` 判定 (`classification / valid / score`) |
|---|---|---|---|---|
| `qwen36_pi_p1_pilot_20261017/episodes` | `officeval_070.r02` | `AttributeError: 'NoneType' object has no attribute 'val'` | `unhealthy_or_ambiguous_official_error / False / None` | `valid_zero / True / 0.0` |
