# R6 研究终点规则版本敏感性只读审计

- 采集区间：`2026-10-17T15:31:01.774122+00:00` 至 `2026-10-17T15:31:05.250642+00:00`。
- 扫描范围：`/Users/jiashen/Desktop/问题诊断/L1_sandbox/C3_Experimental_Sandbox/runs`，递归发现所有 manifest，含嵌套 runs/attempt。
- 覆盖：`{'discovered_manifests': 212, 'processed': 210, 'skipped': 2, 'errors': 0}`；逐包路径、跳过原因、错误与原件前后哈希见 JSON `coverage_manifest`。
- 规则：`['v1.0', 'v1.1']`；默认 `v1.1`。
- 读取期间原件哈希一致：`True`；只证明本次读取前后，不推断历史全过程。
- 源码版本以 JSON `source_sha256` 内容哈希绑定；原始官方分保持只读。实时目录仍可能增长，本报告是上述采集区间的清单。

| 批次 | manifest | 已处理/跳过/错误 | v1.0 有效/无效/均分 | v1.1 有效/无效/均分 | 变化包数 |
|---|---:|---|---|---|---:|
| `deepseek_dev45_20261015` | 47 | 47/0/0 | 45/2/0.25753279400338225 | 45/2/0.25753279400338225 | 0 |
| `deepseek_holdout45_20261017` | 57 | 57/0/0 | 50/7/0.2247485726833553 | 50/7/0.2247485726833553 | 0 |
| `integrated-delivery-001` | 1 | 0/1/0 | 0/0/None | 0/0/None | 0 |
| `l1-accept-001` | 1 | 1/0/0 | 1/0/0.0 | 1/0/0.0 | 0 |
| `l1-accept-033` | 1 | 1/0/0 | 1/0/0.0 | 1/0/0.0 | 0 |
| `l1-accept-31pro` | 2 | 2/0/0 | 1/1/0.0 | 1/1/0.0 | 0 |
| `l1-accept-35lite` | 1 | 1/0/0 | 1/0/0.0 | 1/0/0.0 | 0 |
| `l1-accept-35lite-v2` | 2 | 2/0/0 | 2/0/0.3 | 2/0/0.3 | 0 |
| `l1-accept-38flash` | 1 | 1/0/0 | 1/0/0.0 | 1/0/0.0 | 0 |
| `l1-accept-lhtb-sokoban` | 1 | 0/1/0 | 0/0/None | 0/0/None | 0 |
| `l1-accept-llama8b` | 1 | 1/0/0 | 1/0/0.0 | 1/0/0.0 | 0 |
| `l1-accept-qwen32b` | 1 | 1/0/0 | 0/1/None | 0/1/None | 0 |
| `l1-accept-qwen32b-v2` | 2 | 2/0/0 | 2/0/0.0 | 2/0/0.0 | 0 |
| `l1-live-lhtb-35lite` | 3 | 3/0/0 | 0/3/None | 0/3/None | 0 |
| `l1-multitask-033` | 2 | 2/0/0 | 2/0/0.029411764705882353 | 2/0/0.029411764705882353 | 0 |
| `l1-multitask-060` | 2 | 2/0/0 | 2/0/0.7727272727272727 | 2/0/0.7727272727272727 | 0 |
| `l1-multitask-070` | 2 | 2/0/0 | 2/0/0.0 | 2/0/0.0 | 0 |
| `l1-multitask-090` | 2 | 2/0/0 | 2/0/0.2 | 2/0/0.2 | 0 |
| `l1-resume-001` | 1 | 1/0/0 | 1/0/0.0 | 1/0/0.0 | 0 |
| `l1-resume-stateful-31pro` | 1 | 1/0/0 | 0/1/None | 0/1/None | 0 |
| `l1-resume-stateful-35lite` | 1 | 1/0/0 | 1/0/0.0 | 1/0/0.0 | 0 |
| `lhtb_sokoban_qwen36_cohort` | 3 | 3/0/0 | 3/0/0.0 | 3/0/0.0 | 0 |
| `m1_20261014_2034` | 1 | 1/0/0 | 1/0/0.0 | 1/0/0.0 | 0 |
| `m1_20261014_2034r1` | 4 | 4/0/0 | 4/0/0.13636363636363635 | 4/0/0.13636363636363635 | 0 |
| `qwen36_pi_p1_pilot_20261017` | 16 | 16/0/0 | 15/1/0.2824242424242424 | 16/0/0.26477272727272727 | 1 |
| `swebench_pro_f27_live_20261017` | 1 | 1/0/0 | 1/0/0.0 | 1/0/0.0 | 0 |
| `swebench_pro_qwen36_cohort` | 3 | 3/0/0 | 3/0/0.0 | 3/0/0.0 | 0 |
| `v3_qwen36_pi_dev45_20261017` | 52 | 52/0/0 | 35/17/0.210122230710466 | 36/16/0.20428550207961974 | 1 |

## 版本变化包

| 包路径 | v1.0 | v1.1 |
|---|---|---|
| `qwen36_pi_p1_pilot_20261017/episodes/officeval_070.r02` | `{"study_valid": false, "study_outcome_y": null, "zero_reason": null, "study_classification": "unhealthy_or_ambiguous_official_error", "endpoint_version": "study-outcome-v1", "rule_version": "v1.0", "native_evaluation_validity": "invalid", "native_projection_evaluation_validity": "invalid", "native_status": "error", "native_error": "AttributeError: 'NoneType' object has no attribute 'val'"}` | `{"study_valid": true, "study_outcome_y": 0.0, "zero_reason": "missing_or_damaged_delivery", "study_classification": "valid_zero", "endpoint_version": "study-outcome-v1", "rule_version": "v1.1", "native_evaluation_validity": "invalid", "native_projection_evaluation_validity": "invalid", "native_status": "error", "native_error": "AttributeError: 'NoneType' object has no attribute 'val'"}` |
| `v3_qwen36_pi_dev45_20261017/episodes/officeval_070.r03` | `{"study_valid": false, "study_outcome_y": null, "zero_reason": null, "study_classification": "unhealthy_or_ambiguous_official_error", "endpoint_version": "study-outcome-v1", "rule_version": "v1.0", "native_evaluation_validity": "invalid", "native_projection_evaluation_validity": "invalid", "native_status": "error", "native_error": "AttributeError: 'NoneType' object has no attribute 'val'"}` | `{"study_valid": true, "study_outcome_y": 0.0, "zero_reason": "missing_or_damaged_delivery", "study_classification": "valid_zero", "endpoint_version": "study-outcome-v1", "rule_version": "v1.1", "native_evaluation_validity": "invalid", "native_projection_evaluation_validity": "invalid", "native_status": "error", "native_error": "AttributeError: 'NoneType' object has no attribute 'val'"}` |

## 跳过与错误

| 包路径 | 状态 | 原因 |
|---|---|---|
| `integrated-delivery-001/bundle` | skipped | not_live_recording |
| `l1-accept-lhtb-sokoban` | skipped | not_live_recording |
