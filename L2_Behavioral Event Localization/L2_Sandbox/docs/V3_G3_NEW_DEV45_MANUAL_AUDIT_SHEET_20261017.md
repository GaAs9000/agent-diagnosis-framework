# V3-G3 新开发集 (`v3_qwen36_pi_dev45_20261017`) 20 检查点人工核查表（R20）

- **所属工作项**：`R20`（`V3-G3` 快照评分一致性门禁）
- **状态**：**待人工核查**（严禁以程序自动比对或模板 `PASS` 代替真人签字）
- **程序自动核对结果（单独列报，不抵扣人工核查）**：已完成 L2 快照评分的新 `dev45` 轨迹共 `5` 条，终态快照分与 L1 官方终态分自动比对一致率 `5/5`；P1 探针 9 条轨迹终态快照分与 L1 官方终态分自动比对一致率 `9/9`。
- **当前新 `dev45` 检查点抽样池规模**：`516` 个检查点（固定随机种子 `20261017` 抽取 `20` 个检查点供人工核验）。

## 20 个抽样检查点人工核查记录表（由用户或指定审阅人填写）

| # | 运行 ID (`episode_id`) | 检查点 (`checkpoint_id`) | 快照归档 (`snapshot_id`) | 候选路径 (`candidate`) | 自动快照状态 | 自动快照分/参考 | 审阅人 | 审阅时间 | 人工核查结论 | 备注 |
|---:|---|---|---|---|---|---|---|---|---|---|
| 1 | `officeval_003.r00` | `cp000004` | `s000004` | `(none)` | `no_candidate` | `0.0000` |  |  |  |  |
| 2 | `officeval_003.r02` | `cp000007` | `s000007` | `(pending_l2_snapshot_grade)` | `recorded_on_disk` | `(terminal=0.0000)` |  |  |  |  |
| 3 | `officeval_004.r03` | `cp000008` | `s000008` | `(pending_l2_snapshot_grade)` | `recorded_on_disk` | `(terminal=0.6364)` |  |  |  |  |
| 4 | `officeval_010.r01` | `cp000017` | `s000017` | `(pending_l2_snapshot_grade)` | `recorded_on_disk` | `(terminal=0.0000)` |  |  |  |  |
| 5 | `officeval_010.r02` | `cp000009` | `s000009` | `(pending_l2_snapshot_grade)` | `recorded_on_disk` | `(terminal=0.0000)` |  |  |  |  |
| 6 | `officeval_033.r00` | `cp000001` | `s000001` | `(none)` | `no_candidate` | `0.0000` |  |  |  |  |
| 7 | `officeval_033.r00` | `cp000014` | `s000014` | `submission/家居零售价目册_已调价.docx` | `ok` | `0.0000` |  |  |  |  |
| 8 | `officeval_033.r00` | `cp000016` | `s000016` | `submission/家居零售价目册_已调价.docx` | `ok` | `0.0000` |  |  |  |  |
| 9 | `officeval_033.r00` | `cp000021` | `s000021` | `submission/家居零售价目册_已调价.docx` | `ok` | `0.0000` |  |  |  |  |
| 10 | `officeval_033.r00` | `cp000023` | `s000023` | `submission/家居零售价目册_已调价.docx` | `ok` | `0.0000` |  |  |  |  |
| 11 | `officeval_033.r03` | `cp000015` | `s000015` | `(pending_l2_snapshot_grade)` | `recorded_on_disk` | `(terminal=0.0000)` |  |  |  |  |
| 12 | `officeval_050.r01` | `cp000003` | `s000003` | `(pending_l2_snapshot_grade)` | `recorded_on_disk` | `(terminal=0.0500)` |  |  |  |  |
| 13 | `officeval_050.r02` | `cp000020` | `s000020` | `(pending_l2_snapshot_grade)` | `recorded_on_disk` | `(terminal=0.0500)` |  |  |  |  |
| 14 | `officeval_060.r00` | `cp000005` | `s000005` | `(none)` | `no_candidate` | `0.0000` |  |  |  |  |
| 15 | `officeval_060.r00` | `cp000008` | `s000008` | `(none)` | `no_candidate` | `0.0000` |  |  |  |  |
| 16 | `officeval_070.r01` | `cp000003` | `s000003` | `(pending_l2_snapshot_grade)` | `recorded_on_disk` | `(terminal=0.0000)` |  |  |  |  |
| 17 | `officeval_085.r00` | `cp000010` | `s000010` | `(pending_l2_snapshot_grade)` | `recorded_on_disk` | `(terminal=0.0000)` |  |  |  |  |
| 18 | `officeval_085.r01` | `cp000003` | `s000003` | `(pending_l2_snapshot_grade)` | `recorded_on_disk` | `(terminal=0.0000)` |  |  |  |  |
| 19 | `officeval_090.r00` | `cp000003` | `s000003` | `(pending_l2_snapshot_grade)` | `recorded_on_disk` | `(terminal=0.4000)` |  |  |  |  |
| 20 | `officeval_090.r01` | `cp000000` | `s000000` | `(pending_l2_snapshot_grade)` | `recorded_on_disk` | `(terminal=0.4000)` |  |  |  |  |
