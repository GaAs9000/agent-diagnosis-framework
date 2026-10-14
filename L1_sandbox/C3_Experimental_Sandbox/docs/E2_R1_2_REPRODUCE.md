# E2-R1.2 离线复核

本轮只有一次真实的独立开发资格尝试。G0 为 `BLOCKED_SYSTEM`，没有 OfficeVal 候选、成功检查点或 E2-S 分支。以下步骤读取原始记录，不重新调用模型，不修改已冻结的证据。

在 `C3_Experimental_Sandbox` 项目目录中运行：

```bash
.venv/bin/python scripts/verify_r12_g0.py
.venv/bin/python -m pytest -q
```

独立核验器会校验本地模型文件约 16 GiB 的摘要，读取较慢属于正常现象。当前预期输出是 `audit_status=PASS` 与 `qualification_status=BLOCKED_SYSTEM`；前者不是资格通过。验收报告位于 `runs/e2-r1-2/R1_2_ACCEPTANCE.md`。

原始证据与检查分工：

| 文件 | 复核内容 |
|---|---|
| `system_manifest.json`、`system_qualification/freeze_event.json` | 冻结系统、资格规则、源码与服务身份 |
| `system_qualification/agent/model_calls.jsonl` | 15 次请求、HTTP 字节、SDK 响应、Token 与逐请求服务检查 |
| `system_qualification/agent/events.jsonl`、`protocol_audit.jsonl` | 原生调用、返回观测、末轮未知工具名拒绝 |
| `system_qualification/tool-dispatch/` | 14 次派发对应的代码、退出码和前后快照 |
| `system_qualification/final-workspace/` | 两份输入未变，两份新输出及实际错误 |
| `system_qualification/independent_grade.json` | 冻结确定性评分，不执行模型提交的评分代码 |
| `verification/independent_g0.json` | 独立重算的证据检查、评分与成本 |
| `cleanup.json` | 运行后服务与容器的实际状态 |

`is_synthetic=false` 仅说明资格执行使用真实模型；输入来自合成的独立开发任务。`nontrivial_file_change_verified=false` 表示整体合同未通过，原始文件差分仍证明发生了两次写入。核验器明确区分这两类事实。

`scripts/run_r12_g0.py` 提供事前 `freeze` 和单次 `run` 入口。此次尝试已关闭，不应重复运行、删除目录重跑，或把同一个资格任务当成新候选。系统代码摘要、既存尝试和门禁均有检查；任何未来的新运行需要新的明确协议与输出目录。

依赖位于本项目 `.venv` 和 `.model-venv`，固定文件为 `requirements.local-arm64.lock.txt` 与 `requirements.model.local-arm64.lock.txt`。原始 45 个源码／依赖文件另保存在 `system_qualification/source_snapshot/`。当前核验不要求模型服务或 Docker 容器处于运行状态。

本轮没有重新做真实运行负例。新增离线测试覆盖不完整观测、写入后缺少读回、错误字段、来源或摘要漂移、服务身份改变与未知工具拒绝等路径；它们不能代替成功侧运行验收。
