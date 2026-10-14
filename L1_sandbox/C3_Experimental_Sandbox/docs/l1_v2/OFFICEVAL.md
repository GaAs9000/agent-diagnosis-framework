# OfficeVal：原生接入与官方边界

## 来源是两份，不是一个路径

评估源码：`baidu-frontier-research/OmegaUse-OfficeVal`。
任务数据：`baidu/OmegaUse-OfficeVal` Hugging Face Dataset，包含 tasks/rubrics/task_files。
本配置继承原项目的源码 Commit 和 Dataset Revision，分别锁定；不是拿当前 main 覆盖旧基线。数据下载尚未在本次离线构建环境验证。

`officeval-native` 读取干净 Git checkout 的完整 tracked 源码清单，检查实际 Commit；读取原始任务 JSON、对应 Rubric、输入资产并验证数据锁。`fetch` 只获取选定任务的数据而非预先下载全部任务；任务发现从实际本地定义读取，未下载的任务不会被伪装为已准备好。

`lock-local` 是用户主动接受现有数据，不验证其远端发布真实性；Manifest 保存 `data_origin`。`allow_unlocked_data` 只供显式降级实验，默认关闭且来源标 unlocked，不能与正式锁定数据混为同一可比较基线。

## 随包资产

原项目 `runs/integrated-delivery-001/bundle` 保留八个任务的 tasks、rubrics、inputs、verifiers、source_metadata 和 manifest。`officeval-bundle` 验证这些留存文件，可用于离线导出和工程 smoke 检查；不包含完整 upstream Git 仓库，也不能伪称重新从原生版本取得全部任务。

`configs/l1/officeval.bundle-smoke.json` 使用兼容适配器；本地原生验收脚本要求 `adapter=officeval-native`，因此 bundle 通过不等于 native 验收通过。

## 官方评分方式

本轮调用的是上游**单任务原始 Verifier**：`evaluate(directory: str) -> dict`。调用封装使用独立 Docker 进程；submission 冻结、只读挂载；Verifier 源码、Rubric 不挂载给被测 Agent。worker 只负责加载/调用/保存，不重写评分规则。

本轮没有实现官方完整 `run.py` 批处理行为；例如批处理对缺失提交、运行并行、批量报告等额外逻辑不能宣称被完全复刻。Manifest 明确保存 `official_batch_runner_used=false` / `official_batch_runner=false`。单项原始结果与官方完整批处理结果不能不加说明直接视为同一运行协议。

上游约定 `status="ok"` 表示评分正常结束。官方 `dim1_pass`、`dim2_items`、`total_score`、`max_score` 原样保留；Dim1 未通过也可能有解释性 Dim2 条目，不将其误判为损坏数据。

默认 `outcome_policy=official_dimensions`：Dim1 明确失败映射 failure；其他正常评分保留官方完成率和维度结果，通用 `task_outcome` 可以 unknown。可选 `full_score_v1` 将满分/非零/零映射 success/partial/failure，**这是显式 L1 投影政策，不是宣称 OfficeVal 官方统一成功标签**。原始输出始终存在；`c3_operational_completed` 不参与此投影。

## 环境限制

上游列出的 Windows Office COM 任务 001、008、019、022、023、030、039、074、081 在当前 Linux Docker profile 下拒绝运行。其他任务虽然不属于该硬限制，仍不保证全部视觉/渲染 Rubric 在默认静态模式下完全覆盖。不能把“Verifier 正常返回”扩大解释为“所有评分条件全都测量到”；coverage unknown 会保留。

Dockerfile 包含常用 Office/PDF Python 库及 LibreOffice，但本次没有构建/运行该镜像；某个任务额外依赖不满足时应得到环境/评分失败而非任务能力结论。最终实际镜像 ID、镜像内 Python/安装包版本记录用于复核；不以宿主机的包版本替代容器凭证。
