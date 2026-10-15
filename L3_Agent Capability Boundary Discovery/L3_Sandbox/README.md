# L3 Sandbox

用现有 OfficeVal 任务做预算与公开结构条件实验，复用 L1 原生执行、隔离和官方评分。L3 不导入 L2，也不改变旧 M1 执行包、分支结果或评分。

## 安装与离线准备

在仓库根目录，用已有 L1 虚拟环境安装；无需额外分析服务：

```bash
L1_sandbox/C3_Experimental_Sandbox/.venv/bin/python -m pip install --no-deps -e 'L3_Agent Capability Boundary Discovery/L3_Sandbox'
```

该环境需要安装构建依赖 setuptools/wheel，以及 L1 的 jsonschema。运行计划需要已经下载并锁定的原生 OfficeVal 数据、干净的官方源码和本地 Docker 镜像。计划生成只读取这些本地资产，不调用模型或读取凭据。

```bash
L1_sandbox/C3_Experimental_Sandbox/.venv/bin/l3 plan \
  'L3_Agent Capability Boundary Discovery/L3_Sandbox/configs/budget_pilot.json' \
  --output 'L3_Agent Capability Boundary Discovery/L3_Sandbox/validation/budget_pilot_20261015'

L1_sandbox/C3_Experimental_Sandbox/.venv/bin/l3 run \
  'L3_Agent Capability Boundary Discovery/L3_Sandbox/validation/budget_pilot_20261015'

L1_sandbox/C3_Experimental_Sandbox/.venv/bin/l3 summarize \
  'L3_Agent Capability Boundary Discovery/L3_Sandbox/validation/budget_pilot_20261015'
```

`run` 默认仅验证计划和打印工作量。只有显式追加 `--execute` 才加载凭据、创建新的完整 L1 执行并调用模型。`check-plan <目录>` 可独立检查代码、输入和镜像。

## 预算 pilot

示例选择未在现有 M1 中探索的 004、010、050、085、095，覆盖 Word、PowerPoint、Excel。按已决定的 D3，正式清单 `L1_sandbox/C3_Experimental_Sandbox/benchmarks/officeval/family_split.json` 将 91 个任务划为 28 个操作性语义家族：9 个开发家族（55 任务）、19 个留出家族（36 任务）。五个 pilot 任务属于四个开发家族，085 与 095 同组。计划校验并冻结清单快照与哈希；`family_verified` 表示定义与成员映射已核对，模板谱系仍为 unknown，不能据此宣称天然独立。详见 `docs/FAMILY_SPLIT.md`。

- 5 任务 × 8／35／70 请求 × 3 重复 = 45 个预定执行槽。
- 主比较为 70 对 35 的平均完成度差；0.1 是该开发计划预先选择的最小有意义改善，尚未经试跑校准。
- 所有档位的工具上限统一为 140，墙钟为 10800 秒，其他模型参数不变。该配置不同于旧 35 请求／50 工具的 M1；不能直接混用成同一 M2 基线。
- 调度按任务／重复分块，块内条件打乱，块间也打乱；调度随机种子不会成为模型采样种子。
- 总逻辑请求上界 1695；每请求最多 4 次 HTTP 尝试时，初始槽的 HTTP 上界 6780。最多允许每槽 2 次基础设施替换，理论上界 20340，但全局硬上限仍为 6780，触顶就停止并报告未完成，不自动扩额。
- 模型会看到真实预算告知，测量的是告知与实际上限这一配置组合。

## M2 开发侧基线

`configs/m2_dev_baseline.json` 使用单条件 `design: "baseline"`，覆盖 9 个开发家族代表任务，每任务 5 次，共 45 槽。当前配置为 `officeval-m2-dev-baseline-v2`：SWE 评分与资产锁修复改变了 L1 冻结源码，因此本配置不导入旧预算 pilot 的 12 条记录，45 槽均待执行。旧 pilot 保持原样，仍可离线分析。

```bash
L1_sandbox/C3_Experimental_Sandbox/.venv/bin/l3 plan \
  'L3_Agent Capability Boundary Discovery/L3_Sandbox/configs/m2_dev_baseline.json' \
  --output 'L3_Agent Capability Boundary Discovery/L3_Sandbox/validation/m2_dev_baseline_v2'
L1_sandbox/C3_Experimental_Sandbox/.venv/bin/l3 run \
  'L3_Agent Capability Boundary Discovery/L3_Sandbox/validation/m2_dev_baseline_v2'
```

以上仅冻结计划和预览，不调用模型。正式采样仍需显式 `--execute`。通用 `reuse_from` 功能会在复制前比较源、目标的执行代码、镜像、任务指令、公开资产和私有评分资产；只改变分析代码允许复用，改变执行合同则拒绝。dry-run 的 `workload` 是冻结槽位总数；已复用和待运行槽位可从 `run_results.json` 的有效 receipt 核对。

## 结构信息 × 预算

`configs/structure_budget.json` 是六单元开发示例：原版 O、等字符长度无关材料 P、结构摘要 S，分别与 35／70 请求交叉；同五个开发任务、每格 3 次，共 90 槽。它不是已执行或已冻结的正式验证实验。

安装包注册 `l1.benchmarks/l3-officeval`。该适配器先委托原生 OfficeVal 严格加载，再仅在原任务指令后追加 P 或 S。原版 O 的任务指令字节不变。原资产、任务目标、私有文件和官方评价入口保持不变；生产路径不用 `injected=`，不会伪装测试执行为 live。

结构摘要只读取 `Task.assets`：DOCX 表格／段落结构，XLSX 工作表名、维度、合并范围和公式坐标，PPTX 幻灯片与对象类型计数；不提取单元格值、公式内容或私有评分资料。非 Office 附件仅记录文件名、格式和大小。P 与 S 字符长度一致，UTF-8 字节数另外记录。原任务中所需信息仍可通过原有工具获取。

计划冻结变换版本、附加文本和全部输入哈希；执行时重新从公开输入生成摘要核对。报告分别计算 S−P、P−O、S−O 及两预算间的差中之差，不用“一个显著、另一个不显著”替代交互检验。

## 结果合同与恢复

L1 `evaluation/study.py` 提供共享研究终点；旧 `run_matrix.py` 保留原始官方字段，同时新增 `study_valid`、`study_outcome_y`、分类和零分原因。

- 健康评分器判定未交付／损坏：研究终点为有效零分，原生 invalid/error 状态仍保留。
- 评分器崩溃、缺少原始评分、执行无效或不完整：研究无效，保留证据；按冻结上限换新目录补跑。
- 正常 OfficeVal 完成度采用官方投影；Sokoban 研究终点为 passed，不能用带步数惩罚的 reward 代替。
- 不将交付存在、合法交付与质量得分混称。另报首次持久写入、首次 submission 写入、请求／工具数、模型返回版本、HTTP 尝试和已知 tokens；未知用量和费用保留未知。

每槽保留 `runs/<trial>/attempt_NNN/episode`，失效或中断现场不删除。外层 receipt 绑定配置、任务和证据索引，复用时检查原生封签及重新推导的结果。整个计划有执行锁；HTTP 准入账本跨重启持久化，每次实际 HTTP 尝试均经过共享 RPM／TPM／总次数上限。未知用量保留预算预留。

认证失败触发持久停止标记。认证观察在专用 L3 CLI 进程中临时包装 L1 HTTP 传输；新请求与重试会停止，已在途请求仍可能完成。一次进程只运行一个 L3 执行作用域。

## 报告与研究边界

`summary.json`、`summary.csv`、`report.md` 记录预定分母、有效／缺失槽、无效尝试与替换，不以少跑的数据冒充完整比较。仅完整任务配对进入主效应；家族区间要求核实家族且至少两个家族。模型版本漂移会显示并阻止混合的主对比。

报告保留完整任务／条件基线 cohort、配置哈希、终点版本及模型版本；完整的 baseline 计划另导出 `m2_baseline.json`。`l3 resolve-p0 <计划目录> --parent <父执行包>` 从冻结计划、receipt 和原始密封证据重新计算基线，并排除父轨迹自身；缓存的汇总数值不能代替证据。解析过程不写回源实验，也不调用模型或评分器。

同目录 `summary.json`／`m2_baseline.json` 可作为入口。移到其他目录的导出 JSON 必须携带 `source_binding`，其源目录和计划哈希须可核验，且导出内容须与重算结果一致；没有源证据绑定的 JSON 会被拒绝。L2 的 `--baseline` 使用同一个解析器，检查任务、配置、模型返回版本、当前研究终点版本与逐槽身份；外部基线仍可用显式 `--p0`，不会标为证据已验证。

计划冻结实际执行代码、原任务输入、配置、镜像及家族清单；L3 报告代码和 L1 的研究终点投影 `evaluation/study.py` 只记溯源，不作为执行复用的门禁。修改分析实现后，`summarize` 从封存运行重算；`run` 复用健康的已完成执行，不因派生结局变化付费重跑。原计划及逐次 receipt 不改写，新运行和每次汇总分别记录代码溯源。真正改变执行代码、输入、配置或镜像时，采集需使用新计划；历史汇总无需重装原始执行环境。原始实验及验证产物在本地 `validation/`／`runs/`，不进入 Git。

本版本完成执行与分析工程，不包含已经验证的能力边界、自然任务迁移预测器、前缀检查点评分或剩余预算提醒实验。后几项应按独立协议加入，不能从这轮离线测试推断有效性。
