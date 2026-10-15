# L2 v0.3 运行指南

L2 读取已封签的 L1 父轨迹，提取事实，预测检查点价值，再用原样续跑测量价值曲线和检验行为假说。设计合同见 [DESIGN_INPUT_v0.3.md](DESIGN_INPUT_v0.3.md)，研究约束见 [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md)。

## 安装与输入

在 `L2_Sandbox/` 下使用 Python 3.12+，联合安装本仓库中的 L1 与 L2：

```bash
python -m pip install -e '../../L1_sandbox/C3_Experimental_Sandbox[l1]' -e '.[test]'
python -m pytest -q
l2 --help
```

离线分析也通过 L1 的统一结局模块判定交付与评分，L1 因此是必需依赖；安装时从上述本地路径提供，不假设它已发布到 PyPI。缺少 L1 时请执行联合安装命令。真实续跑和 LLM 分析另需对应镜像、任务资产及环境变量凭证。L2 不下载模型或任务，也不重建官方评分器。输入为 L1 父执行包目录；输出目录必须与父包互不包含。

## 两个命令

所有 `l2 run` 调用（包括 dry-run）都必须显式传入 `--p0`。下例的数值仅演示语法，正式实验应替换为对应配置下、独立 M2 队列测出的基线，并记录来源。

```bash
# 本地事实、规则预测与成本估计；LLM 缓存未命中时标为 deferred。
l2 run /path/to/parent --output runs/episode --analysts rules --p0 0.4 \
  --baseline-provenance 'M2 cohort identifier, configuration hash and parent exclusion'

# --execute 允许付费分析器请求及 L1 续跑。
l2 run /path/to/parent --output runs/episode --p0 0.4 \
  --baseline-provenance 'M2 cohort identifier, configuration hash and parent exclusion' \
  --analysts rules,llm_raw:configs/analyst.gemini.json,llm:configs/analyst.gemini.json \
  --execute --rpm 60 --tpm 100000

l2 summarize runs/episode1 runs/episode2 \
  --holdout-family family1 --output runs/summary.json
```

`c3-l2` 是相同入口的别名。`--p0` 必须为有限的 [0,1] 数；提供该值并不等于程序验证过采样来源。`--baseline-provenance` 写入基线记录，当前仍标记 `cohort_verified: false`。

Python API `report.run_episode` 同样要求显式提供 `p0`；程序化实验应同时记录来源。M1 专用脚本显式传入 0.5，并声明它是工程占位值，历史 M1 记录不改写。

曲线默认参数为 `--n1 12 --n2 24 --stride 4 --alpha 0.05`。`--config` 可提供 `curve.rpm`、`curve.tpm`，对应 CLI 参数优先。模型配置只允许环境变量名引用凭证，禁止内嵌密钥。

## 当前模块与数据流

| 模块 | 职责 |
|---|---|
| `evidence/l1.py`、`evidence/view.py` | 读取并校验父证据，构建分析器可见的父轨迹视图 |
| `facts/build.py`、`facts/docx_probe.py` | 确定性提取 `{turn, type, value}` 事实及 DOCX 结构变化 |
| `analysts/rules.py`、`analysts/llm.py` | 第一次预测检查点价值；集中损失时第二次提出谓词假说 |
| `predicates.py` | 求值 `all`、`any`、`not` 与 `{fact, where}` |
| `curve.py`、`stats.py` | 稀疏网格和区间细化续跑、无效分支处置、置信界及统计检验 |
| `score.py`、`report.py` | 假说检验、技能分、跨家族汇总和报告编排 |
| `cli.py`、`config.py`、`io.py` | 命令、配置、产物绑定及缓存支持 |

A 组 `rules@1` 始终存在；`llm_raw:<config>` 为 B 组，`llm:<config>` 默认是父轨迹加事实的 C 组。R 组用 `--analysts reference --reference-scores scores.json`，必须提供每个检查点 `0..T-1` 的实际官方快照得分。分析器不可读取分支动作、分支事实或逐分支结局；第二次调用只增加允许的检查点级曲线摘要。

## 输出与续跑

本地准备阶段输出 `baseline.json`、`plan.json`、`facts.jsonl`、`values/` 和 `report.md`。执行阶段另产出 `branches/<trial>/`、`outcomes.jsonl`、`curve.json`、`scores.json`；检出集中损失时才提出并记录 `hypotheses/`。`diffuse_loss` 不触发第二次分析器调用，但可能已经完成两阶段曲线采样。

已有有效分支按采样条件幂等复用；父轨迹或采样条件不匹配时拒绝混用。规则与统计结果可重新计算；付费分析器响应按实际模型配置、提示和输入负载缓存，分析实现变化本身不应触发付费重跑。正常任务失败与基础设施异常必须分别处理，未交付不自动视为基础设施故障。缺失假说检验所需回合事实时，结果保留 `untestable`。

## 结果边界

M1 实测概要见 [M1_PILOT_20261014.md](M1_PILOT_20261014.md)，探索性行为诊断见 [L2_POSTHOC_20261014.md](L2_POSTHOC_20261014.md)。工程验收通过、单条程序修复和事后相关性都不等于 L2 独立定位有效性。`diffuse_loss` 只表示当前参数下未检出集中下跌；占位 p0 的技能分不能证明分析器优越。

正式比较须冻结配置和任务家族、测量 M2 基线，并使用独立样本。`summarize` 只有在显式指定留出家族且具备配对 C/A 数据时才计算对应的升级证据。历史 `examples/` 不作为当前 v0.3 运行合同。
