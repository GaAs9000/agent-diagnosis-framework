# L2 v0.3 · 行为事件定位

本模块读取 L1 父轨迹，生成事实、预测检查点价值，测量两阶段续跑曲线，并检验行为谓词。事实、预测和检验结论分开；测试替身不构成真实模型或研究有效性证据。

Python 3.12+，依赖 lxml 和本仓库中的 L1 包。即使只做离线分析，也需要 L1 的统一结局判定模块。在 `L2_Sandbox/` 下联合安装本地包并离线验证：

```bash
python -m pip install -e '../../L1_sandbox/C3_Experimental_Sandbox[l1]' -e '.[test]'
python -m pytest -q
l2 --help
```

上述命令从本地路径安装 L1，不依赖同名包已发布到 PyPI。缺少 L1 时应先执行联合安装命令；真实续跑另需 Docker 镜像、任务资产及模型凭证。

常用命令：

```bash
# 无模型费用：本地规则预测；请求的外部分析器显示 deferred。
l2 run /path/to/parent --output runs/episode --analysts rules,llm:configs/analyst.gemini.json --p0 0.5

# --execute 授权分析器调用及 L1 续跑费用；需要已有 L1 环境。
l2 run /path/to/parent --output runs/episode --analysts rules,llm:configs/analyst.gemini.json --p0 0.5 --execute --rpm 60 --tpm 100000

l2 summarize runs/episode1 runs/episode2 --holdout-family family1 --output runs/summary.json
```

`c3-l2` 是同一入口的别名。规则组 A 始终存在。`llm_raw:<config>` 是 B，`llm:<config>` 默认是 C（原始轨迹加事实）。配置禁止内嵌密钥，只能写环境变量名称。提供的配置模型名与服务地址需由使用者确认；安装不会启动模型服务或下载模型。

`--p0` 与 `--baseline <L3计划目录或绑定的导出JSON>` 二选一，包括 dry-run。`--p0` 是外部提供的有限 [0,1] 基线；示例中的 0.5 仅演示语法，不能替代 M2 测量；`--baseline-provenance` 记录外部来源，程序不声称验证过该外部值。`--baseline` 需要联合安装本仓库的 L3 包，仅在使用此选项时导入 L3；它从冻结计划、receipt 和密封原始证据重算，检查任务／配置／模型／评分终点版本，并排除父轨迹自身。无源证据绑定的独立 JSON 会拒绝，修改缓存汇总不能改变已验证的 p₀。

`--config` 可提供 `curve.rpm`、`curve.tpm`，CLI 选项优先。参照组 R 使用 `--analysts reference --reference-scores scores.json`；JSON 必须含每个检查点 `0..T-1` 的实际官方快照得分，缺失时明确 unavailable，不使用代理值。

输出包含 plan、facts、values、branches、outcomes、curve、hypotheses、scores 和 report。输出必须与源目录互不包含；不匹配已有计划的父轨迹、actor 配置或采样网格会拒绝复用；统计阈值、成本、并发和代码版本变化可复用已付费分支，并从 L1 包重算结局与曲线。规则与参考分析器每次重算；LLM 缓存只绑定实际请求输入、模型配置和提示词。cache_only=True 可禁止因分析变化新增采样。L1 负责完整证据封存，L2 只核对访问分支的 lineage 与 actor，不重复扫描全部历史或归档旧派生产物。分析器只接收父轨迹及允许的曲线摘要，不接收分支行为或逐分支结果。

跨任务汇总只有显式留出家族的配对 C/A 数据才支持研究选择判据；未指定留出家族或数据不足时保留未知。随包 `examples/` 与历史文档是 v0.1 的冻结记录，不能按 v0.3 API 校验，也不证明当前实现已进行真实模型实验。当前设计见 `docs/DESIGN_INPUT_v0.3.md`。
