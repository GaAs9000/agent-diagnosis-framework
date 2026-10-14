# L2 v0.3 · 行为事件定位

本模块读取 L1 父轨迹，生成事实、预测检查点价值，测量两阶段续跑曲线，并检验行为谓词。事实、预测和检验结论分开；测试替身不构成真实模型或研究有效性证据。

Python 3.11+，依赖 lxml。安装与离线验证：

```bash
python -m pip install -e '.[test]'
python -m pytest -q
l2 --help
```

只有两个命令：

```bash
# 无模型费用：本地规则预测；请求的外部分析器显示 deferred。
l2 run /path/to/parent --output runs/episode --analysts rules,llm:configs/analyst.gemini.json

# --execute 授权分析器调用及 L1 续跑费用；需要已有 L1 环境。
l2 run /path/to/parent --output runs/episode --analysts rules,llm:configs/analyst.gemini.json --p0 0.5 --execute --rpm 60 --tpm 100000

l2 summarize runs/episode1 runs/episode2 --holdout-family family1 --output runs/summary.json
```

`c3-l2` 是同一入口的别名。规则组 A 始终存在。`llm_raw:<config>` 是 B，`llm:<config>` 默认是 C（原始轨迹加事实）。配置禁止内嵌密钥，只能写环境变量名称。提供的配置模型名与服务地址需由使用者确认；安装不会启动模型服务或下载模型。

`--p0` 是外部提供的有限 [0,1] 基线；`--baseline-provenance` 记录来源，程序不声称已完成 M2 队列采样或验证父轨迹排除。`--config` 可提供 `curve.rpm`、`curve.tpm`，CLI 选项优先。参照组 R 使用 `--analysts reference --reference-scores scores.json`；JSON 必须含每个检查点 `0..T-1` 的实际官方快照得分，缺失时明确 unavailable，不使用代理值。

输出包含 plan、facts、values、branches、outcomes、curve、hypotheses、scores 和 report。输出必须与源目录互不包含；不匹配已有计划的父轨迹、策略或执行参数会拒绝复用。分析器缓存绑定输入、配置、提示词和代码指纹，旧分析器结果不会混入当前评分。分析器只接收父轨迹及允许的曲线摘要，不接收分支行为或逐分支结果。

跨任务汇总只有显式留出家族的配对 C/A 数据才支持研究选择判据；未指定留出家族或数据不足时保留未知。随包 `examples/` 与历史文档是 v0.1 的冻结记录，不能按 v0.3 API 校验，也不证明当前实现已进行真实模型实验。当前设计见 `docs/DESIGN_INPUT_v0.3.md`。
