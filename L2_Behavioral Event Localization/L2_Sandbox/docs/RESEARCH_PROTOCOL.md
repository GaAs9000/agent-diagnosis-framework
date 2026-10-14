# 研究验证范围（v0.2.1）

九份现有 OfficeVal003 记录属于同一个任务家族，主执行、恢复分支、重试不是独立任务。本版使用它们做接口和工程回归，所有规则输出都是明确的 deterministic baseline，不用这些输出宣称 Analyst 定位准确率。

## 一、先拆解比较问题

同一个基础模型，在原始日志、同权限结构化单次视图、受限检索 Analyst 上的差异需分开测量。当前发布 rules / structured-once / analyst；完整 raw-log 等权限比较需在相同读取预算与不截断前提下单独对比。每个方法实际看过哪些事实、读取了多少字节和发生多少截断应保留，不能把更多信息带来的收益全部归于方法。

规则事实可作为受控测例的位置参照，但自然轨迹中的错误原因没有自动人工因果标签。故障注入的位置标签不等于其对最终结果的因果贡献。

## 二、v0.2 两类可识别对比（无需信息注入接口）

在同一个非退化采样策略 $\pi$（$T = 1.0$、无固定 seed）和完整检查点状态 $c_k$ 下，不需要修改 L1 注入接口即可识别两类对比：

1. **跨检查点价值下跌（`evidence_basis: value_curve`，证据等级 E1）**
   - 比较 $D_{[a,b]} = V(c_a) - V(c_b)$ 与单回合下跌 $d_j = V(c_{j-1}) - V(c_j)$。
   - **预先声明的两阶段样本拆分规则（替代 v0.1 单阶段的 `no_adaptive_stopping`）**：
     - 第一阶段在固定步长网格 $c_0, c_s, c_{2s}, \dots$（与 Analyst 预测无关）上各续跑 $N_1 = 12$ 次，仅用于按 `argmax_interval_drop_earliest_tie` 选出下跌最大的网格区间 $[a, b]$。
     - 第二阶段在选中区间 $[a, b]$ 内的每个合格检查点各全新续跑 $N_2 = 24$ 次。所有统计检验与置信区间**仅使用第二阶段样本**计算，消除“先挑最大值再检验”的选择偏差。
   - **均匀摊派零假设（Uniform-Share Null）**：只分析失败轨迹时，实际路径每一步平均下跌 $(V(c_0) - Y)/T$。区间检验要求单侧下界超过均匀摊派份额 $\frac{b-a}{T}(\hat{V}_1(c_0) - Y)$；单回合检验要求经 Bonferroni 校正的单侧下界超过 $\delta_{\text{turn}} = \max(0.10, 2\hat{V}_1(c_0)/T)$。
2. **同一检查点内的同状态行为分层（`evidence_basis: resample_stratified`，证据等级 E1+）**
   - 在检查点 $c_{j-1}$ 重采样的 $N_2$ 条分支中，按重新采样的第一个回合 $j$ 是否出现预先冻结的 L2a 行为标签 $B$（`tool_execution_anomaly`、`submission_modified`、`assistant_finished_turn`、`submission_readback`、`object_structure_changed`）分组，比较条件期望差 $\Delta_B = \mathbb{E}[Y \mid c_{j-1}, a_j \in B] - \mathbb{E}[Y \mid c_{j-1}, a_j \notin B]$。
   - 特别地，同一状态下比较 `assistant_finished_turn` 与 `submission_readback` 可量化“不检查直接结束”相对“先回读交付物”的结果差。
   - 该对比度量同状态下两类动作条件分布的结果差异（E1+），不直接声称动作 $B$ 的独立干预效应（E2）。每类至少 4 条样本，多标签间做 Holm 校正；跨轨迹按任务家族做 Mantel–Haenszel 合并。

## 三、结果判定、缺失上下界与首请求一致性

1. **终点统一读取 L1 `projection`**：OfficeVal 主终点为 `official_completion_rate`（次终点 `official_dim1`，探索性分项为 `dim2` 评分项 `hit`）；LHTB 主终点为 `passed`。
2. **五类结果判定表**：
   - Agent 正常或因预算/协议停止且评分有效：按 `projection` 取值。
   - 执行有效但 `submission/` 为空：记为有效结果 `0.0`（`official_dim1` 记 `False`）。
   - 执行有效、有交付文件但评分器返回非 ok（文件损坏或缺少预期交付物）：主分析记 `0.0`，损坏无法解析的情形在敏感性分析中改记缺失。
   - 模型传输失败或执行中断（`execution_validity != valid`）：记为缺失（`infrastructure_invalid`）。
   - 评分器自身崩溃/超时或未运行（`official_raw` 为空）：记为缺失（`scorer_infrastructure_invalid`）。
3. **缺失上下界**：缺失分支最多按预声明规则补跑 2 次（保留全部失败回执）；统计复算同时报告缺失全记 `0` 与全记 `1` 的最坏情况上下界，若上下界翻转判定则输出 `not_identifiable`。
4. **首请求逐字节一致性**：每个续跑分支的第一个模型请求体必须与父轨迹在对应检查点后的同编号请求体逐字节相同，否则中止并记为 `policy_identity_violation`。

## 四、已提供的指标函数

- `evaluation.metrics.value_prediction_scores` 与 `evaluate_value_predictions_across_cases`：用第一阶段网格续跑分支的观测结果直接对冻结的价值预测计算 Brier / 有界平方误差，并报告相对**线性衰减基线（linear_decay）**的技能分（Skill Score）及按任务家族聚类的 bootstrap 95% 置信区间。
- `evaluation.metrics.curve_gold` 与 `localization_metrics`：将判定为 `concentrated_turn` 或 `concentrated_interval` 的价值曲线结果映射为 `accepted_event_sets`，评估 Top-k 事件集合精确匹配率；`diffuse_loss`、`out_of_reach` 与 `not_identifiable` 记为 `gold_undefined_not_scored`。
- `family_partition`：必须输入明确 `family_id`；只有一个 family 时拒绝产出貌似独立的 train/holdout 划分。

## 五、后续 E2 干预实验与 v0.3 范围

强策略接管（$V_{\text{strong}}$）、状态级不可逆点以及显式信息注入/动作替换（E2）留待 v0.3，在 L1 提供经审计的换策略续跑与显式注入接口后启用。在此之前，对应干预模板保持 `blocked`。
