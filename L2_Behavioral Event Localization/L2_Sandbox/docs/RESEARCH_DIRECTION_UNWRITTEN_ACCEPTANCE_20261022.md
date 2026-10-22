# 研究方向建议：写不出来的验收标准（2026-10-22）

> 状态：建议稿，待用户决策；未冻结，未授权任何新的 Actor、Analyst、Docker 或评分请求。本文不改写 V3 草案、P4 冻结计划、跨模型 D1 原件或任何历史判定。
>
> 来源：Kiro 对仓库材料、Notion（框架设计、V3 草案、V3 修订任务单、L1、L3、进度台账、V2 归档）与 2023–2026 年相关文献的审阅。文中新增数值均为对已有原件的零成本只读复算，复现方法见第 11 节。

## 1. 结论

1. V3 的主贡献"用状态复制的干预阶梯分解失败"已与 2026 年的公开工作高度重合（FIRE、Ask Early/Late、ORACLE-SWE、PSCI、2610.05572），不能再作为 novelty，只能作为动机和评测工具。
2. "瓶颈是规格而不是算力"在文本场景已由 PSCI、SpecHarness 确立，本项目只能做场景化、按信息来源细分的版本。
3. 仍可辩护的空位是职业交付物里**写不出来的验收标准**：由样例展示（"照图片""和原来一模一样"）或默认惯例给出、而不是写在题面上的要求。现有规格归纳方法都从文字中提取；现有复刻方法优化的是与参考的整体相似度。
4. 建议把第一篇的主线改为"问题（写不出来的验收标准）＋方法（验收编译）＋按来源的缺口闭合评测"，并先跑一个约 2,900 次请求的判定实验，再决定是否投入方法开发。

## 2. 项目内证据（零成本只读复算）

### 2.1 失败主要是系统性的

新 DEV45（Qwen3.6-27B＋L1 Pi harness context-v2，9 家族 × 5 条根轨迹）总扣分 34.9666。其中 29.8664（85.41%）来自"5 条独立根轨迹全部没做对"的 100 个评分项；其余约 14.6% 来自 5 条中有 1–4 条做对的项。逐轨迹定位关键步骤，最多只能解释后者。

5/5 失败不等于成功率为 0：单项真实成功率为 0.1 时，5 次全失败的概率为 0.59。但总体结构是清楚的：失败主要属于"模型 × 评分项"，而不是某条轨迹在某一步走错。

### 2.2 系统性失分按"所需信息在哪里"分类

人工归类，覆盖 9 个开发任务；033 的归类待核。

| 来源 | 评分项数 | 扣分 | 占系统性扣分 | 例子 |
|---|---:|---:|---:|---|
| 样例展示 | 92 | 16.9645 | 56.80% | 010、050 的流程图与幻灯片几何；003 的横线样式；004 图片中的字体格式；070 的字号范围 |
| 默认惯例 | 6 | 6.2353 | 20.88% | 090 要求跨表引用公式与 IFERROR，而不是静态数值；033（待核） |
| 题面写明 | 1 | 1.6667 | 5.58% | 003 "把没有字的空行删掉" |
| 无从得知（评测接口） | 1 | 5.0000 | 16.74% | 085 的输出文件名，见 2.3 |

9 个开发任务里有 6 个（003、004、010、050、070、085）用样例表达要求，例如"横线样式和图片中的一样""按照图片的格式要求""内容要和原来的一模一样""调整成像'临床问题'和'科学假说'这两块的布局"。

### 2.3 085 的文件名要求无从得知

- `officeval_085_verifier.py` 硬性要求 `1_美化版_S11可编辑图表.xlsx`、`2_美化版_S11可编辑图表.xlsx`、`3_美化版_S11可编辑图表.xlsx` 三份文件，缺失即抛出 `FileNotFoundError`，没有回退。
- 题面（`benchmarks/officeval/data/tasks/officeval_085.json`）只说"画完的图分别放置在数据对应的表格里"；官方 runner 的 `build_task_prompt` 没有提供文件名；官方归档按原相对路径复制改动文件，不重命名。
- 因此在官方协议下，085 的交付门槛对任何 Agent 都几乎不可达；新 DEV45 中 085 的 5 条全部因此记 0。
- 跨模型 D1 的 R_implicit 文本同样没有给出这三个文件名，只写了"交付文件数量为3个，分别对应用户提供的1、2、3三个原始表格文件"。所以 085 在 D1 各臂全为 0，主要原因是文件名，不应计作"本协议未恢复"。
- 102 个 verifier 文件中约 25 个含"缺少必需文档"类检查，但并非都没有回退：003 优先找精确文件名，找不到就取目录内任意 .docx；004 找不到精确文件名时按"报告""鉴定"关键字匹配。在 9 个开发任务中，只有 085 是没有回退的硬性文件名要求；其余命中需要逐个核实。
- 建议作为 Benchmark 效度问题记录（必要时向上游反馈），并把"无从得知"的项从能力结论中剔除、单列报告。

### 2.4 跨模型 D1 的描述性结果（探索性，不是同模型 D1）

Qwen 父状态 → DeepSeek V4.1 Flash 续跑，终态时间为 2026-10-21 15:11：计划 2,282 个分支，有效 2,184、无效 82、未启动 16，最后因 provider 429 停止；累计 20,896 次 HTTP，约 11.98 亿记账 tokens（其中约 4.10 亿为未知用量的保守预留），约 $158 用量等值（不是现金账单）。

24 个失败单元的同单元均值（P＝0.049）：

| 臂 | 均值 | 相对 P |
|---|---:|---:|
| D | 0.003 | −0.046 |
| S | 0.040 | −0.010 |
| R_explicit | 0.031 | −0.019 |
| R_implicit（仅归因） | 0.323 | +0.274 |
| R_single（19 个单元） | 0.311 | +0.249 |

对本建议有影响的五点：

1. 可部署臂（P、D、S、R_explicit，以及显式来源的 R_single）平均都不比 P 好。按单元挑最好的可部署臂，相对"一律 P"的样本内、不加权粗算收益约 0.02（交叉拟合只会更低），低于 D2 门槛 0.05；正式 D2 未计算。
2. 阶梯中间级基本为空：9 个任务中有 8 个只有 1 条"明确要求"，就是原题那句话（已核对 010、050、070）。V3 草案要求 R-明确包含 ambiguous 项的可见部分，实现没有放入。实际阶梯接近"什么都不给 → 给出完整评分规格"（R_implicit 文本 652–8,523 字符）。
3. R_single 的文本信息量不一致：090 是完整规则；050 只是短标签（如"+1 第1页arena中部竖向虚线"），而 R_implicit 中同一项带完整坐标与线宽。050 上 R_single＝0、R_implicit≈0.75 是消息构造差异。
4. D 在这批里有害（003.r00：P 0.538 → D 0.077；003.r03：0.500 → 0.000），而 S 和所有 R 臂都嵌套在 D 之后。
5. 010 的 5 个单元和 085.r03、085.r04 锚在第 29 回合、只剩 6 次请求，"救不回"里混有预算不足；085 的 5 个单元还受文件名问题影响（见 2.3）。

### 2.5 其他既有证据

- 新 DEV45 中 38/45 条主动结束；P1 的 9 条全部主动结束，只用了 7–18/35 次请求。
- 旧配置预算 pilot：70 对 35 的任务内平均完成度差为 −0.0155。

## 3. 什么在变化，哪个约束成为瓶颈，现有方法为什么跨不过去

正在变化的三件事：

1. 交付方式从对话里边做边确认，变成整份异步委托：请求者不在场，Agent 自己判断什么时候算做完。
2. 执行能力已经够用：Agent 能做出像样的交付物，并且相信自己做完了（SpecHarness：自称完成率比官方通过率高 28.7–37.9 个百分点）。
3. 验收变成细粒度清单（OfficeVal 平均每题约 20 条带容差的评分项），而请求者多用样例和惯例表达要求。

原本可以忽略、现在变成瓶颈的约束：Agent 能不能自己确定"做到什么程度才算完"，并且检查得了（验收闭合）。以前可以忽略，是因为短任务的规格写全了、代码任务有能跑的测试、对话式使用时人会随时纠正；这三条保障在异步委托下同时消失。

现有方法跨不过去的原因：

| 方法类 | 为什么不够 |
|---|---|
| 加算力、多采样、加预算 | 系统性失败在所有样本上都错；没有对齐的验证器就挑不出对的那个；Agent 会提前收工，多给的预算用不上 |
| 自我检查、反思 | 拿自己的理解当标准；FIRE 中"始终自查"臂与基线相同（39.3%），本项目的 S 臂同样无效 |
| 澄清提问 | Ambig-SWE、Ask Early/Late 都依赖可回答的 oracle 用户；异步委托下请求者不在场，而且往往说不清默认惯例 |
| 从文字抽规格（PSCI、SpecHarness、TICK、RLCF） | 标准不在文字里时抽不出来；PSCI 的规格 95.5% 能在文本中找到出处，SpecHarness 对含糊要求放弃强制 |
| 照参考整体复刻（VisRefiner、DCGen、DrawAI、SlideForge） | 优化整体相似度，不产出评分会逐条核对的具体量，也不处理"其余保持不变"和惯例；DrawAI、SlideForge 仅据摘要判断 |
| 失败归因（Who&When、AgenTracer、AgentDebug） | 假设存在一个出错的关键步骤；系统性缺口没有这样的步骤（2610.05572：同一任务内从轨迹几乎分不出成败） |

## 4. 文献调研与撞车判定

调研分 6 条线并行进行：轨迹归因、规格缺失与澄清、办公与视觉复刻、算力与终止、帮助与惯例记忆、专门找撞车。各线自报筛选约 40–140 篇候选（有重叠）。下表"核对深度"中，"全文"指本人核对过全文关键部分，"摘要"指只核对了摘要，"代理全文"指数字来自调研代理的全文阅读、未逐一复核。

| 工作 | 核对深度 | 做了什么 | 与本项目重合在哪 |
|---|---|---|---|
| FIRE（arXiv 2609.26048） | 全文 | 在曾出现失败的状态触发政策；随机五臂（14 个合格任务）：真实政策 60.7%，同时机假干预 35.7%，始终自查 39.3%，反思 42.9%，基线 39.3%；政策由研究者检查失败轨迹后手写，明确排除"缺能力"的失败 | 干预分解的实验设计；"通用自查无效、内容才有效" |
| Ask Early, Ask Late, Ask Right（arXiv 2605.07937） | 全文 | 在轨迹 10%–90% 处强制注入真实澄清（目标、输入、约束、上下文四类）；目标类信息过了 10% 几乎没价值，任何信息拖过中段都比不给更差；84 个任务变体、6,000+ 次运行 | 晚给信息的折扣；oracle 上限 |
| ORACLE-SWE（arXiv 2604.07789） | 全文（引言与设计） | 在开头注入 5 类 oracle 信息求上限，复现测试贡献最大；再用强模型提取这些信息作为可部署近似 | 按信息类型求上限，并给出现实近似 |
| PSCI（arXiv 2607.09794） | 全文 | CL-Bench 31,592 个评分项中 55.4% 考规格；76.7% 题面没写，但 95.5% 能在文本中找到；先归纳私有规格合同，再生成、对抗检查、修复，+5.59pp；算力匹配的通用批评不能解释增益；单轮长上下文、无工具 | "瓶颈是规格"；归纳→检查→修复 |
| SpecHarness（arXiv 2609.29921） | 全文 | 把 Agent 可见的文字规格编译成义务，没有合格证据就不许结束；评测用 verifier 不参与运行；SkillsBench 通过率 61.1%→73.1%；含糊要求标为建议或弃权 | 不看评分器、按验收标准控制收尾 |
| Disentangling Task Difficulty from Run-Level Failure（arXiv 2610.05572） | 摘要 | 失败预测 AUROC 的样本对 99.93% 以上来自不同任务；只看任务难度可达 0.945；同一任务内接近随机 | 失败以任务级为主 |
| GDPval（arXiv 2510.04374） | 代理全文 | 职业交付物；通用渲染自检＋best-of-N 让专家胜率约 +5pp；上下文变少时表现变差 | 渲染自检基线 |
| DrawAI（2608.00548）、SlideForge（2609.03109）、PPTBench（2609.29718）、ReFigBench（2609.18844） | 摘要 | 图像 → 可编辑对象的复刻 benchmark 与工作流，含渲染后校验 | 测量参考样例这一步的近邻 |
| TAHI（2609.04141）、Feedback-to-Rubrics（2605.29857）、ACE（2510.04618） | 摘要 / 代理全文 | 从人机交互、人工批注或执行反馈中学出评分标准 | 惯例记忆的近邻 |
| Fixed-Pool Diagnostic（arXiv 2607.17531） | 摘要 | 把净增益分解为 oracle 缺口、信号保真度和对已做对输出的伤害 | 缺口归一化与伤害率 |

已确立、必须引用、不可声称：

- 用状态复制或强制注入做干预分解，包括同时机假干预和通用自查对照（FIRE）；同一信息在不同时间点注入的时机效应（Ask Early/Late）；按信息类型注入 oracle 求上限及其可部署近似（ORACLE-SWE）。
- 失败以任务级为主，单次运行的信号很弱（2610.05572）。
- 规格而非内容或算力是瓶颈；先归纳规格合同再检查修复；算力匹配的通用批评解释不了增益（PSCI）。
- 依据可见规格控制收尾（SpecHarness、ECT 2608.23623）；通用渲染自检有小幅收益（GDPval）。
- 从反馈中学习隐性标准（TAHI、Feedback-to-Rubrics、ACE、RubricForge 2608.13564）；oracle 缺口归一化与伤害率（Fixed-Pool Diagnostic）。

仍开放、可以声称（需满足第 5 节的两个前提）：

- 把"由样例展示的验收标准"通过测量编译成带容差的可执行检查，并和"保持不变"检查一起用于修复和收尾。
- 从开发家族的评分反馈中提炼默认惯例，在未见过的家族上按适用条件迁移，不需要请求者在场。
- 按信息来源（写明、展示、默认、无从得知）报告缺口上限和闭合率，并报告对已做对项的伤害。

## 5. Novelty 论证（拟稿）

> 现有工作已经表明，Agent 的失败常常源于不知道"怎样才算做完"，而不是能力或算力不足；也已有方法把规格编译成合同或义务，用来约束生成和收尾（PSCI、SpecHarness）。但这些方法都从文字里提取规格：PSCI 的规格 95.5% 能在文本上下文中找到出处，SpecHarness 对含糊的要求直接放弃强制。职业交付物的验收标准往往不是这样给出的：请求者给一张图、一份原稿、一句"和原来一样"，或者默认对方懂行业惯例。在 OfficeVal 开发集上，系统性失分里约 57% 来自样例中展示的要求，约 21% 来自默认惯例，写明却没做到的只有约 6%。
>
> 我们把"写不出来的验收标准"作为一个独立问题来研究：(1) 用状态复制的干预实验，按信息来源量化缺口的上限，以及现有方法能闭合的比例；(2) 提出验收编译：通过测量把样例编译成带容差的可执行检查，从开发任务的评分反馈中提炼可迁移的惯例检查，再连同文字要求一起约束修复和收尾；(3) 在未见过的任务家族上报告按来源的缺口闭合率，以及对原本做对的项的伤害。和复刻类方法相比，我们产出的是评分会逐条核对的检查，并保持未修改的部分不变；和从人类反馈学习偏好的方法相比，我们不需要请求者在场。

这段论证有两个前提，需要先核实：

1. 读完 DrawAI 与 SlideForge 全文。如果它们已经实现"样例 → 逐条带容差的检查 → 修复"，样例部分就撞车了，贡献收缩为惯例迁移和按来源的缺口闭合评测。
2. 第 2.2 节的约 57% 来自 9 个任务的人工归类，需要在其余开发任务和已暴露的 9 个旧留出家族上复核。10 个未暴露的留出家族只用于最终确认，不用于复核。

## 6. 方法：验收编译（拟）

输入：任务说明、输入文件（含参考图与原稿）、从开发家族学到的惯例库。不使用评分器，也不需要请求者在场。

1. 归纳：按来源生成可执行检查。
   - 写明的：从题面抽取，沿用 PSCI、SpecHarness 的做法，只作为组件，不算贡献。
   - 展示的：先识别题面指向的样例（"照图片""和原来一样""像某某那块"）；再用确定性工具测量样例——读取文档结构（python-docx、python-pptx、openpyxl），渲染后用图像处理与 OCR 测量位置、尺寸、线型、字号、数量和连接关系——生成带容差的检查；另外对比原稿，为题面没要求修改的部分生成"保持不变"检查。
   - 默认惯例：从开发家族的评分反馈中提炼成带适用条件的检查，例如"提取到汇总表时使用跨表引用公式""比值列加除零保护"，在新家族上按适用条件启用。
2. 修复：把没通过的检查自动转成针对性的修复指令，作用类似 FIRE 的定向政策，但不需要人来写。
3. 收尾门控：检查全部通过或修复次数用完，才允许结束。
4. 回归保护：每次修复后重跑全部检查，防止把做对的项改坏。

不处理"无从得知"类要求（如 085 的文件名）；遇到时单列报告。

## 7. 评测设计

- 主指标：按来源的缺口闭合率 GCR_s =（方法 − 原题）/（只给来源 s 评分规格的 oracle − 原题）；同时报告总闭合率和伤害率（原题下做对、方法下做错的评分项比例）。
- 对照：原题、通用自查、GDPval 式渲染自检、PSCI 式文字合同、SpecHarness 式文字义务门控（能复现时）、整体视觉相似度修复、完整 oracle。
- 泛化：10 个未暴露的留出家族；第二个 Benchmark（展示类可考虑 PPTBench 或 DrawAI-Bench，需先确认能用）；第二个 Actor 模型。
- 消融：分别去掉样例编译、惯例库、收尾门控、回归保护。
- L1 检查点的用途：比较"开工前编译"和"收尾前编译"，直接测验收信息的时机（参照 Ask Early/Late）；在同一状态上分叉做伤害检查；对剩余失败做定位。

## 8. 第一个实验：判定"现有方法是否跨不过去"

可执行计划见 [ACCEPTANCE_GAP_EXP1_PLAN_20261022.md](./ACCEPTANCE_GAP_EXP1_PLAN_20261022.md)（2026-10-22）。与本节初稿相比，有四处改动：

- O 不再复用 DEV45 的 45 条，改为与 F/T/V 同期、随机交错地重新运行；DEV45 的 O 只用于漂移检查。原因是 DEV45 之后 L1 改过 3 个文件，GPU 服务也是重新启动的。
- 085 只跑 F。它的文件名要求无从得知，O/T/V 在那里必然为 0。
- 规模：5 个区块 × 33 条 = 165 条，预计约 3,500–4,600 次请求；每个区块单独申请一次 6 小时 GPU 租约。
- 条件文字放在系统提示里，以免被 Pi 的上下文压缩摘要掉。

条件、判据和停止规则都已冻结在 `configs/acceptance_gap_exp1_20261022.json` 中：

- F：完整评分规格，作为信息上限。
- T：PSCI 式文字合同。
- V：GDPval 式渲染自检。
- 判据：F−O ≥ 0.15 且 95% 下界 > 0，同时 T、V 中较好者闭合的差距不到 F−O 的三分之一，才进入方法开发。
- 停止：跑完 3 个区块后，F−O 的上界 < 0.10。

初稿中的可选条件 F_shown、F_assumed 暂不运行；是否需要，等 Exp1 结果出来再定。

## 9. 对现有计划的影响（建议，待用户决定）

- 同模型 D1 不按现设计运行（2,282 个分支，请求上界 34,622）；跨模型 D1 归档为探索性数据。
- 五类单元标签、D2 门槛、A/B/C 分析器移出第一篇；"LLM 读日志能否预测干预结果"留作后续工作。
- R24（接入发布方 scaffold）暂缓。
- 不再新增门禁。从头运行的条件实验不依赖冷恢复；只有"开工前编译 vs 收尾前编译"的比较需要 G1，而 G1 已在新 DEV45 上通过。
- V3 草案第八节的命题需要改写：尺子保留为测量工具；押注改写为"写不出来的验收标准是瓶颈"；推论（诊断价值上限）后移。
- 第 2.3 节的 085 问题应记入 Benchmark 效度说明。

## 10. 开跑前待办

- [ ] 用户决定是否采纳本方向，以及目标会议和截止时间。
- [ ] 确认 aml-b 上的 Qwen 服务可用（同模型实验依赖它）。
- [ ] 读完 DrawAI、SlideForge 全文，核实第 5 节前提 1。
- [ ] 逐个核对约 25 个含文件缺失检查的 verifier，区分有回退和没有回退的（9 个开发任务已核对：只有 085 没有回退）。
- [ ] 在其余开发任务和已暴露的 9 个旧留出家族上复核第 2.2 节的来源分类。
- [x] 写出 F、T、V 的提示文本和运行计划（2026-10-22，见 ACCEPTANCE_GAP_EXP1_PLAN_20261022.md）；待用户审阅后授权。

## 11. 复现

输入原件与 SHA-256（相对仓库根目录）：

| 原件 | SHA-256 |
|---|---|
| `L2_Behavioral Event Localization/L2_Sandbox/diagnoses/prospective_dev45_20261019/*/defects.json`（45 份；按"运行名 空格 sha256"逐行排序后整体哈希） | `fca3b8e8a590f7dc39965680d049909a3a973903767f834423cf097df82df0d9` |
| `L2_Behavioral Event Localization/L2_Sandbox/validation/opencode_length_repair_20261019/continuation_06/terminal_report/summary.json` | `f1c2e70a55b92f00b39a4e9560881e00010c24d5367e5c88081f30f58c8ab2d4` |
| `L2_Behavioral Event Localization/L2_Sandbox/validation/opencode_length_repair_20261019/continuation_06/terminal_report/report.md` | `b20acf5eb9c58caff8e3f1e03bbf0b82ca234cd385dfef9eb5b29bda03160e20` |
| `L2_Behavioral Event Localization/L2_Sandbox/validation/opencode_length_repair_20261019/continuation_06/execution_plan.json` | `4554d52aabc577927ffc73890d9a45522bbd13321332c0e8e6af6f4fb0f46757` |
| `L1_sandbox/C3_Experimental_Sandbox/third_party/OmegaUse-OfficeVal/verifiers/officeval_085_verifier.py` | `19261a63805762270bc5c1bcf23a9e68360cda7c559c8d087c1e997a68f57082` |
| `L1_sandbox/C3_Experimental_Sandbox/benchmarks/officeval/data/tasks/officeval_085.json` | `1173a0be04895506b741e6d728afe43cea45d0ab6f7fbf19349595c2752547a4` |
| `L1_sandbox/C3_Experimental_Sandbox/third_party/OmegaUse-OfficeVal/officeval_runner/src/runner.py` | `43c4ef26577be9984010d8dfcadbcbc443b0dd4b516f4817f002b423d1107c03` |

第 2.1–2.2 节的计算（在仓库根目录运行，只读）：

```python
import json, glob, collections

def source(task, item):
    # 人工归类规则（第 2.2 节）；033 待核
    if task == "officeval_085":
        return "unknowable"
    if task in ("officeval_090", "officeval_033"):
        return "assumed"
    if task == "officeval_003" and item == "文档表格中没有空白行":
        return "written"
    return "shown"

root = "L2_Behavioral Event Localization/L2_Sandbox/diagnoses/prospective_dev45_20261019"
by_task = collections.defaultdict(dict)
for f in sorted(glob.glob(f"{root}/*/defects.json")):
    run = f.split("/")[-2]
    by_task[run.split(".")[0]][run] = {
        d["defect"]: d["deduction"] for d in json.load(open(f))["defects"]
    }

total = systematic = 0.0
by_source = collections.Counter()
for task, runs in by_task.items():
    count, deduction = collections.Counter(), collections.defaultdict(float)
    for items in runs.values():
        for item, value in items.items():
            count[item] += 1
            deduction[item] += value
    total += sum(deduction.values())
    for item, c in count.items():
        if c == len(runs):  # 5 条根轨迹全部未做对
            systematic += deduction[item]
            by_source[source(task, item)] += deduction[item]

print(f"systematic {systematic:.4f} / total {total:.4f} = {systematic / total:.2%}")
for s, v in by_source.items():
    print(s, f"{v:.4f}", f"{v / systematic:.2%}")
```

第 2.4 节的数值读自终态报告；"约 0.02"的可部署选择收益按报告逐单元表，对每个单元取 P、D、S、R_explicit（003 另含 R_single）中的最大均值减去 P 均值，再对 24 个单元等权平均。

## 12. 参考文献

本人核对全文关键部分：

- FIRE: Failure-Informed Runtime Engineering for Reliable Language-Model Agents. arXiv 2609.26048.
- Agentic Context Learning with Self-Discovered Specification（PSCI）. arXiv 2607.09794.
- Who Holds the Pen? Let Specifications, Not Agents, Sign Off（SpecHarness）. arXiv 2609.29921.
- Ask Early, Ask Late, Ask Right: When Does Clarification Timing Matter for Long-Horizon Agents? arXiv 2605.07937.
- ORACLE-SWE: Quantifying the Contribution of Oracle Information Signals on SWE Agents. arXiv 2604.07789.

本人核对摘要：

- Disentangling Task Difficulty from Run-Level Failure in Agent Failure Prediction. arXiv 2610.05572.
- Agentic Rubrics as Contextual Verifiers for SWE Agents. arXiv 2601.04171.
- Inducing Reward-Free Judging Rubrics that Reduce Over-Crediting in Agent Evaluation（RubricForge）. arXiv 2608.13564.
- Oracle Gap and Signal Fidelity: A Fixed-Pool Diagnostic for Test-Time Collaboration. arXiv 2607.17531.
- Implicit Intelligence: Evaluating Agents on What Users Don't Say. arXiv 2602.20424.
- DrawAI: Agentic Benchmark and Workflow for Making Raster Images Editable. arXiv 2608.00548.
- SlideForge: An LLM Agent for Controllable Editing of Slides as Structured Artifacts. arXiv 2609.03109.
- PPTBench: Can Coding Agents Reconstruct the Visual World through Structured, Editable Slides. arXiv 2609.29718.
- ReFigBench: Benchmarking Scientific Figure Reconstruction as Editable PowerPoint Artifacts. arXiv 2609.18844.
- Feedback-to-Rubrics: Can We Learn Expert Criteria from Inline Comments? arXiv 2605.29857.
- Efficient Test-Time Adaptation through Human-AI Interaction（TAHI）. arXiv 2609.04141.

调研代理全文阅读、本人未逐一复核：

- GDPval. arXiv 2510.04374.
- What Prompts Don't Say: Understanding and Managing Underspecification in LLM Prompts. arXiv 2505.13360.
- Interactive Agents to Overcome Underspecificity in Software Engineering（Ambig-SWE）. arXiv 2502.13069.
- Judged Useless, Queried Anyway. arXiv 2610.06191.
- From Confident Closing to Silent Failure: Characterizing False Success in LLM Agents. arXiv 2606.09863.
- When May an Agent Stop? Evidence-Carrying Termination for Tool-Using LLMs. arXiv 2608.23623.
- The Limits of Inference Scaling Through Resampling. arXiv 2411.17501.
- Causal Agent Replay. arXiv 2606.08275；Counterfactual Rollout Replay. arXiv 2609.33875.
- Thinking Is Not Telling: Information Disclosure in User-Service LLM Agents. arXiv 2602.07796.
- Who&When（arXiv 2505.00212）；AgenTracer（2509.03312）；AgentDebug（2509.25370）；AgentDebugX（2607.18754）.
- OmegaUse-OfficeVal. arXiv 2607.27155.
- Agentic Context Engineering（ACE）. arXiv 2510.04618.
- Evaluating Frontier Models for Dangerous Capabilities. arXiv 2403.13793.
- VisRefiner（2602.05998）；DCGen（2406.16386）；AutoPresent（2501.00912）；PPTAgent（2501.03936）.
- Reward Hacking as Equilibrium under Finite Evaluation. arXiv 2603.28063.
- TICK（2410.03608）；Checklists Are Better Than Reward Models（2507.18624）.
