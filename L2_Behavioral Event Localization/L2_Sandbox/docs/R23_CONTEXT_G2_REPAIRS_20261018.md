# 上下文、资源释放与 G2 启动依赖修复（2026-10-18）

本轮是离线工程修复，未新发 Actor／Analyst 请求，未启动 GPU。R22 的原始证据与正式口径仍为 44/45 有效；004.r02 的 HTTP 400 保留为无效，不改评分、不重采 050／010。旧批累计模型 HTTP 仍为 181，G2 为 0，D0 未冻结。

## 上下文管理

新配置 `officeval.qwen36.pi.t1.native-validation.context-v2.timeout600.json` 显式冻结
`harness_revision=l1-pi-native-errors-context-v2`；参数校验仍为 Pi 0.73.1 原生路径。
旧 v1 配置仍可读取验封，开启压缩的新执行必须迁移到 v2。

每次成功响应的 prompt usage 绑定到实际发送的消息列表哈希，压缩后仍保留校准，
不再永久忽略服务端真实 token 数。同一请求的旧 usage 不重复套到压缩后历史。
估计保留观察到的文本比例与图像额外开销高水位，含 10% 经验余量；它是预算启发式，不能保证任意新图像的精确 token 成本。

缩短只在完整 assistant／tool 批次边界进行；保留系统、初始任务、最新完整回合与原生 call ID。
每个候选都核对完整保留历史估计值；不可满足时明确停止为 `pi_context_unsatisfiable`，不冒称成功，也不把完整证据误标成丢失。
模型曾输出的数组／null／数值参数仍原样保存在轨迹里，摘要处理不再因此崩溃。

真实 004 的只读回放：原估计 42,369，下一次真实 prompt 114,090；新策略校准前估计 125,500，选择完整边界后保留 15 条消息，估计 27,856。
这不是新模型运行，不意味着 004 已补齐或官方得分提高。脚本化执行—压缩—冷恢复另做请求字节一致性验证。
G4 重建同步使用相同 usage 类型及消息来源；历史旧压缩轨迹需绑定旧控制器，缺少时保持 unverified。

## GPU 释放

实际故障是 EngineCore 改写进程标题后，`/proc/environ` 不再暴露继承的 owner 标签。
新清理从带标签的 timeout／API 进程核验精确命令、UID、PGID、父子关系、启动时间和 GPU UUID，
允许经证实的无标签 EngineCore，使用 pidfd 在再次核验身份后发送有界 TERM。
同时处理真实的 nice exec、Python shebang 和 EngineCore 标题空格；未知／异主／重用／重挂接进程均拒绝。
未消失的进程或 GPU 占用记 `cleanup_unconfirmed`。

新增 `start_owned_native_service.py`，默认不执行；显式执行需已冻结阶段输入。
保留 CUDA 13 与现有 conda 编译工具链、原始绝对租约、时钟余量和终止宽限；回执新增 `leader_start_ticks`。
回归使用归档的真实命令与检查记录；历史缺失的 birth ticks／父子图在测试中明确为合成字段。
未为验证清理而重启真实 GPU 服务。R22 当时的手动释放回执仍独立保留。

## G2 只等待原定 20 槽

G2 eligibility 与 DEV45 完整性分开。冻结的 20 槽不变，004.r02 不在其中；
010.r00 等任何所需槽缺失／无效／身份不符仍阻断，公共源码与依赖也必须匹配。
统计公式、阈值、G1／G4 结论均不改；eligible 不等于 G2 通过。

G2-only 新阶段通过 `--cohort` 和 `--predecessor-phase` 绑定旧 R22 的计划、freeze、result、
37/0 HTTP admission、三条续接回执、cohort、源配置和全部原件哈希。
保留 v1／legacy 的参考终点；context-v2 只用于新 G2 Actor，报告为恢复与 harness 的联合比较，不能称纯恢复等价。
不重放已完成的 050／010，也不把 invalid004 改成正式样本。

G2-only continuation 请求上限为 0；原 20 槽最多 700 次逻辑请求／2,800 次 HTTP，
含旧 181 次累计最多 2,981，仍小于原 3,360 上限。
默认使用新的 R23 run root；在创建锁、计划或运行标记前，拒绝写入任何旧不可变目录及旧阶段目录。
新阶段仍要求独立计划、源码封签及执行 freeze，旧阶段不得自动重启。

真实旧 cohort 的只读就绪核验已完成：**20/20 ready、eligible=true**，Actor 请求为 0。
预备计划 SHA-256：`49ac604876582de7b69fda7e5d63c0c5ff237ba4d676460aa8eb65d59dda20e9`。
状态为 planned_not_executed，不把就绪核验记作 G2 实跑或 D0 通过。

验收结果见同目录 `R23_REPAIR_VERIFICATION_20261018.json`。源码审阅通过不替代后续真实 G2 或 D1 的研究验证。
