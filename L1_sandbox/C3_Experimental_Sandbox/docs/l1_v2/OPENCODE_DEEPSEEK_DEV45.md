# OpenCode Go / DeepSeek V4.1 Flash 的 L1 开发基线

本批次使用 L1 真实执行 OfficeVal，生成后续 L2 使用的父轨迹、官方评分和密封证据。L3 runner 仅负责冻结、调度和汇总；不调用 L2 Analyst。

## 冻结配置

- L1 配置：`configs/l1/officeval.opencode-go-deepseek-v41flash.t1.json`。
- 批量规范：`L3_Agent Capability Boundary Discovery/L3_Sandbox/configs/l1_deepseek_dev45.json`。
- 模型：`deepseek-v4.1-flash`；端点 `https://opencode.ai/inference/go/v1/chat/completions`，Bearer 凭据由本机 `OPENCODE_API_KEY` 提供。
- 温度 1，无模型 seed；显式 `thinking.type=disabled`，因为上游 thinking 模式不使用 temperature。每响应最多 8192 输出 tokens；图像工具开启。
- 九个开发家族代表任务：003、004、010、033、050、060、070、085、090；每任务 5 次独立初态，共 45 个预定槽。
- 每槽 35 次模型请求、140 次工具调用、10800 秒；并发 2、RPM 30、TPM 10000000；批次 HTTP 准入硬上限 6300，失败尝试和有界替换全部保留。
- 每个 episode 独立生成稳定的 `x-opencode-session`，重试、工具回环和冷恢复保持一致；会话标识不进入模型请求正文。
- 实际返回模型名必须匹配冻结的 `expected_reported_model`；fingerprint 单独记录。执行前／续跑前核验实际加载的适配器类、模块文件 SHA、entry point 和安装分发身份。

这些是本批配置与 API 返回身份的约束，不是模型权重版本或温度实际采样行为的服务端证明。

## 执行与停止

以下路径相对于仓库根目录。先完成本机 doctor、真实 API 工具／图像回环预检，再冻结正式计划；预检单独记账，不计入 45 条基线。

```bash
B=L1_sandbox/C3_Experimental_Sandbox/.venv/bin
D=L1_sandbox/C3_Experimental_Sandbox/runs/deepseek_dev45_20261015

$B/c3-l1 doctor --config L1_sandbox/C3_Experimental_Sandbox/configs/l1/officeval.opencode-go-deepseek-v41flash.t1.json
$B/l3 plan 'L3_Agent Capability Boundary Discovery/L3_Sandbox/configs/l1_deepseek_dev45.json' --output "$D"
$B/l3 check-plan "$D"
$B/l3 run "$D"
$B/l3 run "$D" --execute
$B/l3 summarize "$D"
```

执行器从忽略的 dotenv 文件读取凭据，不需要将密钥写进命令、配置或实验日志。Go 的工作区地区路由必须允许该模型；本次早期预检曾返回“requires Global regions”，该次 HTTP 400 单独保留。Go 订阅用量和可能存在的 Console 余额回退由服务端账号设置决定，不能仅凭端点推断实际扣款金额。

401／403、402／支付错误、终态重试耗尽的 429 或显式模型身份不匹配触发持久停止；不继续消耗排队槽位与替换。重试和 HTTP 消耗在准入账本中累计。进程中断后使用原计划续跑，健康已完成槽位从封存证据核验复用。

## 交付验收

- 45 个预定槽各有一条独立、健康的完整执行及官方评分记录；Agent 未交付或任务未通过仍按统一研究终点记录，不按得分筛掉失败。
- 全部原始尝试保留，包含请求／响应／HTTP 回执、工具输出、状态快照、检查点、交付文件、原始官方评分、manifest 和 evidence index。
- 独立复核全部包的封签、9×5 槽位覆盖、请求目标与返回模型、会话一致性、配置／执行代码／任务资产／镜像绑定；不得用测试替身冒充真实模型执行。
- HTTP 回执与准入账本核对，失败和未知用量单列；协议预检与正式批次分开计数，实际费用无账单证据时保持未知。
- 从全 9 个完整同版本 cohort 导出 p₀，逐父轨迹排除自身；仅有 `exported=true` 不能代替 45/45 完整性检查。
- 运行目录全部位于 L1；Git 只保留框架、配置、测试和说明，不提交凭据、题库、轨迹或交付件。未达到上述条件时不声称 45 条交付完成。

## 接口依据

- [OpenCode Go 接口、地区与客户端要求](https://opencode.ai/v2/docs/console/go/)
- [OpenCode Inference API](https://opencode.ai/v2/docs/console/api/inference/)
- [DeepSeek thinking mode](https://api-docs.deepseek.com/guides/thinking_mode)

离线预检记录在本机 `validation/deepseek_dev45_preflight_20261015/`；本说明不等于该批次已经执行。
