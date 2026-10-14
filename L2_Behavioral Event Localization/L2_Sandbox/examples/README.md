# 示例与资格边界

## 九份实际 L1 输入的离线规则基线

| 目录 | 原始输入 | 模式 |
|---|---|---|
| officeval003/analysis_000 | 3.5 Flash Lite 原恢复分支 | rules，非分析模型 |
| officeval003/analysis_001 | 3.5 Flash Lite cp000030 有状态分支 | rules，非分析模型 |
| officeval003/analysis_002 | 3.5 Flash Lite 主执行 | rules，非分析模型 |
| officeval003/analysis_003 | 3.8 Flash 主执行 | rules，非分析模型 |
| officeval003/analysis_004 | 3.1 Pro 恢复分支 | rules，非分析模型 |
| officeval003/analysis_005 | 3.1 Pro 主执行 | rules，非分析模型 |
| officeval003/analysis_006 | Qwen2.5-32B 恢复分支 | rules，非分析模型 |
| officeval003/analysis_007 | Qwen2.5-32B 主执行 | rules，非分析模型 |
| officeval003/analysis_008 | Llama-3.1-8B 主执行 | rules，非分析模型 |

合计 687 条记录/派生事实、20 个包内文档版本解析、24 个规则候选定位、16 个未验证候选解释。跨包版本可能重复，不是20个独立任务或唯一文件。九份来源属于一个任务家族，不能用于独立泛化准确率声明。

每份可 `c3-l2 verify <目录>`，引用字节子集随包保存；未重新运行用户的模型或官方评分器。原始模型文件按需可读的声明不等于它们已经全部复制进诊断包。

## 其他示例

`analyst_loop_fixture`：显式 test_fixture，模拟模型端口完成检索→原生调用→输出校验→冻结预测。所有模型响应来自确定性测试代码，没有真实 API 或本地权重推理。它用来展示 Analyst trace 的结构，不能当作 M2 真实运行收据。

`legacy_fixture`：使用用户此前提供的 P1 导入器对其合成 fixture 生成 L1 0.1 包，再经 L2 兼容层读取。仅4条保留记录，无对象快照、定位或机制；不补造缺失。

`plans/unchanged_baseline`：针对实际 Lite 主执行的 cp000030 冻结的三次原样续跑计划，未授权执行，也没有执行结果。需要原始完整 L1 包和本机 L1/Docker/模型配置。

`plans/information_intervention_blocked`：明确信息干预接口尚未支持的计划，状态 blocked。没有伪造信息注入实验或支持结论。

所有13个 L2 包均在本次环境实际校验。原始 `/mnt/data/...` 路径只是生成时的来源提示；没有原 L1 包仍能验证被嵌入的引用子集，但不能据此重建完整运行或恢复环境。
