# 上游依据与归档范围

本轮查阅的官方资料（2026-10-12），源码以历史固定 Commit 为主：

- OfficeVal README / 数据与平台要求：
  https://github.com/baidu-frontier-research/OmegaUse-OfficeVal/blob/5f3da616ac6a0bc30ca35e9678f8f1eb255bd91a/README.md
- OfficeVal 原始结果合同：
  https://github.com/baidu-frontier-research/OmegaUse-OfficeVal/blob/5f3da616ac6a0bc30ca35e9678f8f1eb255bd91a/docs/result-format.md
- OfficeVal Verifier 接口：
  https://github.com/baidu-frontier-research/OmegaUse-OfficeVal/blob/5f3da616ac6a0bc30ca35e9678f8f1eb255bd91a/docs/verifier-contract.md
- OfficeVal 数据发布位置：
  https://huggingface.co/datasets/baidu/OmegaUse-OfficeVal
- Google Gemini 原生函数调用：
  https://ai.google.dev/gemini-api/docs/function-calling
- Google Gemini Thought Signatures：
  https://ai.google.dev/gemini-api/docs/thought-signatures
- vLLM OpenAI-compatible server / 原生工具调用：
  https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
  https://docs.vllm.ai/en/latest/features/tool_calling.html

查阅网页不等于成功取得、构建、运行完整上游仓库。本次容器 `git ls-remote` DNS 失败，未取得新下载源码/数据；不能将网页核对代替本机 provenance 资格。HF 当前 main 与默认历史 dataset_revision 不同，配置故意不自动升级。

随包原项目原始 Bundle 中的 `source_metadata`（包括 LICENSE/README/依赖元数据）原样保留。没有补发第三方字体、模型权重或完整第三方仓库。第三方各自许可继续适用；本工程整合不重授上游内容许可。
