"""Plain Markdown reports; claim wording comes from explicitly labelled outputs."""
from __future__ import annotations

def report(manifest,diagnosis,facts,coverage):
    lines=['# L2 行为定位与干预预测报告','',f"分析 ID：`{manifest['analysis_id']}`",'',
        f"来源 Episode：`{manifest['source_episode_id']}`",f"来源索引 SHA-256：`{manifest['source_index_sha256']}`",'',
        f"运行模式：**{manifest['mode']}**；状态：**{manifest['status']}**。",'',
        '**所有候选解释及干预预测均未经过本次分析中的实验验证。结构和引用可核验，不等于自然语言结论被证实。**','',
        diagnosis['summary'],'','## 定位结果','']
    for loc in diagnosis['localizations']:
        lines += [f"### {loc['localization_id']} · {loc['category']}",'',loc['statement'],'',
            f"定位粒度：`{loc['granularity']}`。事件："+(', '.join('`'+x+'`' for x in loc['event_ids']) or '无法定位到单一历史事件。'),'',
            '支持事实：'+', '.join('`'+x+'`' for x in loc['supporting_fact_ids']),
            '反证引用：'+(', '.join('`'+x+'`' for x in loc['counterevidence_fact_ids']) or '未提供；不表示反证不存在。'),'',
            '限制：'+'；'.join(loc['uncertainty']),'']
    if not diagnosis['localizations']:lines+=['没有通过引用检查的定位结果。','']
    lines+=['## 候选机制与预测','']
    for hyp in diagnosis['hypotheses']:
        lines += [f"### {hyp['hypothesis_id']} · untested",'',hyp['mechanism_statement'],'',
                  '竞争解释：'+'；'.join(hyp['alternatives']),'',
                  '必要假设：'+'；'.join(hyp['assumptions']),'',
                  '未知项：'+'；'.join(hyp['unobserved_requirements']),'']
        for p in hyp['predictions']:
            lines += [f"干预 `{p['intervention_template']}` 对照 `{p['control_template']}`；预期方向 `{p['direction']}`。",
                      p['proximal_prediction'],p['endpoint_prediction'],p['uncertainty'],'']
    lines+=['## 弃权与覆盖','',*['- '+x for x in diagnosis['abstentions']],'']
    for x in coverage:lines.append('- '+str(x))
    lines+=['','## 复核入口','',
            '`facts.jsonl` 保存记录/派生事实；`references.json` 定位原始文件、JSON Pointer 或字节范围。',
            '`source_blobs/` 只包含被引用的源字节，不是完整 L1 包；引用锚定原索引。',
            '`derived/objects/` 为本次只读 OOXML 解析，不能冒充当时送达模型的 Observation。',
            '`raw/analyst/` 与 `analyst_trace.jsonl` 保留所有请求、检索与失败尝试。rules 模式没有模型请求。',
            '`predictions/frozen_manifest.json` 固定本次预测摘要；未来实验必须新建验证包，不能改写原预测。','']
    return '\n'.join(lines)
