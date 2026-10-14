"""Deterministic non-LLM reference baseline, not an Analyst reasoning result."""
from __future__ import annotations

def diagnose(builder,max_findings=12):
    facts=builder.facts;by_type={}
    for f in facts:by_type.setdefault(f['fact_type'],[]).append(f)
    task=next(iter(by_type.get('task_instruction',[])),None)
    localizations=[];hypotheses=[];abstentions=[]
    terminal=next(iter(by_type.get('termination',[])),None)
    end_events=terminal['event_ids'] if terminal else []
    def add(category,statement,selected,*,events=None,uncertainty=None,object_scope=None,before=None,after=None,mechanism=None,alternatives=None,template=None):
        if len(localizations)>=max_findings:return
        if terminal and events is None: selected=[*selected,terminal]
        ids=list(dict.fromkeys(f['fact_id'] for f in selected))
        ev=events if events is not None else end_events
        lid=f'loc_{len(localizations):03d}'
        localizations.append({'localization_id':lid,'category':category,
            'granularity':'event' if len(ev)==1 else 'event_set' if ev else 'terminal','event_ids':list(dict.fromkeys(ev)),
            'statement':statement,'object_scope':object_scope or {},'before_version':before,'after_version':after,
            'supporting_fact_ids':ids,'counterevidence_fact_ids':[],
            'task_relation':{'requirement_fact_id':task['fact_id'] if task else None,
                             'interpretation':'与公开任务交付或执行过程有关；不是唯一因果归因。','origin':'analyst_interpretation'},
            'uncertainty':uncertainty or ['规则基线定位的是记录边界，不是已验证的错误原因。']})
        if mechanism:
            predictions=[]
            if template:
                predictions=[{'intervention_template':template,'control_template':'resume_unchanged','direction':'unknown',
                    'proximal_prediction':'目标相关支持是否改变下一次对象检查/操作，须由预先冻结的实验测量。',
                    'endpoint_prediction':'不预测必然修复；与重新采样的未干预组比较。','uncertainty':'规则基线没有估计效应；当前仅形成可检验候选。'}]
            hypotheses.append({'hypothesis_id':f'hyp_{len(hypotheses):03d}','localization_ids':[lid],
                'mechanism_statement':mechanism,'supporting_fact_ids':ids,'counterevidence_fact_ids':[],
                'alternatives':alternatives or ['现有证据不足以唯一判别生成、工具接口、任务理解或预算因素。'],
                'assumptions':['所引用记录在声明范围内有效；非观察到的内部认知不作事实判断。'],
                'unobserved_requirements':['需要受限干预与重新执行的对照来检验机制。'],
                'predictions':predictions,'abstention_reason':'无干预结果，不能确认机制。','verdict':'untested',
                'scope':'this_episode_and_explicit_future_protocol_only'})
    proto=by_type.get('native_protocol_signal',[])+by_type.get('protocol_error',[])
    if proto:
        add('protocol_termination','原生响应或控制器记录了协议异常，定位到对应响应/终止事件；不推断文档理解能力。',proto,
            events=[e for f in proto for e in f['event_ids']],
            mechanism='协议异常与本次续跑中断相伴；其产生机制尚不能唯一识别。',
            alternatives=['响应生成或服务端函数格式校验产生异常。','适配器兼容或协议支持差异可能影响异常处理，需要原生协议对照。'])
    failures=[f for f in by_type.get('tool_exit',[]) if f['value']['exit_code'] not in (0,None) or f['value'].get('stop_reason')]
    if failures:
        add('tool_execution_anomaly',f'记录了 {len(failures)} 个非零退出或停止原因的工具结果；shell 字符串中的每个子命令不视为均已执行。',
            failures,events=[e for f in failures for e in f['event_ids']],
            mechanism='工具级异常可能中断所需操作；对最终任务结果的独立贡献未验证。',
            alternatives=['同一命令中的早期步骤失败，使后续步骤未执行。','异常可能已被后续操作绕开；任务还可能有其他独立阻塞。'])
    inv=next(iter(by_type.get('submission_inventory',[])),None)
    if inv and not inv['value'].get('files'):
        add('submission_missing','最终提交清单没有文件；此处只定位终态观察点，不回填最早失败动作。',[inv],
            mechanism='终态缺少提交文件；预算、执行策略、工具失败或任务要求仍是竞争解释。',
            alternatives=['预算到期前没有完成所需操作。','工具命令失败或提交路径使用不当。','任务操作需求与当前工具/策略不匹配。'],template='generic_reminder')
    grade=next(iter(by_type.get('official_grade',[])),None)
    if grade:
        raw=grade['value'].get('official_raw') or {}
        score=raw.get('total_score'); maximum=raw.get('max_score')
        if raw.get('status')=='ok' and isinstance(score,(int,float)) and isinstance(maximum,(int,float)) and maximum>0 and score<maximum:
            related=[grade]
            if inv:related.append(inv)
            related.extend(f for f in by_type.get('object_property',[]) if any(x['file_path'].startswith('submission/') for x in f['value']['locations']))
            add('final_score_gap',f'原官方返回 {score}/{maximum}，正常评分/可打开性不等于满足全部任务约束。',related,
                mechanism='提交文档未取得全部官方规则分；哪些操作形成当前状态需结合对象版本和后续对照判断。',
                alternatives=['目标对象确实未按要求改变。','直接 XML 属性与实际渲染或评分口径存在差异。','后续修改覆盖了曾满足的状态；需要版本证据，不能默认发生。'],
                template='provide_public_state_summary')
    for f in by_type.get('object_structure_change',[])[:2]:
        v=f['value']
        add('file_version_change','相邻捕获版本中的文档结构计数发生变化；不把相同表格编号当作跨版本语义身份。',[f],events=f['event_ids'],
            object_scope={'file_path':v['file_path']},before=v['before_sha256'],after=v['after_sha256'])
    if by_type.get('restoration_recorded'):
        f=by_type['restoration_recorded'][0]
        add('restoration_observed','本 Episode 记录了恢复后新执行；未读取兄弟分支结果，不作干预效果结论。',[f],events=f['event_ids'])
        if by_type.get('no_recorded_file_change'):
            add('state_stability','所有已捕获文件边界差异为空；不证明没有瞬时写入，也不证明现场写入隔离。',by_type['no_recorded_file_change'])
    if not localizations:
        abstentions.append('没有足够依据产生规则覆盖内的行为定位；不是证明本任务无问题。')
    abstentions += ['所有机制和干预效应均未验证。','对象探针只描述直接 XML，不能判定继承样式、图片相似度或最终排版。']
    return {'summary':'规则基线输出：已观察事件、版本变化及待检验假说。未调用分析模型，不输出唯一根因或能力画像。',
            'localizations':localizations,'hypotheses':hypotheses,'abstentions':abstentions}
