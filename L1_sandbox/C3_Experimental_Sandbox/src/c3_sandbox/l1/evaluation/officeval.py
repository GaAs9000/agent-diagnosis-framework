"""Result projections; upstream raw score and optional L1 labels never conflated."""
from __future__ import annotations
import math


def interpret(raw,task_id:str,policy:str='official_dimensions')->dict:
    out={'evaluation_validity':'invalid','task_outcome':'unknown','outcome_policy':policy,
         'official_dimensions':None,'completion_rate':None,'rubric_coverage':'unknown','reason':'malformed_official_result'}
    if not isinstance(raw,dict): return out
    required={'id','file_name','status','error','dim1_pass','dim1_reason','dim2_items','total_score','max_score'}
    if not required.issubset(raw): return out
    if raw['id']!=task_id[-3:]: return out
    if raw['status']!='ok' or raw['error'] is not None:
        out['reason']='official_'+str(raw['status']); return out
    if type(raw['dim1_pass']) is not bool or not isinstance(raw['dim2_items'],list): return out
    if any(type(raw[k]) not in (int,float) or not math.isfinite(raw[k]) for k in ('total_score','max_score')): return out
    if not isinstance(raw['dim1_reason'],str) or not isinstance(raw['file_name'],(str,list)): return out
    for item in raw['dim2_items']:
        if not isinstance(item,dict) or not {'rule','max_delta','delta','hit'}.issubset(item): return out
        if not isinstance(item['rule'],str) or type(item['hit']) is not bool: return out
        if any(type(item[k]) not in (int,float) or not math.isfinite(item[k]) for k in ('max_delta','delta')): return out
    # Upstream permits explanatory dim2 details when dim1 fails. Do not impose the
    # old C3 "empty dim2 only" assumption or rewrite upstream score ranges.
    if not raw['dim1_pass'] and raw['total_score']!=0: return out
    rate=0.0 if not raw['dim1_pass'] else (max(0,raw['total_score'])/raw['max_score'] if raw['max_score']>0 else None)
    out.update(evaluation_validity='valid',reason='official_result_retained',completion_rate=rate,
        official_dimensions={k:raw[k] for k in ('dim1_pass','total_score','max_score')})
    if not raw['dim1_pass']: out['task_outcome']='failure'
    elif policy=='full_score_v1' and rate is not None:
        out['task_outcome']='success' if rate>=1 else ('partial' if rate>0 else 'failure')
    # official_dimensions intentionally has no invented binary success label.
    return out
