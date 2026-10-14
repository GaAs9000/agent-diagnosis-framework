"""Metrics require external labels. No natural trajectory is auto-labelled causal GT."""
from __future__ import annotations
import math
from ..io import L2Error,digest


def localization_metrics(predictions:list[dict],gold:list[dict],k=3):
    if type(k) is not int or k<1:raise L2Error('k must be positive integer')
    pmap={r['case_id']:r for r in predictions}
    if len(pmap)!=len(predictions):raise L2Error('Duplicate prediction case')
    hits=0;known=0;abstained=0;missing=0;rows=[]
    for g in gold:
        case=g['case_id'];p=pmap.get(case)
        if p is None:missing+=1;rows.append({'case_id':case,'status':'missing_prediction'});continue
        accepted=g.get('accepted_event_sets')
        if not accepted:rows.append({'case_id':case,'status':'gold_undefined_not_scored'});continue
        known+=1
        selected=p.get('event_sets',[])[:k]
        if not selected:abstained+=1
        hit=any(set(x)==set(y) for x in selected for y in accepted)
        hits+=hit;rows.append({'case_id':case,'exact_event_set_topk_hit':hit,'abstained':not bool(selected)})
    return {'metric':'exact_event_set_topk; no broad interval overlap reward','k':k,'labelled_cases':known,
            'hits':hits,'hit_rate':hits/known if known else None,'abstentions':abstained,'missing_predictions':missing,
            'rows':rows,'ground_truth_source':'externally_supplied_not_inferred_from_grade_or_fault_injection_location'}


def brier(rows:list[dict]):
    valid=[];undefined=0
    for r in rows:
        p=r.get('probability');y=r.get('outcome')
        if y is None:undefined+=1;continue
        if type(y) is not bool or type(p) not in (int,float) or not math.isfinite(p) or not 0<=p<=1:raise L2Error('Brier requires finite probabilities and boolean outcomes')
        valid.append((p-int(y))**2)
    return {'n_defined':len(valid),'n_undefined':undefined,'brier':sum(valid)/len(valid) if valid else None}


def family_partition(rows:list[dict],holdout_fraction=.25):
    if not 0<holdout_fraction<1:raise L2Error('Invalid holdout fraction')
    groups={}
    for r in rows:
        # Caller must provide family_id; never use episode ID to split siblings.
        if not r.get('family_id'):raise L2Error('Explicit task-family identity required')
        groups.setdefault(r['family_id'],[]).append(r['case_id'])
    if len(groups)<2:return {'status':'insufficient_independent_families','groups':groups,'train':[],'holdout':[]}
    holdout=[];train=[]
    for fam,ids in sorted(groups.items()):
        (holdout if int(digest(fam)[:8],16)/2**32<holdout_fraction else train).extend(ids)
    return {'status':'partitioned' if train and holdout else 'empty_partition_do_not_claim_evaluation',
            'train':train,'holdout':holdout,'family_count':len(groups),'groups':groups}
