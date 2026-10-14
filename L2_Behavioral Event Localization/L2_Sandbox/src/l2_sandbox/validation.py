"""Structural/ref checks. Does not certify entailment, causal truth or calibration."""
from __future__ import annotations
from functools import lru_cache
from importlib.resources import files
from .io import InvalidOutput, loads

@lru_cache(maxsize=None)
def validator(kind):
    from jsonschema import Draft202012Validator
    doc=loads(files('l2_sandbox').joinpath('schemas/contracts.json').read_bytes())
    return Draft202012Validator({'$schema':doc['$schema'],'$defs':doc['$defs'],'$ref':'#/$defs/'+kind})

def validate_schema(kind,value):
    errors=list(validator(kind).iter_errors(value))
    if errors:
        e=errors[0]
        raise InvalidOutput(f'{kind} schema: {"/".join(map(str,e.path))}: {e.message}')

def check_diagnosis(value, facts, events, *, seen_fact_ids=None, max_findings=12):
    validate_schema('diagnosis',value)
    fact_map={f['fact_id']:f for f in facts};event_map={e['event_id']:e for e in events}
    locs={};hyps={};checks=[]
    if len(value['localizations'])>max_findings or len(value['hypotheses'])>max_findings:
        raise InvalidOutput('Finding budget exceeded')
    allowed=set(fact_map) if seen_fact_ids is None else set(seen_fact_ids)
    def check_refs(record):
        for key in ('supporting_fact_ids','counterevidence_fact_ids'):
            ids=record[key]
            if len(ids)!=len(set(ids)) or any(x not in fact_map or x not in allowed for x in ids):
                raise InvalidOutput('Unknown, unread or duplicate fact reference')
        if set(record['supporting_fact_ids']) & set(record['counterevidence_fact_ids']):
            raise InvalidOutput('Same fact cannot be both support and counterevidence without a separate qualified claim')
    for loc in value['localizations']:
        lid=loc['localization_id']
        if lid in locs:raise InvalidOutput('Duplicate localization ID')
        locs[lid]=loc;check_refs(loc)
        if any(e not in event_map for e in loc['event_ids']):raise InvalidOutput('Unknown event ID')
        if loc['granularity']=='event' and len(loc['event_ids'])!=1:raise InvalidOutput('Single event granularity requires one event')
        if loc['granularity']=='interval':
            seq=[event_map[e]['seq'] for e in loc['event_ids']]
            if len(seq)!=2 or seq[0]>seq[1]:raise InvalidOutput('Interval requires ordered endpoint events')
        support_events={e for fid in loc['supporting_fact_ids'] for e in fact_map[fid]['event_ids']}
        if loc['event_ids'] and not set(loc['event_ids'])<=support_events:
            raise InvalidOutput('Localized event lacks an explicitly supporting event-bound fact')
        req=loc['task_relation']['requirement_fact_id']
        if req is not None and (req not in allowed or req not in fact_map or fact_map[req]['fact_type']!='task_instruction'):
            raise InvalidOutput('Task relation must cite an available public task fact')
        known_versions=set()
        for fid in loc['supporting_fact_ids']:
            v=fact_map[fid]['value']
            if not isinstance(v,dict):continue
            for k in ('before','after'):
                if isinstance(v.get(k),dict) and v[k].get('sha256'):known_versions.add(v[k]['sha256'])
            for k in ('file_sha256','before_sha256','after_sha256'):
                if v.get(k):known_versions.add(v[k])
        for key in ('before_version','after_version'):
            if loc[key] is not None and loc[key] not in known_versions:raise InvalidOutput('Version not supported by cited facts')
        checks.append({'record_id':lid,'reference_check':'passed','scope_check':'passed',
                       'semantic_entailment':'not_automatically_certified','causal_verdict':'untested'})
    for hyp in value['hypotheses']:
        hid=hyp['hypothesis_id']
        if hid in hyps:raise InvalidOutput('Duplicate hypothesis ID')
        hyps[hid]=hyp;check_refs(hyp)
        if any(x not in locs for x in hyp['localization_ids']):raise InvalidOutput('Unknown localization reference')
        checks.append({'record_id':hid,'reference_check':'passed','scope_check':'episode_only',
                       'semantic_entailment':'requires_external_review','causal_verdict':'untested',
                       'prediction_calibration':'not_evaluated'})
    if not value['localizations'] and not value['abstentions']:
        raise InvalidOutput('Empty diagnosis must state an abstention reason')
    return checks
