"""Portable Diagnosis Package sealing, reference and derived-fact verification."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
from . import SCHEMA_VERSION
from .io import (IntegrityError, InvalidOutput, canonical, digest, inventory, read_json, read_jsonl,
                 bytes_read, sha_file, sha_bytes, safe, pointer, loads, check_journal, write_json)
from .validation import validate_schema, check_diagnosis
from .evidence.references import select


def seal(root: Path, kind='diagnosis') -> str:
    index={'schema_version':SCHEMA_VERSION,'package_kind':kind,'files':inventory(root,{'evidence_index.json'}),
           'trust_scope':'Content self-consistency; external trusted digest/signature required for historical authenticity'}
    validate_schema('index',index)
    write_json(root,'evidence_index.json',index)
    return sha_file(root/'evidence_index.json')


def _fact_check(f,selected,refs,source_events,all_facts,objects,probe_config,blob_reader):
    kind=f['fact_type'];v=f['value']
    simple={'task_instruction','episode_status','official_grade','submission_inventory','execution_budget','tool_requested','termination','restoration_recorded','model_output_text'}
    if kind in simple:
        if v!=selected[0]:raise IntegrityError('Recorded fact projection differs: '+f['fact_id'])
    elif kind=='execution_profile':
        if v!={'recording_mode':selected[0],'capabilities':selected[1]}:raise IntegrityError('Execution profile differs')
    elif kind=='tool_exit':
        obs=selected[0];event=selected[-1];cid=v['call_id']
        call=next((e['data'] for e in source_events if e['kind']=='tool.call' and e['data']['call_id']==cid),None)
        delivered=any(e['kind']=='observation.delivered' and e['data']['call_id']==cid for e in source_events)
        expected={'call_id':cid,'name':call.get('name') if call else None,'exit_code':obs.get('exit_code'),
                  'stop_reason':obs.get('stop_reason'),'output_complete':obs.get('output_complete'),
                  'stdout_ref_id':f['source_ref_ids'][1],'stderr_ref_id':f['source_ref_ids'][2],
                  'stderr_preview':obs.get('stderr','')[:800],'delivery_recorded':delivered}
        if event['call_id']!=cid or expected!=v:raise IntegrityError('Tool exit fact projection differs')
    elif kind=='observation_delivery':
        d=selected[0]
        if v!={'call_id':d['call_id'],'request_id':d['request_id'],'observation_ref_id':f['source_ref_ids'][1]}:
            raise IntegrityError('Observation delivery fact differs')
    elif kind=='observation_not_recorded_delivered':
        if selected[0]['call_id']!=v['call_id'] or any(e['kind']=='observation.delivered' and e['data']['call_id']==v['call_id'] for e in source_events):
            raise IntegrityError('False missing-delivery fact')
    elif kind in ('protocol_error','execution_error'):
        e=next(x for x in source_events if x['event_id']==f['event_ids'][0])
        if v!={'kind':e['kind'],**selected[0]}:raise IntegrityError('Error fact projection differs')
    elif kind=='native_protocol_signal':
        d,c=selected[:2]
        expected={'request_id':d['request_id'],'finish_reason':c.get('finishReason',c.get('finish_reason')),
                  'finish_message':c.get('finishMessage'),'source_observation_only':True}
        if v!=expected:raise IntegrityError('Native protocol fact differs')
    elif kind=='file_transition':
        delta,b,a=selected
        event=next(x for x in source_events if x['event_id']==f['event_ids'][0])
        expected={**delta,'before_snapshot_id':b['snapshot_id'],'after_snapshot_id':a['snapshot_id'],
                  'tool_result_event_id':event['links'].get('tool_result')}
        if v!=expected:raise IntegrityError('File transition projection differs')
    elif kind=='no_recorded_file_change':
        changes=[loads(line) for line in selected[0]['text'].splitlines()]
        if v!={'transitions':len(changes),'changed_entries':0} or any(c['changes'] for c in changes):
            raise IntegrityError('False no-file-change fact')
    elif kind=='object_property':
        from .facts.docx_probe import probe
        pid=v['probe_id']
        if pid not in objects:
            raw=blob_reader(refs[f['source_ref_ids'][0]]['local_blob'])
            objects[pid]=probe(raw,probe_config)
        parsed=objects[pid]
        if parsed['probe_id']!=pid or v['file_sha256']!=parsed['source_sha256'] or v['probe_status']!=parsed['status'] or v['summary']!=parsed['summary'] or v['unknowns']!=parsed['unknowns']:
            raise IntegrityError('DOCX derived fact differs from probe recomputation')
        if digest(parsed)!=f['derivation']['output_sha256']:raise IntegrityError('DOCX probe digest differs')
        snapshots={s['snapshot_id']:s for s in selected[1:]}
        for loc in v['locations']:
            snap=snapshots.get(loc['snapshot_id'])
            if not snap or not any(x['path']==loc['file_path'] and x['blob']==loc['blob'] and x['sha256']==v['file_sha256'] for x in snap['files']):
                raise IntegrityError('Object version location is not in cited snapshot')
    elif kind=='object_structure_change':
        ids=v['input_fact_ids'];delta,left,right=(all_facts[x]['value'] for x in ids)
        expected={'file_path':delta['path'],'before_sha256':delta['before']['sha256'],'after_sha256':delta['after']['sha256'],
                  'before_summary':left['summary'],'after_summary':right['summary'],'semantic_object_mapping':'not_inferred','input_fact_ids':ids}
        if v!=expected:raise IntegrityError('Object structure comparison differs')
    elif kind=='legacy_record':
        if v['payload']!=selected[0]:raise IntegrityError('Legacy projection differs')
    else:raise IntegrityError('Unknown fact derivation: '+kind)


def verify(root: Path, *, source: Path | None=None, expected_index: str | None=None) -> dict:
    root=Path(root);errors=[];warnings=[]
    result={'verified':False,'scope':'diagnosis integrity and derivations, not causal truth','errors':errors,'warnings':warnings}
    try:
        idx=read_json(safe(root,'evidence_index.json'));validate_schema('index',idx)
        index_hash=sha_file(root/'evidence_index.json')
        if expected_index and expected_index!=index_hash:raise IntegrityError('External diagnosis index digest differs')
        if idx['files']!=inventory(root,{'evidence_index.json'}):raise IntegrityError('Diagnosis file inventory or digest differs')
        if idx['package_kind']!='diagnosis':
            from .experiments.plans import verify_experiment_package
            return verify_experiment_package(root,idx,index_hash)
        m=read_json(root/'manifest.json');validate_schema('manifest',m)
        config=read_json(root/'analyst_config.json')
        if m['config_sha256']!=digest(config):raise IntegrityError('Config digest mismatch')
        facts=read_jsonl(root/'facts.jsonl');references=read_json(root/'references.json');fact_map={f['fact_id']:f for f in facts}
        if len(fact_map)!=len(facts):raise IntegrityError('Duplicate fact IDs')
        events=read_json(root/'events.json');event_ids={e['event_id'] for e in events}
        trace=read_jsonl(root/'analyst_trace.jsonl');check_journal(trace)
        vals={};ref_map={x['ref_id']:x for x in references}
        if len(ref_map)!=len(references):raise IntegrityError('Duplicate evidence ref IDs')
        source_index=read_json(root/'source_index.json') if (root/'source_index.json').exists() else None
        if source_index is not None and sha_file(root/'source_index.json')!=m['source_index_sha256']:
            raise IntegrityError('Original source index bytes changed')
        source_files={x['path']:x for x in source_index['files']} if source_index else {}
        source_events=[]
        analyst_requests={};analyst_responses={};analyst_calls={}
        for row in trace:
            d=row['data'];kind=row['kind']
            if kind in ('model.request','model.response'):
                raw=bytes_read(safe(root,d['path']))
                if sha_bytes(raw)!=d['sha256']:raise IntegrityError('Raw Analyst request/response differs from journal')
                if kind=='model.request':
                    if d['request_id'] in analyst_requests:raise IntegrityError('Duplicate Analyst request ID')
                    analyst_requests[d['request_id']]=loads(raw)
                else:
                    if d['request_id'] not in analyst_requests or d['request_id'] in analyst_responses:raise IntegrityError('Analyst response pairing differs')
                    try:analyst_responses[d['request_id']]=loads(raw)
                    except IntegrityError:analyst_responses[d['request_id']]=None
            elif kind=='tool.request':
                from .evidence.l1 import native_calls
                body=analyst_responses.get(d['request_id'])
                if not isinstance(body,dict):raise IntegrityError('Tool request lacks a native Analyst response')
                native=native_calls(body,d['request_id'])
                if d not in native:raise IntegrityError('Analyst tool call differs from original native response')
                if d['call_id'] in analyst_calls:raise IntegrityError('Duplicate Analyst tool call')
                analyst_calls[d['call_id']]=d
            elif kind=='tool.result':
                if d['call_id'] not in analyst_calls:raise IntegrityError('Orphan Analyst tool result')
                if digest(read_json(safe(root,d['path'])))!=d['sha256']:raise IntegrityError('Analyst tool result differs from journal')
            elif kind=='output.accepted':
                body=analyst_responses.get(f"analyst-r{d['request_index']:03d}")
                if not isinstance(body,dict):raise IntegrityError('Accepted output lacks a native response')
                candidates=[]
                if d['origin']=='submit_diagnosis':
                    from .evidence.l1 import native_calls
                    for call in native_calls(body,f"analyst-r{d['request_index']:03d}"):
                        if call['name']=='submit_diagnosis':candidates.append(loads(call['arguments']['diagnosis_json']))
                else:
                    if 'candidates' in body:
                        parts=body['candidates'][0].get('content',{}).get('parts',[])
                        text='\n'.join(p['text'] for p in parts if isinstance(p.get('text'),str) and not p.get('thought'))
                    else:text=body['choices'][0]['message'].get('content','')
                    candidates.append(loads(text))
                if not any(digest(c)==d['diagnosis_sha256'] for c in candidates):raise IntegrityError('Accepted diagnosis not in original model response')
        for r in references:
            if r['source_index_sha256']!=m['source_index_sha256']:raise IntegrityError('Cross-source evidence reference')
            original=source_files.get(r['source_path'])
            if not original or original['sha256']!=r['source_sha256'] or original['size_bytes']!=r['size_bytes']:
                raise IntegrityError('Reference not anchored in original source index')
            b=bytes_read(safe(root,r['local_blob']))
            if sha_bytes(b)!=r['source_sha256'] or len(b)!=r['size_bytes']:raise IntegrityError('Source blob differs')
            descriptor={k:r[k] for k in ('source_index_sha256','source_path','source_sha256','size_bytes','selector','event_ids','analyst_visible')}
            if r['ref_id']!='ref_'+digest(descriptor)[:24]:raise IntegrityError('Reference ID differs from descriptor')
            value=select(b,r['selector']);vals[r['ref_id']]=value
            if digest(value)!=r['selected_value_sha256']:raise IntegrityError('Source slice value differs')
            if r['source_path']=='trajectory.jsonl' and not source_events:
                source_events=[loads(x) for x in b.splitlines()]
                if source_events and 'payload' in source_events[0]:source_events=[{**e,'data':e['payload']} for e in source_events]
            if any(x not in event_ids for x in r['event_ids']):raise IntegrityError('Evidence reference cites unavailable event')
        if source_events:
            if len(events)!=len(source_events):raise IntegrityError('Public event view count differs from source')
            for e,original in zip(events,source_events):
                for key in ('event_id','seq','kind'):
                    if e[key]!=original[key]:raise IntegrityError('Public event metadata differs from source')
                if any(fid not in fact_map for fid in e['fact_ids']):raise IntegrityError('Event view references nonexistent fact')
        objects={}
        for f in facts:
            validate_schema('fact',f)
            if f['fact_id']!='fact_'+digest({k:v for k,v in f.items() if k!='fact_id'})[:24]:raise IntegrityError('Fact ID differs from content')
            if any(x not in ref_map for x in f['source_ref_ids']) or any(x not in event_ids for x in f['event_ids']):
                raise IntegrityError('Fact reference missing')
            _fact_check(f,[vals[x] for x in f['source_ref_ids']],ref_map,source_events,fact_map,objects,config['objects'],lambda p:bytes_read(safe(root,p)))
            if f['fact_type']=='object_property':
                p=f['derivation']['output_path']
                if read_json(safe(root,p))!=objects[f['value']['probe_id']]:raise IntegrityError('Saved object probe differs from reparse')
        view=read_json(root/'analysis_view.json')
        if view and (view['view_sha256']!=digest({k:v for k,v in view.items() if k!='view_sha256'}) or set(view['fact_ids'])!=set(fact_map)):
            raise IntegrityError('Analysis view changed')
        diag=read_json(root/'diagnosis.json')
        if m['status'].startswith('completed'):
            seen=None
            if m['mode']!='rules':
                accepted=[r for r in trace if r['kind']=='output.accepted']
                if len(accepted)!=1 or accepted[0]['data']['diagnosis_sha256']!=digest(diag):raise IntegrityError('Accepted prediction does not match diagnosis')
                seen=set()
                for row in trace:
                    if row['kind']=='single_shot_view':
                        single=read_json(root/'raw/analyst/single_shot_view.json')
                        for f in single.get('facts',[]):
                            if fact_map.get(f['fact_id'])!=f:raise IntegrityError('Single-shot fact differs')
                            seen.add(f['fact_id'])
                    elif row['kind']=='tool.result':
                        raw=read_json(safe(root,row['data']['path']))
                        if digest(raw)!=row['data']['sha256']:raise IntegrityError('Analyst tool response digest differs')
                        def walk(v):
                            if isinstance(v,dict):
                                if 'fact_id' in v and 'value' in v and 'source_ref_ids' in v:
                                    if fact_map.get(v['fact_id'])!=v:raise IntegrityError('Returned evidence fact altered')
                                    seen.add(v['fact_id'])
                                for x in v.values():walk(x)
                            elif isinstance(v,list):
                                for x in v:walk(x)
                        walk(raw)
                if set(accepted[0]['data']['seen_fact_ids'])!=seen:raise IntegrityError('Claimed evidence reads do not match tool receipts')
            checks=check_diagnosis(diag,facts,events,seen_fact_ids=seen,max_findings=config['budget']['max_findings'])
            if checks!=read_jsonl(root/'claim_checks.jsonl'):raise IntegrityError('Claim checks differ')
        else:
            if diag['hypotheses'] or diag['localizations']:raise IntegrityError('Failed analysis must not publish accepted findings')
            warnings.append('analysis_not_completed_no_validated_diagnosis')
        if read_jsonl(root/'localizations.jsonl')!=diag['localizations'] or read_jsonl(root/'hypotheses.jsonl')!=diag['hypotheses']:
            raise IntegrityError('Diagnosis projections differ')
        frozen=read_json(root/'predictions/frozen_manifest.json')
        if frozen['diagnosis_sha256']!=digest(diag) or frozen['hypotheses_sha256']!=digest(diag['hypotheses']) or frozen['source_index_sha256']!=m['source_index_sha256']:
            raise IntegrityError('Frozen prediction digest differs')
        if m['counts']['facts']!=len(facts) or m['counts']['localizations']!=len(diag['localizations']) or m['counts']['hypotheses']!=len(diag['hypotheses']):
            raise IntegrityError('Diagnosis counts mismatch')
        if source is not None:
            from .evidence.l1 import L1Reader
            r=L1Reader(source,expected_index=m['source_index_sha256'])
            result['full_source_verified']=r.report
        else:result['full_source_verified']=None
        warnings += ['hypotheses_are_untested','semantic_entailment_not_certified','embedded_cited_source_subset_is_not_a_full_L1_episode']
        if m['mode']=='rules':warnings.append('deterministic_rule_baseline_no_analyst_model_was_called')
        if m['mode']=='test_fixture':warnings.append('test_fixture_not_live_analyst')
        result.update(verified=True,index_sha256=index_hash,analysis_id=m['analysis_id'],status=m['status'],counts=m['counts'])
    except (OSError,KeyError,TypeError,ValueError,IntegrityError,InvalidOutput) as exc:
        errors.append({'type':type(exc).__name__,'message':str(exc)})
    return result
