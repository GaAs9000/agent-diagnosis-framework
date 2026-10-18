"""Live evidence 0.2.0. Legacy 0.1.0 remains readable by the P1 verifier."""
from __future__ import annotations
import json
import os
from pathlib import Path
from ..core.contracts import VERSION, EvidenceError, ProtocolError, L1Error
from ..storage import canonical, digest, file_digest, regular_file, read_json, write_json, write_bytes, confined, utc_now


from functools import lru_cache

@lru_cache(maxsize=None)
def validator(kind):
    from jsonschema import Draft202012Validator, FormatChecker
    from importlib.resources import files
    doc=json.loads(files(__package__).joinpath('schema.json').read_text('utf-8'))
    schema={'$schema':doc['$schema'],'$defs':doc['$defs'],'$ref':'#/$defs/'+kind}
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema,format_checker=FormatChecker())

def validate(kind,value):
    errors=list(validator(kind).iter_errors(value))
    if errors: raise EvidenceError(f'{kind}: {errors[0].message}')

class Journal:
    def __init__(self,root,episode_id):
        self.root=root; self.episode_id=episode_id; self.events=[]
        self.path=write_bytes(root,'trajectory.jsonl',b'')
        self.states=write_bytes(root,'state_changes.jsonl',b'')
    def emit(self,kind,data=None,*,links=None):
        seq=len(self.events)
        event={'schema_version':VERSION,'episode_id':self.episode_id,'event_id':f'{self.episode_id}:e{seq:06d}',
               'seq':seq,'kind':kind,'observed_at':utc_now(),'links':links or {},'data':data or {},
               'previous_sha256':self.events[-1]['sha256'] if self.events else None}
        event['sha256']=digest(event); validate('event',event)
        self._append(self.path,event); self.events.append(event)
        return event
    def state_change(self,value): self._append(self.states,value)
    @staticmethod
    def _append(path,value):
        fd=os.open(path,os.O_WRONLY|os.O_APPEND|os.O_NOFOLLOW)
        with os.fdopen(fd,'ab') as f:
            f.write(canonical(value)+b'\n'); f.flush(); os.fsync(f.fileno())
    def ref(self,relative):
        p=confined(self.root,relative); regular_file(p)
        return {'path':relative,'sha256':file_digest(p),'size_bytes':p.stat().st_size}


def inventory(root):
    if root.is_symlink() or not root.is_dir(): raise EvidenceError('invalid evidence root')
    result=[]
    for p in sorted(root.rglob('*')):
        if p.is_symlink(): raise EvidenceError('symlink in evidence')
        if p.is_dir(): continue
        regular_file(p)
        if p.name=='evidence_index.json' and p.parent==root: continue
        result.append({'path':p.relative_to(root).as_posix(),'sha256':file_digest(p),'size_bytes':p.stat().st_size})
    return result


def seal(root,events):
    manifest=read_json(root/'manifest.json'); validate('manifest',manifest)
    index={'schema_version':VERSION,'episode_id':manifest['episode_id'],'files':inventory(root),
           'event_count':len(events),'trajectory_head_sha256':events[-1]['sha256'] if events else None,
           'trust_scope':'self_consistency; authenticity requires trusted controller and externally anchored index digest'}
    validate('index',index); write_json(root,'evidence_index.json',index)
    return file_digest(root/'evidence_index.json')


def scan(path):
    events=[]; errors=[]; previous=None
    for n,line in enumerate(path.read_bytes().splitlines(keepends=True)):
        try:
            if not line.endswith(b'\n'): raise EvidenceError('unterminated_tail')
            event=json.loads(line); validate('event',event)
            if event['seq']!=n or event['previous_sha256']!=previous or digest({k:v for k,v in event.items() if k!='sha256'})!=event['sha256']:
                raise EvidenceError('broken_event_chain')
            events.append(event); previous=event['sha256']
        except (ValueError,TypeError,EvidenceError) as exc:
            errors.append({'line':n+1,'error':str(exc)}); break
    return {'events':events,'errors':errors,'valid_prefix_events':len(events),'complete':not errors}


def _refs(value):
    if isinstance(value,dict):
        if set(value)=={'path','sha256','size_bytes'}: yield value
        else:
            for x in value.values(): yield from _refs(x)
    elif isinstance(value,list):
        for x in value: yield from _refs(x)


def _verify_response_consumption(root,manifest,events,requests,responses):
    """Bind reused wire evidence to the original, externally anchored prefix."""
    import hashlib
    reused=[e for e in requests.values() if e['data'].get('source')=='saved_response']
    consumed=[e for e in events if e['kind']=='response.consumed']
    if not reused and not consumed:
        if manifest.get('usage',{}).get('saved_response_consumptions',0)!=0:
            raise EvidenceError('Response consumption count without consumed response')
        return
    if len(reused)!=1 or len(consumed)!=1:
        raise EvidenceError('Exactly one saved response may be consumed')
    request=reused[0]; rid=request['data']['request_id']; consumption=consumed[0]
    if (request['data'].get('http_attempts')!=0 or consumption['data'].get('request_id')!=rid or
            consumption['data'].get('new_http_attempts')!=0 or
            consumption['links'].get('request')!=request['event_id'] or
            manifest['usage'].get('saved_response_consumptions')!=1):
        raise EvidenceError('Saved response transport/count binding mismatch')
    lineage=manifest.get('lineage') or {}
    binding=read_json(root/'lineage/response_use.json')
    if binding!=lineage.get('response_use') or binding.get('new_http_attempts_during_consume')!=0:
        raise EvidenceError('Response-use lineage differs from saved binding')
    source_index_path=root/'lineage/source_evidence_index.json'
    if file_digest(source_index_path)!=lineage.get('parent_index_sha256'):
        raise EvidenceError('Consumed response original index hash mismatch')
    source_index=read_json(source_index_path); validate('index',source_index)
    inventory_by_path={row['path']:row for row in source_index['files']}
    def original_file(relative,copy_path):
        row=inventory_by_path.get(relative)
        if row is None or file_digest(copy_path)!=row['sha256'] or copy_path.stat().st_size!=row['size_bytes']:
            raise EvidenceError('Consumed source differs from original index: '+relative)
    original_file('manifest.json',root/'lineage/source_manifest.json')
    original_file('trajectory.jsonl',root/'lineage/source_trajectory.jsonl')
    source_manifest=read_json(root/'lineage/source_manifest.json')
    if source_manifest['episode_id']!=lineage.get('parent_episode_id') or source_index['episode_id']!=source_manifest['episode_id']:
        raise EvidenceError('Consumed source episode identity mismatch')
    source_scan=scan(root/'lineage/source_trajectory.jsonl')
    source_events=source_scan['events']
    if (not source_scan['complete'] or len(source_events)!=source_index['event_count'] or
            not source_events or source_events[-1]['sha256']!=source_index['trajectory_head_sha256'] or
            source_index['trajectory_head_sha256']!=binding.get('source_trajectory_head_sha256')):
        raise EvidenceError('Consumed source trajectory chain mismatch')
    source_rid=binding['source_request_id']
    original_requests=[e for e in source_events if e['kind']=='model.request' and e['data']['request_id']==source_rid]
    original_responses=[e for e in source_events if e['kind']=='model.response' and e['data']['request_id']==source_rid]
    if len(original_requests)!=1 or len(original_responses)!=1:
        raise EvidenceError('Consumed source lacks unique request and response')
    old_request,old_response=original_requests[0],original_responses[0]
    source_calls=[e for e in source_events if e['kind']=='tool.call' and e['data'].get('request_id')==source_rid]
    if (len(source_calls)!=1 or source_calls[0]['data'].get('name')!='edit' or
            source_calls[0]['data'].get('call_id')!=binding.get('source_call_id') or
            old_response['links'].get('request')!=old_request['event_id'] or
            source_calls[0]['links'].get('response')!=old_response['event_id']):
        raise EvidenceError('Consumed source call graph mismatch')
    cid=binding['source_call_id']
    if (sum(e['kind']=='tool.rejected' and e['data'].get('call_id')==cid for e in source_events)!=1 or
            any(e['kind'] in ('tool.started','tool.result') and e['data'].get('call_id')==cid for e in source_events) or
            [e for e in source_events if e['kind']=='model.request'][-1]!=old_request or
            source_manifest.get('recording_mode')!='live' or source_manifest['status']['termination']!='protocol_error'):
        raise EvidenceError('Consumed source is not a final unexecuted rejected call')
    if old_response['sha256']!=binding.get('source_response_event_sha256'):
        raise EvidenceError('Consumed response event hash mismatch')
    source_request=root/'lineage/source_request.bin'; source_response=root/'lineage/source_response.bin'
    original_file(old_request['data']['body']['path'],source_request)
    original_file(old_response['data']['body']['path'],source_response)
    for name,path in [('source_request_sha256',source_request),('source_response_sha256',source_response)]:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=binding.get(name):
            raise EvidenceError('Consumed wire hash differs from binding')
    if (source_request.read_bytes()!=(root/request['data']['body']['path']).read_bytes() or
            rid not in responses or source_response.read_bytes()!=(root/responses[rid]['data']['body']['path']).read_bytes()):
        raise EvidenceError('Consumed child wire bytes differ from original')
    if list((root/request['data']['body']['path']).parent.glob('attempt_*/http.json')):
        raise EvidenceError('Saved response contains a new transport-attempt receipt')
    cp_relative='checkpoints/'+lineage['parent_checkpoint_id']+'.json'
    original_file(cp_relative,root/'lineage/parent_checkpoint.json')
    if file_digest(root/'lineage/parent_checkpoint.json')!=lineage.get('parent_checkpoint_sha256'):
        raise EvidenceError('Consumed parent checkpoint digest mismatch')
    parent_cp=read_json(root/'lineage/parent_checkpoint.json')
    source_offset=int(source_rid.rsplit(':r',1)[1])
    if (parent_cp.get('checkpoint_id')!=lineage['parent_checkpoint_id'] or
            parent_cp.get('budget_used',{}).get('model_requests')!=source_offset or
            parent_cp.get('qualification',{}).get('eligible') is not True):
        raise EvidenceError('Consumed source checkpoint does not immediately precede response')
    original_file('states/snapshots/'+parent_cp['snapshot_id']+'.json',root/'lineage/parent_snapshot.json')
    first_snapshot=next(e for e in events if e['kind']=='state.snapshot')
    restored=read_json(root/first_snapshot['data']['snapshot']['path'])
    parent_snapshot=read_json(root/'lineage/parent_snapshot.json')
    if any(restored.get(k)!=parent_snapshot.get(k) for k in ('content_sha256','files','directories')):
        raise EvidenceError('Consumed response initial workspace differs from original checkpoint')
    prefix=[e for e in events if e['kind']=='response.prefix_accounted']
    if (len(prefix)!=1 or prefix[0]['data'].get('source_call_id')!=binding.get('source_call_id') or
            prefix[0]['data'].get('model_requests')!=int(source_rid.rsplit(':r',1)[1])+1 or
            prefix[0]['data'].get('known_total_tokens')!=source_manifest['usage']['known_total_tokens_including_parent']):
        raise EvidenceError('Consumed prefix budget accounting mismatch')


def context_overflow_source(root, checkpoint_id, request_id):
    """Validate a final context rejection from original indexed source files.

    Works on the full parent or the child's minimal indexed source proof.
    """
    import re
    index=read_json(root/'evidence_index.json'); validate('index',index)
    rows={row['path']:row for row in index['files']}; used=set()
    def load(path, binary=False):
        p=confined(root,path); row=rows.get(path)
        if row is None or file_digest(p)!=row['sha256'] or p.stat().st_size!=row['size_bytes']:
            raise EvidenceError('Context source differs from original index: '+path)
        used.add(path)
        return p.read_bytes() if binary else read_json(p)
    manifest=load('manifest.json'); cp=load('checkpoints/'+checkpoint_id+'.json')
    config=load('raw/resolved_config.json')
    load('trajectory.jsonl',True); history=scan(root/'trajectory.jsonl')
    events=history['events']
    if (not history['complete'] or len(events)!=index['event_count'] or not events or
            events[-1]['sha256']!=index['trajectory_head_sha256'] or
            manifest['episode_id']!=index['episode_id'] or
            any(e['episode_id']!=manifest['episode_id'] for e in events)):
        raise EvidenceError('Context source trajectory/index mismatch')
    if (not re.fullmatch(r'r\d{6}',request_id) or manifest['recording_mode']!='live' or
            manifest['status']['termination']!='model_transport_error' or
            manifest['status']['evidence_completeness']!='complete' or
            cp['qualification']['eligible'] is not True or cp.get('pending_calls') or
            cp['budget_used']['model_requests']!=int(request_id[1:]) or
            manifest['usage']['model_requests_including_parent']!=int(request_id[1:])+1):
        raise EvidenceError('Context resume requires final spent request immediately after qualified checkpoint')
    requests=[e for e in events if e['kind']=='model.request']
    if not requests or requests[-1]['data']['request_id'].split(':')[-1]!=request_id:
        raise EvidenceError('Context resume must target final request')
    request=requests[-1]; rid=request['data']['request_id']
    replies=[e for e in events if e['kind']=='model.response' and e['data']['request_id']==rid]
    if len(replies)!=1 or any(e['kind'] in ('model.parsed','tool.call') and e['data'].get('request_id')==rid for e in events):
        raise EvidenceError('Context rejection cannot have assistant candidates or tool calls')
    response=replies[0]; metadata=load(response['data']['http']['path'])
    body=load(response['data']['body']['path']); error=body.get('error',{})
    if (response['data']['status_code']!=400 or metadata.get('status_code')!=400 or
            metadata.get('body_truncated') or metadata.get('transport_error') or body.get('choices') or
            body.get('candidates') or error.get('type')!='BadRequestError' or
            error.get('param')!='input_tokens' or error.get('code')!=400 or
            'maximum context length' not in error.get('message','').lower()):
        raise EvidenceError('Context resume requires complete HTTP400 input_tokens maximum-context BadRequestError')
    failed_request=load(request['data']['body']['path'])
    if (cp['checkpoint_id']!=checkpoint_id or cp['config_sha256']!=manifest['config_sha256'] or
            digest(config)!=manifest['config_sha256'] or
            cp['budget_used']['tool_calls']!=manifest['usage']['tool_calls_including_parent'] or
            cp['budget_used']['known_total_tokens']!=manifest['usage']['known_total_tokens_including_parent']):
        raise EvidenceError('Context source checkpoint config or spent budget mismatch')
    from ..models.native import OpenAICompatibleBackend
    from ..runtime.tools import tools_for_config
    if (config['model']['adapter']!='openai-compatible' or
            OpenAICompatibleBackend(config['model']).request(cp['conversation'],tools_for_config(config))!=failed_request):
        raise EvidenceError('Context source checkpoint differs from rejected native request')
    parsed=[e for e in events if e['kind']=='model.parsed' and e['seq']<request['seq']]
    if not parsed: raise EvidenceError('Context resume lacks successful prompt usage')
    success=parsed[-1]; success_id=success['data']['request_id']
    if int(success_id.rsplit(':r',1)[1])!=int(request_id[1:])-1:
        raise EvidenceError('Context resume requires prompt usage from immediately preceding request')
    success_request=next(e for e in requests if e['data']['request_id']==success_id)
    success_response=next(e for e in events if e['kind']=='model.response' and e['data']['request_id']==success_id)
    success_http=load(success_response['data']['http']['path'])
    success_body=load(success_response['data']['body']['path'])
    usage=success_body.get('usage',{}); prompt_tokens=usage.get('prompt_tokens')
    if (success_response['data']['status_code']!=200 or success_http.get('body_truncated') or
            success_http.get('transport_error') or type(prompt_tokens) is not int or prompt_tokens<=0 or
            prompt_tokens!=success['data'].get('usage',{}).get('prompt_tokens')):
        raise EvidenceError('Context resume lacks complete successful prompt usage receipt')
    usage_messages=load(success_request['data']['body']['path']).get('messages')
    if not isinstance(usage_messages,list): raise EvidenceError('Context resume requires native messages')
    snapshot=load('states/snapshots/'+cp['snapshot_id']+'.json')
    final_event=[e for e in events if e['kind']=='state.snapshot'][-1]
    final=load(final_event['data']['snapshot']['path'])
    if any(snapshot.get(k)!=final.get(k) for k in ('content_sha256','files','directories')):
        raise EvidenceError('Context checkpoint workspace differs from final workspace')
    binding={'source_request_id':rid,'last_success_request_id':success_id,
             'last_success_prompt_tokens':prompt_tokens,'usage_messages_sha256':digest(usage_messages),
             'source_request_sha256':digest(failed_request), 'new_http_attempts_during_carry':0}
    return dict(binding=binding, files=used, manifest=manifest, checkpoint=cp, config=config,
                failed_request=failed_request, usage_messages=usage_messages, snapshot=snapshot)


def _verify_context_overflow(root,manifest,events,requests):
    lineage=manifest.get('lineage') or {}; binding=lineage.get('context_overflow')
    if binding is None: return
    source=root/'lineage/context_source'
    if file_digest(source/'evidence_index.json')!=lineage['parent_index_sha256']:
        raise EvidenceError('Context source index digest mismatch')
    proof=context_overflow_source(source,lineage['parent_checkpoint_id'],binding['source_request_id'].split(':')[-1])
    if proof['binding']!=binding or proof['manifest']['episode_id']!=lineage['parent_episode_id']:
        raise EvidenceError('Context source lineage mismatch')
    from ..core.orchestrator import _check_context_upgrade, _maybe_compact_pi_conversation
    config=read_json(root/'raw/resolved_config.json')
    if not _check_context_upgrade(proof['config'],config): raise EvidenceError('Invalid context config upgrade')
    usage=proof['manifest']['usage']
    upgrade=lineage.get('config_upgrade',{})
    if (upgrade.get('mode')!='context_overflow' or upgrade.get('applied') is not True or
            upgrade.get('source_config_sha256')!=digest(proof['config']) or
            upgrade.get('target_config_sha256')!=digest(config)):
        raise EvidenceError('Context config upgrade lineage mismatch')
    carried=[e for e in events if e['kind']=='context_overflow.carried']
    if len(carried)!=1 or carried[0]['data']!=binding: raise EvidenceError('Missing context carry receipt')
    child_usage=manifest['usage']
    charged=set(); calls={}; request_count=usage['model_requests_including_parent']
    token_count=usage['known_total_tokens_including_parent']; checkpoint_seconds=usage['agent_phase_seconds']
    inherited_tools=usage['tool_calls_including_parent']
    for event in events:
        kind=event['kind']; data=event['data']
        if kind=='model.request':
            if not data['request_id'].endswith(f':r{request_count:06d}'):
                raise EvidenceError('Context continuation request budget sequence mismatch')
            request_count+=1
        elif kind=='model.parsed':
            u=data.get('usage') or {}; total=u.get('total_tokens',u.get('totalTokenCount'))
            if type(total) is int and total>=0: token_count+=total
        elif kind=='tool.call': calls[data['call_id']]=event['event_id']
        elif kind=='tool.charged':
            cid=data.get('call_id')
            if (cid not in calls or cid in charged or event['links'].get('call')!=calls[cid] or
                    data.get('tool_calls_including_parent')!=inherited_tools+len(charged)+1):
                raise EvidenceError('Context tool charge is duplicate, orphaned or miscounted')
            charged.add(cid)
            if inherited_tools+len(charged)>config['budget']['max_tool_calls']:
                raise EvidenceError('Context tool charge exceeds inherited budget')
        elif kind in ('tool.validation','tool.started','tool.error','tool.result'):
            if data.get('call_id') not in charged:
                raise EvidenceError('Context tool action lacks preceding budget charge')
        elif kind=='checkpoint.created':
            checkpoint=read_json(root/data['checkpoint']['path']); budget=checkpoint['budget_used']
            if (budget['model_requests']!=request_count or budget['tool_calls']!=inherited_tools+len(charged) or
                    budget['known_total_tokens']!=token_count or
                    not checkpoint_seconds<=budget['agent_seconds']<=child_usage['agent_phase_seconds']):
                raise EvidenceError('Context checkpoint spent budget differs from ordered evidence')
            checkpoint_seconds=budget['agent_seconds']
    if (child_usage['model_requests_including_parent']!=request_count or
            child_usage['tool_calls_including_parent']!=inherited_tools+len(charged) or
            child_usage['known_total_tokens_including_parent']!=token_count or
            child_usage['agent_phase_seconds']<usage['agent_phase_seconds'] or
            child_usage.get('saved_response_consumptions')!=0 or
            (usage['all_requests_have_usage'] is False and child_usage['all_requests_have_usage'] is not False)):
        raise EvidenceError('Context continuation cumulative spent budget or usage mismatch')
    restored_events=[e for e in events if e['kind']=='state.snapshot']
    if restored_events:
        restored_snapshot=read_json(root/restored_events[0]['data']['snapshot']['path'])
        if any(restored_snapshot.get(k)!=proof['snapshot'].get(k) for k in ('content_sha256','files','directories')):
            raise EvidenceError('Context restored workspace differs from source checkpoint')
    cp=proof['checkpoint']
    try:
        expected,receipt=_maybe_compact_pi_conversation(
            cp['conversation'],config['agent']['compaction'],binding['last_success_prompt_tokens'],
            proof['manifest']['usage']['model_requests_including_parent'],
            usage_messages=proof['usage_messages'],usage_kind='prompt_tokens')
    except L1Error as exc:
        if (exc.code!='pi_context_unsatisfiable' or manifest['status']['termination']!=exc.code or requests):
            raise EvidenceError('Context compaction failure differs from recorded stop') from exc
        return
    compacted=[e for e in events if e['kind']=='session.compacted' and (not requests or e['seq']<next(iter(requests.values()))['seq'])]
    if receipt is not None and (len(compacted)!=1 or compacted[0]['data']!=receipt):
        raise EvidenceError('Context pre-send compaction receipt mismatch')
    first=next((e for e in events if e['kind']=='checkpoint.created'),None)
    if first is not None:
        restored=read_json(root/first['data']['checkpoint']['path'])
        usage=proof['manifest']['usage']
        if (restored['conversation']!=expected or restored['budget_used']['model_requests']!=usage['model_requests_including_parent'] or
                restored['budget_used']['known_total_tokens']!=usage['known_total_tokens_including_parent']):
            raise EvidenceError('Context restored conversation or budget mismatch')
    fresh=list(requests.values())
    offset=proof['manifest']['usage']['model_requests_including_parent']
    if fresh and not fresh[0]['data']['request_id'].endswith(f':r{offset:06d}'):
        raise EvidenceError('Context continuation reuses spent request')
    if proof['manifest']['usage']['all_requests_have_usage'] is False and manifest['usage']['all_requests_have_usage'] is not False:
        raise EvidenceError('Context continuation fabricated complete usage')


def verify(root:Path,expected_index_sha256=None):
    root=Path(root); errors=[]; warnings=[]; manifest=None
    def error(code,detail): errors.append({'code':code,'detail':str(detail)})
    try:
        index=read_json(root/'evidence_index.json'); validate('index',index)
        actual_hash=file_digest(root/'evidence_index.json')
        if expected_index_sha256 and expected_index_sha256!=actual_hash: error('external_digest_mismatch','evidence_index.json')
        if inventory(root)!=index['files']: error('file_inventory_mismatch','files added, altered, or missing')
        manifest=read_json(root/'manifest.json'); validate('manifest',manifest)
        if manifest['episode_id']!=index['episode_id']: error('episode_mismatch','manifest/index')
        journal=scan(root/'trajectory.jsonl'); events=journal['events']
        if not journal['complete']: error('event_chain',journal['errors'])
        if len(events)!=index['event_count'] or (events[-1]['sha256'] if events else None)!=index['trajectory_head_sha256']:
            error('event_index_mismatch','count/head')
        all_ids={e['event_id']:e for e in events}; requests={}; responses={}; calls={}; results={}; snapshots={}; deliveries=set(); dispositions={}
        for e in events:
            if e['episode_id']!=manifest['episode_id'] or e['event_id']!=f"{manifest['episode_id']}:e{e['seq']:06d}":
                error('event_identity',e['event_id'])
            for key,target in e['links'].items():
                if target not in all_ids or all_ids[target]['seq']>=e['seq']: error('invalid_event_link',target)
            for reference in _refs(e):
                p=confined(root,reference['path']); regular_file(p)
                if file_digest(p)!=reference['sha256'] or p.stat().st_size!=reference['size_bytes']:
                    error('reference_digest',reference['path'])
            d=e['data']; kind=e['kind']
            if kind=='model.request':
                rid=d['request_id']
                if rid in requests: error('duplicate_request',rid)
                requests[rid]=e
            elif kind=='model.response':
                rid=d['request_id']
                if rid not in requests or rid in responses: error('orphan_or_duplicate_response',rid)
                responses[rid]=e
            elif kind=='tool.call':
                cid=d['call_id']
                if cid in calls or d['request_id'] not in responses: error('invalid_call',cid)
                calls[cid]=e
            elif kind=='tool.result':
                cid=d['call_id']
                if cid not in calls or cid in results: error('orphan_or_duplicate_result',cid)
                results[cid]=e
                raw=read_json(root/d['result']['path'])
                if raw['call_id']!=cid: error('result_identity',cid)
                for stream in ('stdout','stderr'):
                    expected=raw['observation'].get(stream)
                    if expected is not None and (root/d[stream]['path']).read_bytes().decode('utf-8','replace')!=expected:
                        error('observation_raw_mismatch',cid+':'+stream)
            elif kind in ('tool.error','tool.skipped','tool.rejected'):
                cid=d['call_id']
                if cid not in calls or cid in dispositions or cid in results: error('invalid_terminal_call_disposition',cid)
                dispositions[cid]=kind
            elif kind=='observation.delivered':
                cid=d['call_id']; rid=d['request_id']
                if cid not in results or rid not in requests or cid in deliveries: error('invalid_delivery',cid)
                deliveries.add(cid)
                # The exact prepared observation must be inside the next native request.
                observation=read_json(root/d['observation']['path'])
                body=read_json(root/requests[rid]['data']['body']['path'])
                if not observation_in_request(body,observation,d.get('native_id'),calls[cid]['data']['name']):
                    error('delivery_payload_mismatch',cid)
                if 'image' in results[cid]['data']:
                    import base64
                    encoded=base64.b64encode((root/results[cid]['data']['image']['path']).read_bytes()).decode('ascii')
                    if encoded not in json.dumps(last_observation_turn(body),ensure_ascii=False):
                        error('delivered_image_mismatch',cid)
            elif kind=='state.snapshot':
                snap=read_json(root/d['snapshot']['path']); sid=snap['snapshot_id']
                if sid in snapshots: error('duplicate_snapshot',sid)
                validate_snapshot(root,snap)
                snapshots[sid]=snap
        if manifest['counts']!={'events':len(events),'model_requests':len(requests),'model_responses':len(responses),
                                'tool_calls':len(calls),'tool_results':len(results),'observations_delivered':len(deliveries),
                                'snapshots':len(snapshots)}: error('counts_mismatch','manifest')
        # Verify native call projection, rather than trusting recorded normalized calls.
        config=read_json(root/'raw/resolved_config.json')
        if digest(config)!=manifest['config_sha256']: error('config_digest_mismatch','resolved config')
        _verify_response_consumption(root,manifest,events,requests,responses)
        _verify_context_overflow(root,manifest,events,requests)
        backend=None
        from ..models.native import GeminiBackend, OpenAICompatibleBackend
        cls={'gemini':GeminiBackend,'openai-compatible':OpenAICompatibleBackend}.get(config['model']['adapter'])
        if cls:
            backend=cls(config['model'])
            backend.native_pi_validation = config.get('agent',{}).get('validation_policy') == 'pi-native-0.73.1'
        else: warnings.append('native_projection_not_supported_for_this_plugin')
        parsed_ids={e['data']['request_id'] for e in events if e['kind']=='model.parsed'}
        if backend:
            for rid in parsed_ids:
                if rid not in responses:
                    error('parsed_without_response',rid); continue
                raw=(root/responses[rid]['data']['body']['path']).read_bytes()
                reply=backend.parse(raw,rid)
                normalized=[{'request_id':rid,'call_id':c.call_id,'native_id':c.native_id,'name':c.name,'arguments':c.arguments} for c in reply.calls]
                recorded=[e['data'] for e in calls.values() if e['data']['request_id']==rid]
                if normalized!=recorded: error('native_call_projection_mismatch',rid)
                parsed=[e['data'] for e in events if e['kind']=='model.parsed' and e['data']['request_id']==rid]
                expected={'request_id':rid,'usage':reply.usage,'reported_model_version':reply.model_version,
                          'finish_reason':reply.finish_reason,'text':reply.text}
                if parsed!=[expected]: error('native_reply_projection_mismatch',rid)
        unaccounted=set(calls)-set(results)-set(dispositions)
        if unaccounted and manifest['status']['evidence_completeness']=='complete': error('unterminated_tool_calls',sorted(unaccounted))
        if events and events[-1]['kind']=='episode.finished' and events[-1]['data']['status']!=manifest['status']:
            error('status_projection_mismatch','manifest vs final event')
        if manifest['task']:
            if read_json(root/'benchmark/description.json')!=manifest['task']: error('task_projection_mismatch','description')
            provenance=manifest['task'].get('provenance',{})
            for relative,sha in provenance.get('private_files_sha256',{}).items():
                p=confined(root,'benchmark/'+relative)
                if file_digest(p)!=sha: error('benchmark_source_digest_mismatch',relative)
        changes=[]
        for line in (root/'state_changes.jsonl').read_bytes().splitlines(keepends=True):
            if not line.endswith(b'\n'): raise EvidenceError('unterminated_state_change')
            changes.append(json.loads(line))
        from ..recording.state import StateStore
        expected_changes=[]
        for e in events:
            if e['kind']=='state.transition':
                d=e['data']; before=snapshots[d['before']]; after=snapshots[d['after']]
                expected_changes.append({'episode_id':manifest['episode_id'],'event_id':e['event_id'],
                    'before_snapshot_id':d['before'],'after_snapshot_id':d['after'],'changes':StateStore.diff(before,after)})
        if changes!=expected_changes: error('state_diff_mismatch','state_changes.jsonl')
        cp_index=read_json(root/'checkpoints/index.json')
        for item in cp_index['checkpoints']:
            cp=read_json(root/item['path']); sid=cp['snapshot_id']
            if sid not in snapshots: error('checkpoint_snapshot_missing',sid)
            if cp['qualification']['eligible'] and (not cp.get('conversation') or not cp['qualification'].get('image_id')):
                error('unqualified_checkpoint',item['path'])
            if cp.get('pending_calls')!=[]: error('checkpoint_pending_calls',item['path'])
            if cp['config_sha256']!=manifest['config_sha256']: error('checkpoint_config_mismatch',item['path'])
            if cp['qualification']['eligible'] and cp['qualification']['image_id']!=(manifest['runtime'] or {}).get('image_id'):
                error('checkpoint_image_mismatch',item['path'])
            cp_events=[e for e in events if e['kind']=='checkpoint.created' and e['data']['checkpoint']['path']==item['path']]
            if len(cp_events)!=1: error('checkpoint_index_coverage',item['path'])
            elif backend:
                following=[e for e in requests.values() if e['seq']>cp_events[0]['seq']]
                if following:
                    nxt=min(following,key=lambda e:e['seq'])
                    from ..runtime.tools import tools_for_config
                    tools=tools_for_config(config)
                    if backend.request(cp['conversation'],tools)!=read_json(root/nxt['data']['body']['path']):
                        error('checkpoint_conversation_mismatch',item['path'])
        grade=read_json(root/'official_grade.json')
        if grade.get('projection') and grade.get('official_raw') is not None:
            from ..evaluation.officeval import interpret
            if manifest['task'] and manifest['task']['benchmark']=='officeval' and grade['projection']['evaluation_validity']=='valid':
                expected=interpret(grade['official_raw'],manifest['task']['task_id'],grade['projection']['outcome_policy'])
                if expected!=grade['projection']: error('grade_projection_mismatch','official_grade.json')
        if grade.get('projection'):
            if grade['projection']['evaluation_validity']!=manifest['status']['evaluation_validity'] or grade['projection']['task_outcome']!=manifest['status']['task_outcome']:
                error('grade_status_mismatch','manifest vs grade')
        if grade.get('official_raw_sha256'):
            raw_path=root/'grading/worker-output/official.json'
            if file_digest(raw_path)!=grade['official_raw_sha256'] or read_json(raw_path)!=grade['official_raw']:
                error('official_raw_projection_mismatch','worker result')
        artifacts=read_json(root/'artifacts/inventory.json')
        if artifacts['snapshot_id'] is not None:
            snapshot=snapshots[artifacts['snapshot_id']]
            expected=[{'path':x['path'][len('submission/'):],'sha256':x['sha256'],'size_bytes':x['size_bytes']} for x in snapshot['files'] if x['path'].startswith('submission/')]
            if expected!=artifacts['files']: error('submission_snapshot_mismatch','artifact inventory')
            for item in artifacts['files']:
                p=confined(root,'artifacts/submission/'+item['path'])
                if file_digest(p)!=item['sha256'] or p.stat().st_size!=item['size_bytes']: error('submission_content_mismatch',item['path'])
        if manifest['status']['evidence_completeness']!='complete': warnings.append('evidence_incomplete')
        if manifest['recording_mode']!='live': warnings.append('test_double_not_live_qualification')
        return {'verified':not errors,'schema_version':VERSION,'episode_id':manifest['episode_id'],
                'index_sha256':actual_hash,'errors':errors,'warnings':warnings,'status':manifest['status']}
    except (OSError,ValueError,KeyError,TypeError,EvidenceError,ProtocolError) as exc:
        error('unreadable_or_invalid_package',exc)
        return {'verified':False,'errors':errors,'warnings':warnings,'schema_version':VERSION}


def observation_in_request(body,obs,native_id,name):
    native_error = None
    if isinstance(obs,dict) and obs.get('isError') is True and isinstance(obs.get('content'),list):
        content=obs['content']
        if all(isinstance(part,dict) and part.get('type')=='text' and isinstance(part.get('text'),str) for part in content):
            native_error='\n'.join(part['text'] for part in content)
    if 'contents' in body:
        for content in last_observation_turn(body):
            for part in content.get('parts',[]):
                fc=part.get('functionResponse')
                if fc and fc.get('name')==name and fc.get('id')==native_id:
                    if fc.get('response')==obs: return True
                    if native_error is not None and fc.get('response')=={'error':native_error}: return True
    for msg in reversed(last_observation_turn(body)):
        if msg.get('role')=='tool' and msg.get('tool_call_id')==native_id:
            if native_error is not None and msg.get('content')==native_error: return True
            try:
                if json.loads(msg['content'])==obs: return True
            except (ValueError,TypeError): pass
        if native_error is not None and msg.get('role')=='user' and isinstance(msg.get('content'),list):
            if any(isinstance(part,dict) and part.get('type')=='tool_result' and
                   part.get('tool_use_id')==native_id and part.get('is_error') is True and
                   part.get('content')==native_error for part in msg['content']): return True
    if native_error is not None:
        for item in body.get('input',[]):
            if isinstance(item,dict) and item.get('type')=='function_call_output' and item.get('call_id')==native_id and item.get('output')==native_error:
                return True
    return False


def validate_snapshot(root,snapshot):
    names=set(); total=0
    for item in snapshot['files']:
        from ..storage import safe_relative
        safe_relative(item['path'])
        if item['path'] in names: raise EvidenceError('duplicate file in snapshot')
        names.add(item['path'])
        p=confined(root,item['blob']); regular_file(p)
        if p.stat().st_size!=item['size_bytes'] or file_digest(p)!=item['sha256']: raise EvidenceError('snapshot blob mismatch')
        total+=item['size_bytes']
    for d in snapshot['directories']:
        from ..storage import safe_relative
        safe_relative(d['path'])
        if d['path'] in names: raise EvidenceError('duplicate path in snapshot')
        names.add(d['path'])
    if total!=snapshot['total_bytes'] or digest({'files':snapshot['files'],'directories':snapshot['directories']})!=snapshot['content_sha256']:
        raise EvidenceError('snapshot content hash mismatch')


def last_observation_turn(body):
    if 'contents' in body:
        contents=body.get('contents',[])
        for i in range(len(contents)-1,-1,-1):
            if contents[i].get('role')=='model': return contents[i+1:]
        return contents[-1:]
    messages=body.get('messages',[])
    for i in range(len(messages)-1,-1,-1):
        if messages[i].get('role')=='assistant': return messages[i+1:]
    return messages
