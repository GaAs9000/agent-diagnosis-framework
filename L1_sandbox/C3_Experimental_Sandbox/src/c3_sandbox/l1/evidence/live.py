"""Live evidence 0.2.0. Legacy 0.1.0 remains readable by the P1 verifier."""
from __future__ import annotations
import json
import os
from pathlib import Path
from ..core.contracts import VERSION, EvidenceError, ProtocolError
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
        backend=None
        from ..models.native import GeminiBackend, OpenAICompatibleBackend
        cls={'gemini':GeminiBackend,'openai-compatible':OpenAICompatibleBackend}.get(config['model']['adapter'])
        if cls: backend=cls(config['model'])
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
                    from ..runtime.tools import TOOLS
                    tools=TOOLS if config['model']['supports_images'] else TOOLS[:1]
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
    if 'contents' in body:
        for content in body['contents'][-1:]:
            for part in content.get('parts',[]):
                fc=part.get('functionResponse')
                if fc and fc.get('name')==name and fc.get('id')==native_id and fc.get('response')==obs: return True
    for msg in reversed(last_observation_turn(body)):
        if msg.get('role')=='tool' and msg.get('tool_call_id')==native_id:
            try:
                if json.loads(msg['content'])==obs: return True
            except (ValueError,TypeError): pass
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
    if 'contents' in body: return body['contents'][-1:]
    messages=body.get('messages',[])
    for i in range(len(messages)-1,-1,-1):
        if messages[i].get('role')=='assistant': return messages[i+1:]
    return messages
