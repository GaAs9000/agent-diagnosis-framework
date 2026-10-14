"""A closed read-only evidence view, never a path-access or shell tool."""
from __future__ import annotations
import copy
from ..io import AccessDenied, BudgetExceeded, canonical, digest

TOOLS = [
    {'name':'get_episode_summary','description':'Read the original episode task, selected status and coverage. This is retrospective, not a prefix prediction.','parameters':{'type':'object','properties':{}}},
    {'name':'list_events','description':'Page event metadata and associated fact IDs; no sibling episodes or private verifier sources.','parameters':{'type':'object','properties':{'start':{'type':'integer'},'limit':{'type':'integer'},'kind':{'type':'string'}}}},
    {'name':'get_requirement_evidence','description':'Retrieve fact records and public task requirements by fact type or a literal search term. Every returned ID is citable.','parameters':{'type':'object','properties':{'start':{'type':'integer'},'limit':{'type':'integer'},'fact_type':{'type':'string'},'query':{'type':'string'}}}},
    {'name':'read_evidence','description':'Read one registered evidence reference ID. Not a filesystem path. Arbitrary/private references are refused.','parameters':{'type':'object','properties':{'ref_id':{'type':'string'}},'required':['ref_id']}},
    {'name':'get_state_diff','description':'Retrieve file transition facts; optional literal workspace-relative path filter, not a filesystem read.','parameters':{'type':'object','properties':{'start':{'type':'integer'},'limit':{'type':'integer'},'file_path':{'type':'string'}}}},
    {'name':'inspect_object_version','description':'Page a registered DOCX probe. Only direct XML properties; semantic cross-version identity and rendering remain unknown.','parameters':{'type':'object','properties':{'probe_id':{'type':'string'},'start':{'type':'integer'},'limit':{'type':'integer'}},'required':['probe_id']}},
    {'name':'list_qualified_checkpoints','description':'View recorded checkpoint qualification and remaining-budget context, not native private conversation or validation branch outcomes.','parameters':{'type':'object','properties':{'start':{'type':'integer'},'limit':{'type':'integer'}}}},
]


class EvidenceView:
    def __init__(self,builder,config):
        self.builder=builder;self.config=config;self.seen_fact_ids=set();self.bytes_read=0;self.calls=0
        task=next((f['value'] for f in builder.facts if f['fact_type']=='task_instruction'),None)
        status=next((f['value'] for f in builder.facts if f['fact_type']=='episode_status'),None)
        self.summary={'analysis_mode':'retrospective','task':task,'status':status,
            'facts':len(builder.facts),'events':len(builder.events),'object_versions':len(builder.objects),
            'coverage':builder.coverage,'causal_status':'untested','view_policy':'original_episode_only_no_private_verifier_or_sibling_results'}
        self.manifest={'profile':'retrospective-public-evidence-with-original-grade',
            'source_index_sha256':builder.reader.index_sha256,'summary':self.summary,
            'allowed_tools':[x['name'] for x in TOOLS],
            'forbidden':['arbitrary_file_paths','benchmark/source','benchmark/rubric.json','sibling_episodes','validation_outcomes','shell','network'],
            'fact_ids':[f['fact_id'] for f in builder.facts],'event_ids':[e['event_id'] for e in builder.events],
            'ref_ids':sorted(set(rid for rid,r in builder.refs.records.items() if r['analyst_visible']) | set(builder.refs.declared)),
            'declared_lazy_source_resources':list(builder.refs.declared.values()),
            'probe_ids':list(builder.objects),'private_rubric_profile':False}
        self.manifest['view_sha256']=digest(self.manifest)

    @staticmethod
    def page(values,args):
        start=args.get('start',0);limit=args.get('limit',8)
        if type(start) is not int or type(limit) is not int or start<0 or not 1<=limit<=30:raise AccessDenied('Invalid pagination; limit is 1..30')
        return {'items':values[start:start+limit],'total':len(values),'start':start,'next_start':start+limit if start+limit<len(values) else None}

    def call(self,name,args):
        if self.calls>=self.config['budget']['max_tool_calls']:raise BudgetExceeded('Evidence tool call budget reached')
        self.calls+=1
        schema=next((t['parameters'] for t in TOOLS if t['name']==name),None)
        if schema is None or not isinstance(args,dict):raise AccessDenied('Unknown read-only tool')
        from jsonschema import Draft202012Validator
        schema={**schema,'additionalProperties':False}
        if list(Draft202012Validator(schema).iter_errors(args)):raise AccessDenied('Invalid tool arguments')
        b=self.builder
        if name=='get_episode_summary':result=copy.deepcopy(self.summary)
        elif name=='list_events':result=self.page([e for e in b.events if not args.get('kind') or e['kind']==args['kind']],args)
        elif name=='get_requirement_evidence':
            selected=[f for f in b.facts if (not args.get('fact_type') or f['fact_type']==args['fact_type'])
                      and (not args.get('query') or args['query'].casefold() in canonical(f).decode().casefold())]
            result=self.page(selected,args)
        elif name=='read_evidence':result={'ref_id':args['ref_id'],'value':b.refs.read(args['ref_id'])}
        elif name=='get_state_diff':
            selected=[f for f in b.facts if f['fact_type'] in ('file_transition','object_structure_change')
                      and (not args.get('file_path') or f['value'].get('path',f['value'].get('file_path'))==args['file_path'])]
            result=self.page(selected,args)
        elif name=='inspect_object_version':
            if args['probe_id'] not in b.objects:raise AccessDenied('Object version is not in view')
            obj=b.objects[args['probe_id']]
            result={'probe_id':obj['probe_id'],'status':obj['status'],'summary':obj['summary'],'unknowns':obj['unknowns'],
                    'objects':self.page(obj['objects'],args)}
        elif name=='list_qualified_checkpoints':result=self.page(b.checkpoints,args)
        else:raise AccessDenied(name)
        limit=self.config['budget']['max_tool_response_bytes']
        # Whole records are omitted rather than silently truncating their values.
        # Oversized single evidence returns an explicit prefix, not a complete object.
        result=copy.deepcopy(result)
        if len(canonical(result))>limit:
            if isinstance(result,dict) and 'items' in result:
                kept=[]
                for item in result['items']:
                    candidate={**result,'items':kept+[item],'truncated':True}
                    if len(canonical(candidate))>limit:break
                    kept.append(item)
                result={**result,'items':kept,'truncated':True,'next_start':result['start']+len(kept),
                        'oversized_next_record':not bool(kept)}
                if not kept:result['advice']='Use a narrower query, read a specific evidence reference or increase the configured tool response budget.'
            else:
                text=canonical(result).decode('utf-8')
                result={'truncated':True,'original_bytes':len(canonical(result)), 'prefix':text[:max(1,limit//4)],
                        'complete_value_returned':False}
        encoded=canonical(result)
        if self.bytes_read+len(encoded)>self.config['budget']['max_total_read_bytes']:raise BudgetExceeded('Evidence read byte budget reached')
        self.bytes_read+=len(encoded)
        def mark(v):
            if isinstance(v,dict):
                if 'fact_id' in v and 'value' in v and 'source_ref_ids' in v:self.seen_fact_ids.add(v['fact_id'])
                for x in v.values():mark(x)
            elif isinstance(v,list):
                for x in v:mark(x)
        mark(result)
        return result
