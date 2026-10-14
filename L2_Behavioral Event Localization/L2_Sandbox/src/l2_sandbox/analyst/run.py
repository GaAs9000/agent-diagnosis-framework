"""Budgeted single Analyst with immutable raw calls, retrieval, repair and rejection logs."""
from __future__ import annotations
import copy
from importlib.resources import files
import time
from ..io import (BudgetExceeded, InvalidOutput, IntegrityError, canonical, digest, loads, write_new, write_json)
from ..validation import check_diagnosis
from ..evidence.view import TOOLS
from .backend import create, provenance, observation, feedback

SUBMIT={'name':'submit_diagnosis','description':'Submit the final diagnosis as strict JSON text matching the supplied diagnosis schema. This is NOT an experiment result.',
        'parameters':{'type':'object','properties':{'diagnosis_json':{'type':'string'}},'required':['diagnosis_json']}}


def run(view, config, output, journal, *, backend=None):
    if backend is None: backend=create(config['model'])
    backend_meta=provenance(backend);write_json(output,'raw/analyst/backend.json',backend_meta)
    budget=config['budget'];started=time.monotonic();deadline=started+budget['wall_seconds'];repairs=0
    schema=loads(files('l2_sandbox').joinpath('schemas/contracts.json').read_bytes())
    system=files('l2_sandbox').joinpath('analyst/system.txt').read_text('utf8')
    system+=f"\nHard budget: {budget['max_model_requests']} model requests, {budget['max_tool_calls']} retrieval calls, {budget['wall_seconds']} seconds. At most {budget['max_findings']} localizations and hypotheses."
    prompt={'task':'Analyze the source episode. Do not use unseen branch results.',
            'view_summary':view.summary,'output_schema':schema['$defs']['diagnosis']}
    # Referenced schema definitions are already inlined by the distributed generator.
    if config['mode']=='structured-once':
        selected=[]
        for fact in view.builder.facts:
            candidate={**prompt,'facts':selected+[fact]}
            if len(canonical(candidate))>budget['max_total_read_bytes']:break
            selected.append(fact);view.seen_fact_ids.add(fact['fact_id'])
        prompt['facts']=selected;prompt['omitted_fact_count']=len(view.builder.facts)-len(selected)
        view.bytes_read=len(canonical(prompt))
        write_json(output,'raw/analyst/single_shot_view.json',prompt)
        journal.emit('single_shot_view',{'fact_ids':sorted(view.seen_fact_ids),'bytes':view.bytes_read,'omitted_facts':prompt['omitted_fact_count']})
    conversation=backend.initial(system,canonical(prompt).decode())
    write_json(output,'raw/analyst/initial_view.json',prompt)
    tools=[SUBMIT] if config['mode']=='structured-once' else [*TOOLS,SUBMIT]
    usage_records=[];request_count=0;tool_results_count=0
    def validate_candidate(text,request_index,origin):
        nonlocal repairs
        n=repairs
        write_new(output,f'raw/analyst/submissions/r{request_index:03d}_attempt{n:03d}.txt',text.encode('utf8'))
        try:
            candidate=loads(text)
            checks=check_diagnosis(candidate,view.builder.facts,view.builder.events,
                                   seen_fact_ids=view.seen_fact_ids,max_findings=budget['max_findings'])
        except (InvalidOutput,IntegrityError) as exc:
            repairs+=1
            journal.emit('output.rejected',{'request_index':request_index,'origin':origin,'error':str(exc),'repair_number':repairs})
            if repairs>budget['max_output_repairs']:raise InvalidOutput('Output repair budget exhausted: '+str(exc)) from exc
            return None,{'accepted':False,'validation_error':str(exc),'instruction':'Revise only using available evidence. Preserve uncertainty and untested verdict.'}
        journal.emit('output.accepted',{'request_index':request_index,'origin':origin,'diagnosis_sha256':digest(candidate),
                                       'seen_fact_ids':sorted(view.seen_fact_ids),'repairs':repairs})
        return (candidate,checks),{'accepted':True,'causal_verdict':'untested'}
    for turn in range(budget['max_model_requests']):
        remaining=deadline-time.monotonic()
        if remaining<=0:raise BudgetExceeded('Analyst wall time budget reached')
        request_count+=1;request_id=f'analyst-r{turn:03d}'
        request=backend.request(conversation,tools);body=canonical(request)
        reqpath=f'raw/analyst/r{turn:03d}/request.bin';write_new(output,reqpath,body)
        journal.emit('model.request',{'request_id':request_id,'path':reqpath,'sha256':digest(request)})
        receipt=backend.send(body,min(remaining,config['model']['timeout_seconds']))
        write_new(output,f'raw/analyst/r{turn:03d}/response.bin',receipt.body)
        write_json(output,f'raw/analyst/r{turn:03d}/http.json',receipt.metadata)
        from ..io import sha_bytes
        journal.emit('model.response',{'request_id':request_id,'path':f'raw/analyst/r{turn:03d}/response.bin',
                                      'sha256':sha_bytes(receipt.body),'status':receipt.status,'body_truncated':receipt.metadata.get('body_truncated',False)})
        if receipt.status is None or not 200<=receipt.status<300 or receipt.metadata.get('body_truncated') or receipt.metadata.get('transport_error'):
            raise InvalidOutput('Analyst response/transport invalid; saved raw bytes retained')
        reply=backend.parse(receipt.body,request_id)
        usage_records.append(reply.usage)
        journal.emit('model.parsed',{'request_id':request_id,'usage':reply.usage,'reported_model_version':reply.model_version,
                                    'finish_reason':reply.finish_reason})
        if not reply.calls:
            if not isinstance(reply.text,str):raise InvalidOutput('Model returned neither tool calls nor JSON diagnosis')
            accepted,msg=validate_candidate(reply.text,turn,'assistant_text')
            if accepted:
                return {'diagnosis':accepted[0],'checks':accepted[1],'usage_records':usage_records,'request_count':request_count,
                        'retrieval_count':view.calls,'read_bytes':view.bytes_read,'repairs':repairs,'backend':backend_meta}
            conversation=backend.accept(conversation,reply,[])
            conversation=feedback(conversation,canonical(msg).decode())
            continue
        results=[]
        submission_calls=[c for c in reply.calls if c.name=='submit_diagnosis']
        for i,call in enumerate(reply.calls):
            journal.emit('tool.request',{'request_id':request_id,'call_id':call.call_id,'native_id':call.native_id,
                                        'name':call.name,'arguments':call.arguments})
            accepted=None
            if submission_calls and len(reply.calls)!=1:
                result={'error':'A final submission must be the only tool in its response.'}
            elif call.name=='submit_diagnosis':
                if set(call.arguments)!={'diagnosis_json'} or not isinstance(call.arguments['diagnosis_json'],str):
                    raise InvalidOutput('Invalid submit_diagnosis arguments')
                accepted,result=validate_candidate(call.arguments['diagnosis_json'],turn,'submit_diagnosis')
            else:
                try:result=view.call(call.name,call.arguments)
                except BudgetExceeded:raise
                except Exception as exc:
                    result={'error':type(exc).__name__,'message':str(exc),'not_executed':True}
            resultpath=f'raw/analyst/tools/r{turn:03d}_c{i:03d}.json';write_json(output,resultpath,result)
            journal.emit('tool.result',{'request_id':request_id,'call_id':call.call_id,'name':call.name,
                                       'path':resultpath,'sha256':digest(result),'seen_fact_ids':sorted(view.seen_fact_ids)})
            tool_results_count+=1
            if accepted:
                return {'diagnosis':accepted[0],'checks':accepted[1],'usage_records':usage_records,'request_count':request_count,
                        'retrieval_count':view.calls,'read_bytes':view.bytes_read,'repairs':repairs,'backend':backend_meta}
            results.append((call,observation(result)))
        conversation=backend.accept(conversation,reply,results)
    raise BudgetExceeded('Analyst model request budget reached before valid output')
