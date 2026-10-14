"""Synthetic source and model fixtures. Never evidence of real L1/model execution."""
from __future__ import annotations
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import uuid
import zipfile
import pytest
from l2_sandbox.io import canonical,digest,sha_bytes,inventory,write_json,write_jsonl,write_new,now,loads
from l2_sandbox.evidence.l1 import diff_snapshots,L1Reader
from l2_sandbox.config import load


def docx_bytes(*,empty_row=True,black=False,style=False,entity=False):
    row='<w:tr><w:tc><w:p><w:r><w:t>text</w:t></w:r></w:p></w:tc></w:tr>'
    if empty_row:row+='<w:tr><w:tc><w:p/></w:tc></w:tr>'
    borders='<w:tblBorders><w:bottom w:val="single" w:color="000000" w:sz="8"/></w:tblBorders>' if black else ''
    st='<w:tblStyle w:val="CustomStyle"/>' if style else ''
    xml=f'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:tbl><w:tblPr>{borders}{st}</w:tblPr>{row}</w:tbl><w:p><w:r><w:t>meeting content</w:t></w:r></w:p></w:body></w:document>'
    if entity:xml='<!DOCTYPE x [<!ENTITY secret SYSTEM "file:///etc/passwd">]>'+xml
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:z.writestr('word/document.xml',xml)
    return out.getvalue()


def make_source(root:Path,*,recording_mode='test_fixture',lineage=None,missing_submission=False,protocol_error=False,tool_exit=0):
    root.mkdir(parents=True,exist_ok=False);eid='ep_'+uuid.uuid4().hex
    source_config={'budget':{'max_model_requests':5,'max_tool_calls':10,'wall_time_seconds':600,'max_total_tokens':None},
        'model':{'adapter':'gemini','kind':'api','model':'SYNTHETIC_TEST_MODEL','base_url':'https://generativelanguage.googleapis.com/v1beta','parameters':{'temperature':0}},
        'agent':{'system_prompt':'Fixture only'},'runtime':{'adapter':'test-only'}}
    write_json(root,'raw/resolved_config.json',source_config)
    public={'benchmark':'officeval','task_id':'officeval_003','prompt':'Remove empty rows in the supplied document.','assets':[]}
    write_json(root,'benchmark/public_task.json',public)
    write_json(root,'benchmark/rubric.json',{'PRIVATE_RUBRIC_SENTINEL':'Do not expose through Analyst tools'})
    write_new(root,'benchmark/source/private.py',b'raise RuntimeError("NEVER EXECUTE THIS VERIFIER")\n')
    old=docx_bytes();new=docx_bytes(empty_row=False,black=True)
    def sf(path,data):
        sha=sha_bytes(data);blob='states/blobs/'+sha
        if not (root/blob).exists():write_new(root,blob,data)
        return {'path':path,'blob':blob,'sha256':sha,'size_bytes':len(data),'mode':0o644}
    initial=[sf('input/测试.docx',old)]
    final=[sf('input/测试.docx',new)]
    if not missing_submission:final.append(sf('submission/结果.docx',new))
    dirs=[{'path':'input','mode':0o755},{'path':'submission','mode':0o755}]
    snaps=[]
    for i,ff in enumerate([initial,final]):
        s={'snapshot_id':f's{i:06d}','label':'fixture','scope':'workspace_files_and_directories','files':ff,'directories':dirs,
           'total_bytes':sum(f['size_bytes'] for f in ff),'content_sha256':digest({'files':ff,'directories':dirs})}
        write_json(root,f"states/snapshots/{s['snapshot_id']}.json",s);snaps.append(s)
    def ref(path):
        b=(root/path).read_bytes();return {'path':path,'sha256':sha_bytes(b),'size_bytes':len(b)}
    events=[]
    def ev(kind,data,links=None):
        x={'schema_version':'0.2.0','episode_id':eid,'event_id':f'{eid}:e{len(events):06d}','seq':len(events),'kind':kind,
           'data':data,'links':links or {},'observed_at':now(),'previous_sha256':events[-1]['sha256'] if events else None}
        x['sha256']=digest(x);events.append(x);return x
    ev('episode.started',{'config':ref('raw/resolved_config.json')})
    ev('state.snapshot',{'snapshot_id':'s000000','snapshot':ref('states/snapshots/s000000.json')})
    conversation={'systemInstruction':{'parts':[{'text':'Fixture only'}]},'contents':[{'role':'user','parts':[{'text':public['prompt']}]}]}
    cp={'schema_version':'0.2.0','episode_id':eid,'checkpoint_id':'cp000000','snapshot_id':'s000000','conversation':copy.deepcopy(conversation),
        'pending_calls':[],'pending_observations':[],'budget_used':{'model_requests':0,'tool_calls':0,'agent_seconds':0,'known_total_tokens':0},
        'qualification':{'eligible':True,'image_id':'sha256:'+'a'*64,'scopes':['workspace_files','native_conversation']},'config_sha256':digest(source_config)}
    write_json(root,'checkpoints/cp000000.json',cp)
    ev('checkpoint.created',{'checkpoint':ref('checkpoints/cp000000.json')})
    rid=eid+':r000000';cid=rid+':c0'
    write_json(root,'raw/model/r000000/request.json',conversation)
    req=ev('model.request',{'request_id':rid,'body':ref('raw/model/r000000/request.json')})
    call={'name':'run','args':{'command':'TEST-AUTHORED, NOT EXECUTED'},'id':'native0'}
    response={'candidates':[{'content':{'role':'model','parts':[{'functionCall':call,'thoughtSignature':'opaque-native-signature'}]},'finishReason':'STOP'}]}
    write_json(root,'raw/model/r000000/response.bin',response)
    write_json(root,'raw/model/r000000/http.json',{'status_code':200,'fixture':True})
    resp=ev('model.response',{'request_id':rid,'body':ref('raw/model/r000000/response.bin'),'http':ref('raw/model/r000000/http.json'),'status_code':200},{'request':req['event_id']})
    ev('model.parsed',{'request_id':rid})
    tc=ev('tool.call',{'request_id':rid,'call_id':cid,'native_id':'native0','name':'run','arguments':call['args']},{'response':resp['event_id']})
    obs={'exit_code':tool_exit,'stdout':'fixture stdout\n','stderr':'fixture error\n' if tool_exit else '', 'output_complete':True,'stop_reason':None}
    write_json(root,'raw/tools/result.json',{'call_id':cid,'observation':obs,'provenance':{'test_fixture':True}})
    write_new(root,'raw/tools/stdout.bin',obs['stdout'].encode());write_new(root,'raw/tools/stderr.bin',obs['stderr'].encode())
    result=ev('tool.result',{'call_id':cid,'result':ref('raw/tools/result.json'),'stdout':ref('raw/tools/stdout.bin'),'stderr':ref('raw/tools/stderr.bin')},{'call':tc['event_id']})
    ev('state.snapshot',{'snapshot_id':'s000001','snapshot':ref('states/snapshots/s000001.json')},{'tool_result':result['event_id']})
    change=ev('state.transition',{'before':'s000000','after':'s000001'},{'tool_result':result['event_id']})
    write_jsonl(root,'state_changes.jsonl',[{'episode_id':eid,'event_id':change['event_id'],'before_snapshot_id':'s000000','after_snapshot_id':'s000001','changes':diff_snapshots(*snaps)}])
    cp1={**cp,'checkpoint_id':'cp000001','snapshot_id':'s000001','budget_used':{'model_requests':1,'tool_calls':1,'agent_seconds':1,'known_total_tokens':3}}
    cp1['conversation']=copy.deepcopy(conversation)
    cp1['conversation']['contents'] += [response['candidates'][0]['content'],{'role':'user','parts':[{'functionResponse':{'id':'native0','name':'run','response':obs}}]}]
    write_json(root,'checkpoints/cp000001.json',cp1);ev('checkpoint.created',{'checkpoint':ref('checkpoints/cp000001.json')})
    write_json(root,'checkpoints/index.json',{'checkpoints':[{'checkpoint_id':'cp000000','path':'checkpoints/cp000000.json','eligible':True},{'checkpoint_id':'cp000001','path':'checkpoints/cp000001.json','eligible':True}]})
    rid2=eid+':r000001';write_json(root,'raw/model/r000001/request.json',cp1['conversation'])
    req2=ev('model.request',{'request_id':rid2,'body':ref('raw/model/r000001/request.json')})
    response2={'candidates':[{'content':{'role':'model','parts':[{'text':'fixture done'}]},'finishReason':'MALFORMED_FUNCTION_CALL' if protocol_error else 'STOP'}]}
    write_json(root,'raw/model/r000001/response.bin',response2);write_json(root,'raw/model/r000001/http.json',{'status_code':200,'fixture':True})
    ev('model.response',{'request_id':rid2,'body':ref('raw/model/r000001/response.bin'),'http':ref('raw/model/r000001/http.json'),'status_code':200},{'request':req2['event_id']})
    write_json(root,'observations/tool0.json',obs)
    ev('observation.delivered',{'call_id':cid,'request_id':rid2,'native_id':'native0','observation':ref('observations/tool0.json')},{'request':req2['event_id'],'result':result['event_id']})
    if protocol_error:ev('episode.error',{'code':'protocol_error','message':'fixture malformed response'})
    else:ev('model.parsed',{'request_id':rid2})
    rawgrade={'status':'error' if missing_submission else 'ok','error':'missing' if missing_submission else None,'id':'003','file_name':'结果.docx',
              'dim1_pass':not missing_submission,'dim1_reason':'','dim2_items':[],'total_score':0,'max_score':0 if missing_submission else 15}
    projection={'evaluation_validity':'invalid' if missing_submission else 'valid','task_outcome':'unknown'}
    write_json(root,'official_grade.json',{'official_raw':rawgrade,'projection':projection})
    art=[]
    if not missing_submission:
        write_new(root,'artifacts/submission/结果.docx',new)
        f=final[-1];art=[{'path':'结果.docx','sha256':f['sha256'],'size_bytes':f['size_bytes']}]
    write_json(root,'artifacts/inventory.json',{'episode_id':eid,'snapshot_id':'s000001','files':art})
    status={'termination':'protocol_error' if protocol_error else 'assistant_finished','execution_validity':'invalid' if protocol_error else 'valid',
            'protocol_validity':'invalid' if protocol_error else 'valid','evaluation_validity':projection['evaluation_validity'],
            'task_outcome':'unknown','evidence_completeness':'complete'}
    ev('episode.finished',{'status':status})
    m={'schema_version':'0.2.0','episode_id':eid,'recording_mode':recording_mode,
       'created_at':now(),'finished_at':now(),
       'task':{'benchmark':'officeval','task_id':'officeval_003','prompt':public['prompt']},
       'agent':{'identity':{'adapter':'gemini','requested_model':'SYNTHETIC_TEST_MODEL'},'model_configuration':{'adapter':'gemini','model':'SYNTHETIC_TEST_MODEL'}},
       'coverage':[{'code':'test_fixture','detail':'synthetic fixture'}],
       'usage':{'all_requests_have_usage':True,'known_total_tokens_including_parent':3},
       'status':status,'config_sha256':digest(source_config),'runtime':{'image_id':'sha256:'+'a'*64},'capabilities':{'test_fixture':True},
       'counts':{'events':len(events),'model_requests':2,'model_responses':2,'tool_calls':1,'tool_results':1,'observations_delivered':1,'snapshots':2},'lineage':lineage}
    write_json(root,'manifest.json',m);write_jsonl(root,'trajectory.jsonl',events)
    write_json(root,'evidence_index.json',{'schema_version':'0.2.0','episode_id':eid,'files':inventory(root),
        'event_count':len(events),'trajectory_head_sha256':events[-1]['sha256'],'trust_scope':'SYNTHETIC FIXTURE'})
    assert L1Reader(root).report['verified']
    return root

@pytest.fixture
def source(tmp_path):return make_source(tmp_path/'source')

@pytest.fixture
def config():return load()


def diagnosis_from_tool_results(body):
    fs=[]
    for message in body.get('contents',[]):
        for part in message.get('parts',[]):
            fr=part.get('functionResponse',{})
            fs += [x for x in fr.get('response',{}).get('items',[]) if 'fact_id' in x and 'value' in x]
    task=next((f for f in fs if f['fact_type']=='task_instruction'),None)
    fact=next(f for f in fs if f['fact_type']=='tool_exit')
    loc={'localization_id':'loc_fixture','category':'tool_execution_anomaly','granularity':'event','event_ids':fact['event_ids'],
         'statement':'A tool exit was recorded; fixture analysis only.','object_scope':{},'before_version':None,'after_version':None,
         'supporting_fact_ids':[fact['fact_id']],'counterevidence_fact_ids':[],
         'task_relation':{'requirement_fact_id':task['fact_id'] if task else None,'interpretation':'Execution observation only.','origin':'analyst_interpretation'},
         'uncertainty':['No causal experiment.']}
    hyp={'hypothesis_id':'hyp_fixture','localization_ids':['loc_fixture'],'mechanism_statement':'The contribution of this tool return remains unknown.',
         'supporting_fact_ids':[fact['fact_id']],'counterevidence_fact_ids':[],'alternatives':['Independent later issue.','Observed operation may be irrelevant.'],
         'assumptions':['Only the supplied evidence is considered.'],'unobserved_requirements':['Controlled results missing.'],
         'predictions':[{'intervention_template':'generic_reminder','control_template':'resume_unchanged','direction':'unknown',
                         'proximal_prediction':'Measure a subsequent action.','endpoint_prediction':'No success guaranteed.','uncertainty':'Untested.'}],
         'abstention_reason':'Cannot determine mechanism.','verdict':'untested','scope':'this_episode_and_explicit_future_protocol_only'}
    return {'summary':'TEST FIXTURE ANALYST, not a live model.','localizations':[loc],'hypotheses':[hyp],'abstentions':['No verified cause.']}


class FixtureBackend:
    """Exercises the L1 port, not a replacement production HTTP implementation."""
    def __init__(self,responses=None):self.responses=list(responses or []);self.requests=[]
    def identity(self):return {'adapter':'fixture','kind':'test','requested_model':'TEST-NOT-A-MODEL'}
    def initial(self,system,prompt):return {'systemInstruction':{'parts':[{'text':system}]},'contents':[{'role':'user','parts':[{'text':prompt}]}]}
    def request(self,conversation,tools):return {**copy.deepcopy(conversation),'tools':[{'functionDeclarations':tools}]}
    def send(self,body,timeout):
        request=loads(body);self.requests.append(request)
        if self.responses:
            item=self.responses.pop(0);response=item(request) if callable(item) else item
        elif len(self.requests)==1:
            response={'calls':[{'name':'get_requirement_evidence','args':{'start':0,'limit':30}}]}
        else:
            response={'calls':[{'name':'submit_diagnosis','args':{'diagnosis_json':canonical(diagnosis_from_tool_results(request)).decode()}}]}
        if isinstance(response,Exception):raise response
        if isinstance(response,bytes):native=response
        else:
            parts=[{'functionCall':{'name':c['name'],'args':c.get('args',{}),'id':f'test-{len(self.requests)}-{i}'},'thoughtSignature':'test-signature'} for i,c in enumerate(response.get('calls',[]))]
            if 'text' in response:parts.append({'text':response['text']})
            native=canonical({'candidates':[{'content':{'role':'model','parts':parts},'finishReason':'STOP'}],'modelVersion':'TEST-FIXTURE','usageMetadata':{'totalTokenCount':10}})
        return SimpleNamespace(status=200,body=native,metadata={'test_fixture':True,'body_truncated':False,'transport_error':None})
    def parse(self,body,request_id):
        v=loads(body);content=v['candidates'][0]['content'];calls=[];text=[]
        for i,p in enumerate(content['parts']):
            if 'functionCall' in p:
                f=p['functionCall'];calls.append(SimpleNamespace(call_id=request_id+':c'+str(i),native_id=f.get('id'),name=f['name'],arguments=f['args']))
            if 'text' in p:text.append(p['text'])
        return SimpleNamespace(calls=calls,native_message=content,usage=v.get('usageMetadata'),model_version=v.get('modelVersion'),finish_reason='STOP',text='\n'.join(text) or None)
    def accept(self,conversation,reply,results):
        c=copy.deepcopy(conversation);c['contents'].append(reply.native_message)
        if results:c['contents'].append({'role':'user','parts':[{'functionResponse':{'name':call.name,'id':call.native_id,'response':r.observation}} for call,r in results]})
        return c
