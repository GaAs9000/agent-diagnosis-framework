from __future__ import annotations
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import pytest
from c3_sandbox.l1.core.config import DEFAULTS
from c3_sandbox.l1.core.contracts import HTTPResult, Result
from c3_sandbox.l1.models.native import GeminiBackend
from c3_sandbox.l1.benchmarks.officeval.adapter import BundleOfficeVal, NativeOfficeVal
from c3_sandbox.l1.benchmarks.officeval.source import lock_local
from c3_sandbox.l1.storage import canonical, write_json, utc_now
from c3_sandbox.l1.evaluation.officeval import interpret

ROOT=Path(__file__).resolve().parents[2]
BUNDLE=ROOT/'runs/integrated-delivery-001/bundle'

@pytest.fixture
def config():
    c=copy.deepcopy(DEFAULTS)
    c['benchmark'].update(adapter='officeval-bundle',task_id='officeval_003',bundle_dir=str(BUNDLE))
    c['model']['api_key_env']=None
    c['budget'].update(max_model_requests=3,max_tool_calls=3,wall_time_seconds=60)
    c['recording']['observation_limit_chars']=20
    return c


def gemini_call(name='run',args=None,*,native_id='native-call-1',signature=True):
    call={'name':name,'args':args if args is not None else {'command':'submit_fixture'}}
    if native_id is not None: call['id']=native_id
    part={'functionCall':call}
    if signature: part['thoughtSignature']='opaque-signature-retain-exactly'
    return {'candidates':[{'content':{'role':'model','parts':[part]},'finishReason':'STOP'}],
            'usageMetadata':{'totalTokenCount':12},'modelVersion':'test-fixture-not-a-real-model'}


def gemini_done():
    return {'candidates':[{'content':{'role':'model','parts':[{'text':'Finished fixture.'}]},'finishReason':'STOP'}],
            'usageMetadata':{'totalTokenCount':7},'modelVersion':'test-fixture-not-a-real-model'}

class ScriptedGemini(GeminiBackend):
    def __init__(self,config,responses=None):
        super().__init__(config); self.responses=list(responses or [gemini_call(),gemini_done()]); self.bodies=[]
    def send(self,body,timeout):
        self.bodies.append(json.loads(body))
        value=self.responses.pop(0)
        if isinstance(value,BaseException): raise value
        if isinstance(value,HTTPResult): return value
        return HTTPResult(200,canonical(value),{'fixture':True,'started_at':utc_now(),'finished_at':utc_now(),
                         'transport_error':None,'body_truncated':False})

class TrustedFixtureRuntime:
    """Only fixed test commands; not registered and never exposed via the CLI."""
    is_quiescent=True
    def __init__(self,*,fail=False,timeout=False,cap=False,symlink=False):
        self.fail=fail; self.timeout=timeout; self.cap=cap; self.symlink=symlink; self.executed=[]
    def preflight(self): return {'adapter':'test_fixture','image_id':'sha256:'+'0'*64,'is_test_double':True}
    def qualification(self): return {'eligible':True,'image_id':'sha256:'+'0'*64,'scopes':['workspace_files','native_conversation'],
                                      'reason_codes':[],'test_only':True}
    def execute(self,call,workspace,timeout):
        self.executed.append(call)
        if self.fail: raise RuntimeError('injected tool failure')
        if self.symlink:
            (workspace/'bad-link').symlink_to('/etc/passwd')
        else:
            # Real subprocess, but a test-authored fixed operation, not agent-authored code.
            script="from pathlib import Path; import shutil; root=Path('.'); src=next((root/'input').glob('*.docx')); shutil.copyfile(src,root/'submission'/src.name); print('fixture execution '+'x'*60)"
            p=subprocess.run([sys.executable,'-c',script],cwd=workspace,capture_output=True,check=True)
            data=p.stdout
            return Result({'stdout':data.decode(),'stderr':'','exit_code':0,
                           'output_complete':not self.cap,'stop_reason':'timeout' if self.timeout else None},data,b'',
                          provenance={'test_fixture':True,'real_fixed_subprocess':True,'quiescent':True})
        return Result({'stdout':'','stderr':'','exit_code':0},provenance={'test_fixture':True,'quiescent':True})

class FixtureEvaluator:
    def __init__(self,outcome='partial',fail=False): self.outcome=outcome; self.fail=fail
    def evaluate(self,spec,submission,output):
        if self.fail: raise RuntimeError('injected grader failure')
        output.mkdir()
        score={'success':5,'partial':2,'failure':0}[self.outcome]
        raw={'id':spec.task_id[-3:],'status':'ok','error':None,'file_name':'fixture.docx',
             'dim1_pass':self.outcome!='failure','dim1_reason':'','dim2_items':[], 'total_score':score,'max_score':5}
        write_json(output,'fixture_grade.json',raw)
        return {'official_raw':raw,'projection':interpret(raw,spec.task_id,'full_score_v1'),'fixture':True}

@pytest.fixture
def components(config):
    return {'model':ScriptedGemini(config['model']),'benchmark':BundleOfficeVal(config['benchmark']),
            'runtime':TrustedFixtureRuntime(),'evaluation':FixtureEvaluator()}

@pytest.fixture
def native_source(tmp_path,config):
    src=tmp_path/'source'; src.mkdir()
    (src/'verifiers').mkdir()
    shutil.copyfile(BUNDLE/'verifiers/officeval_003_verifier.py',src/'verifiers/officeval_003_verifier.py')
    (src/'LICENSE').write_text('Fixture source repository; retained verifier attribution in bundle source_metadata/LICENSE')
    for args in [('init',),('config','user.email','fixture@example.invalid'),('config','user.name','Fixture'),('add','.'),('commit','-m','fixture')]:
        subprocess.run(['git','-C',str(src),*args],check=True,capture_output=True)
    rev=subprocess.check_output(['git','-C',str(src),'rev-parse','HEAD'],text=True).strip()
    data=tmp_path/'data'; assets=tmp_path/'assets'
    for kind in ('tasks','rubrics'):
        (data/kind).mkdir(parents=True)
        shutil.copyfile(BUNDLE/kind/'officeval_003.json',data/kind/'officeval_003.json')
    shutil.copytree(BUNDLE/'inputs/officeval_003',assets/'officeval_003')
    c={**config['benchmark'],'adapter':'officeval-native','source_dir':str(src),'data_dir':str(data),'assets_dir':str(assets),
       'data_lock':str(data/'data.lock.json'),'source_commit':rev}
    lock_local(c,['officeval_003'])
    return c
