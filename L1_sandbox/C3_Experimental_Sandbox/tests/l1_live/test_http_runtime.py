from __future__ import annotations
import copy
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import pytest
from c3_sandbox.l1.core.orchestrator import run
from c3_sandbox.l1.core.contracts import Call
from c3_sandbox.l1.models.native import GeminiBackend, OpenAICompatibleBackend
from c3_sandbox.l1.runtime.docker import bounded_process,DockerRuntime
from c3_sandbox.l1.storage import canonical,read_json
from .conftest import gemini_call,gemini_done
from .test_native_protocols import chat

@pytest.fixture
def server():
    state={'requests':[],'protocol':'gemini','status':200,'payload':None}
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body=self.rfile.read(int(self.headers['Content-Length']))
            state['requests'].append({'path':self.path,'body':body,'headers':dict(self.headers)})
            count=len(state['requests'])
            payload=state['payload']
            if payload is None:payload=(gemini_call() if count==1 else gemini_done()) if state['protocol']=='gemini' else chat(count==1)
            if not isinstance(payload,bytes):payload=canonical(payload)
            self.send_response(state['status']);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(payload)));self.send_header('x-request-id','local-test-http');self.end_headers()
            self.wfile.write(payload)
        def log_message(self,*args):pass
    httpd=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=httpd.serve_forever,daemon=True);thread.start()
    yield state,f'http://127.0.0.1:{httpd.server_port}'
    httpd.shutdown();httpd.server_close();thread.join()

@pytest.mark.parametrize('protocol',['gemini','chat'])
def test_real_loopback_http_episode(server,tmp_path,config,components,protocol,monkeypatch):
    state,url=server;state['protocol']=protocol
    config['model'].update(kind='local',base_url=url,api_key_env='L1_TEST_KEY',adapter='gemini' if protocol=='gemini' else 'openai-compatible')
    monkeypatch.setenv('L1_TEST_KEY','not-a-real-key-TEST-SECRET')
    components['model']=(GeminiBackend if protocol=='gemini' else OpenAICompatibleBackend)(config['model'])
    r=run(config,tmp_path/'http-episode',injected=components)
    assert r['verification']['verified'],r
    assert r['status']['termination']=='assistant_finished' and len(state['requests'])==2
    assert (state['requests'][0]['path'].endswith(':generateContent') if protocol=='gemini' else state['requests'][0]['path']=='/chat/completions')
    for p in (tmp_path/'http-episode').rglob('*'):
        if p.is_file():assert b'not-a-real-key-TEST-SECRET' not in p.read_bytes()
    actual=sorted((tmp_path/'http-episode/raw/model').glob('*/request.json'))
    assert [p.read_bytes() for p in actual]==[r['body'] for r in state['requests']]


def test_http_capped_body_and_no_retry(server,config):
    state,url=server;state.update(payload=b'x'*1000,status=429)
    config['model'].update(kind='local',base_url=url,max_response_bytes=100)
    r=GeminiBackend(config['model']).send(b'{}',3)
    assert r.status==429 and len(r.body)==100 and r.metadata['body_truncated']
    assert len(state['requests'])==1


def test_bounded_process_actual_output():
    r=bounded_process([sys.executable,'-c','import sys;print("real");sys.stderr.write("error")'],3,1000)
    assert r['stdout']==b'real\n' and r['stderr']==b'error' and r['exit_code']==0


def test_bounded_process_timeout():
    r=bounded_process([sys.executable,'-c','import time;time.sleep(10)'],.1,1000)
    assert r['stop_reason']=='timeout' and r['duration_seconds']<3


def test_bounded_process_output_limit():
    r=bounded_process([sys.executable,'-c','print("x"*100000)'],3,200)
    assert not r['output_complete'] and len(r['stdout'])+len(r['stderr'])==200


def test_docker_has_no_host_fallback(config,monkeypatch):
    monkeypatch.setattr('shutil.which',lambda name:None)
    from c3_sandbox.l1.core.contracts import CapabilityError
    with pytest.raises(CapabilityError):DockerRuntime(config['runtime']).preflight()


def test_docker_security_arguments(config,monkeypatch,tmp_path):
    import c3_sandbox.l1.runtime.docker as module
    runtime=DockerRuntime(config['runtime']);runtime.provenance={'image_id':'sha256:'+'1'*64}
    commands=[];cid='a'*64
    def process(args,**kw):
        commands.append(args)
        return subprocess.CompletedProcess(args,0,(cid+'\n').encode() if args[1]=='create' else b'',b'')
    monkeypatch.setattr(module.subprocess,'run',process)
    monkeypatch.setattr(module,'bounded_process',lambda *a,**k:{'stdout':b'out','stderr':b'', 'exit_code':0,'stop_reason':None,'output_complete':True})
    monkeypatch.setattr(module,'docker_json',lambda *a:[{'State':{'Running':False,'ExitCode':0,'OOMKilled':False}}])
    r=runtime.container([(tmp_path,'/workspace',False)],['/bin/sh','-lc','printf test'],3)
    args=commands[0]
    for flag in ('--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--user'):
        assert flag in args
    assert '--privileged' not in args and 'docker.sock' not in ' '.join(args)
    assert len([x for x in args if x=='--mount'])==1 and commands[-1][:3]==['docker','rm','-f']
    assert r['quiescent'] and runtime.is_quiescent

@pytest.mark.docker
@pytest.mark.skipif(os.environ.get('L1_DOCKER_TESTS')!='1',reason='Explicit opt-in required; not a passed live Docker qualification')
def test_real_docker_cold_runtime(config,tmp_path):
    runtime=DockerRuntime(config['runtime']);p=runtime.preflight()
    w=tmp_path/'workspace';w.mkdir();os.chmod(w,0o777)
    r=runtime.execute(Call('t1','run',{'command':'printf hello > file.txt; printf actual'},None),w,10)
    assert r.stdout==b'actual' and (w/'file.txt').read_text()=='hello' and r.provenance['quiescent']
    r=runtime.execute(Call('t2','run',{'command':'cat file.txt'},None),w,10)
    assert r.stdout==b'hello' and runtime.qualification()['image_id']==p['image_id']

@pytest.mark.live_api
@pytest.mark.skipif(not os.environ.get('L1_LIVE_CONFIG'),reason='Explicit live API config and credentials required')
def test_real_model_docker_evidence(tmp_path):
    from c3_sandbox.l1.core.config import resolve
    c=resolve(Path(os.environ['L1_LIVE_CONFIG']))
    result=run(c,tmp_path/'live')
    assert result['verification']['verified'],result
    assert result['status']['execution_validity']=='valid' and result['status']['evaluation_validity']=='valid',result
    assert read_json(tmp_path/'live/manifest.json')['counts']['tool_results']>0
