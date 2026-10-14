"""Docker cold-command runtime. No unrestricted host model-command fallback.

A new read-only-root container is created for each call, killed/removed before
return, and only the episode workspace is writable and persistent. Controller,
credentials, evidence and graders are NEVER mounted into the agent container.
"""
from __future__ import annotations
import os
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path
from ..core.contracts import CapabilityError, L1Error, Result
from ..storage import utc_now, confined, regular_file
from .tools import validate_call


def docker_json(*args):
    try:
        p=subprocess.run(['docker',*args],capture_output=True,timeout=30,check=True)
        return json.loads(p.stdout)
    except (OSError,subprocess.SubprocessError,ValueError) as exc:
        raise CapabilityError('Docker unavailable or requested image missing; run doctor and build the image') from exc


def bounded_process(args:list[str],timeout:float,limit:int,on_abort=None)->dict:
    """Bounded raw stdout/stderr; drain pipes concurrently and stop on overflow."""
    start=time.monotonic(); started=utc_now()
    process=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,stdin=subprocess.DEVNULL)
    streams=[bytearray(),bytearray()]; lock=threading.Lock(); overflow=threading.Event()
    def reader(pipe,index):
        try:
            while True:
                chunk=pipe.read1(65536)
                if not chunk: break
                with lock:
                    used=len(streams[0])+len(streams[1]); left=max(0,limit-used)
                    streams[index].extend(chunk[:left])
                    if len(chunk)>left: overflow.set()
        finally: pipe.close()
    threads=[threading.Thread(target=reader,args=(p,i),daemon=True) for i,p in enumerate((process.stdout,process.stderr))]
    for thread in threads: thread.start()
    reason=None
    try:
        while process.poll() is None:
            if overflow.is_set(): reason='output_limit'; break
            if time.monotonic()-start>=timeout: reason='timeout'; break
            time.sleep(.01)
        if reason:
            if on_abort: on_abort()
            process.kill()
        process.wait(timeout=10)
    except BaseException:
        if on_abort: on_abort()
        process.kill(); process.wait(timeout=10)
        raise
    finally:
        for thread in threads: thread.join(timeout=10)
    if overflow.is_set() and reason is None: reason='output_limit'
    return {'stdout':bytes(streams[0]),'stderr':bytes(streams[1]),'exit_code':process.returncode,
            'stop_reason':reason,'output_complete':not overflow.is_set(),
            'started_at':started,'finished_at':utc_now(),'duration_seconds':time.monotonic()-start}


class DockerRuntime:
    def __init__(self,config):
        self.config=config; self.provenance=None; self.is_quiescent=True; self.event_sink=None
    def set_event_sink(self,sink):
        self.event_sink=sink
    def preflight(self):
        if not shutil.which('docker'): raise CapabilityError('Docker command not installed')
        image=docker_json('image','inspect',self.config['image'])[0]
        info=docker_json('info','--format','{{json .}}')
        if image.get('Os')!='linux' or info.get('OSType')!='linux': raise CapabilityError('docker-cold requires Linux containers')
        self.provenance={'adapter':'docker-cold','execution_model':'one_container_per_tool',
            'image_requested':self.config['image'],'image_id':image['Id'],'repo_digests':image.get('RepoDigests',[]),
            'architecture':image.get('Architecture'),'os':image.get('Os'),'docker_server_version':info.get('ServerVersion'),
            'network':'none','root_filesystem':'read_only','capabilities':'drop_all',
            'security':'no-new-privileges','persistent_scope':['workspace'],
            'transient_scope':['tmp','processes','shell_environment'], 'credential_mount':False,
            'resource_limits':{k:self.config[k] for k in ('memory','cpus','pids_limit')},
            'workspace_quota_enforced':False,'observed_at':utc_now()}
        # Probe the actual resolved image, not the host interpreter or a Dockerfile.
        probe_script="import sys,json,importlib.metadata as m,hashlib,pathlib; p=pathlib.Path('/opt/l1/grade_worker.py'); print(json.dumps({'python':sys.version,'packages':{x.metadata['Name']:x.version for x in m.distributions()},'grader_worker_sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None},sort_keys=True))"
        probe=self.container([],['python','-I','-c',probe_script],30,cwd='/tmp')
        if probe['container_exit_code']!=0 or not probe['output_complete']:
            raise CapabilityError('Resolved image failed Python environment probe')
        self.provenance['environment_probe']={k:v for k,v in probe.items() if k not in ('stdout','stderr')}
        self.provenance['python_environment']=json.loads(probe['stdout'])
        return self.provenance
    def qualification(self):
        return {'eligible':bool(self.provenance),'scopes':['workspace_files','native_conversation'],
                'excluded':['process_memory','tmp','open_descriptors','provider_hidden_state','same_future_response'],
                'reason_codes':[] if self.provenance else ['runtime_not_qualified'],
                'execution_model':'one_container_per_tool','image_id':self.provenance.get('image_id') if self.provenance else None}
    def _flags(self):
        uid = os.getuid() if hasattr(os, 'getuid') and os.getuid() != 0 else 65532
        gid = os.getgid() if hasattr(os, 'getgid') and os.getuid() != 0 else 65532
        flags = ['--network', 'none']
        if self.config.get('read_only_rootfs', True):
            flags.append('--read-only')
        flags += [
            '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
            '--pids-limit', str(self.config['pids_limit']),
            '--memory', self.config['memory'],
            '--cpus', str(self.config['cpus'])
        ]
        user = self.config.get('user')
        if user is not None:
            if user:
                flags += ['--user', str(user)]
        else:
            flags += ['--user', f'{uid}:{gid}']
        tmpfs_opts = (
            'rw,exec,nosuid,nodev,size=256m,mode=1777'
            if not self.config.get('read_only_rootfs', True) or self.config.get('tmpfs_exec', False)
            else 'rw,nosuid,nodev,size=256m,mode=1777'
        )
        flags += [
            '--tmpfs', f'/tmp:{tmpfs_opts}', '--env', 'HOME=/tmp',
            '--env', 'PYTHONDONTWRITEBYTECODE=1', '--env', 'PYTHONUNBUFFERED=1',
            '--env', 'OMEGAUSE_OFFICEVAL_DISABLE_COM=1'
        ]
        return flags
    def container(self,mounts:list[tuple[Path,str,bool]],command:list[str],timeout:float,*,cwd:str='/workspace')->dict:
        if not self.provenance: self.preflight()
        flags=self._flags()
        for source,target,readonly in mounts:
            source=source.resolve()
            if ',' in str(source): raise CapabilityError('Docker bind paths containing commas are unsupported')
            mount=f'type=bind,src={source},dst={target}'+(',readonly' if readonly else '')
            flags+=['--mount',mount]
        args=['docker','create','--label','l1.managed=true',*flags,'--workdir',cwd,'--entrypoint',command[0],self.provenance['image_id'],*command[1:]]
        created=subprocess.run(args,capture_output=True,timeout=30)
        if created.returncode!=0: raise L1Error('Docker create failed: '+created.stderr.decode('utf-8','replace')[-3000:])
        cid=created.stdout.decode('ascii').strip()
        import re
        if not re.fullmatch('[a-f0-9]{64}',cid): raise L1Error('Invalid Docker container ID')
        def kill():
            subprocess.run(['docker','kill',cid],capture_output=True,timeout=15)
        self.is_quiescent=False
        cleanup=False; inspection=None
        try:
            if self.event_sink:
                self.event_sink('runtime.container.created',{'container_id':cid,'image_id':self.provenance['image_id'],
                    'mounts':[{'target':target,'read_only':ro,'source_name':source.name} for source,target,ro in mounts]})
            process=bounded_process(['docker','start','-a',cid],timeout,self.config['output_limit_bytes'],kill)
            inspection=docker_json('inspect',cid)[0]
            state=inspection.get('State',{})
            process['container_exit_code']=state.get('ExitCode')
            process['oom_killed']=state.get('OOMKilled')
            process['container_id']=cid
            process['container_state']={k:state.get(k) for k in ('Status','Running','ExitCode','Error','StartedAt','FinishedAt','OOMKilled')}
            if state.get('Running'): kill()
        finally:
            removed=subprocess.run(['docker','rm','-f',cid],capture_output=True,timeout=20)
            cleanup=removed.returncode==0
            self.is_quiescent=cleanup
            if self.event_sink:
                self.event_sink('runtime.container.removed',{'container_id':cid,'removed':cleanup})
        if not cleanup: raise L1Error(f'Container cleanup failed for {cid}; no quiescent state/checkpoint may be claimed')
        process['quiescent']=True
        return process
    def execute(self,call,workspace,timeout):
        validate_call(call)
        if call.name=='read_image':
            start=utc_now(); name=call.arguments['path']
            if name.startswith('/workspace/'): name=name[len('/workspace/'):]
            p=confined(workspace,name); regular_file(p)
            if p.stat().st_size>self.config['max_image_bytes']: raise L1Error('image_size_limit')
            data=p.read_bytes()
            mime=('image/png' if data.startswith(b'\x89PNG\r\n\x1a\n') else 'image/jpeg' if data.startswith(b'\xff\xd8\xff') else
                  'image/webp' if data.startswith(b'RIFF') and data[8:12]==b'WEBP' else None)
            if not mime: raise L1Error('read_image supports PNG/JPEG/WebP only')
            return Result({'path':name,'mime_type':mime,'size_bytes':len(data),'exit_code':0},image=data,image_mime=mime,
                          provenance={'boundary':'trusted_workspace_image_reader','started_at':start,'finished_at':utc_now(),'quiescent':True})
        p=self.container([(workspace,'/workspace',False)],['/bin/sh','-lc',call.arguments['command']],timeout)
        obs={'stdout':p['stdout'].decode('utf-8','replace'),'stderr':p['stderr'].decode('utf-8','replace'),
             'exit_code':p['container_exit_code'],'stop_reason':p['stop_reason'],'output_complete':p['output_complete'],
             'oom_killed':p['oom_killed']}
        provenance={k:v for k,v in p.items() if k not in ('stdout','stderr')}
        return Result(obs,p['stdout'],p['stderr'],provenance=provenance)
