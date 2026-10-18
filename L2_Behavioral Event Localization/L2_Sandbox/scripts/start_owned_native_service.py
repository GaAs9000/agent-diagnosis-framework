#!/usr/bin/env python3
"""AML remote service launcher. Default is inert; --execute needs a frozen phase payload.

No Actor requests are sent. Source/authorization binding is checked by the phase
controller before invoking this remote entry point; the receipt binds cleanup to
an exact process birth and preserves the original absolute lease.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

SERVICE_ROOT = Path('/home/jiasheren2/projects/agent_diagnosis_d0_20261018/r22_service')
MODEL_ROOT = '/home/jiasheren2/hf_cache/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9'


def leader_identity(pid, proc_root=Path('/proc')):
    fields=(proc_root/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
    if int(fields[2]) != pid:
        raise ValueError('service must own its process group')
    return dict(start_ticks=int(fields[19]))


def launch(v):
    if v.get('phase_authorized') is not True or v.get('phase_frozen') is not True:
        raise ValueError('explicit frozen phase required')
    values=[v['local_start_epoch'],v['service_stop_epoch'],*v['clock_calibration'].values()]
    if any(type(n) not in (int,float) or not math.isfinite(n) for n in values):
        raise ValueError('finite lease and clock calibration required')
    owner=v['plan_sha256']; stop=v['service_stop_epoch']
    assert len(owner)==64 and all(c in '0123456789abcdef' for c in owner)
    assert 645 < stop-v['local_start_epoch'] <= 21000
    uid=os.getuid()
    query=lambda fields:subprocess.check_output(['nvidia-smi',*fields],text=True,timeout=10)
    rows=query(['--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits']).splitlines()
    row=next(x.split(', ') for x in rows if x.startswith('4, '))
    assert row[1]=='GPU-6d3f754b-b139-9354-0cbc-42bf22208180' and int(row[2])<=4 and int(row[3])==0,row
    own=set()
    for line in query(['--query-compute-apps=pid,gpu_uuid','--format=csv,noheader']).splitlines():
     pid,gpu=line.split(', ')
     try:
      if os.stat('/proc/'+pid).st_uid==uid:own.add(gpu)
     except FileNotFoundError:pass
    assert len(own)<=1,own
    with socket.socket() as sock: assert sock.connect_ex(('127.0.0.1',8001))!=0,'port8001 already occupied'
    base=SERVICE_ROOT/owner
    base.mkdir(parents=True,exist_ok=False)
    model=MODEL_ROOT
    assert Path(model,'config.json').is_file()
    cal=v['clock_calibration']
    assert 0 <= cal['local_received_epoch']-cal['local_sent_epoch'] <= 10
    remote_hard_stop=v['local_start_epoch']+21600-(cal['local_received_epoch']-cal['remote_epoch'])-60
    args=['timeout','--signal=TERM','--kill-after=30s','LEASE_PENDING','nice','-n','10','/home/jiasheren2/envs/idea-n1/bin/vllm','serve',model,'--download-dir','/home/jiasheren2/hf_cache','--served-model-name','Qwen/Qwen3.6-27B','Qwen3.6-27B','--port','8001','--host','127.0.0.1','--enable-auto-tool-choice','--tool-call-parser','qwen3_xml','--reasoning-parser','qwen3','--max-model-len','131072','--max-num-seqs','64','--gpu-memory-utilization','0.85','--dtype','bfloat16']
    env=os.environ.copy();env.update(CUDA_HOME='/usr/local/cuda-13.0',PATH='/usr/local/cuda-13.0/bin:/home/jiasheren2/envs/idea-n1/bin:'+os.environ.get('PATH',''),CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='4',OMP_NUM_THREADS='16',OPENBLAS_NUM_THREADS='16',MKL_NUM_THREADS='16',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',C3_R22_SERVICE_OWNER=owner)
    compiler_tools={name:shutil.which(name,path=env['PATH']) for name in ('nvcc','ninja','c++','gcc','g++')}
    assert all(compiler_tools.values()),compiler_tools
    assert compiler_tools['nvcc']=='/usr/local/cuda-13.0/bin/nvcc'
    with (base/'service.log').open('xb') as log:
     remote_launch=time.time()
     duration=int(remote_hard_stop-remote_launch-30-5)
     assert 645 < duration <= 21600
     args[3]=str(duration)+'s'
     process=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    birth=leader_identity(process.pid)
    service=dict(leader_start_ticks=birth['start_ticks'],remote_host='aml-b',device=4,gpu_uuid=row[1],leader_pid=process.pid,pgid=process.pid,owner_tag=owner,server_plan_hash=owner,started_epoch=v['local_start_epoch'],hard_stop_epoch=v['local_start_epoch']+21600,service_stop_epoch=stop,remote_started_epoch=time.time(),command=args,remote_directory=str(base),other_owned_gpu_uuids=sorted(own),compiler_tools=compiler_tools,cuda_home=env['CUDA_HOME'],compiler_path='/usr/local/cuda-13.0/bin/nvcc',clock_calibration=cal,remote_launch_epoch=remote_launch,remote_hard_stop_epoch=remote_hard_stop,timeout_seconds=duration,kill_grace_seconds=30,independent_hard_stop='GNU timeout remaining original lease/KILL+30s')
    (base/'receipt.json').write_text(json.dumps(service,indent=2)+'\n')
    return service


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status='not_executed',requires='--execute and frozen phase JSON on stdin')))
        return
    print(json.dumps(launch(json.load(sys.stdin))))


if __name__=='__main__':
    main()
