"""Finite detached launcher; no retry, renewal, model call or credential handling."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

ROOT=Path('/Users/jiashen/Desktop/问题诊断')
PHASE=Path(__file__).resolve().parent
started=time.time()
record={'started_at_epoch':started,'deadline_epoch':started+86400,
        'max_wall_seconds':86400,'min_free_disk_bytes':8*1024**3,
        'source_commit':'56d2de1fe37e8058b7af59b526587639ea843bc3',
        'supervisor_pid':os.getpid(),'restart_or_renew':False}
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
command=[str(ROOT/'L1_sandbox/C3_Experimental_Sandbox/.venv/bin/python'),
         str(ROOT/'L2_Behavioral Event Localization/L2_Sandbox/scripts/build_opencode_d1_execution.py'),
         '--phase',str(PHASE),'--execute']
if shutil.disk_usage(PHASE).free<record['min_free_disk_bytes']:
    raise SystemExit('insufficient free disk before launch')
with (PHASE/'controller.log').open('xb') as log:
    child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record['controller_pid']=child.pid
(PHASE/'supervisor_started.json').write_text(json.dumps(record,indent=2)+'\n')
reason=None
while child.poll() is None:
    if time.time()>=record['deadline_epoch']:
        reason='wall_cap'
    elif shutil.disk_usage(PHASE).free<record['min_free_disk_bytes']:
        reason='free_disk_floor'
    if reason:
        os.killpg(child.pid,signal.SIGINT)
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)
                child.wait()
        break
    time.sleep(5)
record.update(finished_at_epoch=time.time(),controller_exit_code=child.returncode,supervisor_stop_reason=reason)
(PHASE/'supervisor_result.json').write_text(json.dumps(record,indent=2)+'\n')
