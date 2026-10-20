"""One finite continuation; verify storage identity and floor before/while running."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path('/Users/jiashen/Desktop/问题诊断')
PHASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'L2_Behavioral Event Localization/L2_Sandbox/src'))
from l2_sandbox.v3_execution import file_digest, read_json, verify_seal

plan = read_json(PHASE / 'execution_plan.json')
storage = read_json(PHASE / 'storage_contract.json')
verify_seal(plan)
verify_seal(storage)
volume = Path(storage['volume_path'])
package_root = Path(storage['branch_storage_root'])
if plan.get('branch_storage_root') != str(package_root):
    raise SystemExit('storage contract differs from plan')


def storage_problem():
    try:
        if (not os.path.ismount(volume) or volume.is_symlink()
                or volume.stat().st_dev != storage['volume_device']
                or package_root.stat().st_dev != storage['volume_device']
                or ROOT.stat().st_dev != storage['local_device']):
            return 'storage_identity_changed'
        for path in (ROOT, package_root):
            if shutil.disk_usage(path).free < storage['min_free_disk_bytes']:
                return 'free_disk_floor'
    except OSError:
        return 'storage_unavailable'
    return None


problem = storage_problem()
if problem:
    raise SystemExit(problem)
if '--check-storage' in sys.argv:
    print(json.dumps(dict(storage_ok=True, actor_requests=0,
                          local_free_bytes=shutil.disk_usage(ROOT).free,
                          external_free_bytes=shutil.disk_usage(package_root).free)))
    raise SystemExit(0)
started = time.time()
if not started < plan['deadline_epoch'] <= started + 86400:
    raise SystemExit('phase deadline expired or exceeds 24h')
lock = os.open(PHASE / 'supervisor_launch.lock', os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
os.close(lock)
record = dict(started_at_epoch=started, deadline_epoch=plan['deadline_epoch'],
              max_wall_seconds=plan['deadline_epoch']-started,
              min_free_disk_bytes=storage['min_free_disk_bytes'],
              storage_contract_sha256=file_digest(PHASE/'storage_contract.json'),
              source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              supervisor_pid=os.getpid(),restart_or_renew=False)
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4',
           OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1')
command = [str(ROOT/'L1_sandbox/C3_Experimental_Sandbox/.venv/bin/python'),
           str(ROOT/'L2_Behavioral Event Localization/L2_Sandbox/scripts/build_opencode_d1_execution.py'),
           '--phase',str(PHASE),'--execute']
with (PHASE/'controller.log').open('xb') as log:
    child = subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                             stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record['controller_pid'] = child.pid
(PHASE/'supervisor_started.json').write_text(json.dumps(record,indent=2)+'\n')
reason = None
while child.poll() is None:
    reason = 'wall_cap' if time.time() >= record['deadline_epoch'] else storage_problem()
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
record.update(finished_at_epoch=time.time(),controller_exit_code=child.returncode,
              supervisor_stop_reason=reason)
(PHASE/'supervisor_result.json').write_text(json.dumps(record,indent=2)+'\n')

# Terminal analysis is a separate bounded CPU-only job; never opens D2.
post_started=time.time()
post_error=None
post_code=None
try:
    with (PHASE/'postprocess.log').open('xb') as log:
        post=subprocess.run([str(ROOT/'L1_sandbox/C3_Experimental_Sandbox/.venv/bin/python'),str(ROOT/'L2_Behavioral Event Localization/L2_Sandbox/scripts/archive_opencode_terminal.py'),'--phase',str(PHASE)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=2400)
    post_code=post.returncode
except Exception as exc:
    post_error=type(exc).__name__
(PHASE/'postprocess_supervisor_result.json').write_text(json.dumps(dict(started_at_epoch=post_started,finished_at_epoch=time.time(),returncode=post_code,error_type=post_error,actor_requests=0,automatic_d2=False),indent=2)+'\n')
