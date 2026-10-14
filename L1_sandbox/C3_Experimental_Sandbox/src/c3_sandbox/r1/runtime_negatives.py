"""Explicit controller-run negative acceptance, not a model task or escape proof."""
from __future__ import annotations

import json
from pathlib import Path

from .docker_runtime import DockerRuntime


def run_runtime_negatives(agent_image: str, run_dir: Path) -> dict:
    """Run bounded probes in separate serial containers; persist failures then raise."""
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    private = run_dir / 'controller_private'
    private.mkdir(mode=0o700, exist_ok=True)
    private.chmod(0o700)
    canary = private / 'unmounted_canary.txt'
    canary.write_text('controller-only negative acceptance canary\n')
    canary.chmod(0o600)
    report = {'schema_version': 1, 'agent_image': agent_image, 'cases': [],
              'claim_scope': 'Observed refusal of these operations and unsupported idle states; not proof against kernel attacks or all escapes.'}

    def run_case(name, probe):
        case = {'name': name, 'passed': False}
        try:
            with DockerRuntime(agent_image, run_dir / name) as runtime:
                case.update(probe(runtime))
        except Exception as error:
            case.update(passed=False, error_type=type(error).__name__, error=str(error))
        report['cases'].append(case)

    def execution(runtime, code):
        result = runtime.exec_python(code, timeout=10)
        if result['exit_code'] != 0 or result.get('error') or result.get('truncated'):
            raise RuntimeError('negative probe did not complete cleanly: ' + repr(result))
        return result, json.loads(result['stdout'])

    def inaccessible_and_readonly(runtime):
        targets = ['/opt/officeval/manifest.json', '/opt/grader/r1_grader.py',
                   '/var/run/docker.sock', '/run/docker.sock', str(canary)]
        code = '''# r1-negative: inaccessible_and_readonly
import json,os
results=[]
for target in TARGETS:
 try:
  with open(target,'rb') as stream: stream.read(1)
  results.append({'path':target,'read_succeeded':True})
 except OSError as error:
  results.append({'path':target,'read_succeeded':False,'errno':error.errno})
try:
 fd=os.open('/opt/r1_negative_write_probe',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 os.close(fd)
 write={'write_succeeded':True}
except OSError as error:
 write={'write_succeeded':False,'errno':error.errno}
print(json.dumps({'reads':results,'rootfs_write':write}))
'''.replace('TARGETS', repr(targets))
        result, observed = execution(runtime, code)
        audit = runtime.audit_boundary()
        refused = all(item['read_succeeded'] is False and item.get('errno') in (2, 13, 20) for item in observed['reads'])
        readonly = observed['rootfs_write']['write_succeeded'] is False and observed['rootfs_write'].get('errno') in (13, 30)
        return {'passed': bool(refused and readonly and audit['qualified'] and audit['effective_config_ok']),
                'exec': result, 'observed': observed, 'audit': audit}

    def symlink_export(runtime):
        result, _ = execution(runtime, "# r1-negative: symlink_export\nimport os,json\nos.symlink('/etc/passwd','/workspace/link')\nprint(json.dumps({'created':True}))")
        rejected, error_text = False, None
        try:
            runtime.export_workspace(run_dir / 'symlink_export' / 'rejected_export')
        except (ValueError, RuntimeError) as error:
            error_text = str(error)
            rejected = 'symlink' in error_text.lower()
        return {'passed': rejected, 'exec': result, 'export_rejected': rejected, 'export_error': error_text}

    def background_process(runtime):
        result, observed = execution(runtime, "# r1-negative: background_process\nimport subprocess,json\np=subprocess.Popen(['sleep','60'],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)\nprint(json.dumps({'spawned_pid':p.pid}))")
        audit = runtime.audit_boundary()
        present = any(item['pid'] == observed['spawned_pid'] for item in audit['unexpected_processes'])
        return {'passed': bool(not audit['qualified'] and present and audit['effective_config_ok']), 'exec': result, 'observed': observed, 'audit': audit}

    def tmp_cache(runtime):
        result, _ = execution(runtime, "# r1-negative: tmp_cache\nimport json\nfrom pathlib import Path\nPath('/tmp/r1_transient_cache').write_text('probe')\nprint(json.dumps({'created':True}))")
        audit = runtime.audit_boundary()
        return {'passed': bool(not audit['qualified'] and '/tmp/r1_transient_cache' in audit['tmp_entries'] and audit['effective_config_ok']), 'exec': result, 'audit': audit}

    for name, probe in [('inaccessible_and_readonly', inaccessible_and_readonly),
                        ('symlink_export', symlink_export), ('background_process', background_process), ('tmp_cache', tmp_cache)]:
        run_case(name, probe)
    report['passed'] = all(case['passed'] for case in report['cases'])
    (run_dir / 'runtime_negative_report.json').write_text(json.dumps(report, indent=2, sort_keys=True))
    if not report['passed']:
        failed = ', '.join(case['name'] for case in report['cases'] if not case['passed'])
        raise RuntimeError('runtime negative acceptance failed: ' + failed)
    return report
