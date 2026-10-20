"""Snapshot terminal metadata, summarize on AML CPU, and verify the returned archive."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
L2 = ROOT / 'L2_Behavioral Event Localization/L2_Sandbox'
sys.path.insert(0, str(L2 / 'src'))
from l2_sandbox.v3_execution import exclusive, file_digest, read_json, seal, verify_seal, write_json

SSH = ['ssh', '-oConnectTimeout=10', '-oConnectionAttempts=1', '-oBatchMode=yes', 'aml-b']
RSYNC_SSH = 'ssh -oConnectTimeout=10 -oConnectionAttempts=1 -oBatchMode=yes'
PYTHON = '/home/jiasheren2/envs/agent-diagnosis-d0-py312/bin/python'


def collect_inputs(phase):
    """Read metadata only; never evaluate outcomes or open credentials/packages."""
    plan = read_json(phase / 'execution_plan.json')
    verify_seal(plan)
    ended = read_json(phase / 'supervisor_result.json')
    result = read_json(phase / 'execution/result.json')
    verify_seal(result)
    if (type(ended.get('controller_exit_code')) is not int
            or result.get('process_finished') is not True
            or result.get('plan_sha256') != plan['artifact_sha256']):
        raise ValueError('closed controller and terminal result required')
    files = {}

    def add(path, expected=None):
        path = Path(path)
        if not path.is_absolute() or path.is_symlink() or not path.is_relative_to(ROOT):
            raise ValueError('regular workspace metadata/source required')
        sha = file_digest(path)
        if expected is not None and sha != expected:
            raise ValueError('bound terminal input differs')
        if path in files and files[path] != sha:
            raise ValueError('terminal input changed')
        files[path] = sha

    for name in ('execution_plan.json', 'integrity_preflight.json', 'supervisor_started.json',
                 'supervisor_result.json', 'execution/result.json', 'execution/admission.json'):
        add(phase / name)
    receipts = set((phase / 'execution/receipts').glob('*.json'))
    for path in receipts:
        add(path)
    node, seen = plan, set()
    while node.get('predecessor') is not None:
        if node['artifact_sha256'] in seen or len(seen) >= 8:
            raise ValueError('predecessor cycle or depth limit')
        seen.add(node['artifact_sha256'])
        prior = node['predecessor']
        for key in ('plan', 'result', 'admission', 'supervisor', 'supervisor_result'):
            ref = prior[key]
            if node['bound_files'].get(ref['path']) != ref['sha256']:
                raise ValueError('unbound predecessor metadata')
            add(Path(ref['path']), ref['sha256'])
        node = read_json(Path(prior['plan']['path']))
        verify_seal(node)
    # Analysis sources are independently snapshotted, so later analysis upgrades
    # do not rewrite or invalidate the frozen execution source inventory.
    for directory in (ROOT / 'L1_sandbox/C3_Experimental_Sandbox/src', L2 / 'src'):
        for path in directory.rglob('*'):
            if path.is_file() and path.suffix in ('.py', '.json', '.js', '.mjs') and '__pycache__' not in path.parts:
                add(path)
    for path in (L2 / 'scripts/summarize_opencode_finish.py', Path(__file__).resolve()):
        add(path)
    if receipts != set((phase / 'execution/receipts').glob('*.json')):
        raise ValueError('terminal receipt inventory changed')
    return plan, files


def verify_report(destination, plan_sha256):
    manifest = read_json(destination / 'manifest.json')
    verify_seal(manifest)
    if manifest['plan_sha256'] != plan_sha256:
        raise ValueError('returned report belongs to a different plan')
    for name, sha in manifest['generated_files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or file_digest(destination / relative) != sha:
            raise ValueError('returned report hash differs')
    for ref in manifest['input_snapshots']:
        relative = Path(ref['snapshot_path'])
        if relative.is_absolute() or '..' in relative.parts or file_digest(destination / relative) != ref['sha256']:
            raise ValueError('returned support snapshot differs')
    summary = read_json(destination / 'summary.json')
    verify_seal(summary)
    if summary['artifact_sha256'] != manifest['summary_sha256']:
        raise ValueError('returned summary binding differs')
    return manifest, summary


def archive(phase):
    phase = Path(phase).resolve()
    post = phase / 'postprocess'
    post.mkdir(exist_ok=True)
    with exclusive(post / 'stage.lock'):
        destination = phase / 'terminal_report'
        if destination.exists():
            plan = read_json(phase / 'execution_plan.json')
            verify_seal(plan)
            manifest, summary = verify_report(destination, plan['artifact_sha256'])
            result = seal(dict(success=True, reused_completed_report=True,
                plan_sha256=plan['artifact_sha256'], report_manifest_sha256=manifest['artifact_sha256'],
                terminal_census=summary['census'], actor_requests=0, scorer_invocations=0,
                report_path=str(destination / 'report.md'), automatic_d2=False))
            write_json(post / 'result.json', result)
            return result
        return _archive_attempt(phase, post)


def _archive_attempt(phase, post):
    attempt_id = 'attempt_' + str(time.time_ns())
    attempt = post / attempt_id
    attempt.mkdir()
    status = dict(started_at_epoch=time.time(), actor_requests=0, scorer_invocations=0,
                  analysis_location='aml-b CPU', automatic_d2=False, success=False)
    try:
        plan, files = collect_inputs(phase)
        snapshot = attempt / 'input_snapshot/workspace'
        inventory = []
        for path, sha in sorted(files.items(), key=lambda item: str(item[0])):
            target = snapshot / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            if file_digest(path) != sha or file_digest(target) != sha:
                raise ValueError('terminal snapshot changed during copy')
            inventory.append(dict(source_path=str(path), relative_path=str(path.relative_to(ROOT)), sha256=sha))
        export = seal(dict(plan_sha256=plan['artifact_sha256'], input_files=inventory,
                           scope='terminal metadata and separate analysis sources; no package/credential export'))
        write_json(attempt / 'export_manifest.json', export)
        remote = Path('/home/jiasheren2/projects/agent_diagnosis_terminal_reports') / (
            phase.name + '_' + plan['artifact_sha256'][:16]) / attempt_id
        remote_workspace = remote / 'workspace'
        subprocess.run(SSH + [shlex.join(['mkdir', '-p', str(remote_workspace)])], check=True, timeout=30)
        subprocess.run(['rsync', '-a', '--checksum', '-e', RSYNC_SSH, str(snapshot) + '/',
                        'aml-b:' + str(remote_workspace) + '/'], check=True, timeout=600)
        remote_l2 = remote_workspace / L2.relative_to(ROOT)
        remote_phase = remote_workspace / phase.relative_to(ROOT)
        pythonpath = str(remote_l2 / 'src') + ':' + str(remote_workspace / 'L1_sandbox/C3_Experimental_Sandbox/src')
        command = ['timeout', '900', 'env', 'CUDA_VISIBLE_DEVICES=', 'OMP_NUM_THREADS=4',
                   'OPENBLAS_NUM_THREADS=4', 'MKL_NUM_THREADS=4', 'PYTHONDONTWRITEBYTECODE=1',
                   'PYTHONPATH=' + pythonpath, 'nice', '-n', '10', PYTHON,
                   str(remote_l2 / 'scripts/summarize_opencode_finish.py'),
                   '--phase', str(remote_phase), '--output', str(remote / 'report'),
                   '--path-map', str(ROOT) + '=' + str(remote_workspace)]
        ran = subprocess.run(SSH + [shlex.join(command)], text=True, capture_output=True, timeout=960)
        (attempt / 'server_report.log').write_text(ran.stdout + ran.stderr)
        if ran.returncode:
            raise RuntimeError('server terminal report failed; see server_report.log')
        destination = attempt / 'download'
        destination.mkdir()
        subprocess.run(['rsync', '-a', '-e', RSYNC_SSH, 'aml-b:' + str(remote / 'report') + '/',
                        str(destination) + '/'], check=True, timeout=600)
        manifest, summary = verify_report(destination, plan['artifact_sha256'])
        destination.rename(phase / 'terminal_report')
        status.update(success=True, plan_sha256=plan['artifact_sha256'],
                      report_manifest_sha256=manifest['artifact_sha256'],
                      terminal_census=summary['census'], remote_directory=str(remote),
                      report_path=str(phase / 'terminal_report/report.md'))
    except Exception as exc:
        status.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        status['finished_at_epoch'] = time.time()
        write_json(attempt / 'result.json', seal(status))
        write_json(post / 'result.json', seal(status))
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', type=Path, required=True)
    result = archive(parser.parse_args().phase)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
