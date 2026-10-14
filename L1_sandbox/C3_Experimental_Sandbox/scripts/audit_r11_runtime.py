"""Post-run real negative checks and immutable grader verification; no LLM calls."""
import copy
import hashlib
import json
from pathlib import Path

from c3_sandbox.evidence._storage import encode
from c3_sandbox.replay import ReplayQualification
from c3_sandbox.r1.admission import R1AdmissionAuthority
from c3_sandbox.r1.docker_runtime import DockerRuntime
from c3_sandbox.r1.runtime_negatives import run_runtime_negatives

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/e2-r1-1'


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('xb') as stream:
        stream.write(encode(value))


def main():
    plan = read(RUN / 'candidate_order.json')
    manifest = read(plan['manifest_path'])
    folder = RUN / 'verification'
    folder.mkdir(exist_ok=True)
    evidence, checks = [], []
    negatives = run_runtime_negatives(plan['agent_image'], RUN / 'runtime-negatives')
    for item in negatives['cases']:
        checks.append({'name': 'actual_runtime_' + item['name'], 'passed': item['passed']})
    evidence.append(RUN / 'runtime-negatives/runtime_negative_report.json')
    expected = {'/opt/officeval/' + task['verifier_path']: task['verifier_sha256']
                for task in manifest['tasks'].values()}
    expected['/opt/officeval/manifest.json'] = sha(plan['manifest_path'])
    expected['/opt/grader/officeval.py'] = plan['implementation_lock']['src/c3_sandbox/r1/officeval.py']
    expected['/opt/grader/r1_grader.py'] = plan['implementation_lock']['docker/r1_grader.py']
    code = "import hashlib,json,stat\nfrom pathlib import Path\npaths=" + repr(list(expected)) + "\nprint(json.dumps({p:{'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest(),'mode':stat.S_IMODE(Path(p).stat().st_mode)} for p in paths}))"
    image_dir = RUN / 'grader-image-audit'
    with DockerRuntime(plan['grader_image'], image_dir / 'runtime', is_grader=True) as actor:
        process = actor.exec_fixed(['python', '-I', '-B', '-c', code])
        audit = actor.audit_boundary()
    assert process['exit_code'] == 0 and not process['error'] and not process['truncated']
    observed = json.loads(process['stdout'])
    matches = all(observed[path]['sha256'] == digest and observed[path]['mode'] & 0o222 == 0
                  for path, digest in expected.items())
    write(image_dir / 'report.json', {'image': plan['grader_image'], 'expected': expected,
        'observed': observed, 'hashes_match_and_readonly': matches, 'boundary': audit, 'raw_process': process})
    checks.append({'name': 'actual_grader_image_hashes_and_readonly', 'passed': matches and audit['qualified']})
    evidence.append(image_dir / 'report.json')
    selection = read(RUN / 'coverage_selection.json')['T_candidate']
    assert selection is not None
    baseline = read(RUN / 'originals' / selection['task_id'] / 'result.json')
    admission_root = RUN / 'replay_tool_continuation/admission'
    authority = R1AdmissionAuthority(admission_root / 'private')
    replay = ReplayQualification(Path(baseline['replay_store']), real_admission=authority)
    cp = selection['checkpoint_id']
    target = RUN / 'replay-negative-tests/workspace'
    target.parent.mkdir()
    restored = replay.restore(cp, target)
    assert restored['valid']
    verify = lambda messages, runtime: replay.verify_restoration(cp, target, messages, baseline['system_identity'], runtime)
    good = verify(restored['messages'], restored['runtime'])
    assert good['valid']
    (target / 'unexpected.txt').write_text('explicit negative fixture')
    changed_workspace = verify(restored['messages'], restored['runtime'])
    (target / 'unexpected.txt').unlink()
    messages = copy.deepcopy(restored['messages'])
    messages[-1]['tool_call_id'] = 'tampered-native-result-id'
    changed_messages = verify(messages, restored['runtime'])
    runtime = copy.deepcopy(restored['runtime'])
    runtime['environment']['C3_TAMPER_FIXTURE'] = '1'
    changed_runtime = verify(restored['messages'], runtime)
    report = {'checkpoint_id': cp, 'valid_control': good, 'workspace_tamper': changed_workspace,
              'message_tamper': changed_messages, 'runtime_tamper': changed_runtime,
              'original_checkpoint_unchanged': replay.qualify(cp)['eligible_for_intervention'],
              'model_calls_for_negative_tests': 0}
    write(target.parent / 'report.json', report)
    for name in ('workspace_tamper', 'message_tamper', 'runtime_tamper'):
        checks.append({'name': 'real_checkpoint_' + name + '_refused', 'passed': report[name]['valid'] is False})
    evidence.append(target.parent / 'report.json')
    # No successful S case exists. Reuse only the earlier actual D5 evidence for
    # unchanged modules, with explicit provenance; do not claim fresh E2-S runs.
    old = ROOT / 'runs/e2-r1/scheduler/source_snapshot'
    modules = ('src/c3_sandbox/experiment_scheduler.py', 'src/c3_sandbox/experiment_controller.py',
               'src/c3_sandbox/probe_registry.py')
    reused = {relative: {'current': sha(ROOT / relative), 'prior_actual_run': sha(old / relative)} for relative in modules}
    old_checks_path = ROOT / 'runs/e2-r1/scheduler/integration_checks.json'
    old_checks = read(old_checks_path)
    unchanged = all(item['current'] == item['prior_actual_run'] for item in reused.values())
    write(folder / 'd5_evidence_scope.json', {'modules': reused,
        'unchanged': unchanged, 'historical_actual_checks': old_checks,
        'current_fresh_E2_S_panel_executed': False,
        'current_protocol_tests': 'test_e2_scheduler.py and test_r11_panel.py (synthetic controlled tests)',
        'scope': 'unchanged D5 isolation/accounting implementation, supported by prior actual run plus current adversarial tests; no new success-side result'})
    checks.append({'name': 'D5_unchanged_prior_actual_isolation_evidence', 'passed': unchanged and old_checks['passed']})
    evidence += [folder / 'd5_evidence_scope.json', old_checks_path,
                 ROOT / 'runs/e2-r1/scheduler/controller_audit.private.json']
    assert all(item['passed'] for item in checks)
    write(folder / 'g4_runtime_audit.json', {'status': 'PASS', 'agent_image': plan['agent_image'],
        'grader_image': plan['grader_image'], 'checks': checks,
        'evidence_files': [{'path': str(path), 'sha256': sha(path)} for path in evidence],
        'current_E2_S_coverage': False, 'trust_boundary': 'trusted controller; no malicious-host or kernel-escape proof'})
    print(json.dumps({'G4': 'PASS', 'checks': len(checks), 'grader_files': len(expected),
                      'new_model_calls': 0, 'D5_actual_evidence_reused_for_unchanged_code': True}))


if __name__ == '__main__':
    main()
