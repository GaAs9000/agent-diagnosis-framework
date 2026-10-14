"""One development failure panel through the real D5 acquisition contract.

The two method clients are deterministic engineering drivers, not D6/D7 methods.
No loop retries a model outcome or chooses another seed/task/model.
"""
import hashlib
import argparse
import hmac
import json
from dataclasses import asdict
from pathlib import Path
import secrets

from c3_sandbox.evidence._storage import encode
from c3_sandbox.experiment_controller import RegistryAuthorizer
from c3_sandbox.experiment_scheduler import BranchSpec, ExperimentScheduler, ProtocolError
from c3_sandbox.manifests import canonical_hash
from c3_sandbox.probe_registry import ProbeRegistry
from c3_sandbox.replay import ReplayQualification
from c3_sandbox.r1.admission import R1AdmissionAuthority
from c3_sandbox.r1.branch_executor import RealBranchExecutor

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open('xb') as stream:
        stream.write(encode(value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, default=ROOT / 'runs/e2-r1')
    args = parser.parse_args()
    root = args.run_root.absolute()
    run = root / 'scheduler'
    run.mkdir(mode=0o700)
    plan = read(root / 'frozen_plan.json')
    manifest = read(ROOT / 'runs/r1-build-context/frozen/manifest.json')
    baseline = read(root / 'originals/officeval_002/result.json')
    audit = read(root / 'originals_audit.json')
    case_audit = next(case for case in audit['cases'] if case['task_id'] == baseline['task_id'])
    if not case_audit['admitted_as_failure'] or not baseline['checkpoints']:
        raise RuntimeError('no eligible real failure checkpoint')
    panel = read(root / 'probe-audit-v2/rendered_panel.json')
    tokens = read(root / 'probe-audit-v2/token_audit.json')
    collection = read(root / 'probe-audit-v2/collection_audit.json')
    public = read(root / 'probe-audit-v2/public_evidence.json')
    if not tokens['target_matched_equal'] or not collection['after']['verified'] or not collection['boundary']['qualified']:
        raise RuntimeError('real probe preparation failed')
    if public['inventory']['xml_parse_error_count'] or public['inventory']['docx_count'] != 1:
        raise RuntimeError('selected OfficeVal structure observation is unavailable')
    if any(value['public_evidence'] != public for value in panel.values()):
        raise RuntimeError('rendered panel differs from audited public evidence')
    paths = sorted((ROOT / 'src/c3_sandbox').rglob('*.py'))
    paths += sorted((ROOT / 'scripts').glob('*.py'))
    paths += [ROOT / name for name in ('requirements.local-arm64.lock.txt',
        'requirements.model.local-arm64.lock.txt', 'configs/probes.v1.json')]
    code_lock = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    write(run / 'implementation_lock.json', code_lock)
    task_identity = canonical_hash({'task': manifest['tasks'][baseline['task_id']],
        'frozen_plan': plan, 'implementation': code_lock, 'rendered_panel': panel,
        'scope': 'development-only-real-R1'})
    baseline['task_identity'] = task_identity
    grader_identity = canonical_hash({'image': plan['grader_image'],
        'frozen_manifest': manifest, 'interpretation': code_lock['src/c3_sandbox/r1/officeval.py']})
    cp = baseline['checkpoints'][0]
    prefix = read(Path(cp['messages_path']))
    if collection['checkpoint_id'] != cp['checkpoint_id'] or tokens['same_message_position'] != len(prefix):
        raise RuntimeError('probe evidence belongs to a different checkpoint or insertion position')
    serving = read(root / 'server_observation.json')
    if serving.get('configuration_verified') is not True:
        raise RuntimeError('local serving configuration has not been observed')
    write(run / 'server_observation.json', serving)
    authority = R1AdmissionAuthority(root / 'private/admission')
    replay = ReplayQualification(Path(baseline['replay_store']), real_admission=authority)
    registry = ProbeRegistry.from_file(ROOT / 'configs/probes.v1.json')
    sampling = {'temperature': 0, 'max_tokens': 2048, 'context_window': 16384,
                'max_retries': 0, 'parallel_tool_calls': False, 'top_k': 1,
                'min_p': 0, 'max_turns': 10}
    schedules = {phase: {'0': dict(sampling, seed=plan[phase + '_seed'])}
                 for phase in ('exploration', 'confirmation')}
    case_id = 'officeval_002-development-E2-F'
    review_key = secrets.token_hex(32)
    inputs = [entry['name'] for entry in manifest['tasks'][baseline['task_id']]['input_files']]
    contract = {'task_identity': task_identity, 'line': 'E2-F', 'is_synthetic': False,
        'baseline_probe_id': 'original', 'target_probe_id': 'visibility.expanded',
        'control_probe_id': 'visibility.expanded.matched', 'competing_probe_id': 'feedback.structured',
        'available_inputs': inputs, 'necessary_inputs': inputs,
        'available_tools': ['workspace_python'], 'necessary_tools': ['workspace_python'],
        'observation_capabilities': ['redundant_observation_views', 'public_feedback_fields'],
        'sampling': schedules['exploration']['0'], 'sampling_by_phase': schedules,
        'budgets': {'max_model_calls': 10, 'context_window': 16384},
        'trusted_review_keys': {'r1-closed-public-schema-verifier': review_key}}
    source_review = {'kind': 'automated task-bound dataflow/resource review',
        'task_id': baseline['task_id'], 'checkpoint_identity': cp['checkpoint_id'],
        'task_identity': task_identity, 'panel_hash': canonical_hash(panel),
        'collection_audit_hash': canonical_hash(collection), 'token_audit_hash': canonical_hash(tokens),
        'allowed_payload': 'global public inventory and execution counts only; no object paths, text, target widths, rubric or edits',
        'necessary_resources': 'same full snapshot and native tools for every arm; no resource removed',
        'scope': 'resource-preservation and closed-schema leakage review; not proof of inert control or successful task solution'}
    write(run / 'probe_review_source.json', source_review)
    reviews = {}
    for probe_id in panel:
        review = {'task_identity': task_identity, 'probe_identity': registry.identity(probe_id), 'line': 'E2-F',
            'source': {'kind': 'task_verifier', 'signer_id': 'r1-closed-public-schema-verifier',
                       'record_identity': canonical_hash(source_review)},
            'findings': {'leakage': {'verdict': 'no_task_answer_material',
                                    'evidence_identities': [canonical_hash(panel[probe_id]), canonical_hash(collection)]},
                         'solvability': {'verdict': 'necessary_resources_preserved',
                                       'evidence_identities': [canonical_hash(collection), cp['checkpoint_id']]},
                         'reviewed_inputs': inputs, 'reviewed_tools': ['workspace_python']}}
        signature = hmac.new(review_key.encode(), encode(review), hashlib.sha256).hexdigest()
        review['signature'] = {'scheme': 'hmac-sha256', 'value': signature}
        reviews[(case_id, probe_id)] = review
        if not registry.validate_for_case(probe_id, contract, review)['eligible']:
            raise RuntimeError('probe review was not admitted')
    write(run / 'probe_reviews.json', {probe: review for (case, probe), review in reviews.items()})
    specs = [BranchSpec(case_id, task_identity, 'E2-F', baseline['system_identity'], cp['checkpoint_id'],
        cp['checkpoint_id'], probe, registry.identity(probe), schedules[phase]['0'], 0,
        cp['runtime_identity'], grader_identity, False, phase)
        for phase in ('exploration', 'confirmation')
        for probe in ('original', 'visibility.expanded', 'visibility.expanded.matched', 'feedback.structured')]
    write(run / 'branch_plan.json', {'branches': [asdict(spec) for spec in specs],
        'case_contract_public': {k: v for k, v in contract.items() if k != 'trusted_review_keys'},
        'forecast': 'constant 0.5 engineering placeholders, not scientific method predictions',
        'confirmation_scope': 'fresh execution and sealed acquisition on development case, not research holdout',
        'sampling_interpretation': 'master seed 2026; deterministic phase-derived confirmation seed; temperature 0, no independent-draw claim'})
    executor = RealBranchExecutor(replay, run / 'branches', baseline, manifest,
        plan['agent_image'], plan['grader_image'], panel, baseline['model_qualification'], grader_identity)
    scheduler = ExperimentScheduler(run / 'scheduler.sqlite', qualifier=replay.qualify,
        probe_authorizer=RegistryAuthorizer(registry, reviews), executor=executor,
        real_admission=authority, physical_lock_path=root / 'physical-execution.lock')
    scheduler.enroll_plan(case_id, case_contract=contract, branches=specs)
    method_a = scheduler.register_method('engineering-A', 4)
    method_b = scheduler.register_method('engineering-B-cache-audit', 1)
    checks = []
    def denied(label, action):
        try:
            action()
        except ProtocolError as error:
            checks.append({'check': label, 'passed': True, 'reason': str(error)})
        else:
            raise AssertionError('unauthorized operation accepted: ' + label)
    denied('confirmation_locked_before_exploration', scheduler.finalize_confirmation)
    exploration = [spec for spec in specs if spec.phase == 'exploration']
    probabilities = {spec.key: 0.5 for spec in exploration}
    for index, spec in enumerate(exploration):
        print('START', spec.phase, spec.probe_id, flush=True)
        method_a.commit_prediction(case_id, index, probabilities, spec.key)
        denied('prediction_immutable_round_' + str(index),
               lambda: method_a.commit_prediction(case_id, index, probabilities, spec.key))
        receipt = method_a.request_probe(case_id, index)
        denied('reveal_requires_payment_' + str(index), lambda: method_a.reveal_result(receipt))
        denied('receipt_not_transferable_' + str(index), lambda: method_b.reveal_result(receipt))
        method_a.charge_budget(receipt)
        result = method_a.reveal_result(receipt)
        print('END', spec.phase, spec.probe_id, json.dumps(result['result']['outcome']), flush=True)
        if index == 0:
            assert method_b.observations() == []
            checks.append({'check': 'A_query_absent_from_B_history', 'passed': True})
            before_physical = len(scheduler.audit()['physical'])
            method_b.commit_prediction(case_id, 0, probabilities, spec.key)
            shared_receipt = method_b.request_probe(case_id, 0)
            method_b.charge_budget(shared_receipt)
            shared = method_b.reveal_result(shared_receipt)
            assert len(scheduler.audit()['physical']) == before_physical
            assert shared['result'] == result['result']
            checks.append({'check': 'same_branch_one_physical_execution_two_logical_charges', 'passed': True})
    assert len(method_a.observations()) == 4 and len(method_b.observations()) == 1
    denied('confirmation_locked_until_every_method_closes', scheduler.finalize_confirmation)
    method_a.close_exploration()
    denied('confirmation_locked_while_B_open', scheduler.finalize_confirmation)
    method_b.close_exploration()
    print('START sealed fresh confirmation panel', flush=True)
    confirmation = scheduler.finalize_confirmation()
    assert len(method_a.observations()) == 4 and len(method_b.observations()) == 1
    checks.append({'check': 'confirmation_not_added_to_method_histories', 'passed': True})
    ledger = scheduler.audit()
    assert len(ledger['physical']) == 8
    assert sum(method['spent'] for method in ledger['methods']) == 5
    checks.append({'check': 'eight_physical_five_logical_queries_ledger_consistent', 'passed': True})
    write(run / 'confirmation_results.json', confirmation)
    write(run / 'controller_audit.private.json', ledger)
    write(run / 'integration_checks.json', {'checks': checks, 'passed': all(item['passed'] for item in checks),
        'physical_executions': len(ledger['physical']), 'logical_charges': 5,
        'E2_F_real_panel_completed': True, 'E2_S_real_panel_completed': False,
        'overall_R1_acceptance': 'INCOMPLETE_SUCCESS_SIDE_COVERAGE'})
    print('END integration', len(checks), 'checks; E2-S remains unmet', flush=True)


if __name__ == '__main__':
    main()
