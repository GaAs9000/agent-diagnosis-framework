"""Controller-only R1.1 public probe preparation and eight-branch engineering panel.

prepare_panel is synchronous; call after the original's asyncio.run has finished.
It generates no model response. run_success_panel performs physical model work
only through the existing real executor and D5 serial acquisition broker.
"""
from __future__ import annotations

import asyncio
import copy
from dataclasses import asdict
import hashlib
import hmac
import json
from pathlib import Path
import re
import secrets

from ..evidence._storage import encode
from ..experiment_controller import RegistryAuthorizer
from ..experiment_scheduler import BranchSpec, ExperimentScheduler, ProtocolError
from ..manifests import canonical_hash
from ..probe_registry import ProbeRegistry, VERSION, _vocabulary
from ..replay import ReplayQualification
from .admission import _audit
from .branch_executor import RealBranchExecutor
from .docker_runtime import DockerRuntime, snapshot_workspace, validate_snapshot
from .office_run import _sha, _write
from .probes import render_pair, render_probe, validate_public_evidence
from .public_evidence import collect_public_evidence
from .qualification import TOOL_SCHEMAS
from .tokenizer_client import TokenizerClient

PROBE_IDS = {'E2-S': ('original', 'visibility.compact', 'visibility.compact.matched', 'feedback.aggregate'),
             'E2-F': ('original', 'visibility.expanded', 'visibility.expanded.matched', 'feedback.structured')}
ROLES = ('original', 'target', 'matched', 'competing')
SAMPLING = {'temperature': 0, 'max_tokens': 2048, 'context_window': 16384,
            'max_retries': 0, 'parallel_tool_calls': False, 'top_k': 1,
            'min_p': 0, 'max_turns': 10}


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _read(path):
    return json.loads(Path(path).read_bytes())


def _hash(value, name):
    _require(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value), name + ' requires SHA256')
    return value


def prepare_panel(baseline, selection, root, line):
    """Collect unchanged selected workspace and freeze existing public renderers.

    root is a new preparation directory, normally run_root/probe-preparation.
    Return rendered keyed by roles, rendered_by_probe keyed by registry IDs,
    token_audit, source_review, evidence, collection_audit, and artifact paths.
    """
    _require(line in PROBE_IDS, 'unsupported panel line')
    _require(baseline.get('is_synthetic') is False and baseline.get('execution_valid') is True,
             'real valid baseline required')
    _require(type(baseline.get('task_completed')) is bool and
             line == ('E2-S' if baseline['task_completed'] else 'E2-F'), 'baseline completion/line mismatch')
    reference = selection['reference']
    _require(selection['checkpoint_id'] == reference['checkpoint_id'] and reference in baseline['checkpoints'],
             'selection is not a baseline checkpoint')
    replay = ReplayQualification(Path(baseline['replay_store']))
    record = replay._record(reference['checkpoint_id'])
    _require(record['is_synthetic'] is False and not record['registration_reasons'] and
             not ReplayQualification._input_reasons(record), 'unsupported source checkpoint')
    messages = _read(reference['messages_path'])
    _require(messages == record['messages'] and record['runtime'] == baseline['runtime'] and
             record['system_identity'] == baseline['system_identity'], 'checkpoint prefix/config drift')
    source = _read(reference['source_audit_path'])
    _audit(source, baseline['runtime']['image_digest'])
    events = [event for event in baseline['agent_result']['events']
              if event['event_index'] == reference['event_index']]
    _require(len(events) == 1 and events[0]['type'] == 'tool_result', 'selected boundary lacks actual tool result')
    tool_event = events[0]
    dispatches = [call for call in baseline.get('tool_dispatch', [])
                  if call['tool_call_id'] == tool_event['tool_call_id']]
    _require(len(dispatches) == 1, 'tool result lacks unique snapshot-bound dispatch')
    dispatch = dispatches[0]
    _require(dispatch['raw_result'] == tool_event['raw_result'] and dispatch['arguments'] == tool_event['arguments']
             and dispatch['function'] == tool_event['function'] and not dispatch['error']
             and not dispatch['snapshot_ineligible_reasons'], 'dispatch/result evidence mismatch or incomplete')
    prior = validate_snapshot(_read(dispatch['before_path']))
    after_tool = validate_snapshot(_read(dispatch['after_path']))
    _require(snapshot_workspace(Path(reference['workspace'])) == after_tool,
             'last dispatch after-state differs from selected checkpoint')
    root = Path(root).absolute()
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    restored = replay.workspace_store.restore(record['snapshot_id'], root / 'workspace')
    _write(root / 'host_restore.json', restored.to_dict())
    _require(restored.valid and snapshot_workspace(root / 'workspace') == after_tool, 'checkpoint restore mismatch')
    with DockerRuntime(baseline['runtime']['image_digest'], root / 'runtime') as actor:
        actor.seed_workspace(root / 'workspace')
        before = actor.verify_workspace(root / 'workspace')
        public = collect_public_evidence(actor, checkpoint=prior, last_tool_result=tool_event['raw_result'])
        after = actor.verify_workspace(root / 'workspace')
        audit = actor.audit_boundary()
        _audit(audit, baseline['runtime']['image_digest'])
        _require(before['verified'] is True and after['verified'] is True and
                 audit['inspect']['State']['Pid'] != source['inspect']['State']['Pid'],
                 'collector changed workspace or reused source runtime')
    public = validate_public_evidence(public)
    _require(public['inventory']['xml_parse_error_count'] == 0 and public['inventory']['docx_count'] >= 1,
             'public document structure unavailable')
    collection = {'checkpoint_id': reference['checkpoint_id'], 'before': before, 'after': after,
                  'boundary': audit, 'prior_tool_event': tool_event, 'prior_snapshot_sha256': _sha(Path(dispatch['before_path'])),
                  'dispatch_identity': canonical_hash(dispatch), 'source_boundary_identity': canonical_hash(source),
                  'no_hidden_grader_inputs': True, 'read_only_collection_verified': True,
                  'information_source': 'fixed read-only collector on restored public workspace; last actual tool before-state'}
    tokenizer = TokenizerClient()
    original, target, matched, competing = PROBE_IDS[line]
    panel = render_pair(target, public, token_counter=tokenizer.text_count)
    for probe in (original, competing):
        panel[probe] = render_probe(probe, public, token_counter=tokenizer.text_count)
    rendered = {probe: value.to_dict() for probe, value in panel.items()}
    counts, injected_prefixes = {}, {}
    for probe, value in rendered.items():
        resumed = copy.deepcopy(messages)
        if value['rendered_text']:
            resumed.append({'role': 'user', 'content': value['rendered_text']})
        injected_prefixes[probe] = resumed
        counts[probe] = asyncio.run(tokenizer.count_messages_details(resumed, TOOL_SCHEMAS))
        _require(counts[probe]['tokenizer_fingerprint'] == baseline['model_qualification']['tokenizer_fingerprint'],
                 'tokenizer identity changed')
        _require(counts[probe]['token_count'] <= 16384 - 2048, 'full resumed prompt exceeds frozen context budget')
    _require(counts[target]['token_count'] == counts[matched]['token_count'], 'actual full prompt costs differ')
    token_audit = {'model_fingerprint': baseline['model_qualification']['model_fingerprint'],
                   'tokenizer_fingerprint': baseline['model_qualification']['tokenizer_fingerprint'],
                   'full_prompt_details': counts, 'full_prompt_tokens': {p: c['token_count'] for p, c in counts.items()},
                   'target_matched_equal': True, 'same_message_position': len(messages),
                   'base_prefix_identity': canonical_hash(messages), 'same_context_window': 16384,
                   'no_extra_model_turns_granted': True, 'control_inertness_assumed': False}
    source_review = {'kind': 'automated closed-schema dataflow and resource-preservation review',
                     'task_id': baseline['task_id'], 'checkpoint_identity': reference['checkpoint_id'], 'line': line,
                     'panel_hash': canonical_hash(rendered), 'public_evidence_identity': canonical_hash(public),
                     'collection_audit_hash': canonical_hash(collection), 'token_audit_hash': canonical_hash(token_audit),
                     'checks': {'closed_public_schema': True, 'unchanged_complete_workspace': True,
                                'paired_checkpoint_messages': True, 'full_prompt_costs_matched': True,
                                'same_native_tools': True, 'no_resource_removed': True},
                     'allowed_payload': 'fixed global public inventory/execution counts; no paths, document text, rubric, patches or target values',
                     'scope': 'closed dataflow and necessary-resource preservation; not control inertness or guaranteed solution'}
    evidence = {'public_evidence': public, 'last_tool_event': tool_event, 'dispatch_identity': canonical_hash(dispatch)}
    documents = {'rendered_panel': rendered, 'token_audit': token_audit, 'collection_audit': collection,
                 'public_evidence': public, 'source_review': source_review, 'injected_prefixes': injected_prefixes}
    for name, value in documents.items():
        _write(root / (name + '.json'), value)
    return {'rendered': {role: rendered[probe] for role, probe in zip(ROLES, PROBE_IDS[line])},
            'rendered_by_probe': rendered, 'token_audit': token_audit, 'source_review': source_review,
            'evidence': evidence, 'collection_audit': collection,
            'paths': {name: str(root / (name + '.json')) for name in documents}}


def _verified_preparation(baseline, selection, prepared):
    panel = prepared['rendered_by_probe']
    tokens, collection, review = prepared['token_audit'], prepared['collection_audit'], prepared['source_review']
    public = validate_public_evidence(prepared['evidence']['public_evidence'])
    _require(collection['checkpoint_id'] == selection['checkpoint_id'] and
             tokens['base_prefix_identity'] == canonical_hash(_read(selection['reference']['messages_path'])),
             'preparation belongs to another checkpoint')
    _require(tokens['target_matched_equal'] is True and collection['before']['verified'] is True and
             collection['after']['verified'] is True and collection['boundary']['qualified'] is True and
             set(review['checks']) == {'closed_public_schema', 'unchanged_complete_workspace',
                                      'paired_checkpoint_messages', 'full_prompt_costs_matched',
                                      'same_native_tools', 'no_resource_removed'} and
             all(value is True for value in review['checks'].values()), 'preparation checks failed')
    _require(review['panel_hash'] == canonical_hash(panel) and
             review['collection_audit_hash'] == canonical_hash(collection) and
             review['token_audit_hash'] == canonical_hash(tokens) and
             review['public_evidence_identity'] == canonical_hash(public), 'preparation review drift')
    for probe, value in panel.items():
        _require(value['probe_id'] == probe and value['public_evidence'] == public and
                 hashlib.sha256(value['rendered_text'].encode()).hexdigest() == value['rendered_digest'],
                 'rendered probe drift')
    for name, path in prepared['paths'].items():
        if name in {'rendered_panel', 'token_audit', 'collection_audit', 'public_evidence', 'source_review'}:
            expected = {'rendered_panel': panel, 'token_audit': tokens, 'collection_audit': collection,
                        'public_evidence': public, 'source_review': review}[name]
            _require(_read(path) == expected, 'persisted preparation changed: ' + name)
    return panel


def run_success_panel(baseline, selection, manifest, plan, admission, prepared, root):
    """Run fresh exploration4 + confirmation4 through D5; no method statistics.

    root is the run root; new scheduler/, e2_s_exploration/, e2_s_confirmation/
    directories are created. plan requires immutable agent/grader image IDs,
    candidate_order_sha256, implementation_identity, and disjoint phase seeds.
    Optional implementation_lock maps repo-relative source paths to frozen SHA;
    optional physical_lock_path supplies the shared serial execution lock.
    """
    _require(selection['kind'] == 'S' and baseline.get('task_completed') is True and
             baseline.get('execution_valid') is True and baseline.get('is_synthetic') is False,
             'success panel requires valid real officially complete S baseline')
    _require(prepared['source_review']['line'] == 'E2-S', 'success panel requires E2-S preparation')
    panel = _verified_preparation(baseline, selection, prepared)
    _require(set(panel) == set(PROBE_IDS['E2-S']), 'success panel probe set mismatch')
    pool_hash = _hash(plan['candidate_order_sha256'], 'candidate order')
    code_identity = _hash(plan['implementation_identity'], 'implementation')
    repo = Path(__file__).resolve().parents[3]
    for relative, expected in plan.get('implementation_lock', {}).items():
        path = Path(relative)
        _require(not path.is_absolute() and '..' not in path.parts and _sha(repo / path) == expected,
                 'frozen implementation changed')
    seeds = [plan[phase + '_seed'] for phase in ('exploration', 'confirmation')]
    _require(all(type(seed) is int and seed >= 0 for seed in seeds) and seeds[0] != seeds[1],
             'disjoint phase seed domains required')
    _require(baseline['runtime']['image_digest'] == plan['agent_image'], 'plan actor image differs from baseline')
    root = Path(root).absolute()
    run = root / 'scheduler'
    run.mkdir(parents=True, exist_ok=False, mode=0o700)
    cp = selection['reference']
    freeze = {'task_id': baseline['task_id'], 'task_manifest': manifest['tasks'][baseline['task_id']],
              'full_configuration': copy.deepcopy(plan), 'candidate_order_sha256': pool_hash,
              'rendered_panel_sha256': canonical_hash(panel), 'checkpoint_identity': cp['checkpoint_id'],
              'implementation_identity': code_identity, 'scope': 'development-only-R1.1-engineering-E2-S'}
    task_identity = canonical_hash(freeze)
    grader_identity = canonical_hash({'image': plan['grader_image'], 'manifest': manifest,
                                      'implementation_identity': code_identity})
    _write(run / 'frozen_panel.json', {**freeze, 'task_identity': task_identity, 'grader_identity': grader_identity})
    baseline = copy.deepcopy(baseline)
    baseline['task_identity'] = task_identity
    source_review = {**prepared['source_review'], 'task_identity': task_identity,
                     'frozen_configuration_identity': canonical_hash(freeze)}
    _write(run / 'probe_review_source.json', source_review)
    registry = ProbeRegistry({'version': VERSION, 'probes': list(_vocabulary().values())})
    case_id = baseline['task_id'] + '-development-R1.1-E2-S'
    schedules = {phase: {'0': {**SAMPLING, 'seed': plan[phase + '_seed']}}
                 for phase in ('exploration', 'confirmation')}
    key = secrets.token_hex(32)
    private = run / 'private'
    private.mkdir(mode=0o700)
    key_path = private / 'review.key'
    with key_path.open('x') as stream:
        stream.write(key)
    key_path.chmod(0o600)
    inputs = [entry['name'] for entry in manifest['tasks'][baseline['task_id']]['input_files']]
    contract = {'task_identity': task_identity, 'line': 'E2-S', 'is_synthetic': False,
                'baseline_probe_id': PROBE_IDS['E2-S'][0], 'target_probe_id': PROBE_IDS['E2-S'][1],
                'control_probe_id': PROBE_IDS['E2-S'][2], 'competing_probe_id': PROBE_IDS['E2-S'][3],
                'available_inputs': inputs, 'necessary_inputs': inputs,
                'available_tools': ['workspace_python'], 'necessary_tools': ['workspace_python'],
                'observation_capabilities': ['redundant_observation_views', 'public_feedback_fields'],
                'sampling': schedules['exploration']['0'], 'sampling_by_phase': schedules,
                'budgets': {'max_model_calls': 10, 'context_window': 16384},
                'trusted_review_keys': {'r11-closed-public-schema-verifier': key}}
    reviews = {}
    for probe_id in PROBE_IDS['E2-S']:
        review = {'task_identity': task_identity, 'probe_identity': registry.identity(probe_id), 'line': 'E2-S',
                  'source': {'kind': 'task_verifier', 'signer_id': 'r11-closed-public-schema-verifier',
                             'record_identity': canonical_hash(source_review)},
                  'findings': {'leakage': {'verdict': 'no_task_answer_material',
                                          'evidence_identities': [canonical_hash(panel[probe_id]), canonical_hash(source_review)]},
                               'solvability': {'verdict': 'necessary_resources_preserved',
                                              'evidence_identities': [canonical_hash(prepared['collection_audit']), cp['checkpoint_id']]},
                               'reviewed_inputs': inputs, 'reviewed_tools': ['workspace_python']}}
        review['signature'] = {'scheme': 'hmac-sha256', 'value': hmac.new(key.encode(), encode(review), hashlib.sha256).hexdigest()}
        _require(registry.validate_for_case(probe_id, contract, review)['eligible'], 'task-bound probe review rejected')
        reviews[(case_id, probe_id)] = review
    _write(run / 'probe_reviews.json', {probe: review for (_, probe), review in reviews.items()})
    specs = [BranchSpec(case_id, task_identity, 'E2-S', baseline['system_identity'], cp['checkpoint_id'],
                        cp['checkpoint_id'], probe, registry.identity(probe), schedules[phase]['0'], 0,
                        cp['runtime_identity'], grader_identity, False, phase)
             for phase in ('exploration', 'confirmation') for probe in PROBE_IDS['E2-S']]
    _write(run / 'branch_plan.json', {'branches': [asdict(spec) for spec in specs],
                                    'case_contract_public': {k: v for k, v in contract.items() if k != 'trusted_review_keys'},
                                    'forecast': 'constant 0.5 engineering driver; D6-D8 closed',
                                    'confirmation_scope': 'sealed fresh development engineering coverage, no method effectiveness claim'})
    executor = RealBranchExecutor(admission['replay'], run / 'branches', baseline, manifest,
                                  plan['agent_image'], plan['grader_image'], panel,
                                  baseline['model_qualification'], grader_identity)
    scheduler = ExperimentScheduler(private / 'scheduler.sqlite', qualifier=admission['replay'].qualify,
                                    probe_authorizer=RegistryAuthorizer(registry, reviews), executor=executor,
                                    real_admission=admission['authority'],
                                    physical_lock_path=plan.get('physical_lock_path', root / 'r11.serial.lock'))
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
    exploration_results = {}
    for index, spec in enumerate(exploration):
        method_a.commit_prediction(case_id, index, probabilities, spec.key)
        denied('prediction_immutable_' + str(index), lambda: method_a.commit_prediction(case_id, index, probabilities, spec.key))
        receipt = method_a.request_probe(case_id, index)
        denied('reveal_requires_charge_' + str(index), lambda: method_a.reveal_result(receipt))
        denied('foreign_receipt_denied_' + str(index), lambda: method_b.reveal_result(receipt))
        method_a.charge_budget(receipt)
        revealed = method_a.reveal_result(receipt)
        exploration_results[spec.key] = revealed['result']
        if index == 0:
            _require(method_b.observations() == [], 'A observations leaked to B')
            checks.append({'check': 'A_query_absent_from_B_history', 'passed': True})
            method_b.commit_prediction(case_id, 0, probabilities, spec.key)
            physical_before = len(scheduler.audit()['physical'])
            approval = reviews.pop((case_id, spec.probe_id))
            denied('revoked_approval_denied_even_on_physical_cache', lambda: method_b.request_probe(case_id, 0))
            _require(len(scheduler.audit()['physical']) == physical_before, 'revoked cache request executed a branch')
            reviews[(case_id, spec.probe_id)] = approval
            cached_receipt = method_b.request_probe(case_id, 0)
            denied('cached_reveal_requires_charge', lambda: method_b.reveal_result(cached_receipt))
            method_b.charge_budget(cached_receipt)
            cached = method_b.reveal_result(cached_receipt)
            ledger = scheduler.audit()
            _require(len(ledger['physical']) == physical_before == 1 and cached['result'] == revealed['result']
                     and sum(method['spent'] for method in ledger['methods']) == 2,
                     'physical deduplication must retain two logical charges')
            checks.append({'check': 'same_branch_one_physical_two_logical_charges', 'passed': True})
    _require(len(method_a.observations()) == 4 and len(method_b.observations()) == 1, 'method query counts differ')
    denied('confirmation_denied_until_methods_close', scheduler.finalize_confirmation)
    method_a.close_exploration()
    denied('confirmation_denied_while_B_open', scheduler.finalize_confirmation)
    method_b.close_exploration()
    confirmation = scheduler.finalize_confirmation()
    _require(len(method_a.observations()) == 4 and len(method_b.observations()) == 1,
             'confirmation entered exploration histories')
    checks.append({'check': 'confirmation_absent_from_method_histories', 'passed': True})
    ledger = scheduler.audit()
    logical = sum(method['spent'] for method in ledger['methods'])
    _require(len(ledger['physical']) == 8 and logical == 5 and len(confirmation) == 4,
             'expected eight physical and five logical queries')
    checks.append({'check': 'eight_physical_five_logical_queries', 'passed': True})
    _write(private / 'controller_ledger.json', ledger)
    _write(private / 'confirmation_results.json', confirmation)
    indices, per_arm = {}, {}
    for phase, results in [('exploration', exploration_results), ('confirmation', confirmation)]:
        phase_root = root / ('e2_s_' + phase)
        phase_root.mkdir(mode=0o700)
        rows = []
        for spec in specs:
            if spec.phase != phase:
                continue
            branch = run / 'branches' / spec.key
            result_path, cost_path = branch / 'result.json', branch / 'cost.json'
            result, cost = _read(result_path), _read(cost_path)
            _require(result == results[spec.key], 'branch artifact differs from broker outcome')
            row = {'branch_key': spec.key, 'probe_id': spec.probe_id,
                   'result_path': str(result_path.relative_to(root)), 'result_sha256': _sha(result_path),
                   'cost_path': str(cost_path.relative_to(root)), 'cost_sha256': _sha(cost_path)}
            rows.append(row)
            per_arm[phase + '/' + spec.probe_id] = {'outcome': result['outcome'], 'cost': cost}
        _write(phase_root / 'index.json', {'phase': phase, 'checkpoint_identity': cp['checkpoint_id'], 'branches': rows})
        indices[phase] = str(phase_root / 'index.json')
    fresh_originals = {phase: results[next(s.key for s in specs if s.phase == phase and s.probe_id == 'original')]['outcome']
                       for phase, results in [('exploration', exploration_results), ('confirmation', confirmation)]}
    valid = all(item['outcome']['execution_valid'] is True for item in per_arm.values())
    costs = [item['cost'] for item in per_arm.values()]
    total_cost = {name: None if any(cost[name] is None for cost in costs) else
                       sum(cost[name] for cost in costs) for name in costs[0]}
    summary = {'validity': 'valid' if valid else 'infrastructure_invalid', 'engineering_checks_passed': True,
               'physical_executions': 8, 'logical_charges': logical, 'method_budgets': {'A': 4, 'B': 1},
               'physical_cost_total': total_cost,
               'query_budget': {method['id']: {name: method[name] for name in ('budget', 'spent', 'reserved', 'closed')}
                                for method in ledger['methods']},
               'checks': checks, 'per_arm': per_arm, 'fresh_original_outcome': fresh_originals,
               'unstable_success_baseline': any(flags['execution_valid'] is True and flags['task_completed'] is False
                                                for flags in fresh_originals.values()),
               'phase_indices': indices, 'task_identity': task_identity, 'grader_identity': grader_identity,
               'checkpoint_identity': cp['checkpoint_id'], 'D6_D8': 'CLOSED',
               'claim_scope': 'real replay and scheduler engineering coverage only'}
    _write(run / 'integration_checks.json', summary)
    return summary
