"""Post-hoc metadata admission is versioned and bound to actual source files."""
import json

import pytest

from l2_sandbox.measurement_gates import _v2_authorization, measurement_gate_binding
from l2_sandbox.v3_execution import file_digest
from test_prospective_measurement_gates import prospective  # shared original-v1 evidence fixture


def _authorization(tmp_path):
    source = tmp_path / 'actual_request.txt'
    source.write_text('Adopt v2 post-hoc metadata contract; freeze D0; D1 plan only; zero models.')
    return {
        'authorization_source': 'direct_user_request_attachment',
        'request_attachment': {'path': source.name, 'sha256': file_digest(source)},
        'adoption_scope': {
            'policy': 'bounded-mechanism-contract-v2', 'post_hoc_forensic_extension': True,
            'execution': 'D0_freeze_and_D1_plan_only', 'actor_requests': 0, 'analyst_requests': 0,
        },
    }


def test_attachment_authorization_binds_actual_user_source_bytes(tmp_path):
    authorization = _authorization(tmp_path)
    result = _v2_authorization(authorization, tmp_path / 'authorization.json')
    assert result == {'path': str(tmp_path / 'actual_request.txt'),
                      'sha256': authorization['request_attachment']['sha256']}
    (tmp_path / 'actual_request.txt').write_text('changed')
    with pytest.raises(ValueError, match='attachment_changed'):
        _v2_authorization(authorization, tmp_path / 'authorization.json')


@pytest.mark.parametrize('mutation', ['source', 'missing_attachment', 'hash', 'missing_scope',
                                     'policy', 'post_hoc', 'execute', 'actor', 'analyst', 'bool_zero'])
def test_content_only_or_unbound_authorization_never_suffices(tmp_path, mutation):
    authorization = _authorization(tmp_path)
    if mutation == 'source':
        authorization['authorization_source'] = 'direct_user_message'
    elif mutation == 'missing_attachment':
        authorization.pop('request_attachment')
    elif mutation == 'hash':
        authorization['request_attachment']['sha256'] = '0' * 64
    elif mutation == 'missing_scope':
        authorization.pop('adoption_scope')
    else:
        key, value = {
            'policy': ('policy', 'bounded-mechanism-contract-v1'),
            'post_hoc': ('post_hoc_forensic_extension', False),
            'execute': ('execution', 'execute'),
            'actor': ('actor_requests', 1),
            'analyst': ('analyst_requests', 1),
            'bool_zero': ('actor_requests', False),
        }[mutation]
        authorization['adoption_scope'][key] = value
    with pytest.raises((ValueError, KeyError)):
        _v2_authorization(authorization, tmp_path / 'authorization.json')


def test_policy_string_cannot_upgrade_v1_record(prospective):
    _, docs, audit, update, save = prospective
    docs['g2']['mechanism_gate_pass'] = False
    update('g2')
    before = (save()).read_bytes()
    original, errors = measurement_gate_binding(save())
    assert errors and original['gates']['V3-G2'] is False
    audit['policy'] = 'bounded-mechanism-contract-v2'
    changed, errors = measurement_gate_binding(save())
    assert errors and changed['passed'] is False
    assert docs['g2']['mechanism_gate_pass'] is False
    assert before != save().read_bytes()  # only the audit pointer has changed


def test_v2_schema_without_source_bound_reaudit_does_not_fall_back(prospective):
    _, _, audit, _, save = prospective
    audit.update(schema='prospective_measurement_readiness_v2',
                 policy='bounded-mechanism-contract-v2', post_hoc_forensic_extension=True)
    binding, errors = measurement_gate_binding(save())
    assert errors and not binding['passed']
    assert binding['schema'] == 'prospective_measurement_readiness_v2'
    assert binding['statistical_equivalence'] == 'not_established'
    assert binding['outcome_distribution_equivalence'] == 'not_established'
    assert binding['raw_tool_byte_equivalence'] is False
    assert binding['execution_readiness'] is False
    assert binding['execution_authorized'] is False
    assert binding['current_execution_scope'] == 'D0_freeze_and_D1_plan_only'


@pytest.mark.parametrize('field', ['statistical_equivalence', 'outcome_distribution_equivalence'])
def test_v2_statistical_overclaims_fail_closed(prospective, field):
    _, _, audit, _, save = prospective
    audit.update(schema='prospective_measurement_readiness_v2',
                 policy='bounded-mechanism-contract-v2', post_hoc_forensic_extension=True)
    audit[field] = 'established'
    binding, errors = measurement_gate_binding(save())
    assert not binding['passed']
    assert any('overclaim' in error for error in errors)


def _structural_evidence(prospective):
    from copy import deepcopy
    from l2_sandbox.v3_execution import digest

    _, docs, audit, *_ = prospective
    result = docs['g2']
    plan = docs['g2_plan']
    selected = plan['batches'][0]
    original_replay = result['batches'][0]['replays'][0]
    result['mechanism_gate_pass'] = False
    result['status'] = 'blocked_unexplained_difference'
    result['tool_attempts'] = result['runtime_executions'] = 102
    plan['batches'], result['batches'] = [], []
    for ordinal in range(67):
        name = f'b{ordinal}'
        row = deepcopy(selected)
        row['batch_id'] = name
        plan['batches'].append(row)
        replay = deepcopy(original_replay)
        replay['calls'][0]['exact'] = ordinal >= 35
        result['batches'].append({'batch_id': name, 'replays': [replay] if ordinal >= 35
                                 else [replay, deepcopy(replay)]})
    # The gate binds plan bytes, independent of the old fixture's report.
    refs = deepcopy(audit['refs'])
    refs['g2_plan']['sha256'] = digest(plan)
    result['plan_sha256'] = refs['g2_plan']['sha256']
    return result, plan, refs, docs['cohort']['slots']


def test_v2_structure_preserves_original_blocked_verdict(prospective):
    from l2_sandbox.measurement_gates import _g2_v2_structure

    result, plan, refs, slots = _structural_evidence(prospective)
    _g2_v2_structure(result, plan, refs, slots)
    assert result['mechanism_gate_pass'] is False
    assert result['status'] == 'blocked_unexplained_difference'


@pytest.mark.parametrize('mutation', ['pretend_complete', 'missing_batch', 'duplicate_batch', 'missing_call', 'changed_counter',
                                     'cleanup', 'restore', 'request', 'error', 'v1_upgrade'])
def test_v2_original_coverage_and_error_checks_remain_mandatory(prospective, mutation):
    from l2_sandbox.measurement_gates import _g2_v2_structure

    result, plan, refs, slots = _structural_evidence(prospective)
    replay = result['batches'][0]['replays'][0]
    if mutation == 'pretend_complete':
        result['status'] = 'complete'
    elif mutation == 'missing_batch':
        result['batches'].pop()
    elif mutation == 'duplicate_batch':
        result['batches'].append(result['batches'][0])
    elif mutation == 'missing_call':
        replay['calls'].pop()
    elif mutation == 'changed_counter':
        result['runtime_executions'] += 1
    elif mutation == 'cleanup':
        replay['cleanup']['confirmed'] = False
    elif mutation == 'restore':
        replay['restore_exact'] = False
    elif mutation == 'request':
        replay['request_exact'] = {'raw_sha256': 'changed'}
    elif mutation == 'error':
        result['error'] = 'failed'
    else:
        result['mechanism_gate_pass'] = True
    with pytest.raises((ValueError, IndexError)):
        _g2_v2_structure(result, plan, refs, slots)


def _contract():
    return {
        'schema': 'bounded-mechanism-contract-policy-v2', 'policy': 'bounded-mechanism-contract-v2',
        'derived_from_policy': 'bounded-mechanism-contract-v1', 'post_hoc_forensic_extension': True,
        'preserve_v1_original_decision': True, 'v1_decision': 'blocked_unexplained_difference',
        'reuse_existing_evidence_only': True, 'requires_independent_raw_reaudit': True,
        'inherit_v1_docx_zip_timestamp_exception_only': True, 'raw_tool_output_byte_equivalence': False,
        'statistical_equivalence': 'not_established', 'outcome_distribution_equivalence': 'not_established',
        'current_execution_scope': 'D0_freeze_and_D1_plan_only', 'additional_tool_replays': 0,
        'actor_requests': 0, 'analyst_requests': 0,
        'exemptions': {
            'ls_runtime_metadata_v2': {
                'allowed': ['uid', 'gid', 'mtime', 'directory_total_allocated_blocks'],
                'exact': ['entry_count', 'mode', 'links', 'logical_size', 'name'],
                'requires_after_state_exact': True, 'requires_other_observation_fields_exact': True,
                'forms': ['directory_ls_la', 'single_file_ls_la', 'cd_then_ls_la', 'cp_then_ls_la'],
            },
            'python_object_repr_address_v1': {
                'allowed': ['hexadecimal_address_in_default_python_object_repr'],
                'requires_after_state_exact': True,
                'requires_full_stdout_stderr_and_other_observation_fields_exact_except_named_addresses': True,
            },
        },
    }


def test_v2_approved_contract_has_only_named_exclusions():
    from l2_sandbox.measurement_gates import _v2_contract
    _v2_contract(_contract())


@pytest.mark.parametrize('mutation', ['ls_content', 'ls_mode', 'python_error', 'state', 'observation',
                                     'v1', 'statistical', 'outcome', 'actor', 'tool'])
def test_even_rehashed_contract_cannot_expand_approved_boundaries(mutation):
    from l2_sandbox.measurement_gates import _v2_contract
    contract = _contract()
    if mutation == 'ls_content':
        contract['exemptions']['ls_runtime_metadata_v2']['allowed'].append('content')
    elif mutation == 'ls_mode':
        contract['exemptions']['ls_runtime_metadata_v2']['exact'].remove('mode')
    elif mutation == 'python_error':
        contract['exemptions']['python_object_repr_address_v1']['allowed'].append('error_message')
    elif mutation == 'state':
        contract['exemptions']['ls_runtime_metadata_v2']['requires_after_state_exact'] = False
    elif mutation == 'observation':
        contract['exemptions']['ls_runtime_metadata_v2']['requires_other_observation_fields_exact'] = False
    elif mutation == 'v1':
        contract['preserve_v1_original_decision'] = False
    elif mutation == 'statistical':
        contract['statistical_equivalence'] = 'established'
    elif mutation == 'outcome':
        contract['outcome_distribution_equivalence'] = 'established'
    elif mutation == 'actor':
        contract['actor_requests'] = 1
    else:
        contract['additional_tool_replays'] = 1
    with pytest.raises(ValueError):
        _v2_contract(contract)
