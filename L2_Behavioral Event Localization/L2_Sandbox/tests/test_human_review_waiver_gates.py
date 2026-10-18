import json

import pytest

from l2_sandbox.measurement_gates import measurement_gate_binding, verify_measurement_gate_binding
from l2_sandbox.v3_execution import file_digest, write_json


def waiver_audit(tmp_path):
    authorization = tmp_path / 'authorization.json'
    write_json(authorization, {
        'human_review_policy': 'waived_by_user',
        'authorization_source': 'direct_user_message',
        'exact_text': '跳过人审核',
        'scope': ['G3_human_review'],
    })
    return {
        'human_review_policy': 'waived_by_user',
        'human_review_authorization': {'path': authorization.name, 'sha256': file_digest(authorization)},
        'V3_G1': {'byte_equivalence_verified': True, 'applicable_configuration_verified': True},
        'V3_G2_R19': {'cp000000_all_pre_first_request_valid': True,
                      'completed_valid_pairs': 20, 'ci95_completely_inside_pm_005': True},
        'V3_G3_R20': {'sampled_checkpoint_count': 20, 'restored_checkpoint_count': 20,
                      'missing_restored_directory_count': 0, 'missing_scorer_exit_count': 0,
                      'execution_metadata_verified': True, 'source_provenance_verified': True,
                      'human_review_completed_count': 0},
        'V3_G4_R21': {'normal_and_p_requests_scanned': 80, 'unverified_request_count': 0,
                      'leaked_request_count': 0, 'zero_diagnostic_leakage_verified_on_scanned': True},
    }


def bind(tmp_path, audit):
    path = tmp_path / 'audit.json'
    write_json(path, audit)
    return measurement_gate_binding(path)


def test_explicit_authorization_waives_only_human_requirement(tmp_path):
    audit = waiver_audit(tmp_path)
    binding, errors = bind(tmp_path, audit)
    assert not errors and binding['passed']
    assert binding['human_review_status'] == 'waived_by_user'
    assert 'reviewer' not in binding and 'human_pass' not in binding
    verify_measurement_gate_binding({'measurement_gate_binding': binding})
    audit.pop('human_review_policy')
    strict, errors = bind(tmp_path, audit)
    assert not strict['passed'] and 'measurement_gate_not_passed:V3-G3' in errors


@pytest.mark.parametrize('field,value', [
    ('sampled_checkpoint_count', 19), ('restored_checkpoint_count', 19),
    ('missing_restored_directory_count', 1), ('missing_scorer_exit_count', 1),
    ('execution_metadata_verified', False), ('source_provenance_verified', False),
])
def test_waiver_does_not_replace_technical_g3_evidence(tmp_path, field, value):
    audit = waiver_audit(tmp_path)
    audit['V3_G3_R20'][field] = value
    binding, errors = bind(tmp_path, audit)
    assert not binding['passed'] and 'measurement_gate_not_passed:V3-G3' in errors


@pytest.mark.parametrize('gate,field,value', [
    ('V3_G1', 'applicable_configuration_verified', False),
    ('V3_G2_R19', 'ci95_completely_inside_pm_005', False),
    ('V3_G4_R21', 'leaked_request_count', 1),
])
def test_waiver_does_not_bypass_other_gates(tmp_path, gate, field, value):
    audit = waiver_audit(tmp_path)
    audit[gate][field] = value
    binding, errors = bind(tmp_path, audit)
    assert not binding['passed'] and errors


@pytest.mark.parametrize('change', ['missing', 'tampered', 'wrong_scope', 'not_direct_user', 'wrong_text'])
def test_waiver_requires_bound_direct_user_authorization(tmp_path, change):
    audit = waiver_audit(tmp_path)
    authorization_path = tmp_path / 'authorization.json'
    if change == 'missing':
        audit.pop('human_review_authorization')
    elif change == 'tampered':
        authorization_path.write_text('{}')
    else:
        authorization = {
            'human_review_policy': 'waived_by_user', 'authorization_source': 'direct_user_message',
            'exact_text': '跳过人审核', 'scope': ['G3_human_review'],
        }
        if change == 'wrong_scope':
            authorization['scope'] = ['P0a_human_review']
        elif change == 'not_direct_user':
            authorization['authorization_source'] = 'model_suggestion'
        else:
            authorization['exact_text'] = 'go ahead'
        write_json(authorization_path, authorization)
        audit['human_review_authorization']['sha256'] = file_digest(authorization_path)
    binding, errors = bind(tmp_path, audit)
    assert not binding['passed'] and errors


def test_frozen_waiver_evidence_is_rechecked(tmp_path):
    audit = waiver_audit(tmp_path)
    binding, errors = bind(tmp_path, audit)
    assert not errors
    (tmp_path / 'authorization.json').write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        verify_measurement_gate_binding({'measurement_gate_binding': binding})


def test_unknown_review_policy_cannot_open_gate(tmp_path):
    audit = waiver_audit(tmp_path)
    audit['human_review_policy'] = 'skip'
    binding, errors = bind(tmp_path, audit)
    assert not binding['passed'] and 'human_review_policy_invalid' in errors


def test_requirement_review_uses_same_authorization_with_explicit_scope(tmp_path):
    from l2_sandbox.measurement_gates import human_review_policy_binding
    audit = waiver_audit(tmp_path)
    policy, errors = human_review_policy_binding(
        audit, tmp_path / 'audit.json', required_scope='requirement_catalog_human_review')
    assert errors
    path = tmp_path / 'authorization.json'
    authorization = json.loads(path.read_text())
    authorization['scope'].append('requirement_catalog_human_review')
    write_json(path, authorization)
    audit['human_review_authorization']['sha256'] = file_digest(path)
    policy, errors = human_review_policy_binding(
        audit, tmp_path / 'audit.json', required_scope='requirement_catalog_human_review')
    assert not errors and policy['human_review_status'] == 'waived_by_user'
