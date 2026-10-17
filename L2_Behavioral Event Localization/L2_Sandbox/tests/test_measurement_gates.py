from copy import deepcopy

import pytest

from l2_sandbox.measurement_gates import measurement_gate_binding, verify_measurement_gate_binding
from l2_sandbox.v3_execution import write_json


def audit():
    return {
        "V3_G1": {"byte_equivalence_verified": True, "applicable_configuration_verified": True},
        "V3_G2_R19": {"cp000000_all_pre_first_request_valid": True,
                         "completed_valid_pairs": 20, "ci95_completely_inside_pm_005": True},
        "V3_G3_R20": {"sampled_checkpoint_count": 20, "human_review_completed_count": 20,
                         "missing_restored_directory_count": 0, "missing_scorer_exit_count": 0,
                         "human_conclusions_verified_against_source": True},
        "V3_G4_R21": {"normal_and_p_requests_scanned": 80, "unverified_request_count": 0,
                         "leaked_request_count": 0, "zero_diagnostic_leakage_verified_on_scanned": True},
    }


def test_missing_or_unready_gate_cannot_be_replaced_by_approval(tmp_path):
    path = tmp_path / 'audit.json'
    _, errors = measurement_gate_binding(path)
    assert errors == ['measurement_gate_audit_missing']
    with pytest.raises(ValueError, match='verified measurement gates'):
        verify_measurement_gate_binding({'execution_ready': True})


@pytest.mark.parametrize('gate,field,value', [
    ('V3_G1', 'applicable_configuration_verified', False),
    ('V3_G2_R19', 'completed_valid_pairs', 0),
    ('V3_G3_R20', 'human_review_completed_count', 0),
    ('V3_G4_R21', 'unverified_request_count', 1),
])
def test_each_unresolved_measurement_gate_blocks(tmp_path, gate, field, value):
    document = deepcopy(audit()); document[gate][field] = value
    path = tmp_path / 'audit.json'; write_json(path, document)
    binding, errors = measurement_gate_binding(path)
    assert errors and not binding['passed']


def test_frozen_evidence_is_rechecked_before_production(tmp_path):
    path = tmp_path / 'audit.json'; document = audit(); write_json(path, document)
    binding, errors = measurement_gate_binding(path)
    assert not errors
    verify_measurement_gate_binding({'measurement_gate_binding': binding})
    document['V3_G2_R19']['completed_valid_pairs'] = 21
    write_json(path, document)
    with pytest.raises(ValueError, match='changed'):
        verify_measurement_gate_binding({'measurement_gate_binding': binding})
