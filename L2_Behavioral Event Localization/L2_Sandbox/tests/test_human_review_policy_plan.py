import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('waived_plan', ROOT / 'scripts/v3_d1_ladder.py')
plan_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plan_module)


def test_plan_records_waiver_without_opening_other_gates(tmp_path):
    authorization = ROOT / 'L2_Behavioral Event Localization/L2_Sandbox/configs/human_review_policy_20261018.json'
    plan = plan_module.build_d1_units_and_plan(
        runs_dir=tmp_path / 'absent', diag_dir=tmp_path / 'absent',
        p0a_catalog_path=tmp_path / 'absent.json',
        human_review_authorization_path=authorization,
    )
    assert plan['requirements_review_policy_binding']['human_review_status'] == 'waived_by_user'
    assert not plan['execution_ready']
    assert 'development_cohort_not_complete_45' in plan['readiness_errors']
    assert any(e.startswith('measurement_gate_not_passed:') for e in plan['readiness_errors'])


def test_frozen_requirement_authorization_cannot_change(tmp_path):
    original = ROOT / 'L2_Behavioral Event Localization/L2_Sandbox/configs/human_review_policy_20261018.json'
    authorization = tmp_path / 'authorization.json'
    authorization.write_bytes(original.read_bytes())
    binding, _ = plan_module._bind_requirement_review_policy(authorization)
    plan_module._verify_requirement_review_policy({'requirements_review_policy_binding': binding})
    data = json.loads(authorization.read_text())
    data['source'] += ' changed'
    authorization.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='authorization changed'):
        plan_module._verify_requirement_review_policy({'requirements_review_policy_binding': binding})
