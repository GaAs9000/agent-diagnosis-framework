import copy
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from l3_sandbox import plan
from l3_sandbox.common import L3Error, digest, read, sha_file, write
from l3_sandbox.conditions import (ConditionOfficeVal, TRANSFORM_VERSION,
                                   placebo_context, structure_context)

SANDBOX = Path(__file__).resolve().parents[1]
REPO = SANDBOX.parents[1]


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    spec = read(SANDBOX / 'configs/budget_pilot.json')
    spec['base_config'] = str(REPO / 'L1_sandbox/C3_Experimental_Sandbox/configs/l1/officeval.gemini-35lite.t1.json')
    source = tmp_path / 'source.py'
    source.write_text('source one')
    monkeypatch.setattr(plan, 'source_inventory', lambda: {str(source): sha_file(source)})
    monkeypatch.setattr(plan, 'image_id', lambda image: 'sha256:' + 'a' * 64)
    native = SimpleNamespace(prompt='Public native task', assets=(), private_files={},
                             provenance={'adapter': 'officeval-native'})
    monkeypatch.setattr(plan, 'load_native_task', lambda config, task_id: native)
    path = tmp_path / 'study.json'
    write(path, spec)
    return path, spec, source


def test_pilot_freezes_45_balanced_fresh_slots_and_only_request_budget_varies(prepared, tmp_path):
    from c3_sandbox.l1.storage import digest as l1_digest
    path, _, _ = prepared
    p = plan.create_plan(path, tmp_path / 'p1')
    assert p['workload']['trial_slots'] == 45
    assert p['workload']['logical_request_bound'] == 1695
    assert p['workload']['http_attempt_bound_without_replacements'] == 6780
    assert p['workload']['estimated_actual_cost'] is None
    assert len({j['trial_id'] for j in p['jobs']}) == 45
    assert all(j['config_sha256'] == l1_digest(j['config']) for j in p['jobs'])
    assert all(j['config']['budget']['max_tool_calls'] == 140 for j in p['jobs'])
    assert all(j['family_verified'] is False and j['split'] == 'development' for j in p['jobs'])
    for job in p['jobs']:
        prompt = job['config']['agent']['system_prompt']
        assert f"At most {job['max_model_requests']} model requests" in prompt
        assert 'Do not make dozens of exploratory shell calls' in prompt
        assert '{max_model_requests}' not in prompt
    p2 = plan.create_plan(path, tmp_path / 'p2')
    assert p == p2
    assert plan.verify_plan(tmp_path / 'p1', verify_images=False) == p
    with pytest.raises(L3Error, match='exists'):
        plan.create_plan(path, tmp_path / 'p1')


@pytest.mark.parametrize('target', ['source', 'spec', 'plan'])
def test_freeze_rejects_source_spec_or_plan_drift(prepared, tmp_path, target):
    path, spec, source = prepared
    directory = tmp_path / 'p'
    plan.create_plan(path, directory)
    if target == 'source':
        source.write_text('source two')
    elif target == 'spec':
        spec['replicates'] = 1
        write(path, spec)
    else:
        p = read(directory / 'plan.json'); p['jobs'][0]['config']['budget']['max_model_requests'] = 500
        write(directory / 'plan.json', p)
    with pytest.raises(L3Error):
        plan.verify_plan(directory, verify_images=False)


@pytest.mark.parametrize('kind', ['duplicate_condition', 'excluded_task', 'unverified_holdout', 'mixed_family', 'unknown_field', 'nan_rpm'])
def test_design_errors_fail_before_native_load(prepared, kind):
    _, original, _ = prepared
    s = copy.deepcopy(original)
    if kind == 'duplicate_condition': s['conditions'].append(s['conditions'][0])
    elif kind == 'excluded_task': s['tasks'][0]['task_id'] = 'officeval_003'
    elif kind == 'unverified_holdout': s['tasks'][0]['split'] = 'heldout'
    elif kind == 'mixed_family': s['tasks'][0].update(split='heldout', family_verified=True)
    elif kind == 'unknown_field': s['api_key'] = 'inline-secret-must-never-be-saved'
    else: s['limits']['rpm'] = float('nan')
    with pytest.raises(L3Error): plan.validate_spec(s)


def test_structure_design_has_exact_six_cells_and_native_plugin(prepared, tmp_path):
    path, s, _ = prepared
    s['design'] = 'structure_budget'
    s['conditions'] = [{'condition_id': f'{context}_{budget}', 'max_model_requests': budget, 'context': context}
                       for budget in (35, 70) for context in ('original', 'placebo', 'structure')]
    s['analysis']['primary_comparison'] = ['structure_70', 'placebo_70']
    write(path, s)
    p = plan.create_plan(path, tmp_path / 'structure')
    assert len(p['jobs']) == 90
    assert all(j['config']['benchmark']['adapter'] == 'l3-officeval' for j in p['jobs'])
    assert all(j['config']['plugins']['load_installed'] for j in p['jobs'])
    s['conditions'].pop()
    with pytest.raises(L3Error, match='three contexts'): plan.validate_spec(s)


def test_public_structure_has_coordinates_not_values_formulas_or_private_files(tmp_path):
    from c3_sandbox.l1.core.contracts import Asset
    file = tmp_path / 'public.xlsx'
    with zipfile.ZipFile(file, 'w') as archive:
        archive.writestr('xl/workbook.xml', '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheets><sheet name="Public Sheet"/></sheets></workbook>')
        archive.writestr('xl/worksheets/sheet1.xml', '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1:D9"/><sheetData><row><c r="D9"><f>HIDDEN_FORMULA_TEXT</f><v>HIDDEN_CELL_VALUE</v></c></row></sheetData></worksheet>')
    asset = Asset(file.name, file, sha_file(file), file.stat().st_size)
    task = SimpleNamespace(assets=(asset,), private_files={'rubric': tmp_path / 'MUST_NOT_OPEN'})
    text = structure_context(task)
    assert 'D9' in text and 'A1:D9' in text and 'Public Sheet' in text
    assert 'HIDDEN_FORMULA_TEXT' not in text and 'HIDDEN_CELL_VALUE' not in text
    assert 'MUST_NOT_OPEN' not in text
    assert len(placebo_context(text)) == len(text)


def test_condition_adapter_changes_only_public_prompt_and_provenance(tmp_path, monkeypatch):
    from c3_sandbox.l1.core.contracts import Task, EvaluationSpec
    from c3_sandbox.l1.benchmarks.officeval.adapter import NativeOfficeVal
    task = Task('officeval', 'officeval_004', 'Native instruction', (), {'original': True},
                {'private': tmp_path / 'unread'}, EvaluationSpec('python-verifier', tmp_path, 'verifier.py', 'officeval_004'))
    monkeypatch.setattr(NativeOfficeVal, 'load', lambda self, task_id: task)
    structure = structure_context(task)
    for context in ('original', 'structure', 'placebo'):
        appendix = '' if context == 'original' else structure if context == 'structure' else placebo_context(structure)
        config = {'source_dir': str(tmp_path), 'data_dir': str(tmp_path), 'assets_dir': str(tmp_path),
                  'l3_transform_version': TRANSFORM_VERSION, 'l3_context': context,
                  'l3_appendix': appendix, 'l3_condition_id': context}
        variant = ConditionOfficeVal(config).load(task.task_id)
        assert variant.assets is task.assets and variant.private_files is task.private_files
        assert variant.evaluation is task.evaluation
        assert variant.prompt.startswith(task.prompt)
        assert (variant.prompt == task.prompt) is (context == 'original')
        assert variant.provenance['l3_condition']['appendix_sha256'] == digest(appendix)
        config['l3_appendix'] = 'unexpected'
        with pytest.raises(L3Error, match='differs'): ConditionOfficeVal(config).load(task.task_id)
