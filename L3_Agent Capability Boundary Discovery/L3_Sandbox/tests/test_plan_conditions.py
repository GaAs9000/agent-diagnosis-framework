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
    if 'family_manifest' in spec:
        spec['family_manifest'] = str((SANDBOX / 'configs' / spec['family_manifest']).resolve())
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
    assert all(j['family_verified'] is True and j['split'] == 'development' for j in p['jobs'])
    assert len({j['family_id'] for j in p['jobs']}) == 4
    family = p['family_manifest']
    assert family['manifest']['family_count'] == 28
    assert family['manifest']['heldout_family_count'] == 19
    assert p['inputs_sha256'][family['path']] == family['sha256']
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


def test_frozen_family_input_changes_block_execution_but_not_reanalysis(prepared, tmp_path):
    path, spec, _ = prepared
    original = Path(spec['family_manifest'])
    manifest = read(original)
    manifest['data_lock'] = str((original.parent / manifest['data_lock']).resolve())
    copied = tmp_path / 'families.json'
    write(copied, manifest)
    spec['family_manifest'] = str(copied)
    write(path, spec)
    directory = tmp_path / 'p'
    frozen = plan.create_plan(path, directory)
    copied.write_text(copied.read_text() + '\n')
    with pytest.raises(L3Error, match='Frozen input changed'):
        plan.verify_plan(directory, verify_images=False)
    assert plan.verify_plan(directory, verify_images=False, verify_execution=False) == frozen


def test_plan_family_mismatch_fails_before_native_loading(prepared, tmp_path, monkeypatch):
    path, spec, _ = prepared
    spec['tasks'][0]['family_id'] = 'not-the-frozen-family'
    write(path, spec)
    monkeypatch.setattr(plan, 'load_native_task', lambda *args: pytest.fail('must validate family before loading assets'))
    with pytest.raises(L3Error, match='family metadata mismatches'):
        plan.create_plan(path, tmp_path / 'p')


@pytest.mark.parametrize('kind', ['duplicate_condition', 'excluded_task', 'unverified_holdout', 'mixed_family', 'unknown_field', 'nan_rpm'])
def test_design_errors_fail_before_native_load(prepared, kind):
    _, original, _ = prepared
    s = copy.deepcopy(original)
    if kind == 'duplicate_condition': s['conditions'].append(s['conditions'][0])
    elif kind == 'excluded_task': s['tasks'][0]['task_id'] = 'officeval_003'
    elif kind == 'unverified_holdout': s['tasks'][0].update(split='heldout', family_verified=False)
    elif kind == 'mixed_family':
        s['tasks'][0].update(split='heldout', family_verified=True)
        s['tasks'][1]['family_id'] = s['tasks'][0]['family_id']
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


@pytest.mark.parametrize('kind', ['asset', 'task', 'rubric', 'verifier'])
def test_reuse_compares_frozen_same_path_contents_without_reading_old_files(prepared, tmp_path, kind):
    path, _, _ = prepared
    source = plan.create_plan(path, tmp_path / 'source_plan')
    job = source['jobs'][0]
    benchmark = job['config']['benchmark']
    names = {
        'asset': Path(benchmark['assets_dir']) / job['task_id'] / 'missing.xlsx',
        'task': Path(benchmark['data_dir']) / 'tasks' / (job['task_id'] + '.json'),
        'rubric': Path(benchmark['data_dir']) / 'rubrics' / (job['task_id'] + '.json'),
        'verifier': Path(benchmark['source_dir']) / 'verifiers/missing.py',
    }
    name = str(names[kind])
    source['inputs_sha256'][name] = 'old-content'
    target = copy.deepcopy(source)
    # Historical source is usable offline even when external paths moved away.
    plan._verify_reuse_contract(source, target, job, job)
    target['inputs_sha256'][name] = 'new-content-at-same-path'
    with pytest.raises(L3Error, match='task execution inputs differ'):
        plan._verify_reuse_contract(source, target, job, job)


@pytest.mark.parametrize('field,value', [('family_id', 'relabelled-family'),
                                       ('family_verified', False), ('split', 'heldout')])
def test_reuse_rejects_family_or_split_relabelling(prepared, tmp_path, field, value):
    path, _, _ = prepared
    source = plan.create_plan(path, tmp_path / 'source_plan')
    job = source['jobs'][0]
    target = copy.deepcopy(source)
    target_job = next(j for j in target['jobs'] if j['trial_id'] == job['trial_id'])
    plan._verify_reuse_contract(source, target, job, target_job)
    # A family manifest is optional; frozen slot identity must enforce this too.
    target.pop('family_manifest', None)
    target_job[field] = value
    with pytest.raises(L3Error, match='family assignment differs.*' + field):
        plan._verify_reuse_contract(source, target, job, target_job)


@pytest.mark.parametrize('drift', [None, 'source', 'prompt', 'assets', 'private', 'provenance',
                                   'input_asset', 'input_private', 'runtime_image',
                                   'evaluation_image', 'analysis_legacy', 'metadata'])
def test_m2_dev_baseline_config_and_slot_reuse(prepared, tmp_path, monkeypatch, drift):
    import hashlib
    from l3_sandbox import run
    _, _, _ = prepared
    pilot_dir = tmp_path / 'pilot'
    pilot_spec = read(SANDBOX / 'configs/budget_pilot.json')
    pilot_spec['base_config'] = str(REPO / 'L1_sandbox/C3_Experimental_Sandbox/configs/l1/officeval.gemini-35lite.t1.json')
    pilot_spec['family_manifest'] = str((SANDBOX / 'configs' / pilot_spec['family_manifest']).resolve())
    pilot_spec_path = tmp_path / 'pilot_spec.json'
    write(pilot_spec_path, pilot_spec)
    pilot_plan = plan.create_plan(pilot_spec_path, pilot_dir)

    # Create one completed verified slot in pilot_dir for officeval_004 replicate 0 under requests_035
    match_job = next(j for j in pilot_plan['jobs'] if j['task_id'] == 'officeval_004' and j['condition_id'] == 'requests_035' and j['replicate'] == 0)
    # Replacement attempt 1 is still genuine evidence for destination attempt 0.
    ep = pilot_dir / 'runs' / match_job['trial_id'] / 'attempt_001/episode'
    write(ep / 'manifest.json', {'recording_mode': 'live', 'lineage': None, 'task': {'task_id': 'officeval_004'}, 'config_sha256': match_job['config_sha256']})
    write(ep / 'raw/resolved_config.json', match_job['config'])
    write(ep / 'official_grade.json', {'y': 0.25})
    write(ep / 'artifacts/inventory.json', {})
    (ep / 'trajectory.jsonl').write_text('')
    (ep / 'state_changes.jsonl').write_text('')
    sealed = {name: hashlib.sha256((ep / name).read_bytes()).hexdigest()
              for name in ('official_grade.json', 'artifacts/inventory.json', 'trajectory.jsonl', 'state_changes.jsonl')}
    write(ep / 'evidence_index.json', sealed)
    expected = hashlib.sha256((ep / 'evidence_index.json').read_bytes()).hexdigest()
    monkeypatch.setattr(run, '_verify_package', lambda package, anchor: {'verified': True})
    monkeypatch.setattr(run, '_summarize', lambda package, frozen_job, attempt: {
        'study_valid': True, 'study_outcome_y': 0.25, 'endpoint_version': 'study-outcome-v1', 'model_versions': ['v1']
    })
    original_provenance = {'original-actor.py': 'original-source-digest'}
    row = {**{k: v for k, v in match_job.items() if k != 'config'}, 'attempt': 1,
           'binding': run._binding(pilot_plan, match_job), 'accepted': True, 'status': 'completed',
           'code_provenance': original_provenance,
           'reused_from': {'experiment_id': 'earlier-pilot', 'trial_id': 'original-trial', 'attempt': 1},
           'expected_index_sha256': expected, 'study_valid': True, 'study_outcome_y': 0.25,
           'endpoint_version': 'study-outcome-v1', 'model_versions': ['v1']}
    run._save_receipt(ep.parent / 'receipt.json', row)
    write(pilot_dir / 'run_results.json', {'rows': [row]})

    pilot_plan = copy.deepcopy(pilot_plan)
    task = next(t for t in pilot_plan['tasks'] if t['task_id'] == match_job['task_id'])
    if drift == 'source':
        pilot_plan['source_sha256'][str(prepared[2])] = 'historical-actor'
    elif drift == 'prompt':
        task['public_prompt_sha256'] = digest('different prompt')
    elif drift == 'assets':
        task['public_assets'] = [{'name': 'different.xlsx', 'sha256': 'old', 'size_bytes': 10}]
    elif drift == 'private':
        task['native_provenance']['private_files_sha256'] = {'source/verifier.py': 'old'}
    elif drift == 'provenance':
        task['native_provenance']['source_commit'] = 'historical-commit'
    elif drift in ('input_asset', 'input_private'):
        benchmark = match_job['config']['benchmark']
        name = (Path(benchmark['assets_dir']) / match_job['task_id'] / 'same-path.xlsx'
                if drift == 'input_asset' else Path(benchmark['source_dir']) / 'same-verifier.py')
        # Frozen snapshots are authoritative; these historical paths need not exist.
        pilot_plan['inputs_sha256'][str(name)] = 'historical-content'
    elif drift in ('runtime_image', 'evaluation_image'):
        pilot_plan['images'][drift] = 'sha256:' + 'b' * 64
    elif drift == 'analysis_legacy':
        pilot_plan.pop('source_scope')
        pilot_plan['source_sha256'][str(Path(plan.__file__).parent / 'report.py')] = 'old-analysis'
    elif drift == 'metadata':
        task.update(selection_reason='Different selection rationale', family_id='different-label')
    write(pilot_dir / 'plan.json', pilot_plan)
    (pilot_dir / 'plan.sha256').write_text(sha_file(pilot_dir / 'plan.json'))

    m2_spec = read(SANDBOX / 'configs/m2_dev_baseline.json')
    m2_spec['base_config'] = str(REPO / 'L1_sandbox/C3_Experimental_Sandbox/configs/l1/officeval.gemini-35lite.t1.json')
    m2_spec['family_manifest'] = str((SANDBOX / 'configs' / m2_spec['family_manifest']).resolve())
    m2_spec['reuse_from'] = [str(pilot_dir)]
    m2_spec_path = tmp_path / 'm2_spec.json'
    write(m2_spec_path, m2_spec)

    m2_dir = tmp_path / 'm2'
    if drift not in (None, 'analysis_legacy', 'metadata'):
        monkeypatch.setattr(plan.shutil, 'copytree', lambda *a, **kw: pytest.fail('copy before contract validation'))
        with pytest.raises(L3Error, match='Reused .* differs? from target plan'):
            plan.create_plan(m2_spec_path, m2_dir)
        assert not m2_dir.exists()
        return
    m2_plan = plan.create_plan(m2_spec_path, m2_dir)
    assert m2_plan['design'] == 'baseline'
    assert m2_plan['workload']['trial_slots'] == 45
    assert len({j['family_id'] for j in m2_plan['jobs']}) == 9
    assert m2_plan['reuse_from'] == [str(pilot_dir.resolve())]
    seeded = read(m2_dir / 'run_results.json')['rows']
    assert len(seeded) == 1
    assert seeded[0]['task_id'] == 'officeval_004'
    assert seeded[0]['reused_from']['trial_id'] == match_job['trial_id']
    assert seeded[0]['reused_from']['attempt'] == 1
    assert seeded[0]['reused_from']['reused_from'] == row['reused_from']
    assert seeded[0]['code_provenance'] == original_provenance

