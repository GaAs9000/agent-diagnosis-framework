"""Research denominator gates must never invent a score before grading."""
import json
from pathlib import Path
import pytest
from c3_sandbox.l1.evaluation.officeval import interpret
from c3_sandbox.l1.evaluation.lhtb import interpret_lhtb
from c3_sandbox.l1.evaluation.study import study_outcome


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def package(tmp_path):
    write(tmp_path / 'manifest.json', {'recording_mode': 'live', 'task': {'benchmark': 'officeval', 'task_id': 'officeval_003'},
          'status': {'execution_validity': 'valid', 'evaluation_validity': 'invalid', 'evidence_completeness': 'complete'}})
    raw = {'id': '003', 'file_name': '', 'status': 'error', 'error': 'FileNotFoundError: no docx',
           'dim1_pass': False, 'dim1_reason': '', 'dim2_items': [], 'total_score': 0, 'max_score': 0}
    write(tmp_path / 'official_grade.json', {'official_raw': raw, 'projection': interpret(raw, 'officeval_003'),
          'process': {'container_exit_code': 0, 'stop_reason': None}})
    write(tmp_path / 'grading/worker-output/worker.json', {'official_returned': True, 'exception': None})
    return tmp_path


def test_healthy_missing_delivery_is_study_zero(package):
    result = study_outcome(package)
    assert result['study_valid'] and result['study_outcome_y'] == 0
    assert result['native_evaluation_validity'] == 'invalid'
    assert result['zero_reason'] == 'missing_or_damaged_delivery'


def test_openxml_malformed_line_spacing_attribute_error_is_study_zero(package):
    path = package / 'official_grade.json'
    grade = json.loads(path.read_text())
    grade['official_raw'].update(
        id='070',
        file_name='技术路线图.pptx',
        status='error',
        error="AttributeError: 'NoneType' object has no attribute 'val'",
    )
    manifest = json.loads((package / 'manifest.json').read_text())
    manifest['task']['task_id'] = 'officeval_070'
    write(package / 'manifest.json', manifest)
    grade['projection'] = interpret(grade['official_raw'], 'officeval_070')
    write(path, grade)
    result = study_outcome(package)
    assert result['study_valid'] and result['study_outcome_y'] == 0.0
    assert result['zero_reason'] == 'missing_or_damaged_delivery'



@pytest.mark.parametrize('mutation', ['missing_grade', 'wrong_id', 'process_failure', 'worker_failure', 'infra',
                                     'nan', 'inf', 'bool', 'out_of_range', 'projection_mismatch', 'arbitrary_exception'])
def test_fail_closed(package, mutation):
    path = package / 'official_grade.json'
    grade = json.loads(path.read_text())
    raw = grade['official_raw']
    if mutation == 'missing_grade':
        path.unlink()
    else:
        if mutation == 'wrong_id': raw['id'] = '090'
        if mutation == 'process_failure': grade['process']['container_exit_code'] = 1
        if mutation == 'worker_failure': write(package / 'grading/worker-output/worker.json', {'official_returned': False})
        if mutation == 'infra': grade['error_kind'] = 'infrastructure'
        if mutation in ('nan', 'inf', 'bool', 'out_of_range'):
            raw.update(status='ok', error=None, dim1_pass=True, max_score=15,
                       total_score={'nan': float('nan'), 'inf': float('inf'), 'bool': True, 'out_of_range': 16}[mutation])
        if mutation == 'arbitrary_exception': raw['error'] = 'RuntimeError: unexpected internal failure'
        grade['projection'] = interpret(raw, 'officeval_003')
        if mutation == 'projection_mismatch': grade['projection']['completion_rate'] = .8
        write(path, grade)
    result = study_outcome(package)
    assert not result['study_valid'] and result['study_outcome_y'] is None


@pytest.mark.parametrize('band,case', [(0, 'official_below_pass_band'), (None, 'official_no_moves'), (3, 'official_score')])
def test_lhtb_pass_threshold_is_binary(package, band, case):
    manifest = json.loads((package / 'manifest.json').read_text())
    manifest['task'] = {'task_id': 'sokoban', 'benchmark': 'lhtb'}
    write(package / 'manifest.json', manifest)
    code = 0 if band else 1
    raw = {'status': 'ok', 'error': None, 'task_id': 'sokoban', 'reward': .1 if band else 0,
           'raw_reward': band, 'band': band or 0, 'scoring_case': case, 'protocol': {'pytest_exit': code}}
    write(package / 'official_grade.json', {'official_raw': raw, 'projection': interpret_lhtb(raw, 'sokoban'),
                                            'process': {'container_exit_code': code}})
    if code:
        message = 'AssertionError: No moves found at /submission/moves.log.' if band is None else 'AssertionError: Band 0 &lt; pass threshold 1.'
        (package / 'grading/worker-output/pytest.xml').write_text(f'<testsuite><testcase name="test_sokoban_progress"><failure message="{message}"/></testcase></testsuite>')
    result = study_outcome(package)
    assert result['study_valid']
    assert result['study_outcome_y'] == (1.0 if band else 0.0)


def test_m1_read_only_reconciliation():
    root = Path(__file__).resolve().parents[4]
    parents = root / 'L1_sandbox/C3_Experimental_Sandbox/runs/m1_20261014_2034r1'
    branches = root / 'L2_Behavioral Event Localization/L2_Sandbox/validation/m1_20261014_2034r1'
    packages = [p.parent for r in (parents, branches) for p in r.rglob('manifest.json')]
    if not packages:
        pytest.skip('local M1 evidence is not distributed with source')
    outcomes = [study_outcome(p) for p in packages]
    assert len(outcomes) == 81
    assert all(o['study_valid'] for o in outcomes)
    assert sum(o['native_evaluation_validity'] == 'invalid' and o['study_outcome_y'] == 0 for o in outcomes) == 37


def test_negative_native_rubric_penalty_normalizes_to_zero(package):
    path = package / 'official_grade.json'
    grade = json.loads(path.read_text())
    grade['official_raw'].update(status='ok', error=None, dim1_pass=True,
                                file_name='result.docx', total_score=-1, max_score=15,
                                dim2_items=[{'rule': 'penalty', 'max_delta': 0, 'delta': -1, 'hit': True}])
    grade['projection'] = interpret(grade['official_raw'], 'officeval_003')
    write(path, grade)
    result = study_outcome(package)
    assert grade['official_raw']['total_score'] == -1
    assert grade['projection']['completion_rate'] == 0
    assert result['study_valid'] and result['study_outcome_y'] == 0
    assert result['study_classification'] == 'valid_zero'


@pytest.mark.parametrize('band,code,case', [(2, 1, 'official_below_pass_band'),
                                         (0, 0, 'official_score'),
                                         (False, 1, 'official_below_pass_band')])
def test_lhtb_inconsistent_band_and_pass_fail_closed(package, band, code, case):
    manifest = json.loads((package / 'manifest.json').read_text())
    manifest['task'] = {'task_id': 'sokoban', 'benchmark': 'lhtb'}
    write(package / 'manifest.json', manifest)
    raw = {'status': 'ok', 'error': None, 'task_id': 'sokoban', 'reward': 0,
           'raw_reward': 0, 'band': band, 'scoring_case': case, 'protocol': {'pytest_exit': code}}
    write(package / 'official_grade.json', {'official_raw': raw, 'projection': interpret_lhtb(raw, 'sokoban'),
                                            'process': {'container_exit_code': code}})
    (package / 'grading/worker-output/pytest.xml').write_text('<testsuite><testcase name="test_sokoban_progress"><failure message="AssertionError: Band 0 &lt; pass threshold 1."/></testcase></testsuite>')
    assert not study_outcome(package)['study_valid']


@pytest.mark.parametrize('case', ['missing_grade', 'infra_with_empty_inventory', 'missing_worker',
                                  'unknown_execution', 'incomplete_evidence', 'fixture_mode'])
def test_empty_inventory_never_bypasses_health_gates(package, case):
    write(package / 'artifacts/inventory.json', {'files': []})
    manifest = json.loads((package / 'manifest.json').read_text())
    if case == 'missing_grade':
        (package / 'official_grade.json').unlink()
    elif case == 'infra_with_empty_inventory':
        grade = json.loads((package / 'official_grade.json').read_text())
        grade['error_kind'] = 'infrastructure'
        write(package / 'official_grade.json', grade)
    elif case == 'missing_worker':
        (package / 'grading/worker-output/worker.json').unlink()
    elif case == 'unknown_execution':
        manifest['status']['execution_validity'] = 'unknown'
    elif case == 'incomplete_evidence':
        manifest['status'].pop('evidence_completeness')
    elif case == 'fixture_mode':
        manifest['recording_mode'] = 'test_fixture'
    write(package / 'manifest.json', manifest)
    assert study_outcome(package)['study_valid'] is False


def test_fixture_opt_in_requires_same_grader_health(package):
    manifest = json.loads((package / 'manifest.json').read_text())
    manifest['recording_mode'] = 'test_fixture'
    write(package / 'manifest.json', manifest)
    assert study_outcome(package, allow_test_fixture=True)['study_valid']
    (package / 'grading/worker-output/worker.json').unlink()
    assert not study_outcome(package, allow_test_fixture=True)['study_valid']


def test_shared_l2_matrix_endpoint_and_raw_immutability(package):
    import importlib.util
    from l2_sandbox.evidence.l1 import L1Package
    # No inventory is required to infer a zero: the returned official delivery
    # error and healthy worker provide the evidence. Conversely inventory alone
    # cannot establish grader health (covered above).
    manifest = json.loads((package / 'manifest.json').read_text())
    manifest['episode_id'] = 'shared-outcome-fixture'
    write(package / 'manifest.json', manifest)
    from c3_sandbox.l1.storage import file_digest
    write(package / 'evidence_index.json', {'episode_id': manifest['episode_id'], 'files': [
        {'path': p.relative_to(package).as_posix(), 'sha256': file_digest(p), 'size_bytes': p.stat().st_size}
        for p in sorted(package.rglob('*')) if p.is_file()]})
    before = {p.relative_to(package): p.read_bytes() for p in package.rglob('*') if p.is_file()}
    script = Path(__file__).resolve().parents[2] / 'scripts/l1/run_matrix.py'
    spec = importlib.util.spec_from_file_location('matrix_shared_outcome', script)
    matrix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matrix)
    row = matrix.summarize_episode('officeval_003', 0, package)
    outcome = L1Package(package).outcome()
    assert row['study_outcome_y'] == row['completion_rate'] == outcome['y'] == 0
    assert row['study_valid'] == row['score_eligible'] == outcome['valid'] is True
    assert row['endpoint_version'] == outcome['endpoint_version']
    assert outcome['native_evaluation_validity'] == row['evaluation_validity'] == 'invalid'
    assert outcome['native_status'] == 'error'
    assert before == {p.relative_to(package): p.read_bytes() for p in package.rglob('*') if p.is_file()}
