"""Bounded CPU fixtures only: no Actor, Docker, GPU, or official grader."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/grade_context004_chain.py'
spec = importlib.util.spec_from_file_location('context004_chain_test', SCRIPT)
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class Package:
    def __init__(self, root):
        self.root = root
        self.manifest = self.json('manifest.json')
        self.episode_id = self.manifest['episode_id']
        self.index_sha256 = d.file_digest(root / 'evidence_index.json')
        self.task_id = 'officeval_004'
        self.family_id = 'officeval:officeval_004'
        self.benchmark = 'officeval'
        self.events = self.json('events.json')
        self.local_turns = len(self.events)
        self.start_turn = self.json('checkpoints/cp000000.json')['budget_used']['model_requests']
        self.turns = self.local_turns + self.start_turn

    def json(self, name):
        return json.loads((self.root / name).read_text())

    def checkpoints(self):
        out = {}
        for path in sorted((self.root / 'checkpoints').glob('cp*.json')):
            cp = self.json(str(path.relative_to(self.root)))
            out[int(cp['checkpoint_id'][2:])] = {**cp, 'sha256': d.file_digest(path)}
        return out

    def validate_resume_checkpoint(self, ordinal):
        assert ordinal in self.checkpoints()  # Local ordinal, never a global turn.
        return self.checkpoints()[ordinal]

    def termination(self):
        return 'assistant_finished'


@pytest.fixture
def chain(tmp_path, monkeypatch):
    worker = d.ROOT / 'L1_sandbox/C3_Experimental_Sandbox/src/c3_sandbox/l1/evaluation/worker.py'
    previous = None
    for number, (relative, episode, offset, end) in enumerate(d.SEGMENTS):
        root = tmp_path / relative
        write(root / 'evidence_index.json', {'episode': episode, 'files': []})
        write(root / 'events.json', [{'kind': 'model.request', 'data': {
            'request_id': f'{episode}:r{turn:06d}',
            **({'source': 'saved_response', 'http_attempts': 0} if number == 1 and turn == 7 else {})}}
            for turn in range(offset, end)])
        snap = {'snapshot_id': 's000000', 'files': [], 'directories': [], 'total_bytes': 0,
                'content_sha256': 'empty'}
        write(root / 'states/snapshots/s000000.json', snap)
        for local in range(end - offset):
            write(root / f'checkpoints/cp{local:06d}.json', {
                'checkpoint_id': f'cp{local:06d}', 'snapshot_id': 's000000',
                'budget_used': {'model_requests': offset + local}})
        lineage = None
        if previous is not None:
            cp_id = 'cp000007' if number == 1 else 'cp000013'
            cp_path = previous / f'checkpoints/{cp_id}.json'
            lineage = {'parent_episode_id': d.SEGMENTS[number - 1][1],
                       'parent_index_sha256': d.file_digest(previous / 'evidence_index.json'),
                       'parent_checkpoint_id': cp_id, 'parent_checkpoint_sha256': d.file_digest(cp_path)}
            write(root / 'lineage/parent_checkpoint.json', json.loads(cp_path.read_text()))
        write(root / 'manifest.json', {'episode_id': episode, 'lineage': lineage,
                                      'usage': {'model_requests_including_parent': end}})
        write(root / 'official_grade.json', {'projection': {'completion_rate': None if number == 0 else 8 / 22,
               'evaluation_validity': 'valid', 'official_dimensions': {'dim1_pass': True, 'max_score': 22, 'total_score': 8}},
               'runtime': {'image_id': 'sha256:frozen', 'architecture': 'arm64',
                           'python_environment': {'grader_worker_sha256': d.file_digest(worker)}}})
        write(root / 'raw/resolved_config.json', {'budget': {'max_model_requests': 35}})
        write(root / 'benchmark/description.json', {'task_id': 'officeval_004', 'evaluation': {}})
        previous = root
    monkeypatch.setattr(d, 'L1Package', Package)
    monkeypatch.setattr(d, 'verify_package', lambda *_, **__: {'verified': True})
    monkeypatch.setattr(d.census, 'source_identity', lambda _: {'sha256': 'official-source'})
    monkeypatch.setattr(d, 'analysis_identity', lambda: {'sha256': 'analysis', 'files': {}})
    return tmp_path


def fake_grader(pkg, **kwargs):
    rows = [{'kind': 'terminal_submission', 'k': pkg.turns, 'score': 8 / 22, 'deductions': {'rule': 14 / 22}}]
    for cp in pkg.checkpoints().values():
        boundary = cp['budget_used']['model_requests']
        rows.append({'kind': 'checkpoint', 'k': boundary, 'checkpoint_id': cp['checkpoint_id'],
                     'score': 1 if boundary == 3 else 8 / 22, 'candidate_status': 'ok',
                     'deductions': {'rule': 0 if boundary == 3 else 14 / 22}})
    rows.append({'kind': 'checkpoint', 'k': pkg.turns, 'checkpoint_id': None,
                 'score': 8 / 22, 'candidate_status': 'ok', 'deductions': {'rule': 14 / 22}})
    return rows


def test_exact_plan_preserves_local_checkpoints_and_request_boundaries(chain):
    plan = d.make_plan(chain)
    assert [r['request_offset'] for r in plan['segments']] == [0, 7, 21]
    assert plan['segments'][1]['checkpoints'][-1]['local_checkpoint_id'] == 'cp000013'
    assert plan['segments'][1]['checkpoints'][-1]['request_boundary'] == 20
    assert plan['final_request_boundary'] == 30
    d.validate_plan(plan, chain)


@pytest.mark.parametrize('change', ['index_link', 'checkpoint_link', 'seam', 'request_ids', 'worker', 'replay'])
def test_plan_rejects_broken_chain_evidence(chain, change):
    root = chain / d.SEGMENTS[1][0]
    if change in ('index_link', 'checkpoint_link'):
        path = root / 'manifest.json'
        value = json.loads(path.read_text())
        value['lineage']['parent_' + ('index_sha256' if change == 'index_link' else 'checkpoint_sha256')] = 'wrong'
    elif change == 'seam':
        path = root / 'states/snapshots/s000000.json'
        value = json.loads(path.read_text()); value['content_sha256'] = 'different'
    elif change == 'worker':
        path = root / 'official_grade.json'
        value = json.loads(path.read_text()); value['runtime']['python_environment']['grader_worker_sha256'] = 'wrong'
    else:
        path = root / 'events.json'
        value = json.loads(path.read_text())
        value[0]['data']['request_id' if change == 'request_ids' else 'http_attempts'] = 'wrong' if change == 'request_ids' else 1
    write(path, value)
    with pytest.raises(ValueError):
        d.make_plan(chain)


def test_fake_official_grades_merge_latest_seams_without_synthetic_anchors(chain, tmp_path):
    plan = d.make_plan(chain)
    output = tmp_path / 'output'; output.mkdir()
    calls = []
    def grader(pkg, **kwargs):
        calls.append(kwargs)
        return fake_grader(pkg, **kwargs)
    d.grade_segments(plan, chain, output, grader=grader)
    rows = d.read_rows(output / 'grades.jsonl')
    assert [r['k'] for r in rows[1:]] == list(range(31))
    assert rows[0]['segment_provenance']['episode_id'] == d.SEGMENTS[-1][1]
    assert [c['verify_terminal_submission'] for c in calls] == [False, False, True]
    assert all(str(c['cache_dir']).endswith('cache/analysis') for c in calls)
    assert rows[8]['segment_provenance']['episode_id'] == d.SEGMENTS[1][1]  # boundary7 seam
    assert rows[-1]['checkpoint_id'] is None and rows[-1]['anchor_eligible'] is False
    assert rows[-2]['checkpoint_id'] == 'cp000008' and rows[-2]['anchor_eligible'] is True
    old = d.read_rows(output / 'segment_0_grades.jsonl')[0]
    assert old['kind'] == 'historical_segment_terminal' and old['score'] is None
    defects = d.build_defect_table(Package(chain / d.SEGMENTS[-1][0]), rows, [])
    assert defects['defects'][0]['kind'] == 'broken'  # Prefix success is visible to E1.


def test_local_ordinal_grading_api_is_rejected(chain):
    plan = d.make_plan(chain)
    record = plan['segments'][2]
    pkg = Package(chain / record['path'])
    with pytest.raises(ValueError, match='local ordinal'):
        d.annotate_rows(pkg, record, [{'kind': 'checkpoint', 'checkpoint_id': 'cp000001', 'k': 1}])


def test_missing_boundary_is_not_filled_as_real_evidence(chain):
    plan = d.make_plan(chain)
    rows = [d.annotate_rows(Package(chain / r['path']), r, fake_grader(Package(chain / r['path']))) for r in plan['segments']]
    rows[0] = [r for r in rows[0] if r['k'] != 3]
    with pytest.raises(ValueError, match='contiguous'):
        d.merge_grades(rows)


def test_facts_trim_overridden_eighth_response_without_counting_replay_twice(chain):
    plan = d.make_plan(chain)
    packages = [Package(chain / r['path']) for r in plan['segments']]
    def builder(pkg):
        return [{'turn': turn, 'type': 'fixture'} for turn in range(pkg.start_turn + 1, pkg.turns + 1)]
    facts, historical = d.merge_facts(plan, packages, builder)
    assert [r['turn'] for r in facts] == list(range(1, 31))
    assert len(historical) == 1 and historical[0]['turn'] == 8
    assert historical[0]['segment_provenance']['episode_id'] == d.SEGMENTS[0][1]
    assert facts[7]['segment_provenance']['episode_id'] == d.SEGMENTS[1][1]


def test_default_cli_only_freezes_plan(chain, tmp_path, monkeypatch):
    output = tmp_path / 'plan'
    monkeypatch.setattr('sys.argv', [str(SCRIPT), '--input-root', str(chain), '--output', str(output)])
    monkeypatch.setattr(d, 'grade_official', lambda *_: pytest.fail('official grading was invoked'))
    monkeypatch.setattr(d, 'analyze_cpu', lambda *_: pytest.fail('facts analysis was invoked'))
    d.main()
    assert (output / 'plan.json').is_file()
    with pytest.raises(FileExistsError):
        d.main()


def test_official_stage_reservation_is_once_even_after_failure(chain, tmp_path, monkeypatch):
    plan = d.make_plan(chain); output = tmp_path / 'output'
    plan_path = tmp_path / 'plan.json'; write(plan_path, plan)
    monkeypatch.setattr(d.platform, 'system', lambda: 'Darwin')
    monkeypatch.setattr(d.platform, 'machine', lambda: 'arm64')
    calls = []
    def failed(*_, **__):
        calls.append(1)
        raise TimeoutError('fake hard deadline')
    monkeypatch.setattr(d.owned, 'bounded', failed)
    monkeypatch.setattr(d.owned, 'cleanup_e1', lambda *_: {'status': 'confirmed_empty'})
    with pytest.raises(TimeoutError):
        d.grade_official(plan, plan_path, chain, output)
    with pytest.raises(FileExistsError):
        d.grade_official(plan, plan_path, chain, output)
    assert calls == [1]


def test_analysis_source_change_does_not_match_frozen_plan(chain, monkeypatch):
    plan = d.make_plan(chain)
    monkeypatch.setattr(d, 'analysis_identity', lambda: {'sha256': 'new-analysis', 'files': {}})
    with pytest.raises(ValueError, match='frozen analysis'):
        d.validate_plan(plan, chain)
