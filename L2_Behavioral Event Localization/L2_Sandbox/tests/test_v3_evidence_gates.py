"""Offline adverse evidence fixtures; no model, runtime, or evaluator execution."""
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('v3_evidence_gates', ROOT / 'scripts/v3_evaluate_gates_g2_g3_g4.py')
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def package(path, episode, task='task_a'):
    cfg = {'agent': {'harness': 'pi'}, 'budget': {'max_model_requests': 35}}
    manifest = {'episode_id': episode, 'task': {'benchmark': 'officeval', 'task_id': task},
                'agent': {'identity': {'model': 'test'}, 'configuration': {'harness': 'pi'}},
                'runtime': {'adapter': 'cold', 'image_id': 'image', 'architecture': 'arm64',
                            'network': 'none', 'resource_limits': {'cpus': 1}},
                'config_sha256': g.digest(cfg), 'lineage': None}
    put(path / 'manifest.json', manifest)
    put(path / 'raw/resolved_config.json', cfg)
    put(path / 'checkpoints/index.json', {'checkpoints': [{'checkpoint_id': 'cp000000', 'path': 'checkpoints/cp000000.json'}]})
    put(path / 'checkpoints/cp000000.json', {'episode_id': episode, 'checkpoint_id': 'cp000000',
        'snapshot_id': 's000002', 'config_sha256': g.digest(cfg), 'qualification': {'eligible': True},
        'budget_used': {'model_requests': 0, 'tool_calls': 0},
        'conversation': {'messages': [{'role': 'user', 'content': 'public task'}]},
        'pending_calls': [], 'pending_observations': []})
    put(path / 'states/snapshots/s000002.json', {'snapshot_id': 's000002', 'label': 'initial',
        'files': [], 'directories': [], 'content_sha256': 'initial'})
    return manifest


def pair(tmp_path):
    parent, child = tmp_path / 'p', tmp_path / 'p_c0'
    package(parent, 'parent'); cm = package(child, 'child')
    cm['lineage'] = {'parent_episode_id': 'parent', 'parent_checkpoint_id': 'cp000000',
        'parent_index_sha256': g.sha_file(parent / 'manifest.json'),
        'parent_checkpoint_sha256': g.sha_file(parent / 'checkpoints/cp000000.json')}
    put(child / 'manifest.json', cm)
    return parent, child


def test_bound_pair_requires_seal_and_binds_identity(tmp_path, monkeypatch):
    parent, child = pair(tmp_path)
    seen = []
    def outcome(path, **kwargs):
        seen.append(kwargs)
        return {'study_valid': True, 'study_outcome_y': .5}
    monkeypatch.setattr(g, 'study_outcome', outcome)
    assert g._bound_pair(parent, child)['diff_resume_minus_fresh'] == 0
    assert seen == [{'verify_evidence': True}, {'verify_evidence': True}]
    cm = g._load_json(child / 'manifest.json'); cm['task']['task_id'] = 'wrong'
    put(child / 'manifest.json', cm)
    with pytest.raises(Exception, match='task|checkpoint'):
        g._bound_pair(parent, child)


@pytest.mark.parametrize('field,value', [('parent_episode_id', 'unrelated'),
    ('parent_checkpoint_id', 'cp000001'), ('parent_index_sha256', 'wrong'),
    ('parent_checkpoint_sha256', 'wrong')])
def test_rejects_each_lineage_mismatch(tmp_path, field, value):
    parent, child = pair(tmp_path)
    cm = g._load_json(child / 'manifest.json'); cm['lineage'][field] = value
    put(child / 'manifest.json', cm)
    with pytest.raises(Exception, match='lineage'):
        g._bound_pair(parent, child)


def test_rejects_model_and_config_drift(tmp_path):
    parent, child = pair(tmp_path)
    cm = g._load_json(child / 'manifest.json'); cm['agent']['identity']['model'] = 'other'
    put(child / 'manifest.json', cm)
    with pytest.raises(ValueError, match='agent'):
        g._bound_pair(parent, child)
    cm['agent']['identity']['model'] = 'test'; put(child / 'manifest.json', cm)
    put(child / 'raw/resolved_config.json', {'budget': {'max_model_requests': 99}})
    with pytest.raises(Exception, match='config'):
        g._bound_pair(parent, child)


def test_c0_is_actual_state_not_filename(tmp_path):
    package(tmp_path, 'p')
    cp = g._load_json(tmp_path / 'checkpoints/cp000000.json')
    cp['budget_used']['model_requests'] = 1
    put(tmp_path / 'checkpoints/cp000000.json', cp)
    assert not g._verify_cp000000_before_first_request(tmp_path)['pre_first_request_valid']


def test_task_clusters_and_non_degenerate_constant_samples():
    pairs = [{'task_id': f't{i % 5}', 'diff_resume_minus_fresh': 0} for i in range(20)]
    mean, ci, count = g._cluster_interval(pairs)
    assert mean == 0 and count == 5 and ci[0] < -.05 and ci[1] > .05
    assert g._cluster_interval(pairs * 10) == (mean, ci, count)


def test_g3_reads_metadata_and_preserves_missing_provenance(tmp_path):
    package(tmp_path, 'p')
    row = g._checkpoint_record(tmp_path, {'checkpoint_id': 'cp000000', 'snapshot_id': 's000000',
                                           'candidate': ['submission/a'], 'score': .2})
    assert row['snapshot_id'] == 's000002'
    assert row['snapshot_sha256'] == g.sha_file(tmp_path / 'states/snapshots/s000002.json')
    assert row['grade_snapshot_matches_metadata'] is False
    assert row['restored_directory'] is None and row['scorer_exit_code'] is None
    assert row['reviewer'] == row['human_conclusion'] == ''


def test_boundary_public_overlap_allowed_private_formula_detected():
    public = [{'role': 'user', 'content': 'Use formula =SUM(A1:A8) in the public task'}]
    private = ['Use formula =SUM(A1:A8) in the public task', 'shape_count >= 5; top <= 1.2; height >= 5.0']
    clean = g.audit_request_boundary({'messages': public}, public_messages=public, private_corpus=private)
    assert clean['matches'] == [] and clean['status'] == 'unverified'
    injected = g.audit_request_boundary({'messages': public + [{'role': 'user', 'content': private[1]}]},
                                       public_messages=public, private_corpus=private)
    assert injected['status'] == 'suspected_leakage' and injected['matches']


@pytest.mark.parametrize('payload', [{}, {'messages': []}, {'messages': [3]}])
def test_malformed_requests_never_pass(payload):
    assert g.audit_request_boundary(payload, public_messages=[{'role': 'user', 'content': 'task'}],
                                   private_corpus=['a private rule'])['status'] == 'unverified'


def test_cache_uses_real_candidate_schema_not_absent_hash(tmp_path):
    p = tmp_path / g.COHORT / 'ep/grades.jsonl'; p.parent.mkdir(parents=True)
    rows = [{'kind': 'header', 'grading_elapsed_seconds': 2.5},
            {'kind': 'checkpoint', 'candidate_status': 'no_candidate', 'candidate': [], 'candidates_evaluated': 0},
            {'kind': 'checkpoint', 'candidate_status': 'ok', 'candidate': ['submission/a']},
            {'kind': 'checkpoint', 'candidate_status': 'ok', 'candidate': ['submission/a']},
            {'kind': 'checkpoint'}]
    p.write_text('\n'.join(json.dumps(r) for r in rows))
    stats = g._cache_stats(tmp_path)
    assert stats['no_candidate_instant_zero_checkpoints'] == 1
    assert stats['nonempty_candidate_checkpoints'] == 2
    assert stats['unknown_candidate_checkpoints'] == 1
    assert stats['candidate_cache_status_unknown'] == 2
    assert stats['sha256_dedup_cache_hits'] is None and stats['effective_verifier_skip_rate'] is None
    assert stats['grading_elapsed_seconds_observed_total'] == 2.5


def test_g3_does_not_overwrite_human_history(tmp_path):
    sheet = tmp_path / 'human.md'; sheet.write_text('reviewer signed')
    with pytest.raises(FileExistsError):
        g.evaluate_r20_g3(tmp_path, tmp_path, sheet_path=sheet)
    assert sheet.read_text() == 'reviewer signed'


@pytest.mark.parametrize('field,value', [('conversation', {'messages': [{'role': 'user', 'content': 'injected'}]}),
                                         ('pending_calls', [{'name': 'tool'}])])
def test_rejects_child_c0_conversation_and_pending_state(tmp_path, field, value):
    parent, child = pair(tmp_path)
    cp = g._load_json(child / 'checkpoints/cp000000.json'); cp[field] = value
    put(child / 'checkpoints/cp000000.json', cp)
    with pytest.raises(ValueError, match='c0'):
        g._bound_pair(parent, child)


def test_rejects_invalid_evidence_seal_even_matching_pair(tmp_path, monkeypatch):
    parent, child = pair(tmp_path)
    monkeypatch.setattr(g, 'study_outcome', lambda p, **kw: {'study_valid': False})
    with pytest.raises(ValueError, match='seal'):
        g._bound_pair(parent, child)


def test_g4_malformed_request_is_counted_unverified(tmp_path):
    ep = tmp_path / g.COHORT / 'episodes/ep'
    package(ep, 'p')
    req = ep / 'raw/model/r000001/request.json'; req.parent.mkdir(parents=True); req.write_text('{bad')
    result = g.evaluate_r21_g4(tmp_path, tmp_path / 'diag')
    assert result['normal_and_p_requests_scanned'] == result['unverified_request_count'] == 1
    assert result['zero_diagnostic_leakage_verified_on_scanned'] is False
    assert 'malformed_request' in result['request_findings'][0]['reason']


def test_generated_code_overlap_does_not_claim_confirmed_leakage():
    public = [{'role': 'user', 'content': 'public task'}]
    private = ['shape_count >= 5; top <= 1.2; height >= 5.0']
    result = g.audit_request_boundary({'messages': public + [{'role': 'assistant', 'content': private[0]}]},
                                     public_messages=public, private_corpus=private)
    assert result['matches'] and result['status'] == 'unverified'
    assert result['reason'] == 'private_overlap_in_unverified_generated_history'


def test_public_transcript_cannot_whitelist_injected_c0(tmp_path):
    package(tmp_path, 'p')
    put(tmp_path / 'benchmark/public_task.json', {'prompt': 'public task', 'assets': []})
    cfg = g._load_json(tmp_path / 'raw/resolved_config.json')
    cfg['agent']['system_prompt'] = 'public harness'; put(tmp_path / 'raw/resolved_config.json', cfg)
    with pytest.raises(ValueError, match='public transcript'):
        g._public_messages(tmp_path)


def test_g4_excludes_configured_diagnostic_arms(tmp_path):
    ep = tmp_path / 'qwen36_pi_p1_pilot_20261017/v3_g1_cold_resume/arm'
    package(ep, 'intervention')
    cfg = g._load_json(ep / 'raw/resolved_config.json')
    cfg['agent']['intervention'] = {'at_request': 2, 'message': 'private rubric reminder'}
    put(ep / 'raw/resolved_config.json', cfg)
    result = g.evaluate_r21_g4(tmp_path, tmp_path / 'diag')
    assert result['episodes_scanned'] == result['normal_and_p_requests_scanned'] == 0
    assert len(result['excluded_intervention_episodes']) == 1
