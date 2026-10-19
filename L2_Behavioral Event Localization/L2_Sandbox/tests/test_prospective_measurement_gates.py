"""Versioned recovery readiness never retrofits statistical equivalence."""
import json
from pathlib import Path

import pytest

from l2_sandbox.measurement_gates import measurement_gate_binding, verify_measurement_gate_binding
from l2_sandbox.v3_execution import digest, file_digest, seal


@pytest.fixture
def prospective(tmp_path):
    slots = [f"officeval_{t}.r{r:02}" for t in ("003", "004", "010", "033", "050", "060", "070", "085", "090") for r in range(5)]
    refs = {}
    docs = {}

    def put(name, value):
        docs[name] = value
        p = tmp_path / (name + '.json')
        p.write_text(json.dumps(value))
        refs[name] = {"path": p.name, "sha256": file_digest(p)}
        return refs[name]

    configs = {s.split('.')[0]: {"task": s.split('.')[0]} for s in slots}
    put('prospective_freeze', {"schema": "frozen", "bound_files": {
        '/rubrics/' + task + '.json': digest(task) for task in configs}})
    plan = seal({"schema_version": "prospective-dev45-1", "slots": slots, "configs": configs,
                 "prospective_freeze_sha256": refs['prospective_freeze']['sha256']}, 'plan_sha256')
    put('collection_plan', plan)
    records = {s: {"label": s, "task_id": s.split('.')[0], "status": "completed", "study_valid": True,
                   "episode_id": s, "index_file_sha256": digest(s), "manifest_sha256": digest([s]),
                   "config_sha256": digest(configs[s.split('.')[0]]),
                   "usage": {"model_requests_including_parent": 1},
                   "outcome": {"study_outcome_y": 0.0, "rule_version": "v1.1"}} for s in slots}
    put('cohort', seal({"plan_sha256": plan['plan_sha256'], "slots": records}))
    common = {"cohort": refs['cohort'], "collection_plan": refs['collection_plan'], "freeze": refs['prospective_freeze']}
    put('g1', dict(common, schema="prospective_recovery_census_v1", actor_requests=0,
        mechanical_census_complete=True, restore_exact=True, request_exact=True, request_count=45,
        checkpoint_count=45, compaction_count=1, compression_coverage="observed", slots=[
            {"slot": s, "package_index_sha256": records[s]['index_file_sha256'], "episode_id": s,
             "config_sha256": records[s]['config_sha256'], "request_count": 1, "checkpoint_count": 1,
             "checks": [{"checkpoint_id": "cp0", "eligible": True, "restore_exact": True,
                         "request": {"raw_sha256": digest(s), "size_bytes": 10}}],
             "compactions": [{"checkpoint_id": "cp0", "request_status": "applicable", "event_sha256": digest(s)}] if s == slots[0] else []}
            for s in slots]))
    selected = {"batch_id": "b0", "slot": slots[0], "package": '/episodes/' + slots[0], "calls": [{"call": {"call_id": "c0"}}],
                "request_exact": {"raw_sha256": digest('request'), "size_bytes": 10}}
    put('g2_plan', dict(common, schema="prospective_tool_replay_v1", batches=[selected], limits={"actor_http_requests": 0, "replays_per_batch": 2, "tool_calls": 6300,
                                       "wall_seconds": 21600, "workers": 1},
        package_indices={'/episodes/' + s: records[s]['index_file_sha256'] for s in slots}))
    replay = {"status": "complete", "restore_exact": True, "request_exact": selected['request_exact'],
              "cleanup": {"confirmed": True}, "calls": [{"call_id": "c0", "exact": True,
              "unexplained_state_difference": False, "unexplained_tool_difference": False}]}
    put('g2', {"schema": "prospective_tool_replay_result_v1", "plan_sha256": refs['g2_plan']['sha256'],
               "status": "complete", "mechanism_gate_pass": True, "actor_requests": 0, "analyst_requests": 0,
               "not_run": [], "tool_attempts": 1, "runtime_executions": 1, "native_feedback_only": 0,
               "batches": [{"batch_id": "b0", "replays": [replay]}]})
    put('g4', {"schema": "prospective-g4-post-audit-v1", "g4_verified": True, "actor_requests": 0,
               "cohort_sha256": refs['cohort']['sha256'], "freeze_sha256": refs['prospective_freeze']['sha256'],
               "slot_count": 45, "request_count": 45, "container_count": 45,
               "slots": [{"slot": s, "status": "verified", "index_sha256": records[s]['index_file_sha256'],
                          "episode_id": s, "request_count": 1, "container_count": 1} for s in slots]})
    e1rows = []
    vrows = []
    for s in slots:
        row = records[s]
        binding = {"cohort_sha256": refs['cohort']['sha256'], "parent_index_sha256": row['index_file_sha256'],
                   "manifest_sha256": row['manifest_sha256'], "episode_id": s, "config_sha256": row['config_sha256'],
                   "analysis_sha256": digest({'code.py': digest('code')}), "source_sha256": digest('source'), "study_rule_version": "v1.1"}
        e1rows.append({"slot": s, "binding": binding, "source": {"sha256": digest('source')}})
        vrows.append({"slot": s, "binding": binding, "rubric_sha256": digest(s.split('.')[0]), "Y": 0.0, "terminal_submission_score": 0.0,
                      "receipt_sha256": digest('receipt'), "grading_receipt_sha256": digest('grading'),
                      "outputs": {k: digest(k) for k in ('grades.jsonl', 'facts.jsonl', 'defects.json')},
                      "actual_checkpoint_rows": 1})
    put('e1_manifest', {"full_45_ready": True, "completed_count": 45, "actor_requests": 0,
        "cohort": refs['cohort'], "analysis": {"sha256": digest({'code.py': digest('code')}), "files": {'code.py': digest('code')}}, "slots": e1rows,
        "completed": [{"slot": s} for s in slots]})
    put('e1_verification', {"schema": "prospective_e1_independent_verification_v1", "status": "passed",
        "cohort_sha256": refs['cohort']['sha256'], "manifest_sha256": refs['e1_manifest']['sha256'],
        "analysis_sha256": digest({'code.py': digest('code')}), "coverage_slots": 45, "full_45_ready": True,
        "all_terminal_submission_scores_equal_Y": True, "all_output_and_raw_hashes_valid": True,
        "all_checkpoint_flags_revalidated": True, "slots": vrows})
    put('authorization', {"authorization_source": "direct_user_reply", "exact_text": "采用限定合同，披露变异并继续（推荐）",
        "policy": "bounded-mechanism-contract-v1", "allow_declared_runtime_variance": True,
        "allow_unexplained_differences": False, "scope": "does not establish outcome-distribution equivalence"})
    waiver = put('waiver', {"human_review_policy": "waived_by_user", "authorization_source": "direct_user_message",
                           "exact_text": "跳过人审核", "scope": ["G3_human_review"]})
    audit = {"schema": "prospective_measurement_readiness_v1", "policy": "bounded-mechanism-contract-v1",
             "refs": {k: v for k, v in refs.items() if k != 'waiver'}, "human_review_policy": "waived_by_user",
             "human_review_authorization": waiver}
    p = tmp_path / 'audit.json'

    def save():
        p.write_text(json.dumps(audit))
        return p

    def update(name):
        put(name, docs[name])
        audit['refs'][name] = refs[name]
        return save()

    return save(), docs, audit, update, save


def test_complete_prospective_contract_is_scoped_and_frozen(prospective):
    path, *_ = prospective
    binding, errors = measurement_gate_binding(path)
    assert errors == []
    assert binding['passed'] is True
    assert binding['execution_readiness'] is True
    assert binding['statistical_equivalence'] == 'not_established'
    assert binding['outcome_distribution_equivalence'] == 'not_established'
    verify_measurement_gate_binding({'measurement_gate_binding': binding})
    (path.parent / 'g1.json').write_text('{}')
    with pytest.raises(ValueError, match='changed or remains incomplete'):
        verify_measurement_gate_binding({'measurement_gate_binding': binding})


@pytest.mark.parametrize('reference', ['g1', 'g2_plan', 'e1_verification', 'authorization', 'cohort'])
def test_missing_or_changed_references_fail_closed(prospective, reference):
    path, _, audit, _, save = prospective
    del audit['refs'][reference]
    binding, errors = measurement_gate_binding(save())
    assert errors and not binding['passed']


@pytest.mark.parametrize('mutation', ['unexplained', 'not_run', 'cleanup', 'repeat', 'call', 'request'])
def test_g2_underlying_failures_cannot_be_hidden_by_summary_boolean(prospective, mutation):
    path, docs, _, update, _ = prospective
    result = docs['g2']
    replay = result['batches'][0]['replays'][0]
    if mutation == 'unexplained':
        replay['calls'][0]['unexplained_tool_difference'] = True
    elif mutation == 'not_run':
        result['not_run'] = ['b1']
    elif mutation == 'cleanup':
        replay['cleanup']['confirmed'] = False
    elif mutation == 'repeat':
        result['batches'][0]['replays'].pop()
    elif mutation == 'call':
        replay['calls'][0]['call_id'] = 'other'
    else:
        replay['request_exact'] = {}
    binding, errors = measurement_gate_binding(update('g2'))
    assert errors and binding['gates']['V3-G2'] is False and not binding['passed']


@pytest.mark.parametrize('name,mutation', [('g1', 'compression'), ('g1', 'checkpoint'), ('g4', 'index'),
                                          ('e1_verification', 'terminal'), ('e1_verification', 'binding')])
def test_per_slot_and_compression_evidence_required(prospective, name, mutation):
    _, docs, _, update, _ = prospective
    d = docs[name]
    if mutation == 'compression':
        d['compaction_count'] = 0
        d['slots'][0]['compactions'] = []
    elif mutation == 'checkpoint':
        d['slots'][0]['checks'][0]['restore_exact'] = False
    elif mutation == 'index':
        d['slots'][0]['index_sha256'] = 'wrong'
    elif mutation == 'terminal':
        d['slots'][0]['terminal_submission_score'] = 0.5
    else:
        d['slots'][0]['binding']['config_sha256'] = 'wrong'
    binding, errors = measurement_gate_binding(update(name))
    assert errors and not binding['passed']


def test_unknown_prospective_policy_is_not_legacy_fallback(prospective):
    _, _, audit, _, save = prospective
    audit['policy'] = 'unapproved'
    binding, errors = measurement_gate_binding(save())
    assert errors and not binding['passed']


def test_legacy_ci_remains_required(tmp_path):
    path = tmp_path / 'legacy.json'
    path.write_text(json.dumps({
        'V3_G1': {'byte_equivalence_verified': True, 'applicable_configuration_verified': True},
        'V3_G2_R19': {'cp000000_all_pre_first_request_valid': True, 'completed_valid_pairs': 20,
                     'ci95_completely_inside_pm_005': False},
        'V3_G3_R20': {'sampled_checkpoint_count': 20, 'missing_restored_directory_count': 0,
                     'missing_scorer_exit_count': 0, 'human_review_completed_count': 20,
                     'human_conclusions_verified_against_source': True},
        'V3_G4_R21': {'normal_and_p_requests_scanned': 100, 'unverified_request_count': 0,
                     'leaked_request_count': 0, 'zero_diagnostic_leakage_verified_on_scanned': True}}))
    binding, errors = measurement_gate_binding(path)
    assert binding['gates'] == {'V3-G1': True, 'V3-G2': False, 'V3-G3': True, 'V3-G4': True}
    assert errors == ['measurement_gate_not_passed:V3-G2']
    assert 'statistical_equivalence' not in binding


def test_declared_variance_requires_second_diagnostic_replay(prospective):
    from copy import deepcopy

    _, docs, _, update, _ = prospective
    result = docs['g2']
    replay = result['batches'][0]['replays'][0]
    replay['calls'][0].update(exact=False, zip_timestamp_variance=['document.docx'])
    binding, errors = measurement_gate_binding(update('g2'))
    assert errors and not binding['gates']['V3-G2']
    result['batches'][0]['replays'].append(deepcopy(replay))
    result.update(tool_attempts=2, runtime_executions=2)
    binding, errors = measurement_gate_binding(update('g2'))
    assert errors == [] and binding['passed']


@pytest.mark.parametrize('mutation', ['tool_total', 'runtime_total', 'negative_feedback', 'extra_exact_replay'])
def test_attempt_counts_and_exact_short_circuit_are_enforced(prospective, mutation):
    from copy import deepcopy

    _, docs, _, update, _ = prospective
    result = docs['g2']
    if mutation == 'tool_total':
        result['tool_attempts'] += 1
    elif mutation == 'runtime_total':
        result['runtime_executions'] += 1
    elif mutation == 'negative_feedback':
        result['native_feedback_only'] = -1
    else:
        result['batches'][0]['replays'].append(deepcopy(result['batches'][0]['replays'][0]))
        result.update(tool_attempts=2, runtime_executions=2)
    binding, errors = measurement_gate_binding(update('g2'))
    assert errors and not binding['passed']


@pytest.mark.parametrize('key,value', [('tool_calls', 6301), ('actor_http_requests', 1), ('workers', 2),
                                      ('replays_per_batch', 3), ('wall_seconds', 21601)])
def test_g2_frozen_caps_cannot_expand(prospective, key, value):
    _, docs, audit, update, _ = prospective
    docs['g2_plan']['limits'][key] = value
    update('g2_plan')
    docs['g2']['plan_sha256'] = audit['refs']['g2_plan']['sha256']
    binding, errors = measurement_gate_binding(update('g2'))
    assert errors and not binding['gates']['V3-G2']


def _literal_ls_reclassification(prospective):
    from copy import deepcopy
    import hashlib

    path, docs, audit, update, save = prospective
    docs['g2_plan']['batches'][0]['calls'][0]['call'].update(
        name='bash', arguments={'command': 'ls -la /workspace/input/'})
    update('g2_plan')
    result = docs['g2']
    result['plan_sha256'] = audit['refs']['g2_plan']['sha256']
    before = {'call_id': 'c0', 'exact': False, 'classification': 'unexplained',
              'ls_metadata_variance': False, 'unexplained_tool_difference': True,
              'after_state_exact': True, 'unexplained_state_difference': False,
              'zip_timestamp_variance': [], 'state_diff': [],
              'expected_after_sha256': digest('state'), 'actual_after_sha256': digest('state'),
              'observation_exact': False, 'raw_streams_exact': False,
              'delivered_observation_exact': False, 'image_exact': True}
    first = result['batches'][0]['replays'][0]
    first['calls'] = [before]
    result['batches'][0]['replays'].append(deepcopy(first))
    result.update(tool_attempts=2, runtime_executions=2)
    changes = []
    for n in (1, 2):
        directory = path.parent / f'b0_replay{n}/call_0000'
        raw_before = json.dumps(before, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode() + b'\n'
        original = {'path': str(directory / 'comparison.json'),
                    'sha256': hashlib.sha256(raw_before).hexdigest()}
        after = dict(before, classification='declared_runtime_variance',
                     ls_metadata_variance=True, unexplained_tool_difference=False)
        raw = {name: {'path': str(directory / name), 'sha256': digest(name)}
               for name in ('result.json', 'stdout.bin', 'stderr.bin', 'native_validation.json',
                            'delivered_observation.json')}
        raw['comparison.json'] = original
        changes.append({'batch_id': 'b0', 'call_id': 'c0', 'original_comparison': original,
                        'before': deepcopy(before), 'after': after, 'raw_files': raw,
                        'reason': 'Already-authorized ls UID/GID/timestamp diagnosis; input directory added to literal command allowlist'})
    result['reclassifications'] = changes
    docs['g2_reclassification'] = {'original_receipts_unchanged': True, 'changes': deepcopy(changes)}
    update('g2_reclassification')
    return update('g2')


def test_two_literal_ls_receipts_can_use_bound_independent_reclassification(prospective):
    path = _literal_ls_reclassification(prospective)
    binding, errors = measurement_gate_binding(path)
    assert errors == [] and binding['passed']
    assert 'g2_reclassification' in binding['refs']


@pytest.mark.parametrize('mutation', ['missing_sidecar', 'tampered_sidecar', 'state', 'unknown_command',
                                      'original_hash', 'duplicate', 'raw_hash', 'before', 'replay_path'])
def test_reclassification_cannot_override_unrelated_or_unbound_evidence(prospective, mutation):
    _literal_ls_reclassification(prospective)
    _, docs, audit, update, save = prospective
    change = docs['g2']['reclassifications'][0]
    if mutation == 'missing_sidecar':
        del audit['refs']['g2_reclassification']
        path = save()
    elif mutation == 'tampered_sidecar':
        docs['g2_reclassification']['original_receipts_unchanged'] = False
        path = update('g2_reclassification')
    elif mutation == 'unknown_command':
        docs['g2_plan']['batches'][0]['calls'][0]['call']['arguments']['command'] = 'ls -la /secret/'
        update('g2_plan')
        docs['g2']['plan_sha256'] = audit['refs']['g2_plan']['sha256']
        path = update('g2')
    else:
        if mutation == 'state':
            change['after']['actual_after_sha256'] = digest('changed')
        elif mutation == 'original_hash':
            change['original_comparison']['sha256'] = digest('wrong')
        elif mutation == 'duplicate':
            docs['g2']['reclassifications'].append(change)
        elif mutation == 'raw_hash':
            change['raw_files']['stdout.bin']['sha256'] = ''
        elif mutation == 'before':
            change['before']['observation_exact'] = True
        else:
            change['original_comparison']['path'] = change['original_comparison']['path'].replace('replay1', 'replay9')
        from copy import deepcopy
        docs['g2_reclassification']['changes'] = deepcopy(docs['g2']['reclassifications'])
        update('g2_reclassification')
        path = update('g2')
    binding, errors = measurement_gate_binding(path)
    assert errors and not binding['passed']
