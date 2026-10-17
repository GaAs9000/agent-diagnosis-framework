"""Report regressions use fixtures and existing evidence only, without graders."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_human_reviews_require_exact_identity_and_reject_legacy_template():
    mod = load('build_v3_p0a_audit')
    row = dict(audit_id='V3-G3-01', run_label='run', checkpoint_id='cp1', snapshot_id='s2', human_review_verdict='')
    old = dict(row, human_review_verdict='PASS (in sNNNNNN)', reviewer='')
    mod.preserve_human_reviews([row], {'v3_g3_20_checkpoint_candidate_audit_dossier': [old]}, '')
    assert row['human_review_verdict'] == ''
    md = '| `V3-G3-01` | `run` | `cp1` (`s2`) | 1 | [] | False | False | 0 | Ada | 2026-10-17 | FAIL | wrong candidate |'
    mod.preserve_human_reviews([row], {}, md)
    assert (row['reviewer'], row['human_review_verdict']) == ('Ada', 'FAIL')
    fresh = dict(row, snapshot_id='s3', human_review_verdict='')
    mod.preserve_human_reviews([fresh], {'v3_g3_20_checkpoint_candidate_audit_dossier': [row]}, md)
    assert fresh['human_review_verdict'] == ''


def test_classification_sample_is_retained_verbatim():
    mod = load('build_v3_p0a_audit')
    section = '### 2.2 Human sample\n\n| 1 | Ada | corrected |\n'
    md = section + '\n---\n\n## 3. Other\n'
    assert mod.classification_review_section({}, md) == section.rstrip()


def test_sokoban_first_corner_is_not_turn9_and_formats_share_facts(tmp_path):
    mod = load('run_v3_stage_e_swe_and_sokoban')
    obs = tmp_path / 'observations'
    obs.mkdir()
    for turn, stdout in [(9, '####\n# $ #\n#   #\n#####'),
                         (10, 'message=level 1 reset to its starting layout.'),
                         (15, '#####\n#$  #\n#   #\n#####')]:
        (obs / f'r{turn-1:06d}_c0.json').write_text(json.dumps({'stdout': stdout}))
    facts = mod._detect_deadlock_turns(tmp_path)
    assert facts['first_corner_deadlock_turn_overall'] == 15
    assert facts['first_corner_deadlock_turn_pre_reset'] is None
    ep = dict(episode_id='run_seed42', k_star=0, e1_class='never_succeeded_on_disk', broken_count=0,
              independent_state_damage='corner_deadlock_detected', first_deadlock_turn_overall=15,
              first_deadlock_turn_pre_reset=None, first_deadlock_turn_post_reset=15,
              reset_turns=[10], all_corner_deadlock_turns=[15])
    payload, md = mod.render_report({'episodes': [ep]}, {})
    assert payload['lhtb_sokoban']['episodes'][0] == ep
    assert '| `run_seed42` | `0` | `15` | `None` | `[10]` | `15` | `[15]` |' in md
    assert '第 9 回合' not in md
    assert '不能覆盖全部' in md


def test_r6_recursive_inventory_records_skips_errors_and_read_hashes(tmp_path, monkeypatch):
    mod = load('v3_r6_study_version_audit')
    roots = [tmp_path / 'batch/runs/run/attempt/ep', tmp_path / 'batch/incomplete', tmp_path / 'batch/bad']
    for p in roots:
        p.mkdir(parents=True)
        (p / 'manifest.json').write_text(json.dumps({'recording_mode': 'live'}))
    (roots[0] / 'official_grade.json').write_text('{}')
    (roots[2] / 'manifest.json').write_text('{')
    def both(pkg):
        return {'v1.0': {'study_valid': False}, 'v1.1': {'study_valid': True, 'study_outcome_y': 0, 'study_classification': 'valid_zero'}, 'changed': True}
    monkeypatch.setattr(mod.study, 'study_outcome_by_version', both)
    payload = mod.build_audit(tmp_path)
    assert payload['coverage'] == dict(discovered_manifests=3, processed=1, skipped=1, errors=1)
    assert payload['affected_packages'][0]['path'] == 'batch/runs/run/attempt/ep'
    assert payload['inputs_unchanged_during_read']
    assert payload['source_sha256']
    assert (roots[0] / 'official_grade.json').read_text() == '{}'
    assert 'incomplete_missing_official_grade' in mod.render_markdown(payload)
