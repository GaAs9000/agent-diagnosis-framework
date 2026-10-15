"""Paid evidence survives analysis changes; only request changes authorize cache misses."""
from pathlib import Path
import pytest
from conftest import make_l1_episode
from l2_sandbox import report
from l2_sandbox.curve import build_curve_plan, execute_curve
from l2_sandbox.evidence.l1 import L1Package
from l2_sandbox.io import L2Error, digest, read_json, read_jsonl, sha_file, write_json, write_jsonl
from test_v03_analyst_regressions import fake_curve, fake_score


def test_analysis_and_outcome_change_reuses_legacy_paid_packages(tmp_path, monkeypatch):
    parent = L1Package(make_l1_episode(tmp_path/'parent', num_turns=4, terminal_case='failed_zero'))
    out = tmp_path/'out'
    calls = []
    def runner(p, k, path):
        calls.append(path)
        make_l1_episode(path, num_turns=4, resume_from_k=k, parent_pkg=p, terminal_case='scored', completion_rate=.5)
    plan = build_curve_plan(parent, n1=2, n2=2, stride=2, max_concurrency=1)
    execute_curve(parent, out, plan, resume_fn=runner)
    original_count = len(calls)
    package_hashes = {path: sha_file(path) for path in (out/'branches').rglob('*') if path.is_file()}
    legacy = read_json(out/'plan.json')
    legacy['input_sha256']['implementation_sha256'] = 'old-code'
    legacy['input_sha256']['parent_content_sha256'] = 'old-whole-inventory'
    legacy['code_version'] = 'old-version'
    write_json(out/'plan.json', legacy)
    # Deliberately stale derived rows must not reject intact, already-paid evidence.
    write_jsonl(out/'outcomes.jsonl', [{**r, 'valid': False, 'y': None} for r in read_jsonl(out/'outcomes.jsonl')])
    original_outcome = L1Package.outcome
    def new_policy(self):
        row = original_outcome(self)
        return {**row, 'y': .6} if self.manifest.get('lineage') else row
    monkeypatch.setattr(L1Package, 'outcome', new_policy)
    changed = build_curve_plan(parent, n1=2, n2=2, stride=2, alpha=.1, max_concurrency=3, cost_per_branch_usd=2)
    result = execute_curve(parent, out, changed, resume_fn=runner, cache_only=True)
    assert len(calls) == original_count
    assert all(r['y'] == .6 for r in read_jsonl(out/'outcomes.jsonl') if r['selected'])
    assert package_hashes == {path: sha_file(path) for path in (out/'branches').rglob('*') if path.is_file()}
    assert result['invalid_branches'] == 0


def test_cache_only_missing_selected_sample_never_resumes(tmp_path):
    parent = L1Package(make_l1_episode(tmp_path/'parent', num_turns=2))
    plan = build_curve_plan(parent, n1=1, n2=1, stride=2)
    def never(*args):
        pytest.fail('cache-only may not send')
    with pytest.raises(L2Error, match='missing sample'):
        execute_curve(parent, tmp_path/'out', plan, resume_fn=never, cache_only=True)


def test_llm_actual_request_cache_and_local_recompute(tmp_path, monkeypatch):
    monkeypatch.setattr(report, 'execute_curve', fake_curve)
    monkeypatch.setattr(report, 'score_episode', fake_score)
    parent = make_l1_episode(tmp_path/'parent', num_turns=2)
    cfg = tmp_path/'cfg.json'
    write_json(cfg, {'model': {'parameters': {'temperature': .1}}})
    calls, local = [], []
    real_rules = report.predict_values_rules
    def rules(*args, **kwargs):
        local.append(1)
        return real_rules(*args, **kwargs)
    monkeypatch.setattr(report, 'predict_values_rules', rules)
    def caller(*args):
        calls.append(args)
        return {'values': [.4, .3]}
    def run(**kwargs):
        return report.run_episode(parent, p0=kwargs.pop('p0', .5), output_dir=tmp_path/'out', execute=True,
                                  analysts=(f'llm:{cfg}',), llm_caller=caller, **kwargs)
    run()
    # Unrelated code metadata and execution/accounting config are not model inputs.
    for path in (tmp_path/'out/values').glob('llm*'):
        doc = read_json(path)
        write_json(path, {**doc, 'code_version': 'past', 'code_fingerprint': 'unrelated'})
    write_json(cfg, {'curve': {'alpha': .01}, 'model': {'parameters': {'temperature': .1},
                'timeout_seconds': 99, 'cost_per_call_usd': 123, 'api_key_env': 'OTHER_KEY'}})
    run(alpha=.1)
    assert len(calls) == 1 and len(local) == 2
    run(p0=.7)
    assert len(calls) == 2
    write_json(cfg, {'model': {'parameters': {'temperature': .9}}})
    run(p0=.7)
    assert len(calls) == 3
    from l2_sandbox.analysts import llm
    changed_prompt = report.CALL1_SYSTEM_PROMPT+' changed'
    monkeypatch.setattr(report, 'CALL1_SYSTEM_PROMPT', changed_prompt)
    monkeypatch.setattr(llm, 'CALL1_SYSTEM_PROMPT', changed_prompt)
    run(p0=.7)
    assert len(calls) == 4
    write_json(cfg, {'model': {'model': 'different-model', 'parameters': {'temperature': .9}}})
    run(p0=.7)
    assert len(calls) == 5
    assert not (tmp_path/'out/history').exists()


def test_programmatic_p0_is_required(tmp_path):
    with pytest.raises(TypeError, match='p0'):
        report.run_episode(tmp_path/'absent')


def test_invalid_attempt_bound_is_derived_from_five_percent(tmp_path):
    parent = L1Package(make_l1_episode(tmp_path/'parent', num_turns=4, terminal_case='failed_zero'))
    plan = build_curve_plan(parent, n1=4, n2=4, stride=2, max_concurrency=1)
    # Mmax=8+12=20; one failure is recoverable (1/21), two are not (2/22).
    calls = []
    def broken(p, k, path):
        calls.append(path.name)
        make_l1_episode(path, num_turns=4, resume_from_k=k, parent_pkg=p, terminal_case='execution_invalid')
    curve = execute_curve(parent, tmp_path/'out', plan, resume_fn=broken)
    assert len(calls) == curve['invalid_branches'] == 2
    assert curve['excluded_high_invalid_rate']
    assert all(path.exists() for path in (tmp_path/'out/branches').iterdir())


def test_old_tag_llm_cache_migrates_without_new_call(tmp_path, monkeypatch):
    monkeypatch.setattr(report, 'execute_curve', fake_curve)
    monkeypatch.setattr(report, 'score_episode', fake_score)
    parent = make_l1_episode(tmp_path/'parent', num_turns=2)
    output = tmp_path/'out'
    calls = []
    def caller(*args):
        calls.append(args)
        return {'values': [.4, .3]}
    kwargs = dict(p0=.5, output_dir=output, execute=True, analysts=('llm',), llm_caller=caller)
    report.run_episode(parent, **kwargs)
    cfg = report.load_config()
    old_hash = digest({'c1': report.CALL1_SYSTEM_PROMPT, 'c2': report.CALL2_SYSTEM_PROMPT,
                       'mode': 'facts', 'model_config': cfg['model']})[:16]
    old_id = f"llm:{cfg['model']['model']}@{old_hash}"
    old_spec = {'kind': 'llm', 'id': old_id, 'config': cfg, 'include_facts': True}
    current = next((output/'values').glob('llm*'))
    doc = read_json(current)
    old_binding = {**doc['input_sha256'], 'analyst_config_sha256': digest(old_spec),
                   'implementation_sha256': 'old', 'facts_sha256': 'old-facts', 'parent_index_sha256': 'old-index'}
    old_binding.pop('model_sha256')
    write_json(output/'values'/report._safe_filename(old_id),
               {**doc, 'analyst': old_id, 'input_sha256': old_binding, 'code_version': 'old', 'code_fingerprint': 'old'})
    current.unlink()
    report.run_episode(parent, cache_only=True, **kwargs)
    assert len(calls) == 1
    new_doc = read_json(current)
    assert new_doc['input_sha256'] == doc['input_sha256']
    assert new_doc['analyst'] == doc['analyst']


@pytest.mark.parametrize('name', ['values', 'hypotheses', 'branches'])
def test_owned_output_directory_cannot_redirect_into_parent(tmp_path, name):
    parent = make_l1_episode(tmp_path/'parent', num_turns=2)
    output = tmp_path/'out'
    output.mkdir()
    (output/name).symlink_to(parent, target_is_directory=True)
    before = {path: sha_file(path) for path in parent.rglob('*') if path.is_file()}
    with pytest.raises(L2Error, match='owned output directory'):
        report.run_episode(parent, p0=.5, output_dir=output)
    assert before == {path: sha_file(path) for path in parent.rglob('*') if path.is_file()}
    assert not (output/'plan.json').exists()


def test_old_unselected_valid_retries_cannot_hide_incomplete_curve(tmp_path):
    parent = L1Package(make_l1_episode(tmp_path/'parent', num_turns=2, terminal_case='failed_zero'))
    out = tmp_path/'out'
    plan = build_curve_plan(parent, n1=1, n2=1, stride=2, max_concurrency=1)
    write_json(out/'plan.json', plan)
    for attempt in range(21):
        name = 'k000-s1-i00' + (f'-r{attempt}' if attempt else '')
        make_l1_episode(out/'branches'/name, num_turns=2, resume_from_k=0, parent_pkg=parent,
                        terminal_case='execution_invalid' if attempt == 0 else 'scored', completion_rate=.5)
    def never(*args):
        pytest.fail('D2 must stop new admissions after irrecoverable selected failures')
    curve = execute_curve(parent, out, plan, resume_fn=never)
    assert curve['total_attempted_branches'] == 21
    assert curve['invalid_branches'] == 1
    assert curve['selected_attempted_branches'] == 2
    assert curve['selected_invalid_branches'] == 1
    assert curve['invalid_rate'] == .5
    assert curve['excluded_incomplete_samples'] is True
    assert curve['excluded_high_invalid_rate'] is True
    assert curve['sampling_status'] == 'incomplete'
    assert curve['verdict'] is None
    assert curve['v_hat'] == curve['v2'] == curve['turn_drops'] == {}
    assert 'thresholds' not in curve
