"""Immutable experiment plans; supports unchanged cold resume and two-stage value curves.

Information injection is deliberately blocked against L1's unchanged-resume
contract. A text plan is not an implemented intervention, and a historical retry
is not accepted as a prospective randomized control.
"""
from __future__ import annotations
import copy
from pathlib import Path
from .. import SCHEMA_VERSION
from ..io import (L2Error, IntegrityError, digest, sha_file, read_json, safe, now, write_json, write_new, inventory)
from ..stats.intervals import STATS_POLICY_VERSION
from ..stats.strata import STRATA_TAGS

TEMPLATES = {
    'resume_unchanged': {'implemented': True, 'effect_contrast': False, 'description': 'Fresh continuation of the same checkpoint under inherited policy/budget'},
    'value_curve': {'implemented': True, 'effect_contrast': True, 'description': 'Two-stage sparse value-curve localization and E1+ same-state stratification'},
    'provide_public_state_summary': {'implemented': False, 'effect_contrast': True, 'description': 'Requires an audited L1 experiment-injection API; not a fabricated past Observation'},
    'generic_reminder': {'implemented': False, 'effect_contrast': True, 'description': 'Requires explicit injected-message provenance and matched-cost protocol'},
}
ENDPOINTS = ('submission_present', 'official_dim1', 'official_completion_rate', 'passed', 'official_normalized_score')


def compile_plan(analysis: Path, source: Path, checkpoint: str, output: Path, *, hypothesis_id=None,
                 template='resume_unchanged', replicates=3, endpoint='submission_present'):
    from ..package import verify, seal
    from ..evidence.l1 import L1Reader
    if template == 'value_curve':
        return compile_curve_plan(
            analysis, source, output,
            stage1_replicates=replicates,
            endpoint=endpoint if endpoint != 'submission_present' else 'official_completion_rate',
        )
    if template not in TEMPLATES or endpoint not in ENDPOINTS: raise L2Error('Unknown intervention template or endpoint')
    if type(replicates) is not int or not 1 <= replicates <= 64: raise L2Error('replicates must be integer 1..64')
    if not __import__('re').fullmatch(r'cp[0-9]{6}', checkpoint): raise L2Error('Explicit checkpoint cpXXXXXX required')
    av = verify(analysis)
    if not av['verified'] or not av['status'].startswith('completed'): raise L2Error('A verified accepted diagnosis is required')
    am = read_json(analysis / 'manifest.json'); r = L1Reader(source, expected_index=am['source_index_sha256'])
    if r.version != '0.2.0': raise L2Error('Legacy source has no qualified live experiment bridge')
    diag = read_json(analysis / 'diagnosis.json'); hyp = None
    if hypothesis_id:
        hyp = next((h for h in diag['hypotheses'] if h['hypothesis_id'] == hypothesis_id), None)
        if hyp is None: raise L2Error('Unknown hypothesis ID')
    if template != 'resume_unchanged' and hyp is None: raise L2Error('A mechanism experiment requires a frozen hypothesis')
    cp_path = f'checkpoints/{checkpoint}.json'; cp = r.json(cp_path)
    cfg = r.json('raw/resolved_config.json'); remaining = {
        'model_requests': cfg['budget']['max_model_requests'] - cp['budget_used']['model_requests'],
        'tool_calls': cfg['budget']['max_tool_calls'] - cp['budget_used']['tool_calls'],
        'wall_seconds': cfg['budget']['wall_time_seconds'] - cp['budget_used']['agent_seconds']}
    blockers = []
    if not cp['qualification']['eligible']: blockers += ['source_checkpoint_ineligible']
    if any(v <= 0 for v in remaining.values()): blockers += ['checkpoint_budget_exhausted']
    if not TEMPLATES[template]['implemented']: blockers += ['L1_unchanged_resume_does_not_support_explicit_information_injection']
    if r.manifest.get('recording_mode') != 'live': blockers += ['source_not_live_recording']
    # No final grading validity gate: a failed task may have a qualified checkpoint.
    arms = [{'arm_id': 'control', 'template': 'resume_unchanged', 'replicates': replicates}]
    if template != 'resume_unchanged': arms.append({'arm_id': 'treatment', 'template': template, 'replicates': replicates})
    plan = {'schema_version': SCHEMA_VERSION, 'package_kind': 'experiment_plan', 'created_at': now(),
            'analysis_id': am['analysis_id'], 'analysis_index_sha256': sha_file(analysis / 'evidence_index.json'),
            'prediction_commitment_sha256': sha_file(analysis / 'predictions/frozen_manifest.json'),
            'source_index_sha256': r.index_sha256, 'source_episode_id': r.episode_id,
            'checkpoint_id': checkpoint, 'checkpoint_sha256': r.files[cp_path]['sha256'],
            'checkpoint_snapshot_id': cp['snapshot_id'], 'config_sha256': digest(cfg), 'hypothesis_id': hypothesis_id,
            'hypothesis_sha256': digest(hyp), 'template': template, 'arms': arms, 'endpoint': endpoint,
            'assignment': 'fixed_predeclared_alternating_arms_no_adaptive_stopping', 'replicates_per_arm': replicates,
            'remaining_budget_per_branch': remaining, 'budget_policy': 'inherit_parent_budget_no_reset',
            'failure_policy': 'retain_all_attempts; endpoint undefined on infrastructure_invalid; report undefined count; no causal verdict if missing outcomes',
            'plan_state': 'blocked' if blockers else 'frozen', 'blockers': blockers, 'hypothesis_verdict': 'untested',
            'execution_authorized': False, 'contrast_identifiable': len(arms) > 1 and not blockers,
            'private_rubric_injection': False, 'trusted_external_timestamp': False}
    output = Path(output).absolute()
    for other in (analysis, source):
        if output.resolve().is_relative_to(Path(other).resolve()) or Path(other).resolve().is_relative_to(output.resolve()):
            raise L2Error('Plan output must be separate from source and diagnosis')
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    write_json(output, 'plan.json', plan); write_json(output, 'hypothesis.json', hyp)
    write_new(output, 'prediction_commitment.json', (analysis / 'predictions/frozen_manifest.json').read_bytes())
    write_new(output, 'analysis_index.json', (analysis / 'evidence_index.json').read_bytes())
    write_new(output, 'source_index.json', r.read('evidence_index.json'))
    write_json(output, 'checkpoint_qualification.json', {'checkpoint_id': checkpoint, 'qualification': cp['qualification'], 'budget_used': cp['budget_used']})
    write_new(output, 'plan.md', (
        '# L2 冻结实验计划\n\n状态：' + plan['plan_state'] + '。模板：' + template + '。\n\n'
        '这是计划，不是实验结果；调用 analyze/plan 不会付费调用被测模型。\n\n'
        '仅 resume_unchanged 有内置执行桥；它能建立新基线续跑，但不能单独识别干预效应。\n\n'
        '阻塞项：' + str(blockers) + '\n').encode())
    index = seal(output, 'experiment_plan')
    return {'output': str(output), 'index_sha256': index, 'plan': plan}


def compile_curve_plan(
    analysis: Path,
    source: Path,
    output: Path,
    *,
    stride: int = 4,
    stage1_replicates: int = 12,
    stage2_replicates: int = 24,
    alpha: float = 0.05,
    endpoint: str | None = None,
    max_concurrency: int = 4,
    max_replacements: int = 2,
    require_value_predictions: bool = False,
    strata_kinds: list[str] | None = None,
    min_per_class: int = 4,
    baseline_rate: float = 0.5,
) -> dict:
    """Compiles a frozen v0.2 value_curve_plan package (Section 6.2)."""
    from ..package import verify, seal
    from ..evidence.l1 import L1Reader
    from ..facts.build import extract_turn_tags
    from ..analyst.baselines import compute_all_baselines, extract_terminal_outcome

    if type(stride) is not int or stride < 1: raise L2Error('stride must be positive integer')
    if type(stage1_replicates) is not int or not 1 <= stage1_replicates <= 64: raise L2Error('stage1_replicates must be 1..64')
    if type(stage2_replicates) is not int or not 1 <= stage2_replicates <= 64: raise L2Error('stage2_replicates must be 1..64')
    if not 0.0 < alpha < 0.5: raise L2Error('alpha must be in (0, 0.5)')
    if type(max_concurrency) is not int or not 1 <= max_concurrency <= 16: raise L2Error('max_concurrency must be 1..16')
    if type(max_replacements) is not int or not 0 <= max_replacements <= 8: raise L2Error('max_replacements must be 0..8')

    av = verify(analysis)
    if not av['verified'] or not av['status'].startswith('completed'):
        raise L2Error('A verified accepted diagnosis is required')
    am = read_json(analysis / 'manifest.json')
    r = L1Reader(source, expected_index=am['source_index_sha256'])
    if r.version != '0.2.0':
        raise L2Error('Legacy source has no qualified live value-curve bridge')
    diag = read_json(analysis / 'diagnosis.json')
    cfg = r.json('raw/resolved_config.json')
    bench = (r.manifest.get('task') or {}).get('benchmark', 'officeval')
    resolved_endpoint = endpoint or ('passed' if bench == 'lhtb' else 'official_completion_rate')
    if resolved_endpoint not in ('official_completion_rate', 'official_dim1', 'passed', 'submission_present'):
        raise L2Error('Unsupported value_curve endpoint: ' + str(resolved_endpoint))

    cp_index = r.json('checkpoints/index.json')['checkpoints']
    cp_map = {item['checkpoint_id']: r.json(item['path']) for item in cp_index}
    num_turns = max(1, int(r.manifest.get('counts', {}).get('model_requests', len(cp_index) - 1)))
    grid_turns = [k for k in range(0, num_turns, stride)]
    grid_cps = [f'cp{k:06d}' for k in grid_turns]

    blockers = []
    if r.manifest.get('recording_mode') != 'live':
        blockers.append('source_not_live_recording')

    temp = (cfg.get('model', {}).get('parameters') or {}).get('temperature')
    if isinstance(temp, (int, float)) and not isinstance(temp, bool) and float(temp) <= 0.05:
        blockers.append('degenerate_policy_sampling')

    if require_value_predictions and not diag.get('value_predictions'):
        blockers.append('value_predictions_missing')

    # Verify all checkpoints in the episode up to num_turns-1 that might be used in stage1 or stage2
    grid_qualifications = {}
    for cp_id in grid_cps:
        cp = cp_map.get(cp_id)
        if cp is None:
            blockers.append('grid_checkpoint_ineligible')
            continue
        rem = {
            'model_requests': cfg['budget']['max_model_requests'] - cp['budget_used']['model_requests'],
            'tool_calls': cfg['budget']['max_tool_calls'] - cp['budget_used']['tool_calls'],
            'wall_seconds': cfg['budget']['wall_time_seconds'] - cp['budget_used']['agent_seconds'],
        }
        eligible = bool(cp['qualification'].get('eligible')) and all(v > 0 for v in rem.values()) and cp['config_sha256'] == digest(cfg)
        if not eligible and 'grid_checkpoint_ineligible' not in blockers:
            blockers.append('grid_checkpoint_ineligible')
        grid_qualifications[cp_id] = {
            'checkpoint_id': cp_id,
            'checkpoint_sha256': r.files[f'checkpoints/{cp_id}.json']['sha256'],
            'snapshot_id': cp['snapshot_id'],
            'eligible': eligible,
            'remaining_budget': rem,
            'budget_used': cp['budget_used'],
        }

    all_cp_digests = {
        item['checkpoint_id']: r.files[item['path']]['sha256']
        for item in cp_index
        if item.get('eligible') and int(item['checkpoint_id'][2:]) < num_turns
    }

    kinds = list(strata_kinds or STRATA_TAGS)
    if any(k not in STRATA_TAGS for k in kinds):
        raise L2Error('Unknown strata tag kind')

    turn_tags = extract_turn_tags(r)
    all_candidate_cps = [f'cp{k:06d}' for k in range(num_turns) if f'cp{k:06d}' in cp_map]
    baselines = compute_all_baselines(
        r, turn_tags, all_candidate_cps, baseline_rate=baseline_rate, endpoint=resolved_endpoint
    )
    terminal_y = extract_terminal_outcome(r, endpoint=resolved_endpoint)

    plan = {
        'schema_version': SCHEMA_VERSION,
        'package_kind': 'value_curve_plan',
        'created_at': now(),
        'template': 'value_curve',
        'analysis_id': am['analysis_id'],
        'analysis_index_sha256': sha_file(analysis / 'evidence_index.json'),
        'prediction_commitment_sha256': sha_file(analysis / 'predictions/frozen_manifest.json'),
        'source_index_sha256': r.index_sha256,
        'source_episode_id': r.episode_id,
        'config_sha256': digest(cfg),
        'num_turns': num_turns,
        'terminal_value': terminal_y,
        'endpoint': {'name': resolved_endpoint, 'source': 'l1_projection'},
        'grid': {
            'rule': 'stride',
            'stride': stride,
            'checkpoints': grid_cps,
            'terminal': 'exact_parent_grade',
        },
        'qualified_checkpoint_sha256': all_cp_digests,
        'stage1': {'replicates': stage1_replicates},
        'stage2': {'replicates': stage2_replicates, 'rule': 'argmax_interval_drop_earliest_tie'},
        'inference': {
            'interval_null': 'uniform_share',
            'alpha': alpha,
            'turn_test': 'newcombe_one_sided_bonferroni',
            'delta_turn': 'max(0.10, 2*V0/T)',
        },
        'schedule': {
            'order': 'block_randomized_per_round',
            'seed_from': 'plan_digest',
            'max_concurrency': max_concurrency,
        },
        'failure_policy': {
            'agent_termination': 'valid_outcome',
            'empty_submission': 'score_zero',
            'unscorable_submission': 'score_zero_sensitivity_missing',
            'infrastructure_invalid': 'missing',
            'max_replacements': max_replacements,
            'report': 'bounds_missing_as_0_and_1',
            'policy_identity_check': 'first_request_bytes_equal',
        },
        'strata': {
            'unit': 'first_resampled_turn',
            'min_per_class': min_per_class,
            'multiplicity': 'holm',
            'kinds': kinds,
        },
        'require_value_predictions': require_value_predictions,
        'stats_policy_version': STATS_POLICY_VERSION,
        'plan_state': 'blocked' if blockers else 'frozen',
        'blockers': blockers,
        'execution_authorized': False,
        'private_rubric_injection': False,
    }

    output = Path(output).absolute()
    for other in (analysis, source):
        if output.resolve().is_relative_to(Path(other).resolve()) or Path(other).resolve().is_relative_to(output.resolve()):
            raise L2Error('Plan output must be separate from source and diagnosis')
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    write_json(output, 'plan.json', plan)
    write_json(output, 'baselines.json', baselines)
    write_json(output, 'grid_qualifications.json', grid_qualifications)
    write_new(output, 'prediction_commitment.json', (analysis / 'predictions/frozen_manifest.json').read_bytes())
    write_new(output, 'analysis_index.json', (analysis / 'evidence_index.json').read_bytes())
    write_new(output, 'source_index.json', r.read('evidence_index.json'))
    write_new(output, 'plan.md', (
        f"# L2 冻结两阶段价值曲线计划\n\n状态：{plan['plan_state']}。网格：{grid_cps}，N1={stage1_replicates}，N2={stage2_replicates}。\n\n"
        f"阻塞项：{blockers}\n"
    ).encode('utf-8'))
    index = seal(output, 'value_curve_plan')
    return {'output': str(output), 'index_sha256': index, 'plan': plan}


def verify_experiment_package(root, idx, index_hash):
    kind = idx['package_kind']
    try:
        if kind == 'experiment_plan':
            plan = read_json(root / 'plan.json')
            if plan['package_kind'] != kind or plan['template'] not in TEMPLATES: raise IntegrityError('Invalid plan kind/template')
            if sha_file(root / 'source_index.json') != plan['source_index_sha256'] or sha_file(root / 'analysis_index.json') != plan['analysis_index_sha256']:
                raise IntegrityError('Plan source anchors differ')
            if sha_file(root / 'prediction_commitment.json') != plan['prediction_commitment_sha256'] or digest(read_json(root / 'hypothesis.json')) != plan['hypothesis_sha256']:
                raise IntegrityError('Frozen prediction binding differs')
            if plan['execution_authorized'] or plan['hypothesis_verdict'] != 'untested': raise IntegrityError('Frozen plan cannot assert authorized execution or causal support')
            if not TEMPLATES[plan['template']]['implemented'] and plan['plan_state'] != 'blocked': raise IntegrityError('Unsupported intervention must remain blocked')
            if plan['plan_state'] == 'frozen':
                q = read_json(root / 'checkpoint_qualification.json')
                if plan['blockers'] or not q['qualification']['eligible'] or any(v <= 0 for v in plan['remaining_budget_per_branch'].values()):
                    raise IntegrityError('Blocked/ineligible plan cannot be frozen as executable')
            if len(plan['arms']) == 1 and plan['contrast_identifiable']: raise IntegrityError('Single arm has no intervention contrast')
            status = plan['plan_state']
        elif kind == 'value_curve_plan':
            plan = read_json(root / 'plan.json')
            if plan['package_kind'] != kind or plan['template'] != 'value_curve': raise IntegrityError('Invalid curve plan kind/template')
            if sha_file(root / 'source_index.json') != plan['source_index_sha256'] or sha_file(root / 'analysis_index.json') != plan['analysis_index_sha256']:
                raise IntegrityError('Curve plan source anchors differ')
            if sha_file(root / 'prediction_commitment.json') != plan['prediction_commitment_sha256']:
                raise IntegrityError('Curve plan prediction commitment differs')
            if plan.get('execution_authorized'): raise IntegrityError('Frozen curve plan cannot assert authorized execution')
            quals = read_json(root / 'grid_qualifications.json')
            for cp_id in plan['grid']['checkpoints']:
                if cp_id not in quals: raise IntegrityError('Missing grid checkpoint qualification: ' + cp_id)
            if plan['plan_state'] == 'frozen':
                if plan['blockers'] or any(not quals[c]['eligible'] for c in plan['grid']['checkpoints']):
                    raise IntegrityError('Blocked or ineligible curve plan cannot be frozen')
            status = plan['plan_state']
        elif kind in ('experiment_execution', 'experiment_validation', 'value_curve_execution', 'value_curve_results'):
            from .controller import check_execution_or_validation
            status = check_execution_or_validation(root, kind)
        else: raise IntegrityError('Unknown experiment package kind')
        return {'verified': True, 'index_sha256': index_hash, 'package_kind': kind, 'status': status,
                'warnings': ['integrity_only_not_causal_truth'], 'errors': []}
    except (OSError, KeyError, TypeError, ValueError, IntegrityError) as exc:
        return {'verified': False, 'package_kind': kind, 'errors': [{'type': type(exc).__name__, 'message': str(exc)}]}
