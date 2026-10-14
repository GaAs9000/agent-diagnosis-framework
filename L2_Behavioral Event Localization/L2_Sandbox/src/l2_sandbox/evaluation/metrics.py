"""Metrics require external labels or observed continuation branches. No natural trajectory is auto-labelled causal GT."""
from __future__ import annotations
import math
import random
from ..io import L2Error, digest


def localization_metrics(predictions: list[dict], gold: list[dict], k=3):
    if type(k) is not int or k < 1: raise L2Error('k must be positive integer')
    pmap = {r['case_id']: r for r in predictions}
    if len(pmap) != len(predictions): raise L2Error('Duplicate prediction case')
    hits = 0; known = 0; abstained = 0; missing = 0; rows = []
    for g in gold:
        case = g['case_id']; p = pmap.get(case)
        if p is None: missing += 1; rows.append({'case_id': case, 'status': 'missing_prediction'}); continue
        accepted = g.get('accepted_event_sets')
        if not accepted: rows.append({'case_id': case, 'status': 'gold_undefined_not_scored'}); continue
        known += 1
        selected = p.get('event_sets', [])[:k]
        if not selected: abstained += 1
        hit = any(set(x) == set(y) for x in selected for y in accepted)
        hits += hit; rows.append({'case_id': case, 'exact_event_set_topk_hit': hit, 'abstained': not bool(selected)})
    return {'metric': 'exact_event_set_topk; no broad interval overlap reward', 'k': k, 'labelled_cases': known,
            'hits': hits, 'hit_rate': hits / known if known else None, 'abstentions': abstained, 'missing_predictions': missing,
            'rows': rows, 'ground_truth_source': 'externally_supplied_not_inferred_from_grade_or_fault_injection_location'}


def brier(rows: list[dict]):
    valid = []; undefined = 0
    for r in rows:
        p = r.get('probability'); y = r.get('outcome')
        if y is None: undefined += 1; continue
        if type(y) is not bool or type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1:
            raise L2Error('Brier requires finite probabilities and boolean outcomes')
        valid.append((p - int(y)) ** 2)
    return {'n_defined': len(valid), 'n_undefined': undefined, 'brier': sum(valid) / len(valid) if valid else None}


def family_partition(rows: list[dict], holdout_fraction=.25):
    if not 0 < holdout_fraction < 1: raise L2Error('Invalid holdout fraction')
    groups = {}
    for r in rows:
        # Caller must provide family_id; never use episode ID to split siblings.
        if not r.get('family_id'): raise L2Error('Explicit task-family identity required')
        groups.setdefault(r['family_id'], []).append(r['case_id'])
    if len(groups) < 2: return {'status': 'insufficient_independent_families', 'groups': groups, 'train': [], 'holdout': []}
    holdout = []; train = []
    for fam, ids in sorted(groups.items()):
        (holdout if int(digest(fam)[:8], 16) / 2**32 < holdout_fraction else train).extend(ids)
    return {'status': 'partitioned' if train and holdout else 'empty_partition_do_not_claim_evaluation',
            'train': train, 'holdout': holdout, 'family_count': len(groups), 'groups': groups}


def curve_gold(case_id: str, curve_results: dict, events: list[dict]) -> dict:
    """Maps a two-stage value curve verdict to accepted_event_sets for localization_metrics (Section 5.3)."""
    verdict = curve_results.get('verdict')
    curve = curve_results.get('curve') or {}
    req_events = [e for e in events if e.get('kind') == 'model.request']

    def _turn_events(turn_1based: int) -> list[str]:
        if not 1 <= turn_1based <= len(req_events):
            return []
        start_seq = req_events[turn_1based - 1]['seq']
        end_seq = req_events[turn_1based]['seq'] if turn_1based < len(req_events) else max(e['seq'] for e in events) + 1
        return [e['event_id'] for e in events if start_seq <= e['seq'] < end_seq]

    if verdict == 'concentrated_turn':
        j = int(curve['argmax_turn'])
        ev_ids = _turn_events(j)
        req_id = [req_events[j - 1]['event_id']] if 1 <= j <= len(req_events) else []
        accepted = [s for s in ([req_id, ev_ids] if ev_ids != req_id else [ev_ids]) if s]
        return {
            'case_id': case_id,
            'verdict': verdict,
            'evidence_basis': 'value_curve',
            'localized_turn': j,
            'accepted_event_sets': accepted,
        }
    if verdict == 'concentrated_interval':
        sel = curve['selected_interval']
        a, b = (sel['start_turn'], sel['end_turn']) if isinstance(sel, dict) else sel
        interval_sets = []
        all_interval_events = []
        for t in range(int(a) + 1, int(b) + 1):
            te = _turn_events(t)
            if te:
                interval_sets.append(te)
                all_interval_events.extend(te)
            if 1 <= t <= len(req_events):
                interval_sets.append([req_events[t - 1]['event_id']])
        if all_interval_events:
            interval_sets.append(all_interval_events)
        return {
            'case_id': case_id,
            'verdict': verdict,
            'evidence_basis': 'value_curve',
            'localized_interval': [int(a), int(b)],
            'accepted_event_sets': interval_sets,
        }
    return {
        'case_id': case_id,
        'verdict': verdict,
        'evidence_basis': 'value_curve',
        'accepted_event_sets': [],
        'status': 'gold_undefined_not_scored',
    }


def _normalize_pred_map(raw_preds) -> dict[str, float]:
    if not raw_preds:
        return {}
    if isinstance(raw_preds, dict):
        if 'value_predictions' in raw_preds and isinstance(raw_preds['value_predictions'], (dict, list)):
            return _normalize_pred_map(raw_preds['value_predictions'])
        return {str(k): float(v) for k, v in raw_preds.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    if isinstance(raw_preds, list):
        return {
            str(row['checkpoint_id']): float(row.get('predicted_value', row.get('value')))
            for row in raw_preds
            if isinstance(row, dict) and 'checkpoint_id' in row and ('predicted_value' in row or 'value' in row)
        }
    return {}


def _score_predictor(pred_map: dict[str, float], active_receipts: list[dict]) -> dict:
    sq_errors = []
    undefined = 0
    unpredicted = 0
    by_cp: dict[str, list[float]] = {}
    for r in active_receipts:
        cp_id = r['checkpoint_id']
        out = r.get('outcome') or {}
        defined = bool(out.get('endpoint_defined')) and out.get('value') is not None
        if not defined:
            undefined += 1
        if cp_id not in pred_map:
            unpredicted += 1
            continue
        if not defined:
            continue
        p = float(pred_map[cp_id])
        y = float(out['value'])
        if not math.isfinite(p) or not 0.0 <= p <= 1.0:
            raise L2Error(f'Value prediction out of [0, 1] at {cp_id}: {p}')
        err = (p - y) ** 2
        sq_errors.append(err)
        by_cp.setdefault(cp_id, []).append(err)
    complete = unpredicted == 0
    mean_loss = sum(sq_errors) / len(sq_errors) if sq_errors and complete else None
    cp_losses = {cp: sum(v) / len(v) for cp, v in sorted(by_cp.items()) if v}
    return {
        'status': 'incomplete_predictions' if not complete else ('scored' if sq_errors else 'unscored_no_defined_outcomes'),
        'n_expected_branches': len(active_receipts),
        'n_defined_branches': len(active_receipts) - undefined,
        'prediction_coverage': (len(active_receipts) - unpredicted) / len(active_receipts) if active_receipts else None,
        'n_scored_branches': len(sq_errors),
        'n_undefined_branches': undefined,
        'n_unpredicted_branches': unpredicted,
        'brier_or_mse': mean_loss,
        'checkpoint_losses': cp_losses,
    }


def value_prediction_scores(
    case_id: str,
    family_id: str,
    diagnosis: dict,
    baselines: dict[str, dict],
    receipts: list[dict],
    endpoint: dict | str,
) -> dict:
    """Scores Analyst and baseline value predictions against Stage 1 continuation branches (Section 5.3)."""
    active_s1 = [r for r in receipts if r.get('stage') == 'stage1' and not r.get('replaced_by')]
    if not active_s1:
        active_s1 = [r for r in receipts if not r.get('replaced_by')]

    analyst_preds = _normalize_pred_map(diagnosis.get('value_predictions'))

    scores = {'analyst': _score_predictor(analyst_preds, active_s1)}
    for name, b_entry in sorted((baselines or {}).items()):
        scores[name] = _score_predictor(_normalize_pred_map(b_entry), active_s1)

    linear_loss = (scores.get('linear_decay') or {}).get('brier_or_mse')
    analyst_loss = scores['analyst']['brier_or_mse']
    skill_vs_linear = (
        1.0 - (analyst_loss / linear_loss)
        if analyst_loss is not None and linear_loss is not None and linear_loss > 1e-12
        else None
    )
    return {
        'case_id': case_id,
        'family_id': family_id,
        'endpoint': endpoint,
        'scores': scores,
        'brier_skill_score_vs_linear_decay': skill_vs_linear,
        'proper_scoring_rule': 'quadratic_brier_or_bounded_mse_on_stage1_grid_branches',
    }


def evaluate_value_predictions_across_cases(
    cases: list[dict],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: str = 'l2-v02-family-bootstrap',
) -> dict:
    """Computes family-clustered bootstrap CIs for Analyst Brier/MSE and skill score vs linear decay."""
    if not cases:
        raise L2Error('At least one case required for value prediction evaluation')
    by_family: dict[str, list[dict]] = {}
    for c in cases:
        fam = c.get('family_id')
        if not fam:
            raise L2Error('Explicit family_id required on every case')
        by_family.setdefault(str(fam), []).append(c)

    def _agg(subset: list[dict]) -> tuple[float | None, float | None, float | None]:
        a_vals = [
            c['scores']['analyst']['brier_or_mse']
            for c in subset
            if (c.get('scores', {}).get('analyst') or {}).get('brier_or_mse') is not None
        ]
        l_vals = [
            c['scores']['linear_decay']['brier_or_mse']
            for c in subset
            if (c.get('scores', {}).get('linear_decay') or {}).get('brier_or_mse') is not None
        ]
        mean_a = sum(a_vals) / len(a_vals) if a_vals else None
        mean_l = sum(l_vals) / len(l_vals) if l_vals else None
        paired = [c for c in subset if all(
            (c.get('scores', {}).get(name) or {}).get('brier_or_mse') is not None
            for name in ('analyst', 'linear_decay')
        )]
        paired_a = sum(c['scores']['analyst']['brier_or_mse'] for c in paired) / len(paired) if paired else None
        paired_l = sum(c['scores']['linear_decay']['brier_or_mse'] for c in paired) / len(paired) if paired else None
        skill = (1.0 - paired_a / paired_l) if paired_l and paired_l > 1e-12 else None
        return mean_a, mean_l, skill

    mean_a, mean_l, skill = _agg(cases)
    coverage = {
        'analyst_scored_case_count': sum((c.get('scores', {}).get('analyst') or {}).get('brier_or_mse') is not None for c in cases),
        'analyst_unscored_case_count': sum((c.get('scores', {}).get('analyst') or {}).get('brier_or_mse') is None for c in cases),
        'paired_scored_case_count': sum(all(
            (c.get('scores', {}).get(name) or {}).get('brier_or_mse') is not None
            for name in ('analyst', 'linear_decay')
        ) for c in cases),
    }
    fams = sorted(by_family.keys())
    analyst_fams = [f for f in fams if any(
        (c.get('scores', {}).get('analyst') or {}).get('brier_or_mse') is not None
        for c in by_family[f]
    )]
    paired_fams = [f for f in fams if any(all(
        (c.get('scores', {}).get(name) or {}).get('brier_or_mse') is not None
        for name in ('analyst', 'linear_decay')
    ) for c in by_family[f])]
    coverage.update({
        'analyst_scored_family_count': len(analyst_fams),
        'paired_scored_family_count': len(paired_fams),
    })

    def _bootstrap(eligible_fams: list[str], metric: str, value_index: int) -> list[float]:
        if len(eligible_fams) < 2:
            return []
        rng = random.Random(int(digest({'seed': seed, 'families': eligible_fams, 'metric': metric})[:16], 16))
        values = []
        for _ in range(n_boot):
            sampled_fams = [rng.choice(eligible_fams) for _ in eligible_fams]
            sampled_cases = [item for f in sampled_fams for item in by_family[f]]
            value = _agg(sampled_cases)[value_index]
            if value is not None:
                values.append(value)
        return sorted(values)

    boot_analyst = _bootstrap(analyst_fams, 'analyst_brier_or_mse', 0)
    boot_skills = _bootstrap(paired_fams, 'skill_score_vs_linear_decay', 2)

    def _pct(arr: list[float], p: float) -> float | None:
        if not arr:
            return None
        idx = max(0, min(len(arr) - 1, int(round(p * (len(arr) - 1)))))
        return arr[idx]

    return {
        'status': (
            'insufficient_independent_families_for_cluster_ci' if len(fams) < 2
            else 'incomplete_analyst_predictions' if coverage['analyst_unscored_case_count']
            else 'completed'
        ),
        **coverage,
        'family_count': len(fams),
        'case_count': len(cases),
        'analyst_brier_or_mse': mean_a,
        'analyst_ci_95': [_pct(boot_analyst, alpha / 2.0), _pct(boot_analyst, 1.0 - alpha / 2.0)],
        'linear_decay_brier_or_mse': mean_l,
        'skill_score_vs_linear_decay': skill,
        'skill_ci_95': [_pct(boot_skills, alpha / 2.0), _pct(boot_skills, 1.0 - alpha / 2.0)],
    }
