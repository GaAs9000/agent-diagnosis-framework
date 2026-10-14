"""Hypothesis testing, Brier skill scoring, explanation coverage, and cross-trajectory pooling (L2 v0.3 Section 4 & 5)."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Sequence
from .analysts.rules import linear_decay_values
from .evidence.l1 import L1Package
from .facts.build import build_facts
from .io import digest, read_json, read_jsonl, sha_file, with_header, write_json
from .predicates import RULE_VOCABULARY, eval_predicate_on_turn, validate_predicate
from .stats import clustered_bootstrap, holm_adjust, mantel_haenszel_pool, permutation_one_sided


def brier_on_outcomes(
    values: Sequence[float],
    outcomes: Sequence[dict[str, Any]],
) -> float | None:
    """Computes equal-checkpoint-weighted Brier / squared error against observed branch outcomes (Section 5).

    Each checkpoint k that has valid branches is weighted equally:
      Brier_k = mean_i (values[k] - y_{k, i})^2
      Brier   = mean_k Brier_k
    """
    by_k: dict[int, list[float]] = {}
    for r in outcomes:
        if not r.get('valid') or r.get('y') is None:
            continue
        k = int(r['k'])
        if 0 <= k < len(values):
            by_k.setdefault(k, []).append(float(r['y']))
    if not by_k:
        return None
    cp_losses = []
    for k, ys in sorted(by_k.items()):
        pk = float(values[k])
        cp_losses.append(sum((pk - y) ** 2 for y in ys) / len(ys))
    return sum(cp_losses) / len(cp_losses)


def score_values(
    analyst_values: Sequence[float],
    outcomes: Sequence[dict[str, Any]],
    num_turns: int,
    p0: float,
    terminal_y: float,
) -> dict[str, Any]:
    """Computes Analyst Brier, Linear Decay Brier, and Skill Score = 1 - Brier(analyst) / Brier(linear_decay)."""
    lin_vals = linear_decay_values(num_turns, p0, terminal_y)
    b_analyst = brier_on_outcomes(analyst_values, outcomes)
    b_linear = brier_on_outcomes(lin_vals, outcomes)
    skill = (
        1.0 - (b_analyst / b_linear)
        if b_analyst is not None and b_linear is not None and b_linear > 1e-12
        else None
    )
    return {
        'brier': b_analyst,
        'linear_decay_brier': b_linear,
        'skill_score': skill,
    }


def evaluate_hypotheses(
    hypotheses: Sequence[dict[str, Any]],
    parent_facts: list[dict[str, Any]],
    curve_doc: dict[str, Any],
    branch_facts_by_trial: dict[str, list[dict[str, Any]]],
    outcomes: Sequence[dict[str, Any]],
    *,
    alpha: float = 0.05,
    min_per_group: int = 4,
) -> dict[str, Any]:
    """Tests hypotheses `(j, P, e)` against Stage 2 branches at c_{j-1} (Section 4):
    1. Consistency: P must hold on parent turn j, else `inconsistent`.
    2. Grouping: partition Stage 2 valid branches at k = j - 1 by P on turn j; if either < 4, `untestable`.
    3. Test: one-sided permutation / Fisher exact test + Holm adjustment across hypotheses of the same analyst.
       If p_holm <= alpha and sign(Delta_P) matches sign(e), `confirmed`, else `not_confirmed`.
    4. Report: frequency f_hat, effect Delta_P = mean(Y_P) - mean(Y_not_P), and
       explained_share = (1 - f_hat) * |Delta_P| / d_j_hat.
    """
    v_hat = curve_doc.get('v_hat') or {}
    turn_drops = curve_doc.get('turn_drops') or {}

    results: list[dict[str, Any]] = []
    testable_indices: list[int] = []
    raw_pvals: list[float] = []

    for idx, h in enumerate(hypotheses):
        turn_j = int(h['turn'])
        pred = validate_predicate(h['predicate'])
        pred_effect = float(h.get('effect', -0.1))
        rationale = str(h.get('rationale') or '')

        # Step 1: Consistency on parent turn j
        if not eval_predicate_on_turn(pred, parent_facts, turn_j):
            results.append({
                'turn': turn_j,
                'predicate': pred,
                'predicted_effect': pred_effect,
                'rationale': rationale,
                'status': 'inconsistent',
                'n_p': 0,
                'n_not_p': 0,
                'frequency': None,
                'effect': None,
                'explained_share': None,
                'p_value': None,
                'p_value_holm': None,
                'evidence_level': 'E0',
            })
            continue

        # Step 2: Partition Stage 2 valid branches at c_{j-1} (k = j - 1)
        k_prev = turn_j - 1
        s2_rows = [
            r for r in outcomes
            if r.get('stage') == 's2' and int(r.get('k', -1)) == k_prev and r.get('valid') and r.get('y') is not None
        ]
        # Fallback if Stage 1 branches at k_prev are the only ones available
        if not s2_rows:
            s2_rows = [
                r for r in outcomes
                if int(r.get('k', -1)) == k_prev and r.get('valid') and r.get('y') is not None
            ]

        y_p: list[float] = []
        y_not_p: list[float] = []
        for r in s2_rows:
            b_facts = branch_facts_by_trial.get(r['trial'], [])
            if eval_predicate_on_turn(pred, b_facts, turn_j):
                y_p.append(float(r['y']))
            else:
                y_not_p.append(float(r['y']))

        n_p = len(y_p)
        n_not_p = len(y_not_p)
        total_n = n_p + n_not_p
        f_hat = (n_p / total_n) if total_n > 0 else None

        if n_p < min_per_group or n_not_p < min_per_group:
            results.append({
                'turn': turn_j,
                'predicate': pred,
                'predicted_effect': pred_effect,
                'rationale': rationale,
                'status': 'untestable',
                'n_p': n_p,
                'n_not_p': n_not_p,
                'frequency': f_hat,
                'effect': None,
                'explained_share': None,
                'p_value': None,
                'p_value_holm': None,
                'evidence_level': 'E0',
            })
            continue

        # Step 3 & 4: Compute effect, explained share, and one-sided permutation/Fisher p-value
        mean_p = sum(y_p) / n_p
        mean_not_p = sum(y_not_p) / n_not_p
        delta_p = mean_p - mean_not_p

        d_j = turn_drops.get(str(turn_j))
        if d_j is None:
            v_prev = v_hat.get(str(k_prev))
            v_curr = v_hat.get(str(turn_j))
            if v_prev is not None and v_curr is not None:
                d_j = float(v_prev) - float(v_curr)
        explained = (
            ((1.0 - float(f_hat)) * abs(delta_p) / float(d_j))
            if f_hat is not None and d_j is not None and float(d_j) > 1e-9
            else None
        )

        pval = permutation_one_sided(y_p, y_not_p, direction=-1.0 if pred_effect <= 0 else 1.0)
        testable_indices.append(idx)
        raw_pvals.append(pval)
        results.append({
            'turn': turn_j,
            'predicate': pred,
            'predicted_effect': pred_effect,
            'rationale': rationale,
            'status': 'pending',
            'n_p': n_p,
            'n_not_p': n_not_p,
            'mean_p': mean_p,
            'mean_not_p': mean_not_p,
            'frequency': f_hat,
            'effect': delta_p,
            'explained_share': explained,
            'p_value': pval,
            'p_value_holm': None,
            'evidence_level': 'E0',
        })

    adj_pvals = holm_adjust(raw_pvals)
    for ti, adj_p in zip(testable_indices, adj_pvals):
        row = results[ti]
        row['p_value_holm'] = adj_p
        direction_ok = (row['effect'] < 0.0) if row['predicted_effect'] <= 0.0 else (row['effect'] > 0.0)
        if adj_p <= alpha and direction_ok:
            row['status'] = 'confirmed'
            row['evidence_level'] = 'E1+'
        else:
            row['status'] = 'not_confirmed'
            row['evidence_level'] = 'E0'

    n_proposed = len(results)
    n_confirmed = sum(1 for r in results if r['status'] == 'confirmed')
    confirmed_rate = (n_confirmed / n_proposed) if n_proposed > 0 else None

    is_concentrated = curve_doc.get('verdict') in ('concentrated_turn', 'concentrated_interval')
    covered = (
        any(r['status'] == 'confirmed' and (r.get('explained_share') or 0.0) >= 0.5 for r in results)
        if is_concentrated
        else None
    )

    return {
        'proposed_count': n_proposed,
        'confirmed_count': n_confirmed,
        'confirmation_rate': confirmed_rate,
        'explanation_covered': covered,
        'hypotheses': results,
    }


def score_episode(
    l2_dir: Path,
    *,
    p0: float = 0.5,
    alpha: float = 0.05,
    cost_per_branch_usd: float = 0.01,
    llm_cost_per_call_usd: float = 0.002,
) -> dict[str, Any]:
    """Computes `scores.json` for all analysts in `l2_dir`."""
    l2_dir = Path(l2_dir)
    facts_path = l2_dir / 'facts.jsonl'
    parent_facts = read_jsonl(facts_path) if facts_path.is_file() else []
    plan = read_json(l2_dir / 'plan.json')
    num_turns = int(plan['turns'])
    terminal_y = float(plan['terminal_outcome'])

    outcomes_path = l2_dir / 'outcomes.jsonl'
    outcomes = read_jsonl(outcomes_path) if outcomes_path.is_file() else []
    curve_path = l2_dir / 'curve.json'
    curve_doc = read_json(curve_path) if curve_path.is_file() else {'verdict': None, 'v_hat': {}, 'turn_drops': {}}

    # Load branch facts on demand for Stage 2 branches
    branch_facts_by_trial: dict[str, list[dict[str, Any]]] = {}
    branches_root = l2_dir / 'branches'
    if branches_root.is_dir():
        for r in outcomes:
            if r.get('valid'):
                t_name = r['trial']
                b_dir = branches_root / t_name
                if (b_dir / 'manifest.json').is_file():
                    branch_facts_by_trial[t_name] = build_facts(L1Package(b_dir))

    values_dir = l2_dir / 'values'
    hyps_dir = l2_dir / 'hypotheses'
    analyst_scores: dict[str, Any] = {}
    input_hashes: dict[str, str] = {}
    if facts_path.is_file():
        input_hashes['facts_sha256'] = sha_file(facts_path)
    if outcomes_path.is_file():
        input_hashes['outcomes_sha256'] = sha_file(outcomes_path)
    if curve_path.is_file():
        input_hashes['curve_sha256'] = sha_file(curve_path)

    if values_dir.is_dir():
        for vf in sorted(values_dir.glob('*.json')):
            v_doc = read_json(vf)
            a_name = str(v_doc['analyst'])
            input_hashes[f'values:{vf.name}'] = sha_file(vf)
            v_score = score_values(v_doc['values'], outcomes, num_turns, p0, terminal_y)

            hf = hyps_dir / vf.name
            if hf.is_file():
                h_doc = read_json(hf)
                input_hashes[f'hypotheses:{hf.name}'] = sha_file(hf)
                h_list = h_doc.get('hypotheses') or []
                calls_made = 2
            else:
                h_list = []
                calls_made = 1

            h_eval = evaluate_hypotheses(
                h_list,
                parent_facts,
                curve_doc,
                branch_facts_by_trial,
                outcomes,
                alpha=alpha,
            )
            is_llm = a_name.startswith('llm:')
            cost_usd = (calls_made * llm_cost_per_call_usd) if is_llm else 0.0
            equiv_branches = (cost_usd / cost_per_branch_usd) if cost_per_branch_usd > 0 else 0.0

            analyst_scores[a_name] = {
                'analyst': a_name,
                'value_prediction': v_score,
                'hypothesis_testing': h_eval,
                'cost': {
                    'analyst_calls': calls_made if is_llm else 0,
                    'cost_usd': round(cost_usd, 6),
                    'equivalent_continuations': round(equiv_branches, 4),
                },
            }

    # Also compute vocabulary predicate strata at localized turn for cross-trajectory MH pooling
    vocab_strata: dict[str, dict[str, Any]] = {}
    target_turn = curve_doc.get('turn') or curve_doc.get('candidate_turn')
    if isinstance(target_turn, int) and target_turn >= 1:
        k_prev = target_turn - 1
        s2_rows = [
            r for r in outcomes
            if r.get('stage') == 's2' and int(r.get('k', -1)) == k_prev and r.get('valid') and r.get('y') is not None
        ]
        for v_name, v_pred in RULE_VOCABULARY.items():
            yp = [float(r['y']) for r in s2_rows if eval_predicate_on_turn(v_pred, branch_facts_by_trial.get(r['trial'], []), target_turn)]
            ynp = [float(r['y']) for r in s2_rows if not eval_predicate_on_turn(v_pred, branch_facts_by_trial.get(r['trial'], []), target_turn)]
            if yp and ynp:
                vocab_strata[v_name] = {
                    'turn': target_turn,
                    'n_p': len(yp),
                    'n_not_p': len(ynp),
                    'mean_p': sum(yp) / len(yp),
                    'mean_not_p': sum(ynp) / len(ynp),
                }

    scores_doc = with_header(
        input_hashes,
        {
            'episode_id': plan['episode_id'],
            'family_id': plan.get('family_id', 'unknown_family'),
            'p0': p0,
            'verdict': curve_doc.get('verdict'),
            'excluded_high_invalid_rate': bool(curve_doc.get('excluded_high_invalid_rate', False)),
            'analysts': analyst_scores,
            'vocabulary_strata': vocab_strata,
        },
    )
    write_json(l2_dir / 'scores.json', scores_doc)
    return scores_doc


def summarize_runs(l2_dirs: Sequence[Path]) -> dict[str, Any]:
    """Pools scores and vocabulary predicates across multiple L2 episode directories (`l2 summarize`)."""
    episodes = []
    excluded_episodes = []
    input_hashes: dict[str, str] = {}

    for d in l2_dirs:
        p = Path(d)
        sp = p / 'scores.json'
        if not sp.is_file():
            continue
        input_hashes[str(p.name)] = sha_file(sp)
        s_doc = read_json(sp)
        if s_doc.get('excluded_high_invalid_rate'):
            excluded_episodes.append(s_doc['episode_id'])
            continue
        episodes.append(s_doc)

    analyst_names = sorted({a for ep in episodes for a in (ep.get('analysts') or {}).keys()})
    by_analyst: dict[str, Any] = {}

    for a_name in analyst_names:
        cases = []
        for ep in episodes:
            a_entry = (ep.get('analysts') or {}).get(a_name)
            if a_entry:
                cases.append({
                    'episode_id': ep['episode_id'],
                    'family_id': ep.get('family_id', ep['episode_id']),
                    'verdict': ep.get('verdict'),
                    **a_entry,
                })

        def _skill_stat(subset: Sequence[dict[str, Any]]) -> float | None:
            b_a = [c['value_prediction']['brier'] for c in subset if c['value_prediction'].get('brier') is not None]
            b_l = [c['value_prediction']['linear_decay_brier'] for c in subset if c['value_prediction'].get('linear_decay_brier') is not None]
            if not b_a or not b_l or sum(b_l) <= 1e-12:
                return None
            return 1.0 - (sum(b_a) / len(b_a)) / (sum(b_l) / len(b_l))

        def _confirm_stat(subset: Sequence[dict[str, Any]]) -> float | None:
            prop = sum(c['hypothesis_testing']['proposed_count'] for c in subset)
            conf = sum(c['hypothesis_testing']['confirmed_count'] for c in subset)
            return (conf / prop) if prop > 0 else None

        def _coverage_stat(subset: Sequence[dict[str, Any]]) -> float | None:
            conc = [
                c['hypothesis_testing']['explanation_covered']
                for c in subset
                if c.get('verdict') in ('concentrated_turn', 'concentrated_interval')
                and c['hypothesis_testing'].get('explanation_covered') is not None
            ]
            return (sum(1.0 for x in conc if x) / len(conc)) if conc else None

        skill_ci = clustered_bootstrap(cases, _skill_stat)
        confirm_ci = clustered_bootstrap(cases, _confirm_stat)
        coverage_ci = clustered_bootstrap(cases, _coverage_stat)
        mean_equiv_cost = (
            sum(c['cost']['equivalent_continuations'] for c in cases) / len(cases)
            if cases else 0.0
        )

        by_analyst[a_name] = {
            'episodes_scored': len(cases),
            'skill_score': skill_ci,
            'confirmation_rate': confirm_ci,
            'explanation_coverage': coverage_ci,
            'mean_cost_equivalent_continuations': round(mean_equiv_cost, 4),
        }

    # Pool vocabulary predicates across trajectories via Mantel-Haenszel
    pooled_vocab: dict[str, Any] = {}
    for v_name in RULE_VOCABULARY:
        strata = [
            ep['vocabulary_strata'][v_name]
            for ep in episodes
            if v_name in (ep.get('vocabulary_strata') or {})
        ]
        pooled_vocab[v_name] = mantel_haenszel_pool(strata)

    # Falsifiability check (Section 5): Does any LLM analyst beat rules@1?
    rules_summary = by_analyst.get('rules@1')
    llm_candidates = [k for k in by_analyst if k.startswith('llm:')]
    recommended_mode = 'program_only'
    if rules_summary and llm_candidates:
        r_skill = (rules_summary['skill_score'].get('estimate') or -1e9)
        r_cov = (rules_summary['explanation_coverage'].get('estimate') or 0.0)
        for llm_k in llm_candidates:
            l_skill_lo = (by_analyst[llm_k]['skill_score'].get('ci_95') or [None, None])[0]
            l_cov_lo = (by_analyst[llm_k]['explanation_coverage'].get('ci_95') or [None, None])[0]
            if (l_skill_lo is not None and l_skill_lo > r_skill) or (l_cov_lo is not None and l_cov_lo > r_cov):
                recommended_mode = 'hybrid'
                break

    return with_header(
        input_hashes,
        {
            'included_episodes': [ep['episode_id'] for ep in episodes],
            'excluded_high_invalid_rate_episodes': excluded_episodes,
            'analysts': by_analyst,
            'vocabulary_mantel_haenszel': pooled_vocab,
            'recommended_default_mode': recommended_mode,
        },
    )
