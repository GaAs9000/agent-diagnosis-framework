"""Defect origin scoring, mechanism hypothesis evaluation, and cross-trajectory pooling (L2 v0.4 Section 7)."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Sequence
from .io import L2Error, read_json, read_jsonl, sha_file, with_header, write_json
from .stats import clustered_bootstrap
from .verify import evaluate_intervention, evaluate_partition, evaluate_two_point, load_branch_facts


def _defect_key(d: dict[str, Any]) -> str:
    return str(d.get('defect') or d.get('item_id') or '')


def score_origins(
    predicted_origins: Sequence[dict[str, Any]],
    defects: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluates Call 1 origin predictions `(defect, kind, turn)` against ground-truth `defects` (Section 7.1 & 7.2)."""
    by_defect: dict[str, dict[str, Any]] = {
        _defect_key(o): o for o in predicted_origins if isinstance(o, dict) and _defect_key(o)
    }

    item_evals: list[dict[str, Any]] = []
    kind_hits_unweighted = 0
    kind_total_unweighted = 0
    kind_weight_hit = 0.0
    kind_weight_total = 0.0

    turn_hits_exact = 0
    turn_hits_tol1 = 0
    turn_total_unweighted = 0
    turn_weight_exact = 0.0
    turn_weight_tol1 = 0.0
    turn_weight_total = 0.0

    for d in defects:
        d_id = _defect_key(d)
        true_kind = str(d['kind'])
        true_turn = d.get('turn') if 'turn' in d else d.get('origin_turn')
        ded = float(d.get('deduction', 0.0))

        pred = by_defect.get(d_id)
        pred_kind = str(pred['kind']) if pred and pred.get('kind') is not None else None
        raw_pt = pred.get('turn') if pred and 'turn' in pred else (pred.get('origin_turn') if pred else None)
        pred_turn = int(raw_pt) if isinstance(raw_pt, int) and not isinstance(raw_pt, bool) else None

        kind_match = (pred_kind == true_kind)
        kind_total_unweighted += 1
        kind_weight_total += ded
        if kind_match:
            kind_hits_unweighted += 1
            kind_weight_hit += ded

        turn_eligible = true_kind in ('broken', 'undelivered') and isinstance(true_turn, int) and not isinstance(true_turn, bool)
        turn_exact = False
        turn_tol1 = False
        if turn_eligible:
            turn_total_unweighted += 1
            turn_weight_total += ded
            if isinstance(pred_turn, int):
                turn_exact = (pred_turn == int(true_turn))
                turn_tol1 = (abs(pred_turn - int(true_turn)) <= 1)
                if turn_exact:
                    turn_hits_exact += 1
                    turn_weight_exact += ded
                if turn_tol1:
                    turn_hits_tol1 += 1
                    turn_weight_tol1 += ded

        item_evals.append({
            'defect': d_id,
            'item_id': d_id,
            'deduction': ded,
            'true_kind': true_kind,
            'pred_kind': pred_kind,
            'kind_correct': kind_match,
            'true_origin_turn': true_turn,
            'pred_origin_turn': pred_turn,
            'turn_eligible': turn_eligible,
            'turn_exact_hit': turn_exact if turn_eligible else None,
            'turn_tol1_hit': turn_tol1 if turn_eligible else None,
        })

    return {
        'status': 'scored' if kind_total_unweighted > 0 else 'no_defects',
        'defect_count': kind_total_unweighted,
        'total_deduction': round(kind_weight_total, 6),
        'kind_hits': kind_hits_unweighted,
        'kind_accuracy_unweighted': (kind_hits_unweighted / kind_total_unweighted) if kind_total_unweighted > 0 else None,
        'kind_weight_hit': round(kind_weight_hit, 6),
        'kind_accuracy_weighted': (kind_weight_hit / kind_weight_total) if kind_weight_total > 1e-12 else None,
        'turn_eligible_count': turn_total_unweighted,
        'turn_eligible_deduction': round(turn_weight_total, 6),
        'turn_hits_exact': turn_hits_exact,
        'turn_hits_tol1': turn_hits_tol1,
        'turn_accuracy_exact': (turn_hits_exact / turn_total_unweighted) if turn_total_unweighted > 0 else None,
        'turn_accuracy_tol1': (turn_hits_tol1 / turn_total_unweighted) if turn_total_unweighted > 0 else None,
        'turn_weight_exact': round(turn_weight_exact, 6),
        'turn_weight_tol1': round(turn_weight_tol1, 6),
        'turn_accuracy_exact_weighted': (turn_weight_exact / turn_weight_total) if turn_weight_total > 1e-12 else None,
        'turn_accuracy_tol1_weighted': (turn_weight_tol1 / turn_weight_total) if turn_weight_total > 1e-12 else None,
        'items': item_evals,
    }


def evaluate_mechanisms(
    hypotheses: Sequence[dict[str, Any]],
    defects: Sequence[dict[str, Any]],
    parent_facts: list[dict[str, Any]],
    branch_facts_by_trial: dict[str, list[dict[str, Any]]],
    outcomes: Sequence[dict[str, Any]],
    *,
    alpha: float = 0.05,
    min_per_group: int = 4,
    min_intervention_effect: float = 0.15,
    outcome: str = 'total',
) -> dict[str, Any]:
    """Evaluates Call 2 mechanism hypotheses (both `partition` and `intervention`) and defect coverage (Section 6 & 7.3)."""
    part_eval = evaluate_partition(
        hypotheses,
        parent_facts,
        branch_facts_by_trial,
        outcomes,
        alpha=alpha,
        min_per_group=min_per_group,
        outcome=outcome,
    )
    intv_eval = evaluate_intervention(
        hypotheses,
        outcomes,
        alpha=alpha,
        min_effect=min_intervention_effect,
        outcome=outcome,
    )

    all_hyps = [*part_eval['hypotheses'], *intv_eval['hypotheses']]
    proposed_count = len(all_hyps)
    confirmed_count = sum(1 for h in all_hyps if h['status'] == 'confirmed')
    testable_count = sum(1 for h in all_hyps if h['status'] in ('confirmed', 'not_confirmed'))

    defect_ids = {_defect_key(d) for d in defects if _defect_key(d)}
    tested_defect_ids = {
        _defect_key(h)
        for h in all_hyps
        if h.get('status') in ('confirmed', 'not_confirmed') and _defect_key(h) in defect_ids
    }
    if not tested_defect_ids and testable_count > 0 and defect_ids:
        tested_defect_ids = set(defect_ids)

    confirmed_defect_ids = {
        _defect_key(h)
        for h in all_hyps
        if h.get('status') == 'confirmed' and _defect_key(h) in defect_ids
    }
    if not confirmed_defect_ids and confirmed_count > 0 and defect_ids:
        confirmed_defect_ids = set(defect_ids)

    defect_coverage = (
        len(confirmed_defect_ids) / len(tested_defect_ids)
        if tested_defect_ids else None
    )

    return {
        'status': 'scored' if outcomes else 'unverified',
        'proposed_count': proposed_count,
        'testable_count': testable_count,
        'confirmed_count': confirmed_count,
        'confirmation_rate': (confirmed_count / proposed_count) if proposed_count > 0 else None,
        'testable_confirmation_rate': (confirmed_count / testable_count) if testable_count > 0 else None,
        'tested_defects_count': len(tested_defect_ids),
        'covered_defects_count': len(confirmed_defect_ids),
        'defect_coverage': defect_coverage,
        'partition': part_eval,
        'intervention': intv_eval,
        'hypotheses': all_hyps,
    }


def score_episode(
    l2_dir: Path,
    *,
    alpha: float = 0.05,
    cost_per_branch_usd: float = 0.01,
    llm_cost_per_call_usd: float = 0.002,
    active_analysts: Sequence[str] | None = None,
    expected_input_bindings: dict[str, dict[str, Any]] | None = None,
    reuse_dirs: Sequence[Path] | None = None,
    two_point_pairs: Sequence[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Computes `scores.json` and `verifications.json` for an L2 v0.4 episode directory."""
    l2_dir = Path(l2_dir)
    facts_path = l2_dir / 'facts.jsonl'
    parent_facts = read_jsonl(facts_path) if facts_path.is_file() else []
    defects_path = l2_dir / 'defects.json'
    if not defects_path.is_file():
        raise L2Error(f'Missing defects.json in {l2_dir}; run l2 diagnose first')
    defects_doc = read_json(defects_path)
    defects = defects_doc.get('defects') or []
    episode_id = str(defects_doc['episode_id'])
    family_id = str(defects_doc.get('family_id') or defects_doc.get('task_id') or 'unknown_family')
    terminal_y = defects_doc.get('terminal_outcome') if 'terminal_outcome' in defects_doc else defects_doc.get('terminal_score_y')
    delivery_gap = defects_doc.get('delivery_gap_g') if 'delivery_gap_g' in defects_doc else defects_doc.get('delivery_gap', 0.0)

    outcomes_path = l2_dir / 'outcomes.jsonl'
    if not outcomes_path.is_file() and reuse_dirs:
        for r_dir in reuse_dirs:
            cand_out = Path(r_dir) / 'outcomes.jsonl'
            if cand_out.is_file():
                outcomes_path = cand_out
                break
    outcomes = [r for r in read_jsonl(outcomes_path) if r.get('selected', True)] if outcomes_path.is_file() else []
    branch_facts_by_trial = load_branch_facts(l2_dir, outcomes, reuse_dirs=reuse_dirs) if outcomes else {}

    origins_dir = l2_dir / 'origins'
    hyps_dir = l2_dir / 'hypotheses'
    analyst_scores: dict[str, Any] = {}
    input_hashes: dict[str, str] = {'defects_sha256': sha_file(defects_path)}
    if facts_path.is_file():
        input_hashes['facts_sha256'] = sha_file(facts_path)
    if outcomes_path.is_file():
        input_hashes['outcomes_sha256'] = sha_file(outcomes_path)

    active = set(active_analysts) if active_analysts is not None else None
    bindings = expected_input_bindings or {}

    def validate_output(doc: dict[str, Any], kind: str) -> None:
        if not bindings:
            return
        analyst = str(doc.get('analyst', ''))
        expected = bindings.get(kind, {}).get(analyst)
        if expected and (doc.get('input_sha256') != expected or doc.get('episode_id') != episode_id):
            raise L2Error(f'Stale or unbound {kind} output for {analyst}')

    if origins_dir.is_dir():
        for of in sorted(origins_dir.glob('*.json')):
            o_doc = read_json(of)
            a_name = str(o_doc['analyst'])
            if active is not None and a_name not in active:
                continue
            validate_output(o_doc, 'origins')
            input_hashes[f'origins:{of.name}'] = sha_file(of)
            o_score = score_origins(o_doc.get('origins') or [], defects)

            hf = hyps_dir / of.name
            h_list: list[dict[str, Any]] = []
            calls_made = 1
            if hf.is_file():
                h_doc = read_json(hf)
                if h_doc.get('analyst') == a_name:
                    validate_output(h_doc, 'hypotheses')
                    input_hashes[f'hypotheses:{hf.name}'] = sha_file(hf)
                    h_list = h_doc.get('hypotheses') or []
                    calls_made = 2

            m_eval = evaluate_mechanisms(
                h_list,
                defects,
                parent_facts,
                branch_facts_by_trial,
                outcomes,
                alpha=alpha,
            )
            is_llm = a_name.startswith('llm:')
            cost_usd = (calls_made * llm_cost_per_call_usd) if is_llm else 0.0
            equiv_branches = (cost_usd / cost_per_branch_usd) if cost_per_branch_usd > 0 else 0.0

            analyst_scores[a_name] = {
                'analyst': a_name,
                'analyst_group': o_doc.get('analyst_group'),
                'origin_localization': o_score,
                'mechanism_testing': m_eval,
                'cost': {
                    'basis': 'configured_estimate',
                    'analyst_calls': calls_made if is_llm else 0,
                    'continuation_branches': len(outcomes),
                    'cost_usd': round(cost_usd, 6),
                    'equivalent_continuations': round(equiv_branches, 4),
                },
            }

    two_point_results: list[dict[str, Any]] = []
    if outcomes:
        pairs = list(two_point_pairs or [])
        if not pairs:
            plain_ks = sorted({
                int(r['k']) for r in outcomes
                if (r.get('arm', 'plain') == 'plain' or r.get('stage') in ('s1', 's2')) and r.get('valid')
            })
            for idx in range(len(plain_ks) - 1):
                if plain_ks[idx + 1] - plain_ks[idx] == 1:
                    pairs.append((plain_ks[idx], plain_ks[idx + 1]))
        for k1, k2 in pairs:
            two_point_results.append(evaluate_two_point(outcomes, k1, k2, alpha=alpha))

    if outcomes or two_point_results:
        verifications_doc = with_header(
            input_hashes,
            {
                'episode_id': episode_id,
                'family_id': family_id,
                'total_branches': len(outcomes),
                'valid_branches': sum(1 for r in outcomes if r.get('valid')),
                'reused_branches': sum(1 for r in outcomes if r.get('reused')),
                'executed_branches': sum(1 for r in outcomes if not r.get('reused')),
                'two_point_comparisons': two_point_results,
                'analysts': {
                    a_name: a_data['mechanism_testing']
                    for a_name, a_data in analyst_scores.items()
                },
            },
        )
        write_json(l2_dir / 'verifications.json', verifications_doc)

    scores_doc = with_header(
        input_hashes,
        {
            'episode_id': episode_id,
            'family_id': family_id,
            'terminal_outcome': terminal_y,
            'delivery_gap': delivery_gap,
            'defect_count': len(defects),
            'total_deduction': defects_doc.get('total_deduction', 0.0),
            'analysts': analyst_scores,
            'two_point_comparisons': two_point_results,
        },
    )
    write_json(l2_dir / 'scores.json', scores_doc)
    return scores_doc


def summarize_runs(
    l2_dirs: Sequence[Path],
    *,
    heldout_families: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Pools L2 v0.4 origin accuracy, turn accuracy, and mechanism confirmation across episodes (`l2 summarize`)."""
    episodes: list[dict[str, Any]] = []
    defects_docs: list[dict[str, Any]] = []
    input_hashes: dict[str, str] = {}

    for d in l2_dirs:
        p = Path(d)
        if p.name == 'scores.json' and p.is_file():
            p = p.parent
        sp = p / 'scores.json'
        dp = p / 'defects.json'
        if not sp.is_file():
            continue
        input_hashes[str(p.name)] = sha_file(sp)
        episodes.append(read_json(sp))
        if dp.is_file():
            defects_docs.append(read_json(dp))

    total_deduction = 0.0
    total_delivery_gap = 0.0
    deduction_by_kind = {'undelivered': 0.0, 'broken': 0.0, 'never': 0.0}
    count_by_kind = {'undelivered': 0, 'broken': 0, 'never': 0}
    window_same_target_turns = 0
    window_total_turns = 0

    for dd in defects_docs:
        total_deduction += float(dd.get('total_deduction', 0.0))
        g_val = dd.get('delivery_gap_g') if 'delivery_gap_g' in dd else dd.get('delivery_gap', 0.0)
        total_delivery_gap += float(g_val or 0.0)
        seen_windows: set[tuple[str, int, int]] = set()
        for d in dd.get('defects') or []:
            knd = str(d['kind'])
            ded = float(d.get('deduction', 0.0))
            deduction_by_kind[knd] = deduction_by_kind.get(knd, 0.0) + ded
            count_by_kind[knd] = count_by_kind.get(knd, 0) + 1
            ws = d.get('window') or d.get('uncorrected_window')
            if ws and int(ws.get('window_turns', 0)) > 0:
                wkey = (str(dd.get('episode_id')), int(ws['start_turn']), int(ws['end_turn']))
                if wkey not in seen_windows:
                    seen_windows.add(wkey)
                    same_t = ws.get('inspect_preview_or_same_file_turns')
                    if same_t is None:
                        same_t = ws.get('same_target_turns', 0)
                    window_same_target_turns += int(same_t)
                    window_total_turns += int(ws.get('window_turns', 0))

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
                    **a_entry,
                })

        def _kind_unweighted(subset: Sequence[dict[str, Any]]) -> float | None:
            hits = sum(c['origin_localization'].get('kind_hits', 0) for c in subset)
            tot = sum(c['origin_localization'].get('defect_count', 0) for c in subset)
            return (hits / tot) if tot > 0 else None

        def _kind_weighted(subset: Sequence[dict[str, Any]]) -> float | None:
            w_hit = sum(float(c['origin_localization'].get('kind_weight_hit', 0.0)) for c in subset)
            w_tot = sum(float(c['origin_localization'].get('total_deduction', 0.0)) for c in subset)
            return (w_hit / w_tot) if w_tot > 1e-12 else None

        def _turn_tol1(subset: Sequence[dict[str, Any]]) -> float | None:
            hits = sum(c['origin_localization'].get('turn_hits_tol1', 0) for c in subset)
            tot = sum(c['origin_localization'].get('turn_eligible_count', 0) for c in subset)
            return (hits / tot) if tot > 0 else None

        def _turn_exact(subset: Sequence[dict[str, Any]]) -> float | None:
            hits = sum(c['origin_localization'].get('turn_hits_exact', 0) for c in subset)
            tot = sum(c['origin_localization'].get('turn_eligible_count', 0) for c in subset)
            return (hits / tot) if tot > 0 else None

        def _turn_tol1_weighted(subset: Sequence[dict[str, Any]]) -> float | None:
            w_hit = sum(float(c['origin_localization'].get('turn_weight_tol1', 0.0)) for c in subset)
            w_tot = sum(float(c['origin_localization'].get('turn_eligible_deduction', 0.0)) for c in subset)
            return (w_hit / w_tot) if w_tot > 1e-12 else None

        def _confirm_rate(subset: Sequence[dict[str, Any]]) -> float | None:
            prop = sum(c['mechanism_testing'].get('proposed_count', 0) for c in subset if c['mechanism_testing'].get('status') == 'scored')
            conf = sum(c['mechanism_testing'].get('confirmed_count', 0) for c in subset if c['mechanism_testing'].get('status') == 'scored')
            return (conf / prop) if prop > 0 else None

        def _defect_cov(subset: Sequence[dict[str, Any]]) -> float | None:
            tested = sum(c['mechanism_testing'].get('tested_defects_count', 0) for c in subset if c['mechanism_testing'].get('status') == 'scored')
            cov = sum(c['mechanism_testing'].get('covered_defects_count', 0) for c in subset if c['mechanism_testing'].get('status') == 'scored')
            return (cov / tested) if tested > 0 else None

        defect_cases = [c for c in cases if c['origin_localization'].get('defect_count', 0) > 0]
        turn_cases = [c for c in cases if c['origin_localization'].get('turn_eligible_count', 0) > 0]
        mech_cases = [c for c in cases if c['mechanism_testing'].get('status') == 'scored' and c['mechanism_testing'].get('proposed_count', 0) > 0]

        by_analyst[a_name] = {
            'analyst_group': cases[0].get('analyst_group') if cases else None,
            'episodes_seen': len(cases),
            'episodes_with_defects': len(defect_cases),
            'episodes_with_turn_eligible_defects': len(turn_cases),
            'episodes_with_verified_mechanisms': len(mech_cases),
            'kind_accuracy_unweighted': clustered_bootstrap(defect_cases, _kind_unweighted),
            'kind_accuracy_weighted': clustered_bootstrap(defect_cases, _kind_weighted),
            'turn_accuracy_tol1': clustered_bootstrap(turn_cases, _turn_tol1),
            'turn_accuracy_exact': clustered_bootstrap(turn_cases, _turn_exact),
            'turn_accuracy_tol1_weighted': clustered_bootstrap(turn_cases, _turn_tol1_weighted),
            'mechanism_confirmation_rate': clustered_bootstrap(mech_cases, _confirm_rate),
            'defect_coverage': clustered_bootstrap(mech_cases, _defect_cov),
            'mean_analyst_calls': round(sum(c['cost']['analyst_calls'] for c in cases) / len(cases), 4) if cases else 0.0,
        }

    group_to_analyst: dict[str, str] = {}
    for a_name, info in by_analyst.items():
        grp = info.get('analyst_group')
        if grp:
            group_to_analyst[str(grp)] = a_name

    paired_comparisons: dict[str, Any] = {}
    for g1, g2 in (('C', 'B'), ('C', 'A'), ('B', 'A')):
        a1 = group_to_analyst.get(g1)
        a2 = group_to_analyst.get(g2)
        if not a1 or not a2:
            continue
        paired_eps = [
            ep for ep in episodes
            if a1 in (ep.get('analysts') or {}) and a2 in (ep.get('analysts') or {})
            and int(ep.get('defect_count', 0)) > 0
        ]

        def _diff_kind_unweighted(rows: Sequence[dict[str, Any]], k1: str = a1, k2: str = a2) -> float | None:
            tot = sum(r['analysts'][k1]['origin_localization'].get('defect_count', 0) for r in rows)
            if tot == 0:
                return None
            h1 = sum(r['analysts'][k1]['origin_localization'].get('kind_hits', 0) for r in rows)
            h2 = sum(r['analysts'][k2]['origin_localization'].get('kind_hits', 0) for r in rows)
            return (h1 - h2) / tot

        def _diff_kind_weighted(rows: Sequence[dict[str, Any]], k1: str = a1, k2: str = a2) -> float | None:
            wtot = sum(float(r['analysts'][k1]['origin_localization'].get('total_deduction', 0.0)) for r in rows)
            if wtot <= 1e-12:
                return None
            w1 = sum(float(r['analysts'][k1]['origin_localization'].get('kind_weight_hit', 0.0)) for r in rows)
            w2 = sum(float(r['analysts'][k2]['origin_localization'].get('kind_weight_hit', 0.0)) for r in rows)
            return (w1 - w2) / wtot

        def _diff_turn_tol1(rows: Sequence[dict[str, Any]], k1: str = a1, k2: str = a2) -> float | None:
            tot = sum(r['analysts'][k1]['origin_localization'].get('turn_eligible_count', 0) for r in rows)
            if tot == 0:
                return None
            h1 = sum(r['analysts'][k1]['origin_localization'].get('turn_hits_tol1', 0) for r in rows)
            h2 = sum(r['analysts'][k2]['origin_localization'].get('turn_hits_tol1', 0) for r in rows)
            return (h1 - h2) / tot

        paired_comparisons[f'{g1}_minus_{g2}'] = {
            'analyst_1': a1,
            'analyst_2': a2,
            'paired_episodes': len(paired_eps),
            'kind_accuracy_unweighted_diff': clustered_bootstrap(paired_eps, _diff_kind_unweighted),
            'kind_accuracy_weighted_diff': clustered_bootstrap(paired_eps, _diff_kind_weighted),
            'turn_accuracy_tol1_diff': clustered_bootstrap(paired_eps, _diff_turn_tol1),
        }

    return with_header(
        input_hashes,
        {
            'included_episodes': [ep['episode_id'] for ep in episodes],
            'defect_census': {
                'episodes_count': len(defects_docs),
                'total_deduction': round(total_deduction, 6),
                'total_delivery_gap': round(total_delivery_gap, 6),
                'delivery_gap_share': round(total_delivery_gap / total_deduction, 6) if total_deduction > 1e-12 else 0.0,
                'count_by_kind': count_by_kind,
                'deduction_by_kind': {k: round(v, 6) for k, v in deduction_by_kind.items()},
                'share_by_kind': {
                    k: round(v / total_deduction, 6) if total_deduction > 1e-12 else 0.0
                    for k, v in deduction_by_kind.items()
                },
                'uncorrected_window_same_target_turns': window_same_target_turns,
                'uncorrected_window_total_turns': window_total_turns,
                'uncorrected_window_same_target_ratio': (
                    round(window_same_target_turns / window_total_turns, 6)
                    if window_total_turns > 0 else None
                ),
            },
            'analysts': by_analyst,
            'paired_comparisons': paired_comparisons,
            'heldout_families': sorted(set(heldout_families or [])),
        },
    )
