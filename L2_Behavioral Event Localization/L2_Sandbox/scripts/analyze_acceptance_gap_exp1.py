#!/usr/bin/env python3
"""Pre-registered Exp1 analysis. Zero model requests; reads sealed block outputs only.

Each valid run's official completion is re-derived from its sealed L1 package
(study-outcome-v1) and must equal the block receipt. The decision rules, bootstrap
settings and thresholds come from the frozen spec, not from this script.

Usage:
  analyze_acceptance_gap_exp1.py --blocks <block phase dir> [...] --output <new dir> [--spec P]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import statistics
import sys

SCRIPT = Path(__file__).resolve()
L2 = SCRIPT.parents[1]
ROOT = L2.parents[1]
L1 = ROOT / 'L1_sandbox/C3_Experimental_Sandbox'
sys.path[:0] = [str(L1 / 'src'), str(L2 / 'src')]
from l2_sandbox.v3_execution import digest, file_digest, read_json, seal, verify_seal, write_json  # noqa: E402

SPEC = L2 / 'configs/acceptance_gap_exp1_20261022.json'
SCHEMA = 'acceptance-gap-exp1-analysis-1'
CONTRASTS = ('F', 'T', 'V')


def parse_label(label):
    match = re.fullmatch(r'(officeval_\d{3})\.([OFTV])\.r(\d{2})', label)
    if not match:
        raise ValueError('malformed Exp1 slot label: ' + label)
    return match[1], match[2], int(match[3])


def item_score(item):
    maximum, delta = float(item['max_delta']), float(item['delta'])
    if maximum > 0:
        return max(0.0, min(1.0, delta / maximum))
    if maximum < 0:
        return max(0.0, min(1.0, 1 + delta / abs(maximum)))
    return None


# ---------------------------------------------------------------- loading

def load_spec(path=SPEC):
    spec = read_json(Path(path))
    if spec.get('schema') != 'acceptance-gap-exp1-spec-v1':
        raise ValueError('unexpected Exp1 spec')
    return spec, file_digest(Path(path))


def load_block(phase, spec_sha, reverify=True):
    from c3_sandbox.l1.evaluation.study import study_outcome
    execution = Path(phase) / 'execution'
    plan = read_json(execution / 'executed_plan.json')
    verify_seal(plan, 'plan_sha256')
    cohort = read_json(execution / 'cohort.json')
    verify_seal(cohort)
    if plan['spec_sha256'] != spec_sha or cohort['plan_sha256'] != plan['plan_sha256']:
        raise ValueError('block was not run under this spec: ' + str(phase))
    runs = []
    for label in plan['slots']:
        row = cohort['slots'].get(label) or dict(status='NOT_RUN')
        task, condition, _ = parse_label(label)
        record = dict(block=plan['block'], label=label, task=task, condition=condition,
                      status=row.get('status', 'NOT_RUN'), valid=bool(row.get('study_valid')))
        if record['valid']:
            path = Path(row['path'])
            outcome = study_outcome(path, verify_evidence=True) if reverify else row['outcome']
            if not outcome.get('study_valid') or outcome['study_outcome_y'] != row['outcome']['study_outcome_y']:
                raise ValueError('re-derived outcome differs from the sealed receipt: ' + label)
            raw = read_json(path / 'official_grade.json')['official_raw']
            usage = row.get('usage') or {}
            record.update(y=float(outcome['study_outcome_y']), dim1=raw.get('dim1_pass'), max_score=raw.get('max_score'),
                          items=[{k: item.get(k) for k in ('rule', 'delta', 'max_delta')}
                                 for item in raw.get('dim2_items') or []],
                          requests=usage.get('model_requests_including_parent'),
                          tool_calls=usage.get('tool_calls_including_parent'))
        runs.append(record)
    return dict(block=plan['block'], plan_sha256=plan['plan_sha256'], cohort_sha256=cohort['artifact_sha256']), runs


def load_reference(spec):
    ref = spec['pi']['reference_cohort']
    path = ROOT / ref['path']
    if file_digest(path) != ref['sha256']:
        raise ValueError('reference DEV45 cohort changed')
    cohort = read_json(path)
    return [dict(task=label.split('.')[0], y=float(row['outcome']['study_outcome_y']))
            for label, row in cohort['slots'].items() if row.get('study_valid')]


# ---------------------------------------------------------------- estimation

def cells(runs):
    out = {}
    for run in runs:
        if run['valid']:
            out.setdefault((run['task'], run['condition']), []).append(run['y'])
    return out


def contrast(cell_values, tasks, condition, control='O'):
    diffs = []
    for task in tasks:
        a, b = cell_values.get((task, condition)), cell_values.get((task, control))
        if not a or not b:
            return None
        diffs.append(statistics.fmean(a) - statistics.fmean(b))
    return statistics.fmean(diffs)


def bootstrap(cell_values, tasks, conditions, replicates, seed):
    rng = random.Random(seed)
    draws = {c: [] for c in conditions}
    for _ in range(replicates):
        sample = [rng.choice(tasks) for _ in tasks]
        means = {}
        for index, task in enumerate(sample):
            for condition in ('O', *conditions):
                values = cell_values[(task, condition)]
                means[(index, condition)] = statistics.fmean(rng.choice(values) for _ in values)
        for condition in conditions:
            draws[condition].append(statistics.fmean(means[(i, condition)] - means[(i, 'O')] for i in range(len(sample))))
    return draws


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def decide(estimates, intervals, analysis, blocks_done, total_blocks, min_valid):
    fo, (fo_low, fo_high) = estimates['F'], intervals['F']
    best_condition = max(('T', 'V'), key=lambda c: estimates[c])
    best = estimates[best_condition]
    rule = analysis['g1_information_binding']
    g1 = fo >= rule['min_effect'] and fo_low > rule['lower_bound_must_exceed']
    reasons = dict(f_minus_o=fo, f_minus_o_ci95=[fo_low, fo_high], best_existing=best_condition, best_minus_o=best,
                   g1_information_binding=g1, blocks_done=blocks_done, total_blocks=total_blocks,
                   min_valid_runs_in_primary_o_f_cells=min_valid,
                   share_of_gap_closed=(best / fo) if fo > 0 else None)
    if blocks_done < total_blocks:
        futility = analysis['futility']
        if blocks_done >= futility['min_blocks'] and fo_high < futility['f_minus_o_upper_below']:
            return 'stop_futility', reasons
        return 'interim_continue', reasons
    if min_valid < analysis['min_valid_runs_per_cell']:
        return 'insufficient_data', reasons
    if not g1:
        return 'no_go_information_not_binding', reasons
    if best >= analysis['existing_methods_sufficient_share'] * fo:
        return 'stop_existing_methods_sufficient', reasons
    if best >= analysis['g2_existing_methods_insufficient']['max_share_of_f_minus_o'] * fo:
        return 'partial_requires_user_decision', reasons
    return 'go_method_development', reasons


def drift(cell_values, reference, tasks, threshold):
    per_task = {}
    for task in tasks:
        current = cell_values.get((task, 'O'))
        historical = [r['y'] for r in reference if r['task'] == task]
        if current and historical:
            per_task[task] = statistics.fmean(current) - statistics.fmean(historical)
    mean = statistics.fmean(per_task.values()) if len(per_task) == len(tasks) else None
    return dict(per_task=per_task, task_equal_difference=mean,
                flagged=None if mean is None else abs(mean) > threshold, threshold=threshold)


def run_item_scores(run):
    """Per-item achievement in [0, 1]; an unusable deliverable (dim1 failed) scores every item 0."""
    if not run['dim1']:
        return {}
    out = {}
    for item in run['items']:
        value = item_score(item)
        if value is not None:
            out[item['rule']] = value
    return out


def harm(runs, tasks, conditions, rule):
    """Items the original condition usually gets right: does a condition break them?"""
    report = {c: dict(items=0, harmed=0, mean_drop=None) for c in conditions}
    drops = {c: [] for c in conditions}
    for task in tasks:
        base = [run_item_scores(r) for r in runs if r['valid'] and r['task'] == task and r['condition'] == 'O']
        if len(base) < rule['min_o_valid_runs']:
            continue
        names = sorted({name for scores in base for name in scores})
        base_mean = {name: statistics.fmean(s.get(name, 0.0) for s in base) for name in names}
        stable = [name for name in names if base_mean[name] >= rule['items_with_o_pass_rate_at_least']]
        for condition in conditions:
            other = [run_item_scores(r) for r in runs if r['valid'] and r['task'] == task and r['condition'] == condition]
            if not other:
                continue
            for name in stable:
                drop = base_mean[name] - statistics.fmean(s.get(name, 0.0) for s in other)
                drops[condition].append(drop)
                report[condition]['items'] += 1
                report[condition]['harmed'] += int(drop > rule['flag_mean_drop_above'])
    for condition in conditions:
        if drops[condition]:
            report[condition]['mean_drop'] = statistics.fmean(drops[condition])
    return report


def source_of(rule_map, task, item_rule):
    overrides = rule_map.get('item_overrides', {}).get(task, {})
    for name, source in overrides.items():
        if name in item_rule:
            return source
    return rule_map.get('task_overrides', {}).get(task, rule_map['default'])


def source_deductions(runs, tasks, conditions, rule_map):
    """Descriptive per-source lost share of the official score (task-equal average)."""
    per = {}
    for run in runs:
        if not run['valid'] or run['task'] not in tasks:
            continue
        lost = {}
        if not run['dim1'] or not run.get('max_score'):
            lost['gate_or_no_delivery'] = 1.0 - run['y']
        else:
            for item in run['items']:
                maximum, delta = float(item['max_delta']), float(item['delta'])
                value = (maximum - delta) if maximum > 0 else (-delta if maximum < 0 else 0.0)
                if value > 0:
                    key = source_of(rule_map, run['task'], item['rule'])
                    lost[key] = lost.get(key, 0.0) + value / float(run['max_score'])
        per.setdefault((run['task'], run['condition']), []).append(lost)
    out = {}
    for condition in conditions:
        sources = sorted({k for (t, c), rows in per.items() if c == condition for row in rows for k in row})
        out[condition] = {}
        for source in sources:
            values = [statistics.fmean(row.get(source, 0.0) for row in per[(t, condition)])
                      for t in tasks if (t, condition) in per]
            out[condition][source] = statistics.fmean(values) if values else None
    return out


def analyze(blocks, runs, spec, spec_sha, reference=None, provenance=None):
    analysis = spec['analysis']
    primary = spec['tasks']['primary']
    primary_runs = [r for r in runs if r['task'] in primary]
    values = cells(primary_runs)
    census = {}
    for run in runs:
        cell = census.setdefault(f"{run['task']}.{run['condition']}", dict(planned=0, valid=0))
        cell['planned'] += 1
        cell['valid'] += run['valid']
    complete = all(values.get((t, c)) for t in primary for c in ('O', *CONTRASTS))
    estimates = {c: contrast(values, primary, c) for c in CONTRASTS} if complete else None
    result = dict(schema=SCHEMA, spec_sha256=spec_sha, blocks=blocks, census=census, actor_requests=0,
                  analyst_requests=0, estimates=estimates, provenance=provenance or {})
    decision, reasons, intervals = 'insufficient_data', {}, None
    if complete:
        boot = analysis['bootstrap']
        draws = bootstrap(values, primary, CONTRASTS, boot['replicates'], boot['seed'])
        alpha = (1 - boot['interval']) / 2
        intervals = {c: [percentile(draws[c], alpha), percentile(draws[c], 1 - alpha)] for c in CONTRASTS}
        min_valid = min(len(values[(t, c)]) for t in primary for c in ('O', 'F'))
        decision, reasons = decide(estimates, intervals, analysis, len(blocks), spec['blocks']['count'], min_valid)
    result.update(intervals_ci95=intervals, decision=decision, decision_reasons=reasons,
                  cell_means={f'{t}.{c}': statistics.fmean(v) for (t, c), v in sorted(values.items())})
    if reference is not None:
        result['drift'] = drift(values, reference, primary, analysis['drift']['flag_abs_diff'])
    result['harm'] = harm(primary_runs, primary, CONTRASTS, analysis['harm'])
    result['source_deductions'] = source_deductions(primary_runs, primary, ('O', *CONTRASTS), analysis['source_rule'])
    f_only = [r for r in runs if r['task'] in spec['tasks']['f_only'] and r['valid']]
    result['f_only'] = dict(valid=len(f_only), mean=statistics.fmean(r['y'] for r in f_only) if f_only else None,
                            dim1_pass=sum(bool(r['dim1']) for r in f_only))
    usage = {}
    for run in runs:
        if run['valid'] and run.get('requests') is not None:
            usage.setdefault(run['condition'], []).append(run['requests'])
    result['mean_requests'] = {c: statistics.fmean(v) for c, v in usage.items()}
    return seal(result)


def report_markdown(summary):
    lines = ['# Exp1 analysis', '', f"Decision: **{summary['decision']}**", '']
    if summary.get('estimates'):
        lines += ['| Contrast | Estimate | 95% CI |', '|---|---:|---|']
        for c in CONTRASTS:
            low, high = summary['intervals_ci95'][c]
            lines.append(f"| {c}-O | {summary['estimates'][c]:.4f} | [{low:.4f}, {high:.4f}] |")
        lines.append('')
    lines.append('Blocks: ' + ', '.join(str(b['block']) for b in summary['blocks']))
    if 'drift' in summary:
        lines.append(f"O drift vs DEV45: {summary['drift']['task_equal_difference']} (flagged: {summary['drift']['flagged']})")
    lines.append(f"085 F: valid {summary['f_only']['valid']}, mean {summary['f_only']['mean']}, dim1 {summary['f_only']['dim1_pass']}")
    return '\n'.join(lines) + '\n'


INPUTS_SCHEMA = 'acceptance-gap-exp1-analysis-inputs-1'


def export_inputs(phases, spec_path=SPEC):
    """Read sealed block receipts and graded items locally; no computation beyond collection."""
    spec, spec_sha = load_spec(spec_path)
    blocks, runs = [], []
    for phase in phases:
        meta, block_runs = load_block(phase, spec_sha, reverify=False)
        if any(b['block'] == meta['block'] for b in blocks):
            raise ValueError('duplicate block')
        blocks.append(meta)
        runs += block_runs
    return seal(dict(schema=INPUTS_SCHEMA, spec_sha256=spec_sha, blocks=sorted(blocks, key=lambda b: b['block']),
                     runs=runs, reference=load_reference(spec), evidence_reverified=False,
                     note='outcomes taken from sealed receipts; each package was verified by the driver at slot completion'))


def analyze_inputs(inputs, spec, spec_sha):
    verify_seal(inputs)
    if inputs.get('schema') != INPUTS_SCHEMA or inputs.get('spec_sha256') != spec_sha:
        raise ValueError('analysis inputs were not exported under this spec')
    provenance = dict(inputs_sha256=inputs['artifact_sha256'], evidence_reverified=inputs['evidence_reverified'])
    return analyze(inputs['blocks'], inputs['runs'], spec, spec_sha, inputs['reference'], provenance)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--blocks', type=Path, nargs='+')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--spec', type=Path, default=SPEC)
    parser.add_argument('--export-inputs', type=Path, help='write analysis inputs from sealed receipts and stop')
    parser.add_argument('--inputs', type=Path, help='analyze previously exported inputs (no evidence access)')
    args = parser.parse_args(argv)
    if args.export_inputs:
        if not args.blocks or args.export_inputs.exists():
            parser.error('--export-inputs needs --blocks and a new output file')
        value = export_inputs(args.blocks, args.spec)
        args.export_inputs.parent.mkdir(parents=True, exist_ok=True)
        with args.export_inputs.open('x', encoding='utf-8') as handle:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        print(json.dumps(dict(inputs=str(args.export_inputs), inputs_sha256=value['artifact_sha256']), indent=2))
        return
    if not args.output or args.output.exists():
        parser.error('--output must name a new directory; analysis never overwrites')
    spec, spec_sha = load_spec(args.spec)
    if args.inputs:
        summary = analyze_inputs(read_json(args.inputs), spec, spec_sha)
    else:
        if not args.blocks:
            parser.error('--blocks or --inputs required')
        blocks, runs = [], []
        for phase in args.blocks:
            meta, block_runs = load_block(phase, spec_sha)
            if any(b['block'] == meta['block'] for b in blocks):
                parser.error('duplicate block')
            blocks.append(meta)
            runs += block_runs
        summary = analyze(sorted(blocks, key=lambda b: b['block']), runs, spec, spec_sha, load_reference(spec),
                          dict(evidence_reverified=True))
    args.output.mkdir(parents=True)
    write_json(args.output / 'summary.json', summary)
    (args.output / 'report.md').write_text(report_markdown(summary), encoding='utf-8')
    print(json.dumps(dict(decision=summary['decision'], output=str(args.output),
                          summary_sha256=summary['artifact_sha256']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
