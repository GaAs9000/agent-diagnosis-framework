#!/usr/bin/env python3
"""Build a DEV9-only, unreviewed whole-task catalog without executing verifiers.

All unreviewed rules remain ambiguous: the public half is the original task
instruction and the hidden half is verbatim rubric/verifier text. Historical
labels remain metadata; ten existing limited approvals retain their original signatures only at exact source identities.
Grade records are read only AFTER construction to check alias coverage; they
never select requirements, classifications, or intervention text.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
L1 = Path('L1_sandbox/C3_Experimental_Sandbox')
L2 = Path('L2_Behavioral Event Localization/L2_Sandbox')
DEV9 = tuple(f'officeval_{n}' for n in ('003', '004', '010', '033', '050', '060', '070', '085', '090'))
DEFAULT_OUTPUT = L2 / 'validation/d0_human_waiver_20261018/requirements_catalog.json'
SCORE = re.compile(r'^[+-]\d+(?:\s*[：:]\s*|\s+)')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_score(text: str) -> str:
    return SCORE.sub('', text).strip()


def verifier_aliases(text: str) -> dict[str, list[dict]]:
    """Extract literal scoring descriptions using AST only; never import code."""
    tree = ast.parse(text)
    found: dict[str, list[dict]] = {}

    def add(value: str, node: ast.AST, kind: str) -> None:
        value = re.split(r'\s*[✓✗]', value, maxsplit=1)[0].strip()
        if not value or not re.search(r'[\u3400-\u9fff]', value):
            return
        # Runtime parsers sometimes retain signed prefixes (033), sometimes strip.
        for alias in dict.fromkeys((value, clean_score(value))):
            if not alias:
                continue
            citation = {'line_start': node.lineno, 'line_end': node.end_lineno, 'kind': kind,
                        'quote': ast.get_source_segment(text, node)}
            if citation not in found.setdefault(alias, []):
                found[alias].append(citation)

    def literals(node: ast.AST, kind: str) -> None:
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                if len(child.value) >= 6:
                    add(child.value, child, kind)

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and SCORE.match(node.value):
            add(node.value, node, 'signed_scoring_literal')
        elif isinstance(node, ast.JoinedStr):
            prefix = ''
            for part in node.values:
                if not isinstance(part, ast.Constant) or not isinstance(part.value, str):
                    break
                prefix += part.value
            # Only complete descriptions preceding a literal result marker.
            if SCORE.match(prefix) and re.search('[✓✗]', prefix):
                add(prefix, node, 'scoring_fstring_literal_prefix')
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)]
            if any('RULE' in n or 'CHECK' in n for n in names):
                literals(node.value, 'scoring_table_literal')
            elif any(n.lower() in {'rule', 'rule_text', 'title', 'label', 'desc'} for n in names):
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    add(node.value.value, node.value, 'named_scoring_description')
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == 'rule':
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        add(value.value, value, 'rule_dictionary_literal')
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'item' and len(node.args) > 1:
                if isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                    add(node.args[1].value, node.args[1], 'item_description_literal')
            for kw in node.keywords:
                if kw.arg in {'title', 'label', 'name'} and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    add(kw.value.value, kw.value, 'named_result_literal')
    return found


def build_catalog(root: Path = ROOT, *, validate_grades: bool = True,
                  grades_dir: Path | None = None) -> dict:
    def load(rel: Path) -> dict:
        return json.loads((root / rel).read_text())

    legacy_rel = L2 / 'reports/v3_p0a_audit_20261017.json'
    legacy = load(legacy_rel)['dev45']['delivered_item_wrong_rubric_source']['task_rule_audit']
    approvals_rel = L2 / 'validation/d0_20261018/human_review/approved_batch01.json'
    approvals = load(approvals_rel)
    review_rel = L2 / 'validation/d0_20261018/human_review/rule_review_63.json'
    reviewed_rows = {r['sample_id']: r for r in load(review_rel)['rows']}
    metadata = {
        'schema_version': 'dev9-automatic-requirements-v1', 'status': 'automatic_unreviewed',
        'human_review_policy': 'waived_by_user', 'review_complete': False,
        'reviewer': '', 'reviewed_at': '',
        'construction_scope': list(DEV9),
        'construction_uses_outcomes': False,
        'classification_policy': 'Unreviewed categories ambiguous; ten existing approvals retained only at exact sampled text or verified rubric pointer/hash/quote. No new images inspected or precise rules promoted.',
        'prior_human_approvals': {'path': str(approvals_rel), 'sha256': sha(root / approvals_rel), 'record': approvals,
                                  'signed_rows_source': {'path': str(review_rel), 'sha256': sha(root / review_rel)},
                                  'identity_mappings': []},
        'legacy_label_source': {'path': str(legacy_rel), 'sha256': sha(root / legacy_rel)},
        'sources': [], 'coverage': {},
        'builder_attestation': {'path': str(Path(__file__).resolve().relative_to(ROOT)), 'sha256': sha(Path(__file__))},
        'limitations': [
            'Static descriptions are a sourced catalog, not proof every verifier branch is active.',
            'Rubric 090 retains rules documented as removed by verifier; retained as rubric coverage, not claimed executable.',
            'Verbatim scoring descriptions can describe a penalty condition; no automatic polarity inversion or invented repair instruction.',
            'Reference images are not inspected; public text is restricted to the original instruction.',
            'Dim1 gate is a synthetic aggregate endpoint, covered by full dim1 rubric inventory rather than a hidden task rule.',
        ],
    }
    output = {'metadata': metadata}
    for task in DEV9:
        tp = L1 / f'benchmarks/officeval/data/tasks/{task}.json'
        rp = L1 / f'benchmarks/officeval/data/rubrics/{task}.json'
        vp = L1 / f'third_party/OmegaUse-OfficeVal/verifiers/{task}_verifier.py'
        instruction = load(tp)['instruction'].strip()
        if not instruction:
            raise ValueError(f'{task}: empty public instruction')
        for rel in (tp, rp, vp):
            metadata['sources'].append({'path': str(rel), 'sha256': sha(root / rel)})
        old = legacy.get(f'officeval:{task}', {})
        entries: dict[str, dict] = {}

        def add(rule: str, citation: dict, kind: str) -> None:
            info = entries.setdefault(rule, {
                'source_category': 'ambiguous', 'status': 'automatic_unreviewed', 'lookup_only': True,
                'visible_requirement': instruction,
                'exact_verifier_specification': rule,
                'review_complete': False, 'reviewer': '', 'reviewed_at': '',
                'classification_basis': 'Conservative unreviewed split: whole public instruction versus literal scoring evidence.',
                'source_citation': [], 'source_kinds': [],
                'public_instruction_citation': {'path': str(tp), 'sha256': sha(root / tp), 'json_pointer': '/instruction', 'quote': instruction},
                'prior_proposed_category': (old.get(rule) or old.get(clean_score(rule)) or {}).get('source_category'),
            })
            if citation not in info['source_citation']:
                info['source_citation'].append(citation)
            if kind not in info['source_kinds']:
                info['source_kinds'].append(kind)
            if kind == 'rubric':
                # Keep signed rubric wording in the actual prompt. Stripped
                # penalty descriptions are lookup aliases, never repair orders.
                info['lookup_only'] = False

        rubric = load(rp)['rubrics']
        rows = []
        for dimension, rules in rubric.items():
            for index, rule in enumerate(rules):
                rule = str(rule).strip()
                cite = {'path': str(rp), 'sha256': sha(root / rp), 'json_pointer': f'/rubrics/{dimension}/{index}', 'quote': rule}
                add(rule, cite, 'rubric')
                add(clean_score(rule), cite, 'rubric_score_stripped_alias')
                rows.append({'dimension': dimension, 'rubric_index': index, 'rule': rule, 'covered': rule in entries})
        aliases = verifier_aliases((root / vp).read_text())
        for alias, citations in sorted(aliases.items()):
            for citation in citations:
                add(alias, {'path': str(vp), 'sha256': sha(root / vp), **citation}, 'verifier_static_alias')
        for approval in approvals['rows']:
            if approval['task_id'] != task:
                continue
            signed = reviewed_rows[approval['sample_id']]
            if (signed['task_id'] != task or signed['rule_as_sampled'] != approval['rule_as_sampled']
                    or signed['human_label'] != approval['human_label']
                    or not signed['reviewer'] or not signed['reviewed_at']):
                raise ValueError('Existing human approval identity/signature mismatch')
            mapping = {'sample_id': approval['sample_id'], 'task_id': task, 'mapped_rules': [], 'unmapped': []}
            exact_targets = []
            if approval['rule_as_sampled'] in entries:
                exact_targets.append(approval['rule_as_sampled'])
            else:
                mapping['unmapped'].append('sample_alias_not_extracted')
            evidence = signed['rubric_evidence']
            # A saved semantic pointer may map differently worded aliases. Require
            # the exact original rubric bytes AND pointed quote; never fuzzy-match.
            pointed = load(rp)
            for part in evidence['json_pointer'].strip('/').split('/'):
                pointed = pointed[int(part)] if isinstance(pointed, list) else pointed[part]
            if (evidence['path'] == str(rp) and evidence['sha256'] == sha(root / rp)
                    and evidence['quote'] == pointed and pointed in entries):
                exact_targets.extend([pointed, clean_score(pointed)])
            else:
                mapping['unmapped'].append('rubric_identity_mismatch')
            for rule in dict.fromkeys(exact_targets):
                info = entries[rule]
                info.update({
                    'source_category': signed['human_label'], 'status': 'previously_human_approved',
                    'review_complete': True, 'reviewer': signed['reviewer'], 'reviewed_at': signed['reviewed_at'],
                    'visible_requirement': approval['public_visible_text_exact_quote'],
                    'exact_verifier_specification': approval['precise_hidden_or_unverified_segment'] or rule,
                    'classification_basis': 'Existing limited human approval preserved at exact source identity; no new signature.',
                    'approval_reference': {'path': str(approvals_rel), 'sha256': sha(root / approvals_rel),
                                           'sample_id': approval['sample_id'], 'scope': approvals['approval_scope']},
                })
                mapping['mapped_rules'].append(rule)
            metadata['prior_human_approvals']['identity_mappings'].append(mapping)
        output[task] = dict(sorted(entries.items()))
        metadata['coverage'][task] = {'rubric_rows': rows, 'rubric_row_count': len(rows),
                                      'verifier_static_alias_count': len(aliases), 'catalog_rule_count': len(entries)}
    # Observations cannot mutate the constructed catalog. Read every row, not only failures.
    if validate_grades:
        diag = Path(grades_dir) if grades_dir is not None else L2 / 'diagnoses/d0_20261018'
        if not diag.is_absolute():
            diag = root / diag
        diag = diag.resolve()
        metadata['grade_alias_validation_directory'] = str(diag)
        for task in DEV9:
            observed: set[str] = set()
            validation_sources = []
            paths = sorted(diag.glob(f'{task}.r*/grades.jsonl'))
            if not paths:
                raise ValueError(f'{task}: missing grade validation files in {diag}')
            grade_rows = 0
            for path in paths:
                try:
                    source_path = str(path.relative_to(root))
                except ValueError:
                    source_path = str(path)
                validation_sources.append({'path': source_path, 'sha256': sha(path)})
                file_rows = 0
                for line in path.read_text().splitlines():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f'{path}: grade row must be an object')
                    file_rows += 1
                    observed.update(row.get('deductions', {}))
                if not file_rows:
                    raise ValueError(f'{path}: empty grade validation file')
                grade_rows += file_rows
            missing = sorted(observed - set(output[task]) - {'dim1_gate'})
            metadata['coverage'][task]['observed_grade_alias_validation'] = {
                'observed_distinct_aliases': sorted(observed), 'uncovered': missing,
                'complete': not missing, 'used_for_construction': False, 'validation_sources': validation_sources,
                'grade_row_count': grade_rows,
            }
        metadata['grade_alias_validation_complete'] = all(v['observed_grade_alias_validation']['complete'] for v in metadata['coverage'].values())
    metadata['effective_review_counts'] = {
        'human_approval_decisions': len(approvals['rows']),
        'previously_approved_alias_entries': sum(info['status'] == 'previously_human_approved' for task in DEV9 for info in output[task].values()),
        'automatic_unreviewed_entries': sum(info['status'] == 'automatic_unreviewed' for task in DEV9 for info in output[task].values()),
    }
    metadata['construction_sha256'] = hashlib.sha256(json.dumps({t: output[t] for t in DEV9}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / DEFAULT_OUTPUT)
    parser.add_argument('--grades-dir', type=Path,
                        help='Grade alias validation directory; defaults to legacy diagnoses/d0_20261018.')
    args = parser.parse_args()
    doc = build_catalog(grades_dir=args.grades_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'grade_alias_validation_complete': doc['metadata']['grade_alias_validation_complete'],
                      'uncovered': {t: v['observed_grade_alias_validation']['uncovered'] for t, v in doc['metadata']['coverage'].items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
