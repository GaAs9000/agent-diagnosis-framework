"""Read-only DEV45 export through the evidence-pinned OfficeVal CSV reporter.

Run with L1 and L2 src directories on PYTHONPATH. This invokes no evaluator,
Actor, Analyst, tool, container, or benchmark batch runner.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import re
import shutil
import sys
import types
from pathlib import Path

from c3_sandbox.l1.package import verify_package
from l2_sandbox.v3_execution import digest, file_digest, read_json, verify_seal, write_json


SOURCE_COMMIT = '5f3da616ac6a0bc30ca35e9678f8f1eb255bd91a'
DEV_IDS = frozenset(('003', '004', '010', '033', '050', '060', '070', '085', '090'))
SOURCE_FILES = ('core/config.py', 'core/submission_service.py',
                'core/result_store.py', 'core/csv_reporter.py')
SCOPE = ('DEV9 subset x5; official verifier + CSV projection; '
         'official full batch runner not used')


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_reporter(source: Path, output: Path):
    # Never import the benchmark's core/__init__.py or use a preloaded core.
    package_name = '_officeval_export_' + file_digest(source / 'core/csv_reporter.py')
    names = [package_name] + [package_name + '.' + Path(name).stem for name in SOURCE_FILES]
    _require(not any(name in sys.modules for name in names), 'reporter module already loaded')
    package = types.ModuleType(package_name)
    package.__path__ = [str(source / 'core')]
    sys.modules[package_name] = package
    try:
        modules = []
        for name, relative in zip(names[1:], SOURCE_FILES):
            spec = importlib.util.spec_from_file_location(name, source / relative)
            _require(spec is not None and spec.loader is not None, 'cannot load pinned reporter')
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            # Prevent Python from writing __pycache__ inside sealed evidence.
            exec(compile((source / relative).read_bytes(), str(source / relative), 'exec'), module.__dict__)
            modules.append(module)
        modules[0].RESULTS_DIR = output
        return modules[3], modules[2]
    finally:
        for name in reversed(names):
            sys.modules.pop(name, None)


def _completion(raw: dict, reporter) -> float | None:
    # Exact unrounded numeric counterpart of the pinned renderer. Its actual
    # string stays exclusively owned by the official code in summary.csv.
    rendered = reporter._completion_rate(raw)
    if rendered == '':
        return None
    if raw.get('status') == 'ok' and raw.get('dim1_pass') is False:
        return 0.0
    total, maximum = raw['total_score'], raw['max_score']
    _require(math.isfinite(total) and math.isfinite(maximum), 'non-finite official score')
    return max(0, total) / maximum


def export(cohort_path: Path, runs: Path, output: Path) -> dict:
    _require(not output.exists() and not output.is_symlink(), 'output must be a new directory')
    _require(runs.is_dir() and not runs.is_symlink(), 'runs must be a real directory')
    runs = runs.resolve()
    output = output.resolve()
    _require(not output.is_relative_to(runs), 'output cannot be inside the evidence tree')
    cohort_hash = file_digest(cohort_path)
    cohort = read_json(cohort_path)
    verify_seal(cohort)
    slots = cohort.get('slots')
    _require(isinstance(slots, dict) and len(slots) == 45, 'expected sealed DEV45 cohort')
    _require(set(slots) == {f'officeval_{number}.r{repeat:02d}'
                           for number in DEV_IDS for repeat in range(5)},
             'cohort must contain exactly the frozen nine development tasks')
    groups = {f'r{i:02d}': [] for i in range(5)}
    bindings, source_hashes, source = [], None, None
    # Verify every input before creating the output directory or loading code.
    for label, record in sorted(slots.items()):
        _require(record.get('study_valid') is True
                 and (record.get('outcome') or {}).get('study_valid') is True,
                 f'completed study-valid DEV45 slot required: {label}')
        match = re.fullmatch(r'officeval_(\d{3})\.(r0[0-4])', label)
        _require(match is not None, f'invalid slot label: {label}')
        number, repeat = match.groups()
        path = runs / label
        _require(not path.is_symlink() and path.is_dir() and path.resolve().parent == runs,
                 f'not a direct evidence child: {label}')
        _require(Path(record['path']).name == label and record.get('task_id') == 'officeval_' + number,
                 f'cohort path/task mismatch: {label}')
        index_hash, manifest_hash = record['index_file_sha256'], record['manifest_sha256']
        _require(file_digest(path / 'manifest.json') == manifest_hash, f'manifest binding failed: {label}')
        verified = verify_package(path, expected_index_sha256=index_hash)
        _require(verified.get('verified') is True, f'package verification failed: {label}: {verified.get("errors")}')
        manifest = read_json(path / 'manifest.json')
        _require(manifest['episode_id'] == record['episode_id']
                 and manifest['task']['task_id'] == record['task_id']
                 and manifest['task']['benchmark'] == 'officeval', f'episode/task mismatch: {label}')
        _require(manifest['task']['provenance']['source_commit'] == SOURCE_COMMIT,
                 f'official source revision mismatch: {label}')
        _require(manifest['config_sha256'] == record['config_sha256']
                 == digest(read_json(path / 'raw/resolved_config.json')), f'config binding failed: {label}')
        inventory = {item['path']: item['sha256'] for item in read_json(path / 'evidence_index.json')['files']}
        current_hashes = {name: inventory['benchmark/source/' + name] for name in SOURCE_FILES}
        _require(all(file_digest(path / 'benchmark/source' / name) == value
                     for name, value in current_hashes.items()), f'reporter index binding failed: {label}')
        if source_hashes is None:
            source_hashes, source = current_hashes, path / 'benchmark/source'
        _require(current_hashes == source_hashes, f'CSV reporter differs across slots: {label}')
        grade = read_json(path / 'official_grade.json')
        raw = grade['official_raw']
        _require(isinstance(raw, dict) and raw.get('id') == number, f'official result task mismatch: {label}')
        raw_file = path / 'grading/worker-output/official.json'
        _require(file_digest(raw_file) == grade['official_raw_sha256'] and read_json(raw_file) == raw,
                 f'raw grade file/content mismatch: {label}')
        groups[repeat].append((number, raw, record, grade))
        bindings.append(dict(slot=label, episode_id=record['episode_id'], index_sha256=index_hash,
                             manifest_sha256=manifest_hash, config_sha256=record['config_sha256'],
                             official_grade_file_sha256=file_digest(path / 'official_grade.json'),
                             official_raw_sha256=grade['official_raw_sha256'],
                             official_raw_content_sha256=digest(raw), source_files_sha256=current_hashes))
    ids = [{row[0] for row in values} for values in groups.values()]
    _require(all(len(values) == 9 for values in groups.values())
             and all(group == ids[0] and len(group) == 9 for group in ids), 'expected same nine unique task IDs in five jobs')
    output.mkdir(parents=True, exist_ok=False)
    reporter, result_store = _load_reporter(source, output)
    values, errors, uncomputable, zero_delivery, raw_errors, normalizations = [], [], [], [], [], []
    for repeat, rows in groups.items():
        normalized_rows = []
        for number, raw, record, grade in rows:
            label = f'officeval_{number}.{repeat}'
            process = grade.get('process') or {}
            native = result_store.build_system_result(
                number, raw, raw.get('status'), raw.get('error'),
                process.get('duration_seconds'), process.get('started_at'), retried=False)
            # The upstream constructor uses now(); this export is not a new
            # grading run. Bind timing to the original process, or leave null.
            native['finished_at'] = process.get('finished_at')
            normalized_rows.append(native)
            raw_copy = output / repeat / 'raw_verifier_results' / (number + '.json')
            raw_copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(runs / label / 'grading/worker-output/official.json', raw_copy)
            write_json(output / repeat / (number + '.json'), native)
            changed = {key: {'raw': raw.get(key), 'native': native.get(key)}
                       for key in sorted(set(raw) | set(native))
                       if raw.get(key) != native.get(key) or (key in raw) != (key in native)}
            normalizations.append(dict(slot=label, raw_verifier_content_sha256=digest(raw),
                                       raw_verifier_file_sha256=grade['official_raw_sha256'],
                                       native_result_sha256=digest(native), changed_fields=changed,
                                       timing_origin='original grade.process duration_seconds/started_at/finished_at; absent values null',
                                       retried_origin='false: this export does not retry grading; Actor retries are unrelated'))
            value = _completion(native, reporter)
            if raw.get('status') != 'ok' or raw.get('error'):
                raw_errors.append(label)
            if native.get('status') != 'ok' or native.get('error'):
                errors.append(label)
            if value is None:
                uncomputable.append(label)
            else:
                values.append(value)
                # Only upstream normalization decides whether a missing
                # deliverable becomes a normal dim1-failed zero.
                if native.get('status') == 'ok' and native.get('dim1_pass') is False:
                    zero_delivery.append(label)
            outcome = record.get('outcome') or {}
            if outcome.get('study_valid') is True and value is not None:
                _require(math.isclose(value, outcome['study_outcome_y'], rel_tol=0, abs_tol=1e-12),
                         f'official/study completion discrepancy: {label}')
        reporter.generate_csv(repeat, normalized_rows)
    _require(len(values) == 45 and not uncomputable and not errors,
             'official alignment incomplete: expected 45 numeric normal results; no reduced denominator allowed')
    summary = dict(scope=SCOPE, aggregation_origin='study_defined_arithmetic_mean',
                   mean_official_completion_rate=sum(values) / len(values) if values else None,
                   total_rows=45, completion_rate_denominator=len(values),
                   denominator_policy='all numeric native postprocessed official completion rates; blank rates excluded and listed; raw status/error preserved separately',
                   official_status_error_or_error_text_slots=errors,
                   raw_verifier_error_slots=raw_errors,
                   uncomputable_completion_rate_slots=uncomputable,
                   normal_dim1_failed_zero_count=len(zero_delivery), normal_dim1_failed_zero_slots=zero_delivery,
                   raw_errors_are_not_automatically_study_invalid=True)
    write_json(output / 'study_summary.json', summary)
    write_json(output / 'evidence_bindings.json', dict(cohort_sha256=cohort_hash, slots=bindings))
    write_json(output / 'normalization_provenance.json', dict(
        normalization_scope='upstream result_store.build_system_result postprocessing; official full batch not executed',
        slots=normalizations))
    _require(file_digest(cohort_path) == cohort_hash, 'cohort changed during export')
    receipt = dict(scope=SCOPE, source_commit=SOURCE_COMMIT, source_files_sha256=source_hashes,
                   cohort_file_sha256=cohort_hash, cohort_artifact_sha256=cohort['artifact_sha256'],
                   actor_requests=0, analyst_requests=0, grader_calls=0, tool_calls=0,
                   official_csv_jobs=list(groups), total_rows=45,
                   outputs_sha256={str(p.relative_to(output)): file_digest(p)
                                   for p in sorted(output.rglob('*')) if p.is_file()})
    write_json(output / 'alignment_receipt.json', receipt)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = export(args.cohort, args.runs, args.output)
    print(f'Exported {result["total_rows"]} official rows in five jobs to {args.output}')


if __name__ == '__main__':
    main()
