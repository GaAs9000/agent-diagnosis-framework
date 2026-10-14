"""Prospective, bounded local R1.1 engineering run; no adaptive task/model search."""
import argparse
import asyncio
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import traceback

from c3_sandbox.evidence._storage import encode
from c3_sandbox.experiment_scheduler import BranchSpec
from c3_sandbox.manifests import canonical_hash
from c3_sandbox.probe_registry import ProbeRegistry
from c3_sandbox.r1.branch_executor import RealBranchExecutor
from c3_sandbox.r1.office_run import run_baseline, SYSTEM_PROMPT
from c3_sandbox.r1.qualification import image_identity, model_fingerprint, TOOL_SCHEMAS
from c3_sandbox.r1.r11_panel import SAMPLING, prepare_panel, run_success_panel
from c3_sandbox.r1.r11_replay import admit_checkpoint, select_engineering_candidates

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('xb') as stream:
        stream.write(encode(value))


def journal_records(root):
    path = root / 'run_journal.jsonl'
    records = [json.loads(line) for line in path.read_bytes().splitlines()] if path.exists() else []
    previous = '0' * 64
    for index, record in enumerate(records):
        body = {key: value for key, value in record.items() if key != 'event_sha256'}
        if record['sequence'] != index or body['previous_event_sha256'] != previous or canonical_hash(body) != record['event_sha256']:
            raise RuntimeError('append-only journal integrity failure')
        previous = record['event_sha256']
    return records


def append_event(root, kind, **payload):
    path = root / 'run_journal.jsonl'
    records = journal_records(root)
    previous = records[-1]['event_sha256'] if records else '0' * 64
    event = {'sequence': len(records), 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
             'kind': kind, 'previous_event_sha256': previous, **payload}
    event['event_sha256'] = canonical_hash(event)
    with path.open('ab') as stream:
        stream.write(encode(event) + b'\n')


@contextmanager
def serial(path):
    with Path(path).open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def verify_freeze(root):
    path = root / 'candidate_order.json'
    plan = read(path)
    events = journal_records(root)
    if not events or events[0]['kind'] != 'candidate_pool_frozen' or events[0]['candidate_order_sha256'] != sha(path):
        raise RuntimeError('missing or changed prospective candidate seal')
    for relative, expected in plan['implementation_lock'].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError('frozen implementation changed: ' + relative)
    manifest = read(plan['manifest_path'])
    if sha(plan['manifest_path']) != plan['manifest_sha256']:
        raise RuntimeError('frozen manifest changed')
    for relative, item in manifest['files'].items():
        if sha(Path(manifest['bundle_root']) / relative) != item['sha256']:
            raise RuntimeError('frozen OfficeVal file changed')
    prerequisites = {
        root / 'source_audit-v2/source_review.json': plan['source_review_sha256'],
        root / 'tool_protocol_audit/gate-v2.json': plan['G0_audit_sha256'],
        root / 'grader-dependency-preflight/report.json': plan['dependency_audit_sha256'],
        root / 'server_observation.json': plan['serving_observation_sha256'],
        Path(plan['model_qualification_path']): plan['model_qualification_sha256'],
        Path(plan['model_qualification_trace']): plan['model_qualification_trace_sha256'],
    }
    if any(sha(path) != expected for path, expected in prerequisites.items()):
        raise RuntimeError('frozen prerequisite evidence changed')
    if image_identity('c3-r1-agent:locked') != plan['agent_image'] or image_identity('c3-r11-grader:locked') != plan['grader_image']:
        raise RuntimeError('pinned runtime image changed')
    plan['candidate_order_sha256'] = sha(path)
    return plan, manifest


def freeze(root):
    if (root / 'candidate_order.json').exists():
        raise FileExistsError('candidate pool already frozen')
    g0 = read(root / 'tool_protocol_audit/gate-v2.json')
    review = read(root / 'source_audit-v2/source_review.json')
    dependency = read(root / 'grader-dependency-preflight/report.json')
    serving = read(root / 'server_observation.json')
    if g0['status'] != 'PASS' or serving['configuration_verified'] is not True:
        raise RuntimeError('G0 or observed local serving prerequisite failed')
    qualified = {item['task_id']: item for item in dependency['cases']}
    tasks = [task for task in review['retained_task_ids'] if qualified[task]['validity'] == 'valid']
    excluded = {task: {'reason': 'official verifier cannot grade pristine public input under frozen runtime',
                       'dependency_report': qualified[task]}
                for task in review['retained_task_ids'] if task not in tasks}
    if not 1 <= len(tasks) <= 12 or tasks != sorted(tasks):
        raise RuntimeError('invalid prospective pool')
    manifest_path = root / 'grader-build-context/frozen/manifest.json'
    manifest = read(manifest_path)
    paths = sorted((ROOT / 'src/c3_sandbox').rglob('*.py')) + sorted((ROOT / 'scripts').glob('*.py'))
    paths += sorted((ROOT / 'tests').glob('*.py')) + sorted((ROOT / 'docker').glob('*.py'))
    paths += sorted((ROOT / 'docker').glob('Dockerfile.r1-*'))
    paths += [ROOT / relative for relative in ('requirements.local-arm64.lock.txt',
        'requirements.model.local-arm64.lock.txt', 'configs/probes.v1.json', 'pyproject.toml')]
    implementation = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    fingerprint = model_fingerprint()
    qualification_path = root / 'tool_protocol_audit/live_qualification/qualification.json'
    qualification = read(qualification_path)
    if fingerprint['sha256'] != qualification['model_fingerprint']:
        raise RuntimeError('qualified model fingerprint drift')
    candidates = []
    for index, task_id in enumerate(tasks):
        task = manifest['tasks'][task_id]
        asset_hashes = {item['name']: item['sha256'] for item in task['input_files']}
        candidates.append({'order': index, 'task_id': task_id, 'split': 'development',
            'source_group_id': 'officeval-assets-' + canonical_hash(asset_hashes),
            'source_group_scope': 'observed public asset/template grouping; research-family independence unestablished',
            'asset_hashes': asset_hashes, 'verifier_sha256': task['verifier_sha256'],
            'expected_output': task['expected_output'], 'dependency_qualified': True,
            'excluded_from_future_confirmation': True})
    plan = {'stage': 'C3-E2-R1.1', 'schema_version': 1, 'candidate_count': len(candidates),
        'candidate_limit': 12, 'candidates': candidates, 'model': qualification['model'],
        'model_fingerprint': fingerprint, 'agent_image': image_identity('c3-r1-agent:locked'),
        'grader_image': image_identity('c3-r11-grader:locked'), 'master_seed': 2026,
        'exploration_seed': 2026,
        'confirmation_seed': int.from_bytes(hashlib.sha256(b'C3-E2-R1.1:2026:confirmation').digest()[:4], 'big'),
        'sampling': dict(SAMPLING, seed=2026), 'system_prompt': SYSTEM_PROMPT, 'tools': TOOL_SCHEMAS,
        'manifest_path': str(manifest_path), 'manifest_sha256': sha(manifest_path),
        'model_qualification_path': str(qualification_path),
        'model_qualification_sha256': sha(qualification_path),
        'model_qualification_trace': str(qualification_path.parent / 'agent/model_calls.jsonl'),
        'model_qualification_trace_sha256': qualification['model_trace_sha256'],
        'source_review_sha256': sha(root / 'source_audit-v2/source_review.json'),
        'G0_audit_sha256': sha(root / 'tool_protocol_audit/gate-v2.json'),
        'dependency_audit_sha256': sha(root / 'grader-dependency-preflight/report.json'),
        'serving_observation_sha256': sha(root / 'server_observation.json'),
        'excluded_after_dependency_qualification': excluded,
        'implementation_lock': implementation, 'implementation_identity': canonical_hash(implementation),
        'physical_lock_path': str(ROOT / 'runs/r11-physical.serial.lock'),
        'selection_rule': 'first S and T in the frozen order; all frozen originals attempted once',
        'S_checkpoint_rule': 'first qualified complete-tool boundary in valid fully successful original',
        'T_checkpoint_rule': 'nearest qualified complete-tool boundary before earliest subsequent actual original native dispatch',
        'continuation_failure_rule': 'one seed-2026 Original continuation; no new call => G2 INCOMPLETE, no alternate seed/probe',
        'no_outcome_based_replacements': True, 'D6_D8_open': False,
        'completion_definition': 'unchanged: valid protocol and final official complete assessed positive rules with no negative hits'}
    write(root / 'candidate_order.json', plan)
    for relative in implementation:
        target = root / 'source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    (root / 'candidate_order.json').chmod(0o444)
    append_event(root, 'candidate_pool_frozen', candidate_order_sha256=sha(root / 'candidate_order.json'),
                 task_ids=tasks, implementation_identity=plan['implementation_identity'])
    print(json.dumps({'frozen': True, 'task_ids': tasks, 'excluded_for_dependency': list(excluded)}), flush=True)


async def originals(root):
    plan, manifest = verify_freeze(root)
    qualification = read(plan['model_qualification_path'])
    results = []
    with serial(plan['physical_lock_path']):
        for candidate in plan['candidates']:
            verify_freeze(root)
            task = candidate['task_id']
            folder = root / 'originals' / task
            if folder.exists():
                raise FileExistsError('original attempt already exists; no automatic retry')
            append_event(root, 'original_started', order=candidate['order'], task_id=task,
                         candidate_order_sha256=plan['candidate_order_sha256'])
            print('START original', candidate['order'], task, flush=True)
            try:
                result = await run_baseline(task, manifest, plan['agent_image'], plan['grader_image'],
                    qualification, folder, seed=2026, max_turns=10)
            except Exception as error:
                folder.mkdir(parents=True, exist_ok=True)
                result = {'task_id': task, 'is_synthetic': False, 'execution_valid': False,
                    'task_completed': None, 'target_resolved': None, 'regression_free': None,
                    'outcome': 'execution_error', 'error_type': type(error).__name__, 'error': str(error),
                    'traceback': traceback.format_exc(), 'output_dir': str(folder), 'checkpoints': [],
                    'agent_result': {'protocol_status': 'not_evaluated'}}
                write(folder / 'attempt_failure.json', result)
            results.append(result)
            identity = sha(folder / ('result.json' if (folder / 'result.json').exists() else 'attempt_failure.json'))
            summary = {key: result.get(key) for key in ('task_id', 'execution_valid', 'task_completed', 'outcome')}
            append_event(root, 'original_finished', order=candidate['order'], result_sha256=identity, **summary)
            print('END original', json.dumps(summary), flush=True)
    selection = select_engineering_candidates(results)
    write(root / 'original_results_index.json', {'candidate_order_sha256': plan['candidate_order_sha256'],
        'attempted_count': len(results), 'tasks': [r['task_id'] for r in results],
        'outcome_denominator': len(results), 'result_directories': [r['output_dir'] for r in results]})
    write(root / 'coverage_selection.json', selection)
    append_event(root, 'original_pool_finished', attempted_count=len(results),
        selection_sha256=sha(root / 'coverage_selection.json'),
        S_task=selection['S_candidate']['task_id'] if selection['S_candidate'] else None,
        T_task=selection['T_candidate']['task_id'] if selection['T_candidate'] else None)


def verified_selection(root, plan):
    events = journal_records(root)
    starts = [event for event in events if event['kind'] == 'original_started']
    finishes = [event for event in events if event['kind'] == 'original_finished']
    expected = [(candidate['order'], candidate['task_id']) for candidate in plan['candidates']]
    if ([(event['order'], event['task_id']) for event in starts] != expected or
            [(event['order'], event['task_id']) for event in finishes] != expected):
        raise RuntimeError('original attempt denominator/order differs from frozen pool')
    results = []
    for candidate, finished in zip(plan['candidates'], finishes):
        folder = root / 'originals' / candidate['task_id']
        path = folder / ('result.json' if (folder / 'result.json').exists() else 'attempt_failure.json')
        if sha(path) != finished['result_sha256']:
            raise RuntimeError('original result changed after journal seal')
        results.append(read(path))
    completed = [event for event in events if event['kind'] == 'original_pool_finished']
    if len(completed) != 1 or completed[0]['selection_sha256'] != sha(root / 'coverage_selection.json'):
        raise RuntimeError('coverage selection changed after prospective journal seal')
    selections = read(root / 'coverage_selection.json')
    if selections != select_engineering_candidates(results):
        raise RuntimeError('coverage selection differs from fixed first-S/T rule')
    return selections


def continue_coverage(root):
    plan, manifest = verify_freeze(root)
    selections = verified_selection(root, plan)
    for kind in ('T', 'S'):
        selection = selections[kind + '_candidate']
        folder = root / ('replay_tool_continuation' if kind == 'T' else 'success_preparation')
        folder.mkdir()
        if selection is None:
            write(folder / 'coverage_gap.json', {'gate': 'G2' if kind == 'T' else 'G3',
                'status': 'INCOMPLETE', 'reason': 'no candidate met fixed original coverage requirements',
                'candidate_order_sha256': plan['candidate_order_sha256']})
            if kind == 'S':
                for name in ('e2_s_exploration', 'e2_s_confirmation', 'scheduler'):
                    (root / name).mkdir()
                    write(root / name / 'not_run.json', {'reason': 'no original successful S_candidate', 'status': 'INCOMPLETE'})
            append_event(root, 'coverage_unavailable', kind_requested=kind)
            continue
        baseline = read(root / 'originals' / selection['task_id'] / 'result.json')
        write(folder / 'selection.json', selection)
        append_event(root, 'coverage_started', kind_requested=kind, task_id=baseline['task_id'],
                     checkpoint_id=selection['checkpoint_id'])
        try:
            admission = admit_checkpoint(baseline, selection['reference'], folder / 'admission',
                                         Path(plan['model_qualification_trace']))
            line = 'E2-S' if baseline['task_completed'] else 'E2-F'
            if kind == 'S':
                prepared = prepare_panel(baseline, selection, folder / 'probes', line)
                result = run_success_panel(baseline, selection, manifest, plan, admission, prepared, root)
                write(folder / 'panel_result.json', result)
            else:
                # T is an unassisted Original continuation. No unused probe,
                # document parser or matching requirement may gate this path.
                original = {'probe_id': 'original', 'payload': {}, 'rendered_text': '',
                    'rendered_digest': hashlib.sha256(b'').hexdigest(), 'token_count': 0,
                    'public_evidence': None, 'matching_audit': {'not_applicable': 'no added observation'}}
                task_identity = canonical_hash({'task': manifest['tasks'][baseline['task_id']], 'plan': plan,
                    'selection': selection, 'condition': 'fresh Original only, no helper observation',
                    'scope': 'engineering tool-continuation qualification'})
                baseline = {**baseline, 'task_identity': task_identity}
                grader_id = canonical_hash({'image': plan['grader_image'], 'manifest': manifest,
                                            'implementation': plan['implementation_identity']})
                registry = ProbeRegistry.from_file(ROOT / 'configs/probes.v1.json')
                cp = selection['reference']
                spec = BranchSpec(baseline['task_id'] + '-R1.1-T', task_identity, line, baseline['system_identity'],
                    cp['checkpoint_id'], cp['checkpoint_id'], 'original', registry.identity('original'),
                    dict(SAMPLING, seed=2026), 0, cp['runtime_identity'], grader_id, False, 'exploration')
                write(folder / 'frozen_continuation.json', asdict(spec))
                executor = RealBranchExecutor(admission['replay'], folder / 'branches', baseline, manifest,
                    plan['agent_image'], plan['grader_image'], {'original': original},
                    baseline['model_qualification'], grader_id)
                with serial(plan['physical_lock_path']):
                    outcome = executor(spec)
                count = outcome.result.get('tool_dispatch_count', 0)
                result = {'gate': 'G2', 'status': 'PASS' if count > 0 else 'INCOMPLETE',
                    'reason': 'new model-generated tool call dispatched after verified restoration' if count else 'tool_continuation_not_observed',
                    'new_tool_calls': count, 'branch_key': spec.key,
                    'outcome': outcome.result, 'physical_cost': outcome.physical_cost,
                    'original_suffix_replayed': False, 'alternate_seed_or_probe_attempted': False}
                write(folder / 'result.json', result)
            append_event(root, 'coverage_finished', kind_requested=kind, result_identity=canonical_hash(result))
            print('COVERAGE', kind, json.dumps(result), flush=True)
        except Exception as error:
            failure = {'status': 'INVALID', 'error_type': type(error).__name__, 'error': str(error),
                       'traceback': traceback.format_exc(), 'no_automatic_retry': True}
            write(folder / 'failure.json', failure)
            append_event(root, 'coverage_invalid', kind_requested=kind, error_type=type(error).__name__, error=str(error))
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'originals', 'coverage'))
    parser.add_argument('--run-root', type=Path, default=ROOT / 'runs/e2-r1-1')
    args = parser.parse_args()
    root = args.run_root.absolute()
    if args.action == 'freeze':
        freeze(root)
    elif args.action == 'originals':
        asyncio.run(originals(root))
    else:
        continue_coverage(root)


if __name__ == '__main__':
    main()
