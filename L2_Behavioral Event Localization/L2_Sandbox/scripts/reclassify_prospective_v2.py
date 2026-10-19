#!/usr/bin/env python3
"""Read-only post-hoc re-audit of saved comparisons. Never starts a tool bridge."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from collections import Counter

L2 = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(L2 / 'src'), str(L2.parents[1] / 'L1_sandbox/C3_Experimental_Sandbox/src'), str(L2 / 'scripts')]
import replay_prospective_tools as replay
from l2_sandbox.metadata_contract import VERSION, classify_metadata, validate_projected_report


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reaudit(*, plan, result, forensics, policy, authorization, cohort, freeze,
            logical_root, physical_root):
    """Recompute proof from saved raw files; return report without filesystem writes."""
    logical, physical = Path(logical_root), Path(physical_root)
    def mapped(value):
        path = Path(value)
        if not path.is_absolute():
            path = physical / path
        if path.is_relative_to(logical):
            path = physical / path.relative_to(logical)
        if path.is_symlink() or not path.resolve().is_relative_to(physical.resolve()):
            raise ValueError('input escapes explicit physical root')
        return path
    refs = {}
    def load(value, expected=None):
        path = mapped(value)
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError('bound input changed: ' + str(path))
        return json.loads(path.read_text())
    for key, value in dict(plan=plan, result=result, forensics=forensics, policy=policy,
                           authorization=authorization, cohort=cohort, freeze=freeze).items():
        path = mapped(value)
        refs[key] = {'path': str(path), 'sha256': sha(path), 'size_bytes': path.stat().st_size}
    policy_doc = load(policy); auth_doc = load(authorization)
    if (policy_doc.get('policy') != VERSION or policy_doc.get('post_hoc_forensic_extension') is not True
            or policy_doc.get('reuse_existing_evidence_only') is not True
            or auth_doc.get('authorization_source') != 'direct_user_request_attachment'
            or auth_doc.get('adoption_scope', {}).get('policy') != VERSION
            or auth_doc.get('adoption_scope', {}).get('post_hoc_forensic_extension') is not True
            or auth_doc.get('contract', {}).get('sha256') != refs['policy']['sha256']):
        raise ValueError('explicit bounded v2 policy/authorization missing')
    attachment = mapped(authorization).parent / auth_doc['request_attachment']['path']
    if attachment.is_symlink() or sha(attachment) != auth_doc['request_attachment']['sha256']:
        raise ValueError('direct user request attachment changed')
    frozen = load(plan); aggregate = load(result)
    if aggregate['plan_sha256'] != sha(mapped(plan)):
        raise ValueError('v1 result is not bound to v1 plan')
    for key in ('cohort', 'freeze'):
        if refs[key]['sha256'] != frozen[key]['sha256']:
            raise ValueError('frozen cohort/freeze binding differs')
    # Bind all continuation inventories; no old paths or JSON are rewritten.
    directories = [mapped(result).parent]; ancestors = []; current = frozen
    while current.get('continuation'):
        cont = current['continuation']
        old_result = mapped(cont['prior_result']['path'])
        load(old_result, cont['prior_result']['sha256'])
        old_plan = mapped(cont['prior_plan']['path'])
        parent = load(old_plan, cont['prior_plan']['sha256'])
        for relative, digest in cont['prior_files'].items():
            target = old_result.parent / relative
            if target.resolve().parent != old_result.parent.resolve() and not target.resolve().is_relative_to(old_result.parent.resolve()):
                raise ValueError('unsafe prior inventory path')
            if target.is_symlink() or sha(target) != digest:
                raise ValueError('ancestor raw artifact changed: ' + relative)
        if old_result.parent in directories:
            raise ValueError('cyclic continuation')
        directories.append(old_result.parent)
        ancestors.append({'plan': {'path': str(old_plan), 'sha256': sha(old_plan)},
                          'result': {'path': str(old_result), 'sha256': sha(old_result)},
                          'inventory_files_verified': len(cont['prior_files'])})
        current = parent
    # Package verifier reads sealed indices and event chains, not live models.
    for package, digest in frozen['package_indices'].items():
        replay.mini.verified(mapped(package), digest)
    batches = {b['batch_id']: b for b in frozen['batches']}
    if len(batches) != len(frozen['batches']) or len(aggregate['batches']) != len(batches):
        raise ValueError('selected batch census differs')
    rows = []; raw_refs = {}; seen = set()
    def file_ref(path):
        name = str(path)
        if name not in raw_refs:
            raw_refs[name] = {'path': name, 'sha256': sha(path), 'size_bytes': path.stat().st_size}
        return raw_refs[name]
    for group in aggregate['batches']:
        bid = group['batch_id']
        if bid not in batches or bid in seen or Path(bid).name != bid:
            raise ValueError('duplicate/unknown batch')
        seen.add(bid)
        batch = copy.deepcopy(batches[bid]); batch['package'] = str(mapped(batch['package']))
        selected = replay.select_batch(batch); package = Path(batch['package'])
        if [(r['call'], r['call_event_sha256']) for r in selected['calls']] != [(r['call'], r['call_event_sha256']) for r in batch['calls']]:
            raise ValueError('selected recorded calls differ from frozen plan')
        for ri, saved in enumerate(group['replays'], 1):
            candidates = [d / (bid + '_replay' + str(ri)) for d in directories]
            candidates = [d for d in candidates if (d / 'batch_result.json').exists()
                          and load(d / 'batch_result.json').get('status') == 'complete']
            if len(candidates) != 1:
                raise ValueError('saved complete replay missing or ambiguous: ' + bid)
            directory = candidates[0]
            actual_saved = load(directory / 'batch_result.json')
            if len(saved['calls']) != len(selected['calls']) or len(actual_saved['calls']) != len(saved['calls']):
                raise ValueError('saved sibling census changed')
            events = [load(p) for p in sorted(directory.glob('runtime_event_*.json'))]
            starts = [e['data']['call_id'] for e in events if e['kind'] == 'tool.started']
            if starts != [r['call']['call_id'] for r in selected['calls'] if r['native_validation_ok']]:
                raise ValueError('missing or changed replay tool event')
            cleanup = replay.cleanup_receipt(events)
            if cleanup != saved['cleanup'] or cleanup.get('confirmed') is not True:
                raise ValueError('replay cleanup not proved')
            if load(directory / 'runtime_provenance.json')['image_id'] != selected['checkpoint']['qualification']['image_id']:
                raise ValueError('replay image differs from qualified snapshot')
            before = replay.mini.read_json(replay.mini.referenced(package, selected['before_snapshot']))
            restored_path = directory / 'restore_capture/states/snapshots/s000000.json'
            restored = load(restored_path)
            replay.mini.validate_snapshot(directory / 'restore_capture', restored)
            if not replay.mini.snapshot_equal(before, restored) or saved['request_exact'] != selected['request_exact'] or saved['restore_exact'] is not True:
                raise ValueError('saved restore/request differs')
            for ci, (recorded, prior) in enumerate(zip(selected['calls'], saved['calls'])):
                call_dir = directory / f'call_{ci:04d}'
                disk_prior = load(call_dir / 'comparison.json')
                validation = load(call_dir / 'native_validation.json')
                if validation['ok'] != recorded['native_validation_ok']:
                    raise ValueError('saved native validation differs from recorded call')
                raw = load(call_dir / 'result.json')
                actual = replay.mini.Result(raw['observation'], (call_dir / 'stdout.bin').read_bytes(),
                            (call_dir / 'stderr.bin').read_bytes(),
                            (call_dir / 'image.bin').read_bytes() if (call_dir / 'image.bin').exists() else None,
                            provenance=raw['provenance'])
                state_path = call_dir / 'capture/states/snapshots/s000000.json'
                state = load(state_path)
                replay.mini.validate_snapshot(call_dir / 'capture', state)
                check = json.loads(replay.mini.canonical(replay.compare_call(package, recorded, actual,
                        call_dir / 'capture', state, call_dir, selected['config'], persist=False)))
                if check != prior:
                    sidecars = [r for r in aggregate.get('reclassifications', [])
                                if r['batch_id'] == bid and r['call_id'] == prior['call_id']
                                and r['before'] == prior and r['after'] == check
                                and mapped(r['original_comparison']['path']) == call_dir / 'comparison.json']
                    if len(sidecars) != 1 or sha(mapped(sidecars[0]['original_comparison']['path'])) != sidecars[0]['original_comparison']['sha256']:
                        raise ValueError('v1 effective comparison does not recompute: ' + bid)
                    prior = check
                changed = {'classification', 'ls_metadata_variance', 'unexplained_tool_difference'}
                if disk_prior != prior and not (recorded['call']['arguments'] == {'command': 'ls -la /workspace/input/'}
                    and disk_prior['classification'] == 'unexplained' and prior['classification'] == 'declared_runtime_variance'
                    and {k:v for k,v in disk_prior.items() if k not in changed} == {k:v for k,v in prior.items() if k not in changed}):
                    raise ValueError('original v1 comparison changed')
                expected = load(replay.mini.referenced(package, recorded['after_snapshot']))
                original = recorded['original_result']
                original_stdout = replay.mini.referenced(package, original['stdout']).read_bytes()
                original_stderr = replay.mini.referenced(package, original['stderr']).read_bytes()
                item = classify_metadata(recorded['call'], recorded['original_observation'], actual.observation,
                       original_stdout, actual.stdout, original_stderr, actual.stderr,
                       state_exact=check['after_state_exact'], image_exact=check['image_exact'])
                # A v1 admission is retained only after the existing validator
                # independently reproduces it, including the narrow ZIP check.
                if check['classification'] == 'declared_runtime_variance':
                    item.update(classification='declared_runtime_variance_v1', excluded_fields=[])
                delivered = replay.delivered_observation(actual.observation, selected['config']['recording']['observation_limit_chars'])
                if load(call_dir / 'delivered_observation.json') != delivered:
                    raise ValueError('saved model-visible observation changed')
                if item['classification'] in ('ls_runtime_metadata_v2','python_object_repr_address_v1'):
                    delivered_check = classify_metadata(recorded['call'], recorded['original_delivered'], delivered,
                            original_stdout, actual.stdout, original_stderr, actual.stderr,
                            state_exact=check['after_state_exact'], image_exact=check['image_exact'])
                    if delivered_check['classification'] != item['classification']:
                        item.update(classification='unexplained', excluded_fields=[])
                call_refs = [file_ref(p) for p in sorted(call_dir.iterdir()) if p.is_file()]
                for key in ('result', 'stdout', 'stderr', 'image'):
                    if key in original:
                        call_refs.append(file_ref(replay.mini.referenced(package, original[key])))
                call_refs += [file_ref(state_path), file_ref(replay.mini.referenced(package, recorded['after_snapshot']))]
                # Both immutable states and every referenced blob are bound.
                for base, snap in ((package, expected), (call_dir / 'capture', state)):
                    for f in snap['files']:
                        call_refs.append(file_ref(replay.mini.confined(base, f['blob'])))
                rows.append({'batch_id': bid, 'replay': ri, 'call_id': prior['call_id'],
                    'v1_original_classification': disk_prior['classification'],
                    'v1_effective_classification': prior['classification'],
                    'retained_v1_sidecar': disk_prior != prior,
                    'v2_classification': item.pop('classification'), **item,
                    'after_state_exact': check['after_state_exact'],
                    'expected_after_sha256': check['expected_after_sha256'], 'actual_after_sha256': check['actual_after_sha256'],
                    'original_comparison_sha256': sha(call_dir / 'comparison.json'), 'raw_refs': call_refs})
    selected_census = sum(len(b['calls']) for b in frozen['batches'])
    if selected_census != frozen['recorded_call_count']:
        raise ValueError('frozen first-replay call census differs')
    if len(rows) != aggregate['tool_attempts']:
        raise ValueError('comparison census differs from frozen attempt count')
    counts = dict(Counter(r['v2_classification'] for r in rows))
    report = {'schema': VERSION + '-reaudit', 'posthoc_extension': True, 'post_hoc_forensic_extension': True,
              'references': refs, 'ancestor_chain': ancestors,
              'authorization_source_ref': {'path': str(attachment), 'sha256': sha(attachment)},
              'logical_root': str(logical), 'physical_root': str(physical),
              'source_identity': {str(p.relative_to(L2.parents[1])): sha(p) for p in
                  (Path(__file__), L2 / 'src/l2_sandbox/metadata_contract.py', L2 / 'scripts/replay_prospective_tools.py', L2 / 'scripts/check_recovery_mechanism.py')},
              'actor_requests': 0, 'analyst_requests': 0, 'new_tool_calls': 0,
              'comparisons_reviewed': len(rows), 'batches_reviewed': len(seen), 'counts': counts,
              'rows': rows, 'all_comparisons_admissible': counts.get('unexplained', 0) == 0,
              'raw_byte_equivalence': False, 'statistical_equivalence_established': False,
              'scope': 'Post-hoc bounded restore/request/tool mechanism contract only; no outcome-distribution equivalence.',
              'forensics_used_as_proof': False}
    report['mechanism_gate_pass'] = validate_projected_report(report)
    return report


def verify_reaudit(report_path, *, expected_refs):
    """Fail closed on raw/source/reference tampering; recompute every comparison."""
    path = Path(report_path)
    report = json.loads(path.read_text())
    aliases = {'g2':'result', 'g2_plan':'plan', 'v2_contract':'policy',
               'v2_authorization':'authorization', 'g2_forensics':'forensics',
               'cohort':'cohort', 'prospective_freeze':'freeze'}
    for key, reference in expected_refs.items():
        if key == 'g2_v2':
            if sha(path) != reference['sha256']:
                raise ValueError('v2 report binding changed')
        elif key in aliases:
            if report['references'][aliases[key]]['sha256'] != reference['sha256']:
                raise ValueError('v2 input binding differs: ' + key)
    arguments = {k:v['path'] for k,v in report['references'].items()}
    rebuilt = reaudit(**arguments, logical_root=report['logical_root'], physical_root=report['physical_root'])
    if rebuilt != report or not validate_projected_report(rebuilt):
        raise ValueError('v2 report does not independently recompute')
    return {'tool_attempts': rebuilt['comparisons_reviewed'],
            'unexplained': rebuilt['counts'].get('unexplained', 0),
            'classification_counts': rebuilt['counts']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('plan','result','forensics','policy','authorization','cohort','freeze','logical-root','physical-root','output'):
        parser.add_argument('--' + field, required=True)
    args = vars(parser.parse_args()); output = Path(args.pop('output'))
    if output.exists():
        raise ValueError('new report only; overwrite forbidden')
    report = reaudit(**args)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write('\n')
    print(json.dumps({k:report[k] for k in ('comparisons_reviewed','batches_reviewed','counts','mechanism_gate_pass')}))


if __name__ == '__main__':
    main()
