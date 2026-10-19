"""Close an exited, drained controller's operational census; never run an actor.

Run against the still-frozen controller source. Package contents/wire compatibility
are deliberately left to the successor's mandatory full predecessor verification.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time


def _alive(pid):
    if type(pid) is not int or pid <= 1:
        raise ValueError('positive controller PID required')
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def finalize(phase, drain_request):
    from l2_sandbox import opencode_d1_runtime as runtime
    from l2_sandbox.v3_execution import exclusive, file_digest, read_json, seal, verify_seal, write_json

    phase, drain_request = Path(phase).resolve(), Path(drain_request).resolve()
    output = phase / 'execution'
    closure = output / 'operator_closure'

    def reference(path):
        return dict(path=str(path), sha256=file_digest(path))

    def check_ref(ref):
        path = Path(ref['path'])
        if not path.is_absolute() or path.is_symlink() or file_digest(path) != ref['sha256']:
            raise ValueError('closure reference changed: ' + str(path))

    def exited():
        started = read_json(phase / 'supervisor_started.json')
        ended = read_json(phase / 'supervisor_result.json')
        pid = started.get('controller_pid')
        if (ended.get('controller_pid') != pid
                or type(ended.get('controller_exit_code')) is not int
                or type(ended.get('finished_at_epoch')) not in (int, float)
                or ended['finished_at_epoch'] < started['started_at_epoch']
                or _alive(pid)):
            raise ValueError('controller exit is not established')
        request = read_json(drain_request)
        if request.get('controller_pid') != pid:
            raise ValueError('drain request controller PID differs')
        return pid

    with exclusive(output / 'stage.lock'):
        pid = exited()
        manifest_path = closure / 'manifest.json'
        if closure.exists():
            if not manifest_path.exists():
                raise ValueError('partial closure snapshot; manual inspection required')
            manifest = read_json(manifest_path)
            verify_seal(manifest)
            if (manifest.get('schema') != 'opencode-operator-drain-closure-v1'
                    or manifest.get('controller_pid') != pid
                    or manifest.get('drain_request') != reference(drain_request)):
                raise ValueError('closure identity differs')
            expected_receipts = {row['path'] for row in manifest['receipt_census']}
            if {str(p) for p in (output / 'receipts').glob('*.json')} != expected_receipts:
                raise ValueError('receipt census changed after closure')
            for ref in manifest['frozen_references']:
                check_ref(ref)
            # The prepared manifest is the transaction journal. Recover only an
            # interrupted replacement of exactly these two operational files.
            for name in ('admission', 'result'):
                target = output / (name + '.json')
                item = manifest['operational_files'][name]
                current = file_digest(target)
                if current not in (item['original']['sha256'], item['final']['sha256']):
                    raise ValueError('operational file changed after closure')
                if current != item['final']['sha256']:
                    write_json(target, read_json(Path(item['final']['path'])))
                if file_digest(target) != item['final']['sha256']:
                    raise ValueError('final operational digest differs')
            return manifest

        plan_path = phase / 'execution_plan.json'
        plan = read_json(plan_path)
        runtime._verify_plan(plan)
        jobs = runtime._scheduled_jobs(plan, output)
        by_id = {j['binding']['branch_id']: j for j in jobs}
        preflight_path = phase / 'integrity_preflight.json'
        preflight = read_json(preflight_path)
        verify_seal(preflight)
        if (preflight.get('schema') != 'opencode-d1-integrity-preflight-v1'
                or preflight.get('plan_sha256') != plan['artifact_sha256']
                or preflight.get('planned_jobs') != len(jobs)):
            raise ValueError('preflight binding differs')
        retained = preflight['retained']
        admission = read_json(output / 'admission.json')
        original_result = read_json(output / 'result.json')
        for value in (admission, original_result):
            verify_seal(value)
            if value.get('plan_sha256') != plan['artifact_sha256']:
                raise ValueError('operational plan binding differs')
        if (admission.get('limits') != plan['limits']
                or original_result.get('simulated_offline') is not False
                or original_result.get('d2_gate_open') is not False
                or original_result.get('status') != 'incomplete'
                or admission.get('stop_reason') not in (None, 'closed')):
            raise ValueError('operator drain cannot change an existing stop/error')
        sends = admission.get('sends')
        if (not isinstance(sends, list) or admission.get('requests') != len(sends)
                or any(not isinstance(s, dict) or s.get('status') not in ('settled', 'unknown_usage')
                       or s.get('state') == 'inflight' for s in sends)):
            raise ValueError('uncertain or inflight admission sends')
        counts = dict(completed=0, invalid=0, carried_completed=0, carried_invalid=0)
        census, frozen = [], [reference(plan_path), reference(preflight_path),
            reference(phase / 'supervisor_started.json'), reference(phase / 'supervisor_result.json'),
            reference(drain_request)]
        for inventory in ('source_files', 'bound_files'):
            frozen.extend(dict(path=p, sha256=h) for p, h in plan[inventory].items())
        paths = sorted((output / 'receipts').glob('*.json'))
        receipt_ids = {p.stem for p in paths}
        roots = {output / 'branches', output / 'provenance'}
        if 'branch_storage_root' in plan:
            storage = runtime._branch_storage_root(plan)
            roots.update((storage, storage.parent / 'provenance'))
        for root in roots:
            if root.exists() and any(p.name not in receipt_ids for p in root.iterdir()):
                raise ValueError('uncertain package/provenance without receipt')
        for path in paths:
            receipt = read_json(path)
            verify_seal(receipt)
            job = by_id.get(path.stem)
            if (job is None or job['path'] != path
                    or any(receipt.get(k) != v for k, v in job['binding'].items())
                    or receipt.get('simulated_offline') is not False):
                raise ValueError('receipt binding differs')
            status = receipt.get('status')
            carried = 'predecessor_receipt_ref' in receipt
            if status == 'completed':
                score = receipt.get('outcome_score')
                if (receipt.get('valid_attempt') is not True or type(score) not in (int, float)
                        or not math.isfinite(score) or not 0 <= score <= 1):
                    raise ValueError('invalid completed receipt')
                counts['completed'] += 1
                counts['carried_completed'] += int(carried)
                package = Path(receipt['package_path'])
                if not carried and package != job['package']:
                    raise ValueError('current package path differs')
                if any(key not in receipt for key in ('package_index_sha256', 'official_grade_sha256', 'model_transition_sha256')):
                    raise ValueError('completed receipt lacks package references')
                for filename, key in (('evidence_index.json', 'package_index_sha256'),
                                      ('official_grade.json', 'official_grade_sha256'),
                                      ('lineage/actor_transition.json', 'model_transition_sha256')):
                    if key in receipt:
                        ref = dict(path=str(package / filename), sha256=receipt[key])
                        check_ref(ref)
                        frozen.append(ref)
            elif status in ('incomplete', 'infrastructure_invalid', 'branch_invalid'):
                if (receipt.get('valid_attempt') is not False or 'outcome_score' in receipt
                        or receipt.get('no_resampling') is not True
                        or not isinstance(receipt.get('error_type'), str)):
                    raise ValueError('uncertain failure receipt')
                counts['invalid'] += 1
                counts['carried_invalid'] += int(carried)
            else:
                raise ValueError('inflight or unknown receipt status')
            if carried:
                prior_ref = receipt['predecessor_receipt_ref']
                check_ref(prior_ref)
                predecessor = read_json(Path(prior_ref['path']))
                verify_seal(predecessor)
                if (any(predecessor.get(k) != v for k, v in runtime._logical_binding(job['binding']).items())
                        or predecessor.get('valid_attempt') != receipt['valid_attempt']):
                    raise ValueError('carried receipt differs from predecessor logical binding')
                frozen.append(prior_ref)
            ref = reference(path)
            frozen.append(ref)
            census.append(dict(**ref, branch_id=path.stem, status=status, carried=carried,
                               valid_attempt=receipt['valid_attempt']))
        if (counts['carried_completed'] != retained['completed']
                or counts['carried_invalid'] != retained['infrastructure_invalid']):
            raise ValueError('carried receipt census differs from launch preflight')
        own = dict(requests=admission['requests'], tokens=admission['tokens'],
                   cost_usd=admission['cost_usd'], invalid_attempts=admission['invalid_attempts'],
                   http_send_count=sum(bool(s.get('transport_started')) for s in sends),
                   branch_invalid=admission.get('branch_invalid', 0),
                   fatal_invalid=admission.get('fatal_invalid', admission['invalid_attempts']))
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in own.values()):
            raise ValueError('invalid admission accounting')
        final_admission = seal(dict(admission, stop_reason='closed'))
        final_result = dict(original_result, **own, status='incomplete',
            planned_branches=len(jobs), completed_branches=counts['completed'],
            invalid_or_incomplete_branches=counts['invalid'],
            pending_branches=len(jobs)-len(paths), stop_reason='closed',
            current_phase_completed_branches=counts['completed']-counts['carried_completed'],
            carried_completed_branches=counts['carried_completed'],
            carried_infrastructure_invalid_branches=counts['carried_invalid'],
            aggregate_invalid_branches=counts['invalid'],
            current_phase_branch_invalid=own['branch_invalid'], current_phase_fatal_invalid=own['fatal_invalid'],
            execution_finished=True, process_finished=True,
            schedule_complete=len(paths) == len(jobs), d2_gate_open=False)
        final_result.update({'aggregate_' + k: v + retained[k] for k, v in own.items()})
        final_result = seal(final_result)
        # Read guards again before creating the immutable transaction journal.
        exited()
        for ref in frozen:
            check_ref(ref)
        originals = {name: (output / (name + '.json')).read_bytes() for name in ('result', 'admission')}
        if (json.loads(originals['result']) != original_result
                or json.loads(originals['admission']) != admission):
            raise ValueError('operational files changed during census')
        closure.mkdir()
        files = {}
        for name, final in (('admission', final_admission), ('result', final_result)):
            original_path = closure / ('original_' + name + '.json')
            original_path.write_bytes(originals[name])
            final_path = closure / ('final_' + name + '.json')
            write_json(final_path, final)
            files[name] = dict(original=reference(original_path), final=reference(final_path))
            frozen.extend((files[name]['original'], files[name]['final']))
        manifest = seal(dict(schema='opencode-operator-drain-closure-v1',
            plan_sha256=plan['artifact_sha256'], controller_pid=pid, closed_at_epoch=time.time(),
            verification_scope='operational_census_and_accounting_only',
            successor_full_predecessor_verification_required=True,
            scientific_artifacts_modified=False, actor_requests=0, scorer_invocations=0,
            supervisor_result=reference(phase / 'supervisor_result.json'),
            drain_request=reference(drain_request), receipt_census=census,
            operational_files=files, frozen_references=frozen))
        write_json(manifest_path, manifest)
        for path in closure.iterdir():
            path.chmod(0o444)
        for name in ('admission', 'result'):
            write_json(output / (name + '.json'), read_json(Path(files[name]['final']['path'])))
            if file_digest(output / (name + '.json')) != files[name]['final']['sha256']:
                raise ValueError('operational replacement digest differs')
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--phase', type=Path, required=True)
    parser.add_argument('--drain-request', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo_root / 'L2_Behavioral Event Localization/L2_Sandbox/src'))
    manifest = finalize(args.phase, args.drain_request or args.phase / 'operator_drain_request.json')
    print(json.dumps(dict(closure_sha256=manifest['artifact_sha256'],
                          verification_scope=manifest['verification_scope'],
                          receipts=len(manifest['receipt_census']), actor_requests=0)))


if __name__ == '__main__':
    main()
