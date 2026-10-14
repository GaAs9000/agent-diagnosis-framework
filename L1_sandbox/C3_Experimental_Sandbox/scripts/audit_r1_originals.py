"""Append protocol exclusions and validated D1/D2 views without changing originals."""
import argparse
import hashlib
import json
from pathlib import Path

from c3_sandbox.evidence._storage import encode
from c3_sandbox.r1.trajectory_export import export_baseline, export_manifests

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, default=ROOT / 'runs/e2-r1')
    args = parser.parse_args()
    root = args.run_root.absolute()
    plan = json.loads((root / 'frozen_plan.json').read_bytes())
    qualification = json.loads((root / 'restore-qualification/qualification.json').read_bytes())
    rows, trajectories = [], []
    for task in plan['tasks']:
        folder = root / 'originals' / task
        result = json.loads((folder / 'result.json').read_bytes())
        reasons = []
        records = [json.loads(line) for line in (folder / 'agent/model_calls.jsonl').read_text().splitlines()]
        for record in records:
            if record['type'] != 'response':
                continue
            for choice in record['model_output']['choices']:
                if choice['stop_reason'] == 'tool_calls' and not choice['message'].get('tool_calls'):
                    reasons.append({'turn': record['turn'], 'reason': 'tool_calls_stop_without_native_calls'})
        if result['agent_result']['stopped_reason'] not in ('assistant_finished', 'max_turns'):
            reasons.append({'reason': result['agent_result']['stopped_reason']})
        valid = result['execution_valid'] and not reasons
        rows.append({'task_id': task, 'raw_result_sha256': hashlib.sha256((folder / 'result.json').read_bytes()).hexdigest(),
            'protocol_exclusions': reasons, 'execution_valid_after_audit': bool(valid),
            'admitted_as_failure': bool(valid and result['task_completed'] is False),
            'admitted_as_success': bool(valid and result['task_completed'] is True),
            'native_provider_body_available': any(record['type'] == 'provider_call' for record in records),
            'checkpoint_count': len(result['checkpoints'])})
        relevant = any(cp['checkpoint_id'] == qualification['checkpoint_identity'] for cp in result['checkpoints'])
        trajectories.append(export_baseline(result, root / 'unified-trajectories' / task,
            qualification_report=qualification if relevant else None,
            exclusion_reason=json.dumps(reasons) if reasons else None))
    report = {'schema_version': 'r1-original-audit-1', 'original_files_unchanged': True,
              'cases': rows, 'e2_s_coverage_met': any(row['admitted_as_success'] for row in rows),
              'e2_f_coverage_met': any(row['admitted_as_failure'] for row in rows)}
    with (root / 'originals_audit.json').open('xb') as stream:
        stream.write(encode(report))
    manifest = json.loads((ROOT / 'runs/r1-build-context/frozen/manifest.json').read_bytes())
    export_manifests(trajectories, manifest, root / 'manifests')
    print(json.dumps({'E2_S': report['e2_s_coverage_met'], 'E2_F': report['e2_f_coverage_met']}))


if __name__ == '__main__':
    main()
