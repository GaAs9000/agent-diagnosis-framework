"""Observe a fresh Docker reconstruction before admitting any real branch."""
import copy
import argparse
import json
from pathlib import Path

from c3_sandbox.evidence._storage import encode
from c3_sandbox.manifests import canonical_hash
from c3_sandbox.replay import ReplayQualification
from c3_sandbox.r1.admission import R1AdmissionAuthority
from c3_sandbox.r1.docker_runtime import DockerRuntime
from c3_sandbox.r1.office_run import SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    with path.open('xb') as stream:
        stream.write(encode(value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, default=ROOT / 'runs/e2-r1')
    args = parser.parse_args()
    root = args.run_root.absolute()
    destination = root / 'restore-qualification'
    destination.mkdir(mode=0o700)
    baseline = json.loads((root / 'originals/officeval_002/result.json').read_bytes())
    reference = baseline['checkpoints'][0]
    authority = R1AdmissionAuthority(root / 'private/admission')
    replay = ReplayQualification(Path(baseline['replay_store']), real_admission=authority)
    record = replay._record(reference['checkpoint_id'])
    write(destination / 'checkpoint_record.json', record)
    # This preliminary reconstruction authorizes NO model continuation.
    restored = replay.workspace_store.restore(record['snapshot_id'], destination / 'staging')
    write(destination / 'host_restore.json', restored.to_dict())
    if not restored.valid:
        raise RuntimeError('host snapshot reconstruction failed')
    messages = json.loads(Path(reference['messages_path']).read_bytes())
    source = json.loads(Path(reference['source_audit_path']).read_bytes())
    write(destination / 'source_audit.json', source)
    with DockerRuntime(record['runtime']['image_digest'], destination / 'runtime') as actor:
        actor.seed_workspace(destination / 'staging')
        equality = actor.verify_workspace(destination / 'staging')
        audit = actor.audit_boundary()
        environment = dict(item.split('=', 1) for item in actor._inspect()['Config']['Env'])
        restored_runtime = copy.deepcopy(baseline['runtime'])
        restored_runtime['environment'] = environment
        sampling = {'seed': 2026, 'temperature': 0, 'max_tokens': 2048,
                    'context_window': 16384, 'max_retries': 0, 'parallel_tool_calls': False,
                    'top_k': 1, 'min_p': 0, 'max_turns': 10}
        observed_system = canonical_hash({'model': restored_runtime['model_config'],
            'agent': 'NativeAgent-r1', 'tools': restored_runtime['tool_config'],
            'system_prompt': SYSTEM_PROMPT, 'sampling': sampling, 'runtime': restored_runtime})
        report = {**equality, 'messages_equal': encode(messages) == encode(record['messages']),
            'runtime_equal': encode(restored_runtime) == encode(record['runtime']) and audit['effective_config_ok'],
            'system_equal': observed_system == record['system_identity'],
            'observed_system_identity': observed_system,
            'message_identity': canonical_hash(messages),
            'scope': 'cold tool boundary: complete workspace bytes/types/modes, exact conversation, frozen image/config/environment',
            'excluded_volatile_state': ['wall clock', 'kernel process identifiers', 'fresh namespace identifiers',
                                       'filesystem access/modification times', 'hostname', 'model KV cache'],
            'no_model_continuation_during_qualification': True}
        actor.export_workspace(destination / 'restored-workspace')
    write(destination / 'restore_audit.json', audit)
    write(destination / 'restoration_report.json', report)
    model = baseline['model_qualification']
    write(destination / 'model_qualification.json', model)
    evidence = [destination / name for name in ('checkpoint_record.json', 'source_audit.json',
        'restore_audit.json', 'restoration_report.json', 'model_qualification.json')]
    plan = json.loads((root / 'frozen_plan.json').read_bytes())
    evidence.append(Path(plan.get('qualification_directory', str(ROOT / 'runs/r1-native-qualification-007'))) / 'agent/model_calls.jsonl')
    receipt = authority.issue(reference['checkpoint_id'], record, source, audit, report, model, evidence)
    verdict = replay.qualify(reference['checkpoint_id'])
    write(destination / 'qualification.json', {**verdict, 'receipt_identity': receipt})
    if not verdict['eligible_for_intervention']:
        raise RuntimeError('real admission failed')
    print(json.dumps({'checkpoint_id': reference['checkpoint_id'], 'eligible': True,
                      'verified_paths': report['actual_paths'], 'receipt_identity': receipt}))


if __name__ == '__main__':
    main()
