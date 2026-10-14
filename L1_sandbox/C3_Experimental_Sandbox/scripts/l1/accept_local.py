#!/usr/bin/env python3
"""Explicit, fail-closed local acceptance. Never contacts a model by default.

Run from an installed source checkout. --execute-model is permission for actual
API/local inference. --resume adds a second inference run, possibly paid.
Receipts report operational validity and real new operations, not task high scores.
Decouples engineering validity from official evaluation outcomes and checkpoint eligibility.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

for _p in [Path('.env'), Path(__file__).resolve().parents[2] / '.env']:
    if _p.is_file():
        for _line in _p.read_text().splitlines():
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _k, _v = _line.split('=', 1)
                _k, _v = _k.strip(), _v.strip().strip('"\'')
                if _k not in os.environ:
                    os.environ[_k] = _v

from c3_sandbox.l1.core.config import resolve
from c3_sandbox.l1.core.doctor import doctor
from c3_sandbox.l1.core.orchestrator import run
from c3_sandbox.l1.storage import read_json, write_json


def qualify(result: dict, package: Path) -> dict:
    manifest = read_json(package / 'manifest.json')
    eval_grade = read_json(package / 'official_grade.json') if (package / 'official_grade.json').exists() else None

    # Dimension 1: Execution & Evidence Completeness (沙箱系统工程门禁)
    evidence_checks = {
        'sealed_evidence_verified': result['verification']['verified'],
        'not_test_fixture': manifest['recording_mode'] == 'live',
        'native_officeval': (manifest.get('task') or {}).get('provenance', {}).get('adapter') in ('officeval-native', 'lhtb-native', 'osworld-native'),
        'execution_valid': manifest['status']['execution_validity'] == 'valid',
        'declared_evidence_profile_complete': manifest['status']['evidence_completeness'] == 'complete',
        'new_model_request_recorded': manifest['counts']['model_requests'] > 0,
        'new_actual_tool_result_recorded': manifest['counts']['tool_results'] > 0,
    }
    engineering_passed = all(evidence_checks.values())

    # Dimension 2: Official Evaluation Result (官方评测返回状态，客观记录，不改写错误)
    raw_eval = {}
    if eval_grade:
        raw_eval = eval_grade.get('official_raw') if isinstance(eval_grade.get('official_raw'), dict) else eval_grade

    eval_status = {
        'evaluation_validity': manifest['status']['evaluation_validity'],
        'status': raw_eval.get('status', 'missing') if eval_grade else 'missing',
        'dim1_pass': raw_eval.get('dim1_pass') if eval_grade else None,
        'total_score': raw_eval.get('total_score') if eval_grade else None,
        'max_score': raw_eval.get('max_score') if eval_grade else None,
        'reward': raw_eval.get('reward') if eval_grade else None,
        'raw_reward': raw_eval.get('raw_reward') if eval_grade else None,
        'band': raw_eval.get('band') if eval_grade else None,
        'score': raw_eval.get('score') if eval_grade else None,
        'error': raw_eval.get('error') if eval_grade else None,
    }

    # Dimension 3: Checkpoint Eligibility (检查点恢复能力，解耦官方评分状态)
    idx_path = package / 'checkpoints/index.json'
    eligible_checkpoints = []
    if idx_path.exists():
        cps = read_json(idx_path).get('checkpoints', [])
        eligible_checkpoints = [c['checkpoint_id'] for c in cps if c.get('eligible') and c.get('checkpoint_id') != 'cp000000']

    checkpoint_eligibility = {
        'can_resume': engineering_passed and len(eligible_checkpoints) > 0,
        'eligible_checkpoint_count': len(eligible_checkpoints),
        'has_eligible_post_tool_checkpoint': len(eligible_checkpoints) > 0,
    }

    return {
        'passed': engineering_passed,
        'engineering_gate_passed': engineering_passed,
        'checks': evidence_checks,
        'official_evaluation': eval_status,
        'checkpoint_eligibility': checkpoint_eligibility,
        'task_outcome_not_a_gate': manifest['status']['task_outcome'],
        'episode_id': manifest['episode_id'],
        'package': str(package),
        'trusted_index_sha256': result['verification']['index_sha256'],
    }


def select_checkpoint(package: Path, requested_id: str | None = None) -> tuple[str | None, dict]:
    idx_path = package / 'checkpoints/index.json'
    if not idx_path.exists():
        return None, {'eligible': False, 'reason': 'no_checkpoints_index'}
    checkpoints = read_json(idx_path).get('checkpoints', [])
    eligible = [c for c in checkpoints if c.get('eligible') and c.get('checkpoint_id') != 'cp000000']
    if not eligible:
        return None, {'eligible': False, 'reason': 'no_eligible_post_tool_checkpoint'}

    if requested_id:
        match = [c for c in eligible if c['checkpoint_id'] == requested_id]
        if match:
            return requested_id, {'eligible': True, 'reason': 'explicitly_requested'}
        return None, {'eligible': False, 'reason': f'requested_checkpoint_{requested_id}_not_eligible'}

    # Inspect snapshot file changes compared to s000000 (initial state)
    s0_path = package / 'states/snapshots/s000000.json'
    s0_files = {}
    if s0_path.exists():
        s0_files = {f['path']: f['sha256'] for f in read_json(s0_path).get('files', [])}

    modified_state_cps = []
    for c in eligible:
        cp_file = package / c['path']
        if not cp_file.exists():
            continue
        cp_data = read_json(cp_file)
        snap_id = cp_data.get('snapshot_id')
        snap_file = package / f'states/snapshots/{snap_id}.json'
        if not snap_file.exists():
            continue
        snap_data = read_json(snap_file)
        cur_files = {f['path']: f['sha256'] for f in snap_data.get('files', [])}
        added = set(cur_files) - set(s0_files)
        modified = {p for p in set(cur_files) & set(s0_files) if cur_files[p] != s0_files[p]}
        if added or modified:
            modified_state_cps.append((c['checkpoint_id'], len(added), len(modified)))

    if modified_state_cps:
        # Prefer the earliest checkpoint that has recorded file modifications
        best_id, n_add, n_mod = modified_state_cps[0]
        return best_id, {'eligible': True, 'reason': 'stateful_file_change', 'files_added': n_add, 'files_modified': n_mod}

    # Fallback to earliest post-tool boundary
    return eligible[0]['checkpoint_id'], {'eligible': True, 'reason': 'early_post_tool_fallback', 'files_added': 0, 'files_modified': 0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path, help='New directory only')
    parser.add_argument('--execute-model', action='store_true', help='Authorize real inference, which can incur API charges')
    parser.add_argument('--resume', action='store_true', help='Also attempt a new inference branch from a qualified checkpoint')
    parser.add_argument('--checkpoint-id', type=str, default=None, help='Explicit checkpoint ID to resume from')
    args = parser.parse_args(argv)
    if args.resume and not args.execute_model:
        parser.error('--resume requires --execute-model')
    config = resolve(args.config)
    root = args.output_root.absolute()
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    receipt = {'schema_version': '0.2.0', 'qualification_passed': False,
               'model_execution_authorized': args.execute_model, 'resume_requested': args.resume}
    code = 2
    try:
        receipt['doctor'] = doctor(config)
        write_json(root, 'doctor.json', receipt['doctor'])
        if not receipt['doctor']['ready']:
            receipt['reason'] = 'prerequisites_not_ready'
        elif not args.execute_model:
            receipt['reason'] = 'preflight_only_no_model_execution'
            code = 0
        else:
            result = run(config, root / 'episode')
            receipt['episode'] = qualify(result, root / 'episode')
            receipt['qualification_passed'] = receipt['episode']['passed']
            can_resume = receipt['episode']['checkpoint_eligibility']['can_resume']
            if args.resume and can_resume:
                selected, cp_info = select_checkpoint(root / 'episode', args.checkpoint_id)
                if not selected:
                    receipt['resume'] = {'passed': False, 'reason': cp_info.get('reason')}
                else:
                    branch = run(config, root / 'branch', parent=root / 'episode', checkpoint_id=selected)
                    receipt['resume'] = qualify(branch, root / 'branch')
                    receipt['resume']['checkpoint_id'] = selected
                    receipt['resume']['checkpoint_selection'] = cp_info
                receipt['qualification_passed'] = receipt['episode']['passed'] and receipt['resume']['passed']
            elif args.resume:
                receipt['resume'] = {
                    'passed': False,
                    'reason': 'parent_execution_not_resumable',
                    'detail': {
                        'engineering_gate_passed': receipt['episode']['engineering_gate_passed'],
                        'checkpoint_eligibility': receipt['episode']['checkpoint_eligibility']
                    }
                }
            code = 0 if receipt['qualification_passed'] else 2
    except Exception as exc:
        receipt['error'] = {'type': type(exc).__name__, 'message': str(exc)}
    write_json(root, 'acceptance.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return code


if __name__ == '__main__':
    sys.exit(main())
