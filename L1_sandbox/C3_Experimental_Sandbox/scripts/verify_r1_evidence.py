"""Read-only saved R1 evidence verification; never launches Docker or a model.

Frozen source is authoritative. The trusted controller's HMAC is checked using
its saved private key, which is never emitted. This is not host attestation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from c3_sandbox.evidence._storage import encode
from c3_sandbox.r1.admission import R1AdmissionAuthority, _audit
from c3_sandbox.evidence.workspace import _scan


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(encode(value)).hexdigest()


def require(ok, label):
    if not ok:
        raise ValueError(label)


def check_hash(path, expected):
    require(sha(path) == expected, 'file hash mismatch: ' + Path(path).name)


def check_provider_prefix(native, prefix):
    """Compare the actual SDK body to the intended normalized conversation.

    Inspect converts tool arguments to JSON strings and omits its metadata.
    Bytes in each content field and tool argument values must remain exact.
    """
    actual = native['request']['messages']
    require(len(actual) == len(prefix), 'provider prefix length mismatch')
    for body, intended in zip(actual, prefix):
        require(body['role'] == intended['role'] and
                body.get('content', '').encode() == intended.get('content', '').encode(),
                'provider prefix content mismatch')
        if intended.get('tool_calls'):
            require(len(body['tool_calls']) == len(intended['tool_calls']), 'provider tool count mismatch')
            for call, target in zip(body['tool_calls'], intended['tool_calls']):
                require(call['id'] == target['id'] and call['function']['name'] == target['function'] and
                        json.loads(call['function']['arguments']) == target['arguments'],
                        'provider tool arguments mismatch')
        if intended['role'] == 'tool':
            require(body['tool_call_id'] == intended['tool_call_id'], 'provider observation mismatch')


def workspace_hash(path):
    entries = _scan(Path(path))
    return canonical({k: {**v, 'data': hashlib.sha256(v['data']).hexdigest()} if 'data' in v else v
                      for k, v in entries.items()})


def verify(root=ROOT):
    root = Path(root)
    run = root / 'runs/e2-r1'
    scheduler = run / 'scheduler'
    lock = read(scheduler / 'implementation_lock.json')
    for name, expected in lock.items():
        check_hash(scheduler / 'source_snapshot' / name, expected)
    manifest = read(root / 'runs/r1-build-context/frozen/manifest.json')
    for name, entry in manifest['files'].items():
        check_hash(Path(manifest['bundle_root']) / name, entry['sha256'])
    plan = read(run / 'frozen_plan.json')
    specs = read(scheduler / 'branch_plan.json')['branches']
    ledger = read(scheduler / 'controller_audit.private.json')
    baseline = read(run / 'originals/officeval_002/result.json')
    cp = baseline['checkpoints'][0]
    record = read(cp['checkpoint_record_path'])
    # __init__ creates/chmods evidence directories: avoid it for read-only review.
    authority = R1AdmissionAuthority.__new__(R1AdmissionAuthority)
    authority.root = run / 'private/admission'
    authority.receipts = authority.root / 'receipts'
    authority.key_path = authority.root / 'authority.key'
    authority._key = authority.key_path.read_bytes()
    require(authority.verify(cp['checkpoint_id'], record), 'actual admission receipt invalid')
    receipt = authority.receipt_identity(cp['checkpoint_id'], record)
    source_pid = read(cp['source_audit_path'])['inspect']['State']['Pid']
    require(len(specs) == 8 and len({canonical(s) for s in specs}) == 8, 'eight distinct canonical branches required')
    probes = {'original', 'visibility.expanded', 'visibility.expanded.matched', 'feedback.structured'}
    for phase in ('exploration', 'confirmation'):
        selected = [s for s in specs if s['phase'] == phase]
        require(len(selected) == 4 and {s['probe_id'] for s in selected} == probes, 'phase probe coverage mismatch')
        require({s['sampling']['seed'] for s in selected} == {plan[phase + '_seed']}, 'phase seed mismatch')
    require(plan['exploration_seed'] != plan['confirmation_seed'], 'phase seeds equal')
    physical = {row['branch']: row for row in ledger['physical']}
    outcomes = {row['branch']: row for row in ledger['outcomes']}
    require(len(physical) == len(ledger['physical']) == len(outcomes) == 8, 'physical ledger duplication')
    pids, containers, prefixes, workspaces = set(), set(), set(), set()
    totals = {k: 0 for k in ('input_tokens', 'output_tokens', 'wall_seconds', 'environment_seconds')}
    calls, new_tools, grades = 0, 0, []
    for spec in specs:
        key = canonical(spec)
        b = scheduler / 'branches' / key
        require(read(b / 'attempt.json') == spec, 'branch spec substitution')
        require(spec['is_synthetic'] is False and spec['line'] == 'E2-F', 'real failure branch required')
        boundary = read(b / 'boundary_audit.json')
        _audit(boundary, plan['agent_image'])
        pid = boundary['inspect']['State']['Pid']
        require(pid != source_pid and pid not in pids, 'runtime PID reuse')
        pids.add(pid)
        provenance = read(b / 'actor-runtime/runtime_provenance.json')
        require(provenance['effective_image_id'] == plan['agent_image'], 'actor image mismatch')
        containers.add(provenance['container_id'])
        for name in ('admission_before.json', 'admission_before_model.json', 'verification.json'):
            q = read(b / name)
            require(q['eligible_for_intervention'] is True and q['is_synthetic'] is False and
                    q['admission_receipt'] == receipt and not q['reasons'], 'qualifier receipt mismatch')
        require(read(b / 'seed_verification.json')['verified'] is True, 'seed not verified')
        prefixes.add(sha(b / 'base_prefix.json'))
        base, injected, probe = read(b / 'base_prefix.json'), read(b / 'injected_prefix.json'), read(b / 'probe.json')
        require(base == read(cp['messages_path']), 'base prefix changed')
        require(hashlib.sha256(probe['rendered_text'].encode()).hexdigest() == probe['rendered_digest'], 'probe payload hash mismatch')
        require(injected == base + ([] if spec['probe_id'] == 'original' else
                [{'role': 'user', 'content': probe['rendered_text']}]), 'injected probe mismatch')
        workspaces.add(workspace_hash(b / 'workspace'))
        require(workspace_hash(b / 'workspace') == workspace_hash(cp['workspace']), 'workspace snapshot changed')
        native_rows = [json.loads(line) for line in (b / 'agent/model_calls.jsonl').read_text().splitlines()]
        requested = [r for r in native_rows if r['type'] == 'request']
        native = [r for r in native_rows if r['type'] == 'provider_call']
        response = [r for r in native_rows if r['type'] == 'response']
        require(len(requested) == len(native) == len(response) == 1, 'one real SDK call required')
        require(requested[0]['messages'] == injected and native[0]['error'] is None, 'SDK request mismatch/error')
        check_provider_prefix(native[0], injected)
        require(native[0]['request']['seed'] == spec['sampling']['seed'], 'SDK seed mismatch')
        usage = native[0]['response']['usage']
        normalized = response[0]['usage']
        agent = read(b / 'agent_result.json')
        counts = read(b / 'token_counts.json')
        cost = read(b / 'cost.json')
        require(usage['prompt_tokens'] == normalized['input_tokens'] == agent['tokens']['input_tokens'] ==
                cost['input_tokens'] == counts['injected']['token_count'], 'input token accounting mismatch')
        require(usage['completion_tokens'] == normalized['output_tokens'] == agent['tokens']['output_tokens'] ==
                cost['output_tokens'] > 0, 'output token accounting mismatch')
        require(native[0]['response']['choices'][0]['message']['content'] ==
                response[0]['model_output']['completion'], 'SDK response content mismatch')
        response_events = [e for e in agent['events'] if e['type'] == 'model_response']
        require(len(response_events) == 1 and response_events[0]['usage'] == normalized, 'agent event usage mismatch')
        tools = sum(e['type'] == 'tool_result' for e in agent['events'])
        new_tools += tools
        calls += len(native)
        result = read(b / 'result.json')
        for field, filename in [('agent_result','agent_result.json'), ('final_grade','final_grade.json'),
                                ('base_prefix','base_prefix.json'), ('injected_prefix','injected_prefix.json')]:
            check_hash(b / filename, result['evidence_hashes'][field])
        require(canonical(read(b / 'verification.json')) == result['evidence_hashes']['restoration'], 'restoration hash mismatch')
        runtime = read(b / 'branch_runtime.json')
        expected_runtime = canonical({'base_runtime': runtime['runtime'], 'base_system': spec['system_identity'],
            'probe_identity': spec['probe_identity'], 'rendered_digest': probe['rendered_digest'], 'sampling': runtime['sampling']})
        require(runtime['identity'] == result['evidence_hashes']['runtime'] == expected_runtime, 'runtime hash mismatch')
        grade = read(b / 'final_grade.json')
        require(grade == read(b / 'grade/grade.json'), 'grade copies mismatch')
        gp = read(b / 'grade/runtime/runtime_provenance.json')
        require(gp['effective_image_id'] == grade['grader_image'] == plan['grader_image'], 'grader image mismatch')
        artifact = grade['artifact']
        check_hash(b / 'final-workspace' / artifact['name'], artifact['sha256'])
        require(read(b / 'grade/process.json')['exit_code'] == 0, 'grader failed')
        require(json.loads(physical[key]['cost']) == cost and physical[key]['state'] == 'complete', 'physical cost mismatch')
        require(json.loads(outcomes[key]['result']) == result and outcomes[key]['state'] == 'complete', 'ledger result mismatch')
        require(result['branch_key'] == key and result['raw_score'] == grade['official_raw']['total_score'], 'score/key mismatch')
        grades.append(result['raw_score'])
        for name in totals:
            totals[name] += cost[name]
    require(len(pids) == len(containers) == 8 and len(prefixes) == len(workspaces) == 1, 'runtime/base equality mismatch')
    methods = {m['id']: m for m in ledger['methods']}
    require(len(methods) == 2 and sum(m['spent'] for m in methods.values()) == 5, 'logical charge count mismatch')
    history = {m: [o for o in ledger['observations'] if o['method'] == m] for m in methods}
    require({m: len(o) for m,o in history.items()} == {'engineering-A':4,'engineering-B-cache-audit':1}, 'method cache history mismatch')
    exploration = {canonical(s) for s in specs if s['phase'] == 'exploration'}
    for observation in ledger['observations']:
        body = json.loads(observation['body'])
        require(body['branch'] in exploration, 'confirmation leaked to method history')
        matching = [r for r in ledger['receipts'] if r['token'] == observation['receipt']]
        require(len(matching) == 1 and matching[0]['method'] == observation['method'] and
                matching[0]['charged'] == matching[0]['revealed'] == 1, 'observation charge mismatch')
    audit = read(run / 'originals_audit.json')
    excluded = next(c for c in audit['cases'] if c['task_id'] == 'officeval_003')
    require(excluded['protocol_exclusions'] and not excluded['execution_valid_after_audit'] and
            not excluded['admitted_as_success'] and not excluded['admitted_as_failure'], '003 invalid admission')
    require(audit['e2_s_coverage_met'] is False, 'missing success-side coverage misreported')
    original = read(run / 'originals/officeval_002/result.json')
    original_tools = sum(e['type'] == 'tool_result' for e in original['agent_result']['events'])
    require(original_tools == 1 and new_tools == 0, 'observed task tool coverage changed')
    return {'schema_version':'r1-independent-verification-1','saved_evidence_integrity':'PASS',
            'overall_R1_acceptance':'INCOMPLETE_SUCCESS_SIDE_COVERAGE', 'source_snapshot_files':len(lock),
            'frozen_manifest_files':len(manifest['files']), 'physical_branches':8,'real_sdk_calls':calls,
            'logical_charges':5,'method_history_counts':{m:len(v) for m,v in history.items()},
            'unique_actor_pids':len(pids),'new_branch_tool_calls':new_tools,'original_002_tool_calls':original_tools,'raw_scores':grades,
            'cost_totals':totals,'actual_admission_receipt_verified':True,'protocol_invalid_003_excluded':True,
            'E2_S_coverage_met':False,'limitations':['Eight branches each made one real model call but no renewed tool execution; only original 002 executed one task tool.',
            'Saved image IDs and frozen grader files verified; image contents not independently re-extracted.',
            'Controller HMAC proves trusted local evidence binding, not malicious-host resistance.',
            'Development-only panel, temperature 0; different seeds do not establish independent stochastic draws.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = verify()
    rendered = json.dumps(report, indent=2, sort_keys=True) + '\n'
    if args.output:
        args.output.write_text(rendered)
    print(rendered)


if __name__ == '__main__':
    main()
