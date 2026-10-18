#!/usr/bin/env python3
"""R22 validation-boundary audit. No Actor, scorer, Docker or tool execution."""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

L2 = Path(__file__).resolve().parents[1]
ROOT = L2.parents[1]
L1 = ROOT / 'L1_sandbox/C3_Experimental_Sandbox'
LEGACY = L1 / 'validation/pi_native_alignment_20261018/legacy_source'
EXCLUDED = {'officeval_004.r02', 'officeval_010.r00', 'officeval_050.r04'}
DEV = ('003','004','010','033','050','060','070','085','090')
SLOTS = tuple(f'officeval_{t}.r{r:02d}' for t in DEV for r in range(5) if f'officeval_{t}.r{r:02d}' not in EXCLUDED)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def value_hash(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def strict_equal(a, b):
    # JSON's numeric and boolean types must not collapse under Python equality.
    return canonical(a) == canonical(b)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def check_scope(inventory):
    if set(inventory) != set(SLOTS) or len(inventory) != 42:
        raise ValueError('R22 requires exactly the 42 original other DEV slots; no skips')


def legacy_preprocessor(path):
    # Compile only the archived pure transform; importing tool executors is unnecessary.
    tree = ast.parse(path.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_prepare_edit_arguments']
    if len(functions) != 1:
        raise ValueError('missing archived edit transform')
    namespace = {'json': json}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['_prepare_edit_arguments']


def compare_call(call, outcome, prepare):
    raw = call['arguments']
    effective = prepare(copy.deepcopy(raw)) if call['name'] == 'edit' else copy.deepcopy(raw)
    validated = outcome.get('validated_arguments')
    raw_equal = outcome.get('ok') is True and strict_equal(raw, validated)
    effective_equal = outcome.get('ok') is True and strict_equal(effective, validated)
    return {**call, 'raw_arguments_sha256': value_hash(raw), 'legacy_effective_arguments': effective,
            'legacy_effective_arguments_sha256': value_hash(effective),
            'native': outcome, 'native_validated_arguments_sha256': value_hash(validated),
            'raw_equal': raw_equal, 'legacy_effective_equal': effective_equal,
            'boundary_pass': raw_equal and effective_equal}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_package(path, expected, validator, prepare, verify, reconstruct, read_json, confined, parse_backend):
    def full_verify():
        result = verify(path, expected_index_sha256=expected)
        if not result['verified']:
            raise ValueError('full seal/projection verification failed: ' + repr(result['errors']))
    full_verify()
    events = [json.loads(line) for line in (path / 'trajectory.jsonl').read_text().splitlines()]
    config = read_json(path / 'raw/resolved_config.json')
    manifest = read_json(path / 'manifest.json')
    if manifest.get('lineage'):
        raise ValueError('audit42 must contain original root episodes')
    def read(ref):
        target = confined(path, ref['path'])
        if sha(target) != ref['sha256']:
            raise ValueError('raw reference digest mismatch')
        return target.read_bytes()
    requests = reconstruct(config, read_json(path / 'benchmark/public_task.json'), events, read)
    if not requests or any(r['status'] != 'verified' for r in requests):
        raise ValueError('full raw request reconstruction failed')
    calls = []
    native_batches = 0
    model = parse_backend(config)
    for response in (e for e in events if e['kind'] == 'model.response'):
        rid = response['data']['request_id']
        reply = model.parse(read(response['data']['body']), rid)
        recorded = [e['data'] for e in events if e['kind'] == 'tool.call' and e['data']['request_id'] == rid]
        projected = [{'request_id': rid, 'call_id': c.call_id, 'native_id': c.native_id, 'name': c.name, 'arguments': c.arguments} for c in reply.calls]
        if not strict_equal(recorded, projected):
            raise ValueError('raw response call projection mismatch')
        if not projected:
            continue
        native_batches += 1
        outcomes = validator.validate_batch([{'name': c['name'], 'arguments': c['arguments'], 'id': c['native_id'] or c['call_id']} for c in projected])
        if len(outcomes) != len(projected):
            raise ValueError('native batch result coverage mismatch')
        for c, outcome in zip(projected, outcomes):
            telemetry = outcome.get('telemetry', {})
            if (telemetry.get('actor_requests') != 0 or telemetry.get('native_tool_executions') != 0
                    or telemetry.get('native_loop_observed_responses') != 1
                    or not strict_equal(outcome.get('raw_arguments'), c['arguments'])
                    or outcome.get('provenance') != validator.dependency_identity()):
                raise ValueError('native validation telemetry/input/identity invariant failed')
            calls.append(compare_call({**c, 'raw_response_ref': response['data']['body']}, outcome, prepare))
    if len(calls) != len([e for e in events if e['kind'] == 'tool.call']):
        raise ValueError('tool call coverage incomplete')
    full_verify()
    return {'slot': path.name, 'episode_id': manifest['episode_id'], 'index_sha256': expected,
            'status': 'pass' if all(c['boundary_pass'] for c in calls) else 'protocol_halt',
            'requests': requests, 'calls': calls, 'native_loop_stub_responses': native_batches, 'full_seal_before_after': True}


def prefix_validation(path, last_request_id, validator, prepare):
    """Validate every successful recorded call strictly before the failed response."""
    events = [json.loads(line) for line in (path / 'trajectory.jsonl').read_text().splitlines()]
    responses = [e for e in events if e['kind'] == 'model.response']
    last = [e for e in responses if e['data']['request_id'] == last_request_id]
    if len(last) != 1:
        raise ValueError('failed prefix response binding mismatch')
    all_recorded = [e for e in events if e['kind'] == 'tool.call']
    recorded = [e for e in all_recorded if e['seq'] < last[0]['seq']]
    excluded = [e for e in all_recorded if e['seq'] >= last[0]['seq']]
    successful = {e['data']['call_id'] for e in events if e['kind'] == 'tool.result'}
    if (any(e['data']['call_id'] not in successful for e in recorded)
            or any(e['data'].get('request_id') != last_request_id for e in excluded)):
        raise ValueError('selected parent prefix contains non-successful or post-failure calls')
    rows = []
    batches = 0
    for event in responses:
        if event['seq'] >= last[0]['seq']:
            continue
        ref = event['data']['body']
        target = path / ref['path']
        if sha(target) != ref['sha256']:
            raise ValueError('prefix raw response digest mismatch')
        native = json.loads(target.read_text())['choices'][0]['message'].get('tool_calls') or []
        rid = event['data']['request_id']
        projected = [{'request_id': rid, 'call_id': f'{rid}:c{i}', 'native_id': c['id'],
                      'name': c['function']['name'], 'arguments': json.loads(c['function']['arguments'])}
                     for i, c in enumerate(native)]
        observed = [e['data'] for e in recorded if e['data']['request_id'] == rid]
        if not strict_equal(projected, observed):
            raise ValueError('prefix raw tool call projection mismatch')
        if not projected:
            continue
        batches += 1
        outcomes = validator.validate_batch([{'name': c['name'], 'arguments': c['arguments'], 'id': c['native_id']} for c in projected])
        if len(outcomes) != len(projected):
            raise ValueError('prefix native coverage mismatch')
        for call, outcome in zip(projected, outcomes):
            telemetry = outcome.get('telemetry', {})
            if (telemetry.get('actor_requests') != 0 or telemetry.get('native_tool_executions') != 0
                    or telemetry.get('native_loop_observed_responses') != 1
                    or not strict_equal(outcome.get('raw_arguments'), call['arguments'])
                    or outcome.get('provenance') != validator.dependency_identity()):
                raise ValueError('prefix native invariant failed')
            rows.append(compare_call({**call, 'raw_response_ref': ref}, outcome, prepare))
    if len(rows) != len(recorded):
        raise ValueError('prefix tool call coverage incomplete')
    return {'status': 'pass' if all(c['boundary_pass'] for c in rows) else 'protocol_halt',
            'native_validation_calls': len(rows), 'native_loop_stub_responses': batches,
            'native_validation_failures': sum(c['native'].get('ok') is not True for c in rows),
            'raw_differences': sum(not c['raw_equal'] for c in rows),
            'legacy_effective_differences': sum(not c['legacy_effective_equal'] for c in rows),
            'calls': rows}


def failed_response_proof(plan, validator, verify, prepare):
    rows = []
    for parent in plan['parents']:
        path = ROOT / parent['parent_path']
        expected = parent['index_sha256']
        if not verify(path, expected_index_sha256=expected)['verified']:
            raise ValueError('failed parent seal verification failed')
        checkpoint = path / 'checkpoints' / (parent['checkpoint_id'] + '.json')
        if sha(checkpoint) != parent['checkpoint_sha256']:
            raise ValueError('selected continuation checkpoint changed')
        prefix = prefix_validation(path, parent['last_request_id'], validator, prepare)
        ref = parent['last_response_ref']
        response_path = path / ref['path']
        if sha(response_path) != ref['sha256']:
            raise ValueError('failed response changed')
        response = json.loads(response_path.read_text())
        raw_calls = response['choices'][0]['message']['tool_calls']
        calls = [{'name': c['function']['name'], 'arguments': json.loads(c['function']['arguments']), 'id': c['id']} for c in raw_calls]
        outcomes = validator.validate_batch(calls)
        repeated = validator.validate_batch(calls)
        if len(outcomes) != len(calls) or len(repeated) != len(calls) or not calls:
            raise ValueError('failed response native coverage mismatch')
        proofs = []
        for call, first, second in zip(calls, outcomes, repeated):
            same = strict_equal(first.get('native_result'), second.get('native_result'))
            nested = isinstance(call['arguments'], dict) and 'path' not in call['arguments'] and any(isinstance(e, dict) and 'path' in e for e in call['arguments'].get('edits', []))
            if first.get('ok') is not False or second.get('ok') is not False or not same:
                raise ValueError('expected repeatable native validation error was not established')
            proofs.append({'raw_call': call, 'raw_arguments_sha256': value_hash(call['arguments']), 'native': first, 'repeated_native': second, 'native_error_identical': same, 'observed_argument_shape': 'missing_root_path_with_nested_path' if nested else 'malformed_edit_shape'})
        if not verify(path, expected_index_sha256=expected)['verified']:
            raise ValueError('failed parent changed after proof')
        rows.append({**parent, 'calls': proofs, 'prefix_validation': prefix, 'full_seal_before_after': True})
    if len(rows) != 3:
        raise ValueError('exactly three failed parents required')
    return {'status': 'expected_native_errors_proven', 'actor_requests': 0, 'tool_executions': 0,
            'native_validation_calls': sum(2 * len(r['calls']) for r in rows), 'native_loop_stub_responses': 2 * len(rows),
            'prefix_validation': {'status': 'pass' if all(r['prefix_validation']['status'] == 'pass' for r in rows) else 'protocol_halt',
                                  **{k: sum(r['prefix_validation'][k] for r in rows) for k in ('native_validation_calls', 'native_loop_stub_responses', 'native_validation_failures', 'raw_differences', 'legacy_effective_differences')}}, 'parents': rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--freeze-inventory', action='store_true')
    p.add_argument('--output', type=Path)
    p.add_argument('--bridge-dir', type=Path)
    p.add_argument('--node')
    p.add_argument('--parents', type=Path)
    args = p.parse_args()
    if args.freeze_inventory:
        write(args.inventory, {s: sha(args.runs / s / 'evidence_index.json') for s in SLOTS})
        return
    inventory = json.loads(args.inventory.read_text())
    check_scope(inventory)
    if not args.output:
        p.error('--output required')
    # All historical parsing and request construction use the archived 110-file controller.
    sys.path[:0] = [str(LEGACY / 'L1_sandbox/C3_Experimental_Sandbox/src')]
    from c3_sandbox.l1.package import verify_package
    import c3_sandbox.l1.evidence.live as evidence_live
    # The frozen source archive contains Python files only. Bind the unchanged
    # evidence schema explicitly as its read-only package resource fallback.
    from functools import lru_cache
    from jsonschema import Draft202012Validator, FormatChecker
    schema_path = L1 / 'src/c3_sandbox/l1/evidence/schema.json'
    @lru_cache(maxsize=None)
    def resource_validator(kind):
        doc = json.loads(schema_path.read_text())
        schema = {'$schema': doc['$schema'], '$defs': doc['$defs'], '$ref': '#/$defs/' + kind}
        Draft202012Validator.check_schema(schema)
        return Draft202012Validator(schema, format_checker=FormatChecker())
    evidence_live.validator = resource_validator
    from c3_sandbox.l1.storage import read_json, confined
    g4 = load_module('r22_g4', L2 / 'scripts/audit_g4_provenance.py')
    native_path = L1 / 'src/c3_sandbox/l1/runtime/pi_native_validation.py'
    native = load_module('r22_native', native_path)
    old_tools = LEGACY / 'L1_sandbox/C3_Experimental_Sandbox/src/c3_sandbox/l1/runtime/tools.py'
    prepare = legacy_preprocessor(old_tools)
    archived_manifest = json.loads((L1 / 'validation/pi_native_alignment_20261018/legacy_source_manifest.json').read_text())
    if any(sha(LEGACY / path) != h for path, h in archived_manifest['source_fingerprints'].items()):
        raise ValueError('archived source does not match frozen original manifest')
    source_paths = [L1 / 'src/c3_sandbox/l1/evidence/schema.json', Path(__file__), L2 / 'scripts/audit_g4_provenance.py', native_path,
                    L1 / 'validation/pi_native_alignment_20261018/legacy_source_manifest.json',
                    *sorted(p for p in LEGACY.rglob('*') if p.is_file() and '__pycache__' not in p.parts)]
    sources = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    report = {'schema': 'r22_native_validation_boundary_v1', 'actor_requests': 0,
              'analyst_requests': 0, 'tool_executions': 0, 'scorer_calls': 0,
              'sources': sources, 'inventory': inventory, 'inventory_sha256': sha(args.inventory),
              'scope': 'validation boundary only; does not certify full native Pi harness equivalence',
              'status': 'running_not_certified', 'slots': []}
    write(args.output / 'summary.json', report)
    with native.NativePiValidator(node_executable=args.node, bridge_dir=args.bridge_dir) as validator:
        report['native_identity_before'] = validator.dependency_identity()
        for slot in SLOTS:
            try:
                row = audit_package(args.runs / slot, inventory[slot], validator, prepare,
                                    verify_package, g4.reconstruct_requests, read_json, confined, g4.backend)
            except Exception as exc:
                row = {'slot': slot, 'status': 'protocol_halt', 'error_type': type(exc).__name__, 'error': str(exc)}
            report['slots'].append(row)
            write(args.output / (slot + '.json'), row)
            print(slot, row['status'], flush=True)
        if args.parents:
            plan = json.loads(args.parents.read_text())
            report['failed_response_proof'] = failed_response_proof(plan, validator, verify_package, prepare)
            write(args.parents.parent / 'failed_response_native_proof.json', report['failed_response_proof'])
    # Starting a fresh bridge recomputes the installed dependency tree hash.
    with native.NativePiValidator(node_executable=args.node, bridge_dir=args.bridge_dir) as fresh:
        report['native_identity_after'] = fresh.dependency_identity()
    report['sources_unchanged'] = all(sha(ROOT / path) == h for path, h in sources.items())
    report['inventory_unchanged'] = sha(args.inventory) == report['inventory_sha256']
    report['final_seal_verification'] = {s: verify_package(args.runs / s, expected_index_sha256=inventory[s]) for s in SLOTS}
    report['passed_packages'] = sum(r['status'] == 'pass' for r in report['slots'])
    report['native_validation_failures'] = sum(c['native'].get('ok') is not True for r in report['slots'] for c in r.get('calls', []))
    report['raw_differences'] = sum(not c['raw_equal'] for r in report['slots'] for c in r.get('calls', []))
    report['legacy_effective_differences'] = sum(not c['legacy_effective_equal'] for r in report['slots'] for c in r.get('calls', []))
    report['native_validation_calls'] = sum(len(r.get('calls', [])) for r in report['slots'])
    report['native_loop_stub_responses'] = sum(r.get('native_loop_stub_responses', 0) for r in report['slots'])
    report['status'] = 'pass' if (report['passed_packages'] == 42 and report.get('failed_response_proof', {}).get('prefix_validation', {}).get('status', 'pass') == 'pass' and report['sources_unchanged'] and report['inventory_unchanged'] and report['native_identity_before'] == report['native_identity_after'] and all(v['verified'] for v in report['final_seal_verification'].values())) else 'protocol_halt'
    write(args.output / 'summary.json', report)
    print(json.dumps({k: report[k] for k in ('status','passed_packages','native_validation_failures','raw_differences','legacy_effective_differences','native_validation_calls','native_loop_stub_responses')}), flush=True)


if __name__ == '__main__':
    main()
