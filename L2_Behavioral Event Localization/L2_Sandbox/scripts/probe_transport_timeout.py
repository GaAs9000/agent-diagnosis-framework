"""Single authorized transport diagnostic; never resumes an episode or executes tools."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import urllib.request

from c3_sandbox.l1.models.native import OpenAICompatibleBackend
from c3_sandbox.l1.storage import canonical

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'L2_Behavioral Event Localization/L2_Sandbox/validation/d0_20261018'
ANSWER = '允许单次600秒诊断重放（推荐）'
QUESTION = 'call_5088bbd22d3f47a0b910240eb29270e3'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as handle:
        handle.write(canonical(value)); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_plan(base=BASE):
    proposal_path = base / 'serial_verification/proposed_timeout_diagnostic.json'
    proposal = read(proposal_path)
    serial_path = base / 'serial_resume/plan.json'
    serial = read(serial_path)
    request = ROOT / proposal['input_request_path']
    config = request.parents[2] / 'resolved_config.json'
    # request lives raw/model/rXXXXXX; configuration lives raw/resolved_config.json.
    if sha(request) != proposal['input_request_file_sha256']:
        raise ValueError('source request differs from the reviewed proposal')
    if sha(config) != proposal['resolved_config_sha256']:
        raise ValueError('source config differs from the reviewed proposal')
    bound = {str(proposal_path): sha(proposal_path), str(serial_path): sha(serial_path),
             str(request): sha(request), str(config): sha(config),
             str(Path(__file__).resolve()): sha(__file__)}
    bound.update({str(ROOT / p): digest for p, digest in serial['source_fingerprints'].items()})
    bound.update(serial['predecessor_files'])
    plan = dict(schema_version='transport-timeout-probe-1', d0_frozen=False,
                request_path=str(request), config_path=str(config), bound_files=bound,
                serial_dir=str(base / 'serial_resume'), output_dir=str(base / 'timeout_probe'),
                old_http_attempts=serial['old_http_attempts'], aggregate_http_cap=3360,
                service_stop_epoch=serial['service_stop_epoch'], timeout_seconds=600,
                safety_seconds=645, max_wait_seconds=21600, poll_seconds=15,
                max_http_attempts=1, retry_attempts=0, execute_tools=False,
                copy_into_formal_packages=False, scientific_efficacy='not_tested')
    plan['plan_sha256'] = hashlib.sha256(canonical(plan)).hexdigest()
    return plan


def check_files(plan):
    for path, expected in plan['bound_files'].items():
        if sha(path) != expected:
            raise ValueError('bound input changed: ' + path)


def metrics():
    # Fixed loopback route used by the formal source request. Never infer idle from failure.
    with urllib.request.urlopen('http://127.0.0.1:8001/metrics', timeout=5) as response:
        text = response.read(4 * 1024 * 1024).decode('utf-8')
    values = {}
    for key in ('vllm:num_requests_running', 'vllm:num_requests_waiting'):
        rows = [line for line in text.splitlines() if line.startswith(key + '{') or line.startswith(key + ' ')]
        if not rows:
            raise ValueError('missing server request metric: ' + key)
        samples = [float(line.rsplit(None, 1)[1]) for line in rows]
        if any(not math.isfinite(value) or value < 0 for value in samples):
            raise ValueError('invalid server request metric: ' + key)
        values[key] = sum(samples)
    return values


def validate_request(plan):
    config = read(plan['config_path'])['model']
    body = read(plan['request_path'])
    if (config['kind'] != 'local' or config['model'] != 'Qwen/Qwen3.6-27B'
            or config['base_url'] != 'http://127.0.0.1:8001/v1'
            or config['timeout_seconds'] != 300 or body.get('model') != config['model']
            or body.get('temperature') != 1 or body.get('max_tokens') != 8192
            or 'seed' in body or body.get('stream') is not False or body.get('n') != 1
            or config['parameters'].get('temperature') != 1
            or config['parameters'].get('max_tokens') != 8192
            or 'seed' in config['parameters']):
        raise ValueError('formal request/configuration identity differs')
    return config, canonical(body)


def execute(plan, approval, *, backend_factory=OpenAICompatibleBackend,
            metrics_reader=metrics, now=time.time, sleep=time.sleep):
    unsigned = {k: v for k, v in plan.items() if k != 'plan_sha256'}
    if hashlib.sha256(canonical(unsigned)).hexdigest() != plan['plan_sha256']:
        raise ValueError('plan hash mismatch')
    if (approval.get('approved_plan_sha256') != plan['plan_sha256']
            or approval.get('exact_answer') != ANSWER or approval.get('question_id') != QUESTION
            or approval.get('internal_plan_reviewed') is not True):
        raise ValueError('explicit reviewed one-request authorization required')
    if (plan['timeout_seconds'], plan['safety_seconds'], plan['max_http_attempts'],
            plan['retry_attempts'], plan['aggregate_http_cap'], plan['max_wait_seconds']) != (600, 645, 1, 0, 3360, 21600):
        raise ValueError('diagnostic limits changed')
    output = Path(plan['output_dir']); output.mkdir(parents=True, exist_ok=True)
    if (output / 'attempt_started.json').exists():
        raise ValueError('single attempt already reserved; never retry after crash')
    started = now()
    final = Path(plan['serial_dir']) / 'supervisor_result.json'
    while True:
        check_files(plan)
        if now() - started >= plan['max_wait_seconds'] or plan['service_stop_epoch'] - now() < 645:
            raise RuntimeError('wait/deployment deadline reached; no request sent')
        if final.exists():
            supervisor = read(final)
            if not isinstance(supervisor.get('exit_code'), int) or not supervisor.get('finished_at_utc'):
                raise ValueError('unsealed supervisor result')
            idle = metrics_reader()
            if set(idle) != {'vllm:num_requests_running', 'vllm:num_requests_waiting'}:
                raise ValueError('unknown server activity')
            if all(value == 0 for value in idle.values()):
                break
        sleep(plan['poll_seconds'])
    # The full serial supervisor (including G2) has ended. Seal the remaining budget receipt.
    ledger_path = Path(plan['serial_dir']) / 'admission.json'
    ledger = read(ledger_path)
    requests = ledger.get('requests')
    if type(requests) is not int or requests < 0 or len(ledger.get('sends', [])) != requests:
        raise ValueError('invalid final admission ledger')
    if plan['old_http_attempts'] != 93 or 93 + requests + 1 > 3360:
        raise ValueError('aggregate HTTP budget exhausted')
    config, body = validate_request(plan)
    check_files(plan)
    backend = backend_factory(config)
    reservation = dict(plan_sha256=plan['plan_sha256'], started_epoch=now(),
                       canonical_request_sha256=hashlib.sha256(body).hexdigest(),
                       serial_supervisor_sha256=sha(final), serial_ledger_sha256=sha(ledger_path),
                       old_http_attempts=93, serial_http_attempts=requests,
                       aggregate_including_probe=93 + requests + 1,
                       timeout_seconds=600, no_retries=True)
    if plan['service_stop_epoch'] - now() < 645:
        raise RuntimeError('insufficient service lifetime')
    # O_EXCL is the cross-process/crash barrier: uncertainty consumes the single authorization.
    with (output / 'attempt_started.json').open('xb') as handle:
        handle.write(canonical(reservation)); handle.flush(); os.fsync(handle.fileno())
    save(output / 'executed_plan.json', plan)
    save(output / 'authorization.json', approval)
    try:
        response = backend.send(body, 600)
    except Exception as exc:
        save(output / 'result.json', dict(status='unknown_transport_exception', exception_type=type(exc).__name__,
                                         http_attempts_reserved=1, tools_executed=0))
        raise
    raw_path = output / 'response.bin'
    with raw_path.open('xb') as handle:
        handle.write(response.body); handle.flush(); os.fsync(handle.fileno())
    save(output / 'http_metadata.json', response.metadata)
    result = dict(status='transport_response' if response.status == 200 and not response.metadata.get('transport_error')
                  and not response.metadata.get('body_truncated') else 'transport_failure',
                  http_status=response.status, raw_response_sha256=sha(raw_path),
                  http_attempts=1, tools_executed=0, scientific_efficacy='not_tested')
    try:
        value = json.loads(response.body)
        result['usage'] = value.get('usage')
        result['finish_reasons'] = [c.get('finish_reason') for c in value.get('choices', [])]
    except (ValueError, UnicodeError, TypeError, AttributeError):
        result['raw_json_parse'] = 'failed; original bytes retained'
    save(output / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=BASE / 'timeout_probe/plan.json')
    parser.add_argument('--approval', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        if not args.approval:
            parser.error('--execute requires --approval')
        print(json.dumps(execute(read(args.plan), read(args.approval)), ensure_ascii=False))
    else:
        plan = build_plan(); args.plan.parent.mkdir(parents=True, exist_ok=True)
        save(args.plan, plan)
        print(json.dumps(dict(plan=str(args.plan), plan_sha256=plan['plan_sha256'], executed=False)))


if __name__ == '__main__':
    main()
