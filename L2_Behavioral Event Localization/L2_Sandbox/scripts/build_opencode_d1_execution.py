#!/usr/bin/env python3
"""Freeze a separate Qwen-parent -> OpenCode continuation batch, then execute it."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
L2 = ROOT / 'L2_Behavioral Event Localization/L2_Sandbox'
L1 = ROOT / 'L1_sandbox/C3_Experimental_Sandbox'
LOGICAL = Path('/Users/jiashen/Desktop/问题诊断')
sys.path[:0] = [str(L2 / 'src'), str(L1 / 'src')]
from l2_sandbox.v3_execution import file_digest, read_json, seal, verify_seal, write_json


def mapped(path):
    path = Path(path)
    for prefix in (LOGICAL, Path('/home/jiasheren2/projects/agent_diagnosis_d0_20261018/Users/jiashen/Desktop/问题诊断')):
        if path.is_relative_to(prefix):
            return ROOT / path.relative_to(prefix)
    return path


def logical(path):
    return str(LOGICAL / Path(path).resolve().relative_to(ROOT))


def build(phase: Path):
    from l2_sandbox.opencode_admission import DEFAULT_CAPS
    phase = phase.resolve()
    old = L2 / 'validation/p4_contract_v2_20261019'
    plan_path, freeze_path = old / 'd1_plan.json', old / 'd0_scientific_freeze_v2.json'
    science, freeze = read_json(plan_path), read_json(freeze_path)
    verify_seal(science, 'plan_sha256')
    verify_seal(freeze, 'freeze_sha256')
    if (not freeze.get('d0_frozen') or freeze['plan_sha256'] != science['plan_sha256']
            or not science.get('scientific_plan_ready') or science.get('readiness_errors')
            or science['measurement_gate_binding']['gates'] != {f'V3-G{i}': True for i in range(1,5)}):
        raise ValueError('complete sealed scientific inputs required')
    archive = phase / 'historical_sources.tar'
    if not archive.is_file():
        raise ValueError('historical source archive required before changing execution code')
    historical = phase / 'historical_sources'
    bound, substituted = {}, {}
    for name, expected in freeze['bound_files'].items():
        current = mapped(name)
        if current.is_symlink() or not current.is_file():
            raise ValueError('missing original bound input: ' + str(current))
        if file_digest(current) != expected:
            relative = current.relative_to(ROOT)
            allowed = any(relative.is_relative_to(prefix) for prefix in (
                Path('L1_sandbox/C3_Experimental_Sandbox/src'),
                Path('L2_Behavioral Event Localization/L2_Sandbox/src'),
                Path('L2_Behavioral Event Localization/L2_Sandbox/scripts'), Path('scripts')))
            prior = historical / relative
            if not allowed or not prior.is_file() or prior.is_symlink() or file_digest(prior) != expected:
                raise ValueError('historical input changed without exact source archive: ' + str(relative))
            substituted[logical(current)] = {'path':logical(prior),'sha256':expected}
            current = prior
        bound[logical(current)] = expected
    for path in (plan_path, freeze_path, archive):
        bound[logical(path)] = file_digest(path)
    # Freeze only the model section; preserve native Pi, tools, images and budgets.
    model_template = L1 / 'configs/l1/officeval.opencode-go-deepseek-v41flash.pi.t1.json'
    model = copy.deepcopy(read_json(model_template)['model'])
    model['timeout_seconds'] = 600
    model['max_response_bytes'] = 16777216
    if (model['adapter'], model['provider'], model['model']) != ('opencode-chat','opencode-go','deepseek-v4.1-flash'):
        raise ValueError('unexpected continuation actor')
    bound[logical(model_template)] = file_digest(model_template)
    prospective = read_json(L2 / 'validation/a_plus_i_execute_20261018/prospective_freeze.json')
    base = {k:copy.deepcopy(prospective[k]) for k in ('actor_image_id','actor_image_env','actor_user','actor_image_inspect')}
    base['actor_image_inspect']['path'] = logical(mapped(base['actor_image_inspect']['path']))
    base['native_dependency_identity'] = copy.deepcopy(prospective['native_execution']['dependency_identity'])
    wrapper = L1 / 'validation/pi_native_alignment_20261018/runtime/node-validator-remote.sh'
    base['wrapper'] = {'path':logical(wrapper),'sha256':file_digest(wrapper)}
    bound[logical(wrapper)] = file_digest(wrapper)
    paths = set()
    for directory in (L1/'src', L2/'src'):
        paths.update(p for p in directory.rglob('*') if p.is_file() and p.suffix in ('.py','.json','.js','.mjs') and '__pycache__' not in p.parts)
    for directory in (L2/'scripts',ROOT/'scripts'):
        paths.update(directory.glob('*.py'))
    sources = {logical(p):file_digest(p) for p in sorted(paths)}
    output = seal({
        'schema':'cross-actor-d1-execution-v1',
        'authorization':{'exact_user_text':'开始吧，直接用opencode','execution_authorized':True,
                         'scope':'cross_actor_D1', 'source':'direct_user_message_20261019'},
        'interpretation':'Qwen parent states -> DeepSeek continuation intervention effects; separate from same-actor D1',
        'scientific_plan':science,
        'source_scientific_plan':{'path':logical(plan_path),'sha256':file_digest(plan_path)},
        'source_scientific_freeze':{'path':logical(freeze_path),'sha256':file_digest(freeze_path)},
        'historical_source_commit':'e8c897014a5719e835bb5575b3ccaf3adea6cc49',
        'historical_source_substitutions':substituted,
        'source_files':sources,'bound_files':bound,
        'runs_dir':str(LOGICAL / 'L1_sandbox/C3_Experimental_Sandbox/runs/v3_qwen36_pi_prospective_dev45_20261018/episodes'),
        'model':model,'observer_base':base,'execution_concurrency':3,
        'limits':copy.deepcopy(DEFAULT_CAPS),
        'operational_limits_origin':'conservative operator defaults within user-authorized HTTP/wall limits',
        'cost_scope':'12 USD upper-bound provider usage equivalent, not a claim of additional cash billing; Go quota/balance settings unchanged',
        'pricing_source':'https://docs.opencode.ai/docs/go/',
        'invalid_policy':'stop first invalid attempt; never replace parent or branch sample',
        'execution_location':'remote OpenCode inference and aml-b native validation; existing local ARM64 Docker runtime exception',
        'automatic_d2':False,
    })
    path = phase / 'execution_plan.json'
    if path.exists() and read_json(path) != output:
        raise ValueError('refusing to replace frozen execution plan')
    write_json(path, output)
    print(json.dumps({'plan_sha256':output['artifact_sha256'],'branches':science['budget_and_stop_conditions']['total_planned_branches'],
                      'historical_source_substitutions':len(substituted),'limits':output['limits']},ensure_ascii=False))
    return output


def load_credential():
    name = 'OPENCODE_API_KEY'
    if os.environ.get(name):
        return
    path = L1 / '.env'
    if path.is_symlink() or not path.is_file():
        raise ValueError('local credential file unavailable')
    value = None
    for line in path.read_text().splitlines():
        match = re.fullmatch(r'\s*(?:export\s+)?OPENCODE_API_KEY\s*=\s*(.*?)\s*',line)
        if match:
            value = match[1]
            if value[:1] in ('"',"'"):
                if len(value)<2 or value[-1] != value[0]:
                    raise ValueError('malformed credential')
                value=value[1:-1]
            else:
                value=value.split(' #',1)[0].strip()
    if not value or any(c in value for c in ('\n','\r','$','`')):
        raise ValueError('credential missing or unsupported')
    os.environ[name]=value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.execute:
        build(args.phase)
        return
    from l2_sandbox.opencode_d1_runtime import execute
    plan=read_json(args.phase / 'execution_plan.json')
    load_credential()
    os.environ['C3_PI_NODE']=plan['observer_base']['wrapper']['path']
    os.environ['C3_PI_NATIVE_DIR']=str(Path(os.environ['C3_PI_NODE']).parent / 'pi_native')
    execute(plan, args.phase / 'execution')


if __name__=='__main__':
    main()
