"""Small, explicitly synthetic fixtures. No fixture is a model/sandbox qualification."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import pytest


def encoded(x): return json.dumps(x, ensure_ascii=False, indent=2).encode()
def sha(b): return hashlib.sha256(b).hexdigest()
def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded(value))


@pytest.fixture
def make_source(tmp_path):
    serial = 0
    def make(*, stopped_reason='assistant_finished', completed=False, protocol='valid', initial_validity='valid', final_validity='valid', no_next_request=False, provider_error=False):
        nonlocal serial
        serial += 1
        source = tmp_path / f'synthetic-run-{serial}' / 'target' / 'officeval_fixture'
        for folder, content in [('public-input',b'before'),('final-workspace',b'after')]:
            (source / folder).mkdir(parents=True)
            (source / folder / 'artifact.txt').write_bytes(content)
        grades = {}
        for phase, validity, content in [('initial',initial_validity,b'before'),('final',final_validity,b'after')]:
            process = {'exit_code':0,'stdout':'fixture-grade','stderr':'','error':None,'truncated':False}
            directory = source / (phase + '-grade')
            put(directory / 'process.json', process)
            (directory / 'stdout.txt').write_text(process['stdout'])
            (directory / 'stderr.txt').write_text('')
            g = {'official_raw':{'fixture':True,'dim1_pass':True,'dim2_items':[]},
                 'validity':validity,'c3_operational_completed':completed if phase=='final' else False,
                 'outcome':'task_succeeded' if completed and phase=='final' else 'task_failed',
                 'reason':'fixture_completed' if completed else 'requirements_unmet','coverage_complete':True,'unassessed_rules':[],
                 'grader_image':'sha256:'+'a'*64,'artifact':{'name':'artifact.txt','sha256':sha(content),'size_bytes':len(content)},
                 'raw_process':process,'process_sha256':sha((directory/'process.json').read_bytes()),
                 'stdout_sha256':sha(b'fixture-grade'),'stderr_sha256':sha(b'')}
            put(directory/'grade.json',g)
            grades[phase+'_grade']=g
        fc = {'functionCall':{'name':'workspace_python','args':{'code':"print('synthetic fixture')"},'id':'native-123'}}
        req1 = {'contents':[{'role':'user','parts':[{'text':'Synthetic fixture task'}]}],
                'systemInstruction':{'parts':[{'text':'Fixture; not a live agent'}]},
                'tools':[{'functionDeclarations':[{'name':'workspace_python'}]}]}
        res1 = {'candidates':[{'content':{'parts':[fc]}}], 'usageMetadata':{'totalTokenCount':10},'modelVersion':'fixture-model'}
        delivered = {'exit_code':1,'stdout':'visible prefix','stderr':'visible error','context_output_truncated':True}
        req2 = copy.deepcopy(req1)
        req2['contents'] += [{'role':'model','parts':[fc]}, {'role':'user','parts':[{'functionResponse':{'name':'workspace_python','response':delivered}}]}]
        res2 = {'candidates':[{'content':{'parts':[{'text':'done'}]}}], 'usageMetadata':{'totalTokenCount':12},'modelVersion':'fixture-model'}
        records = []
        pairs = [(req1,res1)] if no_next_request else [(req1,res1),(req2,res2)]
        if provider_error: pairs = [(req1,None)]
        for request,response in pairs:
            rb,sb = encoded(request), encoded(response) if response is not None else b''
            records.append({'request_raw':rb.decode(),'response_raw':sb.decode(), 'request_identity':sha(rb),'response_identity':sha(sb),
                            'usage':response.get('usageMetadata') if response else None,'selector_invalid':False,'error':'transport failed' if provider_error else None})
        delta = {'path':'artifact.txt','before':{'kind':'file','sha256':sha(b'before'),'size_bytes':6},
                 'after':{'kind':'file','sha256':sha(b'after'),'size_bytes':5}}
        raw = {'exit_code':1,'stdout':'visible prefix PLUS raw text','stderr':'visible error PLUS raw stderr', 'elapsed':0.25,'truncated':False}
        dispatch = {'tool_call_id':'legacy-0','name':'workspace_python','arguments':fc['functionCall']['args'],
                    'raw_result':raw,'before_path':'/old/source/tool-dispatch/call-000/before.json','after_path':'/old/source/tool-dispatch/call-000/after.json',
                    'fs_delta':[delta],'snapshot_ineligible_reasons':[],'evidence_kind':'actual_model_tool_dispatch'}
        result = {'task_id':'officeval_fixture','provider':'fixture-provider','model':'fixture-model','seed':2026,'is_synthetic':True,
                  'output_dir':'/old/source','runtime':{'model_fingerprint':'fixture-label','sampling':{'seed':2026,'max_turns':2},'tool_config':{}},
                  **grades,'agent_result':{'stopped_reason':stopped_reason,'protocol_status':protocol,'events':[]},
                  'raw_gemini_records':records,'tool_dispatch':[] if provider_error else [dispatch], 'checkpoints':[],
                  'execution_valid':True,'task_completed':completed,'outcome':'success' if completed else 'failure'}
        put(source/'result.json', result)
        put(source.parent.parent/'config.lock', {'config_sha256':'b'*64})
        return source
    return make
