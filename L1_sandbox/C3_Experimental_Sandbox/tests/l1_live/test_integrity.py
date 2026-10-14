import copy
import json
from pathlib import Path
import pytest
from c3_sandbox.l1.core.orchestrator import run
from c3_sandbox.l1.evidence.live import verify,scan,seal
from c3_sandbox.l1.storage import canonical,digest,read_json
from c3_sandbox.l1.core.registry import Registry
from .conftest import gemini_call,gemini_done


def reseal(root,events=None):
    if events is None:events=scan(root/'trajectory.jsonl')['events']
    previous=None
    for event in events:
        event['previous_sha256']=previous
        event['sha256']=digest({k:v for k,v in event.items() if k!='sha256'});previous=event['sha256']
    (root/'trajectory.jsonl').write_bytes(b''.join(canonical(e)+b'\n' for e in events))
    (root/'evidence_index.json').unlink()
    seal(root,events)


def test_raw_tamper_detected(tmp_path,config,components):
    out=tmp_path/'e';run(config,out,injected=components)
    next((out/'raw/tools').rglob('stdout.bin')).write_bytes(b'forged')
    assert not verify(out)['verified']


def test_native_call_projection_detects_rehashed_lie(tmp_path,config,components):
    out=tmp_path/'e';run(config,out,injected=components)
    events=scan(out/'trajectory.jsonl')['events']
    for e in events:
        if e['kind']=='tool.call':e['data']['arguments']={'command':'invented command'}
    reseal(out,events)
    r=verify(out)
    assert not r['verified'] and any(e['code']=='native_call_projection_mismatch' for e in r['errors'])


def test_state_diff_rehash_does_not_hide_forgery(tmp_path,config,components):
    out=tmp_path/'e';run(config,out,injected=components)
    (out/'state_changes.jsonl').write_bytes(b'')
    reseal(out)
    r=verify(out)
    assert not r['verified'] and any(e['code']=='state_diff_mismatch' for e in r['errors'])


def test_index_external_anchor(tmp_path,config,components):
    out=tmp_path/'e';run(config,out,injected=components)
    assert not verify(out,expected_index_sha256='0'*64)['verified']


def test_snapshot_blob_mismatch_even_after_reseal(tmp_path,config,components):
    out=tmp_path/'e';run(config,out,injected=components)
    next((out/'states/blobs').iterdir()).write_bytes(b'fake')
    reseal(out)
    assert not verify(out)['verified']


def test_valid_prefix_after_torn_write(tmp_path,config,components):
    out=tmp_path/'e';run(config,out,injected=components)
    original=scan(out/'trajectory.jsonl')
    with (out/'trajectory.jsonl').open('ab') as f:f.write(b'{"partial":')
    result=scan(out/'trajectory.jsonl')
    assert not result['complete'] and result['valid_prefix_events']==original['valid_prefix_events']
    assert (out/'trajectory.jsonl').read_bytes().endswith(b'{"partial":')


def test_no_grading_stale_snapshot_when_runtime_not_quiescent(tmp_path,config,components):
    runtime=components['runtime']
    def fail(call,workspace,timeout):
        runtime.is_quiescent=False
        raise RuntimeError('container cleanup failed')
    runtime.execute=fail
    out=tmp_path/'e';r=run(config,out,injected=components)
    assert r['verification']['verified'] and r['status']['evaluation_validity']=='not_run'
    assert read_json(out/'artifacts/inventory.json')['snapshot_id'] is None
    assert r['status']['evidence_completeness']=='incomplete'


def test_extensible_plugin_options_without_core_changes():
    r=Registry('example');r.register('benchmark_b2',lambda cfg:cfg)
    result=r.create('benchmark_b2',{'adapter':'benchmark_b2','options':{'browser_state_path':'browser.json','reset_mode':'transaction'}})
    assert result['reset_mode']=='transaction' and 'options' not in result


def test_checkpoint_byte_tampering_rejected(tmp_path,config,components):
    out=tmp_path/'e';run(config,out,injected=components)
    cp=out/'checkpoints/cp000000.json';v=read_json(cp);v['conversation']['contents'][0]['parts'][0]['text']='forged task'
    cp.write_bytes(canonical(v)+b'\n');reseal(out)
    assert not verify(out)['verified']
