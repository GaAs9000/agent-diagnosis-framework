from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest
from c3_sandbox.l1.benchmarks.officeval.adapter import NativeOfficeVal, BundleOfficeVal,task_name
from c3_sandbox.l1.benchmarks.officeval.source import lock_local
from c3_sandbox.l1.core.contracts import ConfigurationError,CapabilityError,EvidenceError
from c3_sandbox.l1.core.orchestrator import run
from c3_sandbox.l1.recording.state import StateStore,restore
from c3_sandbox.l1.evidence.live import validate_snapshot
from c3_sandbox.l1.storage import read_json,file_digest
from c3_sandbox.l1.evaluation.officeval import interpret
from .conftest import BUNDLE,ROOT,ScriptedGemini,TrustedFixtureRuntime,FixtureEvaluator


def test_native_load_not_bundle(native_source):
    a=NativeOfficeVal(native_source);t=a.load('officeval_003')
    assert t.provenance['adapter']=='officeval-native'
    assert t.provenance['data_lock_status']=='sha256_locked'
    assert t.prompt and len(t.assets)==2
    assert 'source/verifiers/officeval_003_verifier.py' in t.private_files
    assert t.evaluation.entrypoint=='verifiers/officeval_003_verifier.py'
    assert a.list_tasks()[0]['assets_ready']


def test_native_episode_fixture(tmp_path,native_source,config,components):
    config['benchmark']=native_source;components['benchmark']=NativeOfficeVal(native_source)
    result=run(config,tmp_path/'native',injected=components)
    assert result['verification']['verified'],result
    assert read_json(tmp_path/'native/manifest.json')['task']['provenance']['adapter']=='officeval-native'


def test_source_wrong_revision(native_source):
    native_source['source_commit']='0'*40
    with pytest.raises(ConfigurationError):NativeOfficeVal(native_source).load('officeval_003')


def test_source_dirty_rejected(native_source):
    (Path(native_source['source_dir'])/'untracked.py').write_text('bad')
    with pytest.raises(ConfigurationError):NativeOfficeVal(native_source).load('officeval_003')


def test_mutated_locked_asset_rejected(native_source):
    p=next((Path(native_source['assets_dir'])/'officeval_003').glob('*.docx'));p.write_bytes(b'changed')
    with pytest.raises(ConfigurationError):NativeOfficeVal(native_source).load('officeval_003')


def test_local_lock_does_not_silently_update(native_source):
    p=next((Path(native_source['assets_dir'])/'officeval_003').glob('*.docx'));p.write_bytes(b'changed')
    with pytest.raises(ConfigurationError):lock_local(native_source,['officeval_003'])


def test_missing_lock_fails_closed(native_source):
    Path(native_source['data_lock']).unlink()
    with pytest.raises(ConfigurationError):NativeOfficeVal(native_source).load('officeval_003')
    native_source['allow_unlocked_data']=True
    assert NativeOfficeVal(native_source).load('officeval_003').provenance['data_lock_status']=='unlocked'


def test_com_requires_other_runtime(native_source):
    with pytest.raises(CapabilityError):NativeOfficeVal(native_source).load('officeval_001')

@pytest.mark.parametrize('value',['003','officeval_000','officeval_101','../../passwd','officeval_003/../x','officeval_3'])
def test_task_id_safety(value):
    with pytest.raises(ConfigurationError):task_name(value)


def test_file_state_restores_modes_dirs_and_versions(tmp_path):
    w=tmp_path/'w';w.mkdir();(w/'empty').mkdir();(w/'a').write_bytes(b'one');os.chmod(w/'a',0o640)
    r=tmp_path/'e';r.mkdir();store=StateStore(r);before=store.capture(w,'initial')
    (w/'a').write_bytes(b'two');(w/'b').write_bytes(b'new');after=store.capture(w,'after')
    assert [x['change'] for x in store.diff(before,after)]==['modified','created']
    restored=tmp_path/'restored';restore(r,before,restored)
    assert (restored/'a').read_bytes()==b'one' and not (restored/'b').exists()
    assert (restored/'empty').is_dir() and (restored/'a').stat().st_mode & 0o777==0o640
    assert store.capture(restored,'restored')['content_sha256']==before['content_sha256']

@pytest.mark.parametrize('kind',['symlink','hardlink','fifo'])
def test_special_files_never_copied(tmp_path,kind):
    r=tmp_path/'e';r.mkdir();w=tmp_path/'w';w.mkdir();outside=tmp_path/'outside';outside.write_bytes(b'x')
    if kind=='symlink':(w/'bad').symlink_to(outside)
    elif kind=='hardlink':os.link(outside,w/'bad')
    else:os.mkfifo(w/'bad')
    with pytest.raises(EvidenceError):StateStore(r).capture(w,'unsafe')


def test_workspace_limits(tmp_path):
    r=tmp_path/'e';r.mkdir();w=tmp_path/'w';w.mkdir();(w/'large').write_bytes(b'12345678')
    with pytest.raises(EvidenceError):StateStore(r,max_bytes=4).capture(w,'large')


def test_state_blob_tampering(tmp_path):
    r=tmp_path/'e';r.mkdir();w=tmp_path/'w';w.mkdir();(w/'a').write_bytes(b'a');s=StateStore(r).capture(w,'a')
    (r/s['files'][0]['blob']).write_bytes(b'changed')
    with pytest.raises(EvidenceError):restore(r,s,tmp_path/'dest')


def raw_grade():
    return {'id':'003','file_name':'a.docx','status':'ok','error':None,'dim1_pass':True,
            'dim1_reason':'','dim2_items':[],'total_score':3,'max_score':5}


def test_ok_not_success_and_official_policy():
    raw=raw_grade();p=interpret(raw,'officeval_003')
    assert p['evaluation_validity']=='valid' and p['task_outcome']=='unknown' and p['completion_rate']==.6
    assert interpret(raw,'officeval_003','full_score_v1')['task_outcome']=='partial'


def test_dim1_failure_can_retain_explanatory_items():
    raw=raw_grade();raw.update(dim1_pass=False,total_score=0,
      dim2_items=[{'rule':'explanation','max_delta':5,'delta':0,'hit':False}])
    assert interpret(raw,'officeval_003')['task_outcome']=='failure'

@pytest.mark.parametrize('key,value',[('id','004'),('status','timeout'),('error','failure'),('dim1_pass','true'),
 ('total_score',float('nan')),('dim2_items','bad'),('max_score',True)])
def test_invalid_grade_not_quality_failure(key,value):
    raw=raw_grade();raw[key]=value
    p=interpret(raw,'officeval_003')
    assert p['evaluation_validity']=='invalid' and p['task_outcome']=='unknown'


def test_real_retained_official_verifier_worker(tmp_path):
    if not (BUNDLE/'inputs/officeval_003').exists():
        pytest.skip('Retained bundle runs/integrated-delivery-001/bundle not present in this checkout')
    # Trusted retained upstream code, real provided input; no new model or Docker.
    submission=tmp_path/'submission';submission.mkdir()
    source=next((BUNDLE/'inputs/officeval_003').glob('*.docx'))
    shutil.copyfile(source,submission/source.name)
    output=tmp_path/'grade'
    worker=ROOT/'src/c3_sandbox/l1/evaluation/worker.py'
    p=subprocess.run([sys.executable,'-I',str(worker),'--source',str(BUNDLE),
        '--entrypoint','verifiers/officeval_003_verifier.py','--submission',str(submission),'--output',str(output)],capture_output=True,timeout=30)
    assert p.returncode==0,p.stderr
    raw=read_json(output/'official.json')
    assert raw['id']=='003' and raw['status']=='ok'
    assert interpret(raw,'officeval_003')['evaluation_validity']=='valid'
    assert file_digest(source)==file_digest(submission/source.name)


def test_restore_rejects_special_permission_bits(tmp_path):
    r=tmp_path/'e';r.mkdir();w=tmp_path/'w';w.mkdir();(w/'special').mkdir();os.chmod(w/'special',0o2755)
    snapshot=StateStore(r).capture(w,'special')
    with pytest.raises(EvidenceError,match='special permission'):
        restore(r,snapshot,tmp_path/'dest')
