from pathlib import Path
import copy
import json
import os
import pytest
from l2_sandbox.io import IntegrityError,relative,loads,canonical,read_json,sha_file,inventory
from l2_sandbox.evidence.l1 import L1Reader,inspect
from l2_sandbox.pipeline import analyze
from l2_sandbox.package import verify,seal

@pytest.mark.parametrize('path',['../x','/tmp/x','a/../x','a\\x','a//x','a/','C:/a','a\x00b','.',''])
def test_bad_paths(path):
    with pytest.raises(IntegrityError):relative(path)

@pytest.mark.parametrize('data',[b'{"a":1,"a":2}',b'{"x":NaN}',b'{"x":Infinity}',b'{broken',b'\xff'])
def test_bad_json(data):
    with pytest.raises(IntegrityError):loads(data)


def test_real_source_readonly_hashes(source,tmp_path,config):
    before=inventory(source);r=analyze(source,tmp_path/'analysis',config)
    assert r['status']=='completed_rules' and r['verification']['verified'],r
    assert inventory(source)==before
    assert verify(tmp_path/'analysis',source=source)['verified']

@pytest.mark.parametrize('target',['raw/tools/stdout.bin','states/snapshots/s000001.json','benchmark/source/private.py'])
def test_source_tampering_fails(source,target):
    (source/target).write_bytes(b'changed')
    assert not inspect(source)['verified']


def test_source_extra_file_fails(source):
    (source/'extra').write_bytes(b'x');assert not inspect(source)['verified']


def test_source_symlink_fails(source):
    (source/'link').symlink_to('/etc/passwd');assert not inspect(source)['verified']


def test_output_never_inside_source(source,config):
    with pytest.raises(Exception):analyze(source,source/'bad',config)
    assert not (source/'bad').exists()


def test_output_not_overwritten(source,tmp_path,config):
    out=tmp_path/'a';analyze(source,out,config);h=sha_file(out/'evidence_index.json')
    with pytest.raises(FileExistsError):analyze(source,out,config)
    assert sha_file(out/'evidence_index.json')==h


def test_failed_source_retains_failure_package(source,tmp_path,config):
    (source/'raw/tools/stdout.bin').unlink()
    r=analyze(source,tmp_path/'failed',config)
    assert r['status']=='analysis_failed' and r['verification']['verified'],r
    assert not read_json(tmp_path/'failed/diagnosis.json')['localizations']


def test_external_digest_rejects(source,tmp_path,config):
    r=analyze(source,tmp_path/'a',config)
    assert not verify(tmp_path/'a',expected_index='0'*64)['verified']


def reseal(root):
    (root/'evidence_index.json').unlink();seal(root)


def test_rehashed_fabricated_fact_rejected(source,tmp_path,config):
    out=tmp_path/'a';analyze(source,out,config)
    lines=[json.loads(x) for x in (out/'facts.jsonl').read_text().splitlines()]
    f=next(x for x in lines if x['fact_type']=='tool_exit');f['value']['exit_code']=73
    (out/'facts.jsonl').write_bytes(b''.join(canonical(x)+b'\n' for x in lines));reseal(out)
    assert not verify(out)['verified']


def test_derived_probe_tamper_rejected(source,tmp_path,config):
    out=tmp_path/'a';analyze(source,out,config)
    p=next((out/'derived/objects').glob('*.json'));d=read_json(p);d['summary']['tables_recursive']=99;p.write_bytes(canonical(d))
    reseal(out);assert not verify(out)['verified']


def test_missing_source_blob_rejected(source,tmp_path,config):
    out=tmp_path/'a';analyze(source,out,config);next((out/'source_blobs').iterdir()).unlink()
    assert not verify(out)['verified']


def test_untested_cannot_be_promoted(source,tmp_path,config):
    out=tmp_path/'a';analyze(source,out,config);d=read_json(out/'diagnosis.json');d['hypotheses'][0]['verdict']='supported'
    (out/'diagnosis.json').write_bytes(canonical(d));reseal(out);assert not verify(out)['verified']
