from pathlib import Path
import zipfile
import pytest
from l2_sandbox.pipeline import analyze
from l2_sandbox.package import verify
from l2_sandbox.portable import pack,unpack
from l2_sandbox.io import IntegrityError,L2Error,read_json
from l2_sandbox.evaluation.metrics import localization_metrics,brier,family_partition
from l2_sandbox.cli import main


def test_utf8_zip_roundtrip(source,tmp_path,config):
    out=tmp_path/'分析包';analyze(source,out,config)
    r=pack(out,tmp_path/'archive.zip')
    assert r['default_extract_roundtrip_verified']
    unpack(tmp_path/'archive.zip',tmp_path/'unpacked')
    assert verify(tmp_path/'unpacked/分析包')['verified']


def test_zip_traversal_preflight_rejects(tmp_path):
    with zipfile.ZipFile(tmp_path/'bad.zip','w') as z:z.writestr('../escape','data')
    with pytest.raises(IntegrityError):unpack(tmp_path/'bad.zip',tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_zip_duplicate_rejected(tmp_path):
    with zipfile.ZipFile(tmp_path/'bad.zip','w') as z:
        z.writestr('a.txt','1')
        with pytest.warns(UserWarning):z.writestr('a.txt','2')
    with pytest.raises(IntegrityError):unpack(tmp_path/'bad.zip',tmp_path/'out')


def test_zip_size_limit(tmp_path):
    with zipfile.ZipFile(tmp_path/'bad.zip','w') as z:z.writestr('a.txt','large')
    with pytest.raises(IntegrityError):unpack(tmp_path/'bad.zip',tmp_path/'out',max_total_bytes=1)


def test_metrics_do_not_reward_huge_interval():
    pred=[{'case_id':'x','event_sets':[['e1','e2','e3']]}]
    gold=[{'case_id':'x','accepted_event_sets':[['e2']]}]
    assert localization_metrics(pred,gold)['hit_rate']==0


def test_metrics_keep_undefined_and_abstention():
    p=[{'case_id':'x','event_sets':[]},{'case_id':'y','event_sets':[['e']]}]
    g=[{'case_id':'x','accepted_event_sets':[['e']]},{'case_id':'y','accepted_event_sets':[]}]
    r=localization_metrics(p,g)
    assert r['labelled_cases']==1 and r['abstentions']==1


def test_brier_no_undefined_as_zero():
    r=brier([{'probability':.8,'outcome':True},{'probability':.1,'outcome':None}])
    assert r['n_defined']==1 and r['n_undefined']==1 and abs(r['brier']-.04)<1e-10

@pytest.mark.parametrize('value',[-1,2,float('nan'),True])
def test_invalid_probability_rejected(value):
    with pytest.raises(L2Error):brier([{'probability':value,'outcome':True}])


def test_single_family_cannot_be_independent_test_set():
    r=family_partition([{'case_id':str(i),'family_id':'officeval_003'} for i in range(9)])
    assert r['status']=='insufficient_independent_families'


def test_cli_error_is_machine_readable(tmp_path,capsys):
    assert main(['verify',str(tmp_path/'missing')])==2
    assert 'verified' in capsys.readouterr().out


def test_cli_smoke(source,tmp_path,capsys):
    assert main(['analyze','--source',str(source),'--output',str(tmp_path/'a')])==0
    capsys.readouterr()
    assert main(['verify',str(tmp_path/'a')])==0
    capsys.readouterr()
    fid=read_json(tmp_path/'a/diagnosis.json')['localizations'][0]['supporting_fact_ids'][0]
    assert main(['show-fact',str(tmp_path/'a'),'--fact',fid])==0
