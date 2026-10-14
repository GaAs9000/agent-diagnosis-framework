from pathlib import Path
import pytest
from l2_sandbox.pipeline import analyze
from l2_sandbox.experiments.plans import compile_plan
from l2_sandbox.experiments.controller import execute,import_receipts
from l2_sandbox.package import verify
from l2_sandbox.io import L2Error,Unsupported,canonical,read_json,sha_file,inventory
from l2_sandbox.evidence.l1 import L1Reader
from conftest import make_source

@pytest.fixture
def live_labelled_source(tmp_path):
    # Intentionally mimics a live recording label to test gates; it is SYNTHETIC,
    # and the controller is explicitly marked test_fixture via _test_runner.
    return make_source(tmp_path/'live_label_fixture',recording_mode='live')


def test_control_plan_frozen_without_execution(live_labelled_source,tmp_path,config):
    analysis=tmp_path/'a';analyze(live_labelled_source,analysis,config)
    result=compile_plan(analysis,live_labelled_source,'cp000001',tmp_path/'plan')
    assert result['plan']['plan_state']=='frozen'
    assert not result['plan']['execution_authorized'] and not result['plan']['contrast_identifiable']
    assert verify(tmp_path/'plan')['verified']


def test_state_information_intervention_is_blocked_not_faked(live_labelled_source,tmp_path,config):
    analysis=tmp_path/'a';analyze(live_labelled_source,analysis,config)
    hyp=read_json(analysis/'diagnosis.json')['hypotheses'][0]['hypothesis_id']
    p=compile_plan(analysis,live_labelled_source,'cp000001',tmp_path/'plan',hypothesis_id=hyp,template='provide_public_state_summary')
    assert p['plan']['plan_state']=='blocked' and verify(tmp_path/'plan')['verified']
    with pytest.raises(Unsupported):execute(tmp_path/'plan',live_labelled_source,tmp_path/'exec',allow_execution=True)
    assert not (tmp_path/'exec').exists()


def test_fixture_source_cannot_qualify_real_plan(source,tmp_path,config):
    a=tmp_path/'a';analyze(source,a,config)
    p=compile_plan(a,source,'cp000001',tmp_path/'plan')
    assert p['plan']['plan_state']=='blocked' and 'source_not_live_recording' in p['plan']['blockers']


def test_no_implicit_branch_authorization(live_labelled_source,tmp_path,config):
    a=tmp_path/'a';analyze(live_labelled_source,a,config);compile_plan(a,live_labelled_source,'cp000001',tmp_path/'p')
    with pytest.raises(L2Error):execute(tmp_path/'p',live_labelled_source,tmp_path/'e')
    assert not (tmp_path/'e').exists()


def test_parent_grading_error_does_not_block_valid_checkpoint(tmp_path,config):
    source=make_source(tmp_path/'s',recording_mode='live',missing_submission=True)
    a=tmp_path/'a';analyze(source,a,config)
    result=compile_plan(a,source,'cp000001',tmp_path/'plan')
    assert result['plan']['plan_state']=='frozen'


def fixture_runner(config,output,*,parent,checkpoint_id):
    r=L1Reader(parent)
    lineage={'parent_episode_id':r.episode_id,'parent_index_sha256':r.index_sha256,
             'parent_checkpoint_id':checkpoint_id,'parent_checkpoint_sha256':r.files[f'checkpoints/{checkpoint_id}.json']['sha256']}
    make_source(output,recording_mode='live',lineage=lineage)
    return {'fixture':True}


def test_explicit_baseline_controller_and_receipt_import(live_labelled_source,tmp_path,config):
    a=tmp_path/'a';analyze(live_labelled_source,a,config);before=inventory(live_labelled_source)
    compile_plan(a,live_labelled_source,'cp000001',tmp_path/'p',replicates=2)
    result=execute(tmp_path/'p',live_labelled_source,tmp_path/'e',allow_execution=True,_test_runner=fixture_runner)
    assert result['verification']['verified'],result
    assert result['results']['mode']=='test_fixture' and result['results']['hypothesis_verdict']=='untested'
    assert result['results']['groups']['control']['attempted']==2
    assert inventory(live_labelled_source)==before
    imported=import_receipts(a,tmp_path/'e',tmp_path/'v')
    assert imported['verified'],imported
    assert read_json(tmp_path/'v/statistical_analysis.json')['hypothesis_verdict']=='not_identifiable'
    assert verify(a)['verified']


def test_failed_branches_retained_in_denominator(live_labelled_source,tmp_path,config):
    a=tmp_path/'a';analyze(live_labelled_source,a,config);compile_plan(a,live_labelled_source,'cp000001',tmp_path/'p',replicates=2)
    def fail(config,output,*,parent,checkpoint_id):raise RuntimeError('fixture infrastructure failure')
    result=execute(tmp_path/'p',live_labelled_source,tmp_path/'e',allow_execution=True,_test_runner=fail)
    assert result['verification']['verified'],result
    assert result['results']['plan_state']=='execution_invalid'
    assert result['results']['groups']['control']['attempted']==2
    assert result['results']['groups']['control']['endpoint_undefined']==2
    assert result['results']['groups']['control']['mean_on_defined_only'] is None

@pytest.mark.parametrize('template',['rewrite_everything','strong_agent_takeover','shell','../x'])
def test_unregistered_template_refused(template,live_labelled_source,tmp_path,config):
    a=tmp_path/'a';analyze(live_labelled_source,a,config)
    with pytest.raises(L2Error):compile_plan(a,live_labelled_source,'cp000001',tmp_path/'p',template=template)


def test_predicted_hypothesis_must_exist(live_labelled_source,tmp_path,config):
    a=tmp_path/'a';analyze(live_labelled_source,a,config)
    with pytest.raises(L2Error):compile_plan(a,live_labelled_source,'cp000001',tmp_path/'p',hypothesis_id='invented')
