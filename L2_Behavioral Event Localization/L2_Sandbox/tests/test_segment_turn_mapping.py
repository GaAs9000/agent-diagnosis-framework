"""Cumulative segment turns and local checkpoint identity, without models or Docker."""
import pytest

from conftest import make_l1_episode
from l2_sandbox.evidence.l1 import L1Package
from l2_sandbox.facts.build import build_facts
from l2_sandbox.grades import grade_episode_snapshots
from l2_sandbox.io import IntegrityError, L2Error, read_json, write_json
import l2_sandbox.grades as grades


def segment(root, *, start=21, requests=9, parent_cp='cp000013', include_budget=True):
    make_l1_episode(root, num_turns=requests)
    manifest=read_json(root/'manifest.json')
    manifest['lineage']={'parent_checkpoint_id':parent_cp} if parent_cp else None
    write_json(root/'manifest.json',manifest)
    for path in (root/'checkpoints').glob('cp*.json'):
        cp=read_json(path)
        if include_budget:
            cp['budget_used']['model_requests']=start+int(cp['checkpoint_id'][2:])
        else:
            cp['budget_used'].pop('model_requests',None)
        write_json(path,cp)
    write_json(root/'benchmark/description.json',{'task_id':'officeval_003','evaluation':{
        'kind':'officeval-python','entrypoint':'fixture.py','metadata':{}}})
    return L1Package(root)


@pytest.mark.parametrize('start,requests,parent_cp,total',[(21,9,'cp000013',30),(7,14,'cp000007',21),(0,3,None,3)])
def test_segment_offset_and_facts_use_spent_budget(tmp_path,start,requests,parent_cp,total):
    pkg=segment(tmp_path/'episode',start=start,requests=requests,parent_cp=parent_cp)
    assert pkg.start_turn==start and pkg.turns==total and pkg.local_turns==requests
    assert set(pkg.checkpoints())==set(range(requests))
    facts=build_facts(pkg)
    assert min(row['turn'] for row in facts)==start+1
    assert max(row['turn'] for row in facts)==total
    last=pkg.validate_resume_checkpoint(requests-1)
    assert last['checkpoint_id']==f'cp{requests-1:06d}'
    assert last['budget_used']['model_requests']==total-1


def test_budget_offset_needs_no_lineage(tmp_path):
    pkg=segment(tmp_path/'episode',parent_cp=None)
    assert pkg.start_turn==21 and pkg.turns==30


@pytest.mark.parametrize('value',[None,True,-1,1.5,'21'])
def test_declared_malformed_initial_budget_fails_closed(tmp_path,value):
    pkg=segment(tmp_path/'episode')
    path=pkg.root/'checkpoints/cp000000.json'
    cp=read_json(path);cp['budget_used']['model_requests']=value;write_json(path,cp)
    with pytest.raises(IntegrityError,match='invalid cumulative'):
        L1Package(pkg.root)


def test_offline_missing_budget_retains_lineage_and_restore_fallback(tmp_path):
    pkg=segment(tmp_path/'episode',include_budget=False,parent_cp='cp000013')
    assert pkg.start_turn==13 and pkg.turns==22
    manifest=pkg.manifest;manifest['lineage']=None;write_json(pkg.root/'manifest.json',manifest)
    # The fixture's restored event provides the legacy offset when no cp0 budget exists.
    events=pkg.jsonl('trajectory.jsonl')
    from l2_sandbox.io import write_jsonl
    events.insert(0,{'kind':'episode.restored','data':{'parent_checkpoint_id':'cp000005'}})
    write_jsonl(pkg.root/'trajectory.jsonl',events)
    assert L1Package(pkg.root).start_turn==5


def test_nested_branch_validation_uses_parent_budget_but_local_lookup(tmp_path):
    parent=segment(tmp_path/'parent',start=7,requests=14,parent_cp='cp000007')
    child=segment(tmp_path/'child',start=20,requests=2,parent_cp='cp000013')
    cp=parent.validate_resume_checkpoint(13)
    manifest=child.manifest
    manifest['lineage']={'parent_episode_id':parent.episode_id,'parent_checkpoint_id':cp['checkpoint_id'],
        'parent_index_sha256':parent.index_sha256,'parent_checkpoint_sha256':cp['sha256']}
    write_json(child.root/'manifest.json',manifest)
    child=L1Package(child.root)
    child.validate_branch(parent,13)
    assert child.start_turn==20 and child.turns==22
    with pytest.raises(IntegrityError,match='absent'):
        parent.validate_resume_checkpoint(20)
    manifest['task']['task_id']='officeval_004';write_json(child.root/'manifest.json',manifest)
    with pytest.raises(IntegrityError,match='checkpoint/task'):
        L1Package(child.root).validate_branch(parent,13)


def test_ordinary_offline_branch_unchanged(tmp_path):
    parent=L1Package(make_l1_episode(tmp_path/'parent',num_turns=4))
    child=L1Package(make_l1_episode(tmp_path/'child',num_turns=4,resume_from_k=2,parent_pkg=parent))
    child.validate_branch(parent,2)
    assert child.start_turn==2 and child.turns==4


@pytest.fixture
def fake_evaluator(monkeypatch):
    calls=[]
    class Evaluator:
        def __init__(self,config):
            pass
        def evaluate(self,spec,submission,output):
            calls.append(sorted(path.name for path in submission.iterdir()))
            return {'projection':{'completion_rate':.5},'official_raw':{
                'status':'ok','error':None,'dim1_pass':True,'total_score':5,'max_score':10}}
    monkeypatch.setattr(grades,'DockerEvaluator',Evaluator)
    return calls


def test_grade_boundaries_preserve_real_ids_and_terminal_marker(tmp_path,fake_evaluator):
    pkg=segment(tmp_path/'episode')
    rows=grade_episode_snapshots(pkg,verify_terminal_submission=False)
    cps=[row for row in rows if row['kind']=='checkpoint']
    assert rows[0]['k']==30 and len(cps)==10
    assert [row['k'] for row in cps]==list(range(21,31))
    assert [row['checkpoint_id'] for row in cps[:-1]]==[f'cp{k:06d}' for k in range(9)]
    assert cps[-2]['k']==29 and cps[-2]['checkpoint_id']=='cp000008'
    assert cps[-1]['k']==30 and cps[-1]['checkpoint_id'] is None
    assert cps[-1]['synthetic_terminal_checkpoint'] is True
    assert all(row['synthetic_terminal_checkpoint'] is False for row in cps[:-1])
    assert len(fake_evaluator)==10 and all(row['score']==.5 for row in cps)


def test_actual_terminal_checkpoint_is_not_duplicated(tmp_path,fake_evaluator):
    pkg=segment(tmp_path/'episode',start=7,requests=3)
    cp=read_json(pkg.root/'checkpoints/cp000002.json')
    cp.update(checkpoint_id='cp000003',snapshot_id=max(pkg.snapshots))
    cp['budget_used']['model_requests']=10
    write_json(pkg.root/'checkpoints/cp000003.json',cp)
    index=read_json(pkg.root/'checkpoints/index.json')
    index['checkpoints'].append({'checkpoint_id':'cp000003','path':'checkpoints/cp000003.json'})
    write_json(pkg.root/'checkpoints/index.json',index)
    rows=grade_episode_snapshots(L1Package(pkg.root),verify_terminal_submission=False)
    cps=[row for row in rows if row['kind']=='checkpoint']
    assert [row['k'] for row in cps]==[7,8,9,10]
    assert cps[-1]['checkpoint_id']=='cp000003' and not cps[-1]['synthetic_terminal_checkpoint']
    assert len(fake_evaluator)==4


def test_offline_grade_missing_budgets_adds_start_to_local_ordinal(tmp_path,fake_evaluator):
    pkg=segment(tmp_path/'episode',requests=3,parent_cp='cp000005',include_budget=False)
    rows=grade_episode_snapshots(pkg,verify_terminal_submission=False)
    assert [row['k'] for row in rows if row['kind']=='checkpoint']==[5,6,7,8]
    assert len(fake_evaluator)==4


@pytest.mark.parametrize('value',[21,20,40,None,True])
def test_invalid_checkpoint_boundaries_rejected_before_evaluation(tmp_path,fake_evaluator,value):
    pkg=segment(tmp_path/'episode')
    path=pkg.root/'checkpoints/cp000001.json'
    cp=read_json(path);cp['budget_used']['model_requests']=value;write_json(path,cp)
    with pytest.raises(L2Error,match='cumulative request'):
        grade_episode_snapshots(L1Package(pkg.root),verify_terminal_submission=False)
    assert fake_evaluator==[]
