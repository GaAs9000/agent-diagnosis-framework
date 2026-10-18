"""CPU-only recovery contracts; fake L1 runner never contacts an Actor."""
import copy
import importlib.util
from pathlib import Path
import threading
import pytest

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/run_context004_recovery.py'
spec = importlib.util.spec_from_file_location('context004_test_driver', SCRIPT)
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


@pytest.fixture
def env(tmp_path, monkeypatch):
    parent = tmp_path/'old_run/segments'/d.LABEL
    monkeypatch.setattr(d, 'PARENT', parent)
    cfg = d.c.resolve_config(d.c.CONFIG_PATH)
    cfg['benchmark']['task_id'] = 'officeval_004'
    cfg = d.r.native_config(cfg)
    cfg['agent']['harness_revision'] = 'l1-pi-native-errors-v1'
    d.write_json(parent/'raw/resolved_config.json', cfg)
    d.write_json(parent/'checkpoints/cp000013.json', {'budget_used':{'model_requests':20}})
    d.write_json(parent/'manifest.json', {'episode_id':'ep_parent'})
    d.write_json(parent/'evidence_index.json', {'old':'immutable'})
    proof = dict(config=cfg, binding={'source_request_id':'ep_parent:r000020'}, manifest={'episode_id':'ep_parent'})
    monkeypatch.setattr(d, 'context_source', lambda _:copy.deepcopy(proof))
    monkeypatch.setattr(d, 'assert_sources', lambda _:None)
    cohort = {label:dict(status='formal', terminal_endpoint=str(parent), endpoint_index_sha256='old', attempt_id=label,
        origin_prefix='original', continuation_segment='prior-segment') for label in d.r.SLOTS}
    cohort[d.LABEL]['status'] = 'invalid_continuation'
    cohort[d.LABEL]['endpoint_index_sha256'] = d.PARENT_INDEX
    frozen = d.seal(dict(plan_sha256='old-plan', slots=cohort, demoted_controls={}))
    prior = tmp_path/'prior'
    g2_phase = tmp_path/'g2'
    frozen_g2 = [dict(label=x) for x in d.r.G2_SLOTS]
    d.write_json(prior/'cohort.json', frozen)
    d.write_json(g2_phase/'result.json', {'g2':frozen_g2})
    now=d.time.time()
    lease_path=tmp_path/'lease.json'
    lease=dict(started_epoch=now-10,hard_stop_epoch=now-10+21600,service_stop_epoch=now+20000)
    d.write_json(lease_path,lease)
    plan = d.seal(dict(schema_version=d.SCHEMA, selected=[d.LABEL], g2_slots=[], old_http_attempts=578,
        http_cap=56, logical_cap=14, aggregate_http_cap=3360, aggregate_authorized_max=634,
        wall_seconds=21600, request_timeout_seconds=600, send_window_seconds=645, concurrency=1,
        attempt_limit_per_slot=1, service_renewal_authorized=False, d0_frozen=False,
        service_stop_epoch=lease['service_stop_epoch'], service_lease_origin=lease, service_lease_path=str(lease_path),
        bound_files={str(lease_path):d.file_digest(lease_path)}, parent=str(parent), parent_index_sha256=d.PARENT_INDEX,
        checkpoint_id=d.CHECKPOINT, checkpoint_sha256=d.file_digest(parent/'checkpoints/cp000013.json'),
        resume_context_overflow_id=d.FAILED, consumed_logical=21, first_new_request_id='r000021',
        parent_episode_id='ep_parent', context_binding=proof['binding'], native_config=d.r.native_config(cfg),
        frozen_cohort=frozen, frozen_g2=frozen_g2, demoted={},
        predecessor_phase=str(prior), g2_phase=str(g2_phase),
        run_root=str(tmp_path/'new_run'), immutable_trees={str(tmp_path/'old_run'):d.s.tree_digest(tmp_path/'old_run')}), 'plan_sha256')
    freeze = dict(approved_plan_sha256=plan['plan_sha256'], phase_authorized=True, phase_frozen=True,
        internal_plan_reviewed=True, d0_frozen=False, user_authorization_record=dict(exact_text='补004，14 requests', source='fixture'))
    calls=[]
    def package(path):
        return dict(path=str(path), valid=path.exists(), index_file_sha256='child' if path != parent else 'old')
    monkeypatch.setattr(d.p, 'package_record', package)
    monkeypatch.setattr(d.r, 'score_inputs', lambda plan, slots, g2:dict(primary=[], sensitivity_a=[], sensitivity_b=[], g2=[], preserved_g2=copy.deepcopy(g2)))
    def fake(config, path, **kwargs):
        calls.append((config,path,kwargs))
        assert d.read_json(tmp_path/'output/attempt_started.json')['first_new_request_id']=='r000021'
        assert kwargs['resume_context_overflow_id']=='r000020'
        assert kwargs['checkpoint_id']=='cp000013'
        assert kwargs['parent']==parent and kwargs['allow_pi_native_upgrade'] is True
        assert 'resume_response_id' not in kwargs
        assert kwargs['request_limiter'].acquire(1, timeout=700)
        kwargs['request_limiter'].reconcile(1,None)
        d.write_json(path/'manifest.json', {'usage':{'model_requests_including_parent':22}})
    monkeypatch.setattr(d.c, 'l1_run', fake)
    monkeypatch.setattr(d, 'validate_child', lambda *_:dict(new_logical_requests=1,cumulative_logical_requests=22))
    return plan,freeze,calls,tmp_path/'output'


def test_one_recovery_preserves_old_chain_and_g2(env):
    plan,freeze,calls,output=env
    before=copy.deepcopy(plan['frozen_cohort'])
    result=d.execute(plan,freeze,output)
    assert len(calls)==1 and result['g2']==[] and result['preserved_g2_pairs']==20
    assert result['aggregate_http_attempts']==579 and result['all_45_study_valid']
    cohort=d.read_json(output/'cohort.json')
    assert cohort['slots'][d.LABEL]['previous_terminal']==before['slots'][d.LABEL]
    assert all(row==before['slots'][label] for label,row in cohort['slots'].items() if label!=d.LABEL)
    assert plan['frozen_cohort']==before
    assert d.read_json(output/'score_inputs.json')['preserved_g2']==plan['frozen_g2']
    d.assert_originals(plan)
    with pytest.raises(ValueError,match='restart/overwrite'):
        d.execute(plan,freeze,output)
    assert len(calls)==1


@pytest.mark.parametrize('key,value', [('logical_cap',15),('http_cap',57),('consumed_logical',20),('g2_slots',['officeval_004.r02']),('concurrency',2)])
def test_limit_expansion_rejected(env,key,value):
    plan,freeze,calls,output=env
    plan=d.seal({**plan,key:value},'plan_sha256')
    with pytest.raises(ValueError,match='hard limits'):
        d.execute(plan,freeze,output)
    assert calls==[]


def test_predecessor_tamper_and_old_output_block_before_marker(env):
    plan,freeze,calls,output=env
    parent=Path(plan['parent'])
    with pytest.raises(ValueError,match='outside immutable'):
        d.execute(plan,freeze,parent/'new-output')
    assert not (parent/'new-output').exists()
    d.write_json(parent/'manifest.json',{'tampered':True})
    with pytest.raises(ValueError,match='immutable original'):
        d.execute(plan,freeze,output)
    assert not calls and not (output/'attempt_started.json').exists()


def test_child_failure_retains_receipt_and_never_promotes(env,monkeypatch):
    plan,freeze,calls,output=env
    def fail(*_):
        raise ValueError('bad source proof')
    monkeypatch.setattr(d,'validate_child',fail)
    result=d.execute(plan,freeze,output)
    assert not result['all_45_study_valid'] and result['stop_reason']
    assert d.read_json(output/'cohort.json')['slots'][d.LABEL]['status']=='invalid_continuation'
    assert (output/('receipt_context_'+d.LABEL+'.json')).exists()
    assert len(calls)==1 and not (output/'score_inputs.json').exists()


def test_http_cap_and_source_guard(env,monkeypatch,tmp_path):
    plan,*_=env
    admission=d.p.Admission(tmp_path/'limit.json',plan['plan_sha256'],{**d.p.LIMITS,
        'max_requests':56,'max_tokens':56*d.p.GIB,'max_wall_seconds':21600})
    limiter=d.GuardedAdmission(admission,plan,{},d.time.time(),threading.Event())
    for _ in range(56):
        assert limiter.acquire(1,timeout=700)
        limiter.reconcile(1,None)
    assert limiter.acquire(1,timeout=700) is False
    assert admission.snapshot()['requests']==56
    def drift(_):
        raise ValueError('source drift')
    monkeypatch.setattr(d,'assert_sources',drift)
    with pytest.raises(ValueError,match='source drift'):
        limiter.acquire(1,timeout=700)


def test_supervisor_wrapper_rebinds_parent_and_child():
    path=SCRIPT.with_name('supervise_context004_recovery.py')
    spec=importlib.util.spec_from_file_location('context_supervisor_test',path)
    wrapper=importlib.util.module_from_spec(spec); spec.loader.exec_module(wrapper)
    assert wrapper.module.SCRIPT==path
    assert wrapper.module.DRIVER==SCRIPT


def test_bound_predecessor_cohort_tamper_rejected(env):
    plan,freeze,calls,output=env
    path=Path(plan['predecessor_phase'])/'cohort.json'
    value=d.read_json(path); value['slots'][d.LABEL]['status']='formal'
    d.write_json(path,d.seal(value))
    with pytest.raises(ValueError,match='predecessor cohort/G2'):
        d.execute(plan,freeze,output)
    assert not calls and not (output/'attempt_started.json').exists()


def test_changed_sampling_config_rejected(env):
    plan,freeze,calls,output=env
    changed=copy.deepcopy(plan)
    changed['native_config']['model']['parameters']['temperature']=0
    changed=d.seal(changed,'plan_sha256')
    with pytest.raises(ValueError,match='source/config/checkpoint'):
        d.execute(changed,freeze,output)
    assert calls==[]


@pytest.mark.parametrize('used,count', [(36,15),(20,0),(22,2)])
def test_child_cumulative_logical_accounting(env,monkeypatch,tmp_path,used,count):
    import c3_sandbox.l1.evidence.live as live
    plan,*_=env
    monkeypatch.setattr(live,'verify',lambda _:dict(verified=True))
    child=tmp_path/'child'
    lineage=dict(parent_episode_id=plan['parent_episode_id'],parent_index_sha256=plan['parent_index_sha256'],
        parent_checkpoint_id=d.CHECKPOINT,parent_checkpoint_sha256=plan['checkpoint_sha256'],context_overflow=plan['context_binding'])
    d.write_json(child/'manifest.json',dict(lineage=lineage,usage={'model_requests_including_parent':used},counts={'model_requests':count}))
    d.write_json(child/'raw/resolved_config.json',plan['native_config'])
    for path in (Path(plan['parent']),child):
        d.write_json(path/'benchmark/public_task.json',{'task_id':'officeval_004'})
    with pytest.raises(ValueError,match='logical cap'):
        # Restore the actual implementation hidden by env's fake verifier.
        spec=importlib.util.spec_from_file_location('context004_real_child_check',SCRIPT)
        real=importlib.util.module_from_spec(spec);spec.loader.exec_module(real)
        real.validate_child(plan,child)


def load_wrapper():
    path=SCRIPT.with_name('supervise_context004_recovery.py')
    spec=importlib.util.spec_from_file_location('context004_test_supervisor_preflight',path)
    wrapper=importlib.util.module_from_spec(spec);spec.loader.exec_module(wrapper)
    return wrapper


def test_generated_plan_has_bound_lease_and_passes_supervisor(env,monkeypatch,tmp_path):
    previous,_,_,_=env
    wrapper=load_wrapper()
    prior=Path(previous['predecessor_phase']); g2=Path(previous['g2_phase'])
    old=dict(immutable_trees=previous['immutable_trees'],run_root=str(tmp_path/'old_run'),demoted={})
    gp=dict(run_root=str(tmp_path/'old_g2_run'))
    monkeypatch.setattr(d,'predecessor',lambda *_:(old,previous['frozen_cohort'],gp,{'g2':previous['frozen_g2']},{}))
    monkeypatch.setattr(d.r,'BASE',tmp_path/'base'); d.r.BASE.mkdir()
    monkeypatch.setattr(d.r,'dependency_identity',lambda :{'fixture':'native'})
    monkeypatch.setattr(d,'source_files',lambda _:{str(path.relative_to(d.ROOT)):d.file_digest(path)
        for path in (wrapper.module.SCRIPT,wrapper.module.DRIVER,wrapper.module.HELPER)})
    auth=tmp_path/'authorization.json';d.write_json(auth,{'exact_text':'补004'})
    plan=d.build_plan(service_stop_epoch=previous['service_stop_epoch'],authorization_paths=[auth],
        service_lease_path=Path(previous['service_lease_path']),run_root=tmp_path/'new_root',
        cohort_path=prior/'cohort.json',predecessor_phase=prior,g2_phase=g2)
    freeze=dict(approved_plan_sha256=plan['plan_sha256'],phase_authorized=True,phase_frozen=True,internal_plan_reviewed=True,
        d0_frozen=False,user_authorization_record={'exact_text':'补004','source':'fixture'})
    now=d.time.time();lease=plan['service_lease_origin']
    service=dict(**lease,remote_host='aml-b',device=4,owner_tag=plan['plan_sha256'],server_plan_hash=plan['plan_sha256'],
        leader_pid=1234,pgid=1234,clock_calibration=dict(local_sent_epoch=now-2,local_received_epoch=now-1,remote_epoch=now-101),
        remote_launch_epoch=now-100,remote_hard_stop_epoch=lease['hard_stop_epoch']-160,kill_grace_seconds=30)
    service['timeout_seconds']=int(service['remote_hard_stop_epoch']-service['remote_launch_epoch']-35)
    service['command']=['timeout','--signal=TERM','--kill-after=30s',str(service['timeout_seconds'])+'s','qwen-server']
    wrapper.module.validate(plan,freeze,service,now)
    assert plan['service_lease_origin']==d.read_json(Path(previous['service_lease_path']))
    assert plan['bound_files'][previous['service_lease_path']]==d.file_digest(Path(previous['service_lease_path']))


@pytest.mark.parametrize('child',[False,True])
def test_supervisor_rejects_old_phase_before_any_write(env,monkeypatch,tmp_path,child):
    plan,freeze,calls,_=env
    wrapper=load_wrapper();monkeypatch.setattr(wrapper,'driver',d)
    monkeypatch.setattr(wrapper.module,'main',lambda *_:pytest.fail('generic writes reached'))
    phase=Path(plan['parent'])/'forbidden'
    plan_path=tmp_path/'reviewed_plan.json';freeze_path=tmp_path/'freeze.json';service=tmp_path/'service.json'
    d.write_json(plan_path,plan);d.write_json(freeze_path,freeze);d.write_json(service,{})
    args=['--execute','--phase',str(phase),'--plan',str(plan_path),'--freeze',str(freeze_path),'--service',str(service)]
    if child:
        phase.mkdir()
        marker=phase/'supervisor_started.json'
        d.write_json(marker,dict(phase=str(phase),plan=str(plan_path),freeze=str(freeze_path),files={}))
        args=['--execute','--child',str(marker)]
    before={str(path):d.file_digest(path) for path in Path(plan['parent']).rglob('*') if path.is_file()}
    with pytest.raises(ValueError,match='outside immutable'):
        wrapper.main(args)
    after={str(path):d.file_digest(path) for path in Path(plan['parent']).rglob('*') if path.is_file()}
    assert before==after and calls==[]
    if not child:
        assert not phase.exists()


@pytest.mark.parametrize('delta',[-1,1])
def test_service_lease_cannot_be_renewed(env,delta):
    plan,*_=env
    path=Path(plan['service_lease_path']);lease=d.read_json(path)
    lease['hard_stop_epoch']+=delta;d.write_json(path,lease)
    with pytest.raises(ValueError,match='six-hour'):
        d.validate_contract(plan)
