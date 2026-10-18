"""Fresh-pool runner contracts with fixture ports; never invokes live services."""
import copy
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/run_prospective_dev45.py'
spec = importlib.util.spec_from_file_location('prospective_dev45_test_driver', SCRIPT)
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


@pytest.fixture
def env(tmp_path, monkeypatch):
    config = d.resolve(d.CONFIG)
    config_path = tmp_path / 'config.json'
    d.write_json(config_path, config)
    monkeypatch.setattr(d, 'L1', tmp_path / 'L1')
    monkeypatch.setattr(d, 'L2', tmp_path / 'L2')
    source = tmp_path / 'driver.py'
    source.write_text('fixture source')
    monkeypatch.setattr(d, 'source_files', lambda: {'driver.py':d.file_digest(source)})
    configs = d.task_configs(config_path)
    mini, verification, review = (tmp_path / name for name in ('mini.json', 'verification.json', 'review.json'))
    d.write_json(mini, {'actual_slots': 5, 'status':'review_runtime_variance'})
    d.write_json(verification, {'preparation_links_validated':True, 'g4_verified':False})
    d.write_json(review, {'source':'direct human decision', 'scope':'declared variance, never exact equivalence'})
    frozen = dict(source_files=d.source_files(), task_config_sha256={k:d.digest(v) for k,v in configs.items()},
        actor_image_id='sha256:'+'b'*64, provenance_candidate_verified=True,
        mechanism_mini={'result_sha256':d.file_digest(mini), 'reviewed':True},
        provenance_verification={'path':str(verification),'sha256':d.file_digest(verification)},
        mini_review={'path':str(review),'sha256':d.file_digest(review)},
        phase_authorized=True, d0_frozen=False,
        bound_files={str(p):d.file_digest(p) for p in (mini,verification,review)})
    identity = dict(model={key:config['model'][key] for key in ('reported_revision','weight_digest','tokenizer_digest')},
        vllm=dict(executable='/home/jiasheren2/envs/idea-n1/bin/vllm',version='fixture-0.1',script_sha256='c'*64))
    identity['model'].update(resolved_weight_path='/home/jiasheren2/hf_cache/models--Qwen--Qwen3.6-27B/snapshots/'+config['model']['reported_revision'],
        file_manifest={'path':'/home/jiasheren2/identity/files.json','sha256':'d'*64},digest_algorithm='fixture-only',validated_files=10)
    identity['command_after_timeout']=d.service_command(identity)
    identity_path=tmp_path/'service_identity.json';d.write_json(identity_path,copy.deepcopy(identity))
    identity['identity_receipt']={'path':str(identity_path),'sha256':d.file_digest(identity_path)}
    frozen['bound_files'][str(identity_path)]=d.file_digest(identity_path)
    frozen['expected_service_identity']=identity
    freeze_path = tmp_path / 'freeze.json'
    d.write_json(freeze_path, frozen)
    auth = tmp_path / 'authorization.json'
    d.write_json(auth, {'exact_text':'按顺序启动', 'source':'user message'})
    lease_path = tmp_path / 'lease.json'
    now = d.time.time()
    d.write_json(lease_path, {'started_epoch':now-1, 'hard_stop_epoch':now-1+d.WALL, 'service_stop_epoch':now+20000})
    calls, observers, lock = [], [], threading.Lock()

    class Observer:
        def __init__(self, path, expected, external):
            assert path == freeze_path and expected == d.file_digest(freeze_path)
            self.external = external
            with lock:
                observers.append(self)

    module = SimpleNamespace(validate_freeze=lambda path,expected:d.read_json(path), ProspectiveExecutionObserver=Observer)
    monkeypatch.setattr(d, 'observer_module', lambda:module)
    run_root = d.L1 / 'runs/v3_qwen36_pi_prospective_dev45_fixture'
    output = d.L2 / 'validation/a_plus_i_execute_20261018/fresh'

    def build():
        return d.build_plan(prospective_freeze=freeze_path, authorization=auth, service_lease=lease_path,
                            run_root=run_root, config_path=config_path)

    def fake(config, path, **kwargs):
        assert kwargs['parent'] is None and 'checkpoint_id' not in kwargs
        assert isinstance(kwargs['execution_observer'], Observer)
        limiter = kwargs['request_limiter']
        assert limiter.acquire(1, timeout=600)
        limiter.reconcile(1, None)
        with lock:
            calls.append((copy.deepcopy(config), path, kwargs['execution_observer']))
        path.mkdir(parents=True)

    monkeypatch.setattr(d, 'l1_run', fake)
    monkeypatch.setattr(d, 'slot_record', lambda path,task,cfg:dict(study_valid=True, config_sha256=d.digest(cfg),
        path=str(path), usage={'model_requests_including_parent':1}, e1_status='NOT_RUN'))
    plan = build()
    service = tmp_path / 'service.json'
    d.write_json(service, {**plan['service_lease_origin'], 'owner_tag':plan['plan_sha256'],
        'server_plan_hash':plan['plan_sha256'], 'leader_start_ticks':123, 'remote_host':'aml-b',
        'gpu_uuid':'GPU-6d3f754b-b139-9354-0cbc-42bf22208180',
        'identity_receipt_sha256':identity['identity_receipt']['sha256'],
        'command':['timeout','--signal=TERM','--kill-after=30s','20000s',*identity['command_after_timeout']]})
    return SimpleNamespace(plan=plan, build=build, output=output, service=service, frozen=frozen,
        freeze_path=freeze_path, config_path=config_path, source=source, run_root=run_root,
        module=module, calls=calls, observers=observers, lease_path=lease_path, tmp=tmp_path)


def test_plan_is_inert_and_binds_all_45_complete_configs(env):
    assert len(env.plan['slots']) == 45
    assert env.plan['http_cap'] == 6300 and env.plan['logical_cap'] == 1575
    assert not env.output.exists() and not env.run_root.exists()
    assert env.calls == [] and env.observers == []
    assert env.plan['g2_actor_requests'] == 0 and env.plan['d1_execution'] is False
    configs = env.plan['configs']
    normal = []
    for task, cfg in configs.items():
        assert cfg['benchmark']['task_id'] == task and cfg['budget']['max_tool_calls'] == 140
        item = copy.deepcopy(cfg); item['benchmark'].pop('task_id'); normal.append(item)
    assert all(item == normal[0] for item in normal)


def test_fresh_45_one_attempt_observer_per_slot_and_unknown_usage(env):
    result = d.execute(env.plan, env.output, env.service)
    assert len(env.calls) == len(env.observers) == 45
    assert len({id(x) for x in env.observers}) == 45
    assert len({x.external for x in env.observers}) == 45
    assert result['all_45_study_valid'] and result['slots_terminal'] == 45
    assert result['admissions']['requests'] == 45
    assert all(row['status'] == 'unknown_usage' for row in result['admissions']['sends'])
    assert not result['d1_execution'] and result['g2_actor_requests'] == 0 and not result['d0_frozen']
    assert d.read_json(env.output / 'score_inputs.json')['e1_status'] == 'NOT_RUN'
    with pytest.raises(ValueError, match='restart/overwrite'):
        d.execute(env.plan, env.output, env.service)
    assert len(env.calls) == 45


def test_invalid_slot_preserved_without_resampling(env, monkeypatch):
    def invalid(config, path, **kwargs):
        env.calls.append(path)
        raise RuntimeError('fixture invalid attempt')
    monkeypatch.setattr(d, 'l1_run', invalid)
    result = d.execute(env.plan, env.output, env.service)
    assert len(env.calls) == len(set(env.calls)) == 45
    assert result['status'] == 'incomplete_or_invalid'
    rows = d.read_json(env.output / 'cohort.json')['slots']
    assert all(row['status'] == 'execution_invalid' and row['error']['traceback'] for row in rows.values())


@pytest.mark.parametrize('key,value', [('http_cap',6301),('logical_cap',1576),('concurrency',4),
    ('attempt_limit_per_slot',2),('d1_execution',True),('g2_actor_requests',20),('parent','old')])
def test_expanded_or_legacy_execution_contract_rejected(env,key,value):
    plan = d.seal({**env.plan,key:value},'plan_sha256')
    with pytest.raises(ValueError,match='hard limits'):
        d.execute(plan,env.output,env.service)
    assert env.calls == [] and not env.output.exists()


def test_old_pool_and_old_phase_rejected_before_writes(env):
    with pytest.raises(ValueError,match='old pools'):
        d.fresh_paths(d.L1/'runs/v3_qwen36_pi_dev45_20261017')
    with pytest.raises(ValueError,match='new a_plus_i phase'):
        d.execute(env.plan,d.L2/'validation/d0_20261018/old',env.service)
    assert env.calls == []


def test_different_full_pi_is_rejected_even_with_resealed_plan(env):
    plan=copy.deepcopy(env.plan)
    plan['configs']['officeval_003']['recording']['checkpoints']=False
    plan=d.seal(plan,'plan_sha256')
    with pytest.raises(ValueError,match='full task configurations'):
        d.execute(plan,env.output,env.service)
    assert env.calls == []


def test_freeze_flags_cannot_replace_actual_verifier(env):
    env.module.validate_freeze=lambda *args: {'provenance_candidate_verified':True}
    with pytest.raises(ValueError,match='independent prospective'):
        env.build()
    assert env.calls == []


def test_mechanism_variance_requires_bound_review_file(env):
    value=d.read_json(env.freeze_path); value.pop('mini_review')
    d.write_json(env.freeze_path,value)
    with pytest.raises(ValueError,match='explicit mini policy review'):
        env.build()


@pytest.mark.parametrize('which', ['source','freeze','service','lease'])
def test_changes_prevent_all_new_sends(env,which):
    path={'source':env.source,'freeze':env.freeze_path,'service':env.service,'lease':env.lease_path}[which]
    if which=='service':
        value=d.read_json(path); value['owner_tag']='wrong'; d.write_json(path,value)
    else:
        path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError):
        d.execute(env.plan,env.output,env.service)
    assert env.calls == [] and not env.output.exists()


def guarded(env,tmp_path,cap=1):
    admission=d.Admission(tmp_path/'admission.json',env.plan['plan_sha256'],dict(max_requests=cap,
        max_tokens=cap*100,max_cost_usd=0,max_wall_seconds=1000,max_invalid_attempts=46,
        max_tokens_per_request=100,usd_per_token_upper_bound=0))
    return d.GuardedAdmission(admission,env.plan,{},threading.Event())


def test_http_cap_applies_to_retries_and_unknown_is_not_refunded(env,tmp_path):
    limiter=guarded(env,tmp_path)
    assert limiter.acquire(1,600)
    limiter.reconcile(1,None)
    assert not limiter.acquire(1,600)
    assert limiter.admission.snapshot()['requests']==1
    assert limiter.admission.snapshot()['sends'][0]['status']=='unknown_usage'


def test_each_admission_checks_source_and_absolute_window(env,tmp_path):
    limiter=guarded(env,tmp_path,2)
    env.source.write_text('changed')
    with pytest.raises(ValueError,match='source changed'):
        limiter.acquire(1,600)
    assert limiter.admission.snapshot()['requests']==0


def test_no_send_with_less_than_645_seconds_outer_lease(env,tmp_path):
    limiter=guarded(env,tmp_path)
    limiter.plan={**limiter.plan,'service_stop_epoch':d.time.time()+644}
    # Re-anchor only the synthetic plan to isolate the send-window check.
    limiter.plan=d.seal(limiter.plan,'plan_sha256')
    with pytest.raises(RuntimeError,match='safety window'):
        limiter.acquire(1,600)
    assert limiter.admission.snapshot()['requests']==0


def test_changed_service_receipt_blocks_request_and_sets_stop(env,tmp_path):
    limiter=guarded(env,tmp_path)
    limiter.metadata={env.service:d.file_digest(env.service)}
    env.service.write_text(env.service.read_text()+' ')
    with pytest.raises(ValueError,match='metadata changed'):
        limiter.acquire(1,600)
    assert limiter.stop.is_set() and limiter.admission.snapshot()['requests']==0


def test_mid_batch_source_change_retains_all_terminal_error_receipts(env,monkeypatch):
    def source_changed(config,path,**kwargs):
        limiter=kwargs['request_limiter']
        if limiter.acquire(1,600):
            limiter.reconcile(1,None)
            env.source.write_text('changed during fixture run')
            path.mkdir(parents=True)
    monkeypatch.setattr(d,'l1_run',source_changed)
    result=d.execute(env.plan,env.output,env.service)
    assert result['integrity_error'] and not result['all_45_study_valid']
    assert result['slots_terminal']==45
    assert len(list(env.output.glob('receipt_*.json')))==45
    assert (env.output/'cohort.json').exists() and (env.output/'score_inputs.json').exists()


@pytest.mark.parametrize('change',['weightpath','toolparser','reasonparser','maxlen','maxseq','gpumem','dtype','vllmbinary','identitysha','alias'])
def test_actual_service_identity_and_launch_arguments_are_exact(env,change):
    value=d.read_json(env.service)
    argv=value['command']
    if change=='identitysha':value['identity_receipt_sha256']='wrong'
    elif change=='weightpath':argv[9]='/other/weights'
    elif change=='vllmbinary':argv[7]='/other/vllm'
    elif change=='alias':argv[argv.index('--served-model-name')+1]='foreign-model'
    else:
        flag={'toolparser':'--tool-call-parser','reasonparser':'--reasoning-parser','maxlen':'--max-model-len',
            'maxseq':'--max-num-seqs','gpumem':'--gpu-memory-utilization','dtype':'--dtype'}[change]
        argv[argv.index(flag)+1]='changed'
    d.write_json(env.service,value)
    with pytest.raises(ValueError,match='actual service'):
        d.execute(env.plan,env.output,env.service)
    assert env.calls==[] and not env.output.exists()


def test_frozen_vllm_script_hash_cannot_differ_from_actual_identity_artifact(env):
    identity=copy.deepcopy(env.plan['expected_service_identity'])
    identity['vllm']['script_sha256']='e'*64
    with pytest.raises(ValueError,match='independent server'):
        d.validate_service_identity(identity,env.plan['configs'],env.plan['bound_files'])


def test_large_bound_proof_streamed_once_and_changed_metadata_rejected(env,tmp_path,monkeypatch):
    proof=tmp_path/'large_proof.tar';proof.write_bytes(b'a'*524288)
    expected=d.file_digest(proof)
    original=d.hashlib.file_digest
    calls=[]
    def hashed(stream,algorithm):
        calls.append(stream.name)
        return original(stream,algorithm)
    monkeypatch.setattr(d.hashlib,'file_digest',hashed)
    plan=copy.deepcopy(env.plan);plan['bound_files'][str(proof)]=expected
    plan=d.seal(plan,'plan_sha256')
    d.assert_sources(plan);d.assert_sources(plan)
    assert calls.count(str(proof))==1
    old=proof.stat()
    proof.write_bytes(b'b'*524288)
    os.utime(proof,ns=(old.st_atime_ns,old.st_mtime_ns))
    with pytest.raises(ValueError,match='bound configuration'):
        d.assert_sources(plan)
    assert calls.count(str(proof))==2


def test_unfrozen_installed_plugins_cannot_be_enabled(env):
    config=d.read_json(env.config_path)
    config['plugins']['load_installed']=True
    d.write_json(env.config_path,config)
    with pytest.raises(ValueError,match='full Actor configuration'):
        d.task_configs(env.config_path)


def test_actual_raw_sha256_identity_matches_prefixed_config_without_rewriting_receipt(env):
    freeze=d.read_json(env.freeze_path)
    identity=freeze['expected_service_identity']
    for key in ('weight_digest','tokenizer_digest'):
        identity['model'][key]=identity['model'][key].removeprefix('sha256:')
    path=Path(identity['identity_receipt']['path'])
    d.write_json(path,{key:identity[key] for key in ('model','vllm','command_after_timeout')})
    expected=path.read_bytes()
    identity['identity_receipt']['sha256']=d.file_digest(path)
    freeze['bound_files'][str(path)]=d.file_digest(path)
    d.write_json(env.freeze_path,freeze)
    plan=env.build()
    assert plan['expected_service_identity']['model']==identity['model']
    assert len(identity['model']['weight_digest'])==64 and len(identity['model']['tokenizer_digest'])==64
    assert path.read_bytes()==expected


def test_different_raw_digest_is_rejected_against_prefixed_config(env):
    identity=copy.deepcopy(env.plan['expected_service_identity'])
    identity['model']['weight_digest']='0'*64
    with pytest.raises(ValueError,match='weight/tokenizer identity differs'):
        d.validate_service_identity(identity,env.plan['configs'],env.plan['bound_files'])
