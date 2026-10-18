"""Prospective lifecycle wiring with fake processes only, no Docker or GPU."""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/supervise_prospective_dev45.py'
spec = importlib.util.spec_from_file_location('prospective_supervisor_test', SCRIPT)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


@pytest.fixture
def env(tmp_path, monkeypatch):
    d, m = s.driver, s.module
    monkeypatch.setattr(d, 'L1', tmp_path / 'L1')
    monkeypatch.setattr(d, 'L2', tmp_path / 'L2')
    phase = d.L2 / 'validation/a_plus_i_execute_20261018/newphase'
    phase.mkdir(parents=True)
    now = m.time.time()
    origin = dict(started_epoch=now-10, hard_stop_epoch=now-10+21600, service_stop_epoch=now+20000)
    plan = dict(plan_sha256='a'*64, service_stop_epoch=origin['service_stop_epoch'], service_lease_origin=origin,
        run_root=str(d.L1 / 'runs/v3_qwen36_pi_prospective_dev45_fixture'), source_fingerprints={},
        bound_files={str(p):m.sha(p) for p in (s.SCRIPT,m.DRIVER,m.HELPER)})
    cfg=d.resolve(d.CONFIG)
    identity=dict(model={key:cfg['model'][key] for key in ('reported_revision','weight_digest','tokenizer_digest')},
        vllm=dict(executable='/home/jiasheren2/envs/idea-n1/bin/vllm',version='fixture-0.1',script_sha256='c'*64))
    identity['model'].update(resolved_weight_path='/home/jiasheren2/hf_cache/models--Qwen--Qwen3.6-27B/snapshots/'+cfg['model']['reported_revision'],
        file_manifest={'path':'/home/jiasheren2/identity/files.json','sha256':'d'*64},digest_algorithm='fixture-only',validated_files=10)
    identity['command_after_timeout']=d.service_command(identity)
    identity_path=tmp_path/'service_identity.json';m.save(identity_path,identity)
    identity['identity_receipt']={'path':str(identity_path),'sha256':m.sha(identity_path)}
    plan['expected_service_identity']=identity
    plan['bound_files'][str(identity_path)]=m.sha(identity_path)
    plan['configs']={d.TASKS[0]:cfg}
    freeze = dict(approved_plan_sha256=plan['plan_sha256'], phase_authorized=True,
        phase_frozen=True, internal_plan_reviewed=True, d0_frozen=False,
        user_authorization_record={'exact_text':'start bounded batch', 'source':'direct user'})
    service = dict(**origin, owner_tag=plan['plan_sha256'], server_plan_hash=plan['plan_sha256'],
        remote_host='aml-b', device=4, leader_pid=1234, pgid=1234, leader_start_ticks=123,
        gpu_uuid='GPU-6d3f754b-b139-9354-0cbc-42bf22208180')
    service['clock_calibration'] = dict(local_sent_epoch=now-2,local_received_epoch=now-1,remote_epoch=now-101)
    service['remote_launch_epoch'] = now-100
    service['remote_hard_stop_epoch'] = origin['hard_stop_epoch']-100-60
    service['timeout_seconds'] = int(service['remote_hard_stop_epoch']-service['remote_launch_epoch']-35)
    service['kill_grace_seconds'] = 30
    service['identity_receipt_sha256']=identity['identity_receipt']['sha256']
    service['command'] = ['timeout','--signal=TERM','--kill-after=30s',str(service['timeout_seconds'])+'s',*identity['command_after_timeout']]
    for name,value in (('plan.json',plan),('freeze.json',freeze),('service.json',service)):
        m.save(phase/name,value)
    calls=[]
    monkeypatch.setattr(d,'validate_plan',lambda plan:calls.append('validated_plan'))
    monkeypatch.setattr(d,'assert_sources',lambda plan:calls.append('asserted_sources'))
    return phase,plan,freeze,service,calls


def args(phase):
    return (phase,phase/'plan.json',phase/'freeze.json',phase/'service.json')


def test_default_is_inert(monkeypatch,capsys):
    monkeypatch.setattr(s.module,'supervise',lambda *a:pytest.fail('not authorized'))
    s.main([])
    assert json.loads(capsys.readouterr().out)['status']=='not_executed'
    with pytest.raises(SystemExit):
        s.main(['--child','missing'])


def test_preflight_validates_full_lease_and_does_not_write(env):
    phase,plan,freeze,service,calls=env
    before={p.name:s.module.sha(p) for p in phase.iterdir()}
    s.preflight(*args(phase))
    assert calls==['validated_plan','asserted_sources']
    assert {p.name:s.module.sha(p) for p in phase.iterdir()}==before
    assert not Path(plan['run_root']).exists()


@pytest.mark.parametrize('key,value', [('phase_authorized',False),('phase_frozen',False),
    ('internal_plan_reviewed',False),('d0_frozen',True),('approved_plan_sha256','other')])
def test_unapproved_phase_prevents_all_writes(env,key,value):
    phase,plan,freeze,service,calls=env
    freeze[key]=value;s.module.save(phase/'freeze.json',freeze)
    with pytest.raises(ValueError,match='reviewed phase'):
        s.preflight(*args(phase))
    assert not (phase/'supervisor_started.json').exists()


@pytest.mark.parametrize('kind', ['oldphase','sourcealias','oldrun','existingmarker','foreignservice'])
def test_unsafe_destinations_rejected_before_supervisor_writes(env,kind):
    phase,plan,freeze,service,calls=env
    if kind=='oldphase':
        bad=s.driver.L2/'validation/d0_20261018/old'
        with pytest.raises(ValueError,match='fresh a_plus_i'):
            s.preflight(bad,phase/'plan.json',phase/'freeze.json',phase/'service.json')
    elif kind=='sourcealias':
        link=phase.parent/'aliased';link.symlink_to(phase,target_is_directory=True)
        with pytest.raises(ValueError,match='unaliased'):
            s.preflight(link,link/'plan.json',link/'freeze.json',link/'service.json')
    elif kind=='oldrun':
        plan['run_root']=str(s.driver.L1/'runs/v3_qwen36_pi_dev45_20261017')
        s.module.save(phase/'plan.json',plan)
        with pytest.raises(ValueError,match='old pools'):
            s.preflight(*args(phase))
    elif kind=='existingmarker':
        (phase/'supervisor_started.json').write_text('old marker')
        with pytest.raises(ValueError,match='restart/overwrite'):
            s.preflight(*args(phase))
        assert (phase/'supervisor_started.json').read_text()=='old marker'
    else:
        other=phase.parent/'other_service.json';s.module.save(other,service)
        with pytest.raises(ValueError,match='exact phase paths'):
            s.preflight(phase,phase/'plan.json',phase/'freeze.json',other)
    assert not (phase/'supervisor_child.log').exists()


@pytest.mark.parametrize('field,value',[('kill_grace_seconds',0),('hard_stop_epoch',10**12),
    ('remote_hard_stop_epoch',10**12),('owner_tag','foreign'),('pgid',999),('leader_start_ticks',None)])
def test_bad_actual_service_receipt_stops_before_marker(env,field,value):
    phase,plan,freeze,service,calls=env
    service[field]=value;s.module.save(phase/'service.json',service)
    with pytest.raises(ValueError):
        s.preflight(*args(phase))
    assert not (phase/'supervisor_started.json').exists()


def binding(phase,plan):
    paths=(s.SCRIPT,s.module.DRIVER,s.module.HELPER,phase/'plan.json',phase/'freeze.json',phase/'service.json')
    value=dict(phase=str(phase),plan=str(phase/'plan.json'),freeze=str(phase/'freeze.json'),
        files={str(path.resolve()):s.module.sha(path) for path in paths},
        plan_sha256=plan['plan_sha256'],resource_owner='b'*64)
    s.module.save(phase/'supervisor_started.json',value)
    return value


def test_child_passes_new_signature_and_copies_only_new_terminal_artifacts(env,monkeypatch):
    phase,plan,freeze,service,calls=env
    binding(phase,plan)
    @contextmanager
    def containers(helper,actualphase,owner):
        assert actualphase==phase and owner=='b'*64
        calls.append('entered_owned_scope');yield;calls.append('left_owned_scope')
    def execute(actualplan,output,servicepath):
        assert actualplan==plan and output==phase/'execution' and servicepath==phase/'service.json'
        assert calls[-1]=='entered_owned_scope'
        for name in ('result.json','cohort.json','score_inputs.json'):
            s.module.save(output/name,{'source':name,'g2_actor_requests':0,'d1_execution':False})
    monkeypatch.setattr(s.module,'phase_containers',containers)
    monkeypatch.setattr(s.module,'load',lambda *a:object())
    monkeypatch.setattr(s.driver,'execute',execute)
    s.child(phase/'supervisor_started.json')
    assert calls[-1]=='left_owned_scope'
    for name in ('result.json','cohort.json','score_inputs.json'):
        assert s.module.read(phase/name)==s.module.read(phase/'execution'/name)
    assert s.module.read(phase/'executed_freeze.json')==freeze
    assert s.module.read(phase/'executed_plan.json')==plan


@pytest.mark.parametrize('tamper',['file','markerpath','planhash','owner','missingfile'])
def test_child_checks_exact_startup_binding_before_writes(env,tamper):
    phase,plan,freeze,service,calls=env
    value=binding(phase,plan)
    marker=phase/'supervisor_started.json'
    if tamper=='file':
        (phase/'service.json').write_text((phase/'service.json').read_text()+' ')
    elif tamper=='markerpath':
        marker=phase/'other.json';s.module.save(marker,value)
    else:
        if tamper=='planhash':value['plan_sha256']='wrong'
        if tamper=='owner':value['resource_owner']='foreign'
        if tamper=='missingfile':value['files'].pop(str(phase/'service.json'))
        s.module.save(marker,value)
    with pytest.raises(ValueError):s.child(marker)
    assert not (phase/'executed_plan.json').exists()
    assert not (phase/'execution').exists()


def test_child_will_not_overwrite_preexisting_result_or_alias(env):
    phase,plan,freeze,service,calls=env
    binding(phase,plan)
    (phase/'executed_plan.json').symlink_to(phase/'plan.json')
    before=(phase/'plan.json').read_bytes()
    with pytest.raises(ValueError,match='restart/overwrite'):
        s.child(phase/'supervisor_started.json')
    assert (phase/'plan.json').read_bytes()==before


def test_cpu_summary_explicitly_deferred_and_no_calls(env):
    phase,plan,freeze,service,calls=env
    summary=s.cpu_summary(phase,{},0,None)
    assert summary['status']=='NOT_RUN' and summary['e1_required']
    assert summary['actor_requests']==summary['analyst_requests']==0
    assert summary['statistical_summary']=='NOT_RUN' and not summary['d1_execution']


def test_failed_fake_child_still_runs_exact_owner_cleanup(env,monkeypatch):
    phase,plan,freeze,service,calls=env
    class Helper:
        def cleanup_e1(self,actualphase,owner,deadline):
            assert actualphase==phase and len(owner)==64
            calls.append('containers_released');return {'status':'confirmed_empty','actor_requests':0}
    class Process:
        pid=99
        def __init__(self,command,**kwargs):
            assert command[1]==str(s.SCRIPT) and kwargs['start_new_session']
        def wait(self,timeout):return 7
    monkeypatch.setattr(s.module,'load',lambda *a:Helper())
    monkeypatch.setattr(s.module.subprocess,'Popen',Process)
    monkeypatch.setattr(s.module,'stop_service',lambda actual,deadline:calls.append('service_released') or {'status':'confirmed_absent'})
    s.main(['--execute','--phase',str(phase),'--freeze',str(phase/'freeze.json'),'--service',str(phase/'service.json')])
    result=s.module.read(phase/'supervisor_result.json')
    assert result['exit_code']==7 and result['status']=='terminal_or_partial'
    assert result['cpu_summary']['status']=='NOT_RUN'
    assert calls[-2:]==['containers_released','service_released']
