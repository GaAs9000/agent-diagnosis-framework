"""Synthetic receipt test data only: no Docker/model qualification claims."""
import copy
import json
from dataclasses import replace

import pytest

from c3_sandbox.evidence._storage import digest, encode
from c3_sandbox.experiment_scheduler import BranchSpec, ExperimentScheduler, ProtocolError
from c3_sandbox.replay import ReplayQualification, STATE_CONTRACT_V1
from c3_sandbox.r1.admission import R1AdmissionAuthority


@pytest.fixture
def admission(tmp_path):
    authority = R1AdmissionAuthority(tmp_path / 'private')
    service = ReplayQualification(tmp_path / 'store', real_admission=authority)
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'fixture').write_text('synthetic receipt fixture')
    model = dict(provider='synthetic-provider', model='synthetic-receipt-model', model_fingerprint='b' * 64,
                 native_tool_calls=1, prompt_tokens=12, completion_tokens=8,
                 tool_call_id='call', observation_tool_call_id='call', nonce='test-nonce',
                 observed_nonce='test-nonce', independent_nonce='test-nonce')
    runtime = dict(image_digest='sha256:' + 'a' * 64, config={},
                   model_config={k: model[k] for k in ('provider', 'model', 'model_fingerprint')},
                   tool_config={}, environment={})
    checkpoint = service.register_checkpoint(workspace, [{'role': 'user', 'content': 'fixture'}], 'c' * 64, runtime, STATE_CONTRACT_V1, False)
    record = service._record(checkpoint)
    host = dict(NetworkMode='none', ReadonlyRootfs=True, Privileged=False, NanoCpus=2000000000,
                Memory=4294967296, MemorySwap=4294967296, PidsLimit=128, IpcMode='none',
                Binds=None, CapAdd=None, PidMode='', CapDrop=['ALL'], SecurityOpt=['no-new-privileges:true'],
                Tmpfs={'/workspace': 'rw,nosuid,nodev,size=256m,uid=1000,gid=1000,mode=0700',
                       '/tmp': 'rw,nosuid,nodev,noexec,size=64m,uid=1000,gid=1000,mode=0700'})
    source = dict(qualified=True, effective_config_ok=True, inspect=dict(Image=runtime['image_digest'],
                  Config=dict(User='1000:1000', WorkingDir='/workspace'), HostConfig=host,
                  State=dict(Running=True, Pid=100), Mounts=[dict(Type='tmpfs', Destination=p) for p in ('/workspace', '/tmp')]),
                  status='CapEff:\t0000000000000000\nNoNewPrivs:\t1\n',
                  mountinfo='24 1 0:24 / /workspace rw,nosuid,nodev - tmpfs tmpfs rw\n'
                            '25 1 0:25 / /tmp rw,nosuid,nodev,noexec - tmpfs tmpfs rw\n',
                  mount_stats={p: dict(uid=1000, gid=1000, mode=0o700) for p in ('/workspace', '/tmp')},
                  processes=[dict(pid=1, cmdline='sleep infinity')], tmp_entries=[])
    restored = copy.deepcopy(source)
    restored['inspect']['State']['Pid'] = 101
    report = dict(verified=True, expected_paths=1, actual_paths=1,
                  messages_equal=True, runtime_equal=True, system_equal=True)
    return authority, service, checkpoint, record, source, restored, report, model


def issue(admission, tmp_path):
    authority, service, checkpoint, record, source, restored, report, model = admission
    evidence = []
    for i, value in enumerate((source, restored, report, model, record)):
        path = tmp_path / f'evidence-{i}.json'
        path.write_text(json.dumps(value))
        evidence.append(path)
    trace = tmp_path / 'native-trace.jsonl'
    trace.write_text('{"fixture_only":true}\n')
    evidence.append(trace)
    receipt = authority.issue(checkpoint, record, source, restored, report, model, evidence)
    return receipt, evidence


def test_real_checkpoint_requires_receipt_then_reverifies(admission, tmp_path):
    authority, service, checkpoint, record, *_ = admission
    assert record['registration_reasons'] == []
    assert not service.qualify(checkpoint)['eligible_for_intervention']
    receipt, evidence = issue(admission, tmp_path)
    assert authority.verify(checkpoint, record)
    report = service.qualify(checkpoint)
    assert report['eligible_for_intervention'] and report['admission_receipt'] == receipt
    assert not ReplayQualification(service.workspace_store).qualify(checkpoint)['eligible_for_intervention']
    assert R1AdmissionAuthority(authority.root).verify(checkpoint, record)
    assert authority.root.stat().st_mode & 0o777 == 0o700
    assert authority.key_path.stat().st_mode & 0o777 == 0o600
    evidence[-1].write_text('mutated trace')
    assert not service.qualify(checkpoint)['eligible_for_intervention']


@pytest.mark.parametrize('mutation', ['image', 'network', 'root', 'bind', 'caps', 'process', 'tmp', 'readonly', 'pids', 'nonce', 'tokens', 'tool', 'model', 'fresh', 'equality'])
def test_raw_policy_and_model_evidence_fail_closed(admission, tmp_path, mutation):
    authority, service, checkpoint, record, source, restored, report, model = admission
    host = source['inspect']['HostConfig']
    if mutation == 'image': source['inspect']['Image'] = 'sha256:' + 'f' * 64
    elif mutation == 'network': host['NetworkMode'] = 'host'
    elif mutation == 'root': source['inspect']['Config']['User'] = '0:0'
    elif mutation == 'bind': host['Binds'] = ['/host:/workspace']
    elif mutation == 'caps': host['CapAdd'] = ['SYS_ADMIN']
    elif mutation == 'process': source['processes'].append(dict(pid=2, cmdline='daemon'))
    elif mutation == 'tmp': source['tmp_entries'] = ['/tmp/cache']
    elif mutation == 'readonly': host['ReadonlyRootfs'] = False
    elif mutation == 'pids': host['PidsLimit'] = 0
    elif mutation == 'nonce': model['independent_nonce'] = 'other'
    elif mutation == 'tokens': model['prompt_tokens'] = True
    elif mutation == 'tool': model['observation_tool_call_id'] = 'other'
    elif mutation == 'model': model['model_fingerprint'] = 'd' * 64
    elif mutation == 'fresh': restored['inspect']['State']['Pid'] = 100
    else: report['messages_equal'] = False
    with pytest.raises((ValueError, KeyError)):
        issue(admission, tmp_path)
    assert not service.qualify(checkpoint)['eligible_for_intervention']


@pytest.mark.parametrize('target', ['receipt', 'index', 'key', 'evidence', 'record', 'missing'])
def test_tampering_denies_receipt(admission, tmp_path, target):
    authority, service, checkpoint, record, *_ = admission
    receipt, evidence = issue(admission, tmp_path)
    if target == 'record':
        altered = copy.deepcopy(record)
        altered['messages'][0]['content'] = 'wrong prefix'
        assert not authority.verify(checkpoint, altered)
        return
    path = {'receipt': authority.receipts / (receipt + '.json'),
            'index': authority.root / (checkpoint + '.receipt'), 'key': authority.key_path,
            'evidence': evidence[0], 'missing': evidence[-1]}[target]
    if target == 'missing': path.unlink()
    else:
        path.chmod(0o600)
        path.write_bytes(b'tampered')
    assert not service.qualify(checkpoint)['eligible_for_intervention']


def test_boolean_authority_and_unbound_scheduler_rejected(admission, tmp_path):
    authority, service, checkpoint, record, *_ = admission
    service.real_admission = True
    assert not service.qualify(checkpoint)['eligible_for_intervention']
    service.real_admission = authority
    issue(admission, tmp_path)
    qualified = service.qualify(checkpoint)
    spec = BranchSpec('fixture', 'f' * 64, 'E2-S', record['system_identity'], checkpoint, checkpoint,
                      'original', 'e' * 64, {'seed': 1}, 0, record['runtime_identity'], 'd' * 64, False, 'exploration')
    scheduler = ExperimentScheduler(tmp_path / 'scheduler.db', qualifier=lambda _: qualified,
                                    probe_authorizer=lambda *_: None, executor=lambda _: None, real_admission=authority)
    with pytest.raises(ProtocolError, match='bound replay'):
        scheduler._validate(spec, {})
    scheduler.qualifier = service.qualify
    scheduler._validate(spec, {})
    with pytest.raises(ProtocolError, match='system_identity mismatch'):
        scheduler._validate(replace(spec, system_identity='9' * 64), {})


@pytest.mark.parametrize('component', ['messages', 'runtime', 'system_identity', 'snapshot_id'])
def test_receipt_cannot_transfer_to_another_checkpoint(admission, tmp_path, component):
    authority, service, checkpoint, record, *_ = admission
    issue(admission, tmp_path)
    altered = copy.deepcopy(record)
    if component == 'messages': altered['messages'][0]['content'] = 'different'
    elif component == 'runtime': altered['runtime']['model_config']['model'] = 'different'
    else: altered[component] = '8' * 64
    assert not authority.verify(checkpoint, altered)
    assert not authority.verify(digest(encode(altered)), altered)


def test_missing_raw_evidence_documents_denied(admission, tmp_path):
    authority, service, checkpoint, record, source, restored, report, model = admission
    paths = []
    for index in range(6):
        path = tmp_path / f'fake-{index}.json'
        path.write_text('{"qualified":true}')
        paths.append(path)
    with pytest.raises(ValueError, match='documents missing'):
        authority.issue(checkpoint, record, source, restored, report, model, paths)
    assert not authority.verify(checkpoint, record)


def test_empty_docker_mount_list_requires_kernel_proof(admission, tmp_path):
    authority, service, checkpoint, record, source, restored, *_ = admission
    source['inspect']['Mounts'] = []
    restored['inspect']['Mounts'] = []
    issue(admission, tmp_path)
    assert authority.verify(checkpoint, record)
    assert service.qualify(checkpoint)['eligible_for_intervention']


@pytest.mark.parametrize('mutation', ['missing_info', 'missing_stats', 'filesystem', 'noexec', 'owner', 'mode', 'nested', 'bind_mount'])
def test_mount_intent_alone_cannot_admit(admission, tmp_path, mutation):
    authority, service, checkpoint, record, source, *_ = admission
    source['inspect']['Mounts'] = []
    if mutation == 'missing_info': source.pop('mountinfo')
    elif mutation == 'missing_stats': source.pop('mount_stats')
    elif mutation == 'filesystem': source['mountinfo'] = source['mountinfo'].replace('tmpfs tmpfs', 'ext4 /dev/disk')
    elif mutation == 'noexec': source['mountinfo'] = source['mountinfo'].replace(',noexec', '')
    elif mutation == 'owner': source['mount_stats']['/workspace']['uid'] = 0
    elif mutation == 'mode': source['mount_stats']['/tmp']['mode'] = 0o777
    elif mutation == 'nested': source['mountinfo'] += '26 24 0:26 / /workspace/host rw - tmpfs tmpfs rw\n'
    else: source['inspect']['Mounts'] = [dict(Type='bind', Destination='/workspace')]
    with pytest.raises(ValueError, match='kernel mounts'):
        issue(admission, tmp_path)
    assert not service.qualify(checkpoint)['eligible_for_intervention']
