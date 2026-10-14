from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from c3_sandbox.l1.contracts import ExecutionBudget
from c3_sandbox.l1.import_gemini import import_episode
from c3_sandbox.l1.package import seal_package, verify_package
from c3_sandbox.l1.recorder import EventJournal, scan_journal
from c3_sandbox.l1.schema import validate
from c3_sandbox.l1.storage import canonical, digest, file_digest, read_json, safe_relative, write_json


def rewrite(path, value): path.write_bytes(canonical(value)+b'\n')
def reseal(package):
    (package/'evidence_index.json').unlink()
    return seal_package(package, scan_journal(package/'trajectory.jsonl')['events'])
def rechain(package, events):
    previous = None
    for i,event in enumerate(events):
        event['seq']=i
        event['event_id']=f"{event['episode_id']}:e{i:06d}"
        event['prev_event_sha256']=previous
        event.pop('event_sha256',None)
        event['event_sha256']=digest(event)
        previous=event['event_sha256']
    (package/'trajectory.jsonl').write_bytes(b''.join(canonical(e)+b'\n' for e in events))


@pytest.fixture
def package(make_source,tmp_path):
    source=make_source()
    out=tmp_path/'package'
    result=import_episode(source,out,episode_id='fixture.episode')
    assert result['verification']['verified'],result
    return source,out,result


def test_roundtrip_raw_bytes(package):
    source,out,_=package
    assert (source/'result.json').read_bytes()==(out/'raw/result.json').read_bytes()
    raw=read_json(source/'result.json')
    for i,r in enumerate(raw['raw_gemini_records']):
        for side in ('request','response'):
            assert (out/f'raw/model/{i:06d}.{side}.json').read_bytes()==r[side+'_raw'].encode()


def test_independent_status_dimensions(package):
    _,out,_=package
    m=read_json(out/'manifest.json')
    assert m['recording_mode']=='synthetic_fixture'
    assert m['status']['task_outcome']['status']=='failure'
    assert m['status']['evaluation_validity']['status']=='valid'
    assert m['status']['protocol_validity']['status']=='valid'
    assert m['status']['execution_validity']['status']=='unknown'
    assert m['status']['evidence_completeness']=='incomplete'
    assert m['capabilities']['cold_resume_qualified'] is False


def test_raw_and_delivered_are_not_conflated(package):
    _,out,_=package
    ev=scan_journal(out/'trajectory.jsonl')['events']
    raw=next(e for e in ev if e['kind']=='tool.result')['payload']['raw_observation']
    delivered=next(e for e in ev if e['kind']=='observation.delivered')['payload']['delivered_observation']
    assert raw['stdout']=='visible prefix PLUS raw text'
    assert delivered['stdout']=='visible prefix'
    assert raw['exit_code']==1  # agent tool error is not automatic infrastructure invalidity


def test_native_and_dispatch_ids_retained(package):
    _,out,_=package
    call=next(e for e in scan_journal(out/'trajectory.jsonl')['events'] if e['kind']=='tool.call')['payload']
    assert call['native_tool_call_id']=='native-123'
    assert call['source_dispatch_id']=='legacy-0'
    assert call['linkage']=='sequence_and_exact_name_arguments'


def test_missing_snapshot_paths_are_audited(package):
    _,out,_=package
    audit=read_json(out/'raw/source_path_audit.json')
    assert len(audit['entries'])==2
    assert all(not e['retained_in_source'] for e in audit['entries'])
    assert all(e['archived_ref'] is None for e in audit['entries'])


def test_state_differences_have_distinct_assurance(package):
    _,out,_=package
    states=[json.loads(line) for line in (out/'state_changes.jsonl').read_text().splitlines()]
    assert len(states)==2
    assert {s['verification'] for s in states}=={'reported_without_retained_snapshots','recomputed_from_retained_bytes'}
    assert states[0]['object_id']==states[1]['object_id']


def test_no_invented_timing_price_or_seed(package):
    _,out,_=package
    m=read_json(out/'manifest.json')
    assert m['configuration']['declared_seed']==2026
    assert m['configuration']['seed_in_native_requests'] is False
    assert m['configuration']['resolved_config_verified'] is False
    assert all(e['observed_at'] is None for e in scan_journal(out/'trajectory.jsonl')['events'])
    coverage={x['field']:x for x in m['coverage']}
    assert coverage['model_cost_usd']['status']=='unknown'
    assert coverage['model_timestamps_and_latency']['status']=='missing'


def test_no_next_request_does_not_fabricate_delivery(make_source,tmp_path):
    src=make_source(stopped_reason='max_turns',no_next_request=True)
    result=import_episode(src,tmp_path/'out')
    assert result['verification']['verified']
    assert result['counts']['delivered_observations']==0
    assert result['status']['termination']=='budget_exhausted'


@pytest.mark.parametrize('reason,expected',[('assistant_finished','assistant_finished'),('max_turns','budget_exhausted'),('protocol_error','protocol_error'),('tool_execution_error','tool_execution_error'),('timeout','interrupted'),('error','provider_error'),('unrecognized','unknown')])
def test_stop_reason_preserved(make_source,tmp_path,reason,expected):
    result=import_episode(make_source(stopped_reason=reason),tmp_path/'out')
    assert result['verification']['verified']
    assert result['status']['termination']==expected


def test_provider_failure_has_no_fabricated_response(make_source,tmp_path):
    result=import_episode(make_source(stopped_reason='error',protocol='invalid',provider_error=True),tmp_path/'out')
    assert result['verification']['verified'],result
    assert result['counts']['model_requests']==1
    assert result['counts']['model_responses']==0
    assert result['counts']['tool_results']==0
    assert (tmp_path/'out/raw/model/000000.response.json').read_bytes()==b''


def test_initial_grade_failure_does_not_poison_final_evaluation(make_source,tmp_path):
    result=import_episode(make_source(initial_validity='infrastructure_invalid'),tmp_path/'out')
    assert result['verification']['verified']
    assert result['status']['evaluation_validity']['status']=='valid'


def test_final_grade_failure_is_not_task_failure(make_source,tmp_path):
    result=import_episode(make_source(final_validity='infrastructure_invalid'),tmp_path/'out')
    assert result['verification']['verified']
    assert result['status']['evaluation_validity']['status']=='invalid'
    assert result['status']['task_outcome']['status']=='unknown'


def test_success_is_preserved_not_targeted(make_source,tmp_path):
    result=import_episode(make_source(completed=True),tmp_path/'out')
    assert result['verification']['verified']
    assert result['status']['task_outcome']['status']=='success'


def test_missing_grade_files_stay_unknown(make_source,tmp_path):
    src=make_source()
    shutil.rmtree(src/'final-grade')
    result=import_episode(src,tmp_path/'out')
    assert result['verification']['verified']
    assert result['status']['evaluation_validity']['status']=='unknown'


def test_checkpoint_claim_is_not_qualification(make_source,tmp_path):
    src=make_source()
    raw=read_json(src/'result.json')
    raw['checkpoints']=[{'checkpoint_id':'legacy-cp','replay_qualified':True}]
    rewrite(src/'result.json',raw)
    result=import_episode(src,tmp_path/'out')
    assert result['verification']['verified']
    cp=read_json(tmp_path/'out/checkpoints/index.json')
    assert cp['cold_resume_qualified'] is False
    assert cp['items'][0]['qualification']=='unqualified'


def test_wrong_wire_identity_is_detected_and_retained(make_source,tmp_path):
    src=make_source()
    raw=read_json(src/'result.json');raw['raw_gemini_records'][0]['request_identity']='0'*64
    rewrite(src/'result.json',raw)
    result=import_episode(src,tmp_path/'out')
    assert not result['verification']['verified']
    assert read_json(tmp_path/'out/raw/result.json')==raw
    assert (tmp_path/'out/manifest.json').exists()


def test_orphan_dispatch_is_not_dropped(make_source,tmp_path):
    src=make_source()
    raw=read_json(src/'result.json');raw['tool_dispatch'][0]['arguments']={'code':'different'}
    rewrite(src/'result.json',raw)
    result=import_episode(src,tmp_path/'out')
    assert not result['verification']['verified']
    assert result['counts']['tool_results']==1
    assert result['counts']['tool_calls']==2


@pytest.mark.parametrize('path',['raw/model/000000.request.json','trajectory.jsonl','state_changes.jsonl','artifacts/final/artifact.txt','official_grade.json','checkpoints/index.json','manifest.json'])
def test_tampering_is_detected(package,path):
    _,out,report=package
    with (out/path).open('ab') as f:f.write(b'altered')
    assert not verify_package(out,expected_index_sha256=report['index_sha256'])['verified']


def test_external_anchor_rejects_resealed_change(package):
    _,out,report=package
    m=read_json(out/'manifest.json');m['limitations'].append('changed')
    rewrite(out/'manifest.json',m);reseal(out)
    assert not verify_package(out,expected_index_sha256=report['index_sha256'])['verified']


def test_unanchored_checker_reports_its_limit(package):
    _,out,_=package
    v=verify_package(out)
    assert v['verified']
    assert any(w['code']=='no_external_trust_anchor' for w in v['warnings'])


def test_forged_success_rejected_even_after_reseal(package):
    _,out,_=package
    m=read_json(out/'manifest.json');m['status']['task_outcome']['status']='success'
    rewrite(out/'manifest.json',m);reseal(out)
    assert not verify_package(out)['verified']


def test_fixture_cannot_be_upgraded_to_live(package):
    _,out,_=package
    m=read_json(out/'manifest.json');m['recording_mode']='live'
    rewrite(out/'manifest.json',m);reseal(out)
    assert not verify_package(out)['verified']


def test_normalized_tool_result_cannot_override_raw(package):
    _,out,_=package
    events=scan_journal(out/'trajectory.jsonl')['events']
    next(e for e in events if e['kind']=='tool.result')['payload']['raw_observation']['stdout']='fabricated'
    rechain(out,events);reseal(out)
    v=verify_package(out)
    assert not v['verified']
    assert any(e['code']=='raw_observation_projection_mismatch' for e in v['errors'])


def test_relocation_needs_no_original_source(package,tmp_path):
    src,out,report=package
    moved=tmp_path/'moved-package';shutil.copytree(out,moved)
    shutil.rmtree(src.parent.parent)
    assert verify_package(moved,expected_index_sha256=report['index_sha256'])['verified']


def test_export_is_exclusive_and_source_unchanged(package):
    src,out,_=package
    before=file_digest(src/'result.json')
    with pytest.raises(FileExistsError):import_episode(src,out)
    assert file_digest(src/'result.json')==before


def test_source_symlink_fails_preserving_partial_export(make_source,tmp_path):
    src=make_source()
    (src/'public-input/link').symlink_to(src/'result.json')
    out=tmp_path/'out'
    with pytest.raises(ValueError):import_episode(src,out)
    assert (out/'raw/result.json').exists()
    assert read_json(out/'export_error.json')['partial_export_preserved'] is True


def test_package_symlink_is_rejected(package,tmp_path):
    _,out,_=package
    (out/'link').symlink_to(tmp_path/'external')
    assert not verify_package(out)['verified']


@pytest.mark.parametrize('path',['../escape','/absolute','a/../../b','a\\b','a//b','./a','C:/file','\x00','.'])
def test_unsafe_paths_rejected(path):
    with pytest.raises(ValueError):safe_relative(path)


def test_unknown_schema_version_rejected(package):
    _,out,_=package
    m=read_json(out/'manifest.json');m['schema_version']='999.0.0'
    with pytest.raises(ValueError):validate('manifest',m)


def test_unknown_top_level_property_rejected(package):
    _,out,_=package
    m=read_json(out/'manifest.json');m['guessed_cause']='no'
    with pytest.raises(ValueError):validate('manifest',m)


def test_object_ids_are_episode_scoped(make_source,tmp_path):
    src=make_source()
    import_episode(src,tmp_path/'a',episode_id='A')
    import_episode(src,tmp_path/'b',episode_id='B')
    a=read_json(tmp_path/'a/artifacts/inventory.json')['files']
    b=read_json(tmp_path/'b/artifacts/inventory.json')['files']
    assert a[0]['object_id']==a[1]['object_id']
    assert a[0]['object_id']!=b[0]['object_id']


def test_journal_torn_tail_preserved(tmp_path):
    journal=EventJournal(tmp_path,'fixture')
    journal.append('episode.started',{'fixture':True},assurance='synthetic_fixture')
    with journal.path.open('ab') as f:f.write(b'{"partial":')
    before=journal.path.read_bytes()
    report=scan_journal(journal.path)
    assert not report['complete']
    assert report['valid_prefix_events']==1
    assert journal.path.read_bytes()==before


def test_recorder_error_event_needs_details(tmp_path):
    journal=EventJournal(tmp_path,'fixture')
    with pytest.raises(ValueError):journal.append('recorder.error',{})
    assert journal.path.read_bytes()==b''


@pytest.mark.parametrize('value',[0,-1,True,1.5])
def test_invalid_request_budget(value):
    with pytest.raises(ValueError):ExecutionBudget(value,2)


def test_l1_import_does_not_load_agent_or_analyst_modules():
    root=Path(__file__).resolve().parents[2]
    code="""
import sys
import c3_sandbox.l1.cli
import c3_sandbox.l1.import_gemini
import c3_sandbox.l1.package
for name in sys.modules:
    assert not name.startswith(('inspect_ai','mlx','c3_sandbox.integrated','c3_sandbox.gateway')),(name,'unexpected dependency')
"""
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_cli_verify_and_event_read(package):
    _,out,result=package
    from c3_sandbox.l1.cli import main
    assert main(['verify',str(out),'--expected-index-sha256',result['index_sha256']])==0
    assert main(['events',str(out),'--kind','tool.result','--limit','1'])==0


def test_deleted_state_delta_detected_after_reseal(package):
    _,out,_=package
    states=[json.loads(line) for line in (out/'state_changes.jsonl').read_text().splitlines()]
    (out/'state_changes.jsonl').write_bytes(canonical(states[-1])+b'\n')
    m=read_json(out/'manifest.json');m['counts']['state_changes']=1;rewrite(out/'manifest.json',m)
    reseal(out)
    v=verify_package(out)
    assert not v['verified']
    assert any(e['code']=='step_delta_coverage_mismatch' for e in v['errors'])


def test_changed_delivery_binding_is_rejected(package):
    _,out,_=package
    ev=scan_journal(out/'trajectory.jsonl')['events']
    delivery=next(e for e in ev if e['kind']=='observation.delivered')
    delivery['links']['delivered_in_model_call_id']='fixture.episode:m000000'
    rechain(out,ev);reseal(out)
    assert not verify_package(out)['verified']


def test_object_identity_cannot_be_relabelled(package):
    _,out,_=package
    inv=read_json(out/'artifacts/inventory.json');inv['files'][0]['object_id']='forged'
    rewrite(out/'artifacts/inventory.json',inv);reseal(out)
    assert not verify_package(out)['verified']


@pytest.mark.parametrize('value',[float('nan'),float('inf'),float('-inf'),True,0,-1])
def test_invalid_optional_budgets(value):
    with pytest.raises(ValueError):ExecutionBudget(1,1,wall_time_seconds=value)


def test_fractional_token_budget_rejected():
    with pytest.raises(ValueError):ExecutionBudget(1,1,max_tokens=1.5)


def test_known_protocol_invalid_is_not_hidden_by_unknown_environment(make_source,tmp_path):
    result=import_episode(make_source(protocol='invalid',stopped_reason='protocol_error'),tmp_path/'out')
    assert result['verification']['verified']
    assert result['status']['execution_validity']['status']=='invalid'
    assert result['status']['evaluation_validity']['status']=='valid'
