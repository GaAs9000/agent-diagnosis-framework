"""Adversarial, pure tests for the explicitly bounded v2 exclusions."""
import copy
import pytest
from l2_sandbox.metadata_contract import classify_metadata, listing_command, normalize_listing, POINTER_COMMAND

LEFT = 'total 144\ndrwxr-xr-x 1 501 dialout 128 Oct 18 17:00 .\n-rw-r--r-- 1 501 dialout 76311 Oct 18 17:02 file.docx\n'
RIGHT = 'total 92\ndrwxr-xr-x 1 1000 users 128 Oct 19 03:22 .\n-rw-r--r-- 1 1000 users 76311 Oct 19 03:22 file.docx\n'


def observation(stdout):
    return {'stdout': stdout, 'stderr': '', 'output': stdout, 'exit_code': 0,
            'output_complete': True, 'stop_reason': None, 'oom_killed': False}


def classify(left=LEFT, right=RIGHT, command='ls -la /workspace/submission/', **kw):
    return classify_metadata({'name':'bash','arguments':{'command':command}},
         kw.pop('original_obs', observation(left)), kw.pop('actual_obs', observation(right)),
         left.encode(), right.encode(), b'', kw.pop('actual_stderr', b''),
         state_exact=kw.pop('state_exact', True), image_exact=kw.pop('image_exact', True))


def test_metadata_fields_only():
    row = classify()
    assert row['classification'] == 'ls_runtime_metadata_v2'
    assert row['raw_bytes_equal'] is False
    assert 'directory_total_allocated_blocks' in row['excluded_fields']


@pytest.mark.parametrize('old,new', [('file.docx','other.docx'),('76311','76312'),
 ('-rw-r--r--','-rwxr--r--'),('1 1000 users 76311','2 1000 users 76311')])
def test_meaningful_listing_changes_rejected(old,new):
    assert classify(right=RIGHT.replace(old,new))['classification'] == 'unexplained'


@pytest.mark.parametrize('command', ['echo "ls -la /workspace/submission/"',
 'printf fake && ls -la /workspace/submission/', 'ls -la /workspace/submission/; true',
 'ls -la /workspace/submission/ | cat', 'ls -la $(pwd)',
 'ls -la /workspace/../etc', 'cd /workspace/input && python3 -c "print(1)" && ls -la',
 'cp /workspace/input/a /workspace/submission/a && echo injected && ls -la /workspace/submission/'])
def test_arbitrary_command_not_normalized(command):
    assert not listing_command(command)
    assert classify(command=command)['classification'] == 'unexplained'


@pytest.mark.parametrize('command', ['cd /workspace/input && ls -la',
 'cp /workspace/input/a /workspace/submission/a && ls -la /workspace/submission/',
 'cp /workspace/input/a /workspace/submission/a\ncp /workspace/input/b /workspace/submission/b\nls -la /workspace/submission/',
 'ls -la /workspace/submission/a.pptx'])
def test_literal_workspace_operations(command):
    assert listing_command(command)
    assert classify(command=command)['classification'] == 'ls_runtime_metadata_v2'


@pytest.mark.parametrize('mutate', [lambda t:t + '-rw-r--r-- 1 1000 users 1 Oct 19 03:22 extra\n',
 lambda t:t.splitlines(keepends=True)[0] + t.splitlines(keepends=True)[1],
 lambda t:t+'unrelated output\n', lambda t:t.replace('total 92','total garbage')])
def test_listing_census_and_shape(mutate):
    assert classify(right=mutate(RIGHT))['classification'] == 'unexplained'


@pytest.mark.parametrize('field,value', [('exit_code',1),('stop_reason','timeout'),
 ('oom_killed',True),('output_complete',False),('stderr','unexpected'),('unrelated','changed')])
def test_observation_other_fields_must_match(field,value):
    obs=observation(RIGHT);obs[field]=value
    assert classify(actual_obs=obs)['classification'] == 'unexplained'


@pytest.mark.parametrize('kwargs', [{'state_exact':False},{'image_exact':False},{'actual_stderr':b'error'}])
def test_proof_boundary(kwargs):
    assert classify(**kwargs)['classification'] == 'unexplained'


def test_single_file_preserves_total_presence():
    left=LEFT.splitlines(keepends=True)[-1];right=RIGHT.splitlines(keepends=True)[-1]
    assert classify(left,right,command='ls -la /workspace/submission/file.docx')['classification']=='ls_runtime_metadata_v2'
    assert classify(left,'total 92\n'+right)['classification']=='unexplained'


def test_pointer_only_with_literal_frozen_command():
    left='warning <urllib.error.HTTPError object at 0xabc123>\ncode 0xdead\n'
    right=left.replace('0xabc123','0xddd456')
    assert classify(left,right,command=POINTER_COMMAND)['classification']=='python_object_repr_address_v1'
    assert classify(left,right,command='python3 -c "print(fake)"')['classification']=='unexplained'
    assert classify(left,right.replace('0xdead','0xbeef'),command=POINTER_COMMAND)['classification']=='unexplained'


@pytest.mark.parametrize('left,right', [('error at 0xabc','error at 0xddd'),
 ('<HTTPError at 0xabc>','<HTTPError at 0xddd>'),('https://server/0xabc','https://server/0xddd'),
 ('<foo object at 0xabc> x','<foo object at 0xddd> y')])
def test_not_arbitrary_hex_mask(left,right):
    assert classify(left,right,command=POINTER_COMMAND)['classification']=='unexplained'


def test_exact_and_state():
    assert classify(LEFT,LEFT)['classification']=='exact'
    assert classify(LEFT,LEFT,state_exact=False)['classification']=='unexplained'


def test_report_binding_tamper_fails_before_reaudit(tmp_path):
    import importlib.util
    import json
    from pathlib import Path
    script=Path(__file__).resolve().parents[1] / 'scripts/reclassify_prospective_v2.py'
    spec=importlib.util.spec_from_file_location('test_metadata_reaudit_binding',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    report=tmp_path/'report.json';report.write_text(json.dumps({'references':{}}))
    with pytest.raises(ValueError,match='report binding changed'):
        module.verify_reaudit(report,expected_refs={'g2_v2':{'sha256':'0'*64}})


def test_projected_gate_keeps_research_boundary():
    from l2_sandbox.metadata_contract import validate_projected_report, VERSION
    row={'batch_id':'test','replay':1,'call_id':'a','v2_classification':'exact'}
    report={'schema':VERSION+'-reaudit','posthoc_extension':True,'post_hoc_forensic_extension':True,
            'actor_requests':0,'analyst_requests':0,'new_tool_calls':0,
            'comparisons_reviewed':1,'rows':[row],'all_comparisons_admissible':True,
            'statistical_equivalence_established':False}
    assert validate_projected_report(report)
    for field,value in [('statistical_equivalence_established',True),('actor_requests',1),
                        ('post_hoc_forensic_extension',False),('comparisons_reviewed',2)]:
        changed=copy.deepcopy(report);changed[field]=value
        assert not validate_projected_report(changed)
    report['rows'].append(copy.deepcopy(row));report['comparisons_reviewed']=2
    assert not validate_projected_report(report)
