from pathlib import Path
import pytest
from l2_sandbox.config import load
from l2_sandbox.evidence.l1 import L1Reader
from l2_sandbox.evidence.view import EvidenceView
from l2_sandbox.facts.build import FactBuilder
from l2_sandbox.facts.docx_probe import probe
from l2_sandbox.io import AccessDenied,BudgetExceeded,canonical
from conftest import docx_bytes,make_source

@pytest.fixture
def builder(source,tmp_path,config):
    out=tmp_path/'facts';out.mkdir()
    return FactBuilder(L1Reader(source),out,config).build()


def test_version_objects_and_causal_abstention(builder):
    fs=builder.facts
    assert any(f['fact_type']=='object_structure_change' for f in fs)
    assert len(builder.objects)==2
    for p in builder.objects.values():
        assert p['semantic_identity_across_versions']=='not_inferred'
        assert 'inherited_styles' in p['unknowns']


def test_private_paths_not_available(builder,config):
    v=EvidenceView(builder,config)
    for ref in ['benchmark/rubric.json','../episode/official_grade.json','/etc/passwd','source_blobs/secret','PRIVATE_RUBRIC_SENTINEL']:
        with pytest.raises(AccessDenied):v.call('read_evidence',{'ref_id':ref})
    assert 'PRIVATE_RUBRIC_SENTINEL' not in canonical(v.summary).decode()


def test_no_shell_or_network_tool(builder,config):
    v=EvidenceView(builder,config)
    for tool in ('run','bash','read_file','fetch','resume','execute'):
        with pytest.raises(AccessDenied):v.call(tool,{'path':'/etc/passwd'})


def test_sibling_results_never_loaded(source,tmp_path,config):
    sibling=source.parent/'branch';sibling.mkdir();(sibling/'official_grade.json').write_text('{"SECRET_VALIDATION_RESULT":true}')
    out=tmp_path/'facts';out.mkdir();b=FactBuilder(L1Reader(source),out,config).build()
    view=EvidenceView(b,config)
    result=view.call('get_requirement_evidence',{'query':'SECRET_VALIDATION_RESULT'})
    assert result['items']==[]


def test_direct_missing_border_is_not_rendered_absence(config):
    p=probe(docx_bytes(style=True),config['objects'])
    table=next(o for o in p['objects'] if o['kind']=='tbl')
    assert table['properties']['direct_table_borders'] is None and table['properties']['table_style_reference']['val']=='CustomStyle'
    assert table['rendered_properties']=='unknown'


def test_parser_blocks_entity(config):
    p=probe(docx_bytes(entity=True),config['objects'])
    assert p['status']=='unsupported' and not p['objects']


def test_bad_docx_does_not_execute_or_crash(config):
    p=probe(b'not a zip',config['objects'])
    assert p['status']=='unsupported'


def test_probe_size_limit(config):
    config['objects']['max_xml_bytes']=16
    assert probe(docx_bytes(),config['objects'])['status']=='unsupported'


def test_probe_object_limit_declared(config):
    config['objects']['max_objects']=1;p=probe(docx_bytes(),config['objects'])
    assert p['status']=='partial' and len(p['objects'])==1


def test_version_limit_preserved(source,tmp_path,config):
    config['objects']['max_docx_versions']=1;out=tmp_path/'b';out.mkdir()
    b=FactBuilder(L1Reader(source),out,config).build()
    assert len(b.objects)==1 and any(x.get('reason')=='max_docx_versions' for x in b.coverage)


def test_budget_limits(builder,config):
    config['budget']['max_tool_calls']=1;v=EvidenceView(builder,config);v.call('get_episode_summary',{})
    with pytest.raises(BudgetExceeded):v.call('get_episode_summary',{})


def test_byte_budget(builder,config):
    config['budget']['max_total_read_bytes']=10;v=EvidenceView(builder,config)
    with pytest.raises(BudgetExceeded):v.call('get_episode_summary',{})

@pytest.mark.parametrize('args',[{'limit':0},{'start':-1},{'limit':31},{'path':'/etc/passwd'},{'limit':'1'}])
def test_bad_tool_arguments(builder,config,args):
    with pytest.raises(AccessDenied):EvidenceView(builder,config).call('get_requirement_evidence',args)


def test_full_facts_only_marked_seen(builder,config):
    v=EvidenceView(builder,config);r=v.call('get_requirement_evidence',{'limit':1})
    assert v.seen_fact_ids=={r['items'][0]['fact_id']}
    assert v.bytes_read>0


def test_no_empty_delta_failure_invention(source,tmp_path,config):
    # Genuine zero changes needs a source generated accordingly; this fixture does
    # have changes, so the system must not publish a no-change fact.
    out=tmp_path/'b';out.mkdir();b=FactBuilder(L1Reader(source),out,config).build()
    assert not any(f['fact_type']=='no_recorded_file_change' for f in b.facts)


def test_utf16_doctype_rejected(config):
    import io,zipfile
    data=docx_bytes(entity=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:xml=z.read('word/document.xml').decode('utf8')
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w') as z:z.writestr('word/document.xml',xml.encode('utf-16'))
    assert probe(out.getvalue(),config['objects'])['status']=='unsupported'


def test_model_request_body_available_only_by_registered_lazy_reference(builder,config):
    v=EvidenceView(builder,config)
    page=v.call('list_events',{'kind':'model.request','limit':1})
    rid=page['items'][0]['available_evidence_ref_ids'][0]
    assert rid not in builder.refs.records
    result=v.call('read_evidence',{'ref_id':rid})
    assert 'contents' in result['value']
    assert rid in builder.refs.records
    assert builder.refs.records[rid]['source_path'].startswith('raw/model/')


def test_lazy_response_preserves_native_bytes_as_data(builder,config):
    v=EvidenceView(builder,config)
    page=v.call('list_events',{'kind':'model.response','limit':1})
    rid=page['items'][0]['available_evidence_ref_ids'][0]
    result=v.call('read_evidence',{'ref_id':rid})
    assert 'thoughtSignature' in result['value']['text']
