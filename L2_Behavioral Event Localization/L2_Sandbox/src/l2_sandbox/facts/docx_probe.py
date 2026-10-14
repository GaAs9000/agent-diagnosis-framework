"""Read-only, bounded OOXML structural probe. Never executes macros or verifiers.

Outputs DIRECT XML properties, not rendered appearance or computed inherited
styles. Object identity is version + part + XPath; no cross-version semantic ID.
"""
from __future__ import annotations
import io
import zipfile
from lxml import etree
from ..io import IntegrityError, digest, sha_bytes

VERSION = 'docx-direct-xml-0.1.0'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
W = '{' + NS['w'] + '}'


def _attrs(element):
    if element is None: return None
    return {etree.QName(k).localname: v for k, v in element.attrib.items()}


def probe(data: bytes, config: dict) -> dict:
    sha = sha_bytes(data)
    base = {'probe_id': 'docx_' + digest({'sha256': sha, 'version': VERSION, 'config': config})[:24],
            'parser': VERSION, 'source_sha256': sha, 'config_sha256': digest(config),
            'package_part': 'word/document.xml', 'status': 'present',
            'semantic_identity_across_versions': 'not_inferred',
            'unknowns': ['inherited_styles', 'rendered_geometry', 'visual_similarity', 'macros',
                         'external_links', 'headers_footers', 'merged_cell_visual_layout'],
            'summary': {}, 'objects': []}
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            infos = z.infolist()
            if len(infos) > 5000 or len({i.filename for i in infos}) != len(infos):
                raise IntegrityError('DOCX duplicate members or excessive entry count')
            if sum(i.file_size for i in infos) > config['max_zip_uncompressed_bytes']:
                raise IntegrityError('DOCX uncompressed size limit')
            if any(i.flag_bits & 1 for i in infos): raise IntegrityError('Encrypted DOCX unsupported')
            info = z.getinfo('word/document.xml')
            if info.file_size > config['max_xml_bytes']: raise IntegrityError('DOCX XML size limit')
            with z.open(info) as f: xml = f.read(config['max_xml_bytes'] + 1)
        if len(xml) > config['max_xml_bytes'] or b'<!DOCTYPE' in xml or b'<!ENTITY' in xml:
            raise IntegrityError('DTD/entity declarations or oversized XML rejected')
        parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False, huge_tree=False)
        root = etree.fromstring(xml, parser)
        if root.tag != W + 'document': raise IntegrityError('Unexpected OOXML root namespace')
        tree = root.getroottree()
        if tree.docinfo.doctype: raise IntegrityError('DOCTYPE is not supported, including non-UTF8 XML')
        body = root.find('w:body', NS)
        if body is None: raise IntegrityError('Missing OOXML body')
        paragraphs = body.findall('.//w:p', NS)
        tables = body.findall('.//w:tbl', NS)
        rows = body.findall('.//w:tr', NS)
        textless = []
        for row in rows:
            text = ''.join(row.xpath('.//w:t/text()', namespaces=NS))
            if not text.strip() and not row.xpath('.//w:drawing|.//w:pict|.//w:object', namespaces=NS): textless.append(row)
        base['summary'] = {'paragraphs_recursive': len(paragraphs), 'tables_recursive': len(tables),
                           'top_level_tables': len(body.findall('w:tbl', NS)), 'rows_recursive': len(rows),
                           'textless_rows_without_drawing': len(textless),
                           'textless_rows_are_not_rendered_empty_row_judgements': True}
        nodes = [*tables, *paragraphs]
        if len(nodes) > config['max_objects']:
            base['status'] = 'partial'; base['unknowns'].append('object_count_limit_truncated')
        for node in nodes[:config['max_objects']]:
            xpath = tree.getpath(node)
            entry = {'object_id': 'object_' + digest({'file': sha, 'part': base['package_part'], 'xpath': xpath})[:24],
                     'file_sha256': sha, 'locator': xpath, 'kind': etree.QName(node).localname,
                     'text_preview': ''.join(node.xpath('.//w:t/text()', namespaces=NS))[:240],
                     'rendered_properties': 'unknown'}
            if node.tag == W + 'tbl':
                borders = node.find('w:tblPr/w:tblBorders', NS)
                direct_rows = node.findall('w:tr', NS)
                if len(direct_rows) > config['max_objects'] or any(len(r.findall('w:tc', NS)) > config['max_objects'] for r in direct_rows):
                    base['status'] = 'partial'; base['unknowns'].append('table_descendant_limit')
                row_limit = min(len(direct_rows), config['max_objects'])
                entry['properties'] = {'direct_rows': len(direct_rows),
                    'direct_cells_per_row': [len(r.findall('w:tc', NS)) for r in direct_rows],
                    'nested_tables': len(node.findall('.//w:tbl', NS)),
                    'direct_table_borders': {etree.QName(x).localname: _attrs(x) for x in borders} if borders is not None else None,
                    'table_style_reference': _attrs(node.find('w:tblPr/w:tblStyle', NS)),
                    'textless_row_indices': [i for i, r in enumerate(direct_rows) if r in textless],
                    'row_properties': [{'index': i, 'direct_height': _attrs(r.find('w:trPr/w:trHeight', NS)),
                        'cells': [{'index': j, 'span': _attrs(c.find('w:tcPr/w:gridSpan', NS)),
                                   'vertical_merge': _attrs(c.find('w:tcPr/w:vMerge', NS)),
                                   'direct_borders': {etree.QName(v).localname: _attrs(v) for v in c.find('w:tcPr/w:tcBorders', NS)}
                                      if c.find('w:tcPr/w:tcBorders', NS) is not None else None}
                                  for j, c in enumerate(r.findall('w:tc', NS)[:config['max_objects']])]}
                        for i, r in enumerate(direct_rows[:row_limit])]}
            else:
                borders = node.find('w:pPr/w:pBdr', NS)
                entry['properties'] = {'direct_spacing': _attrs(node.find('w:pPr/w:spacing', NS)),
                    'style_reference': _attrs(node.find('w:pPr/w:pStyle', NS)),
                    'direct_borders': {etree.QName(x).localname: _attrs(x) for x in borders} if borders is not None else None,
                    'has_drawing': bool(node.xpath('.//w:drawing|.//w:pict', namespaces=NS))}
            base['objects'].append(entry)
    except (zipfile.BadZipFile, KeyError, etree.XMLSyntaxError, IntegrityError, OSError, RuntimeError) as exc:
        base.update(status='unsupported', error={'type': type(exc).__name__, 'reason': str(exc)}, objects=[], summary={})
    return base
