"""Auditable public-input variants, supplied through L1's benchmark plugin port.

Only Asset paths are passed to the extractor. Neither task.private_files nor
official results are inputs. The native adapter still validates locked assets.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

from .common import L3Error, digest, sha_file

TRANSFORM_VERSION = 'openxml-structure-1'
HEADER = '\n\n[L3 PUBLIC CONTEXT]\n'


def _xml(archive, name):
    info = archive.getinfo(name)
    if info.file_size > 8 * 1024 * 1024:
        raise L3Error('Public XML part exceeds extraction limit')
    return ET.fromstring(archive.read(name))


def describe_asset(asset) -> dict:
    path = Path(asset.path)
    if sha_file(path) != asset.sha256:
        raise L3Error('Public input changed before structure extraction')
    extension = Path(asset.name).suffix.lower()
    out = {'name': asset.name, 'format': extension, 'size_bytes': asset.size_bytes}
    if extension not in ('.docx', '.xlsx', '.pptx'):
        return out
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) > 10000 or sum(i.file_size for i in infos) > 128 * 1024 * 1024:
            raise L3Error('Public Office archive exceeds extraction limit')
        if extension == '.docx':
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            doc = _xml(archive, 'word/document.xml')
            out['paragraph_count'] = len(doc.findall('.//w:p', ns))
            tables = doc.findall('.//w:tbl', ns)
            out['tables'] = [{'rows': len(t.findall('./w:tr', ns)),
                              'cells_per_row': [len(r.findall('./w:tc', ns))
                                                for r in t.findall('./w:tr', ns)]}
                             for t in tables]
        elif extension == '.xlsx':
            ns = {'x': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
            workbook = _xml(archive, 'xl/workbook.xml')
            out['sheet_names'] = [s.get('name') for s in workbook.findall('./x:sheets/x:sheet', ns)]
            out['worksheets'] = []
            for name in sorted(archive.namelist()):
                if not re.fullmatch(r'xl/worksheets/sheet\d+\.xml', name):
                    continue
                sheet = _xml(archive, name)
                dimension = sheet.find('x:dimension', ns)
                formulas = [c.get('r') for c in sheet.findall('.//x:c', ns)
                            if c.find('x:f', ns) is not None]
                out['worksheets'].append({'part': name,
                    'dimension': dimension.get('ref') if dimension is not None else None,
                    'merged_ranges': [m.get('ref') for m in sheet.findall('.//x:mergeCell', ns)],
                    'formula_cell_count': len(formulas), 'formula_coordinates': formulas[:256]})
        else:
            ns = {'p': 'http://schemas.openxmlformats.org/presentationml/2006/main'}
            slides = sorted(n for n in archive.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml', n))
            out['slides'] = []
            for name in slides:
                tree = _xml(archive, name).find('.//p:spTree', ns)
                counts = Counter(c.tag.rsplit('}', 1)[-1] for c in tree) if tree is not None else {}
                out['slides'].append({'part': name, 'object_types': dict(counts)})
    return out


def structure_context(task) -> str:
    import json
    return json.dumps({'transform': TRANSFORM_VERSION,
                       'public_file_structure': [describe_asset(a) for a in task.assets]},
                      ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def placebo_context(structure: str) -> str:
    neutral = 'A quiet garden has a stone path. Clouds pass over a distant hill. '
    return (neutral * (len(structure) // len(neutral) + 1))[:len(structure)]


class ConditionOfficeVal:
    """Installed L1 benchmark adapter; preserves native scorer and public files."""

    def __init__(self, config):
        from c3_sandbox.l1.benchmarks.officeval.adapter import NativeOfficeVal
        self.config = dict(config)
        if self.config.get('l3_transform_version') != TRANSFORM_VERSION:
            raise L3Error('Unrecognized L3 public-input transform')
        self.native = NativeOfficeVal(config)

    def list_tasks(self):
        return self.native.list_tasks()

    def load(self, task_id):
        task = self.native.load(task_id)
        context = self.config.get('l3_context')
        if context not in ('original', 'structure', 'placebo'):
            raise L3Error('Unknown public-input condition')
        structure = structure_context(task)
        expected = '' if context == 'original' else (
            structure if context == 'structure' else placebo_context(structure))
        if expected != self.config.get('l3_appendix'):
            raise L3Error('Frozen public appendix differs from the public input transform')
        prompt = task.prompt + (HEADER + expected if expected else '')
        provenance = {**task.provenance, 'l3_condition': {
            'id': self.config.get('l3_condition_id'), 'context': context,
            'transform_version': TRANSFORM_VERSION, 'base_prompt_sha256': digest(task.prompt),
            'public_prompt_sha256': digest(prompt), 'appendix_sha256': digest(expected),
            'appendix_characters': len(expected), 'appendix_utf8_bytes': len(expected.encode()),
            'position': 'after_native_task_instruction', 'asset_contents_unchanged': True}}
        return replace(task, prompt=prompt, provenance=provenance)
