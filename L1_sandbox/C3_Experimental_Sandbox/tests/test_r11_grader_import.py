"""Synthetic trusted scorer fixture; no OfficeVal/model/Docker execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from c3_sandbox.r1 import officeval


def test_dataclass_scorer_import_and_module_cleanup(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'officeval', officeval)
    script = Path(__file__).resolve().parents[1] / 'docker/r1_grader.py'
    spec = importlib.util.spec_from_file_location('r11_grader_fixture', script)
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    source = '''from __future__ import annotations
from dataclasses import dataclass
@dataclass
class Point:
 value: int
def evaluate(folder):
 assert Point(3).value == 3
 return {"id":"fixture","status":"ok","error":None,"file_name":"answer.docx", "dim1_pass":False,"dim1_reason":"synthetic","dim2_items":[],"total_score":0,"max_score":0}
'''
    folder = tmp_path / 'verifiers'
    folder.mkdir()
    verifier = folder / 'fixture.py'
    verifier.write_text(source)
    manifest = {'tasks': {'fixture': {'official_id': 'fixture', 'verifier_path': 'verifiers/fixture.py',
        'expected_output': 'answer.docx', 'exact_output_name': True, 'max_score': 3}},
        'files': {'verifiers/fixture.py': {'sha256': hashlib.sha256(source.encode()).hexdigest()}}}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    candidate = tmp_path / 'candidate'
    candidate.mkdir()
    (candidate / 'answer.docx').write_bytes(b'synthetic nonempty fixture')
    result = wrapper.grade('fixture', candidate, tmp_path)
    assert result['validity'] == 'valid' and result['reason'] == 'dimension1_failed'
    assert 'official_fixture' not in sys.modules
    (candidate / 'answer.docx').rename(candidate / 'wrong.docx')
    assert wrapper.grade('fixture', candidate, tmp_path)['reason'] == 'invalid_deliverable_inventory'
