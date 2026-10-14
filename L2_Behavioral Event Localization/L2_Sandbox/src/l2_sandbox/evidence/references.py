"""Content-addressed source slices. No arbitrary path read is exposed to Analyst."""
from __future__ import annotations
import copy
from pathlib import Path
from ..io import (AccessDenied, IntegrityError, canonical, digest, loads, pointer, sha_bytes,
                  write_new, write_json)
from .l1 import L1Reader


def select(data: bytes, selector: dict):
    kind = selector.get('kind')
    if kind == 'json': return pointer(loads(data), selector['pointer'])
    if kind == 'jsonl':
        lines = data.splitlines(keepends=True)
        n = selector['line']
        if type(n) is not int or n < 0 or n >= len(lines) or not lines[n].endswith(b'\n'):
            raise IntegrityError('Invalid JSONL selector')
        return pointer(loads(lines[n]), selector['pointer'])
    if kind == 'bytes':
        start, end = selector['start'], selector['end']
        if type(start) is not int or type(end) is not int or not 0 <= start <= end <= len(data):
            raise IntegrityError('Invalid byte selector')
        return {'encoding': 'utf8-replacement', 'text': data[start:end].decode('utf-8', 'replace'),
                'byte_range': [start, end], 'total_bytes': len(data)}
    if kind == 'binary':
        return {'sha256': sha_bytes(data), 'size_bytes': len(data), 'readable_as_text': False}
    raise IntegrityError('Unknown source selector')


class ReferenceStore:
    def __init__(self, reader: L1Reader, output: Path):
        self.reader = reader; self.output = output
        self.records: dict[str, dict] = {}
        self.values: dict[str, object] = {}
        self._copied = set()
        self.declared: dict[str, dict] = {}

    def add(self, path: str, *, json_pointer: str | None = None, line: int | None = None,
            byte_range: tuple[int, int] | None = None, binary: bool = False,
            event_ids: list[str] | None = None, analyst_visible: bool = True) -> str:
        data = self.reader.read(path)
        if binary: selector = {'kind': 'binary'}
        elif byte_range is not None: selector = {'kind': 'bytes', 'start': byte_range[0], 'end': byte_range[1]}
        elif line is not None: selector = {'kind': 'jsonl', 'line': line, 'pointer': json_pointer or ''}
        else: selector = {'kind': 'json', 'pointer': json_pointer or ''}
        value = select(data, selector)
        sha = sha_bytes(data)
        descriptor = {'source_index_sha256': self.reader.index_sha256, 'source_path': path,
                      'source_sha256': sha, 'size_bytes': len(data), 'selector': selector,
                      'event_ids': event_ids or [], 'analyst_visible': analyst_visible}
        rid = 'ref_' + digest(descriptor)[:24]
        if rid not in self.records:
            blob = 'source_blobs/' + sha
            if sha not in self._copied:
                write_new(self.output, blob, data); self._copied.add(sha)
            self.records[rid] = {'ref_id': rid, **descriptor, 'local_blob': blob, 'selected_value_sha256': digest(value)}
            self.values[rid] = value
        return rid

    def declare(self, path: str, *, event_ids=None, as_json=True) -> str:
        # Registration uses the already verified source index. Bytes are copied
        # into the Diagnosis Package only when the Analyst actually reads them.
        row = self.reader.files[path]
        selector = {'kind': 'json', 'pointer': ''} if as_json else {'kind': 'bytes', 'start': 0, 'end': row['size_bytes']}
        descriptor = {'source_index_sha256': self.reader.index_sha256, 'source_path': path,
                      'source_sha256': row['sha256'], 'size_bytes': row['size_bytes'],
                      'selector': selector, 'event_ids': event_ids or [], 'analyst_visible': True}
        rid = 'ref_' + digest(descriptor)[:24]
        self.declared[rid] = {'ref_id': rid, **descriptor}
        return rid

    def event(self, event: dict, json_pointer: str = '/data') -> str:
        return self.add('trajectory.jsonl', line=event['seq'], json_pointer=json_pointer,
                        event_ids=[event['event_id']])

    def read(self, rid: str):
        if rid not in self.records and rid in self.declared:
            d = self.declared[rid]; sel = d['selector']
            materialized = self.add(d['source_path'], json_pointer='' if sel['kind']=='json' else None,
                                    byte_range=(sel['start'], sel['end']) if sel['kind']=='bytes' else None,
                                    event_ids=d['event_ids'])
            if materialized != rid: raise IntegrityError('Declared evidence ID changed while materializing')
        if rid not in self.records or not self.records[rid]['analyst_visible']:
            raise AccessDenied('Evidence reference unavailable in this analysis view')
        r = self.records[rid]
        # Re-read original immutable bytes rather than trusting an external caller cache.
        data = self.reader.read(r['source_path'])
        value = select(data, r['selector'])
        if digest(value) != r['selected_value_sha256']: raise IntegrityError('Reference value changed')
        return copy.deepcopy(value)
