"""Lossless query views of sealed E1 JSON/JSONL, CPU only, no rescoring."""
import hashlib
import json
from pathlib import Path
import sys
import pyarrow as pa
import pyarrow.parquet as pq

pa.set_cpu_count(4)
pa.set_io_thread_count(2)
diagnoses, output = map(Path, sys.argv[1:])
output.mkdir(parents=True, exist_ok=False)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
checkpoints, terminals, defects, sources = [], [], [], {}
for directory in sorted(diagnoses.iterdir()):
    if not directory.is_dir():
        continue
    grade_file, defect_file = directory / 'grades.jsonl', directory / 'defects.json'
    sources[directory.name] = {'grades_sha256': sha(grade_file), 'defects_sha256': sha(defect_file)}
    for line in grade_file.read_text().splitlines():
        value = json.loads(line)
        row = {'slot': directory.name, 'episode_id': value['episode_id'], 'task_id': value['task_id'],
               'kind': value['kind'], 'k': value['k'], 'checkpoint_id': value.get('checkpoint_id'),
               'snapshot_id': value.get('snapshot_id'), 'score': value['score'],
               'dim1_pass': value['dim1_pass'], 'total_score': float(value['total_score']),
               'max_score': float(value['max_score']),
               'synthetic_terminal_checkpoint': value.get('synthetic_terminal_checkpoint', False),
               'anchor_eligible': value.get('anchor_eligible', False),
               'episode_index_sha256': value['episode_index_sha256'],
               'source_file_sha256': sources[directory.name]['grades_sha256'], 'row_json': line}
        (terminals if value['kind'] == 'terminal_submission' else checkpoints).append(row)
    table = json.loads(defect_file.read_text())
    for ordinal, defect in enumerate(table['defects']):
        defects.append({'slot': directory.name, 'episode_id': table['episode_id'],
                        'task_id': table['task_id'], 'defect_ordinal': ordinal,
                        'defect': defect['defect'], 'kind': defect['kind'],
                        'deduction': defect['deduction'], 'k_star': table['k_star'],
                        'episode_index_sha256': table['episode_index_sha256'],
                        'source_file_sha256': sources[directory.name]['defects_sha256'],
                        'row_json': json.dumps(defect, ensure_ascii=False, sort_keys=True, separators=(',', ':'))})
receipt = {'schema': 'e1-lossless-parquet-views-v1', 'pyarrow_version': pa.__version__,
           'actor_requests': 0, 'analyst_requests': 0, 'grader_calls': 0,
           'source_files': sources, 'outputs': {},
           'note': 'JSON/JSONL remains canonical. row_json retains every source field; scalar columns are query views.'}
for name, rows in [('checkpoint_grades', checkpoints), ('terminal_grades', terminals), ('defect_table', defects)]:
    path = output / (name + '.parquet')
    pq.write_table(pa.Table.from_pylist(rows), path, compression='zstd')
    restored = pq.read_table(path).to_pylist()
    assert restored == rows, name + ' roundtrip mismatch'
    receipt['outputs'][name] = {'rows': len(rows), 'sha256': sha(path), 'roundtrip_equal': True}
(output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(receipt['outputs']))
