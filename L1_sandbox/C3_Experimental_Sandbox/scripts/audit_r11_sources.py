"""Public-asset eligibility before the candidate freeze; no model or score access."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from lxml import etree
from c3_sandbox.evidence._storage import encode
from c3_sandbox.r1.docker_runtime import DockerRuntime
from c3_sandbox.r1.qualification import image_identity

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/e2-r1-1'
OLD = ROOT.parent / 'Agent_Latent_Capability_Audit_Kit/data/cache/officeval_assets'
NEW = RUN / 'asset-downloads/task_files'
IDS = ['officeval_' + value for value in ('012', '015', '018', '021', '031', '033', '036', '037')]


def fingerprint(path):
    body = {'path': str(path), 'task_id': path.parent.name, 'name': path.name,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'size_bytes': path.stat().st_size}
    text = ''
    if path.suffix.lower() == '.docx':
        with zipfile.ZipFile(path) as archive:
            parser = etree.XMLParser(resolve_entities=False, no_network=True)
            for name, field in [('word/document.xml', 'body_c14n'), ('word/styles.xml', 'styles_c14n')]:
                info = archive.getinfo(name)
                if info.file_size > 8 * 1024 * 1024:
                    raise ValueError('XML exceeds source-audit limit')
                node = etree.fromstring(archive.read(name), parser)
                body[field] = hashlib.sha256(etree.tostring(node, method='c14n')).hexdigest()
                if field == 'body_c14n':
                    text = ''.join(''.join(node.itertext()).split())
                    # Comparison uses Word text nodes, not namespace/formatting strings.
                    text = ''.join(''.join(node.xpath('//*[local-name()="t"]/text()')).split())
            body['media'] = sorted(hashlib.sha256(archive.read(name)).hexdigest()
                                   for name in archive.namelist() if name.startswith('word/media/') and not name.endswith('/'))
            body['text_characters'] = len(text)
    return body, text


def main():
    folder = RUN / 'source_audit-v2'
    folder.mkdir()
    historical = [fingerprint(path) for task in ('002','003','004','005','006','007')
                  for path in sorted((OLD / ('officeval_' + task)).iterdir()) if path.is_file()]
    downloaded = [fingerprint(path) for task in ['officeval_010'] + IDS
                  for path in sorted((NEW / task).iterdir()) if path.is_file()]
    pairs = []
    for new, newtext in downloaded:
        for old, oldtext in historical:
            evidence = {'new': new['path'], 'historical': old['path'], 'same_file': new['sha256'] == old['sha256']}
            if newtext and oldtext:
                matcher = difflib.SequenceMatcher(None, newtext, oldtext, autojunk=False)
                evidence.update(similarity=matcher.ratio(), longest_shared_run=matcher.find_longest_match().size,
                    same_body=new['body_c14n'] == old['body_c14n'], same_styles=new['styles_c14n'] == old['styles_c14n'],
                    shared_media=sorted(set(new['media']) & set(old['media'])))
            pairs.append(evidence)
    suspect = [pair for pair in pairs if pair['same_file'] or pair.get('same_body') or
               pair.get('longest_shared_run', 0) >= 100]
    if suspect:
        raise RuntimeError('potential historical source/template overlap; refuse candidate freeze')
    image_only = next(item for item, text in downloaded if item['task_id'] == 'officeval_010')
    assert image_only['text_characters'] == 0 and len(image_only['media']) == 1
    staging = folder / 'pdf-input'
    staging.mkdir(mode=0o700)
    for item, _ in downloaded:
        if item['name'].lower().endswith('.pdf') and item['task_id'] in IDS:
            destination = staging / item['task_id']
            destination.mkdir(mode=0o700)
            shutil.copyfile(item['path'], destination / item['name'])
    code = """import fitz,json,hashlib
from pathlib import Path
out=[]
for p in sorted(Path('/workspace').rglob('*.pdf')):
 with fitz.open(p) as doc:
  counts=[len(page.get_text()) for page in doc]
 out.append({'task_id':p.parent.name,'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'pages':len(counts),'page_text_characters':counts})
print(json.dumps(out))
"""
    with DockerRuntime(image_identity('c3-r1-agent:locked'), folder / 'pdf-runtime') as actor:
        actor.seed_workspace(staging)
        process = actor.exec_fixed(['python', '-I', '-B', '-c', code])
        assert process['exit_code'] == 0 and not process['truncated'] and not process['error']
        # PyMuPDF may print native diagnostic lines before the controller's final
        # JSON line. Retain the entire raw process record and parse only that line.
        pdfs = json.loads(process['stdout'].splitlines()[-1])
        assert all(item['pages'] and all(count > 0 for count in item['page_text_characters']) for item in pdfs)
        unchanged = actor.verify_workspace(staging)
    review = {'retained_task_ids': IDS, 'excluded_before_any_model_trial': {'officeval_010': 'image-only flowchart requires unavailable visual interpretation'},
        'known_exposed_task_ids': ['officeval_' + value for value in ('002','003','004','005','006','007')],
        'historical_assets': [item for item, text in historical], 'downloaded_assets': [item for item, text in downloaded],
        'comparisons': pairs, 'historical_overlap_detected': False, 'pdf_text_access': pdfs,
        'pdf_inspection_process': process, 'pdf_inputs_unchanged': unchanged,
        'source_group_policy': 'quarantine each task and any reused public asset/template; reject whole-file/canonical-body matches or >=100-character exact normalized text reuse with exposed assets',
        'limitations': 'no detected reuse under recorded checks; undocumented source-family independence and visual understanding are not established',
        'no_candidate_model_calls_or_scores_used': True}
    with (folder / 'source_review.json').open('xb') as stream:
        stream.write(encode(review))
    print(json.dumps({'eligible_tasks': IDS, 'historical_assets': len(historical),
                      'source_comparisons': len(pairs), 'overlap_detected': False, 'pdf_text_access': pdfs}))


if __name__ == '__main__':
    main()
