"""OfficeVal native definitions and official evaluate(directory) entry points.

Scoring lives outside this adapter. Git and dataset identities are independent.
No verifier imports in the controller; hidden rubrics are private files only.
"""
from __future__ import annotations
import re
import subprocess
from pathlib import Path
from ...core.contracts import Asset, Task, EvaluationSpec, ConfigurationError, CapabilityError
from ...storage import confined, read_json, file_digest, regular_file, digest, safe_relative

COM_REQUIRED={'officeval_'+x for x in ('001','008','019','022','023','030','039','074','081')}
PIN='5f3da616ac6a0bc30ca35e9678f8f1eb255bd91a'
DATA_PIN='cd6ba6d8fb83b3fb551e24eebc20e1fb0bd154a5'


def task_name(value:str)->str:
    if not isinstance(value,str) or not re.fullmatch(r'officeval_\d{3}',value) or not 1<=int(value[-3:])<=100:
        raise ConfigurationError('OfficeVal task ID must be officeval_001 through officeval_100')
    return value


def git(source:Path,*args:str)->str:
    try:
        p=subprocess.run(['git','-C',str(source),*args],capture_output=True,text=True,timeout=30,check=True)
        return p.stdout.strip()
    except (subprocess.SubprocessError,OSError) as e:
        raise ConfigurationError('OfficeVal requires a readable local Git checkout at the pinned commit') from e


def _metadata(root:Path,kind:str,task_id:str)->Path:
    for name in (f'{kind}/{task_id}.json',f'data/{kind}/{task_id}.json'):
        p=confined(root,name)
        if p.is_file(): return p
    raise ConfigurationError(f'missing native {kind}/{task_id}.json in data_dir')


class NativeOfficeVal:
    def __init__(self,config:dict):
        self.config=config
        for key in ('source_dir','data_dir','assets_dir'):
            if key not in config: raise ConfigurationError(f'benchmark.{key} required for native OfficeVal')
        self.source=Path(config['source_dir']); self.data=Path(config['data_dir']); self.assets=Path(config['assets_dir'])
    def list_tasks(self):
        base=self.data/'tasks'
        if not base.is_dir(): base=self.data/'data/tasks'
        if not base.is_dir(): raise ConfigurationError('Native task metadata missing; run benchmark fetch or set data_dir')
        return [{'task_id':task_name(p.stem),'supported_runtime':'windows-office-com' if p.stem in COM_REQUIRED else 'docker-static',
                 'assets_ready':self._assets_ready(p)} for p in sorted(base.glob('officeval_*.json'))]
    def _assets_ready(self,path):
        try:
            raw=read_json(path)
            return all(self._asset_path(path.stem,x['dest']).is_file() for x in raw['origin_files'])
        except (OSError,ValueError,KeyError): return False
    def _asset_path(self,task_id,name):
        safe_relative(name)
        if Path(name).name!=name: raise ConfigurationError('OfficeVal dest must be a basename')
        for relative in (f'{task_id}/{name}',f'task_files/{task_id}/{name}'):
            p=confined(self.assets,relative)
            if p.exists(): return p
        return confined(self.assets,f'{task_id}/{name}')
    def load(self,task_id):
        task_name(task_id)
        if task_id in COM_REQUIRED: raise CapabilityError('This OfficeVal verifier requires Windows Office COM; docker-static does not qualify')
        revision=self.config['source_commit']
        if not re.fullmatch('[0-9a-f]{40}',revision): raise ConfigurationError('source_commit must be full Git SHA')
        if git(self.source,'rev-parse','HEAD')!=revision: raise ConfigurationError('OfficeVal source commit mismatch')
        if git(self.source,'status','--porcelain','--untracked-files=all'):
            raise ConfigurationError('OfficeVal source checkout must be clean (including untracked files)')
        task_path=_metadata(self.data,'tasks',task_id); rubric=_metadata(self.data,'rubrics',task_id)
        raw=read_json(task_path)
        if raw.get('id')!=task_id or not isinstance(raw.get('instruction'),str) or not isinstance(raw.get('origin_files'),list):
            raise ConfigurationError('Malformed native OfficeVal task')
        verifier=confined(self.source,f'verifiers/{task_id}_verifier.py'); regular_file(verifier)
        assets=[]; seen=set()
        for entry in raw['origin_files']:
            name=entry['dest']
            if name in seen: raise ConfigurationError('duplicate origin filename')
            seen.add(name)
            p=self._asset_path(task_id,name); regular_file(p)
            # Git LFS pointer is not the task input.
            with p.open('rb') as f:
                if f.read(100).startswith(b'version https://git-lfs.github.com/spec/v1'):
                    raise ConfigurationError('Unresolved LFS asset pointer')
            assets.append(Asset(name,p,file_digest(p),p.stat().st_size))
        records={'task':file_digest(task_path),'rubric':file_digest(rubric),
                 'assets':{a.name:a.sha256 for a in assets}}
        lock_path=Path(self.config.get('data_lock',self.data/'data.lock.json'))
        lock_status='unlocked'
        if lock_path.exists():
            lock=read_json(lock_path)
            if lock.get('dataset_revision')!=self.config['dataset_revision'] or lock.get('tasks',{}).get(task_id)!=records:
                raise ConfigurationError('Dataset lock mismatch (revision, metadata or assets)')
            lock_status='sha256_locked'
        elif not self.config.get('allow_unlocked_data',False):
            raise ConfigurationError('Missing data.lock.json; fetch pinned assets or explicitly lock local data')
        private={'task.json':task_path,'rubric.json':rubric}
        source_paths=git(self.source,'ls-files','-z').split('\x00')
        for rel in source_paths:
            if not rel: continue
            safe_relative(rel); p=confined(self.source,rel); regular_file(p)
            private['source/'+rel]=p
        provenance={'adapter':'officeval-native','source_commit':revision,'source_tree':git(self.source,'rev-parse','HEAD^{tree}'),
                    'dataset_revision':self.config['dataset_revision'],'data_lock_status':lock_status,
                    'data_origin':lock.get('origins',{}).get(task_id,'unknown') if lock_status!='unlocked' else 'user_unlocked_local_data',
                    'task_sha256':records['task'],'rubric_sha256':records['rubric'],
                    'assets_sha256':records['assets'],'verifier_sha256':file_digest(verifier),
                    'private_files_sha256':{k:file_digest(v) for k,v in private.items()},
                    'evaluation_protocol':'official_single_verifier_evaluate',
                    'official_batch_runner_used':False, 'scoring_coverage':'unknown'}
        # Detect mutation across load. Private copies are also hash-checked by the orchestrator.
        if git(self.source,'status','--porcelain','--untracked-files=all') or git(self.source,'rev-parse','HEAD')!=revision:
            raise ConfigurationError('OfficeVal source changed while loading')
        return Task('officeval',task_id,raw['instruction'],tuple(assets),provenance,private,
                    EvaluationSpec('python-verifier',self.source,f'verifiers/{task_id}_verifier.py',task_id,{'normal_mode':True}),
                    {'runtime':'docker-static','com':False,'submission_directory':'submission'})

class BundleOfficeVal:
    """Explicit offline seed/bundle mode, never labelled native upstream checkout."""
    def __init__(self,config):
        self.config=config
        if 'bundle_dir' not in config: raise ConfigurationError('bundle_dir required')
        self.root=Path(config['bundle_dir']); self.manifest=read_json(self.root/'manifest.json')
    def list_tasks(self):
        return [{'task_id':k,'source':'retained_bundle','assets_ready':all(confined(self.root,a['path']).is_file() and file_digest(confined(self.root,a['path']))==a['sha256'] for a in self.manifest['tasks'][k]['input_files'])} for k in sorted(self.manifest['tasks'])]
    def load(self,task_id):
        task_name(task_id)
        if task_id not in self.manifest['tasks']: raise ConfigurationError('task not present in retained bundle')
        # Validate every input used, including private scorer assets.
        for relative,expected in self.manifest['files'].items():
            p=confined(self.root,relative); regular_file(p)
            if file_digest(p)!=expected['sha256'] or p.stat().st_size!=expected['size_bytes']:
                raise ConfigurationError('bundle file digest mismatch: '+relative)
        task=self.manifest['tasks'][task_id]; raw=read_json(self.root/task['task_path'])
        if raw.get('id')!=task_id or raw.get('instruction')!=task['task_prompt']:
            raise ConfigurationError('bundle task projection mismatch')
        assets=tuple(Asset(a['name'],confined(self.root,a['path']),a['sha256'],a['size_bytes']) for a in task['input_files'])
        private={'task.json':self.root/task['task_path'],'rubric.json':self.root/task['rubric_path'],
                 'source/'+task['verifier_path']:self.root/task['verifier_path']}
        for p in (self.root/'source_metadata').iterdir():
            if p.is_file(): private['source/'+p.name]=p
        provenance={'adapter':'officeval-bundle','source_commit':self.manifest['source_commit'],
            'dataset_revision':self.manifest.get('asset_dataset_revision'),'provenance_scope':'retained_bundle_claim',
            'task_sha256':file_digest(private['task.json']),'rubric_sha256':file_digest(private['rubric.json']),
            'assets_sha256':{a.name:a.sha256 for a in assets},'verifier_sha256':file_digest(self.root/task['verifier_path']),
            'private_files_sha256':{k:file_digest(v) for k,v in private.items()},
            'evaluation_protocol':'official_single_verifier_evaluate','official_batch_runner_used':False,'scoring_coverage':'unknown'}
        return Task('officeval',task_id,raw['instruction'],assets,provenance,private,
            EvaluationSpec('python-verifier',self.root,task['verifier_path'],task_id,{'normal_mode':True}),
            {'runtime':'docker-static','com':False,'submission_directory':'submission'})
