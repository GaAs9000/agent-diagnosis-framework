"""Explicit source acquisition/locking; never substitutes main for a missing pin."""
from __future__ import annotations
import json
import os
import re
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path
from ...core.contracts import ConfigurationError
from ...storage import read_json, file_digest, regular_file, confined, utc_now, canonical
from .adapter import task_name, git, _metadata, NativeOfficeVal

REPOSITORY='https://github.com/baidu-frontier-research/OmegaUse-OfficeVal.git'
DATASET='https://huggingface.co/datasets/baidu-frontier-research/OmegaUse-OfficeVal'


def atomic_json(path:Path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_symlink(): raise ConfigurationError('symlink lock path')
    tmp=path.with_name(path.name+'.tmp-'+__import__('uuid').uuid4().hex)
    fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f: f.write(canonical(value)+b'\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)


def lock_local(config:dict,task_ids:list[str],*,origin='user_accepted_local_files')->dict:
    """Local acceptance/hashes, not proof that bytes were downloaded from upstream."""
    adapter=NativeOfficeVal({**config,'allow_unlocked_data':True})
    data=Path(config['data_dir']); path=Path(config.get('data_lock',data/'data.lock.json'))
    lock=read_json(path) if path.exists() else {'schema_version':'0.2.0','dataset_revision':config['dataset_revision'],'tasks':{},'origins':{}}
    if lock['dataset_revision']!=config['dataset_revision']: raise ConfigurationError('existing lock belongs to different dataset revision')
    for tid in task_ids:
        task_name(tid); task=_metadata(data,'tasks',tid); rubric=_metadata(data,'rubrics',tid)
        raw=read_json(task)
        if raw.get('id')!=tid or not isinstance(raw.get('origin_files'),list): raise ConfigurationError('invalid task metadata')
        assets={}
        for x in raw['origin_files']:
            p=adapter._asset_path(tid,x['dest']); regular_file(p)
            if x['dest'] in assets: raise ConfigurationError('duplicate asset name')
            assets[x['dest']]=file_digest(p)
        record={'task':file_digest(task),'rubric':file_digest(rubric),'assets':assets}
        if tid in lock['tasks'] and lock['tasks'][tid]!=record:
            raise ConfigurationError('local bytes changed under an existing lock; use a new explicit data directory/version')
        lock['tasks'][tid]=record; lock['origins'][tid]=origin
    lock['updated_at']=utc_now(); atomic_json(path,lock)
    return {'lock_path':str(path),'lock_sha256':file_digest(path),'tasks':task_ids,'origin':origin}


def _download(url:str,destination:Path,max_bytes:int=268435456):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.is_symlink(): raise ConfigurationError('symlink download destination')
    tmp=destination.with_name(destination.name+'.download-'+__import__('uuid').uuid4().hex)
    total=0
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'l1-source/0.2.0'}),timeout=60) as response:
            fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
            with os.fdopen(fd,'wb') as out:
                while True:
                    chunk=response.read(1024*1024)
                    if not chunk: break
                    total+=len(chunk)
                    if total>max_bytes: raise ConfigurationError('asset download exceeds max bytes')
                    out.write(chunk)
                out.flush(); os.fsync(out.fileno())
        if not total: raise ConfigurationError('empty upstream asset')
        if destination.exists():
            if file_digest(destination)!=file_digest(tmp): raise ConfigurationError('download conflicts with existing local asset; refusing overwrite')
            tmp.unlink()
        else: os.replace(tmp,destination)
    except BaseException:
        tmp.unlink(missing_ok=True); raise


def fetch(config:dict,task_ids:list[str])->dict:
    for tid in task_ids: task_name(tid)
    source=Path(config['source_dir']); revision=config['source_commit']; data_revision=config['dataset_revision']
    if any(not re.fullmatch('[0-9a-f]{40}',x) for x in (revision,data_revision)):
        raise ConfigurationError('Both source_commit and dataset_revision must be full immutable Git SHAs')
    if not source.exists():
        source.parent.mkdir(parents=True,exist_ok=True)
        tmp=source.with_name(source.name+'.fetch-'+__import__('uuid').uuid4().hex)
        try:
            subprocess.run(['git','clone','--no-checkout',REPOSITORY,str(tmp)],check=True,timeout=300)
            subprocess.run(['git','-C',str(tmp),'checkout','--detach',revision],check=True,timeout=60)
            os.rename(tmp,source)
        except BaseException:
            import shutil; shutil.rmtree(tmp,ignore_errors=True); raise
    if git(source,'rev-parse','HEAD')!=revision or git(source,'status','--porcelain','--untracked-files=all'):
        raise ConfigurationError('source checkout is dirty or not at requested commit; no auto-reset is performed')
    data=Path(config['data_dir']); assets=Path(config['assets_dir'])
    base=f'{DATASET}/resolve/{data_revision}'
    for tid in task_ids:
        for kind in ('tasks','rubrics'):
            _download(f'{base}/{kind}/{tid}.json',confined(data,f'{kind}/{tid}.json'))
        raw=read_json(data/f'tasks/{tid}.json')
        if raw.get('id')!=tid: raise ConfigurationError('upstream task ID mismatch')
        for entry in raw['origin_files']:
            name=entry['dest']
            if Path(name).name!=name: raise ConfigurationError('unsafe upstream asset destination')
            target=confined(assets,f'{tid}/{name}')
            _download(f'{base}/task_files/{tid}/{urllib.parse.quote(name,safe="")}',target)
    result=lock_local(config,task_ids,origin='downloaded_from_explicit_pinned_huggingface_revision')
    return {**result,'source_commit':revision,'dataset_revision':data_revision,'silent_version_fallback':False}
