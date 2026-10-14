"""File-level states captured only at quiescent cold-tool boundaries.

No symlinks, hard links, devices, sockets or FIFOs cross the evidence boundary.
The runtime must stop ALL processes before capture. This is not process/VM memory.
"""
from __future__ import annotations
import os
import stat
from pathlib import Path
from ..storage import canonical, digest, file_digest, confined, regular_file, write_bytes, write_json, read_json, safe_relative
from ..core.contracts import EvidenceError

class StateStore:
    def __init__(self,root: Path,max_bytes:int=268435456,max_files:int=10000):
        self.root=root; self.max_bytes=max_bytes; self.max_files=max_files
        self.snapshots=[]
    def capture(self,workspace: Path,label:str) -> dict:
        if workspace.is_symlink() or not workspace.is_dir(): raise EvidenceError('invalid workspace')
        files=[]; directories=[]; total=0
        for path in sorted(workspace.rglob('*')):
            rel=path.relative_to(workspace).as_posix(); safe_relative(rel)
            s=path.lstat()
            if stat.S_ISDIR(s.st_mode):
                directories.append({'path':rel,'mode':stat.S_IMODE(s.st_mode)})
                if len(directories)>self.max_files: raise EvidenceError('directory_count_limit')
                continue
            try: regular_file(path)
            except ValueError as exc: raise EvidenceError(f'unsafe state entry: {rel}') from exc
            total+=s.st_size
            if total>self.max_bytes or len(files)>=self.max_files: raise EvidenceError('workspace_capture_limit')
            fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
            with os.fdopen(fd,'rb') as f:
                data=f.read(min(s.st_size+1,self.max_bytes+1))
                after=os.fstat(f.fileno())
            if len(data)!=s.st_size or (s.st_ino,s.st_mtime_ns,s.st_size)!=(after.st_ino,after.st_mtime_ns,after.st_size):
                raise EvidenceError('workspace_changed_during_capture')
            import hashlib
            sha=hashlib.sha256(data).hexdigest(); blob=f'states/blobs/{sha}'
            if not (self.root/blob).exists(): write_bytes(self.root,blob,data)
            elif file_digest(self.root/blob)!=sha: raise EvidenceError('corrupt blob store')
            files.append({'path':rel,'sha256':sha,'size_bytes':len(data), 'mode':stat.S_IMODE(s.st_mode),'blob':blob})
        sid=f's{len(self.snapshots):06d}'
        state={'snapshot_id':sid,'label':label,'scope':'workspace_files_and_directories',
               'files':files,'directories':directories,'total_bytes':total,
               'content_sha256':digest({'files':files,'directories':directories})}
        write_json(self.root,f'states/snapshots/{sid}.json',state)
        self.snapshots.append(state)
        return state
    @staticmethod
    def diff(before:dict,after:dict)->list[dict]:
        b={x['path']:x for x in before['files']}; a={x['path']:x for x in after['files']}
        out=[]
        for path in sorted(set(a)|set(b)):
            if a.get(path)==b.get(path): continue
            out.append({'path':path,'change':'created' if path not in b else 'deleted' if path not in a else 'modified',
                        'before':b.get(path),'after':a.get(path)})
        bd={x['path']:x for x in before['directories']}; ad={x['path']:x for x in after['directories']}
        for path in sorted(set(ad)|set(bd)):
            if bd.get(path)!=ad.get(path): out.append({'path':path,'change':'directory_changed','before':bd.get(path),'after':ad.get(path)})
        return out


def restore(root: Path,snapshot:dict,destination:Path)->None:
    if any(item['mode'] & 0o7000 for item in snapshot['files'] + snapshot['directories']):
        raise EvidenceError('special permission bits are outside the portable restore profile')
    if destination.exists() and (destination.is_symlink() or any(destination.iterdir())):
        raise EvidenceError('restore destination must be absent or empty')
    destination.mkdir(parents=True,exist_ok=True,mode=0o700)
    paths=set()
    for d in sorted(snapshot['directories'],key=lambda d:len(Path(d['path']).parts)):
        p=confined(destination,d['path']); p.mkdir(parents=True,exist_ok=True)
    for item in snapshot['files']:
        if item['path'] in paths: raise EvidenceError('duplicate snapshot path')
        paths.add(item['path'])
        source=confined(root,item['blob']); regular_file(source)
        if file_digest(source)!=item['sha256'] or source.stat().st_size!=item['size_bytes']:
            raise EvidenceError('snapshot blob mismatch')
        p=write_bytes(destination,item['path'],source.read_bytes())
        os.chmod(p,item['mode'] & 0o777) # never restore setuid/setgid
    for d in sorted(snapshot['directories'],key=lambda d:-len(Path(d['path']).parts)):
        os.chmod(confined(destination,d['path']),d['mode'] & 0o777)
