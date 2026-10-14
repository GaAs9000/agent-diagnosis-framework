"""UTF-8 portable pack/unpack with explicit limits and roundtrip verification."""
from __future__ import annotations
import hashlib
from pathlib import Path, PurePosixPath
import stat
import tempfile
import zipfile
from .io import L2Error,IntegrityError,relative,safe,write_new,sha_file


def unpack(archive:Path,output:Path,*,repair_utf8_names=False,max_total_bytes=2*1024**3,max_files=30000):
    if output.exists():raise L2Error('Unpack output must not exist')
    plan=[];seen=set();total=0;repairs=[]
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            name=info.filename
            if repair_utf8_names and not info.flag_bits&0x800:
                try:name=name.encode('cp437').decode('utf8')
                except UnicodeError:pass
            if name.startswith('__MACOSX/') or name.endswith('/.DS_Store') or name=='.DS_Store':continue
            stripped=name.rstrip('/')
            relative(stripped)
            if stat.S_ISLNK(info.external_attr>>16) or info.flag_bits&1:raise IntegrityError('Symlink/encrypted archive member rejected')
            if stripped in seen:raise IntegrityError('Duplicate archive path after decoding')
            seen.add(stripped)
            if name!=info.filename:repairs.append({'old':info.filename,'new':name})
            total+=info.file_size
            if total>max_total_bytes or len(seen)>max_files:raise IntegrityError('Archive expansion limit')
            plan.append((info,name))
        output.mkdir(parents=True,exist_ok=False,mode=0o700)
        for info,name in plan:
            if info.is_dir():safe(output,name.rstrip('/')).mkdir(parents=True,exist_ok=True,mode=0o700)
            else:
                # zipfile verifies CRC, and size caps were checked before extraction.
                data=z.read(info)
                if len(data)!=info.file_size:raise IntegrityError('Archive member size differs')
                write_new(output,name,data)
    return {'output':str(output),'files_and_directories':len(plan),'uncompressed_bytes':total,
            'filename_repairs':repairs,'contents_rewritten':False}


def pack(root:Path,output:Path):
    from .package import verify
    check=verify(root)
    if not check['verified']:raise L2Error('Refusing to pack unverified L2 package')
    if output.exists() or output.resolve().is_relative_to(root.resolve()):raise L2Error('New ZIP path outside package required')
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file():z.write(p,root.name+'/'+p.relative_to(root).as_posix())
    with tempfile.TemporaryDirectory(prefix='l2-roundtrip-') as td:
        with zipfile.ZipFile(output) as z:z.extractall(td)
        copied=Path(td).resolve()/root.name
        after=verify(copied,expected_index=check['index_sha256'])
        if not after['verified']:raise IntegrityError('Default ZIP roundtrip did not verify')
    return {'zip':str(output),'sha256':sha_file(output),'default_extract_roundtrip_verified':True,
            'package_index_sha256':check['index_sha256']}
