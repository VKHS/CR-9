"""Immutable arm-experiment output and source/configuration identities."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write_json(path,value):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    fd,temporary=tempfile.mkstemp(dir=path.parent,suffix='.tmp')
    try:
        with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.link(temporary,path)
    finally:Path(temporary).unlink(missing_ok=True)


def fresh_output(path,source):
    p=Path(path).expanduser().absolute();source=Path(source).resolve()
    if any(q.is_symlink() for q in (p,*p.parents)):
        raise ValueError('Output path must not traverse symbolic links')
    p=p.resolve()
    if p==source or p.is_relative_to(source) or source.is_relative_to(p):
        raise ValueError('Use an output outside the immutable input run')
    p.mkdir(parents=True,exist_ok=False)
    return p


def source_identity(root):
    root=Path(root)
    paths=[*root.glob('cm_experiments/*.py'),root/'scripts/run_budget_arms.py']
    return {'files':{p.relative_to(root).as_posix():sha256(p) for p in sorted(paths)},
            'contract':'arm source plus unchanged frozen-plant runtime identity in experiment.json'}


def seal(root):
    root=Path(root)
    entries=[]
    for p in sorted(root.rglob('*')):
        if p.is_symlink():raise ValueError('Cannot seal symbolic links')
        if p.is_file():
            entries.append({'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':sha256(p)})
    write_json(root/'ARTIFACT_MANIFEST.json',{'schema':'cm-arm-artifacts-1','files':entries})


def verify(root):
    root=Path(root)
    obj=json.loads((root/'ARTIFACT_MANIFEST.json').read_text())
    if obj.get('schema')!='cm-arm-artifacts-1':raise ValueError('Unrecognized arm artifact manifest')
    expected=set()
    if not isinstance(obj.get('files'),list) or not obj['files']:
        raise ValueError('Nonempty artifact inventory required')
    if any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('Symbolic links are not evidence artifacts')
    for item in obj['files']:
        name=item['path'];p=root/name
        if Path(name).is_absolute() or '..' in Path(name).parts or p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):
            raise ValueError('Unsafe artifact path')
        if name in expected or not p.is_file() or p.stat().st_size!=item['bytes'] or sha256(p)!=item['sha256']:
            raise ValueError(f'Artifact mismatch: {name}')
        expected.add(name)
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.relative_to(root).as_posix()!='ARTIFACT_MANIFEST.json'}
    if actual!=expected:raise ValueError('Unsealed extra or missing arm artifacts')
    return {'passed':True,'files':len(expected),'schema':obj['schema']}
