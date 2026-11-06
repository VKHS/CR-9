"""Immutable archives and DOI bindings. No archive code is executed on ingestion."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from urllib.parse import urlparse, quote
from urllib.request import Request, urlopen
import zipfile
from .common import (EvidenceError, require, relative, confined, read_json, loads, canonical,
                     sha256, file_hash, valid_hash, directory_inventory, write_json, integer)
from .contract import DATASETS, RECORD_SETS, RUN_ROLES, SET_ROLES

DESCRIPTOR = 'evidence.json'
SCHEMA = 'cross-morse-evidence-archive-1'
MAX_JSON = 16 * 1024 * 1024
ZENODO = re.compile(r'^10\.5281/zenodo\.([1-9][0-9]*)$')

def validate_descriptor(obj: dict) -> None:
    require(isinstance(obj, dict) and obj.get('schema') == SCHEMA, 'Unsupported evidence descriptor')
    require(obj.get('origin') == 'original_experiment', 'Only original_experiment deposits can enter benchmark records')
    require(obj.get('kind') in ('run','record_set'), 'kind must be run or record_set')
    require(isinstance(obj.get('record_id'), str) and re.fullmatch(r'[A-Za-z0-9_-]+', obj['record_id']), 'Invalid record_id')
    datasets = obj.get('datasets')
    declared_dataset_scope = (obj['kind']=='record_set'
        and RECORD_SETS.get(obj['record_id'], {}).get('dataset_scope')=='declared')
    require(isinstance(datasets,list) and (datasets or declared_dataset_scope) and len(datasets)==len(set(datasets))
            and set(datasets)<=set(DATASETS), 'Only exact official benchmark names are accepted')
    source = obj.get('source', {})
    require(isinstance(source,dict) and valid_hash(source.get('tree_sha256')), 'Missing original source tree SHA-256')
    commit = source.get('git_commit')
    require(valid_hash(commit,40) or valid_hash(commit,64), 'Original source commit must be a complete Git object ID')
    url = urlparse(source.get('repository',''))
    require(url.scheme=='https' and url.hostname and url.hostname not in ('example.org','example.com'), 'A real HTTPS source repository is required')
    roles = obj.get('roles')
    require(isinstance(roles,dict), 'Artifact role map required')
    for role, paths in roles.items():
        require(isinstance(role,str) and isinstance(paths,list) and paths, f'Empty role: {role}')
        for p in paths:
            relative(p)
    required = RUN_ROLES if obj['kind']=='run' else SET_ROLES
    require(set(required)<=set(roles), f'Missing roles: {sorted(set(required)-set(roles))}')
    if obj['kind']=='run':
        require(len(datasets)==1 and obj['record_id']==datasets[0], 'One official dataset per run archive')
        require(obj.get('layout')=='sealed_eight_stage', 'Run layout requires the sealed eight-stage adapter; unsupported layouts need an explicit adapter')
        run_paths = obj.get('runs')
        require(isinstance(run_paths,list) and run_paths, 'run archive must declare run directories')
        for p in run_paths:
            relative(p)
        if datasets==['Roman-empire']:
            require(len(run_paths)==10,'Roman-empire archive must contain all ten split runs')
        require(isinstance(obj.get('execution_contract'),str), 'execution_contract JSON path required')
    else:
        name = obj['record_id']
        require(name in RECORD_SETS, f'Unknown record set: {name}')
        # S19 is four control constructions, potentially on independent realizations;
        # dataset provenance is optional there. Its control/quantity coverage is
        # checked from the sealed records by scientific_evidence_audit.
        if not declared_dataset_scope:
            require(set(datasets)==set(RECORD_SETS[name]['datasets']), f'{name}: incomplete dataset coverage')
        require(isinstance(obj.get('study'),str), 'Record set must identify its study.json')
    for key in ('execution_contract','study'):
        if key in obj:
            relative(obj[key])
    require(isinstance(obj.get('files'),list) and obj['files'], 'Missing byte inventory')
    paths=set()
    for row in obj['files']:
        require(isinstance(row,dict) and set(row)=={'path','sha256','bytes'}, 'Invalid inventory row')
        relative(row['path'])
        require(row['path'] != DESCRIPTOR and row['path'] not in paths, 'Duplicate/self-referential inventory path')
        require(valid_hash(row['sha256']) and type(row['bytes']) is int and row['bytes']>=0, 'Invalid hash or byte count')
        paths.add(row['path'])
    for role, members in roles.items():
        require(set(members)<=paths, f'Role {role} references an unsealed file')
    for key in ('execution_contract','study'):
        if key in obj:
            require(obj[key] in paths, f'{key} is not sealed')

def verify_directory(root: Path) -> dict:
    obj = read_json(confined(root,DESCRIPTOR))
    validate_descriptor(obj)
    actual = directory_inventory(root, exclude=(DESCRIPTOR,))
    require(actual==sorted(obj['files'],key=lambda row:row['path']), 'Added, missing or altered evidence file')
    return obj

def seal(source: Path, descriptor: Path, output: Path) -> dict:
    """Build a ZIP from original files plus a sidecar descriptor; preserve inputs."""
    source=source.resolve();output=output.absolute()
    require(source.is_dir(), 'Source folder missing')
    require(not output.resolve().is_relative_to(source) and not source.is_relative_to(output.resolve()),
            'Source/output paths overlap')
    require(not output.exists(), 'Output already exists; choose a distinct archive name')
    require(not (source/DESCRIPTOR).exists(), 'Keep evidence.json descriptor outside source; source contains reserved filename')
    obj=read_json(descriptor)
    obj['files']=directory_inventory(source)
    validate_descriptor(obj)
    output.parent.mkdir(parents=True,exist_ok=True)
    fd, name=tempfile.mkstemp(prefix='.cm-pack-',suffix='.zip',dir=output.parent);os.close(fd)
    temp=Path(name)
    try:
        with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
            for row in obj['files']:
                p=confined(source,row['path'])
                info=zipfile.ZipInfo(row['path'],date_time=(1980,1,1,0,0,0))
                info.compress_type=zipfile.ZIP_DEFLATED
                info.external_attr=(stat.S_IFREG|0o644)<<16
                with p.open('rb') as inp, z.open(info,'w',force_zip64=True) as out:
                    shutil.copyfileobj(inp,out,4*1024*1024)
            info=zipfile.ZipInfo(DESCRIPTOR,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=(stat.S_IFREG|0o644)<<16
            z.writestr(info,canonical(obj))
        require(directory_inventory(source)==obj['files'], 'Source changed during packaging')
        verify_zip(temp)
        os.link(temp,output)
    finally:
        temp.unlink(missing_ok=True)
    return {'archive':str(output),'bytes':output.stat().st_size,'sha256':sha256(output),'record_id':obj['record_id']}

def verify_zip(archive: Path, *, max_bytes: int = 200*1024**3, max_files: int = 2000000) -> dict:
    """Streaming validation prevents path traversal and unexpected artifacts."""
    with zipfile.ZipFile(archive) as z:
        infos=z.infolist()
        require(0<len(infos)<=max_files, 'Archive entry count exceeds limit or is empty')
        require(sum(x.file_size for x in infos)<=max_bytes, 'Archive expansion exceeds configured bound')
        names=set(); by_name={}
        for item in infos:
            name=item.filename.rstrip('/') if item.is_dir() else item.filename
            relative(name)
            require(name not in names, f'Duplicate ZIP entry: {name}');names.add(name)
            mode=(item.external_attr>>16)&0xFFFF
            require(not stat.S_ISLNK(mode), f'Symlink ZIP entry: {name}')
            require(not (item.flag_bits&1), 'Encrypted evidence archives are unsupported')
            if not item.is_dir():
                require(stat.S_IFMT(mode) in (0,stat.S_IFREG), f'Non-regular ZIP entry: {name}')
                by_name[name]=item
        require(DESCRIPTOR in by_name and by_name[DESCRIPTOR].file_size<=MAX_JSON, 'Missing or oversized evidence.json')
        obj=loads(z.read(DESCRIPTOR).decode('utf-8'));validate_descriptor(obj)
        required={r['path'] for r in obj['files']}|{DESCRIPTOR}
        require(set(by_name)==required, 'ZIP entries do not equal sealed inventory')
        for row in obj['files']:
            require(by_name[row['path']].file_size==row['bytes'],f'Byte count mismatch: {row["path"]}')
            h=hashlib.sha256()
            with z.open(row['path']) as inp:
                for part in iter(lambda:inp.read(4*1024*1024),b''):h.update(part)
            require(h.hexdigest()==row['sha256'],f'SHA-256 mismatch: {row["path"]}')
    return obj

def extract(archive: Path, destination: Path) -> dict:
    require(not destination.exists(), 'Evidence extraction is immutable; destination exists')
    obj=verify_zip(archive)
    destination.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix='.cm-unpack-',dir=destination.parent))
    try:
        with zipfile.ZipFile(archive) as z:
            for row in obj['files']+[{'path':DESCRIPTOR}]:
                out=confined(temp,row['path'],exists=False);out.parent.mkdir(parents=True,exist_ok=True)
                with z.open(row['path']) as inp,out.open('xb') as stream:
                    shutil.copyfileobj(inp,stream,4*1024*1024)
        verify_directory(temp)
        require(not destination.exists(),'Destination appeared during extraction')
        # mkdir reserves the name before moving files; never replaces another tree.
        destination.mkdir()
        try:
            for p in temp.iterdir():shutil.move(str(p),str(destination/p.name))
        except BaseException:
            shutil.rmtree(destination)
            raise
    finally:
        shutil.rmtree(temp,ignore_errors=True)
    return obj

def zenodo_id(doi: str) -> str:
    doi=doi.removeprefix('https://doi.org/')
    match=ZENODO.fullmatch(doi)
    require(match is not None, 'Automatic DOI-to-byte binding currently supports published Zenodo version DOIs only')
    return match.group(1)

def get_metadata(doi: str) -> dict:
    record=zenodo_id(doi)
    req=Request(f'https://zenodo.org/api/records/{record}',headers={'User-Agent':'CROSS-MORSE-evidence/1','Accept':'application/json'})
    with urlopen(req,timeout=60) as response:
        require(urlparse(response.url).hostname=='zenodo.org','Unexpected metadata redirect')
        payload=response.read(MAX_JSON+1)
    require(len(payload)<=MAX_JSON,'Metadata response too large')
    data=loads(payload.decode())
    require(str(data.get('id'))==record,'DOI points to a concept/latest record, not the requested immutable version')
    require(data.get('doi')==f'10.5281/zenodo.{record}','Published record DOI mismatch')
    require(data.get('is_published',True) is True and data.get('metadata',{}).get('publication_date'), 'Record is not published')
    return data

def asset_metadata(doi: str, filename: str, data: dict) -> dict:
    record=zenodo_id(doi)
    require(data.get('doi')==f'10.5281/zenodo.{record}' and str(data.get('id'))==record,'Record identity mismatch')
    require('/' not in filename and '\\' not in filename and filename not in ('','.','..'),'Asset filename must be a basename')
    entries=data.get('files',[])
    if isinstance(entries,dict):entries=list(entries.get('entries',{}).values())
    matches=[x for x in entries if x.get('key')==filename]
    require(len(matches)==1,'Exact archive filename is absent or ambiguous in the published DOI')
    f=matches[0]; checksum=f.get('checksum','')
    require(':' in checksum and checksum.split(':',1)[0] in ('md5','sha256'), 'Unsupported/missing publisher checksum')
    size=integer(f.get('size'),'published archive size',minimum=1)
    return {'doi':f'10.5281/zenodo.{record}','filename':filename,'bytes':size,'publisher_checksum':checksum,
            'download_url':f'https://zenodo.org/api/records/{record}/files/{quote(filename,safe="")}/content',
            'metadata_sha256':hashlib.sha256(canonical(data)).hexdigest()}

def bind_local(archive: Path, doi: str, filename: str, *, metadata: dict | None = None) -> dict:
    """Compare published filename/size/checksum with actual local bytes."""
    data=get_metadata(doi) if metadata is None else metadata
    binding=asset_metadata(doi,filename,data)
    require(archive.stat().st_size==binding['bytes'],'Local archive size differs from DOI asset')
    alg,digest=binding['publisher_checksum'].split(':',1)
    require(file_hash(archive,alg)==digest,'Local bytes differ from the DOI asset checksum')
    binding['sha256']=sha256(archive)
    return binding

def download(binding: dict, target: Path, *, max_bytes: int=200*1024**3) -> None:
    require(not target.exists(),'Download destination exists')
    meta=get_metadata(binding['doi']); actual=asset_metadata(binding['doi'],binding['filename'],meta)
    require(actual['bytes']==binding['bytes'] and actual['publisher_checksum']==binding['publisher_checksum'], 'Published asset changed')
    require(actual['bytes']<=max_bytes,'Download exceeds configured size limit')
    target.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.cm-download-',dir=target.parent);os.close(fd);temp=Path(name)
    try:
        req=Request(actual['download_url'],headers={'User-Agent':'CROSS-MORSE-evidence/1'})
        with urlopen(req,timeout=120) as inp,temp.open('wb') as out:
            require(urlparse(inp.url).scheme=='https' and urlparse(inp.url).hostname=='zenodo.org','Unexpected download redirect')
            count=0
            while True:
                chunk=inp.read(4*1024*1024)
                if not chunk:break
                count+=len(chunk);require(count<=actual['bytes'],'Download longer than declared asset');out.write(chunk)
        verify=bind_local(temp,binding['doi'],binding['filename'],metadata=meta)
        require(verify['sha256']==binding['sha256'],'Downloaded SHA-256 differs from repository pin')
        os.link(temp,target)
    finally:temp.unlink(missing_ok=True)
