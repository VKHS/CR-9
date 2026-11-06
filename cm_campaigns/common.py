"""Strict file parsing and non-destructive writes for experimental evidence."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any

class EvidenceError(ValueError):
    """An evidence requirement could not be verified."""

def require(condition: Any, message: str) -> None:
    if not condition:
        raise EvidenceError(message)

def sha256(path: Path) -> str:
    return file_hash(path, 'sha256')

def file_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()

def valid_hash(value: Any, length: int = 64) -> bool:
    return isinstance(value, str) and len(value) == length and all(c in '0123456789abcdef' for c in value)

def relative(value: str) -> PurePosixPath:
    require(isinstance(value, str) and bool(value), 'Empty/non-string artifact path')
    p = PurePosixPath(value)
    require(bool(p.parts) and not p.is_absolute() and p.as_posix() == value and '..' not in p.parts
            and '.' not in p.parts and '\\' not in value and ':' not in value
            and not any(ord(c) < 32 for c in value), f'Unsafe artifact path: {value!r}')
    return p

def confined(root: Path, value: str, *, exists: bool = True) -> Path:
    p = relative(value)
    root = root.resolve()
    out = root.joinpath(*p.parts)
    cursor = root
    for part in p.parts:
        cursor = cursor / part
        require(not cursor.is_symlink(), f'Symlink not permitted: {value}')
    require(out.resolve().is_relative_to(root), f'Path escapes evidence root: {value}')
    if exists:
        require(out.is_file(), f'Missing regular artifact: {value}')
    return out

def _pairs(pairs):
    obj = {}
    for key, value in pairs:
        require(key not in obj, f'Duplicate JSON key: {key}')
        obj[key] = value
    return obj

def _bad_constant(value):
    raise EvidenceError(f'Nonfinite JSON number: {value}')

def loads(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=_bad_constant)

def read_json(path: Path) -> Any:
    return loads(path.read_text(encoding='utf-8'))

def canonical(obj: Any) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()

def write_json(path: Path, obj: Any, *, replace: bool = False) -> None:
    write_bytes(path, canonical(obj), replace=replace)

def write_bytes(path: Path, data: bytes, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.is_symlink(), f'Refusing symlink destination: {path}')
    fd, name = tempfile.mkstemp(prefix='.cm-evidence-', dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temp, path)
        else:
            os.link(temp, path)  # atomic, no-clobber publication
    finally:
        temp.unlink(missing_ok=True)

def rows(path: Path, required: tuple[str, ...] = ()) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames or []
        require(len(names) == len(set(names)) and set(required) <= set(names),
                f'{path.name}: duplicate or missing columns; need {required}')
        result = list(reader)
    require(result, f'{path.name}: empty observations cannot pass validation')
    for i, row in enumerate(result, 2):
        require(None not in row and all(v is not None for v in row.values()),
                f'{path.name}:{i}: malformed CSV row')
    return result

def number(value: Any, name: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    require(not isinstance(value, bool), f'{name}: boolean is not a measurement')
    try:
        x = float(value)
    except (ValueError, TypeError) as exc:
        raise EvidenceError(f'{name}: expected finite number, got {value!r}') from exc
    require(math.isfinite(x), f'{name}: nonfinite number')
    require(minimum is None or x >= minimum, f'{name}: below {minimum}')
    require(maximum is None or x <= maximum, f'{name}: above {maximum}')
    return x

def integer(value: Any, name: str, *, minimum: int = 0) -> int:
    x = number(value, name, minimum=minimum)
    require(x.is_integer(), f'{name}: must be an integer')
    return int(x)

def unique(items, label: str) -> None:
    require(len(items) == len(set(items)), f'Duplicate {label}')

def directory_inventory(root: Path, *, exclude: tuple[str, ...] = ()) -> list[dict]:
    require(root.is_dir() and not root.is_symlink(), f'Invalid source directory: {root}')
    result = []
    for p in sorted(root.rglob('*')):
        require(not p.is_symlink(), f'Symlink not permitted in archive: {p}')
        if p.is_dir():
            continue
        require(p.is_file(), f'Non-regular artifact: {p}')
        rel = p.relative_to(root).as_posix()
        relative(rel)
        if rel in exclude:
            continue
        result.append({'path': rel, 'bytes': p.stat().st_size, 'sha256': sha256(p)})
    require(result, 'An empty evidence directory cannot be deposited')
    return result
