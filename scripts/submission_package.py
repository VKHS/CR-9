#!/usr/bin/env python3
"""Connect existing experiment outputs to the manuscript's availability statement.

inventory: read-only clause and archive-routing report.
verify: strict complete-tier reconstruction and committed-view checks.
finalize: verify existing original folders, optionally publish, then prepare a
          reviewed registration patch in an isolated Git worktree. Never train,
          change measurements, rewrite history, or change repository visibility.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class PackageError(ValueError):
    pass


def read_json(path: Path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise PackageError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise PackageError(f"Nonfinite JSON value: {value}")

    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=invalid)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def safe_path(path: Path) -> Path:
    path = Path(path).expanduser().absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise PackageError("Output and input paths must not traverse symlinks")
    return path.resolve()


def require_separate(work: Path, protected: list[Path]) -> None:
    for item in protected:
        item = safe_path(item)
        if work.is_relative_to(item) or item.is_relative_to(work):
            raise PackageError(f"Output workspace overlaps a protected input: {item}")


def expected_displays(root: Path) -> list[str]:
    rows = read_json(root / 'paper/display_manifest.json').get('displays')
    if not isinstance(rows, list) or not rows:
        raise PackageError("The manuscript display inventory is empty or malformed")
    ids = [row.get('id') if isinstance(row, dict) else None for row in rows]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise PackageError("The display inventory has invalid or duplicate identifiers")
    return ids


def reconstruction_problems(status: dict, expected: int, required: set[str]) -> list[str]:
    """Fail closed: a partial, absent, stale, or 0-of-0 report cannot pass."""
    problems: list[str] = []
    if not isinstance(status, dict):
        return ['Reconstruction status must be an object']
    for key in ('displays', 'complete'):
        if type(status.get(key)) is not int or status[key] != expected or expected <= 0:
            problems.append(f'{key} must equal the registered display count ({expected})')
    for key in ('values_differ', 'values_not_comparable', 'values_no_reference', 'values_missing'):
        if type(status.get(key)) is not int or status[key] != 0:
            problems.append(f'{key} must be zero')
    if type(status.get('values_exact')) is not int or status['values_exact'] <= 0:
        problems.append('A nonempty numerical comparison is required')
    for key in ('incomplete', 'display_content_incomplete', 'artifacts_awaiting_publication'):
        if status.get(key) != []:
            problems.append(f'{key} must be an explicitly empty list')
    for key in ('generator_errors', 'module_errors'):
        if status.get(key) != {}:
            problems.append(f'{key} must be an explicitly empty object')
    if status.get('comparison_coverage_complete') is not True:
        problems.append('Complete comparison coverage is required')
    if status.get('channel') is not None:
        problems.append('Use the strict manuscript command, not an alternate execution channel')
    materialization = status.get('artifact_materialization')
    if not isinstance(materialization, dict):
        return problems + ['Complete-tier artifact materialization is not recorded']
    if materialization.get('requested_tier') != 'complete':
        problems.append('The complete artifact tier is required')
    for key in ('not_registered', 'not_published', 'missing_local', 'incomplete_tiers'):
        if materialization.get(key) != []:
            problems.append(f'artifact_materialization.{key} must be an explicitly empty list')
    declared = materialization.get('required_artifacts')
    if not isinstance(declared, list) or any(not isinstance(i, str) for i in declared) or set(declared) != required or len(declared) != len(required):
        problems.append('Materialization requirements differ from the canonical artifact inventory')
    tiers = materialization.get('verified_tiers')
    if not isinstance(tiers, dict) or any(tiers.get(i) != 'complete' for i in required):
        problems.append('Every required archive must have verified complete-tier bytes')
    return problems


def descriptor_problems(entry: dict | None) -> list[str]:
    if not isinstance(entry, dict):
        return ['No canonical manifest entry']
    problems = []
    if entry.get('status') != 'published':
        problems.append('Publication is not registered')
    complete = (entry.get('tiers') or {}).get('complete') or {}
    parts = complete.get('parts')
    if not isinstance(parts, list) or not parts or any(not isinstance(p, dict) or not p.get('url') or not p.get('sha256') for p in parts):
        problems.append('Complete-tier archive URLs and checksums are not connected')
    if not entry.get('identity') or not entry.get('inventory_sha256'):
        problems.append('Execution identity and file inventory are not connected')
    if entry.get('scientific_only_verification') is False:
        problems.append('This descriptor uses a different validation scope; inspect its originating records')
    return problems


def inventory(root: Path = ROOT) -> dict:
    from scripts import paper_artifacts as pa, statement_check
    manifest_path = root / 'release/paper_artifacts.json'
    manifest = pa.load_manifest(manifest_path)
    contracts = pa.required_artifacts(root)
    entries = {a['id']: a for a in manifest['artifacts']}
    other = {}
    # Locate descriptors, never promote or relabel another execution automatically.
    for candidate in sorted((root / 'release').glob('*artifacts.json')):
        if candidate == manifest_path:
            continue
        try:
            document = read_json(candidate)
            for row in document.get('artifacts', []):
                if row.get('tiers', {}).get('complete', {}).get('parts'):
                    other.setdefault(row['id'], []).append(candidate.relative_to(root).as_posix())
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
    rows = []
    for contract in contracts:
        ident = contract['id']
        entry = entries.get(ident)
        rows.append({'id': ident, 'problems': descriptor_problems(entry),
                     'other_descriptor_locations': other.get(ident, []),
                     'affected_displays': contract.get('displays', [])})
    clauses = statement_check.evaluate(root)
    return {'schema': 'cross-morse-submission-inventory-1',
            'checked_utc': datetime.now(timezone.utc).isoformat(),
            'manifest_sha256': sha256(manifest_path), 'registered_displays': len(expected_displays(root)),
            'required_artifacts': len(rows), 'connected_artifacts': sum(not r['problems'] for r in rows),
            'artifacts': rows, 'statement_items': clauses['items'],
            'ready_for_reconstruction': bool(rows) and not any(r['problems'] for r in rows),
            'public_access_verified': False, 'full_reconstruction_verified': False,
            'scope': 'Read-only inventory and routing check; no experiment is rerun and no data are changed.'}


def markdown(report: dict) -> str:
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    lines = ['# CROSS-MORSE submission package', '',
             f"Status: **{cell(report.get('verdict', 'INVENTORY_ONLY'))}**.", '',
             'Inventory checks and software tests are not a full-manuscript reconstruction certificate.', '']
    inv = report.get('inventory', report)
    if 'artifacts' in inv:
        lines.extend([f"Registered displays: {inv.get('registered_displays')}; connected complete-tier descriptors: "
                      f"{inv.get('connected_artifacts')}/{inv.get('required_artifacts')}.", '',
                      '| Artifact | Connection work | Other descriptor locations |', '|---|---|---|'])
        for row in inv['artifacts']:
            lines.append(f"| {cell(row['id'])} | {cell('; '.join(row['problems']) or 'Connected')} | "
                         f"{cell(', '.join(row.get('other_descriptor_locations', [])) or '—')} |")
    if inv.get('statement_items'):
        lines.extend(['', '## Availability clauses', '', '| Clause | Recorded status | Required connection |', '|---|---|---|'])
        for row in inv['statement_items']:
            lines.append(f"| {cell(row['phrase'])} | {cell(row['status'])} | {cell(row.get('remaining', ''))} |")
    if report.get('error'):
        lines.extend(['', '## Finding', '', cell(report['error'])])
    if report.get('steps'):
        lines.extend(['', '## Executed steps', '', '| Step | Exit code | Log |', '|---|---|---|'])
        for row in report['steps']:
            lines.append(f"| {cell(row['name'])} | {row['exit_code']} | {cell(row['log'])} |")
    lines.extend(['', 'Repository visibility and reviewer permissions remain unchanged. '
                  'A private repository requires reviewer access or an accessible archive.', ''])
    return '\n'.join(lines)


def write_report(folder: Path, report: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (folder / 'REPORT.md').write_text(markdown(report), encoding='utf-8')


def run_step(name: str, argv: list[str], cwd: Path, work: Path, report: dict) -> None:
    log = work / f'{len(report["steps"]):02d}-{name}.log'
    with log.open('w', encoding='utf-8') as out:
        result = subprocess.run(argv, cwd=cwd, stdout=out, stderr=subprocess.STDOUT, check=False)
    report['steps'].append({'name': name, 'exit_code': result.returncode, 'log': log.name})
    if result.returncode:
        raise PackageError(f'{name} did not pass (exit {result.returncode}); see {log.name}')


def python_step(script: str, *args: str) -> list[str]:
    return [sys.executable, '-B', f'scripts/{script}.py', *map(str, args)]


def verify(root: Path, work: Path, report: dict) -> None:
    from scripts import paper_artifacts as pa
    inv = inventory(root)
    report['inventory'] = inv
    if not inv['ready_for_reconstruction']:
        raise PackageError('Connect the listed complete-tier descriptors before requesting manuscript reconstruction')
    manifest = root / 'release/paper_artifacts.json'
    manifest_hash = sha256(manifest)
    run_step('reconstruct', python_step('reconstruct_paper', '--tier', 'complete',
             '--artifacts', work / 'artifacts', '--output-dir', work / 'paper'), root, work, report)
    status = read_json(work / 'paper.status.json')
    required = {r['id'] for r in pa.required_artifacts(root)}
    required.update(a['id'] for a in pa.load_manifest(manifest)['artifacts'] if a.get('required', True))
    problems = reconstruction_problems(status, len(expected_displays(root)), required)
    if sha256(manifest) != manifest_hash:
        problems.append('Artifact manifest changed during reconstruction')
    if problems:
        raise PackageError('; '.join(problems))
    run_step('committed-view-check', python_step('statement_check', '--require-fulfilled'), root, work, report)
    report.update(verdict='RECONSTRUCTION_AND_REGISTRATION_VERIFIED',
                  reconstruction=status, full_reconstruction_verified=True, manifest_sha256=manifest_hash)


GENERATED_FILES = {
    'release/paper_artifacts.json', 'release/release_scope.json', 'STATEMENT_COMPLIANCE.md',
    'release/final_destination.json', 'docs/FINAL_DESTINATION.md', 'AVAILABILITY.md', 'README.md',
    'paper/CLAIMS.md', 'paper/AVAILABILITY_STATEMENT.txt', 'paper/PAPER_SUPPORT.md', 'paper/RESULTS.md',
    'docs/badges/evidence.json', 'release/artifact_catalog.json', 'docs/ARTIFACT_CATALOG.md', 'docs/CONSISTENCY.md',
    'RELEASE_MANIFEST.json', 'SHA256SUMS.txt',
}
GENERATED_PREFIXES = ('manifests/published/', 'environment/observed/', 'paper/source_data/',
                      'experiments/baselines/', 'destination/')


def finalize(root: Path, work: Path, mapping: Path, publish: bool, copy_mode: str, report: dict) -> None:
    """Prepare registrations in a detached worktree; keep the caller's checkout intact."""
    from scripts import paper_artifacts as pa
    document = read_json(mapping)
    if not isinstance(document, dict) or document.get('_fixture'):
        raise PackageError('Supply a map of the result-producing experiment folders')
    required = {r['id'] for r in pa.required_artifacts(root)}
    paths = {k: v for k, v in document.items() if not k.startswith('_')}
    if set(paths) != required or any(not isinstance(v, str) or not v or not Path(v).is_absolute() for v in paths.values()):
        raise PackageError('The input map must name every required artifact with an absolute folder path')
    relationships = document.get('_relationships', {})
    if not isinstance(relationships, dict) or any(relationships.get(k) not in pa.RELATIONSHIPS for k in required):
        raise PackageError('Preserve each artifact\'s original/reproduction relationship in map._relationships')
    require_separate(work, [root, mapping, *[Path(v) for v in paths.values()]])
    clean = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=normal'], cwd=root)
    if clean.strip():
        raise PackageError('Start from a clean committed checkout; existing work will not be overwritten')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if publish and not pa.github_token():
        raise PackageError('Publication requires GH_TOKEN/GITHUB_TOKEN or an existing gh login')
    report.update(source_commit=commit, input_map_sha256=sha256(mapping), publication_requested=publish)
    checkout = work / 'checkout'
    run_step('isolated-checkout', ['git', 'worktree', 'add', '--detach', str(checkout), commit], root, work, report)
    run_step('original-input-preflight', python_step('paper_preflight', '--map', mapping,
             '--work', work / 'preflight', '--copy-mode', copy_mode), checkout, work, report)
    preflight = read_json(work / 'preflight/report.json')
    if preflight.get('verdict') != 'READY_FOR_PUBLICATION' or preflight.get('software_fixture') is not False:
        raise PackageError('The original-input preflight did not establish publication readiness')
    if sha256(mapping) != report['input_map_sha256']:
        raise PackageError('The input map changed during preflight')
    if not publish:
        report.update(verdict='LOCAL_INPUTS_READY_FOR_PUBLICATION', full_reconstruction_verified=False,
                      note='Local preflight passed. No release was uploaded and no public availability is asserted.')
        return
    run_step('pack', python_step('paper_artifacts', 'pack', '--map', mapping, '--output-dir', work / 'staged',
             '--require-complete'), checkout, work, report)
    run_step('publish', python_step('paper_artifacts', 'publish', '--staged', work / 'staged', '--target', commit),
             checkout, work, report)
    # Downloads the just-published bytes and runs the strict full-manuscript checks.
    run_step('reconstruct', python_step('reconstruct_paper', '--tier', 'complete', '--artifacts', work / 'artifacts',
             '--output-dir', work / 'paper'), checkout, work, report)
    status = read_json(work / 'paper.status.json')
    problems = reconstruction_problems(status, len(expected_displays(checkout)), required)
    if problems:
        raise PackageError('; '.join(problems))
    run_step('derive-registrations', python_step('paper_artifacts', 'derive', '--fetched', work / 'artifacts',
             '--workspace', work / 'paper'), checkout, work, report)
    for script, args in (
        ('statement_check', ['--write']), ('final_destination', ['--write']), ('availability', []),
        ('claims_ledger', []), ('manuscript_availability', []), ('paper_support_report', []),
        ('artifact_catalog', []), ('consistency_audit', ['--write']),
        ('create_release_manifest', ['--root', '.', '--output', 'RELEASE_MANIFEST.json']),
    ):
        run_step(f'refresh-{script}', python_step(script, *args), checkout, work, report)
    run_step('availability-fulfilled', python_step('statement_check', '--require-fulfilled'), checkout, work, report)
    run_step('repository-verification', ['make', 'verify', f'PYTHON={sys.executable}'], checkout, work, report)
    changes = set(filter(None, subprocess.check_output(['git', 'diff', '--name-only', '-z'], cwd=checkout).decode().split('\0')))
    changes.update(filter(None, subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '-z'], cwd=checkout).decode().split('\0')))
    unexpected = [p for p in changes if p not in GENERATED_FILES and not p.startswith(GENERATED_PREFIXES)]
    if unexpected:
        raise PackageError('Unexpected generated paths require review: ' + ', '.join(sorted(unexpected)))
    if changes:
        run_step('stage-registration', ['git', 'add', '--', *sorted(changes)], checkout, work, report)
    with (work / 'submission-registration.patch').open('wb') as out:
        subprocess.run(['git', 'diff', '--cached', '--binary', '--full-index', 'HEAD'], cwd=checkout, stdout=out, check=True)
    report.update(verdict='VERIFIED_REGISTRATION_PATCH_READY', full_reconstruction_verified=True,
                  reconstruction=status, registration_commit_published=False,
                  patch='submission-registration.patch', isolated_checkout=str(checkout),
                  note='Review and commit the registration patch. The caller\'s branch and repository visibility are unchanged.')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('inventory', 'verify', 'finalize'):
        p = sub.add_parser(name)
        p.add_argument('--work', type=Path, required=True, help='new output folder outside the checkout and every input')
        if name == 'finalize':
            p.add_argument('--map', type=Path, required=True)
            p.add_argument('--publish', action='store_true', help='explicitly authorize publishing immutable release assets')
            p.add_argument('--copy-mode', choices=('copy', 'hardlink'), default='copy')
    args = parser.parse_args(argv)
    report = {'schema': 'cross-morse-submission-package-1', 'command': args.command,
              'checked_utc': datetime.now(timezone.utc).isoformat(), 'verdict': 'NOT_COMPLETED', 'steps': [],
              'full_reconstruction_verified': False, 'public_access_verified': False}
    work = None
    try:
        candidate = safe_path(args.work)
        require_separate(candidate, [ROOT])
        if candidate.exists():
            raise PackageError('Choose a new workspace; an earlier report will not be overwritten')
        # Validate overlap before creating any directory inside a supplied artifact.
        if args.command == 'finalize':
            mapping = safe_path(args.map)
            doc = read_json(mapping)
            if not isinstance(doc, dict):
                raise PackageError('Input map must be a JSON object')
            listed = [Path(v) for k, v in doc.items() if not k.startswith('_') and isinstance(v, str) and v]
            require_separate(candidate, [ROOT, mapping, *listed])
        candidate.mkdir(parents=True, exist_ok=False)
        work = candidate
        if args.command == 'inventory':
            report['inventory'] = inventory(ROOT)
            report['verdict'] = 'DESCRIPTORS_CONNECTED' if report['inventory']['ready_for_reconstruction'] else 'REGISTRATION_CONNECTIONS_REQUIRED'
            code = 0 if report['inventory']['ready_for_reconstruction'] else 1
        elif args.command == 'verify':
            verify(ROOT, work, report)
            code = 0
        else:
            finalize(ROOT, work, mapping, args.publish, args.copy_mode, report)
            code = 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        report.update(verdict='NOT_COMPLETED', error=str(exc))
        print(f'ERROR: {exc}', file=sys.stderr)
        code = 1 if isinstance(exc, PackageError) else 2
    if work is not None:
        write_report(work, report)
        print(f"{report['verdict']}: {work / 'REPORT.md'}")
    return code


if __name__ == '__main__':
    raise SystemExit(main())
