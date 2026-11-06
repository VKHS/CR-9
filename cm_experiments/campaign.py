"""Export executed arm studies without inventing seeds, hardware records or DOIs.

The partial path is useful while a campaign is running. Only a complete original
campaign can be sealed as budget_matched; computations are repeated on ingestion.
"""
from __future__ import annotations
import csv
import json
import shutil
from pathlib import Path
import numpy as np
from .artifacts import canonical, sha256, write_json, fresh_output, verify, seal
from .protocol import ARMS
from .holdout import ARM_SCHEMA, CAMPAIGN_SCHEMA
from cm_campaigns.contract import ADMITTED

SEED_SCOPE = 'controller_optimization_conditional_on_frozen_parent_ensemble'


def load(path):
    return json.loads(Path(path).read_text())


def resolve_input(base, value):
    p = Path(value).expanduser()
    return (base/p).resolve() if not p.is_absolute() else p.resolve()


def slots(seed_ids):
    if not isinstance(seed_ids, list) or len(seed_ids) != 12 or \
       any(type(s) is not int or s < 0 for s in seed_ids) or len(set(seed_ids)) != 12:
        raise ValueError('Declare twelve distinct nonnegative controller-optimization seeds')
    return {(d, s, sp) for d in ADMITTED for s in seed_ids
            for sp in (range(10) if d == 'Roman-empire' else [0])}


def coverage(entries, seed_ids):
    expected = slots(seed_ids)
    if not isinstance(entries, list) or any(type(e.get('seed')) is not int or type(e.get('split', 0)) is not int for e in entries):
        raise ValueError('Study seed and split identifiers must be integers, not coerced strings or booleans')
    keys = [(e['dataset'], e['seed'], e.get('split', 0)) for e in entries]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate dataset/seed/split study; repeated files are not new replicates')
    unexpected = set(keys)-expected
    if unexpected:
        raise ValueError(f'Unexpected campaign slots: {sorted(unexpected)}')
    return {'required_studies': len(expected), 'observed_studies': len(keys),
            'complete': set(keys) == expected,
            'missing': [dict(dataset=d, seed=s, split=sp) for d, s, sp in sorted(expected-set(keys))]}


def csv_write(path, values):
    if not values:
        raise ValueError('Cannot create an empty observation table')
    with Path(path).open('x', newline='') as f:
        out = csv.DictWriter(f, fieldnames=list(values[0])); out.writeheader(); out.writerows(values)


def export_campaign(manifest, output, *, complete=False):
    """Copy only actual inputs, numerically verify them, and stage a deposit.

    The emitted descriptor is a sidecar for records seal. It does not register a
    DOI, mark the publication gate successful, or make an authenticity claim from
    filenames. `complete` refuses fixtures, posthoc studies and missing originals.
    """
    from .replay import numeric_replay
    from cross_morse.provenance import verify_stage_chain
    manifest = Path(manifest).resolve(); spec = load(manifest); base = manifest.parent
    if spec.get('schema') != 'cm-arm-campaign-1' or spec.get('seed_scope') != SEED_SCOPE:
        raise ValueError('Use cm-arm-campaign-1 with the explicit conditional controller-seed estimand')
    entries = spec.get('studies', []); cov = coverage(entries, spec.get('seed_ids'))
    if not entries:
        raise ValueError('No executed arm studies supplied')
    if complete and not cov['complete']:
        raise ValueError('Campaign incomplete: '+str(len(cov['missing']))+' studies absent; use partial export')
    checked = []; seen_manifests = set()
    for item in entries:
        study = resolve_input(base, item['study']); parent = resolve_input(base, item['parent'])
        verify(study); exp = load(study/'experiment.json')
        if exp.get('schema') != ARM_SCHEMA:
            raise ValueError('Only schema-3 studies with a preallocated D_compare can enter this exporter; rerun older parents from preparation')
        key = (item['dataset'], int(item['seed']), int(item.get('split', 0)))
        cfg = load(parent/'config.json')
        if exp['dataset'] != key[0] or exp['training_seed'] != key[1] or \
           int(cfg['data'].get('split_index', 0)) != key[2]:
            raise ValueError('Campaign slot does not match the actual experiment seed/dataset/split')
        mh = sha256(study/'ARTIFACT_MANIFEST.json')
        if mh in seen_manifests:
            raise ValueError('One study cannot be relabelled as multiple campaign replicates')
        seen_manifests.add(mh)
        if exp['base_configuration_sha256'] != canonical(cfg) or \
           exp['base_confirm_manifest_sha256'] != sha256(parent/'provenance/stages/05_confirm.json'):
            raise ValueError('Parent identities differ from the experiment that actually ran')
        verify_stage_chain(parent, 'confirm', expected_source_sha256=exp['source_runtime_sha256'])
        replay = numeric_replay(study)
        reasons = []
        if exp.get('input_dataset') != key[0] or exp.get('profile') == 'smoke' or exp.get('fixture_teacher_override'):
            reasons.append('software_fixture')
        if exp.get('protocol') != 'pre_admission' or exp.get('prior_admission_or_test_available'):
            reasons.append('retrospective_data_use')
        if exp.get('max_graphs') != 0:
            reasons.append('truncated_dataset')
        if not all(v.get('scored') and v.get('support_qualified') for v in load(study/'results.json')['results'].values()):
            reasons.append('unscored_or_unsupported_arm')
        attachments = {}
        for role in ('accounting', 'job_log'):
            paths = item.get(role, [])
            if not isinstance(paths, list) or not paths:
                reasons.append('missing_'+role)
            else:
                attachments[role] = [resolve_input(base, p) for p in paths]
                if any(not p.is_file() or not p.stat().st_size for p in attachments[role]):
                    raise ValueError('An original accounting/log attachment is absent or empty')
        if complete and reasons:
            raise ValueError(f'{key}: cannot seal as original campaign: {reasons}')
        checked.append((item, study, parent, exp, replay, reasons, attachments))
    # Validate all inputs before creating any outputs. Never overwrite a run.
    for _, spath, ppath, *_ in checked:
        target = Path(output).resolve()
        if any(target == q or target.is_relative_to(q) or q.is_relative_to(target) for q in (spath, ppath)):
            raise ValueError('Campaign output overlaps an immutable input')
    out = fresh_output(output, checked[0][1])
    references = []; role_map = {}; parents = {}; array_rows = []
    try:
        for item, study, parent, exp, replay, reasons, attachments in checked:
            d = item['dataset']; s = int(item['seed']); sp = int(item.get('split', 0))
            rel = f'studies/{d}/seed-{s}/split-{sp}'
            shutil.copytree(study, out/rel)
            ph = exp['base_confirm_manifest_sha256']; prel = 'parents/'+ph
            if ph not in parents:
                # The immutable parent contains its real checkpoints, configurations,
                # split identities, fitting artifacts and source snapshots.
                shutil.copytree(parent, out/prel, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
                verify_stage_chain(out/prel, 'confirm', expected_source_sha256=exp['source_runtime_sha256'])
                parents[ph] = prel
            attached = {}
            for role, paths in attachments.items():
                attached[role] = []
                for i, p in enumerate(paths):
                    name = f'execution/{d}/seed-{s}/split-{sp}/{role}-{i}{p.suffix}'
                    (out/name).parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, out/name)
                    attached[role].append(name); role_map.setdefault(role, []).append(name)
            references.append({'dataset': d, 'seed': s, 'split': sp, 'study': rel, 'parent': prel,
                               'manifest_sha256': sha256(study/'ARTIFACT_MANIFEST.json'),
                               'replay': replay, 'qualification_blockers': reasons, **attached})
            with np.load(study/'paired_unit_losses.npz', allow_pickle=False) as z:
                unit_ids = z['unit_id']; weights = z['unit_weight']
                # Roman's statistical unit is the official split, not a graph core.
                if d == 'Roman-empire':
                    unit_ids = np.asarray([sp]); values = {a: np.asarray([np.average(z[a], weights=weights)])
                         for a in ('qC', 'feedback', *ARMS) if a in z}
                else:
                    values = {a: z[a] for a in ('qC', 'feedback', *ARMS) if a in z}
                for arm in ARMS:
                    if arm not in values: continue
                    for k, unit in enumerate(unit_ids):
                        array_rows.append({'dataset': d, 'seed': s, 'split': sp, 'unit_id': str(unit), 'arm': arm,
                            'qc_loss': float(values['qC'][k]), 'qm_loss': float(values['feedback'][k]),
                            'arm_loss': float(values[arm][k])})
        csv_write(out/'paired_observations.csv', array_rows)
        study_json = {'schema': CAMPAIGN_SCHEMA, 'record_id': 'budget_matched',
            'seed_ids': spec['seed_ids'], 'seed_scope': SEED_SCOPE, 'evaluation_role': 'compare',
            'selection_role': 'Dexplore_selection', 'cost_unit': 'instrumented_deployed_multiply_add_products',
            'expected_work_unit': 'input', 'loss_unit': 'cluster / official split / graph as dataset declares',
            'coverage': cov, 'studies': references, 'observations': 'paired_observations.csv',
            'analysis': spec.get('analysis'), 'run_links': spec.get('run_links', {}),
            'historical_table_reproduction': False, 'registration': 'newly executed arm comparison; replace numerical claims only after remeasurement'}
        write_json(out/'study.json', study_json)
        write_json(out/'coverage.json', cov)
        # No schedule, checkpoint or CPU run is changed into an original GPU run.
        write_json(out/'seed_manifest.json', {'seed_ids': spec['seed_ids'], 'seed_scope': SEED_SCOPE})
        if complete:
            from cm_campaigns.arm_campaign import check_campaign
            result = check_campaign(out, study_json, strict=True)
            write_json(out/'analysis.json', result)
            from cm_campaigns.common import directory_inventory
            from cm_campaigns.archives import validate_descriptor
            source = spec.get('source', {})
            paths = directory_inventory(out)
            roles = {
                'configuration': [r['study']+'/base_config.json' for r in references],
                'environment': [r['study']+'/environment.json' for r in references],
                'split_manifest': [r['study']+'/data_roles.npz' for r in references],
                'seed_manifest': ['seed_manifest.json'], 'raw_observations': ['paired_observations.csv'] +
                    [r['study']+'/predictions.npz' for r in references],
                'statistical_output': ['analysis.json'],
                'code_snapshot': [p['path'] for p in paths if '/code_snapshot/' in p['path']],
                'job_log': role_map['job_log'], 'accounting': role_map['accounting']}
            desc = {'schema': 'cross-morse-evidence-archive-1', 'origin': 'original_experiment',
                'kind': 'record_set', 'record_id': 'budget_matched', 'datasets': list(ADMITTED),
                'source': source, 'study': 'study.json', 'roles': roles, 'files': paths}
            validate_descriptor(desc)
            write_json(out.with_name(out.name+'.descriptor.json'), desc)
        else:
            from cm_campaigns.arm_campaign import check_campaign
            verified = check_campaign(out, study_json, strict=False)
            result = {'passed': verified['passed'], 'publication_deposit': False, 'coverage': cov,
                      'comparisons': len(verified['summary']),
                      'scope': 'partial campaign export; not eligible for the original-record index'}
            write_json(out/'partial_analysis.json', verified)
            write_json(out/'PARTIAL_EXPORT.json', result)
        return {'output': str(out), 'complete_original_campaign': bool(complete), **result}
    except Exception:
        # Preserve a diagnostic partial export, never destroy original sources.
        if not (out/'EXPORT_INCOMPLETE.json').exists():
            write_json(out/'EXPORT_INCOMPLETE.json', {'status': 'FAILED', 'publication_deposit': False})
        raise
