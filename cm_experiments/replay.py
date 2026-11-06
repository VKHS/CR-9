"""Independent score reconstruction and trusted checkpoint reexecution for arms."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .artifacts import verify,source_identity,sha256,canonical
from .holdout import ARM_SCHEMA, comparison_contract, require_arm_contract
from cross_morse.split_contract import RERUN_GUIDANCE


def numeric_replay(study, *, allow_score_fixture=False):
    """Recompute results from saved observations, not printed metric constants."""
    from cross_morse.evaluation import score_observations
    study=Path(study);inventory=verify(study)
    experiment=json.loads((study/'experiment.json').read_text())
    config=json.loads((study/'base_config.json').read_text())
    results=json.loads((study/'results.json').read_text())
    score_fixture = (allow_score_fixture and experiment.get('schema') == 'cm-arm-score-fixture-1')
    if not score_fixture and experiment.get('schema') != ARM_SCHEMA:
        raise ValueError('Legacy/wrong-role arm archive. ' + RERUN_GUIDANCE)
    if canonical(config)!=experiment['base_configuration_sha256']:
        raise ValueError('Base configuration identity mismatch')
    with np.load(study/'predictions.npz',allow_pickle=False) as source:
        data={k:source[k] for k in source.files}
    with np.load(study/'paired_unit_losses.npz',allow_pickle=False) as source:
        units={k:source[k] for k in source.files}
    if not score_fixture:
        with np.load(study/'data_roles.npz', allow_pickle=False) as role_file:
            roles = {k: role_file[k] for k in role_file.files}
        require_arm_contract(experiment, roles, results=results, observations=data['observation_id'])
    selection_replay = None
    if experiment.get('schema') == ARM_SCHEMA:
        from .selection_replay import replay_selection
        selection_replay = replay_selection(study, experiment, config)
        if results['development_selection_sha256'] != sha256(study/'development_selection.json'):
            raise ValueError('Comparison used a different development selection')
    if experiment.get('schema') == ARM_SCHEMA:
        from .accounting import summarize_work
        work=summarize_work(json.loads((study/'timings.json').read_text()),
                            json.loads((study/'execution_traces.json').read_text()),
                            ratio=experiment['budget_ratio'],
                            scored={a:r['scored'] for a,r in results['results'].items()},
                            basis='deployment' if experiment.get('schema') == ARM_SCHEMA else 'qualification_inclusive')
        saved=json.loads((study/'work_comparisons.json').read_text())
        if work != saved:
            raise ValueError('Arm work and ceiling accounting failed numerical replay')
        if experiment.get('schema') == ARM_SCHEMA:
            from .protocol import deployed_work
            traces=json.loads((study/'execution_traces.json').read_text())
            inputs=json.loads((study/'per_input_work.json').read_text())
            observed={(r['arm'],int(r['episode_id'])):r for r in inputs}
            expected={(r['arm'],int(r['episode_id'])):r for r in work['per_input']}
            if len(observed)!=len(inputs) or set(observed)!=set(expected):
                raise ValueError('Input-work coverage differs from executed traces')
            for row in traces:
                key=(row['arm'],int(row['episode_id']))
                amount=sum(deployed_work(t) for t in row['member_traces'])
                if not np.isclose(amount,observed[key]['deployed_work'],rtol=1e-10,atol=1e-8) or not np.isclose(amount,expected[key]['deployed_work'],rtol=1e-10,atol=1e-8):
                    raise ValueError('Input work is not the sum of executed member products')
                if not np.isclose(observed[key]['qc_deployed_work'],expected[key]['comparator_reference_work']):
                    raise ValueError('Input comparator denominator differs')
        for a,row in results['results'].items():
            if row['within_declared_mean_work_ceiling'] != work['summary'][a]['within_mean_ceiling']:
                raise ValueError('Claimed work-ceiling eligibility differs from measured work')
    checks=[]
    def equal(name,a,b):
        passed=bool(np.allclose(a,b,rtol=1e-10,atol=1e-12,equal_nan=False))
        checks.append({'check':name,'passed':passed})
        if not passed:raise ValueError('Numerical arm replay failed: '+name)
    for arm,row in results['results'].items():
        if not row['scored']:
            if arm in data or arm in units:raise ValueError('Unscored arm has fabricated observations')
            continue
        if arm not in data:raise ValueError('Scored arm prediction array missing')
        score=score_observations(data[arm],data['y_true'],data['unit_id'],config)
        equal(arm+'.metric',score['metric'],row['metric'])
        equal(arm+'.loss',score['node_ce'],row['loss'])
        equal(arm+'.unit_losses',score['loss'],units[arm])
        equal(arm+'.unit_ids',score['core_ids'],units['unit_id'])
        equal(arm+'.unit_weights',score['counts'],units['unit_weight'])
        if arm.startswith('E'):
            equal(arm+'.paired_feedback',units[arm]-units['feedback'],units[arm+'_minus_feedback'])
            equal(arm+'.paired_qC',units[arm]-units['qC'],units[arm+'_minus_qC'])
    return {'passed':True,'checks':len(checks),'details':checks,'inventory':inventory,
            'experiment_scope':experiment['scope'],'publication_deposit':False, 'development_selection': selection_replay}


def checkpoint_replay(study,base_run,*,device='cpu',trust=False,atol=0.):
    """Reexecute actual arm checkpoints. This does not train or refit any model."""
    if not trust:raise ValueError('Explicit checkpoint trust required')
    if atol<0 or not np.isfinite(atol):raise ValueError('Nonnegative replay tolerance required')
    import torch
    from cross_morse.data import load_data
    from cross_morse.training import load_plant
    from cross_morse.learning import Controller
    from cross_morse.provenance import verify_stage_chain,current_source_sha256
    from cross_morse.runtime import graph_operator
    from cross_morse.policies import execute_policy
    from .runner import episodes
    from .schedules import load_decoder
    from .execution import execute_arm,execute_budget_ensemble
    study=Path(study).resolve();root=Path(base_run).resolve();numeric=numeric_replay(study)
    config=json.loads((root/'config.json').read_text());exp=json.loads((study/'experiment.json').read_text())
    if canonical(config)!=exp['base_configuration_sha256']:raise ValueError('Replay base configuration differs')
    if current_source_sha256()!=exp['source_runtime_sha256']:raise ValueError('Frozen plant source differs')
    if source_identity(Path(__file__).resolve().parents[1])!=exp['arm_source']:raise ValueError('Arm source differs')
    if sha256(root/'provenance/stages/05_confirm.json')!=exp['base_confirm_manifest_sha256']:
        raise ValueError('Confirmed base run identity differs')
    if config['model'].get('precision')=='bf16' and not str(device).startswith('cuda'):
        raise ValueError('BF16 checkpoint execution requires CUDA')
    verify_stage_chain(root,'confirm',expected_source_sha256=current_source_sha256())
    if exp.get('schema') != ARM_SCHEMA:
        raise ValueError(RERUN_GUIDANCE)
    from .study_execution import StudyBackend
    data=load_data(root).protect_labels()
    if comparison_contract(data, root) != exp.get('comparison_holdout'):
        raise ValueError('Comparison holdout differs from its sealed parent preparation')
    registry=json.loads((study/'variant_registry.json').read_text())
    decisions=json.loads((study/'selection_lock.json').read_text())['decisions']
    selected={a:next((r for r in registry[a] if r['configuration_id']==decisions[a]['selected']),None) for a in registry}
    decoder_ids={r['decoder_id'] for rows in registry.values() for r in rows if r['kind']=='decoder'}
    decoders={k:load_decoder(study/f'decoders/{k}.pt',device)[0] for k in decoder_ids}
    bounds=json.loads((study/'cost_bounds.json').read_text()) if (study/'cost_bounds.json').is_file() else None
    backend=StudyBackend(root,data,config,exp['teacher'],exp['comparator'],exp['ensemble_members'],device,
                         decoders,costs=None if bounds is None else bounds['costs'])
    saved=dict(np.load(study/'predictions.npz',allow_pickle=False))
    observed={a:[] for a in ('qC','feedback','E3','E4','E5','E6')}
    for eid,episode in episodes(data,'compare',exp['max_graphs']):
        ids=episode.splits['compare']
        if not len(ids):continue
        qc=backend.execute(episode,{'kind':'policy','candidate':exp['comparator']})
        values={'qC':qc['prediction'], 'feedback':backend.execute(episode,{'kind':'policy','candidate':exp['teacher']})['prediction']}
        for arm,spec in selected.items():
            values[arm]=(None if spec is None else backend.execute(episode,spec,
                qc_reference_work=qc['deployed_work'],hard_cap_ratio=exp['budget']['hard_cap_ratio'])['prediction'])
        for arm,p in values.items():observed[arm].append(None if p is None else p[ids])
    comparisons=[]
    for arm,parts in observed.items():
        if any(p is None for p in parts):
            if arm in saved:raise ValueError('Replay became unscored for a previously scored arm')
            comparisons.append({'arm':arm,'status':'UNSCORED_REPRODUCED','scored':False});continue
        p=np.concatenate(parts)
        if arm not in saved or p.shape!=saved[arm].shape:raise ValueError('Replay prediction shape differs')
        difference=float(np.max(np.abs(p.astype('float64')-saved[arm])))
        if difference>atol:raise ValueError(f'{arm} replay differs by {difference}')
        comparisons.append({'arm':arm,'status':'REEXECUTED','max_absolute_error':difference,'scored':True})
    verify_stage_chain(root,'confirm',expected_source_sha256=current_source_sha256());verify(study)
    return {'passed':True,'numeric_replay':numeric,'checkpoint_comparisons':comparisons,'atol':atol,
            'device':device,'scope':'same sealed run, data and source; not independent original-benchmark authentication'}
