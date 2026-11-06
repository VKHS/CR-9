"""Strict reducer for development-selected, input-accounted E3--E6 campaigns.

Raw studies remain byte-preserved. Inference work is grouped by actual INPUT;
cluster/graph/split losses use the dataset's statistical unit. They are never
joined by duplicating one whole-input work ledger across its scored clusters.
"""
from __future__ import annotations
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from .common import require, confined, rows, sha256, valid_hash
from .numeric import paired_effect, holm, matrix_from_rows
from .contract import ADMITTED, ARMS, UNITS
from .accounting import read_sacct


def _json(path):
    return json.loads(Path(path).read_text())


def operation_trace(trace):
    """Replay declared-bound affordability, charged work and the held reserve.

    No new native operations are fabricated for a preselected comparator. That
    path is checked against qC predictions/work separately below.

    An old ledger containing only observed costs and ``prechecked=True`` can
    describe executed work, but cannot certify the before-operation check of
    Algorithm S3. Missing original bounds fail here rather than being inferred
    from the paid cost after execution.
    """
    ops = trace.get('operations', [])
    require(isinstance(ops, list) and bool(ops) and all(isinstance(op, dict) for op in ops),
            'An S3 execution lacks its operation ledger')
    def amount(value, name):
        require(not isinstance(value, bool) and isinstance(value, (int, float))
                and np.isfinite(value) and value >= 0, 'Invalid ' + name)
        return float(value)
    reserve = amount(ops[0].get('comparator_reserve'), 'comparator reserve')
    cap = amount(ops[0].get('budget'), 'work cap')
    comparator = amount(ops[0].get('comparator_bound'), 'comparator bound')
    require(reserve >= comparator, 'Restart reserve does not cover the comparator bound')
    work_unit = ops[0].get('work_unit')
    require(isinstance(work_unit, str) and bool(work_unit), 'Missing declared E6 work unit')
    spent = 0.; terminal = False
    for i, op in enumerate(ops):
        require(not terminal and type(op.get('operation_index')) is int and op['operation_index'] == i,
                'Truncated or reordered E6 ledger')
        require(amount(op.get('budget'), 'work cap') == cap
                and amount(op.get('comparator_reserve'), 'comparator reserve') == reserve
                and amount(op.get('comparator_bound'), 'comparator bound') == comparator,
                'E6 changes its cap/reserve/comparator bound')
        require(op.get('work_unit') == work_unit, 'E6 changes its declared work unit')
        cost = amount(op.get('paid_work'), 'paid cost')
        name = op['operation']; outcome = op['outcome']
        tol = 1e-8*max(1, cap)
        require(op.get('prechecked') is True, 'Missing operation affordability record')
        if outcome == 'infeasible':
            require(len(ops) == 1 and name == 'none' and cost == 0 and cap < comparator, 'Invalid infeasible record')
            terminal = True
        else:
            require('operation_bound' in op,
                    'Cannot certify legacy E6 ledger without original operation_bound values')
            bound = amount(op['operation_bound'], 'operation bound')
            # Match the executor's observed-bound and affordability tolerances.
            # A small actual cost does not rescue an unaffordable frozen bound.
            require(cost <= bound + 1e-9*max(1., bound), 'Observed work exceeds its declared operation bound')
            if name == 'prediction':
                require(outcome == 'completed' and bound <= reserve
                        and spent+bound <= cap+1e-10, 'Invalid completion bound')
                terminal = True
            elif name == 'comparator_restart':
                require(outcome in ('static_fallback', 'abort_to_comparator')
                        and bound == comparator and spent+bound <= cap+1e-10,
                        'Invalid comparator restart bound')
                require(outcome != 'static_fallback' or i == 0, 'Static fallback follows paid control work')
                terminal = True
            else:
                require(name in ('initialization', 'cheap_measurement', 'score', 'acquisition', 'action', 'native_update'), 'Unknown operation')
                require(outcome == 'continue' and spent+bound+reserve <= cap+1e-10,
                        'Declared operation bound spends the restart reserve before execution')
        spent += cost
        require(abs(amount(op.get('cumulative_work'), 'cumulative work')-spent) <= tol,
                'Incorrect cumulative charged work')
    require(terminal, 'No terminal E6 decision')
    return {'work': spent, 'outcome': ops[-1]['outcome'], 'cap': cap, 'reserve': reserve,
            'operation_bounds_replayed': True, 'work_unit': work_unit}


def _prediction_identity(root, entry, exp):
    """Verify independent comparison exactly and preserve row pairing and role isolation."""
    p = root/entry['study']
    with np.load(p/'predictions.npz', allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    require({'qC', 'feedback', *ARMS, 'y_true', 'observation_id', 'unit_id', 'input_id'} <= set(data), 'Incomplete six-policy prediction archive')
    n = len(data['y_true']); ids = set(map(str, data['observation_id']))
    require(n == len(ids) and len(data['unit_id']) == n == len(data['input_id']), 'Invalid or duplicated comparison IDs')
    for a in ('qC', 'feedback', *ARMS):
        require(len(data[a]) == n and np.isfinite(data[a]).all() and ((data[a] >= 0) & (data[a] <= 1)).all(), 'Invalid probabilities')
    with np.load(p/'data_roles.npz', allow_pickle=False) as z:
        roles = {k: z[k].copy() for k in z.files}
    # Multi-graph source splits index graphs; the observation IDs index nodes.
    # Validate against the archived parent data, not a guessed node/graph join.
    from cross_morse.data import load_data
    parent = root/entry['parent']; data_object = load_data(parent)
    from cm_experiments.holdout import comparison_contract, require_arm_contract
    require(comparison_contract(data_object, parent) == exp.get('comparison_holdout'),
            'Comparison holdout differs from sealed pre-fitting preparation')
    require_arm_contract(exp, roles,
                         observations=data['observation_id'])
    expected = set(map(str, np.asarray(data_object.splits['compare'])))
    # GraphData.global_sample_ids are preserved by the experimental loader.
    require(ids == expected, 'Arm output does not cover exactly the preallocated comparison observations')
    for role in ('fit', 'explore', 'select', 'confirm', 'admit', 'test'):
        require(not ids.intersection(map(str, np.asarray(data_object.splits[role]))),
                'Independent comparison overlaps ' + role)
    return data


def _scheduler(root, entry, exp):
    env = _json(root/entry['study']/'environment.json')
    jobid = env.get('slurm', {}).get('SLURM_JOB_ID')
    require(jobid and env.get('device_requested', '').startswith('cuda'), 'Original campaign requires its captured CUDA Slurm invocation')
    jobs = {}
    for name in entry.get('accounting', []):
        for key, row in read_sacct(confined(root, name))['allocations'].items():
            require(key not in jobs or jobs[key] == row, 'Conflicting scheduler exports for the same allocation')
            jobs[key] = row
    require(jobs and entry.get('job_log'), 'Original accounting or logs missing')
    # Accounting keys may be namespaced by cluster; accept exact final job ID only.
    matching = [j for k,j in jobs.items() if str(jobid) in {str(j.get('job_id','')), str(j.get('job_id_raw',''))}]
    if not matching:
        matching = [j for k,j in jobs.items() if k == str(jobid) or k.endswith(':'+str(jobid))]
    require(matching, 'Captured arm job ID absent from original sacct export')
    require(all(j['allocated_gpus'] > 0 and j['state'].split()[0] == 'COMPLETED' and j['exit_code'] == '0:0' for j in matching), 'Arm campaign job is not a completed GPU allocation')
    for name in entry['job_log']:
        require(confined(root, name).stat().st_size > 0, 'Empty original job log')


def check_campaign(root, study, *, strict=True):
    from cm_experiments.campaign import coverage, SEED_SCOPE
    from cm_experiments.holdout import ARM_SCHEMA, CAMPAIGN_SCHEMA
    from cm_experiments.replay import numeric_replay
    from cm_experiments.artifacts import verify, canonical
    from cross_morse.provenance import verify_stage_chain
    root = Path(root)
    require(study.get('schema') == CAMPAIGN_SCHEMA, 'Unrecognized campaign schema')
    require(study.get('seed_scope') == SEED_SCOPE, 'Controller replicates must not be called independent backbone retrainings')
    require(study.get('evaluation_role') == 'compare' and study.get('selection_role') == 'Dexplore_selection', 'Invalid campaign data roles')
    cov = coverage(study.get('studies', []), study.get('seed_ids'))
    require(not strict or cov['complete'], 'Incomplete original campaign')
    require(study.get('coverage') == cov, 'Campaign coverage does not recompute')
    recorded = rows(confined(root, study['observations']), ('dataset','seed','split','unit_id','arm','qc_loss','qm_loss','arm_loss'))
    lookup = {(r['dataset'], int(r['seed']), int(r['split']), r['unit_id'], r['arm']): r for r in recorded}
    require(len(lookup) == len(recorded), 'Duplicate paired observations')
    observations = []; work = defaultdict(list); reserve_records = []; seen = set(); means = {}; scopes = {}; selection_types = defaultdict(lambda: defaultdict(int))
    for item in study['studies']:
        d, s, sp = item['dataset'], int(item['seed']), int(item['split'])
        p = root/item['study']
        require(p.resolve().is_relative_to(root.resolve()) and p.is_dir(), 'Study path escapes archive')
        verify(p); exp = _json(p/'experiment.json'); result = _json(p/'results.json')
        require(exp.get('schema') == ARM_SCHEMA and exp.get('evaluation_role') == 'D_compare',
                'Legacy/wrong-role study: rerun preparation with independent D_compare')
        require(sha256(p/'ARTIFACT_MANIFEST.json') == item['manifest_sha256'], 'Changed source arm-study manifest')
        require((exp['dataset'], exp['training_seed']) == (d,s), 'Relabelled study dataset/seed')
        if strict:
            require(exp['input_dataset'] == d and exp['profile'] != 'smoke' and not exp['fixture_teacher_override'], 'Software fixture cannot substantiate benchmark results')
            require(exp['protocol'] == 'pre_admission' and not exp['prior_admission_or_test_available'] and exp['max_graphs'] == 0, 'Retrospective/truncated campaign')
            _scheduler(root, item, exp)
        parent = root/item['parent']; require(parent.resolve().is_relative_to(root.resolve()), 'Parent path escapes archive')
        cfg = _json(parent/'config.json')
        require(canonical(cfg) == exp['base_configuration_sha256'] and int(cfg['data'].get('split_index',0)) == sp, 'Parent configuration/split mismatch')
        require(sha256(parent/'provenance/stages/05_confirm.json') == exp['base_confirm_manifest_sha256'], 'Wrong confirmed parent')
        verify_stage_chain(parent, 'confirm', expected_source_sha256=exp['source_runtime_sha256'])
        numeric_replay(p); raw = _prediction_identity(root, item, exp)
        require(all(r.get('scored') and r.get('support_qualified') for r in result['results'].values()), 'Unscored/unsupported arm cannot be silently dropped')
        with np.load(p/'paired_unit_losses.npz', allow_pickle=False) as z:
            uid = z['unit_id']; vals = {a:z[a] for a in ('qC','feedback',*ARMS)}
            if d == 'Roman-empire':
                uid = np.asarray([sp]); vals={a:np.asarray([np.average(v,weights=z['unit_weight'])]) for a,v in vals.items()}
            for a in ARMS:
                for j,u in enumerate(uid):
                    key=(d,s,sp,str(u),a);require(key in lookup,'Missing canonical per-unit observation'); seen.add(key)
                    row=lookup[key]
                    for field,label in (('qc_loss','qC'),('qm_loss','feedback'),('arm_loss',a)):
                        require(np.isclose(float(row[field]),float(vals[label][j]),rtol=1e-9,atol=1e-10), 'Campaign losses differ from replayed predictions')
                    observations.append(dict(row))
        traces = _json(p/'execution_traces.json'); inputs = _json(p/'per_input_work.json')
        index = {(x['arm'],int(x['episode_id'])): x for x in inputs}
        require(len(index)==len(inputs), 'Duplicated input-level work rows')
        expected_inputs=set(map(int,raw['input_id']))
        require(set(index)=={(a,i) for a in ('qC','feedback',*ARMS) for i in expected_inputs}, 'Work input coverage differs from predictions')
        decisions=_json(p/'selection_lock.json')['decisions']; registry=_json(p/'variant_registry.json')
        e6=next(r for r in registry['E6'] if r['configuration_id']==decisions['E6']['selected'])
        for a in ARMS:
            selected=next(r for r in registry[a] if r['configuration_id']==decisions[a]['selected'])
            label=selected['kind']
            if label=='policy':
                label=('comparator_retained' if selected['candidate']['hash']==exp['comparator']['hash']
                       else 'registered_open_loop:'+selected['candidate']['route'])
            selection_types[d,a][label]+=1
            for i in expected_inputs:
                v=index[a,i];qc=index['qC',i]['deployed_work'];qm=index['feedback',i]['deployed_work']
                require(v['status']=='EXECUTED' and np.isfinite([v['deployed_work'],qc,qm]).all(), 'Invalid work observation')
                require(qc>0 and v['deployed_work']>=0 and qm>=0, 'Nonpositive comparator work')
                work[d,a].append((float(v['deployed_work']),float(qc),float(qm)))
                if a=='E6':
                    trace=next(t for t in traces if t['arm']=='E6' and int(t['episode_id'])==i)
                    if e6['kind']=='policy':
                        require(e6['candidate']['hash']==exp['comparator']['hash'], 'E6 non-S3 variant is not comparator retention')
                        mask=raw['input_id']==i
                        require(np.array_equal(raw['E6'][mask],raw['qC'][mask]) and v['deployed_work']==qc, 'Retained comparator changes predictions or cost')
                        reserve_records.append({'dataset':d,'seed':s,'split':sp,'input_id':i,'outcome':'comparator_selected_on_development','work':float(qc)})
                    else:
                        require(e6['kind']=='s3' and len(trace['member_traces'])==1,'Expected one ensemble-wide S3 ledger')
                        checked=operation_trace(trace['member_traces'][0]);require(np.isclose(checked['work'],v['deployed_work']),'E6 operation sum differs from input cost')
                        reserve_records.append(dict(dataset=d,seed=s,split=sp,input_id=i,**checked))
        contract=exp['budget'];common=(contract['expected_ratio'],contract['tolerance'],contract['cost_basis'])
        require(d not in means or means[d]==common,'A common ceiling/cost basis is not shared across replicates')
        means[d]=common;scopes[d]=exp['input_scope']
    require(seen==set(lookup),'Extra campaign observations without a source study')
    analysis=study.get('analysis', {})
    require(analysis.get('method')=='crossed_bootstrap_signflip' and analysis.get('units')==UNITS and analysis.get('unit_weighting')=='equal','Declare the empirical seed/unit analysis')
    args={k:analysis[k] for k in ('seed','resamples','alpha')}
    summary=[]
    for d in sorted(means):
        for a in ARMS:
            obs=[r for r in observations if r['dataset']==d and r['arm']==a]
            x,seeds,units=matrix_from_rows(obs,lambda r:float(r['arm_loss'])-float(r['qm_loss']))
            if min(x.shape)<2:
                if strict: raise ValueError('At least two seeds and units required for inference')
                continue
            effect=paired_effect(x,**args);w=np.asarray(work[d,a]);rat=float(w[:,0].mean()/w[:,1].mean());ref=float(w[:,2].mean()/w[:,1].mean())
            ceiling,tol,basis=means[d]
            # Keep an over-budget outcome in partial reports but never qualify it.
            eligible=max(rat,ref)<=ceiling*(1+tol)+1e-12
            require(not strict or eligible, f'{d}/{a}: comparison expected work exceeds the common ceiling; no post-comparison reselection')
            summary.append(dict(dataset=d,arm=a,configuration='development-selected per controller replicate',**effect,
                effect_vs_qC=float(np.mean([float(r['arm_loss'])-float(r['qc_loss']) for r in obs])),
                work_ratio=rat,reference_work_ratio=ref,ceiling=ceiling,budget_mismatch=rat/ceiling-1,tolerance=tol,
                work_inputs=len(w),within_ceiling=eligible,seed_scope=study['seed_scope'],input_scope=scopes[d],
                selected_configuration_types=dict(selection_types[d,a]),
                inference_scope=('descriptive dependent-unit comparison' if d in ('ogbn-arxiv','ogbn-proteins','Roman-empire') else 'conditional fixed-parent graph-unit comparison')))
        family=[r for r in summary if r['dataset']==d]
        if family:
            for row,p in zip(family,holm([r['p_raw'] for r in family])):row['p_holm']=p
    require(not strict or len(summary)==24, 'Need 24 complete original contrasts')
    return {'check':'budget_matched','passed':True,'summary':summary,'coverage':cov,'reserve_check':{'episodes':len(reserve_records),'episode_work':reserve_records,'reserve_trace_passed':True},
        'scope':'new experiment: development-selected controller-seed procedure conditional on frozen parent ensembles; not historical S46 reconstruction',
        'seed_scope':study['seed_scope'],'historical_table_reproduction':False,
        'hard_budget_scope':'S3 native input ledgers checked separately from expected-work means; preselected comparator is not an E6 gain'}


def replay_campaign(root, study, *, device='cpu', trust=False):
    """Reexecute every archived arm study; a stored PASS file is not consulted."""
    require(trust, 'Explicit trust required for archived parent/decoder checkpoint execution')
    from cm_experiments.replay import checkpoint_replay
    from cm_experiments.holdout import CAMPAIGN_SCHEMA
    root = Path(root).resolve()
    require(study.get('schema') == CAMPAIGN_SCHEMA, 'Arm replay needs the executable campaign schema')
    require(bool(study.get('studies')), 'No archived arm studies to replay')
    reports = []
    for item in study['studies']:
        folder = (root/item['study']).resolve(); parent = (root/item['parent']).resolve()
        require(folder.is_relative_to(root) and parent.is_relative_to(root), 'Replay path escapes the archive')
        result = checkpoint_replay(folder, parent, device=device, trust=True)
        require(result.get('passed') is True, 'Actual arm checkpoint execution did not pass')
        reports.append({'dataset': item['dataset'], 'seed': item['seed'], 'split': item.get('split',0),
                        'reexecution': result})
    return {'passed': True, 'studies_reexecuted': len(reports), 'reports': reports,
            'scope': 'actual decoder and frozen-predictor execution in this check, not a prewritten replay receipt'}
