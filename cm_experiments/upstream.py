"""Execute pinned upstream baseline adapters with complete attempt accounting.

This is orchestration, not a substitute AMP/GRAMA/Mamba implementation. Adapter
code must call the actual upstream method and emit its development/final outputs.
No reference score, trial duration, checkpoint or scheduler ID is synthesized.
"""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .artifacts import write_json, canonical, sha256, seal, verify
from cm_campaigns.contract import EXTERNAL


def validate_plan(plan):
    if plan.get('schema') != 'cm-upstream-study-1': raise ValueError('Unknown upstream plan')
    if plan.get('dataset') not in EXTERNAL or plan.get('method') not in EXTERNAL[plan['dataset']]:
        raise ValueError('This is not one of the fourteen registered dataset-method comparisons')
    trials=plan.get('trials',[]);seeds=plan.get('seed_ids',[])
    if len(trials)!=48 or any(not isinstance(c,dict) for c in trials):raise ValueError('Provide all 48 actual HPO configurations')
    if len({canonical(c) for c in trials})!=48:raise ValueError('HPO configurations must not be duplicated')
    if len(seeds)!=12 or len(set(seeds))!=12 or any(type(s)is not int or s<0 for s in seeds):raise ValueError('Provide twelve distinct optimization seeds')
    if plan.get('selection_role')!='internal_training_validation' or plan.get('direction') not in ('maximize','minimize'):
        raise ValueError('Declare internal training validation and score direction before the study')
    if plan.get('metric_unit')!='native_0_to_1':raise ValueError('Store development metrics on their native [0,1] scale')
    for name in ('source_commit','adapter_commit'):
        v=plan.get(name)
        if not isinstance(v,str) or len(v) not in (40,64) or any(x not in '0123456789abcdef' for x in v):raise ValueError('Pin the complete '+name)
    cmd=plan.get('adapter_argv')
    if not isinstance(cmd,list) or not cmd or any(not isinstance(x,str) or not x for x in cmd):raise ValueError('Adapter command must be an argument vector, not a shell string')
    if plan['dataset']=='Roman-empire' and plan.get('official_splits')!=list(range(10)):
        raise ValueError('Roman-empire final evaluation must cover all ten official splits')
    return plan


def git_identity(path,commit):
    root=Path(path).resolve()
    actual=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if actual!=commit:raise ValueError('Pinned source/adapter commit differs from working checkout')
    dirty=subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=normal'],text=True)
    if dirty:raise ValueError('Commit upstream changes and adapter patches before executing a recorded study')
    return root


def choose_trial(plan,root):
    observed=[]
    for i in range(48):
        p=Path(root)/'hpo'/f'trial-{i:03d}';verify(p)
        r=json.loads((p/'attempt.json').read_text())
        if r['plan_sha256']!=canonical(plan):raise ValueError('HPO attempts mix different plans')
        if r['phase']!='hpo' or r['index']!=i or r['seed']!=int(plan.get('search_seed',0)):
            raise ValueError('HPO attempt was relabelled')
        actual=json.loads((p/'configuration.json').read_text())
        if actual!=dict(plan['trials'][i],seed=int(plan.get('search_seed',0))):
            raise ValueError('HPO executed configuration differs from the registered trial')
        if r['status']=='COMPLETE':observed.append((float(r['development_score']),i))
    if not observed:raise ValueError('All HPO attempts failed/pruned; no final configuration can be selected')
    factor=-1 if plan['direction']=='maximize' else 1
    score,index=min(observed,key=lambda r:(factor*r[0],r[1]))
    return {'selected_trial':index,'development_score':score,'selected_config':plan['trials'][index],
            'selection_role':plan['selection_role'],'rule':plan['direction']+' development score; tie lowest trial index',
            'all_48_attempts_accounted':True}


def execute_attempt(plan_path,root,*,phase,index,split=0,trust=False):
    if not trust:raise ValueError('Review and explicitly authorize the upstream adapter before executing it')
    plan_path=Path(plan_path).resolve();plan=validate_plan(json.loads(plan_path.read_text()));root=Path(root).resolve()
    base=plan_path.parent
    source=git_identity(base/plan['source_directory'],plan['source_commit'])
    adapter=git_identity(base/plan['adapter_directory'],plan['adapter_commit'])
    if root==source or root.is_relative_to(source) or root==adapter or root.is_relative_to(adapter):raise ValueError('Write experiment outputs outside both source checkouts')
    if phase=='hpo':
        if not 0<=index<48:raise ValueError('HPO index outside the preregistered study')
        config=plan['trials'][index];seed=int(plan.get('search_seed',0));parent=root/'hpo'/f'trial-{index:03d}';selection=None
    elif phase=='final':
        if not 0<=index<12:raise ValueError('Final fit index outside twelve registered seeds')
        if split not in plan.get('official_splits',[0]):raise ValueError('Unregistered split')
        selection=choose_trial(plan,root);config=selection['selected_config'];seed=plan['seed_ids'][index]
        parent=root/'final'/f'seed-{seed}'/f'split-{split}'
    else:raise ValueError('Unknown experiment phase')
    if parent.exists():
        verify(parent);r=json.loads((parent/'attempt.json').read_text())
        if r['plan_sha256']!=canonical(plan) or (r['phase'],r['index'],r['seed'],r['split'])!=(phase,index,seed,split):
            raise ValueError('Existing attempt belongs to another plan/task')
        return r
    parent.mkdir(parents=True,exist_ok=False)
    write_json(parent/'plan.json',plan);write_json(parent/'configuration.json',dict(config,seed=seed))
    if selection:write_json(parent/'development_selection.json',selection)
    # Archives are generated from the pinned commits, never represented by hashes alone.
    for kind,path,commit in [('upstream',source,plan['source_commit']),('adapter',adapter,plan['adapter_commit'])]:
        with (parent/f'{kind}_source.tar').open('xb') as f:
            subprocess.run(['git','-C',str(path),'archive','--format=tar',commit],stdout=f,check=True)
    cmd=[x.replace('{python}',sys.executable) for x in plan['adapter_argv']]+[
        '--phase',phase,'--config',str(parent/'configuration.json'),'--seed',str(seed),
        '--split',str(split),'--output',str(parent/'adapter_output'),'--upstream-root',str(source)]
    write_json(parent/'invocation.json',{'argv':cmd,'source_commit':plan['source_commit'],'adapter_commit':plan['adapter_commit'],
        'slurm':{k:v for k,v in os.environ.items() if k in ('SLURM_JOB_ID','SLURM_ARRAY_JOB_ID','SLURM_ARRAY_TASK_ID')},
        'python':sys.version,'work_directory':'pinned adapter checkout'})
    start=time.perf_counter()
    with (parent/'stdout.log').open('xb') as out,(parent/'stderr.log').open('xb') as err:
        try:rc=subprocess.run(cmd,cwd=adapter,stdout=out,stderr=err,check=False,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')).returncode
        except OSError as exc:err.write(str(exc).encode());rc=127
    elapsed=time.perf_counter()-start;status='FAILED';score=None;reason=None
    receipt=parent/'adapter_output'/'receipt.json'
    if rc==0 and receipt.is_file():
        try:
            r=json.loads(receipt.read_text());status=r.get('status','FAILED')
            if status not in ('COMPLETE','PRUNED','FAILED'):raise ValueError('Adapter returned an invalid status')
            if status=='COMPLETE':
                if phase=='hpo':
                    if r.get('scored_role')!=plan['selection_role']:raise ValueError('HPO receipt uses a prohibited data role')
                    score=float(r['development_score'])
                    if not math.isfinite(score) or not 0<=score<=1:raise ValueError('Nonfinite/out-of-range development metric')
                else:
                    # Artifact contents are scored by the independent reducer.
                    for k in ('predictions','checkpoint','split_manifest','timing_trace'):
                        item=Path(r.get(k,''));target=(receipt.parent/item).resolve()
                        if item.is_absolute() or not target.is_relative_to(receipt.parent.resolve()) or not target.is_file():
                            raise ValueError('Missing/unsafe final baseline artifact: '+k)
        except (OSError,ValueError,KeyError,TypeError) as exc:
            status='FAILED';score=None;reason='invalid adapter receipt: '+str(exc)
    else:reason='adapter failed or did not emit its receipt'
    result={'phase':phase,'index':index,'seed':seed,'split':split,'status':status,'process_exit_code':rc,
        'elapsed_seconds':elapsed,'development_score':score,'plan_sha256':canonical(plan),'reason':reason,
        'accounting_scope':'elapsed process time, NOT GPU-hours; attach terminal sacct allocation export separately'}
    write_json(parent/'attempt.json',result);seal(parent);return result
