"""One Slurm allocation ledger for S5, S15/S15b and S41.

Allocation-hours are not GPU utilization-hours. Job steps are never added to
allocation rows. Typed GPU TRES and generic gres/gpu are reconciled, not summed.
Original failed/pruned jobs remain in total search cost.
"""
from __future__ import annotations
from collections import defaultdict
import csv
from pathlib import Path
import re
from .common import require, EvidenceError, number, integer, sha256, rows

FIELDS=('Cluster','JobIDRaw','JobID','Submit','Start','End','State','ElapsedRaw','AllocTRES','ExitCode')
PHASES={'backbone_training','hpo','explore','controller_fit','select','confirm','compare','admit','evaluate','other'}

def gpu_count(tres: str) -> int:
    values={}
    for token in tres.split(','):
        if not token.strip():continue
        require('=' in token, f'Malformed AllocTRES token: {token}')
        key,value=token.strip().split('=',1)
        require(key not in values, f'Duplicate AllocTRES key: {key}')
        values[key]=value
    typed=[integer(v,'GPU allocation') for k,v in values.items() if k.startswith('gres/gpu:')]
    if 'gres/gpu' in values:
        total=integer(values['gres/gpu'],'GPU allocation')
        require(not typed or sum(typed)==total,'Generic and typed GPU allocations disagree')
        return total
    return sum(typed)

def job_key(row: dict) -> str:
    # JobIDRaw can be a task's internal numeric ID; retain JobID as well.
    return '|'.join(str(row[k]) for k in ('Cluster','JobIDRaw','Submit','Start'))

def read_sacct(path: Path) -> dict:
    with path.open(newline='',encoding='utf-8-sig') as stream:
        reader=csv.DictReader(stream,delimiter='|')
        require(set(FIELDS)<=set(reader.fieldnames or []), f'Slurm export needs columns {FIELDS}')
        data=list(reader)
    require(data,'Empty sacct export')
    allocations={};steps=[];duplicates=0
    for row in data:
        require(None not in row,'Malformed sacct row')
        row={k:(v or '').strip() for k,v in row.items() if k}
        jid=row['JobID']
        if '.' in jid or '.' in row['JobIDRaw']:
            steps.append(row);continue
        require(re.fullmatch(r'\d+(?:_\d+)?(?:\+\d+)?',jid) is not None,
                f'Unexpanded array or unsupported allocation JobID: {jid}')
        require(row['Cluster'] and row['Submit'] and row['Start'], 'Accounting identity lacks Cluster/Submit/Start')
        require('[' not in jid and ']' not in jid,'Expand job arrays in original sacct export')
        elapsed=number(row['ElapsedRaw'],'ElapsedRaw',minimum=0)
        require(row['State'].split()[0] not in ('RUNNING','PENDING','CONFIGURING','COMPLETING'),
                f'Unfinished allocation: {jid}')
        gpus=gpu_count(row['AllocTRES'])
        key=job_key(row)
        computed={'key':key,'job_id':jid,'job_id_raw':row['JobIDRaw'], 'cluster':row['Cluster'],
                  'state':row['State'],'exit_code':row['ExitCode'],'seconds':elapsed,
                  'allocated_gpus':gpus,'gpu_hours':elapsed*gpus/3600.0,'raw':row}
        if key in allocations:
            require(allocations[key]==computed,f'Conflicting repeated allocation row: {key}')
            duplicates+=1
        else:allocations[key]=computed
    require(allocations,'No completed allocation rows in sacct export')
    return {'source_sha256':sha256(path),'allocations':allocations,'ignored_step_rows':len(steps),
            'identical_duplicate_rows':duplicates,
            'total_gpu_hours':sum(x['gpu_hours'] for x in allocations.values()),
            'scope':'allocated GPU count times allocation elapsed time; includes failed and pruned attempts'}

def allocate(path: Path, mapping: Path) -> dict:
    ledger=read_sacct(path)
    mapped=rows(mapping,('job_key','dataset','method','phase','seed','allocation_fraction'))
    allocations=ledger['allocations']; fractions=defaultdict(float); groups=defaultdict(float);seen=set()
    detail=[]
    for row in mapped:
        key=row['job_key'];require(key in allocations, f'Job map references unknown allocation: {key}')
        require(row['phase'] in PHASES, f'Unknown accounting phase: {row["phase"]}')
        fraction=number(row['allocation_fraction'],'allocation_fraction',minimum=0,maximum=1)
        require(fraction>0,'Zero allocation fraction is not an evidence binding')
        ident=(key,row['dataset'],row['method'],row['phase'],row['seed'])
        require(ident not in seen,'Duplicate job-map attribution');seen.add(ident)
        fractions[key]+=fraction
        require(fractions[key]<=1+1e-12,f'Allocation counted more than once: {key}')
        group=(row['dataset'],row['method'],row['phase'],row['seed'])
        hours=allocations[key]['gpu_hours']*fraction;groups[group]+=hours
        detail.append({**row,'gpu_hours':hours,'seconds':allocations[key]['seconds'],
                       'allocated_gpus':allocations[key]['allocated_gpus'],'state':allocations[key]['state']})
    require(set(fractions)==set(allocations),'Some original allocation rows are unattributed; assign them explicitly, including failures')
    require(all(abs(v-1)<=1e-10 for v in fractions.values()),'All allocation fractions must sum to one; explain shared work explicitly')
    return {'sacct_sha256':ledger['source_sha256'],'job_map_sha256':sha256(mapping),
            'ignored_step_rows':ledger['ignored_step_rows'],'duplicate_rows':ledger['identical_duplicate_rows'],
            'total_gpu_hours':ledger['total_gpu_hours'],'detail':detail,
            'groups':[dict(dataset=k[0],method=k[1],phase=k[2],seed=k[3],gpu_hours=v) for k,v in sorted(groups.items())],
            'scope':ledger['scope']}

def table_views(ledger: dict) -> dict:
    """Different table views refer to the same allocation IDs and source hash."""
    s5=defaultdict(list); s15=defaultdict(lambda:defaultdict(float));s41=defaultdict(float)
    for row in ledger['groups']:
        d,m,phase=row['dataset'],row['method'],row['phase'];value=row['gpu_hours']
        s15[(d,m)][phase]+=value
        if phase=='backbone_training':s5[(d,m)].append((row['seed'],value))
        if phase in ('explore','controller_fit','select','confirm','admit'):s41[(d,phase)]+=value
    return {'accounting_source_sha256':ledger['sacct_sha256'],
            'S5':[{'dataset':d,'method':m,'recorded_seed_count':len(v),
                   'mean_gpu_hours_per_seed':sum(x[1] for x in v)/len(v),
                   'sum_gpu_hours_over_recorded_seeds':sum(x[1] for x in v)} for (d,m),v in sorted(s5.items())],
            'S15_accounting':[{'dataset':d,'method':m,'total_allocated_gpu_hours':sum(v.values()),**v} for (d,m),v in sorted(s15.items())],
            'S41':[{'dataset':d,'phase':p,'gpu_hours':v} for (d,p),v in sorted(s41.items())]}


def inclusive_views(ledger: dict, membership: Path, expected: set[tuple[str,str]]) -> list[dict]:
    """Inclusive per-method views can share plant costs without duplicating allocations.

Each original allocation is attributed once by job_map. This separate view map
specifies which components a table row includes. Thus qC and qD can both include
the same frozen plant, but their reported totals are explicitly not additive.
"""
    members=rows(membership,('dataset','method','source_method','phase'))
    groups={(r['dataset'],r['method'],r['phase']):0. for r in ledger['groups']}
    for row in ledger['groups']:groups[row['dataset'],row['method'],row['phase']]+=row['gpu_hours']
    seen=set(); views=defaultdict(lambda:defaultdict(float))
    for row in members:
        key=tuple(row[k] for k in ('dataset','method','source_method','phase'))
        require(key not in seen,'Duplicated cost component within a table view');seen.add(key)
        require((row['dataset'],row['method']) in expected,'Unexpected cost-view row')
        source=row['dataset'],row['source_method'],row['phase']
        require(source in groups,'Cost view references an unobserved accounting component')
        views[row['dataset'],row['method']][row['phase']]+=groups[source]
    require(set(views)==expected,'Cost views do not cover all reported methods')
    return [{'dataset':d,'method':m,'total_allocated_gpu_hours':sum(parts.values()),**parts,
             'cost_scope':'inclusive named components; shared plant cost makes different method rows non-additive',
             'membership_sha256':sha256(membership)} for (d,m),parts in sorted(views.items())]
