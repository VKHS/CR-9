#!/usr/bin/env python3
"""Plan, execute or export a registered controller-seed arm campaign."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('template');q.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('export');q.add_argument('--manifest',type=Path,required=True);q.add_argument('--output',type=Path,required=True);q.add_argument('--complete',action='store_true')
    q=sub.add_parser('check');q.add_argument('--directory',type=Path,required=True);q.add_argument('--partial',action='store_true')
    q=sub.add_parser('execute');q.add_argument('--manifest',type=Path,required=True);q.add_argument('--device',default='cuda');q.add_argument('--threads',type=int,default=2);q.add_argument('--epochs',type=int,default=100);q.add_argument('--trust-local-checkpoints',action='store_true');q.add_argument('--dry-run',action='store_true')
    q=sub.add_parser('replay',help='reexecute every archived decoder and frozen predictor of a complete export');q.add_argument('--directory',type=Path,required=True);q.add_argument('--device',default='cuda');q.add_argument('--trust-local-checkpoints',action='store_true')
    a=p.parse_args(argv)
    try:
        from cm_experiments.artifacts import write_json
        from cm_experiments.campaign import export_campaign, coverage, SEED_SCOPE
        if a.command=='replay':
            from cm_campaigns.arm_campaign import replay_campaign
            result=replay_campaign(a.directory,json.loads((a.directory/'study.json').read_text()),
                                   device=a.device,trust=a.trust_local_checkpoints)
            print(json.dumps(result,indent=2));return 0
        if a.command=='template':
            from cm_campaigns.contract import UNITS
            write_json(a.output, {'schema':'cm-arm-campaign-1','seed_ids':list(range(12)), 'seed_scope':SEED_SCOPE,
                'analysis':{'method':'crossed_bootstrap_signflip','units':UNITS,'unit_weighting':'equal','seed':1701,'resamples':10000,'alpha':.05},
                'studies':[], 'run_links':{}, 'source':{},
                'instructions':'Add actual study/parent paths and original accounting/job_log paths; no completed study is implied by this empty declaration.'})
            result={'written':str(a.output),'executed_studies':0}
        elif a.command=='export':result=export_campaign(a.manifest,a.output,complete=a.complete)
        elif a.command=='check':
            from cm_campaigns.arm_campaign import check_campaign
            result=check_campaign(a.directory,json.loads((a.directory/'study.json').read_text()),strict=not a.partial)
        else:
            if not a.dry_run and not a.trust_local_checkpoints:raise ValueError('Explicit trust required')
            spec=json.loads(a.manifest.read_text())
            if spec.get('schema')!='cm-arm-campaign-1' or spec.get('seed_scope')!=SEED_SCOPE:
                raise ValueError('The campaign must declare its schema and controller-seed scope')
            if not spec.get('studies'):raise ValueError('No campaign studies are registered')
            cov=coverage(spec['studies'],spec['seed_ids']);commands=[]
            for row in spec['studies']:
                base=a.manifest.resolve().parent
                parent=(base/row['parent']).resolve();output=(base/row['study']).resolve()
                cmd=[sys.executable,'-B',str(ROOT/'scripts/run_budget_arms.py'),'--run',str(parent),'--output',str(output),
                     '--seed',str(row['seed']),'--epochs',str(a.epochs),'--threads',str(a.threads),'--device',a.device,'--trust-local-checkpoints']
                for key,flag in (('expected_ratio','--budget-ratio'),('hard_cap_ratio','--hard-cap-ratio')):
                    if key in row:cmd.extend([flag,str(row[key])])
                if row.get('cost_bounds'):cmd.extend(['--cost-bounds',str((base/row['cost_bounds']).resolve())])
                commands.append(cmd)
                if not a.dry_run:
                    if output.exists():
                        from cm_experiments.replay import numeric_replay
                        numeric_replay(output)
                        exp=json.loads((output/'experiment.json').read_text())
                        from cm_experiments.artifacts import canonical, sha256
                        cfg=json.loads((parent/'config.json').read_text())
                        if (exp['training_seed']!=row['seed'] or exp['dataset']!=row['dataset'] or
                            int(cfg['data'].get('split_index',0))!=row.get('split',0) or
                            exp['base_configuration_sha256']!=canonical(cfg) or
                            exp['base_confirm_manifest_sha256']!=sha256(parent/'provenance/stages/05_confirm.json')):
                            raise ValueError('Existing study identity differs from the campaign parent/task')
                        for key,field in (('expected_ratio','expected_ratio'),('hard_cap_ratio','hard_cap_ratio')):
                            if key in row and exp['budget'][field]!=row[key]:
                                raise ValueError('Existing study used a different registered ceiling/cap')
                    else:subprocess.run(cmd,check=True)
            result={'commands':commands,'coverage':cov,'executed':not a.dry_run,'note':'Sequential executions; submit each manifest row as a separate Slurm job for original allocation accounting.'}
        print(json.dumps(result,indent=2));return 0
    except (OSError,ValueError,KeyError,RuntimeError,subprocess.CalledProcessError) as e:
        print('ERROR: '+str(e),file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
