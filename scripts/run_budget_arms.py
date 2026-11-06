#!/usr/bin/env python3
"""Train and execute E3--E6 against a trusted confirmed CROSS-MORSE run."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--verify',type=Path)
    p.add_argument('--replay',type=Path)
    p.add_argument('--report',type=Path)
    p.add_argument('--atol',type=float,default=0.)
    p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--trust-local-checkpoints',action='store_true')
    p.add_argument('--allow-fixture',action='store_true')
    p.add_argument('--allow-posthoc',action='store_true')
    p.add_argument('--max-graphs',type=int,default=0)
    p.add_argument('--oracle-proposals',type=int,default=0)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--patience',type=int,default=15)
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--budget-ratio',type=float,default=None,help='Expected deployed-work ratio; defaults to the dataset S46 ceiling')
    p.add_argument('--hard-cap-ratio',type=float,default=None,help='Separate S3 hard cap; defaults to the expected ratio, never silently enlarged')
    p.add_argument('--budget-tolerance',type=float,default=.01)
    p.add_argument('--selection-fraction',type=float,default=.25)
    p.add_argument('--learning-rates',type=float,nargs='+',default=[.001,.0003])
    p.add_argument('--cost-bounds',type=Path)
    a=p.parse_args(argv)
    existed=bool(a.output and a.output.exists())
    try:
        if a.verify or a.replay:
            from cm_experiments.replay import numeric_replay,checkpoint_replay
            if a.replay:
                if not a.run:p.error('--run is required for checkpoint replay')
                import torch
                torch.set_num_threads(a.threads)
                result=checkpoint_replay(a.replay,a.run,device=a.device,
                    trust=a.trust_local_checkpoints,atol=a.atol)
            else:result=numeric_replay(a.verify)
            if a.report:
                from cm_experiments.artifacts import write_json
                write_json(a.report,result)
            print(json.dumps(result,indent=2));return 0
        if not a.run or not a.output:p.error('--run and --output are required for execution')
        if a.threads<1:raise ValueError('Positive thread count required')
        import torch
        torch.set_num_threads(a.threads)
        from cm_experiments.runner import run_study
        out=run_study(a.run,a.output,device=a.device,trust=a.trust_local_checkpoints,
            allow_fixture=a.allow_fixture,allow_posthoc=a.allow_posthoc,max_graphs=a.max_graphs,
            proposal_limit=a.oracle_proposals,epochs=a.epochs,patience=a.patience,
            training_seed=a.seed,budget_ratio=a.budget_ratio,hard_cap_ratio=a.hard_cap_ratio,
            costs_path=a.cost_bounds,budget_tolerance=a.budget_tolerance,
            selection_fraction=a.selection_fraction,learning_rates=a.learning_rates)
        print(json.dumps({'output':str(out),'experiment_executed':True,'publication_deposit':False},indent=2))
        return 0
    except (ValueError,OSError,RuntimeError) as exc:
        if a.output and not existed and a.output.is_dir() and (a.output/'experiment.json').is_file():
            from cm_experiments.artifacts import write_json
            path=a.output/'FAILURE.json'
            if not path.exists():
                write_json(path,{'status':'FAILED','exception':type(exc).__name__,
                    'message':str(exc),'diagnostics':getattr(exc,'diagnostics',{}),
                    'scope':'partial experiment; no result qualification'})
        print('ERROR: '+str(exc),file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
