#!/usr/bin/env python3
"""Execute one actual pinned-upstream HPO or final-seed job; no substitute models."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT)not in sys.path:sys.path.insert(0,str(ROOT))
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--phase',choices=['hpo','final','select'],required=True);p.add_argument('--index',type=int,default=0);p.add_argument('--split',type=int,default=0);p.add_argument('--trust-upstream-code',action='store_true');a=p.parse_args()
 try:
  from cm_experiments.upstream import validate_plan,choose_trial,execute_attempt
  if a.phase=='select':result=choose_trial(validate_plan(json.loads(a.plan.read_text())),a.output)
  else:result=execute_attempt(a.plan,a.output,phase=a.phase,index=a.index,split=a.split,trust=a.trust_upstream_code)
  print(json.dumps(result,indent=2));return 0 if result.get('status')!='FAILED' else 1
 except (OSError,ValueError,KeyError,RuntimeError)as e:print('ERROR: '+str(e),file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
