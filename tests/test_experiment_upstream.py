"""Exercise orchestration with an explicit test adapter, never a benchmark model."""
import json, subprocess, tempfile, unittest
from pathlib import Path
from cm_experiments.upstream import validate_plan, execute_attempt, choose_trial
from cm_experiments.artifacts import verify

class UpstreamTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name);self.source=self.base/'fixture-adapter';self.source.mkdir()
  code='''import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--phase');p.add_argument('--config');p.add_argument('--seed');p.add_argument('--split');p.add_argument('--output');p.add_argument('--upstream-root');a=p.parse_args()
c=json.loads(Path(a.config).read_text());o=Path(a.output);o.mkdir()
if c['trial']==17:raise SystemExit(3)
r={'status':'COMPLETE','scored_role':'internal_training_validation','development_score':(c['trial']%4)/4}
if c.get('bad_role'):r['scored_role']='test'
if a.phase=='final':
 for k in ('predictions','checkpoint','split_manifest','timing_trace'):
  (o/(k+'.txt')).write_text('unit test only; not a benchmark artifact');r[k]=k+'.txt'
(o/'receipt.json').write_text(json.dumps(r))
'''
  (self.source/'adapter.py').write_text(code)
  for args in (['init','-q'],['config','user.name','Software test'],['config','user.email','test@example.invalid'],['add','adapter.py'],['commit','-qm','Test adapter']):subprocess.run(['git','-C',str(self.source),*args],check=True,stdout=subprocess.DEVNULL)
  commit=subprocess.check_output(['git','-C',str(self.source),'rev-parse','HEAD'],text=True).strip()
  self.plan={'schema':'cm-upstream-study-1','dataset':'ogbn-arxiv','method':'AMP','selection_role':'internal_training_validation','direction':'maximize','metric_unit':'native_0_to_1','source_directory':'fixture-adapter','adapter_directory':'fixture-adapter','source_commit':commit,'adapter_commit':commit,'adapter_argv':['{python}','-S','adapter.py'],'trials':[{'trial':i}for i in range(48)],'seed_ids':list(range(12)),'search_seed':77}
  self.path=self.base/'plan.json';self.path.write_text(json.dumps(self.plan));self.out=self.base/'runs'
 def tearDown(self):self.tmp.cleanup()
 def test_consent_required(self):
  with self.assertRaises(ValueError):execute_attempt(self.path,self.out,phase='hpo',index=0)
 def test_dirty_adapter_rejected(self):
  (self.source/'uncommitted.txt').write_text('change')
  with self.assertRaises(ValueError):execute_attempt(self.path,self.out,phase='hpo',index=0,trust=True)
 def test_failure_attempt_is_preserved(self):
  r=execute_attempt(self.path,self.out,phase='hpo',index=17,trust=True);self.assertEqual(r['status'],'FAILED');self.assertEqual(r['process_exit_code'],3)
  self.assertTrue(verify(self.out/'hpo/trial-017')['passed'])
 def test_wrong_label_role_cannot_be_complete(self):
  self.plan['trials'][0]['bad_role']=True;self.path.write_text(json.dumps(self.plan))
  r=execute_attempt(self.path,self.out,phase='hpo',index=0,trust=True);self.assertEqual(r['status'],'FAILED');self.assertIn('prohibited',r['reason'])
 def test_all_trials_then_deterministic_choice_and_seed_job(self):
  for i in range(48):execute_attempt(self.path,self.out,phase='hpo',index=i,trust=True)
  choice=choose_trial(self.plan,self.out);self.assertEqual(choice['selected_trial'],3)
  r=execute_attempt(self.path,self.out,phase='final',index=0,trust=True);self.assertEqual(r['status'],'COMPLETE')
  again=execute_attempt(self.path,self.out,phase='final',index=0,trust=True);self.assertEqual(r,again)
  self.assertTrue(verify(self.out/'final/seed-0/split-0')['passed'])
 def test_incomplete_search_cannot_select(self):
  execute_attempt(self.path,self.out,phase='hpo',index=0,trust=True)
  with self.assertRaises((ValueError,OSError)):choose_trial(self.plan,self.out)
 def test_duplicate_hpo_configs_rejected(self):
  self.plan['trials'][1]=self.plan['trials'][0]
  with self.assertRaises(ValueError):validate_plan(self.plan)
if __name__=='__main__':unittest.main()
