"""Actual frozen-plant execution and information boundaries for E3--E6."""
import copy,inspect,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
try:import torch
except ImportError:raise unittest.SkipTest('PyTorch required for frozen-plant arm execution')
from cross_morse.core import FrozenPlant,GraphOperator
from cm_experiments.schedules import DecoderSpec,ScheduleDecoder
from cm_experiments.execution import execute_arm,EnsembleEpisode,initial_features
from cm_experiments.budget import run_budget_episode
from cm_experiments.artifacts import seal,verify,write_json,fresh_output
from cm_campaigns.arm_campaign import operation_trace

class Bank:
    def __init__(self):self.counters={'dense_multiply_adds':0}
    def prepare(self,h,p):pass
    def measure(self,h,p,d,L,mass,hops):
        self.counters['dense_multiply_adds']+=5
        y=torch.zeros(25);y[0]=.5;y[5]=d/L;return y
class Controller:
    def __init__(self):self.reset_counters()
    def reset_counters(self):
        self.counters={'approx_dense_multiply_adds':0,'model_forward_calls':0};self.observations=[]
    def start(self,y,previous_action=-1):
        self.counters['approx_dense_multiply_adds']+=3;return SimpleNamespace(y=np.array(y),depth=0,valid=True)
    def observe(self,s,y,d,a):
        self.observations.append(d);return SimpleNamespace(y=np.array(y),depth=d,valid=True)
    def supported(self,y,d):return [0,1,8]
    def score(self,s,actions,**kw):
        self.counters['approx_dense_multiply_adds']+=2
        return {a:{'score':float(a!=1),'sigma':0.} for a in actions}
    def predict(self,s,a):
        self.counters['approx_dense_multiply_adds']+=2;return SimpleNamespace(y=s.y.copy(),depth=s.depth+1,valid=True)
    def stage_cost(self,s,a):return .001*a
    def terminal_value(self,s):return .01

class ArmExecutionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(2);self.x=torch.randn(8,4)
        e=torch.stack([torch.arange(8),(torch.arange(8)+1)%8]);self.g=GraphOperator(torch.cat([e,e.flip(0)],1),8)
        self.m=FrozenPlant(4,8,3,dropout=0).freeze()
        self.cfg={'model':{'depth':4,'precision':'fp32'},'sensor':{'degree':8},
                  'controller':{'max_search_calls':1000,'gamma':.97,'lambda_rho':.0001,'lambda_transport':.0001,'lambda_terminal':1,'top_k':2,'beam_width':2}}
        self.c={'route':'mpc_rh','profile':'full','depth':4,'horizon':2,'ensemble':1}
    def run_arm(self,arm,**kw):
        c=Controller()
        with patch('cm_experiments.execution.sensor_bank',side_effect=lambda *a,**k:Bank()):
            p,t=execute_arm(arm,self.m,self.x,self.g,self.cfg,c,self.c,**kw)
        return p,t,c
    def test_e5_never_assimilates_later_observations(self):
        p,t,c=self.run_arm('E5');self.assertEqual(c.observations,[])
        self.assertEqual([r['depth'] for r in t['measurements']],[0]);self.assertEqual(len(t['decision_trace']),4)
        self.assertEqual(p.shape,(8,3));self.assertTrue(np.allclose(p.sum(1),1))
    def test_e5_changes_no_frozen_weight(self):
        before={k:v.clone() for k,v in self.m.state_dict().items()};self.run_arm('E5')
        self.assertTrue(all(torch.equal(v,self.m.state_dict()[k]) for k,v in before.items()))
    def test_e3_and_e4_decode_once_without_later_measurements(self):
        net=ScheduleDecoder(DecoderSpec(12,8,8,4)).eval().requires_grad_(False)
        for arm in ('E3','E4'):
            p,t,c=self.run_arm(arm,decoder=net);self.assertEqual([r['depth'] for r in t['measurements']],[0])
            self.assertEqual(c.observations,[]);self.assertTrue(t['plan_fixed_before_execution'])
            self.assertEqual(len(t['actions']),4);self.assertNotIn(9,t['actions'])
    def test_unfrozen_model_rejected(self):
        self.m=FrozenPlant(4,8,3,dropout=0)
        with self.assertRaises(ValueError):self.run_arm('E5')
    def test_invalid_arm_rejected(self):
        with self.assertRaises(ValueError):self.run_arm('E7')
    def test_initial_features_are_mean_and_four_graph_statistics(self):
        h=torch.ones(8,8);a=initial_features(h,self.g);self.assertEqual(a.shape,(12,));np.testing.assert_array_equal(a[:8],np.ones(8))
    def test_execution_api_has_no_label_handle(self):
        args=set(inspect.signature(execute_arm).parameters);self.assertFalse(args&{'labels','data','dataset','target','y'})

class Member:
    L=2
    def __init__(self,index):self.index=index;self.calls=[];self.comparator_trace={}
    def initialize(self):self.calls.append('init');return None,1.
    def measure(self,d):self.calls.append('measure');return None,1.
    def score(self,d):return {'supported':[0],'action':0,'recovery':0,'gap':0.},1.
    def acquire(self,d):self.calls.append('acquire');return 0,1.
    def act(self,a,d):self.calls.append('act');return None,1.
    def predict(self):self.calls.append('predict');return np.array([[.2,.8]],dtype=np.float32),1.
    def restart(self):self.calls.append('restart');return np.array([[.8,.2]],dtype=np.float32),5.

class EnsembleTests(unittest.TestCase):
    def test_late_abort_restarts_all_members_not_only_current(self):
        members=[Member(0),Member(1)];backend=EnsembleEpisode(members)
        costs=dict(initialization=2,cheap_measurement=1,score=1,acquisition=1,action=1,prediction=2,comparator=10)
        p,t=run_budget_episode(backend,depth=4,budget=16,costs=costs)
        self.assertEqual(t['status'],'LATE_FALLBACK')
        self.assertEqual(members[0].calls.count('restart'),1);self.assertEqual(members[1].calls.count('restart'),1)
        np.testing.assert_array_equal(p,np.array([[.8,.2]],dtype=np.float32))
        self.assertEqual(t['operations'][-1]['comparator_reserve'],10)
    def test_joint_success_predicts_all_only_after_final_actions(self):
        members=[Member(0),Member(1)];backend=EnsembleEpisode(members)
        costs=dict(initialization=2,cheap_measurement=1,score=1,acquisition=1,action=1,prediction=2,comparator=10)
        p,t=run_budget_episode(backend,depth=4,budget=60,costs=costs)
        self.assertEqual(t['status'],'COMPLETED');self.assertTrue(all(m.calls[-1]=='predict' for m in members))
    def test_member_count_is_checked(self):
        with self.assertRaises(ValueError):EnsembleEpisode([])


class TrainedExampleBudgetTests(unittest.TestCase):
    def test_real_example_ensemble_ledgers_replay_on_success_fallback_and_infeasibility(self):
        """Use actual synthetic-example checkpoints, sensors and observers.

        This is an integration test of accounting and prediction identity, not
        a benchmark claim. It catches stricter replay rejecting valid runtime
        traces that use real product counters and ensemble-wide restarts.
        """
        from cross_morse.data import load_data
        from cross_morse.training import load_plant
        from cross_morse.learning import Controller as TrainedController
        from cross_morse.runtime import graph_operator
        from cross_morse.policies import execute_policy
        from cm_experiments.execution import execute_budget_ensemble
        root=Path(__file__).resolve().parents[1]/'examples/runs/arxiv'
        cfg=json.loads((root/'config.json').read_text())
        data=load_data(root).protect_labels();torch.set_num_threads(1)
        models=[load_plant(data,cfg,root/'train',s,'cpu') for s in cfg['training']['seeds']]
        candidates=json.loads((root/'select/candidates.json').read_text())
        teacher=next(c for c in candidates if c['route']=='mpc_rh' and c['profile']=='full'
                     and c['ensemble']==len(models) and c['horizon']==2)
        comparator=next(c for c in candidates if c['route']=='nominal' and c['ensemble']==len(models))
        with np.load(root/'explore/nominal_reference.npz',allow_pickle=False) as source:
            reference=dict(source)
        def controls(profile):
            return [TrainedController.load(root/f'fit/{profile}/controller.pt','cpu') for _ in models]
        def run(cap):
            return execute_budget_ensemble(models,data.x,graph_operator(data,'cpu'),cfg,
                controls('full'),controls('lean'),teacher,comparator,controls('full'),reference,budget=cap)
        prediction,trace=run(1e15)
        self.assertEqual(trace['status'],'COMPLETED')
        self.assertGreater(trace['acquisitions'],0)
        self.assertEqual(operation_trace(trace)['work'],trace['deployed_multiply_add_work'])
        self.assertTrue(np.isfinite(prediction).all())
        cap=trace['cost_bounds']['comparator']
        fallback,trace=run(cap)
        self.assertEqual(trace['status'],'STATIC_FALLBACK')
        self.assertEqual(operation_trace(trace)['work'],trace['deployed_multiply_add_work'])
        original=[]
        for model,control in zip(models,controls('full')):
            value,_=execute_policy(model,data.x,graph_operator(data,'cpu'),cfg,comparator,control,reference)
            original.append(value)
        expected=np.mean(np.stack(original).astype(np.float64),axis=0).astype(np.float32)
        np.testing.assert_array_equal(fallback,expected)
        prediction,trace=run(cap-1)
        self.assertIsNone(prediction)
        self.assertEqual(trace['status'],'INFEASIBLE')
        self.assertEqual(operation_trace(trace)['work'],0.)

class ArmArtifactTests(unittest.TestCase):
    def test_hash_manifest_rejects_mutations(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);write_json(p/'a.json',{'a':1});seal(p);self.assertEqual(verify(p)['files'],1)
            (p/'a.json').write_text('{}')
            with self.assertRaises(ValueError):verify(p)
    def test_manifest_rejects_extra_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);write_json(p/'a.json',{'a':1});seal(p);(p/'extra').write_text('x')
            with self.assertRaises(ValueError):verify(p)
    def test_no_clobber(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.json';write_json(p,{'a':1})
            with self.assertRaises(FileExistsError):write_json(p,{'a':2})
    def test_output_never_inside_input_or_ancestor(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'source';p.mkdir()
            for q in (p,p/'inside',Path(d)):
                with self.subTest(q=q),self.assertRaises(ValueError):fresh_output(q,p)

if __name__=='__main__':unittest.main()
