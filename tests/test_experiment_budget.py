"""Executable reserve invariants. These are software fixtures, not benchmark records."""
import math
import unittest
from unittest.mock import patch
from cm_experiments.budget import Certificate,WorkLedger,WorkBoundViolation,Unaffordable,run_budget_episode
from cm_campaigns.arm_campaign import operation_trace

class Backend:
    def __init__(self,supported=(0,1),actual=1):
        self.calls=[];self.supported=supported;self.actual=actual
    def op(self,name,value=None,work=None):
        self.calls.append(name);return value,float(self.actual if work is None else work)
    def initialize(self):return self.op('init')
    def measure(self,d):return self.op('measure')
    def score(self,d):return self.op('score',{'supported':list(self.supported),'action':1,'recovery':0 if 0 in self.supported else 8,'gap':1.})
    def acquire(self,d):return self.op('acquire',1)
    def act(self,a,d):return self.op('action:'+str(a))
    def predict(self):return self.op('prediction','controller')
    def restart(self):return self.op('restart','comparator',5)

class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.costs={'initialization':1,'cheap_measurement':1,'score':1,'acquisition':1,'action':1,'prediction':1,'comparator':5}
    def run_backend(self,budget=30,backend=None,depth=3,certificate=None):
        b=backend or Backend();p,t=run_budget_episode(b,depth=depth,budget=budget,costs=self.costs,certificate=certificate)
        self.assertLessEqual(sum(x['paid_work'] for x in t['operations']),budget)
        replay=operation_trace(t)
        self.assertEqual(replay['work'],sum(x['paid_work'] for x in t['operations']))
        self.assertTrue(replay['operation_bounds_replayed'])
        return b,p,t
    def test_all_normal_operations_are_charged(self):
        b,p,t=self.run_backend();self.assertEqual(p,'controller');self.assertEqual(t['status'],'COMPLETED')
        self.assertEqual(t['acquisitions'],3);self.assertEqual(t['operations'][-1]['cumulative_work'],14)
    def test_infeasible_does_not_execute_anything(self):
        b,p,t=self.run_backend(4);self.assertIsNone(p);self.assertEqual(b.calls,[]);self.assertEqual(t['status'],'INFEASIBLE')
    def test_static_fallback_executes_only_original_comparator(self):
        b,p,t=self.run_backend(5);self.assertEqual(b.calls,['restart']);self.assertEqual(p,'comparator');self.assertEqual(t['status'],'STATIC_FALLBACK')
    def test_sensing_and_scoring_are_prechecked_together(self):
        b,p,t=self.run_backend(7);self.assertEqual(b.calls,['init','restart']);self.assertEqual(t['status'],'LATE_FALLBACK')
    def test_unaffordable_acquisition_uses_registered_recovery(self):
        b,p,t=self.run_backend(9);self.assertIn('action:0',b.calls);self.assertNotIn('acquire',b.calls)
        self.assertEqual(p,'comparator');self.assertEqual(t['unaffordable_acquisitions'],1)
    def test_empty_support_recovers_without_acquisition(self):
        b,p,t=self.run_backend(30,Backend(supported=()));self.assertNotIn('acquire',b.calls);self.assertEqual(b.calls.count('action:8'),3)
    def test_valid_certificate_skips_acquisition(self):
        cert=lambda d,s:Certificate(1.,1.,.01,support_invariant=True,recovery_invariant=True,search_invariant=True,bound_source='registered-test-bound')
        b,p,t=self.run_backend(certificate=cert);self.assertEqual(t['certified_skips'],3);self.assertNotIn('acquire',b.calls)
    def test_gap_alone_is_not_certificate(self):
        b,p,t=self.run_backend(certificate=lambda d,s:Certificate(1.,1.,.01));self.assertEqual(t['certified_skips'],0);self.assertEqual(t['acquisitions'],3)
    def test_final_site_does_not_release_reserve_early(self):
        b,p,t=self.run_backend(12,depth=3);self.assertEqual(p,'comparator');self.assertNotIn('prediction',b.calls)
        self.assertLessEqual(t['operations'][-1]['cumulative_work'],12)
    def test_bound_violation_fails_instead_of_claiming_cap(self):
        with self.assertRaises(WorkBoundViolation):self.run_backend(30,Backend(actual=2))
    def test_negative_nan_infinite_work_rejected(self):
        for value in (-1,float('nan'),float('inf'),True):
            with self.subTest(value=value),self.assertRaises(ValueError):WorkLedger(value,5,5)
    def test_restart_bound_not_more_than_reserve(self):
        with self.assertRaises(ValueError):WorkLedger(20,4,5)
    def test_no_callbacks_after_terminal(self):
        ledger=WorkLedger(20,5,5);ledger.restart(lambda:('qC',5))
        with self.assertRaises(ValueError):ledger.pay('action',1,lambda:('bad',1))
    def test_excess_terminal_prediction_rejected(self):
        ledger=WorkLedger(20,5,5)
        with self.assertRaises(ValueError):ledger.pay('prediction',6,lambda:(None,6),terminal=True)
    def test_missing_cost_rejected(self):
        self.costs.pop('score')
        with self.assertRaises(ValueError):self.run_backend()
    def test_unaffordable_callback_not_executed(self):
        ledger=WorkLedger(5,5,5);called=[]
        with self.assertRaises(Unaffordable):ledger.pay('action',1,lambda:called.append(1))
        self.assertEqual(called,[])
    def test_certificate_boundaries(self):
        self.assertFalse(Certificate(.002,0,0,support_invariant=True,recovery_invariant=True,search_invariant=True,bound_source='x').allows_skip())
        self.assertTrue(Certificate(math.inf,1,.1,support_invariant=True,recovery_invariant=True,search_invariant=True,bound_source='x').allows_skip())
        for v in (math.nan,-1):
            with self.assertRaises(ValueError):Certificate(v,1,.1).allows_skip()
    def test_independent_invariance_premises(self):
        for missing in ('support_invariant','recovery_invariant','search_invariant'):
            kw=dict(support_invariant=True,recovery_invariant=True,search_invariant=True,bound_source='bound');kw[missing]=False
            self.assertFalse(Certificate(5.,1.,.1,**kw).allows_skip())
    def test_no_bound_source_no_certificate(self):
        self.assertFalse(Certificate(5.,1.,.1,support_invariant=True,recovery_invariant=True,search_invariant=True).allows_skip())

    @staticmethod
    def certificate(gap):
        return Certificate(gap,1.,.01,support_invariant=True,recovery_invariant=True,
                           search_invariant=True,bound_source='registered-test-bound')

    def test_certificate_cannot_replace_observed_gap(self):
        # The actual margin is one. A callback must not certify a larger
        # margin even if its regional bound and invariance flags are present.
        backend=Backend()
        with self.assertRaisesRegex(ValueError,'exceeds the observed'):
            self.run_backend(backend=backend,certificate=lambda d,s:self.certificate(2.))
        self.assertFalse(any(name.startswith('action:') for name in backend.calls))

    def test_conservative_observed_gap_bound_can_skip(self):
        b,p,t=self.run_backend(certificate=lambda d,s:self.certificate(.5))
        self.assertEqual(t['certified_skips'],3)
        self.assertTrue(all(row['origin']=='certified_skip' for row in t['operations'] if row['operation']=='action'))

    def test_infinite_gap_requires_singleton_supported_set(self):
        backend=Backend()
        with patch.object(backend,'score',side_effect=lambda d:backend.op(
                'score',{'supported':[0,1],'action':1,'recovery':0,'gap':math.inf})):
            with self.assertRaisesRegex(ValueError,'singleton support'):
                self.run_backend(backend=backend,certificate=lambda d,s:self.certificate(math.inf))
        self.assertFalse(any(name.startswith('action:') for name in backend.calls))

    def test_singleton_gap_is_certified_with_actual_singleton_observation(self):
        backend=Backend(supported=(1,))
        with patch.object(backend,'score',side_effect=lambda d:backend.op(
                'score',{'supported':[1],'action':1,'recovery':8,'gap':math.inf})):
            b,p,t=self.run_backend(backend=backend,certificate=lambda d,s:self.certificate(math.inf))
        self.assertEqual(t['certified_skips'],3)
        self.assertEqual(b.calls.count('action:1'),3)

    def test_invalid_observed_gap_never_certifies_a_skip(self):
        for gap in (None,True,'1',math.nan,-1.):
            backend=Backend()
            with self.subTest(gap=gap),patch.object(backend,'score',side_effect=lambda d:backend.op(
                    'score',{'supported':[0,1],'action':1,'recovery':0,'gap':gap})):
                with self.assertRaisesRegex(ValueError,'observed score gap'):
                    self.run_backend(backend=backend,certificate=lambda d,s:self.certificate(1.))
            self.assertFalse(any(name.startswith('action:') for name in backend.calls))

    def test_acquisition_action_ids_are_not_silently_coerced(self):
        for action in (True,1.5,'1',10,-1):
            backend=Backend()
            with self.subTest(action=action),patch.object(backend,'acquire',side_effect=lambda d:backend.op('acquire',action)):
                with self.assertRaisesRegex(ValueError,'Acquired action'):
                    self.run_backend(backend=backend)
            self.assertFalse(any(name.startswith('action:') for name in backend.calls))

    def test_recovery_action_ids_are_not_silently_coerced(self):
        for action in (True,0.5,'0'):
            backend=Backend()
            with self.subTest(action=action),patch.object(backend,'score',side_effect=lambda d:backend.op(
                    'score',{'supported':[0,1],'action':1,'recovery':action,'gap':1.})):
                with self.assertRaisesRegex(ValueError,'Recovery action'):
                    self.run_backend(backend=backend)
            self.assertNotIn('acquire',backend.calls)

    def test_duplicate_support_cannot_claim_a_score_gap(self):
        backend=Backend(supported=(0,1,1))
        with self.assertRaisesRegex(ValueError,'support set'):
            self.run_backend(backend=backend)
        self.assertNotIn('acquire',backend.calls)

    def test_every_executed_site_has_one_origin(self):
        cases=[(30,Backend(),'acquired_policy'),(30,Backend(supported=()),'recovery_no_support'),
               (9,Backend(),'recovery_unaffordable_acquisition')]
        for budget,backend,origin in cases:
            with self.subTest(origin=origin):
                b,p,t=self.run_backend(budget,backend)
                rows=[row for row in t['operations'] if row['operation']=='action']
                self.assertTrue(rows)
                self.assertEqual({row['origin'] for row in rows},{origin})

if __name__=='__main__':unittest.main()
