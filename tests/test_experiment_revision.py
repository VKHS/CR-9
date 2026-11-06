"""Budget coordinates, whole-family proposals and safe campaign export contracts."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from cm_experiments.protocol import (budget_contract,terminal_reserve_requirement,
    choose_development,deployed_work,PAPER_CEILINGS,OPEN_LOOP)
from cm_experiments.campaign import coverage,SEED_SCOPE
from cm_campaigns.arm_campaign import operation_trace
from cm_experiments.upstream import validate_plan

class ContractTests(unittest.TestCase):
    def row(self,name='x',loss=1.,work=1.,**kw):
        return dict(configuration_id=name,role='Dexplore_selection',status='EXECUTED',support_qualified=True,
                    loss=loss,mean_work=work,mean_qc_work=1.,**kw)
    def test_per_dataset_ceilings(self):
        self.assertEqual([budget_contract(d).expected_ratio for d in PAPER_CEILINGS],[1.15,1.17,1.30,1.29,1.33,1.18])
    def test_expected_and_hard_cap_independent(self):
        x=budget_contract('ogbn-arxiv',hard_cap=2.5);self.assertEqual(x.expected_ratio,1.15);self.assertEqual(x.hard_cap_ratio,2.5)
    def test_hard_cap_not_silently_enlarged(self):self.assertEqual(budget_contract('ogbn-arxiv').hard_cap_ratio,1.15)
    def test_products_no_invented_s46_ratio(self):
        with self.assertRaises(ValueError):budget_contract('ogbn-products')
    def test_invalid_coordinates(self):
        for v in (float('nan'),float('inf'),True,-1,.9):
            with self.assertRaises(ValueError):budget_contract('ogbn-arxiv',expected=v)
    def test_prediction_cost_subtracted_in_reserve_bound(self):
        r=terminal_reserve_requirement(1.06,.02,1.)
        self.assertAlmostEqual(r['necessary_cap'],2.04);self.assertFalse(r['is_sufficient'])
    def test_unused_reserve_not_charged(self):self.assertTrue(terminal_reserve_requirement(1.,.01,1.)['unused_reserve_is_not_executed_work'])
    def test_impossible_final_prediction_rejected(self):
        with self.assertRaises(ValueError):terminal_reserve_requirement(1.,1.1,1.)
    def test_best_loss_above_ceiling_cannot_win(self):
        r=choose_development([self.row('a',.01,2),self.row('b',1,1)],budget_contract('ogbn-arxiv'));self.assertEqual(r['selected'],'b')
    def test_unqualified_candidate_cannot_win(self):
        a=self.row('a',.01,1);a['support_qualified']=False
        self.assertEqual(choose_development([a,self.row('b')],budget_contract('ogbn-arxiv'))['selected'],'b')
    def test_no_feasible_candidate_is_not_success(self):
        self.assertIsNone(choose_development([self.row(work=9)],budget_contract('ogbn-arxiv'))['selected'])
    def test_confirm_cannot_select(self):
        a=self.row();a['role']='Dconfirm'
        with self.assertRaises(ValueError):choose_development([a],budget_contract('ogbn-arxiv'))
    def test_deterministic_tie(self):self.assertEqual(choose_development([self.row('b'),self.row('a')],budget_contract('ogbn-arxiv'))['selected'],'a')
    def test_duplicate_config_rejected(self):
        with self.assertRaises(ValueError):choose_development([self.row(),self.row()],budget_contract('ogbn-arxiv'))
    def test_qualification_subtraction_exact(self):self.assertEqual(deployed_work({'multiply_add_work':3,'qualification_multiply_add_work':2,'deployed_multiply_add_work':1}),1)
    def test_false_subtraction_rejected(self):
        with self.assertRaises(ValueError):deployed_work({'multiply_add_work':3,'qualification_multiply_add_work':2,'deployed_multiply_add_work':2})
    def test_campaign_size_includes_roman_splits(self):self.assertEqual(coverage([],list(range(12)))['required_studies'],180)
    def test_campaign_missing_not_complete(self):self.assertFalse(coverage([{'dataset':'ogbn-arxiv','seed':0,'split':0}],list(range(12)))['complete'])
    def test_duplicate_study_slot_rejected(self):
        v={'dataset':'ogbn-arxiv','seed':0,'split':0}
        with self.assertRaises(ValueError):coverage([v,v],list(range(12)))
    def test_coerced_campaign_seed_rejected(self):
        with self.assertRaises(ValueError):coverage([{'dataset':'ogbn-arxiv','seed':'0','split':0}],list(range(12)))
    def test_arm_campaign_replay_needs_explicit_trust(self):
        from cm_campaigns.arm_campaign import replay_campaign
        with self.assertRaises(ValueError):replay_campaign(Path('.'),{'schema':'cm-budget-matched-2'})
    def test_arm_campaign_replay_needs_real_studies(self):
        from cm_campaigns.arm_campaign import replay_campaign
        with self.assertRaises(ValueError):replay_campaign(Path('.'),{'schema':'cm-budget-matched-3','studies':[]},trust=True)
    def test_nonexistent_official_split_rejected(self):
        with self.assertRaises(ValueError):coverage([{'dataset':'Roman-empire','seed':0,'split':10}],list(range(12)))
    def test_seed_count_must_be_twelve(self):
        with self.assertRaises(ValueError):coverage([],list(range(11)))
    def test_reserved_operation_sequence(self):
        rows=[dict(operation_index=0,operation='action',paid_work=.2,operation_bound=.2,cumulative_work=.2,budget=1.3,comparator_reserve=1.,comparator_bound=1.,outcome='continue',prechecked=True,work_unit='multiply_add_products'),
              dict(operation_index=1,operation='prediction',paid_work=.05,operation_bound=.05,cumulative_work=.25,budget=1.3,comparator_reserve=1.,comparator_bound=1.,outcome='completed',prechecked=True,work_unit='multiply_add_products')]
        self.assertEqual(operation_trace({'operations':rows})['work'],.25)
    def test_reserve_released_early_rejected(self):
        row=dict(operation_index=0,operation='action',paid_work=.4,operation_bound=.4,cumulative_work=.4,budget=1.3,comparator_reserve=1.,comparator_bound=1.,outcome='continue',prechecked=True,work_unit='multiply_add_products')
        with self.assertRaises(ValueError):operation_trace({'operations':[row]})
    def test_truncated_ledger_rejected(self):
        row=dict(operation_index=0,operation='action',paid_work=.1,operation_bound=.1,cumulative_work=.1,budget=1.3,comparator_reserve=1.,comparator_bound=1.,outcome='continue',prechecked=True,work_unit='multiply_add_products')
        with self.assertRaises(ValueError):operation_trace({'operations':[row]})
    def test_hpo_all_attempts_required(self):
        with self.assertRaises(ValueError):validate_plan({'schema':'cm-upstream-study-1','dataset':'ogbn-arxiv','method':'AMP','trials':[{}]})
    def test_upstream_unknown_method_rejected(self):
        with self.assertRaises(ValueError):validate_plan({'schema':'cm-upstream-study-1','dataset':'ogbn-arxiv','method':'renamed-MLP'})


class OperationBoundReplayTests(unittest.TestCase):
    def trace(self):
        common=dict(budget=12.,comparator_reserve=10.,comparator_bound=10.,
                    prechecked=True,work_unit='multiply_add_products')
        return {'operations':[
            dict(common,operation_index=0,operation='action',paid_work=1.,operation_bound=2.,
                 cumulative_work=1.,outcome='continue'),
            dict(common,operation_index=1,operation='prediction',paid_work=1.,operation_bound=2.,
                 cumulative_work=2.,outcome='completed')]}

    def test_exact_bound_affordability_is_accepted(self):
        value=operation_trace(self.trace())
        self.assertEqual(value['work'],2.)
        self.assertTrue(value['operation_bounds_replayed'])

    def test_low_actual_work_cannot_rescue_unaffordable_operation_bound(self):
        trace=self.trace();trace['operations'][0]['operation_bound']=3.
        with self.assertRaisesRegex(ValueError,'before execution'):
            operation_trace(trace)

    def test_low_total_work_cannot_rescue_violated_operation_bound(self):
        trace=self.trace();trace['operations'][0]['operation_bound']=.5
        with self.assertRaisesRegex(ValueError,'exceeds its declared'):
            operation_trace(trace)

    def test_legacy_paid_cost_only_ledger_is_not_certified(self):
        trace=self.trace();trace['operations'][0].pop('operation_bound')
        with self.assertRaisesRegex(ValueError,'legacy E6 ledger'):
            operation_trace(trace)

    def test_prediction_bound_must_fit_the_held_reserve(self):
        trace=self.trace();trace['operations'][1]['operation_bound']=11.
        with self.assertRaisesRegex(ValueError,'completion bound'):
            operation_trace(trace)

    def test_comparator_bound_cannot_change_mid_episode(self):
        trace=self.trace();trace['operations'][1]['comparator_bound']=9.
        with self.assertRaisesRegex(ValueError,'changes its cap/reserve/comparator'):
            operation_trace(trace)

    def test_reserve_must_cover_original_comparator_bound(self):
        trace=self.trace()
        for op in trace['operations']:op['comparator_bound']=11.
        with self.assertRaisesRegex(ValueError,'does not cover'):
            operation_trace(trace)

    def test_restart_uses_frozen_comparator_bound(self):
        trace=self.trace();row=trace['operations'][1]
        row.update(operation='comparator_restart',outcome='abort_to_comparator',
                   operation_bound=9.,paid_work=8.,cumulative_work=9.)
        with self.assertRaisesRegex(ValueError,'restart bound'):
            operation_trace(trace)
        row['operation_bound']=10.
        self.assertEqual(operation_trace(trace)['work'],9.)

    def test_static_fallback_cannot_follow_paid_work(self):
        trace=self.trace();trace['operations'][1].update(operation='comparator_restart',
            outcome='static_fallback',operation_bound=10.,paid_work=8.,cumulative_work=9.)
        with self.assertRaisesRegex(ValueError,'Static fallback follows'):
            operation_trace(trace)

    def test_nonfinite_bound_and_cumulative_work_fail_closed(self):
        for field in ('operation_bound','cumulative_work'):
            for value in (True,float('nan'),float('inf'),-1.):
                trace=self.trace();trace['operations'][0][field]=value
                with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                    operation_trace(trace)

    def test_cost_units_must_be_declared_and_unchanged(self):
        trace=self.trace();trace['operations'][0].pop('work_unit')
        with self.assertRaisesRegex(ValueError,'work unit'):
            operation_trace(trace)
        trace=self.trace();trace['operations'][1]['work_unit']='wall_seconds'
        with self.assertRaisesRegex(ValueError,'work unit'):
            operation_trace(trace)

try:import torch
except ImportError:torch=None
@unittest.skipUnless(torch is not None,'PyTorch required for decoder and development tests')
class FamilyTests(unittest.TestCase):
    def test_all_six_families_and_locked_mpc_survive(self):
        from cm_experiments.development import open_loop_family
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'fit').mkdir();r=[dict(route=a,ensemble=2,depth=6,hash=a) for a in OPEN_LOOP]
            (root/'fit/registry.json').write_text(json.dumps(r));qc=dict(r[-1],horizon=4)
            family,audit=open_loop_family(root,qc,depth=6)
            self.assertEqual(set(audit['families']),set(OPEN_LOOP));self.assertTrue(audit['locked_comparator_included'])
            self.assertTrue(any(x['candidate']==qc for x in family))
            with self.assertRaises(ValueError):open_loop_family(root,qc,depth=6,proposal_limit=2)
    def test_development_labels_disjoint(self):
        from cm_experiments.development import development_partition
        data=SimpleNamespace(graph_ptr=None,splits={'explore':torch.arange(10)})
        p=development_partition(data,seed=7);r=p['episodes'][0]
        self.assertFalse(set(r['fit_ids'])&set(r['selection_ids']));self.assertEqual(len(r['fit_ids'])+len(r['selection_ids']),10)
        self.assertEqual(p['unique_input_contexts'],1);self.assertFalse(p['independent_input_collection'])
    def test_variable_decoder_stops_before_full_depth(self):
        from cm_experiments.schedules import DecoderSpec,ScheduleDecoder
        net=ScheduleDecoder(DecoderSpec(6,8,8,4,actions=10))
        # Output parameter names are inspected rather than changing source semantics.
        with torch.no_grad():
            for p in net.parameters():p.zero_()
            head=[m for m in net.modules() if isinstance(m,torch.nn.Linear) and m.out_features==10][-1]
            head.bias[9]=100
        seq,trace=net.decode(np.ones(6));self.assertEqual(len(seq),1);self.assertNotIn(9,seq)
        self.assertLess(trace['decoder_multiply_adds'],net.context_products+4*net.step_products)
    def test_full_candidate_family_in_variant_selection(self):
        from cm_experiments.study_execution import variants
        qc=dict(route='mpc_ol',profile='full',horizon=4,depth=6,ensemble=2,hash='qc')
        family=[dict(candidate=qc,source='locked_comparator')]
        cfg={'model':{'depth':6},'controller':{'horizons':[1,4]}}
        teacher=dict(route='mpc_rh',horizon=4,profile='full',depth=6,ensemble=2)
        r=variants(cfg,teacher,family,{'E3':['e3'],'E4':['e4']})
        self.assertTrue(any(v.get('candidate')==qc and v['kind']=='policy' for v in r['E3']))
        self.assertTrue(any(v['kind']=='s3' for v in r['E6']))
        self.assertTrue(any(v['kind']=='policy' for v in r['E6']))

if __name__=='__main__':unittest.main()
