"""Development-only initial-input sequence learning, actual capacity and replay."""
import tempfile,unittest,inspect
from pathlib import Path
import numpy as np
try:import torch
except ImportError:raise unittest.SkipTest('PyTorch is required for sequence execution')
from cm_experiments.schedules import DecoderSpec,ScheduleDecoder,count_parameters,capacity_match,fit_decoder,save_decoder,load_decoder

class ScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def setUp(self):
        self.spec=DecoderSpec(6,8,8,4);self.net=ScheduleDecoder(self.spec)
        self.x=np.random.default_rng(4).normal(size=(12,6)).astype('float32')
        self.targets=np.tile([0,1,8,2],(12,1));self.groups=np.repeat(['a','b','c','d'],3);self.roles=np.full(12,'Dexplore')
    def fit(self,**kw):
        return fit_decoder(self.x,self.targets,self.groups,self.roles,arm='E3',critic_parameters=2000,epochs=4,patience=2,seed=1,**kw)
    def test_actual_parameter_count_matches_formula(self):self.assertEqual(count_parameters(self.spec),self.net.parameter_count)
    def test_capacity_matching_uses_active_weights(self):
        for n in (2000,29890,70000):
            spec,a=capacity_match(6,4,n);self.assertLessEqual(a['relative_parameter_mismatch'],.01)
            self.assertEqual(ScheduleDecoder(spec).parameter_count,a['decoder_parameters'])
    def test_decode_has_no_measurement_or_label_argument(self):
        self.assertFalse(set(inspect.signature(self.net.decode).parameters)&{'labels','y','plant','measurement','state'})
    def test_complete_sequence_charged_before_execution(self):
        actions,t=self.net.decode(np.ones(6));self.assertEqual(len(actions),4);self.assertTrue(t['plan_fixed_before_execution'])
        self.assertEqual(t['decoder_multiply_adds'],self.net.context_products+4*self.net.step_products)
    def test_masks_are_initial_history_only(self):
        history=[]
        def mask(d,prefix):history.append((d,prefix));return [d]
        actions,_=self.net.decode(np.ones(6),mask);self.assertEqual(actions,[0,1,2,3]);self.assertEqual(history[-1],(3,(0,1,2)))
    def test_empty_mask_direct_source_recovery(self):
        actions,t=self.net.decode(np.ones(6),lambda d,p:[]);self.assertEqual(actions,[8]*4);self.assertEqual(t['empty_support_recovery_depths'],[0,1,2,3])
    def test_dynamic_anchor_mask_rejected(self):
        with self.assertRaises(ValueError):self.net.decode(np.ones(6),lambda d,p:[9])
    def test_duplicate_mask_rejected(self):
        with self.assertRaises(ValueError):self.net.decode(np.ones(6),lambda d,p:[1,1])
    def test_test_or_confirmation_labels_cannot_fit(self):
        for role in ('Dtest','Dconfirm','Dadmit','D_compare','compare'):
            self.roles[0]=role
            with self.assertRaises(ValueError):self.fit()
    def test_group_holdout_is_disjoint(self):
        net,a=self.fit();self.assertFalse(set(a['training_groups'])&set(a['validation_groups']))
        self.assertTrue(all(not p.requires_grad for p in net.parameters()))
    def test_one_group_validation_is_rejected(self):
        self.groups[:]='a'
        with self.assertRaises(ValueError):self.fit()
    def test_reset_target_conversion_is_recorded(self):
        self.targets[0,0]=9;_,a=self.fit();self.assertEqual(a['a9_to_a8_target_conversions'],1)
    def test_checkpoint_safe_load_matches_predictions(self):
        net,a=self.fit()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'decoder.pt';save_decoder(net,a,path);restored,b=load_decoder(path)
            self.assertEqual(net.decode(self.x[0]),restored.decode(self.x[0]));self.assertEqual(a,b)
            with self.assertRaises(FileExistsError):save_decoder(net,a,path)
    def test_seeded_training_replays(self):
        net,a=self.fit();other,b=self.fit();self.assertEqual(a,b)
        for key in net.state_dict():self.assertTrue(torch.equal(net.state_dict()[key],other.state_dict()[key]))
    def test_nonfinite_features_fail(self):
        self.x[0,0]=np.nan
        with self.assertRaises(ValueError):self.fit()
    def test_invalid_decoder_dimensions_fail(self):
        for n in (0,-2,True):
            with self.assertRaises(ValueError):DecoderSpec(n,8,8,4)
    def test_different_outputs_not_claimed_equivalent(self):
        _,a=self.fit();self.assertEqual(a['target_semantics'],'source-reset-normalized complete schedule')

if __name__=='__main__':unittest.main()
