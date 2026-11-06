"""Raw-to-score replay must reject missing observations, tampering and fake scores."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
try:import torch
except ImportError:raise unittest.SkipTest('PyTorch is required by the task evaluator')
from cross_morse.evaluation import score_observations
from cm_experiments.artifacts import canonical,write_json,seal
from cm_experiments.replay import numeric_replay,checkpoint_replay

class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/arxiv_smoke.json').read_text())
        self.cfg=cfg;self.y=np.array([0,1,0,1]);self.u=np.array([0,0,1,1]);self.p=np.array([[.8,.2],[.3,.7],[.6,.4],[.1,.9]],dtype='float32')
        self.cfg['task']={'prediction_level':'node','problem_type':'multiclass','metric':'accuracy','num_outputs':2,'loss':'ce'}
        # Use the existing declared task shape, with only its class count changed.
        self.cfg['task']=dict(cfg['task'],num_outputs=2)
    def tearDown(self):self.tmp.cleanup()
    def pack(self,wrong_score=False,unscored_observation=False,schema='cm-arm-score-fixture-1'):
        p=self.root;c=self.cfg
        write_json(p/'base_config.json',c);write_json(p/'experiment.json',{'schema':schema,'base_configuration_sha256':canonical(c),'scope':'unit test; score reconstruction only'})
        score=score_observations(self.p,self.y,self.u,c)
        results={a:{'scored':True,'metric':score['metric'],'loss':score['node_ce']} for a in ('qC','feedback','E3','E4','E5')}
        results['E6']={'scored':False}
        if wrong_score:results['E3']['metric']+=.1
        write_json(p/'results.json',{'results':results})
        data={'observation_id':np.arange(4),'unit_id':self.u,'y_true':self.y,**{a:self.p for a in results if a!='E6'}}
        if unscored_observation:data['E6']=self.p
        np.savez_compressed(p/'predictions.npz',**data)
        units={'unit_id':score['core_ids'],'unit_weight':score['counts'],**{a:score['loss'] for a in results if a!='E6'}}
        for a in ('E3','E4','E5'):
            units[a+'_minus_feedback']=np.zeros_like(score['loss']);units[a+'_minus_qC']=np.zeros_like(score['loss'])
        np.savez_compressed(p/'paired_unit_losses.npz',**units);seal(p)
    def test_score_fixture_is_not_accepted_as_a_real_arm_archive(self):
        self.pack()
        with self.assertRaisesRegex(ValueError, "Legacy/wrong-role"):
            numeric_replay(self.root)
    def test_legacy_confirmation_archive_requires_new_preparation(self):
        self.pack(schema="cm-budget-arm-experiment-2")
        with self.assertRaisesRegex(ValueError, "never repartition"):
            numeric_replay(self.root)
    def test_metrics_recomputed(self):
        self.pack();self.assertTrue(numeric_replay(self.root, allow_score_fixture=True)['passed'])
    def test_wrong_summary_rejected_even_with_consistent_hashes(self):
        self.pack(wrong_score=True)
        with self.assertRaisesRegex(ValueError,'Numerical'):numeric_replay(self.root, allow_score_fixture=True)
    def test_unscored_arm_cannot_have_fabricated_prediction(self):
        self.pack(unscored_observation=True)
        with self.assertRaisesRegex(ValueError,'Unscored'):numeric_replay(self.root, allow_score_fixture=True)
    def test_checkpoint_loading_requires_explicit_trust(self):
        with self.assertRaisesRegex(ValueError,'trust'):checkpoint_replay(self.root,self.root)
    def test_modified_array_is_rejected_before_scoring(self):
        self.pack();(self.root/'predictions.npz').write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError,'mismatch'):numeric_replay(self.root, allow_score_fixture=True)

if __name__=='__main__':unittest.main()
