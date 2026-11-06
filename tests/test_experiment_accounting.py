import unittest
from cm_experiments.accounting import summarize_work,ARMS

class WorkComparisonTests(unittest.TestCase):
    def rows(self):
        return [{'episode_id':0,'arm':a,'multiply_add_work':100.,'wall_seconds':.01}
                for a in ARMS]
    def call(self,rows=None,traces=None,scored=None):
        return summarize_work(self.rows() if rows is None else rows, traces or [],ratio=1.2,
                              scored={a:True for a in ARMS} if scored is None else scored)
    def test_mean_ceiling_checked(self):
        self.assertTrue(self.call()['summary']['E3']['within_mean_ceiling'])
    def test_separate_qualification_is_charged_in_inclusive_comparison(self):
        value=self.call(traces=[{'episode_id':0,'arm':'E3','actual_state_support_audit':{'qualification_multiply_add_work':30}}])
        self.assertFalse(value['summary']['E3']['within_mean_ceiling'])
        self.assertEqual(value['summary']['E3']['observed_work_ratio'],1.3)
    def test_unscored_zero_work_does_not_pass(self):
        rows=self.rows();rows[-1]['multiply_add_work']=0
        scored={a:a!='E6' for a in ARMS}
        self.assertFalse(self.call(rows=rows,scored=scored)['summary']['E6']['within_mean_ceiling'])
    def test_members_are_summed(self):
        rows=self.rows();rows.append(dict(rows[0]));rows.append(dict(rows[2]))
        report=self.call(rows=rows)
        self.assertEqual(report['summary']['E3']['mean_qualification_inclusive_work'],200.)
    def test_missing_input_reference_rejected(self):
        with self.assertRaises(ValueError):self.call(rows=self.rows()[1:])
    def test_nonfinite_negative_cost_rejected(self):
        for value in (-1.,float('nan'),float('inf')):
            rows=self.rows();rows[0]['multiply_add_work']=value
            with self.assertRaises(ValueError):self.call(rows=rows)
    def test_missing_arm_rejected(self):
        with self.assertRaises(ValueError):self.call(rows=self.rows()[:-1])
    def test_qualification_without_execution_rejected(self):
        with self.assertRaises(ValueError):self.call(traces=[{'episode_id':1,'arm':'E3','actual_state_support_audit':{'qualification_multiply_add_work':1}}])

if __name__=='__main__':unittest.main()
