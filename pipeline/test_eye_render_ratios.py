import unittest
import copy
import numpy as np
from eye_render_ratios import ratio_error,ratio_gates
from capture_eye_alpha_junction import alpha_report


class RatioErrorTest(unittest.TestCase):
    def test_constant_identify_scale_cancels_and_changed_shape_does_not(self):
        self.assertAlmostEqual(ratio_error({'dNas':60},{'dNas':50},{'dNas':24},{'dNas':20},'dNas',100),0)
        self.assertGreater(abs(ratio_error({'dNas':90},{'dNas':50},{'dNas':24},{'dNas':20},'dNas',100)),4)

    def test_signed_delta_uses_projected_change_and_retains_injected_shift(self):
        def q(y):return dict(cI=[10,y],cO=[10,0],E=np.eye(2).tolist(),delta=[0,y/100])
        self.assertAlmostEqual(ratio_error(q(-3),q(-5),q(6),q(4),'deltaY',100),0)
        self.assertEqual(ratio_error(q(7),q(-5),q(6),q(4),'deltaY',100),10)

    def test_alpha_judge_detects_injected_jump_and_distinguishes_far_near(self):
        rows=[dict(yaw=0,pitch=0,left=100,right=200),dict(yaw=.1,pitch=0,left=101,right=210)]
        self.assertTrue(alpha_report([rows],7.24)['passed'])
        rows[1]['right']=700
        report=alpha_report([rows],7.24)
        self.assertFalse(report['passed']);self.assertEqual(report['peaks']['far']['difference'],500)
        self.assertEqual(report['peaks']['near']['difference'],1)

    def test_ratio_gates_reject_nasal_error_and_preserve_approved_exclusion_scope(self):
        def q(scale):return dict(dTemp=30*scale,dNas=20*scale,h=40*scale,a=10*scale,delta=[0,0],cI=[0,0],cO=[0,0],E=np.eye(2).tolist())
        measurement=dict(rest={s:q(1) for s in ['left','right']},K={s:dict(w=100) for s in ['left','right']},records=[])
        records=[]
        for name,scale in [('front',1),('turn',1.2)]:
            for side in ['left','right']:
                measurement['records'].append(dict(name=name,side=side,q=q(scale)))
                records.append(dict(name=name,side=side,v291=q(scale*2),r24=q(scale*2)))
        audit=dict(noise={f:dict(sigma=1) for f in ['dTemp','dNas','h','deltaY','a']},fit={s:dict(dNas=dict(outliers=[])) for s in ['left','right']})
        self.assertTrue(all(r['passed'] for r in ratio_gates(records,measurement,audit)['R2Ratio'].values()))
        bad=copy.deepcopy(records);next(r for r in bad if r['name']=='turn' and r['side']=='left')['v291']['dNas']*=3
        result=ratio_gates(bad,measurement,audit)
        self.assertFalse(result['R2Ratio']['dNas']['passed']);self.assertFalse(result['R3Ratio']['v291']['passed'])
        self.assertTrue(result['R3Ratio']['r24']['passed'])
        audit['fit']['left']['dNas']['outliers']=[dict(eye='turn:left')]
        result=ratio_gates(bad,measurement,audit)
        self.assertFalse(result['R2Ratio']['dNas']['passed']);self.assertTrue(result['R3Ratio']['v291']['passed'])


if __name__=='__main__':unittest.main()
