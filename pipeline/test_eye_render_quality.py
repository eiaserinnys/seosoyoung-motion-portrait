"""The pixel/metric judge rejects injected visible errors before real captures."""
import copy
import unittest
import numpy as np
from PIL import Image
from capture_sam3_eyes import binary, gates, M

class RenderJudgeTest(unittest.TestCase):
    def test_identify_threshold_preserves_visible_pixels(self):
        rgba=np.zeros((2,2,4),np.uint8)
        rgba[0,0]=[255,255,255,255]
        self.assertEqual(int(binary(Image.fromarray(rgba)).sum()),1)

    def test_judge_accepts_reference_and_rejects_injected_rms_and_front_error(self):
        records=[]
        for row in M['records']:
            q=copy.deepcopy(row['q']);old=copy.deepcopy(q)
            for field in ['dTemp','dNas','h','a']:old[field]+=1
            old['delta'][1]+=.01
            records.append(dict(name=row['name'],side=row['side'],v291=q,r24=old))
        front={rev:np.zeros((720,720,4),np.uint8) for rev in ['r24','v291']}
        report=gates(records,front)
        self.assertTrue(all(r['passed'] for r in report['R2'].values()))
        self.assertTrue(report['R5']['passed'])
        for r in records:r['v291']['dTemp']+=20
        front['v291'][400,300,0]=2
        bad=gates(records,front)
        self.assertFalse(bad['R2']['dTemp']['passed'])
        self.assertFalse(bad['R5']['passed'])

if __name__=='__main__':
    unittest.main()
