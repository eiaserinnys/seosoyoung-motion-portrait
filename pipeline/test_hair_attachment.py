import unittest
import json,subprocess
import numpy as np
from fit_hair_attachment import ROOT,area,limit_stage
from hair_attachment_quality import audit

class LocalAreaProjectionTest(unittest.TestCase):
    def test_all_thirteen_poses_keep_S_orientation(self):
        rows=json.loads(subprocess.check_output(['node',str(ROOT/'pipeline/hair_attachment_geometry.mjs')],timeout=60))
        for pose in audit(rows):
            with self.subTest(pose=pose['name']):
                self.assertEqual(pose['reversed']['residual'],0)

    def test_actual_two_dimensional_fold_is_limited_and_clean_vertices_stay_exact(self):
        points=np.array([[0.,0.],[1.,0.],[0.,1.],[10.,0.],[11.,0.],[10.,1.]])
        delta=np.array([[0.,0.],[0.,0.],[2.,-2.],[1.,0.],[1.,0.],[1.,0.]])
        triangles=np.array([[0,1,2],[3,4,5]])
        self.assertLess(area((points+delta)[triangles])[0],0)
        s,report=limit_stage(points,delta,triangles)
        ratio=area((points+delta*s[:,None])[triangles])/area(points[triangles])
        self.assertGreaterEqual(ratio.min(),.2-1e-7)
        np.testing.assert_array_equal(s[3:],1)
        self.assertEqual(report['triggered'],1)

    def test_a_thin_input_triangle_is_not_forced_to_an_absolute_area(self):
        points=np.array([[0.,0.],[1.,0.],[0.,.001]])
        delta=np.array([[0.,0.],[0.,0.],[0.,-.0005]])
        s,report=limit_stage(points,delta,np.array([[0,1,2]]))
        np.testing.assert_array_equal(s,1)
        self.assertEqual(report['triggered'],0)

if __name__=='__main__':unittest.main()
