import unittest,importlib.util
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('surfaces',Path(__file__).parents[1]/'tools/fit_surfaces.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class SurfaceTests(unittest.TestCase):
    def test_training_and_validation_are_disjoint(self):
        self.assertFalse(set(m.TRAIN)&set(m.TEST));self.assertEqual(len(m.TEST),20)
    def test_plane_prediction_and_fit_known_shape(self):
        a=np.array([[x,y] for y in np.linspace(-.5,.5,4) for x in np.linspace(-.4,.4,4)])
        labels=np.full(len(a),'skin');p=np.array([0,0,.75,.75,.4]);weights=np.full(len(a),1/len(a))
        np.testing.assert_allclose(m.predict(a,labels,'plane',p,[(0,0)])[0],a)
        target=m.predict(a,labels,'plane',p,m.TRAIN)
        found,loss,_=m.fit(a,labels,target,weights,'plane',1)
        self.assertLess(loss**.5,.001)
        corrupted=target.copy();corrupted[0,0]+=.5
        self.assertGreater(np.mean((m.predict(a,labels,'plane',p,m.TRAIN)-corrupted)**2),.001)
if __name__=='__main__':unittest.main()
