"""Independent minimum-norm and residual budget contracts."""
import copy
import json
import unittest
import numpy as np
from fit_hair_solve import ROOT,stencil,regularizer,solve_residual,conformal_rows,solve_joint,silhouette_weight,S,Y
from hair_solve_quality import compare_arrays


def assert_budget(data):
    assert len(data['s'])==12 and len(data['y'])==16 and len(data['poses'])==6
    for p in data['poses']:
        du=np.array(p['du'])
        assert du.shape==(192,2) and np.isfinite(du).all()
        assert np.sqrt(np.mean(np.sum(du**2,axis=1)))<=6
        assert np.linalg.norm(du,axis=1).max()<=22
        if not p['yaw']:
            grid=du.reshape(16,12,2)
            np.testing.assert_allclose(grid[:,:,0],-grid[:,::-1,0],atol=1e-10)
            np.testing.assert_allclose(grid[:,:,1],grid[:,::-1,1],atol=1e-10)


class HairSolveTest(unittest.TestCase):
    def test_occluded_and_steep_silhouettes_have_lower_confidence(self):
        self.assertGreater(silhouette_weight(200,0),silhouette_weight(300,0))
        self.assertGreater(silhouette_weight(200,0),silhouette_weight(200,2))

    def test_conformal_energy_permits_similarity_and_penalizes_shear(self):
        points=np.array([[160*s,y] for y in Y for s in S]);cr=conformal_rows()
        for du in [points,points@np.array([[0,1],[-1,0]]),np.ones((192,2))]:
            np.testing.assert_allclose(cr@du.T.reshape(-1),0,atol=1e-12)
        shear=points.copy();shear[:,0]+=points[:,1]
        self.assertGreater(np.linalg.norm(cr@shear.T.reshape(-1)),1)

    def test_joint_lopo_excludes_held_out_observations(self):
        systems=[dict(name=str(yaw),yaw=yaw,pitch=30,c=np.array([stencil(.5,200)]),
                      d=np.array([[4.,2.]]),w=np.ones(1),normal=np.array([[1.,0.]])) for yaw in [-45,0,45]]
        first=solve_joint(systems,regularizer(),held_out='45')
        systems[-1]['d']*=1000
        second=solve_joint(systems,regularizer(),held_out='45')
        for a,b in zip(first,second):np.testing.assert_array_equal(a['du'],b['du'])
        self.assertGreater(np.linalg.norm(first[-1]['du']),0)

    def test_actual_raster_gate_accepts_improvement_and_rejects_regression(self):
        ref=np.zeros((740,1024),bool);ref[40:500,300:700]=True
        face=np.zeros_like(ref);face[200:500,400:600]=True
        prior=np.zeros_like(ref);prior[40:500,320:680]=True
        better=compare_arrays(ref,face,prior,face,ref,face)
        self.assertTrue(better['pass']);self.assertEqual(better['correctedRms'],0)
        worse=np.zeros_like(ref);worse[40:500,350:650]=True
        self.assertFalse(compare_arrays(ref,face,prior,face,worse,face)['pass'])
        self.assertTrue(compare_arrays(ref,face,prior,face,prior,face)['pass'])
    def test_bilinear_partition_and_protected_lower_region(self):
        for s in [-2,-1,0,.4,1,2]:
            for y in [0,120,325,400]:
                a=stencil(s,y)
                self.assertAlmostEqual(float(a.sum()),1)
                self.assertGreaterEqual(a.min(),0)
        for y in [435,470,584]:self.assertEqual(np.count_nonzero(stencil(.5,y)),0)

    def test_solution_satisfies_the_normal_equations_and_improves_observations(self):
        rng=np.random.default_rng(25)
        c=np.array([stencil(s,y) for s,y in zip(np.linspace(-2,2,24),np.linspace(20,390,24))])
        d=rng.normal(size=(24,2));w=np.ones(24)
        normal=rng.normal(size=(24,2));normal/=np.linalg.norm(normal,axis=1)[:,None]
        p=dict(name='fixture',yaw=45,pitch=30,c=c,d=d,w=w,normal=normal)
        r=regularizer();result=solve_residual(p,r);du=np.array(result['du'])
        error=np.sum((c@du-d)*normal,axis=1)
        for axis in [0,1]:np.testing.assert_allclose(c.T@(error*normal[:,axis])+r@du[:,axis],0,atol=1e-10)
        self.assertLess(result['stats']['predictedRms'],result['stats']['priorRms'])

    def test_generated_grid_and_budget(self):
        assert_budget(json.loads((ROOT/'source/viewer/hair-solve.json').read_text()))

    def test_budget_gate_rejects_injected_large_residual(self):
        data=json.loads((ROOT/'source/viewer/hair-solve.json').read_text())
        bad=copy.deepcopy(data);bad['poses'][0]['du'][0]=[23,0]
        with self.assertRaises(AssertionError):assert_budget(bad)


if __name__=='__main__':unittest.main()
