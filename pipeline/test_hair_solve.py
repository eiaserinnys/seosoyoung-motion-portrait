"""Independent minimum-norm and residual budget contracts."""
import copy
import json
import unittest
from unittest.mock import patch
import numpy as np
from fit_hair_solve import ROOT,stencil,regularizer,solve_residual,conformal_rows,solve_joint,silhouette_weight,S,Y
from fit_hair_solve import chart_derivatives,project_orientation
from fit_hair_solve import observations,fit_lower
from hair_solve_quality import compare_arrays,lower_span_comparison,raster_stencil_sampler


def assert_budget(data):
    assert len(data['s'])==12 and len(data['y'])==16 and len(data['poses'])==6
    for p in data['poses']:
        du=np.array(p['du'])
        assert du.shape==(192,2) and np.isfinite(du).all()
        assert np.sqrt(np.mean(np.sum(du**2,axis=1)))<=45
        assert np.linalg.norm(du,axis=1).max()<=115
        if not p['yaw']:
            grid=du.reshape(16,12,2)
            np.testing.assert_allclose(grid[:,:,0],-grid[:,::-1,0],atol=1e-10)
            np.testing.assert_allclose(grid[:,:,1],grid[:,::-1,1],atol=1e-10)


class HairSolveTest(unittest.TestCase):
    def test_attachment_support_stays_exact_under_conflicting_observations(self):
        c=stencil(-1,490);fixed=np.flatnonzero(c).tolist()
        systems=[dict(name=str(yaw),yaw=yaw,pitch=-30,c=np.array([c]),d=np.array([[20.,30.]]),
                      w=np.array([1000.]),normal=np.array([[0.,1.]]),fixed=fixed) for yaw in [-45,0,45]]
        for pose in solve_joint(systems,regularizer()):
            np.testing.assert_array_equal(np.array(pose['du'])[fixed],0)

    def test_spiky_hairline_does_not_apply_horizontal_shear(self):
        import probe_hair_only as probe
        crown=np.zeros((740,1024),bool);crown[100:521,300:700]=True
        face=np.zeros_like(crown);face[350:600,400:600]=True
        ref=face.copy();ref[280:350,400:600]=True
        ref[:410,414]=False  # a narrow gap between two fringe strands
        pose=dict(name='fixture',yaw=45,pitch=30,vertices=[dict(s=(x-500)/200,y=300,xy=[x,350],front=True) for x in range(400,601,10)])
        def mask(path):return crown if path.name.endswith('-crown.png') else face
        with patch.object(probe,'read_mask',side_effect=mask),patch.object(probe,'reference',return_value=dict(head=crown,face=ref)):
            result=observations(pose,{},ROOT/'debug/current')
        hairline=(result['d'][:,1]!=0)&(result['w']<100)
        self.assertGreater(hairline.sum(),20)
        np.testing.assert_array_equal(result['normal'][hairline],np.tile([0,1],(hairline.sum(),1)))
        np.testing.assert_array_equal(result['d'][hairline,1],-70)
        self.assertLessEqual(result['w'][hairline].max(),8)

    def test_tail_fit_rejects_alternating_width_and_center_spikes(self):
        import probe_hair_only as probe
        current=np.zeros((740,1024),bool);tie=np.zeros_like(current);ref=np.zeros_like(current)
        tie[530:555,290:310]=True;tie[530:555,690:710]=True
        for y in range(550,721):
            # Deliberately noisy row measurements must yield one smooth lock.
            shift=55 if (y//24)%2 else -45;width=150 if (y//24)%2 else 30
            for x in [300,700]:
                current[y,x-50:x+50]=True;ref[y,x+shift-width//2:x+shift+width//2]=True
        def mask(path):return tie if path.name.endswith('-ties.png') else current
        with patch.object(probe,'read_mask',side_effect=mask),patch.object(probe,'reference',return_value=dict(mask=ref)):
            data=fit_lower([dict(name='fixture',yaw=45,pitch=30)],{'fixture':{}},ROOT/'debug/current')
        for side in ['left','right']:
            q=np.array(data['poses'][0][side]);np.testing.assert_array_equal(q[:3],[[0,1]]*3)
            self.assertTrue(np.all((q[:,1]>=.75)&(q[:,1]<=1.35)))
            # No alternating turns: curvature is small relative to the lock length.
            self.assertLess(np.abs(np.diff(q[:,0],2)).max(),5)

    def test_raster_correspondence_uses_visible_alpha_and_paint_order(self):
        xy=np.array([[0,0],[10,0],[0,10]]*2,float)
        pose=dict(triangles=[[0,1,2],[3,4,5]],vertices=[dict(uv=[.25,.25] if i<3 else [.75,.75],facing=1,front=True,order=10 if i<3 else 1) for i in range(6)])
        alpha=np.array([[0,0],[0,1]],float)
        with patch('hair_solve_quality.atlas_alpha',return_value=alpha):
            sample=raster_stencil_sampler(pose,xy,np.eye(6),'fixture')
        np.testing.assert_allclose(sample([2,2],True),[0,0,0,.5,.25,.25])
        self.assertIsNone(sample([20,20],True))
        with patch('hair_solve_quality.atlas_alpha',return_value=np.ones((2,2))):
            opaque=raster_stencil_sampler(pose,xy,np.eye(6),'fixture')
        np.testing.assert_allclose(opaque([2,2],True),[.5,.25,.25,0,0,0])
        pose['triangles'][1]=[3,5,4]
        for i,v in enumerate(pose['vertices']):v['order']=1 if i<3 else 10
        with patch('hair_solve_quality.atlas_alpha',return_value=np.ones((2,2))):
            culled=raster_stencil_sampler(pose,xy,np.eye(6),'fixture')
        np.testing.assert_allclose(culled([2,2],True),[.5,.25,.25,0,0,0])

    def test_hairline_observations_survive_a_fully_occluded_row330(self):
        import probe_hair_only as probe
        crown=np.zeros((740,1024),bool);crown[100:521,300:700]=True
        face=np.zeros_like(crown);face[350:600,400:600]=True
        prior=np.zeros_like(crown);prior[280:600,400:600]=True
        ref=prior.copy();ref[250:280,400:600]=True
        prior[330,400:450]=False;prior[330,550:600]=False
        pose=dict(name='fixture',yaw=45,pitch=30,vertices=[dict(s=(x-500)/200,y=300,xy=[x,350],front=True) for x in range(400,601,10)])
        def mask(path):
            if path.parent.name=='hair_solve_prior':return prior
            return crown if path.name.endswith('-crown.png') else face
        with patch.object(probe,'read_mask',side_effect=mask),patch.object(probe,'reference',return_value=dict(head=crown,face=ref)):
            result=observations(pose,{},ROOT/'debug/current')
        vertical=np.all(result['normal']==[0,1],axis=1)
        self.assertGreaterEqual(vertical.sum(),20)
        np.testing.assert_array_equal(result['d'][vertical,1],-100)

    def test_local_orientation_projection_repairs_injected_fold(self):
        ds,dy=chart_derivatives()
        du=np.array([[-s*140,0] for y in Y for s in S])
        def determinant(q):
            ux,vx=(ds@q).T;uy,vy=(dy@q).T
            return ((1+ux)*(1+vy)-uy*vx).min()
        self.assertLess(determinant(du),.25)
        fixed=np.array(project_orientation(du,0,ds,dy))
        self.assertGreaterEqual(determinant(fixed),.25)
        self.assertLessEqual(np.linalg.norm(fixed,axis=1).max(),115)
        zero=np.zeros((192,2));np.testing.assert_array_equal(project_orientation(zero,0,ds,dy),zero)
    def test_lower_span_gate_detects_improvement_regression_and_missing_tail(self):
        ref=np.zeros((740,1024),bool);ref[550:721,200:320]=True;ref[550:721,680:800]=True
        prior=np.zeros_like(ref);prior[550:721,220:300]=True;prior[550:721,700:780]=True
        self.assertTrue(lower_span_comparison(ref,prior,ref)['pass'])
        self.assertEqual(lower_span_comparison(ref,prior,ref)['correctedRms'],0)
        self.assertFalse(lower_span_comparison(ref,ref,prior)['pass'])
        self.assertFalse(lower_span_comparison(ref,prior,np.zeros_like(ref))['pass'])
    def test_span_loss_keeps_horizontal_pixel_units_on_steep_contours(self):
        self.assertGreater(silhouette_weight(200,0),silhouette_weight(300,0))
        for slope in [0,2,10]:
            self.assertAlmostEqual(silhouette_weight(200,slope)/(1+slope**2),silhouette_weight(200,0))

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
            for y in [0,120,325,400,500,515]:
                a=stencil(s,y)
                self.assertAlmostEqual(float(a.sum()),1)
                self.assertGreaterEqual(a.min(),0)
        for y in [545,584]:self.assertEqual(np.count_nonzero(stencil(.5,y)),0)

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
        bad=copy.deepcopy(data);bad['poses'][0]['du'][0]=[116,0]
        with self.assertRaises(AssertionError):assert_budget(bad)


if __name__=='__main__':unittest.main()
