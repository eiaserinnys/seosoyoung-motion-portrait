"""Six bounded contracts for the adopted label-free 2f experiment."""
import signal
import unittest
import numpy as np
from scipy.spatial import cKDTree
from fit_core import read_json, eroded_target, boundary
from fit2d_core import RESULTS as OLD
from fit2f_core import (RESULTS, load_targets_f, taper_amount, surface_contour,
                       pair_rows, geometric_operator, load_surface)


class Contracts(unittest.TestCase):
    def setUp(self):
        previous=signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('30s contract')))
        self.addCleanup(signal.signal,signal.SIGALRM,previous)
        self.addCleanup(signal.alarm,0)
        signal.alarm(30)

    def test_taper(self):
        mask=np.zeros((100,100),np.uint8); mask[:,:20]=1
        p=np.column_stack([np.arange(19,61),np.full(42,50)])
        amount=taper_amount(mask,p,7.)
        self.assertEqual(amount[0],0.)
        self.assertEqual(amount[-1],7.)
        self.assertTrue((np.diff(amount)>=0).all())
        self.assertGreater(np.max(np.abs(7.-amount)),6.)

    def test_skin_conflict_removed(self):
        t=read_json(OLD/'a-shared.json')['thickness_px']
        target=load_targets_f([3],t)[0]
        new=cKDTree(target['head_points_all']).query(target['points'][target['labels']==0])[0]
        old=cKDTree(boundary(eroded_target(target['head'],t))).query(target['points'][target['labels']==0])[0]
        print('CONFLICT old/new median px',np.median(old),np.median(new),flush=True)
        self.assertLessEqual(np.median(new),1.5)
        self.assertGreater(np.median(old),5.)

    def test_open_boundary_not_paired(self):
        v=np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],float)
        tri=np.array([[0,1,2],[0,2,3]])
        pose=np.array([0.,0.,0.,np.log(40),100,100])
        full=surface_contour(v,tri,pose,exclude_boundary=False)
        cut=surface_contour(v,tri,pose)
        self.assertGreater(len(full['xy']),0)
        # attach_contour's +.5 pixel centers can attach two corner samples to
        # the internal diagonal. The contract excludes open-edge attachments,
        # not every sample originating at a boundary raster pixel.
        self.assertLess(len(cut['xy']),len(full['xy']))
        from fit2f_core import boundary_edges
        opened=set(map(tuple,boundary_edges(tri)))
        def open_attachments(model):
            chosen=np.sort(np.take_along_axis(model['ids'],np.argsort(model['weights'],axis=1)[:,-2:],axis=1),axis=1)
            return [tuple(e) in opened for e in chosen]
        self.assertFalse(any(open_attachments(cut)))
        self.assertTrue(any(open_attachments(full)))
        # Tiny GLB triangles with no pixel center cannot own silhouette pixels.
        # Verify the exact z-buffer/owner result before using that rejection.
        from fit2f_core import raster_silhouette
        from fit_core import rasterize
        rng=np.random.default_rng(615)
        q=rng.uniform(5,195,(90,3)); origins=rng.uniform(5,195,(1000,3))
        tiny=origins[:,None,:]+np.array([[0.,0.,0.],[0.,.1,0.],[.1,0.,0.]])[None]
        q=np.vstack([q,tiny.reshape(-1,3)]); triangles=np.arange(len(q)).reshape(-1,3)
        expected_depth,expected_owner=rasterize(q,triangles,cull_backfaces=True)
        actual_depth,actual_owner=raster_silhouette(q,triangles)
        np.testing.assert_array_equal(actual_owner,expected_owner)
        np.testing.assert_array_equal(actual_depth,expected_depth)

    def test_ear_neighborhood_not_paired(self):
        model={'xy':np.array([[0.,0.],[9.,0.],[11.,0.]]),
               'ids':np.array([[0,1,2]]*3),'weights':np.array([[1.,0,0]]*3)}
        target={'points':np.array([[0.,0.],[9.,0.],[11.,0.]]),
                'normals':np.array([[1.,0.]]*3),'labels':np.array([2,0,1]),
                'ear_points':np.array([[0.,0.]])}
        rows=pair_rows(model,target,20.)
        self.assertEqual(set(rows['model_indices']),{2})
        self.assertEqual(set(rows['target_indices']),{2})
        target['ear_points']=np.empty((0,2))
        polluted=pair_rows(model,target,20.)
        self.assertIn(1,polluted['model_indices'])

    def test_geometric_laplacian_bump(self):
        x,y=np.meshgrid(np.arange(-120,121,4.),np.arange(-120,121,4.))
        v=np.column_stack([x.ravel(),y.ravel(),np.zeros(x.size)])
        ids=np.arange(x.size).reshape(x.shape); a=ids[:-1,:-1].ravel()
        tri=np.vstack([np.column_stack([a,a+1,a+x.shape[1]+1]),
                       np.column_stack([a,a+x.shape[1]+1,a+x.shape[1]])])
        lap,mass=geometric_operator(v,tri)
        bump=10*np.exp(-(v[:,0]**2+v[:,1]**2)/(2*30**2))
        curvature=np.abs(lap@bump/mass)
        maximum=float(curvature.max()); print('BUMP curvature max',maximum,flush=True)
        self.assertLessEqual(maximum,2.5*20/900)
        self.assertGreater(maximum,.01)
        self.assertLess(np.max(np.abs(lap@np.ones(len(v))/mass)),1e-10)
        # Cot contributions belong to an edge before nonnegative clamping.
        # On edge (0,1), the opposite cots are -1.75 and +2.4.
        obtuse=np.array([[0.,0,0],[2.,0,0],[.2,.1,0],[1.,-5.,0]])
        triangles=np.array([[0,1,2],[1,0,3]])
        operator,_=geometric_operator(obtuse,triangles)
        self.assertAlmostEqual(float(-operator[0,1]),(-1.75+2.4)/2,places=12)
        obtuse[3,1]=-2.  # Now cot sum is negative and the edge weight is zero.
        operator,_=geometric_operator(obtuse,triangles)
        self.assertEqual(float(operator[0,1]),0.)

    def test_remesh_surface_distance(self):
        info=read_json(RESULTS/'remesh.json')
        self.assertLessEqual(info['distance_px']['max'],1.5)
        self.assertGreater(len(load_surface('surface-a')['vertices']),0)
        bad=dict(info['distance_px'],max=2.)
        self.assertFalse(bad['max']<=1.5)


if __name__=='__main__': unittest.main()
