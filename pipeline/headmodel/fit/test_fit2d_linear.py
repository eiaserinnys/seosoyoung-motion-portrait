"""Six bounded contracts for pixel-unit contour surface editing."""
import unittest
import time
import signal
import cv2
import numpy as np
import torch
from fit_core import ROOT, load_targets, project, read_json
from fit2d_core import (RESULTS, pose_affine, template, linear_camera, normalized_laplacian,
                        classify_face, contour_residual, solve_icp, model_contours)


class LinearContracts(unittest.TestCase):
    def setUp(self):
        previous=signal.signal(signal.SIGALRM,self.timeout)
        self.addCleanup(signal.signal,signal.SIGALRM,previous)
        self.addCleanup(signal.alarm,0)
        signal.alarm(30)

    @staticmethod
    def timeout(signum,frame):
        raise TimeoutError('Individual fit2d test exceeded 30 seconds')

    def test_linear_projection(self):
        rng=np.random.default_rng(21)
        for pose in [[-43.5,24.4,-5.2,3.16,350.,410.],[37.,.47,2.38,3.21,450.,410.]]:
            v=rng.normal(size=(40,3))*20
            a,t=linear_camera(pose)
            q=project(torch.tensor(v),torch.tensor(pose,dtype=torch.float64)).numpy()[:,:2]
            np.testing.assert_allclose(v@a.T+t,q,atol=1e-5,rtol=0)
            self.assertGreater(np.max(np.abs(v@(a*1.125).T+t-q)),1.)

    def test_laplacian_energy(self):
        tri=np.array([[0,1,2],[0,2,3]])
        lap=normalized_laplacian(tri,4)
        constant=np.ones((4,3))*np.array([4.,7.,-3.])
        self.assertLess(np.square(lap@constant).sum(),1e-20)
        constant[0]+=10
        self.assertGreater(np.square(lap@constant).sum(),1.)

    def test_synthetic_bump(self):
        started=time.perf_counter()
        v,tri,_,masks,faces=template()
        v=np.load(RESULTS/'a-state.npz')['vertices']
        document=read_json(RESULTS/'a-poses.json')['frames']
        poses=np.array([document[str(f)] for f in (3,42,264)])
        sref=float(np.exp(np.median([p[3] for p in document.values()])))
        # Smooth compact radial bump in pixel units on the lower central jaw.
        vp=v*sref; center=vp[np.flatnonzero(masks['face'])[np.argmin(vp[masks['face'],1])]]
        radius=np.linalg.norm(vp-center,axis=1)/60.
        bump=np.maximum(1-radius**2,0)**2*10.
        wanted=v.copy(); wanted[:,1]-=bump/sref
        targets=[]; neighborhoods=[]; initial_errors=[]
        from fit_core import rasterize, boundary, eroded_target
        for frame,pose in zip((3,42,264),poses):
            q=project(torch.tensor(wanted),torch.tensor(pose,dtype=torch.float32)).numpy()
            _,owner=rasterize(q,tri[faces['head']],cull_backfaces=True)
            head=(owner>=0).astype(np.uint8)
            # Synthetic contours are rendered from the bumped A surface, using
            # the same exposed-Face ownership contract as the actual solver.
            target={'frame':frame,'head':head,'skin':head,'ears':{},'neck':np.zeros_like(head)}
            from scipy.spatial import cKDTree
            from fit2d_core import contour_normals
            wanted_contours=model_contours(wanted,tri,masks,faces,pose)
            ids=np.flatnonzero(masks['face'][tri].all(1))
            _,face_owner=rasterize(q,tri[ids],cull_backfaces=True)
            face_boundary=boundary(face_owner>=0)
            points=wanted_contours['face']['xy']
            nearest=cKDTree(face_boundary).query(points)[1]
            target['face_points']=points
            target['face_normals']=contour_normals(face_boundary)[nearest]
            target['head_boundary']=boundary(head)
            targets.append(target)
            projected_center=project(torch.tensor(center[None]/sref),torch.tensor(pose)).numpy()[0,:2]
            neighborhoods.append(projected_center)
            local=points[np.linalg.norm(points-projected_center,axis=1)<60]
            baseline=model_contours(v,tri,masks,faces,pose)['face']['xy']
            initial_errors.append(float(np.median(cKDTree(baseline).query(local)[0])))
        result,trace=solve_icp(v,tri,masks,faces,poses,targets,sref,0.,300.,iterations=5)
        medians=[]
        for pose,target,center_xy in zip(poses,targets,neighborhoods):
            contours=model_contours(result,tri,masks,faces,pose)
            target_points=target['face_points']
            target_points=target_points[np.linalg.norm(target_points-center_xy,axis=1)<60]
            points=contours['face']['xy']; points=points[np.linalg.norm(points-center_xy,axis=1)<60]
            medians.append(contour_residual(points,target_points)['median_px'])
        self.assertGreater(max(initial_errors),1.)
        print('SYNTHETIC jaw median px before/after:',initial_errors,medians,flush=True)
        self.assertLessEqual(max(medians),1.)
        self.assertLess(time.perf_counter()-started,30.)

    def test_front_ear_join_excluded(self):
        target=load_targets([3],pose_affine)[0]
        directory=ROOT/'pipeline/headmodel/data/frame_0003'
        ear=(cv2.imread(str(directory/'raw/ear.png'),0)>0).astype(np.uint8)
        neck=(cv2.imread(str(directory/'raw/neck.png'),0)>0).astype(np.uint8)
        points,normals,keep=classify_face(target['skin'],target['head'],ear,neck)
        near=np.linalg.norm(points-np.array([557,425]),axis=1)<12
        self.assertTrue(near.any())
        self.assertFalse(keep[near].any())
        # Deliberate include-all classifier violates the excluded-junction contract.
        self.assertTrue(np.ones_like(keep)[near].any())

    def test_residual_uses_classified_pixels_only(self):
        from unittest.mock import patch
        from fit2d_core import prepare_b_targets, measure_b
        target=prepare_b_targets(load_targets([3],pose_affine),7.)[0]
        model=target['face_points'].copy()
        target['face_all_points']=np.r_[target['face_all_points'],[[90.,90.]]]
        target['face_keep']=np.r_[target['face_keep'],False]
        target['ears']={}
        curves={'head':{'xy':target['head_boundary']},'face':{'xy':model},
                'head_mask':target['head_eroded'],'projected':np.zeros((1,3))}
        documents={'frames':[{'frame':3,'keypoints':[[0.,0.,1.]]*28}],'nose_vertex':0}
        with patch('fit2d_core.model_contours',return_value=curves), \
             patch('fit2d_core.face_interface',return_value=set()), \
             patch('fit2d_core.read_json',return_value=documents):
            exact=measure_b(None,None,None,None,[None],[target])[0]['face']
            self.assertEqual(exact['max_px'],0.)
            self.assertEqual(exact['median_px'],0.)
            # Deliberately contaminate the actual measurement input.
            target['face_points']=target['face_all_points']
            contaminated=measure_b(None,None,None,None,[None],[target])[0]['face']
            self.assertGreater(contaminated['max_px'],100.)

    def test_unconstrained_neck_is_finite(self):
        from scipy import sparse
        from scipy.sparse.linalg import splu
        tri=np.array([[0,1,2],[1,3,2]])
        lap=normalized_laplacian(tri,4)
        # Only vertex 0 is constrained. Vertex 3 represents the free neck.
        c=sparse.csr_matrix(([1.],([0],[0])),shape=(1,12))
        q=300*sparse.kron(lap.T@lap,sparse.eye(3))+.01*sparse.eye(12)+c.T@c
        d=splu(q.tocsc()).solve(np.asarray(c.T@np.array([10.])).ravel()).reshape(-1,3)
        self.assertTrue(np.isfinite(d[3]).all())
        self.assertGreater(abs(d[3,0]),0.)
