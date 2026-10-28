"""Bounded camera refinement contract on a known fixed synthetic surface."""
import signal
import unittest
import numpy as np
import torch
from fit_core import project
from fit2f_core import make_target, raster_silhouette, fit_cameras_f


class CameraContract(unittest.TestCase):
    def test_fixed_shape_camera_translation_and_rotation(self):
        prior=signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('30s contract')))
        self.addCleanup(signal.signal,signal.SIGALRM,prior)
        self.addCleanup(signal.alarm,0); signal.alarm(30)
        import trimesh
        mesh=trimesh.creation.icosphere(subdivisions=1)
        v,tri=np.array(mesh.vertices),np.array(mesh.faces)
        initial=np.array([[12.,7.,2.,np.log(40.),360.,360.]])
        desired=initial.copy(); desired[0,4:6]+=[4.,-3.]
        q=project(torch.tensor(v),torch.tensor(desired[0])).numpy()
        _,owner=raster_silhouette(q,tri); head=(owner>=0).astype(np.uint8)
        zero=np.zeros_like(head); target=make_target(head,head,zero,zero,7.,1)
        before=v.copy()
        fitted,record=fit_cameras_f(v,tri,[target],initial)
        np.testing.assert_array_equal(v,before)
        np.testing.assert_array_equal(fitted[:,:3],initial[:,:3])
        np.testing.assert_allclose(fitted[0,4:6],desired[0,4:6],atol=1.)
        self.assertEqual(record['iterations'],100)
        self.assertEqual(len(record['history']),10)
        self.assertEqual(record['history'][0]['gate_px'],15.)
        self.assertEqual(record['history'][-1]['gate_px'],6.)
        print('CAMERA synthetic recovered',fitted[0].tolist(),flush=True)


if __name__=='__main__': unittest.main()
