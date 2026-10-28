"""Frame duplication must not change the regularization/data balance."""
import contextlib
import io
import signal
import unittest
from unittest.mock import patch
import numpy as np
import torch
from fit_core import project
from fit2f_core import solve_surface


class FrameWeightContract(unittest.TestCase):
    def test_three_frame_equivalent_energy(self):
        previous=signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('30s contract')))
        self.addCleanup(signal.signal,signal.SIGALRM,previous)
        self.addCleanup(signal.alarm,0); signal.alarm(30)
        v=np.array([[0.,0,0],[0,1,0],[1,0,0]])
        tri=np.array([[0,1,2]]); pose=np.zeros(6)
        target={'frame':1,'points':np.array([[1.,0],[1.,-1]]),'normals':np.array([[1.,0],[1.,0]]),
                'labels':np.array([0,1]),'ear_points':np.empty((0,2))}
        # Isolate the normal-equation balance from raster quantization; both
        # skin and hair rows still pass through the real bidirectional pairing.
        def contour(vertices,triangles,camera):
            q=project(torch.tensor(vertices),torch.tensor(camera)).numpy()
            return {'ids':np.array([[0,1,2],[1,0,2]]),
                    'weights':np.array([[1.,0,0],[1.,0,0]]),
                    'xy':q[:2,:2],'boundary_rejected':0}
        with patch('fit2f_core.surface_contour',side_effect=contour),contextlib.redirect_stdout(io.StringIO()):
            _,three,_,_=solve_surface(v,tri,[pose]*3,[target]*3,1.)
            _,many,_,_=solve_surface(v,tri,[pose]*130,[target]*130,1.,reference_frames=3)
            _,unscaled,_,_=solve_surface(v,tri,[pose]*130,[target]*130,1.)
        np.testing.assert_allclose(many,three,rtol=0,atol=1e-7)
        self.assertGreater(float(np.max(np.abs(unscaled-three))),1e-5)
        print('FRAME BALANCE max normalized delta',np.max(np.abs(many-three)),
              'unscaled delta',np.max(np.abs(unscaled-three)),flush=True)


if __name__=='__main__': unittest.main()
