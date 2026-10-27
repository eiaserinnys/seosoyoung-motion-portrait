"""The final sheets use depth testing and reject back-facing projected faces."""
import unittest
import numpy as np
from fit_core import rasterize


class CullingContract(unittest.TestCase):
    def test_closer_backface_is_culled_and_nearest_frontface_wins(self):
        q=np.array([[1.,1.,1.],[1.,6.,1.],[6.,1.,1.],
                    [1.,1.,5.],[1.,6.,5.],[6.,1.,5.],
                    [1.,1.,3.],[1.,6.,3.],[6.,1.,3.]])
        tri=np.array([[0,1,2],[3,5,4],[6,7,8]])
        _,plain=rasterize(q,tri,8)
        self.assertEqual(int(plain[2,2]),1)
        _,culled=rasterize(q,tri,8,cull_backfaces=True)
        self.assertEqual(int(culled[2,2]),2)


if __name__=='__main__':unittest.main()
