"""2c geometry contracts, using tiny bounded inputs before real mesh execution."""
import unittest
import numpy as np
from egg_pca import implicit_smooth, boundary_vertices, nose_observations


class EggContracts(unittest.TestCase):
    def test_fixed_boundary_and_same_linear_operator_on_modes(self):
        edges=np.array([[0,1],[0,2],[0,3],[0,4]])
        fields=np.zeros((5,3)); fields[0,2]=1
        mode=np.arange(15,dtype=float).reshape(5,3)/10
        smooth=implicit_smooth(fields,edges,np.array([0]),1.)
        np.testing.assert_array_equal(smooth[1:],fields[1:])
        self.assertAlmostEqual(smooth[0,2],.2)
        combined=implicit_smooth(fields+2*mode,edges,np.array([0]),1.)
        np.testing.assert_allclose(combined,smooth+2*implicit_smooth(mode,edges,np.array([0]),1.),atol=1e-12)

    def test_face_outside_neighbors_define_fixed_ring(self):
        edges=np.array([[0,1],[1,2],[2,3]])
        np.testing.assert_array_equal(boundary_vertices(np.array([True,True,False,False]),edges),[1])

    def test_nose_uses_point_confidence_without_yaw_gate(self):
        points=[[0.,0.,1.] for _ in range(28)]; points[23]=[40.,60.,.7]
        rows=[{'frame':42,'score':.99,'usable':False,'pose':[-60.,0.,0.,0.,0.,0.],'keypoints':points}]
        result=nose_observations({'frames':rows})
        self.assertTrue(result[42]['used']); self.assertEqual(result[42]['score'],.7)
        points[23][2]=.4
        self.assertFalse(nose_observations({'frames':rows})[42]['used'])


if __name__=='__main__':unittest.main()
