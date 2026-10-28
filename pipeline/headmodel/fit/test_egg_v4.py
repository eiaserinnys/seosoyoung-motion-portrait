"""All feature-hull vertices, including inward rims and cap centers, are free."""
import unittest
import numpy as np
from egg_v4 import feature_mask
from egg_v3 import extended_selection


class EggV4Contracts(unittest.TestCase):
    def test_back_facing_rim_is_free_but_outside_face_or_polygon_is_fixed(self):
        vertices=np.array([[0.,0.,1.],[0.,0.,-1.],[0.,0.,1.],[3.,0.,1.]])
        normals=np.array([[0.,0.,1.],[0.,0.,-1.],[0.,0.,1.],[0.,0.,1.]])
        polygon=np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])
        selected=feature_mask(vertices,normals,np.array([True,True,False,True]),polygon)
        np.testing.assert_array_equal(selected,[True,True,False,False])

    def test_cap_center_is_free_regardless_of_its_normal(self):
        vertices=np.array([[-1.,-1.,0.],[1.,-1.,0.],[0.,1.,0.],[0.,0.,0.]])
        tri=np.array([[1,0,3],[2,1,3],[0,2,3]])  # Center normal -z.
        polygon=np.array([[-2.,-2.],[2.,-2.],[2.,2.],[-2.,2.]])
        selected=extended_selection(np.ones(3,bool),vertices,tri,np.ones(4,bool),polygon,selector=feature_mask)
        np.testing.assert_array_equal(selected,[True,True,True,True])


if __name__=='__main__':unittest.main()
