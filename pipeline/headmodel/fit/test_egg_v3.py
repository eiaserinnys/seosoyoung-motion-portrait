"""Local feature mask must preserve outer facial shape and reject rear surfaces."""
import unittest
import numpy as np
from egg_v3 import expanded_hull, inside_polygon, feature_mask, extended_selection


class EggV3Contracts(unittest.TestCase):
    def test_margin_is_absolute_outward_distance_and_inside_is_inclusive(self):
        square=np.array([[0.,0.],[2.,0.],[2.,2.],[0.,2.],[1.,1.]])
        polygon=expanded_hull(square,.5)
        np.testing.assert_allclose(polygon.min(0),[-.5,-.5])
        np.testing.assert_allclose(polygon.max(0),[2.5,2.5])
        np.testing.assert_array_equal(inside_polygon(np.array([[-.5,1.],[-.51,1.],[1.,1.]]),polygon),[True,False,True])

    def test_mask_rejects_back_facing_outside_face_and_outer_contour(self):
        vertices=np.array([[0.,0.,1.],[0.,0.,-1.],[0.,0.,1.],[3.,0.,1.]])
        normals=np.array([[0.,0.,1.],[0.,0.,-1.],[0.,0.,1.],[0.,0.,1.]])
        polygon=np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])
        mask=feature_mask(vertices,normals,np.array([True,True,False,True]),polygon)
        np.testing.assert_array_equal(mask,[True,False,False,False])

    def test_new_cap_uses_its_own_normal_even_when_original_ring_is_fixed(self):
        vertices=np.array([[-1.,-1.,0.],[1.,-1.,0.],[0.,1.,0.],[0.,0.,0.]])
        tri=np.array([[0,1,3],[1,2,3],[2,0,3]])
        polygon=np.array([[-2.,-2.],[2.,-2.],[2.,2.],[-2.,2.]])
        result=extended_selection(np.array([False]*3),vertices,tri,np.ones(4,bool),polygon)
        np.testing.assert_array_equal(result,[False,False,False,True])


if __name__=='__main__':unittest.main()
