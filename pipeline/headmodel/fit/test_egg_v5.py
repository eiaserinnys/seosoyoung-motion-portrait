"""A single front-facing patch replaces folded cavities without moving its rim."""
import unittest
import numpy as np
from egg_v5 import triangulate_polygon,refine_centroids,interpolate_fields,grid_patch,contour_change


class EggV5Contracts(unittest.TestCase):
    def test_concave_polygon_is_covered_once_and_boundary_edges_survive(self):
        xy=np.array([[0.,0.],[3.,0.],[3.,3.],[1.5,1.5],[0.,3.]])
        tri=triangulate_polygon(xy)
        cross=np.cross(np.pad(xy[tri[:,1]]-xy[tri[:,0]],((0,0),(0,1))),
                       np.pad(xy[tri[:,2]]-xy[tri[:,0]],((0,0),(0,1))))[:,2]
        self.assertTrue((cross>0).all()); self.assertAlmostEqual(float(cross.sum()/2),6.75)
        refined,faces=refine_centroids(xy,tri,2)
        np.testing.assert_array_equal(refined[:len(xy)],xy)
        edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
        unique,counts=np.unique(edges,axis=0,return_counts=True)
        actual=set(map(tuple,unique[counts==1]))
        expected={tuple(sorted((i,(i+1)%len(xy)))) for i in range(len(xy))}
        self.assertEqual(actual,expected)

    def test_fixed_uv_interpolator_is_linear_and_reproduces_xy(self):
        anchors=np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.],[0.,2.]])
        queries=np.array([[0.,0.],[.25,.25]])
        base=np.column_stack([anchors,2.+anchors[:,0]-.2*anchors[:,1]])
        modes=np.column_stack([anchors[:,0]*.1,anchors[:,1]*.2,np.ones(5)*.3])
        result=interpolate_fields(anchors,queries,np.c_[base,modes])
        np.testing.assert_allclose(result[:,:2],queries,atol=1e-12)
        np.testing.assert_allclose(interpolate_fields(anchors,queries,base+2*modes),
                                   result[:,:3]+2*result[:,3:],atol=1e-12)

    def test_grid_patch_preserves_concave_rim_and_rejects_large_contour_change(self):
        polygon=np.array([[0.,0.],[3.,0.],[3.,3.],[1.5,1.5],[0.,3.]])
        xy,tri=grid_patch(polygon,.25)
        edges=np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),axis=1)
        unique,counts=np.unique(edges,axis=0,return_counts=True)
        self.assertEqual(set(map(tuple,unique[counts==1])),{tuple(sorted((i,(i+1)%5))) for i in range(5)})
        self.assertTrue((counts<=2).all())
        reference=np.zeros((100,100),bool); reference[20:80,20:80]=True
        changed=reference.copy(); changed[20:80,80:90]=True
        self.assertFalse(contour_change(reference,changed)['within_five_percent'])
        self.assertTrue(contour_change(reference,reference)['within_five_percent'])


if __name__=='__main__':unittest.main()
