"""Mesh silhouette coverage, including a concave notch and a thin strand."""
import unittest
import numpy as np
import cv2
from PIL import Image
from silhouette_mesh import silhouette_mesh
class SilhouetteTest(unittest.TestCase):
 def test_concavity_and_coverage(self):
  mask=np.zeros((120,120),np.uint8)
  cv2.fillPoly(mask,[np.array([[10,10],[105,10],[105,105],[70,105],[70,50],[48,50],[48,105],[10,105]])],255)
  mask[12:100,110:113]=255
  im=Image.fromarray(np.dstack([mask,mask,mask,mask]))
  result=silhouette_mesh(im,[0,0,120,120],[0,0,120,120],2048)
  raster=np.zeros_like(mask)
  for t in result['triangles']:cv2.fillConvexPoly(raster,np.round(np.array([result['positions'][i] for i in t])).astype(np.int32),255)
  self.assertGreater(np.count_nonzero((mask>0)&(raster>0))/np.count_nonzero(mask),.995)
  self.assertEqual(np.count_nonzero(raster[60:100,53:65]),0)
  self.assertEqual(result['topology'],'alpha-silhouette')
 def test_horizon_is_an_edge_not_a_crossing_triangle(self):
  im=Image.new('RGBA',(100,100),(80,40,20,255))
  seam=[(60,float(y)) for y in range(10,91,2)]
  result=silhouette_mesh(im,[0,0,100,100],[0,0,100,100],2048,seams=[seam])
  points=np.array(result['positions'])
  for indices in result['triangles']:
   tri=points[indices]
   if 10<tri[:,1].mean()<90:
    self.assertFalse(tri[:,0].min()<60-1e-7 and tri[:,0].max()>60+1e-7)
if __name__=='__main__':unittest.main()
