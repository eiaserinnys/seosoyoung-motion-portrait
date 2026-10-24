import unittest
import numpy as np
from hair_albedo import delight, remove_brow_ghost

class HairPixels(unittest.TestCase):
 def test_rim_is_reduced_without_changing_alpha_or_erasing_texture(self):
  a=np.zeros((100,100,4),np.uint8);a[5:95,5:95]=[65,48,44,255]
  a[8:20,10:90,:3]=[220,160,125]
  a[30:60:2,20:80,:3]=[75,57,52]
  b=delight(a,0,False)
  np.testing.assert_array_equal(a[:,:,3],b[:,:,3])
  self.assertLess(b[10:18,20:80,0].mean(),a[10:18,20:80,0].mean()-50)
  self.assertGreater(b[30:60,20:80,0].std(),1)
 def test_only_disconnected_horizontal_brow_ink_is_removed(self):
  a=np.zeros((80,30,4),np.uint8);a[20:75,12:16]=[60,40,35,255];a[33:36,3:26]=[60,40,35,255]
  brow=np.zeros((80,30),np.uint8);brow[32:37,2:27]=255
  b=remove_brow_ghost(a,brow,300)
  self.assertEqual(int(b[34,6,3]),0)
  self.assertEqual(int(b[34,14,3]),255)

if __name__=='__main__':unittest.main()
