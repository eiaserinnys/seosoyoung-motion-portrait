import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from hair_albedo import delight, transfer_redrawn_hair
from fit_outline import symmetrize

class HairPixels(unittest.TestCase):
 def test_rim_is_reduced_without_changing_alpha_or_erasing_texture(self):
  a=np.zeros((100,100,4),np.uint8);a[5:95,5:95]=[65,48,44,255]
  a[8:20,10:90,:3]=[220,160,125]
  a[30:60:2,20:80,:3]=[75,57,52]
  b=delight(a,0,False)
  np.testing.assert_array_equal(a[:,:,3],b[:,:,3])
  self.assertLess(b[10:18,20:80,0].mean(),a[10:18,20:80,0].mean()-50)
  self.assertGreater(b[30:60,20:80,0].std(),1)
 def test_redrawn_rgb_uses_only_visible_matching_hair_and_preserves_alpha(self):
  original=np.full((50,50,4),[180,120,80,255],np.uint8)
  still=original[:,:,:3].copy();redrawn=np.full_like(still,[65,48,44])
  still[10:20,10:20]=[240,180,150] # Hidden face backing does not match still.
  redrawn[25:35,10:20]=[235,175,145] # Skin between slanted strands.
  redrawn[10:20,30:40]=[45,145,200] # Blue accessory.
  original[40:45,:,3]=70
  b,mask=transfer_redrawn_hair(original,original,still,redrawn,0,0)
  np.testing.assert_array_equal(b[:,:,3],original[:,:,3])
  np.testing.assert_array_equal(b[5,5,:3],redrawn[5,5])
  for y,x in [(15,15),(30,15),(15,35)]:
   np.testing.assert_array_equal(b[y,x],original[y,x]);self.assertEqual(mask[y,x],0)
 def test_pitched_far_jaw_cannot_narrow_more_than_clear_horizontal_reference(self):
  rows=[[465,350,650,330,610,390,670,465]]
  poses=[dict(yaw=yaw,pitch=pitch,yTop=465,yChin=616,
              registration={'topEdgeY':[465,465]},rows=[[465,320 if yaw<0 else 420,570 if yaw<0 else 680]])
         for pitch in [-30,30] for yaw in [-45,0,45]]
  symmetrize(rows,poses)
  up=next(p for p in poses if p['pitch']==30 and p['yaw']==-45)
  self.assertEqual(up['rows'][0][2],610)
 def test_built_fringe_and_back_ribbons_keep_the_complete_original_alpha(self):
  root=Path(__file__).resolve().parents[1]
  for key in ['bangs-left','bangs-right','back-hair-left','back-hair-right']:
   original=np.array(Image.open(root/'source/front-hair-inputs'/f'{key}.png'))
   built=np.array(Image.open(root/'parts'/f'{key}.png'))
   np.testing.assert_array_equal(built[:,:,3],original[:,:,3],err_msg=key)

if __name__=='__main__':unittest.main()
