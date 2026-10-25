import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from pipeline.hair_albedo import delight, transfer_redrawn_hair
from pipeline.fit_outline import symmetrize

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
 def test_pitch_outline_observations_are_never_mirror_averaged(self):
  import copy
  rows=[[465,350,650,330,610,390,670,465]]
  poses=[dict(yaw=yaw,pitch=pitch,rows=[[465,465,320+yaw,465,680+yaw]])
         for pitch in [-30,30] for yaw in [-45,0,45]]
  before=copy.deepcopy(poses)
  symmetrize(rows,poses)
  for old,new in zip(before,poses):self.assertEqual(old['rows'],new['rows'])
 def test_measured_chin_registration_and_monotone_profiles(self):
  import json
  root=Path(__file__).resolve().parents[1]
  traces=json.loads((root/'source/viewer/outline-traces.json').read_text())
  for pose in traces['poses']:
   if not pose['pitch']:continue
   r=pose['registration']
   chin=r['eyeMid'][1]+(pose['rawYChin']-r['referenceEyeMid'][1])*r['scale']
   for side in ['left','right']:self.assertAlmostEqual(pose['mapping']['canvasYSpan'][side][1],chin)
   self.assertGreaterEqual(pose['rows'][-1][4]-pose['rows'][-1][2],11-1e-7)
 def test_sclera_boundary_fixture_matches_the_actual_opaque_parts(self):
  import json
  root=Path(__file__).resolve().parents[1]
  rig=json.loads((root/'bundle/assets/seosoyoung-rig.json').read_text())
  fixture=json.loads((root/'pipeline/fixtures/sclera-alpha-boundary.json').read_text())
  for side in ['left','right']:
   key=f'eye-{side}-sclera';mesh=next(m for m in rig['meshes'] if m['id']==key)
   alpha=np.array(Image.open(root/'parts'/f'{key}.png'))[:,:,3]
   points=[]
   for y in range(alpha.shape[0]):
    xs=np.flatnonzero(alpha[y]>127)
    if len(xs):points.extend([[int(x)+mesh['bbox'][0],y+mesh['bbox'][1]] for x in [xs.min(),xs.max()]])
   self.assertEqual(fixture[side],points)
 def test_built_fringe_and_back_ribbons_keep_the_complete_original_alpha(self):
  root=Path(__file__).resolve().parents[1]
  for key in ['bangs-left','bangs-right','back-hair-left','back-hair-right']:
   original=np.array(Image.open(root/'source/front-hair-inputs'/f'{key}.png'))
   built=np.array(Image.open(root/'parts'/f'{key}.png'))
   np.testing.assert_array_equal(built[:,:,3],original[:,:,3],err_msg=key)
 def test_built_lower_fringe_rgb_is_identical_to_original(self):
  import json
  root=Path(__file__).resolve().parents[1]
  parts={p['id']:p for p in json.loads((root/'parts/parts_index.json').read_text())}
  for key in ['bangs-left','bangs-right','side-hair-left']:
   original=np.array(Image.open(root/'source/front-hair-inputs'/f'{key}.png'))
   built=np.array(Image.open(root/'parts'/f'{key}.png'))
   x,y,_,_=parts[key]['bbox'];yy,xx=np.indices(original.shape[:2])
   protected=(yy+y>=310)&(xx+x<640)&(original[:,:,3]>200)
   self.assertTrue(protected.any(),key)
   np.testing.assert_array_equal(built[protected,:3],original[protected,:3],err_msg=key)
 def test_cranium_top_is_opaque_and_lower_edge_fades_out(self):
  root=Path(__file__).resolve().parents[1]
  for side in ['left','right']:
   alpha=np.array(Image.open(root/'parts'/f'side-cranium-{side}.png'))[:,:,3]
   self.assertTrue((alpha[:416]==255).all())
   self.assertTrue((alpha[505:]==0).all())
   self.assertTrue(((alpha[466:505]>0)&(alpha[466:505]<255)).any())
 def test_hairpin_notch_has_opaque_side_hair_and_backing_before_the_ear(self):
  import json
  root=Path(__file__).resolve().parents[1]
  parts={p['id']:p for p in json.loads((root/'parts/parts_index.json').read_text())}
  p=parts['side-hair-right'];x,y,_,_=p['bbox']
  a=np.array(Image.open(root/p['file']))[:,:,3]
  self.assertTrue((a[340-y:407-y,634-x:660-x]==255).all())
  p=parts['back-hair-center'];x,y,w,_=p['bbox']
  self.assertGreaterEqual(x+w,700,'backing crop must contain the requested canvas x700 endpoint')
  a=np.array(Image.open(root/p['file']))[:,:,3];end=min(w,700-x)
  self.assertTrue((a[350-y:408-y,end-2:end]==255).all())

if __name__=='__main__':unittest.main()
