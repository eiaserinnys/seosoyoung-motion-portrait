import json,unittest
from pathlib import Path
import numpy as np
from PIL import Image
from seedance_diagnostics import interpolated_holdout
ROOT=Path(__file__).resolve().parents[1]
class SkinFillTest(unittest.TestCase):
    def test_patch_is_clean_and_opaque_at_the_entire_upper_ribbon(self):
        im=np.array(Image.open(ROOT/'bundle/assets/exposed-skin.png'))
        self.assertEqual(im.shape,(1024,1024,4))
        self.assertEqual(int(im[548:,:,3].max()),0)
        self.assertEqual(int(im[200:500,320:700,3].min()),255)
        self.assertGreater(float(im[200:340,400:600,:3].mean(axis=2).min()),200,
                           'brown hairline pixels must not become forehead RGB')
        visible=(im[:,:,3]>240)&(np.indices(im.shape[:2])[0]<410)
        self.assertGreater(float(im[:,:,:3][visible].mean(axis=1).min()),130)
    def test_rim_is_rgb_repair_not_an_alpha_replacement(self):
        im=np.array(Image.open(ROOT/'bundle/assets/ponytail-skin-alpha.png'))
        data=json.loads((ROOT/'source/viewer/ponytail-seedance-registration.json').read_text())
        y_ear=int(np.array(data['frontConstraint']['landmarks'])[7:9,1].min()+10)
        self.assertEqual(int(im[y_ear:,:,3].max()),0)
        self.assertGreater(int(im[:y_ear,:,3].max()),240)
        self.assertGreater(float(im[:,:,:3][im[:,:,3]>240].mean(axis=1).min()),130)
    def test_interpolated_holdout_uses_only_adjacent_training_cameras(self):
        poses=np.arange(60,dtype=float).reshape(10,6)
        poses[3:6]=900
        result=interpolated_holdout(poses,[3,4,5])
        np.testing.assert_allclose(result[3:6],np.arange(60).reshape(10,6)[3:6])
        np.testing.assert_array_equal(result[[2,6]],poses[[2,6]])

class PixelGateTest(unittest.TestCase):
    def fixture(self):
        rgba=np.zeros((1024,1024,4),np.uint8);rgba[250:350,400:605]=[210,190,170,255]
        identified=np.zeros_like(rgba)
        A=[[450,300,-200],[502,300,-200],[550,300,-200]]
        return rgba,identified,dict(innerRow=[0,1,2],face=A,faceRest=[[450,223],[502,223],[550,223]])
    def test_gates_reject_visible_ear_lobes(self):
        from skin_fill_validation import validate_pixels
        a,b,g=self.fixture();validate_pixels('front',a,b,g)
        b[310,600]=[255,0,0,255]
        with self.assertRaisesRegex(AssertionError,'ear attachment'):validate_pixels('front',a,b,g)
    def test_gates_reject_asymmetric_dome(self):
        from skin_fill_validation import validate_pixels
        a,b,g=self.fixture();validate_pixels('down',a,b,g)
        a[250:270,605:610]=[210,190,170,255]
        with self.assertRaisesRegex(AssertionError,'asymmetry'):validate_pixels('down',a,b,g)
    def test_gates_reject_forehead_luminance_valley(self):
        from skin_fill_validation import validate_pixels
        a,b,g=self.fixture();validate_pixels('front',a,b,g)
        a[300,502,:3]=20
        with self.assertRaisesRegex(AssertionError,'luminance valley'):validate_pixels('front',a,b,g)
