"""Image evidence for opening envelopes and the front nose detector bias."""
import unittest
import numpy as np
from PIL import Image
from fit_features import ROOT,measure,measure_opening,SEEDS

class FeatureMeasurementTest(unittest.TestCase):
    def test_front_nose_bias_is_five_pixels(self):
        rgb=np.array(Image.open(ROOT/'source/references/head_front_ref.png').convert('RGB'))
        nose=measure(rgb,(508-152,497-20),'nose')
        self.assertEqual(497-nose['center'][1],5)

    def test_down_eye_uses_lash_to_liner_height_instead_of_iris_height(self):
        extents=[]
        for name,file,seed in [('front','head_front_ref.png',(264,412)),('down','down_ref_720.png',SEEDS['down'][0])]:
            rgb=np.array(Image.open(ROOT/'source/references'/file).convert('RGB'))
            iris=measure(rgb,seed,'left');opening=measure_opening(rgb,iris['center'])
            self.assertGreater(opening['openingExtent'][1],iris['extent'][1]+5,name)
            extents.append(opening['openingExtent'][1])
        self.assertAlmostEqual(extents[1]/extents[0],51/82)
        self.assertLess(extents[1]/extents[0],.7)

if __name__=='__main__':unittest.main()
