import json,unittest
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
class SkinFillTest(unittest.TestCase):
    def test_patch_is_clean_and_fades_before_the_preserved_lower_jaw(self):
        im=np.array(Image.open(ROOT/'bundle/assets/exposed-skin.png'))
        self.assertEqual(im.shape,(1024,1024,4))
        self.assertEqual(int(im[500:,:,3].max()),0)
        # No eyes, eyebrows or hair ink may be painted on the backing surface.
        visible=(im[:,:,3]>240)&(np.indices(im.shape[:2])[0]<410)
        self.assertGreater(float(im[:,:,:3][visible].mean(axis=1).min()),130)
