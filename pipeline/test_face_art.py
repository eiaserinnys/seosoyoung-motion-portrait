import json
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from face_art import clean_face, rim_pixels


ROOT = Path(__file__).resolve().parents[1]


class FaceArtTest(unittest.TestCase):
    def test_dark_rim_is_filled_without_changing_alpha_or_other_pixels(self):
        art = np.full((64, 64, 4), [240, 210, 190, 255], dtype=np.uint8)
        rim = np.zeros((64, 64), dtype=np.float32)
        rim[:, :12] = 1
        art[20:40, 5:7, :3] = 50
        art[20:40, 30:32, :3] = 50
        result, stroke = clean_face(art, rim)
        self.assertTrue(stroke[25, 5])
        self.assertFalse(stroke[25, 30])
        self.assertGreater(int(result[25, 5, 0]), 180)
        np.testing.assert_array_equal(result[:, :, 3], art[:, :, 3])
        np.testing.assert_array_equal(result[~stroke], art[~stroke])

    def test_clean_assets_preserve_pixels_outside_mesh_rim(self):
        rig = json.loads((ROOT / 'bundle/assets/seosoyoung-rig.json').read_text())
        face = next(mesh for mesh in rig['meshes'] if mesh['id'] == 'face-skin')
        ax, ay, width, height = face['atlas']
        rim = rim_pixels(face)
        for atlas, clean in [('texture_00.png', 'face-clean.png'),
                             ('texture_before_00.png', 'face-clean-before.png')]:
            original = np.asarray(Image.open(ROOT / 'bundle/assets' / atlas).convert('RGBA'))[ay:ay+height, ax:ax+width]
            result = np.asarray(Image.open(ROOT / 'bundle/assets' / clean).convert('RGBA'))
            with self.subTest(atlas=atlas):
                np.testing.assert_array_equal(result[rim <= 1e-7], original[rim <= 1e-7])
                np.testing.assert_array_equal(result[:, :, 3], original[:, :, 3])


if __name__ == '__main__':
    unittest.main()
