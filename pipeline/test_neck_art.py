import json
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from neck_art import clear_background


ROOT = Path(__file__).resolve().parents[1]


class NeckArtTest(unittest.TestCase):
    def test_only_original_transparent_pixels_are_cleared(self):
        artwork = np.array([[[10, 20, 30, 255], [40, 50, 60, 123], [70, 80, 90, 255]]], dtype=np.uint8)
        reference = np.array([[[1, 2, 3, 0], [1, 2, 3, 128], [1, 2, 3, 255]]], dtype=np.uint8)
        result = clear_background(artwork, reference)
        np.testing.assert_array_equal(result[0, :, 3], [0, 123, 255])
        np.testing.assert_array_equal(result[0, :, :3], artwork[0, :, :3])

    def test_neck_and_both_atlases_respect_original_alpha(self):
        parts = json.loads((ROOT / 'parts/parts_index.json').read_text())
        rig = json.loads((ROOT / 'bundle/assets/seosoyoung-rig.json').read_text())
        part = next(part for part in parts if part['id'] == 'neck')
        mesh = next(mesh for mesh in rig['meshes'] if mesh['id'] == 'neck')
        x, y, w, h = part['bbox']
        ax, ay, aw, ah = mesh['atlas']
        self.assertEqual([w, h], [aw, ah])
        still = np.asarray(Image.open(ROOT / 'source/character/still.png').convert('RGBA'))
        background = still[y:y+h, x:x+w, 3] == 0
        for name in ['parts/neck.png', 'bundle/assets/texture_00.png', 'bundle/assets/texture_before_00.png']:
            image = np.asarray(Image.open(ROOT / name).convert('RGBA'))
            if name.startswith('bundle/'):
                image = image[ay:ay+h, ax:ax+w]
            with self.subTest(name=name):
                self.assertEqual(int(np.count_nonzero(image[:, :, 3][background])), 0)


if __name__ == '__main__':
    unittest.main()
