import json
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from neck_art import clear_background, load_removal_mask, removal_mask


ROOT = Path(__file__).resolve().parents[1]


class NeckArtTest(unittest.TestCase):
    def test_neck_margin_face_and_collar_conditions(self):
        neck = np.zeros((12, 12), np.uint8)
        face = np.zeros_like(neck)
        sweater = np.zeros_like(neck)
        neck[4, 4] = 255
        face[3, 8] = 255
        sweater[9, 1:11] = 255
        mask = removal_mask(neck, face, sweater)
        self.assertFalse(mask[4, 6], 'two pixel neck dilation')
        self.assertTrue(mask[4, 7], 'beyond the neck dilation')
        self.assertFalse(mask[3, 8], 'face is protected')
        self.assertTrue(mask[6, 8], 'above the collar minus two')
        self.assertFalse(mask[7, 8], 'collar boundary itself is protected')
        self.assertFalse(mask[10, 8], 'clothing-covered fill is protected')
        self.assertFalse(mask[4, 0], 'columns with no sweater are protected')
        original = np.full((12, 12, 4), 200, dtype=np.uint8)
        result = clear_background(original, mask)
        np.testing.assert_array_equal(result[:, :, :3], original[:, :, :3])
        np.testing.assert_array_equal(result[:, :, 3][~mask], original[:, :, 3][~mask])
        self.assertTrue(np.all(result[:, :, 3][mask] == 0))

    def test_neck_and_both_atlases_respect_mask(self):
        parts = json.loads((ROOT / 'parts/parts_index.json').read_text())
        rig = json.loads((ROOT / 'bundle/assets/seosoyoung-rig.json').read_text())
        part = next(part for part in parts if part['id'] == 'neck')
        mesh = next(mesh for mesh in rig['meshes'] if mesh['id'] == 'neck')
        x, y, w, h = part['bbox']
        ax, ay, aw, ah = mesh['atlas']
        self.assertEqual([w, h], [aw, ah])
        remove = load_removal_mask(ROOT)[y:y+h, x:x+w]
        for name in ['parts/neck.png', 'bundle/assets/texture_00.png', 'bundle/assets/texture_before_00.png']:
            image = np.asarray(Image.open(ROOT / name).convert('RGBA'))
            if name.startswith('bundle/'):
                image = image[ay:ay+h, ax:ax+w]
            with self.subTest(name=name):
                self.assertEqual(int(np.count_nonzero(image[:, :, 3][remove])), 0)
        keep = np.asarray(Image.open(ROOT / 'bundle/assets/neck-keep.png').convert('L'))
        np.testing.assert_array_equal(keep, np.where(remove, 0, 255).astype('uint8'))


if __name__ == '__main__':
    unittest.main()
