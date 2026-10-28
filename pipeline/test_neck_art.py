import json
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from neck_art import apply_reference, crop_inputs


ROOT = Path(__file__).resolve().parents[1]


class NeckArtTest(unittest.TestCase):
    def test_masks_take_priority_and_collar_or_missing_columns_are_untouched(self):
        neck = np.full((8, 8, 4), [60, 70, 80, 255], np.uint8)
        still = np.full_like(neck, [200, 190, 180, 255])
        sweater = np.zeros((8, 8), np.uint8)
        sweater[7, 1:7] = 255  # The cutoff is y=5.
        face = np.zeros((8, 8), bool)
        necklace = np.zeros_like(face)
        hair = np.zeros_like(face)
        face[1, 1] = necklace[1, 1] = hair[1, 1] = True
        necklace[1, 2] = hair[1, 2] = True
        hair[1, 3] = True
        still[1, 4] = [12, 13, 14, 0]
        neck[2, 4] = [3, 4, 5, 0]
        result, keep, counts = apply_reference(neck, still, sweater, face, necklace, hair)
        for x in (1, 2, 3):
            np.testing.assert_array_equal(result[1, x], neck[1, x])
        self.assertEqual(result[1, 4, 3], 0)
        self.assertEqual(keep[1, 4], 0)
        np.testing.assert_array_equal(result[2, 4], still[2, 4])
        np.testing.assert_array_equal(result[5], neck[5])
        np.testing.assert_array_equal(result[:, 0], neck[:, 0])
        self.assertEqual([counts[k] for k in ('face', 'necklace', 'hair', 'background')], [1, 1, 1, 1])

    def test_restored_neck_and_both_atlases_follow_the_reference(self):
        rig = json.loads((ROOT / 'bundle/assets/seosoyoung-rig.json').read_text())
        mesh = next(mesh for mesh in rig['meshes'] if mesh['id'] == 'neck')
        x, y, w, h = mesh['bbox']
        ax, ay, _, _ = mesh['atlas']
        inputs = crop_inputs(ROOT, mesh['bbox'])
        manifest = json.loads((ROOT / 'source/character/manifest.json').read_text())
        source = next(item for item in manifest['leaves'] if item['id'] == 'neck')
        bounds = source['stageBounds']
        raw = Image.open(ROOT / 'source/character' / source['src']).convert('RGBA').resize(
            (bounds['width'], bounds['height']), Image.Resampling.LANCZOS)
        canvas = Image.new('RGBA', (1024, 1536))
        canvas.paste(raw, (bounds['x'], bounds['y']))
        before = np.asarray(canvas.crop((x, y, x+w, y+h)))
        after = np.asarray(Image.open(ROOT / 'parts/neck-untrimmed.png').convert('RGBA'))
        for original, atlas, keep in [(after, 'texture_00.png', 'neck-keep.png'),
                                      (before, 'texture_before_00.png', 'neck-keep-before.png')]:
            with self.subTest(atlas=atlas):
                expected, keep_expected, _ = apply_reference(original, *inputs, origin=(x, y))
                current = np.asarray(Image.open(ROOT / 'bundle/assets' / atlas).convert('RGBA'))[ay:ay+h, ax:ax+w]
                np.testing.assert_array_equal(current, expected)
                keep_image = np.asarray(Image.open(ROOT / 'bundle/assets' / keep).convert('L'))
                np.testing.assert_array_equal(keep_image, keep_expected)


if __name__ == '__main__':
    unittest.main()
