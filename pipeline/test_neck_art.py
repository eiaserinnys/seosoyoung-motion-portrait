import json
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from neck_art import clear_background, removal_mask


ROOT = Path(__file__).resolve().parents[1]


class NeckArtTest(unittest.TestCase):
    def test_outside_candidates_only_above_collar(self):
        neck = np.zeros((12, 12, 4), np.uint8)
        neck[2:10, 2:10] = [220, 190, 170, 255]
        still = np.full_like(neck, [220, 190, 170, 255])
        sweater = np.zeros((12, 12), np.uint8)
        sweater[9, 2:10] = 255
        still[3, 2, :3] = 0  # Candidate touches the outside.
        still[5, 6, :3] = 0  # Interior candidate stays.
        still[7, 2, :3] = 0  # Exactly at the collar cutoff stays.
        neck[4, 2, :3] = 20  # The dark drawn outline stays.
        still[4, 2, :3] = 0
        remove, stats = removal_mask(neck, still, sweater, origin=(0, 0))
        self.assertTrue(remove[3, 2])
        self.assertFalse(remove[5, 6])
        self.assertFalse(remove[7, 2])
        self.assertFalse(remove[4, 2])
        self.assertEqual(stats['removed_components'], 1)
        self.assertEqual(stats['retained_pixels'], 1)
        result = clear_background(neck, remove)
        np.testing.assert_array_equal(result[:, :, :3], neck[:, :, :3])
        np.testing.assert_array_equal(result[:, :, 3][~remove], neck[:, :, 3][~remove])

    def test_component_touching_center_stops_before_removal(self):
        neck = np.zeros((6, 6, 4), np.uint8)
        neck[1:5, 1:5] = [220, 190, 170, 255]
        still = np.full_like(neck, [220, 190, 170, 255])
        still[2, 1, :3] = 0
        sweater = np.zeros((6, 6), np.uint8)
        sweater[5] = 255
        with self.assertRaisesRegex(ValueError, '가운데'):
            removal_mask(neck, still, sweater, origin=(494, 0))

    def test_restored_neck_and_both_atlases_match_their_masks(self):
        rig = json.loads((ROOT / 'bundle/assets/seosoyoung-rig.json').read_text())
        mesh = next(mesh for mesh in rig['meshes'] if mesh['id'] == 'neck')
        x, y, w, h = mesh['bbox']
        ax, ay, _, _ = mesh['atlas']
        still = np.asarray(Image.open(ROOT / 'source/character/still.png').convert('RGBA'))[y:y+h, x:x+w]
        sweater = np.asarray(Image.open(ROOT / 'sam3_masks/sweater_mask.png').convert('L'))[:, x:x+w]
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
                remove, _ = removal_mask(original, still, sweater, origin=(x, y))
                current = np.asarray(Image.open(ROOT / 'bundle/assets' / atlas).convert('RGBA'))[ay:ay+h, ax:ax+w]
                np.testing.assert_array_equal(current, clear_background(original, remove))
                np.testing.assert_array_equal(current[:, 495-x:506-x], original[:, 495-x:506-x])
                keep_image = np.asarray(Image.open(ROOT / 'bundle/assets' / keep).convert('L'))
                np.testing.assert_array_equal(keep_image, np.where(remove, 0, 255).astype('uint8'))


if __name__ == '__main__':
    unittest.main()
