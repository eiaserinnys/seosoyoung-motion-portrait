import json
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from face_art import clean_face, mesh_distances


ROOT = Path(__file__).resolve().parents[1]


class FaceArtTest(unittest.TestCase):
    def test_nearest_interior_color_extends_only_into_the_outer_nine_pixels(self):
        art = np.full((48, 48, 4), [240, 210, 190, 255], dtype=np.uint8)
        art[20:28, 2:10, :3] = 40
        art[20:28, 2:10, 3] = 100
        mesh = {'bbox': [0, 0, 48, 48], 'positions': [[0, 0], [48, 0], [48, 48], [0, 48]],
                'triangles': [[0, 1, 2], [0, 2, 3]]}
        distance, inside = mesh_distances(mesh)
        self.assertAlmostEqual(distance[22, 4], 4.5)
        self.assertAlmostEqual(distance[22, 9], 9.5)
        self.assertAlmostEqual(distance[22, 12], 12.5)
        result, changed = clean_face(art, distance, inside)
        self.assertTrue(changed[22, 4])
        self.assertGreater(int(result[22, 4, 0]), 200)
        np.testing.assert_array_equal(result[:, :, 3], art[:, :, 3])
        np.testing.assert_array_equal(result[distance > 12], art[distance > 12])

    def test_clean_assets_preserve_original_alpha_and_interior(self):
        rig = json.loads((ROOT / 'bundle/assets/seosoyoung-rig.json').read_text())
        face = next(mesh for mesh in rig['meshes'] if mesh['id'] == 'face-skin')
        ax, ay, width, height = face['atlas']
        distance, inside = mesh_distances(face)
        for atlas, clean in [('texture_00.png', 'face-clean.png'),
                             ('texture_before_00.png', 'face-clean-before.png')]:
            original = np.asarray(Image.open(ROOT / 'bundle/assets' / atlas).convert('RGBA'))[ay:ay+height, ax:ax+width]
            result = np.asarray(Image.open(ROOT / 'bundle/assets' / clean).convert('RGBA'))
            with self.subTest(atlas=atlas):
                np.testing.assert_array_equal(result[distance > 12], original[distance > 12])
                np.testing.assert_array_equal(result[:, :, 3], original[:, :, 3])
                self.assertTrue(np.any(inside & (distance > 12)))


if __name__ == '__main__':
    unittest.main()
