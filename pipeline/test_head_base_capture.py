import unittest

import numpy as np
from PIL import Image

from capture_head_base import closed_hole_locations, hole_mesh_samples, neck_gap, neck_opaque_band, still_neck_difference


class NeckGapTest(unittest.TestCase):
    def test_closed_hole_locations_use_canvas_coordinates(self):
        mask = np.zeros((8, 250), bool)
        mask[1:7, 239:249] = True
        mask[3, 243] = False
        self.assertEqual(closed_hole_locations(mask), [[5, 3]])

    def test_hole_sample_reports_source_texel_and_alpha(self):
        mesh = {'triangles': [[0, 1, 2]], 'uvs': [[.125, .125], [.5, .125], [.125, .5]],
                'atlas': [0, 0, 8, 8], 'bbox': [0, 0, 8, 8]}
        atlas = np.full((8, 8, 4), 255, np.uint8)
        atlas[1, 1, 3] = 4
        result = hole_mesh_samples((0, 0), mesh, [[0, 0], [3, 0], [0, 3]], atlas)
        self.assertEqual(result, [{'triangle': 0, 'canvas_texel': [1, 1],
                                   'atlas_texel': [1, 1], 'alpha': 4}])

    def test_first_opaque_row_with_interior_and_gap_pixels(self):
        rgba = np.zeros((10, 1500, 4), dtype=np.uint8)
        rgba[2, 248] = [255, 0, 0, 255]
        rgba[3, 245:252] = [255, 0, 0, 255]
        band = neck_opaque_band(Image.fromarray(rgba))
        self.assertEqual(band, {'row': 3, 'left': 8, 'right': 12})
        head = np.zeros((10, 1500), dtype=bool)
        neck = np.zeros_like(head)
        head[2, 246:251] = True
        neck[6, 246:251] = True
        gap, columns, mask = neck_gap(head, neck, head | neck, (band['left'], band['right']))
        self.assertEqual((gap, columns, int(mask.sum())), (15, 5, 15))

    def test_missing_opaque_art_is_not_reported_as_zero_gap(self):
        rgba = np.zeros((10, 1500, 4), dtype=np.uint8)
        self.assertIsNone(neck_opaque_band(Image.fromarray(rgba)))

    def test_still_difference_uses_only_the_neck_mask_and_aligned_canvas(self):
        still = np.full((8, 8, 4), [150, 120, 100, 255], dtype=np.uint8)
        rendered = still.copy()
        rendered[3, 4, 1] += 7
        rendered[0, 0, 0] += 20
        mask = np.zeros((8, 8), bool)
        mask[3, 4] = True
        result, _, _, _ = still_neck_difference(Image.fromarray(rendered), Image.fromarray(still), mask)
        self.assertEqual(result['pixels'], 1)
        self.assertEqual(result['changed_pixels'], 1)
        self.assertEqual(result['max_channel_difference'], 7)


if __name__ == '__main__':
    unittest.main()
