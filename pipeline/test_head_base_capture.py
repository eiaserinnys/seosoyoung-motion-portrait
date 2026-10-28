import unittest

import numpy as np
from PIL import Image

from capture_head_base import neck_gap, neck_opaque_band, still_neck_difference


class NeckGapTest(unittest.TestCase):
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
