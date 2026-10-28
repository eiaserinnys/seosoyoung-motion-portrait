"""Small checks that head-base measurement gates reject visible regressions."""
import unittest
import numpy as np
from capture_head_base import closed_holes, contour, jacobians, n3_metrics


class MeasurementContracts(unittest.TestCase):
    def test_closed_holes_exclude_background_connected_to_the_edge(self):
        mask = np.zeros((12, 12), bool); mask[2:10, 2:10] = True
        self.assertEqual(closed_holes(mask), 0)
        mask[5:7, 5:7] = False
        self.assertEqual(closed_holes(mask), 4)
        mask[2:5, 5] = False
        self.assertEqual(closed_holes(mask), 0)

    def test_jacobians_detect_reversal_and_anisotropy(self):
        rest = np.array([[0, 220], [1, 220], [0, 221]], float)
        tri = np.array([[0, 1, 2]])
        good = jacobians(rest, rest, tri)
        self.assertAlmostEqual(good['min_det_j'], 1.)
        reversed_points = rest[[0, 2, 1]]
        self.assertLess(jacobians(rest, reversed_points, tri)['min_det_j'], 0.)
        stretched = rest.copy(); stretched[1, 0] = 3.
        self.assertGreater(jacobians(rest, stretched, tri)['max_anisotropy'], 2.)

    def test_bidirectional_contours_reject_an_injected_shift(self):
        mask = np.zeros((40, 40), bool); mask[10:30, 10:30] = True
        points = contour(mask)
        skin, hair = points[points[:, 0] < 20], points[points[:, 0] >= 20]
        good = n3_metrics(mask, skin, hair)
        self.assertTrue(good['passed'])
        shifted = np.zeros_like(mask); shifted[10:30, 18:38] = True
        self.assertFalse(n3_metrics(shifted, skin, hair)['passed'])

if __name__ == '__main__':
    unittest.main()
