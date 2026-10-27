"""Small checks that head-base measurement gates reject visible regressions."""
import unittest
import numpy as np
from capture_head_base import jacobians, n3_pass, protrusion


class MeasurementContracts(unittest.TestCase):
    def test_protrusion_detects_pixels_beyond_one_pixel(self):
        full = np.zeros((12, 12), bool); full[3:8, 3:8] = True
        skin = np.zeros_like(full)
        self.assertTrue(protrusion(full, full, skin)['passed'])
        head = full.copy(); head[5, 9] = True
        result = protrusion(head, full, skin)
        self.assertFalse(result['passed']); self.assertEqual(result['max_depth_px'], 2.)
        skin[5, 9] = True
        self.assertTrue(protrusion(head, full, skin)['passed'])

    def test_jacobians_detect_reversal_and_anisotropy(self):
        rest = np.array([[0, 220], [1, 220], [0, 221]], float)
        tri = np.array([[0, 1, 2]])
        good = jacobians(rest, rest, tri)
        self.assertAlmostEqual(good['min_det_j'], 1.)
        reversed_points = rest[[0, 2, 1]]
        self.assertLess(jacobians(rest, reversed_points, tri)['min_det_j'], 0.)
        stretched = rest.copy(); stretched[1, 0] = 3.
        self.assertGreater(jacobians(rest, stretched, tri)['max_anisotropy'], 2.)

    def test_contour_threshold_rejects_injected_residual(self):
        values = {'head': {'median_px': 1., 'max_px': 5.}, 'face': {'median_px': 1., 'p95_px': 4.}}
        self.assertTrue(n3_pass(values))
        values['face']['p95_px'] = 6.
        self.assertFalse(n3_pass(values))

if __name__ == '__main__':
    unittest.main()
