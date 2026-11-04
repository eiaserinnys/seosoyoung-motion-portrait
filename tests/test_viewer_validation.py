import unittest

from pipeline.verify_viewer import nonfront_passes, holes_within_limits


class ViewerValidationTests(unittest.TestCase):
    def test_nonfront_metrics_allow_exact_v6_limits(self):
        metrics = {
            'value_over_2_percent_visible': 0.6,
            'alpha_over_8': 110,
            'outside_edge_band_value_over_2': 4000,
        }
        self.assertTrue(nonfront_passes(metrics))

    def test_nonfront_metrics_reject_each_exceeded_v6_limit(self):
        limits = {
            'value_over_2_percent_visible': 0.6,
            'alpha_over_8': 110,
            'outside_edge_band_value_over_2': 4000,
        }
        for name, limit in limits.items():
            with self.subTest(metric=name):
                metrics = dict(limits)
                metrics[name] = limit + 0.001
                self.assertFalse(nonfront_passes(metrics))

    def test_hole_pixels_and_components_must_not_exceed_v6_limits(self):
        self.assertTrue(holes_within_limits(24, 6, [24, 6]))
        self.assertFalse(holes_within_limits(25, 6, [24, 6]))
        self.assertFalse(holes_within_limits(24, 7, [24, 6]))


if __name__ == '__main__':
    unittest.main()
