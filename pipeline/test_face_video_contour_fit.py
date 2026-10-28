"""Check observed-arc weighting and fixed-camera depth derivatives."""
import unittest

import numpy as np

from face_video_contour import project
from face_video_contour_fit import (correspondences, depth_jacobian, fit_diagnostics,
                                    overlap_step_allowed, feasible_start)


class DirectContourFitTest(unittest.TestCase):
    def test_hidden_observation_has_no_reverse_weight(self):
        observations = np.array([[0., 0.], [1., 0.], [2., 0.]])
        keep = np.array([True, False, True])
        outline = {'points': np.array([[0., .1], [1., .1], [2., .1]]),
                   'edges': np.array([[0, 1], [1, 2], [2, 3]]), 'edge_t': np.zeros(3)}
        matches = correspondences(outline, observations, keep, frames=1)
        self.assertAlmostEqual(matches['weights'].sum(), 1.)
        self.assertFalse(np.any(np.all(matches['target'] == [1., 0.], axis=1)))
        self.assertEqual(len(matches['target']), 4)

    def test_depth_derivative_matches_the_forward_projection(self):
        pose = [25., -17., 2.]
        local = np.array([[22., 180., 80.]])
        changed = local.copy()
        changed[0, 2] += 1e-4
        numeric = (project(changed, pose) - project(local, pose))[0] / 1e-4
        np.testing.assert_allclose(depth_jacobian(pose), numeric, atol=1e-8)

    def test_fit_diagnostics_report_overlap_and_solver_limits(self):
        stats = [{'frame': 1, 'overlap_pairs': 0}, {'frame': 256, 'overlap_pairs': 0}]
        history = [{'lsqr_stop': 1, 'depth_step_rms': .001}]
        self.assertTrue(fit_diagnostics(stats, history)['valid'])
        stats[1]['overlap_pairs'] = 1
        self.assertEqual(fit_diagnostics(stats, history)['overlap_frames'], [256])
        self.assertFalse(fit_diagnostics(stats, history)['valid'])
        stats[1]['overlap_pairs'] = 0
        history[0]['lsqr_stop'] = 7
        self.assertFalse(fit_diagnostics(stats, history)['valid'])
        self.assertEqual(fit_diagnostics(stats, history)['lsqr_limited_iterations'], [1])
        history.append({'lsqr_stop': 1, 'depth_step_rms': .001})
        self.assertTrue(fit_diagnostics(stats, history)['valid'])
        self.assertEqual(fit_diagnostics(stats, history)['lsqr_limited_iterations'], [1])

    def test_trial_step_cannot_introduce_or_increase_overlap(self):
        old = [{'overlap_pairs': 0, 'overlap_area': 0.},
               {'overlap_pairs': 4, 'overlap_area': 3.2}]
        good = [{'overlap_pairs': 0, 'overlap_area': 0.},
                {'overlap_pairs': 2, 'overlap_area': 1.0}]
        self.assertTrue(overlap_step_allowed(old, good))
        good[0] = {'overlap_pairs': 1, 'overlap_area': .1}
        self.assertFalse(overlap_step_allowed(old, good))
        good[0] = {'overlap_pairs': 0, 'overlap_area': 0.}
        good[1]['overlap_area'] = 3.3
        self.assertFalse(overlap_step_allowed(old, good))

    def test_start_depth_changes_only_the_probed_island_and_preserves_prior(self):
        prior = np.arange(6., dtype=float)
        proposal = {'island_vertices': [0, 1], 'feasible': [{'depth_offset': 4}, {'depth_offset': 8}]}
        start = feasible_start(prior, proposal)
        np.testing.assert_array_equal(prior, np.arange(6.))
        np.testing.assert_array_equal(start, [4., 5., 2., 3., 4., 5.])


if __name__ == '__main__':
    unittest.main()
