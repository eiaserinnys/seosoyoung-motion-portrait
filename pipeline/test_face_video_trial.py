"""Small interface checks for the fixed-angle trial geometry."""
import unittest

import numpy as np

from face_video_trial import (centered_basis, project, ordered_correspondence,
                              visible_prefix, remove_inversions, triangle_determinants)


class FaceVideoTrialTest(unittest.TestCase):
    def test_neutral_projection_preserves_artwork_xy(self):
        local = np.array([[0., 242., 80.], [30., -20., 40.]])
        np.testing.assert_allclose(project(local, [0, 0, 0]), local[:, :2] + [502, 326])
        np.testing.assert_allclose(project(local[:1], [0, 0, 37]), [[502, 568]])

    def test_pitch_then_yaw_carries_y_into_x(self):
        projected = project(np.array([[0., 100., 0.]]), [30, 20, 0])[0]
        self.assertAlmostEqual(projected[0], 502 + 100 * np.sin(np.deg2rad(20)) * .5)
        self.assertAlmostEqual(projected[1], 326 + 100 * np.cos(np.deg2rad(20)))

    def test_pose_basis_is_zero_at_neutral_and_continuous(self):
        self.assertEqual(len(centered_basis(0, 0)), 35)
        np.testing.assert_allclose(centered_basis(0, 0), 0, atol=1e-14)
        self.assertLess(np.max(np.abs(centered_basis(45, 10) - centered_basis(44.999999, 10))), 1e-6)

    def test_correspondence_preserves_point_order(self):
        source = np.array([[0., 0.], [1., 1.], [2., 2.]])
        target = np.array([[0., 0.], [.5, .5], [1., 1.], [1.5, 1.5], [2., 2.]])
        matching = ordered_correspondence(source, target)
        self.assertEqual(len(matching), len(source))
        self.assertTrue(np.all(np.diff(matching) >= 0))
        self.assertEqual((matching[0], matching[-1]), (0, 4))

    def test_observation_only_constrains_the_visible_arc(self):
        source = np.column_stack((np.zeros(11), np.arange(11)))
        target = np.column_stack((np.zeros(6), np.arange(6)))
        self.assertEqual(len(visible_prefix(source, target)), 6)

    def test_inversion_repair_keeps_observed_boundary_fixed(self):
        rest = np.array([[0., 0.], [2., 0.], [2., 2.], [0., 2.], [1., 1.]])
        triangles = [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]
        posed = rest.copy()
        posed[4] = [2.5, 1.]
        fixed = np.array([0, 1, 2, 3])
        repaired = remove_inversions(rest, posed, triangles, fixed)
        np.testing.assert_array_equal(repaired[fixed], posed[fixed])
        self.assertTrue(np.all(triangle_determinants(rest, repaired, triangles) > 0))


if __name__ == '__main__':
    unittest.main()
