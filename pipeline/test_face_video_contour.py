"""Contracts for the front-face silhouette and pose-only contour fit."""
import unittest

import numpy as np

from face_video_contour import (centered_basis, contour, huber_weights, project,
                                valid_viewer_samples)


class FaceVideoContourTest(unittest.TestCase):
    def test_front_triangles_share_an_external_boundary(self):
        points = np.array([[0., 0.], [2., 0.], [2., 2.], [0., 2.]])
        triangles = np.array([[0, 1, 2], [0, 2, 3]])
        result = contour(points, triangles, spacing=.25)
        self.assertEqual(result['back_faces'], 0)
        self.assertEqual(result['overlap_pairs'], 0)
        self.assertAlmostEqual(result['boundary_length'], 8.)
        self.assertGreater(len(result['points']), 4)

    def test_back_face_is_culled_from_the_boundary(self):
        points = np.array([[0., 0.], [2., 0.], [2., 2.], [3., 1.]])
        result = contour(points, np.array([[0, 1, 2], [0, 2, 3]]), spacing=.25)
        self.assertEqual(result['back_faces'], 1)
        self.assertEqual(result['overlap_pairs'], 0)
        self.assertAlmostEqual(result['boundary_length'], 4 + np.sqrt(8))

    def test_two_overlapping_front_faces_are_not_silently_accepted(self):
        points = np.array([[0., 0.], [3., 0.], [0., 3.], [.5, .5], [2.5, .5], [.5, 2.5]])
        result = contour(points, np.array([[0, 1, 2], [3, 4, 5]]), spacing=.25)
        self.assertEqual(result['back_faces'], 0)
        self.assertEqual(result['overlap_pairs'], 1)
        self.assertGreater(result['overlap_area'], 1.)

    def test_hidden_nearest_observation_does_not_contribute(self):
        all_points = np.array([[0., 0.], [3., 0.], [6., 0.]])
        keep = np.array([True, False, True])
        samples = np.array([[.1, 0.], [3.1, 0.], [5.9, 0.]])
        self.assertEqual(valid_viewer_samples(samples, all_points, keep).tolist(), [True, False, True])

    def test_fixed_rotation_huber_and_neutral_basis(self):
        points = np.array([[0., 242., 20.], [0., 100., 0.]])
        np.testing.assert_allclose(project(points, [0, 0, 30])[0], [502, 568])
        self.assertAlmostEqual(project(points, [30, 20, 0])[1, 0], 502 + 50 * np.sin(np.deg2rad(20)))
        np.testing.assert_allclose(huber_weights([1, 2, 8]), [1, 1, .25])
        np.testing.assert_allclose(centered_basis(0, 0), np.zeros(35), atol=1e-14)


if __name__ == '__main__':
    unittest.main()
