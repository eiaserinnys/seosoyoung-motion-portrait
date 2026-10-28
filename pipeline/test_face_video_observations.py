"""Contract checks for the video observation inputs and visible-arc selection."""
import unittest

import numpy as np

from face_video_observations import classify_samples, mask_to_video, split_frames


class ObservationContractTest(unittest.TestCase):
    def test_fit_and_validation_cover_the_video_without_overlap(self):
        fit, validation = split_frames()
        self.assertEqual((len(fit), len(validation)), (271, 90))
        self.assertEqual(set(fit) | set(validation), set(range(1, 362)))
        self.assertFalse(set(fit) & set(validation))
        self.assertTrue({1, 34, 88, 298, 256} <= set(fit))
        self.assertIn(89, validation)

    def test_mask_normalization_is_inverted_into_video_coordinates(self):
        ordinary = np.diag([1.125, 1.125, 1.0])
        np.testing.assert_allclose(mask_to_video(ordinary), np.eye(3))
        normalized = np.array([[1.125, 0, 9], [0, 1.125, -3], [0, 0, 1]])
        original = np.array([100., 250., 1.])
        encoded = normalized @ np.diag([640 / 720, 640 / 720, 1]) @ original
        np.testing.assert_allclose(mask_to_video(normalized) @ encoded, original)

    def test_keep_reasons_distinguish_upper_ear_hair_and_visible_arc(self):
        points = np.array([[5, 10], [15, 20], [30, 20], [45, 20], [60, 20]], float)
        normals = np.tile([1., 0.], (5, 1))
        head = np.zeros((80, 80), np.uint8)
        ear = np.zeros_like(head)
        neck = np.zeros_like(head)
        ear[20, 35] = 255
        head[20, 50] = head[20, 65] = 255
        neck[20, 50] = 255
        keep, reasons = classify_samples(points, normals, head, ear, neck, 4.)
        self.assertEqual(keep.tolist(), [False, True, False, True, False])
        self.assertEqual(reasons.tolist(), [1, 0, 2, 0, 3])


if __name__ == '__main__':
    unittest.main()
