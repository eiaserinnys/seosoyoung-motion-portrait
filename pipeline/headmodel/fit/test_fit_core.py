"""Observable contracts for the three-frame experiment (no video landmarks)."""
import unittest

import cv2
import numpy as np
import torch

from fit_core import (boundary, lower_boundary, clean_mask, eroded_target, define_regions,
                      project, normalize_pose, rasterize, symmetric_distance)


class FitContracts(unittest.TestCase):
    def test_pose_affine_composition(self):
        pose = np.array([-36., 5., 3., .04, 266., 402.])
        angle = np.radians(-3.)
        a = 1.2 * np.array([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle), np.cos(angle)]])
        affine = np.column_stack([a, [80., 20.]])
        vertices = torch.tensor([[2., 3., 5.], [-2., -4., 7.]])
        before = project(vertices, torch.tensor(pose, dtype=torch.float32)).detach().numpy()
        after = project(vertices, torch.tensor(normalize_pose(pose, affine), dtype=torch.float32))
        np.testing.assert_allclose(after.numpy()[:, :2], before[:, :2] @ a.T + affine[:, 2], atol=1e-4)

    def test_axis_sign_and_order(self):
        # ICT Y is up; positive source yaw turns the nose right, pitch turns it up.
        v = torch.tensor([[0., 0., 10.], [0., 2., 0.]])
        p = torch.tensor([30., 20., 0., 0., 0., 0.])
        q = project(v, p)
        self.assertGreater(q[0, 0], 0)
        self.assertLess(q[0, 1], 0)
        self.assertLess(q[1, 1], 0)
        self.assertAlmostEqual(float(q[0, 0]), 10 * .5 * np.cos(np.radians(20)), places=5)

    def test_skin_subtraction_fill_and_lower_only(self):
        face = np.zeros((80, 80), np.uint8)
        cv2.ellipse(face, (40, 30), (22, 20), 0, 0, 360, 1, -1)
        face[20:25, 38:42] = 0
        face[48:77, 34:47] = 1
        neck = np.zeros_like(face); neck[51:, :] = 1
        skin = clean_mask(face & (1-neck))
        self.assertTrue(skin[22, 40])
        self.assertFalse(skin[70, 40])
        bottom = lower_boundary(skin)
        self.assertGreater(bottom[:, 1].min(), 25)
        # Deliberate upper-contour substitution must fail the lower-only assertion.
        self.assertLess(boundary(skin)[:, 1].min(), 20)

    def test_eroded_target_and_symmetric_distance_reject_mismatch(self):
        mask = np.zeros((80, 80), np.uint8); mask[10:71, 10:71] = 1
        eroded = eroded_target(mask, 7.)
        self.assertFalse(eroded[12, 40]); self.assertTrue(eroded[40, 40])
        pts = torch.tensor([[0., 0.], [1., 0.], [2., 0.]])
        exact = symmetric_distance(pts, pts)
        shifted = symmetric_distance(pts, pts + torch.tensor([0., 12.]))
        self.assertLess(float(exact), .001)
        self.assertGreater(float(shifted), 11.9)

    def test_z_buffer_occlusion(self):
        xy = np.array([[5.,5.], [25.,5.], [15.,25.]])
        q = np.r_[np.column_stack([xy, np.ones(3)]),
                  np.column_stack([xy, np.full(3, 3.)])]
        depth, owner = rasterize(q, np.array([[0,1,2], [3,4,5]]), 32)
        self.assertEqual(owner[10,15], 1)
        self.assertAlmostEqual(depth[10,15], 3.)

    def test_report_metrics_reject_shifted_contours_and_disjoint_ears(self):
        from fit_sheets import contour_residual, iou
        line = np.column_stack([np.full(21, 300), np.arange(300, 321)])
        exact = contour_residual(line, line)
        violation = contour_residual(line, line + [30, 0])
        self.assertEqual(exact['max_px'], 0.)
        self.assertAlmostEqual(violation['mean_px'], 30.)
        self.assertGreater(violation['max_px'], 26.)
        ear = np.zeros((80, 80), bool); ear[10:20, 10:20] = True
        disjoint = np.zeros_like(ear); disjoint[40:50, 40:50] = True
        self.assertEqual(iou(ear, ear), 1.)
        self.assertLess(iou(ear, disjoint), .3)

    def test_skin_surface_excludes_upper_face_and_neck(self):
        vertices = np.array([[0.,2.,10.],[1.,2.,10.],[0.,3.,10.],
                             [0.,0.,10.],[1.,0.,10.],[0.,-1.,10.],
                             [0.,-12.,0.],[1.,-12.,0.],[0.,-13.,0.]])
        triangles = np.arange(9).reshape(3,3)
        _, faces = define_regions(vertices, triangles)
        # The above-eye temple surface and excluded neck are not lower skin.
        np.testing.assert_array_equal(faces['face'], [1])


if __name__ == '__main__':
    unittest.main()
