"""Contracts for centered pose fields and their projection derivatives."""
import unittest

import numpy as np
from scipy import sparse

from face_video_contour import centered_basis
from face_video_residual_fit import (basis_transform, center_field, apply_step, pull_step,
                                     linear_system, objective, roll_matrix, step_sizes)
from face_video_contour import rotation


class ResidualFitTest(unittest.TestCase):
    def test_centered_field_has_no_common_translation(self):
        values = np.array([[[1., 3., 5.], [3., 5., 7.], [5., 7., 9.]]])
        result = center_field(values)
        np.testing.assert_allclose(result.mean(axis=1), 0., atol=1e-14)
        np.testing.assert_allclose(result[0, 2] - result[0, 0], values[0, 2] - values[0, 0])

    def test_pose_coefficients_match_samples_and_vanish_at_neutral(self):
        poses = [[-1, 1], [-36, 7], [0, 23], [-28, -32], [25, -17]]
        basis, transform, _ = basis_transform(poses)
        np.testing.assert_allclose(basis @ transform, np.eye(5), atol=1e-9)
        values = np.arange(15.).reshape(5, 3)
        coeff = transform @ values
        np.testing.assert_allclose(basis @ coeff, values, atol=1e-9)
        np.testing.assert_allclose(centered_basis(0, 0) @ coeff, 0, atol=1e-12)

    def test_linear_pose_field_adds_no_curvature_cost(self):
        poses = [[-1, 1], [-36, 7], [0, 23], [-28, -32], [25, -17]]
        _, _, precision = basis_transform(poses)
        linear_yaw = np.array([p[0] for p in poses], float)
        self.assertAlmostEqual(float(linear_yaw @ precision @ linear_yaw), 0., places=7)

    def test_translation_motion_is_included_in_stopping_size(self):
        field = np.zeros((2, 3, 3))
        translation = np.ones((2, 2))
        c_step, t_step = step_sizes(field, translation, .5)
        self.assertEqual(c_step, 0.)
        self.assertAlmostEqual(t_step, np.sqrt(2) / 2)

    def test_projection_and_adjoint_agree_for_edge_interpolation(self):
        rng = np.random.default_rng(4)
        frames = [
            {'edges': np.array([[0, 1], [1, 2]]), 'edge_t': np.array([.2, .7]),
             'jac': np.array([[.8, .1, .5], [0., .9, -.4]]), 'roll': np.eye(2)},
            {'edges': np.array([[1, 2], [0, 2]]), 'edge_t': np.array([.3, .6]),
             'jac': np.array([[.9, .2, -.2], [.1, .8, .4]]), 'roll': np.array([[0., -1.], [1., 0.]])},
        ]
        c, t = rng.normal(size=(2, 3, 3)), rng.normal(size=(2, 2))
        residuals = [rng.normal(size=(2, 2)) for _ in frames]
        forward = apply_step(c, t, frames)
        grad_c, grad_t = pull_step(residuals, frames, nodes=3)
        lhs = sum(np.sum(a * b) for a, b in zip(forward, residuals))
        rhs = np.sum(c * grad_c) + np.sum(t * grad_t)
        self.assertAlmostEqual(lhs, rhs, places=11)

    def test_linearized_system_matches_the_fixed_correspondence_objective(self):
        rng = np.random.default_rng(9)
        frames = []
        for pose in ([20., -10., 3.], [-15., 12., -2.]):
            q = np.array([[0., 0.], [3., 1.]])
            target = np.array([[.5, .2], [0., 0.]])
            pairs = {'q': q, 'target': target, 'weights': np.array([.25, .25]),
                     'edges': np.array([[0, 1], [1, 2]]), 'edge_t': np.array([.2, .6])}
            frames.append({'pose': pose, 'pairs': pairs, 'distance': np.linalg.norm(q - target, axis=1)})
        field, translation = center_field(rng.normal(size=(2, 3, 3))), rng.normal(size=(2, 2))
        incidence = sparse.csr_matrix([[1., -1., 0.], [0., 1., -1.]])
        precision = np.array([[2., .1], [.1, 1.5]])
        system, _, rhs = linear_system(frames, field, translation, .1, incidence, precision)
        direction = rng.normal(size=len(rhs))
        dc = center_field(direction[:18].reshape(2, 3, 3))
        dt = direction[18:].reshape(2, 2)
        specs = [{'edges': f['pairs']['edges'], 'edge_t': f['pairs']['edge_t'],
                  'jac': rotation(*f['pose'])[:2], 'roll': roll_matrix(f['pose'][2])} for f in frames]
        offsets = apply_step(dc, dt, specs)
        eps = 1e-6
        moved = []
        for frame, offset in zip(frames, offsets):
            pairs = dict(frame['pairs'], q=frame['pairs']['q'] + eps * offset)
            moved.append({'pairs': pairs, 'distance': np.linalg.norm(pairs['q'] - pairs['target'], axis=1)})
        before = objective(frames, field, translation, .1, incidence, precision)
        after = objective(moved, field + eps * dc, translation + eps * dt, .1, incidence, precision)
        self.assertAlmostEqual((after - before) / eps, -np.dot(rhs, direction), places=5)
        second = rng.normal(size=len(rhs))
        self.assertAlmostEqual(np.dot(direction, system @ second), np.dot(second, system @ direction), places=10)


if __name__ == '__main__':
    unittest.main()
