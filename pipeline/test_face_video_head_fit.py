"""Small contracts for tracker coordinates and the centered head placement."""
import unittest

import numpy as np
from scipy import sparse
from scipy.optimize import least_squares

from face_video_contour import centered_basis, project
from face_video_contour_fit import edge_incidence
from face_video_head_fit import (FrozenProblem, NECK_PIVOT, SCALE_UNIT, apply_head,
                                actual_objective, evaluate, skin_depth, stable_boundary, tracked_head)


class HeadPlacementTest(unittest.TestCase):
    def test_skin_depth_uses_existing_triangle(self):
        rest = np.array([[0., 0.], [2., 0.], [0., 2.]])
        triangles = np.array([[0, 1, 2]])
        self.assertAlmostEqual(skin_depth([.5, .5], rest, [10., 14., 18.], triangles), 13.)

    def test_tracked_head_uses_body_matrix_once_and_matches_tracker_origin(self):
        pose = [25., -18., 4., np.log(1.2), 130., 240.]
        body = np.array([[2., 0., 12.], [0., 2., -5.]])
        anchor = np.array([501., 430., 105.])
        placement = tracked_head(pose, body, anchor)
        actual_scale = 2.4
        tracked_origin = body[:, :2] @ pose[4:6] + body[:, 2]
        self.assertAlmostEqual(np.exp(placement[2] / SCALE_UNIT), actual_scale)
        origin_before_head = project((anchor - [502., 326., 0.])[None, :], pose[:3])
        np.testing.assert_allclose(apply_head(origin_before_head, placement)[0], tracked_origin, atol=1e-12)

    def test_neutral_basis_produces_identity_placement(self):
        coefficients = np.arange(105., dtype=float).reshape(35, 3)
        placement = centered_basis(0, 0) @ coefficients
        points = np.array([[300., 420.], [700., 610.]])
        np.testing.assert_allclose(apply_head(points, placement), points, atol=1e-12)
        np.testing.assert_allclose(placement, 0, atol=1e-12)

    def test_placement_scales_about_neck_and_moves_once(self):
        points = np.array([NECK_PIVOT, NECK_PIVOT + [20., -40.]])
        placement = np.array([5., -3., SCALE_UNIT * np.log(1.25)])
        expected = np.array([NECK_PIVOT + [5., -3.], NECK_PIVOT + [30., -53.]])
        np.testing.assert_allclose(apply_head(points, placement), expected)

    def test_frozen_jacobian_matches_finite_difference_and_adjoint(self):
        rng = np.random.default_rng(31)
        rest = np.array([[400., 400.], [600., 410.], [500., 600.]])
        frames = []
        for pose in ([20., -10., 3.], [-15., 12., -2.]):
            pairs = {'edges': np.array([[0, 1], [1, 2], [0, 2]]),
                     'edge_t': np.array([.2, .6, .4]),
                     'target': np.array([[410., 410.], [550., 490.], [450., 490.]]),
                     'weights': np.array([.15, .2, .15])}
            frames.append({'pose': pose, 'pairs': pairs, 'distance': np.array([2., 3., 1.])})
        incidence = sparse.csr_matrix([[1., -1., 0.], [0., 1., -1.]])
        factor = np.array([[1., -.1], [.3, .5]])
        problem = FrozenProblem(rest, np.array([100., 110., 90.]), frames, incidence, factor, .1)
        x = rng.normal(size=24)
        direction = rng.normal(size=24)
        jacobian = problem.jacobian(x)
        eps = 1e-6
        numerical = (problem.residual(x + eps * direction) - problem.residual(x - eps * direction)) / (2 * eps)
        np.testing.assert_allclose(jacobian @ direction, numerical, rtol=1e-6, atol=2e-8)
        y = rng.normal(size=problem.residual_size)
        self.assertAlmostEqual(float(y @ (jacobian @ direction)), float(direction @ (jacobian.T @ y)), places=10)

    def test_head_only_solver_recovers_translation_and_scale(self):
        rest = np.array([[400., 400.], [600., 400.], [500., 600.]])
        edges = np.array([[0, 1], [1, 2], [0, 2]])
        fraction = np.array([.2, .6, .4])
        base = rest[edges[:, 0]] * (1 - fraction[:, None]) + rest[edges[:, 1]] * fraction[:, None]
        expected = np.array([12., -17., SCALE_UNIT * np.log(1.12)])
        target = apply_head(base, expected)
        pairs = {'edges': edges, 'edge_t': fraction, 'target': target, 'weights': np.ones(3) / 3}
        frame = {'pose': [0., 0., 0.], 'pairs': pairs, 'distance': np.linalg.norm(base - target, axis=1)}
        incidence = sparse.csr_matrix([[1., -1., 0.]])
        problem = FrozenProblem(rest, np.zeros(3), [frame], incidence, np.zeros((1, 1)), None)
        result = least_squares(problem.residual, np.zeros(3), jac=problem.jacobian, method='trf',
                               tr_solver='lsmr', x_scale=np.full(3, 100.))
        self.assertGreater(result.status, 0)
        np.testing.assert_allclose(result.x, expected, atol=1e-5)

    def test_moving_skin_vertex_has_matching_actual_and_linearized_gradient(self):
        rest = np.array([[492., 316.], [512., 315.], [513., 336.], [491., 337.]])
        triangles = np.array([[0, 1, 2], [0, 2, 3]])
        depth = np.array([20., 21., 19., 20.])
        pose = [10., -8., 3.]
        base = project(np.column_stack((rest - [502, 326], depth)), pose)
        target = apply_head(base, [.7, -.4, 2.62])
        points = [(1 - t) * target[a] + t * target[b]
                  for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]
                  for t in np.linspace(0, 1, 13, endpoint=False)]
        rows = [{'frame': 1, 'pose': pose, 'pointsArt': np.asarray(points).tolist(), 'keep': [True] * len(points)}]
        incidence, factor = edge_incidence(rest, triangles), np.zeros((1, 1))
        field, head = np.zeros((1, 4, 3)), np.zeros((1, 3))
        frames = evaluate(rest, depth, triangles, rows, field, head)
        problem = FrozenProblem(rest, depth, frames, incidence, factor, .1)
        x = np.r_[field.ravel(), head.ravel()]
        gradient = problem.jacobian(x).T @ problem.residual(x)
        direction = np.zeros_like(x)
        direction[7] = 1.
        eps = 1e-4
        def actual(values):
            f, h = problem.unpack(values)
            current = evaluate(rest, depth, triangles, rows, f, h)
            return actual_objective(current, f, h, .1, incidence, factor)
        numeric = (actual(x + eps * direction) - actual(x - eps * direction)) / (2 * eps)
        self.assertAlmostEqual(numeric, gradient[7], places=5)

    def test_boundary_keeps_disconnected_island_and_excludes_holes(self):
        rest = np.array([[0., 0.], [4., 0.], [4., 4.], [0., 4.],
                         [1., 1.], [3., 1.], [3., 3.], [1., 3.],
                         [10., 0.], [12., 0.], [11., 2.]])
        triangles = np.array([[0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                              [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7], [8, 9, 10]])
        result = stable_boundary(rest, rest, triangles)
        edges = {tuple(edge) for edge in result['edges']}
        self.assertEqual(edges, {(0, 1), (1, 2), (2, 3), (0, 3), (8, 9), (9, 10), (8, 10)})
        moved = stable_boundary(rest, rest * 1.1 + [3., 2.], triangles)
        np.testing.assert_array_equal(moved['edges'], result['edges'])
        np.testing.assert_array_equal(moved['edge_t'], result['edge_t'])


if __name__ == '__main__':
    unittest.main()
