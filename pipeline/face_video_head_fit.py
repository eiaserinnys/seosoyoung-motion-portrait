"""Five-frame diagnostic fit of tracker-initialized head placement and skin field."""
import argparse
import hashlib
import json
import resource
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import shapely
from scipy.optimize import least_squares
from scipy.sparse.linalg import LinearOperator

from face_video_contour import centered_basis, contour, huber_weights, project, rotation
from face_video_contour_fit import FRAMES, correspondences, edge_incidence, statistics
from face_video_residual_fit import LAMBDA_C, LAMBDA_SPATIAL, basis_transform, center_field


ROOT = Path(__file__).resolve().parents[1]
HEAD_CENTER = np.array([502., 326.])
NECK_PIVOT = np.array([502., 568.])
# The existing head radius gives scale displacement the same pixel units as H translation.
SCALE_UNIT = 262.
OUTER_LIMIT = 12
INNER_LIMIT = 30
FTOL = 1e-7
XTOL = 1e-7
GTOL = 1e-6
OUTER_STEP = 1e-4


def skin_depth(point, rest, depth, triangles):
    a, b, c = np.asarray(rest)[np.asarray(triangles)].transpose(1, 0, 2)
    p = np.asarray(point)
    cross = lambda u, v: u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    denominator = cross(b - a, c - a)
    w0 = cross(b - p, c - p) / denominator
    w1 = cross(c - p, a - p) / denominator
    w2 = 1 - w0 - w1
    inside = np.flatnonzero((w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9))
    if not len(inside):
        raise ValueError('Eye reference is outside the face-skin mesh')
    index = inside[0]
    return float(np.dot([w0[index], w1[index], w2[index]], np.asarray(depth)[triangles[index]]))


def apply_head(points, placement):
    scale = np.exp(placement[2] / SCALE_UNIT)
    return NECK_PIVOT + scale * (np.asarray(points) - NECK_PIVOT) + placement[:2]


def tracked_head(pose, body_matrix, anchor):
    body = np.asarray(body_matrix)
    target = body[:, :2] @ np.asarray(pose[4:6]) + body[:, 2]
    scale = np.linalg.norm(body[:, 0]) * np.exp(pose[3])
    local = np.asarray(anchor) - np.r_[HEAD_CENTER, 0.]
    base = project(local[None, :], pose[:3])[0]
    shift = target - NECK_PIVOT - scale * (base - NECK_PIVOT)
    return np.r_[shift, np.log(scale) * SCALE_UNIT]


def pose_factor(precision):
    values, vectors = np.linalg.eigh(precision)
    return np.sqrt(np.maximum(values, 0))[:, None] * vectors.T


def stable_boundary(rest, vertices, triangles):
    points = vertices[triangles]
    a, b, c = points.transpose(1, 0, 2)
    area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    front = triangles[area > 0]
    counts = Counter(tuple(sorted((int(t[i]), int(t[(i + 1) % 3])))) for t in front for i in range(3))
    edges = np.asarray(sorted(edge for edge, count in counts.items() if count == 1))
    union = shapely.union_all(shapely.polygons(vertices[front]))
    components = [union] if union.geom_type == 'Polygon' else list(union.geoms)
    exterior = shapely.union_all([p.exterior for p in components if p.geom_type == 'Polygon'])
    midpoints = (vertices[edges[:, 0]] + vertices[edges[:, 1]]) / 2
    edges = edges[shapely.distance(shapely.points(midpoints), exterior) < 1e-7]
    lengths = np.linalg.norm(rest[edges[:, 1]] - rest[edges[:, 0]], axis=1)
    amounts = np.maximum(1, np.ceil(lengths).astype(int))
    indices = np.repeat(np.arange(len(edges)), amounts)
    fractions = np.concatenate([(np.arange(count) + .5) / count for count in amounts])
    sample_edges = edges[indices]
    points = (vertices[sample_edges[:, 0]] * (1 - fractions[:, None]) +
              vertices[sample_edges[:, 1]] * fractions[:, None])
    return {'points': points, 'edges': sample_edges, 'edge_t': fractions}


def evaluate(rest, z, triangles, rows, field, head):
    frames = []
    for i, row in enumerate(rows):
        local = np.column_stack((rest - HEAD_CENTER, z)) + field[i]
        base = project(local, row['pose'])
        q = apply_head(base, head[i])
        surface = contour(q, triangles)
        # With disjoint front triangles, count-one edges are exactly the union boundary.
        # Fixed edge fractions prevent a changing perimeter grid from shifting the objective.
        if surface['overlap_pairs'] == 0:
            surface.update(stable_boundary(rest, q, triangles))
        pairs = correspondences(surface, np.asarray(row['pointsArt']), np.asarray(row['keep']), len(rows))
        distance = np.linalg.norm(pairs['q'] - pairs['target'], axis=1)
        frames.append({'frame': row['frame'], 'pose': row['pose'], 'vertices': q,
                       'surface': surface, 'pairs': pairs, 'distance': distance})
    return frames


def actual_objective(frames, field, head, strength, incidence, factor):
    count, nodes = field.shape[:2]
    value = 0.
    for frame in frames:
        d = frame['distance']
        value += np.sum(frame['pairs']['weights'] * np.where(d <= 2, .5 * d * d, 2 * (d - 1)))
    value += .5 * np.sum((factor @ head) ** 2) / count
    if strength is not None:
        value += .5 * strength * np.sum(field ** 2) / (count * nodes)
        value += .5 * LAMBDA_SPATIAL * sum(np.sum((incidence @ f) ** 2) for f in field) / (count * incidence.shape[0])
        value += .5 * np.sum(np.einsum('st,tnd->snd', factor, field) ** 2) / (count * nodes)
    return float(value)


class FrozenProblem:
    """IRLS residual and sparse operator for currently matched silhouette edges."""

    def __init__(self, rest, z, frames, incidence, factor, strength):
        self.rest, self.z, self.frames = rest, z, frames
        self.incidence, self.factor, self.strength = incidence, factor, strength
        self.count, self.nodes = len(frames), len(rest)
        self.field_size = self.count * self.nodes * 3 if strength is not None else 0
        self.weights = [np.sqrt(f['pairs']['weights'] * huber_weights(f['distance'])) for f in frames]
        self.c_size = np.sqrt(strength / (self.count * self.nodes)) if strength is not None else 0.
        self.c_space = np.sqrt(LAMBDA_SPATIAL / (self.count * incidence.shape[0]))
        self.c_pose = 1 / np.sqrt(self.count * self.nodes)
        self.h_pose = 1 / np.sqrt(self.count)
        self.data_sizes = [2 * len(f['pairs']['target']) for f in frames]
        self.residual_size = sum(self.data_sizes) + self.count * 3
        if self.field_size:
            self.residual_size += self.field_size * 2 + self.count * incidence.shape[0] * 3

    def unpack(self, x):
        if self.field_size:
            field = center_field(x[:self.field_size].reshape(self.count, self.nodes, 3))
        else:
            field = np.zeros((self.count, self.nodes, 3))
        return field, x[self.field_size:].reshape(self.count, 3)

    def base_edges(self, field):
        values = []
        for i, frame in enumerate(self.frames):
            local = np.column_stack((self.rest - HEAD_CENTER, self.z)) + field[i]
            points = project(local, frame['pose'])
            edges, t = frame['pairs']['edges'], frame['pairs']['edge_t'][:, None]
            values.append(points[edges[:, 0]] * (1 - t) + points[edges[:, 1]] * t)
        return values

    def residual(self, x):
        field, head = self.unpack(x)
        base = self.base_edges(field)
        rows = [(self.weights[i][:, None] * (apply_head(base[i], head[i]) - f['pairs']['target'])).ravel()
                for i, f in enumerate(self.frames)]
        if self.field_size:
            rows.extend([(self.c_size * field).ravel(),
                         (self.c_space * np.asarray([self.incidence @ f for f in field])).ravel(),
                         (self.c_pose * np.einsum('st,tnd->snd', self.factor, field)).ravel()])
        rows.append((self.h_pose * (self.factor @ head)).ravel())
        return np.concatenate(rows)

    def jacobian(self, x):
        field, head = self.unpack(x)
        base = self.base_edges(field)
        matrices = [rotation(*frame['pose'])[:2] for frame in self.frames]
        scales = np.exp(head[:, 2] / SCALE_UNIT)

        def matvec(vector):
            df, dh = self.unpack(vector)
            rows = []
            for i, frame in enumerate(self.frames):
                edges, t = frame['pairs']['edges'], frame['pairs']['edge_t'][:, None]
                local = df[i, edges[:, 0]] * (1 - t) + df[i, edges[:, 1]] * t
                step = (scales[i] * (local @ matrices[i].T) + dh[i, :2] +
                        scales[i] * (base[i] - NECK_PIVOT) * (dh[i, 2] / SCALE_UNIT))
                rows.append((self.weights[i][:, None] * step).ravel())
            if self.field_size:
                rows.extend([(self.c_size * df).ravel(),
                             (self.c_space * np.asarray([self.incidence @ f for f in df])).ravel(),
                             (self.c_pose * np.einsum('st,tnd->snd', self.factor, df)).ravel()])
            rows.append((self.h_pose * (self.factor @ dh)).ravel())
            return np.concatenate(rows)

        def rmatvec(vector):
            vector = np.asarray(vector).reshape(-1)
            df, dh = np.zeros((self.count, self.nodes, 3)), np.zeros((self.count, 3))
            offset = 0
            for i, frame in enumerate(self.frames):
                count = self.data_sizes[i]
                result = vector[offset:offset + count].reshape(-1, 2) * self.weights[i][:, None]
                offset += count
                dh[i, :2] = result.sum(axis=0)
                dh[i, 2] = scales[i] * np.sum(result * (base[i] - NECK_PIVOT)) / SCALE_UNIT
                if self.field_size:
                    local = scales[i] * result @ matrices[i]
                    edges, t = frame['pairs']['edges'], frame['pairs']['edge_t']
                    for axis in range(3):
                        df[i, :, axis] += np.bincount(edges[:, 0], weights=local[:, axis] * (1 - t), minlength=self.nodes)
                        df[i, :, axis] += np.bincount(edges[:, 1], weights=local[:, axis] * t, minlength=self.nodes)
            if self.field_size:
                size = self.field_size
                df += self.c_size * vector[offset:offset + size].reshape(df.shape)
                offset += size
                size = self.count * self.incidence.shape[0] * 3
                spatial = vector[offset:offset + size].reshape(self.count, self.incidence.shape[0], 3)
                df += self.c_space * np.asarray([self.incidence.T @ f for f in spatial])
                offset += size
                size = self.field_size
                posed = vector[offset:offset + size].reshape(df.shape)
                df += self.c_pose * np.einsum('ts,tnd->snd', self.factor, posed)
                offset += size
            dh += self.h_pose * self.factor.T @ vector[offset:].reshape(self.count, 3)
            return np.r_[center_field(df).ravel(), dh.ravel()] if self.field_size else dh.ravel()

        return LinearOperator((self.residual_size, len(x)), matvec=matvec, rmatvec=rmatvec, dtype=float)


def fit(rest, z, triangles, rows, initial_head, strength, incidence, factor, transform):
    count, nodes = len(rows), len(rest)
    field = np.zeros((count, nodes, 3))
    head = initial_head.copy()
    frames = evaluate(rest, z, triangles, rows, field, head)
    history, status = [], 'iteration_limit'
    for iteration in range(OUTER_LIMIT):
        frozen = FrozenProblem(rest, z, frames, incidence, factor, strength)
        initial = np.r_[field.ravel(), head.ravel()] if strength is not None else head.ravel()
        scale = np.r_[np.full(frozen.field_size, 10.), np.full(count * 3, 100.)]
        solved = least_squares(frozen.residual, initial, jac=frozen.jacobian, method='trf',
                               tr_solver='lsmr', x_scale=scale,
                               tr_options={'maxiter': 200, 'atol': 1e-5, 'btol': 1e-5},
                               ftol=FTOL, xtol=XTOL, gtol=GTOL, max_nfev=INNER_LIMIT)
        proposed_field, proposed_head = frozen.unpack(solved.x)
        change_field, change_head = proposed_field - field, proposed_head - head
        old_value = actual_objective(frames, field, head, strength, incidence, factor)
        accepted, candidate = None, None
        for step in 2. ** -np.arange(12):
            next_field = center_field(field + step * change_field)
            next_head = head + step * change_head
            trial = evaluate(rest, z, triangles, rows, next_field, next_head)
            if any(row['overlap_pairs'] for row in statistics(trial)):
                continue
            value = actual_objective(trial, next_field, next_head, strength, incidence, factor)
            if value <= old_value + 1e-10:
                accepted, candidate = float(step), trial
                break
        scaled_step = float(np.linalg.norm((solved.x - initial) / scale))
        history.append({'outer': iteration + 1, 'inner_status': int(solved.status),
                        'inner_nfev': int(solved.nfev), 'inner_optimality': float(solved.optimality),
                        'outer_objective': old_value, 'accepted_fraction': accepted,
                        'proposed_scaled_step': scaled_step})
        print(f'{"머리" if strength is None else f"보정 {strength:g}"} {iteration + 1}: '
              f'평균 {[round(v["mean"], 2) for v in statistics(frames)]} · '
              f'반영 {accepted} · 내부 {solved.status}/{solved.nfev}', flush=True)
        if accepted is None:
            status = 'line_search_stalled'
            break
        field = center_field(field + accepted * change_field)
        head += accepted * change_head
        frames = candidate
        if accepted == 1. and scaled_step < OUTER_STEP and solved.status > 0:
            status = 'converged'
            break
    norms = np.linalg.norm(field, axis=2)
    return {'lambda_c': strength, 'head': head.tolist(), 'head_coefficients': (transform @ head).tolist(),
            'C_rms': float(np.sqrt(np.mean(norms ** 2))), 'C_p95': float(np.percentile(norms, 95)),
            'C_max': float(np.max(norms)),
            'C_coefficients': (transform @ field.reshape(count, -1)).reshape(35, nodes, 3).tolist(),
            'stats': statistics(frames), 'status': status, 'history': history}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth-result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
    depth = json.loads(args.depth_result.read_text())
    if depth['frames'] != FRAMES or not depth['diagnostics']['valid']:
        raise ValueError('A valid five-frame depth result is required')
    data = json.loads((ROOT / 'source/viewer/face-solve.json').read_text())
    observation_data = json.loads((ROOT / 'source/viewer/face-video-observations.json').read_text())
    registration = json.loads((ROOT / 'source/viewer/ponytail-seedance-registration.json').read_text())
    features = json.loads((ROOT / 'source/viewer/face-feature-fit.json').read_text())['front']
    by_frame = {r['frame']: r for r in observation_data['observations']}
    raw = {r['frame']: r for r in registration['video']}
    rows = [by_frame[f] for f in FRAMES]
    nodes = data['meshCount']
    rest, triangles, z = np.asarray(data['points'][:nodes]), np.asarray(data['triangles']), np.asarray(depth['depth'])
    eyes = np.asarray([features['left'], features['right']])
    eye_z = np.mean([skin_depth(point, rest, z, triangles) for point in eyes])
    tracker_eye_z = np.mean(np.asarray(registration['model']['points3D'])[:2, 2])
    anchor = np.r_[eyes.mean(axis=0), eye_z - tracker_eye_z]
    if any(not np.allclose(raw[r['frame']]['pose'][:3], r['pose'], rtol=0, atol=1e-10) for r in rows):
        raise ValueError('The stored slider pose must match the tracking data')
    initial_head = np.asarray([tracked_head(raw[r['frame']]['pose'], r['G_after_body'], anchor) for r in rows])
    basis, transform, precision = basis_transform([row['pose'][:2] for row in rows])
    incidence, factor = edge_incidence(rest, triangles), pose_factor(precision)
    result = {'frames': FRAMES, 'depth': z.tolist(), 'depth_stats': depth['stats'],
              'depth_result_sha256': hashlib.sha256(args.depth_result.read_bytes()).hexdigest(),
              'neck_pivot': NECK_PIVOT.tolist(), 'scale_unit': SCALE_UNIT,
              'tracker_anchor': anchor.tolist(), 'skin_eye_depth': eye_z,
              'tracker_eye_depth': tracker_eye_z, 'tracked_head': initial_head.tolist(),
              'basis_rank': int(np.linalg.matrix_rank(basis)),
              'solver': {'method': 'scipy least_squares trf / outer IRLS-ICP', 'ftol': FTOL, 'xtol': XTOL,
                         'gtol': GTOL, 'outer_step': OUTER_STEP, 'outer_limit': OUTER_LIMIT,
                         'inner_limit': INNER_LIMIT, 'field_scale': 10., 'head_scale': 100.},
              'head_only': None, 'solutions': []}

    def save():
        args.output.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':')) + '\n')

    started = time.monotonic()
    head_result = fit(rest, z, triangles, rows, initial_head, None, incidence, factor, transform)
    head_result['seconds'] = time.monotonic() - started
    result['head_only'] = head_result
    save()
    for value in LAMBDA_C:
        started = time.monotonic()
        solution = fit(rest, z, triangles, rows, np.asarray(head_result['head']), value,
                       incidence, factor, transform)
        solution['seconds'] = time.monotonic() - started
        result['solutions'].append(solution)
        save()
    print(f'SAVED {args.output}', flush=True)
    if any(solution['status'] != 'converged' for solution in [head_result, *result['solutions']]):
        sys.exit(1)


if __name__ == '__main__':
    main()
