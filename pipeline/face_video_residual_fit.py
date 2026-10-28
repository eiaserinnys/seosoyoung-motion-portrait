"""Fit a centered pose-only 3D field directly to the five observed skin arcs."""
import argparse
import hashlib
import json
import resource
import sys
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from face_video_contour import (PITCH, YAW, centered_basis, contour, huber_weights,
                                project, rotation)
from face_video_contour_fit import (FRAMES, correspondences, edge_incidence,
                                    overlap_step_allowed, statistics)


ROOT = Path(__file__).resolve().parents[1]
LAMBDA_C = [1., .1, .01]
LAMBDA_SPATIAL = .05
LAMBDA_TRANSLATION = .1
POSE_SMOOTH = .02


def center_field(field):
    return field - np.mean(field, axis=1, keepdims=True)


def basis_transform(poses):
    basis = np.asarray([centered_basis(yaw, pitch) for yaw, pitch in poses])
    grid = np.asarray([[centered_basis(yaw, pitch) for pitch in PITCH] for yaw in YAW])
    second_yaw = np.diff(grid, n=2, axis=0).reshape(-1, 35)
    second_pitch = np.diff(grid, n=2, axis=1).reshape(-1, 35)
    smooth = np.vstack((second_yaw, second_pitch))
    quadratic = POSE_SMOOTH * (smooth.T @ smooth) / len(smooth)
    constraints = np.block([[quadratic, basis.T], [basis, np.zeros((len(poses), len(poses)))]])
    transform = np.linalg.pinv(constraints, rcond=1e-12)[:35, 35:]
    precision = transform.T @ quadratic @ transform
    return basis, transform, (precision + precision.T) * .5


def apply_step(field, translation, frames):
    field = center_field(field)
    result = []
    for index, frame in enumerate(frames):
        edges, t = frame['edges'], frame['edge_t']
        local = field[index, edges[:, 0]] * (1 - t[:, None]) + field[index, edges[:, 1]] * t[:, None]
        result.append(local @ frame['jac'].T + translation[index] @ frame['roll'].T)
    return result


def pull_step(residuals, frames, nodes):
    field = np.zeros((len(frames), nodes, 3))
    translation = np.zeros((len(frames), 2))
    for index, (residual, frame) in enumerate(zip(residuals, frames)):
        local = residual @ frame['jac']
        edges, t = frame['edges'], frame['edge_t']
        for d in range(3):
            field[index, :, d] += np.bincount(edges[:, 0], weights=local[:, d] * (1 - t), minlength=nodes)
            field[index, :, d] += np.bincount(edges[:, 1], weights=local[:, d] * t, minlength=nodes)
        translation[index] = np.sum(residual @ frame['roll'], axis=0)
    return center_field(field), translation


def roll_matrix(angle):
    angle = np.deg2rad(angle)
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def step_sizes(delta_field, delta_translation, scale):
    field = float(np.sqrt(np.mean(np.sum(delta_field ** 2, axis=2)))) * scale
    translation = float(np.sqrt(np.mean(np.sum(delta_translation ** 2, axis=1)))) * scale
    return field, translation


def evaluate_fields(rest, z, triangles, observations, field, translation):
    frames = []
    for index, row in enumerate(observations):
        pose = row['pose']
        local = np.column_stack((rest - [502, 326], z)) + field[index]
        q = project(local, pose) + translation[index] @ roll_matrix(pose[2]).T
        surface = contour(q, triangles)
        pairs = correspondences(surface, np.asarray(row['pointsArt']), np.asarray(row['keep']), len(observations))
        distance = np.linalg.norm(pairs['q'] - pairs['target'], axis=1)
        frames.append({'frame': row['frame'], 'pose': pose, 'vertices': q, 'surface': surface,
                       'pairs': pairs, 'distance': distance})
    return frames


def regularization(field, translation, strength, incidence, precision):
    frames, nodes, _ = field.shape
    flat = field.transpose(1, 0, 2).reshape(nodes, -1)
    laplacian = (incidence.T @ (incidence @ flat)).reshape(nodes, frames, 3).transpose(1, 0, 2)
    grad_field = (strength * field / (frames * nodes) +
                  LAMBDA_SPATIAL * laplacian / (frames * incidence.shape[0]) +
                  np.einsum('st,tnd->snd', precision, field) / (frames * nodes))
    grad_translation = (LAMBDA_TRANSLATION * translation / frames + precision @ translation / frames)
    return center_field(grad_field), grad_translation


def objective(frames, field, translation, strength, incidence, precision):
    data = 0.
    for frame in frames:
        d = frame['distance']
        data += np.sum(frame['pairs']['weights'] * np.where(d <= 2, .5 * d * d, 2 * (d - 1)))
    grad_field, grad_translation = regularization(field, translation, strength, incidence, precision)
    return float(data + .5 * np.sum(field * grad_field) + .5 * np.sum(translation * grad_translation))


def linear_system(frames, field, translation, strength, incidence, precision):
    count, nodes, _ = field.shape
    specs = []
    weighted_residuals = []
    weights = []
    for frame in frames:
        pair = frame['pairs']
        specs.append({'edges': pair['edges'], 'edge_t': pair['edge_t'],
                      'jac': rotation(*frame['pose'])[:2], 'roll': roll_matrix(frame['pose'][2])})
        weight = pair['weights'] * huber_weights(frame['distance'])
        weights.append(weight)
        weighted_residuals.append(-weight[:, None] * (pair['q'] - pair['target']))
    rhs_field, rhs_translation = pull_step(weighted_residuals, specs, nodes)
    regular_field, regular_translation = regularization(field, translation, strength, incidence, precision)
    rhs_field -= regular_field
    rhs_translation -= regular_translation
    size_field = count * nodes * 3
    rhs = np.r_[rhs_field.reshape(-1), rhs_translation.reshape(-1)]

    def operator(vector):
        raw = vector[:size_field].reshape(count, nodes, 3)
        projected = center_field(raw)
        dt = vector[size_field:].reshape(count, 2)
        forwards = apply_step(projected, dt, specs)
        grad_field, grad_translation = pull_step([w[:, None] * p for w, p in zip(weights, forwards)], specs, nodes)
        regular_field, regular_translation = regularization(projected, dt, strength, incidence, precision)
        grad_field += regular_field + (raw - projected)
        grad_translation += regular_translation
        return np.r_[grad_field.reshape(-1), grad_translation.reshape(-1)]

    graph_diagonal = np.asarray(incidence.power(2).sum(axis=0)).ravel()
    diagonal_field = np.empty((count, nodes, 3))
    diagonal_translation = np.empty((count, 2))
    for i, frame in enumerate(specs):
        diagonal_field[i] = (strength / (count * nodes) +
                             LAMBDA_SPATIAL * graph_diagonal[:, None] / (count * incidence.shape[0]) +
                             precision[i, i] / (count * nodes))
        edge, fraction, weight = frame['edges'], frame['edge_t'], weights[i]
        for d in range(3):
            scale = np.sum(frame['jac'][:, d] ** 2)
            diagonal_field[i, :, d] += scale * np.bincount(edge[:, 0], weights=weight * (1 - fraction) ** 2, minlength=nodes)
            diagonal_field[i, :, d] += scale * np.bincount(edge[:, 1], weights=weight * fraction ** 2, minlength=nodes)
        diagonal_translation[i] = LAMBDA_TRANSLATION / count + precision[i, i] / count + weight.sum()
    diagonal = np.r_[diagonal_field.reshape(-1), diagonal_translation.reshape(-1)]
    system = LinearOperator((len(rhs), len(rhs)), matvec=operator, dtype=float)
    preconditioner = LinearOperator((len(rhs), len(rhs)), matvec=lambda v: v / diagonal, dtype=float)
    return system, preconditioner, rhs


def fit_one(rest, z, triangles, observations, strength, incidence, transform, precision, iterations=10):
    count, nodes = len(observations), len(rest)
    field, translation = np.zeros((count, nodes, 3)), np.zeros((count, 2))
    current = evaluate_fields(rest, z, triangles, observations, field, translation)
    history = []
    for iteration in range(iterations):
        system, preconditioner, rhs = linear_system(current, field, translation, strength, incidence, precision)
        solved, info = cg(system, rhs, M=preconditioner, rtol=1e-5, atol=0., maxiter=400)
        delta_field = center_field(solved[:count * nodes * 3].reshape(count, nodes, 3))
        delta_translation = solved[count * nodes * 3:].reshape(count, 2)
        old_stats = statistics(current)
        old_objective = objective(current, field, translation, strength, incidence, precision)
        accepted, candidate = None, None
        for step in 2. ** -np.arange(12):
            next_field = center_field(field + step * delta_field)
            next_translation = translation + step * delta_translation
            trial = evaluate_fields(rest, z, triangles, observations, next_field, next_translation)
            if (overlap_step_allowed(old_stats, statistics(trial)) and
                    objective(trial, next_field, next_translation, strength, incidence, precision) <= old_objective + 1e-10):
                accepted, candidate = float(step), trial
                break
        field_step, translation_step = step_sizes(delta_field, delta_translation, accepted or 1.)
        history.append({'iteration': iteration, 'cg_info': int(info), 'accepted_step': accepted,
                        'field_step_rms': field_step, 'translation_step_rms': translation_step,
                        'objective': old_objective,
                        'stalled': accepted is None})
        print(f'보정 {strength:g} · {iteration + 1}: '
              f'평균 {[round(row["mean"], 2) for row in old_stats]} · 크기 {accepted} · 계산 {info}', flush=True)
        if accepted is None:
            break
        field = center_field(field + accepted * delta_field)
        translation += accepted * delta_translation
        current = candidate
        if max(field_step, translation_step) < .01:
            break
    stats = statistics(current)
    norms = np.linalg.norm(field, axis=2)
    coef = (transform @ field.reshape(count, -1)).reshape(35, nodes, 3)
    tcoef = np.column_stack((transform @ translation, np.zeros(35)))
    return {'lambda_c': strength, 'stats': stats, 'C_rms': float(np.sqrt(np.mean(norms ** 2))),
            'C_p95': float(np.percentile(norms, 95)), 'C_max': float(norms.max()),
            'T_sizes': np.linalg.norm(translation, axis=1).tolist(),
            'C_coefficients': coef.tolist(), 'T_coefficients': tcoef.tolist(),
            'history': history,
            'diagnostics': {'overlap_frames': [r['frame'] for r in stats if r['overlap_pairs']],
                            'cg_info': history[-1]['cg_info'], 'line_search_stalled': history[-1]['stalled'],
                            'small_last_step': max(history[-1]['field_step_rms'], history[-1]['translation_step_rms']) < .01}}


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
    rows = json.loads((ROOT / 'source/viewer/face-video-observations.json').read_text())['observations']
    by_frame = {r['frame']: r for r in rows}
    observations = [by_frame[f] for f in FRAMES]
    nodes = data['meshCount']
    rest, triangles, z = np.asarray(data['points'][:nodes]), np.asarray(data['triangles']), np.asarray(depth['depth'])
    basis, transform, precision = basis_transform([row['pose'][:2] for row in observations])
    incidence = edge_incidence(rest, triangles)
    solutions = [fit_one(rest, z, triangles, observations, value, incidence, transform, precision) for value in LAMBDA_C]
    result = {'frames': FRAMES, 'depth': z.tolist(), 'depth_result_sha256': hashlib.sha256(args.depth_result.read_bytes()).hexdigest(),
              'depth_stats': depth['stats'], 'basis_rank': int(np.linalg.matrix_rank(basis)),
              'lambda_spatial': LAMBDA_SPATIAL, 'lambda_translation': LAMBDA_TRANSLATION,
              'pose_smooth': POSE_SMOOTH, 'pose_penalty': '격자 2차 차분, 동률일 때 최소 계수 norm',
              'solutions': solutions}
    args.output.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':')) + '\n')
    print(f'SAVED {args.output}', flush=True)
    if any(s['diagnostics']['overlap_frames'] or s['diagnostics']['cg_info'] != 0 or
           s['diagnostics']['line_search_stalled'] for s in solutions):
        sys.exit(1)


if __name__ == '__main__':
    main()
