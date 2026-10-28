"""Fit shared depth to visible video arcs with fixed tracked poses."""
import argparse
import hashlib
import json
import resource
import sys
from pathlib import Path

import numpy as np
import shapely
from scipy import sparse
from scipy.sparse.linalg import lsqr
from scipy.spatial import cKDTree

from face_video_contour import contour, huber_weights, project, rotation


ROOT = Path(__file__).resolve().parents[1]
FRAMES = [1, 34, 88, 298, 256]
DEPTH_PRIOR = .05
DEPTH_SMOOTH = .05
OVERLAP_PENALTY = 100.


def arc_labels(keep):
    keep = np.asarray(keep)
    labels = np.full(len(keep), -1, int)
    starts = np.flatnonzero(keep & ~np.roll(keep, 1))
    if not len(starts) and np.any(keep):
        labels[:] = 0
    for label, start in enumerate(starts):
        i = start
        while keep[i]:
            labels[i] = label
            i = (i + 1) % len(keep)
    return labels


def correspondences(outline, all_observations, keep, frames):
    """Both directions, with each visible arc and each frame normalized."""
    q, p, keep = outline['points'], np.asarray(all_observations), np.asarray(keep)
    labels = arc_labels(keep)
    visible = np.flatnonzero(keep)
    forward = cKDTree(q).query(p[visible])[1]
    nearest = cKDTree(p).query(q)[1]
    reverse = np.flatnonzero(keep[nearest])
    qids = np.r_[forward, reverse]
    targets = np.vstack((p[visible], p[nearest[reverse]]))
    groups = np.r_[labels[visible], labels[nearest[reverse]]]
    directions = np.r_[np.zeros(len(forward), int), np.ones(len(reverse), int)]
    weight = np.zeros(len(qids))
    distinct = np.unique(labels[keep])
    for group in distinct:
        present = [d for d in (0, 1) if np.any((groups == group) & (directions == d))]
        for d in present:
            mask = (groups == group) & (directions == d)
            weight[mask] = 1 / (frames * len(distinct) * len(present) * np.count_nonzero(mask))
    return {'q': q[qids], 'target': targets, 'edges': outline['edges'][qids],
            'edge_t': outline['edge_t'][qids], 'weights': weight,
            'forward_count': len(forward), 'reverse_count': len(reverse)}


def depth_jacobian(pose):
    return rotation(*pose)[:2, 2]


def edge_incidence(points, triangles):
    edges = sorted({tuple(sorted((int(t[k]), int(t[(k + 1) % 3])))) for t in triangles for k in range(3)})
    edges = np.asarray(edges)
    length = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    weight = 1 / np.sqrt(np.maximum(length, 1.))
    rows = np.repeat(np.arange(len(edges)), 2)
    columns = edges.reshape(-1)
    values = np.column_stack((weight, -weight)).reshape(-1)
    return sparse.coo_matrix((values, (rows, columns)), shape=(len(edges), len(points))).tocsr()


def evaluate(rest, z, triangles, observations):
    frames = []
    for row in observations:
        q = project(np.column_stack((rest - [502, 326], z)), row['pose'])
        surface = contour(q, triangles)
        pairs = correspondences(surface, np.asarray(row['pointsArt']), np.asarray(row['keep']), len(observations))
        distance = np.linalg.norm(pairs['q'] - pairs['target'], axis=1)
        frames.append({'frame': row['frame'], 'pose': row['pose'], 'vertices': q, 'surface': surface,
                       'pairs': pairs, 'distance': distance})
    return frames


def statistics(frames):
    return [{'frame': f['frame'], 'mean': float(np.mean(f['distance'])),
             'p95': float(np.percentile(f['distance'], 95)), 'max': float(f['distance'].max()),
             'back_faces': f['surface']['back_faces'], 'overlap_pairs': f['surface']['overlap_pairs'],
             'overlap_area': f['surface']['overlap_area'],
             'forward_count': f['pairs']['forward_count'], 'reverse_count': f['pairs']['reverse_count']}
            for f in frames]


def fit_diagnostics(stats, history):
    overlap = [row['frame'] for row in stats if row['overlap_pairs']]
    limited = [i + 1 for i, row in enumerate(history) if row['lsqr_stop'] in (3, 6, 7)]
    stalled = history[-1].get('stalled', False)
    final_limited = history[-1]['lsqr_stop'] in (3, 6, 7)
    return {'valid': not overlap and not final_limited and not stalled, 'overlap_frames': overlap,
            'lsqr_limited_iterations': limited,
            'lsqr_final_limited': final_limited,
            'outer_converged': history[-1]['depth_step_rms'] < .01 and not stalled,
            'line_search_stalled': stalled}


def overlap_step_allowed(old, new):
    return all((a['overlap_pairs'] > 0 or b['overlap_pairs'] == 0) and
               b['overlap_area'] <= a['overlap_area'] + 1e-7 for a, b in zip(old, new))


def feasible_start(prior, proposal):
    choice = min(proposal['feasible'], key=lambda item: abs(item['depth_offset']))
    start = prior.copy()
    start[proposal['island_vertices']] += choice['depth_offset']
    return start


def objective(frames, z, zprior, incidence):
    data = 0.
    for frame in frames:
        distance = frame['distance']
        loss = np.where(distance <= 2, .5 * distance * distance, 2 * (distance - 1))
        data += np.sum(frame['pairs']['weights'] * loss)
    return float(data + .5 * DEPTH_PRIOR * np.mean((z - zprior) ** 2) +
                 .5 * DEPTH_SMOOTH * np.mean((incidence @ z) ** 2) +
                 .5 * OVERLAP_PENALTY * sum(frame['surface']['overlap_area'] for frame in frames))


def overlap_rows(frame, triangles, count):
    """Linearize observed front-face overlap areas with respect to depth."""
    vertices = frame['vertices']
    derivative = depth_jacobian(frame['pose'])
    rows, cols, values, residuals = [], [], [], []
    for pair, area in zip(frame['surface']['overlap_triangles'], frame['surface']['overlap_areas']):
        first, second = triangles[pair]
        ids = np.unique(np.r_[first, second])
        grad = []
        for index in ids:
            h = .1
            def intersection(step):
                moved = vertices.copy()
                moved[index] += step * derivative
                return shapely.Polygon(moved[first]).intersection(shapely.Polygon(moved[second])).area
            grad.append((intersection(h) - intersection(-h)) / (2 * h))
        scale = np.sqrt(OVERLAP_PENALTY) / (2 * np.sqrt(area))
        rows.extend([len(residuals)] * len(ids))
        cols.extend(ids.tolist())
        values.extend((np.asarray(grad) * scale).tolist())
        residuals.append(np.sqrt(OVERLAP_PENALTY * area))
    matrix = sparse.coo_matrix((values, (rows, cols)), shape=(len(residuals), count)).tocsr()
    return matrix, np.asarray(residuals)


def fit_depth(rest, zprior, triangles, observations, z_start, iterations=16):
    z = z_start.copy()
    incidence = edge_incidence(rest, triangles)
    prior_scale = np.sqrt(DEPTH_PRIOR / len(z))
    smooth_scale = np.sqrt(DEPTH_SMOOTH / incidence.shape[0])
    prior = sparse.eye(len(z), format='csr') * prior_scale
    smooth = incidence * smooth_scale
    history = []
    current = evaluate(rest, z, triangles, observations)
    for iteration in range(iterations):
        matrices, rhs = [], []
        for frame in current:
            pairs = frame['pairs']
            residual = pairs['q'] - pairs['target']
            whitening = np.sqrt(pairs['weights'] * huber_weights(frame['distance']))
            m = len(residual)
            edges, t = pairs['edges'], pairs['edge_t']
            derivative = depth_jacobian(frame['pose'])
            row = np.tile(np.arange(2 * m).reshape(m, 2), (1, 2)).reshape(-1)
            col = np.repeat(edges, 2, axis=1).reshape(-1)
            val = np.column_stack(((1 - t) * derivative[0], (1 - t) * derivative[1],
                                   t * derivative[0], t * derivative[1]))
            val *= whitening[:, None]
            matrix = sparse.coo_matrix((val.reshape(-1), (row, col)), shape=(2 * m, len(z))).tocsr()
            matrices.append(matrix)
            rhs.append((-residual * whitening[:, None]).reshape(-1))
            overlap_matrix, overlap_residual = overlap_rows(frame, triangles, len(z))
            matrices.append(overlap_matrix)
            rhs.append(-overlap_residual)
        matrices.extend((prior, smooth))
        rhs.extend((-prior_scale * (z - zprior), -smooth @ z))
        solved = lsqr(sparse.vstack(matrices, format='csr'), np.concatenate(rhs),
                      atol=1e-6, btol=1e-6, iter_lim=1200)
        delta = solved[0]
        stats = statistics(current)
        old_objective = objective(current, z, zprior, incidence)
        accepted, trial = None, None
        for step in 2. ** -np.arange(12):
            candidate = evaluate(rest, z + step * delta, triangles, observations)
            candidate_stats = statistics(candidate)
            candidate_objective = objective(candidate, z + step * delta, zprior, incidence)
            if (overlap_step_allowed(stats, candidate_stats) and
                    candidate_objective <= old_objective + 1e-10):
                accepted, trial = float(step), candidate
                break
        history.append({'iteration': iteration, 'mean': [row['mean'] for row in stats],
                        'overlap_pairs': [row['overlap_pairs'] for row in stats],
                        'depth_step_rms': float(np.sqrt(np.mean(delta * delta))) * (accepted or 1.),
                        'lsqr_stop': int(solved[1]), 'lsqr_iterations': int(solved[2]),
                        'lsqr_residual_norm': float(solved[3]), 'accepted_step': accepted,
                        'objective': old_objective, 'stalled': accepted is None})
        print(f'{iteration + 1}: 평균 {[round(row["mean"], 2) for row in stats]} · '
              f'겹침 {[row["overlap_pairs"] for row in stats]} · 크기 {accepted}', flush=True)
        if accepted is None:
            break
        z += accepted * delta
        current = trial
        if history[-1]['depth_step_rms'] < .01 and not any(row['overlap_pairs'] for row in stats):
            break
    return z, current, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start-probe', type=Path, required=True)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
    data = json.loads((ROOT / 'source/viewer/face-solve.json').read_text())
    all_observations = json.loads((ROOT / 'source/viewer/face-video-observations.json').read_text())['observations']
    by_frame = {r['frame']: r for r in all_observations}
    observations = [by_frame[f] for f in FRAMES]
    n = data['meshCount']
    rest, prior, triangles = np.asarray(data['points'][:n]), np.asarray(data['Z'][:n]), np.asarray(data['triangles'])
    proposal = json.loads(args.start_probe.read_text())
    if proposal['frames'] != FRAMES or not proposal['feasible']:
        raise ValueError('The start probe must contain a feasible candidate for the five trial frames')
    z_start = feasible_start(prior, proposal)
    z, final, history = fit_depth(rest, prior, triangles, observations, z_start)
    stats = statistics(final)
    diagnostics = fit_diagnostics(stats, history)
    output = {'frames': FRAMES, 'depth': z.tolist(), 'depth_prior_weight': DEPTH_PRIOR,
              'depth_smooth_weight': DEPTH_SMOOTH, 'overlap_penalty': OVERLAP_PENALTY,
              'start_probe_sha256': hashlib.sha256(args.start_probe.read_bytes()).hexdigest(),
              'start_island_vertices': proposal['island_vertices'],
              'start_depth_offset': float(z_start[proposal['island_vertices'][0]] - prior[proposal['island_vertices'][0]]),
              'stats': stats, 'history': history, 'diagnostics': diagnostics}
    args.output.write_text(json.dumps(output, ensure_ascii=False, separators=(',', ':')) + '\n')
    print(f'SAVED {args.output}', flush=True)
    print(f'검사: {diagnostics}', flush=True)
    if not diagnostics['valid']:
        sys.exit(1)


if __name__ == '__main__':
    main()
