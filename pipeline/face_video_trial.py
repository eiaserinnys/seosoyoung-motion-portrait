"""Rejected point-target trial retained for diagnosis, not for contour fitting."""
import argparse
import json
import resource
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.interpolate import make_interp_spline
from scipy.sparse.linalg import spsolve


ROOT = Path(__file__).resolve().parents[1]
YAW = np.array([-45, -30, -15, 0, 15, 30, 45], float)
PITCH = np.array([-34, -17, 0, 17, 34], float)
YAW_BASIS = make_interp_spline(YAW, np.eye(len(YAW)), k=3, axis=0)
PITCH_BASIS = make_interp_spline(PITCH, np.eye(len(PITCH)), k=3, axis=0)


def centered_basis(yaw, pitch):
    return (np.outer(YAW_BASIS(yaw), PITCH_BASIS(pitch)) -
            np.outer(YAW_BASIS(0), PITCH_BASIS(0))).reshape(-1)


def rotation(yaw, pitch, roll):
    a, b, c = np.deg2rad([yaw, pitch, roll])
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    rigid = np.array([[ca, sa * sb, sa * cb], [0, cb, -sb], [-sa, ca * sb, ca * cb]])
    rolled = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    return rolled @ rigid


def project(local, pose):
    yaw, pitch, roll = pose
    point = np.asarray(local) @ rotation(yaw, pitch, roll).T
    angle = np.deg2rad(roll)
    pivot = np.array([0., 242.])
    offset = pivot - np.array([[np.cos(angle), -np.sin(angle)],
                               [np.sin(angle), np.cos(angle)]]) @ pivot
    return point[:, :2] + [502, 326] + offset


def ordered_correspondence(source, target):
    """Match two chin-outward arcs while preserving the order on each arc."""
    source, target = np.asarray(source), np.asarray(target)
    tangent = np.gradient(target, axis=0)
    normal = np.column_stack((tangent[:, 1], -tangent[:, 0]))
    normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-8)
    delta = source[:, None, :] - target[None, :, :]
    distance = np.linalg.norm(delta, axis=2)
    normal_distance = np.abs(np.sum(delta * normal[None, :, :], axis=2))
    huber = lambda d: np.where(d <= 2, .5 * d * d, 2 * (d - 1))
    source_length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(source, axis=0), axis=1))]
    target_length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(target, axis=0), axis=1))]
    source_fraction = source_length / source_length[-1]
    target_fraction = target_length / target_length[-1]
    cost = (huber(distance) + huber(normal_distance) +
            100 * (source_fraction[:, None] - target_fraction[None, :]) ** 2)
    n, m = cost.shape
    table = np.full((n, m), np.inf)
    previous = np.zeros((n, m), int)
    table[0, 0] = cost[0, 0]
    for i in range(1, n):
        best, at = np.inf, 0
        for j in range(1, m):
            if table[i - 1, j - 1] < best:
                best, at = table[i - 1, j - 1], j - 1
            table[i, j] = best + cost[i, j]
            previous[i, j] = at
    indices = np.zeros(n, int)
    indices[-1] = m - 1
    for i in range(n - 1, 0, -1):
        indices[i - 1] = previous[i, indices[i]]
    return indices


def visible_prefix(source, target):
    source_length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(source, axis=0), axis=1))]
    target_length = np.linalg.norm(np.diff(target, axis=0), axis=1).sum()
    last = max(1, int(np.argmin(np.abs(source_length - target_length))))
    return np.arange(last + 1)


def mesh_boundary(points, triangles):
    edges = Counter(tuple(sorted((t[k], t[(k + 1) % 3]))) for t in triangles for k in range(3))
    neighbors = defaultdict(list)
    for (a, b), count in edges.items():
        if count == 1:
            neighbors[a].append(b)
            neighbors[b].append(a)
    remaining = set(neighbors)
    components = []
    while remaining:
        stack, component = [remaining.pop()], []
        while stack:
            i = stack.pop()
            component.append(i)
            for j in neighbors[i]:
                if j in remaining:
                    remaining.remove(j)
                    stack.append(j)
        components.append(component)
    boundary = max(components, key=len)
    chin = max(boundary, key=lambda i: (points[i, 1], -abs(points[i, 0] - 502)))
    sides = []
    for first in neighbors[chin]:
        side, previous, current = [chin], chin, first
        while points[current, 1] >= 430 and current != chin:
            side.append(current)
            following = next(i for i in neighbors[current] if i != previous)
            previous, current = current, following
        sides.append(np.asarray(side))
    return sorted(sides, key=lambda side: float(np.mean(points[side, 0])))


def observed_sides(row):
    points = np.asarray(row['pointsArt'])
    keep = np.asarray(row['keep'])
    chin = np.flatnonzero(keep)[np.argmax(points[keep, 1])]
    count = len(keep)
    doubled = np.r_[~keep, ~keep]
    ends = np.flatnonzero(np.diff(np.r_[False, doubled, False]))
    runs = [(start, min(end - start, count)) for start, end in zip(ends[::2], ends[1::2]) if start < count]
    start, length = max(runs, key=lambda item: item[1])
    hidden = {(start + k) % count for k in range(length)}
    sides = []
    for direction in (-1, 1):
        indices, i = [chin], (chin + direction) % count
        while i not in hidden and i != chin:
            if keep[i]:
                indices.append(i)
            i = (i + direction) % count
        sides.append(points[indices])
    return sorted(sides, key=lambda side: float(np.mean(side[:, 0])))


def graph_laplacian(points, triangles):
    edges = {tuple(sorted((t[k], t[(k + 1) % 3]))) for t in triangles for k in range(3)}
    rows, cols, values = [], [], []
    for a, b in edges:
        weight = 1 / max(np.linalg.norm(points[a] - points[b]), 1.)
        rows.extend((a, b, a, b))
        cols.extend((a, b, b, a))
        values.extend((weight, weight, -weight, -weight))
    return sparse.coo_matrix((values, (rows, cols)), shape=(len(points), len(points))).tocsr()


def triangle_determinants(rest, posed, triangles):
    tri = np.asarray(triangles)
    def area(p):
        a, b, c = p[tri].transpose(1, 0, 2)
        return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    return area(posed) / area(rest)


def remove_inversions(rest, posed, triangles, fixed):
    result = posed.copy()
    locked = np.zeros(len(rest), bool)
    locked[fixed] = True
    for _ in range(128):
        changed = False
        for tri in triangles:
            a, b, c = result[tri]
            u, v, w = rest[tri]
            area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            original = (v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0])
            signed = np.sign(original) * area
            minimum = abs(original) * 1e-5
            if signed > minimum:
                continue
            gradient = np.sign(original) * np.array([[b[1] - c[1], c[0] - b[0]],
                                                      [c[1] - a[1], a[0] - c[0]],
                                                      [a[1] - b[1], b[0] - a[0]]])
            gradient[locked[tri]] = 0
            norm = np.sum(gradient * gradient)
            if norm:
                result[tri] += ((minimum - signed) / norm) * gradient
                changed = True
        if not changed:
            break
    return result


def targets(row, rest, zprior, triangles, laplacian):
    q0 = project(np.column_stack((rest - [502, 326], zprior)), row['pose'])
    source_sides = mesh_boundary(rest, triangles)
    target_sides = observed_sides(row)
    fixed = {}
    for source, target in zip(source_sides, target_sides):
        source = source[visible_prefix(q0[source], target)]
        matches = ordered_correspondence(q0[source], target)
        for i, j in zip(source, matches):
            fixed[int(i)] = target[j]
    ids = np.array(sorted(fixed))
    desired = np.array([fixed[i] for i in ids])
    free = np.setdiff1d(np.arange(len(rest)), ids)
    delta = np.zeros_like(rest)
    delta[ids] = desired - q0[ids]
    system = laplacian + sparse.eye(len(rest), format='csr') * 1e-5
    delta[free] = spsolve(system[free][:, free], -system[free][:, ids] @ delta[ids])
    result = remove_inversions(rest, q0 + delta, triangles, ids)
    determinant = triangle_determinants(rest, result, triangles)
    return {'frame': row['frame'], 'pose': row['pose'], 'targets': result.tolist(),
            'boundary_ids': ids.tolist(), 'boundary_targets': desired.tolist(),
            'reversed_triangles': int(np.count_nonzero(determinant <= 0)),
            'minimum_determinant': float(determinant.min())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
    observations = json.loads(args.observations.read_text())['observations']
    data = json.loads((ROOT / 'source/viewer/face-solve.json').read_text())
    count = data['meshCount']
    rest, z = np.asarray(data['points'][:count]), np.asarray(data['Z'][:count])
    triangles = data['triangles']
    laplacian = graph_laplacian(rest, triangles)
    results = [targets(row, rest, z, triangles, laplacian) for row in observations]
    args.output.write_text(json.dumps({'results': results}, ensure_ascii=False, separators=(',', ':')) + '\n')
    for row in results:
        print(f'{row["frame"]}: 대응 {len(row["boundary_ids"])} · 뒤집힘 {row["reversed_triangles"]} · 최소 {row["minimum_determinant"]:.4f}', flush=True)
    print(f'SAVED {args.output}', flush=True)


if __name__ == '__main__':
    main()
